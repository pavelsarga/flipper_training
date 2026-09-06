# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher
import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a receding-horizon (action-chunking) policy inside FTR-Benchmark (Isaac Sim)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a diffusion-policy config yaml")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in config")
    parser.add_argument("--terrain", type=str, default=None, help="Override terrain in config")
    parser.add_argument("--task", type=str, default=None, help="Override task in config (e.g. Marv-Crossing-Direct-v0)")
    parser.add_argument("--play", type=str, default=None, metavar="RUN_DIR",
                        help="Visualise a trained policy instead of training. "
                             "Pass the run directory (e.g. runs/ppo/ftr_ppo_crossing_2026-…). "
                             "Loads policy_final.pth + vecnorm_final.pth from <RUN_DIR>/weights/.")
    AppLauncher.add_app_launcher_args(parser)
    args, unknown_args = parser.parse_known_args()

    # AppLauncher processes some flags (e.g. --gpu) from sys.argv directly without
    # removing them from unknown_args, so they leak into OmegaConf overrides and crash.
    # Strip any --flag / value pairs that are not OmegaConf key=value overrides.
    _filtered, _skip = [], False
    for _a in unknown_args:
        if _skip:
            _skip = False
            continue
        if _a.startswith("--") and "=" not in _a:
            _skip = True  # also drop the following positional value
            continue
        _filtered.append(_a)
    unknown_args = _filtered

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Force-exit on ANY uncaught exception, from here on.
    #
    # The try/except around trainer.train() lower down covers only training. It does not
    # cover the module-level imports in BLOCK 2, which run after Isaac has already started —
    # and an exception there hangs exactly as a training crash does, because the interpreter
    # falls into a normal shutdown and Isaac's atexit handlers deadlock. Observed: job
    # 11498701 died instantly on a ModuleNotFoundError and then sat on a GPU for its full
    # 30-minute walltime.
    #
    # sys.excepthook runs before interpreter shutdown, so os._exit here bypasses atexit for
    # every uncaught exception regardless of where it was raised.
    import sys as _sys

    def _force_exit_on_uncaught(exc_type, exc, tb):
        import os as _os
        import traceback as _tb

        _tb.print_exception(exc_type, exc, tb)
        _sys.stdout.flush()
        _sys.stderr.flush()
        _os._exit(1)

    _sys.excepthook = _force_exit_on_uncaught

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

import torch
from omegaconf import OmegaConf
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import gymnasium

import marv_rl_training  # registers OmegaConf resolvers
from torchrl.envs import CatFrames

from marv_rl_training.environment.chunked_env import ActionChunkEnv
from marv_rl_training.policies.diffusion_dppo import DiffusionChunkActor, make_dppo_loss
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
from marv_rl_training.training.common import make_transformed_env
from marv_rl_training.training.env_type_registry import default_num_env_types
from marv_rl_training.training.eval_data import (
    aggregate_per_env,
    aggregate_per_spot,
    load_env_type_names,
    run_tracked_rollout,
    save_per_spot_csv,
)
from marv_rl_training.utils.cfg_schedulers import _make_cfg_scheduler
from marv_rl_training.utils.logutils import RunLogger, get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device




# ============================================================
# BLOCK 3 — Reward analysis helpers
# ============================================================

_REWARD_ANALYSIS_EXCLUDE = frozenset({
    "rew/total_reward",
    "rew/success_reward_value",
    "rew/fail_reward_value",
    "rew/success_rate_step",
    "rew/fail_rate_step",
    "rew/step_penalty",
})


def _compute_reward_grad_stats(
    flat: "torch.Tensor",
    reward_series: dict[str, list[float]],
    time_steps_per_batch: int,
    execution_horizon: int = 1,
) -> dict[str, float]:
    """Compute per-component variance decomposition and gradient contribution.

    **Variance decomposition** — ``Cov(R_i, R_total) / Var(R_total)``
    Each component's share of total reward variance; values sum to ~1 across
    all components (may be negative when a component is anti-correlated with
    the total reward).

    **Gradient contribution** — ``Cov(R_i, A) / Σ_j |Cov(R_j, A)|``
    Each component's correlation with the GAE advantage that drives policy
    updates; absolute values sum to 1.

    Both statistics use per-step mean time-series (T values = time_steps_per_batch,
    each value being the mean over the healthy env population for that step).

    Args:
        flat: Reshaped rollout tensordict of shape ``(N_clean * T,)``.
        reward_series: Per-CONTROL-step scalar series from
            ``FtrTorchRLEnv.peek_reward_series()`` — NOT yet cleared. Under action chunking
            this is T * T_a long while the rollout is T macro steps, so it is averaged down
            in blocks of T_a before being correlated. Slicing it as ``series[-T:]`` (what
            the un-chunked trainer does) would silently line the last T control steps up
            against all T macro steps and produce meaningless correlations.
        time_steps_per_batch: Number of macro steps T collected per iteration.
        execution_horizon: T_a — control steps per macro step.

    Returns:
        Dict with keys ``var_decomp/{component}`` and ``grad_contrib/{component}``.
        Empty dict if total reward has near-zero variance or no valid components.
    """
    T = time_steps_per_batch
    T_a = max(1, execution_horizon)
    N_clean = flat.shape[0] // T

    # Per-step mean total reward and advantage over healthy envs: (T,)
    r_total = flat[("next", "reward")].squeeze(-1).reshape(N_clean, T).mean(0)
    adv = flat["advantage"].squeeze(-1).reshape(N_clean, T).mean(0)

    r_total_c = r_total - r_total.mean()
    adv_c = adv - adv.mean()
    var_total = (r_total_c ** 2).mean()

    if var_total < 1e-8:
        return {}

    var_decomp_log: dict[str, float] = {}
    cov_with_adv: dict[str, float] = {}

    for key, series in reward_series.items():
        if key in _REWARD_ANALYSIS_EXCLUDE or not key.startswith("rew/"):
            continue
        if len(series) < T * T_a:
            continue
        r_i = torch.tensor(series[-T * T_a :], dtype=torch.float32, device=r_total.device)
        if T_a > 1:
            r_i = r_i.reshape(T, T_a).mean(dim=1)  # control steps -> macro steps
        # Skip zero-variance components (e.g. disabled penalties).
        if r_i.std() < 1e-6:
            continue
        r_i_c = r_i - r_i.mean()
        short_key = key[4:]  # strip "rew/" prefix

        cov_total = (r_i_c * r_total_c).mean()
        var_decomp_log[f"var_decomp/{short_key}"] = (cov_total / var_total).item()

        cov_adv = (r_i_c * adv_c).mean()
        cov_with_adv[f"grad_contrib/{short_key}"] = cov_adv.item()

    # Normalise gradient contribution so absolute values sum to 1
    total_abs = sum(abs(v) for v in cov_with_adv.values())
    if total_abs > 1e-8:
        cov_with_adv = {k: v / total_abs for k, v in cov_with_adv.items()}

    return {**var_decomp_log, **cov_with_adv}


# ============================================================
# BLOCK 4 — Config dataclass for FTR-specific PPO training
# ============================================================

@dataclass
class FtrDiffusionConfig:
    """Config for the receding-horizon (action-chunking) PPO trainer.

    Identical to FtrPPOConfig except for the four horizon/discount fields below. Kept as a
    separate dataclass on purpose: pointing train_ftr.py at a diffusion config (or vice
    versa) must fail loudly at parse time rather than train something subtly different.
    """

    # --- Receding-horizon fields (see docs/diffusion_policy/01_action_chunking.md) ---
    # prediction_horizon (T_p): length of the action trajectory the policy emits.
    # execution_horizon (T_a): how many of those steps are executed before re-planning.
    # history_len (T_o): number of observation frames CatFrames stacks for the encoder.
    # control_gamma: the PER-CONTROL-STEP discount. gae_opts.gamma must be the MACRO
    #   discount control_gamma ** T_a, and env_cfg_overrides.shaping_gamma must stay on
    #   control_gamma — the config derives the former with ${pow:${control_gamma},${T_a}}.

    name: str
    comment: str
    seed: int
    device: str
    training_dtype: torch.dtype
    num_robots: int                    # number of parallel Isaac Sim envs
    task: str                          # gymnasium task ID
    terrain: str                       # FTR terrain name
    total_frames: int
    time_steps_per_batch: int          # steps collected per env per PPO iteration
    epochs_per_batch: int
    frames_per_sub_batch: int
    eval_and_save_every: int
    eval_repeats: int                   # rollouts averaged per mid-training eval checkpoint
    eval_repeats_after_training: int
    max_grad_norm: float
    clip_grad_norm_p: int | str
    optimizer: type
    optimizer_opts: dict[str, Any]
    scheduler: type
    scheduler_opts: dict[str, Any]
    gae_opts: dict[str, Any]
    ppo_opts: dict[str, Any]
    data_collector_opts: dict[str, Any]
    policy_config: type
    policy_opts: dict[str, Any]
    vecnorm_opts: dict[str, Any]
    vecnorm_on_reward: bool
    ftr_obs_encoder_opts: dict[str, Any]
    prediction_horizon: int
    execution_horizon: int
    history_len: int
    control_gamma: float
    # Stop an iteration's epoch loop early once the policy has drifted this far from the
    # rollout policy (mean kl_approx over an epoch). None = disabled.
    #
    # This exists because "how many updates per iteration" has been the dominant failure
    # mode of this trainer, and tuning epochs_per_batch by hand got it wrong twice: ep=5
    # diverged fast (clip_fraction 0.66 by iteration 6), and ep=3 looked stable for seven
    # iterations and then climbed steadily to 0.44. A fixed epoch count cannot adapt as the
    # policy sharpens — log-prob sensitivity grows as scale falls, so the same number of
    # updates moves the policy further later in training. target_kl bounds the drift
    # directly instead of proxying it through an epoch count.
    target_kl: float | None = None
    save_weights_every: int = 0  # 0 = same as eval_and_save_every
    max_eval_steps: int = 0  # 0 = auto: 2 × max_episode_length derived from sim_dt
    # Per-env-type breakdown of mid-training eval success rate — logged to wandb under the
    # "eval_per_env" category. Per-(env-type, depth-col) "spot" breakdown is written to a
    # local CSV only (eval_per_spot.csv in the run directory), never sent to wandb.
    # None = looked up from the terrain's registered layout (env_type_registry.py).
    eval_num_env_types: int | None = None
    eval_env_names_yaml: str | None = None
    use_wandb: bool = False
    use_tensorboard: bool = False
    policy_weights_path: str | None = None
    vecnorm_weights_path: str | None = None
    extra_env_transforms: list = field(default_factory=list)
    # Env config overrides applied via setattr before env creation
    env_cfg_overrides: dict = field(default_factory=dict)
    # Scheduler for step_penalty — anneals the per-step penalty over training.
    # type: 'linear' or 'exponential'. Keys: start_factor, end_factor, total_iters.
    # null = disabled (step_penalty stays constant).
    step_penalty_scheduler: dict | None = None
    # Scheduler for action_bonus_coef — anneals movement bonus over training.
    # type: 'linear' or 'exponential'. Keys: start_factor, end_factor, total_iters.
    # null = disabled (action_bonus_coef stays constant). Silently skipped if action_bonus_coef is null.
    action_bonus_coef_scheduler: dict | None = None
    # Physics tuning (applied to env_cfg.robot / env_cfg.sim before env creation)
    sim_dt: float = 1 / 400
    decimation: int = 5                  # physics steps per policy step; control_freq = 1 / (sim_dt * decimation)
    solver_position_iterations: int = 16
    solver_velocity_iterations: int = 4
    max_depenetration_velocity: float = 0.15
    bounce_threshold_velocity: float = 0.2
    robot_linear_damping: float = 0.05
    robot_angular_damping: float = 0.05
    robot_max_linear_velocity: float = 10.0
    robot_max_angular_velocity: float = 720.0
    physx_gpu_heap_capacity: int = 2**28
    physx_gpu_temp_buffer_capacity: int = 2**26
    physx_gpu_max_num_partitions: int = 8
    physx_gpu_found_lost_aggregate_pairs_capacity: int = 2**27  # 128M; increase if PhysX logs buffer overflow

    log_raw_accel: bool = False
    log_raw_accel_interval: int = 50   # flush every N reward steps; 0 = only at run end


# ============================================================
# BLOCK 4 — FtrDiffusionTrainer
# ============================================================

class FtrDiffusionTrainer:
    """PPO trainer for a receding-horizon policy: one RL step is T_a control steps.

    Structurally identical to FtrPPOTrainer, with three differences that all follow from
    action chunking:

    1. The env is wrapped in ActionChunkEnv, so ``action`` is a flattened T_p x action_dim
       chunk and ``reward`` is the discounted sum over the executed prefix.
    2. A CatFrames transform (after VecNorm) writes the T_o observation window to
       ``obs_history``, which is what the policy reads. It must be a stored tensordict key,
       not a per-env ring buffer, because PPO trains on shuffled minibatches.
    3. ``total_frames`` keeps counting CONTROL steps while the collector and replay buffer
       count macro-step TRANSITIONS; the two differ by a factor of T_a. Keeping
       ``total_frames`` in control steps is what lets slurm/lib/respawn_common.sh budget a
       respawn correctly and keeps the frame axis comparable with the marv_rl baseline.
    """

    def __init__(self, raw_config: "DictConfig", ftr_gym_env: gymnasium.Env, optuna_trial=None):
        self.optuna_trial = optuna_trial
        self.config = FtrDiffusionConfig(**raw_config)
        self.device = set_device(self.config.device)
        self.rng = seed_all(self.config.seed)

        self.run_logger = RunLogger(
            train_config=raw_config,  # RunLogger expects DictConfig
            category="diffusion",
            use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard,
            step_metric_name="collected_frames",
        )
        self._eval_num_env_types = self.config.eval_num_env_types or default_num_env_types(self.config.terrain)
        self._eval_env_type_names = load_env_type_names(self.config.terrain, self.config.eval_env_names_yaml, self._eval_num_env_types)
        self._eval_per_spot_csv = self.run_logger.logpath / "eval_per_spot.csv"
        if self.optuna_trial is not None:
            self.optuna_trial.set_user_attr("logpath", str(self.run_logger.logpath))
        self.term_logger = get_terminal_logger("ftr_ppo_train")

        if self.config.log_raw_accel:
            ftr_gym_env.unwrapped.cfg.log_raw_accel_path = str(self.run_logger.logpath / "raw_accel.npz")
        self.term_logger.info(
            f"Seed: {self.config.seed}  "
            f"(random ✓  numpy ✓  torch ✓  cuda ✓)"
        )

        # ---- environment ----
        # T_a is baked into the wrapper, so from here down one "step" is one macro step.
        # self.ftr_torchrl_env is the CHUNKED env: run_tracked_rollout and the eval path
        # step it, and ActionChunkEnv forwards every accumulator call to the inner env, so
        # all rew/* and state/* numbers keep their per-control-step meaning.
        self.inner_env = FtrTorchRLEnv(
            ftr_gym_env,
            encoder_opts=self.config.ftr_obs_encoder_opts,
            device=self.device,
            shock_scale=self.config.env_cfg_overrides.get("shock_scale"),
        )
        self.T_a = self.config.execution_horizon
        self.ftr_torchrl_env = ActionChunkEnv(
            self.inner_env,
            prediction_horizon=self.config.prediction_horizon,
            execution_horizon=self.T_a,
            control_gamma=self.config.control_gamma,
        )
        self.env = self.ftr_torchrl_env
        _macro_gamma = self.config.gae_opts.get("gamma")
        _expected = self.config.control_gamma ** self.T_a
        if _macro_gamma is not None and abs(_macro_gamma - _expected) > 1e-9:
            get_terminal_logger("ftr_diffusion_train").warning(
                "gae_opts.gamma (%.10f) != control_gamma ** execution_horizon (%.10f). "
                "GAE discounts macro steps, so unless this is deliberate the config should "
                "use gamma: ${pow:${control_gamma},${execution_horizon}}.",
                _macro_gamma, _expected,
            )

        # ---- policy ----
        policy_cfg = self.config.policy_config(**self.config.policy_opts)
        self.actor_value_wrapper, self.optim_groups, policy_transforms = policy_cfg.create(
            env=self.env,
            weights_path=self.config.policy_weights_path,
            device=self.device,
        )
        self.actor_operator = self.actor_value_wrapper.get_policy_operator()
        self.value_operator = self.actor_value_wrapper.get_value_operator()
        if self.config.policy_weights_path is not None:
            self.term_logger.info(f"Loaded policy weights from {self.config.policy_weights_path}")

        # ---- transforms + VecNorm ----
        # CatFrames writes the T_o observation window to "obs_history" AFTER VecNorm, so
        # the stacked frames are already normalised. padding="same" repeats the first frame
        # at episode start, matching the paper and the BC dataset builder.
        obs_history_transform = CatFrames(
            N=self.config.history_len,
            dim=-1,
            in_keys=[OBS_KEY],
            out_keys=["obs_history"],
            padding="same",
        )
        self.env, self.vecnorm = make_transformed_env(
            self.ftr_torchrl_env,
            self.config,
            policy_transforms,
            post_vecnorm_transforms=[obs_history_transform],
        )
        if self.config.vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(
                torch.load(self.config.vecnorm_weights_path, map_location=self.device), strict=False
            )
            self.term_logger.info(f"Loaded vecnorm weights from {self.config.vecnorm_weights_path}")

        # ---- auto-resume from crash checkpoint ----
        self.crash_checkpoint = None
        self._check_and_load_crash_checkpoint()

        # ---- data collection ----
        # The collector counts macro-step transitions; self.config.total_frames counts
        # control steps. They differ by T_a — see the class docstring.
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        self.collector = SyncDataCollector(
            self.env,
            self.actor_operator,
            frames_per_batch=iteration_size,
            total_frames=self.config.total_frames // self.T_a,
            **self.config.data_collector_opts,
            device=self.device,
        )
        # Replay buffer lives on CPU to avoid CUDA memory pressure competing with PhysX
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=iteration_size, ndim=1, device="cpu"),
            sampler=SamplerWithoutReplacement(drop_last=True, shuffle=True),
            batch_size=self.config.frames_per_sub_batch,
            dim_extend=0,
        )

        # ---- GAE + PPO loss ----
        self.advantage_module = GAE(
            **self.config.gae_opts,
            value_network=self.value_operator,
            time_dim=1,
            device=self.device,
            differentiable=False,
        )
        self.advantage_module = self.advantage_module.to(self.config.training_dtype)
        _critic_for_loss = (
            self.actor_value_wrapper.get_value_head()
            if isinstance(self.actor_value_wrapper, ActorValueOperator)
            else self.value_operator
        )
        if isinstance(self.actor_operator, DiffusionChunkActor):
            # Phase 2. ClipPPOLoss cannot score a diffusion actor: it recomputes log-probs
            # via actor.get_dist(td).log_prob(action), and the quantity to score is the
            # stored denoising chain, not the final action. make_dppo_loss also refuses
            # entropy_bonus=True, which is fatal-but-silent here (no tractable entropy).
            self.loss_module = make_dppo_loss(
                policy_cfg, self.actor_operator, _critic_for_loss, **self.config.ppo_opts
            )
            self.term_logger.info(f"Phase 2: using {type(self.loss_module).__name__}")
        else:
            self.loss_module = ClipPPOLoss(self.actor_operator, _critic_for_loss, **self.config.ppo_opts)
        self.loss_module = self.loss_module.to(self.config.training_dtype)

        # ---- optimizer + scheduler ----
        self.optim = self.config.optimizer(self.optim_groups, **(self.config.optimizer_opts or {}))
        self.scheduler = self.config.scheduler(self.optim, **(self.config.scheduler_opts or {}))

        _total_iters = self.config.total_frames // (self.config.time_steps_per_batch * self.config.num_robots * self.T_a)

        # ---- step_penalty scheduler ----
        _sp_init = self.config.env_cfg_overrides.get("step_penalty")
        _sp_sched = self.config.step_penalty_scheduler
        if _sp_sched is not None and _sp_init is not None:
            self.step_penalty_scheduler = _make_cfg_scheduler(
                self.ftr_torchrl_env.ftr_env.unwrapped.cfg,
                "step_penalty",
                _sp_init,
                _sp_sched,
                _total_iters,
            )
        else:
            self.step_penalty_scheduler = None

        # ---- action_bonus_coef scheduler ----
        _abc_init = self.config.env_cfg_overrides.get("action_bonus_coef")
        _abc_sched = self.config.action_bonus_coef_scheduler
        if _abc_sched is not None and _abc_init is not None:
            self.action_bonus_scheduler = _make_cfg_scheduler(
                self.ftr_torchrl_env.ftr_env.unwrapped.cfg,
                "action_bonus_coef",
                _abc_init,
                _abc_sched,
                _total_iters,
            )
        else:
            self.action_bonus_scheduler = None

        # ---- resume optimizer / schedulers / counters (needs all of the above to exist) ----
        self._restore_training_state()

        self.term_logger.info("Initialized FtrDiffusionTrainer.")

    def _check_and_load_crash_checkpoint(self):
        """Check for crash checkpoints or latest step checkpoint and auto-resume.

        Searches the current attempt's weights dir first, then previous attempt dirs
        in reverse order, so a respawned run automatically picks up the best available
        checkpoint from any earlier attempt in the same SLURM job.
        """
        candidate_dirs = self.run_logger.candidate_weight_dirs()

        policy_to_load = None
        vecnorm_to_load = None
        checkpoint_source = None

        for weights_dir in candidate_dirs:
            # Priority 1: crash checkpoint (saved on unhandled exception)
            policy_crash = weights_dir / "policy_crash.pth"
            vecnorm_crash = weights_dir / "vecnorm_crash.pth"
            if policy_crash.exists() and vecnorm_crash.exists():
                policy_to_load = policy_crash
                vecnorm_to_load = vecnorm_crash
                checkpoint_source = f"crash ({weights_dir.parent.name})"
                break

            # Priority 2: latest periodic step checkpoint
            step_policies = sorted(weights_dir.glob("policy_step_*.pth"),
                                   key=lambda p: int(p.stem.split("_")[-1]))
            step_vecnorms = sorted(weights_dir.glob("vecnorm_step_*.pth"),
                                   key=lambda p: int(p.stem.split("_")[-1]))
            if step_policies and step_vecnorms:
                policy_to_load = step_policies[-1]
                vecnorm_to_load = step_vecnorms[-1]
                checkpoint_source = f"step ({weights_dir.parent.name})"
                break

        if policy_to_load and vecnorm_to_load:
            self.term_logger.warning(
                f"Found {checkpoint_source} checkpoint: {policy_to_load.name} / {vecnorm_to_load.name}"
            )
            try:
                policy_state = torch.load(policy_to_load, map_location=self.device)
                self.actor_value_wrapper.load_state_dict(policy_state, strict=False)
                self.term_logger.info(f"Loaded policy from {checkpoint_source} checkpoint")
            except Exception as e:
                self.term_logger.error(f"Failed to load policy: {e}")

            try:
                vecnorm_state = torch.load(vecnorm_to_load, map_location=self.device)
                self.vecnorm.load_state_dict(vecnorm_state, strict=False)
                self.term_logger.info(f"Loaded vecnorm from {checkpoint_source} checkpoint")
            except Exception as e:
                self.term_logger.error(f"Failed to load vecnorm: {e}")

            self.crash_checkpoint = True
            self.term_logger.warning(
                f"Resuming from {checkpoint_source} checkpoint — training may have inconsistent metrics."
            )
        # The optimizer and the schedulers do not exist yet at this point in __init__, so the rest of
        # the training state is restored later, from _restore_training_state().
        self.resume_iteration = None
        self.resume_frames = None

    def _restore_training_state(self):
        """Restore optimizer + scheduler state and the frame/iteration counters.

        Must run *after* the optimizer, the LR scheduler and the config schedulers have been built:
        `_load_training_checkpoint` writes into all of them. Calling it from
        `_check_and_load_crash_checkpoint` (where the policy weights are loaded) is too early — it
        raised ``object has no attribute 'optim'`` (see train_ftr.py, where this was found), which the broad except in
        `_load_training_checkpoint` turned into a warning, so every respawn silently restarted the LR
        and step-penalty schedules from their initial values and reset the frame counter to 0.
        """
        if not self.crash_checkpoint:
            return
        resume_iteration, resume_frames = self._load_training_checkpoint()
        if resume_iteration is not None:
            self.resume_iteration = resume_iteration
            self.resume_frames = resume_frames

    def _save_training_checkpoint(self, iteration: int, total_collected_frames: int):
        """Save training state (optimizers, schedulers, frame count) for resuming."""
        checkpoint = {
            "iteration": iteration,
            "total_collected_frames": total_collected_frames,
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.step_penalty_scheduler is not None:
            checkpoint["step_penalty_scheduler_state_dict"] = self.step_penalty_scheduler.state_dict()
        if self.action_bonus_scheduler is not None:
            checkpoint["action_bonus_scheduler_state_dict"] = self.action_bonus_scheduler.state_dict()
        self.run_logger.save_weights(checkpoint, "training_state")

    def _load_training_checkpoint(self):
        """Load training state and return (iteration, total_collected_frames) to resume from.

        Searches candidate dirs (current attempt, then previous attempts) for training_state.pth.
        """
        training_state_file = None
        for weights_dir in self.run_logger.candidate_weight_dirs():
            candidate = weights_dir / "training_state.pth"
            if candidate.exists():
                training_state_file = candidate
                break
        if training_state_file is None:
            return None, None
        try:
            checkpoint = torch.load(training_state_file, map_location=self.device)
            # Only restore optimizer state if it's non-empty (may be reconstructed from CSV with empty state)
            if checkpoint.get("optimizer_state_dict"):
                try:
                    self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
                except (KeyError, RuntimeError) as e:
                    self.term_logger.warning(
                        f"Failed to load optimizer state (may be OK if reconstructed from CSV): {e}. "
                        f"Optimizer will restart with fresh momentum buffers."
                    )
            else:
                self.term_logger.warning(
                    "Optimizer state is empty (training_state.pth reconstructed from CSV). "
                    "Optimizer momentum will be reset; training may be noisy for first few iterations."
                )
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "step_penalty_scheduler_state_dict" in checkpoint and self.step_penalty_scheduler is not None:
                self.step_penalty_scheduler.load_state_dict(checkpoint["step_penalty_scheduler_state_dict"])
            if "action_bonus_scheduler_state_dict" in checkpoint and self.action_bonus_scheduler is not None:
                self.action_bonus_scheduler.load_state_dict(checkpoint["action_bonus_scheduler_state_dict"])
            self.term_logger.info(
                f"Loaded training checkpoint: resuming from iteration {checkpoint['iteration']}, "
                f"total_collected_frames={checkpoint['total_collected_frames']}"
            )
            return checkpoint["iteration"], checkpoint["total_collected_frames"]
        except Exception as e:
            self.term_logger.warning(f"Failed to load training checkpoint: {e}")
            return None, None

    # ------------------------------------------------------------------
    def train(self):
        try:
            self._train()
            post_log = self._post_training_evaluation()
        except KeyboardInterrupt:
            self.term_logger.info("Training interrupted by user.")
            post_log = None
        except Exception as e:
            self.term_logger.error(f"Training failed: {e}")
            traceback.print_exception(e)
            if "CUDA error" in str(e) or "CUDA out of memory" in str(e) or "CommError" in str(type(e).__name__):
                # CUDA context is dead — saving weights and running atexit/Isaac Sim
                # cleanup handlers will deadlock. Exit immediately so the SLURM slot
                # is freed and the job can be requeued rather than hanging for hours.
                # Also exit on W&B CommError (TLS cert issues) to allow respawn.
                self.term_logger.error(f"{type(e).__name__} detected — calling os._exit(75) to skip cleanup.")
                import os as _os
                _os._exit(75)
            try:
                self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), "policy_crash")
                self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_crash")
                # Save training state so the next attempt restores the correct LR/optimizer position.
                _crash_iter = getattr(self, "_current_iteration", 0)
                _crash_frames = getattr(self, "_current_total_frames", 0)
                self._save_training_checkpoint(_crash_iter, _crash_frames)
                self.term_logger.info(f"Saved crash checkpoint at iter={_crash_iter}, frames={_crash_frames}.")
                if self.config.use_wandb:
                    self.term_logger.info(
                        f"W&B run {self.run_logger.wandb_run_id} will resume when the job restarts."
                    )
            except Exception:
                pass
            raise
        finally:
            if self.run_logger is not None:
                self.run_logger.close()
        return post_log

    def _train(self):
        # Two different units per iteration, deliberately kept apart:
        #   iteration_size    — macro-step transitions; what PPO optimises over.
        #   frames_per_iter   — control steps; what total_frames and the checkpoint names
        #                       count, so the SLURM respawn helper budgets correctly.
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        frames_per_iter = iteration_size * self.T_a
        if iteration_size % self.config.frames_per_sub_batch != 0:
            raise ValueError(
                f"iteration_size ({iteration_size}) must be divisible by frames_per_sub_batch "
                f"({self.config.frames_per_sub_batch})"
            )
        self.term_logger.info(
            f"Iteration size: {iteration_size}, frames_per_sub_batch: {self.config.frames_per_sub_batch}, "
            f"optimisation steps per iter: {iteration_size // self.config.frames_per_sub_batch}"
        )
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        # Initialize progress bar and resume position.
        # iter_offset shifts i so that total_collected_frames and periodic-save
        # cadence are correct without wasting GPU cycles re-running old rollouts.
        resume_iteration = getattr(self, "resume_iteration", None)
        resume_frames = getattr(self, "resume_frames", None)
        iter_offset = resume_iteration if resume_iteration is not None else 0
        # On a respawn the counter carries on from the checkpoint while the collector only collects
        # what is left, so the bar spans resumed + remaining rather than the config's total_frames.
        _pbar_total = self.config.total_frames + (resume_frames or 0)
        pbar = tqdm(total=_pbar_total, desc="FTR receding-horizon training", unit="frames", leave=False)
        if resume_frames is not None:
            pbar.update(resume_frames)
            self.term_logger.info(f"Resuming training — progress bar starting at {resume_frames} frames (iter_offset={iter_offset})")

        for i, tensordict_data in enumerate(self.collector):
            effective_i = i + iter_offset
            total_collected_frames = (effective_i + 1) * frames_per_iter
            # Track for crash checkpoint
            self._current_iteration = effective_i
            self._current_total_frames = total_collected_frames
            pbar.update(frames_per_iter)

            # FTR env does not produce "curr_state"; pop safely.
            tensordict_data.pop("curr_state", None)
            tensordict_data.pop(("next", "curr_state"), None)

            self.actor_operator.train()
            self.value_operator.train()
            self.env.train()

            # Before GAE: remove entire env trajectories that contain any explosion step.
            # An explosion teleports the robot to a "safe" origin; the step immediately
            # before the explosion uses V(safe_state) as the GAE bootstrap, which
            # corrupts advantages for the whole preceding segment.  Filtering at the
            # trajectory level (entire row in [num_envs, time]) removes both the
            # explosion step AND the contaminated pre-explosion bootstrap in one shot.
            n_dirty = 0
            explosion_flags = tensordict_data.get(("next", "explosion"), None)
            if explosion_flags is not None:
                has_explosion = explosion_flags.squeeze(-1).any(dim=1)  # [num_envs]
                n_dirty = int(has_explosion.sum().item())
                if n_dirty > 0:
                    tensordict_data = tensordict_data[~has_explosion]
                    self.term_logger.info(
                        f"Excluded {n_dirty}/{self.config.num_robots} env trajectories with explosions from GAE"
                        f" ({self.config.num_robots - n_dirty} clean)"
                    )

            if tensordict_data.batch_size[0] == 0:
                # Every env exploded this iteration — nothing clean to train on.
                self.term_logger.warning("All env trajectories had explosions — skipping update this iteration.")
                self.scheduler.step()
                log = {
                    **self.ftr_torchrl_env.pop_reward_info(),
                    **self.ftr_torchrl_env.pop_termination_info(),
                    **self.ftr_torchrl_env.pop_state_stats(),
                }
                self.run_logger.log_data(log, total_collected_frames)
                continue

            # GAE computed in chunks to avoid CUDA OOM: the CNN encoder flattens to
            # [N*T, 1, 45, 21] which exhausts memory at full batch (512*128=65536).
            # Chunk across the env axis so each pass is ~frames_per_sub_batch frames.
            # NOTE: TorchRL slice views propagate writes to existing keys but NOT new
            # key insertions, so we cat the GAE-produced keys back into the parent.
            _gae_chunk = max(1, self.config.frames_per_sub_batch // self.config.time_steps_per_batch)
            _n_clean = tensordict_data.batch_size[0]
            _chunks = []
            for _c in range(0, _n_clean, _gae_chunk):
                _chunk = tensordict_data[_c:_c + _gae_chunk]
                self.advantage_module(_chunk)
                _chunks.append(_chunk)
            for _key in ("advantage", "value_target", "state_value"):
                if _key in _chunks[0].keys():
                    tensordict_data[_key] = torch.cat([_ch[_key] for _ch in _chunks], dim=0)

            # Sanitize any remaining NaN/Inf (should be rare after trajectory filter).
            nan_count = 0
            nan_keys = []
            for key in tensordict_data.keys(include_nested=True, leaves_only=True):
                t = tensordict_data[key]
                if t.is_floating_point():
                    bad = ~t.isfinite()
                    if bad.any():
                        nan_count += int(bad.sum().item())
                        nan_keys.append(f"{key}({bad.sum().item()})")
                        tensordict_data[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if nan_count > 0:
                self.term_logger.warning(
                    f"Sanitized {nan_count} non-finite values in rollout tensordict: {', '.join(nan_keys)}"
                )

            flat = tensordict_data.reshape(-1)
            n_valid = flat.shape[0]

            # Reward variance decomposition & gradient contribution (diagnostic).
            # Must be called BEFORE del flat and BEFORE pop_reward_series clears the accum.
            _reward_series = self.ftr_torchrl_env.peek_reward_series()
            reward_grad_stats = (
                _compute_reward_grad_stats(flat, _reward_series, self.config.time_steps_per_batch, self.T_a)
                if _reward_series else {}
            )

            rollout_log_prob = flat["sample_log_prob"].mean().item()
            rollout_adv_mean = flat["advantage"].mean().item()
            rollout_adv_std = flat["advantage"].std().item()
            rollout_mean_reward = flat[("next", "reward")].mean().item()
            rollout_mean_state_value = flat["state_value"].mean().item()
            rollout_value_minus_reward = rollout_mean_state_value - rollout_mean_reward

            # action is the FLATTENED chunk; recover [N, T_p, A] and report statistics over
            # the executed prefix only — the discarded tail never reaches the robot.
            action_dim = self.ftr_torchrl_env.action_dim
            chunk = flat["action"].reshape(-1, self.config.prediction_horizon, action_dim)
            executed = chunk[:, : self.T_a]
            action_log = {
                "action/v_mean":  executed[..., 0].mean().item(),
                "action/v_std":   executed[..., 0].std().item(),
                "action/w_mean":  executed[..., 1].mean().item(),
                "action/w_std":   executed[..., 1].std().item(),
            }
            flipper_num = self.ftr_torchrl_env.ftr_env.unwrapped.flipper_num
            flipper_names = ["fl", "fr", "rl", "rr"] if flipper_num == 4 else ["front", "rear"]
            for fi, fname in enumerate(flipper_names):
                action_log[f"action/flipper_{fname}_mean"] = executed[..., 2 + fi].mean().item()
                action_log[f"action/flipper_{fname}_std"]  = executed[..., 2 + fi].std().item()
            # Within-chunk temporal smoothness — the quantity action chunking exists to
            # improve. Measured over the whole predicted chunk, and over the flipper block
            # separately since that is where snapping would show up first under position
            # control (which removes the env's 5 deg/step slew limit).
            if self.config.prediction_horizon > 1:
                _delta = (chunk[:, 1:] - chunk[:, :-1]).abs()
                action_log["action/chunk_step_delta"] = _delta.mean().item()
                action_log["action/chunk_step_delta_flipper"] = _delta[..., 2:].mean().item()
                action_log["action/chunk_step_delta_max"] = _delta.max().item()
            # ⚠ The three metrics above are computed on SAMPLED actions, so while the policy's
            # scale is still ~1 they measure sampling noise, not the predicted trajectory: for
            # near-independent actions on [-1,1] the expected step delta is 2/3, and the first
            # shakedown duly sat at 0.7239 -> 0.7224 across ten iterations no matter what the
            # network was doing. The mean chunk (loc) is the actual predicted trajectory, and
            # scale_mean says whether the policy is concentrating at all — log both, or a flat
            # chunk_step_delta is uninterpretable.
            if "loc" in flat.keys():
                _loc = flat["loc"].reshape(-1, self.config.prediction_horizon, action_dim)
                if self.config.prediction_horizon > 1:
                    _ld = (_loc[:, 1:] - _loc[:, :-1]).abs()
                    action_log["action/loc_step_delta"] = _ld.mean().item()
                    action_log["action/loc_step_delta_flipper"] = _ld[..., 2:].mean().item()
                action_log["action/loc_abs_mean"] = _loc.abs().mean().item()
            if "scale" in flat.keys():
                action_log["action/scale_mean"] = flat["scale"].mean().item()
            # Phase 2 health. C-TRAC's lesson: a collapsed latent looked fine on the success
            # curve for 22M frames. The analogous silent failure here is a denoising chain
            # that has stopped doing anything — if the late chain_delta_k are ~0 the chain is
            # inert and K_infer is wasted compute; if the early ones are ~0 the schedule is
            # wrong. Logged from the first iteration, not added after something looks off.
            if "denoise_chain" in flat.keys():
                _ch = flat["denoise_chain"]                      # [N, K+1, A, T_p]
                _cd = (_ch[:, 1:] - _ch[:, :-1]).abs().mean(dim=(0, 2, 3))
                for _i, _v in enumerate(_cd.tolist()):
                    action_log[f"diff/chain_delta_k{_i}"] = _v
                action_log["diff/chain_delta_total"] = _cd.sum().item()
            if "sample_log_prob" in flat.keys():
                action_log["diff/logprob_std"] = flat["sample_log_prob"].std().item()
                action_log["diff/logprob_mean"] = flat["sample_log_prob"].mean().item()
            # TanhNormal conditioning. With scale=1 in PRE-tanh space the sampled actions
            # pile up against +-1, where recovering the pre-tanh value (atanh) explodes and
            # log-prob becomes hypersensitive: ClipPPOLoss re-evaluates the STORED action
            # under the updated policy, so a saturated sample can move the ratio far out of
            # the clip band even when loc and scale have barely changed. That is exactly the
            # contradiction seen on attempt 2 (scale 1.0000, loc 0.02, clip_fraction 0.345).
            # sat_frac is the direct measurement; if it is large, lower the initial scale
            # rather than the learning rate.
            _a = flat["action"]
            action_log["action/sat_frac_099"] = (_a.abs() > 0.99).float().mean().item()
            action_log["action/sat_frac_09"] = (_a.abs() > 0.9).float().mean().item()
            action_log["action/abs_mean"] = _a.abs().mean().item()

            self.replay_buffer.extend(flat)
            del tensordict_data, flat
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            n_subbatches = max(1, n_valid // self.config.frames_per_sub_batch)
            skipped_updates = 0
            # Accumulate the loss diagnostics across EVERY sub-batch. Taking them from the
            # loop variable afterwards (what train_ftr.py does) reports only the last
            # sub-batch of the last epoch while calling it "mean_*", which hides how the
            # policy drifted across the 160 updates inside one iteration — exactly the
            # quantity clip_fraction exists to measure.
            _diag_sum: dict[str, float] = {}
            _diag_n = 0
            _epochs_run = 0
            _stopped_on_kl = False
            for _j in range(self.config.epochs_per_batch):
                _epoch_kl, _epoch_kl_n = 0.0, 0
                for _k in range(n_subbatches):
                    sub_batch = self.replay_buffer.sample().to(self.device)
                    loss_vals = self.loss_module(sub_batch)
                    for _dk in ("clip_fraction", "kl_approx", "entropy", "loss_objective", "loss_critic"):
                        if _dk in loss_vals.keys():
                            _diag_sum[_dk] = _diag_sum.get(_dk, 0.0) + loss_vals[_dk].mean().item()
                    _diag_n += 1
                    if "kl_approx" in loss_vals.keys():
                        _epoch_kl += float(loss_vals["kl_approx"].mean().item())
                        _epoch_kl_n += 1
                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
                    if "loss_entropy" in loss_vals.keys():
                        loss_value = loss_value + loss_vals["loss_entropy"]
                    loss_value.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor_value_wrapper.parameters(),
                        self.config.max_grad_norm,
                        error_if_nonfinite=False,
                        norm_type=self.config.clip_grad_norm_p,
                    )
                    if not grad_norm.isfinite():
                        self.term_logger.warning(f"Skipping update: non-finite grad norm ({grad_norm:.4g})")
                        self.optim.zero_grad()
                        skipped_updates += 1
                    else:
                        self.optim.step()
                        self.optim.zero_grad()
                _epochs_run += 1
                # Checked per epoch, not per sub-batch: a single minibatch's kl_approx is
                # noisy enough to trip constantly, and the quantity we care about is the
                # drift accumulated over a pass, not the excursion of one batch.
                if self.config.target_kl is not None and _epoch_kl_n:
                    if (_epoch_kl / _epoch_kl_n) > self.config.target_kl:
                        _stopped_on_kl = True
                        break

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            self.scheduler.step()
            if self.step_penalty_scheduler is not None:
                self.step_penalty_scheduler.step()
            _sp_current = self.step_penalty_scheduler.current_value if self.step_penalty_scheduler is not None else self.config.env_cfg_overrides.get("step_penalty")
            if self.action_bonus_scheduler is not None:
                self.action_bonus_scheduler.step()
            _abc_current = self.action_bonus_scheduler.current_value if self.action_bonus_scheduler is not None else self.config.env_cfg_overrides.get("action_bonus_coef")

            log = {
                **action_log,
                **reward_grad_stats,
                **self.ftr_torchrl_env.pop_reward_info(),
                **self.ftr_torchrl_env.pop_termination_info(),
                **self.ftr_torchrl_env.pop_state_stats(),
                "train/mean_reward": rollout_mean_reward,
                "train/mean_state_value": rollout_mean_state_value,
                "train/value_minus_reward": rollout_value_minus_reward,
                "train/mean_advantage_GAE": rollout_adv_mean,
                "train/mean_action_sample_log_prob": rollout_log_prob,
                "train/mean_critic_loss": _diag_sum.get("loss_critic", 0.0) / max(1, _diag_n),
                "train/mean_objective_loss": _diag_sum.get("loss_objective", 0.0) / max(1, _diag_n),
                "train/mean_entropy_loss": (
                    loss_vals["loss_entropy"].mean().item() if "loss_entropy" in loss_vals.keys() else 0.0
                ),
                "train/mean_entropy": _diag_sum.get("entropy", 0.0) / max(1, _diag_n),
                "train/mean_kl_approx": _diag_sum.get("kl_approx", 0.0) / max(1, _diag_n),
                "train/mean_clip_fraction": _diag_sum.get("clip_fraction", 0.0) / max(1, _diag_n),
                # Last sub-batch only — the END-of-iteration drift, vs the mean above.
                "train/final_clip_fraction": loss_vals["clip_fraction"].mean().item(),
                "train/final_kl_approx": loss_vals["kl_approx"].mean().item(),
                "train/mean_advantage": rollout_adv_mean,
                "train/std_advantage": rollout_adv_std,
                "train/total_grad_norm": grad_norm.item(),
                **{f"train/{g['name']}_lr": g["lr"] for g in self.optim.param_groups},
                "train/step_penalty": _sp_current if _sp_current is not None else 0.0,
                "train/action_bonus_coef": _abc_current if _abc_current is not None else 0.0,
                "train/epochs_run": _epochs_run,
                "train/stopped_on_kl": float(_stopped_on_kl),
                "explosions/dirty_envs": n_dirty,
                "explosions/skipped_updates": skipped_updates,
                "explosions/skipped_updates_frac": skipped_updates / (n_subbatches * self.config.epochs_per_batch),
            }

            save_every = self.config.save_weights_every or self.config.eval_and_save_every
            if effective_i % save_every == 0:
                self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{total_collected_frames}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{total_collected_frames}")
                self._save_training_checkpoint(effective_i, total_collected_frames)
            if effective_i % self.config.eval_and_save_every == 0 and effective_i > 0:
                try:
                    eval_log = self._get_eval_rollout_results()
                    for _ in range(self.config.eval_repeats - 1):
                        for k, v in self._get_eval_rollout_results().items():
                            eval_log[k] += v
                    if self.config.eval_repeats > 1:
                        for k in eval_log:
                            eval_log[k] /= self.config.eval_repeats
                    log.update(eval_log)

                    if self.optuna_trial is not None:
                        eval_step = i // self.config.eval_and_save_every
                        success_rate = eval_log.get("eval/success_rate", 0.0)
                        self.optuna_trial.report(success_rate, eval_step)

                        if self.optuna_trial.should_prune():
                            self.term_logger.info(
                                f"Trial pruned at iteration {i} "
                                f"(eval_step={eval_step}, success_rate={success_rate:.3f})"
                            )
                            raise optuna.TrialPruned()
                except optuna.TrialPruned:
                    raise  # must propagate — don't catch in RuntimeError handler
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.term_logger.warning(f"Eval CUDA error — GPU context corrupted. Exiting immediately to avoid cleanup hang.")
                        import os as _os
                        _os._exit(75)
                    self.term_logger.warning(f"Eval rollout failed (physics explosion): {e}. Skipping eval metrics this checkpoint.")

            self.run_logger.log_data(log, total_collected_frames)

        # Name the final checkpoint by the CUMULATIVE frame count, not config.total_frames.
        # They are the same on a fresh run, but on a cross-job resume config.total_frames is
        # only the ADDITIONAL frames requested — so the final checkpoint would be misnamed
        # and could collide with one from the job being resumed, corrupting the frame
        # accounting (attempt_record_progress reads these names) on any further resume.
        _final_frames = getattr(self, "_current_total_frames", None) or self.config.total_frames
        _final_iter = getattr(self, "_current_iteration", 0)
        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), "policy_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")
        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{_final_frames}")
        self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{_final_frames}")
        # Save the matching training state too. Without this the run ends with weights one
        # save-interval ahead of the optimizer/scheduler state that training_state.pth holds,
        # so a resume restores mismatched pairs — measured on the resume test: weights from
        # frame 32768 but schedules from 28672, silently losing two iterations of schedule
        # progress on every resume.
        self._save_training_checkpoint(_final_iter, _final_frames)

    def _get_eval_rollout_results(self) -> dict[str, float]:
        self.env.eval()
        self.actor_operator.eval()
        # max_episode_length is in control steps; the rollout loop counts macro steps.
        # eval/rollout_steps in the logs is therefore also in macro steps.
        max_eval_steps = self.config.max_eval_steps or max(
            1, (self.ftr_torchrl_env.ftr_env.unwrapped.max_episode_length * 2) // self.T_a
        )
        self.ftr_torchrl_env.enable_per_env_tracking()
        results, episode_records = run_tracked_rollout(
            self.env, self.ftr_torchrl_env, self.ftr_torchrl_env.ftr_env, self.actor_operator, max_eval_steps,
            num_env_types=self._eval_num_env_types,
            env_type_names=self._eval_env_type_names,
            terrain=self.config.terrain,
        )
        self.ftr_torchrl_env.disable_per_env_tracking()
        _eval_reward_info = {k: results.pop(k) for k in list(results) if k.startswith("rew/") or k.startswith("shock/")}
        _shock_keys = (
            "shock/accel_magnitude", "shock/shock_normalised",
            "shock/accel_p90", "shock/accel_p95", "shock/accel_p99",
            "shock/shock_p90_normalised", "shock/shock_p95_normalised", "shock/shock_p99_normalised",
        )
        for _shock_key in _shock_keys:
            if _shock_key in _eval_reward_info:
                results[f"eval/{_shock_key.split('/', 1)[1]}"] = _eval_reward_info[_shock_key]

        if episode_records:
            per_env_rows = aggregate_per_env(
                episode_records=episode_records, env_type_names=self._eval_env_type_names,
                eval_id="train", policy=self.run_logger.run_name, terrain=self.config.terrain,
                repeat=1, obs_stats=results,
            )
            for row in per_env_rows:
                results[f"eval_per_env/{row.env_type_name}_success_rate"] = row.success_rate

            per_spot_rows = aggregate_per_spot(
                episode_records=episode_records, env_type_names=self._eval_env_type_names,
                num_depth_cols=10, eval_id="train", policy=self.run_logger.run_name,
                terrain=self.config.terrain, repeat=1,
            )
            save_per_spot_csv(self._eval_per_spot_csv, per_spot_rows)
        return results

    def _post_training_evaluation(self) -> dict[str, float]:
        self.term_logger.info(f"Training finished. Running {self.config.eval_repeats_after_training} final eval(s).")
        avg = self._get_eval_rollout_results()
        for _ in range(self.config.eval_repeats_after_training - 1):
            for k, v in self._get_eval_rollout_results().items():
                avg[k] += v
        for k in avg:
            avg[k] /= self.config.eval_repeats_after_training
        print("\nFinal evaluation results:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
        return avg


# ============================================================
# BLOCK 5 — Entry point
# ============================================================

def _load_raw_config(config_path: str, cli_overrides: list[str]):
    """Load and merge config YAML, returning a raw OmegaConf DictConfig."""
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


if __name__ == "__main__":
    # --play mode: load weights from a finished run directory and visualise
    if args.play is not None:
        play_dir = Path(args.play)
        # Load the config that was saved alongside that run
        saved_cfg_path = play_dir / "config.yaml"
        if not saved_cfg_path.exists():
            raise FileNotFoundError(f"No config.yaml found in {play_dir}")
        raw_cfg = _load_raw_config(str(saved_cfg_path), unknown_args)
        # Override weights paths to point at the saved weights
        weights_dir = play_dir / "weights"
        raw_cfg.policy_weights_path = str(weights_dir / "policy_final.pth")
        raw_cfg.vecnorm_weights_path = str(weights_dir / "vecnorm_final.pth")
        # Sensible play defaults: few envs, no logging
        raw_cfg.use_wandb = False
        raw_cfg.use_tensorboard = False
    else:
        # On SLURM respawn, prefer the config saved by the previous attempt so that
        # reward weights, LR schedule, and all other hyperparameters are identical
        # to what the checkpoint was trained with. Fall back to args.config only when
        # no previous attempt exists (i.e. this is the first run of the job).
        prev_cfg_path = RunLogger.latest_attempt_config()
        if prev_cfg_path is not None:
            print(f"[INFO] Respawn detected — loading config from previous attempt: {prev_cfg_path}", flush=True)
            raw_cfg = _load_raw_config(str(prev_cfg_path), unknown_args)
            if not raw_cfg:
                print(f"[WARNING] Previous attempt config at {prev_cfg_path} is empty — falling back to {args.config}", flush=True)
                raw_cfg = _load_raw_config(args.config, unknown_args)
        else:
            raw_cfg = _load_raw_config(args.config, unknown_args)

    # Apply CLI overrides for num_envs / terrain / task before constructing FtrDiffusionConfig
    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs
    if args.terrain is not None:
        raw_cfg.terrain = args.terrain
    if args.task is not None:
        raw_cfg.task = args.task

    # Verify CUDA is accessible before importing FTR tasks.
    # Importing ftr_envs.tasks triggers wp.init() (via omni.isaac.lab.envs chain),
    # which crashes with an unhelpful RuntimeError if the CUDA context is dead.
    # Use os._exit() — not sys.exit/raise — so Isaac Sim's atexit handlers are bypassed
    # and the apptainer process terminates immediately instead of hanging for minutes.
    import os
    import torch
    if not torch.cuda.is_available():
        print(
            "FATAL: torch.cuda.is_available() returned False after AppLauncher init.\n"
            "Isaac Sim failed to create a CUDA context (check .err for 'CUDA error 46').\n"
            "This is usually a node-level GPU issue — try a different compute node.",
            flush=True,
        )
        os._exit(1)

    # Import FTR task registrations (must happen after AppLauncher)
    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    # Use FtrDiffusionConfig only to read the num_robots / task / terrain fields needed for env setup
    _cfg = FtrDiffusionConfig(**raw_cfg)

    # Dynamically resolve the env config class from the gymnasium task registry
    # so that --task Marv-Crossing-Potential-v0 (or any registered task) uses the right config.
    spec = gymnasium.spec(_cfg.task)
    _env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
    if isinstance(_env_cfg_entry, str) and ":" in _env_cfg_entry:
        import importlib
        _mod_path, _cls_name = _env_cfg_entry.rsplit(":", 1)
        _EnvCfgClass = getattr(importlib.import_module(_mod_path), _cls_name)
    elif isinstance(_env_cfg_entry, type):
        _EnvCfgClass = _env_cfg_entry
    else:
        from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg
        _EnvCfgClass = CrossingEnvCfg

    env_cfg = _EnvCfgClass()
    env_cfg.scene.num_envs = _cfg.num_robots
    env_cfg.terrain_name = _cfg.terrain

    # --- Simulation timestep and decimation ---
    env_cfg.sim.dt = _cfg.sim_dt
    env_cfg.decimation = _cfg.decimation

    # --- Rigid body properties ---
    env_cfg.robot.spawn.rigid_props.max_linear_velocity = _cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = _cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = _cfg.max_depenetration_velocity
    env_cfg.robot.spawn.rigid_props.linear_damping = _cfg.robot_linear_damping
    env_cfg.robot.spawn.rigid_props.angular_damping = _cfg.robot_angular_damping

    # --- Per-articulation solver iterations ---
    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = _cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = _cfg.solver_velocity_iterations

    # --- Scene-wide PhysX solver (matched to per-articulation values) ---
    env_cfg.sim.physx.min_position_iteration_count = _cfg.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = _cfg.solver_velocity_iterations
    env_cfg.sim.physx.bounce_threshold_velocity = _cfg.bounce_threshold_velocity
    env_cfg.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = _cfg.physx_gpu_found_lost_aggregate_pairs_capacity

    # Apply arbitrary direct-attribute overrides (e.g. potential reward params)
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    # Raw accel logging: enable flag + interval now; path is set by FtrDiffusionTrainer after
    # RunLogger is created (logpath not known until then).
    if _cfg.log_raw_accel:
        env_cfg.log_raw_accel = True
        env_cfg.log_raw_accel_interval = _cfg.log_raw_accel_interval
        # log_raw_accel_path stays None until trainer patches it below

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    if args.play is not None:
        # Visualisation-only: build env + policy, run forever in deterministic mode
        from marv_rl_training.training.common import make_transformed_env
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        env = ActionChunkEnv(
            FtrTorchRLEnv(
                ftr_gym_env,
                encoder_opts=_cfg.ftr_obs_encoder_opts,
                device=_cfg.device,
                shock_scale=_cfg.env_cfg_overrides.get("shock_scale"),
            ),
            prediction_horizon=_cfg.prediction_horizon,
            execution_horizon=_cfg.execution_horizon,
            control_gamma=_cfg.control_gamma,
        )
        policy_cfg = _cfg.policy_config(**_cfg.policy_opts)
        actor_value_wrapper, _, policy_transforms = policy_cfg.create(
            env=env, weights_path=_cfg.policy_weights_path, device=_cfg.device
        )
        env, vecnorm = make_transformed_env(
            env,
            _cfg,
            policy_transforms,
            post_vecnorm_transforms=[
                CatFrames(N=_cfg.history_len, dim=-1, in_keys=[OBS_KEY], out_keys=["obs_history"], padding="same")
            ],
        )
        if _cfg.vecnorm_weights_path:
            vecnorm.load_state_dict(
                torch.load(_cfg.vecnorm_weights_path, map_location=_cfg.device), strict=False
            )
        actor = actor_value_wrapper.get_policy_operator().eval()
        print("Running policy — close the Isaac Sim window to stop.")
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = env.reset()
            while simulation_app.is_running():
                td = actor(td)
                td = env.step(td)
                td = td["next"]
    else:
        trainer = FtrDiffusionTrainer(raw_cfg, ftr_gym_env)
        try:
            trainer.train()
        except BaseException as _exc:  # noqa: BLE001 — deliberately catching everything, see below
            # ⚠ An exception must NOT be allowed to propagate out of here.
            #
            # The same reason simulation_app.close() is skipped below applies to ordinary
            # interpreter shutdown: Isaac Sim's atexit handlers deadlock. FtrDiffusionTrainer
            # .train() force-exits with os._exit(75) on CUDA/W&B errors, but any OTHER
            # exception is re-raised, and an uncaught exception here means Python runs a
            # normal shutdown -> Isaac's atexit -> hang. The job then holds its node until
            # walltime instead of failing, which on amdgpulong is up to 24 h of a GPU per
            # crash. Observed for real: job 11493997 crashed on a tensor-shape bug at
            # 00:10:20, wrote its crash checkpoint, closed the RunLogger — and was still
            # RUNNING 15 minutes later, doing nothing, until it was cancelled by hand.
            #
            # Exit 1, not 75: 75 means "transient, respawn me", and a crash that reaches
            # here is a bug that will recur. The sbatch loop treats non-75 as terminal.
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            import os as _os
            _os._exit(_exc.code if isinstance(_exc, SystemExit) and isinstance(_exc.code, int) else 1)

    # Skip simulation_app.close() — Isaac Sim's shutdown re-initialises GPU foundation
    # and frequently deadlocks, keeping the SLURM slot busy for hours.
    import os as _os
    _os._exit(0)
