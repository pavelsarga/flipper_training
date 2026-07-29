# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher
import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICM-D3QN (Pan et al. 2023) flipper policy inside FTR-Benchmark (Isaac Sim)")
    parser.add_argument("--config", type=str, required=True, help="Path to an ICM-D3QN config yaml (see FtrICMD3QNConfig)")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in config")
    parser.add_argument("--terrain", type=str, default=None, help="Override terrain in config")
    parser.add_argument("--task", type=str, default=None, help="Override task in config (e.g. Ftr-Crossing-Direct-v0)")
    parser.add_argument("--play", type=str, default=None, metavar="RUN_DIR",
                        help="Visualise a trained policy instead of training. "
                             "Pass the run directory. Loads policy_final.pth + vecnorm_final.pth "
                             "from <RUN_DIR>/weights/.")
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

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import copy
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, RandomSampler, TensorDictReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import gymnasium

import marv_rl_training  # registers OmegaConf resolvers
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
from marv_rl_training.utils.cfg_schedulers import _LinearCfgScheduler
from marv_rl_training.utils.logutils import RunLogger, get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.icmd3qn.icmd3qn_policy import ICMD3QNPolicy
from rl_modules.icmd3qn.icmd3qn_icm import ICMD3QNCuriosityModule


# ============================================================
# BLOCK 3 — Config dataclass for FTR-specific ICM-D3QN training
# ============================================================

@dataclass
class FtrICMD3QNConfig:
    """ICM-D3QN config for FTR-Benchmark training. Structurally identical to
    FtrD3QNConfig (AT-D3QN, train_d3qn.py) plus the curiosity-module fields Eq. 11-14
    need: icm_opts (ICMD3QNCuriosityModule kwargs), icm_optimizer_opts, and the forward/
    inverse loss weights beta_F/beta_I (Eq. 15).
    """

    name: str
    comment: str
    seed: int
    device: str
    training_dtype: torch.dtype
    num_robots: int                    # number of parallel Isaac Sim envs
    task: str                          # gymnasium task ID
    terrain: str                       # FTR terrain name
    total_frames: int
    time_steps_per_batch: int          # steps collected per env per collector iteration

    # Off-policy replay + optimisation
    replay_buffer_capacity: int
    min_replay_size: int                # no updates until the buffer holds at least this many transitions
    batch_size: int                     # transitions sampled per gradient step
    updates_per_batch: int              # gradient steps performed per collected iteration
    gamma: float
    target_update_interval: int         # gradient steps between hard target-network syncs

    # Epsilon-greedy exploration schedule (linear decay over training)
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_iters: int | None    # None = decay over the full run

    eval_and_save_every: int
    eval_repeats: int                   # rollouts averaged per mid-training eval checkpoint
    eval_repeats_after_training: int
    max_grad_norm: float
    clip_grad_norm_p: int | str
    optimizer: type
    optimizer_opts: dict[str, Any]
    scheduler: type
    scheduler_opts: dict[str, Any]
    data_collector_opts: dict[str, Any]
    policy_opts: dict[str, Any]         # ICMD3QNPolicy kwargs (track_vel, hm_hidden, hm_out, fusion_hidden, ...)
    vecnorm_opts: dict[str, Any]
    vecnorm_on_reward: bool
    ftr_obs_encoder_opts: dict[str, Any]

    # ICM (Eq. 11-15): trained jointly with the Q-network on every sampled DQN batch.
    icm_opts: dict[str, Any] = field(default_factory=dict)          # ICMD3QNCuriosityModule kwargs
    icm_optimizer_opts: dict[str, Any] = field(default_factory=dict)  # reuses `optimizer` (e.g. AdamW)
    icm_beta_forward: float = 1.0   # beta_F (Eq. 15)
    icm_beta_inverse: float = 1.0   # beta_I (Eq. 15)
    icm_weights_path: str | None = None

    save_weights_every: int = 0  # 0 = same as eval_and_save_every
    max_eval_steps: int = 0  # 0 = auto: 2 x max_episode_length derived from sim_dt
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
    # Env config overrides applied via setattr before env creation.
    # Must include `module_name: icmd3qn` so FtrEnv computes ICMD3QN's 18-D obs/reward
    # (see rl_modules/registry.py) and FtrTorchRLEnv picks the matching ICMD3QNObservation.
    env_cfg_overrides: dict = field(default_factory=dict)
    # Physics tuning (applied to env_cfg.robot / env_cfg.sim before env creation)
    sim_dt: float = 1 / 400
    decimation: int = 5
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
    physx_gpu_found_lost_aggregate_pairs_capacity: int = 2**27


# ============================================================
# BLOCK 4 — FtrICMD3QNTrainer
# ============================================================

class FtrICMD3QNTrainer:
    """Off-policy Double-Dueling DQN + ICM (ICM-D3QN, Pan et al. 2023) trainer for
    FTR-Benchmark. Structurally identical to FtrD3QNTrainer (train_d3qn.py) — same env
    setup, RunLogger, crash-recovery, replay buffer + target network + epsilon-greedy
    loop — plus a curiosity module trained jointly on every sampled batch: R_t =
    R^e_t + R^i_t (Eq. 13), where R^e_t comes from ICMD3QNModule.get_reward_components()
    and R^i_t from this trainer's ICMD3QNCuriosityModule.
    """

    def __init__(self, raw_config: "DictConfig", ftr_gym_env: gymnasium.Env, optuna_trial=None):
        self.optuna_trial = optuna_trial
        self.config = FtrICMD3QNConfig(**raw_config)
        self.device = set_device(self.config.device)
        self.rng = seed_all(self.config.seed)

        self.run_logger = RunLogger(
            train_config=raw_config,
            category="icmd3qn",
            use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard,
            step_metric_name="collected_frames",
        )
        self._eval_num_env_types = self.config.eval_num_env_types or default_num_env_types(self.config.terrain)
        self._eval_env_type_names = load_env_type_names(self.config.terrain, self.config.eval_env_names_yaml, self._eval_num_env_types)
        self._eval_per_spot_csv = self.run_logger.logpath / "eval_per_spot.csv"
        if self.optuna_trial is not None:
            self.optuna_trial.set_user_attr("logpath", str(self.run_logger.logpath))
        self.term_logger = get_terminal_logger("ftr_icmd3qn_train")
        self.term_logger.info(f"Seed: {self.config.seed}  (random check numpy check torch check cuda check)")

        # ---- environment ----
        self.ftr_torchrl_env = FtrTorchRLEnv(
            ftr_gym_env,
            encoder_opts=self.config.ftr_obs_encoder_opts,
            device=self.device,
            shock_scale=self.config.env_cfg_overrides.get("shock_scale"),
        )
        self.env = self.ftr_torchrl_env

        # ---- policy (Q-network + epsilon-greedy action decode) ----
        self.policy = ICMD3QNPolicy(epsilon=self.config.epsilon_start, **self.config.policy_opts).to(self.device)
        if self.config.policy_weights_path is not None:
            self.policy.q_network.load_state_dict(
                torch.load(self.config.policy_weights_path, map_location=self.device), strict=False
            )
            self.term_logger.info(f"Loaded policy weights from {self.config.policy_weights_path}")
        self.policy_operator = TensorDictModule(self.policy, in_keys=[OBS_KEY], out_keys=["action", "action_idx"])

        # Target network for Double DQN bootstrap — frozen copy, hard-synced periodically.
        self.target_q_network = copy.deepcopy(self.policy.q_network).to(self.device)
        self.target_q_network.eval()
        for p in self.target_q_network.parameters():
            p.requires_grad_(False)

        # ---- curiosity module (Eq. 11-15) — trained jointly, not part of action selection ----
        self.icm = ICMD3QNCuriosityModule(**self.config.icm_opts).to(self.device)
        if self.config.icm_weights_path is not None:
            self.icm.load_state_dict(torch.load(self.config.icm_weights_path, map_location=self.device), strict=False)
            self.term_logger.info(f"Loaded ICM weights from {self.config.icm_weights_path}")

        # ---- transforms + VecNorm ----
        self.env, self.vecnorm = make_transformed_env(self.ftr_torchrl_env, self.config, policy_transforms=[])
        if self.config.vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(
                torch.load(self.config.vecnorm_weights_path, map_location=self.device), strict=False
            )
            self.term_logger.info(f"Loaded vecnorm weights from {self.config.vecnorm_weights_path}")

        # ---- auto-resume from crash checkpoint ----
        self.crash_checkpoint = None
        self._check_and_load_crash_checkpoint()

        # ---- data collection ----
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        self.collector = SyncDataCollector(
            self.env,
            self.policy_operator,
            frames_per_batch=iteration_size,
            total_frames=self.config.total_frames,
            **self.config.data_collector_opts,
            device=self.device,
        )
        # Off-policy: the buffer persists and accumulates across iterations (unlike PPO's
        # per-iteration on-policy buffer), so capacity is independent of iteration_size.
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config.replay_buffer_capacity, ndim=1, device="cpu"),
            sampler=RandomSampler(),
            batch_size=self.config.batch_size,
        )

        # ---- optimizers + scheduler ----
        self.optim = self.config.optimizer(self.policy.q_network.parameters(), **(self.config.optimizer_opts or {}))
        self.icm_optim = self.config.optimizer(self.icm.parameters(), **(self.config.icm_optimizer_opts or self.config.optimizer_opts or {}))
        self.scheduler = self.config.scheduler(self.optim, **(self.config.scheduler_opts or {}))

        _total_iters = self.config.total_frames // iteration_size

        # ---- epsilon-greedy schedule ----
        self.epsilon_scheduler = _LinearCfgScheduler(
            self.policy,
            "epsilon",
            init_value=self.config.epsilon_start,
            start_factor=1.0,
            end_factor=(self.config.epsilon_end / self.config.epsilon_start) if self.config.epsilon_start > 0 else 0.0,
            total_iters=self.config.epsilon_decay_iters or _total_iters,
        )

        self._grad_steps = 0
        self.term_logger.info("Initialized FtrICMD3QNTrainer.")

    # ------------------------------------------------------------------
    # Crash recovery — mirrors FtrD3QNTrainer, plus the ICM network.
    # ------------------------------------------------------------------
    def _check_and_load_crash_checkpoint(self):
        candidate_dirs = self.run_logger.candidate_weight_dirs()

        policy_to_load = None
        vecnorm_to_load = None
        icm_to_load = None
        checkpoint_source = None

        for weights_dir in candidate_dirs:
            policy_crash = weights_dir / "policy_crash.pth"
            vecnorm_crash = weights_dir / "vecnorm_crash.pth"
            icm_crash = weights_dir / "icm_crash.pth"
            if policy_crash.exists() and vecnorm_crash.exists():
                policy_to_load, vecnorm_to_load = policy_crash, vecnorm_crash
                icm_to_load = icm_crash if icm_crash.exists() else None
                checkpoint_source = f"crash ({weights_dir.parent.name})"
                break

            step_policies = sorted(weights_dir.glob("policy_step_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
            step_vecnorms = sorted(weights_dir.glob("vecnorm_step_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
            step_icms = sorted(weights_dir.glob("icm_step_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
            if step_policies and step_vecnorms:
                policy_to_load, vecnorm_to_load = step_policies[-1], step_vecnorms[-1]
                icm_to_load = step_icms[-1] if step_icms else None
                checkpoint_source = f"step ({weights_dir.parent.name})"
                break

        if policy_to_load and vecnorm_to_load:
            self.term_logger.warning(f"Found {checkpoint_source} checkpoint: {policy_to_load.name} / {vecnorm_to_load.name}")
            try:
                self.policy.q_network.load_state_dict(torch.load(policy_to_load, map_location=self.device), strict=False)
                self.target_q_network.load_state_dict(self.policy.q_network.state_dict())
                self.term_logger.info(f"Loaded policy from {checkpoint_source} checkpoint")
            except Exception as e:
                self.term_logger.error(f"Failed to load policy: {e}")
            try:
                self.vecnorm.load_state_dict(torch.load(vecnorm_to_load, map_location=self.device), strict=False)
                self.term_logger.info(f"Loaded vecnorm from {checkpoint_source} checkpoint")
            except Exception as e:
                self.term_logger.error(f"Failed to load vecnorm: {e}")
            if icm_to_load is not None:
                try:
                    self.icm.load_state_dict(torch.load(icm_to_load, map_location=self.device), strict=False)
                    self.term_logger.info(f"Loaded ICM from {checkpoint_source} checkpoint")
                except Exception as e:
                    self.term_logger.error(f"Failed to load ICM: {e}")

            self.crash_checkpoint = True
            self.term_logger.warning(f"Resuming from {checkpoint_source} checkpoint — training may have inconsistent metrics.")
            resume_iteration, resume_frames = self._load_training_checkpoint()
            self.resume_iteration = resume_iteration
            self.resume_frames = resume_frames
        else:
            self.resume_iteration = None
            self.resume_frames = None

    def _save_training_checkpoint(self, iteration: int, total_collected_frames: int):
        checkpoint = {
            "iteration": iteration,
            "total_collected_frames": total_collected_frames,
            "optimizer_state_dict": self.optim.state_dict(),
            "icm_optimizer_state_dict": self.icm_optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epsilon_scheduler_state_dict": self.epsilon_scheduler.state_dict(),
            "grad_steps": self._grad_steps,
        }
        self.run_logger.save_weights(checkpoint, "training_state")

    def _load_training_checkpoint(self):
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
            if checkpoint.get("optimizer_state_dict"):
                try:
                    self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
                except (KeyError, RuntimeError) as e:
                    self.term_logger.warning(f"Failed to load optimizer state: {e}. Optimizer will restart fresh.")
            if checkpoint.get("icm_optimizer_state_dict"):
                try:
                    self.icm_optim.load_state_dict(checkpoint["icm_optimizer_state_dict"])
                except (KeyError, RuntimeError) as e:
                    self.term_logger.warning(f"Failed to load ICM optimizer state: {e}. ICM optimizer will restart fresh.")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "epsilon_scheduler_state_dict" in checkpoint:
                self.epsilon_scheduler.load_state_dict(checkpoint["epsilon_scheduler_state_dict"])
            self._grad_steps = checkpoint.get("grad_steps", 0)
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
                self.term_logger.error(f"{type(e).__name__} detected — calling os._exit(75) to skip cleanup.")
                import os as _os
                _os._exit(75)
            try:
                self.run_logger.save_weights(self.policy.q_network.state_dict(), "policy_crash")
                self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_crash")
                self.run_logger.save_weights(self.icm.state_dict(), "icm_crash")
                _crash_iter = getattr(self, "_current_iteration", 0)
                _crash_frames = getattr(self, "_current_total_frames", 0)
                self._save_training_checkpoint(_crash_iter, _crash_frames)
                self.term_logger.info(f"Saved crash checkpoint at iter={_crash_iter}, frames={_crash_frames}.")
            except Exception:
                pass
            raise
        finally:
            if self.run_logger is not None:
                self.run_logger.close()
        return post_log

    def _icm_update(self, obs: torch.Tensor, action_idx: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Trains psi/F/I on this batch (Eq. 11, 14 -> Eq. 15) and returns the intrinsic
        reward (Eq. 12) for the same transitions, using the just-updated ICM."""
        psi_t1, psi_t1_hat, action_logits = self.icm(obs, action_idx, next_obs)
        l_forward, l_inverse = self.icm.losses(psi_t1, psi_t1_hat, action_logits, action_idx)
        icm_loss = self.config.icm_beta_forward * l_forward + self.config.icm_beta_inverse * l_inverse

        self.icm_optim.zero_grad()
        icm_loss.backward()
        self.icm_optim.step()

        with torch.no_grad():
            psi_t1, psi_t1_hat, _ = self.icm(obs, action_idx, next_obs)
            intrinsic_reward = self.icm.intrinsic_reward(psi_t1, psi_t1_hat)

        return intrinsic_reward, {
            "icm_loss_forward": l_forward.item(),
            "icm_loss_inverse": l_inverse.item(),
            "icm_intrinsic_reward_mean": intrinsic_reward.mean().item(),
        }

    def _dqn_update(self, batch) -> dict[str, float]:
        """One gradient step of Double DQN TD learning on a sampled transition batch,
        with the ICM's intrinsic reward (Eq. 13) added to the extrinsic env reward
        before building the TD target."""
        obs = batch[OBS_KEY]
        next_obs = batch[("next", OBS_KEY)]
        action_idx = batch["action_idx"].long()
        extrinsic_reward = batch[("next", "reward")].reshape(-1)
        terminated = batch[("next", "terminated")].reshape(-1).float()

        intrinsic_reward, icm_log = self._icm_update(obs, action_idx, next_obs)
        reward = extrinsic_reward + intrinsic_reward  # R_t = R^e_t + R^i_t (Eq. 13)

        q_values = self.policy.q_network(obs)  # (B, 9)
        q_sa = q_values.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_online = self.policy.q_network(next_obs)
            next_actions = next_q_online.argmax(dim=-1)  # Double DQN: select with online net
            next_q_target = self.target_q_network(next_obs).gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = reward + self.config.gamma * (1.0 - terminated) * next_q_target

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.q_network.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=False,
            norm_type=self.config.clip_grad_norm_p,
        )
        if grad_norm.isfinite():
            self.optim.step()
        else:
            self.term_logger.warning(f"Skipping update: non-finite grad norm ({grad_norm:.4g})")
            self.optim.zero_grad()

        self._grad_steps += 1
        if self._grad_steps % self.config.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.policy.q_network.state_dict())

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if grad_norm.isfinite() else float("nan"),
            "q_mean": q_values.mean().item(),
            "target_q_mean": target.mean().item(),
            "extrinsic_reward_mean": extrinsic_reward.mean().item(),
            **icm_log,
        }

    def _train(self):
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        self.term_logger.info(f"Iteration size: {iteration_size}, batch_size: {self.config.batch_size}, updates/iter: {self.config.updates_per_batch}")
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        resume_iteration = getattr(self, "resume_iteration", None)
        resume_frames = getattr(self, "resume_frames", None)
        iter_offset = resume_iteration if resume_iteration is not None else 0
        pbar = tqdm(total=self.config.total_frames, desc="FTR ICM-D3QN Training", unit="frames", leave=False)
        if resume_frames is not None:
            pbar.update(resume_frames)
            self.term_logger.info(f"Resuming training — progress bar starting at {resume_frames} frames (iter_offset={iter_offset})")

        for i, tensordict_data in enumerate(self.collector):
            effective_i = i + iter_offset
            total_collected_frames = (effective_i + 1) * iteration_size
            self._current_iteration = effective_i
            self._current_total_frames = total_collected_frames
            pbar.update(iteration_size)

            self.policy.train()
            self.env.train()

            flat = tensordict_data.reshape(-1)

            # Drop exploded transitions — the post-explosion "safe" teleport state is not a
            # meaningful next_obs to bootstrap from. Per-transition (not per-trajectory like
            # PPO/GAE) is sufficient here since DQN transitions are trained independently.
            explosion_flags = flat.get(("next", "explosion"), None)
            n_dirty = 0
            if explosion_flags is not None:
                clean_mask = ~explosion_flags.reshape(-1)
                n_dirty = int((~clean_mask).sum().item())
                if n_dirty > 0:
                    flat = flat[clean_mask]

            if flat.shape[0] == 0:
                self.term_logger.warning("All transitions exploded this iteration — skipping.")
                self.scheduler.step()
                self.epsilon_scheduler.step()
                log = {
                    **self.ftr_torchrl_env.pop_reward_info(),
                    **self.ftr_torchrl_env.pop_termination_info(),
                    **self.ftr_torchrl_env.pop_state_stats(),
                }
                self.run_logger.log_data(log, total_collected_frames)
                continue

            # Sanitize any remaining NaN/Inf before they corrupt the replay buffer.
            nan_count = 0
            for key in flat.keys(include_nested=True, leaves_only=True):
                t = flat[key]
                if t.is_floating_point():
                    bad = ~t.isfinite()
                    if bad.any():
                        nan_count += int(bad.sum().item())
                        flat[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if nan_count > 0:
                self.term_logger.warning(f"Sanitized {nan_count} non-finite values in rollout tensordict.")

            rollout_mean_reward = flat[("next", "reward")].mean().item()
            actions = flat["action"]
            action_log = {
                "action/v_mean": actions[:, 0].mean().item(),
                "action/w_mean": actions[:, 1].mean().item(),
                "action/front_mean": actions[:, 2].mean().item(),
                "action/rear_mean": actions[:, 3].mean().item(),
            }

            self.replay_buffer.extend(flat)
            del tensordict_data, flat
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            update_log = {}
            if len(self.replay_buffer) >= self.config.min_replay_size:
                for _ in range(self.config.updates_per_batch):
                    sub_batch = self.replay_buffer.sample().to(self.device)
                    update_log = self._dqn_update(sub_batch)

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            self.scheduler.step()
            self.epsilon_scheduler.step()

            log = {
                **action_log,
                **self.ftr_torchrl_env.pop_reward_info(),
                **self.ftr_torchrl_env.pop_termination_info(),
                **self.ftr_torchrl_env.pop_state_stats(),
                "train/mean_reward": rollout_mean_reward,
                "train/epsilon": self.policy.epsilon,
                "train/replay_buffer_size": len(self.replay_buffer),
                **{f"train/{g.get('name', 'q_network')}_lr": g["lr"] for g in self.optim.param_groups},
                "explosions/dirty_transitions": n_dirty,
                **{f"train/{k}": v for k, v in update_log.items()},
            }

            save_every = self.config.save_weights_every or self.config.eval_and_save_every
            if effective_i % save_every == 0:
                self.run_logger.save_weights(self.policy.q_network.state_dict(), f"policy_step_{total_collected_frames}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{total_collected_frames}")
                self.run_logger.save_weights(self.icm.state_dict(), f"icm_step_{total_collected_frames}")
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
                            self.term_logger.info(f"Trial pruned at iteration {i} (success_rate={success_rate:.3f})")
                            raise optuna.TrialPruned()
                except optuna.TrialPruned:
                    raise
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        self.term_logger.warning("Eval CUDA error — GPU context corrupted. Exiting immediately.")
                        import os as _os
                        _os._exit(75)
                    self.term_logger.warning(f"Eval rollout failed (physics explosion): {e}. Skipping eval metrics.")

            self.run_logger.log_data(log, total_collected_frames)

        self.run_logger.save_weights(self.policy.q_network.state_dict(), "policy_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")
        self.run_logger.save_weights(self.icm.state_dict(), "icm_final")
        self.run_logger.save_weights(self.policy.q_network.state_dict(), f"policy_step_{self.config.total_frames}")
        self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{self.config.total_frames}")
        self.run_logger.save_weights(self.icm.state_dict(), f"icm_step_{self.config.total_frames}")

    def _get_eval_rollout_results(self) -> dict[str, float]:
        self.env.eval()
        self.policy.eval()  # disables epsilon-greedy exploration (see ICMD3QNPolicy.forward)
        max_eval_steps = self.config.max_eval_steps or (self.ftr_torchrl_env.ftr_env.unwrapped.max_episode_length * 2)
        self.ftr_torchrl_env.enable_per_env_tracking()
        results, episode_records = run_tracked_rollout(
            self.env, self.ftr_torchrl_env, self.ftr_torchrl_env.ftr_env, self.policy_operator, max_eval_steps,
            num_env_types=self._eval_num_env_types,
            env_type_names=self._eval_env_type_names,
            terrain=self.config.terrain,
        )
        self.ftr_torchrl_env.disable_per_env_tracking()
        self._eval_reward_info = {k: results.pop(k) for k in list(results) if k.startswith("rew/") or k.startswith("shock/")}

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
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


if __name__ == "__main__":
    if args.play is not None:
        play_dir = Path(args.play)
        saved_cfg_path = play_dir / "config.yaml"
        if not saved_cfg_path.exists():
            raise FileNotFoundError(f"No config.yaml found in {play_dir}")
        raw_cfg = _load_raw_config(str(saved_cfg_path), unknown_args)
        weights_dir = play_dir / "weights"
        raw_cfg.policy_weights_path = str(weights_dir / "policy_final.pth")
        raw_cfg.vecnorm_weights_path = str(weights_dir / "vecnorm_final.pth")
        raw_cfg.use_wandb = False
        raw_cfg.use_tensorboard = False
    else:
        prev_cfg_path = RunLogger.latest_attempt_config()
        if prev_cfg_path is not None:
            print(f"[INFO] Respawn detected — loading config from previous attempt: {prev_cfg_path}", flush=True)
            raw_cfg = _load_raw_config(str(prev_cfg_path), unknown_args)
            if not raw_cfg:
                print(f"[WARNING] Previous attempt config at {prev_cfg_path} is empty — falling back to {args.config}", flush=True)
                raw_cfg = _load_raw_config(args.config, unknown_args)
        else:
            raw_cfg = _load_raw_config(args.config, unknown_args)

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs
    if args.terrain is not None:
        raw_cfg.terrain = args.terrain
    if args.task is not None:
        raw_cfg.task = args.task

    import os
    import torch
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() returned False after AppLauncher init.", flush=True)
        os._exit(1)

    try:
        import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls
    except Exception as _e:
        print(f"FATAL: failed to import ftr_envs.tasks: {_e}", flush=True)
        os._exit(1)

    _cfg = FtrICMD3QNConfig(**raw_cfg)

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

    env_cfg.sim.dt = _cfg.sim_dt
    env_cfg.decimation = _cfg.decimation

    env_cfg.robot.spawn.rigid_props.max_linear_velocity = _cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = _cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = _cfg.max_depenetration_velocity
    env_cfg.robot.spawn.rigid_props.linear_damping = _cfg.robot_linear_damping
    env_cfg.robot.spawn.rigid_props.angular_damping = _cfg.robot_angular_damping

    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = _cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = _cfg.solver_velocity_iterations

    env_cfg.sim.physx.min_position_iteration_count = _cfg.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = _cfg.solver_velocity_iterations
    env_cfg.sim.physx.bounce_threshold_velocity = _cfg.bounce_threshold_velocity
    env_cfg.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = _cfg.physx_gpu_found_lost_aggregate_pairs_capacity

    # Must set module_name: icmd3qn here (via env_cfg_overrides in the yaml) so FtrEnv
    # routes observations/rewards through ICMD3QNModule instead of the default marv_rl one.
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    if args.play is not None:
        env = FtrTorchRLEnv(
            ftr_gym_env,
            encoder_opts=_cfg.ftr_obs_encoder_opts,
            device=_cfg.device,
            shock_scale=_cfg.env_cfg_overrides.get("shock_scale"),
        )
        policy = ICMD3QNPolicy(epsilon=0.0, **_cfg.policy_opts).to(_cfg.device)
        policy.q_network.load_state_dict(torch.load(_cfg.policy_weights_path, map_location=_cfg.device), strict=False)
        policy_operator = TensorDictModule(policy, in_keys=[OBS_KEY], out_keys=["action", "action_idx"])
        env, vecnorm = make_transformed_env(env, _cfg, policy_transforms=[])
        if _cfg.vecnorm_weights_path:
            vecnorm.load_state_dict(torch.load(_cfg.vecnorm_weights_path, map_location=_cfg.device), strict=False)
        policy.eval()
        print("Running policy — close the Isaac Sim window to stop.")
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = env.reset()
            while simulation_app.is_running():
                td = policy_operator(td)
                td = env.step(td)
                td = td["next"]
    else:
        trainer = FtrICMD3QNTrainer(raw_cfg, ftr_gym_env)
        trainer.train()

    import os as _os
    _os._exit(0)
