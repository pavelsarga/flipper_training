# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher
import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO policy inside FTR-Benchmark (Isaac Sim)")
    parser.add_argument("--config", type=str, required=True, help="Path to ftr_config.yaml")
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

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
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

import flipper_training  # registers OmegaConf resolvers
from flipper_training.environment.ftr_env_adapter import FtrTorchRLEnv
from flipper_training.experiments.ppo.common import make_transformed_env
from flipper_training.utils.logutils import RunLogger, get_terminal_logger
from flipper_training.utils.torch_utils import seed_all, set_device




# ============================================================
# BLOCK 3 — Config dataclass for FTR-specific PPO training
# ============================================================

@dataclass
class FtrPPOConfig:
    """Minimal PPO config for FTR-Benchmark training (no physics engine fields)."""

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
    save_weights_every: int = 0  # 0 = same as eval_and_save_every
    max_eval_steps: int = 0  # 0 = auto: 2 × max_episode_length derived from sim_dt
    use_wandb: bool = False
    use_tensorboard: bool = False
    policy_weights_path: str | None = None
    vecnorm_weights_path: str | None = None
    extra_env_transforms: list = field(default_factory=list)
    # Env config overrides applied via setattr before env creation
    env_cfg_overrides: dict = field(default_factory=dict)
    # Physics tuning (applied to env_cfg.robot / env_cfg.sim before env creation)
    sim_dt: float = 1 / 400
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


# ============================================================
# BLOCK 4 — FtrPPOTrainer
# ============================================================

class FtrPPOTrainer:
    """PPO trainer that uses FTR-Benchmark as the physics backend instead of flipper_training's engine."""

    def __init__(self, raw_config: "DictConfig", ftr_gym_env: gymnasium.Env, optuna_trial=None):
        self.optuna_trial = optuna_trial
        self.config = FtrPPOConfig(**raw_config)
        self.device = set_device(self.config.device)
        self.rng = seed_all(self.config.seed)

        self.run_logger = RunLogger(
            train_config=raw_config,  # RunLogger expects DictConfig
            category="ppo",
            use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard,
            step_metric_name="collected_frames",
        )
        if self.optuna_trial is not None:
            self.optuna_trial.set_user_attr("logpath", str(self.run_logger.logpath))
        self.term_logger = get_terminal_logger("ftr_ppo_train")

        # ---- environment ----
        self.ftr_torchrl_env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=self.config.ftr_obs_encoder_opts, device=self.device)
        self.env = self.ftr_torchrl_env

        # ---- policy ----
        policy_cfg = self.config.policy_config(**self.config.policy_opts)
        self.actor_value_wrapper, self.optim_groups, policy_transforms = policy_cfg.create(
            env=self.env,
            weights_path=self.config.policy_weights_path,
            device=self.device,
        )
        self.actor_operator = self.actor_value_wrapper.get_policy_operator()
        self.value_operator = self.actor_value_wrapper.get_value_operator()

        # ---- transforms + VecNorm ----
        self.env, self.vecnorm = make_transformed_env(self.ftr_torchrl_env, self.config, policy_transforms)
        if self.config.vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(
                torch.load(self.config.vecnorm_weights_path, map_location=self.device), strict=False
            )

        # ---- data collection ----
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
        self.collector = SyncDataCollector(
            self.env,
            self.actor_operator,
            frames_per_batch=iteration_size,
            total_frames=self.config.total_frames,
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
        self.loss_module = ClipPPOLoss(
            self.actor_operator,
            self.actor_value_wrapper.get_value_head()
            if isinstance(self.actor_value_wrapper, ActorValueOperator)
            else self.value_operator,
            **self.config.ppo_opts,
        )
        self.loss_module = self.loss_module.to(self.config.training_dtype)

        # ---- optimizer + scheduler ----
        self.optim = self.config.optimizer(self.optim_groups, **(self.config.optimizer_opts or {}))
        self.scheduler = self.config.scheduler(self.optim, **(self.config.scheduler_opts or {}))

        self.term_logger.info("Initialized FtrPPOTrainer.")

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
            if "CUDA error" in str(e) or "CUDA out of memory" in str(e):
                # CUDA context is dead — saving weights and running atexit/Isaac Sim
                # cleanup handlers will deadlock. Exit immediately so the SLURM slot
                # is freed and the job can be requeued rather than hanging for hours.
                self.term_logger.error("CUDA error detected — calling os._exit(75) to skip cleanup.")
                import os as _os
                _os._exit(75)
            try:
                self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), "policy_crash")
                self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_crash")
                self.term_logger.info("Saved crash checkpoint.")
            except Exception:
                pass
            raise
        finally:
            if self.run_logger is not None:
                self.run_logger.close()
        return post_log

    def _train(self):
        iteration_size = self.config.time_steps_per_batch * self.config.num_robots
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

        pbar = tqdm(total=self.config.total_frames, desc="FTR PPO Training", unit="frames", leave=False)
        for i, tensordict_data in enumerate(self.collector):
            total_collected_frames = (i + 1) * iteration_size
            pbar.update(iteration_size)

            # FTR env does not produce "curr_state"; pop safely.
            tensordict_data.pop("curr_state", None)
            tensordict_data.pop(("next", "curr_state"), None)

            self.actor_operator.train()
            self.value_operator.train()
            self.env.train()

            # GAE computed once; extend CPU replay buffer; free the CUDA rollout tensor
            # before PPO training so PhysX can reclaim GPU memory for the next collection.
            self.advantage_module(tensordict_data)

            # Sanitize all float tensors before extending the replay buffer.
            # NaN/Inf in sample_log_prob, advantage, or value estimates (from physics
            # explosions that slip through the env-side guards) would poison every
            # sub-batch sampled from the buffer, causing all gradient updates to be
            # skipped for the entire iteration.
            nan_count = 0
            for key in tensordict_data.keys(include_nested=True, leaves_only=True):
                t = tensordict_data[key]
                if t.is_floating_point():
                    bad = ~t.isfinite()
                    if bad.any():
                        nan_count += int(bad.sum().item())
                        tensordict_data[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if nan_count > 0:
                self.term_logger.warning(f"Sanitized {nan_count} non-finite values in rollout tensordict before replay buffer extension")

            self.replay_buffer.extend(tensordict_data.reshape(-1))
            rollout_log_prob = tensordict_data["sample_log_prob"].mean().item()
            rollout_adv_mean = tensordict_data["advantage"].mean().item()
            rollout_adv_std = tensordict_data["advantage"].std().item()
            rollout_mean_reward = tensordict_data["next", "reward"].mean().item()
            rollout_mean_state_value = tensordict_data["state_value"].mean().item()
            rollout_value_minus_reward = rollout_mean_state_value - rollout_mean_reward
            del tensordict_data
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            for _j in range(self.config.epochs_per_batch):
                for _k in range(iteration_size // self.config.frames_per_sub_batch):
                    sub_batch = self.replay_buffer.sample().to(self.device)
                    loss_vals = self.loss_module(sub_batch)
                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
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
                    else:
                        self.optim.step()
                        self.optim.zero_grad()

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            self.scheduler.step()
            log = {
                **self.ftr_torchrl_env.pop_reward_info(),
                **self.ftr_torchrl_env.pop_termination_info(),
                "train/mean_reward": rollout_mean_reward,
                "train/mean_state_value": rollout_mean_state_value,
                "train/value_minus_reward": rollout_value_minus_reward,
                "train/mean_advantage_GAE": rollout_adv_mean,
                "train/mean_action_sample_log_prob": rollout_log_prob,
                "train/mean_critic_loss": loss_vals["loss_critic"].mean().item(),
                "train/mean_objective_loss": loss_vals["loss_objective"].mean().item(),
                "train/mean_entropy_loss": loss_vals["loss_entropy"].mean().item(),
                "train/mean_entropy": loss_vals["entropy"].mean().item(),
                "train/mean_kl_approx": loss_vals["kl_approx"].mean().item(),
                "train/mean_clip_fraction": loss_vals["clip_fraction"].mean().item(),
                "train/mean_advantage": rollout_adv_mean,
                "train/std_advantage": rollout_adv_std,
                "train/total_grad_norm": grad_norm.item(),
                **{f"train/{g['name']}_lr": g["lr"] for g in self.optim.param_groups},
            }

            save_every = self.config.save_weights_every or self.config.eval_and_save_every
            if i % save_every == 0:
                self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{total_collected_frames}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{total_collected_frames}")
            if i % self.config.eval_and_save_every == 0 and i > 0:
                try:
                    eval_log = self._get_eval_rollout_results()
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

        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), "policy_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")
        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{self.config.total_frames}")
        self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{self.config.total_frames}")

    def _get_eval_rollout_results(self) -> dict[str, float]:
        self.env.eval()
        self.actor_operator.eval()
        max_eval_steps = self.config.max_eval_steps or (self.ftr_torchrl_env.ftr_env.unwrapped.max_episode_length * 2)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            eval_rollout = self.env.rollout(
                max_eval_steps,
                self.actor_operator,
                break_when_all_done=True,
                auto_reset=True,
            )
        results = {
            "eval/mean_step_reward": eval_rollout["next", "reward"].mean().item(),
            "eval/max_step_reward": eval_rollout["next", "reward"].max().item(),
            "eval/min_step_reward": eval_rollout["next", "reward"].min().item(),
            "eval/pct_terminated": eval_rollout["next", "terminated"].float().mean().item(),
            "eval/pct_truncated": eval_rollout["next", "truncated"].float().mean().item(),
        }
        del eval_rollout
        results.update({"eval/" + k.split("/", 1)[-1]: v for k, v in self.ftr_torchrl_env.pop_termination_info().items()})
        self.ftr_torchrl_env.pop_reward_info()  # discard eval reward info; don't mix into train log
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
        raw_cfg = _load_raw_config(args.config, unknown_args)

    # Apply CLI overrides for num_envs / terrain / task before constructing FtrPPOConfig
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

    # Use FtrPPOConfig only to read the num_robots / task / terrain fields needed for env setup
    _cfg = FtrPPOConfig(**raw_cfg)

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

    # --- Simulation timestep ---
    env_cfg.sim.dt = _cfg.sim_dt

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

    # Apply arbitrary direct-attribute overrides (e.g. potential reward params)
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    if args.play is not None:
        # Visualisation-only: build env + policy, run forever in deterministic mode
        from flipper_training.experiments.ppo.common import make_transformed_env
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=_cfg.ftr_obs_encoder_opts, device=_cfg.device)
        policy_cfg = _cfg.policy_config(**_cfg.policy_opts)
        actor_value_wrapper, _, policy_transforms = policy_cfg.create(
            env=env, weights_path=_cfg.policy_weights_path, device=_cfg.device
        )
        env, vecnorm = make_transformed_env(env, _cfg, policy_transforms)
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
        trainer = FtrPPOTrainer(raw_cfg, ftr_gym_env)
        trainer.train()

    # Skip simulation_app.close() — Isaac Sim's shutdown re-initialises GPU foundation
    # and frequently deadlocks, keeping the SLURM slot busy for hours.
    import os as _os
    _os._exit(0)
