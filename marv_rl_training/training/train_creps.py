# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CREPS (Pecka et al. 2016) flipper policy inside FTR-Benchmark (Isaac Sim)")
    parser.add_argument("--config", type=str, required=True, help="Path to a CREPS config yaml (see FtrCREPSConfig)")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in config")
    parser.add_argument("--terrain", type=str, default=None, help="Override terrain in config")
    parser.add_argument("--task", type=str, default=None, help="Override task in config (e.g. Ftr-Crossing-Direct-v0)")
    parser.add_argument("--play", type=str, default=None, metavar="RUN_DIR",
                        help="Visualise the current best policy (omega_mean, no sampling) instead of training. "
                             "Pass the run directory. Loads creps_state_final.pth from <RUN_DIR>/weights/.")
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

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type

import gymnasium

import marv_rl_training  # registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
from marv_rl_training.utils.logutils import RunLogger, get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.creps.creps_algorithm import (
    compute_sample_weights,
    effective_sample_size,
    solve_creps_dual,
    weighted_gaussian_fit,
)
from rl_modules.creps.creps_policy import CREPSLowerLevelPolicy

OMEGA_DIM = 6


# ============================================================
# BLOCK 3 — Config dataclass for FTR-specific CREPS training
# ============================================================

@dataclass
class FtrCREPSConfig:
    """Config for Pecka, Salansky, Zimmermann & Svoboda 2016's Constrained REPS (CREPS).

    Unlike every other trainer in this project (PPO/D3QN, gradient-trained neural
    networks), CREPS has NO optimizer, scheduler, replay buffer, or training_dtype -- the
    entire trainable state is the upper-level Gaussian q(omega) = N(omega_mean,
    omega_cov) over the 6-D lower-level linear policy's parameters, refit each outer
    iteration via a closed-form weighted maximum-likelihood update
    (rl_modules/creps/creps_algorithm.py), not gradient descent.
    """

    name: str
    comment: str
    seed: int
    device: str
    num_robots: int              # = total parallel envs = num_omega_samples * num_executions_with_same_omega
    task: str
    terrain: str
    num_iterations: int          # K in the paper (10-30)

    kl_epsilon: float            # eta's KL bound
    safety_delta: float          # lower bound on expected safety compliance (paper: 0.8)
    mechanical_delta: float      # lower bound on expected mechanical-limit compliance

    initial_omega_mean: list     # len 6
    initial_omega_std: list      # len 6, sqrt of the initial covariance's diagonal

    eval_and_save_every: int
    ftr_obs_encoder_opts: dict[str, Any]
    env_cfg_overrides: dict = field(default_factory=dict)   # must include module_name: creps
    cov_min_std: float = 1e-3    # per-dimension floor on the refit covariance's std
    solver_eta_init: float = 1.0
    # Reference repo's CREPSParameters.eta_min/dual_var_max (constrained_reps.py:682-683),
    # and its num_executions_with_same_omega (default 3, gazebo_safe_exploration.py) --
    # each sampled omega is executed this many times and its reward/safety/mechanical
    # values AVERAGED across the repeats before the dual solve (variance reduction).
    # num_robots must be evenly divisible by this.
    num_executions_with_same_omega: int = 3
    solver_eta_min: float = 1e-2
    solver_dual_var_max: float = 1e6
    solver_max_optimization_trials: int = 10
    use_wandb: bool = False
    use_tensorboard: bool = False
    omega_weights_path: str | None = None
    # Physics tuning (applied to env_cfg.robot / env_cfg.sim before env creation) --
    # copied verbatim from FtrHFCConfig/FtrD3QNConfig.
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
# BLOCK 4 — FtrCREPSTrainer
# ============================================================

class FtrCREPSTrainer:
    """Constrained REPS (Pecka et al. 2016) outer-loop trainer.

    Each outer iteration: sample num_omega_samples = num_robots //
    num_executions_with_same_omega different omega vectors from the current upper-level
    Gaussian q(omega)=N(mu,Sigma) (context-free -- see creps_module.py's docstring),
    broadcast each one to num_executions_with_same_omega consecutive parallel envs (a
    variance-reduction technique matched to the reference implementation's
    `CREPSParameters.num_executions_with_same_omega`, see
    paper_repos/tradr-simulation/safe_exploration/src/safe_exploration/
    gazebo_safe_exploration.py), run every one of the num_robots parallel envs for
    exactly one fixed-length episode (_run_creps_episode), average each omega's repeated
    (distance, safety_ok, mechanical_ok) triples down to one value per omega, solve the
    CREPS dual (creps_algorithm.solve_creps_dual), compute sample weights (Eq. 11
    analogue, creps_algorithm.compute_sample_weights), and refit (mu, Sigma) via
    closed-form weighted maximum likelihood (creps_algorithm.weighted_gaussian_fit). No
    replay buffer, no value network, no autograd anywhere in this loop -- the "policy"
    is a (num_omega_samples, 6) tensor of numbers and "training" is closed-form weighted
    statistics repeated num_iterations times.
    """

    def __init__(self, raw_config: "DictConfig", ftr_gym_env: gymnasium.Env):
        self.config = FtrCREPSConfig(**raw_config)
        assert self.config.num_robots % self.config.num_executions_with_same_omega == 0, (
            f"num_robots ({self.config.num_robots}) must be evenly divisible by "
            f"num_executions_with_same_omega ({self.config.num_executions_with_same_omega})"
        )
        self.num_omega_samples = self.config.num_robots // self.config.num_executions_with_same_omega
        self.device = set_device(self.config.device)
        seed_all(self.config.seed)
        # A dedicated numpy Generator (not the legacy global np.random state seed_all
        # also sets) for omega sampling, so its state can be checkpointed/restored
        # exactly on crash-respawn (see _check_and_load_crash_checkpoint/_save_checkpoint).
        self.omega_rng = np.random.default_rng(self.config.seed)

        self.run_logger = RunLogger(
            train_config=raw_config,
            category="creps",
            use_wandb=self.config.use_wandb,
            use_tensorboard=self.config.use_tensorboard,
            step_metric_name="outer_iteration",
        )
        self.term_logger = get_terminal_logger("ftr_creps_train")
        self.term_logger.info(f"Seed: {self.config.seed}")

        # ---- environment ----
        self.ftr_torchrl_env = FtrTorchRLEnv(
            ftr_gym_env,
            encoder_opts=self.config.ftr_obs_encoder_opts,
            device=self.device,
            shock_scale=self.config.env_cfg_overrides.get("shock_scale"),
        )
        # No make_transformed_env/VecNorm wrapping -- CREPS needs the raw, unnormalized
        # 2-D obs (see creps_observation.py's supports_vecnorm=False docstring) and has
        # no reward-vecnorm need either (distance is not a gradient-descent target here).
        self.env = self.ftr_torchrl_env

        self.max_episode_length = self.ftr_torchrl_env.ftr_env.unwrapped.max_episode_length

        # ---- lower-level policy (deterministic linear, no learnable params) ----
        self.policy = CREPSLowerLevelPolicy(num_envs=self.config.num_robots, device=self.device).to(self.device)
        self.policy_operator = TensorDictModule(self.policy, in_keys=[OBS_KEY], out_keys=["action"])

        if self.config.omega_weights_path is not None:
            state = torch.load(self.config.omega_weights_path, map_location=self.device)
            self.omega_mean = np.asarray(state["omega_mean"], dtype=np.float64)
            self.omega_cov = np.asarray(state["omega_cov"], dtype=np.float64)
            self.term_logger.info(f"Loaded omega_mean/omega_cov from {self.config.omega_weights_path}")
        else:
            self.omega_mean = np.array(self.config.initial_omega_mean, dtype=np.float64)
            self.omega_cov = np.diag(np.array(self.config.initial_omega_std, dtype=np.float64) ** 2)
        # eta/gamma are warm-started across outer iterations (solve_creps_dual's x0).
        self.eta = self.config.solver_eta_init
        self.gamma = np.zeros(2)  # [safety, mechanical]

        # ---- auto-resume from crash checkpoint ----
        self.resume_iteration = 0
        self.crash_checkpoint = None
        self._check_and_load_crash_checkpoint()

        self.term_logger.info("Initialized FtrCREPSTrainer.")

    # ------------------------------------------------------------------
    # Crash recovery -- same structure as every other trainer's
    # (candidate_weight_dirs search, crash-file-then-latest-step), but the payload
    # collapses to a single dict: there's no optimizer/scheduler/target-network state
    # to save, since nothing here is gradient-trained.
    # ------------------------------------------------------------------
    def _check_and_load_crash_checkpoint(self):
        candidate_dirs = self.run_logger.candidate_weight_dirs()
        state_to_load = None
        checkpoint_source = None

        for weights_dir in candidate_dirs:
            crash = weights_dir / "creps_state_crash.pth"
            if crash.exists():
                state_to_load = crash
                checkpoint_source = f"crash ({weights_dir.parent.name})"
                break
            step_states = sorted(
                weights_dir.glob("creps_state_step_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])
            )
            if step_states:
                state_to_load = step_states[-1]
                checkpoint_source = f"step ({weights_dir.parent.name})"
                break

        if state_to_load is None:
            return

        self.term_logger.warning(f"Found {checkpoint_source} checkpoint: {state_to_load.name}")
        try:
            checkpoint = torch.load(state_to_load, map_location=self.device)
            self.omega_mean = np.asarray(checkpoint["omega_mean"], dtype=np.float64)
            self.omega_cov = np.asarray(checkpoint["omega_cov"], dtype=np.float64)
            self.eta = float(checkpoint.get("eta", self.eta))
            self.gamma = np.asarray(checkpoint.get("gamma", self.gamma), dtype=np.float64)
            if "rng_state" in checkpoint:
                self.omega_rng.bit_generator.state = checkpoint["rng_state"]
            self.resume_iteration = int(checkpoint["iteration"]) + 1
            self.crash_checkpoint = True
            self.term_logger.warning(
                f"Resuming from {checkpoint_source} checkpoint at iteration {self.resume_iteration} "
                f"-- metrics before this point may look discontinuous."
            )
        except Exception as e:
            self.term_logger.error(f"Failed to load checkpoint: {e}")
            self.resume_iteration = 0

    def _save_checkpoint(self, iteration: int, name: str):
        checkpoint = {
            "iteration": iteration,
            "omega_mean": self.omega_mean,
            "omega_cov": self.omega_cov,
            "eta": self.eta,
            "gamma": self.gamma,
            "rng_state": self.omega_rng.bit_generator.state,
        }
        self.run_logger.save_weights(checkpoint, name)

    # ------------------------------------------------------------------
    def train(self):
        try:
            self._train()
        except KeyboardInterrupt:
            self.term_logger.info("Training interrupted by user.")
        except Exception as e:
            self.term_logger.error(f"Training failed: {e}")
            traceback.print_exception(e)
            if "CUDA error" in str(e) or "CUDA out of memory" in str(e):
                self.term_logger.error(f"{type(e).__name__} detected -- calling os._exit(75) to skip cleanup.")
                import os as _os
                _os._exit(75)
            try:
                self._save_checkpoint(getattr(self, "_current_iteration", 0), "creps_state_crash")
                self.term_logger.info("Saved crash checkpoint.")
            except Exception:
                pass
            raise
        finally:
            if self.run_logger is not None:
                self.run_logger.close()

    def _run_creps_episode(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One CREPS outer-iteration rollout: every one of the num_robots parallel envs
        runs EXACTLY one fixed-length episode (self.max_episode_length steps, the
        env's own episode_length_s=30 default -- matching the paper's fixed 30s
        horizon), each under its own omega already loaded into self.policy.omega.

        Returns (distance, safety_ok, mechanical_ok), each shape (num_robots,):
          - distance: cumulative "distance_reward" (creps_module.py) up to (and
            including) each env's own first `done` step -- R_sw in the paper.
          - safety_ok: 1.0 unless env._fail_mask | env._explosion_mask fired at ANY
            step up to that env's own first done (rollover / out-of-range / high-
            deceleration shock / physics explosion -- the same signals every other
            module already treats as "unsafe/terminal"; used directly here rather than
            reimplementing the underlying physical thresholds a second time).
          - mechanical_ok: 1.0 unless the RAW (pre-clamp) commanded flipper angle
            (env.last_action[:, 2:]) exceeded [-1, 1] at any step up to that env's own
            first done.

        IsaacLab auto-resets a done env INSIDE env.step() -- envs finishing early
        silently start a NEW episode in the same window with the SAME still-loaded
        omega. `already_done` freezes each env's own accumulators exactly once it first
        goes done, so residual post-reset steps for that env are never counted.
        """
        num_envs = self.env.batch_size[0]
        device = self.device
        unwrapped = self.ftr_torchrl_env.ftr_env.unwrapped

        distance = torch.zeros(num_envs, device=device)
        safety_violation = torch.zeros(num_envs, dtype=torch.bool, device=device)
        mechanical_violation = torch.zeros(num_envs, dtype=torch.bool, device=device)
        already_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = self.env.reset()
            for _ in range(self.max_episode_length):
                td = self.policy_operator(td)
                td = self.env.step(td)
                nxt = td["next"]

                active = ~already_done

                step_reward = nxt["reward"].squeeze(-1)
                distance = distance + torch.where(active, step_reward, torch.zeros_like(step_reward))

                unsafe_now = unwrapped._fail_mask | unwrapped._explosion_mask
                safety_violation = safety_violation | (unsafe_now & active)

                mech_bad = (unwrapped.last_action[:, 2:].abs() > 1.0).any(dim=-1)
                mechanical_violation = mechanical_violation | (mech_bad & active)

                dones = nxt["done"].squeeze(-1)
                already_done = already_done | dones
                td = nxt

        safety_ok = (~safety_violation).float()
        mechanical_ok = (~mechanical_violation).float()
        return distance, safety_ok, mechanical_ok

    def _train(self):
        num_robots = self.config.num_robots
        num_repeats = self.config.num_executions_with_same_omega
        num_omega_samples = self.num_omega_samples
        delta = np.array([self.config.safety_delta, self.config.mechanical_delta])
        self.term_logger.info(
            f"CREPS: {num_omega_samples} omega samples x {num_repeats} repeats "
            f"({num_robots} rollouts) per iteration, {self.config.num_iterations} iterations, "
            f"{self.max_episode_length} steps/episode."
        )

        for it in range(self.resume_iteration, self.config.num_iterations):
            self._current_iteration = it

            # Sample num_omega_samples distinct omegas, then broadcast each one to
            # num_repeats consecutive envs (reference repo's
            # num_executions_with_same_omega variance-reduction technique) -- np.repeat
            # groups consecutive envs [0,0,0,1,1,1,...] matching the .view() reshape below.
            omega_unique = self.omega_rng.multivariate_normal(self.omega_mean, self.omega_cov, size=num_omega_samples)
            omega_broadcast = np.repeat(omega_unique, num_repeats, axis=0)  # (num_robots, 6)
            self.policy.set_omega(torch.from_numpy(omega_broadcast).float())

            distance, safety_ok, mechanical_ok = self._run_creps_episode()

            # Average each omega's num_repeats executions down to one value per omega.
            distance_avg = distance.view(num_omega_samples, num_repeats).mean(dim=1)
            safety_avg = safety_ok.view(num_omega_samples, num_repeats).mean(dim=1)
            mechanical_avg = mechanical_ok.view(num_omega_samples, num_repeats).mean(dim=1)

            rewards = distance_avg.cpu().numpy().astype(np.float64)
            constraints = torch.stack([safety_avg, mechanical_avg], dim=-1).cpu().numpy().astype(np.float64)

            eta, gamma, solve_info = solve_creps_dual(
                rewards, constraints, self.config.kl_epsilon, delta,
                eta_init=self.eta, gamma_init=self.gamma,
                eta_min=self.config.solver_eta_min, dual_var_max=self.config.solver_dual_var_max,
                max_optimization_trials=self.config.solver_max_optimization_trials,
                rng=self.omega_rng,
            )
            if solve_info["success"]:
                self.eta, self.gamma = eta, gamma
            else:
                self.term_logger.warning(
                    f"Iteration {it}: CREPS dual solver did not converge after "
                    f"{solve_info['attempts']} attempts ({solve_info['message']}) "
                    f"-- keeping previous (eta, gamma) rather than the runaway solver output "
                    f"(likely the requested safety_delta/mechanical_delta is infeasible within "
                    f"kl_epsilon for the current sample batch)."
                )

            weights = compute_sample_weights(rewards, constraints, self.eta, self.gamma)
            self.omega_mean, self.omega_cov = weighted_gaussian_fit(
                omega_unique, weights, min_std=self.config.cov_min_std,
            )

            log = {
                "train/mean_reward": float(rewards.mean()),
                "train/mean_safety": float(constraints[:, 0].mean()),
                "train/mean_mechanical_ok": float(constraints[:, 1].mean()),
                "train/eta": float(self.eta),
                "train/gamma_safety": float(self.gamma[0]),
                "train/gamma_mechanical": float(self.gamma[1]),
                "train/ess": effective_sample_size(weights),
                "train/solver_success": float(solve_info["success"]),
                **{f"train/omega_mean_{i}": float(v) for i, v in enumerate(self.omega_mean)},
                **{f"train/omega_std_{i}": float(np.sqrt(self.omega_cov[i, i])) for i in range(OMEGA_DIM)},
            }
            self.run_logger.log_data(log, step=it)
            self.term_logger.info(
                f"iter {it}: mean_reward={log['train/mean_reward']:.4f} "
                f"mean_safety={log['train/mean_safety']:.3f} "
                f"mean_mechanical_ok={log['train/mean_mechanical_ok']:.3f} "
                f"eta={self.eta:.4f} gamma={self.gamma}"
            )

            if it % self.config.eval_and_save_every == 0:
                self._save_checkpoint(it, f"creps_state_step_{it}")

        self._save_checkpoint(self.config.num_iterations - 1, "creps_state_final")


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
        raw_cfg.omega_weights_path = str(play_dir / "weights" / "creps_state_final.pth")
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

    _cfg = FtrCREPSConfig(**raw_cfg)

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

    # Must set module_name: creps here (via env_cfg_overrides in the yaml) so FtrEnv
    # routes observations/rewards through CREPSModule instead of the default marv_rl one.
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
        policy = CREPSLowerLevelPolicy(num_envs=_cfg.num_robots, device=_cfg.device).to(_cfg.device)
        state = torch.load(_cfg.omega_weights_path, map_location=_cfg.device)
        omega_mean = torch.from_numpy(np.asarray(state["omega_mean"], dtype=np.float32))
        policy.set_omega(omega_mean.unsqueeze(0).expand(_cfg.num_robots, -1))
        policy_operator = TensorDictModule(policy, in_keys=[OBS_KEY], out_keys=["action"])
        print("Running CREPS policy (omega_mean, deterministic) — close the Isaac Sim window to stop.")
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = env.reset()
            while simulation_app.is_running():
                td = policy_operator(td)
                td = env.step(td)
                td = td["next"]
    else:
        trainer = FtrCREPSTrainer(raw_cfg, ftr_gym_env)
        trainer.train()

    import os as _os
    _os._exit(0)
