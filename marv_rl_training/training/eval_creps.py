# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a trained CREPS (Pecka et al. 2016) flipper policy (Isaac Sim backend)."
)
parser.add_argument(
    "--rundir", type=str, required=True, metavar="RUN_DIR",
    help="Path to the run directory (must contain config.yaml and weights/).",
)
parser.add_argument(
    "--policy", type=str, default="creps_state_final.pth",
    help="CREPS state checkpoint filename inside <run>/weights/ (contains omega_mean/"
         "omega_cov/eta/gamma — NOT a neural-net state_dict). Default: creps_state_final.pth. "
         "SLURM runs that timed out often have no *_final.pth — pass e.g. "
         "creps_state_step_29.pth (see <run>/weights/ for available steps) instead.",
)
parser.add_argument(
    "--num_envs", type=int, default=None,
    help="Override num_robots from config.",
)
parser.add_argument(
    "--repeats", type=int, default=1,
    help="Number of independent eval rollouts to run and average. (default: 1)",
)
parser.add_argument(
    "--max_steps", type=int, default=None,
    help="Override the eval rollout horizon (default: 2x the env's max_episode_length, "
         "so each env gets ~2 independent episodes per repeat for a meaningful success rate).",
)
parser.add_argument(
    "--map", type=str, default=None, metavar="TERRAIN",
    help="Override the terrain from the saved config (e.g. pecka_pallet).",
)
parser.add_argument(
    "--output_dir", type=str, default=None, metavar="DIR",
    help="Directory to save CSV results (eval_summary.csv, eval_per_env.csv, eval_episodes.csv, "
         "eval_per_spot.csv). If omitted, prints metrics only.",
)
parser.add_argument(
    "--num_env_types", type=int, default=None,
    help="Number of distinct env types cycling across robots. Default: looked up from the "
         "terrain's registered layout (env_type_registry.py) — 1 for pecka_pallet's single "
         "raised_platform row.",
)
parser.add_argument(
    "--env_names_yaml", type=str, default=None, metavar="YAML",
    help="Path to YAML file mapping env-type index → name (list or dict), overriding the "
         "terrain's registered default names.",
)
parser.add_argument(
    "--eval_id", type=str, default=None,
    help="Identifier for this eval run (default: auto UTC timestamp).",
)
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
import importlib
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.nn import TensorDictModule

import gymnasium

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
from marv_rl_training.training.env_type_registry import default_num_depth_cols, default_num_env_types
from marv_rl_training.training.terrain_assets import write_terrain_manifest
from marv_rl_training.training.eval_data import (
    SummaryRow,
    aggregate_per_env,
    aggregate_per_spot,
    load_env_type_names,
    make_eval_id,
    run_tracked_rollout,
    save_eval_csvs,
)
# FtrCREPSConfig parses CREPS's own YAML shape (num_iterations, kl_epsilon, safety_delta,
# num_executions_with_same_omega, ...) which FtrPPOConfig/FtrD3QNConfig both reject.
from marv_rl_training.training.train_creps import FtrCREPSConfig, OMEGA_DIM
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.creps.creps_policy import CREPSLowerLevelPolicy


# ============================================================
# BLOCK 3 — Eval logic
# ============================================================

logger = get_terminal_logger("eval_creps")


def _print_results(results: dict[str, float], header: str) -> None:
    print(f"\n{'=' * 60}")
    print(header)
    print('=' * 60)
    for k, v in sorted(results.items()):
        print(f"  {k:<45} {v:.6f}")


def run_eval(
    raw_cfg: OmegaConf,
    ftr_gym_env: gymnasium.Env,
    max_steps: int,
    repeats: int,
    output_dir: "Path | None" = None,
    num_env_types: "int | None" = None,
    env_names_yaml: "str | None" = None,
    eval_id: "str | None" = None,
    policy_label: "str | None" = None,
) -> None:
    cfg = FtrCREPSConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)
    logger.info(f"Seed: {cfg.seed}  (random ✓  numpy ✓  torch ✓  cuda ✓)")

    # Build the raw TorchRL env directly (mirrors train_creps.py's FtrCREPSTrainer.__init__
    # / --play, NOT the PPO actor-critic or D3QN Q-network path). No make_transformed_env/
    # VecNorm — CREPS trains and evaluates on the raw, unnormalized 2-D obs (see
    # creps_observation.py's supports_vecnorm=False docstring).
    ftr_torchrl_env = FtrTorchRLEnv(
        ftr_gym_env,
        encoder_opts=cfg.ftr_obs_encoder_opts,
        device=device,
        shock_scale=(cfg.env_cfg_overrides or {}).get("shock_scale"),
    )
    env = ftr_torchrl_env

    if max_steps == 0:
        max_steps = ftr_gym_env.unwrapped.max_episode_length * 2

    num_envs = ftr_gym_env.unwrapped.num_envs

    # Deterministic eval policy: broadcast the trained omega_mean to every env (no
    # sampling from omega_cov — that's only for exploration during training). This
    # mirrors train_creps.py's --play mode.
    policy = CREPSLowerLevelPolicy(num_envs=num_envs, device=device).to(device)
    state = torch.load(cfg.omega_weights_path, map_location=device)
    omega_mean = torch.from_numpy(np.asarray(state["omega_mean"], dtype=np.float32))
    assert omega_mean.shape == (OMEGA_DIM,), f"omega_mean shape {omega_mean.shape} != ({OMEGA_DIM},)"
    policy.set_omega(omega_mean.unsqueeze(0).expand(num_envs, -1))
    logger.info(f"Loaded omega_mean from {cfg.omega_weights_path}: {omega_mean.tolist()}")
    policy_operator = TensorDictModule(policy, in_keys=[OBS_KEY], out_keys=["action"])

    policy.eval()
    env.eval()

    # Resolve CSV-output settings
    _output_dir = Path(output_dir) if output_dir else None
    _eval_id    = eval_id or make_eval_id()
    _terrain    = cfg.terrain
    num_env_types = num_env_types if num_env_types is not None else default_num_env_types(_terrain)
    _env_names  = load_env_type_names(_terrain, env_names_yaml, num_env_types)
    _depth_cols = default_num_depth_cols(_terrain)
    _policy_lbl = policy_label or (cfg.omega_weights_path or "unknown")

    # run_tracked_rollout depends on per-env termination bookkeeping (pop_per_env_termination),
    # which is only populated while tracking is enabled — required even without CSV output.
    ftr_torchrl_env.enable_per_env_tracking()
    if _output_dir:
        logger.info(f"CSV output: {_output_dir}  eval_id={_eval_id}")
        write_terrain_manifest(
            _output_dir, _eval_id, _terrain, _env_names, _depth_cols, policy=_policy_lbl,
        )
        logger.info(f"Terrain '{_terrain}': {num_env_types} env types x {_depth_cols} depth cols "
                    f"→ assets copied to {_output_dir}/terrain")

    all_results: list[dict[str, float]] = []

    try:
        for r in range(repeats):
            logger.info(f"Running eval rollout {r + 1}/{repeats} (max_steps={max_steps}) ...")
            results, episode_records = run_tracked_rollout(
                env, ftr_torchrl_env, ftr_gym_env, policy_operator, max_steps,
                repeat=r + 1,
                eval_id=_eval_id,
                policy_label=_policy_lbl,
                terrain=_terrain,
                num_env_types=num_env_types,
                env_type_names=_env_names,
            )

            _print_results(results, f"Repeat {r + 1}/{repeats}")
            all_results.append(results)

            if _output_dir and episode_records:
                timestamp = datetime.now(timezone.utc).isoformat()
                summary = SummaryRow(
                    eval_id=_eval_id,
                    policy=_policy_lbl,
                    terrain=_terrain,
                    num_envs=num_envs,
                    num_env_types=num_env_types,
                    repeat=r + 1,
                    timestamp=timestamp,
                    success_rate=results.get("eval/success_rate", float("nan")),
                    failure_rate=results.get("eval/failure_rate", float("nan")),
                    explosion_rate=results.get("eval/explosion_rate", float("nan")),
                    mean_step_reward=results.get("eval/mean_step_reward", float("nan")),
                    shock_mean=results.get("shock/accel_magnitude", float("nan")),
                    shock_p90=results.get("shock/accel_p90", float("nan")),
                    shock_p95=results.get("shock/accel_p95", float("nan")),
                    shock_p99=results.get("shock/accel_p99", float("nan")),
                )
                per_env_rows = aggregate_per_env(
                    episode_records=episode_records,
                    env_type_names=_env_names,
                    eval_id=_eval_id,
                    policy=_policy_lbl,
                    terrain=_terrain,
                    repeat=r + 1,
                    obs_stats=results,
                )
                per_spot_rows = aggregate_per_spot(
                    episode_records=episode_records,
                    env_type_names=_env_names,
                    num_depth_cols=_depth_cols,
                    eval_id=_eval_id,
                    policy=_policy_lbl,
                    terrain=_terrain,
                    repeat=r + 1,
                )
                save_eval_csvs(_output_dir, [summary], per_env_rows, per_spot_rows, episode_records)
                logger.info(f"Saved repeat {r + 1} CSV → {_output_dir}")
    finally:
        ftr_torchrl_env.disable_per_env_tracking()

    if repeats > 1 and all_results:
        averaged = {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}
        _print_results(averaged, f"AVERAGE over {repeats} repeats")

    if _output_dir:
        logger.info(f"Eval complete. Results saved to {_output_dir}  (eval_id={_eval_id})")


# ============================================================
# BLOCK 4 — Entry point
# ============================================================

if __name__ == "__main__":
    import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls

    run_dir = Path(args.rundir)
    saved_cfg_path = run_dir / "config.yaml"
    if not saved_cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")

    raw_cfg = OmegaConf.load(saved_cfg_path)
    if unknown_args:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(unknown_args))

    # Point the weights path at the requested checkpoint (contains omega_mean/omega_cov,
    # not a neural-net state_dict — see CREPSLowerLevelPolicy/FtrCREPSTrainer).
    weights_dir = run_dir / "weights"
    raw_cfg.omega_weights_path = str(weights_dir / args.policy)

    # Disable logging backends — this is eval only
    raw_cfg.use_wandb = False
    raw_cfg.use_tensorboard = False

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs

    if args.map is not None:
        raw_cfg.terrain = args.map
        logger.info(f"Terrain overridden: {args.map}")

    max_steps = args.max_steps if args.max_steps is not None else 0

    # Build the config just to read task/terrain/env fields for gymnasium.make
    _cfg = FtrCREPSConfig(**raw_cfg)

    spec = gymnasium.spec(_cfg.task)
    _env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
    if isinstance(_env_cfg_entry, str) and ":" in _env_cfg_entry:
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

    # --- Simulation timestep + control decimation ---
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

    # Scale down GPU PhysX buffers for small env counts (e.g. local eval on laptop GPUs).
    # The defaults in FTR_SIM_CFG are sized for thousands of envs on server GPUs.
    if _cfg.num_robots <= 64:
        env_cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 22

    # Must apply env_cfg_overrides (module_name: creps, sync_flipper_control: true,
    # fixed_forward_vel: ..., ...) so FtrEnv routes obs/rewards through CREPSModule.
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    run_eval(
        raw_cfg, ftr_gym_env,
        max_steps=max_steps,
        repeats=args.repeats,
        output_dir=args.output_dir,
        num_env_types=args.num_env_types,
        env_names_yaml=args.env_names_yaml,
        eval_id=args.eval_id,
    )

    os._exit(0)
