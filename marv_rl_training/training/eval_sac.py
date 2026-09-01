# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained C-TRAC (SAC) flipper policy (Isaac Sim backend).")
parser.add_argument("--rundir", type=str, required=True, metavar="RUN_DIR", help="Path to the run directory (must contain config.yaml and weights/).")
parser.add_argument("--policy", type=str, default="policy_final.pth", help="Actor checkpoint filename inside <run>/weights/. (default: policy_final.pth)")
parser.add_argument("--vecnorm", type=str, default="vecnorm_final.pth", help="VecNorm checkpoint filename inside <run>/weights/. (default: vecnorm_final.pth)")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots from config.")
parser.add_argument("--repeats", type=int, default=1, help="Number of independent eval rollouts to run and average. (default: 1)")
parser.add_argument("--max_steps", type=int, default=None, help="Override max_eval_steps from config.")
parser.add_argument("--map", type=str, default=None, metavar="TERRAIN", help="Override the terrain from the saved config.")
parser.add_argument("--output_dir", type=str, default=None, metavar="DIR", help="Directory to save CSV results. If omitted, prints metrics only.")
parser.add_argument("--num_env_types", type=int, default=None, help="Number of distinct env types cycling across robots. Default: looked up from the terrain's registered layout.")
parser.add_argument("--env_names_yaml", type=str, default=None, metavar="YAML", help="Path to YAML file mapping env-type index -> name, overriding the terrain's registered default names.")
parser.add_argument("--eval_id", type=str, default=None, help="Identifier for this eval run (default: auto UTC timestamp).")
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()

_filtered, _skip = [], False
for _a in unknown_args:
    if _skip:
        _skip = False
        continue
    if _a.startswith("--") and "=" not in _a:
        _skip = True
        continue
    _filtered.append(_a)
unknown_args = _filtered

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import gymnasium

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv
from marv_rl_training.training.common import make_transformed_env
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
from marv_rl_training.training.train_sac import FtrSACConfig
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.ctrac.ctrac_policy import CTRACPolicyConfig

logger = get_terminal_logger("eval_sac")


def _print_results(results: dict[str, float], header: str) -> None:
    print(f"\n{'=' * 60}")
    print(header)
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k:<45} {v:.6f}")


def run_eval(raw_cfg, ftr_gym_env, max_steps, repeats, output_dir=None, num_env_types=None,
             env_names_yaml=None, eval_id=None, policy_label=None) -> None:
    cfg = FtrSACConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)
    logger.info(f"Seed: {cfg.seed}")

    ftr_torchrl_env = FtrTorchRLEnv(
        ftr_gym_env, encoder_opts=cfg.ftr_obs_encoder_opts, device=device,
        shock_scale=(cfg.env_cfg_overrides or {}).get("shock_scale"),
    )

    if max_steps == 0:
        max_steps = ftr_gym_env.unwrapped.max_episode_length * 2

    # Only the actor is needed for eval — CTRACPolicyConfig.create()'s weights_path loads
    # a flat policy_operator.state_dict() (see ctrac_policy.py's docstring on this).
    #
    # cvae_weights_path is forced to None here, the same as
    # ctrac_policy_inference_module.py does for the ROS2 node, and for the same reason: the
    # C-VAE is a submodule of CTRACActorNet, so the trained one is already inside
    # policy_final.pth (22 of its 28 keys, verified byte-identical to the separately saved
    # cvae_final.pth). Loading the path from the training config would pull in the STAGE I
    # pretrained C-VAE, which is a training-time artifact and not what this policy ran with.
    #
    # It is also a dependency eval has no business having. That path points into the
    # collection dataset directory, so evaluating an archived experiment required a file
    # that may have been cleaned up, moved, or -- as happened here -- half-transferred,
    # producing "PytorchStreamReader failed reading zip archive: failed finding central
    # directory" from a truncated 5.3 MB copy of a 9.3 MB checkpoint.
    policy_opts = {**cfg.policy_opts, "cvae_weights_path": None}
    policy_cfg = CTRACPolicyConfig(**policy_opts)
    policy_operator, _qvalue_operator, _cvae, _optim_groups = policy_cfg.create(
        ftr_torchrl_env, device=device, weights_path=cfg.policy_weights_path,
    )
    logger.info(f"Loaded actor weights from {cfg.policy_weights_path}")

    env, vecnorm = make_transformed_env(ftr_torchrl_env, cfg, policy_transforms=[])
    env.reset()

    if cfg.vecnorm_weights_path:
        try:
            import torch
            vecnorm.load_state_dict(torch.load(cfg.vecnorm_weights_path, map_location=device), strict=False)
            logger.info("Loaded vecnorm weights.")
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Skipping vecnorm weights (incompatible keys): {e}")

    policy_operator.eval()
    env.eval()

    _output_dir = Path(output_dir) if output_dir else None
    _eval_id = eval_id or make_eval_id()
    _terrain = cfg.terrain
    num_env_types = num_env_types if num_env_types is not None else default_num_env_types(_terrain)
    _env_names = load_env_type_names(_terrain, env_names_yaml, num_env_types)
    _depth_cols = default_num_depth_cols(_terrain)
    _policy_lbl = policy_label or (cfg.policy_weights_path or "unknown")

    ftr_torchrl_env.enable_per_env_tracking()
    if _output_dir:
        logger.info(f"CSV output: {_output_dir}  eval_id={_eval_id}")
        write_terrain_manifest(_output_dir, _eval_id, _terrain, _env_names, _depth_cols, policy=_policy_lbl)

    all_results = []
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for r in range(repeats):
                logger.info(f"Running eval rollout {r + 1}/{repeats} (max_steps={max_steps}) ...")
                results, episode_records = run_tracked_rollout(
                    env, ftr_torchrl_env, ftr_gym_env, policy_operator, max_steps,
                    repeat=r + 1, eval_id=_eval_id, policy_label=_policy_lbl, terrain=_terrain,
                    num_env_types=num_env_types, env_type_names=_env_names,
                )
                _print_results(results, f"Repeat {r + 1}/{repeats}")
                all_results.append(results)

                if _output_dir and episode_records:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    summary = SummaryRow(
                        eval_id=_eval_id, policy=_policy_lbl, terrain=_terrain,
                        num_envs=ftr_gym_env.unwrapped.num_envs, num_env_types=num_env_types, repeat=r + 1,
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
                        episode_records=episode_records, env_type_names=_env_names, eval_id=_eval_id,
                        policy=_policy_lbl, terrain=_terrain, repeat=r + 1, obs_stats=results,
                    )
                    per_spot_rows = aggregate_per_spot(
                        episode_records=episode_records, env_type_names=_env_names, num_depth_cols=_depth_cols,
                        eval_id=_eval_id, policy=_policy_lbl, terrain=_terrain, repeat=r + 1,
                    )
                    save_eval_csvs(_output_dir, [summary], per_env_rows, per_spot_rows, episode_records)
                    logger.info(f"Saved repeat {r + 1} CSV -> {_output_dir}")
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
    run_dir = Path(args.rundir)
    saved_cfg_path = run_dir / "config.yaml"
    if not saved_cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")
    raw_cfg = OmegaConf.load(saved_cfg_path)
    if unknown_args:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(unknown_args))

    weights_dir = run_dir / "weights"
    raw_cfg.policy_weights_path = str(weights_dir / args.policy)
    raw_cfg.vecnorm_weights_path = str(weights_dir / args.vecnorm)
    raw_cfg.use_wandb = False
    raw_cfg.use_tensorboard = False

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs
    if args.map is not None:
        raw_cfg.terrain = args.map

    import os
    import torch
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() returned False after AppLauncher init.", flush=True)
        os._exit(1)
    import ftr_envs.tasks  # noqa: F401

    _cfg = FtrSACConfig(**raw_cfg)
    spec = gymnasium.spec(_cfg.task)
    _env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
    if isinstance(_env_cfg_entry, str) and ":" in _env_cfg_entry:
        import importlib
        _mod_path, _cls_name = _env_cfg_entry.rsplit(":", 1)
        _EnvCfgClass = getattr(importlib.import_module(_mod_path), _cls_name)
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

    # Scale down GPU PhysX buffers for small env counts (e.g. local eval on laptop GPUs) —
    # mirrors eval_ftr.py's own scaling exactly. FTR_SIM_CFG's defaults are sized for 4096
    # envs on server GPUs; requesting that scale of PhysX GPU buffers on an 8 GB laptop
    # card is what was actually causing "Failed to create simulation view: no active
    # physics scene found" locally — confirmed by direct comparison against eval_ftr.py,
    # which already has this scaling and works fine locally at small env counts, while
    # eval_sac.py (missing it entirely) failed even at num_envs=4. Not a ContactSensor,
    # terrain, or SAC-specific issue — this scaling was simply never carried over from
    # eval_ftr.py's reference pattern into any of the ctrac scripts.
    if _cfg.num_robots <= 64:
        env_cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 22
    elif _cfg.num_robots > 512:
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 27

    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    run_eval(
        raw_cfg, ftr_gym_env,
        max_steps=args.max_steps or 0, repeats=args.repeats, output_dir=args.output_dir,
        num_env_types=args.num_env_types, env_names_yaml=args.env_names_yaml, eval_id=args.eval_id,
        policy_label=run_dir.name,
    )

    os._exit(0)
