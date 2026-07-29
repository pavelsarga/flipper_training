# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
import os
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a random policy baseline in the FTR-benchmark (Isaac Sim backend)."
)
parser.add_argument(
    "--config", type=str, required=True,
    metavar="CONFIG",
    help="Path to the eval config YAML (e.g. configs/rand_policy_eval.yaml).",
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
    help="Override max_eval_steps from config.",
)
parser.add_argument(
    "--plot_heightmap", action="store_true",
    help="Save heightmap plots to /tmp/ftr_eval_<timestamp>/. Requires num_envs=1.",
)
parser.add_argument(
    "--plot_interval", type=int, default=1,
    help="Save a heightmap every N steps (default: 1 = every step).",
)
parser.add_argument(
    "--accel_out", type=str, default=None,
    help="Path for raw_accel.npz output. Overrides config log_raw_accel_path. "
         "Defaults to /tmp/ftr_eval_rand_<timestamp>/raw_accel.npz when log_raw_accel is true.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    metavar="DIR",
    help="Directory to save CSV results (eval_summary.csv, eval_per_env.csv, eval_episodes.csv). "
         "Enables per-robot tracking via a manual step loop. If omitted, prints only (fast path).",
)
parser.add_argument(
    "--num_env_types", type=int, default=None,
    help="Number of distinct env types cycling across robots. Default: looked up from the "
         "terrain's registered layout (env_type_registry.py); 16 if the terrain is unregistered.",
)
parser.add_argument(
    "--env_names_yaml", type=str, default=None,
    metavar="YAML",
    help="Path to YAML file mapping env-type index → name (list or dict), overriding the "
         "terrain's registered default names.",
)
parser.add_argument(
    "--eval_id", type=str, default=None,
    help="Identifier for this eval run (default: auto UTC timestamp).",
)
AppLauncher.add_app_launcher_args(parser)
args, unknown_args = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
import importlib
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import marv_rl_training  # registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv
from marv_rl_training.training.common import make_transformed_env
from marv_rl_training.training.env_type_registry import default_num_depth_cols, default_num_env_types
from marv_rl_training.training.terrain_assets import write_terrain_manifest
from marv_rl_training.training.eval_data import (
    EpisodeRecord,
    PerSpotRow,
    SummaryRow,
    _OBS_SLICES,
    _compute_obs_stats,
    aggregate_per_env,
    aggregate_per_spot,
    load_env_type_names,
    make_eval_id,
    run_tracked_rollout,
    save_eval_csvs,
)
from marv_rl_training.training.train_ftr import FtrPPOConfig
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

import gymnasium


# ============================================================
# BLOCK 3 — Eval helpers (self-contained, no import from eval_ftr.py)
# ============================================================

logger = get_terminal_logger("eval_ftr_rand")


def _save_heightmap(ftr_gym_env, step: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    unwrapped = ftr_gym_env.unwrapped
    hmap = unwrapped.current_frame_height_maps[0].cpu().numpy()
    pos = unwrapped.positions[0].cpu()
    lin_vel = unwrapped.robot_lin_velocities[0].cpu().norm().item()
    ang_vel = unwrapped.robot_ang_velocities[0].cpu().norm().item()
    dist = (unwrapped.target_positions[0, :2] - unwrapped.positions[0, :2]).cpu().norm().item()

    fig, ax = plt.subplots(figsize=(5, 9))
    im = ax.imshow(hmap, origin="upper", cmap="terrain", aspect="auto")
    plt.colorbar(im, ax=ax, label="height (m)")
    cy, cx = hmap.shape[0] // 2, hmap.shape[1] // 2
    ax.plot(cx, cy, "r^", markersize=10, label="robot")
    ax.set_title(
        f"step={step:04d}  pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})\n"
        f"lin_vel={lin_vel:.2f}m/s  ang_vel={ang_vel:.2f}rad/s  dist_goal={dist:.2f}m"
    )
    ax.set_xlabel("← −y (left)   +y (right) →")
    ax.set_ylabel("rear (bottom) ↑ front (top)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "heightmap.png", dpi=80)
    fig.savefig(out_dir / f"step_{step:04d}.png", dpi=80)
    plt.close(fig)


def _print_lin_vels(ftr_gym_env, label: str = "Linear velocities") -> None:
    unwrapped = ftr_gym_env.unwrapped
    vels = unwrapped.robot_lin_velocities.cpu()
    speeds = vels.norm(dim=-1)
    logger.info(f"{label}:")
    logger.info(f"  {'Robot':>5}  {'vx (m/s)':>9}  {'vy (m/s)':>9}  {'vz (m/s)':>9}  {'speed':>7}")
    for i in range(vels.shape[0]):
        logger.info(
            f"  {i:>5}  {vels[i, 0]:>9.4f}  {vels[i, 1]:>9.4f}  {vels[i, 2]:>9.4f}  {speeds[i]:>7.4f}"
        )
    logger.info(
        f"  Summary — mean={speeds.mean():.4f}  max={speeds.max():.4f}  "
        f"min={speeds.min():.4f}  std={speeds.std():.4f} m/s"
    )


def _run_single_rollout_with_heightmap(
    env, ftr_torchrl_env: FtrTorchRLEnv, ftr_gym_env, actor, max_steps: int,
    out_dir: Path, plot_interval: int,
) -> dict[str, float]:
    logger.info(f"Saving heightmap plots to {out_dir} every {plot_interval} step(s)")
    out_dir.mkdir(parents=True, exist_ok=True)

    td = env.reset()
    total_reward = 0.0
    n_steps = 0

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        for step in range(max_steps):
            if plot_interval > 0 and step % plot_interval == 0:
                _save_heightmap(ftr_gym_env, step, out_dir)
            td = actor(td)
            td = env.step(td)
            total_reward += td["next", "reward"].mean().item()
            n_steps += 1
            if td["next", "done"].all():
                break
            td = td["next"]

    _save_heightmap(ftr_gym_env, n_steps, out_dir)

    try:
        import imageio.v2 as imageio
        frames = sorted(out_dir.glob("step_*.png"))
        gif_path = out_dir / "heightmap.gif"
        imgs = [imageio.imread(str(f)) for f in frames]
        imageio.mimsave(str(gif_path), imgs, fps=10)
        logger.info(f"Saved GIF: {gif_path}")
    except Exception as e:
        logger.info(f"Could not create GIF ({e}). Individual PNGs are in {out_dir}")

    results: dict[str, float] = {
        "eval/mean_step_reward": total_reward / max(n_steps, 1),
        "eval/rollout_steps": float(n_steps),
    }
    term_info = ftr_torchrl_env.pop_termination_info()
    results.update({("eval/explosion_rate" if k == "explosions/rate" else "eval/" + k.split("/", 1)[-1]): v for k, v in term_info.items()})
    results.update(ftr_torchrl_env.pop_reward_info())
    return results


def _run_single_rollout(env, ftr_torchrl_env: FtrTorchRLEnv, ftr_gym_env, actor, max_steps: int) -> dict[str, float]:
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        rollout = env.rollout(max_steps, actor, auto_reset=True, break_when_all_done=True)
    results: dict[str, float] = {
        "eval/mean_step_reward": rollout["next", "reward"].mean().item(),
        "eval/max_step_reward":  rollout["next", "reward"].max().item(),
        "eval/min_step_reward":  rollout["next", "reward"].min().item(),
        "eval/pct_terminated":   rollout["next", "terminated"].float().mean().item(),
        "eval/pct_truncated":    rollout["next", "truncated"].float().mean().item(),
        "eval/rollout_steps":    float(rollout.shape[1]),
    }
    del rollout
    term_info = ftr_torchrl_env.pop_termination_info()
    results.update({("eval/explosion_rate" if k == "explosions/rate" else "eval/" + k.split("/", 1)[-1]): v for k, v in term_info.items()})
    results.update(ftr_torchrl_env.pop_reward_info())
    return results


def _print_results(results: dict[str, float], header: str) -> None:
    print(f"\n{'=' * 60}")
    print(header)
    print("=" * 60)
    groups: dict[str, list[tuple[str, float]]] = {}
    for k, v in sorted(results.items()):
        prefix = k.split("/")[0]
        groups.setdefault(prefix, []).append((k, v))
    for prefix, items in groups.items():
        for k, v in items:
            print(f"  {k:<45} {v:.6f}")


# ============================================================
# BLOCK 4 — Main eval loop
# ============================================================

def run_eval_rand(
    raw_cfg: OmegaConf,
    ftr_gym_env: gymnasium.Env,
    max_steps: int,
    repeats: int,
    plot_heightmap: bool = False,
    plot_interval: int = 1,
    output_dir: "Path | None" = None,
    num_env_types: "int | None" = None,
    env_names_yaml: "str | None" = None,
    eval_id: "str | None" = None,
) -> dict[str, float]:
    cfg = FtrPPOConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)
    logger.info(f"Seed: {cfg.seed}  (random ✓  numpy ✓  torch ✓  cuda ✓)")

    ftr_torchrl_env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=cfg.ftr_obs_encoder_opts, device=device)

    if max_steps == 0:
        max_steps = ftr_gym_env.unwrapped.max_episode_length * 2

    policy_cfg = cfg.policy_config(**cfg.policy_opts)
    actor_value_wrapper, _, policy_transforms = policy_cfg.create(
        env=ftr_torchrl_env,
        weights_path=cfg.policy_weights_path if cfg.policy_weights_path else None,
        device=device,
    )
    actor = actor_value_wrapper.get_policy_operator()

    env, vecnorm = make_transformed_env(ftr_torchrl_env, cfg, policy_transforms)
    if cfg.vecnorm_weights_path:
        vecnorm.load_state_dict(
            torch.load(cfg.vecnorm_weights_path, map_location=device), strict=False
        )

    actor_value_wrapper.eval()
    env.eval()

    if plot_heightmap and ftr_gym_env.unwrapped.num_envs != 1:
        raise ValueError("--plot_heightmap requires num_envs=1 (pass --num_envs 1)")

    # CSV-output settings
    _output_dir = Path(output_dir) if output_dir else None
    _eval_id    = eval_id or make_eval_id()
    _terrain    = cfg.terrain
    num_env_types = num_env_types if num_env_types is not None else default_num_env_types(_terrain)
    _env_names  = load_env_type_names(_terrain, env_names_yaml, num_env_types)
    _depth_cols = default_num_depth_cols(_terrain)
    _policy_lbl = "random"

    if _output_dir:
        ftr_torchrl_env.enable_per_env_tracking()
        logger.info(f"Per-env tracking enabled → CSV output: {_output_dir}  eval_id={_eval_id}")
        # Record the terrain (and copy its gen_config / preview plot) before the
        # first rollout, so the results are self-describing even if eval crashes.
        write_terrain_manifest(
            _output_dir, _eval_id, _terrain, _env_names, _depth_cols, policy=_policy_lbl,
        )
        logger.info(f"Terrain '{_terrain}': {num_env_types} env types x {_depth_cols} depth cols "
                    f"→ assets copied to {_output_dir}/terrain")

    all_results: list[dict[str, float]] = []

    for r in range(repeats):
        logger.info(f"Running eval rollout {r + 1}/{repeats} (max_steps={max_steps}) ...")
        if plot_heightmap:
            from datetime import datetime
            out_dir = Path(f"/tmp/ftr_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_r{r+1}")
            results = _run_single_rollout_with_heightmap(
                env, ftr_torchrl_env, ftr_gym_env, actor, max_steps,
                out_dir=out_dir, plot_interval=plot_interval,
            )
            episode_records: list[EpisodeRecord] = []
        elif _output_dir:
            results, episode_records = run_tracked_rollout(
                env, ftr_torchrl_env, ftr_gym_env, actor, max_steps,
                repeat=r + 1,
                eval_id=_eval_id,
                policy_label=_policy_lbl,
                terrain=_terrain,
                num_env_types=num_env_types,
                env_type_names=_env_names,
            )
        else:
            results = _run_single_rollout(env, ftr_torchrl_env, ftr_gym_env, actor, max_steps)
            episode_records = []

        _print_results(results, f"Repeat {r + 1}/{repeats}")
        all_results.append(results)

        if _output_dir and episode_records:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
            summary = SummaryRow(
                eval_id=_eval_id,
                policy=_policy_lbl,
                terrain=_terrain,
                num_envs=ftr_gym_env.unwrapped.num_envs,
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
            logger.info(f"Saved repeat {r+1} CSV → {_output_dir}")

    if repeats > 1:
        averaged = {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}
        _print_results(averaged, f"AVERAGE over {repeats} repeats")
        if _output_dir:
            logger.info(f"Eval complete. Results saved to {_output_dir}  (eval_id={_eval_id})")
        return averaged

    if _output_dir:
        logger.info(f"Eval complete. Results saved to {_output_dir}  (eval_id={_eval_id})")
    return all_results[0]


# ============================================================
# BLOCK 5 — Entry point
# ============================================================

if __name__ == "__main__":
    import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw_cfg = OmegaConf.load(config_path)
    if unknown_args:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(unknown_args))

    raw_cfg.use_wandb = False
    raw_cfg.use_tensorboard = False

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs

    max_steps = args.max_steps if args.max_steps is not None else raw_cfg.get("max_eval_steps", 0)

    _cfg = FtrPPOConfig(**raw_cfg)

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

    env_cfg.sim.dt = _cfg.sim_dt
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

    if _cfg.log_raw_accel:
        from datetime import datetime
        accel_path = args.accel_out or f"/tmp/ftr_eval_rand_{datetime.now().strftime('%Y%m%d_%H%M%S')}/raw_accel.npz"
        env_cfg.log_raw_accel = True
        env_cfg.log_raw_accel_interval = _cfg.log_raw_accel_interval
        env_cfg.log_raw_accel_path = accel_path
        logger.info(f"Raw accel logging enabled → {accel_path}")

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    run_eval_rand(
        raw_cfg, ftr_gym_env,
        max_steps=max_steps,
        repeats=args.repeats,
        plot_heightmap=args.plot_heightmap,
        plot_interval=args.plot_interval,
        output_dir=args.output_dir,
        num_env_types=args.num_env_types,
        env_names_yaml=args.env_names_yaml,
        eval_id=args.eval_id,
    )

    if _cfg.log_raw_accel:
        ftr_gym_env.unwrapped._flush_raw_accel()
        logger.info(f"Raw accel data saved → {env_cfg.log_raw_accel_path}")


    os._exit(0)
