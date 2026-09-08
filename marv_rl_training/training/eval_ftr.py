# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
from marv_rl_training.training.cli import eval_arg_parser, launch_isaac_app

parser = eval_arg_parser("Evaluate a trained FTR PPO policy (Isaac Sim backend).")
parser.add_argument("--plot_heightmap", action="store_true",
                    help="Save heightmap plots to /tmp/ftr_eval_<timestamp>/. Requires num_envs=1.")
parser.add_argument("--plot_interval", type=int, default=1,
                    help="Save a heightmap every N steps (default: 1 = every step).")
parser.add_argument("--print_actions", action="store_true",
                    help="Print the policy's action vector each step for env 0. Requires num_envs=1.")
parser.add_argument("--const_linear_vel", type=float, default=None,
                    help="Override action[:,0] (linear velocity) with this constant value in [-1,1]. "
                         "Use for policies trained without forward command control.")
parser.add_argument("--invert_rear_flippers", action="store_true",
                    help="Multiply rear flipper actions (action[:,4:]) by -1.")
args, unknown_args, simulation_app = launch_isaac_app(parser)

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
from pathlib import Path

import torch
from omegaconf import OmegaConf

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv
from marv_rl_training.training.common import make_transformed_env
from marv_rl_training.training.env_setup import build_ftr_gym_env, import_ftr_tasks
from marv_rl_training.training.env_type_registry import default_num_depth_cols, default_num_env_types
from marv_rl_training.training.eval_common import (
    exit_flushed,
    ActionOverrideWrapper,
    print_results,
    remap_native_to_ftr_weights,
    run_single_rollout,
    run_single_rollout_print_actions,
    run_single_rollout_with_heightmap,
)
from marv_rl_training.training.terrain_assets import write_terrain_manifest
from marv_rl_training.training.eval_data import (
    EpisodeRecord,
    SummaryRow,
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
# BLOCK 3 — Eval logic
# ============================================================

logger = get_terminal_logger("eval_ftr")


def run_eval(
    raw_cfg: OmegaConf,
    ftr_gym_env: gymnasium.Env,
    max_steps: int,
    repeats: int,
    plot_heightmap: bool = False,
    plot_interval: int = 1,
    const_linear_vel: float | None = None,
    invert_rear_flippers: bool = False,
    output_dir: "Path | None" = None,
    num_env_types: "int | None" = None,
    env_names_yaml: "str | None" = None,
    eval_id: "str | None" = None,
    policy_label: "str | None" = None,
    print_actions: bool = False,
) -> None:
    cfg = FtrPPOConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)
    logger.info(f"Seed: {cfg.seed}  (random ✓  numpy ✓  torch ✓  cuda ✓)")

    # Build TorchRL env + transforms + policy (mirrors FtrPPOTrainer.__init__)
    ftr_torchrl_env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=cfg.ftr_obs_encoder_opts, device=device)

    if max_steps == 0:
        max_steps = ftr_gym_env.unwrapped.max_episode_length * 2

    policy_cfg = cfg.policy_config(**cfg.policy_opts)
    flipper_style = (cfg.ftr_obs_encoder_opts or {}).get("flipper_style", False)
    actor_value_wrapper, _, policy_transforms = policy_cfg.create(
        env=ftr_torchrl_env,
        weights_path=cfg.policy_weights_path,
        device=device,
        key_remapper=remap_native_to_ftr_weights if flipper_style else None,
    )
    actor = actor_value_wrapper.get_policy_operator()

    if const_linear_vel is not None or invert_rear_flippers:
        if const_linear_vel is not None:
            logger.info(f"Overriding linear velocity with constant: {const_linear_vel}")
        if invert_rear_flippers:
            logger.info("Inverting flipper actions (multiplying by -1)")
        actor = ActionOverrideWrapper(actor, const_linear_vel=const_linear_vel,
                                      invert_rear_flippers=invert_rear_flippers)

    env, vecnorm = make_transformed_env(ftr_torchrl_env, cfg, policy_transforms)

    # Prime VecNorm's internal tensordict before loading weights or calling env.eval().
    # Without this, _td is empty when env.eval() locks it, and the first rollout reset
    # crashes trying to initialise _td against a locked tensordict.
    env.reset()

    if cfg.vecnorm_weights_path:
        try:
            vecnorm.load_state_dict(
                torch.load(cfg.vecnorm_weights_path, map_location=device), strict=False
            )
            logger.info("Loaded vecnorm weights.")
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Skipping vecnorm weights (incompatible keys — native→FTR transfer?): {e}")

    actor.eval()
    env.eval()

    if plot_heightmap and ftr_gym_env.unwrapped.num_envs != 1:
        raise ValueError("--plot_heightmap requires num_envs=1 (pass --num_envs 1)")
    if print_actions and ftr_gym_env.unwrapped.num_envs != 1:
        raise ValueError("--print_actions requires num_envs=1 (pass --num_envs 1)")

    # Resolve CSV-output settings
    _output_dir = Path(output_dir) if output_dir else None
    _eval_id    = eval_id or make_eval_id()
    _terrain    = cfg.terrain
    num_env_types = num_env_types if num_env_types is not None else default_num_env_types(_terrain)
    _env_names  = load_env_type_names(_terrain, env_names_yaml, num_env_types)
    _depth_cols = default_num_depth_cols(_terrain)
    _policy_lbl = policy_label or (cfg.policy_weights_path or "unknown")

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
    all_episodes: list[EpisodeRecord]   = []
    all_per_env_rows: list = []
    all_per_spot_rows: list = []

    for r in range(repeats):
        logger.info(f"Running eval rollout {r + 1}/{repeats} (max_steps={max_steps}) ...")
        if print_actions:
            results = run_single_rollout_print_actions(env, ftr_torchrl_env, actor, max_steps, logger=logger)
            episode_records: list[EpisodeRecord] = []
        elif plot_heightmap:
            from datetime import datetime
            out_dir = Path(f"/tmp/ftr_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_r{r+1}")
            results = run_single_rollout_with_heightmap(
                env, ftr_torchrl_env, ftr_gym_env, actor, max_steps,
                out_dir=out_dir, plot_interval=plot_interval, logger=logger,
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
            results = run_single_rollout(env, ftr_torchrl_env, actor, max_steps)
            episode_records = []

        print_results(results, f"Repeat {r + 1}/{repeats}")
        all_results.append(results)
        all_episodes.extend(episode_records)

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
            all_per_env_rows.extend(per_env_rows)
            all_per_spot_rows.extend(per_spot_rows)
            save_eval_csvs(_output_dir, [summary], per_env_rows, per_spot_rows, episode_records)
            logger.info(f"Saved repeat {r+1} CSV → {_output_dir}")

    if repeats > 1:
        averaged = {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}
        print_results(averaged, f"AVERAGE over {repeats} repeats")

    if _output_dir:
        logger.info(f"Eval complete. Results saved to {_output_dir}  (eval_id={_eval_id})")


# ============================================================
# BLOCK 4 — Entry point
# ============================================================

if __name__ == "__main__":
    import_ftr_tasks()

    run_dir = Path(args.rundir)
    saved_cfg_path = run_dir / "config.yaml"
    if not saved_cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")

    raw_cfg = OmegaConf.load(saved_cfg_path)
    if unknown_args:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(unknown_args))

    # Point weights paths at the requested checkpoints
    weights_dir = run_dir / "weights"
    raw_cfg.policy_weights_path = str(weights_dir / args.policy)
    raw_cfg.vecnorm_weights_path = str(weights_dir / args.vecnorm)

    # Disable logging backends — this is eval only
    raw_cfg.use_wandb = False
    raw_cfg.use_tensorboard = False

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs

    if args.map is not None:
        raw_cfg.terrain = args.map
        logger.info(f"Terrain overridden: {args.map}")

    max_steps = args.max_steps if args.max_steps is not None else raw_cfg.get("max_eval_steps", 0)

    # FtrPPOConfig is built here only to read the env fields build_ftr_gym_env needs.
    _cfg = FtrPPOConfig(**raw_cfg)
    ftr_gym_env = build_ftr_gym_env(_cfg, set_decimation=False, physx_buffers="auto")

    if _cfg.log_raw_accel:
        accel_path = run_dir / "raw_accel_eval.npz"
        accel_path.unlink(missing_ok=True)  # remove stale/corrupted file from a previous run
        ftr_gym_env.unwrapped.cfg.log_raw_accel_path = str(accel_path)

    run_eval(
        raw_cfg, ftr_gym_env,
        max_steps=max_steps,
        repeats=args.repeats,
        plot_heightmap=args.plot_heightmap,
        plot_interval=args.plot_interval,
        const_linear_vel=args.const_linear_vel,
        invert_rear_flippers=args.invert_rear_flippers,
        output_dir=args.output_dir,
        num_env_types=args.num_env_types,
        env_names_yaml=args.env_names_yaml,
        eval_id=args.eval_id,
        print_actions=args.print_actions,
    )

    exit_flushed()
