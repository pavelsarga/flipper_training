# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse

from marv_rl_training.training.cli import add_eval_output_args, launch_isaac_app

parser = argparse.ArgumentParser(
    description="Evaluate a random policy baseline in the FTR-benchmark (Isaac Sim backend)."
)
parser.add_argument("--config", type=str, required=True, metavar="CONFIG",
                    help="Path to the eval config YAML (e.g. configs/random_policy/rand_policy_eval.yaml).")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots from config.")
parser.add_argument("--repeats", type=int, default=1,
                    help="Number of independent eval rollouts to run and average. (default: 1)")
parser.add_argument("--max_steps", type=int, default=None, help="Override max_eval_steps from config.")
parser.add_argument("--plot_heightmap", action="store_true",
                    help="Save heightmap plots to /tmp/ftr_eval_<timestamp>/. Requires num_envs=1.")
parser.add_argument("--plot_interval", type=int, default=1,
                    help="Save a heightmap every N steps (default: 1 = every step).")
parser.add_argument("--accel_out", type=str, default=None,
                    help="Path for raw_accel.npz output. Overrides config log_raw_accel_path. "
                         "Defaults to /tmp/ftr_eval_rand_<timestamp>/raw_accel.npz when "
                         "log_raw_accel is true.")
add_eval_output_args(parser)
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
from marv_rl_training.training.eval_common import (
    exit_flushed,
    print_results,
    run_single_rollout,
    run_single_rollout_with_heightmap,
)
from marv_rl_training.training.env_type_registry import default_num_depth_cols, default_num_env_types
from marv_rl_training.training.terrain_assets import write_terrain_manifest
from marv_rl_training.training.eval_data import (
    EpisodeRecord,
    load_env_type_names,
    make_eval_id,
    run_tracked_rollout,
    save_repeat,
)
from marv_rl_training.training.train_ftr import FtrPPOConfig
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

import gymnasium


# ============================================================
# BLOCK 3 — Eval helpers (self-contained, no import from eval_ftr.py)
# ============================================================

logger = get_terminal_logger("eval_ftr_rand")


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
            results = run_single_rollout_with_heightmap(
                env, ftr_torchrl_env, ftr_gym_env, actor, max_steps,
                out_dir=out_dir, plot_interval=plot_interval,
                collect_obs_stats=False, logger=logger,
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
            results = run_single_rollout(env, ftr_torchrl_env, actor, max_steps,
                                         collect_obs_stats=False)
            episode_records = []

        print_results(results, f"Repeat {r + 1}/{repeats}")
        all_results.append(results)

        if _output_dir and episode_records:
            save_repeat(
                _output_dir, results, episode_records,
                eval_id=_eval_id, policy=_policy_lbl, terrain=_terrain,
                num_envs=ftr_gym_env.unwrapped.num_envs, num_env_types=num_env_types,
                env_type_names=_env_names, num_depth_cols=_depth_cols, repeat=r + 1,
            )
            logger.info(f"Saved repeat {r+1} CSV → {_output_dir}")

    if repeats > 1:
        averaged = {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}
        print_results(averaged, f"AVERAGE over {repeats} repeats")
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
    import_ftr_tasks()

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

    # FtrPPOConfig is built here only to read the env fields build_ftr_gym_env needs.
    _cfg = FtrPPOConfig(**raw_cfg)

    accel_path = None
    if _cfg.log_raw_accel:
        from datetime import datetime
        accel_path = args.accel_out or (
            f"/tmp/ftr_eval_rand_{datetime.now().strftime('%Y%m%d_%H%M%S')}/raw_accel.npz")
        logger.info(f"Raw accel logging enabled → {accel_path}")

    ftr_gym_env = build_ftr_gym_env(_cfg, set_decimation=False, physx_buffers="auto",
                                    log_raw_accel_path=accel_path)

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
        logger.info(f"Raw accel data saved → {accel_path}")


    exit_flushed()
