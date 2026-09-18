# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
from marv_rl_training.training.cli import eval_arg_parser, launch_isaac_app

parser = eval_arg_parser(
    "Evaluate a trained C-TRAC (SAC) flipper policy (Isaac Sim backend).",
    policy_help="Actor checkpoint filename inside <run>/weights/. (default: policy_final.pth)",
)
args, unknown_args, simulation_app = launch_isaac_app(parser)

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
from pathlib import Path

from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv
from marv_rl_training.training.common import make_transformed_env
from marv_rl_training.training.env_setup import build_ftr_gym_env, import_ftr_tasks
from marv_rl_training.training.eval_common import exit_flushed, print_results
from marv_rl_training.training.env_type_registry import default_num_depth_cols, default_num_env_types
from marv_rl_training.training.terrain_assets import write_terrain_manifest
from marv_rl_training.training.eval_data import (
    load_env_type_names,
    make_eval_id,
    run_tracked_rollout,
    save_repeat,
)
from marv_rl_training.training.train_sac import FtrSACConfig
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.ctrac.ctrac_policy import CTRACPolicyConfig

logger = get_terminal_logger("eval_sac")


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
                print_results(results, f"Repeat {r + 1}/{repeats}")
                all_results.append(results)

                if _output_dir and episode_records:
                    save_repeat(
                        _output_dir, results, episode_records,
                        eval_id=_eval_id, policy=_policy_lbl, terrain=_terrain,
                        num_envs=ftr_gym_env.unwrapped.num_envs, num_env_types=num_env_types,
                        env_type_names=_env_names, num_depth_cols=_depth_cols, repeat=r + 1,
                    )
                    logger.info(f"Saved repeat {r + 1} CSV -> {_output_dir}")
    finally:
        ftr_torchrl_env.disable_per_env_tracking()

    if repeats > 1 and all_results:
        averaged = {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}
        print_results(averaged, f"AVERAGE over {repeats} repeats")

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
    import_ftr_tasks()

    _cfg = FtrSACConfig(**raw_cfg)
    ftr_gym_env = build_ftr_gym_env(_cfg, physx_buffers="auto")

    run_eval(
        raw_cfg, ftr_gym_env,
        max_steps=args.max_steps or 0, repeats=args.repeats, output_dir=args.output_dir,
        num_env_types=args.num_env_types, env_names_yaml=args.env_names_yaml, eval_id=args.eval_id,
        policy_label=run_dir.name,
    )

    exit_flushed()
