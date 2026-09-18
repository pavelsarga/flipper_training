# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
from marv_rl_training.training.cli import eval_arg_parser, launch_isaac_app

parser = eval_arg_parser(
    "Evaluate a trained AT-D3QN / ICM-D3QN flipper policy (Isaac Sim backend).",
    policy_help="Policy (Q-network) checkpoint filename inside <run>/weights/. "
                "(default: policy_final.pth)",
)
args, unknown_args, simulation_app = launch_isaac_app(parser)

# ============================================================
# BLOCK 2 — All other imports (Isaac Sim is now running)
# ============================================================
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.nn import TensorDictModule

import gymnasium

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
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
# FtrD3QNConfig / FtrICMD3QNConfig parse the D3QN-family YAML that FtrPPOConfig rejects
# (replay_buffer_capacity, epsilon_*, target-network fields, ...). icmd3qn adds icm_* fields.
from marv_rl_training.training.train_d3qn import FtrD3QNConfig
from marv_rl_training.training.train_icmd3qn import FtrICMD3QNConfig
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.atd3qn.atd3qn_policy import ATD3QNPolicy
from rl_modules.icmd3qn.icmd3qn_policy import ICMD3QNPolicy


# ============================================================
# BLOCK 3 — Eval logic
# ============================================================

logger = get_terminal_logger("eval_d3qn")


def _select_module(raw_cfg: OmegaConf):
    """Pick (ConfigClass, PolicyClass) from env_cfg_overrides.module_name.

    Both D3QN families share the Q-network / policy_operator interface; the ICM curiosity
    module is training-only, so at eval time icmd3qn just needs the greedy Q-network.
    """
    module_name = (raw_cfg.get("env_cfg_overrides") or {}).get("module_name", "atd3qn")
    if module_name == "icmd3qn":
        return FtrICMD3QNConfig, ICMD3QNPolicy
    return FtrD3QNConfig, ATD3QNPolicy


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
    config_cls=FtrD3QNConfig,
    policy_cls=ATD3QNPolicy,
) -> None:
    cfg = config_cls(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)
    logger.info(f"Seed: {cfg.seed}  (random ✓  numpy ✓  torch ✓  cuda ✓)")

    # Build TorchRL env + transforms + policy (mirrors FtrD3QNTrainer.__init__ /
    # train_d3qn.py --play, NOT the PPO actor-critic path).
    ftr_torchrl_env = FtrTorchRLEnv(
        ftr_gym_env,
        encoder_opts=cfg.ftr_obs_encoder_opts,
        device=device,
        shock_scale=(cfg.env_cfg_overrides or {}).get("shock_scale"),
    )

    if max_steps == 0:
        max_steps = ftr_gym_env.unwrapped.max_episode_length * 2

    # epsilon=0.0 → greedy (argmax Q) action selection; policy.eval() also disables
    # exploration inside the D3QN policy's forward.
    policy = policy_cls(epsilon=0.0, **cfg.policy_opts).to(device)
    policy.q_network.load_state_dict(
        torch.load(cfg.policy_weights_path, map_location=device), strict=False
    )
    logger.info(f"Loaded Q-network weights from {cfg.policy_weights_path}")
    policy_operator = TensorDictModule(policy, in_keys=[OBS_KEY], out_keys=["action", "action_idx"])

    env, vecnorm = make_transformed_env(ftr_torchrl_env, cfg, policy_transforms=[])

    # Prime VecNorm's internal tensordict before loading weights or calling env.eval()
    # (see eval_ftr.run_eval for the rationale — locked-td crash on first reset otherwise).
    env.reset()

    if cfg.vecnorm_weights_path:
        try:
            vecnorm.load_state_dict(
                torch.load(cfg.vecnorm_weights_path, map_location=device), strict=False
            )
            logger.info("Loaded vecnorm weights.")
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Skipping vecnorm weights (incompatible keys): {e}")

    policy.eval()
    env.eval()

    # Resolve CSV-output settings
    _output_dir = Path(output_dir) if output_dir else None
    _eval_id    = eval_id or make_eval_id()
    _terrain    = cfg.terrain
    num_env_types = num_env_types if num_env_types is not None else default_num_env_types(_terrain)
    _env_names  = load_env_type_names(_terrain, env_names_yaml, num_env_types)
    _depth_cols = default_num_depth_cols(_terrain)
    _policy_lbl = policy_label or (cfg.policy_weights_path or "unknown")

    # run_tracked_rollout depends on per-env termination bookkeeping (pop_per_env_termination),
    # which is only populated while tracking is enabled — required even without CSV output.
    ftr_torchrl_env.enable_per_env_tracking()
    if _output_dir:
        logger.info(f"CSV output: {_output_dir}  eval_id={_eval_id}")
        # Record the terrain (and copy its gen_config / preview plot) before the
        # first rollout, so the results are self-describing even if eval crashes.
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

            print_results(results, f"Repeat {r + 1}/{repeats}")
            all_results.append(results)

            if _output_dir and episode_records:
                save_repeat(
                    _output_dir, results, episode_records,
                    eval_id=_eval_id, policy=_policy_lbl, terrain=_terrain,
                    num_envs=ftr_gym_env.unwrapped.num_envs, num_env_types=num_env_types,
                    env_type_names=_env_names, num_depth_cols=_depth_cols, repeat=r + 1,
                )
                logger.info(f"Saved repeat {r + 1} CSV → {_output_dir}")
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

    # atd3qn vs icmd3qn — picks the right config dataclass + greedy policy class.
    _config_cls, _policy_cls = _select_module(raw_cfg)
    logger.info(f"D3QN module: {_config_cls.__name__} / {_policy_cls.__name__}")

    # Build the config just to read task/terrain/env fields for gymnasium.make
    _cfg = _config_cls(**raw_cfg)
    ftr_gym_env = build_ftr_gym_env(_cfg, set_decimation=False, physx_buffers="small")

    run_eval(
        raw_cfg, ftr_gym_env,
        max_steps=max_steps,
        repeats=args.repeats,
        output_dir=args.output_dir,
        num_env_types=args.num_env_types,
        env_names_yaml=args.env_names_yaml,
        eval_id=args.eval_id,
        config_cls=_config_cls,
        policy_cls=_policy_cls,
    )

    exit_flushed()
