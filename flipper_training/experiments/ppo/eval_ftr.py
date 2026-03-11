# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a trained FTR PPO policy (Isaac Sim backend)."
)
parser.add_argument(
    "--rundir", type=str, required=True,
    metavar="RUN_DIR",
    help="Path to the run directory (must contain config.yaml and weights/).",
)
parser.add_argument(
    "--policy", type=str, default="policy_final.pth",
    help="Policy checkpoint filename inside <run>/weights/. (default: policy_final.pth)",
)
parser.add_argument(
    "--vecnorm", type=str, default="vecnorm_final.pth",
    help="VecNorm checkpoint filename inside <run>/weights/. (default: vecnorm_final.pth)",
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

import flipper_training  # registers OmegaConf resolvers
from flipper_training.environment.ftr_env_adapter import FtrTorchRLEnv
from flipper_training.experiments.ppo.common import make_transformed_env
from flipper_training.experiments.ppo.train_ftr import FtrPPOConfig
from flipper_training.utils.logutils import get_terminal_logger
from flipper_training.utils.torch_utils import seed_all, set_device

import gymnasium


# ============================================================
# BLOCK 3 — Eval logic
# ============================================================

logger = get_terminal_logger("eval_ftr")


def _run_single_rollout(env, ftr_torchrl_env: FtrTorchRLEnv, actor, max_steps: int) -> dict[str, float]:
    """Run one deterministic rollout and return a flat dict of metrics."""
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

    # Termination stats (success / failure rates per episode)
    term_info = ftr_torchrl_env.pop_termination_info()
    results.update({"eval/" + k.split("/", 1)[-1]: v for k, v in term_info.items()})

    # Per-component reward means
    results.update(ftr_torchrl_env.pop_reward_info())

    return results


def _print_results(results: dict[str, float], header: str) -> None:
    print(f"\n{'=' * 60}")
    print(header)
    print('=' * 60)
    # Group by prefix for readability
    groups: dict[str, list[tuple[str, float]]] = {}
    for k, v in sorted(results.items()):
        prefix = k.split("/")[0]
        groups.setdefault(prefix, []).append((k, v))
    for prefix, items in groups.items():
        for k, v in items:
            print(f"  {k:<45} {v:.6f}")


def run_eval(raw_cfg: OmegaConf, ftr_gym_env: gymnasium.Env, max_steps: int, repeats: int) -> None:
    cfg = FtrPPOConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)

    # Build TorchRL env + transforms + policy (mirrors FtrPPOTrainer.__init__)
    ftr_torchrl_env = FtrTorchRLEnv(ftr_gym_env, encoder_opts=cfg.ftr_obs_encoder_opts, device=device)

    policy_cfg = cfg.policy_config(**cfg.policy_opts)
    actor_value_wrapper, _, policy_transforms = policy_cfg.create(
        env=ftr_torchrl_env,
        weights_path=cfg.policy_weights_path,
        device=device,
    )
    actor = actor_value_wrapper.get_policy_operator()

    env, vecnorm = make_transformed_env(ftr_torchrl_env, cfg, policy_transforms)
    if cfg.vecnorm_weights_path:
        vecnorm.load_state_dict(
            torch.load(cfg.vecnorm_weights_path, map_location=device), strict=False
        )

    actor.eval()
    env.eval()

    all_results: list[dict[str, float]] = []
    for r in range(repeats):
        logger.info(f"Running eval rollout {r + 1}/{repeats} (max_steps={max_steps}) ...")
        results = _run_single_rollout(env, ftr_torchrl_env, actor, max_steps)
        _print_results(results, f"Repeat {r + 1}/{repeats}")
        all_results.append(results)

    if repeats > 1:
        averaged = {k: sum(d[k] for d in all_results) / repeats for k in all_results[0]}
        _print_results(averaged, f"AVERAGE over {repeats} repeats")


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

    # Point weights paths at the requested checkpoints
    weights_dir = run_dir / "weights"
    raw_cfg.policy_weights_path = str(weights_dir / args.policy)
    raw_cfg.vecnorm_weights_path = str(weights_dir / args.vecnorm)

    # Disable logging backends — this is eval only
    raw_cfg.use_wandb = False
    raw_cfg.use_tensorboard = False

    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs

    max_steps = args.max_steps if args.max_steps is not None else raw_cfg.max_eval_steps

    # Build FtrPPOConfig just to read task/terrain/env fields for gymnasium.make
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
    env_cfg.robot.spawn.rigid_props.max_linear_velocity = _cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = _cfg.robot_max_angular_velocity
    env_cfg.sim.physx.gpu_heap_capacity = _cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = _cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = _cfg.physx_gpu_max_num_partitions
    for k, v in (_cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(_cfg.task, cfg=env_cfg)

    run_eval(raw_cfg, ftr_gym_env, max_steps=max_steps, repeats=args.repeats)

    simulation_app.close()
