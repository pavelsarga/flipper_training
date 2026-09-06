# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect a (obs_history, action_chunk) dataset for BC-pretraining the Phase 2 "
                "diffusion policy, by rolling out an already-trained marv_rl policy."
)
parser.add_argument("--config", type=str, required=True, help="Path to a chunk-dataset config yaml")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots")
parser.add_argument("--episode_repeats", type=int, default=None, help="Override episode_repeats")
parser.add_argument("--output", type=str, default=None, help="Override output_path")
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

# Force-exit on any uncaught exception — Isaac's atexit handlers deadlock on a normal
# interpreter shutdown, so without this a failure holds the node until walltime.
import sys as _sys


def _force_exit_on_uncaught(exc_type, exc, tb):
    import os as _os
    import traceback as _tb

    _tb.print_exception(exc_type, exc, tb)
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(1)


_sys.excepthook = _force_exit_on_uncaught

# ============================================================
# BLOCK 2 — everything else (Isaac Sim is now running)
# ============================================================
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs import CatFrames
from torchrl.envs.utils import ExplorationType, set_exploration_type

import gymnasium

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
from marv_rl_training.policies.mlp_policy import MLPPolicyConfig
from marv_rl_training.training.chunk_dataset_utils import build_chunks, flipper_angles_to_position_action
from marv_rl_training.training.common import make_transformed_env
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

logger = get_terminal_logger("collect_chunk_dataset")
_WS_ROOT = Path(__file__).resolve().parents[4]


def _resolve_ws_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _WS_ROOT / path


@dataclass
class ChunkDatasetConfig:
    """Rolls out a trained marv_rl policy and stores (obs_history, action_chunk) pairs.

    Two decisions are load-bearing and easy to get wrong:

    1. The rollout runs through the ORDINARY, UN-CHUNKED env with the full training
       transform stack (VecNorm -> CatFrames). obs_history is therefore produced by exactly
       the code path the Phase 2 policy will see at training time. Building it by hand — as
       collect_ctrac_dataset.py must, because CTRACObservation is a different observation —
       is what risks the pretrained model seeing an input distribution the RL stage never
       reproduces. That mismatch is the documented reason C-TRAC's C-VAE trained on a
       constant history while the rollout gave it a real one.

    2. Chunks are formed OFFLINE by sliding a T_p-long window over consecutive control
       steps, and any window crossing an episode boundary is dropped. Chunking the env
       instead would give one chunk per macro step and throw away T_a-1 of every T_a
       possible windows, for no benefit — BC data is per-control-step.
    """

    name: str
    comment: str
    seed: int
    device: str
    num_robots: int
    task: str
    terrain: str
    output_path: str

    prediction_horizon: int          # T_p — must match the Phase 2 config
    history_len: int                 # T_o — must match the Phase 2 config

    marv_rl_policy_opts: dict[str, Any]
    marv_rl_weights_path: str
    marv_rl_vecnorm_weights_path: str
    ftr_obs_encoder_opts: dict[str, Any]
    vecnorm_opts: dict[str, Any]
    vecnorm_on_reward: bool = True
    extra_env_transforms: list = field(default_factory=list)
    env_cfg_overrides: dict = field(default_factory=dict)

    episode_repeats: int = 4
    shard_size_steps: int = 200

    sim_dt: float = 0.005
    decimation: int = 20
    solver_position_iterations: int = 5
    solver_velocity_iterations: int = 1
    max_depenetration_velocity: float = 0.1235
    robot_max_linear_velocity: float = 10.0
    robot_max_angular_velocity: float = 720.0
    physx_gpu_heap_capacity: int = 2**31
    physx_gpu_temp_buffer_capacity: int = 2**30
    physx_gpu_max_num_partitions: int = 8


def main() -> None:
    raw = OmegaConf.load(args.config)
    if unknown_args:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(unknown_args))
    if args.num_envs is not None:
        raw.num_robots = args.num_envs
    if args.episode_repeats is not None:
        raw.episode_repeats = args.episode_repeats
    if args.output is not None:
        raw.output_path = args.output
    cfg = ChunkDatasetConfig(**raw)

    device = set_device(cfg.device)
    seed_all(cfg.seed)

    import ftr_envs.tasks  # noqa: F401 — gymnasium registrations

    spec = gymnasium.spec(cfg.task)
    entry = spec.kwargs.get("env_cfg_entry_point", "")
    if isinstance(entry, str) and ":" in entry:
        import importlib

        mod, cls = entry.rsplit(":", 1)
        EnvCfgClass = getattr(importlib.import_module(mod), cls)
    else:
        from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg

        EnvCfgClass = CrossingEnvCfg

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = cfg.num_robots
    env_cfg.terrain_name = cfg.terrain
    env_cfg.sim.dt = cfg.sim_dt
    env_cfg.decimation = cfg.decimation
    env_cfg.robot.spawn.rigid_props.max_linear_velocity = cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = cfg.max_depenetration_velocity
    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = cfg.solver_velocity_iterations
    env_cfg.sim.physx.gpu_heap_capacity = cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = cfg.physx_gpu_max_num_partitions
    for k, v in (cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)
    # ⚠ The demonstrator was trained in velocity mode and must be ROLLED OUT in velocity
    # mode; only the recorded labels are converted to position mode. Forcing position
    # control here would feed its rate outputs in as absolute targets and produce garbage.
    env_cfg.flipper_control_mode = "velocity"

    ftr_gym_env = gymnasium.make(cfg.task, cfg=env_cfg)
    inner = FtrTorchRLEnv(ftr_gym_env, encoder_opts=cfg.ftr_obs_encoder_opts, device=device)

    policy_cfg = MLPPolicyConfig(**cfg.marv_rl_policy_opts)
    wrapper, _, policy_transforms = policy_cfg.create(
        env=inner, device=device, weights_path=cfg.marv_rl_weights_path
    )
    actor = wrapper.get_policy_operator().eval()

    env, vecnorm = make_transformed_env(
        inner, cfg, policy_transforms,
        post_vecnorm_transforms=[CatFrames(
            N=cfg.history_len, dim=-1, in_keys=[OBS_KEY], out_keys=["obs_history"], padding="same")],
    )
    vecnorm.load_state_dict(torch.load(cfg.marv_rl_vecnorm_weights_path, map_location=device), strict=False)
    # eval() freezes the running statistics. Without it VecNorm keeps adapting to this
    # rollout, so the observations the dataset records drift away from the ones the
    # demonstrator was trained on — and away from what the Phase 2 policy will be given.
    env.eval()

    unwrapped = ftr_gym_env.unwrapped
    low, high = unwrapped.flipper_angle_bounds()
    if low is None:
        raise RuntimeError("flippers are locked (flipper_angle_bounds returned None) — nothing to imitate")
    low, high = low.to(device), high.to(device)

    n_steps = cfg.episode_repeats * unwrapped.max_episode_length
    out_dir = _resolve_ws_path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Collecting {n_steps} control steps x {cfg.num_robots} envs -> {out_dir} "
        f"(T_o={cfg.history_len}, T_p={cfg.prediction_horizon}, shard every {cfg.shard_size_steps} steps)"
    )

    obs_buf, act_buf, th_buf, done_buf = [], [], [], []
    shard_idx, total_pairs = 0, 0

    def flush():
        nonlocal shard_idx, total_pairs, obs_buf, act_buf, th_buf, done_buf
        if len(act_buf) < cfg.prediction_horizon:
            return
        o, a = build_chunks(
            torch.stack(obs_buf), torch.stack(act_buf), torch.stack(th_buf),
            torch.stack(done_buf), cfg.prediction_horizon, low, high,
        )
        if o is None:
            obs_buf, act_buf, th_buf, done_buf = [], [], [], []
            return
        p = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(p, obs_history=o.cpu().numpy().astype(np.float32),
                            action_chunk=a.cpu().numpy().astype(np.float32))
        logger.info(f"  wrote {p.name}: {o.shape[0]} pairs")
        shard_idx += 1
        total_pairs += o.shape[0]
        # Keep the last T_p-1 steps so windows spanning a shard boundary are not lost.
        keep = cfg.prediction_horizon - 1
        obs_buf, act_buf = obs_buf[-keep:], act_buf[-keep:]
        th_buf, done_buf = th_buf[-keep:], done_buf[-keep:]

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        td = env.reset()
        for step in range(n_steps):
            td = actor(td)
            obs_buf.append(td["obs_history"].detach().clone())
            act_buf.append(td["action"].detach().clone())
            td = env.step(td)
            nxt = td["next"]
            # Angle AFTER the step — the target the position-mode action is asking for.
            th_buf.append(unwrapped.flipper_positions.detach().clone().to(device))
            done_buf.append(nxt["done"].squeeze(-1).detach().clone())
            td = nxt
            if len(act_buf) >= cfg.shard_size_steps:
                flush()
    flush()
    logger.info(f"Done: {total_pairs} (obs_history, action_chunk) pairs in {shard_idx} shards under {out_dir}")


if __name__ == "__main__":
    main()
    import os as _os

    _os._exit(0)
