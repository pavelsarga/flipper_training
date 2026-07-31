# ============================================================
# BLOCK 1 — AppLauncher MUST be initialised before any omni.* imports
# ============================================================
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect a C-TRAC C-VAE pretraining dataset (Stage I) by rolling out an "
                "already-trained marv_rl policy through the CTRAC-configured env, logging "
                "CTRACModule's own observations/ground-truth contacts each step."
)
parser.add_argument("--config", type=str, required=True, help="Path to a dataset-collection config yaml")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_robots in config")
parser.add_argument("--total_steps", type=int, default=None, help="Override total_steps in config")
parser.add_argument("--output", type=str, default=None, help="Override output_path in config")
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

import gymnasium

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY, FtrTorchRLEnv
from marv_rl_training.policies.mlp_policy import MLPPolicyConfig
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

from rl_modules.marv_rl.marv_rl_flat_observation import MarvRLFlatObservation
from rl_modules.marv_rl.marv_rl_module import MarvRLModule
from rl_modules.ctrac.ctrac_policy import CTRACObsHistory
from rl_modules.ctrac.ctrac_observation import PARTIAL_DIM

logger = get_terminal_logger("collect_ctrac_dataset")

MARV_RL_OBS_KEY = "MarvRLFlatObservation"


@dataclass
class CTRACDatasetCollectionConfig:
    """Drives collect_ctrac_dataset.py. Not a training config (no optimizer/loss fields) —
    just enough env/policy-loading fields to roll out a fixed, already-trained marv_rl
    policy through the CTRAC-configured env and log what CTRACModule computes each step."""

    name: str
    comment: str
    seed: int
    device: str
    num_robots: int
    task: str
    terrain: str
    total_steps: int                 # env steps to collect (x num_robots transitions logged)
    output_path: str                 # where to write the collected dataset (.pt)
    cvae_history_len: int            # must match ctrac_config.yaml's policy_opts.history_len

    marv_rl_policy_opts: dict[str, Any]     # MLPPolicyConfig kwargs for the demonstrator
    marv_rl_weights_path: str               # trained marv_rl checkpoint (policy_final.pth or similar)
    marv_rl_encoder_opts: dict[str, Any] = field(default_factory=dict)

    env_cfg_overrides: dict = field(default_factory=dict)
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


def _load_raw_config(config_path: str, cli_overrides: list[str]):
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


if __name__ == "__main__":
    raw_cfg = _load_raw_config(args.config, unknown_args)
    if args.num_envs is not None:
        raw_cfg.num_robots = args.num_envs
    if args.total_steps is not None:
        raw_cfg.total_steps = args.total_steps
    if args.output is not None:
        raw_cfg.output_path = args.output

    import os
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() returned False after AppLauncher init.", flush=True)
        os._exit(1)
    import ftr_envs.tasks  # noqa: F401 — triggers gymnasium.register calls

    cfg = CTRACDatasetCollectionConfig(**raw_cfg)
    device = set_device(cfg.device)
    seed_all(cfg.seed)

    from ftr_envs.tasks.crossing.crossing_env import CrossingEnvCfg
    env_cfg = CrossingEnvCfg()
    env_cfg.scene.num_envs = cfg.num_robots
    env_cfg.terrain_name = cfg.terrain
    env_cfg.sim.dt = cfg.sim_dt
    env_cfg.decimation = cfg.decimation
    env_cfg.robot.spawn.rigid_props.max_linear_velocity = cfg.robot_max_linear_velocity
    env_cfg.robot.spawn.rigid_props.max_angular_velocity = cfg.robot_max_angular_velocity
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = cfg.max_depenetration_velocity
    env_cfg.robot.spawn.rigid_props.linear_damping = cfg.robot_linear_damping
    env_cfg.robot.spawn.rigid_props.angular_damping = cfg.robot_angular_damping
    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = cfg.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = cfg.solver_velocity_iterations
    env_cfg.sim.physx.min_position_iteration_count = cfg.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = cfg.solver_velocity_iterations
    env_cfg.sim.physx.bounce_threshold_velocity = cfg.bounce_threshold_velocity
    env_cfg.sim.physx.gpu_heap_capacity = cfg.physx_gpu_heap_capacity
    env_cfg.sim.physx.gpu_temp_buffer_capacity = cfg.physx_gpu_temp_buffer_capacity
    env_cfg.sim.physx.gpu_max_num_partitions = cfg.physx_gpu_max_num_partitions
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = cfg.physx_gpu_found_lost_aggregate_pairs_capacity

    # module_name: ctrac drives CrossingEnv._get_observations/_get_rewards through
    # CTRACModule (and attaches the ContactSensor, ftr_env.py) — this is what makes the
    # logged obs/contacts CTRAC-shaped, independent of which policy is actually driving.
    for k, v in (cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(cfg.task, cfg=env_cfg)
    env = FtrTorchRLEnv(ftr_gym_env, encoder_opts={}, device=device, shock_scale=cfg.env_cfg_overrides.get("shock_scale"))

    # ---- load the marv_rl demonstrator policy ----
    # MLPPolicyConfig.create(env, ...) reads env.observations (name-keyed) to build its
    # EncoderCombiner — env.observations is currently [CTRACObservation(...)] (module_name:
    # ctrac), which is the WRONG shape for marv_rl's actor. Temporarily swap in a
    # MarvRLFlatObservation instance (purely for this construction call — env.observations
    # is restored immediately after) so the demonstrator's network architecture matches
    # what it was actually trained with; env.action_spec is unaffected (it depends only on
    # sync_flipper_control/flipper_style, identical between marv_rl's and ctrac's configs).
    original_observations = env.observations
    env.observations = [MarvRLFlatObservation(env=env, encoder_opts=cfg.marv_rl_encoder_opts)]
    try:
        marv_rl_wrapper, _, _ = MLPPolicyConfig(**cfg.marv_rl_policy_opts).create(
            env, device=device, weights_path=cfg.marv_rl_weights_path,
        )
    finally:
        env.observations = original_observations
    marv_rl_policy_operator = marv_rl_wrapper.get_policy_operator()
    marv_rl_policy_operator.eval()
    logger.info(f"Loaded marv_rl demonstrator from {cfg.marv_rl_weights_path}")

    # marv_rl_module is only ever used to call get_observations() directly against the live
    # underlying env — its reward/scanned-heightmap methods are never invoked, since
    # CTRACModule (the env's actual active module) owns reward/observation bookkeeping.
    marv_rl_module = MarvRLModule(env.ftr_env.unwrapped)

    obs_history_builder = CTRACObsHistory(num_envs=cfg.num_robots, history_len=cfg.cvae_history_len).to(device)

    all_obs_history, all_obs, all_next_obs = [], [], []

    with torch.inference_mode():
        td = env.reset()
        n_steps = cfg.total_steps
        pbar = tqdm(total=n_steps, desc="Collecting C-TRAC dataset", unit="steps")
        for step in range(n_steps):
            marv_obs = torch.nan_to_num(marv_rl_module.get_observations()["policy"].to(device))
            marv_td = TensorDict({MARV_RL_OBS_KEY: marv_obs}, batch_size=[cfg.num_robots], device=device)
            marv_td = marv_rl_policy_operator(marv_td)
            action = marv_td["action"]

            ctrac_obs = td[OBS_KEY]  # this step's CTRAC-shaped packed observation
            partial = ctrac_obs[..., :PARTIAL_DIM]
            fresh_mask = partial[..., -1:]
            obs_hist = obs_history_builder(partial, fresh_mask)

            step_td = TensorDict({"action": action}, batch_size=[cfg.num_robots], device=device)
            td = env.step(step_td)["next"]
            next_obs = td[OBS_KEY]

            all_obs_history.append(obs_hist.cpu())
            all_obs.append(ctrac_obs.cpu())
            all_next_obs.append(next_obs.cpu())
            pbar.update(1)
        pbar.close()

    dataset = {
        "obs_history": torch.cat(all_obs_history, dim=0),
        "obs": torch.cat(all_obs, dim=0),
        "next_obs": torch.cat(all_next_obs, dim=0),
    }
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)
    logger.info(f"Saved {dataset['obs'].shape[0]} transitions to {out_path}")

    import os as _os
    _os._exit(0)
