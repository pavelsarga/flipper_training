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
parser.add_argument("--episode_repeats", type=int, default=None,
                     help="Override episode_repeats in config (full episode resets per env — see "
                          "CTRACDatasetCollectionConfig's docstring for why this is preferred over --total_steps)")
parser.add_argument("--total_steps", type=int, default=None, help="Override total_steps in config (raw control-step count)")
parser.add_argument("--output", type=str, default=None, help="Override output_path in config")
parser.add_argument("--log_every_n_steps", type=int, default=None, help="Override log_every_n_steps in config")
parser.add_argument("--shard_size_steps", type=int, default=None, help="Override shard_size_steps in config")
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
from torchrl.envs import VecNorm
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

# Workspace root: .../<ws>/src/flipper_training/marv_rl_training/training/<this file>. Under
# apptainer the workspace is bound at /ws, so this resolves to /ws there and to the checkout
# root locally — without depending on the process's cwd, which is NOT reliable in the SLURM
# jobs: the .sbatch does `cd $WS` on the *host*, but $WS doesn't exist inside the container
# (only the /ws bind does), so apptainer silently starts elsewhere and a relative
# output_path would land somewhere unintended.
_WS_ROOT = Path(__file__).resolve().parents[4]


def _resolve_ws_path(p: str) -> Path:
    """Absolute paths pass through; relative ones resolve against the workspace root so a
    config's `output_path: experiments/...` means the same thing locally and on SLURM."""
    path = Path(p)
    return path if path.is_absolute() else _WS_ROOT / path


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
    # Directory the dataset is written to, as numbered shard files (shard_00000.pt, ...) —
    # NOT a single file. Raising episode_repeats for diversity (see below) multiplies total
    # transitions by num_robots x steps-per-episode; at num_robots in the hundreds/thousands
    # this reaches tens-to-hundreds of GB, which held entirely in RAM before one final
    # torch.save (the old behaviour) OOMs long before that. Sharding bounds peak RAM to one
    # shard's worth regardless of how long the run goes; pretrain_ctrac_cvae.py reads the
    # whole shard directory back with a bounded in-memory cap (max_dataset_rows).
    output_path: str
    cvae_history_len: int            # must match ctrac_config.yaml's policy_opts.history_len

    marv_rl_policy_opts: dict[str, Any]     # MLPPolicyConfig kwargs for the demonstrator
    marv_rl_weights_path: str               # trained marv_rl checkpoint (policy_final.pth or similar)
    # MarvRLFlatObservation.supports_vecnorm = True (unlike CTRACObservation) — the marv_rl
    # policy was trained on VecNorm-normalized observations (see make_transformed_env /
    # marv_config_marv_rl.yaml's vecnorm_opts), so its actor expects the same normalization
    # at inference. This script calls MarvRLModule.get_observations() directly against the
    # raw env (bypassing FtrTorchRLEnv's transform pipeline entirely — see the comment at
    # the marv_rl_wrapper construction below), so the loaded VecNorm stats must be applied
    # by hand each step; feeding it raw observations would make the demonstrator's actions
    # close to garbage, silently corrupting the whole point of using a trained policy as
    # the data source. Required — no default, since a stale/missing vecnorm here fails
    # quietly (a valid-shaped but wrong-scale observation, not an error) rather than loudly.
    marv_rl_vecnorm_weights_path: str
    marv_rl_encoder_opts: dict[str, Any] = field(default_factory=dict)
    marv_rl_vecnorm_opts: dict[str, Any] = field(default_factory=lambda: {"decay": 0.99, "eps": 1e-4})

    # How long to run: episode_repeats (recommended) computes total control-steps as
    # episode_repeats * max_episode_length once the real env is built, so every one of the
    # num_robots parallel envs completes roughly that many full episode resets (distinct
    # start/target draws + domain-randomization noise) before collection stops — the number
    # that actually determines dataset diversity, not a raw step count that silently depends
    # on episode_length_s/sim_dt/decimation. total_steps is a raw control-step count and
    # wins if both are set (escape hatch for exact reproducibility). Setting neither is an
    # error — cheap step counts (e.g. 2000 steps at the default 300-step episode length is
    # only ~6-7 repeats per env) are exactly the "not enough repeats" trap this exists to
    # avoid making you compute by hand.
    episode_repeats: "int | None" = None
    total_steps: "int | None" = None

    # Dataset-size controls, independent of episode_repeats (which should stay large — it's
    # what buys diversity). log_every_n_steps subsamples which control steps get logged at
    # all (default 1 = every step); shard_size_steps caps how many *logged* steps accumulate
    # in RAM before being flushed to a new shard file and cleared.
    log_every_n_steps: int = 1
    shard_size_steps: int = 200

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
    if args.episode_repeats is not None:
        raw_cfg.episode_repeats = args.episode_repeats
    if args.total_steps is not None:
        raw_cfg.total_steps = args.total_steps
    if args.output is not None:
        raw_cfg.output_path = args.output
    if args.log_every_n_steps is not None:
        raw_cfg.log_every_n_steps = args.log_every_n_steps
    if args.shard_size_steps is not None:
        raw_cfg.shard_size_steps = args.shard_size_steps

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

    # Scale down GPU PhysX buffers for small env counts (e.g. local debug collection runs
    # on laptop GPUs) — mirrors eval_ftr.py's own scaling exactly, never carried over into
    # this file originally. FTR_SIM_CFG's defaults are sized for 4096 envs on server GPUs;
    # this is what actually caused local ctrac runs to fail scene creation regardless of
    # env count or ContactSensor/terrain settings (confirmed by direct comparison against
    # eval_ftr.py, which already scales these and works locally).
    if cfg.num_robots <= 64:
        env_cfg.sim.physx.gpu_max_rigid_contact_count = 2 ** 20
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 20
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 18
        env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 22
    elif cfg.num_robots > 512:
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 27

    # module_name: ctrac drives CrossingEnv._get_observations/_get_rewards through
    # CTRACModule (and attaches the ContactSensor, ftr_env.py) — this is what makes the
    # logged obs/contacts CTRAC-shaped, independent of which policy is actually driving.
    for k, v in (cfg.env_cfg_overrides or {}).items():
        setattr(env_cfg, k, v)

    ftr_gym_env = gymnasium.make(cfg.task, cfg=env_cfg)
    env = FtrTorchRLEnv(ftr_gym_env, encoder_opts={}, device=device, shock_scale=cfg.env_cfg_overrides.get("shock_scale"))

    if cfg.total_steps is not None:
        n_steps = cfg.total_steps
        if cfg.episode_repeats is not None:
            logger.warning(f"Both total_steps ({cfg.total_steps}) and episode_repeats ({cfg.episode_repeats}) "
                            "are set — total_steps wins.")
    elif cfg.episode_repeats is not None:
        max_episode_length = ftr_gym_env.unwrapped.max_episode_length
        n_steps = cfg.episode_repeats * max_episode_length
        logger.info(f"episode_repeats={cfg.episode_repeats} x max_episode_length={max_episode_length} "
                    f"-> collecting {n_steps} control steps ({n_steps * cfg.num_robots} transitions across {cfg.num_robots} envs).")
    else:
        raise ValueError("Set either episode_repeats (recommended) or total_steps in the config — see "
                          "CTRACDatasetCollectionConfig's docstring.")

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

    # ---- load the demonstrator's VecNorm observation-normalization stats ----
    # MarvRLFlatObservation.supports_vecnorm = True — the demonstrator's actor expects
    # normalized input, not raw. Loading VecNorm directly and calling it each step would
    # keep UPDATING its running stats on this rollout's data (that's what it's designed to
    # do during training); to_observation_norm() freezes the loaded stats into a static
    # ObservationNorm affine transform instead — the same "use at inference, don't keep
    # adapting" pattern eval_ftr.py's own eval rollouts rely on via env.eval().
    # strict=False: the saved checkpoint's VecNorm also tracked "reward" stats (marv_rl's
    # vecnorm_on_reward: true), which this single-key, observation-only instance doesn't
    # declare and doesn't need.
    marv_rl_vecnorm = VecNorm(in_keys=[MARV_RL_OBS_KEY], **cfg.marv_rl_vecnorm_opts)
    marv_rl_vecnorm.load_state_dict(torch.load(cfg.marv_rl_vecnorm_weights_path, map_location=device), strict=False)
    marv_rl_obs_norm = marv_rl_vecnorm.to_observation_norm().to(device)
    logger.info(f"Loaded marv_rl demonstrator VecNorm from {cfg.marv_rl_vecnorm_weights_path}")

    # marv_rl_module is only ever used to call get_observations() directly against the live
    # underlying env — its reward/scanned-heightmap methods are never invoked, since
    # CTRACModule (the env's actual active module) owns reward/observation bookkeeping.
    marv_rl_module = MarvRLModule(env.ftr_env.unwrapped)

    obs_history_builder = CTRACObsHistory(num_envs=cfg.num_robots, history_len=cfg.cvae_history_len).to(device)

    out_dir = _resolve_ws_path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing dataset shards to {out_dir} (from output_path={cfg.output_path!r})")
    shard_idx = 0
    total_logged_steps = 0
    total_transitions = 0

    def _flush_shard(obs_history_buf, obs_buf, next_obs_buf):
        global shard_idx, total_transitions
        if not obs_buf:
            return
        shard = {
            "obs_history": torch.cat(obs_history_buf, dim=0),
            "obs": torch.cat(obs_buf, dim=0),
            "next_obs": torch.cat(next_obs_buf, dim=0),
        }
        shard_path = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(shard, shard_path)
        total_transitions += shard["obs"].shape[0]
        logger.info(f"Flushed shard {shard_idx} ({shard['obs'].shape[0]} transitions) -> {shard_path}")
        shard_idx += 1

    all_obs_history, all_obs, all_next_obs = [], [], []

    with torch.inference_mode():
        td = env.reset()
        pbar = tqdm(total=n_steps, desc="Collecting C-TRAC dataset", unit="steps")
        for step in range(n_steps):
            marv_obs = torch.nan_to_num(marv_rl_module.get_observations()["policy"].to(device))
            marv_td = TensorDict({MARV_RL_OBS_KEY: marv_obs}, batch_size=[cfg.num_robots], device=device)
            marv_td = marv_rl_obs_norm(marv_td)  # raw -> VecNorm-normalized, matching training
            marv_td = marv_rl_policy_operator(marv_td)
            action = marv_td["action"]

            ctrac_obs = td[OBS_KEY]  # this step's CTRAC-shaped packed observation
            partial = ctrac_obs[..., :PARTIAL_DIM]
            fresh_mask = partial[..., -1:]
            # obs_history_builder must see EVERY step (its ring buffer is temporally
            # stateful per env), even on steps that don't get logged — only the logging
            # below is subsampled by log_every_n_steps.
            obs_hist = obs_history_builder(partial, fresh_mask)

            step_td = TensorDict({"action": action}, batch_size=[cfg.num_robots], device=device)
            td = env.step(step_td)["next"]
            next_obs = td[OBS_KEY]

            if step % cfg.log_every_n_steps == 0:
                all_obs_history.append(obs_hist.cpu())
                all_obs.append(ctrac_obs.cpu())
                all_next_obs.append(next_obs.cpu())
                total_logged_steps += 1
                if total_logged_steps % cfg.shard_size_steps == 0:
                    _flush_shard(all_obs_history, all_obs, all_next_obs)
                    all_obs_history, all_obs, all_next_obs = [], [], []
            pbar.update(1)
        pbar.close()

    _flush_shard(all_obs_history, all_obs, all_next_obs)  # final partial shard
    logger.info(f"Saved {total_transitions} transitions across {shard_idx} shards to {out_dir}")

    import os as _os
    _os._exit(0)
