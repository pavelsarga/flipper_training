"""
FTR-compatible PPO training script for the flipper_training physics engine.

Reads a config in FTR format (ftr_config_new_v5.yaml style) and trains a policy
using the differentiable physics engine instead of Isaac Sim, while matching:
  • FTR observation format  (FtrCompatObservation, 968-D)
  • FTR CNN encoder         (FtrCNNFlatEncoder with state_dim=23)
  • FTR reward structure    (FtrCompatReward — all v5 terms)
  • FTR terrain             (FtrTerrainPatchGenerator for cur_mixed, etc.)

Usage:
    python -m flipper_training.experiments.ppo.train_ftr_compat \
        --local configs/ftr_compat_config_template.yaml

    # Override any key:
    python -m flipper_training.experiments.ppo.train_ftr_compat \
        --local configs/ftr_compat_config_template.yaml num_robots=64 device=cuda:0

The script accepts the same --local / --wandb flags as train.py.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from argparse import ArgumentParser
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import Compose, VecNorm, StepCounter, TransformedEnv
from torchrl.modules import ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import flipper_training  # noqa: F401 — registers OmegaConf resolvers
from flipper_training.configs.terrain_config import TerrainConfig
from flipper_training.configs.engine_config import PhysicsEngineConfig
from flipper_training.configs.robot_config import RobotModelConfig
from flipper_training.environment.env import Env
from flipper_training.environment.transforms import RawRewardSaveTransform
from flipper_training.heightmaps.ftr_terrain import FtrTerrainPatchGenerator
from flipper_training.heightmaps.flat import FlatHeightmapGenerator
from flipper_training.heightmaps.stairs import StairsHeightmapGenerator
from flipper_training.heightmaps.barrier import BarrierHeightmapGenerator
from flipper_training.heightmaps.mixed import MixedHeightmapGenerator
from flipper_training.observations.ftr_compat_obs import FtrCompatObservation
from flipper_training.rl_objectives.random_nav import RandomNavigationObjective
from flipper_training.rl_rewards.ftr_compat_reward import FtrCompatReward
from flipper_training.utils.logutils import RunLogger, get_terminal_logger, LocalRunReader, WandbRunReader
from flipper_training.utils.torch_utils import seed_all, set_device

if TYPE_CHECKING:
    from tensordict import TensorDict

# ---------------------------------------------------------------------------
# Terrain name → heightmap generator mapping
# ---------------------------------------------------------------------------
_FTR_MAP_DIR = Path(__file__).parents[5] / "src" / "FTR-benchmark" / "ftr_envs" / "assets" / "terrain" / "map"

_TERRAIN_MAP: dict[str, type] = {
    "cur_mixed":       FtrTerrainPatchGenerator,
    "cur_base":        FlatHeightmapGenerator,
    "cur_stairs_up":   StairsHeightmapGenerator,
    "cur_stairs_down": StairsHeightmapGenerator,
    "cur_steps_up":    BarrierHeightmapGenerator,
    "cur_steps_down":  BarrierHeightmapGenerator,
    "cur_waves":       MixedHeightmapGenerator,
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_heightmap_gen(cfg: DictConfig):
    """Return (heightmap_gen_cls, heightmap_gen_opts_dict) from config."""
    if OmegaConf.select(cfg, "heightmap_gen") is not None:
        # Explicit override in config (e.g. FtrTerrainPatchGenerator via ${cls:...})
        gen_cls = cfg.heightmap_gen
        opts = dict(OmegaConf.to_container(cfg.heightmap_gen_opts, resolve=True)) if OmegaConf.select(cfg, "heightmap_gen_opts") else {}
        return gen_cls, opts

    terrain_name = OmegaConf.select(cfg, "terrain", default="cur_mixed")
    gen_cls = _TERRAIN_MAP.get(terrain_name)
    if gen_cls is None:
        raise ValueError(
            f"Unknown terrain name {terrain_name!r}.  "
            f"Supported: {list(_TERRAIN_MAP.keys())}.  "
            f"Or set 'heightmap_gen' explicitly in the config."
        )

    # For FTR terrain types, automatically resolve the .map file path.
    opts: dict = dict(OmegaConf.to_container(cfg.heightmap_gen_opts, resolve=True)) if OmegaConf.select(cfg, "heightmap_gen_opts") else {}
    if issubclass(gen_cls, FtrTerrainPatchGenerator) and "map_path" not in opts:
        map_file = _FTR_MAP_DIR / f"{terrain_name}.map"
        if not map_file.exists():
            raise FileNotFoundError(
                f"FTR terrain map not found: {map_file}.  "
                f"Set 'heightmap_gen_opts.map_path' explicitly in the config."
            )
        opts["map_path"] = str(map_file)
    return gen_cls, opts


def _build_env(cfg: DictConfig, rng: torch.Generator) -> tuple[Env, torch.device]:
    """Build the FTR-compat environment from config."""
    device = set_device(cfg.device)

    # ── Terrain ──────────────────────────────────────────────────────────
    grid_res  = float(OmegaConf.select(cfg, "grid_res",  default=0.05))
    max_coord = float(OmegaConf.select(cfg, "max_coord", default=8.0))

    gen_cls, gen_opts = _resolve_heightmap_gen(cfg)
    gen = gen_cls(**gen_opts)
    x, y, z, extras = gen(grid_res, max_coord, cfg.num_robots, rng)

    world_opts = dict(OmegaConf.to_container(cfg.world_opts, resolve=True)) if OmegaConf.select(cfg, "world_opts") else {}
    world_defaults = {"k_stiffness": 30000.0, "k_friction_lon": 0.8, "k_friction_lat": 0.5}
    world_opts = world_defaults | world_opts

    terrain_cfg = TerrainConfig(
        x_grid=x, y_grid=y, z_grid=z,
        grid_extras=extras,
        grid_res=grid_res,
        max_coord=max_coord,
        **world_opts,
    ).to(device)

    # ── Robot & physics ───────────────────────────────────────────────────
    robot_opts = dict(OmegaConf.to_container(cfg.robot_model_opts, resolve=True)) if OmegaConf.select(cfg, "robot_model_opts") else {}
    robot_defaults = {
        "kind": "marv",
        "mesh_voxel_size": 0.01,
        "points_per_driving_part": 192,
        "points_per_body": 256,
    }
    robot_opts = robot_defaults | robot_opts
    robot_cfg = RobotModelConfig(**robot_opts).to(device)

    engine_opts = dict(OmegaConf.to_container(cfg.engine_opts, resolve=True)) if OmegaConf.select(cfg, "engine_opts") else {}
    engine_defaults = {"dt": 0.007, "damping_alpha": 5.0}
    engine_opts = engine_defaults | engine_opts
    phys_cfg = PhysicsEngineConfig(num_robots=cfg.num_robots, **engine_opts).to(device)

    # ── Objective ─────────────────────────────────────────────────────────
    engine_iters = int(OmegaConf.select(cfg, "engine_iters_per_env_step", default=4))
    obj_opts = dict(OmegaConf.to_container(cfg.objective_opts, resolve=True)) if OmegaConf.select(cfg, "objective_opts") else {}
    obj_defaults = {
        "higher_allowed": 1.5,
        "min_dist_to_goal": 3.0,
        "max_dist_to_goal": max_coord * 1.2,
        "goal_reached_threshold": 0.25,
        "start_z_offset": 0.3,
        "goal_z_offset": 0.0,
        "iteration_limit_factor": 1000,
        "max_feasible_roll": 0.5,
        "max_feasible_pitch": 0.5,
        "start_position_orientation": "towards_goal",
        "init_joint_angles": "random",
        "cache_size": cfg.num_robots * 200,
    }
    obj_opts = obj_defaults | obj_opts
    objective_factory = RandomNavigationObjective.make_factory(rng=rng, **obj_opts)

    # ── Reward ────────────────────────────────────────────────────────────
    env_cfg = dict(OmegaConf.to_container(cfg.env_cfg_overrides, resolve=True)) if OmegaConf.select(cfg, "env_cfg_overrides") else {}
    # Map scheduler sub-dicts to FtrCompatReward's scheduler params.
    sched_keys = {
        "step_penalty_scheduler":      "step_penalty_scheduler",
        "action_bonus_coef_scheduler": "action_bonus_coef_scheduler",
    }
    reward_opts: dict = {}
    for ftr_key, rew_key in sched_keys.items():
        sched_val = OmegaConf.select(cfg, ftr_key, default=None)
        if sched_val is not None:
            reward_opts[rew_key] = dict(OmegaConf.to_container(sched_val, resolve=True))
    # Copy all env_cfg_overrides that are valid FtrCompatReward fields.
    _rew_fields = FtrCompatReward.__dataclass_fields__
    for k, v in env_cfg.items():
        if k in _rew_fields and k not in {"env"}:
            reward_opts[k] = v
    # Apply default track_vel_max from robot config.
    reward_opts.setdefault("track_vel_max", robot_cfg.v_max)
    reward_factory = FtrCompatReward.make_factory(**reward_opts)

    # ── Observation ───────────────────────────────────────────────────────
    flipper_max_deg = float(env_cfg.get("flipper_pos_max_deg", 60.0))
    enc_opts = dict(OmegaConf.to_container(cfg.ftr_obs_encoder_opts, resolve=True)) if OmegaConf.select(cfg, "ftr_obs_encoder_opts") else {"output_dim": 128}
    obs_factory = FtrCompatObservation.make_factory(
        flipper_pos_max_deg=flipper_max_deg,
        encoder_opts=enc_opts,
    )

    # ── Environment ───────────────────────────────────────────────────────
    training_dtype = cfg.training_dtype if OmegaConf.select(cfg, "training_dtype") else torch.float32
    compile_opts = dict(OmegaConf.to_container(cfg.engine_compile_opts, resolve=True)) if OmegaConf.select(cfg, "engine_compile_opts") else {}
    env = Env(
        objective_factory=objective_factory,
        reward_factory=reward_factory,
        observation_factories=[obs_factory],
        terrain_config=terrain_cfg,
        physics_config=phys_cfg,
        robot_model_config=robot_cfg,
        device=device,
        batch_size=[cfg.num_robots],
        differentiable=False,
        engine_compile_opts=compile_opts if compile_opts else None,
        out_dtype=training_dtype,
        return_derivative=False,
        engine_iters_per_step=engine_iters,
        generator=rng,
    )
    return env, device


def _make_transformed_env(env: Env, cfg: DictConfig, policy_transforms) -> tuple[TransformedEnv, VecNorm]:
    vecnorm_keys = [o.name for o in env.observations if o.supports_vecnorm]
    if OmegaConf.select(cfg, "vecnorm_on_reward", default=True):
        vecnorm_keys.append("reward")
    vecnorm_opts = dict(OmegaConf.to_container(cfg.vecnorm_opts, resolve=True)) if OmegaConf.select(cfg, "vecnorm_opts") else {}
    vecnorm_opts.setdefault("decay", 0.99)
    vecnorm_opts.setdefault("eps", 0.0001)
    vecnorm = VecNorm(in_keys=vecnorm_keys, **vecnorm_opts)
    transforms = [StepCounter(), RawRewardSaveTransform()]
    transforms += policy_transforms
    transforms.append(vecnorm)
    return TransformedEnv(env, Compose(*transforms)), vecnorm


def _train_step_log(rollout_td, loss_td, grad_norm, optim) -> dict[str, float]:
    return {
        "train/mean_action_sample_log_prob": rollout_td.get("sample_log_prob", rollout_td.get("action_log_prob")).mean().item(),
        "train/mean_critic_loss":            loss_td["loss_critic"].mean().item(),
        "train/mean_objective_loss":         loss_td["loss_objective"].mean().item(),
        "train/mean_entropy_loss":           loss_td["loss_entropy"].mean().item(),
        "train/mean_entropy":                loss_td["entropy"].mean().item(),
        "train/mean_kl_approx":              loss_td["kl_approx"].mean().item(),
        "train/mean_clip_fraction":          loss_td["clip_fraction"].mean().item(),
        "train/mean_advantage":              rollout_td["advantage"].mean().item(),
        "train/std_advantage":               rollout_td["advantage"].std().item(),
        "train/total_grad_norm":             grad_norm.item(),
        **{f"train/{g['name']}_lr": g["lr"] for g in optim.param_groups},
    }


def _eval_log(eval_rollout) -> dict[str, float]:
    last_step_count = eval_rollout["step_count"][:, -1].float()
    last_succeeded = eval_rollout["next", "succeeded"][:, -1].float().mean().item()
    last_failed    = eval_rollout["next", "failed"][:, -1].float().mean().item()
    d = {
        "eval/mean_step_reward": eval_rollout["next", "reward"].mean().item(),
        "eval/mean_step_count":  last_step_count.mean().item(),
        "eval/pct_succeeded":    last_succeeded,
        "eval/pct_failed":       last_failed,
        "eval/pct_truncated":    1 - last_succeeded - last_failed,
    }
    if "raw_reward" in eval_rollout["next"]:
        d["eval/mean_raw_step_reward"] = eval_rollout["next", "raw_reward"].mean().item()
    return d


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FtrCompatTrainer:
    """PPO trainer that reads FTR-style configs and uses the flipper_training physics engine."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = get_terminal_logger("ftr_compat_train")
        self.run_logger = RunLogger(
            train_config=cfg,
            category="ppo",
            use_wandb=bool(OmegaConf.select(cfg, "use_wandb", default=False)),
            use_tensorboard=bool(OmegaConf.select(cfg, "use_tensorboard", default=False)),
            step_metric_name="collected_frames",
        )
        rng = seed_all(cfg.seed)

        # Build environment
        base_env, self.device = _build_env(cfg, rng)
        self.reward: FtrCompatReward = base_env.reward  # type: ignore[assignment]

        # Build policy
        training_dtype = OmegaConf.select(cfg, "training_dtype", default=torch.float32)
        policy_cls = cfg.policy_config
        policy_opts = dict(OmegaConf.to_container(cfg.policy_opts, resolve=True))
        policy_cfg_obj = policy_cls(**policy_opts)
        self.actor_value_wrapper, self.optim_groups, policy_transforms = policy_cfg_obj.create(
            env=base_env,
            weights_path=OmegaConf.select(cfg, "policy_weights_path", default=None),
            device=self.device,
        )
        self.actor_operator = self.actor_value_wrapper.get_policy_operator()
        self.value_operator = self.actor_value_wrapper.get_value_operator()

        # Wrap environment with transforms
        self.env, self.vecnorm = _make_transformed_env(base_env, cfg, policy_transforms)
        vn_path = OmegaConf.select(cfg, "vecnorm_weights_path", default=None)
        if vn_path is not None:
            self.vecnorm.load_state_dict(torch.load(vn_path, map_location=self.device), strict=False)

        # Collector
        dc_opts = dict(OmegaConf.to_container(cfg.data_collector_opts, resolve=True)) if OmegaConf.select(cfg, "data_collector_opts") else {}
        dc_opts.setdefault("split_trajs", False)
        dc_opts.setdefault("exploration_type", "RANDOM")
        self.collector = SyncDataCollector(
            self.env,
            self.actor_operator,
            frames_per_batch=cfg.time_steps_per_batch * cfg.num_robots,
            total_frames=cfg.total_frames,
            **dc_opts,
            device=self.device,
        )

        # Replay buffer
        iter_size = cfg.time_steps_per_batch * cfg.num_robots
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=iter_size, ndim=1, device=self.device, compilable=True),
            sampler=SamplerWithoutReplacement(drop_last=True, shuffle=True),
            batch_size=cfg.frames_per_sub_batch,
            dim_extend=0,
            compilable=True,
        )

        # PPO modules
        gae_opts = dict(OmegaConf.to_container(cfg.gae_opts, resolve=True))
        self.advantage_module = GAE(
            **gae_opts, value_network=self.value_operator,
            time_dim=1, device=self.device, differentiable=False,
        ).to(training_dtype)

        ppo_opts = dict(OmegaConf.to_container(cfg.ppo_opts, resolve=True))
        self.loss_module = ClipPPOLoss(
            self.actor_operator,
            self.actor_value_wrapper.get_value_head() if isinstance(self.actor_value_wrapper, ActorValueOperator) else self.value_operator,
            **ppo_opts,
        ).to(training_dtype)

        # Optimizer & scheduler
        optim_cls = cfg.optimizer
        optim_opts = dict(OmegaConf.to_container(cfg.optimizer_opts, resolve=True)) if OmegaConf.select(cfg, "optimizer_opts") else {}
        self.optim = optim_cls(self.optim_groups, **optim_opts)

        sched_cls = cfg.scheduler
        sched_opts = dict(OmegaConf.to_container(cfg.scheduler_opts, resolve=True)) if OmegaConf.select(cfg, "scheduler_opts") else {}
        self.scheduler = sched_cls(self.optim, **sched_opts)

        # Eval config
        self.max_eval_steps = int(OmegaConf.select(cfg, "max_eval_steps", default=1000))
        self.eval_repeats_after_training = int(OmegaConf.select(cfg, "eval_repeats_after_training", default=10))
        self.eval_and_save_every = int(cfg.eval_and_save_every)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.clip_grad_norm_p = OmegaConf.select(cfg, "clip_grad_norm_p", default=2)

        self.logger.info("FtrCompatTrainer initialised")
        print(OmegaConf.to_yaml(cfg, sort_keys=True))

    def train(self):
        try:
            self._train()
            post_eval = self._post_training_eval()
        except KeyboardInterrupt:
            self.logger.info("Training interrupted.")
            post_eval = None
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            traceback.print_exception(e)
            raise
        finally:
            self.run_logger.close()
        return post_eval

    def _train(self):
        iter_size = self.cfg.time_steps_per_batch * self.cfg.num_robots
        if iter_size % self.cfg.frames_per_sub_batch != 0:
            raise ValueError(
                f"time_steps_per_batch * num_robots must be divisible by frames_per_sub_batch, "
                f"got {iter_size} % {self.cfg.frames_per_sub_batch} = {iter_size % self.cfg.frames_per_sub_batch}"
            )
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        pbar = tqdm(total=self.cfg.total_frames, desc="FtrCompat PPO", unit="frames", leave=False)
        for i, tensordict_data in enumerate(self.collector):
            total_frames = (i + 1) * iter_size
            pbar.update(iter_size)

            # Anneal reward schedulers.
            self.reward.scheduler_step(i)

            tensordict_data.pop(Env.STATE_KEY, None)
            tensordict_data.pop(("next", Env.STATE_KEY), None)
            self.actor_operator.train()
            self.value_operator.train()
            self.env.train()

            for _ in range(self.cfg.epochs_per_batch):
                self.advantage_module(tensordict_data)
                self.replay_buffer.extend(tensordict_data.reshape(-1))
                for _ in range(iter_size // self.cfg.frames_per_sub_batch):
                    sub_batch = self.replay_buffer.sample()
                    loss_vals = self.loss_module(sub_batch)
                    loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    loss.backward()
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor_value_wrapper.parameters(),
                            self.max_grad_norm,
                            error_if_nonfinite=True,
                            norm_type=self.clip_grad_norm_p,
                        )
                    except Exception as e:
                        self.logger.error(f"Gradient norm error: {e}")
                        raise
                    self.optim.step()
                    self.optim.zero_grad()

            self.scheduler.step()
            log = _train_step_log(tensordict_data, loss_vals, grad_norm, self.optim)

            if i % self.eval_and_save_every == 0:
                eval_log = self._eval()
                log.update(eval_log)
                self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{total_frames}")
                self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{total_frames}")

            self.run_logger.log_data(log, total_frames)

        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), "policy_final")
        self.run_logger.save_weights(self.vecnorm.state_dict(), "vecnorm_final")
        self.run_logger.save_weights(self.actor_value_wrapper.state_dict(), f"policy_step_{self.cfg.total_frames}")
        self.run_logger.save_weights(self.vecnorm.state_dict(), f"vecnorm_step_{self.cfg.total_frames}")

    def _eval(self) -> dict[str, float]:
        self.env.eval()
        self.actor_operator.eval()
        with (
            set_exploration_type(ExplorationType.DETERMINISTIC),
            torch.inference_mode(),
        ):
            rollout = self.env.rollout(self.max_eval_steps, self.actor_operator, break_when_all_done=True, auto_reset=True)
        results = _eval_log(rollout)
        del rollout
        return results

    def _post_training_eval(self) -> dict[str, float]:
        self.logger.info(f"Running final evaluation ({self.eval_repeats_after_training} repeats).")
        avg = self._eval()
        for _ in range(self.eval_repeats_after_training - 1):
            for k, v in self._eval().items():
                avg[k] += v
        for k in avg:
            avg[k] /= self.eval_repeats_after_training
        print(f"\nFinal evaluation ({self.eval_repeats_after_training} repeats):")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
        return avg


# ---------------------------------------------------------------------------
# Config loading (same interface as train.py)
# ---------------------------------------------------------------------------

def parse_and_load_config() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--local", type=Path, required=False, default=None)
    parser.add_argument("--wandb", type=Path, required=False, default=None)
    parser.add_argument("--weight_step", type=str, required=False, default=None)
    args, unknown = parser.parse_known_args()
    if args.local is None and args.wandb is None:
        raise ValueError("Either --local or --wandb must be provided.")
    if args.local is not None and "yaml" in args.local.name:
        cfg = OmegaConf.load(args.local)
    else:
        reader = (
            WandbRunReader(args.wandb, category="ppo") if args.wandb
            else LocalRunReader(Path("runs/ppo") / args.local)
        )
        cfg = reader.load_config()
        if args.weight_step is not None:
            step = args.weight_step
            tag = f"policy_step_{step}" if step.isdigit() else f"policy_{step}"
            cfg["policy_weights_path"] = reader.get_weights_path(tag)
            cfg["vecnorm_weights_path"] = reader.get_weights_path(tag.replace("policy", "vecnorm"))
    cli_cfg = OmegaConf.from_dotlist(unknown)
    return OmegaConf.merge(cfg, cli_cfg)


if __name__ == "__main__":
    config = parse_and_load_config()
    trainer = FtrCompatTrainer(config)
    trainer.train()
