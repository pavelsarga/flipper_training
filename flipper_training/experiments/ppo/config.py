from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from flipper_training.policies import PolicyConfig

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from omegaconf import DictConfig
    from torchrl.envs import Transform
    from flipper_training.observations import Observation
    from flipper_training.rl_objectives import BaseObjective
    from flipper_training.rl_rewards.rewards import Reward
    from flipper_training.heightmaps import BaseHeightmapGenerator

import hashlib
from typing import Type, TypedDict, List
from omegaconf import OmegaConf

import torch


class ObservationConfig(TypedDict):
    cls: "Type[Observation]"
    opts: dict[str, Any] | None


class EnvTransformConfig(TypedDict):
    cls: "Type[Transform]"
    opts: dict[str, Any] | None


def hash_omegaconf(omegaconf: "DictConfig") -> str:
    """Hashes the omegaconf config to a string."""
    s = OmegaConf.to_yaml(omegaconf, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class PPOExperimentConfig:
    name: str
    comment: str
    training_dtype: torch.dtype
    use_wandb: bool
    use_tensorboard: bool
    engine_iters_per_env_step: int
    eval_repeats_after_training: int
    seed: int
    device: str
    num_robots: int
    grid_res: float
    max_coord: float
    robot_model_opts: dict[str, Any]
    optimizer: "Type[Optimizer]"
    max_grad_norm: float
    total_frames: int
    time_steps_per_batch: int
    heightmap_gen: "Type[BaseHeightmapGenerator]"
    world_opts: dict[str, float]
    engine_opts: dict[str, Any]
    observations: List[ObservationConfig]
    objective: "Type[BaseObjective]"
    objective_opts: dict[str, Any]
    reward: "Type[Reward]"
    reward_opts: dict[str, Any]
    eval_and_save_every: int
    max_eval_steps: int
    epochs_per_batch: int
    frames_per_sub_batch: int
    gae_opts: dict[str, Any]
    ppo_opts: dict[str, Any]
    data_collector_opts: dict[str, Any]
    policy_config: type[PolicyConfig]
    policy_opts: dict[str, Any]
    vecnorm_opts: dict[str, Any]
    vecnorm_on_reward: bool
    scheduler: "Type[LRScheduler]"
    scheduler_opts: dict[str, Any]
    gae_compile_opts: str | None = None
    ppo_compile_opts: str | None = None
    policy_weights_path: str | None = None
    vecnorm_weights_path: str | None = None
    clip_grad_norm_p: str | int = 2
    extra_env_transforms: list[EnvTransformConfig] = field(default_factory=list)
    optimizer_opts: dict[str, Any] = field(default_factory=dict)
    heightmap_gen_opts: dict[str, Any] = field(default_factory=dict)
    engine_compile_opts: dict[str, Any] = field(default_factory=dict)
    # Also compute+emit a noiseless mirror of every observation under a "clean" sub-key each step
    # (-> Env.emit_clean_observations). Off by default (byte-identical to before this field
    # existed); C-TRAC's "full" config turns it on to supply the C-VAE denoising loss its clean
    # reconstruction target (Pan et al. 2025 Sec. IV-C) -- see ctrac_policy.py /
    # experiments/ctrac/train.py. No effect on any other experiment/trainer.
    emit_clean_observations: bool = False
