from typing import TYPE_CHECKING, Literal, Tuple

import torch
from collections import defaultdict
from torchrl.envs.utils import check_env_specs

from flipper_training.configs.terrain_config import TerrainConfig
from flipper_training.configs.engine_config import PhysicsEngineConfig
from flipper_training.configs.robot_config import RobotModelConfig
from flipper_training.environment.env import Env
from flipper_training.utils.torch_utils import seed_all, set_device
from argparse import ArgumentParser
from pathlib import Path
from flipper_training.experiments.ppo.config import PPOExperimentConfig
from omegaconf import DictConfig, OmegaConf
from flipper_training.utils.logutils import LocalRunReader, WandbRunReader
from flipper_training.environment.transforms import RawRewardSaveTransform
from torchrl.envs import Compose, VecNorm, StepCounter, TransformedEnv, Transform

if TYPE_CHECKING:
    from config import PPOExperimentConfig
    from tensordict import TensorDict

EVAL_LOG_OPT = {
    "precision": {
        "step_reward": 4,
        "step_count": 0,
        "succeeded": 3,
        "failed": 3,
        "truncated": 3,
    },
    "groups": [
        ["step_reward"],
        ["step_count"],
        ["succeeded", "failed", "truncated"],
    ],
    "group_labels": {
        0: "Eval per-step reward",
        1: "Eval step count",
        2: "Eval state statistics",
    },
}


def prepare_configs(rng: torch.Generator, cfg: "PPOExperimentConfig") -> Tuple[TerrainConfig, PhysicsEngineConfig, RobotModelConfig, torch.device]:
    heightmap_gen = cfg.heightmap_gen(**cfg.heightmap_gen_opts if cfg.heightmap_gen_opts else {})
    x, y, z, extras = heightmap_gen(cfg.grid_res, cfg.max_coord, cfg.num_robots, rng)
    terrain_config = TerrainConfig(
        x_grid=x,
        y_grid=y,
        z_grid=z,
        grid_extras=extras,
        **cfg.world_opts,
        grid_res=cfg.grid_res,
        max_coord=cfg.max_coord,
    )
    robot_model = RobotModelConfig(**cfg.robot_model_opts)
    physics_config = PhysicsEngineConfig(num_robots=cfg.num_robots, **cfg.engine_opts)
    device = set_device(cfg.device)
    robot_model.to(device)
    terrain_config.to(device)
    physics_config.to(device)
    return terrain_config, physics_config, robot_model, device


def prepare_env(
    train_config: "PPOExperimentConfig", mode: Literal["train", "eval"], force_return_derivative: bool = False
) -> tuple["Env", torch.device, torch.Generator]:
    """Args:
    force_return_derivative: force ``Env(return_derivative=True)`` regardless of
        ``mode``. Normally only ``mode="eval"`` needs ``PhysicsStateDer``
        (``xdd``/``f_spring``/``f_friction``/...) in the observation
        tensordict; ``experiments/creps/train.py`` also needs it during
        TRAINING (``mode="train"``) for its Sec III hard-impact safety
        criterion, hence this separate flag instead of misusing
        ``mode="eval"`` (which would be confusing for a training loop and
        could silently start meaning something else if "eval" mode ever grows
        other side effects here).
    """
    # Init configs and RL-related objects
    rng = seed_all(train_config.seed)
    terrain_config, physics_config, robot_model, device = prepare_configs(rng, train_config)
    # Create environment
    base_env = Env(
        objective_factory=train_config.objective.make_factory(
            **train_config.objective_opts,
            rng=rng,
        ),
        reward_factory=train_config.reward.make_factory(**train_config.reward_opts),
        observation_factories=[o["cls"].make_factory(**(o.get("opts", None) or {})) for o in train_config.observations],
        terrain_config=terrain_config,
        physics_config=physics_config,
        robot_model_config=robot_model,
        device=device,
        batch_size=[train_config.num_robots],
        differentiable=False,
        engine_compile_opts=train_config.engine_compile_opts,
        out_dtype=train_config.training_dtype,
        return_derivative=mode == "eval" or force_return_derivative,  # needed for evaluation, or when explicitly forced
        engine_iters_per_step=train_config.engine_iters_per_env_step,
        generator=rng,
        emit_clean_observations=train_config.emit_clean_observations,
    )
    check_env_specs(base_env)
    return base_env, device, rng


def make_formatted_str_lines(eval_log: dict[str, int | float], format_dict: dict, init_eval_log: dict[str, int | float] | None = None) -> list[str]:
    var_dict = defaultdict(list)
    for key, value in eval_log.items():
        statistic = key.split("/")[1]  # remove the prefix specifying the type of value
        var_name = statistic.split("_", maxsplit=1)[-1]  # remove the prefix specifying the type of value
        if var_name in format_dict["precision"]:
            prec = format_dict["precision"].get(var_name, 4)
            s = f"{key}: {value:.{prec}f}"
            if init_eval_log is not None and key in init_eval_log:
                s += f" (init {init_eval_log[key]:.{prec}f})"
            var_dict[var_name].append(s)

    lines = []
    for i, group in enumerate(format_dict["groups"]):
        s = f"{format_dict['group_labels'][i]}: "
        for var_name in group:
            for entry in var_dict[var_name]:
                s += f"{entry} "
        lines.append(s)
    return lines


def log_from_eval_rollout(eval_rollout: "TensorDict") -> dict[str, int | float]:
    """
    Computes the statistics from the evaluation rollout and returns them as a dictionary.
    """
    last_step_count = eval_rollout["step_count"][:, -1].float()
    last_succeeded_mean = eval_rollout["next", "succeeded"][:, -1].float().mean().item()
    last_failed_mean = eval_rollout["next", "failed"][:, -1].float().mean().item()
    l = {
        "eval/mean_step_reward": eval_rollout["next", "reward"].mean().item(),
        "eval/max_step_reward": eval_rollout["next", "reward"].max().item(),
        "eval/min_step_reward": eval_rollout["next", "reward"].min().item(),
        "eval/mean_step_count": last_step_count.mean().item(),
        "eval/max_step_count": last_step_count.max().item(),
        "eval/min_step_count": last_step_count.min().item(),
        "eval/pct_succeeded": last_succeeded_mean,
        "eval/pct_failed": last_failed_mean,
        "eval/pct_truncated": 1 - last_succeeded_mean - last_failed_mean,  # eithered none of the states at the end of the rollout
    }
    if "raw_reward" in eval_rollout["next"]:
        l["eval/mean_raw_step_reward"] = eval_rollout["next", "raw_reward"].mean().item()
        l["eval/max_raw_step_reward"] = eval_rollout["next", "raw_reward"].max().item()
        l["eval/min_raw_step_reward"] = eval_rollout["next", "raw_reward"].min().item()
    return l


def make_transformed_env(env: "Env", train_config: "PPOExperimentConfig", policy_transforms: list[Transform]) -> tuple[TransformedEnv, VecNorm]:
    vecnorm_keys = [o.name for o in env.observations if o.supports_vecnorm]
    if getattr(env, "emit_clean_observations", False):
        # Normalise the "clean" mirror (see Env.emit_clean_observations / C-TRAC's denoising
        # target) with the SAME vecnorm-eligibility as its noisy counterpart, so the C-VAE's
        # reconstruction target stays in the same representation the actor/critic actually see.
        # Each nested ("clean", name) key gets its OWN running mean/std (VecNorm tracks stats
        # per in_key) rather than literally sharing the noisy key's statistics -- since
        # observation noise is zero-mean, the two independently-tracked estimates converge to
        # (and in practice track very closely) the same statistics; a deliberate, documented
        # approximation rather than plumbing a stat-sharing mechanism VecNorm doesn't expose.
        vecnorm_keys += [("clean", o.name) for o in env.observations if o.supports_vecnorm]
    if train_config.vecnorm_on_reward:
        vecnorm_keys.append("reward")
    vecnorm = VecNorm(
        in_keys=vecnorm_keys,
        **train_config.vecnorm_opts,
    )
    transforms = [StepCounter()]
    transforms += [RawRewardSaveTransform()]
    transforms += [t["cls"](**(t["opts"] or {})) for t in train_config.extra_env_transforms]
    transforms += policy_transforms
    transforms.append(vecnorm)
    return TransformedEnv(env, Compose(*transforms)), vecnorm


def download_config_and_paths(reader: WandbRunReader | LocalRunReader, weight_step: str | None) -> "DictConfig":
    run_omegaconf = reader.load_config()
    if not isinstance(run_omegaconf, DictConfig):
        raise ValueError("Config must be a DictConfig")
    if weight_step is not None:
        run_omegaconf["policy_weights_path"] = reader.get_weights_path(
            f"policy_step_{weight_step}" if weight_step.isdigit() else f"policy_{weight_step}"
        )
        run_omegaconf["vecnorm_weights_path"] = reader.get_weights_path(
            f"vecnorm_step_{weight_step}" if weight_step.isdigit() else f"vecnorm_{weight_step}"
        )
    return run_omegaconf


def parse_and_load_config() -> "DictConfig":
    parser = ArgumentParser()
    parser.add_argument("--local", type=Path, required=False, default=None, help="Path to the local run directory")
    parser.add_argument("--wandb", type=Path, required=False, default=None, help="Name of the run to evaluate")
    parser.add_argument("--weight_step", type=str, required=False, help="Step from which to load the weights", default=None)
    args, unknown = parser.parse_known_args()
    if args.local is None and args.wandb is None:
        raise ValueError("Either --local or --wandb must be provided")
    if args.local is not None and args.wandb is not None:
        raise ValueError("Only one of --local or --wandb must be provided")
    if args.local is not None and "yaml" in args.local.name:
        parsed_omegaconf = OmegaConf.load(args.local)
    else:
        run_reader = WandbRunReader(args.wandb, category="ppo") if args.wandb else LocalRunReader(Path("runs/ppo") / args.local)
        parsed_omegaconf = download_config_and_paths(run_reader, args.weight_step)
    cli_omegaconf = OmegaConf.from_dotlist(unknown)
    merged_omegaconf = OmegaConf.merge(parsed_omegaconf, cli_omegaconf)
    return merged_omegaconf
