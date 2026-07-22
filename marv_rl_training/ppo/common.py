from typing import TYPE_CHECKING

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from torchrl.envs import Compose, VecNorm, StepCounter, TransformedEnv, Transform

from marv_rl_training.environment.transforms import RawRewardSaveTransform
from marv_rl_training.utils.logutils import LocalRunReader, WandbRunReader

if TYPE_CHECKING:
    from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv
    from marv_rl_training.ppo.train_ftr import FtrPPOConfig


def make_transformed_env(env: "FtrTorchRLEnv", train_config: "FtrPPOConfig", policy_transforms: list[Transform]) -> tuple[TransformedEnv, VecNorm]:
    vecnorm_keys = [o.name for o in env.observations if o.supports_vecnorm]
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
