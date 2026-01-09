import torch
import numpy as np
from flipper_training.experiments.ppo.common import (
    prepare_env,
    make_transformed_env,
)
from tensordict import TensorDict
from flipper_training.experiments.ppo.config import PPOExperimentConfig, OmegaConf
from pathlib import Path
from flipper_training.utils.logutils import get_terminal_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type


class PPOPolicyInferenceModule:
    """
    A module for policy inference using PPO to be integrated onto the physical robot.
    """

    def __init__(
        self, train_config_path: Path | str, policy_weights_path: Path | str, vecnorm_weights_path: Path | str | None = None, device: str = "cpu"
    ):
        config = OmegaConf.load(train_config_path)
        train_config = PPOExperimentConfig(**config)
        train_config.device = device
        train_config.num_robots = 1  # Single robot for inference
        train_config.engine_compile_opts = None  # Disable engine compilation for inference on robot
        if train_config.objective_opts.get("cache_size") is not None:
            train_config.objective_opts["cache_size"] = 1  # Disable objective caching for inference on robot
        full_env, self.device, self.rng = prepare_env(train_config, mode="eval")
        policy_config = train_config.policy_config(**train_config.policy_opts)
        self.logger = get_terminal_logger("policy_inference_module")
        actor_value_wrapper, optim_groups, policy_transforms = policy_config.create(
            env=full_env,
            weights_path=policy_weights_path,
            device=self.device,
        )
        self.actor_operator = actor_value_wrapper.get_policy_operator()
        self.world_interface_env = full_env._to_realistic_env()
        self.env, self.vecnorm = make_transformed_env(self.world_interface_env, train_config, policy_transforms)
        self.logger.info(f"Environment transforms: {self.env.transform}")
        if vecnorm_weights_path is not None:
            self.vecnorm.load_state_dict(torch.load(vecnorm_weights_path, map_location=self.device))
            self.logger.info(f"Loaded VecNorm weights from {vecnorm_weights_path}")
        self.actor_operator.eval()
        self.env.eval()

    def infer_action(self, **kwargs) -> np.ndarray:
        """
        Infers the action for the given observation tensordict.

        Args:
            **kwargs: Keyword arguments to be passed to the environment's step function in a tensordict format.
        Returns:
            torch.Tensor: The inferred action tensor.
        """
        with (
            set_exploration_type(ExplorationType.DETERMINISTIC),
            torch.inference_mode(),
        ):
            world_td = TensorDict(
                {k: torch.tensor(v, device=self.device).unsqueeze(0) for k, v in kwargs.items()},
                batch_size=[1],
                device=self.device,
            )
            world_td.set("step_count", torch.zeros([1], dtype=torch.long, device=self.device))  # some envs need step_count
            env_td = self.env.step(world_td)
            true_action_td = self.actor_operator(env_td["next"])

        return true_action_td["action"].squeeze(0).cpu().numpy()


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    policy_weights_path = sys.argv[2]
    vecnorm_weights_path = sys.argv[3] if len(sys.argv) > 3 else None

    ppo_inference_module = PPOPolicyInferenceModule(
        train_config_path=config_path,
        policy_weights_path=policy_weights_path,
        vecnorm_weights_path=vecnorm_weights_path,
        device="cpu",
    )
