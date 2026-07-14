import dataclasses

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

# PPOExperimentConfig is a strict dataclass (no **kwargs catch-all). Several native
# trainers store extra, trainer-specific top-level keys in their saved config alongside
# the PPOExperimentConfig-schema ones -- e.g. C-TRAC's gamma/target_tau/alpha_init/.../
# n_iters (see experiments/ctrac/train.py's module docstring's "Additional (optional)
# top-level config keys" list), plus DQN's/C-REPS's own trainer-specific keys. Loading
# ANY such config's raw dict straight into PPOExperimentConfig(**config) crashes on the
# first unrecognised key ("unexpected keyword argument") -- this is what made every
# C-TRAC checkpoint UNDEPLOYABLE via this module before this fix (found while live-sim
# testing a C-TRAC checkpoint against flipper_policy_node: crashed at construction,
# before even reaching the policy/env). Filtering to only PPOExperimentConfig's own
# declared fields mirrors the exact pattern experiments/ctrac/train.py (and dqn/creps)
# already use when constructing the SAME dataclass from the SAME kind of config for
# training, and is a strict no-op for any config whose raw dict was already
# PPOExperimentConfig-shaped (plain PPO-trained configs are unaffected).
_PPO_CFG_FIELDS = {f.name for f in dataclasses.fields(PPOExperimentConfig)}


class PPOPolicyInferenceModule:
    """
    A module for policy inference using PPO to be integrated onto the physical robot.
    """

    def __init__(
        self, train_config_path: Path | str, policy_weights_path: Path | str, vecnorm_weights_path: Path | str | None = None, device: str = "cpu"
    ):
        config = OmegaConf.load(train_config_path)
        train_config = PPOExperimentConfig(**{k: v for k, v in config.items() if k in _PPO_CFG_FIELDS})
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

        # stash recurrent carries the actor declares as ("next", <key>) outputs (GRUModule
        # convention, also used by the HFC state machine) so the deploy node can feed them
        # back as plain kwargs on the next tick — without this, recurrent policies are
        # silently stateless at deployment (live-sim-found, 2026-07-14)
        self.recurrent_carry = {}
        for ok in getattr(self.actor_operator, "out_keys", []):
            if isinstance(ok, tuple) and len(ok) == 2 and ok[0] == "next":
                v = true_action_td.get(ok, None)
                if isinstance(v, torch.Tensor):
                    self.recurrent_carry[ok[1]] = v.squeeze(0).cpu().numpy()

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
