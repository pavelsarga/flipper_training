import torch
from torchrl.envs import Transform
from tensordict import TensorDict


class RawRewardSaveTransform(Transform):
    """
    Save the raw (pre-normalisation) reward in the tensordict as 'raw_reward'.
    """

    def __init__(self):
        super().__init__(in_keys=["reward"], out_keys=["reward", "raw_reward"])

    def _call(self, tensordict):
        # Clone so VecNorm cannot corrupt raw_reward when it normalises reward in-place.
        tensordict["raw_reward"] = tensordict["reward"].clone()
        return tensordict

    def transform_reward_spec(self, reward_spec):
        # Register raw_reward in the spec so TorchRL does not warn about unknown keys.
        reward_spec["raw_reward"] = reward_spec["reward"].clone()
        return reward_spec
