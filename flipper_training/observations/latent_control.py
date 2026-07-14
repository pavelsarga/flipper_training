from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from . import Observation, ObservationEncoder


class IdentityEncoder(ObservationEncoder):
    def __init__(self, output_dim: int):
        super(IdentityEncoder, self).__init__(output_dim)

    def forward(self, x):
        return x


@dataclass
class LatentControlParameter(Observation):
    supports_vecnorm = False

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        curr_params = getattr(self.env, "latent_control_params", None)
        if curr_params is None:
            raise ValueError("Latent control parameters not found in the environment. Make sure to set them before calling this observation.")
        return curr_params

    def from_realistic_world(self, tensordict) -> torch.Tensor:
        """Deployment path: the latent control parameter must be SUPPLIED — it is an external
        command (not a sensor reading), set during training by the objective/env. Sources, in
        order: ``tensordict["latent_control"]`` (the deploy node's ``latent_control`` ROS
        parameter), else ``env.latent_control_params`` if the rebuilt eval env carries one.
        Raising (rather than silently feeding zeros) is deliberate: a policy trained WITH a
        latent command is meaningless to deploy without saying which command to run.
        """
        lc = tensordict.get("latent_control", None)
        if lc is not None:
            return lc.to(self.env.device).view(1, self.dim).to(self.env.out_dtype)
        env_params = getattr(self.env, "latent_control_params", None)
        if env_params is not None:
            return env_params.view(1, self.dim).to(self.env.out_dtype)
        raise ValueError(
            "LatentControlParameter: no 'latent_control' in the deploy tensordict and the env has no "
            "latent_control_params — set the flipper_policy_node 'latent_control' parameter."
        )

    @property
    def dim(self) -> int:
        """
        The dimension of the observation vector.
        """
        return 1

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> IdentityEncoder:
        return IdentityEncoder(1)
