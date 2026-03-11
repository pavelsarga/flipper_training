from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from flipper_training.observations import Observation, ObservationEncoder
from flipper_training.policies import MLP


class FtrFlatEncoder(ObservationEncoder):
    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super().__init__(output_dim)
        self.mlp = MLP(in_dim=input_dim, out_dim=output_dim, activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@dataclass(kw_only=True)
class FtrFlatObservation(Observation):
    """115-dimensional flat observation vector produced by FTR-Benchmark CrossingEnv.

    This observation is filled directly by FtrTorchRLEnv; the __call__ method is never
    invoked during FTR training.
    """

    supports_vecnorm = True
    dim: int = 115

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("FtrFlatObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> FtrFlatEncoder:
        return FtrFlatEncoder(
            input_dim=self.dim,
            **self.encoder_opts,
        )
