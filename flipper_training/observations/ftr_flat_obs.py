from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from flipper_training.observations import Observation, ObservationEncoder
from flipper_training.observations.heightmap import FTR_HeightmapEncoder
from flipper_training.policies import MLP


class FtrFlatEncoder(ObservationEncoder):
    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super().__init__(output_dim)
        self.mlp = MLP(in_dim=input_dim, out_dim=output_dim, activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class FtrCNNFlatEncoder(ObservationEncoder):
    HM_SIZE = (45, 21)   # 945 values
    STATE_DIM = 21       # 2+3+3+4+3+6

    def __init__(self, output_dim: int, cnn_output_dim: int = 128, input_dim: int = None, **mlp_kwargs):
        super().__init__(output_dim)
        self.cnn = FTR_HeightmapEncoder(self.HM_SIZE, output_dim=cnn_output_dim)
        self.mlp = MLP(in_dim=cnn_output_dim + self.STATE_DIM, out_dim=output_dim,
                       activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 966]
        hm = x[..., :945].view(*x.shape[:-1], 1, 45, 21)  # [N,1,45,21]
        state = x[..., 945:]                                # [N,21]
        latent = self.cnn(hm)                               # [N,cnn_output_dim]
        return self.mlp(torch.cat([latent, state], dim=-1))

@dataclass(kw_only=True)
class FtrFlatObservation(Observation):
    """966-dimensional flat observation vector produced by FTR-Benchmark CrossingEnv.

    This observation is filled directly by FtrTorchRLEnv; the __call__ method is never
    invoked during FTR training.
    """

    supports_vecnorm = True
    dim: int = 966  # 945 hmap + 2 orient + 3 lin_vel + 3 ang_vel + 4 joints + 3 goal_vec + 6 prev_action

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("FtrFlatObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> FtrFlatEncoder:
        return FtrCNNFlatEncoder(
            input_dim=self.dim,
            **self.encoder_opts,
        )
    
