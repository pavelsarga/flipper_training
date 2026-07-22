import torch

from marv_rl_training.observations import ObservationEncoder
from marv_rl_training.observations.heightmap import HeightmapEncoder
from marv_rl_training.policies import MLP


class FtrFlatEncoder(ObservationEncoder):
    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super().__init__(output_dim)
        self.mlp = MLP(in_dim=input_dim, out_dim=output_dim, activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class FtrFlipperStyleEncoder(ObservationEncoder):
    """Encodes FTR 4117-D flat obs using the flipper_training split architecture.

    CNN for the 64*64 heightmap → cnn_output_dim, MLP with layernorm for the 21-D state
    vector → state_output_dim.  Both branches are concatenated; output_dim must equal
    cnn_output_dim + state_output_dim.  No fusion MLP — matches the native flipper_training
    pattern where each observation encoder feeds directly into the actor/critic.
    """

    HM_DIM = 4096
    STATE_DIM = 15
    HM_SIZE = (64, 64)

    def __init__(
        self,
        output_dim: int,
        cnn_output_dim: int = 64,
        state_output_dim: int = 64,
        state_hidden_dim: int = 64,
        state_num_hidden: int = 2,
        state_layernorm: bool = True,
        input_dim: int | None = None,  # noqa: ARG002 — accepted for API parity with MarvRLCNNFlatEncoder
    ):
        assert output_dim == cnn_output_dim + state_output_dim, (
            f"output_dim ({output_dim}) must equal cnn_output_dim ({cnn_output_dim}) + state_output_dim ({state_output_dim})"
        )
        super().__init__(output_dim)
        self.cnn = HeightmapEncoder(self.HM_SIZE, output_dim=cnn_output_dim)
        self.state_encoder = MLP(
            in_dim=self.STATE_DIM,
            out_dim=state_output_dim,
            hidden_dim=state_hidden_dim,
            num_hidden=state_num_hidden,
            layernorm=state_layernorm,
            activate_last_layer=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hm = x[..., : self.HM_DIM].view(*x.shape[:-1], 1, 64, 64)
        state = x[..., self.HM_DIM : self.HM_DIM+self.STATE_DIM]
        return torch.cat([self.cnn(hm), self.state_encoder(state)], dim=-1)