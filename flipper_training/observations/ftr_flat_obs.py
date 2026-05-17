from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from flipper_training.observations import Observation, ObservationEncoder
from flipper_training.observations.heightmap import FTR_HeightmapEncoder, HeightmapEncoder
from flipper_training.policies import MLP


class FtrFlatEncoder(ObservationEncoder):
    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super().__init__(output_dim)
        self.mlp = MLP(in_dim=input_dim, out_dim=output_dim, activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class FtrCNNFlatEncoder(ObservationEncoder):
    HM_SIZE = (45, 21)   # 945 values
    HM_DIM = 945

    def __init__(self, output_dim: int, state_dim: int = 21, cnn_output_dim: int = 128, state_proj_dim: int | None = None, input_dim: int | None = None, **mlp_kwargs):
        super().__init__(output_dim)
        self.state_dim = state_dim
        self.cnn = FTR_HeightmapEncoder(self.HM_SIZE, output_dim=cnn_output_dim)
        # Optional learned projection to bring state up to a comparable scale before fusion.
        # Without this, the 21-dim state is structurally dominated by the 128-dim CNN output.
        if state_proj_dim is not None:
            self.state_proj = MLP(in_dim=state_dim, out_dim=state_proj_dim, hidden_dim=state_proj_dim, num_hidden=1, layernorm=False, activate_last_layer=True)
            fusion_dim = cnn_output_dim + state_proj_dim
        else:
            self.state_proj = None
            fusion_dim = cnn_output_dim + state_dim
        self.mlp = MLP(in_dim=fusion_dim, out_dim=output_dim, activate_last_layer=True, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, HM_DIM + state_dim]
        hm = x[..., :self.HM_DIM].view(*x.shape[:-1], 1, 45, 21)  # [N,1,45,21]
        state = x[..., self.HM_DIM:]                                # [N, state_dim]
        latent = self.cnn(hm)                                        # [N,cnn_output_dim]
        if self.state_proj is not None:
            state = self.state_proj(state)                           # [N, state_proj_dim]
        return self.mlp(torch.cat([latent, state], dim=-1))

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
        input_dim: int | None = None,  # noqa: ARG002 — accepted for API parity with FtrCNNFlatEncoder
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


@dataclass(kw_only=True)
class FtrFlatObservation(Observation):
    """Flat observation vector produced by FTR-Benchmark CrossingEnv.

    Default (flipper_style=False): 966-D  (945 hmap + 21 state, 6 prev_action)
    flipper_style=True:           4119-D  (4096 hmap + 15 state + 8 prev_action)

    This observation is filled directly by FtrTorchRLEnv; the __call__ method is never
    invoked during FTR training.
    """

    supports_vecnorm = True
    dim: int = 966  # default; overridden in __post_init__ for flipper_style

    def __post_init__(self):
        opts = self.encoder_opts or {}
        if opts.get("flipper_style", False):
            self.dim = 4119  # 4096 hmap + 15 state + 8 prev_action (4 track vels + 4 flipper angles)

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("FtrFlatObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        opts: dict = {**(self.encoder_opts or {})}
        if opts.pop("flipper_style", False):
            return FtrFlipperStyleEncoder(input_dim=4117, **opts)
        return FtrCNNFlatEncoder(input_dim=self.dim, **opts)
    
