from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchrl.modules import ActorCriticWrapper, ActorValueOperator
    from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv
    from torchrl.envs.transforms import Transform
    from marv_rl_training.observations import ObservationEncoder


class PolicyConfig(ABC):
    """
    Base class for all policy configurations.
    """

    @abstractmethod
    def create(self, env: "FtrTorchRLEnv", **kwargs) -> "tuple[ActorCriticWrapper | ActorValueOperator, list[dict], list[Transform]]":
        """
        Create a policy, value or action-value function wrapped into a TorchRL native object, optimizer groups from the configuration and optionally additional transforms for the environment.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class EncoderCombiner(torch.nn.Module):
    def __init__(self, encoders: dict[str, "ObservationEncoder"]):
        super(EncoderCombiner, self).__init__()
        self.encoders = torch.nn.ModuleDict(encoders)
        self.output_dim = sum([encoder.output_dim for encoder in encoders.values()])

    def forward(self, **kwargs):
        y_encs = []
        for key, encoder in self.encoders.items():
            y_enc = encoder(kwargs[key])
            y_encs.append(y_enc)
        return torch.cat(y_encs, dim=-1)


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | list[int],
        num_hidden: int,
        out_dim: int,
        layernorm: bool,
        activate_last_layer: bool = False,
        dropout: float | list[float | None] | None = None,
        activation: type[torch.nn.Module] = torch.nn.Tanh,
    ):
        super(MLP, self).__init__()
        if isinstance(hidden_dim, list):
            num_hidden = len(hidden_dim)
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_hidden
        if not isinstance(dropout, list):
            dropout = [dropout] * num_hidden
        if len(hidden_dim) != num_hidden:
            raise ValueError(f"hidden_dim should be a list of length {num_hidden}, but got {len(hidden_dim)}")
        if len(dropout) != num_hidden:
            raise ValueError(f"dropout should be a list of length {num_hidden}, but got {len(dropout)}")
        layers = []
        for i in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim[i]))
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(activation())
            if dropout[i] and dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            in_dim = hidden_dim[i]
        layers.append(nn.Linear(in_dim, out_dim))
        if activate_last_layer:
            if layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
