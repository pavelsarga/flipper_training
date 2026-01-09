from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Callable
from functools import wraps

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer

if TYPE_CHECKING:
    from flipper_training.environment.env import Env
    from tensordict import TensorDictBase


class ObservationEncoder(torch.nn.Module):
    """
    Abstract class for observation encoders.

    Args:
        output_dim (int): The output dimension of the encoder.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim


@dataclass(kw_only=True)
class Observation(ABC):
    """
    Abstract class for observation generators.

    Args:
        env (Env): The environment.
    """

    env: "Env"
    encoder_opts: dict | None = None
    apply_noise: bool = False
    noise_scale: float | torch.Tensor | None = None
    supports_vecnorm: ClassVar[bool] = NotImplemented

    @property
    def name(self) -> str:
        """
        Name of the observation generator.
        """
        return self.__class__.__name__

    @abstractmethod
    def __call__(
        self, prev_state: PhysicsState, action: torch.Tensor, prev_state_der: PhysicsStateDer, curr_state: PhysicsState
    ) -> torch.Tensor | TensorDict:
        """
        Generate observations from the current state of the environment.

        Args:
            prev_state (PhysicsState): The previous state of the environment.
            action (torch.Tensor): The action taken in previous state.
            prev_state_der (PhysicsStateDer): The derivative of the previous state.
            curr_state (PhysicsState): The current state of the environment.

        Returns:
            The observation tensor.
        """
        pass

    @abstractmethod
    def get_spec(self) -> Bounded | Unbounded | Composite:
        """
        Get the observation spec.

        Returns:
            The observation spec.
        """
        pass

    @abstractmethod
    def get_encoder(self) -> ObservationEncoder:
        """
        Get the encoder for the observation.

        Returns:
            The observation encoder.
        """
        pass

    @classmethod
    def make_factory(cls, **opts):
        """
        Factory method to create a reward function with the given options.
        """

        @wraps(cls)
        def factory(env: "Env", **kwargs):
            return cls(env=env, **opts | kwargs)

        return factory

    def from_realistic_world(self, tensordict: "TensorDictBase") -> torch.Tensor:
        """
        Convert observation from realistic world to local state vector format.
        Args:
            tensordict (TensorDictBase): TensorDict containing the observation from the realistic world.
        Returns:
            torch.Tensor: Converted observation tensor.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement from_realistic_world method.")


ObservationFactory = Callable[["Env"], Observation]
