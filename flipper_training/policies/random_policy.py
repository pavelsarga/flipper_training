"""Random policy baseline for benchmarking trained PPO policies.

Ignores all observations. Drives at (optionally randomized) constant linear speed
and moves flippers with per-robot sinusoidal velocity commands, producing
oscillating flipper positions over time.

Action layout expected by the FTR-benchmark env:
  [v, w, flipper_0, ..., flipper_N]
where v ∈ [-1, 1], w ∈ [-1, 1], flipper_i are velocity commands in [-1, 1].

The action space dimension is taken directly from env.action_spec at creation time.

Flipper synchronisation (sync_flippers=True in policy_opts):
  Front-left == front-right and rear-left == rear-right.
  Only 2 sine parameter sets are sampled; the values are broadcast to all 4
  flipper slots regardless of the action-space dimension seen by the env.
  This works independently of the env's sync_flipper_control setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from flipper_training.policies import PolicyConfig


class _RandomActorModule(nn.Module):
    """Observation-blind actor: constant speed + sinusoidal flipper velocities.

    When sync_flippers=True, only 2 sine parameter sets are sampled (front, rear)
    and their values are tiled to fill all flipper slots:
      action[:, 2] = action[:, 3] = front_sine   (FL = FR)
      action[:, 4] = action[:, 5] = rear_sine    (RL = RR)
    """

    def __init__(
        self,
        n_robots: int,
        n_flipper_actions: int,
        linear_speeds: torch.Tensor,  # (n_robots,) — per-robot forward speed
        angular_speed: float,
        sync_flippers: bool,
        amp_min: float,
        amp_max: float,
        freq_min: float,
        freq_max: float,
        seed: int,
    ) -> None:
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)

        self.register_buffer("linear_speeds", linear_speeds)  # (n_robots,)
        self.register_buffer("angular_speed", torch.full((n_robots,), angular_speed))

        # When syncing, sample only 2 parameter sets (front pair, rear pair)
        n_sine_sets = 2 if sync_flippers else n_flipper_actions
        freqs = torch.empty(n_robots, n_sine_sets).uniform_(freq_min, freq_max, generator=rng)
        amps = torch.empty(n_robots, n_sine_sets).uniform_(amp_min, amp_max, generator=rng)
        phases = torch.empty(n_robots, n_sine_sets).uniform_(0.0, 2.0 * math.pi, generator=rng)
        self.register_buffer("freqs", freqs)
        self.register_buffer("amps", amps)
        self.register_buffer("phases", phases)

        self._action_dim = 2 + n_flipper_actions
        self._sync_flippers = sync_flippers
        self._n_flipper_actions = n_flipper_actions
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    def forward(self) -> torch.Tensor:
        t = float(self._step.item())
        self._step += 1

        # Sinusoidal flipper velocity: amp * sin(freq * t + phase)
        # The env integrates this to get oscillating positions.
        sine = self.amps * torch.sin(self.freqs * t + self.phases)  # (n_robots, n_sine_sets)

        if self._sync_flippers:
            # Tile front and rear values to all flipper slots
            # Assumes layout: [FL, FR, RL, RR] → front=sine[:,0], rear=sine[:,1]
            front = sine[:, 0:1]  # (n_robots, 1)
            rear = sine[:, 1:2]   # (n_robots, 1)
            n_front = self._n_flipper_actions // 2
            n_rear = self._n_flipper_actions - n_front
            flipper_vel = torch.cat([front.expand(-1, n_front), rear.expand(-1, n_rear)], dim=1)
        else:
            flipper_vel = sine

        action = torch.empty(
            self.linear_speeds.shape[0], self._action_dim,
            dtype=torch.float32, device=self.linear_speeds.device,
        )
        action[:, 0] = self.linear_speeds
        action[:, 1] = self.angular_speed
        action[:, 2:] = flipper_vel
        return action


class _RandomWrapper:
    """Thin wrapper satisfying the ActorCriticWrapper interface used in eval_ftr.py."""

    def __init__(self, actor_tdm: TensorDictModule) -> None:
        self._actor_tdm = actor_tdm

    def get_policy_operator(self) -> TensorDictModule:
        return self._actor_tdm

    def eval(self) -> "_RandomWrapper":
        self._actor_tdm.eval()
        return self

    def parameters(self):
        return iter([])


@dataclass
class RandomPolicyConfig(PolicyConfig):
    """Configuration for the random benchmark policy.

    All float parameters that can be set to ``None`` will instead be sampled
    randomly per robot from the corresponding ``_min`` / ``_max`` bounds.
    Express ``null`` in YAML to trigger this behaviour.

    Args:
        linear_speed: Forward speed in [-1, 1]. ``None`` → sample uniformly
            from ``[linear_speed_min, linear_speed_max]`` per robot.
        linear_speed_min: Lower bound for random linear speed sampling.
        linear_speed_max: Upper bound for random linear speed sampling.
        angular_speed: Angular (turn) velocity, same for all robots.
        sync_flippers: If True, front-left==front-right and rear-left==rear-right.
            Only 2 sine parameter sets are sampled; values are broadcast to all
            flipper slots. Works regardless of the env's sync_flipper_control setting.
        amp_min: Minimum flipper-velocity sinusoid amplitude.
        amp_max: Maximum flipper-velocity sinusoid amplitude.
        freq_min: Minimum sinusoid frequency in rad/step. ~1 cycle per
            ``2π / freq_max`` steps.
        freq_max: Maximum sinusoid frequency in rad/step.
        seed: RNG seed for reproducible parameter sampling.
    """

    linear_speed: float | None = None
    linear_speed_min: float = 0.5
    linear_speed_max: float = 1.0
    angular_speed: float = 0.0
    sync_flippers: bool = False
    amp_min: float = 0.3
    amp_max: float = 0.9
    freq_min: float = 0.005   # ~1 cycle per 1257 steps
    freq_max: float = 0.05    # ~1 cycle per 126 steps
    seed: int = 0

    def create(self, env, **kwargs):
        action_spec = env.action_spec
        n_robots: int = action_spec.shape[0]
        action_dim: int = action_spec.shape[-1]
        n_flipper_actions: int = action_dim - 2

        device = getattr(env, "device", torch.device("cpu"))

        rng = torch.Generator()
        rng.manual_seed(self.seed)

        if self.linear_speed is None:
            linear_speeds = torch.empty(n_robots).uniform_(
                self.linear_speed_min, self.linear_speed_max, generator=rng
            ).to(device)
        else:
            linear_speeds = torch.full((n_robots,), self.linear_speed, device=device)

        module = _RandomActorModule(
            n_robots=n_robots,
            n_flipper_actions=n_flipper_actions,
            linear_speeds=linear_speeds,
            angular_speed=self.angular_speed,
            sync_flippers=self.sync_flippers,
            amp_min=self.amp_min,
            amp_max=self.amp_max,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            seed=self.seed + 1,  # offset so flipper params differ from speed params
        ).to(device)

        actor_tdm = TensorDictModule(module, in_keys=[], out_keys=["action"])
        return _RandomWrapper(actor_tdm), [], []
