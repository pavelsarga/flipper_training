"""AT-D3QN: dueling double-DQN flipper controller (Pan et al. 2023).

Pan, H. et al. "Deep Reinforcement Learning for Flipper Control of Tracked
Robots," arXiv:2306.10352 (AT-D3QN) and "... in Urban Rescuing Environments,"
Remote Sensing 15(18):4616 (ICM-D3QN = this + the curiosity module in
``icm.py``).

Discrete-action baseline expressed with the SAME libraries as the rest of the
repo (TorchRL ``QValueActor`` + the shared ``MLP``/``EncoderCombiner``), so it
trains with ``torchrl.objectives.DQNLoss`` / ``DoubleDQN`` in your loop.

Architecture (faithful to AT-D3QN):
* shared observation encoder (``EncoderCombiner``),
* **dueling head**: a value stream V(s) and an advantage stream A(s,a),
  combined as ``Q = V + (A - mean_a A)``,
* a **discrete flipper-configuration action set**: each of the 4 flippers is
  binned into ``n_bins`` angle levels across the joint limits, giving
  ``n_bins**4`` discrete configurations; the chosen index maps to a continuous
  env action (constant forward track velocity + the selected flipper joints).
  The continuous mapping table is exposed as ``wrapper.discrete_to_continuous``
  (an ``[M, action_dim]`` buffer) for the collector/env bridge.

Double-DQN is a *training* choice (``DQNLoss(..., double_dqn=True)`` or
``SoftUpdate`` target net), not an architecture change — the dueling network
here is exactly what AT-D3QN learns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import QValueActor

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["D3QNPolicyConfig"]


class _DuelingQ(nn.Module):
    """Encoder -> dueling (V, A) -> Q over ``n_actions`` discrete configs."""

    def __init__(self, encoder: EncoderCombiner, n_actions: int, mlp_opts: dict):
        super().__init__()
        self.encoder = encoder
        hidden = dict(mlp_opts)
        self.value = MLP(in_dim=encoder.output_dim, out_dim=1, **hidden)
        self.adv = MLP(in_dim=encoder.output_dim, out_dim=n_actions, **hidden)

    def forward(self, *obs_tensors):
        # EncoderCombiner.forward takes kwargs; rebuild the mapping by order.
        enc = self.encoder(**dict(zip(self._obs_keys, obs_tensors)))
        v = self.value(enc)
        a = self.adv(enc)
        return v + (a - a.mean(dim=-1, keepdim=True))


class _D3QNWrapper(nn.Module):
    def __init__(self, q_actor, discrete_to_continuous: torch.Tensor):
        super().__init__()
        self._q_actor = q_actor
        self.register_buffer("discrete_to_continuous", discrete_to_continuous)

    def get_policy_operator(self):
        return self._q_actor

    def get_qvalue_network(self):       # for DQNLoss(value_network=...)
        return self._q_actor

    def eval(self):
        self._q_actor.eval()
        return self


@dataclass
class D3QNPolicyConfig(PolicyConfig):
    """Dueling double-DQN over a discretized flipper-configuration action set.

    Args:
        mlp_opts: shared ``MLP`` kwargs for the value/advantage streams.
        optimizer_opts: optimizer kwargs for the single Q-network group.
        n_bins: angle levels per flipper (``n_bins**4`` discrete actions).
        forward_track_velocity: constant normalized forward command written into
            the track-velocity slots of the continuous action (flippers are the
            learned part, as in Pan et al.).
    """

    mlp_opts: dict
    optimizer_opts: dict
    n_bins: int = 3
    forward_track_velocity: float = 0.6
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("D3QNPolicyConfig")

    def _build_action_table(self, action_spec):
        """Map each discrete config -> a full continuous action vector.

        Continuous action layout is ``[tracks..., flipper_joints...]`` (see
        ``Env._make_action_spec``): we fill the track slots with a constant
        forward command and the flipper-joint slots with the binned config.
        """
        adim = action_spec.shape[1]
        n_drive = adim // 2                         # tracks == flipper joints count
        low = action_spec.space.low[0]
        high = action_spec.space.high[0]
        # flipper-joint slots are the second half of the action vector
        joint_low = low[n_drive:]
        joint_high = high[n_drive:]
        n_flip = joint_low.shape[0]
        # per-flipper bin levels
        levels = [torch.linspace(joint_low[i].item(), joint_high[i].item(), self.n_bins)
                  for i in range(n_flip)]
        configs = list(product(*[range(self.n_bins)] * n_flip))
        table = torch.zeros(len(configs), adim)
        # constant forward velocity on the track slots (clamped to spec)
        v = float(self.forward_track_velocity)
        table[:, :n_drive] = torch.clamp(
            torch.tensor(v), low[:n_drive], high[:n_drive])
        for r, cfg in enumerate(configs):
            for i, b in enumerate(cfg):
                table[r, n_drive + i] = levels[i][b]
        return table

    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec
        encoders = {o.name: o.get_encoder() for o in env.observations}
        encoder = EncoderCombiner(encoders)
        table = self._build_action_table(action_spec)
        n_actions = table.shape[0]

        qnet = _DuelingQ(encoder, n_actions, self.mlp_opts)
        qnet._obs_keys = list(encoder.encoders.keys())
        q_module = TensorDictModule(
            qnet, in_keys=qnet._obs_keys, out_keys=["action_value"])
        q_actor = QValueActor(
            q_module, in_keys=qnet._obs_keys, action_space="one_hot",
            action_value_key="action_value")

        wrapper = _D3QNWrapper(q_actor, table)
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        optim_groups = [{"params": q_actor.parameters(), "name": "qnet",
                         **self.optimizer_opts}]
        n_params = sum(p.numel() for p in q_actor.parameters())
        self.logger.info(
            f"AT-D3QN: dueling Q over {n_actions} discrete flipper configs "
            f"(n_bins={self.n_bins}), {n_params:,} params. "
            f"Train with DQNLoss(double_dqn=True); env bridge via "
            f"wrapper.discrete_to_continuous [{table.shape[0]}, {table.shape[1]}].")
        return wrapper, optim_groups, []
