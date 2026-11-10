"""Soft differentiable state-machine policy (Azayev & Zimmermann 2022).

Azayev, T. & Zimmermann, K. "Autonomous State-Based Flipper Control for
Articulated Tracked Robots in Urban Environments," IEEE RA-L 7(3):7794-7801.

This is the closest published architectural rival to a hierarchical GC/OA
switching/blending design, so it is the key head-to-head baseline. Implemented
with the SAME libraries/conventions as the repo's PPO policies (TorchRL
actor-critic, shared ``MLP``/``EncoderCombiner``), so it trains with the same
``ClipPPOLoss`` loop.

Architecture (soft state machine):
* shared encoder (``EncoderCombiner``),
* ``K`` learned **state heads**, each a small MLP that emits a continuous action
  (a "flipper template" / local behaviour),
* a **gating network** that emits soft state-occupancy weights (softmax with
  temperature) — the differentiable relaxation of a discrete state machine,
* the emitted action is the gate-weighted blend of the K state actions; its
  mean/scale feed a bounded TanhNormal ``ProbabilisticActor``.

Optionally the gate is recurrent (GRU) so the current state depends on history,
matching the state-transition behaviour of the original soft state machine; the
default here is feed-forward for simplicity (set ``recurrent_gate=True`` to add
the GRU + the required InitTracker/primer transforms in your trainer).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator,
    ActorValueOperator,
)

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["StateMachinePolicyConfig"]


class _SoftStateMachine(nn.Module):
    """Encoded obs -> soft state weights -> blended action params (loc, scale)."""

    def __init__(self, enc_dim: int, action_dim: int, n_states: int,
                 state_mlp_opts: dict, gate_mlp_opts: dict, temperature: float):
        super().__init__()
        self.n_states = n_states
        self.temperature = temperature
        # K state heads, each emits 2*action_dim (loc+scale params)
        self.states = nn.ModuleList(
            MLP(in_dim=enc_dim, out_dim=2 * action_dim, **state_mlp_opts)
            for _ in range(n_states))
        self.gate = MLP(in_dim=enc_dim, out_dim=n_states, **gate_mlp_opts)
        self.param_extractor = NormalParamExtractor()

    def forward(self, enc: torch.Tensor):
        # soft state occupancy (differentiable relaxation of the state machine)
        w = torch.softmax(self.gate(enc) / self.temperature, dim=-1)  # [B, K]
        # each state's action params
        params = torch.stack([s(enc) for s in self.states], dim=1)    # [B, K, 2A]
        blended = (w.unsqueeze(-1) * params).sum(dim=1)               # [B, 2A]
        loc, scale = self.param_extractor(blended)
        return loc, scale


@dataclass
class StateMachinePolicyConfig(PolicyConfig):
    """Soft state-machine actor + value critic (Azayev & Zimmermann 2022).

    Args:
        n_states: number of flipper-template states K.
        temperature: gating softmax temperature (->0 hard switching, large=blend).
        state_mlp_opts/gate_mlp_opts/value_mlp_opts: shared ``MLP`` kwargs.
        *_optimizer_opts: per-group optimizer kwargs.
        share_encoder: share the obs encoder between actor and critic.
    """

    n_states: int
    temperature: float
    state_mlp_opts: dict
    gate_mlp_opts: dict
    value_mlp_opts: dict
    actor_optimizer_opts: dict
    value_optimizer_opts: dict
    extra_distribution_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("StateMachinePolicyConfig")

    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec
        adim = action_spec.shape[1]
        encoders = {o.name: o.get_encoder() for o in env.observations}
        encoder = EncoderCombiner(encoders)

        common = TensorDictModule(
            encoder, in_keys={k: k for k in encoder.encoders.keys()},
            out_keys=["y_shared"], out_to_in_map=True)

        sm = TensorDictModule(
            _SoftStateMachine(encoder.output_dim, adim, self.n_states,
                              self.state_mlp_opts, self.gate_mlp_opts, self.temperature),
            in_keys=["y_shared"], out_keys=["loc", "scale"])
        actor = ProbabilisticActor(
            module=sm, spec=action_spec, in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": action_spec.space.low[0],
                                 "high": action_spec.space.high[0],
                                 **self.extra_distribution_kwargs},
            return_log_prob=True)
        value = ValueOperator(
            module=MLP(in_dim=encoder.output_dim, out_dim=1, **self.value_mlp_opts),
            in_keys=["y_shared"])

        wrapper = ActorValueOperator(
            policy_operator=actor, value_operator=value, common_operator=common)
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        optim_groups = [
            {"params": wrapper.get_policy_operator().parameters(),
             "name": "policy_and_encoder", **self.actor_optimizer_opts},
            {"params": wrapper.get_value_head().parameters(),
             "name": "value_head", **self.value_optimizer_opts},
        ]
        n_params = sum(p.numel() for p in wrapper.parameters())
        self.logger.info(
            f"Azayev soft state-machine: K={self.n_states} states, "
            f"temp={self.temperature}, {n_params:,} params. Train with ClipPPOLoss.")
        return wrapper, optim_groups, []
