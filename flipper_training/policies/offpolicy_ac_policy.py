"""Off-policy actor + twin-Q critic architecture (SAC / TD3 / DDPG).

Covers the off-policy half of the FTR-Bench monolithic-RL baseline set
(SAC, TD3, DDPG) using the SAME libraries and conventions as the in-repo
PPO policies (``MLPPolicyConfig``): TorchRL operators, the shared ``MLP`` /
``EncoderCombiner`` building blocks, and the ``PolicyConfig.create(env)``
contract.

What this builds (all TorchRL-native, so it drops straight into a TorchRL
``SACLoss`` / ``TD3Loss`` / ``DDPGLoss`` in your trainer):

* **actor**  – ``ProbabilisticActor`` (TanhNormal, bounded by ``action_spec``)
  for SAC, or a deterministic ``TanhModule`` actor for TD3/DDPG. Same encoder
  pattern as the PPO actor.
* **twin Q critic** – two independent state-action value networks
  ``Q(obs, action) -> state_action_value`` (TD3/SAC clipped-double-Q). DDPG
  uses a single Q (``twin_q=False``).

The returned wrapper exposes ``get_policy_operator()`` and
``get_qvalue_operator()`` (a list) plus ``eval()/parameters()`` so it composes
with both your training loop and the eval utilities. ``create`` returns the
usual ``(wrapper, optim_groups, transforms)`` triple.

Pairs with (your trainer picks the loss):
    SAC  -> torchrl.objectives.SACLoss(actor, qvalue=qvalue_ops)
    TD3  -> torchrl.objectives.TD3Loss(actor, qvalue=qvalue_ops)  (mode='deterministic')
    DDPG -> torchrl.objectives.DDPGLoss(actor, value_network=qvalue_ops[0])
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    TanhModule,
    NormalParamExtractor,
    AdditiveGaussianModule,
)

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["OffPolicyACConfig"]


class _ConcatQ(nn.Module):
    """Q-network body: concat encoded-obs and action, MLP -> scalar Q."""

    def __init__(self, enc_dim: int, action_dim: int, mlp_opts: dict):
        super().__init__()
        self.mlp = MLP(in_dim=enc_dim + action_dim, out_dim=1, **mlp_opts)

    def forward(self, enc: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([enc, action], dim=-1))


class _OffPolicyWrapper(nn.Module):
    """Bundles actor + (twin) Q operators with the accessor API used elsewhere."""

    def __init__(self, actor, qvalue_ops: list):
        super().__init__()
        self._actor = actor
        self._qvalue = nn.ModuleList(qvalue_ops)

    def get_policy_operator(self):
        return self._actor

    def get_qvalue_operator(self):
        return list(self._qvalue)

    # convenience: many TorchRL losses accept a single qvalue op
    def get_value_operator(self):
        return self._qvalue[0]

    def eval(self):
        self._actor.eval()
        self._qvalue.eval()
        return self


@dataclass
class OffPolicyACConfig(PolicyConfig):
    """Actor + (twin) Q-critic for SAC/TD3/DDPG.

    Args:
        actor_mlp_opts / qvalue_mlp_opts: kwargs for the shared ``MLP`` (e.g.
            ``hidden_dim``, ``num_hidden``, ``layernorm``, ``activation``).
            ``in_dim``/``out_dim`` are derived from the env and forbidden here.
        actor_optimizer_opts / qvalue_optimizer_opts: per-group optimizer kwargs.
        deterministic_actor: ``True`` -> TD3/DDPG style deterministic Tanh actor;
            ``False`` -> SAC style squashed-Gaussian stochastic actor.
        twin_q: two Q networks (SAC/TD3) vs one (DDPG).
        exploration_sigma: exploration noise std for the deterministic actor
            (added via ``AdditiveGaussianModule`` as a transform; ignored for SAC).
    """

    actor_mlp_opts: dict
    qvalue_mlp_opts: dict
    actor_optimizer_opts: dict
    qvalue_optimizer_opts: dict
    deterministic_actor: bool = False        # False=SAC, True=TD3/DDPG
    twin_q: bool = True                      # False=DDPG
    exploration_sigma: float = 0.1
    extra_distribution_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("OffPolicyACConfig")

    # ---------------------------------------------------------------- helpers
    def _encoder_module(self, encoder: EncoderCombiner, out_key: str):
        return TensorDictModule(
            deepcopy(encoder),
            in_keys={k: k for k in encoder.encoders.keys()},
            out_keys=[out_key],
            out_to_in_map=True,
        )

    def _build_actor(self, action_spec, encoder: EncoderCombiner):
        enc_dim = encoder.output_dim
        adim = action_spec.shape[1]
        enc_mod = self._encoder_module(encoder, "y_actor")
        if self.deterministic_actor:
            head = TensorDictModule(
                MLP(in_dim=enc_dim, out_dim=adim, **self.actor_mlp_opts),
                in_keys=["y_actor"], out_keys=["param"],
            )
            actor = TensorDictSequential(
                enc_mod, head,
                TanhModule(in_keys=["param"], out_keys=["action"],
                           low=action_spec.space.low[0], high=action_spec.space.high[0]),
            )
            return actor
        # stochastic (SAC)
        head = TensorDictModule(
            nn.Sequential(MLP(in_dim=enc_dim, out_dim=2 * adim, **self.actor_mlp_opts),
                          NormalParamExtractor()),
            in_keys=["y_actor"], out_keys=["loc", "scale"],
        )
        return ProbabilisticActor(
            module=TensorDictSequential(enc_mod, head),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": action_spec.space.low[0],
                                 "high": action_spec.space.high[0],
                                 **self.extra_distribution_kwargs},
            return_log_prob=True,
        )

    def _build_qnet(self, action_spec, encoder: EncoderCombiner):
        enc_mod = self._encoder_module(encoder, "y_q")
        body = TensorDictModule(
            _ConcatQ(encoder.output_dim, action_spec.shape[1], self.qvalue_mlp_opts),
            in_keys=["y_q", "action"], out_keys=["state_action_value"],
        )
        return TensorDictSequential(enc_mod, body)

    # ------------------------------------------------------------------- main
    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec
        encoders = {o.name: o.get_encoder() for o in env.observations}
        encoder = EncoderCombiner(encoders)

        actor = self._build_actor(action_spec, encoder)
        qvalue_ops = [self._build_qnet(action_spec, encoder)]
        if self.twin_q:
            qvalue_ops.append(self._build_qnet(action_spec, encoder))

        wrapper = _OffPolicyWrapper(actor, qvalue_ops)
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        optim_groups = [
            {"params": actor.parameters(), "name": "actor", **self.actor_optimizer_opts},
            {"params": wrapper._qvalue.parameters(), "name": "qvalue",
             **self.qvalue_optimizer_opts},
        ]

        transforms = []
        if self.deterministic_actor:
            # exploration noise for TD3/DDPG collection
            transforms.append(AdditiveGaussianModule(
                spec=action_spec, sigma_init=self.exploration_sigma,
                sigma_end=self.exploration_sigma))

        if weights_path := kwargs.get("weights_path", None):
            mu = wrapper.load_state_dict(
                torch.load(weights_path, map_location=kwargs.get("device", "cpu")),
                strict=False)
            self.logger.info(f"Loaded weights from {weights_path}")
            if mu.missing_keys:
                self.logger.warning(f"Missing keys: {mu.missing_keys}")

        n_actor = sum(p.numel() for p in actor.parameters())
        n_q = sum(p.numel() for p in wrapper._qvalue.parameters())
        self.logger.info(
            f"OffPolicyAC: actor={n_actor:,} params, "
            f"{'twin' if self.twin_q else 'single'}-Q={n_q:,} params, "
            f"actor={'deterministic (TD3/DDPG)' if self.deterministic_actor else 'stochastic (SAC)'}")
        return wrapper, optim_groups, transforms
