"""Intrinsic Curiosity Module (ICM) — turns AT-D3QN into ICM-D3QN.

Pathak et al. 2017, as used by Pan et al. (Remote Sensing 15(18):4616) on top
of the dueling DQN flipper controller. Same libraries (torch/tensordict); the
module is trainer-agnostic — query :meth:`intrinsic_reward` each step and add it
to the extrinsic reward, and add :meth:`loss` to your optimizer.

Components (over a learned feature space ``phi(obs)``):
* inverse model  ``g(phi_t, phi_{t+1}) -> action``  (encourages controllable features)
* forward model  ``f(phi_t, action)   -> phi_{t+1}`` (its error is the curiosity reward)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EncoderCombiner, MLP

__all__ = ["ICM"]


class ICM(nn.Module):
    def __init__(self, encoder: EncoderCombiner, action_dim: int, feat_dim: int = 128,
                 mlp_opts: dict | None = None, beta: float = 0.2, eta: float = 0.5):
        super().__init__()
        mlp_opts = mlp_opts or dict(hidden_dim=256, num_hidden=2, layernorm=True)
        self.encoder = encoder
        self.feat = MLP(in_dim=encoder.output_dim, out_dim=feat_dim, **mlp_opts)
        self.inverse = MLP(in_dim=2 * feat_dim, out_dim=action_dim, **mlp_opts)
        self.forward_model = MLP(in_dim=feat_dim + action_dim, out_dim=feat_dim, **mlp_opts)
        self.beta = beta      # forward/inverse loss weighting
        self.eta = eta        # intrinsic reward scale

    def _phi(self, obs: dict) -> torch.Tensor:
        return self.feat(self.encoder(**obs))

    @torch.no_grad()
    def intrinsic_reward(self, obs, action, next_obs) -> torch.Tensor:
        phi, phi_next = self._phi(obs), self._phi(next_obs)
        pred_next = self.forward_model(torch.cat([phi, action], dim=-1))
        return self.eta * 0.5 * (pred_next - phi_next).pow(2).sum(dim=-1, keepdim=True)

    def loss(self, obs, action, next_obs) -> torch.Tensor:
        phi, phi_next = self._phi(obs), self._phi(next_obs)
        pred_action = self.inverse(torch.cat([phi, phi_next], dim=-1))
        pred_next = self.forward_model(torch.cat([phi, action], dim=-1))
        inv_loss = F.mse_loss(pred_action, action)
        fwd_loss = F.mse_loss(pred_next, phi_next.detach())
        return (1 - self.beta) * inv_loss + self.beta * fwd_loss
