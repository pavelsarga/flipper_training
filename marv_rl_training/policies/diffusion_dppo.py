"""Phase 2: diffusion actor over the action chunk, trained with DPPO.

Two pieces:

  DiffusionChunkActor  — runs the DDIM chain with ConditionalUnet1D as eps_theta and emits
                         the executed action, the full denoising chain, and the chain's
                         log-probability.
  DPPOClipLoss         — ClipPPOLoss with the log-probability recomputed from the STORED
                         chain instead of from a freshly drawn one.

Why a subclass and not plain ClipPPOLoss. ClipPPOLoss gets the current log-prob by calling
`actor.get_dist(td)` and asking that distribution for `log_prob(action)`. A diffusion actor
cannot answer it: the object to score is the denoising chain that was actually sampled, not
the final action, and calling the actor again just draws a different chain. torchrl
anticipates this — PPOLoss._get_cur_log_prob raises NotImplementedError with the advice to
"augment [the loss] by implementing your own logic in _get_cur_log_prob", which is exactly
what happens below. Everything downstream (the ratio, the clipping, kl_approx, the critic)
is inherited untouched.

DPPO (Ren et al. 2024) is what makes this tractable: each DDIM step is a Gaussian
    pi(A^{k-1} | A^k, s) = N( mu_theta(A^k, s, k), sigma_k^2 I )
so the chain has a closed-form likelihood, the product of its per-step Gaussians. This file
implements the chain-level ratio (one importance weight per macro-step). Per-denoising-step
clipping — what DPPO actually advocates — is a further change, worth making only if the
chain-level KL proves unstable.

⚠ entropy_bonus MUST be false. Diffusion entropy is not tractable, `dist` is returned as
None here, and ClipPPOLoss only touches it under `if self.entropy_bonus`. Exploration comes
from the sampling sigma floor (`min_sampling_std` on the schedule) instead.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.objectives import ClipPPOLoss

from marv_rl_training.policies.diffusion_policy import ConditionalUnet1D, ObsHistoryEncoder, SinusoidalPosEmb
from marv_rl_training.policies.diffusion_schedule import DiffusionSchedule

__all__ = ["DiffusionChunkActor", "DPPOClipLoss"]


class DiffusionChunkActor(TensorDictModuleBase):
    """obs_history -> DDIM chain -> (action, denoise_chain, sample_log_prob).

    The chain is stored as ``[N, K+1, A, T_p]``, oldest (pure noise) first, so
    ``chain[:, i]`` is ``A^{k_i}`` and ``chain[:, i+1]`` the transition's outcome. It has to
    be carried in the tensordict because the loss must score the transitions that were
    actually taken; recomputing them from the action alone is impossible.
    """

    def __init__(
        self,
        encoder: ObsHistoryEncoder,
        schedule: DiffusionSchedule,
        action_dim: int,
        prediction_horizon: int,
        down_dims: list[int],
        kernel_size: int = 5,
        n_groups: int = 8,
        step_embed_dim: int = 64,
        obs_history_key: str = "obs_history",
    ):
        super().__init__()
        self.in_keys = [obs_history_key]
        self.out_keys = ["action", "denoise_chain", "sample_log_prob"]
        self.obs_history_key = obs_history_key
        self.encoder = encoder
        self.schedule = schedule
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon

        self.step_embed = nn.Sequential(
            SinusoidalPosEmb(step_embed_dim),
            nn.Linear(step_embed_dim, step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(step_embed_dim * 4, step_embed_dim),
        )
        self.unet = ConditionalUnet1D(
            in_channels=action_dim,
            out_channels=action_dim,           # epsilon-prediction
            horizon=prediction_horizon,
            cond_dim=encoder.output_dim + step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    # ------------------------------------------------------------------ helpers

    def _cond(self, obs_emb: torch.Tensor, k: int) -> torch.Tensor:
        n = obs_emb.shape[0]
        kt = torch.full((n,), float(k), device=obs_emb.device)
        return torch.cat([obs_emb, self.step_embed(kt)], dim=-1)

    def _chain_to_action(self, x: torch.Tensor) -> torch.Tensor:
        # [N, A, T_p] -> [N, T_p, A] -> flat, matching ActionChunkEnv's reshape(N, T_p, A).
        return x.transpose(1, 2).reshape(x.shape[0], -1)

    # ------------------------------------------------------------------ rollout

    def sample_chain(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_emb = self.encoder(obs_history)
        n = obs_emb.shape[0]
        x = torch.randn(n, self.action_dim, self.prediction_horizon, device=obs_emb.device)
        chain = [x]
        logp = torch.zeros(n, device=obs_emb.device)
        for k, k_prev in self.schedule.step_pairs():
            eps = self.unet(x, self._cond(obs_emb, k))
            mean, std = self.schedule.ddim_step(x, eps, k, k_prev)
            x = mean + std * torch.randn_like(mean)
            logp = logp + torch.distributions.Normal(mean, std).log_prob(x).sum(dim=(-1, -2))
            chain.append(x)
        return self._chain_to_action(x), torch.stack(chain, dim=1), logp

    def forward(self, tensordict: TensorDict) -> TensorDict:
        action, chain, logp = self.sample_chain(tensordict.get(self.obs_history_key))
        tensordict.set("action", action)
        tensordict.set("denoise_chain", chain)
        tensordict.set("sample_log_prob", logp)
        return tensordict

    # ------------------------------------------------------------------ scoring

    def chain_log_prob(self, tensordict: TensorDict) -> torch.Tensor:
        """Log-probability of the STORED chain under the current parameters.

        Deterministic given the chain — no resampling — which is what makes the PPO ratio
        well defined for a diffusion policy.
        """
        obs_emb = self.encoder(tensordict.get(self.obs_history_key))
        chain = tensordict.get("denoise_chain")
        logp = torch.zeros(obs_emb.shape[0], device=obs_emb.device)
        for i, (k, k_prev) in enumerate(self.schedule.step_pairs()):
            x_k = chain[:, i]
            x_next = chain[:, i + 1]
            eps = self.unet(x_k, self._cond(obs_emb, k))
            mean, std = self.schedule.ddim_step(x_k, eps, k, k_prev)
            logp = logp + torch.distributions.Normal(mean, std).log_prob(x_next).sum(dim=(-1, -2))
        return logp


class DPPOClipLoss(ClipPPOLoss):
    """ClipPPOLoss scoring the stored denoising chain rather than a fresh sample."""

    def _get_cur_log_prob(self, tensordict):
        actor = self.actor_network
        with self.actor_network_params.to_module(actor) if self.functional else contextlib.nullcontext():
            log_prob = actor.chain_log_prob(tensordict)
        # dist=None is safe: ClipPPOLoss only uses it under `if self.entropy_bonus`, which
        # must be false for a diffusion policy (entropy is not tractable).
        return log_prob, None, False
