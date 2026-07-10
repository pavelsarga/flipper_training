"""Intrinsic Curiosity Module (ICM) -- turns AT-D3QN into ICM-D3QN.

Pathak et al. 2017's ICM formulation as adapted by Pan et al. 2023 (Remote
Sensing 15(18):4616, Sec. 3.5 + Fig. 6) on top of the dueling-DQN flipper
controller (``d3qn_policy.py``). Same libraries (torch/tensordict).

This module does not step the env or touch the replay buffer itself, but it
is NOT a fire-and-forget decoration either -- the caller
(``experiments/dqn/train.py``) MUST:

1. call :meth:`intrinsic_reward` under ``torch.no_grad()`` each env step and
   ADD it to the stored transition reward BEFORE the transition goes into the
   replay buffer, so ``R_t = R^e_t + R^i_t`` (Eq. 13) is what ``DQNLoss``
   bootstraps on. Historically this call site did not exist anywhere in the
   trainer -- :meth:`intrinsic_reward` was never invoked, only :meth:`loss`
   was, which silently degenerated ICM-D3QN into plain AT-D3QN (the curiosity
   term never reached the RL objective, only its own aux loss).
2. add :meth:`loss` to the optimizer alongside the DQN loss (Eq. 15).
3. EITHER (default, ``separate_encoder=False``) construct this module with the
   D3QN Q-network's OWN encoder instance (e.g. ``d3qn_wrapper.get_encoder()``),
   NOT a freshly built ``EncoderCombiner``, so the Q-network and the ICM
   genuinely train one encoder -- NOTE: this is a deliberate implementation
   simplification, not the paper's literal wiring (Fig. 7 draws two separate
   encoders: the D3QN's fused feature module and the ICM's own raw-state
   encoder); sharing means curiosity gradients shape the policy
   representation, and it is one encoder trained by both the TD loss and
   ICM's inverse/forward losses -- not two decoupled copies that happen to
   have the same architecture. Because the encoder is then a submodule of
   both the Q-network and this module, its parameters appear in
   `ICM.parameters()` too -- the caller must exclude them when building the
   ICM optimizer param group, or the shared weights receive two Adam updates
   per step (autograd already sums the encoder's gradient contribution from
   both losses correctly; it's the *optimizer* step that must not
   double-apply). OR (``separate_encoder=True``, the paper-literal wiring)
   construct this module with ``encoder=None`` and ``raw_state_dim=`` the
   ``PanTerrainState`` observation's ``.dim`` (18 for the paper's N=15): ICM
   then builds its OWN Fig. 7 3-layer raw-state encoder over that
   observation's raw tensor, no sharing, no dedup needed -- ALL of ICM's
   params are its own optimizer group. ``experiments/dqn/train.py`` resolves
   ``raw_state_dim`` automatically from the env's observations when this mode
   is selected via ``icm_opts: {separate_encoder: true}``.

Components (over a learned feature space ``psi(obs) = feat(encoder(obs))``,
``feat`` a no-op ``Identity`` when ``separate_encoder=True`` since Fig. 7's
``psi`` already ends at ``feat_dim`` -- no extra projection head on top):
* inverse model  ``I(psi_t, psi_{t+1}) -> action logits``  (Eq. 14: cross-entropy
  against the discrete one-hot action -- encourages controllable features)
* forward model  ``F(psi_t, action)   -> psi_hat_{t+1}``   (Eq. 11: its MSE
  against ``psi_{t+1}`` is both a training loss and, scaled by ``eta`` and
  normalized by the feature dimension M, the intrinsic reward, Eq. 12)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EncoderCombiner, MLP

__all__ = ["ICM"]


class ICM(nn.Module):
    """Args:
        encoder: the observation encoder to REUSE when ``separate_encoder=False``
            (default) -- pass the D3QN Q-network's encoder instance directly
            (``d3qn_wrapper.get_encoder()``) so it is shared, not copied (one
            encoder instance feeding both the Q-network and the curiosity
            module). See the module docstring's point 3 for the optimizer-side
            implication of sharing by reference. Must be ``None`` when
            ``separate_encoder=True`` (ICM builds its own instead).
        action_dim: size of the one-hot discrete action space -- same as the
            D3QN's ``wrapper.n_actions`` (9 for the paper's paired front/rear
            delta set, Eq. 3, or ``n_bins**n_flip`` for the absolute table).
        feat_dim: dimensionality M of the psi feature space (Eq. 12's M).
            ``None`` (default) resolves to 128 when ``separate_encoder=False``
            (this module's original default) or 10 when ``separate_encoder=True``
            (Fig. 7's literal psi output width).
        mlp_opts: shared-mode (``separate_encoder=False``) only -- kwargs for
            the ``feat``/``inverse``/``forward_model`` heads. Ignored (and
            warned about, if explicitly non-empty) when ``separate_encoder=True``,
            which uses Fig. 7's fixed widths instead (see below).
        beta_forward / beta_inverse: independent weights on the forward and
            inverse losses in the combined ICM loss (Eq. 15,
            ``L_zeta = beta_forward * L_F + beta_inverse * L_I``, both in
            [0, 1] and NOT required to sum to 1 -- the paper decouples them,
            unlike Pathak's original ``(1-beta, beta)`` convention). Defaults
            (0.8, 0.2) are Table 2's beta_F / beta_I.
        eta: intrinsic-reward scale (Eq. 12's eta). Exposed as a config option
            via ``icm_opts: {eta: ...}`` in the dqn trainer's yaml. NOT given a
            numeric value in either paper's text or Table 2 (checked via
            ``pdftotext`` over both PDFs) -- the default 0.5 is our own choice.
        separate_encoder: build the paper's OWN Fig. 7 raw-state encoder
            (``psi``: ``Dense(raw_state_dim,32) -> LeakyReLU -> Dense(32,64) ->
            LeakyReLU -> Dense(64,feat_dim)``, 3 dense layers, matching Fig. 7's
            literal ``18 -> 32 -> 64 -> 10`` for the paper's own raw-state dim)
            instead of reusing the D3QN's encoder. The forward/inverse heads
            also switch to Fig. 7's fixed 1-hidden-layer-of-32 LeakyReLU shape
            (2 dense layers each, matching Fig. 7's ``19x32``/``20x32`` blocks)
            instead of ``mlp_opts``.
        raw_state_key: which observation ``separate_encoder=True`` reads its
            raw (unencoded) state tensor from. Default ``"PanTerrainState"``.
        raw_state_dim: dimensionality of that raw state (required when
            ``separate_encoder=True``; e.g. a ``PanTerrainState`` observation's
            ``.dim``, 18 for the paper's N=15 -- ``experiments/dqn/train.py``
            resolves this automatically from the env's observations).
    """

    def __init__(
        self,
        encoder: EncoderCombiner | None,
        action_dim: int,
        feat_dim: int | None = None,
        mlp_opts: dict | None = None,
        beta_forward: float = 0.8,  # paper Table 2: beta_F = 0.8
        beta_inverse: float = 0.2,  # paper Table 2: beta_I = 0.2
        eta: float = 0.5,
        separate_encoder: bool = False,
        raw_state_key: str = "PanTerrainState",
        raw_state_dim: int | None = None,
    ):
        super().__init__()
        self.separate_encoder = separate_encoder
        if separate_encoder:
            if encoder is not None:
                raise ValueError("separate_encoder=True builds its OWN encoder -- pass encoder=None (got a real encoder instance).")
            if raw_state_dim is None:
                raise ValueError(
                    "separate_encoder=True needs raw_state_dim (the raw-state observation's .dim, e.g. a "
                    "PanTerrainState's dim = n_heights + 3 = 18 for the paper's N=15). "
                    "experiments/dqn/train.py resolves this automatically from the env's observations; pass "
                    "it explicitly if constructing ICM by hand."
                )
            if mlp_opts:
                raise ValueError(
                    "separate_encoder=True uses Fig. 7's FIXED layer widths (32-wide, LeakyReLU), not mlp_opts "
                    f"-- got a non-empty mlp_opts={mlp_opts}. Drop it (or set separate_encoder=False)."
                )
            self.raw_state_key = raw_state_key
            feat_dim = 10 if feat_dim is None else feat_dim  # Fig. 7's literal psi output width
            # Fig. 7's psi: Dense(raw,32) -> LeakyReLU -> Dense(32,64) -> LeakyReLU -> Dense(64,feat_dim) -- 3 dense layers.
            self.encoder = MLP(in_dim=raw_state_dim, hidden_dim=[32, 64], num_hidden=2, out_dim=feat_dim, layernorm=False, activation=nn.LeakyReLU)
            self.feat = nn.Identity()  # psi already IS the feature (no extra projection, unlike shared mode)
            # Fig. 7's F / I: Dense(in,32) -> LeakyReLU -> Dense(32,out) -- 2 dense layers each, fixed width 32.
            head_opts = dict(hidden_dim=32, num_hidden=1, layernorm=False, activation=nn.LeakyReLU)
        else:
            if encoder is None:
                raise ValueError("separate_encoder=False (shared mode, default) needs a real `encoder` -- pass d3qn_wrapper.get_encoder().")
            self.raw_state_key = None
            feat_dim = 128 if feat_dim is None else feat_dim
            head_opts = mlp_opts or dict(hidden_dim=256, num_hidden=2, layernorm=True)
            self.encoder = encoder
            self.feat = MLP(in_dim=encoder.output_dim, out_dim=feat_dim, **head_opts)
        self.inverse = MLP(in_dim=2 * feat_dim, out_dim=action_dim, **head_opts)
        self.forward_model = MLP(in_dim=feat_dim + action_dim, out_dim=feat_dim, **head_opts)
        self.feat_dim = feat_dim
        self.beta_forward = beta_forward    # weight on L_F in Eq. 15
        self.beta_inverse = beta_inverse    # weight on L_I in Eq. 15
        self.eta = eta                      # intrinsic reward scale, Eq. 12

    def _phi(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.separate_encoder:
            return self.feat(self.encoder(obs[self.raw_state_key]))
        return self.feat(self.encoder(**obs))

    @torch.no_grad()
    def intrinsic_reward(self, obs: dict[str, torch.Tensor], action: torch.Tensor, next_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """R^i_t = eta * (1/M) * ||psi_hat(s_{t+1}) - psi(s_{t+1})||^2 (Eq. 12,
        M = feat_dim) -- i.e. eta times the forward model's error, EXACTLY as
        in Eq. 11's MSELoss (mean over the feature dim), just evaluated
        per-transition instead of batch-reduced so it can be added onto a
        per-transition reward (Eq. 13). Already ``@torch.no_grad()``; callers
        do not need to wrap this call again.

        ``action`` must be the one-hot action actually taken (same convention
        as :meth:`loss`). Returns shape ``[B, 1]`` so it adds directly onto a
        ``[B, 1]`` env reward tensor.
        """
        phi, phi_next = self._phi(obs), self._phi(next_obs)
        pred_next = self.forward_model(torch.cat([phi, action], dim=-1))
        return self.eta * (pred_next - phi_next).pow(2).mean(dim=-1, keepdim=True)

    def loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor, next_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combined ICM loss (Eq. 15): ``beta_forward * L_F + beta_inverse * L_I``.

        * ``L_F`` (Eq. 11): MSE between the forward model's prediction and the
          encoder's OWN next-state feature, target detached -- the Pathak
          convention: the forward loss must not be able to trivially shrink by
          dragging the ``psi_{t+1}`` target around too, only by getting better
          at predicting it (it still trains ``F`` and, through ``psi_t``, the
          shared encoder).
        * ``L_I`` (Eq. 14): cross-entropy of the inverse head's output (treated
          as class logits over the discrete one-hot action space) against the
          index of the action actually taken -- NOT an MSE regression to the
          one-hot vector, which is what this used to compute.
        """
        phi, phi_next = self._phi(obs), self._phi(next_obs)
        pred_action_logits = self.inverse(torch.cat([phi, phi_next], dim=-1))
        pred_next = self.forward_model(torch.cat([phi, action], dim=-1))
        inv_loss = F.cross_entropy(pred_action_logits, action.argmax(dim=-1))
        fwd_loss = F.mse_loss(pred_next, phi_next.detach())
        return self.beta_forward * fwd_loss + self.beta_inverse * inv_loss
