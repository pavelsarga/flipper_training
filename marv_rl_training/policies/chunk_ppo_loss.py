"""ClipPPOLoss for a Gaussian action chunk whose executed prefix varies per sample.

Why this exists. The Phase 1 actor is a TanhNormal over the flattened T_p x A chunk, and
``ClipPPOLoss`` scores the likelihood ratio of the WHOLE chunk. The steps after the executed
prefix never touched the environment, so their sampled values are independent of the
advantage: they contribute zero expected gradient but full variance to the ratio — with
T_p=16 / T_a=4 that is 72 of the 96 ratio dimensions, enough on their own to push a sample
outside the clip band and zero the gradient of the 24 dimensions that mattered.

This loss masks the ratio (and the entropy bonus) to the executed prefix of each sample,
``exec_len`` steps long, which ActionChunkEnv reports per macro step (it is T_a for a fixed
horizon and a random draw when ``execution_horizon_std`` is set). Optionally it adds the
TAIL-CONSISTENCY term: the unexecuted steps of chunk t are regressed onto the policy's own
later plan for the same control steps (the mean action of chunk t+1, stop-gradient), so
each output slice has exactly one loss — PPO for the prefix, consistency for the tail — and
the two cannot fight over the same dimensions.

Both old and new per-dimension log-probs are computed from (loc, scale): the rollout-time
ones stored by the ProbabilisticActor, and the current network's. Independent dimensions
make the prefix log-prob a plain masked sum.
"""

from __future__ import annotations

import contextlib

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.modules import TanhNormal
from torchrl.objectives import ClipPPOLoss

__all__ = ["PrefixMaskedClipPPOLoss", "per_dim_tanh_normal", "chunk_step_mask"]


def per_dim_tanh_normal(dist_like, loc: torch.Tensor, scale: torch.Tensor) -> TanhNormal:
    """A TanhNormal with the same support/upscale as ``dist_like`` but per-dimension log-probs."""
    return TanhNormal(
        loc=loc,
        scale=scale,
        low=dist_like.low,
        high=dist_like.high,
        upscale=dist_like.upscale,
        event_dims=0,
        tanh_loc=False,  # dist_like.loc is already the transformed location when tanh_loc was set
    )


def chunk_step_mask(exec_len: torch.Tensor, prediction_horizon: int, action_dim: int) -> torch.Tensor:
    """[N, T_p * A] boolean mask of the executed prefix: chunk step j < exec_len."""
    steps = torch.arange(prediction_horizon, device=exec_len.device)
    m = steps[None, :] < exec_len.reshape(-1, 1)  # [N, T_p]
    return m.repeat_interleave(action_dim, dim=1)


class PrefixMaskedClipPPOLoss(ClipPPOLoss):
    def __init__(
        self,
        actor,
        critic,
        *,
        prediction_horizon: int,
        action_dim: int,
        tail_consistency_coef: float = 0.0,
        **ppo_opts,
    ):
        super().__init__(actor, critic, **ppo_opts)
        self.prediction_horizon = int(prediction_horizon)
        self.action_dim = int(action_dim)
        self.tail_consistency_coef = float(tail_consistency_coef)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
        advantage = advantage.reshape(-1)

        action = tensordict.get(self.tensor_keys.action)
        exec_len = tensordict.get("exec_len", None)
        if exec_len is None:
            exec_len = tensordict.get(("next", "exec_len"))
        mask = chunk_step_mask(exec_len, self.prediction_horizon, self.action_dim).to(action.dtype)

        # Read the rollout-time parameters BEFORE get_dist: the actor writes its current
        # loc/scale into the tensordict it is given.
        old_loc, old_scale = tensordict.get("loc"), tensordict.get("scale")
        if old_loc.requires_grad or old_scale.requires_grad:
            raise RuntimeError("stored loc/scale require grad")
        with self.actor_network_params.to_module(self.actor_network) if self.functional else contextlib.nullcontext():
            dist = self.actor_network.get_dist(tensordict)
        cur = per_dim_tanh_normal(dist, dist.loc, dist.scale)
        old = per_dim_tanh_normal(dist, old_loc, old_scale)

        logp_cur = cur.log_prob(action)   # [N, T_p*A]
        with torch.no_grad():
            logp_old = old.log_prob(action)
        log_weight = ((logp_cur - logp_old) * mask).sum(dim=-1)   # ratio over the executed prefix only
        kl_approx = ((logp_old - logp_cur) * mask).sum(dim=-1)

        gain1 = log_weight.exp() * advantage
        lw_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = lw_clip.exp() * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1).values

        reduction = getattr(self, "reduction", "mean")

        def _red(x):
            return x.mean() if reduction == "mean" else (x.sum() if reduction == "sum" else x)

        td_out = TensorDict({"loss_objective": _red(-gain)}, batch_size=[])
        td_out.set("clip_fraction", (lw_clip != log_weight).to(log_weight.dtype).mean().detach())
        td_out.set("kl_approx", kl_approx.mean().detach())
        td_out.set("exec_len_mean", exec_len.to(log_weight.dtype).mean().detach())

        if self.entropy_bonus:
            # torchrl's estimator (-log_prob of a fresh sample), per dimension, over the prefix.
            x = cur.rsample((self.samples_mc_entropy,))
            ent = (-cur.log_prob(x).mean(0) * mask).sum(dim=-1)
            td_out.set("entropy", ent.detach().mean())
            td_out.set("loss_entropy", _red(-self.entropy_coef * ent))

        if self.tail_consistency_coef > 0:
            target = tensordict.get("tail_target", None)
            tmask = tensordict.get("tail_mask", None)
            if target is None or tmask is None:
                raise KeyError("tail_consistency_coef > 0 needs tail_target / tail_mask in the batch")
            tmask = tmask.to(action.dtype)
            mode = cur.deterministic_sample  # mean action of the CURRENT policy, in action space
            n = tmask.sum(dim=-1).clamp_min(1.0)
            per_sample = (((mode - target) ** 2) * tmask).sum(dim=-1) / n
            has_tail = (tmask.sum(dim=-1) > 0).to(action.dtype)
            td_out.set("loss_tail", _red(self.tail_consistency_coef * per_sample * has_tail))
            td_out.set("tail_mse", (per_sample * has_tail).sum().detach() / has_tail.sum().clamp_min(1.0))

        if self._has_critic:
            _lc = self.loss_critic(tensordict)
            if isinstance(_lc, tuple):
                loss_critic, value_clip_fraction = _lc
            else:
                loss_critic, value_clip_fraction = _lc, None
            td_out.set("loss_critic", _red(loss_critic))
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        self._clear_weakrefs(
            tensordict, td_out,
            "actor_network_params", "critic_network_params",
            "target_actor_network_params", "target_critic_network_params",
        )
        return td_out
