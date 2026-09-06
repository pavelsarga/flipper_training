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
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.objectives import ClipPPOLoss

from marv_rl_training.policies import PolicyConfig
from marv_rl_training.policies.diffusion_policy import ConditionalUnet1D, ObsHistoryEncoder, SinusoidalPosEmb
from marv_rl_training.policies.diffusion_schedule import DiffusionSchedule
from marv_rl_training.utils.logutils import get_terminal_logger

_log = get_terminal_logger("DiffusionPolicyPhase2")

__all__ = ["DiffusionChunkActor", "DPPOClipLoss", "DPPOPerStepClipLoss",
           "DiffusionPolicyPhase2Config", "make_dppo_loss"]


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
        self.out_keys = ["action", "denoise_chain", "sample_log_prob", "denoise_logp_steps"]
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

    def sample_chain(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_emb = self.encoder(obs_history)
        n = obs_emb.shape[0]
        x = torch.randn(n, self.action_dim, self.prediction_horizon, device=obs_emb.device)
        chain = [x]
        steps: list[torch.Tensor] = []
        logp = torch.zeros(n, device=obs_emb.device)
        for k, k_prev in self.schedule.step_pairs():
            eps = self.unet(x, self._cond(obs_emb, k))
            mean, std = self.schedule.ddim_step(x, eps, k, k_prev)
            x = mean + std * torch.randn_like(mean)
            step_lp = torch.distributions.Normal(mean, std).log_prob(x).sum(dim=(-1, -2))
            steps.append(step_lp)
            logp = logp + step_lp
            chain.append(x)
        return self._chain_to_action(x), torch.stack(chain, dim=1), logp, torch.stack(steps, dim=1)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        action, chain, logp, steps = self.sample_chain(tensordict.get(self.obs_history_key))
        tensordict.set("action", action)
        tensordict.set("denoise_chain", chain)
        tensordict.set("sample_log_prob", logp)
        tensordict.set("denoise_logp_steps", steps)
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

    def chain_log_prob_steps(self, tensordict: TensorDict) -> torch.Tensor:
        """Per-denoising-step log-probabilities of the stored chain, ``[N, K]``."""
        obs_emb = self.encoder(tensordict.get(self.obs_history_key))
        chain = tensordict.get("denoise_chain")
        out = []
        for i, (k, k_prev) in enumerate(self.schedule.step_pairs()):
            eps = self.unet(chain[:, i], self._cond(obs_emb, k))
            mean, std = self.schedule.ddim_step(chain[:, i], eps, k, k_prev)
            out.append(torch.distributions.Normal(mean, std).log_prob(chain[:, i + 1]).sum(dim=(-1, -2)))
        return torch.stack(out, dim=1)


class DPPOClipLoss(ClipPPOLoss):
    """ClipPPOLoss scoring the stored denoising chain rather than a fresh sample."""

    def _get_cur_log_prob(self, tensordict):
        actor = self.actor_network
        with self.actor_network_params.to_module(actor) if self.functional else contextlib.nullcontext():
            log_prob = actor.chain_log_prob(tensordict)
        # dist=None is safe: ClipPPOLoss only uses it under `if self.entropy_bonus`, which
        # must be false for a diffusion policy (entropy is not tractable).
        return log_prob, None, False


class DPPOPerStepClipLoss(DPPOClipLoss):
    """DPPO with the PPO ratio formed and clipped PER DENOISING STEP.

    This is what DPPO actually advocates, and the default here rather than a fallback.
    ``DPPOClipLoss`` forms one ratio from the whole chain's log-probability, which sums
    ``K * T_p * A`` Gaussian terms; measured on this configuration (see
    training/test_dppo.py), a 1e-3 perturbation of eps_theta moves that sum by 3.2 nats
    against a clip threshold of log(1.2) = 0.182, so every sample clips at any sensible
    learning rate and the objective goes flat. One ratio per step spans ``T_p * A`` dims
    instead, which keeps each one inside the trust region.

    Formulation: the denoising chain is an MDP whose intermediate transitions carry zero
    reward, so every denoising step shares the environment advantage. The objective is the
    mean over steps of the usual clipped surrogate.
    """

    def forward(self, tensordict):
        from tensordict import TensorDict as _TD

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
            # Inline rather than torchrl.objectives.utils._standardize, which is private and
            # not exported in 0.8.1 — importing it made the loss fail at call time.
            advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)

        prev = tensordict.get("denoise_logp_steps")            # [N, K], from collection
        if prev is None:
            raise KeyError("denoise_logp_steps missing — the actor must store per-step log-probs")
        if prev.requires_grad:
            raise RuntimeError("stored denoise_logp_steps requires grad")

        import contextlib as _c
        with self.actor_network_params.to_module(self.actor_network) if self.functional else _c.nullcontext():
            cur = self.actor_network.chain_log_prob_steps(tensordict)   # [N, K]

        log_weight = cur - prev                                 # [N, K]
        adv = advantage.reshape(-1, 1)                          # same advantage for every step
        gain1 = log_weight.exp() * adv
        lw_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = lw_clip.exp() * adv
        gain = torch.stack([gain1, gain2], -1).min(dim=-1).values

        # torchrl's losses apply self.reduction at the end of forward, and every caller
        # relies on it: the trainer does loss_objective.backward() directly, which needs a
        # scalar. DPPOClipLoss inherits ClipPPOLoss.forward and gets this for free; this
        # class overrides forward, so it has to reduce explicitly. Without it the loss comes
        # back shaped [N] and backward() raises "grad can be implicitly created only for
        # scalar outputs" on the first update.
        reduction = getattr(self, "reduction", "mean")

        def _red(x):
            if reduction == "mean":
                return x.mean()
            if reduction == "sum":
                return x.sum()
            return x

        td_out = _TD({"loss_objective": _red(-gain.mean(dim=1))}, batch_size=[])
        td_out.set("clip_fraction", (lw_clip != log_weight).to(log_weight.dtype).mean().detach())
        td_out.set("kl_approx", (prev - cur).mean().detach())
        # Per-step diagnostics: if the later steps carry all the drift, K_infer is too long.
        td_out.set("clip_fraction_per_step", (lw_clip != log_weight).to(log_weight.dtype).mean(dim=0).detach())
        if self.critic_coef is not None:
            _lc = self.loss_critic(tensordict)
            # loss_critic returns (loss, value_clip_fraction) in some versions and a bare
            # tensor in others — accept both rather than assume.
            if isinstance(_lc, tuple):
                loss_critic, value_clip_fraction = _lc
            else:
                loss_critic, value_clip_fraction = _lc, None
            td_out.set("loss_critic", _red(loss_critic))
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        return td_out


@dataclass
class DiffusionPolicyPhase2Config(PolicyConfig):
    """Phase 2 actor-critic: DDIM diffusion actor + the same MLP critic as Phase 1.

    Deliberately mirrors DiffusionPolicyConfig so a Phase 1 checkpoint is a usable
    initialisation: same ObsHistoryEncoder, same ConditionalUnet1D geometry. Only the
    U-Net's output width changes (A instead of 2A — epsilon rather than loc/scale) and the
    conditioning vector gains the denoising-step embedding.

    Args:
        num_train_timesteps / num_inference_steps: K_train and K_infer.
        eta: DDIM stochasticity. Must be > 0 — a deterministic chain has no log-prob.
        min_sampling_std: Floor on the per-step sigma; the exploration knob that replaces
            entropy_coef. Measured to be a weak lever on ratio sensitivity (0.02 -> 0.2
            moved it 0.007 -> 0.004 nats), so treat it as exploration only.
        per_step_clipping: Use DPPOPerStepClipLoss instead of DPPOClipLoss. Chain-level is
            the default because it measured comfortably inside the trust region at
            realistic step sizes; this is the safety valve if KL misbehaves.
    """

    actor_optimizer_opts: dict
    value_optimizer_opts: dict
    value_mlp_opts: dict
    prediction_horizon: int = 4
    history_len: int = 2
    down_dims: list = field(default_factory=lambda: [64, 128])
    kernel_size: int = 5
    n_groups: int = 8
    step_embed_dim: int = 64
    num_train_timesteps: int = 100
    num_inference_steps: int = 8
    eta: float = 1.0
    min_sampling_std: float = 0.02
    per_step_clipping: bool = False
    obs_history_key: str = "obs_history"

    def create(self, env, **kwargs):
        from torchrl.modules import ActorCriticWrapper
        from tensordict.nn import TensorDictModule
        from marv_rl_training.policies.diffusion_policy import ChunkCriticNet

        chunk_dim = env.action_spec.shape[-1]
        action_dim = chunk_dim // self.prediction_horizon
        if action_dim * self.prediction_horizon != chunk_dim:
            raise ValueError(
                f"action_spec last dim ({chunk_dim}) is not prediction_horizon "
                f"({self.prediction_horizon}) x action_dim — is the env wrapped in ActionChunkEnv?"
            )
        observation = env.observations[0]
        device = kwargs.get("device", None)

        def _enc():
            return ObsHistoryEncoder(observation.get_encoder(), observation.dim, self.history_len)

        schedule = DiffusionSchedule(
            num_train_timesteps=self.num_train_timesteps,
            num_inference_steps=self.num_inference_steps,
            eta=self.eta,
            min_sampling_std=self.min_sampling_std,
        )
        actor = DiffusionChunkActor(
            encoder=_enc(), schedule=schedule, action_dim=action_dim,
            prediction_horizon=self.prediction_horizon, down_dims=list(self.down_dims),
            kernel_size=self.kernel_size, n_groups=self.n_groups,
            step_embed_dim=self.step_embed_dim, obs_history_key=self.obs_history_key,
        )
        critic = TensorDictModule(
            ChunkCriticNet(_enc(), dict(self.value_mlp_opts)),
            in_keys=[self.obs_history_key], out_keys=["state_value"],
        )
        wrapper = ActorCriticWrapper(policy_operator=actor, value_operator=critic)
        if device is not None:
            wrapper.to(device)

        optim_groups = [
            {"params": list(actor.parameters()), "name": "policy_operator", **self.actor_optimizer_opts},
            {"params": list(critic.parameters()), "name": "value_operator", **self.value_optimizer_opts},
        ]
        if weights_path := kwargs.get("weights_path", None):
            sd = torch.load(weights_path, map_location=device or "cpu")
            mu = wrapper.load_state_dict(sd, strict=False)
            _log.info(f"Loaded weights from {weights_path}")
            if mu.missing_keys:
                _log.warning(f"Missing keys: {mu.missing_keys}")
            if mu.unexpected_keys:
                _log.warning(f"Unexpected keys: {mu.unexpected_keys}")

        _log.info(
            "Diffusion policy (Phase 2): T_p=%d T_o=%d A=%d K_train=%d K_infer=%d eta=%.2f "
            "sigma_floor=%.3f per_step_clip=%s | actor %s params, critic %s params",
            self.prediction_horizon, self.history_len, action_dim, self.num_train_timesteps,
            self.num_inference_steps, self.eta, self.min_sampling_std, self.per_step_clipping,
            f"{sum(p.numel() for p in actor.parameters() if p.requires_grad):,}",
            f"{sum(p.numel() for p in critic.parameters() if p.requires_grad):,}",
        )
        return wrapper, optim_groups, []


def make_dppo_loss(policy_cfg, actor, critic, **ppo_opts):
    """Pick the DPPO loss class the policy config asks for, and refuse a fatal misconfig."""
    if ppo_opts.get("entropy_bonus", False):
        raise ValueError(
            "entropy_bonus must be false for a diffusion policy: its entropy is not tractable, "
            "the loss returns dist=None, and ClipPPOLoss would call dist.entropy(). Use the "
            "schedule's min_sampling_std for exploration instead."
        )
    cls = DPPOPerStepClipLoss if getattr(policy_cfg, "per_step_clipping", False) else DPPOClipLoss
    return cls(actor, critic, **ppo_opts)
