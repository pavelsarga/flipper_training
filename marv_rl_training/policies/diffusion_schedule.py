"""Squared-cosine DDPM schedule + DDIM sampler, for the Phase 2 diffusion actor.

Written rather than pulled from `diffusers` on purpose: the apptainer image
`containers/isaaclab_optuna.sif` is fixed, and rebuilding it to add a pip package is a
separate job with its own risk. The subset actually needed here — betas, alphas_cumprod,
the forward noising step, and a DDIM update that exposes its Gaussian mean AND std — is
small and easy to test directly.

Exposing the std is the whole point. DPPO (Ren et al. 2024) makes PPO applicable to a
diffusion policy by treating each denoising step as a Gaussian transition

    pi(A^{k-1} | A^k, s) = N( mu_theta(A^k, s, k), sigma_k^2 I )

so the denoising chain has a tractable likelihood. A sampler that only returns the next
sample is useless for that; `ddim_step` returns (mean, std) and lets the caller sample and
score. That also means eta must be > 0 — a deterministic DDIM (eta=0) has sigma=0 and no
log-prob at all — and that a floor on sigma is what replaces PPO's entropy bonus as the
exploration knob, since diffusion entropy is not tractable.

Conventions: epsilon-prediction (the paper's Eq. 5), actions normalised to [-1, 1], and the
predicted A^0 clipped to that range at every step (the paper does this too, and without it
the early steps of a short inference chain can leave the action space).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["DiffusionSchedule"]


class DiffusionSchedule(nn.Module):
    """Squared-cosine (iDDPM) schedule with a strided DDIM sampler.

    Args:
        num_train_timesteps: K_train. The paper uses 100 for every task.
        num_inference_steps: K_infer, a strided subset of the training steps. The paper
            reports 0.1 s for 10 steps of a 67M-param network; ours is ~1.3M.
        eta: DDIM stochasticity. 0 is deterministic (and unusable under DPPO — no
            log-prob); 1 reproduces the DDPM posterior variance.
        min_sampling_std: Floor on the per-step sigma. This is the exploration knob that
            replaces entropy_coef in Phase 2 (DPPO's `min_sampling_denoising_std`).
        s: Offset in the cosine schedule, from Nichol & Dhariwal.
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 8,
        eta: float = 1.0,
        min_sampling_std: float = 0.02,
        s: float = 0.008,
        max_beta: float = 0.999,
    ):
        super().__init__()
        if not 1 <= num_inference_steps <= num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) must be in [1, num_train_timesteps ({num_train_timesteps})]"
            )
        if eta <= 0.0:
            raise ValueError("eta must be > 0: a deterministic DDIM has sigma=0 and no tractable log-prob, which DPPO needs")
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.eta = float(eta)
        self.min_sampling_std = float(min_sampling_std)

        # Squared-cosine alpha_bar, Nichol & Dhariwal Eq. 17. The paper found this worked
        # best of the schedules it tried.
        t = torch.arange(self.num_train_timesteps + 1, dtype=torch.float64) / self.num_train_timesteps
        f = torch.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = (f / f[0]).clamp(min=1e-8)
        betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=max_beta)

        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod[1:].float(), persistent=False)

        # Strided inference timesteps, descending, always ending at 0.
        stride = self.num_train_timesteps / self.num_inference_steps
        steps = [int(round(self.num_train_timesteps - 1 - i * stride)) for i in range(self.num_inference_steps)]
        steps = sorted({max(0, s_) for s_ in steps}, reverse=True)
        self.register_buffer("timesteps", torch.tensor(steps, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------ training

    def q_sample(self, x0: torch.Tensor, noise: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Forward noising: x_k = sqrt(a_k) x0 + sqrt(1 - a_k) * noise.

        `k` is a per-sample integer tensor; broadcasting is over the leading batch dim, so
        x0/noise may be [N, C, T] and k [N].
        """
        a = self.alphas_cumprod[k].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def loss_target(self, noise: torch.Tensor) -> torch.Tensor:
        """Epsilon-prediction: the regression target is the noise itself (paper Eq. 5)."""
        return noise

    # ------------------------------------------------------------------ sampling

    def _sigma(self, k: int, k_prev: int) -> torch.Tensor:
        a_k = self.alphas_cumprod[k]
        a_prev = self.alphas_cumprod[k_prev] if k_prev >= 0 else torch.ones_like(a_k)
        sigma = self.eta * ((1 - a_prev) / (1 - a_k)).sqrt() * (1 - a_k / a_prev).clamp(min=0).sqrt()
        return sigma.clamp(min=self.min_sampling_std)

    def ddim_step(
        self, x_k: torch.Tensor, eps: torch.Tensor, k: int, k_prev: int, clip_x0: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One denoising transition, returned as its Gaussian (mean, std).

        The caller samples ``x_{k-1} = mean + std * z`` and can score that draw with an
        ordinary Normal log-prob — which is exactly what makes the chain PPO-able.
        """
        a_k = self.alphas_cumprod[k]
        a_prev = self.alphas_cumprod[k_prev] if k_prev >= 0 else torch.ones_like(a_k)

        x0 = (x_k - (1 - a_k).sqrt() * eps) / a_k.sqrt()
        if clip_x0:
            # Actions live in [-1, 1]; without this a short chain can wander outside it.
            x0 = x0.clamp(-1.0, 1.0)

        sigma = self._sigma(k, k_prev)
        # Re-derive eps from the (possibly clipped) x0 so mean and x0 stay consistent.
        eps_c = (x_k - a_k.sqrt() * x0) / (1 - a_k).sqrt()
        dir_xt = (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps_c
        mean = a_prev.sqrt() * x0 + dir_xt
        return mean, sigma.expand_as(mean)

    def step_pairs(self) -> list[tuple[int, int]]:
        """(k, k_prev) pairs for a full inference chain, ending with k_prev = -1 (clean)."""
        ts = self.timesteps.tolist()
        return [(k, ts[i + 1] if i + 1 < len(ts) else -1) for i, k in enumerate(ts)]


if __name__ == "__main__":  # smoke check
    torch.manual_seed(0)
    sch = DiffusionSchedule(num_train_timesteps=100, num_inference_steps=8)
    print("timesteps:", sch.timesteps.tolist())
    print("alphas_cumprod: a_0=%.4f a_-1=%.6f (must fall monotonically to ~0)"
          % (sch.alphas_cumprod[0], sch.alphas_cumprod[-1]))
    assert torch.all(sch.alphas_cumprod[1:] < sch.alphas_cumprod[:-1]), "alphas_cumprod must be decreasing"
    assert torch.all(sch.betas > 0) and torch.all(sch.betas < 1)

    x0 = torch.rand(4, 6, 4) * 2 - 1
    noise = torch.randn_like(x0)
    k = torch.randint(0, 100, (4,))
    xk = sch.q_sample(x0, noise, k)
    print("q_sample:", tuple(xk.shape))

    # Denoise with an oracle eps: the chain must return roughly to x0.
    x = torch.randn_like(x0)
    for kk, kp in sch.step_pairs():
        a = sch.alphas_cumprod[kk]
        eps_oracle = (x - a.sqrt() * x0) / (1 - a).sqrt()
        mean, std = sch.ddim_step(x, eps_oracle, kk, kp)
        x = mean + std * torch.randn_like(mean)
        assert torch.isfinite(x).all()
    print("oracle-eps chain end error: %.4f (should be small)" % (x - x0).abs().mean().item())
    print("all schedule checks passed")
