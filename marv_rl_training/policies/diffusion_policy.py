"""Receding-horizon (Diffusion-Policy-style) actor-critic for MARV.

Implements the CNN backbone of Chi et al., *Diffusion Policy*: a 1-D temporal U-Net over
the action-horizon axis, conditioned by FiLM on an observation embedding (and, in Phase 2,
on the denoising step). The time-series diffusion transformer is deliberately not
implemented — the paper's own recommendation is to start with the CNN variant, which needs
far less hyperparameter tuning and won on their real-robot task.

Phase 1 (this file, ``DiffusionPolicyConfig``) uses the U-Net as an ordinary Gaussian head
over the flattened T_p x action_dim chunk, so ClipPPOLoss and GAE apply unchanged. Phase 2
reuses the very same ``ConditionalUnet1D`` as the noise predictor eps_theta; only the
output channel count and the conditioning vector change, so a Phase 1 checkpoint is a
usable initialisation for it.

The observation encoder is NOT new: ``MarvRLFlatObservation.get_encoder()`` (i.e.
``MarvRLCNNFlatEncoder``, configured by ``ftr_obs_encoder_opts``) is applied per history
frame with shared weights, matching the paper's "images in each timestep are encoded
independently and then concatenated". The added depth of this architecture is the 1-D
U-Net along the horizon axis, not a second spatial CNN.

See docs/diffusion_policy/03_network.md and 04_phase1_gaussian.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ActorCriticWrapper, NormalParamExtractor, ProbabilisticActor, TanhNormal

from marv_rl_training.policies import MLP, PolicyConfig
from marv_rl_training.utils.logutils import get_terminal_logger

__all__ = ["DiffusionPolicyConfig", "ConditionalUnet1D", "ObsHistoryEncoder"]

_log = get_terminal_logger("DiffusionPolicy")


# ----------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Standard transformer sinusoidal embedding of the (scalar) denoising step k."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb dim must be even, got {dim}")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / (half - 1)
        )
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish, the paper's basic temporal block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int):
        super().__init__()
        if out_channels % n_groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by n_groups ({n_groups})")
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning applied after the first, plus a 1x1 skip.

    FiLM (Perez et al.): ``h <- gamma * h + beta`` where ``[gamma, beta] = Linear(cond)``,
    broadcast across the temporal axis and applied channel-wise. This is the mechanism that
    lets the observation (and the denoising step) condition a network whose input is only
    the action sequence, which in turn is what allows the observation encoder to run once
    per chunk instead of once per denoising iteration.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.out_channels = out_channels
        self.block0 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.block1 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * out_channels))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.block0(x)
        scale, bias = self.cond_encoder(cond).view(-1, 2, self.out_channels, 1).unbind(dim=1)
        out = scale * out + bias
        out = self.block1(out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1-D temporal U-Net over the action horizon, FiLM-conditioned on ``cond``.

    Args:
        in_channels: Channels of the input sequence (= action_dim).
        out_channels: Channels of the output sequence. ``2 * action_dim`` for the Phase 1
            Gaussian head (loc and scale), ``action_dim`` for the Phase 2 noise predictor.
        horizon: T_p. Must be divisible by ``2 ** (len(down_dims) - 1)``.
        cond_dim: Width of the conditioning vector.
        down_dims: Channel width per resolution level.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        horizon: int,
        cond_dim: int,
        down_dims: list[int],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        n_levels = len(down_dims)
        if horizon % (2 ** (n_levels - 1)) != 0:
            raise ValueError(
                f"horizon ({horizon}) must be divisible by 2**(len(down_dims)-1) = {2 ** (n_levels - 1)}"
            )

        dims = [in_channels, *down_dims]
        self.down_modules = nn.ModuleList()
        for i in range(n_levels):
            d_in, d_out = dims[i], dims[i + 1]
            is_last = i >= n_levels - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(d_in, d_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(d_out, d_out, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(d_out, d_out, 3, stride=2, padding=1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.up_modules = nn.ModuleList()
        for i in range(n_levels - 1):
            # Walk back up: level n_levels-1-i, skip connection doubles the input channels.
            d_out = down_dims[n_levels - 2 - i]
            d_in = down_dims[n_levels - 1 - i]
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(d_in + d_in, d_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(d_out, d_out, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(d_out, d_out, 4, stride=2, padding=1),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], out_channels, 1),
        )
        # Zero-init the output layer so the head starts centred, matching the
        # small-output-layer initialisation the MLP policy gets from apply_baselines_init.
        nn.init.zeros_(self.final_conv[-1].weight)
        nn.init.zeros_(self.final_conv[-1].bias)

    def forward(self, sample: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """sample: [N, in_channels, T_p]; cond: [N, cond_dim] -> [N, out_channels, T_p]."""
        x = sample
        skips = []
        for res0, res1, downsample in self.down_modules:
            x = res0(x, cond)
            x = res1(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res0, res1, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res0(x, cond)
            x = res1(x, cond)
            x = upsample(x)

        # One skip is left over: the top level was never downsampled, so its activation has
        # already been carried through unchanged. Matches the reference implementation.
        return self.final_conv(x)


class ObsHistoryEncoder(nn.Module):
    """Applies the per-frame observation encoder across the T_o history with shared weights.

    Input is the flat ``obs_history`` key written by the CatFrames transform,
    ``[N, T_o * obs_dim]``; output is ``[N, T_o * frame_dim]``, the paper's "encode each
    timestep independently and concatenate".
    """

    def __init__(self, frame_encoder: nn.Module, obs_dim: int, history_len: int):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.output_dim = frame_encoder.output_dim * history_len

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        n = obs_history.shape[0]
        frames = obs_history.view(n * self.history_len, self.obs_dim)
        return self.frame_encoder(frames).view(n, -1)


# ----------------------------------------------------------------------------------
# Phase 1: Gaussian chunk actor
# ----------------------------------------------------------------------------------


class ChunkGaussianActorNet(nn.Module):
    """obs_history -> FiLM U-Net -> (loc, scale_raw) over the flattened T_p x A chunk.

    The U-Net's input is a learned constant sequence: with no denoising step to condition
    on, all the information arrives through FiLM. Phase 2 replaces that constant with the
    noisy action sequence A^k and adds the step embedding to ``cond`` — the architecture
    below is otherwise unchanged, which is what makes a Phase 1 checkpoint reusable.
    """

    def __init__(
        self,
        encoder: ObsHistoryEncoder,
        action_dim: int,
        prediction_horizon: int,
        down_dims: list[int],
        kernel_size: int,
        n_groups: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        self.query = nn.Parameter(torch.zeros(1, action_dim, prediction_horizon))
        self.unet = ConditionalUnet1D(
            in_channels=action_dim,
            out_channels=2 * action_dim,
            horizon=prediction_horizon,
            cond_dim=encoder.output_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.param_extractor = NormalParamExtractor()

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.encoder(obs_history)
        n = cond.shape[0]
        out = self.unet(self.query.expand(n, -1, -1), cond)  # [N, 2A, T_p]
        loc_c, scale_c = out[:, : self.action_dim], out[:, self.action_dim :]
        # [N, A, T_p] -> [N, T_p, A] -> flat, so the chunk is laid out step-major and the
        # env wrapper's reshape(N, T_p, A) recovers it.
        loc = loc_c.transpose(1, 2).reshape(n, -1)
        scale_raw = scale_c.transpose(1, 2).reshape(n, -1)
        return self.param_extractor(torch.cat([loc, scale_raw], dim=-1))


class ChunkCriticNet(nn.Module):
    """obs_history -> its own encoder copy -> MLP -> one state value per macro-step."""

    def __init__(self, encoder: ObsHistoryEncoder, mlp_opts: dict):
        super().__init__()
        self.encoder = encoder
        self.mlp = MLP(in_dim=encoder.output_dim, out_dim=1, **mlp_opts)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs_history))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass
class DiffusionPolicyConfig(PolicyConfig):
    """Phase 1 receding-horizon actor-critic (Gaussian head over the action chunk).

    Args:
        prediction_horizon: T_p — length of the predicted action trajectory. Must be
            divisible by ``2 ** (len(down_dims) - 1)``.
        history_len: T_o — number of stacked observation frames the CatFrames transform
            produces. Must match the transform's ``N``.
        down_dims: U-Net channel width per level. ``[64, 128, 256]`` is the default;
            ``[64, 128]`` is the small variant. Choose with scripts/bench_diffusion_head.py.
        kernel_size: Temporal conv kernel (the paper uses 5).
        n_groups: GroupNorm groups (the paper uses 8).
        value_mlp_opts: Options for the critic MLP (in/out features come from the encoder).
        actor_optimizer_opts / value_optimizer_opts: Per-param-group optimiser settings.
        obs_history_key: TensorDict key the CatFrames transform writes.
    """

    actor_optimizer_opts: dict[str, Any]
    value_optimizer_opts: dict[str, Any]
    value_mlp_opts: dict[str, Any]
    prediction_horizon: int = 16
    history_len: int = 2
    down_dims: list[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 5
    n_groups: int = 8
    extra_distribution_kwargs: dict = field(default_factory=dict)
    obs_history_key: str = "obs_history"

    def create(self, env, **kwargs):
        action_spec = env.action_spec
        chunk_dim = action_spec.shape[-1]
        action_dim = chunk_dim // self.prediction_horizon
        if action_dim * self.prediction_horizon != chunk_dim:
            raise ValueError(
                f"action_spec last dim ({chunk_dim}) is not prediction_horizon "
                f"({self.prediction_horizon}) x action_dim — is the env wrapped in ActionChunkEnv?"
            )

        observation = env.observations[0]
        obs_dim = observation.dim
        device = kwargs.get("device", None)

        def _build_encoder() -> ObsHistoryEncoder:
            # Reuses the observation's own factory, so ftr_obs_encoder_opts still configures
            # the per-frame CNN+MLP exactly as it does for the baseline MLP policy.
            return ObsHistoryEncoder(observation.get_encoder(), obs_dim, self.history_len)

        actor_net = ChunkGaussianActorNet(
            encoder=_build_encoder(),
            action_dim=action_dim,
            prediction_horizon=self.prediction_horizon,
            down_dims=list(self.down_dims),
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )
        # A separate encoder instance for the critic — the baseline runs share_encoder: false
        # and we keep that, so the value loss cannot drag the actor's perception around.
        critic_net = ChunkCriticNet(encoder=_build_encoder(), mlp_opts=dict(self.value_mlp_opts))

        actor_module = ProbabilisticActor(
            module=TensorDictModule(actor_net, in_keys=[self.obs_history_key], out_keys=["loc", "scale"]),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low[0],
                "high": action_spec.space.high[0],
                **self.extra_distribution_kwargs,
            },
            return_log_prob=True,
        )
        value_operator = TensorDictModule(
            critic_net, in_keys=[self.obs_history_key], out_keys=["state_value"]
        )
        wrapper = ActorCriticWrapper(policy_operator=actor_module, value_operator=value_operator)
        if device is not None:
            wrapper.to(device)

        optim_groups = [
            {"params": list(actor_module.parameters()), "name": "policy_operator", **self.actor_optimizer_opts},
            {"params": list(value_operator.parameters()), "name": "value_operator", **self.value_optimizer_opts},
        ]

        if weights_path := kwargs.get("weights_path", None):
            sd = torch.load(weights_path, map_location=device or "cpu")
            missing_unexpected = wrapper.load_state_dict(sd, strict=False)
            _log.info(f"Loaded policy weights from {weights_path}")
            if missing_unexpected.missing_keys:
                _log.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                _log.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        _log.info(
            "Diffusion policy (Phase 1): T_p=%d T_o=%d action_dim=%d down_dims=%s | "
            "actor %s params (encoder %s, unet %s), critic %s params",
            self.prediction_horizon,
            self.history_len,
            action_dim,
            list(self.down_dims),
            f"{count_parameters(actor_net):,}",
            f"{count_parameters(actor_net.encoder):,}",
            f"{count_parameters(actor_net.unet):,}",
            f"{count_parameters(critic_net):,}",
        )
        return wrapper, optim_groups, []
