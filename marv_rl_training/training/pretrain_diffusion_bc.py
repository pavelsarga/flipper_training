"""BC-pretrain the Phase 2 diffusion actor on chunks from a trained marv_rl policy.

Stage 1 of the warm start: fit eps_theta to demonstration chunks with the paper's
epsilon-MSE objective (Eq. 5), so RL fine-tuning begins from a policy that already produces
plausible action trajectories. DPPO is normally applied to a BC-pretrained diffusion model
rather than from scratch — a randomly initialised eps_theta emits noise chunks, and the
gradients through an 8-step denoising chain from that starting point are high variance.

Deliberately needs NO Isaac Sim: the observation encoder and the diffusion modules import
without `omni`, so this runs as a plain script on any GPU (and on CPU for tests). Only the
dataset collection needs a simulator.

Reads the shards written by collect_chunk_dataset.py and writes a checkpoint that
`DiffusionPolicyPhase2Config(bc_weights_path=...)` loads straight into the actor.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.policies.diffusion_dppo import DiffusionChunkActor
from marv_rl_training.policies.diffusion_policy import ObsHistoryEncoder
from marv_rl_training.policies.diffusion_schedule import DiffusionSchedule
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all, set_device

logger = get_terminal_logger("pretrain_diffusion_bc")
_WS_ROOT = Path(__file__).resolve().parents[4]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else _WS_ROOT / q


@dataclass
class BCPretrainConfig:
    name: str
    dataset_path: str
    output_path: str
    obs_dim: int = 966
    action_dim: int = 6
    prediction_horizon: int = 4
    history_len: int = 2
    down_dims: list = field(default_factory=lambda: [64, 128])
    kernel_size: int = 5
    n_groups: int = 8
    step_embed_dim: int = 64
    num_train_timesteps: int = 100
    encoder_opts: dict[str, Any] = field(
        default_factory=lambda: dict(num_hidden=3, hidden_dim=256, output_dim=128, layernorm=True)
    )
    # Paper's CNN-variant optimiser settings.
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple = (0.95, 0.999)
    warmup_steps: int = 500
    epochs: int = 100
    batch_size: int = 256
    ema_power: float = 0.75
    ema_max: float = 0.9999
    val_frac: float = 0.1
    seed: int = 42
    device: str = "cuda"
    log_every: int = 50
    max_rows: int | None = None


class EMA:
    """Exponential moving average of the weights, as the paper uses.

    The EMA copy is what gets saved: DDPM training is noisy per-batch and the averaged
    weights are consistently the better initialisation.
    """

    def __init__(self, model: nn.Module, power: float, max_value: float):
        self.power, self.max_value = power, max_value
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.step = 0

    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = min(self.max_value, (1 + self.step) / (10 + self.step) ** self.power)
        decay = min(max(decay, 0.0), self.max_value)
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(decay).add_(v.detach().float(), alpha=1 - decay)

    def state_dict(self, model: nn.Module) -> dict:
        out = dict(model.state_dict())
        for k, v in self.shadow.items():
            out[k] = v.to(out[k].dtype)
        return out


def load_shards(path: Path, max_rows: int | None):
    files = sorted(glob.glob(str(path / "shard_*.npz"))) if path.is_dir() else [str(path)]
    if not files:
        raise FileNotFoundError(f"no shard_*.npz under {path}")
    obs, act, n = [], [], 0
    for f in files:
        d = np.load(f)
        obs.append(d["obs_history"])
        act.append(d["action_chunk"])
        n += obs[-1].shape[0]
        if max_rows is not None and n >= max_rows:
            break
    o = torch.from_numpy(np.concatenate(obs)).float()
    a = torch.from_numpy(np.concatenate(act)).float()
    if max_rows is not None:
        o, a = o[:max_rows], a[:max_rows]
    logger.info(f"loaded {o.shape[0]} pairs from {len(files)} shard(s) under {path}")
    return o, a


def build_actor(cfg: BCPretrainConfig) -> DiffusionChunkActor:
    # Imported here so the module still imports when FTR-Benchmark is not on the path
    # (e.g. when only the unit tests are being run).
    from rl_modules.marv_rl.marv_rl_cnn_flat_encoder import MarvRLCNNFlatEncoder

    enc = ObsHistoryEncoder(
        MarvRLCNNFlatEncoder(input_dim=cfg.obs_dim, **cfg.encoder_opts), cfg.obs_dim, cfg.history_len
    )
    sched = DiffusionSchedule(num_train_timesteps=cfg.num_train_timesteps, num_inference_steps=8)
    return DiffusionChunkActor(
        encoder=enc, schedule=sched, action_dim=cfg.action_dim,
        prediction_horizon=cfg.prediction_horizon, down_dims=list(cfg.down_dims),
        kernel_size=cfg.kernel_size, n_groups=cfg.n_groups, step_embed_dim=cfg.step_embed_dim,
    )


def epsilon_loss(actor: DiffusionChunkActor, obs: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 5: predict the noise added to a clean action chunk at a random step k."""
    n = obs.shape[0]
    # [N, T_p, A] -> [N, A, T_p]; the U-Net convolves along the horizon.
    x0 = chunk.transpose(1, 2)
    k = torch.randint(0, actor.schedule.num_train_timesteps, (n,), device=obs.device)
    noise = torch.randn_like(x0)
    xk = actor.schedule.q_sample(x0, noise, k)
    obs_emb = actor.encoder(obs)
    kt = k.to(obs_emb.dtype)
    cond = torch.cat([obs_emb, actor.step_embed(kt)], dim=-1)
    return nn.functional.mse_loss(actor.unet(xk, cond), noise)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="override dataset_path")
    ap.add_argument("--output", default=None, help="override output_path")
    args, unknown = ap.parse_known_args()

    raw = OmegaConf.load(args.config)
    if unknown:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(unknown))
    if args.dataset:
        raw.dataset_path = args.dataset
    if args.output:
        raw.output_path = args.output
    cfg = BCPretrainConfig(**raw)

    device = set_device(cfg.device)
    seed_all(cfg.seed)

    obs, act = load_shards(_resolve(cfg.dataset_path), cfg.max_rows)
    if act.shape[1] != cfg.prediction_horizon or act.shape[2] != cfg.action_dim:
        raise ValueError(
            f"dataset chunks are {tuple(act.shape[1:])} but the config expects "
            f"({cfg.prediction_horizon}, {cfg.action_dim}) — the collector and this config must agree"
        )
    if obs.shape[1] != cfg.history_len * cfg.obs_dim:
        raise ValueError(
            f"obs_history is {obs.shape[1]} wide but history_len*obs_dim is "
            f"{cfg.history_len * cfg.obs_dim} — T_o mismatch between collector and config"
        )

    n_val = max(1, int(obs.shape[0] * cfg.val_frac))
    perm = torch.randperm(obs.shape[0])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    logger.info(f"train {len(tr_idx)} / val {len(val_idx)} pairs")

    actor = build_actor(cfg).to(device)
    n_par = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    logger.info(f"actor {n_par:,} params | T_p={cfg.prediction_horizon} T_o={cfg.history_len} "
                f"K_train={cfg.num_train_timesteps} down_dims={list(cfg.down_dims)}")

    opt = torch.optim.AdamW(actor.parameters(), lr=cfg.lr, betas=tuple(cfg.betas),
                            weight_decay=cfg.weight_decay, eps=1e-8)
    steps_per_epoch = max(1, len(tr_idx) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = EMA(actor, cfg.ema_power, cfg.ema_max)

    out = _resolve(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    gstep = 0

    for epoch in range(cfg.epochs):
        actor.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        run = 0.0
        for i in range(steps_per_epoch):
            b = order[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            loss = epsilon_loss(actor, obs[b].to(device), act[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(actor)
            run += loss.item()
            gstep += 1
            if gstep % cfg.log_every == 0:
                logger.info(f"  epoch {epoch} step {gstep}/{total_steps} loss {run / (i + 1):.5f} "
                            f"lr {sched.get_last_lr()[0]:.2e}")
        # Validate on the EMA weights — those are what gets saved and used.
        actor.eval()
        backup = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        actor.load_state_dict(ema.state_dict(actor))
        with torch.no_grad():
            vb = val_idx[: cfg.batch_size * 4]
            vloss = epsilon_loss(actor, obs[vb].to(device), act[vb].to(device)).item()
        if vloss < best:
            best = vloss
            torch.save(actor.state_dict(), out)
            logger.info(f"epoch {epoch}: val {vloss:.5f} (best) -> saved {out}")
        else:
            logger.info(f"epoch {epoch}: val {vloss:.5f}")
        actor.load_state_dict(backup)

    logger.info(f"done. best val epsilon-MSE {best:.5f}; checkpoint at {out}")
    logger.info("load it with DiffusionPolicyPhase2Config(bc_weights_path=...)")


if __name__ == "__main__":
    main()
