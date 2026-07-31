"""Stage I: supervised C-VAE pretraining for C-TRAC (Pan et al. 2025), Eq. 9-14.

Deliberately independent of Isaac Sim / AppLauncher / gymnasium — this trains purely on a
dataset file collect_ctrac_dataset.py already produced (obs_history/obs/next_obs tensors),
so it's runnable on any machine (including this sandbox, unlike every other train_*.py in
this project) and doesn't need a GPU-backed physics scene at all.

Usage:
    python -m marv_rl_training.training.pretrain_ctrac_cvae --config <cvae_pretrain>.yaml
"""
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from rl_modules.ctrac.ctrac_cvae import CTRACCVAE, contact_est_loss, contact_geo_loss, contact_prob_loss, vae_loss
from rl_modules.ctrac.ctrac_observation import CONTACT_POINTS_OFFSET, CONTACT_PROB_OFFSET, PARTIAL_DIM

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
_log = logging.getLogger("pretrain_ctrac_cvae")


@dataclass
class CTRACCVAEPretrainConfig:
    name: str
    comment: str
    seed: int
    device: str
    dataset_path: str
    output_path: str
    total_steps: int
    batch_size: int
    history_len: int
    latent_dim: int = 32
    encoder_hidden: tuple = (512, 256, 128)
    decoder_hidden: tuple = (128, 256, 128)
    vae_beta: float = 1.0
    prob_weight: float = 1.0
    est_weight: float = 1.0
    geo_weight: float = 1.0
    max_reach: float = 0.8
    max_grad_norm: float = 0.5
    optimizer: type = torch.optim.Adam
    optimizer_opts: dict[str, Any] = field(default_factory=lambda: {"lr": 3e-4})
    log_every: int = 100
    save_every: int = 1000


def _load_raw_config(config_path: str, cli_overrides: list[str]):
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


def train(cfg: CTRACCVAEPretrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    dataset = torch.load(cfg.dataset_path, map_location="cpu")
    obs_history, obs, next_obs = dataset["obs_history"], dataset["obs"], dataset["next_obs"]
    n = obs.shape[0]
    _log.info(f"Loaded {n} transitions from {cfg.dataset_path}")

    cvae = CTRACCVAE(
        history_len=cfg.history_len, partial_dim=PARTIAL_DIM,
        encoder_hidden=tuple(cfg.encoder_hidden), decoder_hidden=tuple(cfg.decoder_hidden),
        latent_dim=cfg.latent_dim,
    ).to(device)
    optim = cfg.optimizer(cvae.parameters(), **(cfg.optimizer_opts or {}))

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(cfg.total_steps), desc="C-VAE pretraining", unit="step")
    for step in pbar:
        idx = torch.randint(0, n, (cfg.batch_size,))
        batch_hist = obs_history[idx].to(device)
        batch_obs = obs[idx].to(device)
        batch_next_obs = next_obs[idx].to(device)

        target_points = batch_obs[..., CONTACT_POINTS_OFFSET:CONTACT_PROB_OFFSET].reshape(-1, 4, 3)
        target_prob = batch_obs[..., CONTACT_PROB_OFFSET:CONTACT_PROB_OFFSET + 4]
        target_recon = batch_next_obs[..., :PARTIAL_DIM]

        _z, mu, logvar, pred_points, pred_prob, pred_recon = cvae(batch_hist, sample=True)

        loss_vae, recon_l, kl_l = vae_loss(pred_recon, target_recon, mu, logvar, beta=cfg.vae_beta)
        loss_prob = contact_prob_loss(pred_prob, target_prob)
        loss_est = contact_est_loss(pred_points, target_points, target_prob)
        loss_geo = contact_geo_loss(pred_points, target_prob, target_points.mean(dim=1), cfg.max_reach)
        loss = loss_vae + cfg.prob_weight * loss_prob + cfg.est_weight * loss_est + cfg.geo_weight * loss_geo

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), cfg.max_grad_norm, error_if_nonfinite=False)
        optim.step()

        if step % cfg.log_every == 0:
            pbar.set_postfix(loss=loss.item(), recon=recon_l.item(), kl=kl_l.item(), prob=loss_prob.item(), est=loss_est.item(), geo=loss_geo.item())
        if step % cfg.save_every == 0 and step > 0:
            torch.save(cvae.state_dict(), out_path)

    torch.save(cvae.state_dict(), out_path)
    _log.info(f"Saved final C-VAE checkpoint to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage I: supervised C-VAE pretraining for C-TRAC")
    parser.add_argument("--config", type=str, required=True, help="Path to a C-VAE pretrain config yaml")
    args, unknown_args = parser.parse_known_args()
    raw_cfg = _load_raw_config(args.config, unknown_args)
    cfg = CTRACCVAEPretrainConfig(**raw_cfg)
    train(cfg)
