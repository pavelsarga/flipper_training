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

from rl_modules.ctrac.ctrac_cvae import (
    CTRACCVAE,
    contact_est_loss,
    contact_geo_loss,
    contact_prob_loss,
    latent_diagnostics,
    vae_loss,
)
from rl_modules.ctrac.ctrac_observation import CONTACT_POINTS_OFFSET, CONTACT_PROB_OFFSET, PARTIAL_DIM

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
_log = logging.getLogger("pretrain_ctrac_cvae")

# See collect_ctrac_dataset.py's copy of this: the SLURM .sbatch does `cd $WS` on the host,
# but $WS doesn't exist inside the apptainer container (only the /ws bind does), so cwd is
# not a reliable base for relative config paths. Resolve them against the workspace root
# instead, which is /ws in the container and the checkout root locally.
_WS_ROOT = Path(__file__).resolve().parents[4]


def _resolve_ws_path(p: str) -> Path:
    """Absolute paths pass through; relative ones resolve against the workspace root."""
    path = Path(p)
    return path if path.is_absolute() else _WS_ROOT / path


@dataclass
class CTRACCVAEPretrainConfig:
    name: str
    comment: str
    seed: int
    device: str
    dataset_path: str        # a directory of shard_*.pt files (collect_ctrac_dataset.py's
                              # output_path), or a single legacy .pt file
    output_path: str
    total_steps: int
    batch_size: int
    history_len: int
    # Caps how many transitions get loaded into RAM at once, regardless of how large the
    # collected dataset on disk is — shards beyond this budget are proportionally
    # subsampled rather than dropped, so the loaded set still spans the whole collection
    # run (not just the first few shards). None = load everything (fine for small/legacy
    # single-file datasets, risky for a large sharded collection run).
    max_dataset_rows: "int | None" = 2_000_000
    latent_dim: int = 32
    encoder_hidden: tuple = (512, 256, 128)
    decoder_hidden: tuple = (128, 256, 128)
    # See train_sac.py's cvae_vae_beta / cvae_free_bits and vae_loss()'s docstring: at
    # beta=1.0 the KL term swamps a reconstruction loss whose natural scale here is ~0.08
    # and the latent collapses to the prior. Stage I must be kept consistent with Stage II
    # on both of these, otherwise the pretrained encoder handed to the actor is already dead
    # (or is trained under a different objective than the one that continues it).
    vae_beta: float = 0.01
    free_bits: float = 0.5
    prob_weight: float = 1.0
    est_weight: float = 1.0
    geo_weight: float = 1.0
    max_reach: float = 0.8
    max_grad_norm: float = 0.5
    # Exclude episode-boundary transitions from the dataset (see train()). Keep it on: with
    # it off, a small learning rate never learns contact existence at all.
    skip_reset_frames: bool = True
    optimizer: type = torch.optim.Adam
    optimizer_opts: dict[str, Any] = field(default_factory=lambda: {"lr": 3e-4})
    log_every: int = 100
    save_every: int = 1000


def _load_raw_config(config_path: str, cli_overrides: list[str]):
    parsed = OmegaConf.load(config_path)
    if cli_overrides:
        parsed = OmegaConf.merge(parsed, OmegaConf.from_dotlist(cli_overrides))
    return parsed


def _load_dataset(dataset_path: str, max_rows: "int | None"):
    """Loads either a directory of collect_ctrac_dataset.py shard_*.pt files or a single
    legacy .pt file. If max_rows is set and the on-disk total exceeds it, each shard is
    proportionally subsampled (not just the first N shards) so the loaded set still spans
    the whole collection run rather than only its earliest episodes.

    Two passes over the shards, BOTH using torch.load(..., mmap=True): a memory-mapped
    load only pages in the bytes actually touched, so Pass 1 (reading .shape[0] to size
    things) costs next to nothing regardless of how big any individual shard file is —
    this matters because two prior fixes at this OOM (accumulate-then-cat, then a
    pre-allocated-but-still-fully-materialized reload) both still assumed shard sizes in
    the ~1 GB range; if the dataset on disk doesn't actually match that assumption (e.g. a
    collection run from before shard_size_steps/log_every_n_steps existed, one giant
    single-file "shard"), a plain torch.load() of even ONE such file can OOM outright
    before this function logs anything. Pass 2 copies each shard's (possibly subsampled)
    rows into a PRE-ALLOCATED destination tensor sized to the final row count — with mmap,
    only the copied bytes get paged in, so peak RAM is bounded by max_rows (the
    destination) plus whatever fraction of the current shard is actually being copied,
    not by shard size at all.
    """
    p = _resolve_ws_path(dataset_path)
    if p.is_dir():
        shard_paths = sorted(p.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {p}")
    else:
        shard_paths = [p]
    _log.info(f"Found {len(shard_paths)} shard file(s) under {dataset_path}.")

    # Pass 1: shard sizes only (mmap — doesn't materialize tensor data).
    shard_sizes = []
    for i, sp in enumerate(shard_paths):
        n_rows = torch.load(sp, map_location="cpu", mmap=True)["obs"].shape[0]
        shard_sizes.append(n_rows)
        _log.info(f"  shard {i + 1}/{len(shard_paths)} ({sp.name}): {n_rows} rows")
    total_rows = sum(shard_sizes)

    if max_rows is not None and total_rows > max_rows:
        keep_fraction = max_rows / total_rows
        keep_counts = [max(1, round(n * keep_fraction)) for n in shard_sizes]
        _log.info(f"Dataset has {total_rows} transitions across {len(shard_paths)} shards; "
                  f"subsampling to ~{sum(keep_counts)} (keep_fraction={keep_fraction:.4f}).")
    else:
        keep_counts = shard_sizes

    final_n = sum(keep_counts)
    first_shard = torch.load(shard_paths[0], map_location="cpu", mmap=True)
    obs_history_shape, obs_history_dtype = first_shard["obs_history"].shape[1:], first_shard["obs_history"].dtype
    obs_shape, obs_dtype = first_shard["obs"].shape[1:], first_shard["obs"].dtype
    next_obs_shape, next_obs_dtype = first_shard["next_obs"].shape[1:], first_shard["next_obs"].dtype
    del first_shard

    row_bytes = (torch.Size(obs_history_shape).numel() * obs_history_dtype.itemsize
                 + torch.Size(obs_shape).numel() * obs_dtype.itemsize
                 + torch.Size(next_obs_shape).numel() * next_obs_dtype.itemsize)
    _log.info(f"Allocating destination tensors for {final_n} rows (~{final_n * row_bytes / 1e9:.2f} GB).")

    obs_history_out = torch.empty(final_n, *obs_history_shape, dtype=obs_history_dtype)
    obs_out = torch.empty(final_n, *obs_shape, dtype=obs_dtype)
    next_obs_out = torch.empty(final_n, *next_obs_shape, dtype=next_obs_dtype)

    # Pass 2: reload each shard (mmap'd, one at a time), copy its (subsampled) rows in.
    cursor = 0
    for i, (sp, n_rows, keep_n) in enumerate(zip(shard_paths, shard_sizes, keep_counts)):
        shard = torch.load(sp, map_location="cpu", mmap=True)
        if keep_n < n_rows:
            idx = torch.randperm(n_rows)[:keep_n]
            obs_history_out[cursor:cursor + keep_n] = shard["obs_history"][idx]
            obs_out[cursor:cursor + keep_n] = shard["obs"][idx]
            next_obs_out[cursor:cursor + keep_n] = shard["next_obs"][idx]
        else:
            obs_history_out[cursor:cursor + keep_n] = shard["obs_history"]
            obs_out[cursor:cursor + keep_n] = shard["obs"]
            next_obs_out[cursor:cursor + keep_n] = shard["next_obs"]
        cursor += keep_n
        del shard
        _log.info(f"  loaded shard {i + 1}/{len(shard_paths)} ({cursor}/{final_n} rows so far)")

    return obs_history_out, obs_out, next_obs_out


def train(cfg: CTRACCVAEPretrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    obs_history, obs, next_obs = _load_dataset(cfg.dataset_path, cfg.max_dataset_rows)
    _log.info(f"Loaded {obs.shape[0]} transitions from {cfg.dataset_path}")

    # Drop episode-boundary rows once, up front, rather than per batch — same filter and
    # same three reasons as train_sac.py's _cvae_update (o_{t+1} crosses into a different
    # episode; the obs_history window is 16 copies of one frame; and on any dataset
    # collected before ftr_env._refresh_state_after_reset existed the row also carries a
    # stale robot pose, putting ground-truth contact points up to 87 m from the base and a
    # mask-weighted L_est of 405 on it against 0.006 for a clean row).
    #
    # Keeping them is not survivable at a small learning rate. Measured on shards 0-2 from
    # random init, 4000 steps: at lr 3e-4 the contact head reaches 73.7% accuracy with them
    # and 71.5% at lr 1e-5 WITHOUT them — but only 56.5% at lr 1e-5 WITH them, i.e. it never
    # leaves the 55.8% chance level. The corruption is survivable only if the step size is
    # large enough to punch through it.
    # Kept as an INDEX, never as a filtered copy. `obs_history[keep]` with a boolean mask
    # allocates a second tensor ~97% the size of the first, which would double the peak RAM
    # this function was carefully written to bound (see _load_dataset: mmap + pre-allocated
    # destination, so peak is the dataset itself). At max_dataset_rows 3e6 and history_len
    # 16 that is 78 GB -> 155 GB, i.e. an OOM instead of a filter. Indexing costs 8 B/row.
    if cfg.skip_reset_frames:
        keep = (obs[:, PARTIAL_DIM - 1] == 0) & (next_obs[:, PARTIAL_DIM - 1] == 0)
        row_idx = keep.nonzero(as_tuple=False).squeeze(1)
        n_drop = int(obs.shape[0] - row_idx.numel())
        _log.info(f"Dropped {n_drop} episode-boundary transitions "
                  f"({100.0 * n_drop / max(obs.shape[0], 1):.2f}%); {row_idx.numel()} remain")
        del keep
    else:
        row_idx = torch.arange(obs.shape[0])
    n = row_idx.numel()

    cvae = CTRACCVAE(
        history_len=cfg.history_len, partial_dim=PARTIAL_DIM,
        encoder_hidden=tuple(cfg.encoder_hidden), decoder_hidden=tuple(cfg.decoder_hidden),
        latent_dim=cfg.latent_dim,
    ).to(device)
    optim = cfg.optimizer(cvae.parameters(), **(cfg.optimizer_opts or {}))

    out_path = _resolve_ws_path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(cfg.total_steps), desc="C-VAE pretraining", unit="step")
    for step in pbar:
        idx = row_idx[torch.randint(0, n, (cfg.batch_size,))]
        batch_hist = obs_history[idx].to(device)
        batch_obs = obs[idx].to(device)
        batch_next_obs = next_obs[idx].to(device)

        target_points = batch_obs[..., CONTACT_POINTS_OFFSET:CONTACT_PROB_OFFSET].reshape(-1, 4, 3)
        target_prob = batch_obs[..., CONTACT_PROB_OFFSET:CONTACT_PROB_OFFSET + 4]
        target_recon = batch_next_obs[..., :PARTIAL_DIM]

        _z, mu, logvar, pred_points, pred_prob, pred_recon = cvae(batch_hist, sample=True)

        loss_vae, recon_l, kl_l = vae_loss(pred_recon, target_recon, mu, logvar, beta=cfg.vae_beta, free_bits=cfg.free_bits)
        loss_prob = contact_prob_loss(pred_prob, target_prob)
        loss_est = contact_est_loss(pred_points, target_points, target_prob)
        loss_geo = contact_geo_loss(pred_points, target_prob, cfg.max_reach)
        loss = loss_vae + cfg.prob_weight * loss_prob + cfg.est_weight * loss_est + cfg.geo_weight * loss_geo

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), cfg.max_grad_norm, error_if_nonfinite=False)
        optim.step()

        if step % cfg.log_every == 0:
            # active_dims is the collapse tell — if it reaches 0 the encoder is emitting
            # pure N(0, I) and the checkpoint this stage produces is worthless downstream.
            diag = latent_diagnostics(mu, logvar)
            pbar.set_postfix(
                loss=loss.item(), recon=recon_l.item(), kl=kl_l.item(), prob=loss_prob.item(),
                est=loss_est.item(), geo=loss_geo.item(), active_dims=diag["cvae_latent_active_dims"],
            )
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
