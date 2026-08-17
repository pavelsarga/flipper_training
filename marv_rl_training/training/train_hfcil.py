"""Stage 1 for HFC-IL: supervised training of the state-transition classifier on
demonstrations recorded from the MARV reactive controller's mode-8 state machine.

Pure PyTorch — no Isaac Sim, no environment stepping, CPU-runnable. It reads the .npz
shards described in paper_repos/HFC-IL-data-collection.md and writes a checkpoint
(encoder + classifier state_dicts) intended to eventually warm-start an
HFCILActorNet for rollout/eval — that wiring (an HFCILPolicyConfig, analogous to
hfc_policy.HFCPolicyConfig) doesn't exist yet; only training does so far.

    python -m marv_rl_training.training.train_hfcil --config configs/baselines/marv_config_hfcil.yaml

What is and isn't trained here
-----------------------------
Trained: HFCTerrainStateEncoder + HFCILClassifier (the parts with an imitation signal).
NOT trained: HFCActionDecoder's kp/kd/escape_mods. The demonstrations contain state labels
only — nothing in them identifies a roll-stabilisation gain or escape magnitude, so there
is no gradient for those here regardless. They are FIXED constants now (see
rl_modules.hfc.hfc_policy's DEFAULT_KP_RAD/DEFAULT_KD_RAD/DEFAULT_ESCAPE_RAD — sourced from
the real deployed reference controller, the paper's own Eq. 8 value where it conflicts),
not something any training stage optimises any more; an earlier version of this project
trained them via PPO instead and that did not work out.

The three things that make this different from a stock classification script
---------------------------------------------------------------------------
1. **Class imbalance is extreme and structural.** ~4 transitions per ~500-sample traversal
   means ~99% of pairs are "stay". Argmax accuracy is a useless metric (99% by predicting
   stay forever) and unweighted cross-entropy converges to exactly that degenerate policy.
   Handled by `transition_oversample` (a WeightedRandomSampler that raises the sampling rate
   of state-changing pairs) and by reporting per-edge recall, never bare accuracy.
2. **Splits must be by episode.** Consecutive samples at 10-20 Hz are near-duplicates; a
   random split puts near-copies of the same moment on both sides and inflates validation
   scores enormously. `_split_by_episode` partitions on the episode id.
3. **Legality is structural.** Illegal successors are already -inf from the classifier's
   mask, so cross-entropy is implicitly over the legal set only. A label that is illegal
   given its predecessor would produce -inf - (-inf) = NaN rather than a loud failure, so
   it is asserted against explicitly before training rather than trusting that the dataset
   validator was run.
"""
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers

from rl_modules.hfcil.hfcil_policy import HFCILClassifier
from rl_modules.hfc.hfc_policy import HFCTerrainStateEncoder
from rl_modules.hfc import hfc_policy as _hfc
from rl_modules.hfcil.hfcil_transitions import (
    LEGAL_SUCCESSORS,
    NUM_STATES,
    STATE_SHORT_NAMES,
    is_legal,
    legal_mask,
)
from rl_modules.hfcil import hfcil_dataset as D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s][%(levelname)s]: %(message)s")
_log = logging.getLogger("train_hfcil")

# Same workspace-root resolution as collect_ctrac_dataset.py / pretrain_ctrac_cvae.py: the
# .sbatch does `cd $WS` on the host, but $WS doesn't exist inside the container (only the
# /ws bind does), so cwd is not a reliable base for relative config paths.
_WS_ROOT = Path(__file__).resolve().parents[4]


def _resolve_ws_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _WS_ROOT / path


@dataclass
class HFCILPretrainConfig:
    dataset_path: str          # directory of shard_*.npz, or a single .npz
    output_path: str

    # Architecture — must match marv_config_hfcil.yaml's policy_opts or the checkpoint
    # won't load into the rollout policy.
    hm_hidden: int = 32       # HFCTerrainStateEncoder's terrain MLP width
    hm_out: int = 4           # terrain embedding width
    fusion_hidden: int = 32   # fused feature width == classifier input dim
    hidden_dim: int = 64      # HFCILClassifier hidden width

    total_epochs: int = 40
    batch_size: int = 256
    val_episode_frac: float = 0.2
    seed: int = 42
    device: str = "cpu"

    optimizer: Any = torch.optim.AdamW
    optimizer_opts: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3, "weight_decay": 1e-4})

    # Imbalance handling. transition_oversample = how many times more likely a
    # state-changing pair is to be drawn than a "stay" pair. 1.0 disables it (and will train
    # a degenerate always-stay classifier on real data — see this module's docstring).
    transition_oversample: float = 50.0
    # Additional per-class weighting inside the loss, on top of the sampler. Applied to the
    # *target* state. "balanced" = inverse frequency over the training split, GLOBALLY --
    # this can fight the sampler rather than help it: N is by far the most common s_next
    # overall (mostly AR->N/DR->N, which are already well represented), so "balanced" pushes
    # every prediction of N down, including the already-rare AF->N/DF->N abort edges, making
    # them matter LESS to the loss rather than more. "balanced_edges" fixes this by
    # normalizing per (current_state, next_state) instead of per next_state alone: every
    # current state's own set of legal successors (its "which branch do I take" decision)
    # gets equal total loss mass, regardless of how common that current state or its
    # successors are elsewhere in the dataset. null = off.
    class_weighting: "str | None" = "balanced_edges"

    early_stop_patience: int = 8


def _split_by_episode(episode: np.ndarray, val_frac: float, seed: int):
    """Episode-wise split — see this module's docstring for why a random per-sample split
    is invalid here."""
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_frac))) if len(eps) > 1 else 0
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episode])
    return ~is_val, is_val


def _assert_labels_legal(state: np.ndarray, next_state: np.ndarray) -> None:
    bad = np.array([not is_legal(int(a), int(b)) for a, b in zip(state, next_state)])
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{bad.sum()} training pairs have an illegal transition, e.g. "
            f"{STATE_SHORT_NAMES[state[i]]}->{STATE_SHORT_NAMES[next_state[i]]} at index {i}. "
            "The classifier masks illegal successors to -inf, so these would produce NaN loss "
            "rather than a clear error. Run hfcil_dataset.validate() on the recording — this "
            "means the label stream doesn't match Control.cpp's state machine."
        )


def _encode_inputs(obs: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    """HFCTerrainStateEncoder.forward takes the FULL 24-D observation and slices it itself
    (terrain / flippers / roll / pitch) using hfc_policy's own index constants — so hand it
    the whole vector rather than re-slicing here, which would risk the two disagreeing."""
    return encoder(obs)


@torch.no_grad()
def evaluate(encoder, classifier, obs, state, next_state, device, batch_size=4096) -> dict:
    """Per-edge recall + transition-only accuracy. Bare accuracy is deliberately NOT the
    headline metric — see this module's docstring."""
    encoder.eval(), classifier.eval()
    preds = []
    for i in range(0, len(obs), batch_size):
        o = obs[i : i + batch_size].to(device)
        s = state[i : i + batch_size].to(device)
        preds.append(classifier(_encode_inputs(o, encoder), s).argmax(-1).cpu())
    pred = torch.cat(preds)

    changed = state != next_state
    out = {
        "acc_all": float((pred == next_state).float().mean()),
        "acc_transitions": float((pred[changed] == next_state[changed]).float().mean()) if changed.any() else float("nan"),
        "acc_stay": float((pred[~changed] == next_state[~changed]).float().mean()) if (~changed).any() else float("nan"),
        "pred_transition_rate": float((pred != state).float().mean()),
        "true_transition_rate": float(changed.float().mean()),
    }
    per_edge = {}
    for s in range(NUM_STATES):
        for d in LEGAL_SUCCESSORS[s]:
            if d == s:
                continue
            m = (state == s) & (next_state == d)
            if m.any():
                per_edge[f"{STATE_SHORT_NAMES[s]}->{STATE_SHORT_NAMES[d]}"] = (
                    float((pred[m] == d).float().mean()), int(m.sum())
                )
    out["per_edge_recall"] = per_edge
    # Never predicts an illegal successor by construction; assert it rather than assume.
    lm = legal_mask()
    assert bool(lm[state, pred].all()), "classifier emitted an illegal successor"
    return out


def train(cfg: HFCILPretrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ds_path = _resolve_ws_path(cfg.dataset_path)
    obs_np, state_np, ep_np = D.load(ds_path)
    stats = D.validate(obs_np, state_np, ep_np)  # raises on illegal transitions
    _log.info(f"Loaded {stats['timesteps']} samples / {stats['episodes']} episodes from {ds_path}")
    _log.info(f"  {stats['transitions']} transitions, stay_fraction={stats['stay_fraction']:.4f}")
    _log.info("\n" + D.coverage_report(stats))

    o, s, s_next = D.build_pairs(obs_np, state_np, ep_np)
    ep_pairs = ep_np[:-1][ep_np[:-1] == ep_np[1:]]
    _assert_labels_legal(s, s_next)

    tr_m, va_m = _split_by_episode(ep_pairs, cfg.val_episode_frac, cfg.seed)
    _log.info(f"Split by episode: {tr_m.sum()} train / {va_m.sum()} val pairs "
              f"({len(np.unique(ep_pairs[tr_m]))} / {len(np.unique(ep_pairs[va_m]))} episodes)")

    t = lambda a, dt: torch.as_tensor(a, dtype=dt)
    obs_tr, s_tr, sn_tr = t(o[tr_m], torch.float32), t(s[tr_m], torch.long), t(s_next[tr_m], torch.long)
    obs_va, s_va, sn_va = t(o[va_m], torch.float32), t(s[va_m], torch.long), t(s_next[va_m], torch.long)

    # --- imbalance handling -------------------------------------------------
    changed_tr = (s_tr != sn_tr)
    weights = torch.where(changed_tr, torch.full_like(changed_tr, cfg.transition_oversample, dtype=torch.float32),
                          torch.ones(len(s_tr)))
    sampler = WeightedRandomSampler(weights.double(), num_samples=len(s_tr), replacement=True)
    _log.info(f"Transition oversampling x{cfg.transition_oversample:g}: effective transition rate "
              f"{float((weights * changed_tr).sum() / weights.sum()):.3f} (raw {float(changed_tr.float().mean()):.4f})")

    class_w = None
    sample_w = None
    if cfg.class_weighting == "balanced":
        counts = torch.bincount(sn_tr, minlength=NUM_STATES).float()
        class_w = torch.where(counts > 0, counts.sum() / (counts.clamp_min(1) * (counts > 0).sum()), torch.zeros_like(counts))
        _log.info("Class weights: " + ", ".join(f"{STATE_SHORT_NAMES[i]}={class_w[i]:.2f}" for i in range(NUM_STATES) if counts[i] > 0))
        class_w = class_w.to(device)
    elif cfg.class_weighting == "balanced_edges":
        # Per (s, s_next) edge, not per s_next alone: within EACH current state's own set of
        # observed successors, every successor gets equal total loss mass. A raw sample
        # weight of 1/count(s, s_next) makes each edge's total contribution exactly 1, so
        # normalizing by the number of distinct successors observed for that s makes every
        # source state's *choice* equally weighted too, independent of how many raw samples
        # or how many branches it has.
        edge_id = s_tr * NUM_STATES + sn_tr
        edge_counts = torch.bincount(edge_id, minlength=NUM_STATES * NUM_STATES).float()
        n_edges_per_state = (edge_counts.view(NUM_STATES, NUM_STATES) > 0).sum(dim=1).float()
        sample_w = 1.0 / edge_counts[edge_id]
        sample_w = sample_w / n_edges_per_state[s_tr]
        sample_w = sample_w * (len(sample_w) / sample_w.sum())  # keep loss magnitude comparable to unweighted
        sample_w = sample_w.to(device)
        for si in range(NUM_STATES):
            row = edge_counts[si * NUM_STATES:(si + 1) * NUM_STATES]
            if row.sum() == 0:
                continue
            edges = ", ".join(f"{STATE_SHORT_NAMES[di]}={int(row[di])}" for di in range(NUM_STATES) if row[di] > 0)
            _log.info(f"Edge weights from {STATE_SHORT_NAMES[si]}: {edges} (each gets equal total loss mass)")

    loader = DataLoader(TensorDataset(obs_tr, s_tr, sn_tr, torch.arange(len(s_tr))),
                         batch_size=cfg.batch_size, sampler=sampler, drop_last=True)

    encoder = HFCTerrainStateEncoder(hm_hidden=cfg.hm_hidden, hm_out=cfg.hm_out, fusion_hidden=cfg.fusion_hidden).to(device)
    classifier = HFCILClassifier(in_dim=cfg.fusion_hidden, hidden_dim=cfg.hidden_dim).to(device)
    optim = cfg.optimizer(list(encoder.parameters()) + list(classifier.parameters()), **(cfg.optimizer_opts or {}))

    out_path = _resolve_ws_path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log.info(f"Writing checkpoint to {out_path}")

    best, best_epoch, since = -1.0, -1, 0
    for epoch in range(cfg.total_epochs):
        encoder.train(), classifier.train()
        tot, nb = 0.0, 0
        for ob, st, nx, idx in loader:
            ob, st, nx = ob.to(device), st.to(device), nx.to(device)
            logits = classifier(_encode_inputs(ob, encoder), st)
            if sample_w is not None:
                per_sample = nn.functional.cross_entropy(logits, nx, reduction="none")
                loss = (per_sample * sample_w[idx.to(device)]).mean()
            else:
                loss = nn.functional.cross_entropy(logits, nx, weight=class_w)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1.0)
            optim.step()
            tot += float(loss)
            nb += 1

        m = evaluate(encoder, classifier, obs_va, s_va, sn_va, device)
        # Selection metric is transition recall AND stay accuracy, geometric-mean'd -- not
        # bare accuracy (a checkpoint that never transitions scores ~0.99 accuracy and is
        # worthless), but ALSO not acc_transitions alone (the opposite degenerate: a
        # checkpoint that transitions on nearly every tick scores acc_transitions ~0.93 while
        # firing on 90%+ of true-stay pairs -- acc_stay ~0.08 -- which is just as worthless in
        # rollout, and was silently winning "best" here before this was a geometric mean).
        # Geometric mean rather than average because it collapses to ~0 if EITHER side is
        # near-degenerate, whereas an average of 0.99/0.08 still looks like a mediocre 0.5.
        at = m["acc_transitions"] if not np.isnan(m["acc_transitions"]) else 0.0
        asy = m["acc_stay"] if not np.isnan(m["acc_stay"]) else 0.0
        score = float(np.sqrt(at * asy))
        _log.info(f"epoch {epoch:3d}  loss {tot/max(nb,1):.4f}  "
                  f"val acc_all {m['acc_all']:.4f}  score(sqrt(trans*stay)) {score:.4f}  "
                  f"acc_transitions {at:.4f}  acc_stay {asy:.4f}  "
                  f"pred_trans_rate {m['pred_transition_rate']:.4f} (true {m['true_transition_rate']:.4f})")
        if score > best:
            best, best_epoch, since = score, epoch, 0
            torch.save({"encoder": encoder.state_dict(), "classifier": classifier.state_dict(),
                        "hm_hidden": cfg.hm_hidden, "hm_out": cfg.hm_out,
                        "fusion_hidden": cfg.fusion_hidden, "hidden_dim": cfg.hidden_dim,
                        "val_acc_transitions": at, "val_acc_stay": asy, "val_score": score,
                        "epoch": epoch}, out_path)
        else:
            since += 1
            if since >= cfg.early_stop_patience:
                _log.info(f"Early stop: no score improvement for {since} epochs")
                break

    _log.info(f"Best score {best:.4f} at epoch {best_epoch} -> {out_path}")
    ck = torch.load(out_path, map_location=device)
    encoder.load_state_dict(ck["encoder"]), classifier.load_state_dict(ck["classifier"])
    final = evaluate(encoder, classifier, obs_va, s_va, sn_va, device)
    _log.info("Per-edge recall on held-out episodes (n = support):")
    for k, (r, n) in sorted(final["per_edge_recall"].items()):
        _log.info(f"   {k:<12} {r:.3f}  (n={n})")
    if final["pred_transition_rate"] < 0.5 * final["true_transition_rate"]:
        _log.warning(
            "Predicted transition rate is far below the true rate — the classifier is biased "
            "toward 'stay'. Raise transition_oversample, or collect more transitions."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation-learn the HFC state-transition classifier from mode-8 demonstrations")
    parser.add_argument("--config", type=str, required=True, help="Path to an HFC-IL pretrain config yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset_path in config")
    parser.add_argument("--output", type=str, default=None, help="Override output_path in config")
    args, overrides = parser.parse_known_args()

    raw = OmegaConf.load(args.config)
    if overrides:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist([o.lstrip("-") for o in overrides]))
    if args.dataset:
        raw.dataset_path = args.dataset
    if args.output:
        raw.output_path = args.output
    cfg = HFCILPretrainConfig(**OmegaConf.to_container(raw, resolve=True))
    train(cfg)
