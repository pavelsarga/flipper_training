"""BC-pretrain the Phase 1 receding-horizon actor for flipper state mode on a converted HFC-IL dataset.

Input: the .npz written by hfcil_to_chunk_dataset.py (per-step obs, action targets, valid mask,
state, episode) and the training config the policy will be fine-tuned with, so the actor is
built by the same DiffusionPolicyConfig.create call train_diffusion.py makes and its weights
load there via policy_weights_path.

Loss per chunk step: cross-entropy of the state targets on the pre-tanh loc of the state-score
dims (the env takes their argmax), plus MSE of tanh(loc) against the v / w targets. The Gaussian
scale is not trained, so PPO starts with its usual exploration noise.

Normalisation: the observations are normalised with --reference_vecnorm, a VecNorm checkpoint
from a real training run of the same observation layout, for the dims both share (heightmap,
state, v, w). PPO's VecNorm replaces its statistics with real-data ones within the first update,
so statistics computed from the synthetic observations would make the BC input space differ from
the one fine-tuning runs in. The state-score dims, new in state mode, use the dataset's statistics.
The saved vecnorm_bc.pth carries those statistics and the reference run's reward statistics.

Needs no Isaac Sim; runs on CPU or GPU inside the container.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Bounded

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY
from marv_rl_training.policies.diffusion_policy import ChunkGaussianActorNet, ChunkMLPActorNet
from marv_rl_training.utils.logutils import get_terminal_logger
from marv_rl_training.utils.torch_utils import seed_all
from rl_modules.marv_rl.marv_rl_flat_observation import MarvRLFlatObservation

logger = get_terminal_logger("pretrain_state_bc")


class _StepEnvStub:
    """What MarvRLFlatObservation reads: a per-step action spec."""

    def __init__(self, action_dim: int, device: torch.device):
        self.batch_size = torch.Size([1])
        self.device = device
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(1, action_dim), device=device, dtype=torch.float32)


class _ChunkEnvStub:
    """What DiffusionPolicyConfig.create reads: the chunk action spec and the observation."""

    def __init__(self, observation, action_dim: int, horizon: int, device: torch.device):
        self.batch_size = torch.Size([1])
        self.device = device
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(1, horizon * action_dim), device=device, dtype=torch.float32)
        self.observations = [observation]


def windows(valid: np.ndarray, episode: np.ndarray, T_o: int, T_p: int):
    """(obs index (M, T_o), target index (M, T_p)) for every t whose T_p targets stay in its episode."""
    T = len(valid)
    starts = np.zeros(T, dtype=np.int64)
    new_ep = np.r_[True, episode[1:] != episode[:-1]]
    starts[new_ep] = np.nonzero(new_ep)[0]
    starts = np.maximum.accumulate(starts)
    cs = np.r_[0, np.cumsum(valid)]
    t = np.arange(T - T_p + 1)
    ok = (cs[t + T_p] - cs[t]) == T_p  # all T_p targets valid, i.e. inside one episode
    t = t[ok]
    hist = np.maximum(t[:, None] - np.arange(T_o - 1, -1, -1)[None, :], starts[t][:, None])
    tgt = t[:, None] + np.arange(T_p)[None, :]
    return hist, tgt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help=".npz from hfcil_to_chunk_dataset.py")
    p.add_argument("--config", required=True, help="training config the policy is fine-tuned with (state mode)")
    p.add_argument("--reference_vecnorm", required=True, help="vecnorm_*.pth of a real run with the 966-D layout")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vw_coef", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    T_o, T_p = cfg["history_len"], cfg["prediction_horizon"]
    data = np.load(args.dataset)
    obs, act, valid, state, episode = data["obs"], data["action"], data["valid"], data["state"], data["episode"]
    A = act.shape[1]
    K = A - 2
    n_states = len(cfg["env_cfg_overrides"]["flipper_states_deg"])
    if K != n_states or obs.shape[1] != 960 + A:
        raise ValueError(f"dataset is {obs.shape[1]}-D obs / {A}-D action, config has {n_states} states")

    # -- policy, built exactly as train_diffusion does
    observation = MarvRLFlatObservation(env=_StepEnvStub(A, device), encoder_opts=cfg["ftr_obs_encoder_opts"])
    wrapper, _, _ = cfg["policy_config"](**cfg["policy_opts"]).create(_ChunkEnvStub(observation, A, T_p, device), device=device)
    actor_net = next(m for m in wrapper.modules() if isinstance(m, (ChunkMLPActorNet, ChunkGaussianActorNet)))
    actor_params = list(wrapper.get_policy_operator().parameters())

    # -- normalisation (see module docstring)
    ref = torch.load(args.reference_vecnorm, map_location="cpu", weights_only=False)["_extra_state"]
    r_sum, r_ssq, r_cnt = (ref[f"{OBS_KEY}_{s}"].double() for s in ("sum", "ssq", "count"))
    n_shared = 962
    if r_sum.numel() < n_shared:
        raise ValueError(f"reference vecnorm has {r_sum.numel()} dims, need >= {n_shared}")
    mean = np.zeros(obs.shape[1])
    var = np.zeros(obs.shape[1])
    mean[:n_shared] = (r_sum[:n_shared] / r_cnt).numpy()
    var[:n_shared] = (r_ssq[:n_shared] / r_cnt).numpy() - mean[:n_shared] ** 2
    mean[n_shared:] = obs[:, n_shared:].mean(0)
    var[n_shared:] = obs[:, n_shared:].var(0)
    eps = cfg["vecnorm_opts"].get("eps", 1e-4)
    std = np.sqrt(np.maximum(var, eps))
    obs_n = torch.from_numpy(((obs - mean) / np.maximum(std, eps)).astype(np.float32))

    # -- windows, split by episode
    hist, tgt = windows(valid, episode, T_o, T_p)
    rng = np.random.default_rng(args.seed)
    eps_ids = np.unique(episode)
    val_eps = set(rng.choice(eps_ids, size=max(1, int(len(eps_ids) * args.val_frac)), replace=False).tolist())
    is_val = np.array([episode[i] in val_eps for i in hist[:, -1]])
    tgt_state = torch.from_numpy(act[:, 2:].argmax(1))
    vw = torch.from_numpy(act[:, :2])
    # first chunk step is a transition when the target differs from the state the window ends in
    first_trans = (act[tgt[:, 0], 2:].argmax(1) != state[hist[:, -1]])
    logger.info(f"{len(hist)} windows ({int(is_val.sum())} val), T_o={T_o} T_p={T_p}, {int(first_trans.sum())} transition windows")

    def batch(idx):
        h = torch.from_numpy(hist[idx])
        g = torch.from_numpy(tgt[idx])
        x = obs_n[h].reshape(len(idx), -1).to(device)
        return x, tgt_state[g].to(device), vw[g].to(device)

    def forward(x):
        loc, _ = actor_net(x)
        return loc.reshape(x.shape[0], T_p, A)

    def loss_fn(loc, s, v):
        ce = nn.functional.cross_entropy(loc[..., 2:].reshape(-1, K), s.reshape(-1))
        mse = (torch.tanh(loc[..., :2]) - v).pow(2).mean()
        return ce + args.vw_coef * mse, ce, mse

    @torch.no_grad()
    def evaluate(idx):
        correct = np.zeros(len(idx), dtype=bool)
        tot = 0.0
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            x, s, v = batch(b)
            loc = forward(x)
            tot += loss_fn(loc, s, v)[0].item() * len(b)
            correct[i:i + len(b)] = (loc[:, 0, 2:].argmax(-1) == s[:, 0]).cpu().numpy()
        tr = first_trans[idx]
        acc_t = correct[tr].mean() if tr.any() else float("nan")
        acc_s = correct[~tr].mean() if (~tr).any() else float("nan")
        return tot / len(idx), acc_t, acc_s

    train_idx, val_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    optim = torch.optim.AdamW(actor_params, lr=args.lr, weight_decay=1e-4)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, bad = -1.0, 0
    for epoch in range(args.epochs):
        actor_net.train()
        perm = rng.permutation(train_idx)
        run = [0.0, 0.0, 0.0]
        for i in range(0, len(perm), args.batch_size):
            x, s, v = batch(perm[i:i + args.batch_size])
            loss, ce, mse = loss_fn(forward(x), s, v)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, 1.0)
            optim.step()
            run = [run[0] + loss.item(), run[1] + ce.item(), run[2] + mse.item()]
        n = math.ceil(len(perm) / args.batch_size)
        actor_net.eval()
        vloss, acc_t, acc_s = evaluate(val_idx)
        score = math.sqrt(max(acc_t, 0.0) * max(acc_s, 0.0))
        logger.info(f"epoch {epoch}: train loss {run[0] / n:.4f} (ce {run[1] / n:.4f}, v/w mse {run[2] / n:.4f}) | "
                    f"val loss {vloss:.4f} acc transitions {acc_t:.3f} stays {acc_s:.3f} score {score:.3f}")
        if score > best:
            best, bad = score, 0
            torch.save(wrapper.state_dict(), out / "policy_bc.pth")
        else:
            bad += 1
            if bad >= args.patience:
                break

    cnt = r_cnt.float()
    m, v2 = torch.from_numpy(mean).float(), torch.from_numpy(var).float()
    vecnorm = {"_extra_state": {
        f"{OBS_KEY}_sum": m * cnt,
        f"{OBS_KEY}_ssq": (v2 + m.pow(2)) * cnt,
        f"{OBS_KEY}_count": cnt.clone(),
        **{k: v.clone() for k, v in ref.items() if k.startswith("reward_")},
    }}
    torch.save(vecnorm, out / "vecnorm_bc.pth")
    # round-trip: the saved policy must load strictly into a freshly built actor-critic
    fresh, _, _ = cfg["policy_config"](**cfg["policy_opts"]).create(_ChunkEnvStub(observation, A, T_p, device), device=device)
    fresh.load_state_dict(torch.load(out / "policy_bc.pth", map_location=device), strict=True)
    logger.info(f"best val score {best:.3f}; wrote {out / 'policy_bc.pth'} and {out / 'vecnorm_bc.pth'}")


if __name__ == "__main__":
    main()
