"""Checks for the Phase 2 diffusion actor and the DPPO loss.

No Isaac, no GPU. Run:
    PYTHONPATH=src/flipper_training:src/FTR-Benchmark python -m marv_rl_training.training.test_dppo

The load-bearing property is that `chain_log_prob` re-scores the STORED chain
deterministically: if it did not reproduce the log-prob recorded at sampling time (with the
same parameters), the PPO ratio would be ~1 by construction at the first epoch and garbage
afterwards, and nothing downstream would reveal it.
"""

import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marv_rl_training.policies.diffusion_dppo import DiffusionChunkActor, DPPOClipLoss  # noqa: E402
from marv_rl_training.policies.diffusion_policy import ChunkCriticNet, ObsHistoryEncoder  # noqa: E402
from marv_rl_training.policies.diffusion_schedule import DiffusionSchedule  # noqa: E402
from rl_modules.marv_rl.marv_rl_cnn_flat_encoder import MarvRLCNNFlatEncoder  # noqa: E402

OBS, A, T_O, T_P, K = 966, 6, 2, 4, 8
ENC = dict(num_hidden=3, hidden_dim=256, output_dim=128, layernorm=True)
_fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        _fails.append(name)


torch.manual_seed(0)
enc = lambda: ObsHistoryEncoder(MarvRLCNNFlatEncoder(input_dim=OBS, **ENC), OBS, T_O)
sched = DiffusionSchedule(num_train_timesteps=100, num_inference_steps=K)
actor = DiffusionChunkActor(enc(), sched, A, T_P, [64, 128]).eval()

N = 5
obs = torch.randn(N, T_O * OBS)

# --- 1. shapes -------------------------------------------------------------------
with torch.no_grad():
    action, chain, logp = actor.sample_chain(obs)
check("action is the flattened T_p x A chunk", tuple(action.shape) == (N, T_P * A), str(tuple(action.shape)))
check("chain is [N, K+1, A, T_p]", tuple(chain.shape) == (N, K + 1, A, T_P), str(tuple(chain.shape)))
check("one log-prob per sample", tuple(logp.shape) == (N,) and bool(torch.isfinite(logp).all()))
check("action equals the final chain element, same layout",
      torch.allclose(action, chain[:, -1].transpose(1, 2).reshape(N, -1)))

# --- 2. re-scoring the stored chain reproduces the sampling log-prob ---------------
td = TensorDict({"obs_history": obs, "denoise_chain": chain}, batch_size=[N])
with torch.no_grad():
    relogp = actor.chain_log_prob(td)
check("chain_log_prob reproduces the sampled log-prob under unchanged params",
      torch.allclose(relogp, logp, atol=1e-4), f"max diff {float((relogp - logp).abs().max()):.2e}")
with torch.no_grad():
    again = actor.chain_log_prob(td)
check("chain_log_prob is deterministic (no resampling)", torch.allclose(again, relogp, atol=1e-6))

# --- 3. it actually responds to parameter changes ---------------------------------
before = relogp.clone()
with torch.no_grad():
    for p in actor.unet.final_conv[-1].parameters():
        p.add_(torch.randn_like(p) * 0.05)
    after = actor.chain_log_prob(td)
check("perturbing eps_theta changes the chain log-prob (the ratio is live)",
      not torch.allclose(after, before, atol=1e-3), f"mean |d| {float((after - before).abs().mean()):.3f}")

# --- 4. DPPO loss end to end ------------------------------------------------------
critic = TensorDictModule(ChunkCriticNet(enc(), dict(num_hidden=2, hidden_dim=64, layernorm=True)),
                          in_keys=["obs_history"], out_keys=["state_value"])
loss = DPPOClipLoss(actor, critic, clip_epsilon=0.2, entropy_bonus=False,
                    critic_coef=1.0, loss_critic_type="smooth_l1", normalize_advantage=True)
# Score the chain under the CURRENT params so prev_log_prob matches, i.e. the realistic
# first-epoch condition where the ratio is 1. Using the pre-perturbation logp here made
# log_weight ~ 80 nats, which saturates the clip — and clamp has zero gradient outside its
# bounds, so the objective went flat and eps_theta received no gradient at all.
with torch.no_grad():
    logp_cur = actor.chain_log_prob(td)
td_loss = TensorDict({
    "obs_history": obs,
    "denoise_chain": chain,
    "action": action,
    "sample_log_prob": logp_cur,
    "advantage": torch.randn(N, 1),
    "value_target": torch.randn(N, 1),
}, batch_size=[N])
try:
    out = loss(td_loss)
    keys = set(out.keys())
    check("DPPOClipLoss returns an objective and a critic loss",
          {"loss_objective", "loss_critic"} <= keys, str(sorted(keys)))
    check("clip_fraction is reported", "clip_fraction" in keys)
    check("no entropy terms (entropy_bonus off, dist is None)", "loss_entropy" not in keys)
    tot = out["loss_objective"].mean() + out["loss_critic"].mean()
    tot.backward()
    # torchrl wraps the actor functionally, so gradients land on loss.actor_network_params,
    # NOT on actor.parameters() — checking the latter reports ~0 and looks like a pass
    # against a naive `> 0` threshold. Check the params the loss actually differentiates,
    # and require a magnitude rather than mere non-zero.
    grads = [p.grad for p in loss.parameters() if p.grad is not None]
    total = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)) if grads else torch.tensor(0.0)
    check("gradients reach the loss's actor params through the whole chain",
          len(grads) > 0 and bool(torch.isfinite(total)) and float(total) > 1e-8,
          f"{len(grads)} grad tensors, norm {float(total):.6g}")
    # And specifically that eps_theta itself is differentiated, not only the critic.
    unet_grads = [p.grad for n, p in loss.named_parameters() if "unet" in n and p.grad is not None]
    un = torch.sqrt(sum((g.double() ** 2).sum() for g in unet_grads)) if unet_grads else torch.tensor(0.0)
    check("eps_theta (unet) specifically receives gradient",
          len(unet_grads) > 0 and float(un) > 1e-8, f"{len(unet_grads)} tensors, norm {float(un):.6g}")
except Exception as e:  # noqa: BLE001
    check("DPPOClipLoss runs", False, f"{type(e).__name__}: {e}")

# --- 5. how sensitive is the chain log-prob to a REALISTIC parameter step? ------------
# This is the risk that decides chain-level (2A) vs per-denoising-step (2B) clipping. The
# chain log-prob sums K x T_p x A Gaussian terms, so its sensitivity compounds; if a single
# optimiser-sized step moves it by more than ~log(1.2)=0.18 nats, every sample clips
# immediately and the chain-level objective has no gradient.
base = actor.chain_log_prob(td).detach()
for scale in (1e-4, 1e-3, 1e-2):
    bak = [p.detach().clone() for p in actor.unet.parameters()]
    with torch.no_grad():
        for p in actor.unet.parameters():
            p.add_(torch.randn_like(p) * scale)
        moved = actor.chain_log_prob(td).detach()
        for p, b in zip(actor.unet.parameters(), bak):
            p.copy_(b)
    d = float((moved - base).abs().mean())
    print(f"  INFO  perturb eps_theta by {scale:g}: mean |d log-prob| = {d:.3f} nats "
          f"({'clips immediately' if d > 0.182 else 'within the trust region'})")

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}")
    sys.exit(1)
print("all checks passed")
