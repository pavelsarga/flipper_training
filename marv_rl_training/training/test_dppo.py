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

from marv_rl_training.policies.diffusion_dppo import (  # noqa: E402
    DiffusionChunkActor, DPPOClipLoss, DPPOPerStepClipLoss,
    DiffusionPolicyPhase2Config, make_dppo_loss)
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
    action, chain, logp, steps = actor.sample_chain(obs)
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
_bak0 = [q.detach().clone() for q in actor.unet.final_conv[-1].parameters()]
with torch.no_grad():
    for q in actor.unet.final_conv[-1].parameters():
        q.add_(torch.randn_like(q) * 0.05)
    after = actor.chain_log_prob(td)
    # RESTORE. Leaving this perturbation in place silently invalidated every later
    # comparison against `logp`/`steps`, which were captured before it — it made the 2A/2B
    # sensitivity numbers read 80 and 10.5 nats when the true values are ~1 and ~0.03.
    for q, b in zip(actor.unet.final_conv[-1].parameters(), _bak0):
        q.copy_(b)
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
    check("2A loss_objective is a scalar too", out["loss_objective"].dim() == 0,
          f"shape {tuple(out['loss_objective'].shape)}")
    tot = out["loss_objective"] + out["loss_critic"]
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

# --- 6. per-step clipping (2B) — the intended default ---------------------------------
check("per-step log-probs are [N, K] and sum to the chain log-prob",
      tuple(steps.shape) == (N, K) and torch.allclose(steps.sum(1), logp, atol=1e-4),
      f"{tuple(steps.shape)}, max diff {float((steps.sum(1)-logp).abs().max()):.2e}")
# NB `steps` was captured before section 3 perturbed the actor, so it is a stale reference.
# The meaningful invariant at the current parameters is that the per-step log-probs sum to
# the chain log-prob — that is what makes 2A and 2B two views of the same quantity.
with torch.no_grad():
    re_steps = actor.chain_log_prob_steps(td)
    re_chain = actor.chain_log_prob(td)
check("chain_log_prob_steps reproduces the sampled per-step log-probs",
      torch.allclose(re_steps, steps, atol=1e-4), f"max diff {float((re_steps-steps).abs().max()):.2e}")
check("per-step log-probs sum to the chain log-prob at the current parameters",
      torch.allclose(re_steps.sum(1), re_chain, atol=1e-4),
      f"max diff {float((re_steps.sum(1)-re_chain).abs().max()):.2e}")

loss2 = DPPOPerStepClipLoss(actor, critic, clip_epsilon=0.2, entropy_bonus=False,
                            critic_coef=1.0, loss_critic_type="smooth_l1", normalize_advantage=True)
td2 = td_loss.clone()
td2.set("denoise_logp_steps", re_steps)
for p_ in loss2.parameters():
    p_.grad = None
try:
    o2 = loss2(td2)
    check("DPPOPerStepClipLoss returns objective + critic", {"loss_objective","loss_critic"} <= set(o2.keys()),
          str(sorted(o2.keys())))
    check("per-step clip_fraction is reported for every denoising step",
          tuple(o2["clip_fraction_per_step"].shape) == (K,), str(tuple(o2["clip_fraction_per_step"].shape)))
    # Backward EXACTLY as the trainer does — no .mean() here. An earlier version of this
    # test reduced first, which made a [N]-shaped loss_objective look fine and let a
    # "grad can be implicitly created only for scalar outputs" failure reach the cluster.
    check("loss_objective is a scalar, as backward() requires",
          o2["loss_objective"].dim() == 0, f"shape {tuple(o2['loss_objective'].shape)}")
    check("loss_critic is a scalar", o2["loss_critic"].dim() == 0, f"shape {tuple(o2['loss_critic'].shape)}")
    (o2["loss_objective"] + o2["loss_critic"]).backward()
    ug = [p_.grad for n_, p_ in loss2.named_parameters() if "unet" in n_ and p_.grad is not None]
    un2 = torch.sqrt(sum((g.double()**2).sum() for g in ug)) if ug else torch.tensor(0.0)
    check("eps_theta receives gradient under per-step clipping", len(ug) > 0 and float(un2) > 1e-8,
          f"{len(ug)} tensors, norm {float(un2):.6g}")
except Exception as e:  # noqa: BLE001
    check("DPPOPerStepClipLoss runs", False, f"{type(e).__name__}: {e}")

# 2A vs 2B under a realistic parameter step: how much of the batch clips?
bak = [q.detach().clone() for q in actor.unet.parameters()]
with torch.no_grad():
    for q in actor.unet.parameters():
        q.add_(torch.randn_like(q) * 3e-4)
    chain_lw = (actor.chain_log_prob(td) - logp).abs()
    step_lw = (actor.chain_log_prob_steps(td) - steps).abs()
    for q, b in zip(actor.unet.parameters(), bak):
        q.copy_(b)
thr = torch.log(torch.tensor(1.2))
print(f"  INFO  after a 3e-4 step — 2A chain ratio |log w| = {float(chain_lw.mean()):.3f} nats "
      f"({float((chain_lw > thr).float().mean())*100:.0f}% of samples clip)")
print(f"  INFO  after a 3e-4 step — 2B per-step |log w| = {float(step_lw.mean()):.3f} nats "
      f"({float((step_lw > thr).float().mean())*100:.0f}% of steps clip)")

# --- 7. sigma floor directly controls ratio sensitivity -------------------------------
# log N(x; mu, sigma) sensitivity to a shift in mu scales as 1/sigma^2, so min_sampling_std
# is not only an exploration knob — it decides whether PPO ratios are usable at all.
from marv_rl_training.policies.diffusion_schedule import DiffusionSchedule as _DS
for floor in (0.02, 0.05, 0.1, 0.2):
    a2 = DiffusionChunkActor(enc(), _DS(100, K, min_sampling_std=floor), A, T_P, [64, 128]).eval()
    a2.load_state_dict(actor.state_dict())
    with torch.no_grad():
        _, ch2, _, st2 = a2.sample_chain(obs)
        td2s = TensorDict({"obs_history": obs, "denoise_chain": ch2}, batch_size=[N])
        bak2 = [q.detach().clone() for q in a2.unet.parameters()]
        for q in a2.unet.parameters():
            q.add_(torch.randn_like(q) * 3e-4)
        d2 = (a2.chain_log_prob_steps(td2s) - st2).abs()
        for q, b in zip(a2.unet.parameters(), bak2):
            q.copy_(b)
    print(f"  INFO  min_sampling_std={floor:<5g} per-step |log w| after a 3e-4 step = "
          f"{float(d2.mean()):7.3f} nats  ({float((d2 > thr).float().mean())*100:3.0f}% clip)")

# --- 8. the Phase 2 policy config assembles and runs through the collector's path -------
from torchrl.data import Bounded  # noqa: E402


class _StubObs:
    dim = OBS
    def get_encoder(self):
        return MarvRLCNNFlatEncoder(input_dim=OBS, **ENC)


class _StubEnv:
    action_spec = Bounded(low=-1.0, high=1.0, shape=(N, T_P * A))
    observations = [_StubObs()]


cfg = DiffusionPolicyPhase2Config(
    actor_optimizer_opts={"lr": 1e-4}, value_optimizer_opts={"lr": 1e-3},
    value_mlp_opts={"num_hidden": 2, "hidden_dim": 64, "layernorm": True},
    prediction_horizon=T_P, history_len=T_O, down_dims=[64, 128], num_inference_steps=K,
)
try:
    wrapper, groups, transforms = cfg.create(_StubEnv())
    check("Phase 2 config builds an actor-critic wrapper",
          hasattr(wrapper, "get_policy_operator") and hasattr(wrapper, "get_value_operator"))
    check("two optimiser groups, named as the trainer expects",
          [g["name"] for g in groups] == ["policy_operator", "value_operator"], str([g["name"] for g in groups]))
    td_roll = TensorDict({"obs_history": obs}, batch_size=[N])
    out_roll = wrapper.get_policy_operator()(td_roll)
    check("actor writes action, chain, log-prob and per-step log-probs into the tensordict",
          {"action", "denoise_chain", "sample_log_prob", "denoise_logp_steps"} <= set(out_roll.keys()),
          str(sorted(out_roll.keys())))
    check("critic produces one value per macro step",
          tuple(wrapper.get_value_operator()(td_roll)["state_value"].shape) == (N, 1))
except Exception as e:  # noqa: BLE001
    check("Phase 2 config builds", False, f"{type(e).__name__}: {e}")

# entropy_bonus is a fatal misconfiguration for a diffusion policy — it must not be silent.
try:
    make_dppo_loss(cfg, actor, critic, clip_epsilon=0.2, entropy_bonus=True)
    check("make_dppo_loss rejects entropy_bonus=True", False, "it did not raise")
except ValueError:
    check("make_dppo_loss rejects entropy_bonus=True", True)
l_default = make_dppo_loss(cfg, actor, critic, clip_epsilon=0.2, entropy_bonus=False)
cfg2 = DiffusionPolicyPhase2Config(actor_optimizer_opts={}, value_optimizer_opts={},
                                   value_mlp_opts={"num_hidden": 1, "hidden_dim": 32, "layernorm": False},
                                   per_step_clipping=True)
l_perstep = make_dppo_loss(cfg2, actor, critic, clip_epsilon=0.2, entropy_bonus=False)
check("make_dppo_loss selects the loss class from per_step_clipping",
      type(l_default).__name__ == "DPPOClipLoss" and type(l_perstep).__name__ == "DPPOPerStepClipLoss",
      f"{type(l_default).__name__} / {type(l_perstep).__name__}")

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}")
    sys.exit(1)
print("all checks passed")
