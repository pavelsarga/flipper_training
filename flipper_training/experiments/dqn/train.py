"""DQN trainer for the AT-D3QN / ICM-D3QN baselines (Pan et al.).

Adds the value-based (DQN) loop MARV_RL lacks. The d3qn policy is a dueling
double-DQN over a discrete flipper-action set (absolute per-flipper bins by
default, or the paper's 9-action paired-delta set with ``policy_opts:
incremental: true`` -- see ``d3qn_policy.py``); this trainer optimises its
QValueActor with ``torchrl.objectives.DQNLoss(double_dqn=True)`` and,
when ``use_icm: true`` (ICM-D3QN, ``icm.py``), the curiosity module's
intrinsic reward AND aux loss -- see ``icm.py``'s module docstring for the
three things a correct ICM integration requires; all three now hold here:
the encoder is shared with the Q-network (``wrapper.get_encoder()``, not a
fresh copy), the intrinsic reward is computed under ``no_grad`` each step and
folded into the stored transition reward BEFORE it enters the replay buffer
(``R_t = R^e_t + R^i_t``, Eq. 13 -- this call did not previously exist
anywhere in this loop, so ICM's curiosity term never reached the RL
objective), and ``icm.loss()`` is added to the optimized loss. The env is
continuous-action, so each step the one-hot Q action is mapped to the
continuous env action via ``wrapper.action_from_onehot(onehot, td)`` (works
for both action-space modes -- absolute is a static table lookup, incremental
reads the current flipper angles back out of ``td``) while the one-hot is
stored for the TD loss (and is also exactly the "action taken" ICM's
inverse/forward models need).

Run:  python -m flipper_training.experiments.dqn.train --local <dqn_config.yaml>
Config: policy_config: ${cls:...d3qn_policy.D3QNPolicyConfig}; policy_opts
{mlp_opts, optimizer_opts, n_bins} or, for the paper-faithful action set,
{mlp_opts, optimizer_opts, incremental: true, delta} -- add fig5_topology: true
for the literal Fig. 5 network (needs a PanTerrainState observation); optional
gamma, eps_start/eps_end, use_icm, icm_opts ({feat_dim, mlp_opts, beta_forward,
beta_inverse, eta} -- see icm.ICM -- or, for the paper-faithful Fig. 7 encoder,
{separate_encoder: true, raw_state_key, raw_state_dim} with mlp_opts/feat_dim
left unset so Fig. 7's fixed widths apply), steps_per_iter, n_iters, batch_size,
replay_buffer_size (optional, defaults to the previous steps_per_iter*num_robots*50).
See configs/baselines/at_d3qn_full.yaml / icm_d3qn_full.yaml for complete,
paper-faithful example configs (PanTerrainState + pan_reward.PanReward +
fig5_topology + incremental actions + Table-2 hyperparameters).

Partial-reset note (checked, not implemented): ``Env._reset`` reads the
standard TorchRL ``"_reset"`` mask and correctly LEAVES the live physics state
of robots NOT in the mask untouched, but -- verified empirically against a
real ``Env`` -- it does NOT reset the physical ``PhysicsState`` (x/q/thetas/
...) of robots that ARE in the mask back to the freshly generated
``env.start``; it only refreshes bookkeeping (start/goal targets,
step_limits, step_count). Feeding a partial ``"_reset"`` mask here would
therefore silently leave "done" robots' pose stale (still wherever the
episode ended) while their step counter/goal look freshly restarted --
corrupting the replay buffer rather than truncating live episodes, which is
the worse failure mode of the two. Until ``Env._reset`` masks the physical
state too, this loop pays for correctness with a full-batch reset whenever
ANY robot is done (see the reset call below).
"""
from __future__ import annotations

import dataclasses

import torch
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, RandomSampler

from flipper_training.experiments.ppo.common import (
    prepare_env, make_transformed_env, parse_and_load_config,
)
from flipper_training.experiments.ppo.config import PPOExperimentConfig
from flipper_training.utils.logutils import RunLogger, get_terminal_logger

log = get_terminal_logger("dqn_train")


def _cfg(c, k, d):
    return c[k] if k in c else d


_PPO_CFG_FIELDS = {f.name for f in dataclasses.fields(PPOExperimentConfig)}


def main():
    config = parse_and_load_config()
    # This trainer's own hyperparameters (gamma, n_iters, steps_per_iter, ...; see module
    # docstring) live alongside the shared PPOExperimentConfig fields in the SAME yaml, so
    # filter down to the fields PPOExperimentConfig actually declares before constructing it
    # -- otherwise any config that used the documented dqn-only keys would fail to load at all.
    cfg = PPOExperimentConfig(**{k: v for k, v in config.items() if k in _PPO_CFG_FIELDS})
    env, device, rng = prepare_env(cfg, mode="train")

    wrapper, _optim_groups, policy_transforms = cfg.policy_config(**cfg.policy_opts).create(env, device=device)
    qnet = wrapper.get_qvalue_network()          # QValueActor: obs -> one-hot "action" + "action_value"
    env, _ = make_transformed_env(env, cfg, policy_transforms)  # threads e.g. d3qn's incremental-mode RenameTransform

    obs_list = getattr(env, "observations", None) or getattr(env, "base_env", env).observations
    obs_keys = [o.name for o in obs_list]

    loss = DQNLoss(qnet, action_space="one_hot", delay_value=True, double_dqn=True).to(device)
    loss.make_value_estimator(gamma=float(_cfg(config, "gamma", 0.99)))
    target_updater = SoftUpdate(loss, eps=float(_cfg(config, "target_eps", 0.995)))
    opt = torch.optim.AdamW(qnet.parameters(),
                            **(cfg.policy_opts.get("optimizer_opts") or {"lr": 1e-3}))

    icm = None
    if bool(_cfg(config, "use_icm", False)):          # ICM-D3QN variant (intrinsic curiosity)
        from flipper_training.policies.icm import ICM
        icm_opts = dict(_cfg(config, "icm_opts", {}))
        separate_encoder = bool(icm_opts.pop("separate_encoder", False))
        if separate_encoder:
            # Paper-literal wiring (Fig. 7): ICM builds its OWN raw-state encoder over the
            # PanTerrainState observation instead of sharing the Q-network's encoder. This is
            # the ONLY option compatible with policy_opts.fig5_topology=true, which has no
            # EncoderCombiner to share (see d3qn_policy.py's get_encoder()). raw_state_dim is
            # auto-resolved from the env's observations unless icm_opts already supplies one.
            raw_state_key = icm_opts.get("raw_state_key", "PanTerrainState")
            if "raw_state_dim" not in icm_opts:
                obs_by_name = {o.name: o for o in obs_list}
                pan_obs = obs_by_name.get(raw_state_key)
                if pan_obs is None:
                    raise ValueError(
                        f"use_icm + icm_opts.separate_encoder=true needs a '{raw_state_key}' observation in "
                        f"the env (found {sorted(obs_by_name)}), or an explicit icm_opts.raw_state_dim."
                    )
                icm_opts["raw_state_dim"] = pan_obs.dim
            icm = ICM(encoder=None, action_dim=wrapper.n_actions, separate_encoder=True, **icm_opts).to(device)
            # No sharing with the Q-network here (separate_encoder builds its own from scratch) --
            # every ICM param is genuinely its own, no dedup against qnet.parameters() needed.
            opt.add_param_group({"params": list(icm.parameters())})
        else:
            # Share the Q-network's OWN encoder (not a fresh EncoderCombiner) -- a deliberate
            # simplification (the paper's Fig. 7 draws two separate encoders); sharing lets
            # curiosity gradients shape the policy representation. See
            # icm.py's module docstring point 3 / d3qn_policy.py's get_encoder() docstring.
            icm = ICM(wrapper.get_encoder(), action_dim=wrapper.n_actions, **icm_opts).to(device)
            # icm.encoder IS qnet's encoder (registered as a submodule of both), so icm.parameters()
            # re-yields the encoder's params, which are already in `opt` via qnet.parameters() above.
            # Add only ICM's OWN new heads (feat/inverse/forward_model) here -- the shared encoder still
            # gets its combined gradient (autograd sums the contribution from both losses through `l`
            # below), just applied by exactly one Adam step instead of two.
            icm_new_params = [p for n, p in icm.named_parameters() if not n.startswith("encoder.")]
            opt.add_param_group({"params": icm_new_params})

    n_iters = int(_cfg(config, "n_iters", 200))
    steps = int(_cfg(config, "steps_per_iter", 32))
    batch_size = int(_cfg(config, "batch_size", cfg.num_robots))
    eps0, eps1 = float(_cfg(config, "eps_start", 1.0)), float(_cfg(config, "eps_end", 0.05))
    # replay_buffer_size is optional (defaults to the previous hardcoded steps*num_robots*50 formula)
    # -- added so a config can state Table 2's literal replay buffer size (8e6) explicitly rather than
    # only being reachable by accident via steps_per_iter/num_robots.
    buffer_size = int(_cfg(config, "replay_buffer_size", steps * cfg.num_robots * 50))
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size, device=device),
        sampler=RandomSampler(), batch_size=batch_size)
    logger = RunLogger(train_config=config, use_wandb=cfg.use_wandb, use_tensorboard=cfg.use_tensorboard, category="dqn")

    td = env.reset()
    for it in range(n_iters):
        eps = max(eps1, eps0 - (eps0 - eps1) * it / n_iters)
        icm_r_sum, icm_r_n = 0.0, 0  # diagnostics: mean intrinsic reward this iter (icm is None otherwise)
        for _ in range(steps):
            with torch.no_grad():
                out = qnet(td.clone())
                onehot = out["action"]
                # epsilon-greedy over the M discrete configs
                rand = torch.rand(onehot.shape[0], device=device) < eps
                if rand.any():
                    ridx = torch.randint(0, onehot.shape[-1], (int(rand.sum()),), device=device)
                    onehot = onehot.clone()
                    onehot[rand] = torch.zeros_like(onehot[rand]).scatter_(-1, ridx[:, None], 1.0)
                cont_action = wrapper.action_from_onehot(onehot, td)  # continuous action for the physics step
            step_td = td.clone()
            step_td["action"] = cont_action
            next_td = env.step(step_td)
            if icm is not None:
                # R_t = R^e_t + R^i_t (Eq. 13): fold the curiosity bonus into the stored reward
                # BEFORE it enters the replay buffer, so DQNLoss bootstraps on the combined reward.
                # Must happen here, at collection time -- the buffer stores rewards, not raw
                # obs/next_obs pairs recomputed later, and `onehot` here is the actual
                # (possibly epsilon-greedy-exploratory) action taken, which is what the curiosity
                # module should be evaluated against.
                with torch.no_grad():
                    obs_t = {k: td[k] for k in obs_keys}
                    obs_next = {k: next_td["next"][k] for k in obs_keys}
                    r_i = icm.intrinsic_reward(obs_t, onehot, obs_next)  # [B, 1], eta already applied
                    reward = next_td["next", "reward"]
                    next_td["next", "reward"] = reward + r_i.to(reward.dtype).reshape(reward.shape)
                    icm_r_sum += float(r_i.sum()); icm_r_n += r_i.numel()
            trans = td.clone()
            trans["action"] = onehot                      # store one-hot for the TD loss
            trans["next"] = next_td["next"]
            buffer.extend(trans)
            td = next_td["next"]
            if td.get("done", torch.zeros(1, dtype=torch.bool)).any():
                # Full-batch reset by design -- see module docstring "Partial-reset
                # note": Env._reset's "_reset" mask does not restore the physical
                # state of just-reset robots, so a masked reset here would silently
                # corrupt their transitions rather than merely truncate others' episodes.
                td = env.reset()

        for _ in range(int(_cfg(config, "grad_steps", 8))):
            batch = buffer.sample().to(device)
            loss_td = loss(batch)
            l = loss_td["loss"]
            if icm is not None:
                obs = {k: batch[k] for k in obs_keys}
                next_obs = {k: batch["next"][k] for k in obs_keys}
                l = l + icm.loss(obs, batch["action"], next_obs)
            opt.zero_grad(); l.backward(); opt.step()
            target_updater.step()
        loss_value = float(l.detach())
        log_payload = {"dqn/loss": loss_value, "dqn/eps": eps}
        icm_msg = ""
        if icm is not None and icm_r_n:
            icm_r_mean = icm_r_sum / icm_r_n
            log_payload["dqn/icm_intrinsic_reward_mean"] = icm_r_mean
            icm_msg = f" icm_r={icm_r_mean:.4f}"
        log.info(f"[dqn] iter {it+1}/{n_iters} eps={eps:.3f} loss={loss_value:.4f} buf={len(buffer)}{icm_msg}")
        logger.log_data(log_payload, step=it)

    logger.save_weights(wrapper.state_dict(), name="policy_final")
    log.info("DQN done; saved policy_final.pth (deploy via run_flipper_policy_sim.sh)")


if __name__ == "__main__":
    main()
