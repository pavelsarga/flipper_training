# Trainable baseline policies

These are the learned flipper-control **baselines** from `baselines_todo.md`,
implemented as `PolicyConfig` subclasses using the SAME stack as the repo's PPO
policies — TorchRL operators, `tensordict`, and the shared `MLP`/`EncoderCombiner`
building blocks. Each obeys the standard contract:

```python
policy_config = SomePolicyConfig(**policy_opts)
wrapper, optim_groups, transforms = policy_config.create(env, device=...)
```

so they plug into the existing experiment config exactly like `MLPPolicyConfig`
(set `policy_config:` + `policy_opts:`). You only choose the matching TorchRL
**loss** in your trainer — the architectures are ready.

| Baseline (paper) | Module · class | TorchRL loss | Notes |
|---|---|---|---|
| **FTR SAC / TD3 / DDPG** (FTR-Bench) | `offpolicy_ac_policy.OffPolicyACConfig` | `SACLoss` / `TD3Loss` / `DDPGLoss` | `deterministic_actor=False`→SAC; `True`+`twin_q`→TD3; `twin_q=False`→DDPG. Actor + (twin) Q(obs,act). |
| **AT-D3QN** (Pan 2023) | `d3qn_policy.D3QNPolicyConfig` | `DQNLoss(double_dqn=True)` | Dueling Q over `n_bins**4` discrete flipper configs; `wrapper.discrete_to_continuous` is the `[M, action_dim]` env-action table. |
| **ICM-D3QN** (Pan 2023) | `d3qn_policy` + `icm.ICM` | `DQNLoss` + ICM aux loss | Add `ICM.intrinsic_reward()` to the reward and `ICM.loss()` to the optimizer. |
| **Azayev soft state-machine** (RA-L 2022) | `state_machine_policy.StateMachinePolicyConfig` | `ClipPPOLoss` | K soft-gated flipper-template states; the key architectural rival to GC/OA switching. |
| FTR PPO/TRPO (on-policy) | *existing* `mlp_policy` / `gru_policy` / `lstm_policy` | `ClipPPOLoss` / TRPO | already in the repo. |

Still to add (same pattern): `bc_policy` (Behavior Cloning / DAgger — actor as a
regression head, supervised loss), `diffusion_policy` (conditional DDPM action
head), `ctrac_policy` (C-TRAC: asymmetric SAC + contact-VAE), Pecka C-REPS
discrete 5-config selector. GC-only / OA-only are *ablations* of your own
contribution, not external baselines.

## Example `policy_opts` (SAC)

```yaml
policy_config: flipper_training.policies.offpolicy_ac_policy.OffPolicyACConfig
policy_opts:
  deterministic_actor: false
  twin_q: true
  actor_mlp_opts:   {hidden_dim: 256, num_hidden: 2, layernorm: true}
  qvalue_mlp_opts:  {hidden_dim: 256, num_hidden: 2, layernorm: true}
  actor_optimizer_opts:  {lr: 3.0e-4}
  qvalue_optimizer_opts: {lr: 3.0e-4}
```

## Validate (no GPU needed)

```bash
flipper_venv/bin/python src/flipper_training/test_baseline_policies.py
# constructs each policy against a mock Env, runs a forward pass, and builds the
# matching TorchRL loss — proves training-compatibility.
```
