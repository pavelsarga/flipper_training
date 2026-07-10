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

## Master table

Config paths are relative to the MARV_RL repo root (one level above this
`flipper_training` tree). "Status" is `full` (a faithful reproduction of the
paper's core algorithm, documented deviations only where this repo genuinely
can't match it — e.g. no demonstrator, no arm), `adaptation` (deliberately
reuses this repo's shared machinery — obs/action/reward/policy shape — as a
documented stand-in for paper-specific mechanisms that don't exist here),
`blocked here` (the architecture exists and is deploy-compliant, but no
trainer can currently run it in this environment), or `pending` (owned by a
concurrently-running agent at the time this table was last verified — see
"Verified file existence" below).

| Baseline (paper) | Policy module · class | Trainer entry point | Config path | Status |
|---|---|---|---|---|
| Azayev & Zimmermann, RA-L 2022 (soft state-machine / HFC) | `state_machine_policy.StateMachinePolicyConfig` | native `flipper_training/experiments/ppo/train.py` (`ClipPPOLoss`) | `configs/baselines/azayev.yaml` | **full** — verified loads + trains + evals end-to-end this session |
| Mitriakov et al., IEEE RA Mag. 2021 (staircase negotiation) | `mlp_policy.MLPPolicyConfig` (swap to `gru_policy.GRUPolicyConfig` for the recurrent variant) | native `flipper_training/experiments/ppo/train.py` (`ClipPPOLoss`) | `configs/baselines/mitriakov.yaml` | **adaptation** — see the file's header comment for the exact per-term mapping to the paper's Eq. 1-6; verified loads + trains + evals end-to-end this session |
| AT-D3QN, Pan et al. 2023 | `d3qn_policy.D3QNPolicyConfig` (`incremental=True` + `fig5_topology=True` for the paper's action set + network) + `observations/pan_terrain.PanTerrainState` + `rl_rewards/pan_reward.PanReward` | native `flipper_training/experiments/dqn/train.py` (Double DQN) | `configs/baselines/at_d3qn_full.yaml` | **full** — paper state (Eq. 1-2)/action (Eq. 3)/reward (Eq. 4/6/7)/network (Fig. 5) + Table 2 hyperparams; verified constructs + trains end-to-end this session (see "Status notes") |
| ICM-D3QN, Pan et al. 2023 | AT-D3QN's config/network + `icm.ICM` (`use_icm: true`, `icm_opts.separate_encoder: true` for the paper's own Fig. 7 raw-state encoder) | native `flipper_training/experiments/dqn/train.py` | `configs/baselines/icm_d3qn_full.yaml` | **full** — AT-D3QN full repro + Fig. 7's dedicated 3-layer psi encoder (not shared with the Q-network); verified constructs + trains end-to-end this session (see "Status notes") |
| Pecka et al., IROS 2016 (safety-constrained flipper control) | `pecka_policy.PeckaLinearPolicyConfig` + `observations/robot_state_with_terrain_lookahead.LocalStateVectorWithTerrainHeightAhead` | native `flipper_training/experiments/creps/train.py` (context-free Constrained REPS, gradient-free — no TorchRL loss) | `configs/baselines/pecka_full.yaml` | **full** — full paper phi(s)=[pitch, height_ahead, 1] (6 params) + full Sec III safety (tip-over OR hard-impact; "delicate parts" N/A, documented); verified end-to-end this session (see "Status notes" below) |
| C-TRAC, Pan et al. 2025 IROS | `ctrac_policy.CTRACConfig` | native `flipper_training/experiments/ctrac/train.py` (two-stage 1:5 SAC : C-VAE) | `configs/baselines/ctrac_full.yaml` (checklist: `configs/baselines/ctrac_NOTE.md`) | **pending** — owned by a concurrent agent, see "Verified file existence" |
| FTR SAC / TD3 / DDPG | `offpolicy_ac_policy.OffPolicyACConfig` | **none native.** FTR-Bench `scripts/ftr_algo/train.py --algo {sac,td3,ddpg}` (requires Isaac Lab) | N/A — see `configs/baselines/OFFPOLICY_NOTE.md` | **blocked here** — policy architecture is ready/deploy-compliant; no native trainer exists for it, and Isaac Lab (`omni.isaac.lab`) is not installed in this environment (confirmed by import test) |
| FTR PPO/TRPO (on-policy, pre-existing) | existing `mlp_policy` / `gru_policy` / `lstm_policy` | native `experiments/ppo/train.py` (`ClipPPOLoss`) / FTR-Bench TRPO | pre-existing `test_configs/*.yaml`, `final_training_configs/*.yaml`, `sota_configs/*.yaml` | **full** — predates this baselines effort, not touched by it |

GC-only / OA-only are *ablations* of your own contribution, not external baselines.

NOTE — the imitation-learning family (Behavior Cloning / DAgger / Diffusion Policy)
was DROPPED: those are IL methods that require an expert demonstrator, and this
project has no natural demonstrator (see progress.md §22.5). Only reward-driven
baselines are kept.

### Verified file existence

Any remaining `pending` rows above are other agents' concurrently-in-progress
work. Checked at the end of THIS session (AT-D3QN/ICM-D3QN):

```
$ ls configs/baselines/
at_d3qn_full.yaml  azayev.yaml  ctrac_NOTE.md  icm_d3qn_full.yaml  mitriakov.yaml  OFFPOLICY_NOTE.md  pecka_full.yaml
```

`at_d3qn_full.yaml`/`icm_d3qn_full.yaml` are this session's own deliverable
(AT-D3QN/ICM-D3QN full reproduction: `observations/pan_terrain.py`,
`rl_rewards/pan_reward.py`, `d3qn_policy.py`'s `fig5_topology`, `icm.py`'s
`separate_encoder`) — see "Status notes" below for the verified end-to-end
summary. `ctrac_full.yaml` was **not present** as of the last check (a
different concurrent agent's scope). Re-run the `ls` above (or grep this
table's config paths) to check current status; update any remaining `pending`
rows to `full`/`adaptation` once those files land, rather than re-deriving the
whole table.

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

This architecture is ready (see the master table) but has no native trainer to
plug it into yet — see `configs/baselines/OFFPOLICY_NOTE.md` before trying to
train it.

## Validate (no GPU needed)

```bash
flipper_venv/bin/python src/flipper_training/test_baseline_policies.py
# constructs each policy against a mock Env, runs a forward pass, and builds the
# matching TorchRL loss — proves training-compatibility.
```

For an end-to-end check against a *real* `Env` (not a mock) through the actual
trainer entry point, run the target config with tiny overrides, e.g.:

```bash
cd src/flipper_training/flipper_training/experiments/ppo   # plain-script form, see gotcha below
../../../../../flipper_venv/bin/python train.py --local ../../../../../configs/baselines/azayev.yaml \
  use_wandb=false num_robots=4 time_steps_per_batch=8 frames_per_sub_batch=16 \
  epochs_per_batch=1 eval_and_save_every=1 max_eval_steps=5 eval_repeats_after_training=1 \
  total_frames=64 device=cpu objective_opts.cache_size=500
```

(`objective_opts.cache_size=500` only speeds up the smoke test — the checked-in
configs use the proven `cache_size: 40000`, which takes noticeably longer to
build at real `num_robots` scale but is unrelated to correctness.)

**Invocation gotcha (fixed 2026-07, but worth knowing):** `experiments/ppo/train.py`
used to import its own config dataclass with a bare `from config import
PPOExperimentConfig, OmegaConf`, which only resolves when the file is run as a
plain script from *within* `experiments/ppo/` (script-dir `sys.path[0]`
aliases the sibling `config.py`). It raised `ModuleNotFoundError: No module
named 'config'` under the CLAUDE.md-documented `python -m
flipper_training.experiments.ppo.train --local ...` form, since `-m` does not
add the submodule's own directory to `sys.path`. Now uses fully-qualified
imports, so both invocation styles work.

**Compile gotcha:** `engine_compile_opts` must be explicit `null` to actually
skip `torch.compile`, not merely omitted — `Env._compile_engine()` triggers on
`engine_compile_opts is not None`, and `PPOExperimentConfig`'s own dataclass
default is `{}` (empty dict, which *is* "not None"). Omitting the key still
compiles the engine with the default 100 correctness-check iterations, which
is slow and — verified empirically on this machine's CPU backend — can
hard-crash the run on a real compiled-vs-eager numerical mismatch. Both
`azayev.yaml` and `mitriakov.yaml` set `engine_compile_opts: null` explicitly
for this reason; copy real compile opts from `final_training_configs/*.yaml`
only after confirming they pass the correctness check on your target device.

## Making a baseline trainable in MARV_RL AND runnable in sim

Both paths are generic over `PolicyConfig`:
* **Train** — `experiments/ppo/train.py` does `policy_config(**policy_opts).create(env)`.
* **Deploy** — `flipper_policy_node` (via the inference module) does the same, then reads
  `get_policy_operator()(td)["action"]` and publishes /cmd_vel + /flippers_cmd_vel/*.

So a baseline is trainable+runnable iff:
1. its `PolicyConfig.get_policy_operator()` outputs a **continuous `"action"`** matching
   `env.action_spec` (discrete policies must map argmax→continuous — see d3qn_policy
   `_DiscreteToContinuous`), and
2. it has a MARV_RL training **config** whose algorithm MARV_RL supports.

Algorithm support: **native** experiments = PPO / grad / MPPI; **FTR-Bench** =
PPO/SAC/TD3/DDPG/TRPO/MAPPO/HAPPO/HATRPO. → PPO-family + off-policy AC fit; **DQN
(AT-D3QN/ICM-D3QN), C-REPS (Pecka), and C-TRAC (asymmetric SAC + C-VAE) get their own
trainers** (`experiments/dqn`, `experiments/creps`, `experiments/ctrac`). SAC/TD3/DDPG's
"FTR-Bench fit" is Isaac-Lab-gated and not runnable in this environment — see
`configs/baselines/OFFPOLICY_NOTE.md`.

Recipe for a new **native-PPO** baseline (Azayev/Mitriakov's path — most baselines want
this one): copy `configs/baselines/mitriakov.yaml` or any `final_training_configs/*.yaml`
(these use `flipper_training.experiments.ppo.config.PPOExperimentConfig`'s actual schema:
`heightmap_gen`/`world_opts`/`engine_opts`/`observations`/`objective(_opts)`/`reward(_opts)`/
`policy_config`+`policy_opts`/... — a full field list is in `experiments/ppo/config.py`),
swap the `policy_config:` + `policy_opts:` block for your `PolicyConfig`, and swap
`heightmap_gen`/`objective`/`reward` for whatever terrain/task/shaping the paper needs.
Train with that config via `experiments/ppo/train.py` (see the invocation gotcha above),
then deploy the produced `policy_final.pth` via `run_flipper_policy_sim.sh`.

Do **NOT** copy `configs/ftr_config_marv_potential_3.yaml` (or any other top-level
`configs/ftr_config_*.yaml`) as a starting point for a native-PPO baseline — those use an
entirely different schema (`task:`/`terrain:`/`env_cfg_overrides:`/`physx_gpu_*`) consumed
by the FTR-compat trainer (`marv_rl_training/ppo/train_ftr*.py`, Isaac-Sim-flavored
gymnasium task registry), not `PPOExperimentConfig`. Mixing the two schemas is exactly the
bug `azayev.yaml`/`mitriakov.yaml` had (both previously copied `ftr_config_marv_potential_3.yaml`
verbatim and would not construct against `PPOExperimentConfig` at all) — see their file
headers for the full fix writeup.

## Status notes

Azayev implementation done (hand-coded primitives + learned SDSM gate, PPO-trainable,
Algorithm-1 hard inference at eval — see `state_machine_policy.py` module docstring).
`configs/baselines/azayev.yaml` was rewritten onto the correct `PPOExperimentConfig`
schema and verified to load, train a few iterations, and evaluate end-to-end against
`experiments/ppo/train.py` (previously it was written against the unrelated FTR-compat
config schema — a different bug from the `policy_opts.state_mlp_opts` leftover once
flagged here, though that leftover field, from an older learned-action-head design that
predates `StateMachinePolicyConfig`, is also gone now). D3QN deploy-compliant
(continuous action), trains with real Double DQN (`DQNLoss(double_dqn=True)` — the trainer previously omitted
the flag and silently ran vanilla DQN + target net), and now offers the paper's incremental 9-action
paired-delta scheme alongside the original absolute per-flipper-bin table (`incremental=True`, see
`d3qn_policy.py` module docstring) + `experiments/dqn` trainer (also fixed: `PPOExperimentConfig(**config)`
rejected the trainer's own documented dqn-only keys like `n_iters`/`gamma`, and two `RunLogger` calls used a
nonexistent `.log`/`.save` API — either bug alone made every run of this trainer crash immediately). ICM-D3QN
(`icm.py` + the `use_icm`/`icm_opts` path in `experiments/dqn/train.py`) was previously inert: the intrinsic
reward was computed nowhere in the trainer (only `ICM.loss()` was wired in, so curiosity never reached the RL
objective and ICM-D3QN silently degenerated into plain AT-D3QN), the inverse-model loss MSE'd the inverse
head's output against the one-hot action instead of treating it as logits for cross-entropy (Eq. 14), and the
curiosity module built its own fresh encoder fully decoupled from the Q-network instead of sharing the Q-network's
single psi. All three are now fixed: the trainer adds `ICM.intrinsic_reward()` to the stored transition reward
before it reaches the replay buffer (`R_t = R^e + R^i`, Eq. 13), the inverse loss is real cross-entropy, and
`icm.ICM` is constructed from `d3qn_wrapper.get_encoder()` (the literal same encoder instance as the
Q-network, with the optimizer's param groups deduped so the shared weights get exactly one Adam step per
iteration despite contributing gradient to both losses). Pecka is
`pecka_policy.PeckaLinearPolicyConfig`
(rewritten from the previous fabricated discrete-selector version — continuous 2-dim positional
action, no encoder/MLP) + its dedicated `experiments/creps` trainer (context-free Constrained
REPS per the paper's own zero-context experiments). It now implements the FULL paper method
(previously only a 4-parameter fallback + tip-over-only safety):
* **6-parameter policy** (Sec IV-A "d) Policy"): a new observation,
  `observations/robot_state_with_terrain_lookahead.LocalStateVectorWithTerrainHeightAhead`
  (`LocalStateVector` + terrain height ``lookahead_dist`` ahead of the robot, sampled from the
  sim's ground-truth heightmap grid in the robot's yaw frame — see that module's docstring for
  the honest gap vs. the paper's real octomap sensor), finishes `pecka_policy.py`'s
  `extra_feature_idx` hook: point `obs_key` at it and set `extra_feature_idx: -1` to get the full
  `phi(s) = [pitch, height_ahead, 1]` (6 params); `create()` now validates the request against
  `env.observation_spec` and falls back to the 4-parameter `[pitch, 1]` policy with a logged
  warning if the observation is absent/misconfigured, instead of crashing or silently misreading
  a column.
* **Full Sec III safety** (`experiments/creps/train.py`): the per-episode safety `S` is now
  tip-over (`max_tilt`, pre-existing) OR hard-impact (new) — hard-impact reads the
  contact-induced part of the robot's CoG acceleration straight out of `PhysicsStateDer`
  (`(f_spring + f_friction) / mass`, i.e. `xdd - gravity_vector`, the same "specific force" an
  onboard accelerometer reads), thresholded by `max_impact_accel` (m/s^2); the trainer forces
  `prepare_env(..., force_return_derivative=True)` (`experiments/ppo/common.py`, new flag) to get
  `PhysicsStateDer` during training, not just eval. A short `impact_warmup_steps` window
  right after reset is excluded from the check — the soft-contact engine's spawn-settling
  transient (confirmed empirically to spike several g) would otherwise misclassify literally
  every episode's opening step as unsafe. "Delicate parts" (paper: e.g. sensors) has no MARV/engine
  analogue — documented as N/A, not silently dropped.
* **Example config**, `configs/baselines/pecka_full.yaml` (EUR-pallet crossing, matching the
  paper's own Fig. 3 task): loads, constructs a real `Env`, and was smoke-tested end-to-end this
  session for a couple of C-REPS iterations at `num_robots=4` (verified both safety branches fire
  independently — forcing `max_tilt`/`max_impact_accel` low each drives `mean_safety` to 0 on
  their own, and both relaxed gives `mean_safety=1.0`), producing `policy_final.pth` +
  `safety_gate.json`. See the config's own header comment for the exact CLI recipe used.

AT-D3QN / ICM-D3QN (Pan et al. 2023, arXiv:2306.10352 + Remote Sens. 15(18):4616) are
now FULL reproductions, not just a deploy-compliant network shape trained on generic
observations/reward. Three new/changed pieces, each config-selectable so the previous
generic modes remain available as documented alternatives:
* **State** (Eq. 1-2), `observations/pan_terrain.py`'s new `PanTerrainState`: 15
  downsampled local terrain heights H (mean per x-bin — confirmed against the
  ICM-D3QN journal PDF's `pdftotext` output, which disambiguates a `min`-vs-`mean`
  glyph the arXiv PDF renders ambiguously) + robot state E = {theta_f1, theta_f2,
  theta_R}. Frame and sign conventions are NOT free choices — see the module
  docstring for both derivations: the [L] frame is read as yaw-only-heading-aligned
  with a world-vertical Z (not chassis-normal, or a constant slope would read as
  flat); the front/rear flipper "up" sign and chassis pitch "nose-up" sign were
  verified empirically against this engine's actual `rot_Y` joint-rotation matrix,
  the robot mesh's bbox extents, and a constructed test quaternion (front raw
  positive = DOWN, rear raw positive = UP, engine `quaternion_to_pitch` positive =
  nose-DOWN — all three needed a sign flip to reach the paper's convention).
* **Reward** (Eq. 4/6/7, Eq. 10 minus the journal-only R_contact/kappa_3 — out of
  scope, see the module docstring), `rl_rewards/pan_reward.py`'s new `PanReward`:
  R_flipper's "candidate angle" (Fig. 4, never given a closed form) and R_end's
  "stuck" condition (never given ANY formula in either paper, checked via
  `pdftotext`) are the two things that had to be operationalized rather than
  transcribed — both documented explicitly, including the one genuinely ambiguous
  reading required for Eq. 4's "+-pi/36" notation (a dead-zone, not a fixed offset).
  Also documents an architectural limit worth knowing before reusing this reward:
  only the env's `Objective` can end an episode here, not a `Reward`, so R_end's
  |theta_R|>=pi/3 and "stuck" branches can re-fire every qualifying step rather than
  exactly once unless the `Objective` is configured to terminate at the same
  condition (the checked-in configs do this: `max_feasible_pitch: pi/3`).
* **Network** (Fig. 5), `d3qn_policy.py`'s new `fig5_topology: true` option
  (`_Fig5DuelingQ`): the literal terrain-branch-MLP -> fusion -> 16-dim S_t' ->
  dueling-heads topology, replacing the generic `EncoderCombiner` path (which
  remains the default, `fig5_topology: false`). Required resolving a genuine
  inconsistency between the two papers' own Fig. 5/7 diagrams (which annotate the
  fusion layer's "E" input as 4-dim, with an unexplained "contact" label never
  mentioned in either paper's prose) and Eq. 2 (which defines E as a 3-tuple) — the
  Fig. 7 diagram's OWN labeled ICM raw-state input (18 = 15 + 3) only reconciles
  with the 3-dim reading, which is what's built; see `_Fig5DuelingQ`'s docstring.
* **ICM's own encoder** (Fig. 7), `icm.py`'s new `separate_encoder: true` option:
  builds the paper's literal 3-layer psi (Dense 18->32->64->10, LeakyReLU) over the
  raw `PanTerrainState` tensor instead of sharing the Q-network's encoder (the
  pre-existing, still-default `separate_encoder: false` path) — required whenever
  `fig5_topology: true` (which has no `EncoderCombiner` to share). Found and fixed
  a real bug while smoke-testing this session: `experiments/dqn/train.py` popped
  `separate_encoder` out of `icm_opts` (correctly, to avoid a duplicate/unknown
  kwarg) but never re-passed it into the `ICM(...)` constructor call, so it silently
  reverted to the shared-encoder default and crashed on `encoder=None` — one-line fix.
* **Table 2 hyperparameters** reproduced where given (lr=5e-4, batch_size=256,
  replay_buffer_size=8e6 — added as a new optional `experiments/dqn/train.py` config
  key, previously hardcoded — lambda1=0.1, lambda2=0.33, kappa=0.005,
  beta_F=0.8/beta_I=0.2); gamma/eta/epsilon-greedy schedule/target-update
  rate/settlement-reward magnitude/t_max are NOT given numeric values by either
  paper (checked via `pdftotext` grep over both PDFs) and are documented, our-own
  defaults. Two paper-given PHYSICAL constants are also reproduced in the checked-in
  configs: `forward_track_velocity: 0.2` (Sec IV-A's stated test speed) and
  `incremental_vel_cmd` = 25 deg/s (Sec IV-A's stated flipper rotation rate), with
  `engine_iters_per_env_step` chosen so one RL decision, held at that rate, produces
  exactly one paper-sized `delta_f = pi/12` increment.
* **Verified this session**: a real-`Env` construct-test (shapes, finiteness, the
  sign-convention checks above, a full ICM forward+backward pass) plus running
  `experiments/dqn/train.py` end-to-end against BOTH checked-in configs (tiny
  `num_robots`/`n_iters` overrides) for both AT-D3QN and ICM-D3QN, producing
  `policy_final.pth` with no errors/NaNs in either case.

Mitriakov (PPO) is now a **native** `experiments/ppo/train.py` config
(`configs/baselines/mitriakov.yaml`, `MLPPolicyConfig`), not an FTR-side config as a
previous version of this note claimed — see that file's header comment for the full,
honest mapping from the paper's Eq. 1-6 (observation/action/reward) to what's actually
wired here (`LocalStateVector` + `Heightmap` observations, `StairCrossing` objective over
a randomized `StairsHeightmapGenerator` staircase, `PotentialGoalWithPenaltiesConfigurable`
reward as a stability-proxy stand-in for the paper's phase-conditional NESM/COG-deviation
and pitch-rate terms). Verified to load, train, and evaluate end-to-end this session.
Its `Heightmap` observation resolves to `marv_rl_training.observations.heightmap.Heightmap`
(NOT `flipper_training.observations.heightmap.Heightmap`, which several older
`test_configs/*.yaml`/`final_training_configs/*.yaml` still reference — the "restructure
and cleanup" commit (`4a54e89`) moved that class out of `flipper_training.observations`
without updating those configs; they are stale in the same way `azayev.yaml`/
`mitriakov.yaml` were, just a different symptom — out of scope to mass-fix here, flagged
for whoever touches those configs next).

SAC/TD3/DDPG have a ready, deploy-compliant architecture
(`offpolicy_ac_policy.OffPolicyACConfig`) but **no runnable trainer in this environment**:
the only place in this repo that instantiates `SACLoss`/`TD3Loss`/`DDPGLoss` against a
plugin `PolicyConfig` is `experiments/ctrac/train.py`, and it is hard-coupled to
`CTRACConfig` + a `GroundTruthContacts` observation (refuses to start without it) — not a
generic harness `OffPolicyACConfig` can be dropped into. The actual algorithm
implementations live in FTR-Bench (`src/FTR-Benchmark/ftr_algo/algorithms/rl/{sac,td3,ddpg}/`,
driven by `scripts/ftr_algo/train.py --algo {sac,td3,ddpg}`), which requires Isaac Lab
(`from omni.isaac.lab.app import AppLauncher` at import time) — confirmed not installed in
this environment (`ModuleNotFoundError: No module named 'omni'`). Full writeup, the exact
FTR-Bench invocation, and what a native trainer would need: `configs/baselines/OFFPOLICY_NOTE.md`.

C-TRAC is native, NOT FTR-side: `ctrac_policy.py` was rewritten
(the previous version's C-VAE was a dangling auxiliary — its latent fed nothing, each
component had its own `deepcopy` encoder, the privileged critic was dropped, the hybrid
loss was collapsed to plain MSE+KL) so the actor now genuinely conditions on the C-VAE's
estimates through one shared encoder, the critic reads the ground-truth
`GroundTruthContacts` observation (`observations/contacts.py`, new) when present (else a
documented symmetric fallback, never silent), and the hybrid loss implements Eq. 12-14
(BCE + dynamic-mask-weighted MSE + geometric-feasibility penalty) with a real two-stage
1:5 SAC:C-VAE trainer, `experiments/ctrac/train.py`. One torchrl-version gotcha is worth
knowing before touching either file: `SACLoss` in the installed torchrl (0.8.1) crashes on
a list of separately-built Q modules, so the Q-critic is a single template expanded via
`num_qvalue_nets` — see `configs/baselines/ctrac_NOTE.md` and both files' module
docstrings. IL family (BC/DAgger/Diffusion) dropped — no demonstrator (§22.5).
