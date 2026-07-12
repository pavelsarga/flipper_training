# Trainable baseline policies

These are the learned flipper-control **baselines** tracked for this project (external
papers reproduced or adapted onto this repo's engine — see the table below), implemented
as `PolicyConfig` subclasses using the SAME stack as the repo's PPO policies — TorchRL
operators, `tensordict`, and the shared `MLP`/`EncoderCombiner` building blocks. (A
previous revision of this line cited a `baselines_todo.md` as the source; no such file
exists anywhere in this repo's git history, in `MARV_RL` or the `flipper_training`
submodule — corrected here rather than left as a dangling reference.) Each obeys the
standard contract:

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
| Azayev & Zimmermann, RA-L 2022 (soft state-machine / HFC) | `state_machine_policy.StateMachinePolicyConfig` | native `flipper_training/experiments/ppo/train.py` (`ClipPPOLoss`) | `configs/baselines/azayev.yaml` | **full** — loads + trains + evals end-to-end, re-verified with a fresh `--local` run on 2026-07-10 (which also found+fixed a real `device: cuda` → `cpu` bug — see the config's own "RE-VERIFIED 2026-07" header note and "Status notes" below) |
| Mitriakov et al., IEEE RA Mag. 2021 (staircase negotiation) | `mlp_policy.MLPPolicyConfig` (swap to `gru_policy.GRUPolicyConfig` for the recurrent variant) | native `flipper_training/experiments/ppo/train.py` (`ClipPPOLoss`) | `configs/baselines/mitriakov.yaml` | **adaptation** — see the file's header comment for the exact per-term mapping to the paper's Eq. 1-6 (cross-checked against the actual Mitriakov_2021.pdf text on 2026-07-10, not just re-read from this file); loads + trains + evals end-to-end, re-verified 2026-07-10 (same `device` fix as azayev.yaml) |
| AT-D3QN, Pan et al. 2023 | `d3qn_policy.D3QNPolicyConfig` (`incremental=True` + `fig5_topology=True` for the paper's action set + network) + `observations/pan_terrain.PanTerrainState` + `rl_rewards/pan_reward.PanReward` | native `flipper_training/experiments/dqn/train.py` (Double DQN) | `configs/baselines/at_d3qn_full.yaml` | **full** — INDEPENDENTLY RE-VERIFIED 2026-07-10 (a prior table revision had this as self-reported only). Re-derived the sign/frame conventions from scratch (matched the original claims); found+fixed a real bug in `pan_reward.py`'s candidate-angle geometry (Fig. 4's angle was in the wrong reference frame — see "Status notes" below); fixed `device: cuda` → `cpu` (this host's sm_61 eGPU crashes on the flipper_venv's PyTorch build, confirmed by actually running it — same root cause as azayev/mitriakov); fixed a wrong `cwd` in the config's own header comment; re-ran the end-to-end smoke test after each fix. **Re-re-verified 2026-07-12** (fresh, independent pass — Table 2/Eq. 1-10/Fig. 5 re-derived directly from the PDFs incl. reading Figs. 3/4/5 as page images, not re-trusting the prior pass's text): found+fixed a real bug in `d3qn_policy.py`'s incremental action bridge — the front-pair delta was applied straight to the raw engine velocity with no paper-vs-raw sign correction, silently INVERTING the front flipper's response to `i=+1`/`i=-1` (confirmed empirically: before the fix, selecting `i=+1` measurably moved `theta_f1` *down*; the rear pair was unaffected, since MARV's raw/paper sign happens to already agree there) — see `BASELINE_AUDIT.md`'s "ROUND 4 continued" section for the full empirical before/after. Fresh end-to-end smoke run after the fix: exit 0, finite/reasonable loss. |
| ICM-D3QN, Pan et al. 2023 | AT-D3QN's config/network + `icm.ICM` (`use_icm: true`, `icm_opts.separate_encoder: true` for the paper's own Fig. 7 raw-state encoder) | native `flipper_training/experiments/dqn/train.py` | `configs/baselines/icm_d3qn_full.yaml` | **full** — INDEPENDENTLY RE-VERIFIED 2026-07-10, same pass and shared fixes as AT-D3QN above (`pan_reward.py`/`device`/header-comment); ICM's own Fig. 7 encoder dimension math (18/19/20-wide layers) was independently recomputed against the paper's figure and matches what's built. **Re-re-verified 2026-07-12** alongside AT-D3QN above (shares `d3qn_policy.py`'s incremental bridge, so the same front-pair sign bug affected this config too — same fix); Fig. 7's `18->32->64->10` / `19->32->10` / `20->32->9` layer widths independently re-confirmed by reading the actual figure image (not text alone) and by reading `MLP`'s `num_hidden` semantics in `policies/__init__.py` directly rather than assuming them. Fresh end-to-end smoke run after the fix: exit 0, finite loss, intrinsic reward (`icm_r`) positive and non-degenerate across all iterations. |
| Pecka et al., IROS 2016 (safety-constrained flipper control) | `pecka_policy.PeckaLinearPolicyConfig` + `observations/robot_state_with_terrain_lookahead.LocalStateVectorWithTerrainHeightAhead` | native `flipper_training/experiments/creps/train.py` (context-free Constrained REPS, gradient-free — no TorchRL loss) | `configs/baselines/pecka_full.yaml` | **full** — independently re-verified 2026-07-10 (separate pass from the one that built it, see "Status notes"): fresh `--local` runs confirm the full 6-parameter `phi(s)=[pitch,height_ahead,1]` engages (`phi(s) dim=3 ... omega dim=6` in the startup log, not the 4-param fallback), the 4-param fallback itself works both when deliberately requested and when misconfigured (out-of-range `extra_feature_idx` logs a warning and falls back rather than crashing), and both Sec III safety clauses are independently load-bearing (forcing `max_tilt`/`max_impact_accel` low alone each drives `mean_safety`→0.0/`certified_safe=False`; both relaxed → `mean_safety=1.0`). `device: cpu` (already fixed from `cuda:0` — same sm_61 issue as azayev/mitriakov below). Re-re-verified 2026-07-12 (third, independent pass): fresh end-to-end runs today reproduce all of the above unchanged (see "Status notes") plus a documented, worked-around `/tmp` debug-plot permission gotcha in the shared `heightmaps/pallets.py` (environment-level, not a Pecka defect) |
| C-TRAC, Pan et al. 2025 IROS | `ctrac_policy.CTRACConfig` | native `flipper_training/experiments/ctrac/train.py` (two-stage 1:5 SAC : C-VAE) | `configs/baselines/ctrac_full.yaml` (checklist: `configs/baselines/ctrac_NOTE.md`) | **full** — independently re-verified 2026-07-10 (round 4, this pass): multi-frame history `o_t^H`, clean denoising target, wide privileged heightmap `h_t^f`, an optional 5-D effective action (Eq. 3, off by default for cross-baseline comparability), and LeakyReLU-everywhere are now all implemented and config-selectable, cross-checked line-by-line against the actual paper text (Sec. III-B/IV/V-A.1/V-A.3), smoke-tested end-to-end against a real `Env` (3 full training runs + a dedicated deploy-contract test), AND live-sim-verified through the real `flipper_policy_node.py` against the Gazebo sim — which found and fixed 4 real pre-existing deploy-path bugs (heightmap shape handling x2, a `TensorDictPrimer`-placeholder crash in the new history buffer, `policy_inference_module.py` rejecting every C-TRAC config outright) plus a missing-vecnorm-checkpoint gap, none reachable by `Env`-only testing — see "Status notes" below. Only remaining, explicitly-kept divergence: Stage-1 C-VAE pretraining still uses random-init rollouts, not the paper's expert trajectories (no experts exist in this repo; kept by explicit user instruction, not an oversight) |
| FTR SAC / TD3 / DDPG | `offpolicy_ac_policy.OffPolicyACConfig` | **none native.** FTR-Bench `scripts/ftr_algo/train.py --algo {sac,td3,ddpg}` (requires Isaac Lab) | N/A — see `configs/baselines/OFFPOLICY_NOTE.md` | **blocked here** — policy architecture is ready/deploy-compliant; no native trainer exists for it, and Isaac Lab (`omni.isaac.lab`) is not installed in this environment (confirmed by import test) |
| FTR PPO/TRPO (on-policy, pre-existing) | existing `mlp_policy` / `gru_policy` / `lstm_policy` | native `experiments/ppo/train.py` (`ClipPPOLoss`) / FTR-Bench TRPO | pre-existing `test_configs/*.yaml`, `final_training_configs/*.yaml`, `sota_configs/*.yaml` | **full** — predates this baselines effort, not touched by it |

GC-only / OA-only are *ablations* of your own contribution, not external baselines.

NOTE — the imitation-learning family (Behavior Cloning / DAgger / Diffusion Policy)
was DROPPED: those are IL methods that require an expert demonstrator, and this
project has no natural demonstrator (see progress.md §22.5). Only reward-driven
baselines are kept.

### Verified file existence

This table is co-maintained by several concurrently-running agents, each scoped to
different baselines; a `pending` row means the owning agent's config hadn't landed yet
as of the last check. Re-checked at the end of THIS pass (2026-07-10, scope: re-verify
azayev.yaml/mitriakov.yaml + refresh this table — see "Status notes" below for exactly
what this pass did and did not do):

```
$ ls configs/baselines/
at_d3qn_full.yaml  azayev.yaml  ctrac_full.yaml  ctrac_NOTE.md  icm_d3qn_full.yaml
mitriakov.yaml  OFFPOLICY_NOTE.md  pecka_full.yaml
```

All four of `at_d3qn_full.yaml` / `icm_d3qn_full.yaml` / `pecka_full.yaml` /
`ctrac_full.yaml` now exist (`ctrac_full.yaml` did not yet exist earlier in this same
overall effort — an older revision of this section said so; that is now stale and is
corrected here). None of the four are this pass's work — each was produced by a
separate concurrently-running agent outside this pass's scope (AT-D3QN/ICM-D3QN/Pecka/
C-TRAC respectively; see each file's own header comment and its corresponding paragraph
in "Status notes" below, written by the agent that built it). This pass did not
independently re-verify their internals, only confirmed the files exist and glanced at
their self-reported status for the master table above. No row is `pending` any more as
of this check; re-run the `ls` above before trusting that if more time has passed.

One thing this pass DID verify and IS worth another agent's attention: `azayev.yaml`
and `mitriakov.yaml` both shipped with `device: cuda`, which crashes immediately on
this host (`torch.AcceleratorError: no kernel image is available for execution on the
device` — this host's GPU is a GTX 1050 / sm_61, and flipper_venv's torch 2.9.1+cu128
only ships kernels for sm_70+). Both were fixed to `device: cpu` here (see their
headers). `at_d3qn_full.yaml` (`device: cuda`) and `icm_d3qn_full.yaml` (`device: cuda`)
still request CUDA and were NOT checked for the same failure in this pass (out of
scope) — `ctrac_full.yaml` already uses `device: cpu` with its own "override to
cuda:0 for a real run" comment, i.e. the same fix, independently arrived at by that
config's own author. UPDATE (2026-07-10, later Pecka full-fidelity pass): `pecka_full.yaml`
hit this identical sm_61 crash and has SINCE been fixed to `device: cpu` too (see its own
header comment and the Pecka paragraph in "Status notes") — it no longer belongs in the
"still request CUDA" group above; re-run the `ls`/grep yourself before trusting this for
`at_d3qn_full.yaml`/`icm_d3qn_full.yaml`, which remain unchecked.

UPDATE (2026-07-12, azayev/mitriakov scope, re-run of the freshness check only): re-ran
`grep -H '^device:' configs/baselines/*.yaml` — `at_d3qn_full.yaml` and `icm_d3qn_full.yaml`
now ALSO read `device: cpu` (each with its own "RE-VERIFIED 2026-07-10" comment), so the
"remain unchecked" / "still request CUDA" caveat above is now stale too: as of this check,
all six `configs/baselines/*_full.yaml` + `azayev.yaml` + `mitriakov.yaml` files use
`device: cpu`, none still request `cuda` unconditionally. This was the owning AT-D3QN/
ICM-D3QN agent's own fix, made sometime after the paragraph above was written — not
something this pass did (out of this pass's scope, per every prior paragraph here) — only
independently observed via the same one-line grep. Working tree also currently shows
`at_d3qn_full.yaml`/`icm_d3qn_full.yaml`/`pecka_full.yaml`/`ctrac_NOTE.md` as locally
modified-but-uncommitted (`git status`), consistent with their owning agents still being
active concurrently; not opened or evaluated beyond this device-line grep, per scope.

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

**Correction (2026-07-10):** this section used to open with a
`flipper_venv/bin/python src/flipper_training/test_baseline_policies.py` snippet
("constructs each policy against a mock Env... proves training-compatibility"). That
script does not exist and never has — `git log --all` for
`**/test_baseline_policies.py` in both `MARV_RL` and the `flipper_training` submodule
returns nothing, and there is no mock-`Env` helper anywhere in the tree either. Removed
rather than left as a copy-pasteable command that 404s. The block below it also silently
assumed `flipper_venv/` lives inside the `MARV_RL` repo (`flipper_venv/bin/python` /
`../../../../../flipper_venv/bin/python`); it actually lives two directories above
`MARV_RL` root, at `/home/cnuc/marv_ws/flipper_venv` — the 5-level-up form resolved to
`MARV_RL/flipper_venv/bin/python`, which doesn't exist (`No such file or directory`,
reproduced verbatim before fixing). Fixed below with absolute paths, which don't rot
under `cd`/directory-depth changes; re-verified by actually running it, exit code 0.

The only validation that currently exists in this repo is a real (not mock) `Env`
smoke run through the actual trainer entry point — which is CPU-only and takes ~5-10s
at these sizes, so "no GPU needed" still holds. Run the target config with tiny
overrides, e.g.:

```bash
cd /home/cnuc/marv_ws/src/MARV_RL/src/flipper_training/flipper_training/experiments/ppo
/home/cnuc/marv_ws/flipper_venv/bin/python train.py \
  --local /home/cnuc/marv_ws/src/MARV_RL/configs/baselines/azayev.yaml \
  use_wandb=false num_robots=4 time_steps_per_batch=8 frames_per_sub_batch=16 \
  epochs_per_batch=1 eval_and_save_every=1 max_eval_steps=5 eval_repeats_after_training=1 \
  total_frames=64 device=cpu objective_opts.cache_size=500
```

(`objective_opts.cache_size=500` only speeds up the smoke test — the checked-in
configs use the proven `cache_size: 40000`, which takes noticeably longer to
build at real `num_robots` scale but is unrelated to correctness.)

Equivalently, the `-m` form (see the invocation gotcha below) doesn't need to run from
inside `experiments/ppo/` — just from wherever `flipper_training` resolves on
`sys.path` (its own repo root, shown below, or set `PYTHONPATH`). This is what was
actually used for the fresh re-verification runs logged at the bottom of this file:

```bash
cd /home/cnuc/marv_ws/src/MARV_RL/src/flipper_training   # so `flipper_training` resolves on sys.path
/home/cnuc/marv_ws/flipper_venv/bin/python -m flipper_training.experiments.ppo.train \
  --local /home/cnuc/marv_ws/src/MARV_RL/configs/baselines/mitriakov.yaml \
  use_wandb=false num_robots=4 time_steps_per_batch=8 frames_per_sub_batch=16 \
  epochs_per_batch=1 eval_and_save_every=1 max_eval_steps=5 eval_repeats_after_training=1 \
  total_frames=64 objective_opts.cache_size=500
```

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

**Independent re-verification (2026-07-10, later pass, separate agent from the one that
built the above)**: re-ran the config header's own CLI recipe fresh, in new run
directories, rather than trusting the prior agent's `runs/creps/pecka_full_freshsmoke_*`
output. Confirmed:
* The startup log reports `phi(s) dim=3 (pitch + height_ahead + bias), omega dim=6` —
  the full 6-parameter policy genuinely engages by default with the checked-in config
  (not the 4-parameter fallback) — and a 2-iteration run completes with no
  errors/NaNs, producing a `certified_safe` safety gate.
* Both `extra_feature_idx` fallback paths work as documented: setting it to `null`
  (deliberate 4-param request) silently gives `phi(s) dim=2, omega dim=4` with no
  warning; setting it to an out-of-range value (`999`) logs
  `extra_feature_idx=999 is out of range for obs_key='...' (width=16) -- falling back
  to the 4-parameter phi(s)=[pitch,1] policy` and still completes the run rather than
  crashing.
* Both Sec III safety clauses are independently load-bearing, not just written but
  inert: `max_tilt=0.001, max_impact_accel=1000` (tip-over-only, effectively
  impossible to satisfy) → `mean_safety=0.000`; `max_tilt=1000, max_impact_accel=0.5`
  (hard-impact-only, below the ~1g / 9.81 m/s² resting-contact baseline the
  accelerometer-style proxy reads at steady state) → `mean_safety=0.000`; both relaxed
  (`max_tilt=1000, max_impact_accel=1000`) → `mean_safety=1.000`. The
  `max_tilt=0.001` run also exercised `solve_creps_dual`'s "no safe samples in the
  batch" saturation branch (logged warning, uniform fallback weights, no crash).
* Traced `f_spring`/`f_friction` into `engine.py: forward_kinematics` to confirm they
  are genuinely per-contact-point tensors `(B, n_pts, 3)` at runtime (matching
  `PhysicsStateDer.dummy()`'s shapes), so `impact_accel()`'s `.sum(dim=1)` in
  `experiments/creps/train.py` sums over contact points as intended. Worth flagging
  for whoever next touches `engine_state.py`: that file's own `PhysicsStateDer`
  attribute docstring claims `f_spring`/`f_friction` are shape `(num_robots, 3)` —
  stale and wrong (pre-existing, not introduced by this baseline, not fixed here as
  out of scope) — but this does NOT indicate a bug in the hard-impact criterion, which
  uses the real runtime shape correctly.
* `device: cpu` in the checked-in config was independently confirmed to actually run
  (the `cuda:0` alternative was not re-tried here — the config header already documents
  why it fails on this host's sm_61 GPU, matching azayev/mitriakov's independently
  diagnosed root cause).
* Spot-checked the docstrings' specific empirical claim ("a ~1.5m drop peaks at ~90g;
  ordinary resting/settling stays under ~5g") with a standalone script (not checked into
  the repo) that spawns the robot 1.5m above the pallet-crossing start height and prints
  the exact `impact_accel` quantity `experiments/creps/train.py` computes, every step.
  Result: exactly 0 (contact-free, correctly excluding gravity) throughout the ~0.5s
  free-fall — matching the expected free-fall time for 1.5m (`sqrt(2*1.5/9.81)=0.55s`) —
  then a single-step spike to **~104g** at touchdown, decaying back down within a step or
  two. Same order of magnitude as the docstring's "~90g", corroborating it rather than
  contradicting it (exact number depends on contact stiffness/point density, which the
  docstring's own config differs slightly from this ad hoc script's CLI overrides).

**Second independent re-verification (2026-07-12, third pass overall, separate agent
from both passes above)**: re-read Pecka_2016.pdf Sec III and Sec IV-A directly from the
PDF (not from either prior pass's paraphrase) and cross-checked every code claim above
against it (state = `[pitch, height ~20cm ahead]`; policy linear in state, 3
params/pair -> 6 total; safety = tips over, hits hard [measured as deceleration], or hits
delicate parts) — no mismatch found. Independently traced, line-by-line, that
`engine.py`'s `F_cog = self._F_g + act_force.sum(dim=1)` / `xdd = F_cog / total_mass`
with `act_force = F_spring + F_friction` algebraically gives
`(f_spring+f_friction).sum(dim=1)/mass == xdd - gravity_vector`, i.e. `impact_accel()` in
`experiments/creps/train.py` really does read the paper's "deceleration" quantity, not
just a plausible-sounding docstring claim. Then ran three FRESH end-to-end
`experiments.creps.train` invocations today (new `runs/creps/pecka_full_freshverify_*`
directories, not reusing the 2026-07-10 runs), reproducing all three headline claims
independently:
* Checked-in config, default `policy_opts` (`num_robots=4 n_iters=2 steps_per_iter=20`):
  startup log `phi(s) dim=3 (pitch + height_ahead + bias), omega dim=6` — full 6-param
  policy still engages by default — completes cleanly, `certified_safe=True`.
* `max_tilt=0.001` (tip-over-only, effectively unsatisfiable): `mean_safety=0.000`,
  `certified_safe=False`, and the dual solver's saturation fallback fires as documented
  (`gamma` pinned at its `1e8` box bound in the log).
* `policy_opts.extra_feature_idx=null` (explicit 4-param request): startup log
  `phi(s) dim=2 (pitch + bias), omega dim=4` — the fallback path still works on demand.

One NEW finding, environment-level rather than a Pecka code defect: the first fresh-run
attempt crashed in `heightmaps/pallets.py`'s debug-plot save (`PermissionError` on
`/tmp/training_heightmap_debug.png`) because that fixed, unnamespaced `/tmp` path had
been left behind, owned by `root`, by a non-`cnuc` process at some earlier point — any
later run as `cnuc` (the required user for this repo, see top-level CLAUDE.md) then
fails to overwrite it. Not a regression in anything touched by this baseline (the pallet
heightmap generator is shared, pre-existing, unrelated code; `pecka_full.yaml` merely
consumes it), so left unfixed as out of this task's scope — worked around by deleting
the two stale root-owned files (`training_heightmap_debug.png`,
`standalone_pallet_test.png`); they now regenerate `cnuc`-owned. Flagging here so the
next agent that hits an inexplicable `PermissionError` on those exact paths doesn't
waste time on it — it is a multi-user `/tmp` collision in shared code, not a `pecka_full`
or C-REPS bug.
Conclusion: all four of the original task's items — 6-parameter policy via the terrain-
height-ahead observation, the full tip-over-OR-hard-impact Sec III safety criterion,
a complete/runnable `pecka_full.yaml`, and honest docstrings/README — were already
correctly implemented by the passes above; this pass found zero Pecka-specific defects
and changed no `flipper_training` source, only added this paragraph.

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
* **Independently re-verified in a LATER pass (2026-07-10)**, adversarially, against
  the actual PDFs rather than re-reading the above claims: re-derived the front/rear
  flipper and chassis-pitch sign conventions from scratch (engine `rot_Y` matrix +
  `robots/marv.yaml` wheel offsets + a hand-built test quaternion run through the
  real `rotate_vector_by_quaternion`/`quaternion_to_pitch`) and the Fig. 5/Fig. 7
  E-dimension resolution (independently recomputed the 18/19/20-wide ICM layer
  inputs from Fig. 7's own numbers) — both matched the original claims exactly, no
  changes needed. Found and fixed one real bug, in two parts, in
  `pan_reward.py`'s `_candidate_angle` (Fig. 4's candidate-angle geometry):
  1. the angle was computed directly in `[L]` (the yaw-only, world-vertical frame
     `sample_terrain_points_relative` samples in) instead of `[R]` (the full chassis
     body frame Eq. 2 and Sec. III-D's own prose specify — "the vector and the
     ROBOT coordinate system are calculated") — a chassis already perfectly aligned
     with a slope would see a spuriously large candidate angle instead of ~0;
  2. less obviously, even after rotating into `[R]`, the angle was measured FROM the
     chassis origin rather than from the hinge's own true (pitch-shifted) world
     position — a forward hinge on a nose-up chassis sits measurably above the
     chassis origin purely from the rotation (`hinge_z=0` in the body frame does
     NOT mean level-with-origin once tilted), and a first attempt at the fix missed
     this, plus separately used the chassis's absolute +X as the angle's reference
     axis for BOTH flippers when the rear's own "outward" axis is -X (a perfectly
     aligned rear flipper read ~180 deg instead of ~0 deg until this was caught).
     The final fix rotates the hinge's own body-frame offset by the FULL chassis
     quaternion (not a pitch-only shortcut) to locate it in world space, builds the
     hinge-to-candidate vector from there, rotates that into `[R]`, and measures the
     angle against each flipper's own outward axis. Verified numerically against a
     real constructed `Env` (not just re-derived by hand): with a synthetic chassis
     resting exactly flush on a constant slope (hinge exactly on the slope line,
     pitch exactly matching), the fixed method returns EXACTLY 0.0 deg at every
     sampled point for BOTH the front and rear flipper, and returns correctly-signed
     non-zero angles when the chassis is perturbed off that resting pose. Also fixed,
     smaller: (a) `_r_pitch`'s smoothness-window average divided by a constant
     `(pitch_smoothness_window - 1)` even during the first few steps of an episode
     when the trailing buffer is still zero-padded, systematically UNDERESTIMATING
     `Delta_theta_R^k` (and so under-penalizing) right after every reset — now
     divides by the actual valid-sample count, identical to the paper's literal
     `1/(k-1)` once the buffer is warmed up; (b) both configs hard-coded
     `device: cuda`, which actually crashes on this host's eGPU (sm_61 unsupported by
     the flipper_venv's PyTorch build — confirmed by running it, not inferred, same
     root cause as the azayev/mitriakov `device` fix) — changed to `cpu`; (c) both
     configs' header comments said to run from `src/flipper_training`, but
     `configs/` lives at the MARV_RL repo root (same as every other
     `configs/baselines/*.yaml`) — confirmed by reproducing the resulting
     `FileNotFoundError` before fixing the comment. Re-ran the end-to-end smoke test
     (`experiments/dqn/train.py`, tiny `num_robots`, this time via the config's own
     now-fixed `device: cpu` default) against both configs after every fix — still
     no errors/NaNs, finite/reasonable losses (`icm_r` intrinsic reward positive and
     non-degenerate for ICM-D3QN).

**Independent re-audit, 2026-07-12** (scope: AT-D3QN/ICM-D3QN only, a fresh pass, not a
continuation of the same session as the two passes above — see `BASELINE_AUDIT.md`'s
"ROUND 4 continued — Pan" section for the full writeup). Re-derived every numeric/
architectural claim directly from the two PDFs (`pdftotext` over the full text of both,
PLUS reading Figs. 3/4/5 (`Pan_2023_AT-D3QN.pdf`) and Fig. 7 (`Pan_2023_ICM-D3QN.pdf`) as
actual page images — not trusting the prior passes' transcriptions) and re-ran fresh
construct + smoke tests against the code as it stands today, 2 days after the pass being
re-audited (code changes elsewhere in the repo could in principle have bit-rotted it since
— confirmed they had not, modulo the bug below). Everything already claimed (Eq. 1's
"mean" not "min", Eq. 2's 3-dim E and its yaw-only/world-vertical `[L]` frame, Eq. 3's
9-action structure, Eq. 4/6/7/10's reward transcription including Eq. 5's paper-literal
`sum-of-k / (k-1)` normalization, Fig. 5's 4-vs-3-dim `E` inconsistency and its resolution,
Fig. 7's `18/19/20`-wide ICM layers, every Table 2 value, the `0.2 m/s`/`25 deg/s` Sec.
IV-A constants) re-confirmed exactly, independently, from the primary sources — no
contradictions found. One genuine bug WAS found and fixed this pass: the incremental
action bridge (`d3qn_policy.py`'s `_IncrementalToContinuous`, used by both checked-in
configs via `incremental: true`) applied Eq. 3's paper-frame delta (`i*delta_f`, defined
against `theta_f1`, Eq. 2's positive-up convention) straight to the engine's raw
angular-velocity slots with no sign correction. `pan_terrain.py`'s own established
raw-vs-paper convention (front: `theta_f1 = -raw`; rear: `theta_f2 = +raw`, already
paper-signed) meant this was a silent no-op for the rear pair but INVERTED the front pair
— confirmed empirically against a real `Env`: applying the one-hot action for `i=+1,j=0`
for 5 sub-steps from a random reset state moved `theta_f1` by **-1.31 / -1.25 rad**
(strongly negative, i.e. the front flipper went DOWN when "i=+1" should raise it per a
literal Eq. 3 reading) — same test after the fix gives **+0.70 / +1.03 rad** (positive, as
intended), with the rear pair's behavior provably byte-identical before/after (its
correction factor is `+1.0`, a no-op). Fixed via a new, documented
`front_rear_action_sign` field on `D3QNPolicyConfig` (default `(-1.0, 1.0)`, mirroring
`pan_terrain.py`'s own convention, overridable for a different robot). Re-ran both
configs' own documented smoke-test CLI invocations fresh after the fix: both exit 0,
finite/reasonable losses, ICM-D3QN's intrinsic reward positive and non-degenerate across
all iterations. NOT independently re-derived this pass (spot-checked only — no
NaN/crash/~180-deg misreadings across a level and a +20-deg-pitch case, corroborating but
not a fresh proof): `pan_reward.py`'s `_candidate_angle` (Fig. 4) geometry, which the
2026-07-10 pass already verified with a specific numeric test (flush-on-slope → exactly
0.0 deg) that this pass judged not worth re-deriving a third time given no contradicting
evidence turned up anywhere else.

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

**2026-07-10 re-verification pass** (scope: confirm azayev.yaml/mitriakov.yaml still
load+train+eval end-to-end against `experiments/ppo/train.py`, tighten Mitriakov's
staircase terrain, add SAC/TD3/DDPG configs if a native trainer exists, refresh this
table — explicitly NOT the AT-D3QN/ICM-D3QN/Pecka/C-TRAC configs above, which were
someone else's concurrent scope). Findings, all from actually running things rather
than re-reading prior claims:
* **Both configs re-run fresh, end-to-end, exactly as checked in** (`--local
  configs/baselines/{azayev,mitriakov}.yaml` + only smoke-scale overrides —
  `num_robots=4`, `total_frames=16`, etc. — no `device=` override): both completed
  training + evaluation with real numeric output, exit code 0.
* **Found and fixed a real bug this way that a schema/field-name check alone would not
  have caught**: both files' checked-in `device: cuda` crashes immediately on this
  host — `torch.AcceleratorError: CUDA error: no kernel image is available for
  execution on the device`. Root cause: this host's GPU is a Thunderbolt eGPU, GTX
  1050 (Pascal, compute capability sm_61); `flipper_venv`'s installed `torch==2.9.1+cu128`
  wheel embeds kernels only for sm_70 and above (confirmed via the wheel's own runtime
  warning: "Minimum and Maximum cuda capability supported by this version of PyTorch is
  (7.0) - (12.0)"), so there is no PTX to JIT down to sm_61 either — a hard,
  unconditional incompatibility on this specific box, not a config mistake. Changed
  both files' `device:` to `cpu` (what was actually proven to run), with a header
  comment explaining exactly why and how to switch back on sm_70+ hardware. This
  matches the independent choice `ctrac_full.yaml` (a different agent's config) already
  made for the same reason. `at_d3qn_full.yaml`/`icm_d3qn_full.yaml`/`pecka_full.yaml`
  still say `device: cuda`/`cuda:0`; not checked whether they hit the identical failure
  (out of this pass's scope — the crash happens inside `DPhysicsEngine.__init__`,
  common to every trainer, so it plausibly affects them too).
* **Mitriakov staircase terrain**: the file's own header already documented this as
  fixed (`heightmap_gen: StairsHeightmapGenerator` + `objective:
  rl_objectives.stair_crossing.StairCrossing`, replacing an old `terrain: exp_step40_up`
  TODO from a different config schema entirely). Independently confirmed real by reading
  `heightmaps/stairs.py` and `rl_objectives/stair_crossing.py`: both classes exist, and
  every field the config passes in `heightmap_gen_opts`/`objective_opts` matches those
  classes' actual dataclass fields exactly. There is no `terrain: <name>` string-keyed
  registry anywhere in this native trainer's config schema (`PPOExperimentConfig`) — that
  convention belongs to the separate FTR-compat/Isaac-Sim-facing trainer
  (`marv_rl_training/ppo/train_ftr*.py`, `configs/ftr_config_*.yaml`); the native trainer
  this baseline uses always names a `heightmap_gen` **class** directly, so "register a
  named staircase terrain" (as originally asked) does not apply to this trainer and
  nothing needed adding beyond what the header already fixed.
* **Mitriakov reward/observation paper-mapping honesty check**: read
  `/home/cnuc/obsidian/baselines/pdfs/Mitriakov_2021.pdf` directly (Problem Description +
  Reward Function Design sections, Eq. 1-6) rather than trusting the config header's own
  paraphrase. Every specific claim in the header checked out against the actual paper
  text (9-D obs with pitch deliberately excluded per the paper's own stated reasoning;
  5/3-DoF action with yaw explicitly disabled; Eq. 3-4 fractional-progress reward; Eq. 5
  COG-vs-geometric-center ascent penalty; Eq. 6 pitch-rate descent penalty; tip-over at
  |pitch|>pi/2; PPO chosen over SAC/TD3 for the paper's own stated reasons) — and,
  independently, `rl_rewards/rewards.py`'s `PotentialGoalWithPenaltiesConfigurable` code
  was read to confirm it really is phase-agnostic (no ascent/descent branching) and really
  does route tip-over through `reward[fail] += failed_reward`, and a repo-wide grep
  confirmed there is no NESM/support-polygon/COG-projection implementation anywhere in
  `rl_rewards/`/`rl_objectives/` — so the header's "this is a documented proxy, not the
  paper's literal mechanism" framing is accurate, not just asserted.
* **SAC/TD3/DDPG**: `configs/baselines/OFFPOLICY_NOTE.md` already existed with this
  conclusion reached; spot-checked its two load-bearing claims rather than re-deriving
  them — `grep -r "SACLoss\|TD3Loss\|DDPGLoss"` confirms `experiments/ctrac/train.py` is
  the only trainer in this tree that instantiates one of those losses (and it's
  hard-coupled to `CTRACConfig`), and `python -c "import omni"` confirms Isaac Lab is
  genuinely not installed in `flipper_venv`. No `configs/baselines/{sac,td3,ddpg}.yaml`
  added, for the same reason the note already gives.

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

**Round-4 full-fidelity pass (2026-07-10, separate agent instance, scope: C-TRAC only —
take the "single-frame" architecture above from REASONABLE-APPROXIMATION/FAITHFUL-
DISTILLATION to the paper's FULL method, item by item, per `BASELINE_AUDIT.md`'s
divergence table).** All five previously-open items now implemented in
`ctrac_policy.py`/`observations/heightmap.py`, config-selectable, default OFF so
construction stays byte-identical unless opted in — see that file's module docstring for
the complete writeup of each, `ctrac_NOTE.md` for the config knobs, and the paper
cross-checks below (`Pan_2025_C-TRAC.pdf`, read directly, not paraphrased):
* **Multi-frame history `o_t^H`** (`policy_opts.history_len`): a parameter-free ring
  buffer (`_FrameHistoryBuffer`) INSIDE the policy operator, carried via
  `("next", HISTORY_KEY)` + `TensorDictPrimer`/`InitTracker()`, the same convention
  `state_machine_policy`/`gru_policy`/`lstm_policy` already use for recurrent state — chosen
  over a `CatFrames` env-side transform because the generic deploy node calls the actor
  operator directly, never through `env.transform`, so `CatFrames` would silently never
  fire at deploy time. Confirmed by reading the actual paper: Sec. III-B defines
  `o_t^H` symbolically but genuinely never states a numeric H anywhere (checked
  III-B/IV-C/V-A.1) — `ctrac_full.yaml`'s `history_len: 5` is a documented placeholder, not
  a paper constant.
* **Clean denoising target**: new `Env.emit_clean_observations` flag
  (`experiments/ppo/config.py`/`environment/env.py`) emits a parallel noiseless mirror of
  every observation under `("next", "clean", ...)` each step; the C-VAE's reconstruction
  loss uses it as the target when present, else the pre-existing plain-next-obs behaviour —
  matches Sec. IV-C's "denoising autoencoder... from noisy multi-frame interaction data"
  literally (noisy input, clean target).
* **Wide privileged heightmap `h_t^f`**: new `PrivilegedHeightmap` class
  (`observations/heightmap.py`, alongside `Heightmap` for the actor's own `h_t^l`) — Eq. 1's
  `[0.4,1.0]x[-0.5,0.5]` and Eq. 2's `[-1.0,1.4]` length range are literal matches to the
  paper text; width for `h_t^f` is undocumented by the paper and kept equal to `h_t^l`'s,
  disclosed as a choice, not a paper number.
* **5-D effective action** (Eq. 3, `policy_opts.effective_action_5d`, OFF by default per
  explicit task instruction "keep the 8-D default for cross-baseline comparability"): an
  env-side `Transform` (`EffectiveActionTransform`), not a policy-operator submodule — this
  is a hard requirement, not a style choice: `torchrl` 0.8.1's `SACLoss` reads
  `dist.rsample()` straight off the actor distribution on every loss code path
  (`_actor_loss`/`_compute_target_v2`), never re-running anything downstream, so the
  Q-critic must be built at the ACTOR's own width (5-D in this mode) and the 8-D expansion
  must happen strictly outside the actor, at the env boundary. Verified against the
  installed `torchrl/envs/transforms/transforms.py` and reproduced empirically (an earlier
  revision that built an 8-D critic unconditionally crashed the instant this mode trained).
  Consequence disclosed prominently: `get_policy_operator()(td)["action"]` returns the 5-D
  action in this mode, not env-native 8-D — the one documented exception to this repo's
  usual deploy contract.
* **LeakyReLU everywhere** (Sec. V-A.1): `MLP`/`HeightmapEncoder` already took an
  `activation` constructor kwarg (repo-wide Tanh/ReLU defaults preserved); `ctrac_full.yaml`
  sets `activation: ${cls:torch.nn.LeakyReLU}` on every actor/critic/C-VAE `*_mlp_opts` and
  every heightmap `encoder_opts`, matching "All neural networks employ LeakyReLU activation
  functions" exactly. All five paper-stated Sec. V-A.1 hyperparameters (entropy coef 1.5,
  target-net tau 0.01, gamma 0.99, KL weight 1.0, grad-clip 0.5) and the Sec. V-A.3 noise
  stds (heightmap: 0.08 m, a literal match) were independently re-checked against the PDF
  text too — all already correct pre-round-4, unchanged here.

Verification actually run (not just claimed): `py_compile` on every touched file;
`experiments/ctrac/train.py --local ctrac_full.yaml` with the config's own documented
smoke-test overrides, to completion (both stages + eval + `policy_final.pth`), THREE times
— as-checked-in, with `effective_action_5d=true` added, and with the round-4 additions
forced back off (`history_len=1 emit_clean_observations=false`, the pre-round-4-equivalent
regression check) — all exit 0, finite losses; and a dedicated deploy-contract script
(bare/transform-free multi-tick actor calls in both action-width modes, confirming the
history buffer's documented graceful degradation; a separate carried-state run confirming
the buffer genuinely accumulates a real sliding window and not just repeats; the
`history_len=1` exact-identity guarantee; `EffectiveActionTransform.expand()` standalone;
the `emit_clean_observations` on/off "clean"-key gating) — full detail and exact commands
in `ctrac_NOTE.md`'s "Validate" section. One honest gap FOUND (not introduced) while tracing
the deploy path for the history-buffer claim: `policy_inference_module.py`'s
`infer_action()` builds a fresh tensordict every control tick and never threads
`("next", HISTORY_KEY)` forward, so history does not actually accumulate across REAL ROS
control ticks today (deployed behaviour there is `history_len=1`-equivalent regardless of
config) — a pre-existing property of that wrapper shared identically by GRU/LSTM/
`state_machine`'s own recurrent carry, not new here and out of this pass's scope to fix;
training/eval (every collector/rollout usage) is unaffected and genuinely uses real
multi-frame windows. Documented in both `ctrac_policy.py`'s docstring and `ctrac_NOTE.md`.

**Live-sim pass, after the above**: none of the checks so far ever exercise
`Env._to_realistic_env()`/`from_realistic_world()` — the code path ONLY the real ROS deploy
node uses. Actually running `flipper_policy_node.py` against the live `marv_flipper_eval`
Gazebo sim (a fresh smoke-trained C-TRAC checkpoint) surfaced FOUR real, pre-existing bugs
no `Env`-based test could have caught, plus one non-crashing gap — all fixed, then
re-verified: (1) `Heightmap.from_realistic_world` assumed an unbatched 2-D input, but
`infer_action()`'s calling convention adds a batch dim, producing a 5-D tensor into
`grid_sample` and crashing it; (2) the same method's return value was erroneously squeezed
one dimension short of what `get_spec()`/`HeightmapEncoder` expect, surfacing downstream as
an opaque matmul-shape `RuntimeError`; (3) `_FrameHistoryBuffer` crashed on a
`NonTensorData` placeholder that `TensorDictPrimer` leaves at `HISTORY_KEY` when its proper
zero-fill (computed only on `env.reset()`) never runs — because `infer_action()` never calls
`reset()` — fixed defensively (treat anything non-`Tensor` as "not primed"); the identical
`TensorDictPrimer` pattern in `state_machine_policy`'s `RECURRENT_KEY` would hit the same
issue, not verified there, flagged for its owner; (4) `policy_inference_module.py` could not
load ANY C-TRAC config at all (`PPOExperimentConfig(**raw_config)` rejects C-TRAC's
trainer-only top-level keys like `gamma`/`n_iters`) — meaning the "deploy exactly like every
other baseline" claim was false for every C-TRAC checkpoint, this round's or any earlier
one's, until fixed (the same field-filter `experiments/ctrac/train.py` already uses, applied
once more on the deploy side; a no-op for every other baseline). Plus: (5)
`experiments/ctrac/train.py` never saved `vecnorm_final.pth` (unlike `experiments/ppo/train.py`)
— fixed to match. Re-verified after all five fixes: `flipper_policy_node.py` against the live
sim, goal published, 28 consecutive 10 Hz control ticks, zero errors, correct elevation-map
ingestion, well-formed `/cmd_vel` + `/flippers_cmd_vel/*` commands throughout. Full detail:
`ctrac_NOTE.md`'s new "Bugs found and fixed this round" section.

The one item deliberately NOT touched: Stage-1 C-VAE pretraining still bootstraps off
random-init-policy rollouts rather than the paper's expert trajectories — the task's own
instruction was to keep this one, so it is kept, and remains disclosed as such.

**Independent follow-up pass (2026-07-10, separate agent instance from the
"2026-07-10 re-verification pass" above, same scope: azayev/mitriakov + Mitriakov
terrain + SAC/TD3/DDPG + this table).** Rather than accept the prior pass's writeup
at face value, re-derived every claim in its scope from scratch:
* **azayev.yaml / mitriakov.yaml re-run a third and fourth time, fresh, both
  invocation styles**: `-m flipper_training.experiments.ppo.train --local ...` (per
  CLAUDE.md) for both configs (`runs/ppo/azayev_independent_reverify_2026-07-10_19-22-49_275861`,
  `runs/ppo/mitriakov_independent_reverify_2026-07-10_19-23-22_276210`), and the
  plain-script form for azayev (`runs/ppo/azayev_readme_doc_verify_2026-07-10_19-30-04_279840`).
  All three: exit code 0, real `eval/mean_step_reward` etc. output, no errors/NaNs in
  the full log (grepped for `error|traceback|nan|exception`, only hit was an unrelated
  `sentry_sdk` deprecation warning from a dependency). Confirms the checked-in `device:
  cpu` still works and both invocation styles genuinely both work, right now, not just
  in the prior pass's session.
* **Field-drift check redone independently, by reading the dataclasses, not by
  re-trusting the header comments**: `PPOExperimentConfig` (`experiments/ppo/config.py`)
  against every top-level key in both files; `BarrierHeightmapGenerator`/`BarrierCrossing`
  against azayev's `heightmap_gen_opts`/`objective_opts`; `StairsHeightmapGenerator`/
  `StairCrossing` against mitriakov's; `PotentialGoalWithPenaltiesConfigurable` against
  both `reward_opts`; `StateMachinePolicyConfig`/`MLPPolicyConfig` against both
  `policy_opts`; `marv_rl_training.observations.heightmap.Heightmap` (imported and
  `dataclasses.fields()`-inspected directly in `flipper_venv`) against mitriakov's second
  observation block. Every field name matches exactly; no drift found.
* **Mitriakov paper mapping independently re-read from the PDF itself** (not from the
  config header): `Read` on `Mitriakov_2021.pdf` pages 1-8 (Problem Description +
  Reward Function Design + Fig. 1/2). Confirms, against the actual typeset text/figures:
  Eq. 2's 9-D state and the "pitch deliberately excluded... (2) already indirectly
  enables the capture of that parameter" reasoning; Eq. 1's 5-DoF action and the yaw-lock
  reasoning; Eq. 3/4's `R_t = x_t/D_max + r_t` split; Eq. 5's ascent COG-deviation
  penalty (`D_t = sqrt(d^2+h^2)`, explicitly framed by the paper itself as jointly
  optimizing "the SM and the NESM" rather than computing literal NESM) and Eq. 6's
  descent pitch-velocity penalty; the pi/2 tip-over threshold; Fig. 1's `Dense(64,tanh)
  x2` architecture; and the paper's own stated PPO-over-SAC/TD3 reasoning ("more
  straightforward to implement, less sensitive to hyperparameter changes"). One nuance
  the header doesn't flag: Eq. 6 is typeset as `-K_W * W_t` (no explicit absolute value),
  which if taken completely literally would reward one sign of pitch rate — the
  surrounding prose ("mitigate such events" / shaking) and this repo's actual
  implementation (`pitch_rate.abs()`) both make clear the intended quantity is
  `|W_t|`; noted here in case the ambiguity matters to someone reading the equation
  typography literally.
* **Mitriakov terrain-registry claim independently re-derived by grep, not re-asserted**:
  `grep -rn "TERRAIN_REGISTRY\|terrain_registry" flipper_training/` returns nothing; the
  only `terrain: <name>` string-keyed dispatch in this whole repo
  (`_TERRAIN_MAP`/`OmegaConf.select(cfg, "terrain", ...)`) lives in
  `marv_rl_training/ppo/train_ftr_compat.py`, which consumes a different dataclass
  (`FtrPPOConfig`, with its own `terrain: str` field) than `PPOExperimentConfig`. This
  independently confirms the prior pass's conclusion that "register a named staircase
  terrain" doesn't apply to the trainer this baseline actually uses.
* **OFFPOLICY_NOTE.md's claims independently re-run, not re-read**: `grep -rn
  "SACLoss\|TD3Loss\|DDPGLoss" --include=*.py .` again finds only `ctrac_policy.py`
  (docstrings) and `experiments/ctrac/train.py` (the actual `SACLoss(...)` call) as real
  instantiations; `offpolicy_ac_policy.py`'s mentions are docstring-only. `import omni`
  in `flipper_venv` still raises `ModuleNotFoundError`. `sed -n '1,20p'
  src/FTR-Benchmark/scripts/ftr_algo/train.py` confirms `from omni.isaac.lab.app import
  AppLauncher` really is line 17, at module scope, before argument parsing — matches the
  note's citation exactly. No `experiments/sac|td3|ddpg/` directory exists. Conclusion
  unchanged: no fake `configs/baselines/{sac,td3,ddpg}.yaml` added.
* **Two stale/incorrect pieces of documentation found in this file and fixed** (neither
  introduced by any baselines pass — both predate `azayev.yaml`/`mitriakov.yaml` and
  trace to the original `README_baselines.md` skeleton, confirmed via `git show
  HEAD:.../README_baselines.md`): (1) the opening paragraph cited a `baselines_todo.md`
  that has never existed in this repo's git history; (2) the "Validate" section's first
  command referenced a `test_baseline_policies.py` that has likewise never existed (no
  mock-`Env` helper exists anywhere either), and its second command's
  `../../../../../flipper_venv/bin/python` resolved to `MARV_RL/flipper_venv/bin/python`
  — reproduced the literal `bash: ... No such file or directory` before fixing;
  `flipper_venv` actually lives at `/home/cnuc/marv_ws/flipper_venv`, two directories
  above `MARV_RL` root, not inside it. Both fixed in place with real, just-executed
  commands rather than deleted outright, so "Validate" still has a working recipe.
* **Also separately confirmed accurate and left untouched** (adjacent facts checked
  because they were cheap to check, not because they were in doubt): `MLP`'s default
  `activation` really is `torch.nn.Tanh` (`policies/__init__.py`); a repo-wide grep for
  `nesm|support.polygon|cog.*project` under `rl_rewards/`/`rl_objectives/` is empty;
  `PotentialGoalWithStepAscentBonus` genuinely has no roll/pitch fields; the produced
  `runs/ppo/*/weights/policy_final.pth` + `vecnorm_final.pth` files are real, non-empty,
  and owned `cnuc:cnuc`.
* **Not re-touched, explicitly out of this pass's scope, same as the pass before it**:
  the AT-D3QN / ICM-D3QN / Pecka / C-TRAC "Status notes" paragraphs above (each written
  and owned by its own concurrent agent) — only their **file existence** and the
  **`device:` line** of each (a one-line, objectively grep-checkable fact, not a claim
  about their algorithmic fidelity) were re-checked, both here and below.

**Fresh file-existence + `device:` check, right before finishing this pass** (supersedes
the `ls`/device notes above if any more time has passed since — always prefer re-running
this over trusting either writeup):

```
$ ls configs/baselines/
at_d3qn_full.yaml  azayev.yaml  ctrac_full.yaml  ctrac_NOTE.md  icm_d3qn_full.yaml
mitriakov.yaml  OFFPOLICY_NOTE.md  pecka_full.yaml

$ grep -H '^device:' configs/baselines/*.yaml
at_d3qn_full.yaml:device: cuda        # unchanged, not this baseline's scope
azayev.yaml:device: cpu               # this pass's scope, re-verified working
ctrac_full.yaml:device: cpu           # unchanged, not this baseline's scope
icm_d3qn_full.yaml:device: cuda       # unchanged, not this baseline's scope
mitriakov.yaml:device: cpu            # this pass's scope, re-verified working
pecka_full.yaml:device: cpu           # unchanged since the Pecka agent's own fix above,
                                       # not this baseline's scope
```

All eight files exist; nothing is `pending`. This also means the "2026-07-10
re-verification pass" bullet above that says `pecka_full.yaml` "still say[s] `device:
cuda`/`cuda:0`" is now stale (the Pecka-owning agent fixed it in a later, separate edit
— see its own "Independent re-verification" paragraph and the "UPDATE" note in
"Verified file existence" above); left in place rather than rewritten, per this file's
own convention of appending corrections instead of editing other agents' historical
paragraphs.

**2026-07-12 pass (yet another separate agent instance, identical scope: verify+fix
azayev.yaml/mitriakov.yaml against `experiments/ppo/train.py`, tighten Mitriakov's
staircase terrain, add SAC/TD3/DDPG configs or explain why not, refresh this table —
explicitly not AT-D3QN/ICM-D3QN/Pecka/C-TRAC).** Worth being explicit about method here:
every check below was run and its output read BEFORE this pass opened this file and read
the two "2026-07-10" passes above — so the fact that it lands on the same conclusions
(down to independently flagging the identical Eq. 6 `|W_t|` typography nuance) is
genuine convergent verification, not this pass copying the prior write-ups. Nothing in
this scope needed fixing; everything below is confirmation, plus the one freshness delta
already folded into the "UPDATE (2026-07-12...)" note above.
* **Both configs run fresh, a further time, in new run directories**
  (`runs/ppo/azayev_freshcheck_indep_1783865129_2026-07-12_16-05-37_689012`,
  `runs/ppo/mitriakov_freshcheck_indep_1783865158_2026-07-12_16-06-06_689201`), via the
  `-m flipper_training.experiments.ppo.train --local ...` form, checked-in `device: cpu`
  untouched (no override). Both exit 0; mitriakov's "Environment Summary" table logs
  `Observations: LocalStateVector, Heightmap` / `Objective: StairCrossing` /
  `Reward: PotentialGoalWithPenaltiesConfigurable`, i.e. the staircase wiring is what
  actually constructs, not just what the YAML says.
* **Field-drift re-checked by reading source, not by re-trusting either prior pass**:
  `PPOExperimentConfig` (`experiments/ppo/config.py`) against every top-level key in both
  files; `StairsHeightmapGenerator`/`StairCrossing` (`heightmaps/stairs.py`,
  `rl_objectives/stair_crossing.py`) against mitriakov's `heightmap_gen_opts`/
  `objective_opts`, including confirming `heightmap_gen_opts` is byte-identical to the
  proven `final_training_configs/stairs_from_scratch_extra_reward.yaml`;
  `marv_rl_training/observations/heightmap.py`'s `Heightmap` against the config's second
  observation block; `LocalStateVector` (`observations/robot_state.py`) index order
  (roll, pitch, xd[3], omega[3], thetas[num_driving_parts], goal[3]) against
  `robots/marv.yaml`'s 4 `driving_parts` entries, confirming azayev's
  `flipper_angle_idx: [8,9,10,11]`/`roll_idx: 0`/`roll_rate_idx: 5`/`vx_idx: 2` are the
  correct offsets, not just plausible-looking numbers; `PotentialGoalWithPenaltiesConfigurable`
  (`rl_rewards/rewards.py`) against both `reward_opts`. No drift found anywhere.
* **sm_61 claim independently re-derived from hardware, not re-quoted**: `nvidia-smi
  --query-gpu=name,compute_cap` reports the GTX 1050 at `6.1`; `flipper_venv`'s
  `torch.cuda.get_arch_list()` returns `['sm_70', 'sm_75', ..., 'sm_120']` — sm_61 is
  genuinely absent, confirming `device: cpu` is a real hardware-forced fix, not a config
  author's guess.
* **Mitriakov paper mapping re-read from the primary source a third time**
  (`Read` on `Mitriakov_2021.pdf` pages 1-8, before opening this README): Eq. 1's 5-DoF
  action + explicit yaw exclusion, Eq. 2's 9-D state with pitch deliberately excluded,
  Eq. 3/4's `R_t = x_t/D_max + r_t`, Eq. 5's ascent COG/support-polygon-deviation penalty
  (the paper's own words: COG tip-over "only when its projection point Cx crosses the
  lowermost edge of the robot support polygon"), Eq. 6's descent pitch-velocity penalty
  (typeset without explicit `|W_t|` bars, though "mitigate such events"/shaking framing
  and this repo's actual `pitch_rate.abs()` both point at the intended magnitude reading),
  the `|pitch| > pi/2` tip-over threshold, Fig. 1's `Dense(64,tanh) x2` network, and the
  paper's own stated PPO-over-SAC/TD3 rationale — all independently confirmed to match
  `mitriakov.yaml`'s header comment, line for line.
* **Terrain-registry claim re-derived by grep, scoped slightly wider than the prior
  pass** (whole `src/flipper_training` tree, not just `flipper_training/`):
  `grep -rniE "terrain_registry|TERRAIN_MAP|register.*terrain"` finds exactly one hit,
  `marv_rl_training/ppo/train_ftr_compat.py`'s `_TERRAIN_MAP`, gated behind that file's
  own `FtrPPOConfig` (not `PPOExperimentConfig`). A separate grep for `^terrain:` across
  every config directory finds it in ~39 files, all under `configs/` at the MARV_RL root,
  ZERO under `test_configs/`/`final_training_configs/`/`sota_configs/` (the
  `PPOExperimentConfig`-schema directories azayev/mitriakov's own configs belong to).
  Checked the `configs/` hits aren't secretly native despite non-`ftr_config_*` names
  (`teleop_marv.yaml`, and — worth flagging since the name is actively misleading —
  `marv_native_coldstart.yaml`): every single one also carries `task: Ftr-Crossing-Direct-v0`
  and/or `env_cfg_overrides:`, the FTR-compat schema's own signature keys, confirming
  they're FTR-side despite filenames that could suggest otherwise. Confirms, a third
  independent way, that "register a named staircase terrain" doesn't apply to the trainer
  `azayev.yaml`/`mitriakov.yaml` use.
* **OFFPOLICY_NOTE.md re-checked, not re-read**: `grep -rl "SACLoss\|TD3Loss\|DDPGLoss"
  flipper_training/` still returns only `ctrac_policy.py`, `offpolicy_ac_policy.py`
  (docstring only), `README_baselines.md` (prose only) and `experiments/ctrac/train.py`
  (the one real instantiation); `flipper_venv/bin/python -c "import omni.isaac.lab"` still
  raises `ModuleNotFoundError: No module named 'omni'`; `sed -n '1,25p'
  src/FTR-Benchmark/scripts/ftr_algo/train.py` still shows `from omni.isaac.lab.app import
  AppLauncher` at line 17, before `argparse` even runs. No `experiments/sac|td3|ddpg/`
  directory exists. Conclusion unchanged: no `configs/baselines/{sac,td3,ddpg}.yaml` added.
* **File existence + `device:` re-checked at the end of this pass, same as the
  convention established above** — see the "UPDATE (2026-07-12...)" note earlier in this
  section for the one delta found (`at_d3qn_full.yaml`/`icm_d3qn_full.yaml` now also
  `device: cpu`, someone else's fix, not this pass's). All eight files in
  `configs/baselines/` still present; still nothing `pending`.
* **Net effect on `azayev.yaml`/`mitriakov.yaml` this pass**: none — both files are
  byte-identical to `HEAD` (`git status` shows no modification to either), because every
  independent check above reproduced the prior passes' findings exactly rather than
  surfacing anything to fix. This paragraph is itself the "fix, or confirm none needed"
  the task asked for.
