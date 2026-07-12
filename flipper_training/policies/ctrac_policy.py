"""C-TRAC: Terrain-Adaptive Control via Contact-Aware RL (Pan et al., IROS 2025).

Contact-modelling variational autoencoder (C-VAE) + asymmetric Soft Actor-Critic,
built with MARV_RL's stack (TorchRL operators, ``tensordict``, shared
``MLP``/``EncoderCombiner``) so it trains with ``torchrl.objectives.SACLoss`` via
``experiments/ctrac/train.py`` and deploys through the generic
``flipper_policy_node`` (``get_policy_operator()(td)["action"]``).

What is actually built (matches the paper Sec. IV; every option below defaults to
OFF/single-frame/8-D so the default construction is BYTE-IDENTICAL to before these
options existed — pass ``configs/baselines/ctrac_full.yaml`` to get the paper's
full architecture with everything turned on):

* **Contact-conditioned actor** — the paper's ``pi(a_t | o_t, z_t, c~_t, c~_t^prob)``
  (Sec. IV-A.3). The policy operator is a ``TensorDictSequential``:

      obs keys --[shared encoder psi]--> "y_enc" (current frame o_t)
      "y_enc" + carried history --[ring buffer]--> "y_enc_hist" (o_t^H, Eq. 9)
      "y_enc_hist" --[C-VAE trunk+heads]--> "contact_latent" z, "contact_est" c~ (3A),
                                             "contact_prob" c~^prob (A, sigmoid)
      ["y_enc", z, c~, c~^prob] --[Gaussian head]--> loc/scale --TanhNormal--> "action"

  Everything runs inside ``get_policy_operator()``; the actor consumes NO
  privileged inputs, so the deploy contract holds unchanged (EXCEPT in the
  ``effective_action_5d`` mode — see that section below, the one deliberate,
  documented exception). ``z`` is sampled with the reparameterisation trick in
  ``train()`` mode and equals the posterior mean in ``eval()`` mode
  (deterministic deployment).

* **ONE shared observation encoder** between the actor and the C-VAE: the
  ``EncoderCombiner`` instance appears once in the policy operator and its
  ``"y_enc"`` output is consumed by both the history ring buffer (below) and the
  actor head directly. Both the SAC actor loss and the C-VAE hybrid loss
  backpropagate into it (joint optimisation) — this is precisely the fix for the
  audited bug (each component previously had its own ``deepcopy`` encoder, so
  gradients from contact learning never reached the policy). The critics keep
  their own independent encoders.

  Honest note on paper fidelity here: Fig. 2 depicts the actor consuming the
  raw current observation o_t directly through its own small dedicated MLP,
  separate from the C-VAE's encoder (which runs on the history buffer o_t^H).
  The paper never states the two share weights. Every policy in this
  repo's ``policies/`` package routes observations through an
  ``EncoderCombiner`` before any head, so unifying that encoder between the
  actor and the C-VAE is this codebase's convention applied to close the
  dangling-gradient bug, not a literal reproduction of Fig. 2's wiring.

* **Multi-frame history o_t^H** (Sec. III-B / Fig. 2's "History Obs" -> Encoder
  path, Eq. 9): a parameter-free ring-buffer module, ``_FrameHistoryBuffer``,
  sits between the shared encoder and the C-VAE trunk. It stacks the last
  ``history_len`` values of "y_enc" (NOT the raw per-frame observations — see
  below) into "y_enc_hist" (dim ``enc_dim * history_len``), which becomes the
  C-VAE trunk's input instead of the single-frame "y_enc". The actor head is
  UNAFFECTED — it still reads "y_enc" (the current frame only), matching Fig. 2's
  separate "Current Obs o_t -> Actor" arrow.

  Two documented design choices:
  - **Buffering "y_enc" (the shared encoder's OUTPUT) rather than re-encoding H
    raw historical observation frames each step.** Mathematically equivalent
    within a rollout (the encoder's weights are fixed between gradient updates,
    so "encode frame t-k now" == "encode frame t-k when it was collected, and
    remember the result") but far cheaper (O(1) update per step instead of
    O(history_len) re-encodes), and it is what naturally falls out of carrying
    state the same way GRUModule/state_machine's recurrent state already do in
    this repo.
  - **A self-contained ring-buffer module living INSIDE the policy operator**
    (carried via ``("next", HISTORY_KEY)``, primed with ``TensorDictPrimer`` +
    ``InitTracker()``, with an ``is_init``/missing-buffer reset exactly like
    ``state_machine_policy._HFCActorModule``'s recurrent ``p``) rather than a
    ``torchrl.envs.transforms.CatFrames``-style ENV-side transform. A CatFrames
    env transform only stacks whatever the wrapped env happens to be handed to
    it; the generic deployment node (``flipper_policy_node`` /
    ``policy_inference_module.py``) calls ``get_policy_operator()(td)["action"]``
    directly on a bare, transform-free tensordict (verified: it never routes the
    actor call through the training env's transform ``Compose``), so a CatFrames
    env transform would silently do nothing at deploy time and the actor would
    need its OWN fallback logic anyway — putting the ring buffer inside the
    policy operator from the start means there is only ONE code path, and a bare
    call degrades gracefully to "history = current frame repeated"
    (``history_len=1``-equivalent), the same graceful-missing-state behaviour
    ``state_machine_policy`` already documents for its own recurrent carry.

  Concretely, not hypothetically: traced ``policy_inference_module.py``'s
  ``infer_action()`` (what ``ros2/flipper_policy_node.py``'s
  ``control_callback`` calls every control tick) end to end. It builds a
  FRESH tensordict from raw kwargs each call, does ``env_td =
  self.env.step(world_td)``, then a bare ``self.actor_operator(env_td["next"])``
  — nothing captures ``env_td["next", HISTORY_KEY]`` (or GRU/LSTM/
  ``state_machine``'s own ``RECURRENT_KEY``) back into the NEXT tick's
  ``world_td``. So today, through THIS specific node, multi-frame history
  does not actually accumulate across real control ticks — every tick hits
  the bare-call fallback above, i.e. live-deployed behaviour is
  ``history_len=1``-equivalent regardless of the configured ``history_len``,
  identically to how GRU/LSTM/``state_machine``'s own recurrent carry already
  behaves through this same node (this is a pre-existing property of
  ``infer_action()``'s per-tick calling convention, not introduced here, and
  it affects every recurrent policy in this repo the same way, not just
  C-TRAC). History DOES accumulate for real wherever a caller threads
  ``next -> current`` across steps — every ``SyncDataCollector`` /
  ``Env.rollout()`` usage does this automatically (training, this trainer's
  own eval rollouts, ``check_env_specs``), so the C-VAE genuinely trains and
  evaluates on real multi-frame windows; only this one ROS inference
  wrapper's per-tick convention is memoryless. Fixing that wrapper (for every
  recurrent policy in this repo, not just C-TRAC) is out of scope here.

  One level deeper, live-sim-verified: that memorylessness was supposed to be
  HARMLESS (degrade to ``history_len=1``-equivalent, per ``_FrameHistoryBuffer``'s
  own docstring below) but, before this pass, actually CRASHED instead — a real bug,
  not just a fidelity gap, only reachable by running the actual ROS node (a plain
  unit test that calls the actor operator directly on a truly bare tensordict, with
  no ``TensorDictPrimer``/``InitTracker`` in the picture at all, does NOT reproduce
  this: that is an easier case than what ``infer_action()`` actually produces).
  ``TensorDictPrimer``'s zero-fill only ever gets computed by its reset-time code
  path; ``infer_action()`` never calls ``env.reset()``, so every tick instead hits
  the primer's weaker step-time path, which left a non-tensor placeholder at
  ``HISTORY_KEY`` that crashed the ring buffer's ``buf.abs()`` check. Fixed
  defensively in ``_FrameHistoryBuffer.forward`` itself (treat anything that isn't
  literally a ``torch.Tensor`` as "not primed", not just ``None``) rather than by
  fighting ``TensorDictPrimer`` internals — see that class's docstring for the full
  trace. Re-verified end to end after the fix: `flipper_policy_node.py` against the
  live Gazebo sim, 28 consecutive control ticks, zero errors, well-formed published
  `/cmd_vel` + `/flippers_cmd_vel/*` commands throughout.

  ``history_len=1`` (the default) is an exact identity — the ring buffer always
  holds exactly the current frame, so "y_enc_hist" == "y_enc" bit-for-bit and
  training reduces exactly to this file's previous single-frame behaviour. This
  is a genuine paper-fidelity gap worth stating precisely: **the paper defines
  o_t^H = [o_t, ..., o_{t-H}] symbolically (Sec. III-B) but never gives a numeric
  value for H anywhere in the text** (checked Sec. III-B, IV-C, and the Sec. V-A.1
  hyperparameter list — none state it). ``configs/baselines/ctrac_full.yaml`` sets
  ``history_len: 5`` as a documented, reasonable placeholder — NOT a paper-derived
  constant; do not cite it as such.

* **Clean denoising target** (Sec. IV-C: "a denoising autoencoder mechanism to
  learn robot-terrain interaction dynamics from noisy multi-frame interaction
  data"; Sec. V-A.3 domain randomization adds Gaussian noise to the
  observations). A real denoising objective needs a NOISY input and a CLEAN
  reconstruction target — this repo's per-observation ``apply_noise`` already
  gives the noisy input, but by itself gives the C-VAE's reconstruction head no
  clean target to denoise TOWARD (it would just be reconstructing whatever the
  env emits, noisy or not — plain autoencoding, not denoising). Fixed by
  ``Env.emit_clean_observations`` (``environment/env.py`` /
  ``experiments/ppo/config.py``'s ``emit_clean_observations`` field): when set,
  the env additionally computes a parallel NOISELESS copy of every observation
  each step under the nested key ``("clean", <observation name>)`` (aliasing the
  already-computed value — no extra compute — for observations that were not
  noised to begin with). ``experiments/ctrac/train.py``'s ``_vae_update`` uses
  ``batch["next", "clean"]`` as the C-VAE reconstruction target when present,
  falling back to plain ``batch["next"]`` (today's behaviour) when absent — so
  with ``emit_clean_observations=False`` (the default) or with no observation
  actually noised, this degenerates EXACTLY to the previous behaviour (noisy
  target when ``apply_noise`` is set on the actor observations, plain next-obs
  reconstruction otherwise).

* **Asymmetric privileged twin-Q critic** — ``Q(o_t, c_t, h_t^f, a_t)``
  (Sec. IV-A Eq. 2): the Q network TEMPLATE (``get_qvalue_operator()``) consumes
  ALL environment observations, including the privileged ones listed in
  ``privileged_observations`` (by default just the ``GroundTruthContacts``
  observation = ``flipper_training.observations.contacts.GroundTruthContacts``,
  which reads per-flipper ground-truth contact positions + flags straight from
  the physics engine's ``PhysicsStateDer``), plus the action, through its own
  independent encoder (never the actor/C-VAE's shared one). The critic is
  SINGLE-FRAME (Eq. 2 / Fig. 2 feed the privileged obs to the critic directly,
  with no history-encoder box on that path — unlike the C-VAE's o_t^H input, see
  above), so it is unaffected by ``history_len``. The actor and C-VAE never see
  the privileged keys. If no configured privileged observation exists in the
  env, ``create`` falls back to a SYMMETRIC ``Q(o_t, a_t)`` critic and logs a
  loud warning — it never silently claims asymmetry (the C-TRAC trainer refuses
  to run without the ground-truth contact observation anyway, since it also
  provides the C-VAE targets).

  ONE template, not two: ``twin_q`` sets ``wrapper.num_qvalue_nets`` (2 if
  ``True`` else 1) for ``SACLoss(..., num_qvalue_nets=wrapper.num_qvalue_nets)``
  to expand internally, rather than this module building two separate
  ``nn.Module`` instances itself. This is a version-compatibility fix, not a
  style choice: the installed ``torchrl`` (0.8.1) ``SACLoss._set_in_keys``
  unconditionally does ``self.qvalue_network.in_keys``, which crashes on a
  plain ``list``/``tuple`` of separately-built modules — passing several
  independent module instances (the natural first attempt, and what this file
  did until this fix) is not supported end-to-end by this version despite the
  docstring suggesting otherwise. torchrl's ``num_qvalue_nets`` expansion
  reinitialises ``N`` independent parameter copies **disconnected from the
  template's own parameter identity** (verified empirically — the template's
  ``.grad`` stays ``None`` and its weights never change); the template's
  ARCHITECTURE (independent encoder over ``critic_obs``) is what's preserved
  and expanded, not its specific init values. See
  ``experiments/ctrac/train.py``'s module docstring for exactly where the
  trainable copies end up (``sac_loss.qvalue_network_params``) and why
  ``wrapper.state_dict()``'s Q-net weights are consequently NOT the trained
  ones (training-only; never read by the deployed actor).

  **Wide privileged heightmap h_t^f** (Eq. 2): list
  ``flipper_training.observations.heightmap.PrivilegedHeightmap`` in
  ``privileged_observations`` (alongside an ordinary
  ``flipper_training.observations.heightmap.Heightmap`` entry, NOT listed as
  privileged, for the actor's own ``h_t^l``, Eq. 1) and it is routed to the
  critic like any other privileged observation — no code change needed here,
  the mechanism was always generic over whatever names you configure; only the
  heightmap Observation classes themselves (``observations/heightmap.py``) were
  missing from the canonical tree until now. ``configs/baselines/ctrac_full.yaml``
  wires both by default.

* **C-VAE** (Sec. IV-C, Eq. 9-14) — encoder trunk ``y_enc_hist -> (mu, logvar)``
  and a multi-head decoder from ``z``:
  contact positions (``3 * n_areas``), contact-probability logits
  (``n_areas``), and a next-observation reconstruction head (the paper's
  denoising head for ``o~_{t+1}``). The hybrid loss (``ContactVAE.loss``):

      L = w_recon * MSE(o^_{t+1}, o_{t+1}_CLEAN)       # VAE reconstruction (denoising)
        + beta   * KL(q(z|o_t^H) || N(0, I))           # beta-VAE ELBO term
        + w_prob * BCE(prob logits, c^prob_gt)         # Eq. 12
        + w_est  * sum_i m_i * MSE(c~_i, c_i)          # Eq. 13, dynamic mask
        + w_geo  * sum_i m_i * relu(||c~_i|| - reach)  # Eq. 14, distilled

  with the dynamic mask ``m_i = c_i^prob / sum_j c_j^prob`` (zero when no area
  is in contact). Ground-truth targets ``c_i``/``c_i^prob`` come from the
  ``GroundTruthContacts`` observation stored in the rollout; ``o_{t+1}_CLEAN``
  is ``batch["next", "clean"]`` when ``emit_clean_observations`` is on (see
  above), else plain ``batch["next"]`` (today's behaviour).

  Remaining deliberate simplification (documented, config-tunable):
  - **Geometric feasibility region** ``Omega`` (Eq. 14) is distilled from "on
    the robot's articulated body" to an origin-centred reach sphere: radius =
    ``max(body point norms, max_part(||joint_position|| + max local-point
    norm))`` (derived from ``env.robot_cfg``; override with
    ``contact_reach_m``). By the triangle inequality this is a safe UPPER
    BOUND on the true max reach of any collision point over all flipper
    angles for a single-axis revolute joint (exact only when the joint's
    rotation plane happens to contain the origin) — it never falsely
    penalises a genuinely-reachable contact estimate, though it can be
    slightly generous. The penalty is the Euclidean distance outside that
    sphere, 0 inside — exactly the paper's ``I(c, Omega)`` for the spherical
    ``Omega``.

* **5-D effective action option** (Eq. 3: ``a_t = [v_t, dtheta_fl, dtheta_rl,
  dtheta_rr, dtheta_fr]``, OFF by default — the env-native 8-D
  ``[4 track velocities, 4 flipper velocities]`` remains the default action
  space "for cross-baseline comparability" across this repo's other baselines).
  Set ``effective_action_5d=True`` to make the actor's OWN decision variable
  genuinely 5-dimensional: one shared desired chassis speed ``v`` (broadcast to
  all track slots — Eq. 3 has no per-track differential term, matching the
  paper's "straight-path traversal" framing) plus one delta-angle per flipper,
  converted to the angular-velocity command the engine actually integrates via
  ``theta_dot_i = dtheta_i / env.effective_dt`` ("the constant angular velocity
  that realises this one-step angle delta"). The flipper ORDER follows this
  repo's native convention ``[FL, FR, RL, RR]`` (``env.robot_cfg`` /
  ``env.action_spec``'s own layout), not the paper's listing order
  ``[fl, rl, rr, fr]`` in Eq. 3 — both name the same 4 quantities (one delta per
  flipper); only the presentation order differs, which carries no semantic
  weight (it is not a track/kinematic-chain ordering, just how the paper's
  authors chose to write the vector).

  **Why this MUST be an env-side transform, not a policy-operator submodule**
  (unlike the history ring buffer above), AND why the critic's action input
  tracks the actor's width instead of staying env-native — both driven by the
  same fact: torchrl 0.8.1's ``SACLoss`` computes both the actor loss and the
  SAC target value via ``dist = self.actor_network.get_dist(td); a =
  dist.rsample()`` and feeds ``a`` STRAIGHT to ``self.qvalue_network``
  (verified against the installed ``torchrl/objectives/sac.py``'s
  ``_actor_loss``/``_compute_target_v2`` — and reproduced empirically: an
  earlier revision of this file built the critic at the env-native width
  unconditionally and hit exactly this crash, ``RuntimeError`` out of the
  vmapped ``_ConcatQ`` linear layer, the moment ``effective_action_5d=True``
  reached its first SAC update). It NEVER re-runs the actor's ``forward()``
  after sampling, so:

  1. A deterministic "expand 5-D to 8-D" module placed after the
     ``ProbabilisticActor`` inside the same ``TensorDictSequential`` would be
     silently skipped on exactly those two code paths (``dist.rsample()`` is
     read directly off the actor's distribution, not off anything downstream
     of it) — ruling out a policy-operator submodule as the fix.
  2. Consequently, ``self.qvalue_network`` MUST accept whatever width
     ``dist.rsample()`` produces — the actor's OWN ``actor_adim`` (5 when this
     option is on, 8 otherwise) — on every SAC code path (``_actor_loss``,
     ``_compute_target_v2``, and the plain ``_qvalue_v2_loss`` update, which
     reads the REPLAY-STORED "action"). So ``get_qvalue_operator()``'s
     ``_ConcatQ`` is built with ``actor_adim``, not ``env_adim`` — this is also
     the MORE paper-faithful choice independent of the torchrl constraint:
     Sec. IV-A.3 defines the critic as ``Q_phi(s_t, a_t)`` using the SAME
     ``a_t`` as Eq. 3 (5-D in the paper), never an 8-D physical expansion of
     it.
  3. The physics engine still needs a valid 8-D command every step, and that
     translation has to happen SOMEWHERE — expanding INSIDE the actor operator
     is ruled out by point 1, so it happens OUTSIDE the actor entirely, as an
     env-side action transform, exactly the use case
     ``torchrl.envs.transforms.Transform``'s ``in_keys_inv``/``_inv_call``
     exists for. ``EffectiveActionTransform`` (below) does this:
     ``TransformedEnv._step`` calls ``transform.inv()`` on a CLONE of the
     incoming tensordict to build what is actually handed to the base env's
     ``_step`` (verified in ``torchrl/envs/transforms/transforms.py``) — the
     ORIGINAL tensordict (whose "action" is untouched, still 5-D) is what gets
     "next"-merged and returned, i.e. what the collector yields and the replay
     buffer stores. So collection, the replay buffer, the (now 5-D-native)
     critic, and every ``SACLoss`` code path all consistently see the 5-D
     action; only the (transparent, internal) physics step — which
     ``EffectiveActionTransform`` alone touches — receives the expanded 8-D
     command. Nothing "reconciles" an 8-D critic with a 5-D actor because
     there is no such mismatch to reconcile: the critic is 5-D-native in this
     mode, full stop, and the env-side transform's only job is feeding the
     physics engine, not bridging an actor/critic width gap.

  **Consequence for the deploy contract**: with ``effective_action_5d=True``,
  ``get_policy_operator()(td)["action"]`` returns the 5-D EFFECTIVE action, NOT
  the env-native 8-D one — a deliberate, DOCUMENTED exception to this repo's
  usual "the bare policy operator call alone produces an env-shaped action"
  contract, forced by the ``SACLoss`` constraint above (there is no way to
  satisfy both "actor distribution is genuinely 5-D for correct SAC training"
  and "a bare policy-operator call emits an 8-D action" simultaneously under
  this torchrl version). A caller that wants to execute the result directly
  outside the training env (e.g. a future ROS integration for this specific
  mode) must apply the SAME expansion —
  ``EffectiveActionTransform.expand(action_5d, effective_dt, num_driving_parts,
  action_low, action_high)`` is a plain stateless staticmethod, no env
  required. This mode is OFF by default; with it off (the default), the actor
  IS the env-native 8-D ``ProbabilisticActor`` exactly as before, and none of
  this applies — ``get_policy_operator()(td)["action"]`` matches
  ``env.action_spec`` unconditionally, as for every other baseline in this repo.

* **LeakyReLU** (Sec. V-A.1: "All neural networks employ LeakyReLU activation
  functions"), while this repo's shared ``MLP`` (and the heightmap CNN encoder,
  ``observations/heightmap.py``) default to Tanh/ReLU respectively. To match the
  paper, pass ``activation: ${cls:torch.nn.LeakyReLU}`` in ``actor_mlp_opts``/
  ``qvalue_mlp_opts``/``cvae_mlp_opts`` (shared ``MLP``) AND in each heightmap
  observation's ``encoder_opts`` (``HeightmapEncoder`` — a separate conv
  encoder, not the shared ``MLP``); ``configs/baselines/ctrac_full.yaml`` sets
  all of these. The default here deliberately keeps the repo-wide Tanh/ReLU
  convention so C-TRAC stays architecturally comparable to the sibling
  baselines when these options are left off.

* **Training** — ``experiments/ctrac/train.py``: stage 1 pretrains the C-VAE on
  rollouts of the randomly initialised policy, stage 2 alternates 1 SAC update
  with ``vae_updates_per_sac_update`` (default 5) C-VAE updates, per the
  paper's Fig. 2. Pairs with ``torchrl.objectives.SACLoss(actor_network=actor,
  qvalue_network=wrapper.get_qvalue_operator(),
  num_qvalue_nets=wrapper.num_qvalue_nets)``. Divergence kept by explicit user
  instruction (not resolved by this revision): the paper pretrains the C-VAE on
  "successful locomotion trajectories performed by terrain-specific ... flipper
  control policies" (expert data this repo has no equivalent of); this trainer
  uses rollouts of the random-init policy instead — see
  ``experiments/ctrac/train.py``'s module docstring and
  ``configs/baselines/ctrac_NOTE.md``.

Environment-level divergence from the paper shared by every baseline in this
repo, not introduced here: the reward terms of Sec. IV-B are the env config's
business, not this policy's (action space is no longer an environment-level-only
divergence — see the 5-D effective action option above).

``create(env)`` returns the usual ``(wrapper, optim_groups, transforms)``.
Optimizer groups are disjoint: ``actor`` (Gaussian head), ``qvalue`` (critics),
``contact_vae`` (shared encoder + C-VAE trunk/heads). Note the SAC actor loss
also deposits gradients on the shared encoder/C-VAE parameters; they are
applied with the ``contact_vae`` group's optimizer settings. The history ring
buffer and (when enabled) the 5-D action transform hold NO learnable
parameters, so they need no optimizer group of their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs.transforms import InitTracker, TensorDictPrimer, Transform
from torchrl.modules import ProbabilisticActor, TanhNormal, NormalParamExtractor

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

if TYPE_CHECKING:
    from tensordict import TensorDictBase

__all__ = ["CTRACConfig", "ContactVAE", "EffectiveActionTransform", "flatten_observations", "HISTORY_KEY"]

DEFAULT_CONTACT_OBS_NAME = "GroundTruthContacts"
HISTORY_KEY = "ctrac_frame_history"


def flatten_observations(td, keys: list[str]) -> torch.Tensor:
    """Flatten and concatenate raw observation tensors in canonical (env) key order.

    Used to build the C-VAE reconstruction target for ``o_{t+1}`` — pass the
    ``"next"`` (or ``"next", "clean"``, see the module docstring's "Clean
    denoising target" section) sub-tensordict and the wrapper's
    ``actor_obs_keys``.
    """
    return torch.cat([td[k].flatten(1) for k in keys], dim=-1)


class ContactVAE(nn.Module):
    """C-VAE trunk + multi-head decoder (paper Sec. IV-C).

    The observation-encoding body (obs -> ``y_enc``) and the multi-frame history
    ring buffer (``y_enc`` -> ``y_enc_hist``, Eq. 9's o_t^H) are the SHARED
    policy pipeline and live *outside* this module (registered once, in the
    actor operator); this module maps ``y_enc_hist`` to the latent and the
    decoder heads.

    Honest note on paper fidelity (decoder head count, found on a round-4 re-audit,
    pre-existing from an earlier pass, not touched by round 4's checklist): Sec. IV-C's
    text says the multi-head decoder's "first head estimates c~_t and c~_t^prob"
    (singular "first head" for both quantities) and "the second reconstructs and denoises
    the observation" -- i.e. TWO heads total, one shared between contact-position and
    contact-probability. This class instead gives contact position (``dec_contact``),
    contact probability (``dec_prob``), and denoising reconstruction (``dec_denoise``)
    THREE fully independent single-purpose MLPs (no shared sub-trunk between
    ``dec_contact``/``dec_prob``) — every output the paper describes is still produced and
    supervised by exactly the loss terms Eq. 12-14 specify, so this does not change what
    is learned or how it is scored, only whether position/probability share decoder
    parameters before their respective final layers. Not resolved here: changing it would
    touch the single-frame architecture this round was told to extend, not regress.

    ``forward(y_enc_hist) -> (z, contact_est, contact_prob)`` is the policy-path
    inference used inside the actor operator (reparameterised sample of ``z``
    in train mode, posterior mean in eval mode).

    ``loss(y_enc_hist, contact_target, next_obs_flat)`` is the hybrid training
    loss (Eq. 10-14); see the module docstring for the exact terms.
    ``contact_target`` must have the ``GroundTruthContacts`` layout
    ``[pos (3A), prob (A)]``; ``next_obs_flat`` should be the CLEAN next
    observation when available (module docstring's "Clean denoising target").
    """

    def __init__(
        self,
        in_dim: int,
        obs_recon_dim: int,
        n_areas: int,
        latent_dim: int,
        mlp_opts: dict,
        reach: float,
        beta: float = 1.0,
        w_recon: float = 1.0,
        w_prob: float = 1.0,
        w_est: float = 1.0,
        w_geo: float = 1.0,
    ):
        super().__init__()
        self.n_areas = n_areas
        self.latent_dim = latent_dim
        self.reach = float(reach)
        self.beta, self.w_recon, self.w_prob, self.w_est, self.w_geo = beta, w_recon, w_prob, w_est, w_geo
        self.trunk = MLP(in_dim=in_dim, out_dim=2 * latent_dim, **mlp_opts)
        self.dec_contact = MLP(in_dim=latent_dim, out_dim=3 * n_areas, **mlp_opts)
        self.dec_prob = MLP(in_dim=latent_dim, out_dim=n_areas, **mlp_opts)
        self.dec_denoise = MLP(in_dim=latent_dim, out_dim=obs_recon_dim, **mlp_opts)

    def encode(self, y_enc_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.trunk(y_enc_hist).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, y_enc_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy-path estimate: (z, contact positions, contact probabilities)."""
        mu, logvar = self.encode(y_enc_hist)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        contact_est = self.dec_contact(z)
        contact_prob = torch.sigmoid(self.dec_prob(z))
        return z, contact_est, contact_prob

    def loss(self, y_enc_hist: torch.Tensor, contact_target: torch.Tensor, next_obs_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Hybrid C-VAE loss (Eq. 10-14). Returns each term plus ``"loss_vae_total"``."""
        a = self.n_areas
        mu, logvar = self.encode(y_enc_hist)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        pos_target = contact_target[..., : 3 * a].view(-1, a, 3)
        prob_target = contact_target[..., 3 * a :]
        # beta-VAE terms: denoised next-obs reconstruction + KL
        recon = F.mse_loss(self.dec_denoise(z), next_obs_flat)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()
        # L_prob (Eq. 12): BCE on contact existence
        prob_logits = self.dec_prob(z)
        l_prob = F.binary_cross_entropy_with_logits(prob_logits, prob_target)
        # Dynamic mask (Eq. 13): m_i = c_i^prob / sum_j c_j^prob (all-zero rows -> zero mask)
        mask = prob_target / prob_target.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, A)
        # L_est (Eq. 13): mask-weighted MSE on contact positions
        contact_est = self.dec_contact(z).view(-1, a, 3)
        l_est = (mask * (contact_est - pos_target).pow(2).mean(dim=-1)).sum(dim=-1).mean()
        # L_geo (Eq. 14): distance outside the reach-sphere Omega, mask-weighted
        l_geo = (mask * (contact_est.norm(dim=-1) - self.reach).clamp_min(0.0)).sum(dim=-1).mean()
        total = self.w_recon * recon + self.beta * kl + self.w_prob * l_prob + self.w_est * l_est + self.w_geo * l_geo
        return {
            "loss_vae_total": total,
            "loss_vae_recon": recon,
            "loss_vae_kl": kl,
            "loss_contact_prob": l_prob,
            "loss_contact_est": l_est,
            "loss_contact_geo": l_geo,
        }


class _FrameHistoryBuffer(TensorDictModuleBase):
    """Parameter-free ring buffer: stacks the last ``history_len`` values of "y_enc" into
    "y_enc_hist" (paper's o_t^H, Sec. III-B / Eq. 9). See the module docstring's "Multi-frame
    history" section for the full design rationale (why "y_enc" and not raw frames, why a
    self-contained module and not a CatFrames env transform).

    Carried state ``HISTORY_KEY`` (shape ``(history_len, enc_dim)``) follows the exact same
    convention as ``state_machine_policy._HFCActorModule``'s recurrent ``p``: written to
    ``("next", HISTORY_KEY)`` for the collector to feed back next step (paired with a
    ``TensorDictPrimer`` + ``InitTracker()`` in the returned transforms), and gracefully
    re-initialised (buffer <- the CURRENT frame repeated ``history_len`` times) whenever the
    carried buffer is absent (bare/transform-free call), all-zero (never primed), or ``is_init``
    fires. ``history_len=1`` is an exact identity regardless of any of the above (see class
    call-site docs) — "y_enc_hist" always equals "y_enc" bit-for-bit.

    Live-sim-found bug (round 4): also treats a carried buffer as "not primed" whenever it is
    not literally a ``torch.Tensor`` — needed because ``TensorDictPrimer``'s zero-fill only
    ever gets computed by its ``_reset_func`` (a real ``torch.full(spec.shape, 0.0, ...)``
    call), which runs on ``env.reset()``; its weaker ``_step()`` path (taken on every bare
    ``env.step()`` call with no preceding reset in the SAME process, e.g. every single tick of
    ``policy_inference_module.infer_action()``, which never calls ``reset()`` at all) just
    tries to copy forward whatever the PREVIOUS "current" tensordict held at this key, and on
    the very first such call there is nothing to copy — verified empirically that this leaves
    a ``tensordict.tensorclass.NonTensorData`` placeholder at ``HISTORY_KEY`` instead of a zero
    tensor, which crashed ``buf.abs()`` with ``RuntimeError: Tensor list must have at least one
    tensor`` (``NonTensorData`` has no tensor leaves to reduce) the instant this was exercised
    through the real ROS node — never reachable via a collector/``Env.rollout()``, both of
    which always call ``reset()`` first. Not a ``TensorDictPrimer`` misuse on this class's
    part (state_machine_policy's ``RECURRENT_KEY`` primer uses the identical
    ``default_value={KEY: 0.0}`` call pattern and would hit the exact same issue under the
    same reset-less calling convention) — fixed here, defensively, at the point of use.
    """

    def __init__(self, enc_dim: int, history_len: int):
        super().__init__()
        self.in_keys = ["y_enc", HISTORY_KEY, "is_init"]
        self.out_keys = ["y_enc_hist", ("next", HISTORY_KEY)]
        self.enc_dim = enc_dim
        self.history_len = history_len

    def forward(self, tensordict: "TensorDictBase") -> "TensorDictBase":
        y_enc = tensordict.get("y_enc")
        batch_shape = y_enc.shape[:-1]
        fresh = y_enc.unsqueeze(-2).expand(*batch_shape, self.history_len, self.enc_dim).clone()

        buf = tensordict.get(HISTORY_KEY, None)
        if not isinstance(buf, torch.Tensor):
            # Absent (bare call), or a non-tensor primer placeholder (see docstring above) --
            # both mean "no real history was ever carried in", so degrade the same way.
            buf = None
        if buf is None:
            new_buf = fresh
        else:
            buf = buf.reshape(*batch_shape, self.history_len, self.enc_dim)
            reset = buf.abs().sum(dim=(-2, -1)) <= 0.0  # never-primed (all-zero) buffer
            is_init = tensordict.get("is_init", None)
            if is_init is not None:
                reset = reset | is_init.reshape(*batch_shape, -1).any(dim=-1)
            # history_len==1: buf[..., 1:, :] is an empty slice, so shifted == fresh always,
            # making the reset/is_init branching above a no-op -- the exact-identity guarantee.
            shifted = torch.cat([buf[..., 1:, :], y_enc.unsqueeze(-2)], dim=-2)
            new_buf = torch.where(reset.reshape(*batch_shape, 1, 1), fresh, shifted)

        tensordict.set("y_enc_hist", new_buf.reshape(*batch_shape, self.history_len * self.enc_dim))
        tensordict.set(("next", HISTORY_KEY), new_buf)
        return tensordict


class EffectiveActionTransform(Transform):
    """Env-side expansion of the 5-D Eq. 3 effective action to the env-native 8-D action.

    See the module docstring's "5-D effective action option" section for the full rationale
    (in particular WHY this must be an env-side ``Transform`` rather than a policy-operator
    submodule, unlike ``_FrameHistoryBuffer`` above). Registered with
    ``in_keys_inv=out_keys_inv=["action"]``; ``TransformedEnv._step`` calls this transform's
    ``.inv()`` on a CLONE of the tensordict to build what physically steps the base env, leaving
    the ORIGINAL (collected/stored/replayed) "action" untouched at 5-D.
    """

    def __init__(self, effective_dt: float, num_driving_parts: int, action_low: torch.Tensor, action_high: torch.Tensor):
        super().__init__(in_keys=[], out_keys=[], in_keys_inv=["action"], out_keys_inv=["action"])
        self.effective_dt = float(effective_dt)
        self.num_driving_parts = int(num_driving_parts)
        self.register_buffer("action_low", action_low.detach().clone())
        self.register_buffer("action_high", action_high.detach().clone())

    @staticmethod
    def expand(
        action_5d: torch.Tensor,
        effective_dt: float,
        num_driving_parts: int,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[v, dtheta_0..3] (..., 5) -> [v]*num_driving_parts + [dtheta_i / effective_dt] (..., 2*num_driving_parts)."""
        v = action_5d[..., :1]
        dtheta = action_5d[..., 1:]
        track = v.expand(*v.shape[:-1], num_driving_parts)
        flipper_vel = dtheta / effective_dt
        expanded = torch.cat([track, flipper_vel], dim=-1)
        if action_low is not None:
            expanded = expanded.clamp(action_low, action_high)
        return expanded

    def _inv_call(self, tensordict: "TensorDictBase") -> "TensorDictBase":
        a5 = tensordict.get("action")
        a8 = self.expand(a5, self.effective_dt, self.num_driving_parts, self.action_low, self.action_high)
        tensordict.set("action", a8)
        return tensordict


class _ConcatGaussianHead(nn.Module):
    """Concat variadic inputs -> MLP -> NormalParamExtractor -> (loc, scale)."""

    def __init__(self, in_dim: int, action_dim: int, mlp_opts: dict):
        super().__init__()
        self.mlp = MLP(in_dim=in_dim, out_dim=2 * action_dim, **mlp_opts)
        self.extractor = NormalParamExtractor()

    def forward(self, *xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.extractor(self.mlp(torch.cat(xs, dim=-1)))


class _ConcatQ(nn.Module):
    """Q body: concat encoded (obs + privileged) features and action -> scalar Q."""

    def __init__(self, enc_dim: int, action_dim: int, mlp_opts: dict):
        super().__init__()
        self.mlp = MLP(in_dim=enc_dim + action_dim, out_dim=1, **mlp_opts)

    def forward(self, enc: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([enc, action], dim=-1))


class _CTRACWrapper(nn.Module):
    """Bundles the contact-conditioned actor + privileged Q-network TEMPLATE.

    ``contact_vae``, ``encoder_operator`` and ``history_operator`` are *references* to modules
    that are registered inside the actor operator (single registration — one ``state_dict``
    entry, one optimizer membership); they are exposed for the trainer's C-VAE updates.

    ``get_qvalue_operator()`` returns a SINGLE Q-network module — not ``num_qvalue_nets`` of
    them — meant to be passed straight to ``SACLoss(..., qvalue_network=...,
    num_qvalue_nets=wrapper.num_qvalue_nets)``, which expands it internally. See the module
    docstring's "Asymmetric privileged twin-Q critic" section for why (torchrl version
    compatibility) and ``experiments/ctrac/train.py`` for how the trainer sources the
    actually-trainable Q-parameters that result.
    """

    def __init__(
        self,
        actor,
        qvalue_op,
        contact_vae,
        encoder_operator,
        history_operator,
        actor_obs_keys,
        priv_obs_keys,
        contact_target_key,
        num_qvalue_nets,
        history_len,
        effective_action_5d,
    ):
        super().__init__()
        self._actor = actor
        self._qvalue = qvalue_op
        self.num_qvalue_nets = num_qvalue_nets
        # bypass nn.Module registration: these live inside self._actor already
        object.__setattr__(self, "contact_vae", contact_vae)
        object.__setattr__(self, "encoder_operator", encoder_operator)
        object.__setattr__(self, "history_operator", history_operator)
        self.actor_obs_keys = list(actor_obs_keys)
        self.priv_obs_keys = list(priv_obs_keys)
        self.contact_target_key = contact_target_key
        self.history_len = history_len
        self.effective_action_5d = effective_action_5d

    def get_policy_operator(self):
        """obs -> continuous "action" (contact conditioning included) — the deploy operator.

        Returns the 5-D Eq. 3 EFFECTIVE action, not the env-native 8-D one, when
        ``effective_action_5d=True`` — see the module docstring's "5-D effective action option"
        section for why and for the expansion formula (``EffectiveActionTransform.expand``).
        """
        return self._actor

    def get_qvalue_operator(self):
        """The Q-network template — pass to ``SACLoss`` with ``num_qvalue_nets=self.num_qvalue_nets``.

        Always built for the env-native action width, regardless of ``effective_action_5d``
        (see the module docstring's "5-D effective action option" section for why the critic
        cannot mirror a smaller actor action space under this torchrl version's ``SACLoss``).
        """
        return self._qvalue

    def get_value_operator(self):
        return self._qvalue

    def eval(self):
        self._actor.eval()
        self._qvalue.eval()
        return self


@dataclass
class CTRACConfig(PolicyConfig):
    """C-TRAC policy factory (contact-conditioned SAC actor + privileged twin-Q + C-VAE).

    Args:
        actor_mlp_opts / qvalue_mlp_opts / cvae_mlp_opts: kwargs for the shared
            ``MLP`` block of the Gaussian head, Q bodies, and C-VAE trunk/heads.
            Set ``activation: ${cls:torch.nn.LeakyReLU}`` here for the paper's
            Sec. V-A.1 (default: this repo's Tanh convention).
        actor_optimizer_opts / qvalue_optimizer_opts / vae_optimizer_opts:
            per-group optimizer kwargs (groups: Gaussian head / Q template /
            shared encoder + C-VAE). ``qvalue_optimizer_opts`` only matters if
            you build an optimizer directly from the "qvalue" ``optim_groups``
            entry; ``experiments/ctrac/train.py`` does not (see
            ``get_qvalue_operator``'s docstring) — pass the SAC critic's
            learning rate there instead (its ``qvalue_optim`` is built from
            ``sac_loss.qvalue_network_params``, not from this config).
        n_contact_areas: number of contact areas A (overridden by the
            ``GroundTruthContacts`` spec when that observation is in the env).
        latent_dim: C-VAE latent size.
        history_len: number of frames stacked into the C-VAE's o_t^H input
            (Sec. III-B / Eq. 9) — see the module docstring's "Multi-frame
            history" section, including why the default (1, meaning "no
            history, C-VAE reads the current frame only" — this file's
            previous, single-frame-only behaviour) is NOT the paper's own
            value (the paper never states one numerically).
        effective_action_5d: use the paper's 5-D Eq. 3 action space
            (``[v, 4 flipper deltas]``) as the actor's OWN decision variable
            instead of the env-native 8-D one — see the module docstring's
            "5-D effective action option" section, INCLUDING the deploy-contract
            exception this implies. Default False ("keep the 8-D default for
            cross-baseline comparability").
        twin_q: sets ``wrapper.num_qvalue_nets`` to 2 (SAC default) vs 1 — the
            Q-network template is always a SINGLE module; pass it to
            ``SACLoss(..., num_qvalue_nets=wrapper.num_qvalue_nets)``, which
            expands it internally (see the module docstring).
        privileged_observations: env observation names routed to the CRITIC
            ONLY. Empty/missing names degrade to a symmetric critic with a loud
            warning (never silent). Include
            ``flipper_training.observations.heightmap.PrivilegedHeightmap``'s
            class name here for the paper's wide ``h_t^f`` (Eq. 2).
        contact_target_key: name of the ground-truth contact observation used
            as the C-VAE target (and, when present, as a privileged critic input).
        contact_reach_m: radius of the geometric-feasibility sphere for L_geo;
            ``None`` derives it from ``env.robot_cfg`` collision points (falls
            back to 1.0 m with a warning if the env has no robot config).
        vae_loss_opts: ContactVAE loss kwargs (``beta``, ``w_recon``, ``w_prob``,
            ``w_est``, ``w_geo``).
        extra_distribution_kwargs: extra kwargs merged into the actor's
            ``TanhNormal`` ``distribution_kwargs`` (beyond ``low``/``high``).
    """

    actor_mlp_opts: dict
    qvalue_mlp_opts: dict
    actor_optimizer_opts: dict
    qvalue_optimizer_opts: dict
    vae_optimizer_opts: dict
    cvae_mlp_opts: dict | None = None
    n_contact_areas: int = 4
    latent_dim: int = 32
    history_len: int = 1
    effective_action_5d: bool = False
    twin_q: bool = True
    privileged_observations: tuple[str, ...] = (DEFAULT_CONTACT_OBS_NAME,)
    contact_target_key: str = DEFAULT_CONTACT_OBS_NAME
    contact_reach_m: float | None = None
    vae_loss_opts: dict = field(default_factory=dict)
    extra_distribution_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("CTRACConfig")
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")

    @staticmethod
    def _enc_mod(observations: list, out_key: str) -> TensorDictModule:
        encoder = EncoderCombiner({o.name: o.get_encoder() for o in observations})
        return TensorDictModule(
            encoder,
            in_keys={k: k for k in encoder.encoders.keys()},
            out_keys=[out_key],
            out_to_in_map=True,
        )

    @staticmethod
    def _obs_flat_dim(obs) -> int:
        spec = obs.get_spec()
        if not hasattr(spec, "shape"):
            raise ValueError(f"Observation {obs.name} has a non-tensor spec; the C-VAE denoise head needs flat tensor observations.")
        return int(math.prod(spec.shape[1:]))

    def _derive_reach(self, env: "Env") -> float:
        if self.contact_reach_m is not None:
            return float(self.contact_reach_m)
        robot_cfg = getattr(env, "robot_cfg", None)
        if robot_cfg is None:
            self.logger.warning("Env has no robot_cfg; using default contact reach of 1.0 m for L_geo (set contact_reach_m to override).")
            return 1.0
        body_r = robot_cfg.body_points.norm(dim=-1).max()
        part_r = (robot_cfg.joint_positions.norm(dim=-1) + robot_cfg.joint_local_driving_part_pts.norm(dim=-1).amax(dim=-1)).max()
        return float(torch.maximum(body_r, part_r))

    def _make_effective_action_spec(self, env: "Env", env_action_spec) -> Bounded:
        """The actor's OWN 5-D Eq. 3 action spec — see the module docstring's "5-D effective
        action option" section for why this differs from ``env_action_spec``."""
        if env.robot_cfg.num_driving_parts != 4:
            raise ValueError(
                f"effective_action_5d assumes 4 flippers (paper Eq. 3: [v, 4 flipper deltas]), "
                f"but env.robot_cfg.num_driving_parts={env.robot_cfg.num_driving_parts}."
            )
        device = env_action_spec.device
        v_max = float(env.robot_cfg.v_max)
        dtheta_max = (env.robot_cfg.joint_max_pivot_vels.to(device=device, dtype=torch.float32) * env.effective_dt).abs()
        low5 = torch.cat([torch.tensor([-v_max], device=device, dtype=torch.float32), -dtheta_max])
        high5 = torch.cat([torch.tensor([v_max], device=device, dtype=torch.float32), dtheta_max])
        n_robots = env_action_spec.shape[0]
        return Bounded(
            low=low5.unsqueeze(0).expand(n_robots, -1).clone(),
            high=high5.unsqueeze(0).expand(n_robots, -1).clone(),
            shape=(n_robots, 5),
            device=device,
            dtype=torch.float32,
        )

    def create(self, env: "Env", **kwargs):
        env_action_spec = env.action_spec
        env_adim = env_action_spec.shape[1]

        # ----- observation split: actor/C-VAE (proprio) vs critic-only (privileged)
        obs_by_name = {o.name: o for o in env.observations}
        priv_names = [n for n in self.privileged_observations if n in obs_by_name]
        missing_priv = [n for n in self.privileged_observations if n not in obs_by_name]
        actor_obs = [o for o in env.observations if o.name not in priv_names]
        if not actor_obs:
            raise ValueError("C-TRAC needs at least one non-privileged observation for the actor.")
        if missing_priv:
            self.logger.warning(
                f"Privileged observation(s) {missing_priv} not found in the env — the critic will "
                f"{'only use ' + str(priv_names) if priv_names else 'be SYMMETRIC Q(obs, action), i.e. NOT the paper asymmetric critic'}. "
                "Add flipper_training.observations.contacts.GroundTruthContacts to the env observations for the paper architecture."
            )

        # ----- contact-area count: the GroundTruthContacts spec is the source of truth when present
        n_areas = self.n_contact_areas
        if self.contact_target_key in obs_by_name:
            gt_dim = self._obs_flat_dim(obs_by_name[self.contact_target_key])
            if gt_dim % 4 != 0:
                raise ValueError(f"{self.contact_target_key} dim {gt_dim} is not 4*n_areas — incompatible contact layout.")
            if gt_dim // 4 != n_areas:
                self.logger.info(f"Deriving n_contact_areas={gt_dim // 4} from the {self.contact_target_key} spec (config said {n_areas}).")
                n_areas = gt_dim // 4

        # ----- shared encoder psi (actor + C-VAE) — FIX: one instance, registered once
        enc_mod = self._enc_mod(actor_obs, "y_enc")
        shared_encoder = enc_mod.module
        obs_recon_dim = sum(self._obs_flat_dim(o) for o in actor_obs)

        # ----- multi-frame history ring buffer: "y_enc" (current frame) -> "y_enc_hist"
        # (o_t^H, Eq. 9). history_len=1 is an exact identity -- see _FrameHistoryBuffer.
        hist_mod = _FrameHistoryBuffer(shared_encoder.output_dim, self.history_len)

        # ----- C-VAE (runs inside the policy operator; trunk reads the HISTORY-STACKED
        # encoding, latent + estimates feed the actor)
        contact_vae = ContactVAE(
            in_dim=shared_encoder.output_dim * self.history_len,
            obs_recon_dim=obs_recon_dim,
            n_areas=n_areas,
            latent_dim=self.latent_dim,
            mlp_opts=self.cvae_mlp_opts or dict(self.qvalue_mlp_opts),
            reach=self._derive_reach(env),
            **self.vae_loss_opts,
        )
        cvae_mod = TensorDictModule(contact_vae, in_keys=["y_enc_hist"], out_keys=["contact_latent", "contact_est", "contact_prob"])

        # ----- actor head: pi(a | enc(o_t) [CURRENT frame only, Fig. 2's separate
        # "Current Obs o_t -> Actor" arrow], z, c_est, c_prob)
        head_in_dim = shared_encoder.output_dim + self.latent_dim + 4 * n_areas
        actor_adim = 5 if self.effective_action_5d else env_adim
        head_mod = TensorDictModule(
            _ConcatGaussianHead(head_in_dim, actor_adim, self.actor_mlp_opts),
            in_keys=["y_enc", "contact_latent", "contact_est", "contact_prob"],
            out_keys=["loc", "scale"],
        )

        # ----- actor's OWN action spec: env-native 8-D by default, or the paper's 5-D Eq. 3
        # space (see the module docstring's "5-D effective action option" section for why the
        # critic below stays env-native regardless).
        action_spec = self._make_effective_action_spec(env, env_action_spec) if self.effective_action_5d else env_action_spec

        actor = ProbabilisticActor(
            module=TensorDictSequential(enc_mod, hist_mod, cvae_mod, head_mod),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low[0],
                "high": action_spec.space.high[0],
                **self.extra_distribution_kwargs,
            },
            return_log_prob=True,
        )

        # ----- asymmetric Q TEMPLATE: Q(obs, privileged, action); own encoder, never the
        # actor/C-VAE's shared one. SINGLE-FRAME (Eq. 2 / Fig. 2 feed the privileged obs to the
        # critic directly, no history-encoder box on that path). Action width matches the
        # ACTOR's own ``actor_adim`` (env-native 8-D by default, or the paper's literal 5-D
        # Eq. 3 width when effective_action_5d=True) — NOT unconditionally env-native; see the
        # module docstring's "5-D effective action option" section for why this MUST track the
        # actor (torchrl 0.8.1's SACLoss._actor_loss/_compute_target_v2 feed the actor's freshly
        # --rsampled action straight to this Q-network on every SAC loss code path, so a width
        # mismatch is a hard crash, not just an inconsistency) — and for why this is also the
        # MORE paper-faithful choice regardless (Sec. IV-A.3: "the critic employs ... Q_phi(s_t,
        # a_t)" with the SAME a_t as Eq. 3, i.e. 5-D in the paper, not an 8-D expansion of it).
        # ONE template only — SACLoss(..., num_qvalue_nets=N) expands it into N independent
        # parameter sets internally (see the module docstring's "Asymmetric privileged twin-Q
        # critic" section for why this file does NOT build N separate nn.Module instances
        # itself: this torchrl version's SACLoss._set_in_keys crashes on a plain list of
        # modules).
        critic_obs = actor_obs + [obs_by_name[n] for n in priv_names]
        q_enc = self._enc_mod(critic_obs, "y_q")
        qvalue_op = TensorDictSequential(
            q_enc,
            TensorDictModule(
                _ConcatQ(q_enc.module.output_dim, actor_adim, self.qvalue_mlp_opts),
                in_keys=["y_q", "action"],
                out_keys=["state_action_value"],
            ),
        )
        num_qvalue_nets = 2 if self.twin_q else 1

        wrapper = _CTRACWrapper(
            actor,
            qvalue_op,
            contact_vae,
            enc_mod,
            hist_mod,
            actor_obs_keys=[o.name for o in actor_obs],
            priv_obs_keys=priv_names,
            contact_target_key=self.contact_target_key if self.contact_target_key in obs_by_name else None,
            num_qvalue_nets=num_qvalue_nets,
            history_len=self.history_len,
            effective_action_5d=self.effective_action_5d,
        )
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        # Disjoint optimizer groups. NOTE 1: the SAC actor loss also backprops into the
        # shared encoder + C-VAE (joint optimisation); those gradients are applied with
        # the "contact_vae" group's settings (the history ring buffer has no parameters of its
        # own -- gradients merely pass through it). NOTE 2: the "qvalue" group below holds the Q
        # TEMPLATE's own parameters — exposed for generic param-count/introspection use,
        # but the C-TRAC trainer does NOT optimize through this group: SACLoss's
        # num_qvalue_nets expansion trains independent parameter copies disconnected from
        # this template's identity, so the trainer builds its qvalue optimizer from
        # ``sac_loss.qvalue_network_params`` instead (see experiments/ctrac/train.py).
        head_params = list(head_mod.parameters())
        vae_params = list(shared_encoder.parameters()) + list(contact_vae.parameters())
        optim_groups = [
            {"params": head_params, "name": "actor", **self.actor_optimizer_opts},
            {"params": list(wrapper._qvalue.parameters()), "name": "qvalue", **self.qvalue_optimizer_opts},
            {"params": vae_params, "name": "contact_vae", **self.vae_optimizer_opts},
        ]

        if weights_path := kwargs.get("weights_path", None):
            mu = wrapper.load_state_dict(torch.load(weights_path, map_location=kwargs.get("device", "cpu")), strict=False)
            self.logger.info(f"Loaded weights from {weights_path}")
            if mu.missing_keys:
                self.logger.warning(f"Missing keys: {mu.missing_keys}")

        # ----- transforms: history-buffer carry (always -- degenerates to a no-op at
        # history_len=1) + the optional 5-D->8-D env-side action expansion.
        transforms = [
            InitTracker(),
            TensorDictPrimer(
                primers=Composite({HISTORY_KEY: Unbounded(shape=(self.history_len, shared_encoder.output_dim), dtype=torch.float32)}),
                default_value={HISTORY_KEY: 0.0},
                expand_specs=True,
            ),
        ]
        if self.effective_action_5d:
            transforms.append(
                EffectiveActionTransform(
                    effective_dt=env.effective_dt,
                    num_driving_parts=env.robot_cfg.num_driving_parts,
                    action_low=env_action_spec.space.low[0],
                    action_high=env_action_spec.space.high[0],
                )
            )

        n_a = sum(p.numel() for p in actor.parameters())
        n_q = sum(p.numel() for p in wrapper._qvalue.parameters())
        self.logger.info(
            f"C-TRAC: contact-conditioned SAC actor={n_a:,} params (shared encoder + "
            f"{self.history_len}-frame history + C-VAE[{n_areas} areas, latent {self.latent_dim}, "
            f"reach {contact_vae.reach:.2f} m] + head), "
            f"action={'5-D effective (Eq. 3)' if self.effective_action_5d else f'{env_adim}-D env-native'}, "
            f"{'asymmetric' if priv_names else 'SYMMETRIC (no privileged obs!)'} "
            f"Q template={n_q:,} params x num_qvalue_nets={num_qvalue_nets} (privileged keys: {priv_names}). "
            "Train with experiments/ctrac/train.py (SACLoss + hybrid C-VAE loss, 1:N alternation); deploy = actor only."
        )
        return wrapper, optim_groups, transforms
