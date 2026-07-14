"""RL adaptation of the Azayev & Zimmermann (2022) hybrid flipper controller (HFC).

Azayev, T. & Zimmermann, K. "Autonomous State-Based Flipper Control for
Articulated Tracked Robots in Urban Environments," IEEE RA-L 7(3):7794-7801.

Faithful-to-the-paper parts (Sec III-IV):

* **Hand-coded local flipper policies pi^q with ZERO learnable parameters** (Sec III,
  Sec IV-B). Each of the ``K`` states holds a fixed flipper-angle *template*
  (Fig. 1: neutral / ascending-front / up-stairs / ascending-rear / descending-front /
  down-stairs / descending-rear), overlaid with a PD **body-roll stabilization** term
  (Eq. 7, left group [1,3] gets ``-(phi*kp - phid*kd)``, right group [2,4] the opposite
  sign) and a **stagnation escape** term linear in a stagnation feature ``st in [0,1]``
  (Eq. 8); the three overlays are summed per Eq. 9. The paper's templates are target
  *positions*; this env is velocity-controlled, so the summed target angles are
  converted to flipper angular-velocity commands with a proportional gain
  (``flipper_kp * (theta_target - theta)``) using the current flipper angles read back
  from the observation vector. Track velocity is a constant forward command (the paper
  controls flippers only; steering belongs to the path follower).
* **The ONLY learnable actor component is the state-transition gate**, the
  soft-differentiable state machine (SDSM, Eq. 6): ``K`` per-state transition networks
  ``mu_i`` map the (shared-encoder) observation encoding to a distribution over next
  states, and the carried state distribution is propagated as
  ``p_{t+1} = sum_i p_t^i * mu_i(o_t)``. ``p_t`` is carried between steps as recurrent
  state in the tensordict (key ``"recurrent_state_p"``, zero-primed + ``is_init``-reset
  to the one-hot initial state ``[1, 0, ..., 0]`` of Algorithm 1, following the same
  InitTracker/TensorDictPrimer conventions as ``gru_policy``/``lstm_policy``).
* **Hard argmax inference** (Algorithm 1): with ``hard_inference=True`` (default) and
  the module in ``eval()`` mode, only the argmax-state primitive is executed;
  the state distribution itself is still propagated softly per Eq. 6, exactly as the
  paper describes SDSM inference.

Honest deviations from the paper (documented, deliberate):

* **Training signal**: the paper trains the gate by imitation on human demonstrations
  (Eq. 2) and explicitly forgoes RL (Sec III). This project has NO demonstrator (the
  IL baseline family was dropped), so the gate is trained with PPO — which the paper
  itself names as future work enabled by the SDSM's differentiability (Sec VII:
  "jointly learn parameters in the state transition ... would require training using
  Reinforcement Learning"). During training the emitted action is the ``p_{t+1}``-
  weighted soft blend of the K primitive actions, so ``ClipPPOLoss`` gradients flow
  through the gate; the primitives contribute no parameters.
* **Action readout timing**: Algorithm 1 reads the state from the distribution
  computed at the previous step (``q_t = argmax p_t``). Here the action at step t is
  read from the freshly-updated ``p_{t+1}`` (which already includes ``o_t``); with the
  flattened/shuffled PPO batches used by this trainer, this one-step relabelling is
  what lets gradients reach the gate within a single stored transition.
* The paper's per-template torque levels (Fig. 1) are not modelled (no torque
  interface in this engine).

VecNorm note: primitives must read *raw* observations. ``create()`` therefore returns
a ``RenameTransform(create_copy=True)`` that stashes the raw observation under
``"hfc_raw_obs"`` *before* the trainer-appended VecNorm normalizes it (policy
transforms precede VecNorm in ``make_transformed_env``); the actor reads the stash
and falls back to the plain observation key when the stash is absent (e.g. the bare
deploy-contract call ``wrapper.get_policy_operator()(td)["action"]``).

Deployment: stateless single-tick calls work out of the box (``p`` re-initialised to
the Algorithm-1 one-hot each call). For *stateful* deployment feed
``td["next", "recurrent_state_p"]`` back as ``td["recurrent_state_p"]`` on the next
tick — the same carry the generic node needs for GRU/LSTM hidden states.

Alignment with the author's own code (2026-07 pass)
====================================================

The paper's text is symbolic (Eq. 6-10); the actual numeric constants and network
shapes live only in the author's own repo, ``silverjoda/augmented_robot_trackers``
(silverjoda = Teymur Azayev), cloned read-only at
``/home/cnuc/upstream_refs/azayev2022``. That repo is a ROS1 stack for the same
CTU MARV/TRADR robot family this project targets, and is treated as ground truth
below wherever the paper text alone was ambiguous. Every constant changed in this
pass, with its source ``file:line``:

* **State count/order/names (K=7)** — ``src/envs/marv_dataset_flipper_env.py:9-15``
  and ``src/control/marv_flipper_controller.py:33-39`` both enumerate the SAME 7
  states in the SAME order: NEUTRAL, ASCENDING_FRONT, UP_STAIRS, ASCENDING_REAR,
  DESCENDING_FRONT, DOWN_STAIRS, DESCENDING_REAR. ``DEFAULT_TEMPLATES`` below is
  now keyed/ordered identically (``neutral, ascending_front, up_stairs,
  ascending_rear, descending_front, down_stairs, descending_rear`` — previously
  this repo used a different order/naming, ``stairs_up``/``stairs_down``).
* **Flipper-angle templates** (Eq. 9's ``a_temp``) —
  ``src/control/configs/marv_flipper_controller_config.yaml:23-30``
  (``FLIPPERS_<STATE>``, order ``[front_left, front_right, rear_left, rear_right]``).
  Transcribed verbatim into ``DEFAULT_TEMPLATES`` under the working hypothesis that
  his raw sign convention matches this repo's documented one (front: -pi/2=up,
  +pi/2=down; rear: +pi/2=up, -pi/2=down, per this package's own CLAUDE.md) — cross-
  checked qualitatively against Fig. 1's per-state descriptions (e.g. NEUTRAL folds
  both ends toward their "up" limit; ASCENDING_REAR (NOTE: Fig. 1 arguably draws the AR front flipper folded UP/retracted, not pressed down -- verifier finding 2026-07-13; the numeric template is transcribed from his config regardless) was previously justified as pressing the front down onto the
  obstacle top while extending the rear down to reach it) and found consistent for
  at most ~4 of 7 states on an honest reading (Fig. 1 is stylized/non-metric; an independent render-and-inspect pass found AR and DESCENDING_FRONT visually ambiguous too); DESCENDING_REAR's sign is the state where the qualitative check most clearly
  is inconclusive (documented here rather than silently assumed). No URDF for his
  robot was available in the cloned repo to derive the mapping analytically. His
  ``NEUTRAL`` front value (``-2`` rad) exceeds this engine's joint limit
  (``robots/marv.yaml`` joint_limits=[-1.57, 1.57]) — read as "commanded past the
  mechanical hard stop to rest fully retracted," so it is clamped to ``-pi/2`` here
  (same physical intent, different actuator limit).
* **PD roll-stabilization gains (Eq. 7)** —
  ``src/control/configs/marv_flipper_modulator_config.yaml:25-26``:
  ``roll_stabilization_p=1.4``, ``roll_stabilization_d=0.0`` (old repo defaults were
  0.5/0.1, unsourced guesses). The per-flipper sign pattern this module already used
  (``stab_signs = [-1, +1, +1, -1]`` for ``[FL, FR, RL, RR]``) was independently
  re-derived from ``src/control/marv_flipper_modulator.py:130-138`` and found to
  ALREADY match exactly (his code pairs FL with RR and FR with RL, a diagonal
  grouping — not the simple "left/right" grouping the paper prose describes in
  Eq. 7 — this repo's sign pattern was already faithful to the *code*; only the
  gains kp/kd were unsourced guesses, now fixed).
* **Escape-maneuver constants (Eq. 8)** —
  ``src/control/marv_flipper_modulator.py:143-157``. The paper text only gives a
  worked example for ASCENDING_REAR (Eq. 8 itself: front ``-0.3*st``, rear
  ``+0.5*st``); the ACTUAL DEPLOYED constants in his modulator code differ from the
  paper's own worked example (front ``+0.5*st``, rear ``-0.7*st`` for
  ASCENDING_REAR) AND, critically, escape terms exist for 5 of 7 states, not just
  ASCENDING_REAR — this repo previously modelled only the ascending-rear case as
  non-zero. Per the mission's own precedence rule (author's code overrides paper
  text where they conflict — here they flatly disagree in both sign and magnitude),
  the CODE values are used, transcribed into ``DEFAULT_ESCAPE_DELTAS`` for all 7
  states (front-pair / rear-pair corrections, replicated across left/right exactly
  as his code does — his ``front_flipper_correction``/``rear_flipper_correction``
  are shared, non-differential terms, unlike the roll-stabilization overlay).
* **SDSM transition-network architecture** —
  ``src/policies/policies.py``: ``MiniMLP`` (``:89-101``) is
  ``Linear(feat_dim, 64) -> LeakyReLU -> Linear(64, out_dim)`` (ONE hidden layer,
  hidden width 64, leaky-ReLU, Xavier-normal init) and is the architecture actually
  deployed (``"linear": False`` in
  ``src/control/configs/marv_flipper_controller_config.yaml:10``, i.e. the
  non-linear per-state net, not the linear fallback). ``DSM.__init__``
  (``:216-240``) builds one such net per source state ``k``, each with
  ``out_dim = len(state_transition_dict[k])`` — i.e. his transition graph is
  SPARSE (each state can only reach a small, hand-specified subset of the other
  states), not the dense/fully-connected K-to-K gate this repo previously used
  unconditionally. `gate_mlp_opts` default in `configs/baselines/azayev.yaml` is
  now ``num_hidden: 1, hidden_dim: 64, activation: torch.nn.LeakyReLU`` to match
  (previously 2 hidden layers, Tanh, unsourced). The sparse topology itself —
  ``src/envs/marv_dataset_flipper_env.py:33-39`` /
  ``src/control/marv_flipper_controller.py:53-59`` (``state_transition_dict``,
  identical in both files) — is reproduced as ``DEFAULT_TRANSITION_TOPOLOGY``
  below and applied as a ``-inf``-logit mask before the gate's softmax whenever the
  7 canonical state names are in use (``restrict_topology=True`` by default, the
  same restricted-softmax construction his sparse per-state output layers
  implement, just parameterized densely for implementation simplicity — the two
  are mathematically equivalent: masking disallowed entries to zero probability
  before renormalizing is the same distribution family as never having those output
  units in the first place).
* **Inference rule (Algorithm 1)** — ``src/policies/policies.py:257``
  (``current_state = argmax(new_state_distrib)``) confirms the primitive is always
  selected by hard argmax of the (softly-propagated) state distribution, both
  during his training AND at inference — this module's existing ``hard_inference``
  flag + "Action readout timing" deviation above already implement this correctly
  (soft blend only during PPO training, for gradient flow the paper's IL loss never
  needed since it backprops through a state-classification cross-entropy, not
  through the zero-parameter primitive's action).

What was **NOT** transferred, and why:

* **Observation feature vector / per-flipper elevation bounding boxes (Eq. 10)**.
  His ``feat_dim=9`` gate input is ``[pitch] + frontal_low_feat(4) +
  rear_low_feat(4)`` (``src/envs/marv_dataset_flipper_env.py:98``,
  ``src/control/marv_flipper_controller.py:199``), where each 4-vector is
  ``(avg_height, min_bnd, max_bnd, point_count_intensity)`` computed by cropping a
  local traversability point cloud to a box and taking robust height statistics
  (``src/perception/marv_feature_processor.py:230-256``,
  ``get_pc_feat``/``get_bnd_pts``). The box extents themselves
  (``src/perception/configs/marv_feature_processor_config.yaml:10-16``, all in the
  zero-roll-pitch base-link frame, ``[x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]`` metres):
  ``front_low_feat_bnd=[0.35,0.7,-0.35,0.35,-0.3,0.4]``,
  ``front_mid_feat_bnd=[0.35,0.7,-0.35,0.35,0.4,0.6]``,
  ``rear_low_feat_bnd=[-0.4,-0.0,-0.35,0.35,-0.2,0.3]``, plus 4 narrower
  per-flipper boxes (``fl/fr/rl/rr_flipper_feat_bnd``, e.g.
  ``[0.35,0.7,0.05,0.35,-0.3,0.4]``). This project's ``StateMachinePolicyConfig``
  gate instead reads this repo's shared ``y_shared`` encoder output (here, an MLP
  over ``LocalStateVector`` — roll/pitch/velocity/flipper-angles/goal-vector, NO
  local-terrain channel) because there is no equivalent Observation class in
  ``flipper_training/observations/`` that crops a local heightmap patch into these
  7 boxes and reduces it to (mean/min/max-height, coverage) statistics per box —
  building one is a real, non-trivial new Observation type (new encoder pipeline,
  new heightmap-sampling code, its own tests) rather than a constant swap, and is
  out of scope for this alignment pass. The exact extents above are recorded here
  so a future pass can build ``ElevationBoxFeatures`` (or similar) against this
  repo's heightmap grid and wire it as ``obs_key`` without re-deriving Eq. 10 from
  scratch. GAP CLOSED (2026-07-14, later the same day):
  ``flipper_training.observations.elevation_boxes.ElevationBoxFeatures`` now
  implements the author's DEPLOYED terrain input (his two pooled boxes, robust
  median statistics, verbatim bounds) and ``configs/baselines/azayev.yaml``
  includes it -- the gate now sees pitch (via LocalStateVector) + both boxes,
  matching the deployed 9-D information content. The note below documents why
  the DEPLOYED input (not Eq. 10) is the right target. Code-vs-paper audit note (2026-07-14): the gap is SMALLER than the
  paper implies — the paper's Eq. 1 / Fig. 5 four per-flipper feature vectors are
  computed by his feature processor but NEVER consumed by the deployed classifier
  (``src/control/marv_flipper_controller.py:169-206`` reads only pitch + the two
  pooled full-width boxes). His deployed gate, like ours, sees proprioception plus
  coarse pooled terrain statistics, not per-flipper terrain channels — so this
  repo's ``LocalStateVector``-based gate input deviates from his DEPLOYED system
  mainly by the two pooled elevation boxes, not by four missing per-flipper
  channels. See ``AUTHOR_CODE_FINDINGS.md`` §2.2.
* **Stagnation-feature construction** (the ``st`` fed into Eq. 8, as opposed to
  Eq. 8's linear escape mapping itself, which IS transferred above). His ``st`` is
  a persistent integrator, not an instantaneous ratio:
  ``src/perception/marv_feature_processor.py:552-560`` — increments by 0.10/tick
  while ``avg_lin_vel < 0.04 m/s`` AND the commanded track speed exceeds
  ``0.15 m/s`` (i.e., commanded to move but not actually moving), decays by
  0.05/tick otherwise, resets to 0 on every state change, and is clipped to
  ``[0, 1]``. This module's ``st = clamp(1 - vx/track_velocity, 0, 1)`` is a
  stateless proxy requiring no extra recurrent carry. Replicating his integrator
  faithfully would need a new persistent tensordict key (analogous to
  ``recurrent_state_p``) plus the two velocity-average queues his code keeps —
  a real architectural addition, not a constant change, so it is documented here
  as an honest gap rather than silently approximated further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.data import Composite, Unbounded
from torchrl.envs.transforms import InitTracker, RenameTransform, TensorDictPrimer
from torchrl.modules import ActorValueOperator, ProbabilisticActor, TanhNormal, ValueOperator

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["StateMachinePolicyConfig", "DEFAULT_TEMPLATES", "DEFAULT_ESCAPE_DELTAS", "DEFAULT_TRANSITION_TOPOLOGY"]

RECURRENT_KEY = "recurrent_state_p"
RAW_OBS_STASH_KEY = "hfc_raw_obs"

# Flipper-angle templates (Eq. 9's a_temp), transcribed verbatim from the author's own
# deployed config (src/control/configs/marv_flipper_controller_config.yaml:23-30 in
# silverjoda/augmented_robot_trackers, cloned at /home/cnuc/upstream_refs/azayev2022) —
# see the module docstring's "Alignment with the author's own code" section for the
# sign-convention cross-check and the NEUTRAL front joint-limit clamp. Order
# [front-left, front-right, rear-left, rear-right]; angle 0 = horizontal; this repo's
# convention (CLAUDE.md): FRONT -pi/2=fully up/+pi/2=fully down, REAR +pi/2=fully
# up/-pi/2=fully down. Key order matches his state_list exactly (state 0, "neutral",
# is the Algorithm-1 initial state).
DEFAULT_TEMPLATES: dict[str, tuple[float, float, float, float]] = {
    "neutral": (-1.57, -1.57, 1.5, 1.5),  # his [-2,-2,1.5,1.5]; front clamped to this engine's joint_limits
    # VERIFIER CORRECTION (2026-07-13): the author repo ships TWO values for this
    # state's front template: marv_flipper_controller_config.yaml:25 says -0.4 and
    # marv_flipper_modulator_config.yaml:9 says -0.5. Tracing the runtime shows the
    # CONTROLLER node only publishes the DSM state name; the MODULATOR node is what
    # actually combines template + PD roll-stab + escape into the flipper command --
    # so -0.5 (the modulator's value) is what the real system acts on, and it also
    # keeps the template co-sourced with the PD/escape constants taken from the same
    # node. An earlier revision used -0.4 from the controller-node file.
    "ascending_front": (-0.5, -0.5, 0.0, 0.0),
    "up_stairs": (0.1, 0.1, -0.1, -0.1),
    "ascending_rear": (0.1, 0.1, -0.6, -0.6),
    "descending_front": (0.35, 0.35, -0.7, -0.7),
    "down_stairs": (0.0, 0.0, 0.05, 0.05),
    "descending_rear": (-0.3, -0.3, 0.4, 0.4),
}

# Escape-maneuver overlay per unit stagnation (Eq. 8), transcribed verbatim from
# src/control/marv_flipper_modulator.py:143-157 (his ACTUAL deployed constants, which
# differ from the paper text's own ASCENDING_REAR worked example — see docstring).
# front_flipper_correction is replicated across [FL, FR], rear_flipper_correction
# across [RL, RR], exactly as his code does (shared, non-differential terms, unlike
# the roll-stabilization overlay). States not listed here (neutral, descending_rear)
# have zero escape overlay in his code too.
DEFAULT_ESCAPE_DELTAS: dict[str, tuple[float, float, float, float]] = {
    "ascending_front": (-0.2, -0.2, 0.2, 0.2),
    "up_stairs": (0.10, 0.10, -0.20, -0.20),
    "ascending_rear": (0.5, 0.5, -0.7, -0.7),
    "descending_front": (0.4, 0.4, -0.5, -0.5),
    "down_stairs": (0.10, 0.10, -0.20, -0.20),
}

# Sparse state-transition topology (his DSM's actual per-state output arity), from
# src/envs/marv_dataset_flipper_env.py:33-39 / src/control/marv_flipper_controller.py:53-59
# (state_transition_dict, identical in both). Each entry lists the states reachable
# FROM the key state (always includes itself, the "stay" self-loop). Applied as a
# -inf-logit mask on the gate's softmax (see _SoftStateMachineGate) when the 7
# canonical state names are in use and restrict_topology=True (default).
DEFAULT_TRANSITION_TOPOLOGY: dict[str, list[str]] = {
    "neutral": ["neutral", "ascending_front", "descending_front"],
    "ascending_front": ["ascending_front", "neutral", "up_stairs", "ascending_rear"],
    "up_stairs": ["up_stairs", "ascending_rear"],
    "ascending_rear": ["ascending_rear", "neutral"],
    "descending_front": ["descending_front", "neutral", "down_stairs", "descending_rear"],
    "down_stairs": ["down_stairs", "descending_rear"],
    "descending_rear": ["descending_rear", "neutral"],
}


class _HandCodedPrimitives(nn.Module):
    """The K local flipper control policies pi^q (Eq. 7-9). NO learnable parameters.

    ``forward(raw_obs) -> [..., K, action_dim]``: per-state env actions built from
    fixed angle templates + PD roll stabilization + stagnation escape, converted to
    flipper velocity commands by a P-controller on the observed flipper angles.
    """

    def __init__(
        self,
        templates: torch.Tensor,  # [K, 4] target flipper angles (rad)
        escape_deltas: torch.Tensor,  # [K, 4] angle overlay per unit stagnation (rad)
        action_low: torch.Tensor,  # [action_dim] env action lower bound
        action_high: torch.Tensor,  # [action_dim] env action upper bound
        flipper_angle_idx: tuple[int, int, int, int],
        flipper_angle_scale: torch.Tensor,  # [4] obs -> rad: theta = obs * scale + offset
        flipper_angle_offset: torch.Tensor,  # [4]
        roll_idx: int,
        roll_scale: float,
        roll_rate_idx: int | None,
        roll_rate_scale: float,
        vx_idx: int | None,
        vx_scale: float,
        roll_kp: float,
        roll_kd: float,
        flipper_kp: float,
        track_velocity: float,
        track_dim_values: list[float] | None,
    ):
        super().__init__()
        if templates.shape != escape_deltas.shape or templates.shape[-1] != 4:
            raise ValueError(f"templates/escape_deltas must be [K, 4], got {templates.shape} / {escape_deltas.shape}")
        action_dim = action_low.shape[-1]
        if action_dim < 5:
            raise ValueError(f"action_dim must be >= 5 (leading track dims + 4 flipper dims), got {action_dim}")
        self.flipper_angle_idx = tuple(flipper_angle_idx)
        self.roll_idx = roll_idx
        self.roll_rate_idx = roll_rate_idx
        self.vx_idx = vx_idx
        self.roll_scale = roll_scale
        self.roll_rate_scale = roll_rate_scale
        self.vx_scale = vx_scale
        self.roll_kp = roll_kp
        self.roll_kd = roll_kd
        self.flipper_kp = flipper_kp
        self.track_velocity = track_velocity

        n_track_dims = action_dim - 4
        if track_dim_values is None:
            track_vals = torch.full((n_track_dims,), float(track_velocity))
        else:
            if len(track_dim_values) != n_track_dims:
                raise ValueError(f"track_dim_values must have {n_track_dims} entries (action_dim - 4), got {len(track_dim_values)}")
            track_vals = torch.tensor([float(v) for v in track_dim_values])

        self.register_buffer("templates", templates.float())
        self.register_buffer("escape_deltas", escape_deltas.float())
        self.register_buffer("action_low", action_low.float())
        self.register_buffer("action_high", action_high.float())
        self.register_buffer("track_values", track_vals.clamp(action_low[:n_track_dims], action_high[:n_track_dims]))
        self.register_buffer("theta_scale", flipper_angle_scale.float())
        self.register_buffer("theta_offset", flipper_angle_offset.float())
        self.register_buffer("theta_low", flipper_angle_offset.float())
        self.register_buffer("theta_high", flipper_angle_offset.float() + flipper_angle_scale.float())
        # Eq. 7 sign pattern: independently re-derived from the author's ACTUAL code
        # (src/control/marv_flipper_modulator.py:130-138 in augmented_robot_trackers),
        # which pairs (FL, RR) with one sign and (FR, RL) with the opposite -- a
        # diagonal grouping, NOT the simple "left group [FL,RL] / right group [FR,RR]"
        # the paper's prose (Eq. 7) describes. This tensor already matched his code
        # exactly before this alignment pass (only roll_kp/roll_kd were unsourced).
        self.register_buffer("stab_signs", torch.tensor([-1.0, 1.0, 1.0, -1.0]))
        assert sum(p.numel() for p in self.parameters()) == 0, "primitive local policies must have no learnable parameters (Sec III)"

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        # current flipper angles (rad) read back from the raw observation
        idx = torch.tensor(self.flipper_angle_idx, device=raw_obs.device)
        theta = raw_obs.index_select(-1, idx) * self.theta_scale + self.theta_offset  # [..., 4]
        # Eq. 7 — PD roll stabilization overlay (shared by all states)
        roll = raw_obs[..., self.roll_idx] * self.roll_scale
        if self.roll_rate_idx is not None:
            roll_rate = raw_obs[..., self.roll_rate_idx] * self.roll_rate_scale
        else:
            roll_rate = torch.zeros_like(roll)
        stab = self.roll_kp * roll - self.roll_kd * roll_rate  # [...]
        stab_delta = stab.unsqueeze(-1) * self.stab_signs  # [..., 4]
        # Eq. 8 — stagnation feature and per-state escape overlay
        if self.vx_idx is not None and self.track_velocity > 0:
            vx = raw_obs[..., self.vx_idx] * self.vx_scale
            st = (1.0 - vx / self.track_velocity).clamp(0.0, 1.0)  # [...]
        else:
            st = torch.zeros_like(roll)
        escape = st.unsqueeze(-1).unsqueeze(-1) * self.escape_deltas  # [..., K, 4]
        # Eq. 9 — target angles = template + stabilization + escape, kept within joint limits
        targets = (self.templates + stab_delta.unsqueeze(-2) + escape).clamp(self.theta_low, self.theta_high)  # [..., K, 4]
        # position targets -> velocity commands (P-controller on the observed angles)
        flipper_cmd = (self.flipper_kp * (targets - theta.unsqueeze(-2))).clamp(self.action_low[-4:], self.action_high[-4:])
        track_cmd = self.track_values.expand(*flipper_cmd.shape[:-1], -1)  # [..., K, n_track_dims]
        return torch.cat([track_cmd, flipper_cmd], dim=-1).clamp(self.action_low, self.action_high)  # [..., K, action_dim]


class _SoftStateMachineGate(nn.Module):
    """The learnable SDSM transition classifier (Eq. 6).

    K per-state transition networks ``mu_i`` (separate parameters phi_i, as in the
    paper) map the encoded observation to a distribution over next states;
    ``forward`` returns ``p_next = sum_i p[..., i] * softmax(mu_i(enc) / T)``.

    ``transition_mask`` (optional ``[K, K]`` bool, row i = states reachable from i)
    reproduces the author's SPARSE per-state output arity (his ``DSM`` gives each
    source state's network only as many outputs as it has allowed transitions,
    ``src/policies/policies.py:216-240`` in ``augmented_robot_trackers`` — see the
    module docstring). Masking disallowed entries to ``-inf`` before the softmax is
    the same distribution family as never having those output units in the first
    place, just parameterized densely for implementation simplicity. ``None``
    (default when the topology can't be resolved) keeps the fully-connected gate.
    """

    def __init__(self, enc_dim: int, n_states: int, gate_mlp_opts: dict, temperature: float, transition_mask: torch.Tensor | None = None):
        super().__init__()
        self.n_states = n_states
        self.temperature = temperature
        self.transition_mlps = nn.ModuleList(MLP(in_dim=enc_dim, out_dim=n_states, **gate_mlp_opts) for _ in range(n_states))
        self.register_buffer("transition_mask", transition_mask.bool() if transition_mask is not None else None)

    def forward(self, enc: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([m(enc) for m in self.transition_mlps], dim=-2)  # [..., K (cond. state i), K (next state)]
        if self.transition_mask is not None:
            logits = logits.masked_fill(~self.transition_mask, float("-inf"))
        trans = torch.softmax(logits / self.temperature, dim=-1)
        return (p.unsqueeze(-1) * trans).sum(dim=-2)  # Eq. 6: [..., K]


class _HFCActorModule(TensorDictModuleBase):
    """Ties gate + primitives together and emits TanhNormal params (loc, scale).

    Reads: shared encoding, raw observation (stash key with fallback to the plain
    observation key), carried state distribution ``recurrent_state_p`` and ``is_init``
    (both optional — missing/zero/reset entries re-initialise to Algorithm 1's
    ``p_1 = [1, 0, ..., 0]``). Writes: ``loc``/``scale`` for the ProbabilisticActor,
    the propagated distribution to ``("next", "recurrent_state_p")`` (collector
    carry-over, GRUModule convention) and a detached copy to ``"state_probs"`` for
    logging/analysis.

    Detach-through-time note (code-vs-paper audit, 2026-07-14): the carried ``p``
    is detached before being written to ``("next", "recurrent_state_p")`` below —
    gradients flow into the gate only within a single step's Eq.-6 blend, never
    across timesteps. The paper claims the SDSM's advantage is backpropagation
    "through time", but the author's own deployed training code detaches the state
    distribution at every step too (``augmented_robot_trackers``
    ``src/policies/policies.py:242-258``, ``calculate_next_state_diff``, his
    comment: "Detach here for no gradient propagation through time"; the
    non-detached variant is dead code). So this implementation matches the
    AUTHOR'S CODE exactly; under PPO's flattened minibatches a non-detached carry
    could not propagate across collector steps anyway, so no flag is offered —
    see ``AUTHOR_CODE_FINDINGS.md`` §2.2.
    """

    def __init__(
        self,
        gate: _SoftStateMachineGate,
        primitives: _HandCodedPrimitives,
        obs_key: str,
        raw_obs_stash_key: str,
        action_std: float,
        learnable_std: bool,
        hard_inference: bool,
        enc_key: str = "y_shared",
    ):
        super().__init__()
        self.in_keys = [enc_key, raw_obs_stash_key, obs_key, RECURRENT_KEY, "is_init"]
        self.out_keys = ["loc", "scale", "state_probs", ("next", RECURRENT_KEY)]
        self.gate = gate
        self.primitives = primitives
        self.obs_key = obs_key
        self.raw_obs_stash_key = raw_obs_stash_key
        self.enc_key = enc_key
        self.hard_inference = hard_inference
        log_std = torch.full((primitives.action_low.shape[-1],), math.log(action_std))
        if learnable_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

    def _initial_p(self, batch_shape: torch.Size, device: torch.device) -> torch.Tensor:
        p0 = torch.zeros(*batch_shape, self.gate.n_states, device=device)
        p0[..., 0] = 1.0  # Algorithm 1: p_1 = [1, 0, ..., 0]
        return p0

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        enc = tensordict.get(self.enc_key)
        raw = tensordict.get(self.raw_obs_stash_key, None)
        if raw is None:
            raw = tensordict.get(self.obs_key)
        batch_shape = enc.shape[:-1]
        # --- carried state distribution (recurrent state), Algorithm-1 init/reset ---
        p = tensordict.get(RECURRENT_KEY, None)
        if p is None:
            p = self._initial_p(batch_shape, enc.device)
        else:
            p = p.float()
            reset = p.sum(dim=-1, keepdim=True) <= 1e-6  # zero-primed entries (episode start)
            is_init = tensordict.get("is_init", None)
            if is_init is not None:
                if is_init.shape != p.shape[:-1]:
                    is_init = is_init.view(*p.shape[:-1], -1).any(dim=-1)
                reset = reset | is_init.unsqueeze(-1)
            p = torch.where(reset, self._initial_p(p.shape[:-1], p.device), p)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # --- Eq. 6 soft transition, then action readout ---
        p_next = self.gate(enc, p)
        per_state_actions = self.primitives(raw)  # [..., K, action_dim]
        if self.hard_inference and not self.training:
            # Algorithm 1: run only the argmax-state primitive (p itself stays soft)
            weights = nn.functional.one_hot(p_next.argmax(dim=-1), self.gate.n_states).to(p_next.dtype)
        else:
            weights = p_next  # soft blend -> PPO gradients flow into the gate
        blended = (weights.unsqueeze(-1) * per_state_actions).sum(dim=-2)  # [..., action_dim]
        # map the (in-bounds) blended action to TanhNormal loc so that the
        # deterministic sample tanh(loc) reproduces it exactly
        low, high = self.primitives.action_low, self.primitives.action_high
        normalized = (2.0 * (blended - low) / (high - low) - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        tensordict.set("loc", torch.atanh(normalized))
        tensordict.set("scale", self.log_std.exp().expand(*batch_shape, -1))
        tensordict.set("state_probs", p_next.detach())
        # detached carry — matches the author's own per-step .detach() (see class docstring)
        tensordict.set(("next", RECURRENT_KEY), p_next.detach())
        return tensordict


@dataclass
class StateMachinePolicyConfig(PolicyConfig):
    """RL adaptation of the Azayev & Zimmermann (2022) HFC — see module docstring.

    Hand-coded primitive local policies (zero learnable action parameters) + the
    learnable soft-differentiable state-transition gate (Eq. 6) trained with
    ``ClipPPOLoss`` (the paper's Sec-VII future-work variant; the paper itself trains
    the gate by imitation, impossible here for lack of a demonstrator). Value head on
    the shared encoder. At eval time the primitive of the argmax state is executed
    (Algorithm 1) when ``hard_inference=True``.

    Args:
        gate_mlp_opts: shared ``MLP`` kwargs for each of the K per-state transition
            networks mu_i (the only learnable actor component).
        value_mlp_opts: ``MLP`` kwargs for the value head (reads the shared encoding).
        actor_optimizer_opts / value_optimizer_opts: per-group optimizer kwargs.
        n_states: number K of state-machine states. Defaults to all 7 paper states;
            if ``templates`` is None the first ``n_states`` entries of
            ``DEFAULT_TEMPLATES`` are used (neutral first — the Alg.-1 initial state).
        templates: optional explicit [K][4] flipper-angle templates (rad, repo
            convention [FL, FR, RL, RR]; front -pi/2=up, rear +pi/2=up). Overrides
            ``n_states``.
        escape_deltas: optional [K][4] escape-maneuver angle overlay per unit
            stagnation (Eq. 8). Default: ``DEFAULT_ESCAPE_DELTAS`` (author's code
            values for 5 of 7 states; zero for neutral/descending_rear, matching his).
        restrict_topology: mask the gate's softmax to the author's sparse
            state-transition graph (``DEFAULT_TRANSITION_TOPOLOGY``, his
            ``state_transition_dict``) whenever the 7 canonical state names are in
            use and ``transition_topology`` is not given explicitly. ``False``
            restores the fully-connected gate this module used before this
            alignment pass. Falls back to fully-connected (with a warning) if the
            state names don't match the canonical set and no explicit
            ``transition_topology`` is given.
        transition_topology: optional explicit ``{state_name: [reachable states]}``
            adjacency (overrides the ``DEFAULT_TRANSITION_TOPOLOGY`` lookup).
        temperature: gate softmax temperature.
        hard_inference: argmax-state primitive at eval time (Algorithm 1). Training
            mode always soft-blends so gradients flow.
        action_std / learnable_std: exploration std of the TanhNormal head (RL
            machinery only, NOT a primitive parameter; fixed buffer by default so the
            primitive/action path stays parameter-free).
        obs_key: observation key holding the proprioceptive vector the primitives
            read (name of the Observation, e.g. ``"LocalStateVector"``).
        flipper_angle_idx: indices of the 4 flipper angles [FL, FR, RL, RR] inside
            that vector (LocalStateVector: 8-11).
        flipper_angle_scale / flipper_angle_offset: affine map obs -> radians,
            ``theta = obs * scale + offset`` (scalar or per-flipper list). Default
            (None): derived from ``env.robot_cfg.joint_limits`` when available, else
            LocalStateVector's ``(theta - lo) / (hi - lo)`` with lo/hi = -/+ pi/2.
        roll_idx / roll_scale: roll entry and obs -> radians factor (LocalStateVector:
            index 0, scale pi).
        roll_rate_idx / roll_rate_scale: roll-rate entry (local angular velocity x;
            LocalStateVector: index 5, scale pi). None disables the D-term.
        vx_idx / vx_scale: forward-velocity entry for the stagnation feature
            (LocalStateVector: index 2; scale defaults to the env's ``max_dist`` when
            available, else 1.0). None disables escape maneuvers.
        roll_kp / roll_kd: Eq. 7 PD gains (rad target offset per rad roll / rad/s).
            Defaults 1.4/0.0 are the author's own deployed values
            (``marv_flipper_modulator_config.yaml``; his D-term is effectively
            disabled in the shipped config).
        flipper_kp: P-gain converting target-angle error to velocity command (1/s).
        track_velocity: constant forward track command (m/s; the paper's HFC controls
            flippers only).
        track_dim_values: explicit values for the leading non-flipper action dims
            (default: fill all with ``track_velocity``; use e.g. ``[v, 0.0]`` for a
            [v, w, flippers...] action layout). The LAST 4 action dims are always the
            flipper commands.
        extra_distribution_kwargs: forwarded to TanhNormal.
    """

    gate_mlp_opts: dict
    value_mlp_opts: dict
    actor_optimizer_opts: dict
    value_optimizer_opts: dict
    n_states: int = 7
    templates: list[list[float]] | None = None
    escape_deltas: list[list[float]] | None = None
    restrict_topology: bool = True
    transition_topology: dict[str, list[str]] | None = None
    temperature: float = 1.0
    hard_inference: bool = True
    action_std: float = 0.3
    learnable_std: bool = False
    obs_key: str = "LocalStateVector"
    flipper_angle_idx: tuple[int, int, int, int] = (8, 9, 10, 11)
    flipper_angle_scale: float | list[float] | None = None
    flipper_angle_offset: float | list[float] | None = None
    roll_idx: int = 0
    roll_scale: float = math.pi
    roll_rate_idx: int | None = 5
    roll_rate_scale: float = math.pi
    vx_idx: int | None = 2
    vx_scale: float | None = None
    roll_kp: float = 1.4
    roll_kd: float = 0.0
    flipper_kp: float = 3.0
    track_velocity: float = 0.4
    track_dim_values: list[float] | None = None
    extra_distribution_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("StateMachinePolicyConfig")

    def _build_templates(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if self.templates is not None:
            templates = torch.tensor(self.templates, dtype=torch.float32)
            names = [f"state_{i}" for i in range(templates.shape[0])]
        else:
            if not 1 <= self.n_states <= len(DEFAULT_TEMPLATES):
                raise ValueError(f"n_states must be in [1, {len(DEFAULT_TEMPLATES)}] when using default templates, got {self.n_states}")
            names = list(DEFAULT_TEMPLATES)[: self.n_states]
            templates = torch.tensor([DEFAULT_TEMPLATES[n] for n in names], dtype=torch.float32)
        k = templates.shape[0]
        if self.escape_deltas is not None:
            escape = torch.tensor(self.escape_deltas, dtype=torch.float32)
            if escape.shape != templates.shape:
                raise ValueError(f"escape_deltas shape {tuple(escape.shape)} must match templates shape {tuple(templates.shape)}")
        else:
            escape = torch.zeros_like(templates)
            for i, name in enumerate(names):
                if name in DEFAULT_ESCAPE_DELTAS:  # Eq. 8, author's code values (see docstring)
                    escape[i] = torch.tensor(DEFAULT_ESCAPE_DELTAS[name])
        return templates, escape, names[:k]

    def _build_transition_mask(self, state_names: list[str]) -> torch.Tensor | None:
        """Sparse-topology mask for the gate's softmax (author's state_transition_dict).

        Returns ``None`` (fully-connected gate, this module's pre-alignment behavior)
        when disabled, or when the state names don't match a resolvable topology.
        """
        if not self.restrict_topology:
            return None
        topology = self.transition_topology
        if topology is None:
            if set(state_names) == set(DEFAULT_TRANSITION_TOPOLOGY):
                topology = DEFAULT_TRANSITION_TOPOLOGY
            else:
                self.logger.warning(
                    f"restrict_topology=True but state names {state_names} do not match the author's "
                    f"canonical 7-state set {list(DEFAULT_TRANSITION_TOPOLOGY)} and no explicit "
                    "transition_topology was given -- falling back to a fully-connected gate."
                )
                return None
        n = len(state_names)
        mask = torch.zeros(n, n, dtype=torch.bool)
        for i, name in enumerate(state_names):
            allowed = topology.get(name, state_names)  # unresolved state name -> fully-connected row
            for j, other in enumerate(state_names):
                if other in allowed:
                    mask[i, j] = True
        return mask

    def _resolve_angle_affine(self, env: Env) -> tuple[torch.Tensor, torch.Tensor]:
        """obs -> radians affine for the flipper angles: theta = obs * scale + offset."""

        def as4(v: float | list[float]) -> torch.Tensor:
            t = torch.as_tensor(v, dtype=torch.float32).flatten()
            return t.repeat(4) if t.numel() == 1 else t

        if self.flipper_angle_scale is not None and self.flipper_angle_offset is not None:
            return as4(self.flipper_angle_scale), as4(self.flipper_angle_offset)
        joint_limits = getattr(getattr(env, "robot_cfg", None), "joint_limits", None)
        if joint_limits is not None and self.flipper_angle_scale is None and self.flipper_angle_offset is None:
            lo, hi = joint_limits[0].detach().cpu().float(), joint_limits[1].detach().cpu().float()
            self.logger.info(f"Flipper-angle affine derived from env.robot_cfg.joint_limits: scale={(hi - lo).tolist()}, offset={lo.tolist()}")
            return hi - lo, lo
        scale = as4(self.flipper_angle_scale) if self.flipper_angle_scale is not None else torch.full((4,), math.pi)
        offset = as4(self.flipper_angle_offset) if self.flipper_angle_offset is not None else torch.full((4,), -math.pi / 2)
        return scale, offset

    def _resolve_vx_scale(self, env: Env) -> float:
        if self.vx_scale is not None:
            return self.vx_scale
        max_coord = getattr(getattr(env, "terrain_cfg", None), "max_coord", None)
        if max_coord is not None:  # LocalStateVector normalizes velocities by max_dist
            return float(max_coord) * 2**1.5
        return 1.0

    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec
        templates, escape_deltas, state_names = self._build_templates()
        n_states = templates.shape[0]

        encoders = {o.name: o.get_encoder() for o in env.observations}
        if self.obs_key not in encoders:
            self.logger.warning(f"obs_key '{self.obs_key}' is not among the env observations {list(encoders)} — primitives will fail at runtime.")
        encoder = EncoderCombiner(encoders)
        common = TensorDictModule(
            encoder,
            in_keys={k: k for k in encoder.encoders.keys()},
            out_keys=["y_shared"],
            out_to_in_map=True,
        )

        angle_scale, angle_offset = self._resolve_angle_affine(env)
        primitives = _HandCodedPrimitives(
            templates=templates,
            escape_deltas=escape_deltas,
            action_low=action_spec.space.low[0].detach().cpu(),
            action_high=action_spec.space.high[0].detach().cpu(),
            flipper_angle_idx=self.flipper_angle_idx,
            flipper_angle_scale=angle_scale,
            flipper_angle_offset=angle_offset,
            roll_idx=self.roll_idx,
            roll_scale=self.roll_scale,
            roll_rate_idx=self.roll_rate_idx,
            roll_rate_scale=self.roll_rate_scale,
            vx_idx=self.vx_idx,
            vx_scale=self._resolve_vx_scale(env),
            roll_kp=self.roll_kp,
            roll_kd=self.roll_kd,
            flipper_kp=self.flipper_kp,
            track_velocity=self.track_velocity,
            track_dim_values=self.track_dim_values,
        )
        transition_mask = self._build_transition_mask(state_names)
        gate = _SoftStateMachineGate(encoder.output_dim, n_states, self.gate_mlp_opts, self.temperature, transition_mask=transition_mask)
        hfc_module = _HFCActorModule(
            gate=gate,
            primitives=primitives,
            obs_key=self.obs_key,
            raw_obs_stash_key=RAW_OBS_STASH_KEY,
            action_std=self.action_std,
            learnable_std=self.learnable_std,
            hard_inference=self.hard_inference,
        )
        actor = ProbabilisticActor(
            module=hfc_module,
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
        value = ValueOperator(
            module=MLP(in_dim=encoder.output_dim, out_dim=1, **self.value_mlp_opts),
            in_keys=["y_shared"],
        )
        wrapper = ActorValueOperator(policy_operator=actor, value_operator=value, common_operator=common)
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        if weights_path := kwargs.get("weights_path", None):
            missing_unexpected = wrapper.load_state_dict(torch.load(weights_path, map_location=kwargs.get("device", "cpu")), strict=False)
            self.logger.info(f"Loaded weights from {weights_path}")
            if missing_unexpected.missing_keys:
                self.logger.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                self.logger.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        optim_groups = [
            {"params": wrapper.get_policy_operator().parameters(), "name": "policy_and_encoder", **self.actor_optimizer_opts},
            {"params": wrapper.get_value_head().parameters(), "name": "value_head", **self.value_optimizer_opts},
        ]

        # p_t carry (zero-primed; the actor resets zero/is_init entries to Alg.-1 one-hot),
        # is_init tracking, and the raw-observation stash copied out BEFORE VecNorm.
        transforms = [
            InitTracker(),
            TensorDictPrimer(
                primers=Composite({RECURRENT_KEY: Unbounded(shape=(n_states,), dtype=torch.float32)}),
                default_value={RECURRENT_KEY: 0.0},
                expand_specs=True,
            ),
            RenameTransform(in_keys=[self.obs_key], out_keys=[RAW_OBS_STASH_KEY], create_copy=True),
        ]

        n_gate = sum(p.numel() for p in gate.parameters())
        n_primitive = sum(p.numel() for p in primitives.parameters())
        n_encoder = sum(p.numel() for p in encoder.parameters())
        n_value = sum(p.numel() for p in value.parameters())
        self.logger.info(
            f"Azayev HFC (RL adaptation): K={n_states} hand-coded primitives {state_names} "
            f"({n_primitive} learnable params — must be 0), SDSM transition gate {n_gate:,} params, "
            f"topology={'sparse (author DSM)' if transition_mask is not None else 'fully-connected'}, "
            f"encoder {n_encoder:,}, value head {n_value:,}. temp={self.temperature}, "
            f"hard_inference={self.hard_inference} (Alg. 1 argmax at eval). Train with ClipPPOLoss."
        )
        return wrapper, optim_groups, transforms
