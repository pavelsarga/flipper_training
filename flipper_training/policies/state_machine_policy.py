"""RL adaptation of the Azayev & Zimmermann (2022) hybrid flipper controller (HFC).

Azayev, T. & Zimmermann, K. "Autonomous State-Based Flipper Control for
Articulated Tracked Robots in Urban Environments," IEEE RA-L 7(3):7794-7801.

Faithful-to-the-paper parts (Sec III-IV):

* **Hand-coded local flipper policies pi^q with ZERO learnable parameters** (Sec III,
  Sec IV-B). Each of the ``K`` states holds a fixed flipper-angle *template*
  (Fig. 1: neutral / ascending-front / ascending-rear / stairs-up / descending-front /
  descending-rear / stairs-down), overlaid with a PD **body-roll stabilization** term
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
  interface in this engine), and the stagnation feature compares the observed forward
  speed against the constant commanded track speed instead of a path-follower target.

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

__all__ = ["StateMachinePolicyConfig", "DEFAULT_TEMPLATES"]

RECURRENT_KEY = "recurrent_state_p"
RAW_OBS_STASH_KEY = "hfc_raw_obs"

# Default flipper-angle templates distilled from Fig. 1 / Sec. I of the paper, in this
# repo's angle convention (see CLAUDE.md): order [front-left, front-right, rear-left,
# rear-right]; angle 0 = horizontal; FRONT: -pi/2 = fully up, +pi/2 = fully down;
# REAR: +pi/2 = fully up, -pi/2 = fully down. State 0 (neutral) is the Algorithm-1
# initial state. Values are heuristic per-robot tuning, overridable via ``templates``.
DEFAULT_TEMPLATES: dict[str, tuple[float, float, float, float]] = {
    "neutral": (-0.4, -0.4, 0.4, 0.4),  # all flippers slightly raised (travel)
    "ascending_front": (-1.0, -1.0, -0.2, -0.2),  # front raised onto obstacle, rear pushing
    "ascending_rear": (0.2, 0.2, -1.0, -1.0),  # front pressing the top, rear extended down
    "stairs_up": (-0.6, -0.6, -0.3, -0.3),  # long support polygon along ascending slope
    "descending_front": (0.9, 0.9, 0.3, 0.3),  # front reaching down over the edge
    "descending_rear": (-0.2, -0.2, -0.8, -0.8),  # rear down holding the upper edge
    "stairs_down": (0.6, 0.6, -0.4, -0.4),  # long support polygon along descending slope
}

# Escape-maneuver overlay per unit stagnation (Eq. 8 magnitudes, repo sign convention):
# defined in the paper for the ascending-rear state only — lift front (0.3), lower rear
# (0.5); front lift = negative angle delta, rear lower = negative angle delta.
_DEFAULT_ESCAPE_AR = (-0.3, -0.3, -0.5, -0.5)


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
        # Eq. 7 sign pattern in repo convention: per-side push-down = (-stab, +stab) for
        # (left, right) groups; angle delta per unit push-down = +1 front / -1 rear.
        # [FL, FR, RL, RR] -> side (-1,+1,-1,+1) * down-direction (+1,+1,-1,-1).
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
    """

    def __init__(self, enc_dim: int, n_states: int, gate_mlp_opts: dict, temperature: float):
        super().__init__()
        self.n_states = n_states
        self.temperature = temperature
        self.transition_mlps = nn.ModuleList(MLP(in_dim=enc_dim, out_dim=n_states, **gate_mlp_opts) for _ in range(n_states))

    def forward(self, enc: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([m(enc) for m in self.transition_mlps], dim=-2)  # [..., K (cond. state i), K (next state)]
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
            stagnation (Eq. 8). Default: zeros except the ascending-rear default state.
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
    roll_kp: float = 0.5
    roll_kd: float = 0.1
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
            if "ascending_rear" in names:  # Eq. 8 is defined for the ascending-rear state
                escape[names.index("ascending_rear")] = torch.tensor(_DEFAULT_ESCAPE_AR)
        return templates, escape, names[:k]

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
        gate = _SoftStateMachineGate(encoder.output_dim, n_states, self.gate_mlp_opts, self.temperature)
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
            f"encoder {n_encoder:,}, value head {n_value:,}. temp={self.temperature}, "
            f"hard_inference={self.hard_inference} (Alg. 1 argmax at eval). Train with ClipPPOLoss."
        )
        return wrapper, optim_groups, transforms
