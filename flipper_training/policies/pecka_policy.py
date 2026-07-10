"""Pecka, Salansky, Zimmermann & Svoboda (2016, IROS): "Autonomous Flipper
Control with Safety Constraints" -- linear positional flipper policy.

Paper focus: Sec. II (Constrained REPS, eqs 1-11), Sec. IV-A ("Safe Traversal"
task / policy representation), Sec. III (cautious-simulator safety gate). This
module implements ONLY the lower-level policy representation (Sec IV-A "d)
Policy"); the upper-level Constrained REPS search that learns it lives in
``experiments/creps/train.py`` (see that module's docstring for eqs 1-11).

Faithful-to-the-paper parts
----------------------------
* **Continuous, 2-dimensional action** (Sec IV-A "b) Actions"): the four
  flippers are actuated as two symmetric PAIRS -- front (FL, FR) and rear (RL,
  RR) -- each pair sharing one positional target, the two pairs controlled
  independently. This file previously implemented a discrete 5-preset
  selector network; that was a fabrication not present anywhere in the paper
  (the paper never discretizes the action) and has been replaced outright.
* **The policy is linear in a small state-feature vector** (Sec IV-A "d)
  Policy"): ``target_angle = W @ phi(s)``, ``W`` a ``[2, F]`` matrix (2 pairs
  x F features). With the paper's full 2-dim state (body pitch + terrain
  height ~20cm ahead) this is F=3 (2 state features + bias) => 6 total
  parameters, ``omega = (omega_1, ..., omega_6)``.
* **omega (the entries of W) is the object Constrained REPS searches over**,
  via an episodic upper-level Gaussian ``q(omega) = N(mu, Sigma)``. There is
  deliberately NO gradient-trained ``nn.Parameter`` anywhere in this module:
  ``W`` is a plain buffer that ``experiments/creps/train.py`` overwrites
  directly (never via ``.backward()``/optimizer step), matching the paper's
  black-box episodic policy search.

Bridging this repo's action space (deliberate, documented adaptation)
-----------------------------------------------------------------------
This repo's ``Env`` action space is VELOCITY-controlled, not
POSITION-controlled like the paper's: ``engine.py`` clamps the joint half of
the action (``thetas_d``) to ``robot_cfg.joint_max_pivot_vels`` and integrates
it, i.e. the action IS a pivot angular velocity command (matches CLAUDE.md's
"Policy Output Actions": "4 flipper rotational velocities"). We convert the
policy's positional target to a velocity command with a proportional gain,
``cmd = flipper_kp * (target - theta_current)``, clamped to the true pivot
velocity limit -- the same position-to-velocity bridge this repo's other
positional baseline (``state_machine_policy``) uses for the identical
train/deploy-action-space mismatch. The non-flipper ("track") action dims get
a constant forward command (paper Sec IV-A: "the robot is automatically
driven forward by a constant speed").

Full paper phi(s) is now wired up (6 parameters)
--------------------------------------------------
``observations/robot_state_with_terrain_lookahead.py`` implements the paper's
2nd state feature -- "height of the terrain approximately 20cm in front of the
robot body" (Sec IV-A "a) States") -- as ``LocalStateVectorWithTerrainHeightAhead``
(``LocalStateVector`` + one trailing feature, sampled from the sim's own
ground-truth terrain grid; see that module's docstring for the honest gap vs.
the paper's real octomap sensor and the exact frame convention). Point
``obs_key`` at that class's name and set ``extra_feature_idx: -1`` (the
appended feature is always last) to get the full ``phi = [pitch, height_ahead,
1]`` (F=3, 6 total parameters).

Honest simplification of phi(s) when the extra feature is absent (deliberate,
NOT a silent shortcut)
-----------------------------------------------------------------------------
* Default ``phi(s) = [pitch, 1]`` (F=2, 4 total parameters): ``extra_feature_idx``
  defaults to ``None`` so EXISTING configs (predating the observation above)
  keep working unchanged.
* If ``extra_feature_idx`` IS set but ``obs_key`` doesn't actually have that
  many features (e.g. still pointed at plain ``LocalStateVector``, or a typo'd
  index), ``create()`` validates the request against
  ``env.observation_spec`` and FALLS BACK to the 4-parameter policy with a
  logged warning, rather than crashing inside ``features()`` with an
  out-of-bounds index or silently reading the wrong column
  (``_resolve_extra_feature_idx``).
* Context is NOT modelled: phi depends only on the (non-contextual) state s,
  never on a per-episode "context" vector. The paper's own Sec IV-A "e)
  Context" experiments also fix context to zero ("we did not make use of the
  context -- it was always set to zeros"), so this is a faithful
  specialization of Contextual REPS down to context-free (Constrained) REPS,
  not an omission -- see ``experiments/creps/train.py`` for where that
  specialization is spelled out in the dual.

Deploy contract
----------------
``get_policy_operator()`` applies the upper-level distribution's MEAN omega
(``mu``, the maximum-likelihood point estimate Constrained REPS converged to)
-- deterministic, matching how the paper evaluates "the optimal upper-level
policy" (Table I, Sec IV-D) as a single point estimate rather than a further
stochastic draw. Output is always a continuous action matching
``env.action_spec``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.transforms import RenameTransform

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig

__all__ = ["PeckaLinearPolicyConfig", "RAW_OBS_STASH_KEY"]

RAW_OBS_STASH_KEY = "pecka_raw_obs"


class _PeckaLinearCore(nn.Module):
    """Pure tensor computation: phi(s) -> W @ phi -> P-control -> full env action.

    Holds no tensordict-key knowledge, so it is reusable AS-IS by both the
    deployed actor (below, called with ``W = self.W``, the upper-level mean)
    and by ``experiments/creps/train.py``'s vectorized rollout (called with an
    explicit, per-robot-row ``W`` batch -- one sampled omega per row). Sharing
    this exact function is what guarantees training and deployment can never
    silently compute the action differently.
    """

    def __init__(
        self,
        n_features: int,
        pitch_idx: int,
        pitch_scale: float,
        extra_idx: int | None,
        extra_scale: float,
        extra_offset: float,
        flipper_obs_idx: tuple[int, ...],
        flipper_angle_scale: torch.Tensor,
        flipper_angle_offset: torch.Tensor,
        n_drive: int,
        flipper_kp: float,
        joint_vel_limit: torch.Tensor,
        track_values: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        if len(flipper_obs_idx) != n_drive:
            raise ValueError(f"flipper_obs_idx must have {n_drive} entries, got {len(flipper_obs_idx)}")
        self.n_features = n_features
        self.pitch_idx = pitch_idx
        self.pitch_scale = pitch_scale
        self.extra_idx = extra_idx
        self.extra_scale = extra_scale
        self.extra_offset = extra_offset
        self.n_drive = n_drive
        self.n_pair = n_drive // 2
        self.flipper_kp = flipper_kp
        self.register_buffer("W", torch.zeros(2, n_features))  # [pair(front/rear), F] -- the upper-level mean omega
        self.register_buffer("flipper_obs_idx", torch.as_tensor(flipper_obs_idx, dtype=torch.long))
        self.register_buffer("flipper_angle_scale", flipper_angle_scale)
        self.register_buffer("flipper_angle_offset", flipper_angle_offset)
        self.register_buffer("joint_vel_limit", joint_vel_limit)
        self.register_buffer("track_values", track_values)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """phi(s): ``[..., F] = [pitch, (extra,) 1]``."""
        pitch = obs[..., self.pitch_idx] * self.pitch_scale
        feats = [pitch.unsqueeze(-1)]
        if self.extra_idx is not None:
            extra = obs[..., self.extra_idx] * self.extra_scale + self.extra_offset
            feats.append(extra.unsqueeze(-1))
        feats.append(torch.ones_like(pitch).unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def targets(self, obs: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """``W @ phi(s) -> [..., 2]`` (front, rear) target angles, radians.

        ``W`` broadcasts: ``[2, F]`` (the shared deploy-time mean) or
        ``[..., 2, F]`` (a per-robot-row omega batch from the C-REPS trainer).
        """
        phi = self.features(obs)  # [..., F]
        return (phi.unsqueeze(-2) * W).sum(-1)  # [...,1,F] * [...,2,F] -> [...,2,F] -> [...,2]

    def compute_action(self, obs: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """``[..., obs_dim], W -> [..., action_dim]`` full env action (tracks + flippers)."""
        target = self.targets(obs, W)  # [..., 2] rad, (front, rear)
        theta = obs.index_select(-1, self.flipper_obs_idx) * self.flipper_angle_scale + self.flipper_angle_offset  # [..., n_drive]
        target_per_flipper = torch.cat(
            [
                target[..., 0:1].expand(*target.shape[:-1], self.n_pair),  # front target -> FL, FR
                target[..., 1:2].expand(*target.shape[:-1], self.n_pair),  # rear target -> RL, RR
            ],
            dim=-1,
        )  # [..., n_drive]
        vel_cmd = self.flipper_kp * (target_per_flipper - theta)
        vel_cmd = torch.clamp(vel_cmd, -self.joint_vel_limit, self.joint_vel_limit)
        track_cmd = self.track_values.expand(*vel_cmd.shape[:-1], -1)
        action = torch.cat([track_cmd, vel_cmd], dim=-1)
        return torch.clamp(action, self.action_low, self.action_high)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.compute_action(obs, self.W)


class _PeckaActorModule(TensorDictModuleBase):
    """Tensordict glue: reads the raw-obs stash (falling back to the plain obs
    key for the bare deploy-contract call with no transform pipeline), writes
    "action". See module docstring / ``PeckaLinearPolicyConfig.create`` for
    why the stash exists (VecNorm normalizes ``obs_key`` in place; phi()/theta
    readout need the untouched physical values).
    """

    def __init__(self, core: _PeckaLinearCore, obs_key: str, raw_obs_stash_key: str) -> None:
        super().__init__()
        self.in_keys = [raw_obs_stash_key, obs_key]
        self.out_keys = ["action"]
        self.core = core
        self.obs_key = obs_key
        self.raw_obs_stash_key = raw_obs_stash_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raw = tensordict.get(self.raw_obs_stash_key, None)
        if raw is None:
            raw = tensordict.get(self.obs_key)
        tensordict.set("action", self.core(raw))
        return tensordict


class _PeckaWrapper(nn.Module):
    """Minimal deploy wrapper -- no value head (C-REPS is black-box episodic
    search, not actor-critic). Satisfies the same minimal
    ``get_policy_operator()`` / ``eval()`` surface as this repo's other
    hand-derived baselines (``heightmap_flipper_policy._HeightmapWrapper``,
    ``d3qn_policy._D3QNWrapper``).
    """

    def __init__(self, actor: _PeckaActorModule, obs_key: str, raw_obs_stash_key: str) -> None:
        super().__init__()
        self.actor = actor  # registers `actor` and (transitively) `actor.core` exactly once
        self.obs_key = obs_key
        self.raw_obs_stash_key = raw_obs_stash_key

    @property
    def module(self) -> _PeckaLinearCore:
        """The pure-tensor core (phi/W/P-control). ``experiments/creps/train.py``
        reads/writes ``.W`` directly and calls ``.compute_action(obs, W_batch)``
        with a per-robot-row omega batch during training. Exposed as a
        ``@property`` (not a plain attribute) so ``core`` is registered in the
        module tree exactly once, via ``actor`` -- avoids duplicate/aliased
        entries in ``state_dict()``.
        """
        return self.actor.core

    def get_policy_operator(self) -> _PeckaActorModule:
        return self.actor

    def eval(self) -> "_PeckaWrapper":
        self.actor.eval()
        return self


@dataclass
class PeckaLinearPolicyConfig(PolicyConfig):
    """Pecka et al. (2016) linear positional flipper policy -- see module docstring.

    Args:
        obs_key: observation holding the raw proprioceptive vector (default:
            the ``LocalStateVector`` Observation's name/class).
        pitch_idx: index of body pitch inside ``obs_key`` (LocalStateVector: 1).
        pitch_scale: obs -> radians factor for pitch (LocalStateVector divides
            by pi, so the inverse scale is pi).
        extra_feature_idx: optional index of a 3rd phi feature inside the SAME
            ``obs_key`` vector -- e.g. ``-1`` with
            ``obs_key: LocalStateVectorWithTerrainHeightAhead`` (its trailing
            feature), restoring the paper's full 6-parameter policy. ``None``
            (default) uses the 2-feature fallback ``phi = [pitch, 1]`` (4
            total parameters). If set but not actually resolvable against
            ``obs_key`` (missing from ``env.observation_spec``, or the index
            is out of range), ``create()`` logs a warning and falls back to
            the 4-parameter policy instead of crashing -- see module
            docstring "Honest simplification".
        extra_feature_scale / extra_feature_offset: affine obs -> feature-units
            map applied to the extra feature, ``feature = obs*scale + offset``.
        flipper_angle_idx: indices of the ``n_drive`` current flipper angles
            inside ``obs_key`` (order: front-left, front-right, rear-left,
            rear-right, matching ``robot_cfg.driving_part_names`` /
            ``Env._make_action_spec``). ``None`` (default): derived as
            ``range(8, 8 + n_drive)`` (LocalStateVector's layout: roll(0),
            pitch(1), xd(2:5), omega(5:8), thetas(8:8+n_drive), goal(...)).
        flipper_angle_scale / flipper_angle_offset: affine obs -> radians map
            for those angles, ``theta = obs*scale + offset``. ``None``
            (default): derived from ``env.robot_cfg.joint_limits``
            (LocalStateVector normalizes joint angles to
            ``(theta - lo) / (hi - lo)``); falls back to a symmetric
            ``[-pi/2, pi/2]`` guess if the env exposes no ``robot_cfg``.
        flipper_kp: P-gain converting target-angle error to a pivot-velocity
            command (1/s), clamped to ``robot_cfg.joint_max_pivot_vels`` (the
            engine's real velocity limit -- NOT ``action_spec``'s declared
            bound for that half, which is actually the wider joint-ANGLE
            range; see ``Env._make_action_spec`` vs ``engine.py``).
        track_velocity: constant forward command written to every track-drive
            action dim (paper: "automatically driven forward by a constant
            speed"). Overridden per-dim by ``track_dim_values`` if given.
        track_dim_values: explicit per-track values (length ``n_drive``);
            default: fill all with ``track_velocity``.
        init_std: NOT read by ``experiments/creps/train.py`` -- that trainer
            seeds ``Sigma`` from its OWN top-level YAML key ``init_std:``
            (sibling of ``policy_config:``/``policy_opts:``, see that module's
            docstring "Config" list), not from this field. This field is
            unused at pure deploy time too (a deployed Pecka policy is a fixed
            mean ``omega``, no ``Sigma``). Kept purely as a documentation
            placeholder so the dataclass mentions the concept next to
            ``seed_omega`` -- set the trainer's top-level ``init_std:`` to
            actually change it. (A previous revision of this docstring
            incorrectly claimed creps reads this field; it does not.)
        seed_omega: optional explicit warm-start for the mean omega (length
            ``2 * n_features``, row-major ``[front_params..., rear_params...]``).
            The paper notes initialization has "large influence" on the
            safety of the converged policy (Fig. 4 caption).
    """

    obs_key: str = "LocalStateVector"
    pitch_idx: int = 1
    pitch_scale: float = math.pi
    extra_feature_idx: int | None = None
    extra_feature_scale: float = 1.0
    extra_feature_offset: float = 0.0
    flipper_angle_idx: tuple[int, ...] | None = None
    flipper_angle_scale: float | list[float] | None = None
    flipper_angle_offset: float | list[float] | None = None
    flipper_kp: float = 3.0
    track_velocity: float = 0.4
    track_dim_values: list[float] | None = None
    init_std: float = 0.3
    seed_omega: list[float] | None = None

    def __post_init__(self) -> None:
        self.logger = get_terminal_logger("PeckaLinearPolicyConfig")

    def _n_drive(self, env: Env) -> int:
        adim = env.action_spec.shape[-1]
        n_drive = adim // 2
        if n_drive != 4 or n_drive * 2 != adim:
            raise ValueError(
                "PeckaLinearPolicyConfig assumes 2 flipper pairs (front, rear) = 4 driving parts in a "
                f"[tracks(n_drive), joints(n_drive)] action layout; got action_spec shape {tuple(env.action_spec.shape)} (n_drive={n_drive})."
            )
        return n_drive

    def _resolve_extra_feature_idx(self, env: Env) -> int | None:
        """Validates ``extra_feature_idx`` against ``obs_key``'s ACTUAL width in
        ``env.observation_spec`` (finishes the "extra_feature hook"). Returns
        the index to use, or ``None`` to fall back to the 4-parameter policy
        -- logging a warning whenever the fallback fires, so an absent/typo'd
        terrain-height observation degrades loudly instead of crashing deep
        inside ``features()`` or (worse) silently reading an unrelated column.
        """
        if self.extra_feature_idx is None:
            return None
        try:
            obs_width = int(env.observation_spec[self.obs_key].shape[-1])
        except KeyError:
            self.logger.warning(
                f"extra_feature_idx={self.extra_feature_idx} was requested but obs_key={self.obs_key!r} is not in "
                "env.observation_spec -- falling back to the paper's 4-parameter phi(s)=[pitch,1] policy. For the "
                "full 6-parameter phi(s)=[pitch,height_ahead,1] policy, add "
                "flipper_training.observations.robot_state_with_terrain_lookahead.LocalStateVectorWithTerrainHeightAhead "
                "to the env's `observations:` list and set obs_key to its name (extra_feature_idx: -1)."
            )
            return None
        if not (-obs_width <= self.extra_feature_idx < obs_width):
            self.logger.warning(
                f"extra_feature_idx={self.extra_feature_idx} is out of range for obs_key={self.obs_key!r} "
                f"(width={obs_width}) -- falling back to the 4-parameter phi(s)=[pitch,1] policy."
            )
            return None
        return self.extra_feature_idx

    def _resolve_flipper_obs_idx(self, n_drive: int) -> tuple[int, ...]:
        if self.flipper_angle_idx is not None:
            if len(self.flipper_angle_idx) != n_drive:
                raise ValueError(f"flipper_angle_idx must have {n_drive} entries, got {len(self.flipper_angle_idx)}")
            return tuple(self.flipper_angle_idx)
        return tuple(range(8, 8 + n_drive))

    def _resolve_angle_affine(self, env: Env, n_drive: int) -> tuple[torch.Tensor, torch.Tensor]:
        """obs -> radians affine for the flipper angles: theta = obs * scale + offset."""

        def asn(v: float | list[float]) -> torch.Tensor:
            t = torch.as_tensor(v, dtype=torch.float32).flatten()
            return t.repeat(n_drive) if t.numel() == 1 else t

        if self.flipper_angle_scale is not None and self.flipper_angle_offset is not None:
            return asn(self.flipper_angle_scale), asn(self.flipper_angle_offset)
        joint_limits = getattr(getattr(env, "robot_cfg", None), "joint_limits", None)
        if joint_limits is not None and self.flipper_angle_scale is None and self.flipper_angle_offset is None:
            lo, hi = joint_limits[0].detach().cpu().float(), joint_limits[1].detach().cpu().float()
            self.logger.info(f"Flipper-angle affine derived from env.robot_cfg.joint_limits: scale={(hi - lo).tolist()}, offset={lo.tolist()}")
            return hi - lo, lo
        scale = asn(self.flipper_angle_scale) if self.flipper_angle_scale is not None else torch.full((n_drive,), math.pi)
        offset = asn(self.flipper_angle_offset) if self.flipper_angle_offset is not None else torch.full((n_drive,), -math.pi / 2)
        return scale, offset

    def _resolve_joint_vel_limit(self, env: Env, high: torch.Tensor, n_drive: int) -> torch.Tensor:
        limit = getattr(getattr(env, "robot_cfg", None), "joint_max_pivot_vels", None)
        if limit is not None:
            return limit.detach().cpu().float()
        self.logger.warning("env.robot_cfg.joint_max_pivot_vels not found; falling back to action_spec's (possibly mis-scaled) joint bound.")
        return high[n_drive:].abs()

    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec
        n_drive = self._n_drive(env)
        low = action_spec.space.low[0].detach().cpu().float()
        high = action_spec.space.high[0].detach().cpu().float()

        flipper_obs_idx = self._resolve_flipper_obs_idx(n_drive)
        flipper_scale, flipper_offset = self._resolve_angle_affine(env, n_drive)
        joint_vel_limit = self._resolve_joint_vel_limit(env, high, n_drive)
        extra_idx = self._resolve_extra_feature_idx(env)

        if self.track_dim_values is None:
            track_vals = torch.full((n_drive,), float(self.track_velocity))
        else:
            if len(self.track_dim_values) != n_drive:
                raise ValueError(f"track_dim_values must have {n_drive} entries, got {len(self.track_dim_values)}")
            track_vals = torch.tensor([float(v) for v in self.track_dim_values])
        track_vals = track_vals.clamp(low[:n_drive], high[:n_drive])

        n_features = 3 if extra_idx is not None else 2
        core = _PeckaLinearCore(
            n_features=n_features,
            pitch_idx=self.pitch_idx,
            pitch_scale=self.pitch_scale,
            extra_idx=extra_idx,
            extra_scale=self.extra_feature_scale,
            extra_offset=self.extra_feature_offset,
            flipper_obs_idx=flipper_obs_idx,
            flipper_angle_scale=flipper_scale,
            flipper_angle_offset=flipper_offset,
            n_drive=n_drive,
            flipper_kp=self.flipper_kp,
            joint_vel_limit=joint_vel_limit,
            track_values=track_vals,
            action_low=low,
            action_high=high,
        )
        if self.seed_omega is not None:
            omega0 = torch.tensor([float(v) for v in self.seed_omega], dtype=torch.float32)
            if omega0.numel() != 2 * n_features:
                raise ValueError(f"seed_omega must have {2 * n_features} entries (2 pairs x {n_features} features), got {omega0.numel()}")
            with torch.no_grad():
                core.W.copy_(omega0.view(2, n_features))

        actor = _PeckaActorModule(core, self.obs_key, RAW_OBS_STASH_KEY)
        wrapper = _PeckaWrapper(actor, self.obs_key, RAW_OBS_STASH_KEY)
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        if weights_path := kwargs.get("weights_path", None):
            missing_unexpected = wrapper.load_state_dict(torch.load(weights_path, map_location=kwargs.get("device", "cpu")), strict=False)
            self.logger.info(f"Loaded weights from {weights_path}")
            if missing_unexpected.missing_keys:
                self.logger.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                self.logger.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        # Raw-obs stash BEFORE VecNorm normalizes obs_key (see module docstring
        # + state_machine_policy's identical VecNorm note): phi()/theta readout
        # need LocalStateVector's real physical scaling, not VecNorm's
        # running-statistics whitening. `make_transformed_env` places
        # policy-returned transforms before the VecNorm it appends, so this
        # RenameTransform(create_copy=True) always sees the untouched value.
        transforms = [RenameTransform(in_keys=[self.obs_key], out_keys=[RAW_OBS_STASH_KEY], create_copy=True)]

        n_params = 2 * n_features
        self.logger.info(
            f"Pecka linear positional flipper policy: phi(s) dim={n_features} "
            f"(pitch{' + height_ahead' if extra_idx is not None else ''} + bias), "
            f"omega dim={n_params} (2 pairs x {n_features} features), 0 gradient-trained params. "
            "Train episodically with experiments/creps/train.py (context-free Constrained REPS, "
            "Pecka et al. 2016 eqs 1-11)."
        )
        return wrapper, [], transforms
