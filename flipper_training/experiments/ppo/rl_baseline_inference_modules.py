"""Isaac-free inference wrappers for the MARV_RL-trained baseline RL modules
(atd3qn, icmd3qn, hfc, mitriakov, creps) — the counterparts, for those five, of
FtrPolicyInferenceModule (which only ever handled the ``marv_rl`` module).

Each of these five was trained inside MARV_RL/src/FTR-Benchmark's Isaac-Lab
harness with its OWN observation formula and (for atd3qn/icmd3qn/hfc/mitriakov)
its own bespoke policy network class, all living in the ``rl_modules.<algo>``
namespace package (MARV_RL/src/FTR-Benchmark) and importing observation/policy
base classes from ``marv_rl_training`` (MARV_RL/src/flipper_training). Neither
package needs Isaac Sim at import time for inference — only the *training*
entry points (``marv_rl_training.training.train_*``) launch
``omni.isaac.lab.app.AppLauncher``; the policy/observation classes themselves
have zero Isaac imports (verified by reading every file below). Both packages
must be on ``sys.path``/``PYTHONPATH`` for the imports in this file to resolve
(see ``run_flipper_policy_sim.sh``).

Every class below exposes the same tiny contract:

    module.reset()                                    # call once per episode
    action = module.infer_action(
        heightmap=..., heightmap_extent=..., goal_vec_local=..., xd_local=...,
        omega_local=..., thetas=..., quat=..., robot_world_z=..., dt=...,
    )
    # action: RLBaselineAction(v, w, flipper_mode, flippers)

``flipper_policy_node.py`` reads ``action.flipper_mode`` to decide whether to
publish ``action.flippers`` (rad/s) to ``/flippers_cmd_vel/*`` or (rad,
absolute target) to ``/flippers_cmd_pos/*``.

Shared conventions carried over unmodified from ``ftr_policy_inference_module.py``
(the marv_rl module, already live-verified in this deployment):
  * heightmap is cropped to FTR's 45x21 window (0.05 m/cell, +/-1.10 m x
    +/-0.50 m about the robot) before use -- see ftr_heightmap_window;
  * flipper joint angles read from ROS are converted to the FTR/logical sign
    convention via ``thetas * [-1, -1, 1, 1]`` (front pair sign-flipped) before
    they're fed to any observation or position-error computation — this
    mirrors ``FtrWheelArticulation.get_all_flipper_positions()``.

That conversion is now applied in BOTH directions. The two frames disagree only
on the front pair:

    FTR-Benchmark (what every policy in this file was trained in, and what its
    action therefore means):  positive = UP for all four flippers.
    ROS / Gazebo (``/flippers_cmd_pos/*``, ``/joint_states``, and this project's
    documented convention in flipper_training/CLAUDE.md):
        front: negative = up, positive = down
        rear:  positive = up, negative = down

Until 2026-08-18 the flip was applied on the way IN but not on the way OUT, so
every commanded FRONT flipper target left this file with the wrong sign — a
policy asking for "front up" drove the front pair down by the same amount, and
vice versa (the rear pair, whose sign agrees between the frames, was correct).
Both the commanded targets and the ``flipper_limits_rad`` each module exposes
(which ``flipper_policy_node`` compares against raw ROS ``/joint_states``
angles) are converted by ``_ros_frame_flippers`` / ``_ros_frame_limits`` below.

Which modules need the outward flip is NOT uniform, because the underlying
impls differ — check before adding a new one:
  * hfc, mitriakov      -- targets built inline here in FTR frame  -> flip needed
  * creps               -- CREPSPolicyInferenceModule returns FTR frame
                           (``front_target`` straight from the position map)
                                                                   -> flip needed
  * atd3qn, icmd3qn     -- D3QNPolicyInferenceModule ALREADY converts back
                           (``fl = fr = -front_target``)           -> NO flip
``flipper_limits_rad`` is a separate question and needs converting for all four:
every one of them builds it from FTR-frame front_low/front_high bounds.

NOTE: none of this applies to ``ftr_policy_inference_module.py`` (the ``marv_rl``
module). That path deliberately applies no sign flip in either direction and
documents its own live Gazebo verification — it is not touched here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.data import Bounded
from torchrl.envs.utils import ExplorationType, set_exploration_type

# NOTE: flipper_training/__init__.py (imported next) patches
# OmegaConf.register_new_resolver to tolerate MARV_RL's marv_rl_training package
# re-registering the same resolver names -- see that file's comment. Importing
# flipper_training here, before anything that might transitively import
# marv_rl_training/rl_modules, is required for that patch to be in place in time.
import flipper_training  # noqa: F401 -- registers OmegaConf resolvers
from flipper_training.experiments.ppo.ftr_heightmap_window import ftr_heightmap_window
from flipper_training.utils.logutils import get_terminal_logger

# --------------------------------------------------------------------------- #
# Shared constants / helpers
# --------------------------------------------------------------------------- #

# FTR terrain scan grid: 45 rows x 21 cols, 0.05 m/cell (same convention as
# FtrPolicyInferenceModule / ftr_env.py's calc_current_frame_height_maps()).
_HM_ROWS = 45
_HM_COLS = 21
_HM_RES = 0.05

# ftr_env.py's MARV default (env.track_wheel_radius) -- used to convert the raw
# elevation heightmap into "height relative to the track/wheel ground-contact
# plane", the reference frame atd3qn/icmd3qn/hfc/creps's terrain features use.
TRACK_WHEEL_RADIUS = 0.1165

_FTR_SIGN = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)  # [FL, FR, RL, RR]

# Default max flipper rate (rad/s) used to convert an absolute position target
# into a velocity command for the position-mode baselines (hfc/mitriakov/creps),
# which this deployment only has a velocity actuation interface for on
# /flippers_cmd_vel/*. /flippers_cmd_pos/* (a real, separately-actuated
# interface -- see marv_xu_hto/controller_base.py, marv_oehler_baseline) is used
# instead when available, so this constant only matters as an anti-windup clamp.
_DEFAULT_MAX_FLIPPER_RATE = math.radians(180.0)


def _resample_heightmap(module, heightmap: np.ndarray, extent=None) -> np.ndarray:
    """Crop the deployment elevation map onto the FTR 45x21 observation window.

    Delegates to the shared ftr_heightmap_window -- see that module for why this
    must crop rather than resize. Until 2026-08-10 this did a bare cv2.resize of
    whatever map it was handed, which fed all five baselines an 8 m x 8 m map
    squashed into the 2.25 m x 1.05 m window they were trained on.

    Also records the result on the calling module as ``last_policy_heightmap``,
    which flipper_policy_node publishes to /policy_heightmap_image -- so the RViz
    image shows the real policy input for the baselines too, not the full map.
    """
    hm = ftr_heightmap_window(heightmap, extent)
    module.last_policy_heightmap = hm
    return hm


def _terrain_profile_15(heightmap_45x21: np.ndarray, robot_world_z: float | None) -> np.ndarray:
    """Row-mean-pooled (45 -> 15) terrain profile, base-robot-frame if
    ``robot_world_z`` is given -- mirrors atd3qn_module.py/icmd3qn_module.py/
    hfc_module.py's ``calc_scanned_height_maps(base_robot_frame=True)``:
    reshape to (45, 21), subtract (robot_world_z - track_wheel_radius), then
    mean-pool each group of 3 rows AND all 21 columns -> (15,).
    """
    hm = heightmap_45x21.astype(np.float32)
    if robot_world_z is not None:
        hm = hm - (float(robot_world_z) - TRACK_WHEEL_RADIUS)
    # "n (h k) w -> n h", reduction="mean", k=3: mean over each 3-row group AND over w.
    return hm.reshape(15, 3, _HM_COLS).mean(axis=(1, 2)).astype(np.float32)


def _roll_pitch_from_ros_quat(quat: np.ndarray) -> tuple[float, float]:
    """Same formula as FtrPolicyInferenceModule._build_obs (ROS x,y,z,w convention)."""
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
    roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    return roll, pitch


def _ftr_frame_flippers(thetas: np.ndarray) -> np.ndarray:
    """Raw ROS [FL,FR,RL,RR] radians -> FTR/logical sign convention."""
    return np.asarray(thetas, dtype=np.float32) * _FTR_SIGN


def _ros_frame_flippers(targets: np.ndarray) -> np.ndarray:
    """FTR/logical [FL,FR,RL,RR] radians -> raw ROS sign convention.

    Exact inverse of _ftr_frame_flippers (the sign vector is its own inverse).
    Every policy in this file emits FTR-frame angles, where positive = up for
    all four flippers; ROS wants front-positive = DOWN. Without this the front
    pair is commanded backwards. See the module docstring.
    """
    return np.asarray(targets, dtype=np.float32) * _FTR_SIGN


def _ros_frame_limits(limits_ftr: np.ndarray) -> np.ndarray:
    """(4, 2) [low, high] angle limits, FTR frame -> raw ROS frame.

    Negating a front row also REVERSES it: FTR [-front_up, +front_down] becomes
    ROS [-front_down, +front_up]. Swapping is not optional — flipper_policy_node
    reads these as (low, high) and would clamp against an inverted interval
    otherwise. Rear rows are unchanged (both frames agree there).
    """
    out = np.asarray(limits_ftr, dtype=np.float64).copy()
    out[:2] = -out[:2, ::-1]  # front pair: negate and swap low/high
    return out


def _position_target(cmd: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse of ftr_env.py's position-control mapping: target = low + unit*(high-low),
    unit = (clamp(cmd,-1,1)+1)/2."""
    unit = (np.clip(cmd, -1.0, 1.0) + 1.0) / 2.0
    return low + unit * (high - low)


def _flipper_bounds_rad(front_up_deg: float, front_down_deg: float, back_up_deg: float, back_down_deg: float):
    """(front_low, front_high, rear_low, rear_high) in radians, ftr_env.py's convention."""
    front_low = -math.radians(front_up_deg)
    front_high = math.radians(front_down_deg)
    rear_low = -math.radians(back_down_deg)
    rear_high = math.radians(back_up_deg)
    return front_low, front_high, rear_low, rear_high


@dataclass
class RLBaselineAction:
    """Uniform output contract for all rl_baselines inference modules."""

    v: float
    w: float
    flipper_mode: str  # "velocity" (rad/s, /flippers_cmd_vel) or "position" (rad, /flippers_cmd_pos)
    flippers: np.ndarray  # (4,) [FL, FR, RL, RR], RAW ROS sign convention (front
    # positive = down, rear positive = up) -- i.e. already passed through
    # _ros_frame_flippers, ready to publish unmodified. See the module docstring.


class _MinimalActionSpecEnv:
    """Minimal env stub exposing only what these policy `.create()`/constructors read:
    ``batch_size``, ``device``, ``action_spec``. No Isaac Sim, no observations list --
    every rl_baselines observation is fully self-describing (fixed dim), unlike the
    generic FtrFlatObservation path FtrPolicyInferenceModule stubs with real
    Observation objects.
    """

    def __init__(self, num_actions: int, device: torch.device):
        self.batch_size = torch.Size([1])
        self.device = device
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(1, num_actions), device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# atd3qn / icmd3qn -- discrete D3QN family (increment-mode, sync front/rear pairs)
#
# Thin RLBaselineAction adapters. The actual observation/action logic (position-mode
# targets integrated onto the live measured angle, terrain bands, front-sign
# inversion, heightmap-window cropping) lives ONLY in MARV_RL now --
# marv_rl_training.training.d3qn_policy_inference_module.D3QNPolicyInferenceModule --
# so it isn't duplicated and can't drift between the two repos again. That module
# auto-detects atd3qn vs icmd3qn from env_cfg_overrides.module_name itself, so one
# adapter class serves both entries in MODULE_NAME_TO_CLASS below.
# --------------------------------------------------------------------------- #


class ATD3QNInferenceModule:
    """Adapter around MARV_RL's D3QNPolicyInferenceModule (handles both atd3qn and
    icmd3qn -- see module docstring above)."""

    def __init__(self, config_path: str | Path, policy_weights_path: str | Path, device: str = "cpu"):
        from marv_rl_training.training.d3qn_policy_inference_module import D3QNPolicyInferenceModule

        self._impl = D3QNPolicyInferenceModule(config_path=config_path, policy_weights_path=policy_weights_path, device=device)
        self.flipper_limits_rad = _ros_frame_limits(np.array(
            [[self._impl.front_low, self._impl.front_high], [self._impl.front_low, self._impl.front_high],
             [self._impl.rear_low, self._impl.rear_high], [self._impl.rear_low, self._impl.rear_high]], dtype=np.float64))
        self.last_policy_heightmap = None

    def reset(self):
        self._impl.reset()

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent,
        goal_vec_local,
        xd_local,
        omega_local,
        thetas: np.ndarray,
        quat: np.ndarray,
        robot_world_z: float | None,
        dt: float,
    ) -> RLBaselineAction:
        action = self._impl.infer_action(
            heightmap=heightmap, heightmap_extent=heightmap_extent, thetas=thetas,
            robot_z=robot_world_z if robot_world_z is not None else 0.0, quat=quat,
        )
        self.last_policy_heightmap = self._impl.last_policy_heightmap
        # NO _ros_frame_flippers here, unlike every other module in this file:
        # D3QNPolicyInferenceModule.infer_action already returns raw ROS-frame
        # angles (it ends with `fl = fr = -front_target`, "Back to raw Gazebo
        # joint convention"). Applying the flip again would re-invert the front
        # pair. Its front_low/front_high, however, ARE still FTR-frame, hence the
        # _ros_frame_limits above.
        return RLBaselineAction(
            v=float(action[0]), w=float(action[1]), flipper_mode="position",
            flippers=np.asarray(action[2:6], dtype=np.float32),
        )


class ICMD3QNInferenceModule(ATD3QNInferenceModule):
    """Same adapter as ATD3QNInferenceModule -- D3QNPolicyInferenceModule branches
    on env_cfg_overrides.module_name internally, no separate logic needed here."""


# --------------------------------------------------------------------------- #
# hfc -- PPO actor/critic, own PolicyConfig, stateful SDSM belief
# --------------------------------------------------------------------------- #


class HFCInferenceModule:
    """Hybrid Flipper Controller (Azayev & Zimmermann 2022). Uses the real
    HFCPolicyConfig.create() factory (torchrl ProbabilisticActor over TanhNormal) --
    only the observation vector needs to be hand-built. Stateful: HFCSDSMBelief's
    belief buffer must be carried across calls (persistent=False in the checkpoint) --
    reset() re-seeds it to one-hot NEUTRAL, matching env reset semantics.
    """

    def __init__(self, config_path: str | Path, policy_weights_path: str | Path, device: str = "cpu"):
        self.logger = get_terminal_logger("HFCInferenceModule")
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(config_path)

        from rl_modules.hfc.hfc_policy import HFCPolicyConfig
        from marv_rl_training.environment.ftr_env_adapter import OBS_KEY

        self._obs_key = OBS_KEY
        policy_opts = dict(self.cfg.policy_opts)
        self.track_vel = float(policy_opts.get("track_vel", self.cfg.env_cfg_overrides.get("fixed_forward_vel", 0.6)))
        policy_cfg = HFCPolicyConfig(**policy_opts)

        fake_env = _MinimalActionSpecEnv(num_actions=6, device=self.device)
        actor_value_wrapper, _, _ = policy_cfg.create(env=fake_env, weights_path=str(policy_weights_path), device=self.device)
        self.actor = actor_value_wrapper.get_policy_operator().eval()

        eco = self.cfg.env_cfg_overrides
        self.front_low, self.front_high, self.rear_low, self.rear_high = _flipper_bounds_rad(
            eco["marv_flipper_front_up_deg"], eco["marv_flipper_front_down_deg"],
            eco["marv_flipper_back_up_deg"], eco["marv_flipper_back_down_deg"],
        )
        # Same bounds in the per-corner [FL, FR, RL, RR] x [low, high] form
        # flipper_policy_node clamps its published commands against, so the
        # deployed travel matches the trained travel instead of a hardcoded +/-90.
        self.flipper_limits_rad = _ros_frame_limits(np.array(
            [[self.front_low, self.front_high], [self.front_low, self.front_high],
             [self.rear_low, self.rear_high], [self.rear_low, self.rear_high]], dtype=np.float64))
        self._fresh = True  # next infer_action() call is the first of an episode
        self.logger.info(f"HFCInferenceModule ready -- device={device}, track_vel={self.track_vel}")

    def reset(self):
        self._fresh = True

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent,
        goal_vec_local,
        xd_local,
        omega_local,
        thetas: np.ndarray,
        quat: np.ndarray,
        robot_world_z: float | None,
        dt: float,
    ) -> RLBaselineAction:
        hm = _resample_heightmap(self, heightmap, heightmap_extent)
        terrain = _terrain_profile_15(hm, robot_world_z)
        flippers_ftr = _ftr_frame_flippers(thetas)  # raw rad, FTR sign convention, [FL,FR,RL,RR]
        roll, pitch = _roll_pitch_from_ros_quat(quat)
        roll_rate = float(np.asarray(omega_local, dtype=np.float32)[0])
        fwd_vel = float(np.asarray(xd_local, dtype=np.float32)[0])
        reset_flag = 1.0 if self._fresh else 0.0
        self._fresh = False

        obs = np.concatenate([
            terrain,
            flippers_ftr,
            np.array([roll, roll_rate, pitch, fwd_vel, reset_flag], dtype=np.float32),
        ])  # 15+4+1+1+1+1+1 = 24
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        from tensordict import TensorDict

        td = TensorDict({self._obs_key: obs_t}, batch_size=[1], device=self.device)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = self.actor(td)
        action = td["action"].squeeze(0).detach().cpu().numpy()  # [v,w,FL,FR,RL,RR] in [-1,1]

        low = np.array([self.front_low, self.front_low, self.rear_low, self.rear_low], dtype=np.float32)
        high = np.array([self.front_high, self.front_high, self.rear_high, self.rear_high], dtype=np.float32)
        targets = _position_target(action[2:6], low, high)
        return RLBaselineAction(v=self.track_vel, w=0.0, flipper_mode="position",
                                flippers=_ros_frame_flippers(targets))


# --------------------------------------------------------------------------- #
# mitriakov -- PPO actor/critic, own PolicyConfig; observation needs stair-edge
# geometry that isn't available from raw ROS sensors -- best-effort heuristic
# --------------------------------------------------------------------------- #


class MitriakovInferenceModule:
    """Mitriakov et al. 2021 staircase-negotiation controller.

    DISCLOSED APPROXIMATION: the paper's/training observation's p_x/p_y front/rear
    features are offsets to the nearest staircase step edge, computed during training
    from ground-truth analytic staircase geometry (mitriakov_module.yaml's per-terrain
    lookup tables) -- there is no step-edge perception anywhere in this codebase, and
    none is reconstructable from goal_vec/odom/quat alone. Here they are instead
    estimated from the live elevation heightmap itself: the along-track (row) height
    profile at the robot's centerline is scanned for the largest forward/backward
    height jumps, used as a stand-in "next/previous edge" (x = along-track distance,
    y = height jump). This is NOT what the network was trained on and is a materially
    weaker signal (single-frame local geometry vs known global stair geometry) --
    treat this baseline's live behaviour as indicative only, consistent with this
    project's "reasonable-approximation, disclosed" baseline policy (see
    BASELINE_AUDIT.md).
    """

    def __init__(self, config_path: str | Path, policy_weights_path: str | Path, device: str = "cpu"):
        self.logger = get_terminal_logger("MitriakovInferenceModule")
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(config_path)

        from rl_modules.mitriakov.mitriakov_policy import MitriakovPolicyConfig
        from marv_rl_training.environment.ftr_env_adapter import OBS_KEY

        self._obs_key = OBS_KEY
        policy_opts = dict(self.cfg.policy_opts)
        policy_cfg = MitriakovPolicyConfig(**policy_opts)

        class _FakeObs:
            dim = 8

        fake_env = _MinimalActionSpecEnv(num_actions=4, device=self.device)
        fake_env.observations = [_FakeObs()]
        actor_value_wrapper, _, _ = policy_cfg.create(env=fake_env, weights_path=str(policy_weights_path), device=self.device)
        self.actor = actor_value_wrapper.get_policy_operator().eval()

        eco = self.cfg.env_cfg_overrides
        self.front_low, self.front_high, self.rear_low, self.rear_high = _flipper_bounds_rad(
            eco["marv_flipper_front_up_deg"], eco["marv_flipper_front_down_deg"],
            eco["marv_flipper_back_up_deg"], eco["marv_flipper_back_down_deg"],
        )
        # Same bounds in the per-corner [FL, FR, RL, RR] x [low, high] form
        # flipper_policy_node clamps its published commands against, so the
        # deployed travel matches the trained travel instead of a hardcoded +/-90.
        self.flipper_limits_rad = _ros_frame_limits(np.array(
            [[self.front_low, self.front_high], [self.front_low, self.front_high],
             [self.rear_low, self.rear_high], [self.rear_low, self.rear_high]], dtype=np.float64))
        self.logger.info(f"MitriakovInferenceModule ready -- device={device} (heuristic stair-edge obs, see class docstring)")

    def reset(self):
        pass

    @staticmethod
    def _edge_offsets(heightmap_45x21: np.ndarray) -> tuple[float, float, float, float, bool]:
        """Along-centerline row profile -> heuristic (px_front, py_front, px_rear,
        py_rear, is_ascending). Row 0 = far front, row 44 = far back (matches the
        established heightmap orientation, CLAUDE.md: "objects in front are in the
        upper part"); center row = robot position; col = center column.
        """
        col = _HM_COLS // 2
        center_row = _HM_ROWS // 2
        profile = heightmap_45x21[:, col]  # (45,)
        ahead = profile[:center_row]  # rows in front, far->near
        behind = profile[center_row + 1:]  # rows behind, near->far

        # Largest |height jump| ahead / behind -> treated as the nearest step edge.
        def _nearest_edge(segment: np.ndarray, rows_from_center: np.ndarray) -> tuple[float, float]:
            if segment.shape[0] < 2:
                return 0.0, 0.0
            jumps = np.abs(np.diff(segment))
            if jumps.size == 0 or jumps.max() < 1e-3:
                return float(rows_from_center[-1] * _HM_RES), 0.0
            k = int(np.argmax(jumps))
            dx = float(rows_from_center[k] * _HM_RES)
            dy = float(segment[k + 1] - segment[0])
            return dx, dy

        ahead_rows = np.arange(center_row, 0, -1)[: ahead.shape[0]]
        behind_rows = np.arange(1, behind.shape[0] + 1)
        px_front, py_front = _nearest_edge(ahead[::-1], ahead_rows[::-1])
        px_rear, py_rear = _nearest_edge(behind, behind_rows)

        is_ascending = py_front > 0.0
        return px_front, py_front, -px_rear, py_rear, is_ascending

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent,
        goal_vec_local,
        xd_local,
        omega_local,
        thetas: np.ndarray,
        quat: np.ndarray,
        robot_world_z: float | None,
        dt: float,
    ) -> RLBaselineAction:
        hm = _resample_heightmap(self, heightmap, heightmap_extent)
        px_f, py_f, px_r, py_r, is_ascending = self._edge_offsets(hm)
        v_s = float(np.asarray(xd_local, dtype=np.float32)[0])
        flippers_ftr = _ftr_frame_flippers(thetas)
        psi_front = float(flippers_ftr[0])  # front pair representative angle
        psi_rear = float(flippers_ftr[2])  # rear pair representative angle

        obs = np.array(
            [px_f, py_f, px_r, py_r, v_s, psi_front, psi_rear, 1.0 if is_ascending else 0.0],
            dtype=np.float32,
        )
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        from tensordict import TensorDict

        td = TensorDict({self._obs_key: obs_t}, batch_size=[1], device=self.device)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = self.actor(td)
        action = td["action"].squeeze(0).detach().cpu().numpy()  # [v,w,front,rear] in [-1,1]

        low = np.array([self.front_low, self.rear_low], dtype=np.float32)
        high = np.array([self.front_high, self.rear_high], dtype=np.float32)
        fr = _position_target(action[2:4], low, high)
        flippers = np.array([fr[0], fr[0], fr[1], fr[1]], dtype=np.float32)  # sync -> 4 corners
        return RLBaselineAction(v=self.cfg.env_cfg_overrides.get("fixed_forward_vel", 0.6), w=0.0,
                                 flipper_mode="position", flippers=_ros_frame_flippers(flippers))


# --------------------------------------------------------------------------- #
# creps -- deterministic 6-parameter linear controller (no neural net)
#
# Thin RLBaselineAction adapter -- the actual law (front/rear linear-in-state
# computation, height_ahead_row derivation, heightmap-window cropping) lives ONLY
# in MARV_RL now: marv_rl_training.training.creps_policy_inference_module.
# CREPSPolicyInferenceModule. See that module for the omega/row logic and the
# last_diag snapshot this adapter forwards to flipper_policy_node's RViz markers.
# --------------------------------------------------------------------------- #


class CREPSInferenceModule:
    def __init__(self, config_path: str | Path, policy_weights_path: str | Path, device: str = "cpu"):
        from marv_rl_training.training.creps_policy_inference_module import CREPSPolicyInferenceModule

        self._impl = CREPSPolicyInferenceModule(config_path=config_path, policy_weights_path=policy_weights_path, device=device)
        self.flipper_limits_rad = _ros_frame_limits(np.array(
            [[self._impl.front_low, self._impl.front_high], [self._impl.front_low, self._impl.front_high],
             [self._impl.rear_low, self._impl.rear_high], [self._impl.rear_low, self._impl.rear_high]], dtype=np.float64))
        self.last_policy_heightmap = None
        self.last_diag = None

    def reset(self):
        self._impl.reset()

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent,
        goal_vec_local,
        xd_local,
        omega_local,
        thetas: np.ndarray,
        quat: np.ndarray,
        robot_world_z: float | None,
        dt: float,
    ) -> RLBaselineAction:
        action = self._impl.infer_action(
            heightmap=heightmap, heightmap_extent=heightmap_extent,
            robot_z=robot_world_z if robot_world_z is not None else 0.0, quat=quat,
        )
        self.last_policy_heightmap = self._impl.last_policy_heightmap
        self.last_diag = self._impl.last_diag
        return RLBaselineAction(
            v=float(action[0]), w=float(action[1]), flipper_mode="position",
            flippers=_ros_frame_flippers(action[2:6]),
        )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

MODULE_NAME_TO_CLASS = {
    "atd3qn": ATD3QNInferenceModule,
    "icmd3qn": ICMD3QNInferenceModule,
    "hfc": HFCInferenceModule,
    "mitriakov": MitriakovInferenceModule,
    "creps": CREPSInferenceModule,
}


def detect_rl_baseline_module_name(config_path: str | Path) -> str | None:
    """Return env_cfg_overrides.module_name if it names one of the rl_baselines
    modules handled by this file, else None (caller falls back to marv_rl/FTR/native)."""
    try:
        cfg = OmegaConf.load(config_path)
        name = cfg.get("env_cfg_overrides", {}).get("module_name")
        return name if name in MODULE_NAME_TO_CLASS else None
    except Exception:
        return None


def build_rl_baseline_inference_module(module_name: str, config_path: str | Path, policy_weights_path: str | Path, device: str = "cpu"):
    cls = MODULE_NAME_TO_CLASS[module_name]
    return cls(config_path=config_path, policy_weights_path=policy_weights_path, device=device)
