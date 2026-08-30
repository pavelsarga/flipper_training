"""Inference module for the C-TRAC flipper controller (Pan et al. 2025, "C-TRAC:
Terrain-Adaptive Control for Articulated Tracked Robots via Contact-Aware
Reinforcement Learning").

Asymmetric SAC actor with an embedded C-VAE (rl_modules/ctrac/ctrac_policy.py's
CTRACActorNet): pi(a | o_t, z_t, c~_t, c~prob_t). Only the ACTOR runs here — the
privileged critic and the ground-truth contact extractor are training-only, and the
C-VAE's whole purpose is to let the actor estimate contact from partial observations
alone, which is exactly the deployment case.

Three things about this checkpoint that differ from the other baselines:

  * **cvae_weights_path is forced to None.** The C-VAE lives INSIDE CTRACActorNet, so
    policy_final.pth already contains it (22 of its 28 tensors are `...cvae.*`).
    The config's Stage I path is a training-time warm start, relative to the MARV_RL
    checkout, and would either fail to resolve here or overwrite the trained weights
    with the pretrained ones.
  * **The actor is stateful.** CTRACObsHistory is a per-env ring buffer of the last
    `history_len` partial-observation frames (o^H_t, Eq. 9) and is non-persistent, so
    it is not in the checkpoint. Built with num_envs=1 it is exactly right for a live
    single-robot rollout: pass obs_history=None and let the buffer feed itself. It
    re-seeds from the current frame whenever the observation's reset flag is 1.0,
    which is what reset() arms.
  * **The action is a flipper VELOCITY command** (`flipper_control_mode: velocity`),
    not a position target — unlike hfc/creps/mitriakov. flipper_policy_node's FTR
    branch already implements exactly this actuation
    (`action[2:6] * [-1,-1,1,1] * deg2rad(5) / dt`, matching ftr_env.py's
    `flipper_delta = flipper_cmd * flipper_dt` with flipper_dt=5 deg/step), so this
    module returns the RAW [-1, 1] policy output and lets the node convert it.

Observation (Eq. 1's partial obs o_t, 251-D — see rl_modules/ctrac/ctrac_observation.py):
    fwd_vel (1) | flipper angles, PAPER order [FL,RL,RR,FR] (4) | roll, pitch, yaw (3)
    | goal-relative XY, body frame, raw metres (2) | local heightmap 12x20 covering
    robot-frame x in [0.4, 1.0] m ahead, y in [-0.5, 0.5] (240) | reset flag (1)

The privileged slice (976 more columns) is zero-filled: CTRACActorNet slices
`obs[..., :PARTIAL_DIM]` and never reads past it. Only the critic does, and the critic
is not built here.
"""

import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from marv_rl_training.training.ftr_heightmap_window import HM_COLS, HM_RES, HM_ROWS, ftr_heightmap_window
from marv_rl_training.utils.logutils import get_terminal_logger

# ftr_env.py's MARV default (env.track_wheel_radius): converts the raw elevation map into
# "height relative to the track/wheel ground-contact plane", the reference frame
# ctrac_module.calc_scanned_height_maps(base_robot_frame=True) uses.
TRACK_WHEEL_RADIUS = 0.1165

# Raw ROS [FL,FR,RL,RR] radians -> FTR/logical sign convention (front positive = up).
_FTR_SIGN = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)

# rl_modules/ctrac/ctrac_module.py's local-heightmap window (Eq. 1's h^l_t).
_LOCAL_ROWS, _LOCAL_COLS = 12, 20
_LOCAL_X_LO, _LOCAL_X_HI = 0.4, 1.0
_LOCAL_Y_LO = -0.5


def _crop_local_window(hm: np.ndarray) -> np.ndarray:
    """Port of ctrac_module._crop_and_pad for the local window, NumPy and 2-D.

    Kept in step with that function deliberately rather than imported: ctrac_module.py
    does `from omni.isaac.lab.envs import VecEnvObs` at module scope, so it cannot be
    imported outside Isaac Sim — which is the entire point of this file.

    ROW INDEX DECREASES TOWARD THE FRONT (ftr_env.calc_current_frame_height_maps stores
    `local_map.flip(0)`, and ftr_heightmap_window reproduces that convention), so
    row = center_row - x/res. Getting this backwards silently samples the MIRROR of the
    requested range — the strip BEHIND the robot — which is a bug this project has
    already shipped once; see the _crop_and_pad docstring.
    """
    h, w = hm.shape
    center_row, center_col = h // 2, w // 2
    row_lo = center_row - round(_LOCAL_X_HI / HM_RES)
    row_hi = center_row - round(_LOCAL_X_LO / HM_RES)
    col_lo = center_col + round(_LOCAL_Y_LO / HM_RES)
    col_hi = col_lo + _LOCAL_COLS

    clamp_row_lo, clamp_row_hi = max(row_lo, 0), min(row_hi, h)
    clamp_col_lo, clamp_col_hi = max(col_lo, 0), min(col_hi, w)
    crop = hm[clamp_row_lo:clamp_row_hi, clamp_col_lo:clamp_col_hi]

    pad_top, pad_left = clamp_row_lo - row_lo, clamp_col_lo - col_lo
    pad_bottom = _LOCAL_ROWS - crop.shape[0] - pad_top
    pad_right = _LOCAL_COLS - crop.shape[1] - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((max(pad_top, 0), max(pad_bottom, 0)), (max(pad_left, 0), max(pad_right, 0))),
            mode="edge",
        )
    return crop[:_LOCAL_ROWS, :_LOCAL_COLS].astype(np.float32)


def _rpy_from_ros_quat(quat: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) radians from a ROS (x, y, z, w) quaternion. Same roll/pitch
    formula as FtrPolicyInferenceModule._build_obs, with yaw added."""
    qx, qy, qz, qw = (float(v) for v in quat)
    pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
    roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return roll, pitch, yaw


class _ObsDimStub:
    def __init__(self, dim: int):
        self.dim = dim


class _MinimalEnv:
    """Only what CTRACPolicyConfig.create() reads: batch_size, device, action_spec and
    observations[0].dim (the latter sizes the critic, which is built and discarded)."""

    def __init__(self, num_actions: int, obs_dim: int, device: torch.device):
        from torchrl.data import Bounded

        self.batch_size = torch.Size([1])
        self.device = device
        self.observations = [_ObsDimStub(obs_dim)]
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(1, num_actions), device=device, dtype=torch.float32)


class CTRACPolicyInferenceModule:
    """Inference module for a ctrac checkpoint (train_sac.py / Isaac Sim).

    Usage::

        module = CTRACPolicyInferenceModule(
            config_path="rl_baselines/ctrac/config.yaml",
            policy_weights_path="rl_baselines/ctrac/weights/policy_final.pth",
        )
        action = module.infer_action(
            heightmap=hm, heightmap_extent=extent, goal_vec_local=goal,
            xd_local=xd, omega_local=om, thetas=thetas, quat=quat, robot_z=z,
        )
        # action: (6,) float32 [v, w, fl, fr, rl, rr] — v in m/s, w always 0.0, and
        # fl..rr the RAW [-1, 1] flipper velocity commands in FTR order [FL,FR,RL,RR].
    """

    def __init__(self, config_path: str | Path, policy_weights_path: str | Path, device: str = "cpu", **_ignored):
        self.logger = get_terminal_logger("CTRACPolicyInferenceModule")
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(config_path)

        from rl_modules.ctrac.ctrac_observation import PARTIAL_DIM, TOTAL_DIM
        from rl_modules.ctrac.ctrac_policy import CTRACPolicyConfig
        from marv_rl_training.environment.ftr_env_adapter import OBS_KEY

        self._obs_key = OBS_KEY
        self._partial_dim = PARTIAL_DIM
        self._total_dim = TOTAL_DIM

        policy_opts = OmegaConf.to_container(self.cfg.policy_opts, resolve=True)
        # See the module docstring: the trained checkpoint already carries the C-VAE.
        policy_opts["cvae_weights_path"] = None
        policy_cfg = CTRACPolicyConfig(**policy_opts)

        fake_env = _MinimalEnv(num_actions=6, obs_dim=TOTAL_DIM, device=self.device)
        policy_operator, _qvalue, _cvae, _groups = policy_cfg.create(
            env=fake_env, weights_path=str(policy_weights_path), device=self.device
        )
        self.actor = policy_operator.eval()
        # sample=self.training inside CTRACActorNet: eval() makes the C-VAE use mu
        # rather than a reparameterised draw, so inference is deterministic.

        eco = self.cfg.env_cfg_overrides
        front_up, front_down = float(eco["marv_flipper_front_up_deg"]), float(eco["marv_flipper_front_down_deg"])
        back_up, back_down = float(eco["marv_flipper_back_up_deg"]), float(eco["marv_flipper_back_down_deg"])
        # ftr_env.flipper_angle_bounds() convention, FTR frame, per corner [FL,FR,RL,RR].
        self.flipper_limits_rad = np.array(
            [
                [-math.radians(front_up), math.radians(front_down)],
                [-math.radians(front_up), math.radians(front_down)],
                [-math.radians(back_down), math.radians(back_up)],
                [-math.radians(back_down), math.radians(back_up)],
            ],
            dtype=np.float64,
        )
        self.track_vel_max = float(self.cfg.get("track_vel_max", 0.7))

        self.last_policy_heightmap = None
        self._fresh = True
        self.logger.info(
            f"CTRACPolicyInferenceModule ready — device={device}, history_len={policy_opts['history_len']}, "
            f"latent_dim={policy_opts['latent_dim']}, track_vel_max={self.track_vel_max}"
        )

    def reset(self):
        """Arm the next infer_action() as the first frame of an episode, so
        CTRACObsHistory re-seeds its window from that frame instead of carrying the
        previous episode's history across the discontinuity."""
        self._fresh = True

    def _build_partial_obs(self, heightmap, heightmap_extent, goal_vec_local, xd_local, thetas, quat, robot_z):
        hm45 = ftr_heightmap_window(heightmap, heightmap_extent)  # (45, 21), FTR convention
        self.last_policy_heightmap = hm45
        # base_robot_frame=True, same reference every other module's terrain feature uses.
        hm45 = hm45.astype(np.float32) - (float(robot_z) - TRACK_WHEEL_RADIUS)
        local_hmap = _crop_local_window(hm45)  # (12, 20)

        flippers_ftr = np.asarray(thetas, dtype=np.float32) * _FTR_SIGN  # [FL,FR,RL,RR]
        # ctrac_module.get_observations reorders to the paper's [FL,RL,RR,FR].
        flippers_paper = flippers_ftr[[0, 2, 3, 1]]

        roll, pitch, yaw = _rpy_from_ros_quat(quat)

        # goal_vec_local arrives de-rotated by the FULL orientation (flipper_policy_node's
        # `rot.inv().apply(goal_vec_world)`), but training de-rotates by YAW ONLY. The two
        # differ by cos(pitch) on the x component — 13% at the 30 deg pitch this robot
        # reaches mid-climb — so rotate back to world and redo it the training way.
        from scipy.spatial.transform import Rotation

        goal_world = Rotation.from_quat(np.asarray(quat, dtype=np.float64)).apply(
            np.asarray(goal_vec_local, dtype=np.float64)
        )
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        goal_bx = cos_y * goal_world[0] + sin_y * goal_world[1]
        goal_by = -sin_y * goal_world[0] + cos_y * goal_world[1]

        # Training feeds the WORLD yaw here, but every training config sets
        # spawn_yaw_range: 0.0 on lanes running along world +X, so the policy only ever
        # saw yaw ~= 0 plus drift — and with the goal straight ahead, world yaw equals
        # -atan2(goal_by, goal_bx) exactly. Deployment's world frame is arbitrary, so
        # feeding the raw odom yaw would be badly out of distribution. Feeding the
        # goal-relative heading instead reproduces the quantity the policy was trained
        # on, in a frame that does not depend on where the map's origin happens to be.
        yaw_obs = -math.atan2(goal_by, goal_bx)

        fwd_vel = float(np.asarray(xd_local, dtype=np.float32)[0])
        reset_flag = 1.0 if self._fresh else 0.0
        self._fresh = False

        partial = np.concatenate(
            [
                np.array([fwd_vel], dtype=np.float32),
                flippers_paper.astype(np.float32),
                np.array([roll, pitch, yaw_obs], dtype=np.float32),
                np.array([goal_bx, goal_by], dtype=np.float32),
                local_hmap.reshape(-1),
                np.array([reset_flag], dtype=np.float32),
            ]
        )
        if partial.shape[0] != self._partial_dim:
            raise RuntimeError(f"ctrac partial obs is {partial.shape[0]}-D, expected {self._partial_dim}")
        return partial

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent=None,
        goal_vec_local=None,
        xd_local=None,
        omega_local=None,
        thetas: np.ndarray = None,
        quat: np.ndarray = None,
        robot_z: float = 0.0,
        **_ignored,
    ) -> np.ndarray:
        partial = self._build_partial_obs(heightmap, heightmap_extent, goal_vec_local, xd_local, thetas, quat, robot_z)

        # Privileged slice zero-filled — CTRACActorNet reads obs[..., :PARTIAL_DIM] only.
        obs = np.zeros(self._total_dim, dtype=np.float32)
        obs[: self._partial_dim] = partial
        obs_t = torch.from_numpy(obs).to(self.device).unsqueeze(0)

        from tensordict import TensorDict
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        td = TensorDict({self._obs_key: obs_t}, batch_size=[1], device=self.device)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
            td = self.actor(td)
        action = td["action"].squeeze(0).detach().cpu().numpy()  # [v, w, FL, FR, RL, RR]

        v = float(np.clip(action[0], -self.track_vel_max, self.track_vel_max))
        # w is pinned to ~0 by CTRACActorNet (the paper's task is straight-path only) and
        # this platform has ~3.6% yaw authority anyway — see CLAUDE.md's skid-steer note.
        return np.concatenate([[v, 0.0], action[2:6]]).astype(np.float32)
