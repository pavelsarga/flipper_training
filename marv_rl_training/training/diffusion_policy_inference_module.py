"""Inference module for receding-horizon (Phase 1 diffusion-style) policies trained with
train_diffusion.py — for the ROS deployment node, no Isaac Sim at runtime.

What differs from FtrPolicyInferenceModule, and why each piece is here:

* The policy reads an OBSERVATION HISTORY, not one observation. Training stacks the last
  T_o observations with torchrl CatFrames (dim=-1, padding="same") placed AFTER VecNorm, on
  the ActionChunkEnv — so the frames are the normalised observations at the RE-PLAN points
  (one per macro step, i.e. every T_a control steps), oldest first, and right after a reset
  every slot holds the first observation. This module keeps exactly that window.
* The policy outputs a CHUNK of T_p actions and only the first T_a are executed open-loop
  before it re-plans. infer_action() is still called once per control tick (the node's
  timer), and returns the next action of the current chunk; it only builds an observation
  and runs the network on re-plan ticks. Sensor data passed on the other ticks is ignored,
  as it is in training.
* The per-step 966-D observation itself is identical to the single-step marv_rl policy, so
  it is built by FtrPolicyInferenceModule._build_obs (heightmap window, lateral mirror,
  flipper sign convention, goal/velocity scaling). The "previous action" slot is the action
  executed on the previous control tick — what CrossingEnv.last_action holds when the
  chunked env hands back the observation at the end of a macro step.
* The diffusion configs run flipper_control_mode: position — the flipper outputs are
  ABSOLUTE targets in [-1, 1] mapped onto the per-corner [low, high] limits, not the
  increments the PPO policies emit. flipper_joint_targets() does that map exactly as
  FtrEnv._pre_physics_step does and converts it to the ROS joint frame.

Deploy at the training control rate (sim_dt x decimation = 0.005 x 20 = 10 Hz): the chunk is
T_a consecutive control ticks, so any other rate stretches or compresses the plan in time.
"""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Bounded
from torchrl.envs import VecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type

import marv_rl_training  # noqa: F401 — registers OmegaConf resolvers
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY
from marv_rl_training.training.ftr_policy_inference_module import FtrPolicyInferenceModule
from marv_rl_training.utils.logutils import get_terminal_logger
from rl_modules.marv_rl.marv_rl_flat_observation import MarvRLFlatObservation

# FtrWheelArticulation.get_all_flipper_positions / set_all_flipper_position_targets: the
# FTR logical flipper angle is the joint angle times this, in both directions.
_FTR_TO_JOINT = np.array([-1.0, -1.0, 1.0, 1.0])


class _MinimalChunkEnv:
    """Env stub exposing what DiffusionPolicyConfig.create reads: the flattened chunk action
    spec (T_p * A, as ActionChunkEnv declares it) and the per-frame observation."""

    def __init__(self, num_actions: int, prediction_horizon: int, device: torch.device, encoder_opts: dict):
        self.batch_size = torch.Size([1])
        self.device = device
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1, prediction_horizon * num_actions), device=device, dtype=torch.float32,
        )
        # MarvRLFlatObservation sizes its prev_action slot from ftr_env.unwrapped.cfg.num_actions,
        # and falls back to action_spec without one -- which here is the whole T_p * A chunk, a
        # 48-wide slot instead of 6 that no checkpoint matches. Give it the per-step count.
        self.ftr_env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=SimpleNamespace(num_actions=num_actions)))
        self.observations = [MarvRLFlatObservation(env=self, encoder_opts=encoder_opts)]


class DiffusionPolicyInferenceModule(FtrPolicyInferenceModule):
    """Same call signature as FtrPolicyInferenceModule.infer_action, one call per control tick.

    Only __init__ and the inference loop differ; _build_obs is inherited unchanged, which is
    why __init__ sets the attributes it reads (joint_limit, track_wheel_radius, _prev_action,
    device) instead of calling the parent constructor, which would build a single-step policy.
    """

    def __init__(
        self,
        config_path: str | Path,
        policy_weights_path: str | Path,
        vecnorm_weights_path: str | Path | None = None,
        device: str = "cpu",
        num_actions: int = 6,
    ):
        self.logger = get_terminal_logger("DiffusionPolicyInferenceModule")
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(config_path)
        overrides = self.cfg.get("env_cfg_overrides", {}) or {}

        # --- attributes FtrPolicyInferenceModule._build_obs reads ---------------------
        self.joint_limit = float(np.deg2rad(overrides.get("flipper_pos_max_deg", 90.0)))
        self.track_wheel_radius = float(overrides.get("render_radius", 0.1165))
        self.num_actions = num_actions
        self._prev_action = torch.zeros(num_actions, dtype=torch.float32, device=self.device)

        # --- receding horizon ---------------------------------------------------------
        self.prediction_horizon = int(self.cfg.prediction_horizon)
        self.execution_horizon = int(self.cfg.execution_horizon)
        self.history_len = int(self.cfg.history_len)
        if not 1 <= self.execution_horizon <= self.prediction_horizon:
            raise ValueError(f"execution_horizon {self.execution_horizon} not in [1, {self.prediction_horizon}]")
        self.flipper_control_mode = str(overrides.get("flipper_control_mode", "velocity"))

        # --- policy -------------------------------------------------------------------
        fake_env = _MinimalChunkEnv(num_actions, self.prediction_horizon, self.device, self.cfg.ftr_obs_encoder_opts)
        policy_cfg = self.cfg.policy_config(**self.cfg.policy_opts)
        if int(policy_cfg.history_len) != self.history_len:
            raise ValueError(f"policy_opts.history_len {policy_cfg.history_len} != history_len {self.history_len}")
        actor_value_wrapper, _, _ = policy_cfg.create(env=fake_env, weights_path=str(policy_weights_path), device=self.device)
        self.actor = actor_value_wrapper.get_policy_operator().eval()
        self._obs_history_key = policy_cfg.obs_history_key

        # --- VecNorm on the per-frame observation (before the history, as in training) --
        self.vecnorm = VecNorm(in_keys=[OBS_KEY], **self.cfg.vecnorm_opts)
        if vecnorm_weights_path:
            # Prime the lazily-built stats tensordict before loading into it.
            self.vecnorm(TensorDict({OBS_KEY: torch.zeros(1, fake_env.observations[0].dim, device=self.device)},
                                    batch_size=[1], device=self.device))
            self.vecnorm.load_state_dict(torch.load(str(vecnorm_weights_path), map_location=self.device), strict=False)
            self.logger.info(f"Loaded VecNorm from {vecnorm_weights_path}")
        else:
            self.logger.warning("No vecnorm_weights_path provided — running without normalisation.")
        # freeze(), not just eval(): in torchrl 0.8.1 VecNorm keeps updating its running
        # statistics unless `frozen` is set (eval() does not set it), so every tick would pull
        # the training normalisation toward whatever the robot is currently seeing.
        self.vecnorm.eval()
        self.vecnorm.freeze()

        # --- flipper position-mode limits (FTR logical frame), FtrEnv.flipper_angle_bounds
        marv = [overrides.get(k) for k in ("marv_flipper_front_up_deg", "marv_flipper_front_down_deg",
                                           "marv_flipper_back_up_deg", "marv_flipper_back_down_deg")]
        if overrides.get("robot_type") == "marv" and all(v is not None for v in marv):
            fu, fd, bu, bd = (np.deg2rad(float(v)) for v in marv)
            self.flipper_low_ftr = np.array([-fu, -fu, -bd, -bd])
            self.flipper_high_ftr = np.array([fd, fd, bu, bu])
        else:
            self.flipper_low_ftr = np.full(4, -self.joint_limit)
            self.flipper_high_ftr = np.full(4, self.joint_limit)
        # Same interval in the ROS joint frame: negating a front row also swaps its ends.
        lo_j, hi_j = self.flipper_low_ftr * _FTR_TO_JOINT, self.flipper_high_ftr * _FTR_TO_JOINT
        self.flipper_limits_ros = np.stack([np.minimum(lo_j, hi_j), np.maximum(lo_j, hi_j)], axis=1)

        self._history: deque[torch.Tensor] = deque(maxlen=self.history_len)
        self._chunk: torch.Tensor | None = None  # [T_p, A]
        self._cursor = 0
        self.replanned = False  # True if the last infer_action call ran the network

        self.logger.info(
            f"DiffusionPolicyInferenceModule ready — T_o={self.history_len} T_p={self.prediction_horizon} "
            f"T_a={self.execution_horizon}, head={policy_cfg.head}, flipper_control_mode={self.flipper_control_mode}, "
            f"device={device}"
        )

    def reset(self):
        """Start a new episode: empty history, no plan, zero previous action."""
        self._prev_action.zero_()
        self._history.clear()
        self._chunk = None
        self._cursor = 0

    @property
    def chunk(self) -> np.ndarray | None:
        """The current plan, [T_p, A] (debug)."""
        return None if self._chunk is None else self._chunk.cpu().numpy()

    @property
    def cursor(self) -> int:
        """Index in the current chunk of the NEXT action to be returned."""
        return self._cursor

    def _obs_history(self, frame: torch.Tensor) -> torch.Tensor:
        """Push a normalised [1, obs_dim] frame and return the [1, T_o * obs_dim] window,
        oldest first; an empty window is filled with this frame (CatFrames padding='same')."""
        if not self._history:
            self._history.extend([frame] * self.history_len)
        else:
            self._history.append(frame)
        return torch.cat(list(self._history), dim=-1)

    def infer_action(
        self,
        heightmap: np.ndarray,
        heightmap_extent: list | None,
        goal_vec_local: np.ndarray,
        xd_local: np.ndarray,
        omega_local: np.ndarray,
        thetas: np.ndarray,
        quat: np.ndarray,
        robot_z: float = 0.0,
    ) -> np.ndarray:
        """Next control-tick action, (6,) float32 [v, w, fl, fr, rl, rr] in [-1, 1].

        Re-plans (builds the observation, runs the network) when the executed prefix of the
        current chunk is used up; otherwise returns the next planned action. Arguments are as
        in FtrPolicyInferenceModule.infer_action.
        """
        self.replanned = self._chunk is None or self._cursor >= self.execution_horizon
        if self.replanned:
            obs = self._build_obs(heightmap, heightmap_extent, goal_vec_local, xd_local, omega_local, thetas, quat, robot_z)
            td = self.vecnorm(TensorDict({OBS_KEY: obs}, batch_size=[1], device=self.device))
            td = TensorDict({self._obs_history_key: self._obs_history(td[OBS_KEY])}, batch_size=[1], device=self.device)
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
                td = self.actor(td)
            self._chunk = td["action"].reshape(self.prediction_horizon, self.num_actions).detach()
            self._cursor = 0
        action = self._chunk[self._cursor]
        self._cursor += 1
        self._prev_action = action.to(self.device).clone()
        return action.cpu().numpy()

    def flipper_joint_targets(self, action: np.ndarray) -> np.ndarray:
        """Position mode: (4,) absolute flipper joint targets [FL, FR, RL, RR] in radians, ROS
        joint frame — FtrEnv's [-1, 1] -> [low, high] map, then the FTR->joint sign flip."""
        unit = (np.clip(np.asarray(action, dtype=np.float64)[2:6], -1.0, 1.0) + 1.0) * 0.5
        return (self.flipper_low_ftr + unit * (self.flipper_high_ftr - self.flipper_low_ftr)) * _FTR_TO_JOINT
