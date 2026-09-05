"""Action-chunking wrapper: one RL step == ``execution_horizon`` control steps.

Implements the receding-horizon control scheme of Chi et al., *Diffusion Policy*: the
policy emits a ``prediction_horizon`` (T_p) step action trajectory, of which only the
first ``execution_horizon`` (T_a) steps are executed open-loop before it re-plans. The
discarded tail exists to give the network a longer context to shape the executed prefix.

The action is exposed **flattened** as ``(num_envs, T_p * action_dim)`` rather than
``(num_envs, T_p, action_dim)``. That keeps NormalParamExtractor / TanhNormal /
ProbabilisticActor / ClipPPOLoss on their ordinary code paths and makes
``sample_log_prob`` one scalar per macro-step.

Reward is aggregated as ``sum_i gamma_ctrl^i * r_i`` over the executed prefix, so a macro
MDP discounted at ``gamma_ctrl ** T_a`` has the same discounted return as the underlying
control-step MDP. The training config must therefore set ``gae_opts.gamma`` to the macro
discount while leaving ``env_cfg_overrides.shaping_gamma`` on the per-control-step value
(see docs/diffusion_policy/01_action_chunking.md).

⚠ Mid-chunk termination. FtrEnv (IsaacLab DirectRLEnv) auto-resets terminated envs inside
step(), and the env is batch-locked, so the remaining sub-steps of a chunk would drive an
already-respawned robot with actions meant for the previous episode. From the sub-step
after an env goes done we therefore feed it a NEUTRAL action (zero track velocity, flipper
command that holds the current angle) and stop accumulating its reward. The residual cost
is that a fresh episode loses up to T_a-1 control steps to idling. That is the main
correctness compromise of this design; ``T_a=1`` reproduces the un-chunked env exactly and
is the reference for measuring it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from torchrl.data import Bounded
from torchrl.envs import EnvBase

from marv_rl_training.environment.ftr_env_adapter import OBS_KEY

if TYPE_CHECKING:
    from marv_rl_training.environment.ftr_env_adapter import FtrTorchRLEnv


class ActionChunkEnv(EnvBase):
    """Wraps FtrTorchRLEnv so that one TorchRL step executes a chunk of control steps.

    Args:
        inner: The wrapped single-step environment.
        prediction_horizon: T_p — the number of action steps the policy predicts.
        execution_horizon: T_a — how many of them are executed before re-planning.
            Must satisfy ``1 <= T_a <= T_p``.
        control_gamma: The per-control-step discount used to sum rewards within a chunk.
            This is NOT the discount GAE should use; that one is ``control_gamma ** T_a``.
    """

    _batch_locked = True

    def __init__(
        self,
        inner: "FtrTorchRLEnv",
        prediction_horizon: int,
        execution_horizon: int,
        control_gamma: float,
    ):
        if not 1 <= execution_horizon <= prediction_horizon:
            raise ValueError(
                f"execution_horizon ({execution_horizon}) must be in [1, prediction_horizon ({prediction_horizon})]"
            )
        super().__init__(device=inner.device, batch_size=inner.batch_size)

        self.inner = inner
        self.prediction_horizon = int(prediction_horizon)
        self.execution_horizon = int(execution_horizon)
        self.control_gamma = float(control_gamma)

        num_envs = inner.batch_size[0]
        self.action_dim = inner.action_spec.shape[-1]

        # Flattened chunk. Bounds are the inner spec's, repeated over the horizon.
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(num_envs, self.prediction_horizon * self.action_dim),
            device=inner.device,
            dtype=torch.float32,
        )
        # Observation / reward / done shapes are unchanged — a macro step still yields one
        # observation, one reward and one set of done flags per env.
        self.observation_spec = inner.observation_spec.clone()
        self.reward_spec = inner.reward_spec.clone()
        self.done_spec = inner.done_spec.clone()

        # Discount weights for the executed prefix, precomputed on device.
        self._reward_weights = torch.tensor(
            [self.control_gamma**i for i in range(self.execution_horizon)],
            device=inner.device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Neutral action for envs that terminated part-way through a chunk
    # ------------------------------------------------------------------

    def _neutral_action(self) -> torch.Tensor:
        """Action that keeps a just-respawned robot still for the rest of the chunk.

        Track velocity and yaw rate are zero. The flipper command holds the current angle:
        in ``position`` mode that means inverting the env's ``[-1, 1] -> [low, high]`` map
        on the measured angle, in ``velocity``/``increment`` mode a zero command already
        integrates to no change.
        """
        env = self.inner.ftr_env.unwrapped
        num_envs = self.batch_size[0]
        action = torch.zeros(num_envs, self.action_dim, device=self.device, dtype=torch.float32)

        if env.cfg.flipper_control_mode != "position":
            return action

        low, high = env.flipper_angle_bounds()
        if low is None:
            return action  # flippers locked horizontal — zero is already neutral

        # Same offset _pre_physics_step uses to slice the flipper block out of the action.
        flipper_offset = 4 if env.cfg.flipper_style else 2
        span = (high - low).clamp_min(1e-6)
        unit = 2.0 * (env.flipper_positions.to(self.device) - low.to(self.device)) / span.to(self.device) - 1.0
        action[:, flipper_offset:] = unit.clamp(-1.0, 1.0).to(action.dtype)
        return action

    # ------------------------------------------------------------------
    # EnvBase interface
    # ------------------------------------------------------------------

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        return self.inner._reset(tensordict)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        num_envs = self.batch_size[0]
        chunk = tensordict["action"].reshape(num_envs, self.prediction_horizon, self.action_dim)
        executed = chunk[:, : self.execution_horizon]

        alive = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        reward = torch.zeros(num_envs, 1, device=self.device, dtype=torch.float32)
        terminated = torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)
        explosion = torch.zeros_like(terminated)
        obs = None

        for i in range(self.execution_horizon):
            action = executed[:, i]
            if not bool(alive.all()):
                action = torch.where(alive.unsqueeze(-1), action, self._neutral_action())

            sub = self.inner._step(
                TensorDict({"action": action}, batch_size=self.batch_size, device=self.device)
            )

            alive_f = alive.unsqueeze(-1)
            reward = reward + self._reward_weights[i] * sub["reward"] * alive_f
            # Flags are only meaningful while the env is still on the episode we started
            # the chunk with; after that they belong to a freshly respawned robot.
            terminated |= sub["terminated"] & alive_f
            truncated |= sub["truncated"] & alive_f
            explosion |= sub["explosion"] & alive_f
            alive = alive & ~sub["done"].squeeze(-1)
            obs = sub[OBS_KEY]

        return TensorDict(
            {
                OBS_KEY: obs,
                "reward": reward,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
                "explosion": explosion,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        self.inner._set_seed(seed)

    # ------------------------------------------------------------------
    # Pass-through to the inner env
    #
    # run_tracked_rollout, the trainers and eval_ftr.py all call these on what they think
    # is the FtrTorchRLEnv. Delegated explicitly rather than via __getattr__ so a missing
    # method fails loudly instead of silently resolving through nn.Module's lookup.
    # Every number they return keeps its PER-CONTROL-STEP meaning: the inner env
    # accumulates once per sub-step, not once per macro step.
    # ------------------------------------------------------------------

    @property
    def ftr_env(self):
        return self.inner.ftr_env

    @property
    def observations(self):
        return self.inner.observations

    def peek_reward_series(self) -> dict[str, list[float]]:
        return self.inner.peek_reward_series()

    def pop_reward_info(self) -> dict[str, float]:
        return self.inner.pop_reward_info()

    def pop_state_stats(self) -> dict[str, float]:
        return self.inner.pop_state_stats()

    def pop_termination_info(self) -> dict[str, float]:
        return self.inner.pop_termination_info()

    def enable_per_env_tracking(self) -> None:
        self.inner.enable_per_env_tracking()

    def pop_per_env_termination(self) -> dict[int, dict[str, int]]:
        return self.inner.pop_per_env_termination()

    def disable_per_env_tracking(self) -> None:
        self.inner.disable_per_env_tracking()
