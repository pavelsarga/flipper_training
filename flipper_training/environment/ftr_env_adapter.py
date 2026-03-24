"""TorchRL EnvBase adapter wrapping an FTR-Benchmark DirectRLEnv gymnasium environment.

Usage:
    # Instantiate AFTER AppLauncher has started Isaac Sim
    ftr_gym_env = gymnasium.make("FTR-Crossing-Direct-v0", ...)
    env = FtrTorchRLEnv(ftr_gym_env, encoder_opts={...}, device="cuda:0")
    check_env_specs(env)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

if TYPE_CHECKING:
    from gymnasium import Env as GymEnv

# Key under which the flat observation is stored in TensorDicts.
# Must match FtrFlatObservation.__class__.__name__ so the encoder TensorDictModule finds it.
OBS_KEY = "FtrFlatObservation"


class FtrTorchRLEnv(EnvBase):
    """Wraps an FTR-Benchmark DirectRLEnv (gymnasium API) as a TorchRL EnvBase.

    FTR's DirectRLEnv auto-resets done environments internally at every step and returns
    the post-reset observation for those environments.  TorchRL's SyncDataCollector will
    call _reset() for done envs after each step; at that point FTR has already reset them,
    so _reset() returns the stored last observation without re-triggering a reset.
    """

    _batch_locked = True

    def __init__(
        self,
        ftr_env: GymEnv,
        encoder_opts: dict,
        device: str | torch.device = "cuda:0",
    ):
        num_envs: int = ftr_env.unwrapped.num_envs
        super().__init__(device=device, batch_size=[num_envs])

        self.ftr_env = ftr_env
        self._last_obs: torch.Tensor | None = None
        self._reward_info_accum: dict[str, list[float]] = {}
        self._term_success: int = 0
        self._term_failure: int = 0
        self._term_explosion: int = 0
        self._term_total: int = 0

        # Instantiate the observation descriptor so make_transformed_env can build VecNorm keys.
        from flipper_training.observations.ftr_flat_obs import FtrFlatObservation

        self.observations = [FtrFlatObservation(env=self, encoder_opts=encoder_opts)]

        # ------------------------------------------------------------------ specs
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(num_envs, 4),
            device=device,
            dtype=torch.float32,
        )
        self.observation_spec = Composite(
            {OBS_KEY: Unbounded(shape=(num_envs, 115), device=device, dtype=torch.float32)},
            shape=(num_envs,),
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1), device=device, dtype=torch.float32)
        self.done_spec = Composite(
            {
                "done": Binary(shape=(num_envs, 1), device=device, dtype=torch.bool),
                "terminated": Binary(shape=(num_envs, 1), device=device, dtype=torch.bool),
                "truncated": Binary(shape=(num_envs, 1), device=device, dtype=torch.bool),
            },
            shape=(num_envs,),
        )

    # ------------------------------------------------------------------
    # EnvBase interface
    # ------------------------------------------------------------------

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Reset some or all environments.

        FTR auto-resets done envs during step(), so for partial resets TorchRL triggers
        after a done signal we return the already-stored post-reset observation.  A full
        reset (None tensordict, e.g. at training start) calls ftr_env.reset() properly.
        """
        if tensordict is not None and "_reset" in tensordict.keys():
            # Partial reset — FTR already handled this in the last step().
            if self._last_obs is not None:
                return TensorDict(
                    {OBS_KEY: self._last_obs.to(self.device)},
                    batch_size=self.batch_size,
                    device=self.device,
                )

        # Full reset (initial collection or explicit full reset).
        obs_dict, _ = self.ftr_env.reset()
        obs = torch.nan_to_num(obs_dict["policy"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        self._last_obs = obs
        return TensorDict({OBS_KEY: obs}, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].detach().contiguous().to(self.ftr_env.unwrapped.device)
        # Sanitize: NaN/Inf actions would be written as joint targets into PhysX, causing GPU
        # memory corruption (CUDA error 700).  This can happen when a robot's quaternion goes
        # invalid (escaped terrain bounds), propagating NaN through observations → policy → action.
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        obs_dict, reward, terminated, truncated, _info = self.ftr_env.step(action)

        if isinstance(_info, dict):
            if "reward_components" in _info:
                for k, v in _info["reward_components"].items():
                    self._reward_info_accum.setdefault(k, []).append(v)
            if "success" in _info and "failure" in _info:
                episodes_done = (terminated | truncated).long().sum().item()
                self._term_total += episodes_done
                self._term_success += _info["success"].long().sum().item()
                self._term_failure += _info["failure"].long().sum().item()
                if "explosion" in _info:
                    self._term_explosion += _info["explosion"].long().sum().item()

        # Sanitize observation: NaN from invalid robot state (fallen off terrain) must not reach
        # the policy or VecNorm running statistics on the next step.
        obs = torch.nan_to_num(obs_dict["policy"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        self._last_obs = obs  # store for subsequent _reset() calls

        reward = torch.nan_to_num(reward.to(self.device).to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
        terminated = terminated.to(self.device).unsqueeze(-1)
        truncated = truncated.to(self.device).unsqueeze(-1)
        done = terminated | truncated

        return TensorDict(
            {
                OBS_KEY: obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def pop_reward_info(self) -> dict[str, float]:
        """Return per-component reward means accumulated since the last call, then clear."""
        if not self._reward_info_accum:
            return {}
        result = {k: sum(v) / len(v) for k, v in self._reward_info_accum.items()}
        self._reward_info_accum.clear()
        return result

    def pop_termination_info(self) -> dict[str, float]:
        """Return success/failure rates accumulated since the last call, then clear."""
        if self._term_total == 0:
            return {}
        result = {
            "env/success_rate": self._term_success / self._term_total,
            "env/failure_rate": self._term_failure / self._term_total,
            "env/explosion_rate": self._term_explosion / self._term_total,
        }
        self._term_success = self._term_failure = self._term_explosion = self._term_total = 0
        return result

    def _set_seed(self, seed: int | None) -> None:
        pass  # FTR environments do not support external seeding via this API
