"""Standalone checks for ActionChunkEnv + the CatFrames observation window.

No Isaac Sim, no GPU — the inner env is a fake that records what it was asked to do. Run:

    PYTHONPATH=src/flipper_training python -m marv_rl_training.training.test_action_chunk_env

Covers the parts of docs/diffusion_policy/01_action_chunking.md that are easy to get
silently wrong: the number of inner steps per macro step, the chunk memory layout, the
discounted reward sum, done/explosion OR-ing, the neutral action after a mid-chunk
termination, T_a=1 equivalence with the un-chunked env, and CatFrames' per-row reset.
"""

import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.envs import CatFrames, Compose, EnvBase, StepCounter, TransformedEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marv_rl_training.environment.chunked_env import ActionChunkEnv  # noqa: E402
from marv_rl_training.environment.ftr_env_adapter import OBS_KEY  # noqa: E402

N_ENVS, ACTION_DIM, OBS_DIM = 3, 6, 8

_fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        _fails.append(name)


# ----------------------------------------------------------------------------------
# Fake inner env
# ----------------------------------------------------------------------------------


class _FakeCfg:
    flipper_control_mode = "velocity"
    flipper_style = False


class _FakeUnwrapped:
    def __init__(self, device):
        self.cfg = _FakeCfg()
        self.flipper_positions = torch.zeros(N_ENVS, 4, device=device)
        self._device = device

    def flipper_angle_bounds(self):
        low = torch.full((4,), -1.0, device=self._device)
        high = torch.full((4,), 1.0, device=self._device)
        return low, high


class _FakeGymEnv:
    def __init__(self, device):
        self.unwrapped = _FakeUnwrapped(device)


class FakeInnerEnv(EnvBase):
    """Records every action it is stepped with; terminates env `die_env` at step `die_at`."""

    _batch_locked = True

    def __init__(self, device="cpu", die_env=None, die_at=None):
        super().__init__(device=device, batch_size=[N_ENVS])
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(N_ENVS, ACTION_DIM), device=device, dtype=torch.float32)
        self.observation_spec = Composite(
            {OBS_KEY: Unbounded(shape=(N_ENVS, OBS_DIM), device=device, dtype=torch.float32)}, shape=(N_ENVS,)
        )
        self.reward_spec = Unbounded(shape=(N_ENVS, 1), device=device, dtype=torch.float32)
        self.done_spec = Composite(
            {k: Binary(shape=(N_ENVS, 1), device=device, dtype=torch.bool) for k in
             ("done", "terminated", "truncated", "explosion")},
            shape=(N_ENVS,),
        )
        self.ftr_env = _FakeGymEnv(device)
        self.observations = []
        self.seen_actions = []      # one [N, A] tensor per inner step
        self.n_steps = 0
        self.die_env, self.die_at = die_env, die_at

    def _reset(self, tensordict=None):
        return TensorDict({OBS_KEY: torch.zeros(N_ENVS, OBS_DIM)}, batch_size=self.batch_size)

    def _step(self, tensordict):
        action = tensordict["action"]
        self.seen_actions.append(action.clone())
        self.n_steps += 1

        terminated = torch.zeros(N_ENVS, 1, dtype=torch.bool)
        explosion = torch.zeros_like(terminated)
        if self.die_env is not None and self.n_steps == self.die_at:
            terminated[self.die_env] = True
            explosion[self.die_env] = True
        # Reward = step index, so a discounted sum is trivial to verify by hand.
        reward = torch.full((N_ENVS, 1), float(self.n_steps))
        return TensorDict(
            {
                OBS_KEY: torch.full((N_ENVS, OBS_DIM), float(self.n_steps)),
                "reward": reward,
                "done": terminated,
                "terminated": terminated,
                "truncated": torch.zeros_like(terminated),
                "explosion": explosion,
            },
            batch_size=self.batch_size,
        )

    def _set_seed(self, seed):
        pass

    # accumulator API the trainers call through the wrapper
    def peek_reward_series(self):
        return {"rew/x": [1.0]}

    def pop_reward_info(self):
        return {"rew/x": 1.0}

    def pop_state_stats(self):
        return {"state/x": 1.0}

    def pop_termination_info(self):
        return {"env/success_rate": 1.0}

    def enable_per_env_tracking(self):
        self._tracking = True

    def pop_per_env_termination(self):
        return {}

    def disable_per_env_tracking(self):
        self._tracking = False


def make_chunk(T_p, values):
    """values[t][a] -> flattened chunk of shape [N, T_p*A], identical across envs."""
    chunk = torch.zeros(N_ENVS, T_p, ACTION_DIM)
    for t in range(T_p):
        for a in range(ACTION_DIM):
            chunk[:, t, a] = values(t, a)
    return chunk.reshape(N_ENVS, -1)


# ----------------------------------------------------------------------------------
# 1. Spec, step count and chunk memory layout
# ----------------------------------------------------------------------------------

T_p, T_a, GAMMA = 8, 4, 0.9
inner = FakeInnerEnv()
env = ActionChunkEnv(inner, prediction_horizon=T_p, execution_horizon=T_a, control_gamma=GAMMA)

check("action spec is the flattened chunk", tuple(env.action_spec.shape) == (N_ENVS, T_p * ACTION_DIM),
      str(tuple(env.action_spec.shape)))
check("observation/reward/done specs are unchanged",
      env.observation_spec[OBS_KEY].shape == inner.observation_spec[OBS_KEY].shape
      and env.reward_spec.shape == inner.reward_spec.shape)

# Encode the (t, a) index into the value so the layout is recoverable from what the inner
# env saw. This is the contract between ChunkGaussianActorNet's transpose-then-flatten and
# the wrapper's reshape(N, T_p, A) — get it wrong and every action goes to the wrong joint.
td = env._step(TensorDict({"action": make_chunk(T_p, lambda t, a: t * 10 + a)}, batch_size=[N_ENVS]))
check("executes exactly T_a inner steps", inner.n_steps == T_a, f"{inner.n_steps}")
layout_ok = all(
    torch.allclose(inner.seen_actions[t], torch.tensor([[t * 10 + a for a in range(ACTION_DIM)]] * N_ENVS, dtype=torch.float32))
    for t in range(T_a)
)
check("chunk is step-major: reshape(N, T_p, A) recovers the actor's layout", layout_ok)
check("the discarded tail is never executed", len(inner.seen_actions) == T_a)

expected_r = sum(GAMMA**i * (i + 1) for i in range(T_a))
check("reward is the control-step-discounted sum over the executed prefix",
      torch.allclose(td["reward"], torch.full((N_ENVS, 1), expected_r)), f"{td['reward'][0].item():.6f} vs {expected_r:.6f}")
check("observation is the last inner observation",
      torch.allclose(td[OBS_KEY], torch.full((N_ENVS, OBS_DIM), float(T_a))))
check("no spurious done flags", not td["done"].any() and not td["explosion"].any())

# ----------------------------------------------------------------------------------
# 2. Mid-chunk termination: flags OR-ed, reward stops, neutral action afterwards
# ----------------------------------------------------------------------------------

DIE_ENV, DIE_AT = 1, 2  # env 1 terminates on the 2nd of 4 sub-steps
inner = FakeInnerEnv(die_env=DIE_ENV, die_at=DIE_AT)
env = ActionChunkEnv(inner, prediction_horizon=T_p, execution_horizon=T_a, control_gamma=GAMMA)
td = env._step(TensorDict({"action": make_chunk(T_p, lambda t, a: 0.5)}, batch_size=[N_ENVS]))

check("terminated is OR-ed across sub-steps", bool(td["terminated"][DIE_ENV]) and not bool(td["terminated"][0]))
check("explosion is OR-ed across sub-steps (the GAE trajectory filter reads it)",
      bool(td["explosion"][DIE_ENV]) and not bool(td["explosion"][0]))
check("done is terminated | truncated", bool(td["done"][DIE_ENV]))

r_dead = sum(GAMMA**i * (i + 1) for i in range(DIE_AT))
r_live = sum(GAMMA**i * (i + 1) for i in range(T_a))
check("reward stops accumulating at the terminating sub-step",
      abs(float(td["reward"][DIE_ENV]) - r_dead) < 1e-5, f"{float(td['reward'][DIE_ENV]):.6f} vs {r_dead:.6f}")
check("other envs keep accumulating the full chunk",
      abs(float(td["reward"][0]) - r_live) < 1e-5, f"{float(td['reward'][0]):.6f} vs {r_live:.6f}")

after = torch.stack([inner.seen_actions[t][DIE_ENV] for t in range(DIE_AT, T_a)])
check("terminated env gets the neutral action for the rest of the chunk (velocity mode -> zeros)",
      torch.allclose(after, torch.zeros_like(after)), str(after.tolist()))
still = torch.stack([inner.seen_actions[t][0] for t in range(DIE_AT, T_a)])
check("live envs keep their commanded action", torch.allclose(still, torch.full_like(still, 0.5)))

# Position mode: neutral holds the current angle, not zero.
inner = FakeInnerEnv(die_env=DIE_ENV, die_at=DIE_AT)
inner.ftr_env.unwrapped.cfg.flipper_control_mode = "position"
inner.ftr_env.unwrapped.flipper_positions = torch.full((N_ENVS, 4), 0.5)  # bounds are [-1, 1]
env = ActionChunkEnv(inner, prediction_horizon=T_p, execution_horizon=T_a, control_gamma=GAMMA)
env._step(TensorDict({"action": make_chunk(T_p, lambda t, a: 0.9)}, batch_size=[N_ENVS]))
neutral = inner.seen_actions[DIE_AT][DIE_ENV]
check("position mode: neutral holds the measured flipper angle",
      torch.allclose(neutral[:2], torch.zeros(2)) and torch.allclose(neutral[2:], torch.full((4,), 0.5)),
      str(neutral.tolist()))

# ----------------------------------------------------------------------------------
# 3. T_a = 1 reproduces the un-chunked env
# ----------------------------------------------------------------------------------

inner_a = FakeInnerEnv()
env_a = ActionChunkEnv(inner_a, prediction_horizon=1, execution_horizon=1, control_gamma=GAMMA)
td_a = env_a._step(TensorDict({"action": torch.full((N_ENVS, ACTION_DIM), 0.3)}, batch_size=[N_ENVS]))
inner_b = FakeInnerEnv()
td_b = inner_b._step(TensorDict({"action": torch.full((N_ENVS, ACTION_DIM), 0.3)}, batch_size=[N_ENVS]))
check("T_a=1, T_p=1 is a pass-through (the reference for measuring the chunking artifact)",
      torch.allclose(td_a["reward"], td_b["reward"]) and torch.allclose(td_a[OBS_KEY], td_b[OBS_KEY])
      and torch.allclose(inner_a.seen_actions[0], inner_b.seen_actions[0]))

# ----------------------------------------------------------------------------------
# 4. Validation and pass-through
# ----------------------------------------------------------------------------------

try:
    ActionChunkEnv(FakeInnerEnv(), prediction_horizon=4, execution_horizon=5, control_gamma=GAMMA)
    check("execution_horizon > prediction_horizon is rejected", False)
except ValueError:
    check("execution_horizon > prediction_horizon is rejected", True)

env = ActionChunkEnv(FakeInnerEnv(), prediction_horizon=T_p, execution_horizon=T_a, control_gamma=GAMMA)
check("accumulator API is forwarded to the inner env (run_tracked_rollout depends on it)",
      env.pop_reward_info() == {"rew/x": 1.0}
      and env.pop_state_stats() == {"state/x": 1.0}
      and env.pop_termination_info() == {"env/success_rate": 1.0}
      and env.peek_reward_series() == {"rew/x": [1.0]}
      and env.ftr_env is env.inner.ftr_env)

# ----------------------------------------------------------------------------------
# 5. CatFrames window: shape, "same" padding, and per-row reset
# ----------------------------------------------------------------------------------

T_o = 2
inner = FakeInnerEnv(die_env=DIE_ENV, die_at=T_a)  # dies on the LAST sub-step of macro step 1
chunked = ActionChunkEnv(inner, prediction_horizon=T_p, execution_horizon=T_a, control_gamma=GAMMA)
tenv = TransformedEnv(
    chunked,
    Compose(StepCounter(), CatFrames(N=T_o, dim=-1, in_keys=[OBS_KEY], out_keys=["obs_history"], padding="same")),
)
td = tenv.reset()
check("obs_history has shape [N, T_o * obs_dim]", tuple(td["obs_history"].shape) == (N_ENVS, T_o * OBS_DIM),
      str(tuple(td["obs_history"].shape)))
check("padding='same' repeats the first frame at episode start",
      torch.allclose(td["obs_history"][:, :OBS_DIM], td["obs_history"][:, OBS_DIM:]))

td["action"] = make_chunk(T_p, lambda t, a: 0.0)
# step_and_maybe_reset is the collector's own path: step, then partially reset the rows
# that reported done. Doing it this way rather than calling reset() by hand is the point —
# it is the sequence CatFrames actually sees during training.
root, td_next = tenv.step_and_maybe_reset(td)
win = root["next"]["obs_history"]
check("window holds [previous, current] after one macro step",
      torch.allclose(win[0, :OBS_DIM], torch.zeros(OBS_DIM)) and torch.allclose(win[0, OBS_DIM:], torch.full((OBS_DIM,), float(T_a))),
      str(win[0].tolist()))
check("the macro step reported done for the env that terminated mid-chunk",
      bool(root["next"]["done"][DIE_ENV]) and not bool(root["next"]["done"][0]))

w = td_next["obs_history"]
check("the partial reset clears only the terminated env's window",
      torch.allclose(w[DIE_ENV, :OBS_DIM], w[DIE_ENV, OBS_DIM:])
      and not torch.allclose(w[0, :OBS_DIM], w[0, OBS_DIM:]),
      f"dead={w[DIE_ENV].tolist()} live={w[0].tolist()}")

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}")
    sys.exit(1)
print("all checks passed")
