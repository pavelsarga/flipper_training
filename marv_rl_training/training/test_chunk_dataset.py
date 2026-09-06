"""Checks for the BC chunk-dataset helpers. No Isaac, no GPU."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marv_rl_training.training.chunk_dataset_utils import (  # noqa: E402
    build_chunks, flipper_angles_to_position_action)

_fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        _fails.append(name)


low = torch.tensor([-1.0472, -1.0472, -1.3963, -1.3963])   # MARV asymmetric limits, radians
high = torch.tensor([1.3963, 1.3963, 1.0472, 1.0472])

# --- label conversion round-trips against the env's own [-1,1] -> [low,high] map ---
theta = torch.rand(7, 4) * (high - low) + low
u = flipper_angles_to_position_action(theta, low, high)
back = low + (u + 1.0) * 0.5 * (high - low)                # exactly what ftr_env.py does
check("velocity-mode angles convert to position actions that map back exactly",
      torch.allclose(back, theta, atol=1e-5), f"max err {float((back-theta).abs().max()):.2e}")
check("the limits map to the endpoints of [-1, 1]",
      torch.allclose(flipper_angles_to_position_action(low.clone(), low, high), -torch.ones(4), atol=1e-5)
      and torch.allclose(flipper_angles_to_position_action(high.clone(), low, high), torch.ones(4), atol=1e-5))
check("out-of-range angles clamp rather than extrapolate",
      float(flipper_angles_to_position_action(high * 2, low, high).max()) <= 1.0)

# --- sliding window ---
T, N, A, T_p = 10, 3, 6, 4
acts = torch.arange(T * N * A, dtype=torch.float32).reshape(T, N, A)
ths = torch.zeros(T, N, 4)
obs = torch.arange(T * N * 5, dtype=torch.float32).reshape(T, N, 5)
dones = torch.zeros(T, N, dtype=torch.bool)

o, a = build_chunks(obs, acts, ths, dones, T_p, low, high)
check("no resets -> (T - T_p + 1) windows per env", o.shape[0] == (T - T_p + 1) * N, f"{o.shape[0]}")
check("chunk shape is [M, T_p, A]", tuple(a.shape) == ((T - T_p + 1) * N, T_p, A), str(tuple(a.shape)))
check("obs_history row is the window's FIRST step (what the policy conditions on)",
      torch.allclose(o[0], obs[0, 0]), f"{o[0].tolist()} vs {obs[0,0].tolist()}")
check("track v/w are taken from the demonstrator unchanged",
      torch.allclose(a[0, :, :2], acts[0:T_p, 0, :2]))

# --- episode boundaries ---
dones2 = torch.zeros(T, N, dtype=torch.bool)
dones2[5, 1] = True                     # env 1 ends its episode on step 5
o2, a2 = build_chunks(obs, acts, ths, dones2, T_p, low, high)
lost = o.shape[0] - o2.shape[0]
# Window t spans steps t..t+T_p-1. A done at step s makes s+1 a new episode, so the window
# is only spoiled when s is among steps t..t+T_p-2 — a done on the window's LAST step leaves
# the chunk entirely within one episode and is fine. For a done at step 5 that is
# t in {3,4,5}: three windows, for one env only.
check("windows spanning a reset are dropped, and only for the env that reset",
      lost == 3, f"dropped {lost}, expected 3")

dones3 = torch.zeros(T, N, dtype=torch.bool)
dones3[T_p - 1, 0] = True               # done exactly on window t=0's LAST step
o3, a3 = build_chunks(obs, acts, ths, dones3, T_p, low, high)
# Assert the specific window survives rather than counting: window t=0 for env 0 should
# still be present, identifiable by its obs_history row.
kept_first = any(torch.allclose(o3[i], obs[0, 0]) for i in range(o3.shape[0]))
check("a reset on the window's final step is kept (chunk stays within one episode)",
      kept_first and o3.shape[0] == o.shape[0] - 3, f"{o3.shape[0]} rows, first-window kept={kept_first}")

check("too-short input returns nothing rather than raising",
      build_chunks(obs[:2], acts[:2], ths[:2], dones[:2], T_p, low, high) == (None, None))

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}")
    sys.exit(1)
print("all checks passed")
