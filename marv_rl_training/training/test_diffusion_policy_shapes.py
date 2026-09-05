"""Shape/rank checks for the receding-horizon policy networks.

No Isaac Sim, no env — just the modules, plus a real GAE call, which is where the ranks
actually bite. Run:

    PYTHONPATH=src/flipper_training:src/FTR-Benchmark \
        python -m marv_rl_training.training.test_diffusion_policy_shapes

The case that matters: GAE evaluates the critic as ``vmap(value_net, (0,))`` over an
``[envs, time]`` tensordict, so the encoder sees two batch dims plus a vmap dim. A version
of ObsHistoryEncoder that reshaped using ``shape[0]`` passed every other path — collection,
PPO minibatches, eval — and died only inside GAE, one full smoke run in. Hence the direct
GAE test at the bottom rather than only unit checks on the modules.
"""

import sys
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marv_rl_training.policies.diffusion_policy import (  # noqa: E402
    ChunkCriticNet,
    ChunkGaussianActorNet,
    ObsHistoryEncoder,
)
from rl_modules.marv_rl.marv_rl_cnn_flat_encoder import MarvRLCNNFlatEncoder  # noqa: E402

OBS_DIM, ACTION_DIM = 966, 6
T_O, T_P = 2, 16
DOWN_DIMS = [64, 128]
# Matches ftr_obs_encoder_opts in the configs.
ENCODER_OPTS = dict(num_hidden=3, hidden_dim=256, output_dim=128, layernorm=True)

_fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        _fails.append(name)


def make_encoder():
    return ObsHistoryEncoder(MarvRLCNNFlatEncoder(input_dim=OBS_DIM, **ENCODER_OPTS), OBS_DIM, T_O)


torch.manual_seed(0)

# ----------------------------------------------------------------------------------
# 1. ObsHistoryEncoder is agnostic to leading dimensions
# ----------------------------------------------------------------------------------

enc = make_encoder().eval()
check("output_dim = T_o * frame_dim", enc.output_dim == T_O * 128, str(enc.output_dim))

with torch.no_grad():
    x2 = torch.randn(8, T_O * OBS_DIM)
    y2 = enc(x2)
    check("2-D input [N, T_o*obs] -> [N, T_o*frame]", tuple(y2.shape) == (8, T_O * 128), str(tuple(y2.shape)))

    x3 = torch.randn(4, 5, T_O * OBS_DIM)
    y3 = enc(x3)
    check("3-D input [envs, time, ...] keeps both leading dims", tuple(y3.shape) == (4, 5, T_O * 128), str(tuple(y3.shape)))

    check("3-D result equals the 2-D result on the same rows",
          torch.allclose(enc(x3.reshape(-1, T_O * OBS_DIM)).reshape(4, 5, -1), y3, atol=1e-6))

    # Frame ordering: the window is [oldest | newest] laid out contiguously on the last dim,
    # so the embedding must be cat(enc(oldest), enc(newest)) in that order.
    a, b = torch.randn(3, OBS_DIM), torch.randn(3, OBS_DIM)
    win = torch.cat([a, b], dim=-1)
    expect = torch.cat([enc.frame_encoder(a), enc.frame_encoder(b)], dim=-1)
    check("frames are encoded independently and concatenated oldest-first",
          torch.allclose(enc(win), expect, atol=1e-6))

    # The exact call GAE makes.
    stacked = torch.stack([torch.randn(4, 5, T_O * OBS_DIM), torch.randn(4, 5, T_O * OBS_DIM)], dim=0)
    yv = torch.vmap(enc, (0,))(stacked)
    check("survives vmap over a leading dim (this is how GAE calls the critic)",
          tuple(yv.shape) == (2, 4, 5, T_O * 128), str(tuple(yv.shape)))

# ----------------------------------------------------------------------------------
# 2. Actor: shapes and the step-major chunk layout
# ----------------------------------------------------------------------------------

actor = ChunkGaussianActorNet(make_encoder(), ACTION_DIM, T_P, DOWN_DIMS, 5, 8).eval()
with torch.no_grad():
    loc, scale = actor(torch.randn(8, T_O * OBS_DIM))
check("actor emits loc/scale over the flattened T_p x A chunk",
      tuple(loc.shape) == (8, T_P * ACTION_DIM) and tuple(scale.shape) == (8, T_P * ACTION_DIM),
      f"{tuple(loc.shape)} / {tuple(scale.shape)}")
check("initial scale is 1.0 everywhere (zero-init output conv, as the baseline head is)",
      torch.allclose(scale, torch.ones_like(scale)), f"min {scale.min():.4f} max {scale.max():.4f}")
check("loc is finite", bool(torch.isfinite(loc).all()))
check("loc reshapes to [N, T_p, A] — the layout ActionChunkEnv assumes",
      tuple(loc.reshape(8, T_P, ACTION_DIM).shape) == (8, T_P, ACTION_DIM))

# ----------------------------------------------------------------------------------
# 3. Critic
# ----------------------------------------------------------------------------------

critic = ChunkCriticNet(make_encoder(), dict(num_hidden=2, hidden_dim=256, layernorm=True)).eval()
with torch.no_grad():
    check("critic 2-D -> one value per row", tuple(critic(torch.randn(8, T_O * OBS_DIM)).shape) == (8, 1))
    check("critic 3-D -> one value per (env, step)", tuple(critic(torch.randn(4, 5, T_O * OBS_DIM)).shape) == (4, 5, 1))

# ----------------------------------------------------------------------------------
# 4. The real thing: GAE over an [envs, time] rollout
# ----------------------------------------------------------------------------------

N_ENV, T = 4, 5
value_operator = TensorDictModule(critic, in_keys=["obs_history"], out_keys=["state_value"])
gae = GAE(gamma=0.98, lmbda=0.95, value_network=value_operator, time_dim=1,
          average_gae=False, skip_existing=False, differentiable=False)

td = TensorDict(
    {
        "obs_history": torch.randn(N_ENV, T, T_O * OBS_DIM),
        "next": TensorDict(
            {
                "obs_history": torch.randn(N_ENV, T, T_O * OBS_DIM),
                "reward": torch.randn(N_ENV, T, 1),
                "done": torch.zeros(N_ENV, T, 1, dtype=torch.bool),
                "terminated": torch.zeros(N_ENV, T, 1, dtype=torch.bool),
            },
            batch_size=[N_ENV, T],
        ),
    },
    batch_size=[N_ENV, T],
)

try:
    gae(td)
    ok = tuple(td["advantage"].shape) == (N_ENV, T, 1) and tuple(td["value_target"].shape) == (N_ENV, T, 1)
    check("GAE runs the critic over an [envs, time] batch and returns advantages",
          ok and bool(torch.isfinite(td["advantage"]).all()), str(tuple(td["advantage"].shape)))
except Exception as e:  # noqa: BLE001 — the regression this file exists for
    check("GAE runs the critic over an [envs, time] batch and returns advantages", False, f"{type(e).__name__}: {e}")

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}")
    sys.exit(1)
print("all checks passed")
