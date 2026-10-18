"""Motion-smoothness tracker (FTR-Benchmark ftr_envs/tasks/crossing/smoothness.py).

Loaded by path: importing the ftr_envs package would start the Isaac task registration.
Each test drives synthetic robots along +x over an analytic height profile z(x), with the
body pitched to the local slope, sampled at the physics rate (200 Hz).
"""

import importlib.util
import math
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[3] / "FTR-Benchmark" / "ftr_envs" / "tasks" / "crossing" / "smoothness.py"
_spec = importlib.util.spec_from_file_location("smoothness", _PATH)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

DT = 0.005
CH = {c: i for i, c in enumerate(sm.CHANNELS)}


def quat_pitch(p: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) of a rotation by p about body y."""
    return torch.stack([torch.cos(p / 2), torch.zeros_like(p), torch.sin(p / 2), torch.zeros_like(p)], -1)


def drive(profile, speeds, length=3.0, hold=0.0):
    """Drive len(speeds) envs over z = profile(x) at constant speeds; returns episode cost (N, C)."""
    v = torch.tensor(speeds, dtype=torch.float32)
    n = len(speeds)
    tr = sm.SmoothnessTracker(n, "cpu", DT)

    def state(x):
        # float64 for the finite-difference slope: in float32 its rounding noise (~3e-4 rad
        # at x = 3 m) is what the spatial second difference would measure
        xd, h = x.double(), 1e-4
        z = profile(xd)
        p = torch.atan((profile(xd + h) - profile(xd - h)) / (2 * h))
        return torch.stack([xd, torch.zeros_like(xd), z], -1).float(), p.float()

    x = torch.zeros(n)
    pos, p = state(x)
    tr.reset(torch.arange(n), pos, quat_pitch(p), torch.zeros(n, 3), torch.stack([v, 0 * v, 0 * v], -1))
    # every env keeps its speed until the slowest has covered `length`; `hold` > 0 then
    # stops them all (the robots on the flank of a profile stay pitched)
    n_move = int(math.ceil(length / float(v.min()) / DT))
    p_prev = p
    for k in range(n_move + int(hold / DT)):
        moving = k < n_move
        if moving:
            x = x + v * DT
        pos, p = state(x)
        w = torch.stack([0 * v, (p - p_prev) / DT, 0 * v], -1)
        lin = torch.stack([v if moving else 0 * v, 0 * v, 0 * v], -1)
        tr.substep(pos, quat_pitch(p), w, lin)
        p_prev = p
    return tr.episode_cost, tr


def smooth_bump(x, h=0.2, c=1.5, wdt=0.25):
    return h * torch.exp(-((x - c) / wdt) ** 2)


def test_flat_and_constant_slope_cost_nothing():
    cost, _ = drive(lambda x: 0.0 * x, [0.5])
    assert cost.abs().max() < 1e-6
    cost, _ = drive(lambda x: 0.3 * x, [0.5])  # 16.7 deg ramp, steady pitch
    assert cost[0, CH["pitch_ss"]] < 1e-6 and cost[0, CH["pitch_acc"]] < 1e-6
    assert cost[0, CH["z_ss"]] < 1e-6


def test_edge_costs_more_than_bump():
    edge = lambda x: 0.15 * torch.sigmoid((x - 1.5) / 0.01)   # near-step, 15 cm
    soft = lambda x: 0.15 * torch.sigmoid((x - 1.5) / 0.15)   # same rise, gentle
    c_edge, _ = drive(edge, [0.5])
    c_soft, _ = drive(soft, [0.5])
    for ch in ("z_ss", "pitch_ss", "pitch_acc"):
        assert c_edge[0, CH[ch]] > 3 * c_soft[0, CH[ch]] > 0, ch


def test_spatial_speed_invariant_temporal_grows_with_speed():
    cost, _ = drive(smooth_bump, [0.3, 0.6])
    sp_ratio = cost[1, CH["pitch_ss"]] / cost[0, CH["pitch_ss"]]
    tm_ratio = cost[1, CH["pitch_acc"]] / cost[0, CH["pitch_acc"]]
    assert 0.8 < sp_ratio < 1.25, sp_ratio
    assert 5.0 < tm_ratio < 11.0, tm_ratio  # ~ v^3 = 8 for a speed-independent path


def test_standing_still_is_free_and_finite():
    cost, tr = drive(smooth_bump, [0.5], length=1.0, hold=3.0)  # stops dead on the bump's flank
    assert cost[0, CH["v_jerk"]] > 0  # the instantaneous stop is a jerk
    still = cost.clone()
    pos = tr.prev_pos.clone()
    q = quat_pitch(torch.tensor([0.3]))
    for _ in range(400):
        tr.substep(pos, q, torch.zeros(1, 3), torch.zeros(1, 3))
    assert torch.isfinite(tr.episode_cost).all()
    assert (tr.episode_cost[0, CH["z_ss"]] - still[0, CH["z_ss"]]).abs() < 1e-9


def test_teleport_restarts_spatial_history():
    _, tr = drive(lambda x: 0.0 * x, [0.5], length=1.0)
    before = tr.episode_cost.clone()
    tr.substep(torch.tensor([[5.0, 0.0, 1.0]]), quat_pitch(torch.zeros(1)), torch.zeros(1, 3), torch.tensor([[0.5, 0, 0]]))
    assert tr.n_grid.item() == 0
    assert (tr.episode_cost[0, CH["z_ss"]] - before[0, CH["z_ss"]]).abs() < 1e-9


def test_reset_is_per_env_and_pop_clears():
    _, tr = drive(smooth_bump, [0.4, 0.4])
    tr.step_cost.fill_(1.0)
    ep1 = tr.episode_cost[1].clone()
    tr.reset(torch.tensor([0]), torch.zeros(1, 3), quat_pitch(torch.zeros(1)), torch.zeros(1, 3), torch.zeros(1, 3))
    assert tr.episode_cost[0].abs().max() == 0 and torch.equal(tr.episode_cost[1], ep1)
    assert tr.pop_step()[1].sum() > 0 and tr.step_cost.abs().max() == 0


def test_eval_under_inference_mode_then_training_step():
    """The eval rollout runs under inference_mode, the training rollout and resets outside it
    (job 11807492 died at its first eval: a buffer rebound inside became an inference tensor)."""
    _, tr = drive(smooth_bump, [0.4, 0.4], length=0.5)
    args = (torch.tensor([[0.6, 0.0, 0.0]] * 2), quat_pitch(torch.zeros(2)), torch.zeros(2, 3), torch.tensor([[0.4, 0, 0]] * 2))
    with torch.inference_mode():
        tr.substep(*args)
        tr.pop_step()
    with torch.no_grad():
        tr.substep(*args)
        tr.reset(torch.tensor([0]), *(a[:1] for a in args))
        tr.pop_step()
    assert not any(t.is_inference() for t in vars(tr).values() if isinstance(t, torch.Tensor))
