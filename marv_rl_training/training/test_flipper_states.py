"""Discrete flipper-state logic (FTR-Benchmark ftr_envs/tasks/crossing/flipper_states.py).

Loaded by path: importing the ftr_envs package would start the Isaac task registration.
"""

import importlib.util
import math
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[3] / "FTR-Benchmark" / "ftr_envs" / "tasks" / "crossing" / "flipper_states.py"
_spec = importlib.util.spec_from_file_location("flipper_states", _PATH)
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)

# MARV limits as the diffusion configs set them (front 60/80, rear 60/80), logical convention.
LOW = torch.tensor([-math.radians(60), -math.radians(60), -math.radians(80), -math.radians(80)])
HIGH = torch.tensor([math.radians(80), math.radians(80), math.radians(60), math.radians(60)])


def _scores(idx: list[int], k: int = 7) -> torch.Tensor:
    s = -torch.ones(len(idx), k)
    s[torch.arange(len(idx)), torch.tensor(idx)] = 1.0
    return s


def test_pose_convention():
    poses = torch.rad2deg(fs.state_pose_targets(fs.DEFAULT_FLIPPER_STATES_DEG, LOW, HIGH))
    names = fs.state_names(fs.DEFAULT_FLIPPER_STATES_DEG)
    # ROS front -60 (up) -> logical +60; rear +60 (up) stays +60. Pairs share the angle.
    assert torch.allclose(poses[names.index("N")], torch.tensor([60.0, 60.0, 60.0, 60.0]))
    # DF: front 60 down -> logical -60, exactly the front's lower limit.
    assert torch.allclose(poses[names.index("DF")], torch.tensor([-60.0, -60.0, 0.0, 0.0]))
    assert torch.allclose(poses[names.index("AR")], torch.tensor([-30.0, -30.0, -80.0, -80.0]))
    assert (poses >= torch.rad2deg(LOW) - 1e-4).all() and (poses <= torch.rad2deg(HIGH) + 1e-4).all()


def test_pose_clamped_to_limits():
    poses = fs.state_pose_targets((("X", 90.0, -90.0),), LOW, HIGH)
    assert torch.allclose(poses[0], torch.tensor([LOW[0], LOW[1], LOW[2], LOW[3]]))


def test_switch_then_cooldown_blocks_then_allows():
    state = torch.zeros(1, dtype=torch.long)
    cd = torch.zeros(1, dtype=torch.long)
    # t0: N -> DF accepted
    state, cd, sw, bl = fs.step_flipper_state(_scores([4]), state, cd, cooldown_steps=2)
    assert state.item() == 4 and sw.item() and not bl.item()
    # t1: DF -> DR blocked (200 ms not elapsed)
    state, cd, sw, bl = fs.step_flipper_state(_scores([6]), state, cd, cooldown_steps=2)
    assert state.item() == 4 and not sw.item() and bl.item()
    # t2: allowed
    state, cd, sw, bl = fs.step_flipper_state(_scores([6]), state, cd, cooldown_steps=2)
    assert state.item() == 6 and sw.item()


def test_staying_is_never_blocked_and_any_order_allowed():
    state = torch.tensor([3])  # AR
    cd = torch.tensor([2])
    state, cd, sw, bl = fs.step_flipper_state(_scores([3]), state, cd, cooldown_steps=2)
    assert state.item() == 3 and not sw.item() and not bl.item()
    # AR -> DF is not a legal HFC-IL transition, but has no order constraint here.
    state, cd, sw, bl = fs.step_flipper_state(_scores([4]), state, cd, cooldown_steps=2)
    assert state.item() == 4 and sw.item()


def test_zero_cooldown_and_batch_independence():
    state = torch.tensor([0, 0])
    cd = torch.tensor([0, 2])
    state, cd, sw, _ = fs.step_flipper_state(_scores([1, 1]), state, cd, cooldown_steps=0)
    assert state.tolist() == [1, 0] and sw.tolist() == [True, False]
    state, cd, sw, _ = fs.step_flipper_state(_scores([2, 1]), state, cd, cooldown_steps=0)
    assert state.tolist() == [2, 1]


def test_neutral_zero_scores_select_n():
    state, _, _, _ = fs.step_flipper_state(torch.zeros(1, 7), torch.tensor([0]), torch.tensor([0]), 2)
    assert state.item() == 0
