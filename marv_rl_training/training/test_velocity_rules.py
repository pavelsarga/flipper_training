"""Rule-based speed controller (FTR-Benchmark ftr_envs/tasks/crossing/velocity_rules.py) on synthetic inputs.

The module is loaded by file path, as the Gazebo node loads it, so this runs without Isaac:
    python -m pytest src/flipper_training/marv_rl_training/training/test_velocity_rules.py
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "velocity_rules", ROOT / "src/FTR-Benchmark/ftr_envs/tasks/crossing/velocity_rules.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

DT = 0.05
CFG_DIR = ROOT / "configs/velocity_rule"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cfg(**criteria):
    return {"v_default": 1.0, "v_min": 0.1, "criteria": criteria}


def inputs(n=1, hm=None, v=0.5, flip_vel=0.0, roll=0.0, pitch=0.0, wx=0.0, wy=0.0, w=0.0, x=0.0, yaw=0.0):
    return R.Inputs(hm=torch.zeros(n, R.HM_ROWS, R.HM_COLS) if hm is None else hm,
                    v_meas=torch.full((n,), v), flip_vel=torch.full((n, 4), flip_vel),
                    rpy=torch.tensor([[roll, pitch, yaw]] * n), ang_vel=torch.tensor([[wx, wy, 0.0]] * n),
                    w_cmd=torch.full((n,), w),
                    pos=torch.tensor([[x * math.cos(yaw), x * math.sin(yaw)]] * n))


def step_map(edge_x, height):
    """Flat, then `height` from robot-frame x = edge_x forward (row 0 = front)."""
    x = (R.CENTER_ROW - torch.arange(R.HM_ROWS)) * R.HM_RES
    return torch.where(x >= edge_x - 1e-6, height, 0.0)[:, None].expand(R.HM_ROWS, R.HM_COLS)[None].clone()


def target(rs, x, ticks=1):
    """Speed target (before the slew limiter) after `ticks` ticks on input x."""
    for _ in range(ticks):
        rs(x, DT)
    return float(rs.debug["target"][0])


STEP = {"weight": 0.6, "mode": "bang", "trip": 0.08, "release": 0.05, "hold_s": 0.0,
        "lookahead": 1.0, "behind": 0.10, "baseline": 0.10}


def test_default_config_loads_and_flat_is_v_default():
    c = R.load_rule_config(CFG_DIR / "default.yaml")
    assert target(R.RuleSpeed(c), inputs()) == pytest.approx(c["v_default"])


def test_mapless_extends_default_and_needs_no_map():
    c = R.load_rule_config(CFG_DIR / "mapless.yaml")
    base = R.load_rule_config(CFG_DIR / "default.yaml")
    assert c["predictive"] is False and c["v_min"] == base["v_min"]
    assert c["criteria"]["turning"] == base["criteria"]["turning"]
    for k in ("tilt", "shake", "flippers"):                                            # stronger without the map
        assert c["criteria"][k]["weight"] >= base["criteria"][k]["weight"]
    assert c["criteria"]["flippers"]["weight"] > base["criteria"]["flippers"]["weight"]
    assert c["criteria"]["shake"]["weight"] > base["criteria"]["shake"]["weight"]
    assert c["criteria"]["tilt"]["hi"] == base["criteria"]["tilt"]["hi"]              # deep merge keeps the rest
    rs = R.RuleSpeed(c)
    assert not rs.uses_map and {k.name for k in rs.criteria} == {"tilt", "shake", "flippers", "turning"}
    assert target(rs, inputs()._replace(hm=None)) == pytest.approx(c["v_default"])
    shake = c["criteria"]["shake"]
    bump = inputs(hm=None, wx=2.0)                                                     # 0.5 s, through the EMA
    assert target(rs, bump, ticks=10) < c["v_default"] * (1 - 0.5 * shake["weight"])


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="unknown"):
        R.RuleSpeed(cfg(stepup=STEP))
    with pytest.raises(ValueError, match="unknown keys"):
        R.RuleSpeed(cfg(step_up={**STEP, "lookahaed": 1.0}))
    with pytest.raises(ValueError, match="unknown keys"):
        R.RuleSpeed({**cfg(), "slew": {"acel": 1.0}})


def test_step_up_and_down_ahead():
    rs = R.RuleSpeed(cfg(step_up=STEP, step_down={**STEP, "weight": 0.7}))
    assert target(rs, inputs(hm=step_map(0.6, 0.15))) == pytest.approx(0.4)
    rs.reset()
    assert target(rs, inputs(hm=step_map(0.6, -0.15))) == pytest.approx(0.3)
    rs.reset()
    assert target(rs, inputs(hm=step_map(1.5, 0.15))) == pytest.approx(1.0)      # beyond lookahead


def test_step_reduction_holds_until_edge_is_behind_the_centre():
    """The robot drives over a step: the edge moves back through the window tick by tick."""
    rs = R.RuleSpeed(cfg(step_up=STEP))
    held = {}
    for edge in torch.arange(0.80, -0.40, -0.05).tolist():
        held[round(edge, 2)] = target(rs, inputs(hm=step_map(edge, 0.15))) < 1.0
    assert all(held[e] for e in held if e >= 0.05)       # ahead and just past the centre: reduced
    assert not any(held[e] for e in held if e <= -0.15)  # `behind` (0.10) + baseline past it: released


def test_ramp_is_not_a_step():
    x = (R.CENTER_ROW - torch.arange(R.HM_ROWS)) * R.HM_RES
    ramp = (x.clamp_min(0) * math.tan(math.radians(20)))[:, None].expand(R.HM_ROWS, R.HM_COLS)[None].clone()
    rs = R.RuleSpeed(cfg(step_up=STEP))
    assert target(rs, inputs(hm=ramp)) == pytest.approx(1.0)


def test_roughness_scales_separate_pixel_noise_from_patches():
    g = torch.Generator().manual_seed(0)
    pixel = (torch.rand(1, R.HM_ROWS, R.HM_COLS, generator=g) - 0.5) * 0.06
    patch = torch.repeat_interleave(torch.repeat_interleave(
        (torch.rand(1, 12, 6, generator=g) - 0.5) * 0.06, 4, dim=1), 4, dim=2)[:, :R.HM_ROWS, :R.HM_COLS]
    rough = {"weight": 0.5, "mode": "linear", "lo": 0.0, "hi": 0.1, "quantile": 0.5, "x": [-0.35, 0.6]}

    def metric(hm, scales):
        rs = R.RuleSpeed(cfg(roughness={**rough, "scales": scales}))
        rs(inputs(hm=hm), DT)
        return float(rs.debug["roughness"][0][0])

    assert metric(pixel, [[1, 1.0]]) > 2 * metric(patch, [[1, 1.0]])    # scale 1 sees per-cell noise
    assert metric(pixel, [[4, 1.0]]) < 0.5 * metric(pixel, [[1, 1.0]])  # scale 4 averages it out
    assert metric(patch, [[4, 1.0]]) > 1.5 * metric(pixel, [[4, 1.0]])  # ... and sees the 4-cell patches
    assert metric(pixel, [[1, 0.0], [4, 1.0]]) == pytest.approx(metric(pixel, [[4, 1.0]]))   # weight 0 drops a band
    x = (R.CENTER_ROW - torch.arange(R.HM_ROWS)) * R.HM_RES
    plane = (0.3 * x[:, None] + 0.1 * torch.linspace(-0.5, 0.5, R.HM_COLS)[None, :])[None]
    assert metric(plane, [[1, 1.0], [2, 1.0], [4, 1.0]]) == pytest.approx(0.0, abs=1e-6)   # slope is not rough


def test_canopy_is_mismatch_and_distrusts_the_map():
    c = cfg(map_mismatch={"weight": 0.6, "mode": "linear", "lo": 0.05, "hi": 0.20},
            roughness={"weight": 0.5, "mode": "linear", "lo": 0.0, "hi": 0.01, "scales": [[1, 1.0]]})
    rs = R.RuleSpeed({**c, "distrust_map_at": 0.7})
    g = torch.Generator().manual_seed(1)
    grass = 0.3 + 0.03 * torch.rand(1, R.HM_ROWS, R.HM_COLS, generator=g)    # canopy above the tracks
    assert target(rs, inputs(hm=grass)) == pytest.approx(0.4)          # mismatch only: roughness distrusted
    rs = R.RuleSpeed(c)
    assert target(rs, inputs(hm=grass)) == pytest.approx(0.4 * 0.5)    # without distrust both act
    assert target(R.RuleSpeed(c), inputs()) == pytest.approx(1.0)


def test_mismatch_follows_the_body_plane():
    """A pitched robot on a matching slope is consistent; air under the body never counts."""
    rs = R.RuleSpeed(cfg(map_mismatch={"weight": 0.6, "mode": "linear", "lo": 0.05, "hi": 0.20}))
    pitch = math.radians(-20)                                           # nose up
    x = (R.CENTER_ROW - torch.arange(R.HM_ROWS)) * R.HM_RES
    slope = (x * math.tan(-pitch))[:, None].expand(R.HM_ROWS, R.HM_COLS)[None].clone()
    assert target(rs, inputs(hm=slope, pitch=pitch)) == pytest.approx(1.0)
    roll = math.radians(15)                                             # left side up
    y = (torch.arange(R.HM_COLS) - R.CENTER_COL) * R.HM_RES
    side = (y * math.tan(roll))[None, :].expand(R.HM_ROWS, R.HM_COLS)[None].clone()
    assert target(rs, inputs(hm=side, roll=roll)) == pytest.approx(1.0)
    assert target(rs, inputs(hm=torch.full((1, R.HM_ROWS, R.HM_COLS), -0.3))) == pytest.approx(1.0)


def test_tilt_roll_weighs_more_than_pitch():
    rs = R.RuleSpeed(cfg(tilt={"weight": 0.6, "mode": "linear", "lo": 0.0, "hi": 30.0, "roll_k": 1.0, "pitch_k": 0.5}))
    a = math.radians(20)
    assert target(rs, inputs(roll=a)) < target(rs, inputs(pitch=a))
    assert target(rs, inputs(roll=a)) == pytest.approx(1 - 0.6 * 20 / 30)


def test_tilt_nose_down_pitch_has_its_own_weight():
    rs = R.RuleSpeed(cfg(tilt={"weight": 0.6, "mode": "linear", "lo": 0.0, "hi": 30.0, "pitch_k": 0.5,
                               "pitch_down_k": 1.0}))
    a = math.radians(20)
    assert target(rs, inputs(pitch=a)) == pytest.approx(1 - 0.6 * 20 / 30)      # + = nose down: full weight
    assert target(rs, inputs(pitch=-a)) == pytest.approx(1 - 0.6 * 10 / 30)     # nose up: pitch_k
    rs = R.RuleSpeed(cfg(tilt={"weight": 0.6, "mode": "linear", "lo": 0.0, "hi": 30.0, "pitch_k": 0.5}))
    assert target(rs, inputs(pitch=a)) == target(rs, inputs(pitch=-a))         # without it: symmetric


def test_flippers_bang_with_hold():
    rs = R.RuleSpeed(cfg(flippers={"weight": 0.5, "mode": "bang", "trip": 0.3, "release": 0.1, "hold_s": 0.5}))
    assert target(rs, inputs(flip_vel=0.5)) == pytest.approx(0.5)
    assert target(rs, inputs(flip_vel=0.2)) == pytest.approx(0.5)        # hysteresis: between off and on
    assert target(rs, inputs(flip_vel=0.0), ticks=9) == pytest.approx(0.5)   # 0.45 s of hold
    assert target(rs, inputs(flip_vel=0.0)) == pytest.approx(1.0)        # 0.50 s: released


def test_linear_peak_hold():
    rs = R.RuleSpeed(cfg(turning={"weight": 0.4, "mode": "linear", "lo": 0.0, "hi": 1.0, "hold_s": 0.25}))
    assert target(rs, inputs(w=1.0)) == pytest.approx(0.6)
    assert target(rs, inputs(w=0.5), ticks=4) == pytest.approx(0.6)     # 0.20 s: the peak is held
    assert target(rs, inputs(w=0.5)) == pytest.approx(0.8)              # 0.25 s: follows again
    assert target(rs, inputs(w=0.8)) == pytest.approx(0.68)             # a new peak is taken at once


def test_turning_linear_and_shake_ema():
    rs = R.RuleSpeed(cfg(turning={"weight": 0.4, "mode": "linear", "lo": 0.2, "hi": 1.0}))
    assert target(rs, inputs(w=0.6)) == pytest.approx(0.8)
    assert target(rs, inputs(w=-1.5)) == pytest.approx(0.6)
    rs = R.RuleSpeed(cfg(shake={"weight": 0.4, "mode": "linear", "lo": 0.0, "hi": 1.0, "tau": 0.3}))
    assert target(rs, inputs(wx=0.6, wy=0.8)) == pytest.approx(0.6)     # first tick: EMA starts at the metric
    first = target(rs, inputs())
    assert 0.6 < first < 1.0                                            # then decays, not jumps


def test_compounding_floor_and_cap():
    flat = {"mode": "linear", "lo": 0.0, "hi": 1.0}
    rs = R.RuleSpeed(cfg(turning={**flat, "weight": 0.9}, flippers={**flat, "weight": 0.9}))
    assert target(rs, inputs(w=1.0, flip_vel=1.0)) == pytest.approx(0.1)    # 0.01 -> v_min
    rs = R.RuleSpeed(cfg(turning={**flat, "weight": 0.9},
                         tilt={"weight": 0.5, "mode": "linear", "lo": 0.0, "hi": 30.0, "cap": 0.05}))
    assert target(rs, inputs(w=1.0, roll=math.radians(35))) == pytest.approx(0.05)   # special case below v_min
    assert target(rs, inputs(w=1.0, roll=math.radians(25))) == pytest.approx(0.1)    # not fully triggered


def test_predictive_combine_strongest():
    crit = {"step_up": STEP, "roughness": {"weight": 0.5, "mode": "linear", "lo": 0.0, "hi": 0.01, "scales": [[1, 1.0]]},
            "turning": {"weight": 0.5, "mode": "linear", "lo": 0.0, "hi": 1.0}}
    g = torch.Generator().manual_seed(3)
    rough_step = step_map(0.6, 0.15) + 0.05 * torch.rand(1, R.HM_ROWS, R.HM_COLS, generator=g)
    x = inputs(hm=rough_step, w=1.0)
    assert target(R.RuleSpeed(cfg(**crit)), x) == pytest.approx(0.4 * 0.5 * 0.5)                     # all compound
    assert target(R.RuleSpeed({**cfg(**crit), "predictive_combine": "strongest"}), x) == pytest.approx(0.4 * 0.5)
    with pytest.raises(ValueError, match="predictive_combine"):
        R.RuleSpeed({**cfg(**crit), "predictive_combine": "max"})


def test_slew_limits():
    rs = R.RuleSpeed({**cfg(turning={"weight": 0.9, "mode": "linear", "lo": 0.0, "hi": 1.0}),
                      "slew": {"accel": 0.4, "decel": 2.0}})
    v = [float(rs(inputs(), DT)) for _ in range(5)]
    assert v == pytest.approx([0.02 * (k + 1) for k in range(5)])        # 0.4 m/s^2 from rest
    rs.v_prev[:] = 1.0
    assert float(rs(inputs(w=1.0), DT)) == pytest.approx(0.9)            # 2 m/s^2 down


STALL = {"window_s": 2.0, "progress_m": 0.2, "release_m": 0.3, "max_boost_s": 3.0, "boost": [0.4, 0.6]}
SLOW = {"turning": {"weight": 0.9, "mode": "linear", "lo": 0.0, "hi": 1.0}}   # w = 1 -> target 0.1 = v_min


def drive(rs, xs, yaw=0.0):
    """One tick per position along the heading; the speed targets."""
    return [target(rs, inputs(w=1.0, x=x, yaw=yaw)) for x in xs]


def test_stall_guard_boosts_held_until_progress():
    rs = R.RuleSpeed({**cfg(**SLOW), "stall": STALL})
    out = drive(rs, [0.0] * 50)                                       # 2.5 s without moving
    first = next(i for i, v in enumerate(out) if v > 0.2)
    assert 1.95 <= first * DT <= 2.1 and out[first] == pytest.approx(0.4)   # window full, no progress -> level 1
    assert drive(rs, [0.1, 0.2])[-1] == pytest.approx(0.4)            # held: 0.2 m < release_m
    assert drive(rs, [0.31])[-1] == pytest.approx(0.1, rel=1e-5)      # 0.31 m: released, rules again
    assert rs.stall.level[0] == 0                                     # got over it: escalation reset


def test_stall_guard_rocking_is_not_progress_and_slow_climbing_is():
    rs = R.RuleSpeed({**cfg(**SLOW), "stall": STALL})
    rock = [0.15 * math.sin(2 * math.pi * k * DT) for k in range(60)]      # +-0.15 m at 1 Hz, going nowhere
    assert max(drive(rs, rock)) == pytest.approx(0.4)
    rs = R.RuleSpeed({**cfg(**SLOW), "stall": STALL})
    creep = [0.15 * k * DT for k in range(100)]                       # 0.15 m/s: 0.3 m per window
    assert max(drive(rs, creep, yaw=1.0)) == pytest.approx(0.1, rel=1e-5)


def test_stall_guard_escalates_when_a_boost_runs_out():
    rs = R.RuleSpeed({**cfg(**SLOW), "stall": STALL})
    out = [round(v, 3) for v in drive(rs, [0.0] * 220)]             # 11 s stuck
    first_low, first_high = out.index(0.4), out.index(0.6)
    assert first_low * DT == pytest.approx(2.0, abs=0.1)
    assert (first_high - first_low) * DT == pytest.approx(3.0, abs=0.1)    # ran out: straight to the next level
    pause = [k for k in range(first_high, len(out)) if out[k] == 0.1]       # the top level ran out: a new window
    assert pause and (pause[0] - first_high) * DT == pytest.approx(3.0, abs=0.1)
    assert out[pause[0] + int(2.0 / DT) + 2] == 0.6                          # stuck after it: the top level again


def test_stall_guard_roll_limit_keeps_the_first_level():
    rs = R.RuleSpeed({**cfg(**SLOW), "stall": {**STALL, "max_roll_deg": 15.0}})
    level = lambda roll: target(rs, inputs(w=1.0, roll=math.radians(roll)))
    out = [round(level(0.0), 3) for _ in range(110)]                 # 5.5 s stuck, level: escalated to 0.6
    assert out[-1] == 0.6
    assert round(level(20.0), 3) == 0.4                               # rolled past 15 deg: first level only
    assert round(level(5.0), 3) == 0.6                                # level again once the roll is back


def test_stall_guard_nose_down_keeps_the_first_level():
    rs = R.RuleSpeed({**cfg(**SLOW), "stall": {**STALL, "max_pitch_down_deg": 10.0}})
    level = lambda pitch: target(rs, inputs(w=1.0, pitch=math.radians(pitch)))
    out = [round(level(0.0), 3) for _ in range(110)]                 # 5.5 s stuck, level: escalated to 0.6
    assert out[-1] == 0.6
    assert round(level(20.0), 3) == 0.4                               # nose down over a lip: first level only
    assert round(level(-20.0), 3) == 0.6                              # nose up (climbing): not limited


def test_batched_envs_are_independent_and_reset_per_env():
    rs = R.RuleSpeed(cfg(flippers={"weight": 0.5, "mode": "bang", "trip": 0.3, "hold_s": 10.0}), num_envs=3)
    x = inputs(n=3)._replace(flip_vel=torch.tensor([[0.5] * 4, [0.0] * 4, [0.5] * 4]))
    rs(x, DT)
    assert rs.debug["target"].tolist() == pytest.approx([0.5, 1.0, 0.5])
    rs.reset(torch.tensor([2]))
    rs(inputs(n=3), DT)
    assert rs.debug["target"].tolist() == pytest.approx([0.5, 1.0, 1.0])  # env 0 held, env 2 reset
    assert rs.factors(0) == pytest.approx({"flippers": 0.5})


def test_isaac_adapter_matches_the_gazebo_input_path(monkeypatch):
    """velocity_control.RuleVelocity (Isaac) only unpacks the env's tensors: fed the same data as
    velocity_rule_node (Gazebo) builds its Inputs from, it gives the identical speed."""
    for name in ("ftr_envs", "ftr_envs.tasks", "ftr_envs.tasks.crossing"):     # skip the Isaac package inits
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ftr_envs.tasks.crossing.velocity_rules", R)
    VC = load(ROOT / "src/FTR-Benchmark/ftr_envs/tasks/crossing/velocity_control.py", "velocity_control")
    F = load(ROOT / "src/flipper_training/marv_rl_training/training/velocity_il_features.py", "velocity_il_features")

    n, g = 3, torch.Generator().manual_seed(2)
    isaac = VC.make_velocity_controller({"type": "rule", "config": str(CFG_DIR / "default.yaml"), "history": 1}, n, "cpu")
    gazebo = R.RuleSpeed(R.load_rule_config(CFG_DIR / "default.yaml"), n)
    for t in range(20):
        hm = 0.05 * torch.randn(n, R.HM_ROWS, R.HM_COLS, generator=g) + step_map(0.8 - 0.05 * t, 0.15)
        v, v_prev, rpy, ang_vel = (torch.rand(n, generator=g), torch.rand(n, generator=g),
                                   0.4 * torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g))
        flip_vel, angle, w = (torch.randn(n, 4, generator=g), torch.randn(n, 4, generator=g), torch.randn(n, generator=g))
        proprio = torch.stack([torch.from_numpy(F.proprio_frame(float(v[i]), float(v_prev[i]), flip_vel[i].numpy(),
                                                                angle[i].numpy())) for i in range(n)])
        pos = torch.rand(n, 2, generator=g)
        v_isaac = isaac(hm[:, None], proprio[:, None], {"rpy": rpy, "ang_vel": ang_vel, "w_cmd": w, "pos": pos, "dt": DT})
        v_gazebo = gazebo(R.Inputs(hm=hm, v_meas=v, flip_vel=flip_vel, rpy=rpy, ang_vel=ang_vel, w_cmd=w, pos=pos), DT)
        assert torch.equal(v_isaac, v_gazebo)
    isaac.reset(torch.tensor([1]))
    assert isaac.rules.v_prev[1] == 0.0 and isaac.rules.v_prev[0] > 0.0

    mapless = VC.make_velocity_controller({"type": "rule", "config": str(CFG_DIR / "default.yaml"),
                                           "overrides": {"predictive": False, "criteria": {"tilt": {"weight": 0.9}}}}, n, "cpu")
    assert not mapless.rules.uses_map
    assert {c.name: c.weight for c in mapless.rules.criteria}["tilt"] == 0.9


def test_bearing_heading_distance_ramp(monkeypatch):
    for name in ("ftr_envs", "ftr_envs.tasks", "ftr_envs.tasks.crossing"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ftr_envs.tasks.crossing.velocity_rules", R)
    VC = load(ROOT / "src/FTR-Benchmark/ftr_envs/tasks/crossing/velocity_control.py", "velocity_control")
    start = torch.tensor([[0.0, 0.0, 0.2]] * 3)
    target = torch.tensor([[-6.0, 0.0, 0.2]] * 3)                         # the course runs along -x
    pos = torch.tensor([[-0.0, 0.3, 0.2], [-3.0, 0.3, 0.2], [-6.5, 0.3, 0.2]])   # start, half way, past the goal
    yaw = torch.full((3,), math.pi)
    full = VC.bearing_heading(pos, yaw, target, gain=0.5, max_w=10.0)
    ramp = VC.bearing_heading(pos, yaw, target, gain=0.5, max_w=10.0, distance_ramp=True, starts=start)
    assert ramp[0] == pytest.approx(0.0) and ramp[1] == pytest.approx(0.5 * full[1], rel=1e-5)
    assert ramp[2] == pytest.approx(full[2], rel=1e-5)                    # clamped at 1 past the goal


def test_state_survives_an_inference_mode_eval():
    """train_diffusion runs its mid-training eval under torch.inference_mode and the training rollout
    under no_grad only: state rebound during the eval must stay writable in place afterwards (the
    stall guard's `armed_s += dt` killed job 12073363 on the first training step after its first eval)."""
    n = 3
    rs = R.RuleSpeed(R.load_rule_config(CFG_DIR / "default.yaml"), n)
    with torch.no_grad():
        for t in range(5):
            rs(inputs(n, x=0.02 * t), DT)
    with torch.inference_mode():
        for t in range(30):
            rs(inputs(n, x=0.1 + 0.001 * t), DT)          # creeping: the stall guard rebinds its state
        rs.reset(torch.tensor([0]))
    with torch.no_grad():
        for t in range(5):
            rs(inputs(n, x=0.2 + 0.02 * t), DT)
        rs.reset(torch.tensor([1]))
        assert torch.isfinite(rs(inputs(n, x=0.4), DT)).all()
