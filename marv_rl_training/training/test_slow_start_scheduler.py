"""SlowStartCoefScheduler: phases, loss handling, cooldown, timeout, resume, and a replay of the
smoothness x3 runs' eval SR (open loop: the SR the runs actually produced under the old
SRGatedCoefScheduler, fed to the new controller to check when it would have reacted).

Pure Python, no Isaac needed:
  env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robot/conda/envs/isaaclab/bin/python \
      -m pytest src/flipper_training/marv_rl_training/training/test_slow_start_scheduler.py -q
"""
from types import SimpleNamespace

import pytest

from marv_rl_training.utils.cfg_schedulers import SlowStartCoefScheduler

M = 1_000_000

# eval (collected_frames, success_rate), logs/diff_p1_smooth_ramp0_temporal_x3_11853664 / _x3_11853665
X3_TEMPORAL = [(2359296, 0.3056), (4456448, 0.4401), (6553600, 0.4418), (8650752, 0.4703), (10747904, 0.4828),
               (12845056, 0.521), (14942208, 0.5804), (17039360, 0.6282), (19136512, 0.711), (21233664, 0.7875),
               (23330816, 0.8098), (25427968, 0.7958), (27525120, 0.785), (29622272, 0.8311), (31719424, 0.807),
               (33816576, 0.7992), (35913728, 0.786), (38010880, 0.7906), (40108032, 0.7809), (42205184, 0.76),
               (44302336, 0.6906)]
X3_ALL = [(2359296, 0.4084), (4456448, 0.447), (6553600, 0.3837), (8650752, 0.4474), (10747904, 0.4541),
          (12845056, 0.4751), (14942208, 0.5247), (17039360, 0.556), (19136512, 0.5303), (21233664, 0.6522),
          (23330816, 0.7476), (25427968, 0.7223), (27525120, 0.7666), (29622272, 0.7807), (31719424, 0.811),
          (33816576, 0.7978), (35913728, 0.7575), (38010880, 0.7385), (40108032, 0.6674)]


def make(**kw):
    cfg = SimpleNamespace()
    opts = dict(target_value=0.20, step=0.015, ssthresh_init=4, min_frames=15 * M, sr_min=0.45,
                slope_window=3, drop_slope=0.015, linear_drop_steps=2, cooldown_evals=2)
    opts.update(kw)
    return cfg, SlowStartCoefScheduler(cfg, "smooth_coef", 0.0, **opts)


def progress_trace(s, evals):
    out = []
    for frames, sr in evals:
        s.on_eval(sr, frames=frames)
        out.append(s._progress)
    return out


def test_waits_for_both_frames_and_sr():
    cfg, s = make()
    s.on_eval(0.9, frames=10 * M)          # SR fine, too early
    s.on_eval(0.40, frames=16 * M)         # late enough, SR too low
    assert cfg.smooth_coef == 0.0 and s._phase == "wait"
    s.on_eval(0.46, frames=18 * M)
    assert s._phase == "slow_start" and cfg.smooth_coef == pytest.approx(0.015)


def test_slow_start_doubles_then_linear_then_cap():
    cfg, s = make()
    trace = progress_trace(s, [(16 * M + i, 0.8) for i in range(20)])
    assert trace[:6] == [1, 2, 4, 5, 6, 7]
    assert max(trace) == 14 and cfg.smooth_coef == pytest.approx(0.20)   # 14 * 0.015 = 0.21, clipped


def test_max_progress_float_rounding():
    _, s = make(target_value=0.15, step=0.015)
    assert s._max_progress == 10


def test_loss_in_slow_start_halves_ssthresh():
    _, s = make(target_value=1.0, ssthresh_init=32)
    # enter and double 1 -> 2 -> 4 -> 8; then window (.80 .80 .75) slope -0.025 -> loss while
    # still in slow start: ssthresh = 8 // 2 = 4, progress = 4, continue linearly
    trace = progress_trace(s, [(16 * M, 0.80), (17 * M, 0.80), (18 * M, 0.80), (19 * M, 0.80), (20 * M, 0.75)])
    assert trace == [1, 2, 4, 8, 4]
    assert s._phase == "linear" and s._ssthresh == 4 and s._losses == 1


def test_linear_loss_drops_two_and_is_counted_once():
    _, s = make()
    rising = [(16 * M + i, 0.80) for i in range(8)]          # progress 1 2 4 5 6 7 8 9
    progress_trace(s, rising)
    assert s._progress == 9 and s._phase == "linear"
    falling = [(30 * M + i, 0.80 - 0.04 * k) for i, k in enumerate(range(1, 6))]
    trace = progress_trace(s, falling)
    # eval 1: window (.80 .80 .76) slope -0.02 -> loss, 9 - 2 = 7, history cleared
    # evals 2, 3: cooldown, hold -- the windows (.80 .76 .72), (.76 .72 .68) that still
    #             contain the drop are never formed
    # eval 4: window of three post-loss evals (.72 .68 .64) slope -0.04 -> loss again, 5
    # eval 5: cooldown
    assert trace == [7, 7, 7, 5, 5]
    assert s._losses == 2


def test_timeout_resets_to_zero_and_reenters_with_half_ssthresh():
    cfg, s = make()
    progress_trace(s, [(16 * M + i, 0.80) for i in range(8)])   # progress 9
    s.on_eval(0.40, frames=40 * M)
    assert cfg.smooth_coef == 0.0 and s._phase == "wait" and s._ssthresh == 4
    trace = progress_trace(s, [(41 * M, 0.50), (42 * M, 0.55), (43 * M, 0.60), (44 * M, 0.62)])
    assert trace == [1, 2, 4, 5]


def test_resume_from_state_dict_continues_identically():
    evals = X3_ALL
    _, ref = make()
    ref_trace = progress_trace(ref, evals)
    for cut in range(1, len(evals)):
        _, a = make()
        progress_trace(a, evals[:cut])
        _, b = make()
        b.load_state_dict(a.state_dict())
        assert progress_trace(b, evals[cut:]) == ref_trace[cut:], f"diverged after resuming at eval {cut}"


def test_replay_x3_temporal():
    cfg, s = make()
    trace = progress_trace(s, X3_TEMPORAL)
    frames = [f for f, _ in X3_TEMPORAL]
    # enters at 17.0 M (14.9 M is below min_frames), 1 2 4 slow start, then linear
    assert trace[frames.index(14942208)] == 0 and trace[frames.index(17039360)] == 1
    # first loss: SR 0.831 -> 0.807 -> 0.799 at 33.8 M (slope -0.016), progress 9 -> 7
    assert trace[frames.index(31719424)] == 9 and trace[frames.index(33816576)] == 7
    # cooldown holds 35.9 / 38.0, then grows; second loss at 42.2 M (slope -0.0153)
    assert trace[frames.index(35913728):] == [7, 7, 8, 6, 6]
    assert max(trace) == 9                     # kappa peaked at 0.135, under the 0.20 cap


def test_replay_x3_all_channels():
    _, s = make()
    trace = progress_trace(s, X3_ALL)
    frames = [f for f, _ in X3_ALL]
    assert trace[frames.index(33816576)] == 10
    # loss at the first real drop, 0.798 -> 0.758 at 35.9 M (slope -0.027): 10 -> 8, then cooldown
    assert trace[frames.index(35913728):] == [8, 8, 8]
