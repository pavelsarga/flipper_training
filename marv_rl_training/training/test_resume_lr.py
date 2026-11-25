"""reconcile_resumed_lrs: a resume must keep the schedule's progress, and a deliberate config
LR change must rescale it rather than restart it. CPU only.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest src/flipper_training/marv_rl_training/training/test_resume_lr.py
"""

import torch

from marv_rl_training.training.trainer_common import reconcile_resumed_lrs

TOTAL = 100
LR = 5e-4


def _make(lr=LR):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p], "lr": lr, "name": "policy"}])
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.2, total_iters=TOTAL)
    return opt, sched


def _step(opt, sched, n):
    for _ in range(n):
        opt.step()
        sched.step()


def _resume(opt, sched, cfg_lr):
    """Save + rebuild + load + reconcile, as a relaunch does."""
    o_state, s_state = opt.state_dict(), sched.state_dict()
    opt2, sched2 = _make(cfg_lr)
    opt2.load_state_dict(o_state)
    sched2.load_state_dict(s_state)
    msgs = reconcile_resumed_lrs(opt2, sched2, {"policy": cfg_lr})
    return opt2, sched2, msgs


def test_unchanged_config_keeps_decay_across_many_relaunches():
    ref_opt, ref_sched = _make()
    _step(ref_opt, ref_sched, TOTAL)
    expected = ref_opt.param_groups[0]["lr"]  # 0.2 * LR

    opt, sched = _make()
    for _ in range(10):  # ten relaunches, 10 steps each
        _step(opt, sched, 10)
        opt, sched, msgs = _resume(opt, sched, LR)
        assert msgs == []
    assert abs(opt.param_groups[0]["lr"] - expected) < 1e-12
    assert abs(expected - 0.2 * LR) < 1e-12


def test_changed_config_rescales_at_same_progress():
    opt, sched = _make()
    _step(opt, sched, 50)
    before = opt.param_groups[0]["lr"]  # 0.6 * LR
    opt, sched, msgs = _resume(opt, sched, 2 * LR)
    assert len(msgs) == 1
    assert abs(opt.param_groups[0]["lr"] - 2 * before) < 1e-12
    _step(opt, sched, 50)
    assert abs(opt.param_groups[0]["lr"] - 0.2 * 2 * LR) < 1e-12  # ends at end_factor of the NEW base
    assert sched.base_lrs == [2 * LR]
