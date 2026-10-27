"""VelocityAgent (velocity_rl.py): rewards, episode bookkeeping, PPO update, export. CPU, no Isaac.

    python -m pytest src/flipper_training/marv_rl_training/training/test_velocity_rl.py
"""
import torch
from omegaconf import OmegaConf

from marv_rl_training.policies.velocity_controller import PROPRIO_DIM, VelocityControllerExport, VelocityNet
from marv_rl_training.training.velocity_rl import EXTRA_DIM, VelocityAgent, VelocityRewardConfig, VelocityRLConfig

T, N, DT = 5, 8, 0.05
NO_TERMS = dict(mean_coef=0.0, std_coef=0.0, jerk_coef=0.0, rate_coef=0.0, stall_coef=0.0, fail_penalty=0.0)


def make_il_dir(tmp_path):
    """A small VelocityNet saved the way train_velocity_il.py saves one."""
    torch.manual_seed(0)
    kw = dict(history=T, channels=(8, 8, 8), proprio_hidden=16, head_hidden=32, head_layers=1, v_max=1.0,
              drop_proprio=(1,))
    net = VelocityNet(**kw)
    d = tmp_path / "il"
    d.mkdir(parents=True)
    torch.save(net.state_dict(), d / "net_best.pth")
    OmegaConf.save(OmegaConf.create(dict(history=T, channels=[8, 8, 8], proprio_hidden=16, head_hidden=32,
                                         head_layers=1, drop_proprio=[1])), d / "config.yaml")
    exp = VelocityControllerExport(net.eval(), 0.3, torch.zeros(PROPRIO_DIM), torch.ones(PROPRIO_DIM))
    torch.jit.script(exp).save(str(d / "velocity_controller.pt"))
    return str(d)


def agent(tmp_path, only=None, **kw):
    """`only`: reward terms to switch on (dict of weights); every other term is 0. None = defaults."""
    reward = VelocityRewardConfig(**{**NO_TERMS, **only}) if only is not None else VelocityRewardConfig()
    cfg = VelocityRLConfig(il_dir=make_il_dir(tmp_path), record_envs=N, minibatch=64, reward=reward, **kw)
    return VelocityAgent(cfg, N, DT, "cpu")


def obs():
    return torch.randn(N, T, 45, 21) * 0.05, torch.randn(N, T, PROPRIO_DIM) * 0.1


def step(a, v=None, ref=None, xy=None, success=None, fail=None, timeout=None, rates=None):
    """One control step; `v` / `ref` override the applied and the IL net's speed (rewards in isolation)."""
    a.act(*obs())
    if v is not None:
        a.v = torch.full((N,), float(v))
    if ref is not None:
        a.v_ref = torch.full((N,), float(ref))
    z = torch.zeros(N, dtype=torch.bool)
    return a.end_step(torch.zeros(N, 2) if xy is None else xy, torch.zeros(N, 3) if rates is None else rates,
                      z if success is None else success, z if fail is None else fail, z if timeout is None else timeout)


# ---------------------------------------------------------------------------- reward terms
def test_no_reward_for_speed_or_success(tmp_path):
    """Default weights, policy == IL net at a constant speed: going faster gains nothing."""
    rewards = {}
    for v in (0.2, 0.6, 1.0):
        a = agent(tmp_path / str(v))
        rewards[v] = [float(step(a, v=v, ref=v)[0]) for _ in range(50)]
    assert rewards[0.2] == rewards[0.6] == rewards[1.0] and all(r == 0.0 for r in rewards[1.0])
    a = agent(tmp_path / "s")
    assert float(step(a, v=0.6, ref=0.6, success=torch.ones(N, dtype=torch.bool))[0]) == 0.0


def test_anchor_penalises_a_sustained_offset_from_the_il_mean(tmp_path):
    a = agent(tmp_path, only=dict(mean_coef=2.0))
    assert all(float(step(a, v=0.5, ref=0.5)[0]) == 0.0 for _ in range(20))
    b = agent(tmp_path / "b", only=dict(mean_coef=2.0))
    assert all(abs(float(step(b, v=0.4, ref=0.5)[0]) + 2.0 * 0.01) < 1e-6 for _ in range(20))
    c = agent(tmp_path / "c", only=dict(mean_coef=2.0))       # faster than the IL net costs the same way
    assert all(abs(float(step(c, v=0.7, ref=0.5)[0]) + 2.0 * 0.04) < 1e-6 for _ in range(20))


def test_a_short_slowdown_barely_moves_the_anchor(tmp_path):
    """1 s at 0.2 m/s where the IL net does 0.5, inside a 10 s window: < 10 % of a sustained 0.1 offset."""
    a = agent(tmp_path, only=dict(mean_coef=2.0))
    for _ in range(200):
        step(a, v=0.5, ref=0.5)
    worst = min(float(step(a, v=0.2, ref=0.5)[0]) for _ in range(20))
    assert worst > -0.1 * 2.0 * 0.01, worst


def test_range_rewards_the_spread_up_to_the_il_spread(tmp_path):
    def run(v_lo, v_hi, r_lo, r_hi, sub):
        a = agent(tmp_path / sub, only=dict(std_coef=1.0))
        for i in range(3000):
            r = step(a, v=v_hi if i % 2 else v_lo, ref=r_hi if i % 2 else r_lo)
        return float(r[0])
    assert abs(run(0.3, 0.7, 0.45, 0.55, "wide") - 0.05) < 5e-3        # spread 0.2 > IL 0.05: capped at 0.05
    assert abs(run(0.45, 0.55, 0.3, 0.7, "narrow") - 0.05) < 5e-3      # spread 0.05 < IL 0.2: paid 0.05
    assert abs(run(0.5, 0.5, 0.3, 0.7, "const")) < 1e-6                # constant speed: nothing


def test_jerk_only_no_penalty_for_a_steady_speed_change(tmp_path):
    a = agent(tmp_path, only=dict(jerk_coef=1.0))
    assert all(abs(float(step(a, v=0.1 * i, ref=0.5)[0])) < 1e-6 for i in range(1, 8))   # constant accel: free
    b = agent(tmp_path / "b", only=dict(jerk_coef=1.0))
    assert float(step(b, v=0.2)[0]) == 0.0 and float(step(b, v=0.4)[0]) == 0.0   # first two steps: no 2nd diff
    assert abs(float(step(b, v=0.4)[0]) + 0.04) < 1e-6                 # 0.4 - 0.8 + 0.2 = -0.2 -> -0.04


def test_attitude_rate_penalty(tmp_path):
    a = agent(tmp_path, only=dict(rate_coef=0.5))
    rates = torch.tensor([[1.0, 2.0, 5.0]]).repeat(N, 1)              # yaw rate is not penalised
    assert torch.allclose(step(a, v=0.3, ref=0.3, rates=rates), torch.full((N,), -0.5 * 5.0))


def test_stall_only_after_two_seconds_below_threshold(tmp_path):
    a = agent(tmp_path, only=dict(stall_coef=1.0))
    rewards = [float(step(a, v=0.05, ref=0.05)[0]) for _ in range(45)]
    assert all(r == 0.0 for r in rewards[:39]), rewards[:40]          # 39 steps = 1.95 s: not yet
    assert all(r == -1.0 for r in rewards[41:]), rewards[40:]         # past 2 s: stalled
    assert float(step(a, v=0.2, ref=0.2)[0]) == 0.0                   # moving again clears it at once
    b = agent(tmp_path / "b", only=dict(stall_coef=1.0))
    assert all(float(step(b, v=0.075, ref=0.075)[0]) == 0.0 for _ in range(60))   # at the threshold: never


def test_tipover_and_timeout_are_failures(tmp_path):
    a = agent(tmp_path, only=dict(fail_penalty=7.0))
    fail, timeout = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool)
    fail[0], timeout[1] = True, True
    r = step(a, v=0.3, ref=0.3, fail=fail, timeout=timeout)
    assert r[0] == -7.0 and r[1] == -7.0 and torch.all(r[2:] == 0.0)


def test_rolling_stats_restart_at_an_episode_start(tmp_path):
    a = agent(tmp_path, only=dict(mean_coef=2.0))
    for _ in range(50):
        step(a, v=0.3, ref=0.6)                                       # big offset in the old episode
    a.reset(torch.arange(N))
    assert float(step(a, v=0.5, ref=0.5)[0]) == 0.0                   # a new episode starts clean


def test_abort_ends_the_episode_without_a_penalty(tmp_path):
    a = agent(tmp_path, only={})
    abort = torch.zeros(N, dtype=torch.bool)
    abort[0] = True
    z = torch.zeros(N, dtype=torch.bool)
    a.act(*obs())
    r = a.end_step(torch.zeros(N, 2), torch.zeros(N, 3), z, z, z, abort)
    assert torch.all(r == 0.0) and bool(a._buf[-1]["done"][0]) and not bool(a._buf[-1]["done"][1])


# ---------------------------------------------------------------------------- critic / learning
def test_critic_sees_the_rolling_state(tmp_path):
    a = agent(tmp_path)
    hm, pr = obs()
    h, p = a._norm(hm, pr)
    e0 = torch.zeros(N, EXTRA_DIM)
    e1 = e0.clone()
    e1[:, 0] = 1.0
    with torch.no_grad():
        assert not torch.allclose(a.critic(h, p, e0), a.critic(h, p, e1))
    a.act(hm, pr)
    assert a._pending["extra"].shape == (N, EXTRA_DIM)


def test_evaluating_records_nothing_and_uses_the_mean(tmp_path):
    a = agent(tmp_path)
    hm, pr = obs()
    with a.evaluating():
        v1 = a.act(hm, pr)
        a.end_step(torch.zeros(N, 2), torch.zeros(N, 3), *(torch.zeros(N, dtype=torch.bool),) * 3)
        v2 = a.act(hm, pr)
    assert torch.equal(v1, v2) and a._buf == [] and a.pop_stats()["vel_rl/v_mean"] == 0.0


def test_export_reproduces_the_actor_mean(tmp_path):
    a = agent(tmp_path)
    path = str(tmp_path / "exp.pt")
    a.export(path)
    hm, pr = obs()
    with a.evaluating():
        v = a.act(hm, pr)
    assert torch.allclose(torch.jit.load(path)(hm, pr), v, atol=1e-6)


def _train(a, reward_fn, iters, steps=40):
    hm, pr = obs()
    for _ in range(iters):
        for _ in range(steps):
            a.act(hm, pr)
            a._buf.append(dict(**a._pending, reward=reward_fn(a.v).clone(), done=torch.zeros(N, dtype=torch.bool)))
            a._pending = None
        a.update()
    with a.evaluating():
        return a.act(hm, pr).mean()


def test_ppo_follows_the_reward(tmp_path):
    """A reward that pays for low speed must lower the mean (sign of the policy gradient). gamma 0:
    the reward is immediate, so the advantage is the step's own reward -- a clean sign check."""
    a = agent(tmp_path, actor_lr=3e-3, critic_warmup=0, anchor_coef=0.0, init_std=0.1, gamma=0.0, gae_lambda=0.0)
    hm, pr = obs()
    with a.evaluating():
        before = a.act(hm, pr).mean()
    after = _train(a, lambda v: -v, 15)
    assert after < before - 0.05, (float(before), float(after))


def test_loss_anchor_holds_the_mean_near_the_il_net(tmp_path):
    a = agent(tmp_path, actor_lr=3e-3, critic_warmup=0, anchor_coef=1e4, init_std=0.1)
    hm, pr = obs()
    with a.evaluating():
        before = a.act(hm, pr).mean()
    assert abs(float(_train(a, lambda v: -v, 5) - before)) < 0.03


def test_critic_warmup_freezes_the_actor(tmp_path):
    a = agent(tmp_path, critic_warmup=2)
    w0 = {k: v.clone() for k, v in a.actor.state_dict().items()}
    for _ in range(2):
        for _ in range(10):
            step(a)
        assert a.update()["vel_rl/learning_actor"] == 0.0
    assert all(torch.equal(w0[k], v) for k, v in a.actor.state_dict().items())
    for _ in range(10):
        step(a)
    assert a.update()["vel_rl/learning_actor"] == 1.0


def test_state_dict_round_trip(tmp_path):
    a = agent(tmp_path)
    for _ in range(10):
        step(a)
    a.update()
    b = agent(tmp_path / "b")
    b.load_state_dict(a.state_dict())
    hm, pr = obs()
    with a.evaluating(), b.evaluating():
        assert torch.allclose(a.act(hm, pr), b.act(hm, pr))
    assert b.updates == a.updates and b.frames == a.frames
