"""Unit checks for the prefix-masked chunk PPO loss and the random execution horizon (no Isaac).

  python -m pytest src/flipper_training/marv_rl_training/training/test_chunk_ppo_loss.py
"""

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from marv_rl_training.environment.chunked_env import execution_horizon_probs
from marv_rl_training.policies.chunk_ppo_loss import PrefixMaskedClipPPOLoss, chunk_step_mask, per_dim_tanh_normal

T_P, A, OBS = 4, 3, 8
D = T_P * A


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(OBS, 2 * D)

    def forward(self, obs):
        out = self.lin(obs)
        return out[..., :D], torch.nn.functional.softplus(out[..., D:]) + 0.05


def _actor_critic(seed=0):
    torch.manual_seed(seed)
    actor = ProbabilisticActor(
        module=TensorDictModule(_Head(), in_keys=["obs"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"], distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0}, return_log_prob=True,
    )
    critic = ValueOperator(torch.nn.Linear(OBS, 1), in_keys=["obs"])
    return actor, critic


def _rollout(actor, critic, N=64, seed=1):
    torch.manual_seed(seed)
    td = TensorDict({"obs": torch.randn(N, OBS)}, batch_size=[N])
    with torch.no_grad():
        actor(td)                                  # loc, scale, action, sample_log_prob
        critic(td)
    # perturb so old != new
    for p in actor.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    td["advantage"] = torch.randn(N, 1)
    td["value_target"] = torch.randn(N, 1)
    return td


def test_masked_loss_equals_clip_ppo_on_full_chunk():
    actor, critic = _actor_critic()
    td = _rollout(actor, critic)
    td["exec_len"] = torch.full((td.shape[0], 1), T_P, dtype=torch.int64)
    ref = ClipPPOLoss(actor, critic, entropy_bonus=False, normalize_advantage=True, loss_critic_type="smooth_l1")
    ours = PrefixMaskedClipPPOLoss(actor, critic, prediction_horizon=T_P, action_dim=A,
                                   entropy_bonus=False, normalize_advantage=True, loss_critic_type="smooth_l1")
    r, o = ref(td.clone()), ours(td.clone())
    for k in ("loss_objective", "loss_critic", "kl_approx", "clip_fraction"):
        assert torch.allclose(r[k], o[k], atol=1e-5, rtol=1e-4), (k, r[k], o[k])


def test_prefix_mask_drops_tail_from_ratio():
    actor, critic = _actor_critic()
    td = _rollout(actor, critic)
    td["exec_len"] = torch.full((td.shape[0], 1), 1, dtype=torch.int64)
    ours = PrefixMaskedClipPPOLoss(actor, critic, prediction_horizon=T_P, action_dim=A,
                                   entropy_bonus=False, normalize_advantage=False)
    o = ours(td.clone())
    # the ratio computed by hand over the first A dims
    dist = actor.get_dist(td.clone())
    cur = per_dim_tanh_normal(dist, dist.loc, dist.scale).log_prob(td["action"])[:, :A].sum(-1)
    old = per_dim_tanh_normal(dist, td["loc"], td["scale"]).log_prob(td["action"])[:, :A].sum(-1)
    lw = (cur - old).detach()
    adv = td["advantage"].squeeze(-1)
    gain = torch.minimum(lw.exp() * adv, lw.clamp(*ours._clip_bounds).exp() * adv)
    assert torch.allclose(o["loss_objective"], -gain.mean(), atol=1e-5)
    m = chunk_step_mask(td["exec_len"], T_P, A)
    assert m[:, :A].all() and not m[:, A:].any()


def test_variable_gamma_gae_matches_torchrl_for_constant_horizon():
    from marv_rl_training.training.train_diffusion import FtrDiffusionTrainer

    actor, critic = _actor_critic()
    N, T, tau, gamma_ctrl, lmbda = 5, 7, 3, 0.99, 0.9
    torch.manual_seed(2)
    td = TensorDict({
        "obs": torch.randn(N, T, OBS),
        "next": TensorDict({
            "obs": torch.randn(N, T, OBS),
            "reward": torch.randn(N, T, 1),
            "done": torch.rand(N, T, 1) < 0.15,
            "exec_len": torch.full((N, T, 1), tau, dtype=torch.int64),
        }, batch_size=[N, T]),
    }, batch_size=[N, T])
    td["next", "terminated"] = td["next", "done"].clone()

    class _Cfg:
        gae_opts = {"gamma": gamma_ctrl ** tau, "lmbda": lmbda}
        control_gamma = gamma_ctrl

    fake = FtrDiffusionTrainer.__new__(FtrDiffusionTrainer)
    fake.config = _Cfg()
    fake.value_operator = critic
    ours = td.clone()
    FtrDiffusionTrainer._variable_gamma_gae(fake, ours)
    ref = td.clone()
    GAE(gamma=gamma_ctrl ** tau, lmbda=lmbda, value_network=critic, time_dim=1)(ref)
    assert torch.allclose(ours["advantage"], ref["advantage"], atol=1e-5), (ours["advantage"] - ref["advantage"]).abs().max()
    assert torch.allclose(ours["value_target"], ref["value_target"], atol=1e-5)


def test_tail_targets_align_with_next_chunk():
    from marv_rl_training.training.train_diffusion import FtrDiffusionTrainer

    actor, critic = _actor_critic()
    N, T = 3, 4
    td = TensorDict({"obs": torch.randn(N, T, OBS)}, batch_size=[N, T])
    with torch.no_grad():
        actor(td)
    exec_len = torch.tensor([1, 3, 4, 2]).view(1, T, 1).expand(N, T, 1).clone()
    done = torch.zeros(N, T, 1, dtype=torch.bool)
    done[1, 1] = True   # env 1's episode ends at t=1 -> no target there
    td["next", "exec_len"], td["next", "done"] = exec_len, done

    class _Cfg:
        prediction_horizon = T_P

    class _Env:
        action_dim = A

    fake = FtrDiffusionTrainer.__new__(FtrDiffusionTrainer)
    fake.config, fake.ftr_torchrl_env, fake.actor_operator = _Cfg(), _Env(), actor
    FtrDiffusionTrainer._add_tail_targets(fake, td)
    dist = actor.get_dist(td[:1, :1])
    mode = per_dim_tanh_normal(dist, td["loc"], td["scale"]).deterministic_sample.reshape(N, T, T_P, A)
    tgt, msk = td["tail_target"].reshape(N, T, T_P, A), td["tail_mask"].reshape(N, T, T_P, A)
    # t=0, tau=1: steps 1..3 of chunk 0 <- steps 0..2 of chunk 1
    assert msk[0, 0, 0].sum() == 0 and msk[0, 0, 1:].all()
    assert torch.allclose(tgt[0, 0, 1:], mode[0, 1, :3])
    # t=1, tau=3: step 3 <- step 0 of chunk 2 ; env 1 masked (done)
    assert torch.allclose(tgt[0, 1, 3], mode[0, 2, 0]) and msk[0, 1, 3].all() and not msk[0, 1, :3].any()
    assert not msk[1, 1].any()
    # t=2, tau=4: nothing to regress; last step: nothing
    assert not msk[:, 2].any() and not msk[:, 3].any()


def test_execution_horizon_probs():
    p = execution_horizon_probs(8, 2, 6.0)
    assert p.shape == (8,) and abs(float(p.sum()) - 1) < 1e-6
    assert p.argmax().item() == 1 and p[7] / p[1] > 0.5
    assert torch.equal(execution_horizon_probs(8, 2, 0.0).nonzero().flatten(), torch.tensor([1]))


def test_sampled_execution_horizon_is_one_based():
    from marv_rl_training.environment.chunked_env import ActionChunkEnv

    env = ActionChunkEnv.__new__(ActionChunkEnv)
    env.prediction_horizon, env.execution_horizon = 8, 2
    env._exec_probs = execution_horizon_probs(8, 2, 6.0)
    torch.nn.Module.__init__(env)
    env.train()
    draws = {env.sample_execution_horizon() for _ in range(2000)}
    assert min(draws) >= 1 and max(draws) == 8, draws
    env.eval()
    assert env.sample_execution_horizon() == 2
