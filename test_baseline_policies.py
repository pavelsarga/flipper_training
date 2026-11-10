#!/usr/bin/env python3
"""Validate the trainable-baseline PolicyConfigs construct + forward + pair with
their TorchRL loss, using a lightweight mock Env (no physics engine needed).

This is the "easily adjust to our training" check: each baseline uses the same
PolicyConfig.create(env) contract as the repo's PPO policies and yields
TorchRL-native operators that the corresponding TorchRL loss accepts.
"""
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Unbounded

from flipper_training.observations import ObservationEncoder

N, N_DRIVE, OBS_IN, ENC = 6, 4, 16, 32
ADIM = 2 * N_DRIVE
OBS_KEY = "proprio"


class _Enc(ObservationEncoder):
    def __init__(self):
        super().__init__(ENC)
        self.net = torch.nn.Linear(OBS_IN, ENC)

    def forward(self, x):
        return self.net(x)


class _Obs:
    name = OBS_KEY
    def get_encoder(self):
        return _Enc()
    def get_spec(self):
        return Unbounded(shape=(N, OBS_IN))


class MockEnv:
    def __init__(self):
        low = -torch.ones(N, ADIM)
        self.action_spec = Bounded(low=low, high=-low, shape=(N, ADIM),
                                   dtype=torch.float32)
        self.observations = [_Obs()]
        self.device = torch.device("cpu")


def sample_td():
    return TensorDict({OBS_KEY: torch.randn(N, OBS_IN)}, batch_size=[N])


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    assert ok, f"{name} failed: {detail}"


MLP = dict(hidden_dim=64, num_hidden=2, layernorm=True)
OPT = dict(lr=3e-4)


def test_offpolicy_sac():
    from flipper_training.policies.offpolicy_ac_policy import OffPolicyACConfig
    cfg = OffPolicyACConfig(actor_mlp_opts=MLP, qvalue_mlp_opts=MLP,
                            actor_optimizer_opts=OPT, qvalue_optimizer_opts=OPT,
                            deterministic_actor=False, twin_q=True)
    w, groups, tr = cfg.create(MockEnv())
    td = w.get_policy_operator()(sample_td())
    check("SAC actor forward -> action", "action" in td.keys(),
          f"action shape {tuple(td['action'].shape)}")
    check("SAC twin-Q present", len(w.get_qvalue_operator()) == 2)
    # TorchRL SACLoss takes ONE qvalue net and vmaps it into the twin internally
    # (num_qvalue_nets=2); our wrapper also exposes the explicit pair if needed.
    from torchrl.objectives import SACLoss
    SACLoss(actor_network=w.get_policy_operator(),
            qvalue_network=w.get_qvalue_operator()[0], num_qvalue_nets=2)
    check("SACLoss accepts actor+Q (twin via vmap)", True)


def test_offpolicy_td3():
    from flipper_training.policies.offpolicy_ac_policy import OffPolicyACConfig
    cfg = OffPolicyACConfig(actor_mlp_opts=MLP, qvalue_mlp_opts=MLP,
                            actor_optimizer_opts=OPT, qvalue_optimizer_opts=OPT,
                            deterministic_actor=True, twin_q=True)
    w, groups, tr = cfg.create(MockEnv())
    td = w.get_policy_operator()(sample_td())
    check("TD3 deterministic actor -> action", "action" in td.keys())
    check("TD3 exploration transform returned", len(tr) == 1)


def test_d3qn():
    from flipper_training.policies.d3qn_policy import D3QNPolicyConfig
    cfg = D3QNPolicyConfig(mlp_opts=MLP, optimizer_opts=OPT, n_bins=3)
    w, groups, tr = cfg.create(MockEnv())
    td = w.get_policy_operator()(sample_td())
    check("D3QN forward -> action_value", "action_value" in td.keys(),
          f"{w.discrete_to_continuous.shape[0]} discrete configs")
    check("D3QN action one-hot", "action" in td.keys())
    check("discrete_to_continuous maps to full action_dim",
          w.discrete_to_continuous.shape[1] == ADIM)
    from torchrl.objectives import DQNLoss
    DQNLoss(value_network=w.get_qvalue_network(), action_space="one_hot")
    check("DQNLoss accepts dueling Q-actor", True)


def test_state_machine():
    from flipper_training.policies.state_machine_policy import StateMachinePolicyConfig
    cfg = StateMachinePolicyConfig(n_states=4, temperature=1.0, state_mlp_opts=MLP,
                                   gate_mlp_opts=MLP, value_mlp_opts=MLP,
                                   actor_optimizer_opts=OPT, value_optimizer_opts=OPT)
    w, groups, tr = cfg.create(MockEnv())
    td = w.get_policy_operator()(sample_td())
    check("Azayev SM actor -> action", "action" in td.keys())
    vd = w.get_value_operator()(sample_td())
    check("Azayev SM critic -> state_value", "state_value" in vd.keys())
    from torchrl.objectives import ClipPPOLoss
    ClipPPOLoss(actor_network=w.get_policy_operator(), critic_network=w.get_value_operator())
    check("ClipPPOLoss accepts SM actor+critic", True)


def test_icm():
    from flipper_training.policies import EncoderCombiner
    from flipper_training.policies.icm import ICM
    enc = EncoderCombiner({OBS_KEY: _Enc()})
    icm = ICM(enc, action_dim=ADIM)
    obs = {OBS_KEY: torch.randn(N, OBS_IN)}
    nxt = {OBS_KEY: torch.randn(N, OBS_IN)}
    act = torch.randn(N, ADIM)
    r = icm.intrinsic_reward(obs, act, nxt)
    loss = icm.loss(obs, act, nxt)
    check("ICM intrinsic_reward shape", tuple(r.shape) == (N, 1))
    check("ICM loss is scalar", loss.dim() == 0)


if __name__ == "__main__":
    for fn in [test_offpolicy_sac, test_offpolicy_td3, test_d3qn,
               test_state_machine, test_icm]:
        print(f"== {fn.__name__} ==")
        fn()
    print("\nALL BASELINE-POLICY TESTS PASSED")
