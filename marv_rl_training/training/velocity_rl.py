"""RL fine-tuning of the velocity-IL controller, trained alongside the flipper policy (co-training).

The forward-speed network (policies/velocity_controller.py, imitation-learned from Gazebo
operator rides by train_velocity_il.py) keeps driving inside FtrEnv -- `velocity_controller:
{type: rl}` -- while train_diffusion.py trains the flipper policy. This module is the speed
agent's learner: it samples the speed, records its own transitions, computes its own reward and
runs a low-LR PPO update after every flipper batch. The flipper policy is part of its
environment and improves meanwhile, so the speed net also meets early, badly acting flippers.

Policy: the IL network's mean, plus a learnable exploration spread (init `init_std`; the IL
head's own spread is the operator's variance, far too noisy to explore with). The applied speed
is the sample clamped to [0, v_max]; log-probabilities are of the unclamped sample.
Anchor: `anchor_coef * (mean - mean_IL)^2` against a frozen copy of the IL net, plus a low actor
LR -- the fine-tune polishes the imitation instead of drifting away from it. Critic: the IL
net's encoder with a fresh value head, trained alone for `critic_warmup` updates before the
actor moves (an untrained critic's advantages would otherwise undo the imitation first).

Reward, per control step. There is NO reward for speed (any speed reward, even a mean over a
crossing, drifts to full throttle) and no success bonus (discounted, it pays for finishing
sooner). The speed is anchored to what the IL net does on the same states -- the frozen copy
runs alongside every step -- and the only reason to leave it is stability:
    anchor     - mean_coef * (m_pi - m_IL)^2       rolling (window_s) mean speed vs the IL net's
    range      + std_coef * min(s_pi, s_IL)        rolling speed std, rewarded up to the IL net's
                                                   (keeps the range; nothing for going beyond it)
    jerk       - jerk_coef * (v_t - 2 v_{t-1} + v_{t-2})^2      low weight: cannot block speed changes
    attitude   - rate_coef * (roll_rate^2 + pitch_rate^2)       a slow-down through an impact pays,
                                                                a global slow-down does not
    stall      - stall_coef per step once v < stall_v has lasted more than stall_after_s
    failure    - fail_penalty on tip-over / out of range / shock, and on timeout
The rolling statistics are exponential averages (time constant window_s) restarted at every
episode start; a short slow-down at one obstacle barely moves them, a sustained offset does.
They are hidden state of the reward, so the critic (not the actor) also gets
[m_pi, m_IL, s_pi, s_IL, stall timer] -- an asymmetric critic.

Recording: only the first `record_envs` envs store transitions (a 5-frame heightmap history per
step is ~9 kB; 524k steps per flipper batch would not fit). Every env drives with the policy.

The env calls, per control step: act() from _pre_physics_step (through the controller), then
end_step() once the dones are known and before the resets, then reset() for the reset envs.
"""
import contextlib
import math
import os
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf
from torch import nn

from marv_rl_training.policies.velocity_controller import VelocityControllerExport, VelocityNet


@dataclass
class VelocityRewardConfig:
    window_s: float = 10.0              # time constant of the rolling mean / std
    mean_coef: float = 2.0              # x (m_pi - m_IL)^2 [(m/s)^2]
    std_coef: float = 0.05              # x min(s_pi, s_IL) [m/s]
    jerk_coef: float = 0.2              # x (second difference per step)^2
    rate_coef: float = 0.02             # x (roll rate^2 + pitch rate^2) [(rad/s)^2]
    fail_penalty: float = 10.0          # tip-over, out of range, shock -- and timeout
    stall_coef: float = 0.02            # per step, once stalled
    stall_v: float = 0.075              # m/s
    stall_after_s: float = 2.0


@dataclass
class VelocityRLConfig:
    il_dir: str = ""                    # experiments/velocity_il/<name> (config.yaml, net_best.pth, export)
    v_max: float = 1.0
    record_envs: int = 256
    init_std: float = 0.03              # m/s, exploration spread
    min_std: float = 0.005
    max_std: float = 0.1
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    gamma: float = 0.995                # per 20 Hz control step (~10 s horizon)
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    anchor_coef: float = 1.0            # x (mean - mean_IL)^2 [(m/s)^2]
    entropy_coef: float = 0.0
    critic_warmup: int = 5              # updates with the actor frozen
    start_after_frames: int = 0         # control steps of the flipper run before the speed agent learns
    max_grad_norm: float = 1.0
    reward: VelocityRewardConfig = field(default_factory=VelocityRewardConfig)


EXTRA_DIM = 5   # critic-only inputs: m_pi, m_IL, s_pi, s_IL, stall timer


class ValueNet(VelocityNet):
    """VelocityNet's encoder (the IL net's encoder weights are copied in) with a scalar value head
    that also sees `extra_dim` critic-only inputs -- the reward's hidden rolling state."""

    def __init__(self, extra_dim: int = EXTRA_DIM, **kwargs):
        super().__init__(**kwargs)
        hidden = [m for m in self.head if isinstance(m, nn.Linear)][:-1]
        d_in = [m for m in self.head if isinstance(m, nn.Linear)][0].in_features + extra_dim
        layers = []
        for lin in hidden:
            layers += [nn.Linear(d_in, lin.out_features), nn.SiLU()]
            d_in = lin.out_features
        self.head = nn.Sequential(*layers, nn.Linear(d_in, 1))

    def forward(self, hm, proprio, extra):  # noqa: D102
        z = torch.cat([self.cnn(hm), self.proprio((proprio * self.proprio_keep).flatten(1)), extra], dim=-1)
        return self.head(z)[:, 0]


def load_il(il_dir: str, v_max: float, device):
    """(VelocityNet with the best-val weights, hm_scale, proprio_mean, proprio_std, net kwargs)."""
    cfg = OmegaConf.load(os.path.join(il_dir, "config.yaml"))
    kwargs = dict(history=int(cfg.history), channels=tuple(cfg.channels), proprio_hidden=int(cfg.proprio_hidden),
                  head_hidden=int(cfg.head_hidden), head_layers=int(cfg.head_layers), v_max=float(v_max),
                  drop_proprio=tuple(cfg.get("drop_proprio", [])))
    net = VelocityNet(**kwargs)
    weights = os.path.join(il_dir, "net_best.pth")
    if not os.path.exists(weights):
        weights = os.path.join(il_dir, "net_final.pth")
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    export = torch.jit.load(os.path.join(il_dir, "velocity_controller.pt"), map_location="cpu")
    return (net.to(device), float(export.hm_scale), export.proprio_mean.clone().to(device),
            export.proprio_std.clone().to(device), kwargs)


class VelocityAgent:
    def __init__(self, cfg: VelocityRLConfig, num_envs: int, step_dt: float, device):
        self.cfg = cfg
        self.rw = cfg.reward
        self.num_envs = num_envs
        self.step_dt = float(step_dt)
        self.device = torch.device(device)
        self.K = min(int(cfg.record_envs), num_envs)

        self.actor, self.hm_scale, self.pr_mean, self.pr_std, self.net_kwargs = load_il(cfg.il_dir, cfg.v_max, self.device)
        self.ref, *_ = load_il(cfg.il_dir, cfg.v_max, self.device)
        self.ref.requires_grad_(False).eval()
        self.critic = ValueNet(**self.net_kwargs).to(self.device)
        enc = {k: v for k, v in self.actor.state_dict().items() if not k.startswith("head.")}
        self.critic.load_state_dict(enc, strict=False)
        self.log_std = nn.Parameter(torch.tensor(math.log(cfg.init_std), device=self.device))
        self.actor_opt = torch.optim.Adam([*self.actor.parameters(), self.log_std], lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.deterministic = False
        self.recording = True
        self.updates = 0
        self.frames = 0                 # control steps seen (all envs count once per step)
        self._buf = []                  # per step: dict of [K] / [K, ...] tensors
        self._pending = None            # this step's recorded obs/action, closed by end_step
        n = num_envs
        self.v = torch.zeros(n, device=self.device)          # applied speed this step
        self.v1 = torch.zeros(n, device=self.device)         # previous step
        self.v2 = torch.zeros(n, device=self.device)         # two steps back
        self.age = torch.zeros(n, device=self.device)        # steps since this env's episode start
        self.stall_t = torch.zeros(n, device=self.device)    # s below stall_v
        self.start_xy = torch.zeros(n, 2, device=self.device)
        self.v_ref = torch.zeros(n, device=self.device)      # the IL net's speed on this step's state
        # rolling (EMA, window_s) first and second moments of the policy's and the IL net's speed
        self.ema = {k: torch.zeros(n, device=self.device) for k in ("v", "v2", "r", "r2")}
        self._stats = {}

    # ------------------------------------------------------------------ acting
    def _norm(self, hm, proprio):
        return hm / self.hm_scale, (proprio - self.pr_mean) / self.pr_std

    @torch.no_grad()
    def act(self, hm_hist: torch.Tensor, proprio_hist: torch.Tensor) -> torch.Tensor:
        """[N] applied speed for this control step; records the first K envs' transitions."""
        hm_hist, proprio_hist = hm_hist.float(), proprio_hist.float()
        mean, _ = self.actor(*self._norm(hm_hist, proprio_hist))
        if self.deterministic:
            raw = mean
        else:
            std = self.log_std.exp().clamp(self.cfg.min_std, self.cfg.max_std)
            raw = mean + std * torch.randn_like(mean)
        v = raw.clamp(0.0, self.cfg.v_max)
        self.v2, self.v1, self.v = self.v1, self.v, v
        self.v_ref = self.ref(*self._norm(hm_hist, proprio_hist))[0]
        if self.recording and not self.deterministic:
            k = self.K
            extra = self._critic_extra()[:k]
            self._pending = dict(hm=hm_hist[:k].half(), proprio=proprio_hist[:k].clone(), raw=raw[:k].clone(),
                                 logp=self._logp(mean[:k], raw[:k]), extra=extra,
                                 value=self.critic(*self._norm(hm_hist[:k], proprio_hist[:k]), extra))
        return v

    def _rolling(self):
        """(m_pi, m_IL, s_pi, s_IL) [N] from the EMA moments."""
        e = self.ema
        return (e["v"], e["r"], (e["v2"] - e["v"] ** 2).clamp_min(0.0).sqrt(),
                (e["r2"] - e["r"] ** 2).clamp_min(0.0).sqrt())

    def _critic_extra(self):
        """[N, EXTRA_DIM] the reward's hidden state at decision time (before this step's update);
        a fresh episode reads as zero deviation and zero spread, as its first reward will."""
        m, m_il, s, s_il = self._rolling()
        fresh = self.age == 0
        m = torch.where(fresh, self.v, m)
        m_il = torch.where(fresh, self.v_ref, m_il)
        s = torch.where(fresh, torch.zeros_like(s), s)
        s_il = torch.where(fresh, torch.zeros_like(s_il), s_il)
        return torch.stack([m, m_il, 4.0 * s, 4.0 * s_il, self.stall_t / self.rw.stall_after_s], dim=-1)

    def _logp(self, mean, raw):
        std = self.log_std.exp().clamp(self.cfg.min_std, self.cfg.max_std)
        return torch.distributions.Normal(mean, std).log_prob(raw)

    # ------------------------------------------------------------------ reward
    @torch.no_grad()
    def end_step(self, positions_xy, ang_vel_body, success, fail, timeout, abort=None):
        """Reward and done for this step, once the env's dones are known (before its resets).
        positions_xy [N, 2]; ang_vel_body [N, 3] (roll, pitch, yaw rate in the base frame);
        success / fail / timeout [N] bool; abort [N] bool: ended for a reason that is not the
        policy's (a physics explosion) -- done, without a penalty. Returns the [N] reward."""
        rw = self.rw
        first = self.age == 0
        self.start_xy[first] = positions_xy[first]
        # rolling moments, restarted at an episode's first step
        a = min(self.step_dt / rw.window_s, 1.0)
        for key, x in (("v", self.v), ("v2", self.v ** 2), ("r", self.v_ref), ("r2", self.v_ref ** 2)):
            self.ema[key] = torch.where(first, x, (1.0 - a) * self.ema[key] + a * x)
        m, m_il, s, s_il = self._rolling()
        # no second difference across an episode start (from the 3rd step of an episode on)
        jerk = torch.where(self.age <= 1, torch.zeros_like(self.v), self.v - 2.0 * self.v1 + self.v2)
        self.age += 1
        self.stall_t = torch.where(self.v < rw.stall_v, self.stall_t + self.step_dt, torch.zeros_like(self.stall_t))
        stalled = self.stall_t > rw.stall_after_s

        elapsed = self.age * self.step_dt
        mean_speed = (positions_xy - self.start_xy).norm(dim=-1) / elapsed.clamp_min(self.step_dt)   # diagnostic only
        terms = {
            "anchor": -rw.mean_coef * (m - m_il) ** 2,
            "range": rw.std_coef * torch.minimum(s, s_il),
            "jerk": -rw.jerk_coef * jerk ** 2,
            "attitude": -rw.rate_coef * (ang_vel_body[:, 0] ** 2 + ang_vel_body[:, 1] ** 2),
            "stall": -rw.stall_coef * stalled.float(),
            "fail": -rw.fail_penalty * (fail | timeout).float(),
        }
        reward = sum(terms.values())
        done = success | fail | timeout
        if abort is not None:
            done = done | abort
        if not self.deterministic:
            self.frames += self.num_envs
            self._accumulate(terms, reward, done, success, fail, timeout, mean_speed, stalled, (m, m_il, s, s_il))
        if self._pending is not None:
            k = self.K
            self._buf.append(dict(**self._pending, reward=reward[:k].clone(), done=done[:k].clone()))
            self._pending = None
        return reward

    def reset(self, env_ids):
        """The env reset these envs: their next step starts a new episode."""
        self.age[env_ids] = 0
        self.stall_t[env_ids] = 0.0

    def _accumulate(self, terms, reward, done, success, fail, timeout, mean_speed, stalled, rolling):
        s = self._stats
        for name, x in zip(("m_pi", "m_il", "s_pi", "s_il"), rolling):
            s[name] = s.get(name, 0.0) + float(x.sum())
        s["v_ref_sum"] = s.get("v_ref_sum", 0.0) + float(self.v_ref.sum())
        for k, v in terms.items():
            s[f"rew/{k}"] = s.get(f"rew/{k}", 0.0) + float(v.sum())
        s["rew/total"] = s.get("rew/total", 0.0) + float(reward.sum())
        s["n_steps"] = s.get("n_steps", 0) + int(reward.numel())
        s["v_sum"] = s.get("v_sum", 0.0) + float(self.v.sum())
        s["stalled"] = s.get("stalled", 0) + int(stalled.sum())
        s["episodes"] = s.get("episodes", 0) + int(done.sum())
        s["success"] = s.get("success", 0) + int(success.sum())
        s["fail"] = s.get("fail", 0) + int(fail.sum())
        s["timeout"] = s.get("timeout", 0) + int(timeout.sum())
        s["success_speed_sum"] = s.get("success_speed_sum", 0.0) + float(mean_speed[success].sum())

    def pop_stats(self) -> dict:
        """Per-batch means for logging, then reset."""
        s, self._stats = self._stats, {}
        n = max(s.get("n_steps", 0), 1)
        ep = max(s.get("episodes", 0), 1)
        # one "/" per key: RunLogger writes <prefix>.csv per first path element
        out = {f"vel_rl/rew_{k[4:]}": v / n for k, v in s.items() if k.startswith("rew/")}
        out.update({
            "vel_rl/v_mean": s.get("v_sum", 0.0) / n,
            "vel_rl/v_il_mean": s.get("v_ref_sum", 0.0) / n,
            **{f"vel_rl/{k}": s.get(k, 0.0) / n for k in ("m_pi", "m_il", "s_pi", "s_il")},
            "vel_rl/stalled_frac": s.get("stalled", 0) / n,
            "vel_rl/success_rate": s.get("success", 0) / ep,
            "vel_rl/fail_rate": s.get("fail", 0) / ep,
            "vel_rl/timeout_rate": s.get("timeout", 0) / ep,
            "vel_rl/success_mean_speed": s.get("success_speed_sum", 0.0) / max(s.get("success", 0), 1),
            "vel_rl/std": float(self.log_std.exp().clamp(self.cfg.min_std, self.cfg.max_std)),
        })
        return out

    @contextlib.contextmanager
    def evaluating(self):
        """Mean speed, nothing recorded or accumulated (eval rollouts)."""
        prev = self.deterministic, self.recording
        self.deterministic, self.recording = True, False
        try:
            yield
        finally:
            self.deterministic, self.recording = prev
            self._pending = None

    # ------------------------------------------------------------------ learning
    def update(self) -> dict:
        """GAE + PPO over the transitions recorded since the last update; clears them."""
        buf, self._buf = self._buf, []
        if len(buf) < 3:
            return {}
        learn_actor = self.updates >= self.cfg.critic_warmup and self.frames >= self.cfg.start_after_frames
        self.updates += 1
        # stack [S, K, ...]; the inference-mode tensors of the rollout are cloned into normal ones
        st = {k: torch.stack([b[k] for b in buf]).clone() for k in buf[0]}
        reward, done, value = st["reward"], st["done"].float(), st["value"]
        S = len(buf) - 1                # the last step only bootstraps the one before it
        adv = torch.zeros(S, self.K, device=self.device)
        gae = torch.zeros(self.K, device=self.device)
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        for t in reversed(range(S)):
            nonterminal = 1.0 - done[t]
            delta = reward[t] + g * value[t + 1] * nonterminal - value[t]
            gae = delta + g * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + value[:S]
        flat = lambda x: x[:S].reshape(S * self.K, *x.shape[2:])  # noqa: E731
        # the heightmaps stay half precision ([S*K, T, 45, 21] in float32 is GBs); float per minibatch
        hm, pr, raw, logp_old = flat(st["hm"]), flat(st["proprio"]), flat(st["raw"]), flat(st["logp"])
        extra = flat(st["extra"])
        adv, ret = adv.reshape(-1), ret.reshape(-1)
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        n = hm.shape[0]

        with torch.no_grad():
            mean_ref = torch.cat([self.ref(*self._norm(hm[i:i + 8192].float(), pr[i:i + 8192]))[0]
                                  for i in range(0, n, 8192)])

        log = dict(pg=0.0, anchor=0.0, vf=0.0, clipfrac=0.0, n=0)
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.cfg.minibatch):
                idx = perm[i:i + self.cfg.minibatch]
                h, p = self._norm(hm[idx].float(), pr[idx])
                vf = (self.critic(h, p, extra[idx]) - ret[idx]).pow(2).mean()
                self.critic_opt.zero_grad()
                vf.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()
                log["vf"] += float(vf)
                if learn_actor:
                    mean, _ = self.actor(h, p)
                    logp = self._logp(mean, raw[idx])
                    ratio = (logp - logp_old[idx]).exp()
                    a = adv[idx]
                    pg = -torch.min(ratio * a, ratio.clamp(1 - self.cfg.clip, 1 + self.cfg.clip) * a).mean()
                    anchor = (mean - mean_ref[idx]).pow(2).mean()
                    loss = pg + self.cfg.anchor_coef * anchor - self.cfg.entropy_coef * self.log_std
                    self.actor_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_([*self.actor.parameters(), self.log_std], self.cfg.max_grad_norm)
                    self.actor_opt.step()
                    log["pg"] += float(pg)
                    log["anchor"] += float(anchor)
                    log["clipfrac"] += float(((ratio - 1).abs() > self.cfg.clip).float().mean())
                log["n"] += 1
        m = max(log.pop("n"), 1)
        out = {f"vel_rl/loss_{k}": v / m for k, v in log.items()}
        out.update({"vel_rl/learning_actor": float(learn_actor), "vel_rl/updates": float(self.updates),
                    "vel_rl/adv_samples": float(n), "vel_rl/return_mean": float(ret.mean())})
        return out

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), log_std=self.log_std.detach().clone(),
                    actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    updates=self.updates, frames=self.frames)

    def load_state_dict(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        with torch.no_grad():
            self.log_std.copy_(sd["log_std"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.updates, self.frames = int(sd["updates"]), int(sd["frames"])

    def export(self, path: str):
        """TorchScript VelocityControllerExport of the current actor (mean speed): what FtrEnv's
        velocity_controller {type: il} and the Gazebo velocity node load."""
        net = VelocityNet(**self.net_kwargs)
        net.load_state_dict({k: v.detach().cpu() for k, v in self.actor.state_dict().items()})
        exp = VelocityControllerExport(net.eval(), self.hm_scale, self.pr_mean.cpu(), self.pr_std.cpu())
        torch.jit.script(exp).save(path)


def make_agent(cfg_dict: dict, num_envs: int, step_dt: float, device) -> VelocityAgent:
    """VelocityAgent from the config's `velocity_rl:` block (a plain dict)."""
    d = dict(cfg_dict)
    reward = VelocityRewardConfig(**(d.pop("reward", {}) or {}))
    return VelocityAgent(VelocityRLConfig(**d, reward=reward), num_envs, step_dt, device)
