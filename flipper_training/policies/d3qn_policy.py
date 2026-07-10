"""AT-D3QN: dueling double-DQN flipper controller (Pan et al. 2023).

Pan, H. et al. "Deep Reinforcement Learning for Flipper Control of Tracked
Robots," arXiv:2306.10352 (AT-D3QN) and "... in Urban Rescuing Environments,"
Remote Sensing 15(18):4616 (ICM-D3QN = this + the curiosity module in
``icm.py``).

Discrete-action baseline expressed with the SAME libraries as the rest of the
repo (TorchRL ``QValueActor`` + the shared ``MLP``/``EncoderCombiner``), so it
trains with ``torchrl.objectives.DQNLoss(double_dqn=True)`` in your loop (see
``experiments/dqn/train.py``, which now passes the flag -- earlier revisions
of this trainer silently ran vanilla DQN + target net).

Architecture, two selectable topologies:

* ``fig5_topology=False`` (**generic**, default): a shared observation encoder
  (``EncoderCombiner``, over WHATEVER observations the env has -- not
  paper-specific), exposed post-hoc via ``wrapper.get_encoder()`` so
  ICM-D3QN's curiosity module (``icm.py``) can reuse the SAME instance
  instead of building its own -- a deliberate implementation simplification,
  NOT paper-mandated (the ICM-D3QN paper's Fig. 7 actually draws TWO separate
  encoders: the D3QN's fused feature-extraction module and the ICM's own
  raw-state encoder; we share one instance so curiosity shapes the policy
  representation); see ``experiments/dqn/train.py``, which passes this into
  ``icm.ICM(encoder=..., ...)`` when ``use_icm=True`` and ``separate_encoder=False``,
  and dedupes it out of ICM's own optimizer param group (the shared params
  are already covered by the Q-network's group; autograd sums their gradient
  from both losses, so only the *optimizer step* must not double-apply).
  Layer sizes come from ``mlp_opts``, NOT the paper's Fig. 5 widths.
* ``fig5_topology=True`` (**paper-literal**, ``_Fig5DuelingQ``): the LITERAL
  AT-D3QN Fig. 5 network (fixed layer widths: terrain branch 15->32->4,
  fusion (4+3)->16, dueling heads 16->1 / 16->9, all LeakyReLU) over the raw
  ``PanTerrainState`` observation (``observations/pan_terrain.py``), required
  present (validated, clear error otherwise). Has NO ``EncoderCombiner`` to
  share -- pair it with ``icm.ICM(separate_encoder=True, ...)``, which builds
  the paper's OWN Fig. 7 raw-state encoder instead of reusing this one; see
  ``_Fig5DuelingQ``'s own docstring for the honest note on Fig. 5/7's
  E-dimension inconsistency (3-dim per Eq. 2 vs. 4-dim as literally drawn)
  and how it's resolved.

Both topologies share:
* **dueling head**: a value stream V(s) and an advantage stream A(s,a),
  combined as ``Q = V + (A - mean_a A)``,
* a discrete action set mapped to a continuous env action (constant forward
  track velocity + flipper-joint command) via a deploy bridge exposed as
  ``wrapper.get_policy_operator()`` (continuous ``"action"``, matching
  ``env.action_spec`` -- the deploy contract the generic node relies on) and,
  for the hand-rolled training loop, ``wrapper.action_from_onehot(onehot, td)``.

Two action-space modes, selected by ``incremental`` (default ``False``):

* ``incremental=False`` (**absolute**, default, NOT the paper's action space):
  each of the ``n_flip`` flippers is binned into ``n_bins`` levels spanning the
  joint limits, giving ``n_bins**n_flip`` discrete configs; the chosen index
  maps to a FIXED continuous action via the ``[M, action_dim]`` table exposed
  as ``wrapper.discrete_to_continuous``. This is simple and index-static, but
  it is a deliberate divergence from the paper (see below) -- kept as the
  default only for backward compatibility with anything already built against
  it; ``n_bins`` defaults low (3) to keep the action count small regardless.
* ``incremental=True`` (**paper-faithful**, Sec. III-C Eq. 3): the action set
  is the paper's 9 paired increments ``A = {a_i,j} = {i*delta_f, j*delta_f}``,
  ``i, j in {-1, 0, 1}`` (front pair clockwise/hold/counter-clockwise x rear
  pair clockwise/hold/counter-clockwise), ``delta_f = pi/12``. Unlike the
  absolute table, the resulting continuous action is NOT a fixed per-index
  lookup: the deploy bridge reads the CURRENT front/rear flipper angles back
  out of the ``angle_obs_key`` observation (default ``"LocalStateVector"``,
  offsets ``flipper_angle_idx`` -- see ``state_machine_policy.py`` for the
  identical convention), adds the selected ``(front_delta, rear_delta)`` pair,
  clamps to the joint limits, and writes the result into the flipper-joint
  action slots. Because VecNorm normalizes ``angle_obs_key`` for training
  (``LocalStateVector.supports_vecnorm = True``), ``create()`` also returns a
  ``RenameTransform`` that stashes a pre-VecNorm raw copy (mirroring
  ``state_machine_policy.py``'s ``hfc_raw_obs`` pattern exactly); the bridge
  prefers that stash and falls back to the plain (unnormalized) observation
  key for bare deploy-contract calls with no transform chain at all.

Environment-level divergence shared by every baseline in this repo (not
introduced here, see ``ctrac_policy.py``): the env's flipper action slots are
actuated as an angular VELOCITY command (clamped to
``robot_cfg.joint_max_pivot_vels``), not an absolute position, even though
``action_spec`` bounds them by the wider position ``joint_limits``. Both modes
above write an angle-scaled value into those slots (matching the pre-existing
absolute-table convention); for MARV (``joint_limits=+-1.57``,
``max_pivot_vel=1.0``) this means large commanded angles saturate to max
velocity in the commanded sign direction rather than literally teleporting the
joint -- a pre-existing engine trait, not something either mode here corrects.

Double-DQN is a *training* choice (``DQNLoss(..., double_dqn=True)`` + a
``SoftUpdate``/``HardUpdate`` target net), not an architecture change -- the
dueling network here is exactly what AT-D3QN learns, in both action modes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.envs.transforms import RenameTransform
from torchrl.modules import QValueActor

from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["D3QNPolicyConfig"]

D3QN_RAW_ANGLE_STASH_KEY = "d3qn_raw_angle_obs"


class _DuelingQ(nn.Module):
    """Encoder -> dueling (V, A) -> Q over ``n_actions`` discrete configs."""

    def __init__(self, encoder: EncoderCombiner, n_actions: int, mlp_opts: dict):
        super().__init__()
        self.encoder = encoder
        hidden = dict(mlp_opts)
        self.value = MLP(in_dim=encoder.output_dim, out_dim=1, **hidden)
        self.adv = MLP(in_dim=encoder.output_dim, out_dim=n_actions, **hidden)

    def forward(self, *obs_tensors):
        # EncoderCombiner.forward takes kwargs; rebuild the mapping by order.
        enc = self.encoder(**dict(zip(self._obs_keys, obs_tensors)))
        v = self.value(enc)
        a = self.adv(enc)
        return v + (a - a.mean(dim=-1, keepdim=True))


class _Fig5DuelingQ(nn.Module):
    """Literal AT-D3QN Fig. 5 / ICM-D3QN Fig. 7-block-(1,2) topology, used when
    ``D3QNPolicyConfig(fig5_topology=True)``. Consumes the RAW ``PanTerrainState`` observation
    tensor directly (bypassing ``EncoderCombiner``/per-observation encoders -- there is exactly
    one input tensor, sliced internally into H and E) and reproduces:

    * terrain branch:  ``Linear(n_terrain, 32) -> LeakyReLU -> Linear(32, 4)``        (H', Fig. 5)
    * fusion:          ``Linear(4 + n_e, 16) -> LeakyReLU``                           (S_t', Fig. 5)
    * dueling heads:   ``V = Linear(16, 1)``, ``A = Linear(16, n_actions)``, single Dense each
      (Fig. 5/7 draw no hidden layer inside either head), combined as ``Q = V + (A - mean(A))``
      (the standard Dueling-DQN combination, Wang et al. 2016 -- cited by the paper as [19]/[31]
      but not restated as an equation).

    **Honest E-dimension note** (see also ``d3qn_policy.py`` module docstring and
    ``observations/pan_terrain.py``): AT-D3QN's Eq. 2 defines the robot state as the 3-tuple
    ``E = {theta_f1, theta_f2, theta_R}``, and this is what the task/paper prose (Sec. III-B)
    describes. Figs. 5 and 7 (BOTH papers) instead annotate the "E" box feeding this fusion
    layer as 4-dimensional (with an unexplained "contact" label next to it in Fig. 5, never
    described in either paper's prose) and size the fusion Dense layer as ``Dense(8, 16)``
    (= H'(4) + E(4)). We resolve this in favor of the 3-dim E: ICM-D3QN's Fig. 7 separately
    labels its OWN raw-state encoder's input as 18-dim, which only reconciles with
    ``H(15) + E(3) = 18`` (NOT ``15 + 4 = 19``) -- i.e. the one place the paper's diagrams give
    an independently-checkable total, it confirms the 3-dim E of Eq. 2. We therefore build the
    fusion layer as ``Linear(4 + 3, 16)`` (7, not 8) and treat the diagrams' "8" / unexplained
    4th "E"/"contact" feature as an inconsistency in the source paper we cannot reproduce
    bit-for-bit, rather than silently matching one diagram number (8) while contradicting both
    the other diagram's number (18) and the equation (Eq. 2).
    """

    def __init__(self, n_terrain: int, n_e: int, n_actions: int, negative_slope: float = 0.01):
        super().__init__()
        self.n_terrain = n_terrain
        self.n_e = n_e
        self.terrain_branch = nn.Sequential(
            nn.Linear(n_terrain, 32),
            nn.LeakyReLU(negative_slope),
            nn.Linear(32, 4),
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 + n_e, 16),
            nn.LeakyReLU(negative_slope),
        )
        self.value = nn.Linear(16, 1)
        self.adv = nn.Linear(16, n_actions)

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        h = raw_state[..., : self.n_terrain]
        e = raw_state[..., self.n_terrain : self.n_terrain + self.n_e]
        h_prime = self.terrain_branch(h)
        s_t = self.fusion(torch.cat([h_prime, e], dim=-1))
        v = self.value(s_t)
        a = self.adv(s_t)
        return v + (a - a.mean(dim=-1, keepdim=True))


class _DiscreteToContinuous(nn.Module):
    """Deploy bridge for ABSOLUTE mode: map the one-hot discrete action to the
    continuous env action via a fixed discrete_to_continuous table, so
    get_policy_operator() emits a continuous "action" (what the deployment
    inference module / sim expects)."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table)  # [M, action_dim]

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        idx = one_hot.argmax(dim=-1)
        return self.table.to(one_hot.device)[idx]


def _pair_expand_matrix(front_pair: tuple[int, ...], rear_pair: tuple[int, ...], n_flip: int) -> torch.Tensor:
    """[n_flip, 2] one-hot: row i is (1, 0) if flipper i is in front_pair, (0, 1) if in rear_pair."""
    m = torch.zeros(n_flip, 2)
    for i in front_pair:
        m[i, 0] = 1.0
    for j in rear_pair:
        m[j, 1] = 1.0
    return m


class _IncrementalToContinuous(TensorDictModuleBase):
    """Deploy bridge for INCREMENTAL mode: the paper-faithful paired-delta action
    set (Sec. III-C, Eq. 3). Reads the CURRENT front/rear flipper angles back out
    of the angle observation (preferring the pre-VecNorm raw stash; falling back
    to the plain observation key for bare deploy-contract calls with no transform
    chain -- the same fallback ``state_machine_policy.py``'s HFC actor uses for
    its identical ``hfc_raw_obs`` stash) and maps the selected
    ``(front_delta, rear_delta)`` pair to the flipper-joint slots of the
    continuous action. The engine actuates those slots as ANGULAR VELOCITY
    commands (engine.py: ``thetas_d = controls[...].clamp(+-max_pivot_vels)``),
    so the bridge emits ``sign(delta) * vel_cmd`` per flipper: Eq. 3's three
    motion elements become rotate-ccw / HOLD (exactly zero velocity) /
    rotate-cw, and a command that would push a flipper past its joint limit is
    gated to zero using the current angle. (An earlier revision wrote the
    absolute target angle ``theta + delta`` into the velocity slot, which made
    the "hold" action spin the flipper at ~theta rad/s instead of holding --
    audit round-3 finding.) The track slots get the constant forward command.

    A full ``TensorDictModuleBase`` (not a plain ``nn.Module`` wrapped in
    ``TensorDictModule``) because the raw-stash-or-plain-key fallback needs the
    whole tensordict, not a fixed positional set of tensors.
    """

    def __init__(
        self,
        angle_obs_key: str,
        raw_stash_key: str,
        delta_pairs: torch.Tensor,  # [9, 2]  (front_delta, rear_delta), Eq. 3
        angle_idx: torch.Tensor,  # [n_flip] long, offsets of the flipper angles inside angle_obs_key
        angle_scale: torch.Tensor,  # [n_flip]  obs -> rad affine: theta = obs * scale + offset
        angle_offset: torch.Tensor,  # [n_flip]
        delta_expand: torch.Tensor,  # [n_flip, 2] one-hot (is_front, is_rear) per flipper
        joint_low: torch.Tensor,  # [n_flip]
        joint_high: torch.Tensor,  # [n_flip]
        track_cmd: torch.Tensor,  # [n_drive]
        vel_cmd: torch.Tensor,  # scalar: flipper rotation rate [rad/s] for the +-delta actions
    ):
        super().__init__()
        self.angle_obs_key = angle_obs_key
        self.raw_stash_key = raw_stash_key
        self.in_keys = ["action", raw_stash_key, angle_obs_key]
        self.out_keys = ["action"]
        self.register_buffer("delta_pairs", delta_pairs)
        self.register_buffer("angle_idx", angle_idx)
        self.register_buffer("angle_scale", angle_scale)
        self.register_buffer("angle_offset", angle_offset)
        self.register_buffer("delta_expand", delta_expand)
        self.register_buffer("joint_low", joint_low)
        self.register_buffer("joint_high", joint_high)
        self.register_buffer("track_cmd", track_cmd)
        self.register_buffer("vel_cmd", vel_cmd)

    def _current_theta(self, tensordict: TensorDictBase) -> torch.Tensor:
        raw = tensordict.get(self.raw_stash_key, None)
        if raw is None:  # bare deploy-contract call: no VecNorm/RenameTransform chain ran
            raw = tensordict.get(self.angle_obs_key)
        obs = raw.index_select(-1, self.angle_idx.to(raw.device))
        return obs * self.angle_scale.to(raw.device) + self.angle_offset.to(raw.device)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        one_hot = tensordict.get("action")
        dev = one_hot.device
        idx = one_hot.argmax(dim=-1)
        pair = self.delta_pairs.to(dev)[idx]  # [B, 2] = (front_delta, rear_delta)
        deltas = pair @ self.delta_expand.to(dev).T  # [B, n_flip]
        theta = self._current_theta(tensordict)
        # Velocity semantics (see class docstring): hold -> 0, +-delta -> a
        # constant rotation rate, zeroed when the flipper is already at the
        # joint ANGLE limit the command would push past. The angle limits come
        # from the obs affine (theta = obs*scale + offset with scale = hi-lo,
        # offset = lo); joint_low/joint_high are the action-spec bounds of the
        # VELOCITY slots and only clamp the emitted rate.
        vel = torch.sign(deltas) * self.vel_cmd.to(dev)
        ang_lo = self.angle_offset.to(dev)
        ang_hi = self.angle_offset.to(dev) + self.angle_scale.to(dev)
        at_high = (theta >= ang_hi - 1e-6) & (vel > 0)
        at_low = (theta <= ang_lo + 1e-6) & (vel < 0)
        vel = torch.where(at_high | at_low, torch.zeros_like(vel), vel)
        vel = vel.clamp(self.joint_low.to(dev), self.joint_high.to(dev))
        track = self.track_cmd.to(dev).expand(one_hot.shape[0], -1)
        tensordict.set("action", torch.cat([track, vel], dim=-1))
        return tensordict


class _D3QNWrapper(nn.Module):
    def __init__(
        self,
        q_actor,
        bridge: nn.Module,
        incremental: bool,
        n_actions: int,
        discrete_to_continuous: torch.Tensor | None = None,
        qnet_module: "_DuelingQ | None" = None,
    ):
        super().__init__()
        self._q_actor = q_actor
        self._incremental = incremental
        self._bridge = bridge
        self._qnet_module = qnet_module  # raw _DuelingQ (pre-QValueActor); exposes .encoder via get_encoder()
        self.n_actions = n_actions  # size of the one-hot action space (9 incremental / n_bins**n_flip absolute) -- e.g. for ICM's action_dim
        if incremental:
            self._deploy_actor = TensorDictSequential(q_actor, bridge)  # bridge is a TensorDictModuleBase
        else:
            self.register_buffer("discrete_to_continuous", discrete_to_continuous)
            self._deploy_actor = TensorDictSequential(
                q_actor,
                TensorDictModule(bridge, in_keys=["action"], out_keys=["action"]),
            )

    def get_policy_operator(self):
        # continuous action matching env.action_spec (deploy + collection)
        return self._deploy_actor

    def get_qvalue_network(self):  # for DQNLoss(value_network=...)
        return self._q_actor

    def get_encoder(self) -> EncoderCombiner:
        """The Q-network's observation encoder (the SAME live instance, not a
        copy). Pass this into ``icm.ICM(encoder=..., ...)`` for ICM-D3QN so
        the curiosity module shares the Q-network's feature mapping instead of
        learning its own from scratch (a deliberate simplification -- the
        paper's Fig. 7 draws two separate encoders). See the module
        docstring's "Architecture" bullet for the optimizer-side dedup this
        implies.

        Raises when built with ``fig5_topology=True``: that mode has no
        ``EncoderCombiner`` to share (``_Fig5DuelingQ`` reads the raw
        ``PanTerrainState`` tensor directly) -- pair it with
        ``icm.ICM(separate_encoder=True, ...)`` instead, which builds the
        paper's OWN Fig. 7 raw-state encoder rather than sharing this one.
        """
        if self._qnet_module is None:
            raise RuntimeError(
                "This wrapper was constructed without a qnet_module reference, so it has no encoder "
                "to expose. D3QNPolicyConfig.create() always passes one -- if you're seeing this, the "
                "wrapper was built some other way."
            )
        encoder = getattr(self._qnet_module, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                "This wrapper was built with fig5_topology=True (_Fig5DuelingQ), which has no "
                "EncoderCombiner to share -- there is nothing for ICM to reuse. Construct the ICM "
                "module with separate_encoder=True instead (its own Fig. 7 raw-state encoder over "
                "the 'PanTerrainState' observation), not get_encoder()."
            )
        return encoder

    def action_from_onehot(self, onehot: torch.Tensor, td: TensorDictBase | None = None) -> torch.Tensor:
        """Manual-loop helper (experiments/dqn/train.py): map a one-hot Q action to
        the continuous env action, for EITHER mode. Absolute mode is a static
        lookup and ignores `td`; incremental mode needs `td` (the CURRENT
        observation, pre-action) to read back the flipper angles the deltas
        apply to.
        """
        if not self._incremental:
            return self.discrete_to_continuous[onehot.argmax(dim=-1)]
        if td is None:
            raise ValueError("incremental D3QN needs `td` (the current observation) to map one-hot -> continuous action.")
        scratch = td.copy()
        scratch.set("action", onehot)
        return self._bridge(scratch).get("action")

    def eval(self):
        self._q_actor.eval()
        return self


@dataclass
class D3QNPolicyConfig(PolicyConfig):
    """Dueling double-DQN over a discrete flipper-configuration action set.

    Args:
        mlp_opts: shared ``MLP`` kwargs for the value/advantage streams.
        optimizer_opts: optimizer kwargs for the single Q-network group.
        n_bins: angle levels per flipper in ABSOLUTE mode (``n_bins**n_flip``
            discrete actions). Unused when ``incremental=True``.
        forward_track_velocity: constant normalized forward command written into
            the track-velocity slots of the continuous action (flippers are the
            learned part, as in Pan et al.).
        incremental: use the paper's paired-delta action set (Sec. III-C Eq. 3,
            9 actions) instead of the default ABSOLUTE per-flipper bin table.
            See the module docstring for the honest trade-off between the two.
        delta: Delta_f, the paper's per-step angle increment (rad). Only used
            when ``incremental=True``. Default pi/12 (Eq. 3).
        angle_obs_key: name of the env Observation holding the flipper angles
            the incremental bridge reads back (default matches
            ``state_machine_policy.py``'s convention).
        flipper_angle_idx: offsets of the ``n_flip`` flipper angles inside
            ``angle_obs_key``'s vector (default ``(8, 9, 10, 11)`` = the
            ``LocalStateVector`` convention: [FL, FR, RL, RR]).
        flipper_angle_scale / flipper_angle_offset: affine map obs -> radians,
            ``theta = obs * scale + offset`` (scalar or per-flipper list).
            Default (``None``): derived from ``env.robot_cfg.joint_limits``.
        front_pair / rear_pair: indices (0-based, into ``flipper_angle_idx``'s
            order, which is also the order of the action vector's flipper-joint
            slots) of the flippers sharing the front / rear delta. Must
            partition ``range(n_flip)``. Default ``(0, 1)`` / ``(2, 3)`` =
            [FL, FR] front, [RL, RR] rear.
        fig5_topology: build the LITERAL AT-D3QN Fig. 5 network (``_Fig5DuelingQ``:
            terrain-branch MLP -> fusion -> 16-dim S_t' -> dueling V/A heads, all
            LeakyReLU) instead of the generic ``EncoderCombiner`` + ``mlp_opts``
            path. Requires the env to have a ``PanTerrainState`` observation
            (``observations/pan_terrain.py``) -- ``create()`` raises a clear
            ``ValueError`` otherwise. ``mlp_opts`` is ignored in this mode (the
            paper's layer widths are fixed, not configurable). See
            ``_Fig5DuelingQ``'s docstring for the honest note on Fig. 5/7's
            E-dimension inconsistency and how we resolved it.
        fig5_obs_key: name of the ``PanTerrainState`` observation to read when
            ``fig5_topology=True``.
        fig5_negative_slope: LeakyReLU negative slope for the Fig. 5 network.
            NOT given a numeric value by the paper; default is PyTorch's own
            ``nn.LeakyReLU`` default (0.01).
    """

    mlp_opts: dict
    optimizer_opts: dict
    n_bins: int = 3
    forward_track_velocity: float = 0.6
    # Flipper rotation rate [rad/s] emitted for the +-delta actions in incremental
    # mode (hold = 0). Clamped by the action spec / engine max_pivot_vels. Tune so
    # rate * control-dt matches the paper's per-step increment (pi/12 per step).
    incremental_vel_cmd: float = 1.0
    incremental: bool = False
    delta: float = math.pi / 12
    angle_obs_key: str = "LocalStateVector"
    flipper_angle_idx: tuple[int, int, int, int] = (8, 9, 10, 11)
    flipper_angle_scale: float | list[float] | None = None
    flipper_angle_offset: float | list[float] | None = None
    front_pair: tuple[int, ...] = (0, 1)
    rear_pair: tuple[int, ...] = (2, 3)
    fig5_topology: bool = False
    fig5_obs_key: str = "PanTerrainState"
    fig5_negative_slope: float = 0.01
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("D3QNPolicyConfig")
        if self.incremental and self.n_bins != 3:
            self.logger.warning(f"n_bins={self.n_bins} is ignored in incremental mode (fixed 9 paired actions, Eq. 3).")
        if self.fig5_topology and self.mlp_opts:
            self.logger.warning("fig5_topology=True: mlp_opts is ignored (the paper's Fig. 5 layer widths are fixed, not configurable).")

    def _build_action_table(self, action_spec):
        """ABSOLUTE mode: map each discrete config -> a full continuous action vector.

        Continuous action layout is ``[tracks..., flipper_joints...]`` (see
        ``Env._make_action_spec``): we fill the track slots with a constant
        forward command and the flipper-joint slots with the binned config.
        """
        adim = action_spec.shape[1]
        n_drive = adim // 2  # tracks == flipper joints count
        low = action_spec.space.low[0]
        high = action_spec.space.high[0]
        # flipper-joint slots are the second half of the action vector
        joint_low = low[n_drive:]
        joint_high = high[n_drive:]
        n_flip = joint_low.shape[0]
        # per-flipper bin levels
        levels = [torch.linspace(joint_low[i].item(), joint_high[i].item(), self.n_bins) for i in range(n_flip)]
        configs = list(product(*[range(self.n_bins)] * n_flip))
        table = torch.zeros(len(configs), adim)
        # constant forward velocity on the track slots (clamped to spec)
        v = float(self.forward_track_velocity)
        table[:, :n_drive] = torch.clamp(torch.tensor(v), low[:n_drive], high[:n_drive])
        for r, cfg in enumerate(configs):
            for i, b in enumerate(cfg):
                table[r, n_drive + i] = levels[i][b]
        return table

    def _build_delta_pairs(self) -> torch.Tensor:
        """INCREMENTAL mode, Eq. 3: A = {a_i,j} = {i*delta_f, j*delta_f}, i,j in {-1,0,1}."""
        ijs = list(product((-1, 0, 1), (-1, 0, 1)))  # 9 combos, i=front sign, j=rear sign
        return torch.tensor(ijs, dtype=torch.float32) * float(self.delta)  # [9, 2]

    def _resolve_angle_affine(self, env: Env, n_flip: int) -> tuple[torch.Tensor, torch.Tensor]:
        """obs -> radians affine for the flipper angles: theta = obs * scale + offset.
        Mirrors state_machine_policy.py's identical helper (same convention/defaults).
        """

        def as_n(v: float | list[float]) -> torch.Tensor:
            t = torch.as_tensor(v, dtype=torch.float32).flatten()
            return t.repeat(n_flip) if t.numel() == 1 else t

        if self.flipper_angle_scale is not None and self.flipper_angle_offset is not None:
            return as_n(self.flipper_angle_scale), as_n(self.flipper_angle_offset)
        joint_limits = getattr(getattr(env, "robot_cfg", None), "joint_limits", None)
        if joint_limits is not None:
            lo = joint_limits[0].detach().cpu().float()
            hi = joint_limits[1].detach().cpu().float()
            self.logger.info(f"Flipper-angle affine derived from env.robot_cfg.joint_limits: scale={(hi - lo).tolist()}, offset={lo.tolist()}")
            return hi - lo, lo
        self.logger.warning("env.robot_cfg.joint_limits unavailable; falling back to a generic +-pi/2 flipper-angle affine.")
        return torch.full((n_flip,), math.pi), torch.full((n_flip,), -math.pi / 2)

    def _build_incremental_bridge(self, env: Env, action_spec) -> _IncrementalToContinuous:
        adim = action_spec.shape[1]
        n_drive = adim // 2
        n_flip = adim - n_drive
        if len(self.flipper_angle_idx) != n_flip:
            raise ValueError(
                f"incremental=True needs flipper_angle_idx to have {n_flip} entries (one per flipper-joint "
                f"action slot), got {len(self.flipper_angle_idx)}: {self.flipper_angle_idx}."
            )
        front, rear = set(self.front_pair), set(self.rear_pair)
        if front & rear or (front | rear) != set(range(n_flip)):
            raise ValueError(
                f"front_pair={self.front_pair} and rear_pair={self.rear_pair} must partition range({n_flip}) "
                "exactly -- one shared delta per flipper pair (paper Eq. 3)."
            )
        obs_names = {o.name for o in env.observations}
        if self.angle_obs_key not in obs_names:
            self.logger.warning(
                f"angle_obs_key='{self.angle_obs_key}' is not among the env observations {sorted(obs_names)} -- "
                "the incremental deploy bridge will KeyError at runtime. Add it to the env or override angle_obs_key."
            )
        low = action_spec.space.low[0].detach().cpu()
        high = action_spec.space.high[0].detach().cpu()
        joint_low, joint_high = low[n_drive:], high[n_drive:]
        angle_scale, angle_offset = self._resolve_angle_affine(env, n_flip)
        v = float(self.forward_track_velocity)
        track_cmd = torch.clamp(torch.tensor(v), low[:n_drive], high[:n_drive])
        return _IncrementalToContinuous(
            angle_obs_key=self.angle_obs_key,
            raw_stash_key=D3QN_RAW_ANGLE_STASH_KEY,
            delta_pairs=self._build_delta_pairs(),
            angle_idx=torch.tensor(self.flipper_angle_idx, dtype=torch.long),
            angle_scale=angle_scale,
            angle_offset=angle_offset,
            delta_expand=_pair_expand_matrix(self.front_pair, self.rear_pair, n_flip),
            joint_low=joint_low,
            joint_high=joint_high,
            track_cmd=track_cmd,
            vel_cmd=torch.tensor(float(self.incremental_vel_cmd)),
        )

    def _build_fig5_qnet(self, env: Env, n_actions: int) -> "_Fig5DuelingQ":
        """fig5_topology=True: build the literal Fig. 5 network over the raw
        ``PanTerrainState`` observation (validated present) instead of an ``EncoderCombiner``.
        """
        obs_by_name = {o.name: o for o in env.observations}
        pan_obs = obs_by_name.get(self.fig5_obs_key)
        if pan_obs is None:
            raise ValueError(
                f"fig5_topology=True requires a '{self.fig5_obs_key}' observation (PanTerrainState) in the "
                f"env; found observations {sorted(obs_by_name)}. Add "
                "'{cls: ${cls:flipper_training.observations.pan_terrain.PanTerrainState}, opts: {...}}' to "
                "the config's observations list, or set fig5_obs_key to match your observation's name."
            )
        n_terrain = getattr(pan_obs, "n_heights", None)
        if n_terrain is None:
            raise ValueError(
                f"fig5_topology=True requires the observation named '{self.fig5_obs_key}' to be a "
                f"PanTerrainState instance (which exposes 'n_heights'); got {type(pan_obs).__name__}."
            )
        n_e = 3  # theta_f1, theta_f2, theta_R (Eq. 2) -- see _Fig5DuelingQ's E-dimension note
        if pan_obs.dim != n_terrain + n_e:
            raise ValueError(
                f"fig5_topology=True expects the '{self.fig5_obs_key}' observation's layout to be "
                f"[H(n_heights), theta_f1, theta_f2, theta_R] (dim = n_heights + 3); got dim={pan_obs.dim} "
                f"for n_heights={n_terrain}. Is this really a (stock) PanTerrainState?"
            )
        return _Fig5DuelingQ(n_terrain=n_terrain, n_e=n_e, n_actions=n_actions, negative_slope=self.fig5_negative_slope)

    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec

        transforms = []
        if self.incremental:
            bridge = self._build_incremental_bridge(env, action_spec)
            table = None
            n_actions = bridge.delta_pairs.shape[0]
            # VecNorm normalizes angle_obs_key during training (LocalStateVector.supports_vecnorm
            # = True); stash a pre-VecNorm raw copy the bridge can read the true angles from. Policy
            # transforms run before VecNorm in make_transformed_env, same as state_machine_policy.py.
            transforms = [RenameTransform(in_keys=[self.angle_obs_key], out_keys=[D3QN_RAW_ANGLE_STASH_KEY], create_copy=True)]
        else:
            table = self._build_action_table(action_spec)
            bridge = _DiscreteToContinuous(table)
            n_actions = table.shape[0]

        if self.fig5_topology:
            qnet = self._build_fig5_qnet(env, n_actions)
            qnet._obs_keys = [self.fig5_obs_key]
        else:
            encoders = {o.name: o.get_encoder() for o in env.observations}
            encoder = EncoderCombiner(encoders)
            qnet = _DuelingQ(encoder, n_actions, self.mlp_opts)
            qnet._obs_keys = list(encoder.encoders.keys())
        q_module = TensorDictModule(qnet, in_keys=qnet._obs_keys, out_keys=["action_value"])
        q_actor = QValueActor(q_module, in_keys=qnet._obs_keys, action_space="one_hot", action_value_key="action_value")

        wrapper = _D3QNWrapper(
            q_actor, bridge, incremental=self.incremental, n_actions=n_actions, discrete_to_continuous=table, qnet_module=qnet
        )
        if kwargs.get("device", None) is not None:
            wrapper.to(kwargs["device"])

        if weights_path := kwargs.get("weights_path", None):
            missing_unexpected = wrapper.load_state_dict(torch.load(weights_path, map_location=kwargs.get("device", "cpu")), strict=False)
            self.logger.info(f"Loaded weights from {weights_path}")
            if missing_unexpected.missing_keys:
                self.logger.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                self.logger.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        optim_groups = [{"params": q_actor.parameters(), "name": "qnet", **self.optimizer_opts}]
        n_params = sum(p.numel() for p in q_actor.parameters())
        topology_msg = (
            f"Fig.-5-literal topology ({self.fig5_obs_key}: {qnet.n_terrain} terrain + {qnet.n_e} robot-state -> "
            f"H'(4)+E({qnet.n_e})->S_t'(16)->dueling)"
            if self.fig5_topology
            else f"generic EncoderCombiner + mlp_opts={self.mlp_opts}"
        )
        if self.incremental:
            self.logger.info(
                f"AT-D3QN (incremental, paper-faithful Eq. 3): dueling Q over 9 paired front/rear flipper-delta "
                f"actions (delta={self.delta:.4f} rad = {math.degrees(self.delta):.1f} deg), {n_params:,} params, "
                f"{topology_msg}. angle_obs_key='{self.angle_obs_key}'{list(self.flipper_angle_idx)}, "
                f"front={self.front_pair} rear={self.rear_pair}. Train with DQNLoss(double_dqn=True); deploy bridge "
                f"reads current flipper angles back from the observation each step (see "
                f"wrapper.action_from_onehot for the manual-loop path)."
            )
        else:
            self.logger.info(
                f"AT-D3QN (absolute, divergent from the paper -- see module docstring): dueling Q over {n_actions} "
                f"discrete flipper configs (n_bins={self.n_bins}), {n_params:,} params, {topology_msg}. Train with "
                f"DQNLoss(double_dqn=True); env bridge via wrapper.discrete_to_continuous [{table.shape[0]}, "
                f"{table.shape[1]}]. Set incremental=True for the paper's 9-action paired-delta scheme (Eq. 3)."
            )
        return wrapper, optim_groups, transforms
