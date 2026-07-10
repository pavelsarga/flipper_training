"""Privileged ground-truth contact observation (per-area contact positions + flags).

Extracts, from the physics engine's own contact model, the C-TRAC (Pan et al.
2025, Sec. IV-A Eq. 2) privileged contact vector ``c_t``: for each *driving
part* (flipper) — the paper's "4 areas" — a contact position (robot frame) and
a binary contact-existence flag.

Source of truth: ``PhysicsStateDer.in_contact`` (soft per-collision-point
contact indicator, ``sigmoid(-penetration/temperature)``, shape
``(B, n_pts, 1)``) and ``PhysicsStateDer.robot_points`` (world-frame collision
points). The engine concatenates points *driving-parts-first*
(``engine.py: torch.cat((driving_parts_world, body_world))``), each part
contributing ``points_per_driving_part`` consecutive points, so area ``i``
occupies the point slice ``[i*P, (i+1)*P)``.

Per area ``i``:
* ``prob_i``  = 1.0 if the *max* soft contact indicator over the area's points
  exceeds ``prob_threshold`` (default 0.5 = zero penetration depth), else 0.0.
* ``pos_i``   = contact-weighted mean of the area's points, expressed in the
  ROBOT frame (world points shifted by ``prev_state.x`` and rotated by
  ``prev_state.q``-inverse — the same state the derivative was computed from),
  zeroed when ``prob_i == 0``.

Layout of the emitted vector (``dim = 4 * n_areas``):
``[pos_0.xyz, pos_1.xyz, ..., pos_{A-1}.xyz, prob_0, ..., prob_{A-1}]``.

Intended consumers (C-TRAC):
* the **asymmetric critic** — Q(obs, c_t, a_t) reads this key; the actor must
  NOT list it among its inputs;
* the **C-VAE hybrid loss** — this vector is the training target for the
  contact-position and contact-probability decoder heads (Eq. 12-13).

Timing note: ``prev_state_der`` describes the transition *into* ``curr_state``
(the first engine substep from ``prev_state``), so the contact labels lag the
current observation by one physics substep. This is the exact ground truth the
simulator has; there is no fresher contact label without stepping the engine.

Deployment: this is privileged simulation-only information.
``from_realistic_world`` returns ZEROS — safe because only the critic and the
C-VAE loss (both training-only) consume the key; the deployed actor never
reads it.

VecNorm: ``supports_vecnorm = False`` — the values are metric targets for the
C-VAE loss and must stay in meters / {0, 1}.
"""

from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from tensordict import TensorDictBase

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import inverse_quaternion, rotate_vector_by_quaternion
from . import Observation, ObservationEncoder
from .latent_control import IdentityEncoder
from .robot_state import LocalStateVectorEncoder


@dataclass
class GroundTruthContacts(Observation):
    """Per-driving-part ground-truth contact positions (robot frame) + binary contact flags.

    Args:
        prob_threshold: soft-contact value above which an area counts as "in contact"
            (0.5 corresponds to zero penetration depth in the engine's sigmoid model).
        encoder_opts: if ``None`` (default) the encoder is the identity (the raw
            metric vector is concatenated into the consumer, e.g. the critic);
            otherwise kwargs for an MLP encoder (``hidden_dim``, ``num_hidden``,
            ``output_dim``, ``layernorm``, ...).
    """

    supports_vecnorm = False
    prob_threshold: float = 0.5

    def __post_init__(self):
        if self.apply_noise:
            raise ValueError("GroundTruthContacts is a privileged ground-truth signal and must not be noised (apply_noise=True is invalid).")
        self.n_areas = self.env.robot_cfg.num_driving_parts
        self.pts_per_area = self.env.robot_cfg.points_per_driving_part

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        a, p = self.n_areas, self.pts_per_area
        n = a * p
        soft = prev_state_der.in_contact[:, :n, 0]  # (B, A*P) soft contact indicator in [0, 1]
        pts_world = prev_state_der.robot_points[:, :n]  # (B, A*P, 3), world frame, pose = prev_state
        # Express contact points in the robot frame of the state the derivative was computed from
        inv_q = inverse_quaternion(prev_state.q)  # (B, 4)
        pts_local = rotate_vector_by_quaternion(pts_world - prev_state.x.unsqueeze(1), inv_q)  # (B, A*P, 3)
        soft = soft.view(-1, a, p)
        pts_local = pts_local.view(-1, a, p, 3)
        # Binary contact-existence flag per area (paper's c^prob)
        prob = (soft.amax(dim=2) > self.prob_threshold).to(self.env.out_dtype)  # (B, A)
        # Contact-weighted mean contact position per area (paper's c), zeroed for non-contact areas
        w = soft.unsqueeze(-1)  # (B, A, P, 1)
        pos = (pts_local * w).sum(dim=2) / w.sum(dim=2).clamp_min(1e-6)  # (B, A, 3)
        pos = pos * prob.unsqueeze(-1)
        return torch.cat([pos.flatten(1), prob], dim=1).to(self.env.out_dtype)  # (B, 4*A)

    def from_realistic_world(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Privileged sim-only signal — unavailable on the real robot; returns zeros.

        Only the critic and the C-VAE loss (training-only) consume this key, so the
        zeros are never read by the deployed actor.
        """
        return torch.zeros((1, self.dim), device=self.env.device, dtype=self.env.out_dtype)

    @property
    def dim(self) -> int:
        return 4 * self.n_areas  # 3 position coords + 1 contact flag per area

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> ObservationEncoder:
        if not self.encoder_opts:
            return IdentityEncoder(self.dim)
        return LocalStateVectorEncoder(input_dim=self.dim, **self.encoder_opts)
