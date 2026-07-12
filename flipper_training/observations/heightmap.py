"""Robot-centric local terrain heightmap observation.

Used by the C-TRAC baseline (Pan et al. 2025, IROS, Sec. IV-A.1) for both of the
paper's heightmap terms:

* ``h_t^l`` (Eq. 1) — part of the ACTOR observation ``o_t``: "Height map h_t^l is
  centered on the robot and spans a length range of [0.4 m, 1.0 m] and a width
  range of [-0.5 m, 0.5 m]" (forward-only patch). Use the plain :class:`Heightmap`
  class for this, listed as a normal (non-privileged) entry in ``observations:``.
* ``h_t^f`` (Eq. 2) — part of the CRITIC-ONLY privileged state
  ``s_t = [o_t, c_t, h_t^f]``: "a larger terrain height map of range
  [-1.0 m, 1.4 m]" (extends behind AND ahead of the robot, unlike ``h_t^l``). The
  paper states only this one (length) interval for ``h_t^f``; it never says the
  WIDTH also grows, so :class:`PrivilegedHeightmap`'s default width matches
  :class:`Heightmap`'s and only the length range is widened to the literal paper
  numbers — see that class's docstring. Use :class:`PrivilegedHeightmap` (a
  distinct class purely so it gets a distinct :attr:`Observation.name`, since
  ``Observation.name`` is the class name and two differently-configured
  instances of the same class cannot coexist as separate ``env.observations``
  entries) listed in ``privileged_observations`` so
  ``flipper_training.policies.ctrac_policy.CTRACConfig`` routes it to the
  asymmetric critic only, never the actor.

Math (both classes): a robot-centered, yaw-rotated 2D sampling grid is built once
at construction (``percep_shape`` points spanning ``percep_extent`` in the robot's
local XY frame, X=forward/length, Y=left-right/width); each step the grid is
rotated by the robot's current yaw and translated to the robot's current XY
position, terrain height is bilinearly interpolated at those world points from
``env.terrain_cfg.z_grid`` (:func:`flipper_training.utils.environment.interpolate_grid`),
and the robot's own Z is subtracted so the map reads "height relative to the
robot" (positive = terrain above the robot's base). This is this repo's
established pattern for local heightmap observations (the same construction used
by the alternate FTR-Bench-facing ``marv_rl_training.observations.heightmap.Heightmap``,
which already reuses these exact canonical ``flipper_training.utils`` helpers);
this module is the canonical-tree version referenced by
``flipper_training.observations.heightmap.Heightmap`` in the (previously stale —
the module did not exist yet) example config
``test_configs/deterministic_flats_debug_with_heightmap.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded
from tensordict import TensorDictBase

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import planar_rot_from_q
from . import Observation, ObservationEncoder

__all__ = ["Heightmap", "PrivilegedHeightmap", "HeightmapEncoder"]


class HeightmapEncoder(ObservationEncoder):
    """Small strided-CNN encoder mapping a ``(1, H, W)`` heightmap patch to a flat embedding.

    ``activation`` is a constructor kwarg (default ``nn.ReLU``, this repo's usual
    conv-net default, e.g. the FTR-facing ``marv_rl_training`` heightmap encoders)
    so C-TRAC's "full" config can set it to ``nn.LeakyReLU`` for every network per
    the paper's Sec. V-A.1 ("All neural networks employ LeakyReLU activation
    functions") — pass ``activation: ${cls:torch.nn.LeakyReLU}`` in ``encoder_opts``.
    """

    def __init__(
        self,
        img_shape: tuple[int, int],
        output_dim: int,
        activate_output: bool = False,
        activation: type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        super().__init__(output_dim)
        self.img_shape = img_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(4 * 32, output_dim),
            activation() if activate_output else nn.Identity(),
        )

    def forward(self, hm: torch.Tensor) -> torch.Tensor:
        if hm.ndim > 4:
            # optional leading time dimension, e.g. (B, T, 1, H, W) -- flatten, encode, restore
            b, t = hm.shape[:2]
            c, h, w = hm.shape[2:]
            y = self.encoder(hm.reshape(b * t, c, h, w))
            return y.view(b, t, -1)
        return self.encoder(hm)


@dataclass
class Heightmap(Observation):
    """Robot-centric local terrain heightmap sampled from ``env.terrain_cfg.z_grid``.

    Defaults match the paper's ``h_t^l`` (Eq. 1): forward-only patch, length range
    [0.4 m, 1.0 m], width range [-0.5 m, 0.5 m] (``percep_extent`` below). Intended
    as a normal (non-privileged) ``observations:`` entry, visible to the actor.

    Args:
        percep_shape: ``(rows, cols)`` resolution of the sampled patch.
        percep_extent: ``(x_start, y_start, x_end, y_end)`` in the robot's local
            frame (X=forward, Y=left), meters. Row 0 samples ``x_start``, row -1
            samples ``x_end``; likewise columns for Y. Paper default: forward-only,
            [0.4, 1.0] x [-0.5, 0.5].
        interval: ``(min, max)`` meters the (robot-relative) height is clamped to.
        normalize_to_interval: if True, divide the clamped height by
            ``interval[1] - interval[0]`` (scales to roughly [-1, 1] wide).
    """

    percep_shape: tuple[int, int] = (32, 32)
    percep_extent: tuple[float, float, float, float] = (0.4, -0.5, 1.0, 0.5)
    interval: tuple[float, float] = (-1.0, 1.0)
    normalize_to_interval: bool = False
    supports_vecnorm = False

    def __post_init__(self):
        if self.apply_noise and not isinstance(self.noise_scale, (float, torch.Tensor)):
            raise ValueError("Noise scale must be specified if apply_noise is True and must be a float or tensor.")
        self._initialize_perception_grid()

    def _initialize_perception_grid(self) -> None:
        x_space = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0])
        y_space = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1])
        px, py = torch.meshgrid(x_space, y_space, indexing="ij")
        grid_points_2d = torch.dstack([px, py]).reshape(-1, 2)  # (N, 2)
        self._grid_points_2d = grid_points_2d.unsqueeze(0).repeat(self.env.n_robots, 1, 1).to(self.env.device)  # (B, N, 2)

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        b = curr_state.x.shape[0]
        r_yaw = planar_rot_from_q(curr_state.q)  # (B, 2, 2)
        rotated = torch.bmm(self._grid_points_2d, r_yaw.transpose(1, 2))  # (B, N, 2)
        global_pts = rotated + curr_state.x[..., :2].unsqueeze(1)  # (B, N, 2)
        z = interpolate_grid(self.env.terrain_cfg.z_grid, global_pts, self.env.terrain_cfg.max_coord)  # (B, N, 1)
        hm = z.reshape(b, 1, self.percep_shape[0], self.percep_shape[1]) - curr_state.x[..., 2].reshape(-1, 1, 1, 1)
        if self.apply_noise:
            noise = torch.randn_like(hm) * self.noise_scale
            hm = hm + noise
        hm = hm.clamp(self.interval[0], self.interval[1])
        if self.normalize_to_interval:
            hm = hm / (self.interval[1] - self.interval[0])
        return hm.to(self.env.out_dtype)

    def from_realistic_world(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Resample a real elevation map (``/elevation_map``, see ``flipper_policy_node.py``) onto ``percep_extent``.

        Expects ``tensordict["heightmap"]`` (H_src, W_src) up to leading singleton dims,
        robot-local-frame, and ``tensordict["heightmap_extent"]`` = (x_max, y_max, x_min,
        y_min) — the same convention documented for the ROS deploy node's elevation-map
        callback. Raises if the supplied extent does not fully cover ``percep_extent``.

        Live-sim-found bug (round 4): ``policy_inference_module.infer_action()`` builds its
        input tensordict via a blanket ``torch.tensor(v).unsqueeze(0)`` over EVERY kwarg
        (adding a batch dim of 1), so ``tensordict["heightmap"]`` actually arrives as
        ``(1, H_src, W_src)``, not the bare ``(H_src, W_src)`` this docstring used to claim
        without qualification — ``.unsqueeze(0).unsqueeze(0)`` on that made a 5-D tensor and
        crashed ``grid_sample`` ("expected grid to have size 3 in last dimension") the
        instant this was exercised through the real ROS node (never reached by the
        differentiable-physics ``Env``, which never calls this method — only caught by
        actually running ``flipper_policy_node.py`` against the live sim). Fixed by
        unconditionally squeezing down to exactly 2-D first, so both the bare-2-D case
        (e.g. a hand-built test tensordict) and the batched-3-D real deploy case work
        identically.
        """
        hm: torch.Tensor = tensordict["heightmap"].to(self.env.device)
        while hm.ndim > 2:
            hm = hm.squeeze(0)
        extent = tensordict["heightmap_extent"]
        if isinstance(extent, torch.Tensor):
            extent = extent.cpu().squeeze().tolist()
        if extent[0] < self.percep_extent[0] or extent[1] < self.percep_extent[1] or extent[2] > self.percep_extent[2] or extent[3] > self.percep_extent[3]:
            raise ValueError(f"Real-world heightmap extent {extent} does not fully contain percep_extent {self.percep_extent}.")
        x_target = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0], device=hm.device)
        y_target = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1], device=hm.device)
        px, py = torch.meshgrid(x_target, y_target, indexing="ij")
        grid_v = 2 * (px - extent[0]) / (extent[2] - extent[0]) - 1
        grid_u = 2 * (py - extent[1]) / (extent[3] - extent[1]) - 1
        grid = torch.stack((grid_u, grid_v), dim=-1).unsqueeze(0)  # (1, H_p, W_p, 2)
        # grid_sample(input=(N=1,C=1,H_src,W_src), grid=(N=1,H_p,W_p,2)) -> (N=1,C=1,H_p,W_p),
        # already exactly get_spec()'s (n_robots, 1, H_p, W_p) 4-D shape (matches __call__'s
        # (b, 1, H, W) too) -- do NOT squeeze this, a second live-sim-found bug (round 4):
        # a now-removed `.squeeze(0)` here silently dropped the leading dim, so the actor's
        # EncoderCombiner received a 3-D (1, H_p, W_p) heightmap instead of 4-D (1, 1, H_p,
        # W_p) -- one dim short of what HeightmapEncoder's Conv2d stack was built for, which
        # surfaced downstream as an opaque "mat1 and mat2 shapes cannot be multiplied"
        # RuntimeError inside the flattened Linear layer, not as an obvious shape error at
        # the source. Only reachable via the real ROS deploy path (`_to_realistic_env`) --
        # the differentiable-physics `Env`'s own `Heightmap.__call__` never calls this method,
        # so no amount of training/eval-rollout testing could have caught it; only found by
        # actually running `flipper_policy_node.py` against the live sim and inspecting the
        # tensordict shapes at the crash site.
        z = torch.nn.functional.grid_sample(hm.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        z = z.clamp(self.interval[0], self.interval[1])
        if self.normalize_to_interval:
            z = z / (self.interval[1] - self.interval[0])
        return z.to(self.env.out_dtype)  # (1, 1, H_p, W_p) -- n_robots=1, channel=1

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, 1, self.percep_shape[0], self.percep_shape[1]),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> HeightmapEncoder:
        return HeightmapEncoder(self.percep_shape, **(self.encoder_opts or {}))


@dataclass
class PrivilegedHeightmap(Heightmap):
    """The paper's wider, critic-only heightmap ``h_t^f`` (Eq. 2).

    Identical to :class:`Heightmap` in every respect (same math, same encoder) —
    the ONLY reason this is a separate class is that ``Observation.name`` is the
    class name, so a second, differently-configured ``Heightmap`` instance (wider
    extent) needs its own class to coexist with the actor's ``Heightmap`` entry in
    ``env.observations``. List this one in ``policy_opts.privileged_observations``
    (``CTRACConfig``) so it reaches the asymmetric critic only, never the actor.

    Default ``percep_extent`` widens the LENGTH (X) range to the paper's literal
    ``h_t^f`` numbers, [-1.0 m, 1.4 m] (extending behind the robot too, unlike
    ``h_t^l``'s forward-only [0.4, 1.0]); the paper states only this one interval
    for ``h_t^f`` and never says the WIDTH (Y) also grows, so the width is kept
    identical to :class:`Heightmap`'s default ([-0.5 m, 0.5 m]) rather than
    guessing a number the paper does not give — an explicit, documented choice,
    not a claim that the paper specifies a wider width. Override ``percep_extent``
    directly if a wider width is wanted.

    Privileged/training-only: ``from_realistic_world`` returns ZEROS (like
    ``GroundTruthContacts``) rather than attempting real-world resampling — the
    deployed actor never reads this key, and its wide extent may exceed whatever
    the real elevation map (``/elevation_map``) happens to cover, so attempting a
    real resample here would risk spuriously crashing sim-to-real deployment for
    a value nothing downstream of deployment actually consumes.
    """

    percep_shape: tuple[int, int] = (64, 32)
    percep_extent: tuple[float, float, float, float] = (-1.0, -0.5, 1.4, 0.5)

    def from_realistic_world(self, tensordict: TensorDictBase) -> torch.Tensor:
        return torch.zeros(
            (1, 1, self.percep_shape[0], self.percep_shape[1]),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )
