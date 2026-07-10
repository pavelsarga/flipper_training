"""``LocalStateVector`` + the Pecka et al. (2016, IROS) "Autonomous Flipper
Control with Safety Constraints" Sec IV-A state feature (ii): "height of the
terrain approximately 20 cm in front of the robot body".

The paper reads this quantity from an octomap built online from the robot's
own laser scans (a REAL sensor pipeline). This simulator has no octomap, but
it DOES have the analytic ground-truth heightmap grid the terrain was
generated from (``env.terrain_cfg.z_grid``, the same grid the physics engine
itself queries for ground contact -- see ``engine.py: find_contact_points``),
which is the noise-free version of exactly the same signal. Sampling that grid
at a point ahead of the robot is therefore a faithful (if idealized -- no
octomap noise/occlusion) stand-in for the paper's feature, not a fabrication:
the paper's own experiments assume the octomap has already resolved a clean
height at that point.

Frame convention (matches ``Heightmap`` (``observations/heightmap.py``), the
only other terrain-sampling observation in this repo): "in front of the robot"
means displaced by ``lookahead_dist`` along the robot's local +X axis
(``robot_cfg.driving_direction``), rotated into world coordinates using ONLY
the robot's YAW (``utils.geometry.planar_rot_from_q``) -- i.e. horizontal
ground distance in the direction the robot is heading, not the full 3D body
+X axis. This deliberately does NOT foreshorten with pitch/roll: on a steep
slope, the full-3D body-frame +X axis tilts down/up and its horizontal (XY)
projection shrinks by cos(pitch), which would silently pull the lookahead
point closer exactly when climbing steep terrain (i.e. exactly when this
feature matters most for flipper control) -- probably not what "20cm in front
of the robot" is meant to describe, and not how the paper's own octomap-slice
would behave either (an online-built map is queried in world/ground
coordinates, not body coordinates).

The sampled height is returned RELATIVE to the robot's own current height
(``z_terrain_ahead - curr_state.x[:,2]``), matching how ``Heightmap`` reports
heights (relative to the robot, not absolute world Z) -- i.e. positive = a
step UP ahead, negative = a step DOWN ahead. The paper does not specify the
frame of its own height feature precisely; robot-relative is the physically
meaningful choice for a *linear* policy (a fixed weight then means "respond to
a step of a given size", independent of the robot's absolute Z) and matches
this repo's own convention for every other quantity in ``LocalStateVector``
(goal vector, velocities -- all robot-relative, never raw world coordinates).

Consumer: ``flipper_training.policies.pecka_policy.PeckaLinearPolicyConfig``
via its ``extra_feature_idx`` hook -- point ``obs_key`` at this class's name
and set ``extra_feature_idx: -1`` (the appended feature is always last) to
restore the paper's full 6-parameter policy phi(s) = [pitch, height_ahead, 1].
"""

from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from tensordict import TensorDictBase

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import planar_rot_from_q
from .robot_state import LocalStateVector

__all__ = ["LocalStateVectorWithTerrainHeightAhead"]


@dataclass
class LocalStateVectorWithTerrainHeightAhead(LocalStateVector):
    """``LocalStateVector`` with one extra trailing feature: terrain height
    ``lookahead_dist`` meters ahead of the robot (robot-yaw frame), relative to
    the robot's current height. See module docstring for the exact frame
    convention and the honest gap vs. the paper's octomap-based sensor
    (analytic ground-truth grid here, not a simulated noisy/occluded scan).

    Args:
        lookahead_dist: distance ahead of the robot to sample, in meters
            (paper Sec IV-A: "approximately 20 cm").
    """

    lookahead_dist: float = 0.20

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        base = super().__call__(prev_state, action, prev_state_der, curr_state)  # (B, LocalStateVector.dim)
        height_ahead = self._height_ahead(curr_state)  # (B, 1)
        return torch.cat([base, height_ahead.to(self.env.out_dtype)], dim=1)

    def _height_ahead(self, curr_state: PhysicsState) -> torch.Tensor:
        """Terrain height ``lookahead_dist`` m ahead of the robot (yaw-only
        forward direction), relative to the robot's current Z. Shape (B, 1).
        """
        B = curr_state.x.shape[0]
        R_yaw = planar_rot_from_q(curr_state.q)  # (B, 2, 2)
        local_fwd = curr_state.x.new_tensor([self.lookahead_dist, 0.0]).expand(B, 2).unsqueeze(1)  # (B, 1, 2)
        world_offset = torch.bmm(local_fwd, R_yaw.transpose(1, 2))  # (B, 1, 2), same convention as Heightmap.__call__
        query_xy = curr_state.x[:, :2].unsqueeze(1) + world_offset  # (B, 1, 2)
        z_ahead = interpolate_grid(self.env.terrain_cfg.z_grid, query_xy, self.env.terrain_cfg.max_coord)  # (B, 1, 1)
        return z_ahead.squeeze(-1) - curr_state.x[:, 2:3]  # (B, 1), relative to robot height

    def from_realistic_world(self, tensordict: TensorDictBase) -> torch.Tensor:
        """NOT implemented -- deliberately raises rather than silently emitting a
        wrong-shaped or fabricated value.

        Real-robot deployment would need the lookahead sample wired to the live
        elevation-mapping pipeline (this repo's ROS bridge exposes
        ``/elevation_map_filtered``, see the top-level CLAUDE.md), which is out
        of scope here. Returning e.g. zeros would silently feed a wrong
        "flat ground ahead" reading into a real flipper controller -- worse
        than failing loudly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.from_realistic_world is not implemented: real-robot terrain-height-ahead "
            "requires bridging the elevation_map pipeline, which hasn't been wired up. Use plain LocalStateVector "
            "(4-parameter Pecka policy) for real-robot deployment, or implement this bridge first."
        )

    @property
    def dim(self) -> int:
        return super().dim + 1

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )
