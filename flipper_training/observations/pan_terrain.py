"""AT-D3QN / ICM-D3QN paper state (Pan et al. 2023): local terrain heights H + robot state E.

Pan, H. et al. "Deep Reinforcement Learning for Flipper Control of Tracked Robots,"
arXiv:2306.10352 (AT-D3QN), Sec. III-B "State Space", Eq. 1-2, Fig. 3; and "... in Urban
Rescuing Environments," Remote Sensing 15(18):4616 (ICM-D3QN, the journal extension of the
same paper), Sec. 3.2, Eq. 1-2, Fig. 3 -- the two papers' state-space sections are the same
formulation, the journal version just OCRs more cleanly (see "mean vs. min" note below), and
we treat it as the primary source for the exact wording.

State = ``[H (n_heights,), theta_f1, theta_f2, theta_R]``, dim = ``n_heights + 3``
(paper default ``n_heights = 15``, Sec. III-B). This EXACT layout (H first, then the 3-tuple
E in front/rear/pitch order) is a contract other paper-faithful components depend on:
``d3qn_policy.py``'s ``fig5_topology`` mode slices this same tensor into H'/E branches, and
``icm.py``'s ``separate_encoder`` mode feeds the WHOLE vector into the paper's own raw-state
encoder (Fig. 7's ``psi``, input dim 18 = 15 + 3) -- do not reorder these fields.

Local terrain heights H (Eq. 1)
--------------------------------
``H = {h_i} = { mean_{(xT,zT) in T_i} (zT) }``, ``i = 1..N``, where ``T_i`` are ``N`` equally
spaced x-bins spanning ``[-N/2*d, N/2*d]`` in the paper's ``[L]`` frame (Sec. III-B: "the
center of the robot chassis serves as the origin, the X-axis representing the robot's
forward direction and the Z-axis indicating the opposite direction of gravity").

* **mean, not min**: the arXiv (AT-D3QN) PDF's Eq. 1 typesets a glyph that looks like `min`
  under the aggregation; the Remote Sensing (ICM-D3QN) journal PDF -- same equation, same
  authors, cleaner OCR -- renders it unambiguously as `mean`, and the surrounding prose in
  BOTH papers says "N *average* heights" and uses an over-bar (h-bar = conventional "mean")
  notation. We implement `mean`, confirmed against `pdftotext` output of the journal PDF
  (grep "mean" hits the exact equation line).
* **frame**: we read "[L]"'s axes as YAW-ONLY heading-aligned (X = the robot's yaw-projected
  forward direction in the horizontal plane) with a WORLD-VERTICAL Z (Sec. III-B says Z is
  "the opposite direction of gravity", not "normal to the chassis") -- i.e. roll/pitch are
  NOT applied when placing the sample grid, only yaw. This is a deliberate reading, not
  explicitly disambiguated by the paper: it is the only interpretation under which H reflects
  the true upcoming terrain profile (a chassis-fixed frame that tilts with pitch would make a
  constant slope look perfectly flat in H, defeating the point of feeding terrain shape to the
  network). It also matches how CLAUDE.md describes this repo's own heightmap convention
  elsewhere (gravity/world-aligned, yaw-rotated to the robot heading).
* **bin width `d`**: neither paper gives `d` a numeric value anywhere (checked via
  `pdftotext` grep over both PDFs) -- Fig. 3 only labels it symbolically. Default
  ``bin_width=0.1`` m (so the ``N=15`` window spans +-0.75 m) is OUR choice, sized so the
  window comfortably covers MARV's flipper reach (pivot at x=+-0.256 m, pivot->tip 0.3815 m,
  so tips sit near +-0.64 m from chassis center) with a bit of look-ahead/behind margin.
  Override via config for other robots/terrains.
* **per-bin averaging**: the paper averages over a literal lidar point CLOUD falling in each
  bin; we have a continuous ground-truth heightmap instead (bilinearly interpolated via
  ``flipper_training.utils.environment.interpolate_grid``), so we approximate "mean over T_i"
  by averaging a dense ``x_samples_per_bin * y_samples_per_bin`` sub-grid of interpolated
  samples spanning the bin's x-width and a ``y_window`` strip (see ``sample_binned_terrain_heights``).
  ``y_window`` defaults to the robot's own footprint half-width (derived from
  ``robot_cfg.body_bbox`` / driving-part bboxes) rather than the paper's implicit "collapse
  all Y" (Sec. III-A: "project the terrain outline ... onto the robot's lateral side"),
  because sampling literally infinite Y would let unrelated off-path terrain leak into H.

Robot state E (Eq. 2)
----------------------
``E = {theta_f1, theta_f2, theta_R}``, all in ``[-pi/3, pi/3]``: front flipper angle, rear
flipper angle (both POSITIVE = flipper above the chassis plane, Sec. III-B), and chassis
pitch (POSITIVE = chassis nose above the local x-axis, i.e. nose-UP / climbing).

**Sign conventions are NOT the same as this engine's raw state and must be corrected here:**
this was verified against the actual engine code (``engine.py``'s ``assemble_and_transform_robot``
rotates each driving part's joint-local points by ``rot_Y(theta)``, whose matrix is
``[[cos,0,sin],[0,1,0],[-sin,0,cos]]``) plus the robot's mesh geometry (``robots/marv.yaml``:
front-flipper driving-part points sit at LOCAL +X relative to their joint/pivot, rear-flipper
points sit at LOCAL -X), and cross-checked empirically (constructing a quaternion for a known
+10 deg rotation about world +Y and calling ``rotate_vector_by_quaternion``/``quaternion_to_pitch``
directly):

* FRONT flipper: `Ry(theta) @ (+X)` moves the tip toward -Z as `theta` increases -> raw
  ``theta`` POSITIVE means DOWN. Paper wants positive = up, so ``theta_f1 = -mean(raw FL, FR)``.
* REAR flipper: `Ry(theta) @ (-X)` moves the tip toward +Z as `theta` increases -> raw
  ``theta`` POSITIVE already means UP, matching the paper directly: ``theta_f2 = +mean(raw RL, RR)``.
* Chassis pitch: empirically, ``quaternion_to_pitch(q)`` is POSITIVE when the body-forward
  vector rotates toward -Z (nose DOWN). Paper wants positive = nose-UP, so
  ``theta_R = -quaternion_to_pitch(q)``.
* This matches this package's own ``CLAUDE.md`` ("Flipper Angle Conventions": front fully-up
  = -pi/2, rear fully-up = +pi/2 in the raw/engine convention), which independently confirms
  the front/rear asymmetry derived above.

Front/rear flipper grouping is auto-detected from ``robot_cfg.driving_part_names`` (substring
match on "front" / "back"|"rear", case-insensitive); pass ``front_indices``/``rear_indices``
explicitly for robots whose names don't follow that convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import quaternion_to_pitch, quaternion_to_yaw
from flipper_training.policies import MLP

from . import Observation, ObservationEncoder

__all__ = ["PanTerrainState", "PanTerrainStateEncoder", "resolve_front_rear_indices", "resolve_front_rear_hinges", "sample_terrain_points_relative", "sample_binned_terrain_heights"]


def resolve_front_rear_indices(
    robot_cfg,
    front_indices: tuple[int, ...] | None,
    rear_indices: tuple[int, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolves the (front-pair, rear-pair) driving-part index tensors used to collapse MARV's
    4 independent flippers into the paper's 2 paired DOFs (front flipper / rear flipper --
    the paper's NuBot-Rescue robot has "advantageous central symmetry", i.e. left/right always
    move together; Sec. III-A).

    If either override is ``None``, auto-detects by matching "front" / "back"|"rear"
    (case-insensitive substring) against ``robot_cfg.driving_part_names``; raises a clear
    ``ValueError`` if that doesn't cleanly partition all driving parts (e.g. a robot whose
    parts aren't named this way) -- pass both overrides explicitly in that case.
    """
    n = robot_cfg.num_driving_parts
    if front_indices is None or rear_indices is None:
        names = [str(n_).lower() for n_ in robot_cfg.driving_part_names]
        auto_front = tuple(i for i, nm in enumerate(names) if "front" in nm)
        auto_rear = tuple(i for i, nm in enumerate(names) if ("back" in nm or "rear" in nm))
        if front_indices is None:
            front_indices = auto_front
        if rear_indices is None:
            rear_indices = auto_rear
        if not front_indices or not rear_indices or set(front_indices) & set(rear_indices) or set(front_indices) | set(rear_indices) != set(range(n)):
            raise ValueError(
                f"Could not auto-detect a front/rear partition of the {n} driving parts from their names "
                f"{robot_cfg.driving_part_names!r} (got front={front_indices}, rear={rear_indices}). Pass "
                "front_indices=/rear_indices= explicitly (0-based indices into driving_part_names, must "
                "partition range(num_driving_parts))."
            )
    front_t = torch.as_tensor(front_indices, dtype=torch.long)
    rear_t = torch.as_tensor(rear_indices, dtype=torch.long)
    return front_t, rear_t


def resolve_front_rear_hinges(robot_cfg, front_idx: torch.Tensor, rear_idx: torch.Tensor) -> tuple[float, float, float, float]:
    """Mean (x, z) hinge/pivot position (body frame, meters) of the front pair and rear pair,
    from ``robot_cfg.joint_positions`` (the driving parts' pivot points, ``robots/<kind>.yaml``'s
    ``joint_position``). Returns ``(front_x, front_z, rear_x, rear_z)``.
    """
    jp = robot_cfg.joint_positions.detach().cpu()
    front_xy = jp[front_idx]
    rear_xy = jp[rear_idx]
    return (
        float(front_xy[:, 0].mean()),
        float(front_xy[:, 2].mean()),
        float(rear_xy[:, 0].mean()),
        float(rear_xy[:, 2].mean()),
    )


def _resolve_y_half_width(robot_cfg) -> float:
    """Robot footprint half-width in Y (meters), from body/driving-part bboxes, used as the
    default per-bin averaging window for H. Falls back to 0.35 m (MARV's approximate track
    half-width) if no bbox is available on this robot config.
    """
    extents = []
    bb = getattr(robot_cfg, "body_bbox", None)
    if bb is not None:
        extents.append(bb[[1, 4]].abs().max().item())
    dpb = getattr(robot_cfg, "driving_part_bboxes", None)
    if dpb is not None and dpb.numel() > 0:
        extents.append(dpb[:, [1, 4]].abs().max().item())
    if not extents:
        return 0.35
    return max(extents)


def sample_terrain_points_relative(env, curr_state: PhysicsState, local_x: torch.Tensor, local_y: torch.Tensor) -> torch.Tensor:
    """Bilinearly samples ``env.terrain_cfg.z_grid`` at a SHARED (batch-independent) set of
    ``M`` local-frame ``(x, y)`` offsets, one query per robot per offset, and returns each
    sample's height RELATIVE to that robot's own chassis height (``curr_state.x[..., 2]``) --
    the raw per-point primitive Eq. 1's per-bin mean (``sample_binned_terrain_heights``) and
    ``pan_reward.PanReward``'s candidate-angle geometry (Eq. 4) both build on.

    The local frame is [L] (see module docstring): origin at the chassis xy position, X-axis
    = yaw-only forward heading (roll/pitch NOT applied), Z = world-vertical.

    Args:
        local_x, local_y: ``(M,)`` local-frame offsets in meters, shared across the batch.
    Returns:
        ``(B, M)`` relative terrain heights (meters).
    """
    device = curr_state.x.device
    B = curr_state.x.shape[0]
    M = local_x.numel()
    yaw = quaternion_to_yaw(curr_state.q)  # (B,)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    lx = local_x.to(device).view(1, M).expand(B, M)
    ly = local_y.to(device).view(1, M).expand(B, M)
    world_dx = cos_y.view(B, 1) * lx - sin_y.view(B, 1) * ly
    world_dy = sin_y.view(B, 1) * lx + cos_y.view(B, 1) * ly
    query_x = curr_state.x[:, 0:1] + world_dx
    query_y = curr_state.x[:, 1:2] + world_dy
    query = torch.stack([query_x, query_y], dim=-1)  # (B, M, 2)
    z = interpolate_grid(env.terrain_cfg.z_grid, query, env.terrain_cfg.max_coord).squeeze(-1)  # (B, M)
    return z - curr_state.x[:, 2:3]


def sample_binned_terrain_heights(
    env,
    curr_state: PhysicsState,
    bin_centers: torch.Tensor,
    bin_width: float,
    y_window: float,
    x_samples_per_bin: int,
    y_samples_per_bin: int,
) -> torch.Tensor:
    """Eq. 1's H: the per-bin MEAN of a dense sub-grid of terrain samples (see module
    docstring's "per-bin averaging" note). ``bin_centers``: ``(N,)`` local-frame x offsets.
    Returns ``(B, N)``.
    """
    device = bin_centers.device
    N = bin_centers.numel()
    sub_x = torch.linspace(-bin_width / 2, bin_width / 2, x_samples_per_bin, device=device) if x_samples_per_bin > 1 else torch.zeros(1, device=device)
    sub_y = torch.linspace(-y_window, y_window, y_samples_per_bin, device=device) if y_samples_per_bin > 1 else torch.zeros(1, device=device)
    local_x = (bin_centers.view(N, 1, 1) + sub_x.view(1, -1, 1)).expand(N, sub_x.numel(), sub_y.numel()).reshape(-1)
    local_y = sub_y.view(1, 1, -1).expand(N, sub_x.numel(), sub_y.numel()).reshape(-1)
    z = sample_terrain_points_relative(env, curr_state, local_x, local_y)  # (B, N*xs*ys)
    return z.view(z.shape[0], N, -1).mean(dim=-1)


class PanTerrainStateEncoder(ObservationEncoder):
    """Generic MLP encoder for ``PanTerrainState``, used by the repo's GENERIC policy path
    (``EncoderCombiner``, e.g. PPO or ``D3QNPolicyConfig(fig5_topology=False)``). NOT used by
    ``D3QNPolicyConfig(fig5_topology=True)``, which reads the raw ``PanTerrainState`` tensor
    directly and builds the paper's literal Fig. 5 branches itself (see ``d3qn_policy.py``).
    """

    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.mlp = MLP(**mlp_kwargs | {"in_dim": input_dim, "out_dim": output_dim, "activate_last_layer": True})

    def forward(self, x):
        return self.mlp(x)


@dataclass
class PanTerrainState(Observation):
    """AT-D3QN / ICM-D3QN paper state (Sec. III-B / Sec. 3.2): ``[H(n_heights), theta_f1,
    theta_f2, theta_R]``. See module docstring for the exact equations, frame convention, and
    sign-convention derivation.

    Args:
        n_heights: N in Eq. 1 (paper default 15).
        bin_width: d in Fig. 3/Eq. 1 (meters). NOT given a numeric value by the paper --
            see module docstring.
        y_window: half-width (meters) of the per-bin Y-averaging strip. ``None`` (default)
            derives it from the robot's own footprint bboxes (module docstring).
        x_samples_per_bin / y_samples_per_bin: sub-sampling resolution used to approximate
            Eq. 1's per-bin point-cloud mean from the continuous ground-truth heightmap.
        front_indices / rear_indices: 0-based driving-part indices for the front/rear pair.
            ``None`` auto-detects from ``robot_cfg.driving_part_names``.
    """

    supports_vecnorm = True

    n_heights: int = 15
    bin_width: float = 0.1
    y_window: float | None = None
    x_samples_per_bin: int = 3
    y_samples_per_bin: int = 3
    front_indices: tuple[int, ...] | None = None
    rear_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.n_heights < 1:
            raise ValueError(f"n_heights must be >= 1, got {self.n_heights}.")
        if self.bin_width <= 0:
            raise ValueError(f"bin_width must be > 0, got {self.bin_width}.")
        if self.x_samples_per_bin < 1 or self.y_samples_per_bin < 1:
            raise ValueError("x_samples_per_bin and y_samples_per_bin must both be >= 1.")
        if self.apply_noise:
            if not isinstance(self.noise_scale, (float, torch.Tensor)):
                raise ValueError("Noise scale must be specified if apply_noise is True and must be a float or tensor.")
            if isinstance(self.noise_scale, float):
                self.noise_scale = torch.tensor([self.noise_scale], dtype=self.env.out_dtype, device=self.env.device)
            if self.noise_scale.shape[0] not in (1, self.dim):
                raise ValueError(f"Noise scale tensor must have shape (1,) or ({self.dim},) but got {self.noise_scale.shape}.")
        self.front_idx, self.rear_idx = resolve_front_rear_indices(self.env.robot_cfg, self.front_indices, self.rear_indices)
        self.front_idx = self.front_idx.to(self.env.device)
        self.rear_idx = self.rear_idx.to(self.env.device)
        if self.y_window is None:
            self.y_window = _resolve_y_half_width(self.env.robot_cfg)
        self.bin_centers = ((torch.arange(self.n_heights, dtype=torch.float32) - (self.n_heights - 1) / 2) * self.bin_width).to(self.env.device)

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        h = sample_binned_terrain_heights(
            self.env, curr_state, self.bin_centers, self.bin_width, self.y_window, self.x_samples_per_bin, self.y_samples_per_bin
        )  # (B, n_heights)
        theta_f1 = -curr_state.thetas[:, self.front_idx].mean(dim=-1)  # sign flip, see module docstring
        theta_f2 = curr_state.thetas[:, self.rear_idx].mean(dim=-1)
        theta_r = -quaternion_to_pitch(curr_state.q)  # sign flip, see module docstring
        obs = torch.cat([h, theta_f1.unsqueeze(-1), theta_f2.unsqueeze(-1), theta_r.unsqueeze(-1)], dim=-1).to(self.env.out_dtype)
        if self.apply_noise:
            noise = torch.randn_like(obs) * self.noise_scale.view(1, -1)
            obs = obs + noise
        return obs

    def from_realistic_world(self, tensordict) -> torch.Tensor:
        """Deployment path (``flipper_policy_node`` -> ``_to_realistic_env``): compute the SAME
        ``[H, theta_f1, theta_f2, theta_R]`` from the node's raw tensordict instead of the
        engine's ground-truth ``z_grid``.

        Inputs (same conventions as ``heightmap.Heightmap.from_realistic_world``, which is the
        live-sim-tested reference for consuming the node's map):
        * ``heightmap`` (H_src, W_src) up to leading singleton dims — robot-LOCAL yaw-aligned
          frame (X forward = rows), heights RELATIVE to the robot base. That matches Eq. 1's
          [L] frame (yaw-only heading alignment, module docstring) and
          ``sample_terrain_points_relative``'s chassis-relative z exactly.
        * ``heightmap_extent`` = (x_max, y_max, x_min, y_min) in meters.
        * ``thetas`` (num_driving_parts,) raw joint angles, engine/robot sign convention —
          the same paper-vs-engine sign corrections as ``__call__`` apply (module docstring).
        * ``quat`` (4,) ROS-order (x, y, z, w). ROS REP-103 pitch (asin(2(wy - zx))) is
          positive nose-DOWN about +Y — the same convention the module docstring derives for
          the engine's ``quaternion_to_pitch`` — so the same ``theta_R = -pitch`` flip applies.
        """
        hm: torch.Tensor = tensordict["heightmap"].to(self.env.device)
        while hm.ndim > 2:
            hm = hm.squeeze(0)
        extent = tensordict["heightmap_extent"]
        if isinstance(extent, torch.Tensor):
            extent = extent.cpu().squeeze().tolist()
        need_x = float(self.bin_centers.abs().max()) + self.bin_width / 2
        if extent[0] < need_x or extent[1] < self.y_window or extent[2] > -need_x or extent[3] > -self.y_window:
            raise ValueError(
                f"Real-world heightmap extent {extent} does not cover the Eq.-1 sampling window "
                f"(x +-{need_x:.2f} m, y +-{self.y_window:.2f} m)."
            )
        # dense per-bin subgrid in the local frame (same layout as sample_binned_terrain_heights)
        dev = hm.device
        sub_x = torch.linspace(-self.bin_width / 2, self.bin_width / 2, self.x_samples_per_bin, device=dev) if self.x_samples_per_bin > 1 else torch.zeros(1, device=dev)
        sub_y = torch.linspace(-self.y_window, self.y_window, self.y_samples_per_bin, device=dev) if self.y_samples_per_bin > 1 else torch.zeros(1, device=dev)
        bc = self.bin_centers.to(dev)
        px = (bc.view(-1, 1, 1) + sub_x.view(1, -1, 1)).expand(self.n_heights, sub_x.numel(), sub_y.numel())
        py = sub_y.view(1, 1, -1).expand_as(px)
        # local (x, y) -> normalized grid_sample coords (heightmap.py's mapping)
        grid_v = 2 * (px - extent[0]) / (extent[2] - extent[0]) - 1
        grid_u = 2 * (py - extent[1]) / (extent[3] - extent[1]) - 1
        grid = torch.stack((grid_u, grid_v), dim=-1).reshape(1, self.n_heights, -1, 2)
        z = torch.nn.functional.grid_sample(hm.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        h = z.view(self.n_heights, -1).mean(dim=-1).view(1, self.n_heights)  # Eq. 1 per-bin mean
        # E (Eq. 2) with the module-docstring sign corrections
        thetas = tensordict["thetas"].to(self.env.device).view(-1)
        theta_f1 = -thetas[self.front_idx].mean().view(1, 1)
        theta_f2 = thetas[self.rear_idx].mean().view(1, 1)
        q = tensordict["quat"].to(self.env.device).view(-1)  # ROS (x, y, z, w)
        sin_p = torch.clamp(2 * (q[3] * q[1] - q[2] * q[0]), -1.0, 1.0)
        theta_r = -torch.asin(sin_p).view(1, 1)  # positive = nose-UP (paper convention)
        return torch.cat([h, theta_f1, theta_f2, theta_r], dim=-1).to(self.env.out_dtype)

    def viz_geometry(self) -> dict:
        """Read-only introspection for deployment visualization (plain Python floats/ints
        only -- no torch tensors, no ROS): the Eq.-1 bin layout this instance samples.
        Consumed by the ROS deploy node's debug-marker publisher (``ros2/obs_viz.py``) to
        render the 15-bin terrain profile along the robot heading. Adds NO new math; bin
        order matches the leading ``n_heights`` entries of ``__call__`` /
        ``from_realistic_world`` output (followed by theta_f1, theta_f2, theta_R).
        """
        return {
            "bin_centers": [float(c) for c in self.bin_centers.detach().cpu().tolist()],  # local-frame x offsets (m)
            "bin_width": float(self.bin_width),
            "y_window": float(self.y_window),
            "n_heights": int(self.n_heights),
        }

    @property
    def dim(self) -> int:
        return self.n_heights + 3

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> PanTerrainStateEncoder:
        return PanTerrainStateEncoder(
            input_dim=self.dim,
            **self.encoder_opts,
        )
