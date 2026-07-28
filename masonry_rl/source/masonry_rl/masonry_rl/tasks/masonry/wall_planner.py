"""Deterministic wall/slot planner for the masonry task.

This module is intentionally free of any Isaac Sim / Isaac Lab dependency: it is
pure geometry on top of ``torch`` tensors, so it can be unit tested on a CPU-only
machine without a simulator.

Layer 3 of the three-layer decomposition (see docs/isaaclab-masonry-rl-plan.md):
the *order* in which bricks are laid is not learned. This planner turns a
``WallSpec`` into target slot poses, and also emits the poses of the pre-built
(``base_courses``) bricks that are spawned as kinematic bodies.

Frames
------
All poses are produced in the *wall frame* first (origin at the left end of the
wall, on the floor, x along the wall, z up), then transformed into the
environment frame by ``spec.origin`` and ``spec.yaw``.

Heights (with the default spec)::

    course pitch = brick_height + joint = 0.057 + 0.010 = 0.067
    bed base   of course c = c * course_pitch                 # mortar is laid here
    brick bottom of course c = joint + c * course_pitch       # "work plane"
    brick top    of course c = joint + c * course_pitch + brick_height

So ``base_courses=3`` gives a work plane at 0.010 + 3*0.067 = 0.211 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["WallSpec", "WallPlanner"]


@dataclass
class WallSpec:
    """Geometry of the wall to be built."""

    # brick geometry [m] - Korean standard brick
    brick_length: float = 0.190
    brick_width: float = 0.090
    brick_height: float = 0.057
    # nominal mortar joint thickness [m]
    joint_thickness: float = 0.010

    # layout
    bricks_per_course: int = 4
    num_courses: int = 3
    """Number of courses the *policy* builds."""
    base_courses: int = 0
    """Number of pre-built courses spawned as kinematic bodies underneath.

    This is the single knob that drives both the height curriculum
    (12 -> 6 -> 3 -> 0) and the fallback demo condition (freeze at 3).
    """
    bond: str = "running"
    """``"running"`` staggers odd courses by half a column pitch; ``"stack"``
    aligns every course."""

    # placement of the wall in the environment frame
    origin: tuple[float, float, float] = (0.45, -0.395, 0.0)
    """Left end of the wall, on the floor, in the environment frame."""
    yaw: float = math.pi / 2
    """Rotation of the wall about +z [rad].

    The default lays the wall *across* the robot's frontal plane rather than
    running away from it. The robot is stationary, so a wall extending along +x
    puts its far end ~1.4 m out - well beyond any humanoid's reach. Laid
    laterally and centred, both ends sit at ~0.6 m. ``scripts/preview_wall.py``
    checks this.
    """

    def __post_init__(self) -> None:
        if self.bond not in ("running", "stack"):
            raise ValueError(f"unknown bond pattern: {self.bond!r}")
        if self.bricks_per_course < 1 or self.num_courses < 1:
            raise ValueError("bricks_per_course and num_courses must be >= 1")
        if self.base_courses < 0:
            raise ValueError("base_courses must be >= 0")

    # -- derived geometry ---------------------------------------------------

    @property
    def course_pitch(self) -> float:
        """Vertical distance between two consecutive brick bottoms."""
        return self.brick_height + self.joint_thickness

    @property
    def column_pitch(self) -> float:
        """Horizontal distance between two consecutive brick centers."""
        return self.brick_length + self.joint_thickness

    @property
    def wall_length(self) -> float:
        return (
            self.bricks_per_course * self.brick_length
            + (self.bricks_per_course - 1) * self.joint_thickness
        )

    @property
    def num_slots(self) -> int:
        """Number of bricks the policy has to place."""
        return self.bricks_per_course * self.num_courses

    @property
    def num_base_bricks(self) -> int:
        return self.bricks_per_course * self.base_courses

    @property
    def total_courses(self) -> int:
        return self.base_courses + self.num_courses

    def bed_base_height(self, course: int) -> float:
        """Height of the surface the mortar bed for ``course`` is laid on."""
        return course * self.course_pitch

    def work_plane_height(self, course: int | None = None) -> float:
        """Height of the *bottom face* of a brick in ``course``.

        With ``course=None`` this returns the work plane of the first course the
        policy has to build, i.e. the height the curriculum is actually tuning.
        """
        if course is None:
            course = self.base_courses
        return self.joint_thickness + course * self.course_pitch

    def top_height(self) -> float:
        """Height of the top face of the finished wall."""
        return self.work_plane_height(self.total_courses - 1) + self.brick_height

    def course_x_offset(self, course: int) -> float:
        if self.bond == "running" and course % 2 == 1:
            return 0.5 * self.column_pitch
        return 0.0


class WallPlanner:
    """Turns a :class:`WallSpec` into batched slot poses.

    All returned tensors live on ``device``. Poses are ``(position, quaternion)``
    with quaternions in Isaac Lab's ``(w, x, y, z)`` convention.
    """

    def __init__(
        self,
        spec: WallSpec,
        num_envs: int,
        device: str | torch.device = "cpu",
        env_origins: torch.Tensor | None = None,
    ) -> None:
        self.spec = spec
        self.num_envs = num_envs
        self.device = torch.device(device)
        # per-environment offset (Isaac Lab's ``scene.env_origins``); zero if the
        # caller already works in per-environment local coordinates.
        if env_origins is None:
            env_origins = torch.zeros(num_envs, 3, device=self.device)
        self.env_origins = env_origins.to(self.device)

        self._slot_pos_local, self._slot_course = self._build_slot_table()
        self._base_pos_local = self._build_base_table()
        self._quat_local = self._yaw_quat(spec.yaw)

    # -- construction -------------------------------------------------------

    def _wall_to_env(self, xyz: torch.Tensor) -> torch.Tensor:
        """Rotate by ``spec.yaw`` about +z and translate by ``spec.origin``."""
        c, s = math.cos(self.spec.yaw), math.sin(self.spec.yaw)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        out = torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)
        origin = torch.tensor(self.spec.origin, device=self.device, dtype=out.dtype)
        return out + origin

    def _brick_center_local(self, course: int, column: int) -> tuple[float, float, float]:
        sp = self.spec
        x = column * sp.column_pitch + 0.5 * sp.brick_length + sp.course_x_offset(course)
        y = 0.5 * sp.brick_width
        z = sp.work_plane_height(course) + 0.5 * sp.brick_height
        return x, y, z

    def _build_slot_table(self) -> tuple[torch.Tensor, torch.Tensor]:
        sp = self.spec
        centers, courses = [], []
        for k in range(sp.num_slots):
            course = sp.base_courses + k // sp.bricks_per_course
            column = k % sp.bricks_per_course
            centers.append(self._brick_center_local(course, column))
            courses.append(course)
        pos = torch.tensor(centers, device=self.device, dtype=torch.float32)
        return self._wall_to_env(pos), torch.tensor(courses, device=self.device, dtype=torch.long)

    def _build_base_table(self) -> torch.Tensor:
        sp = self.spec
        centers = [
            self._brick_center_local(course, column)
            for course in range(sp.base_courses)
            for column in range(sp.bricks_per_course)
        ]
        if not centers:
            return torch.zeros(0, 3, device=self.device)
        pos = torch.tensor(centers, device=self.device, dtype=torch.float32)
        return self._wall_to_env(pos)

    def _yaw_quat(self, yaw: float) -> torch.Tensor:
        return torch.tensor(
            [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
            device=self.device,
            dtype=torch.float32,
        )

    # -- queries ------------------------------------------------------------

    @property
    def num_slots(self) -> int:
        return self.spec.num_slots

    def slot_pose(self, slot_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Target pose of the brick for ``slot_idx`` (one index per environment).

        Returns ``(pos[num_envs, 3], quat[num_envs, 4])`` in world coordinates.
        """
        slot_idx = slot_idx.to(self.device).long()
        pos = self._slot_pos_local[slot_idx] + self.env_origins
        quat = self._quat_local.expand(slot_idx.shape[0], 4)
        return pos, quat

    def slot_course(self, slot_idx: torch.Tensor) -> torch.Tensor:
        return self._slot_course[slot_idx.to(self.device).long()]

    def all_slot_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(pos[num_slots, 3], quat[num_slots, 4])`` for a single environment."""
        quat = self._quat_local.expand(self.num_slots, 4)
        return self._slot_pos_local.clone(), quat

    def base_brick_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Poses of the pre-built (kinematic) bricks, single environment."""
        n = self._base_pos_local.shape[0]
        return self._base_pos_local.clone(), self._quat_local.expand(n, 4)

    # -- mortar support -----------------------------------------------------

    def slot_cell_positions(self, slot_idx: torch.Tensor, num_cells: int) -> torch.Tensor:
        """World positions of the mortar bed sample cells for ``slot_idx``.

        Cells are spread along the brick's length on the *bed base* plane (the
        surface the mortar is extruded onto), returning
        ``[num_envs, num_cells, 3]``.
        """
        sp = self.spec
        slot_idx = slot_idx.to(self.device).long()
        pos, _ = self.slot_pose(slot_idx)
        course = self.slot_course(slot_idx)

        # local offsets along the wall's x axis, cell centers
        k = torch.arange(num_cells, device=self.device, dtype=torch.float32)
        du = (k + 0.5) / num_cells - 0.5  # in [-0.5, 0.5)
        dx = du * sp.brick_length
        c, s = math.cos(sp.yaw), math.sin(sp.yaw)
        offs = torch.stack([dx * c, dx * s, torch.zeros_like(dx)], dim=-1)  # [C, 3]

        cells = pos.unsqueeze(1) + offs.unsqueeze(0)  # [E, C, 3]
        bed_z = course.to(torch.float32) * sp.course_pitch
        cells[..., 2] = bed_z.unsqueeze(1) + self.env_origins[:, 2].unsqueeze(1)
        return cells

    def bed_base_height(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """World height of the mortar bed base plane for ``slot_idx``."""
        course = self.slot_course(slot_idx).to(torch.float32)
        return course * self.spec.course_pitch + self.env_origins[:, 2]
