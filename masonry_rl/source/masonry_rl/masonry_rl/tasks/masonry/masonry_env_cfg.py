"""Environment configuration for the masonry task.

This file is the single place where the plan's numbers become code. It is
deliberately robot-agnostic: it takes a :class:`RobotPreset` and addresses
joints through that preset's named groups, never by index or by a hard-coded
robot name.

STATUS (P2 scaffold): scene entities, term wiring and physics settings are
declared; the manager term bodies live in ``mdp/`` and are still TODO. Field
names must be checked against the pinned Isaac Lab version in P0 - the manager
API moves between releases.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from ...robots import ATLAS_DRC_PRESET, RobotPreset
from .mortar import MortarCfg
from .wall_planner import WallSpec


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@configclass
class MasonrySimCfg:
    """Physics settings.

    The GPU buffer sizes are not optional. A stack of bricks plus bipedal
    contact overflows the PhysX defaults and the run dies with a
    ``PxgDynamicsMemoryConfig`` error partway through training - usually hours
    in, on the first environment that reaches a tall course.
    """

    dt: float = 1.0 / 120.0
    """Large timesteps make brick stacks explode. Do not raise this."""
    render_interval: int = 4

    solver_position_iteration_count: int = 24
    solver_velocity_iteration_count: int = 1

    # PhysX GPU buffers - see section 8.2 of the plan
    gpu_max_rigid_contact_count: int = 2**23
    gpu_max_rigid_patch_count: int = 2**20
    gpu_found_lost_pairs_capacity: int = 2**22
    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    gpu_total_aggregate_pairs_capacity: int = 2**22
    gpu_collision_stack_size: int = 2**28
    gpu_heap_capacity: int = 2**26
    gpu_temp_buffer_capacity: int = 2**24
    gpu_max_num_partitions: int = 8

    brick_static_friction: float = 0.9
    brick_dynamic_friction: float = 0.8
    brick_restitution: float = 0.0
    foot_static_friction: float = 1.0
    foot_dynamic_friction: float = 0.9
    contact_offset: float = 0.002
    rest_offset: float = 0.0005
    max_depenetration_velocity: float = 1.0
    enable_ccd: bool = False


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------


@configclass
class MasonryRewardWeights:
    """Section 5.3 of the plan.

    The two dominant terms are load-bearing:
    ``release_stable`` (+30) defines what a successful placement means, and
    ``fall`` (-200) sits an order of magnitude above everything else so that
    falling is never worth one more brick.
    """

    reach: float = 1.0
    grasp: float = 2.0
    lift: float = 5.0
    mortar_bed: float = 8.0
    align: float = 8.0
    seat_force: float = 6.0
    joint_thickness: float = 15.0
    release_stable: float = 30.0

    fall: float = -200.0
    collapse: float = -50.0
    com_margin: float = -5.0
    self_collision: float = -5.0
    squeeze_out: float = -5.0
    foot_slip: float = -2.0
    posture_deviation: float = -0.5
    excess_force: float = -0.1
    action_rate: float = -0.01
    joint_vel: float = -0.01


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@configclass
class MasonryEnvCfg:
    """Top-level config. Subclassed per phase (see ``__init__.py``)."""

    robot: RobotPreset = ATLAS_DRC_PRESET
    """Swap to ``G1_PRESET`` to run the whole pipeline without the Atlas asset."""

    wall: WallSpec = WallSpec()
    mortar: MortarCfg = MortarCfg()
    sim: MasonrySimCfg = MasonrySimCfg()
    rewards: MasonryRewardWeights = MasonryRewardWeights()

    # -- scene --------------------------------------------------------------
    num_envs: int = 512
    env_spacing: float = 4.0
    pallet_origin: tuple[float, float, float] = (0.30, -0.55, 0.0)
    """Front-right of the robot: the right arm handles bricks, and this keeps
    the pallet clear of the wall's lateral span."""
    num_source_bricks: int = 16

    # -- episode ------------------------------------------------------------
    decimation: int = 4
    seconds_per_brick: float = 15.0
    """Longer than a fixed-base arm needs: mortar application and posture
    changes both cost time."""

    @property
    def episode_length_s(self) -> float:
        return self.seconds_per_brick * self.wall.num_slots

    # -- staged capability ---------------------------------------------------
    fix_pelvis: bool = True
    """P3: weld the pelvis and learn manipulation alone. Released at L4."""
    enable_pelvis_action: bool = False
    """P5: 4-D pelvis command into the frozen balance policy."""
    enable_left_arm: bool = False
    """P6: bimanual, left arm carries the mortar nozzle."""
    learn_extrusion: bool = False
    """P6: promote mortar application from scripted trajectory to policy action."""
    balance_checkpoint: str = MISSING
    """Frozen balance policy from P4. Required once ``enable_pelvis_action``."""

    # -- tolerances ----------------------------------------------------------
    position_tolerance: float = 0.005
    yaw_tolerance: float = 0.035  # ~2 degrees
    stable_steps: int = 30

    def __post_init__(self) -> None:
        self.robot.validate()
        self._check_reachability()

    def _check_reachability(self) -> None:
        """Fail early if any slot or the pallet is outside the robot's reach.

        Cheap, and it catches the mistake that otherwise shows up as a policy
        that plateaus at a low success rate for no visible reason. The check
        uses the planner's actual slot positions rather than assuming the wall
        runs along one axis, so it stays correct for any ``yaw``.
        """
        from .wall_planner import WallPlanner

        planner = WallPlanner(self.wall, num_envs=1)
        slot_pos, _ = planner.all_slot_poses()
        worst = float(slot_pos[:, :2].norm(dim=-1).max())

        px, py = self.pallet_origin[0], self.pallet_origin[1]
        checks = (("furthest slot", worst), ("pallet", (px**2 + py**2) ** 0.5))
        for name, distance in checks:
            if distance > self.robot.arm_reach:
                raise ValueError(
                    f"{name} at {distance:.2f} m exceeds {self.robot.name} reach "
                    f"({self.robot.arm_reach:.2f} m). The robot is stationary, so "
                    "shorten the wall, lay it laterally, or move it closer."
                )
