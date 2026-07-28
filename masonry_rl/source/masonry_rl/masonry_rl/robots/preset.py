"""Robot-agnostic preset interface.

The masonry environment never names a robot. It asks a :class:`RobotPreset` for
joint groups, end-effector frames and body names, all addressed by *regex over
joint names* rather than by index, so nothing here depends on the DoF count.
That is what makes swapping DRC Atlas (~30 DoF) for Unitree G1 (29 DoF) - or for
a future electric Atlas (56 DoF) - a one-line config change.

The dataclass itself imports nothing from Isaac Lab so it can be inspected and
unit tested without a simulator. Only :meth:`RobotPreset.articulation_cfg`
touches ``isaaclab``, and it imports lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from isaaclab.assets import ArticulationCfg

__all__ = ["ActuatorGroup", "RobotPreset"]


@dataclass
class ActuatorGroup:
    """Implicit actuator settings for one group of joints."""

    joint_expr: list[str]
    """Regex patterns matching joint names."""
    stiffness: float
    damping: float
    effort_limit: float | None = None
    velocity_limit: float | None = None


@dataclass
class RobotPreset:
    """Everything the masonry task needs to know about a robot."""

    name: str

    # -- asset source -------------------------------------------------------
    usd_path: str | None = None
    """Pre-converted USD. Takes precedence over ``urdf_path`` when set."""
    urdf_path: str | None = None
    """Source URDF; converted by Isaac Lab's URDF importer during P0."""

    # -- joint groups (regex over joint names) ------------------------------
    joint_groups: dict[str, list[str]] = field(default_factory=dict)
    """Named groups, e.g. ``right_arm``, ``left_arm``, ``torso``, ``legs``,
    ``right_gripper``. The environment only ever refers to these names."""

    # -- frames and bodies --------------------------------------------------
    ee_frames: dict[str, str] = field(default_factory=dict)
    """Body name per end effector, e.g. ``{"right": "r_hand", "left": "l_hand"}``."""
    pelvis_body: str = ""
    """Floating-base body used for balance observations."""
    foot_bodies: list[str] = field(default_factory=list)
    contact_bodies: list[str] = field(default_factory=list)
    """Bodies that need a contact sensor (feet, fingers, wrists)."""

    # -- kinematics / posture ----------------------------------------------
    nominal_pelvis_height: float = 0.0
    """Standing pelvis height [m]; the balance policy's neutral command."""
    arm_reach: float = 0.0
    """Approximate single-arm reach [m]; used to sanity check wall placement."""
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    """Regex -> angle [rad]. Unmatched joints default to 0."""

    # -- actuation ----------------------------------------------------------
    actuators: dict[str, ActuatorGroup] = field(default_factory=dict)

    # -- gripper ------------------------------------------------------------
    gripper_open: float = 0.0
    gripper_closed: float = 0.0
    """Joint targets [m or rad] for the binary gripper action."""

    # -- pelvis command limits (the 4-D action into the balance policy) ------
    pelvis_height_range: tuple[float, float] = (0.0, 0.0)
    pelvis_pitch_range: tuple[float, float] = (0.0, 0.0)
    pelvis_shift_range: tuple[float, float] = (0.0, 0.0)
    """Fore/aft and lateral CoM shift limits [m]."""

    # -- validation ---------------------------------------------------------

    REQUIRED_GROUPS = ("right_arm", "right_gripper", "torso", "legs")

    def validate(self) -> None:
        """Fail loudly on an incomplete preset rather than deep inside the sim."""
        if not self.usd_path and not self.urdf_path:
            raise ValueError(f"{self.name}: needs usd_path or urdf_path")
        missing = [g for g in self.REQUIRED_GROUPS if g not in self.joint_groups]
        if missing:
            raise ValueError(f"{self.name}: missing joint groups {missing}")
        if "right" not in self.ee_frames:
            raise ValueError(f"{self.name}: ee_frames must define 'right'")
        if not self.pelvis_body:
            raise ValueError(f"{self.name}: pelvis_body is required")
        if not self.foot_bodies:
            raise ValueError(f"{self.name}: foot_bodies is required")
        if self.nominal_pelvis_height <= 0.0:
            raise ValueError(f"{self.name}: nominal_pelvis_height must be > 0")

    @property
    def is_bimanual(self) -> bool:
        return "left_arm" in self.joint_groups and "left" in self.ee_frames

    def group(self, name: str) -> list[str]:
        try:
            return self.joint_groups[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"{self.name}: no joint group {name!r}") from exc

    # -- Isaac Lab bridge ---------------------------------------------------

    def articulation_cfg(self, prim_path: str = "{ENV_REGEX_NS}/Robot", **kwargs: Any) -> "ArticulationCfg":
        """Build the Isaac Lab ``ArticulationCfg`` for this robot.

        Imports ``isaaclab`` lazily so the rest of this package stays importable
        on a machine without Isaac Sim.
        """
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.sim import UsdFileCfg

        self.validate()
        if not self.usd_path:
            raise RuntimeError(
                f"{self.name}: urdf_path must be converted to USD during P0; "
                "run scripts/convert_asset.py and set usd_path"
            )

        actuators = {
            key: ImplicitActuatorCfg(
                joint_names_expr=grp.joint_expr,
                stiffness=grp.stiffness,
                damping=grp.damping,
                effort_limit=grp.effort_limit,
                velocity_limit=grp.velocity_limit,
            )
            for key, grp in self.actuators.items()
        }
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=UsdFileCfg(usd_path=self.usd_path),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.nominal_pelvis_height),
                joint_pos=dict(self.default_joint_pos),
            ),
            actuators=actuators,
            **kwargs,
        )
