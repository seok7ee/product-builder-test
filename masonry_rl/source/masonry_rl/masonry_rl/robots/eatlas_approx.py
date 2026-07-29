"""Electric Atlas *approximation* - an unofficial model built from published specs.

WHAT THIS IS NOT
----------------
This is **not** a Boston Dynamics asset and is not derived from their CAD,
meshes or any proprietary data. Boston Dynamics has not released a URDF or USD
for the electric Atlas. This module generates a parametric humanoid whose gross
dimensions, mass, degree-of-freedom budget and joint ranges match the figures
Boston Dynamics and the trade press have published, so that the masonry
pipeline has something Atlas-shaped to develop against before a real asset
exists.

Fidelity, honestly:

* link shapes are boxes and cylinders - it will look blocky, not like Atlas
* segment lengths come from Winter's anthropometric table scaled to the
  published height, then the arms are stretched to hit the published span
* segment masses come from Winter's mass fractions, normalised to the published
  total. A real robot carries battery and compute in the torso, so the true
  distribution is more torso-heavy than this
* joint limits are human ranges, widened to continuous where Boston Dynamics
  has stated 360-degree motion (hip, waist, neck)
* actuator dynamics, joint friction and link inertia detail are estimates

Use it for workspace, reachability, balance and contact bring-up. Do not use it
for anything that depends on true Atlas dynamics, and do not present results
from it as Atlas results.

PUBLISHED SPECS USED (CES 2026 production Atlas)
-----------------------------------------------
Sources disagree; these are the values that recur most consistently across
reporting of the CES 2026 production machine. The conflicts are recorded in
``docs/asset-notes.md``.

===================  ==========================================
height               1.9 m
mass                 90 kg
degrees of freedom   56
reach (span)         2.3 m  <- fingertip to fingertip, NOT single-arm reach
arms                 7 DoF each, five-fingered hands
payload              30 kg sustained, 50 kg burst
joint range          360 deg at hip, waist and neck
actuators            custom electric, high torque density
===================  ==========================================

DoF budget summing to 56::

    legs   2 x 6 = 12    hip yaw/roll/pitch, knee, ankle pitch/roll
    waist        =  3    yaw (continuous), roll, pitch
    neck         =  3    yaw (continuous), pitch, roll
    arms   2 x 7 = 14    shoulder p/r/y, elbow, wrist y/p/r
    hands  2 x12 = 24    thumb 3, index 3, middle/ring/pinky 2 each
                   --
                   56

For masonry we do not want 24 finger joints, so ``hand="parallel"`` swaps both
hands for a single-DoF gripper, giving a 34-DoF model. ``hand="five_finger"``
gives the full 56.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .preset import ActuatorGroup, RobotPreset

__all__ = ["EAtlasSpec", "build_urdf", "link_frames", "EATLAS_APPROX_PRESET"]


# Winter, "Biomechanics and Motor Control of Human Movement": segment lengths as
# a fraction of stature, and segment masses as a fraction of body mass.
_LEN = {
    "shoulder_z": 0.818,
    "hip_z": 0.530,
    "shoulder_width": 0.259,
    "hip_width": 0.191,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
    "thigh": 0.245,
    "shank": 0.246,
    "foot_len": 0.152,
}
_MASS = {
    "head": 0.081,
    "torso": 0.355,
    "pelvis": 0.142,
    "upper_arm": 0.028,
    "forearm": 0.016,
    "hand": 0.006,
    "thigh": 0.100,
    "shank": 0.0465,
    "foot": 0.0145,
}

_CONTINUOUS = math.inf

# Vertical offsets inside the trunk. They must be subtracted from the shoulder
# and neck origins, otherwise every landmark above the waist drifts upward by
# the same amount and the published shoulder height is quietly wrong.
_WAIST_OFFSET = 0.03
_NECK_GAP = 0.04


@dataclass
class EAtlasSpec:
    """Published-spec inputs. Everything else is derived."""

    height: float = 1.90
    mass: float = 90.0
    span: float = 2.30
    """Fingertip-to-fingertip, the published '2.3 m reach'. Single-arm reach is
    derived from this and is roughly 0.9 m - a distinction that matters,
    because the masonry reachability check uses single-arm reach."""
    hand: str = "parallel"
    """``"parallel"`` (1 DoF gripper, 34 DoF total) or ``"five_finger"`` (56)."""
    dummy_link_mass: float = 0.01
    """Massless intermediate links upset some importers; give them a token mass."""

    def __post_init__(self) -> None:
        if self.hand not in ("parallel", "five_finger"):
            raise ValueError(f"unknown hand type: {self.hand!r}")
        for name, value in (("height", self.height), ("mass", self.mass), ("span", self.span)):
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.span <= _LEN["shoulder_width"] * self.height:
            raise ValueError("span must exceed shoulder width")

    # -- derived geometry ---------------------------------------------------

    @property
    def hip_z(self) -> float:
        return _LEN["hip_z"] * self.height

    @property
    def shoulder_z(self) -> float:
        return _LEN["shoulder_z"] * self.height

    @property
    def shoulder_width(self) -> float:
        return _LEN["shoulder_width"] * self.height

    @property
    def arm_scale(self) -> float:
        """Atlas has longer arms than human proportion; stretch to hit the span."""
        human = (_LEN["upper_arm"] + _LEN["forearm"] + _LEN["hand"]) * self.height
        return (self.span - self.shoulder_width) / 2.0 / human

    @property
    def upper_arm(self) -> float:
        return _LEN["upper_arm"] * self.height * self.arm_scale

    @property
    def forearm(self) -> float:
        return _LEN["forearm"] * self.height * self.arm_scale

    @property
    def hand_len(self) -> float:
        return _LEN["hand"] * self.height * self.arm_scale

    @property
    def arm_total(self) -> float:
        """Single-arm reach from the shoulder. This is the number the masonry
        reachability check wants, not the 2.3 m span."""
        return self.upper_arm + self.forearm + self.hand_len

    @property
    def thigh(self) -> float:
        return _LEN["thigh"] * self.height

    @property
    def shank(self) -> float:
        return _LEN["shank"] * self.height

    @property
    def foot_height(self) -> float:
        return self.hip_z - self.thigh - self.shank

    @property
    def torso_len(self) -> float:
        return self.shoulder_z - self.hip_z


# ---------------------------------------------------------------------------
# Model description
# ---------------------------------------------------------------------------


@dataclass
class _Link:
    name: str
    geom: str = "none"  # "box" | "cylinder" | "none"
    size: tuple[float, ...] = ()
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 0.0


@dataclass
class _Joint:
    name: str
    parent: str
    child: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float = 0.0
    upper: float = 0.0
    jtype: str = "revolute"
    effort: float = 200.0
    velocity: float = 10.0

    @property
    def is_continuous(self) -> bool:
        return self.jtype == "continuous"


def _chain(prefix: str, parent: str, origin, axes, limits, effort) -> tuple[list, list]:
    """A multi-DoF joint as a chain of single-DoF joints with dummy links."""
    links: list[_Link] = []
    joints: list[_Joint] = []
    current = parent
    for i, (suffix, axis, (lo, hi)) in enumerate(zip(axes[0], axes[1], limits)):
        child = f"{prefix}_{suffix}_link"
        links.append(_Link(child))
        jtype = "continuous" if math.isinf(hi) else "revolute"
        joints.append(
            _Joint(
                name=f"{prefix}_{suffix}_joint",
                parent=current,
                child=child,
                origin=origin if i == 0 else (0.0, 0.0, 0.0),
                axis=axis,
                lower=0.0 if jtype == "continuous" else lo,
                upper=0.0 if jtype == "continuous" else hi,
                jtype=jtype,
                effort=effort,
            )
        )
        current = child
    return links, joints


X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def build_model(spec: EAtlasSpec) -> tuple[list[_Link], list[_Joint]]:
    """Return the link and joint description at zero configuration.

    All joint origins are pure translations, so forward kinematics at zero
    configuration is a plain translation sum - which keeps both the model and
    its validation readable.
    """
    s = spec
    links: list[_Link] = []
    joints: list[_Joint] = []

    def add(l=None, j=None):
        if l:
            links.extend(l if isinstance(l, list) else [l])
        if j:
            joints.extend(j if isinstance(j, list) else [j])

    # -- pelvis (root) ------------------------------------------------------
    hip_half = _LEN["hip_width"] * s.height / 2
    add(_Link("pelvis", "box", (0.22, hip_half * 2, 0.16), (0, 0, 0)))

    # -- waist: yaw (continuous) / roll / pitch -----------------------------
    l, j = _chain(
        "waist",
        "pelvis",
        (0.0, 0.0, _WAIST_OFFSET),
        (("yaw", "roll", "pitch"), (Z, X, Y)),
        [(0, _CONTINUOUS), (-0.6, 0.6), (-0.7, 1.0)],
        effort=600.0,
    )
    add(l, j)
    torso_origin_z = 0.0
    trunk = s.torso_len - _WAIST_OFFSET  # waist joint to shoulder line
    add(_Link("torso", "box", (0.24, 0.32, trunk * 0.92), (0, 0, trunk / 2)))
    add(j=_Joint("torso_fixed", "waist_pitch_link", "torso", (0, 0, torso_origin_z), Z, jtype="fixed"))

    # -- neck: yaw (continuous) / pitch / roll ------------------------------
    head_h = 0.22
    l, j = _chain(
        "neck",
        "torso",
        (0.0, 0.0, trunk + _NECK_GAP),
        (("yaw", "pitch", "roll"), (Z, Y, X)),
        [(0, _CONTINUOUS), (-0.7, 0.5), (-0.5, 0.5)],
        effort=40.0,
    )
    add(l, j)
    head_center = s.height - head_h / 2 - (s.hip_z + s.torso_len + _NECK_GAP)
    add(_Link("head", "box", (0.18, 0.16, head_h), (0, 0, head_center), 0.0))
    add(j=_Joint("head_fixed", "neck_roll_link", "head", (0, 0, 0), Z, jtype="fixed"))

    # -- arms: 7 DoF each ---------------------------------------------------
    for side, sign in (("l", 1.0), ("r", -1.0)):
        l, j = _chain(
            f"{side}_shoulder",
            "torso",
            (0.0, sign * s.shoulder_width / 2, trunk),
            (("pitch", "roll", "yaw"), (Y, X, Z)),
            [(-3.1, 1.6), (-1.6, 1.6) if sign > 0 else (-1.6, 1.6), (-2.6, 2.6)],
            effort=200.0,
        )
        add(l, j)
        add(_Link(f"{side}_upper_arm", "cylinder", (0.055, s.upper_arm), (0, 0, -s.upper_arm / 2)))
        add(j=_Joint(f"{side}_upper_arm_fixed", f"{side}_shoulder_yaw_link",
                     f"{side}_upper_arm", (0, 0, 0), Z, jtype="fixed"))

        add(j=_Joint(f"{side}_elbow_pitch_joint", f"{side}_upper_arm", f"{side}_forearm",
                     (0, 0, -s.upper_arm), Y, lower=-2.6, upper=0.0, effort=150.0))
        add(_Link(f"{side}_forearm", "cylinder", (0.048, s.forearm), (0, 0, -s.forearm / 2)))

        l, j = _chain(
            f"{side}_wrist",
            f"{side}_forearm",
            (0.0, 0.0, -s.forearm),
            (("yaw", "pitch", "roll"), (Z, Y, X)),
            [(-2.0, 2.0), (-1.2, 1.2), (-1.5, 1.5)],
            effort=60.0,
        )
        add(l, j)
        add(_Link(f"{side}_hand", "box", (0.05, 0.09, s.hand_len), (0, 0, -s.hand_len / 2)))
        add(j=_Joint(f"{side}_hand_fixed", f"{side}_wrist_roll_link", f"{side}_hand",
                     (0, 0, 0), Z, jtype="fixed"))
        add(l=_hand_links(s, side), j=_hand_joints(s, side))

    # -- legs: 6 DoF each ---------------------------------------------------
    for side, sign in (("l", 1.0), ("r", -1.0)):
        l, j = _chain(
            f"{side}_hip",
            "pelvis",
            (0.0, sign * hip_half, 0.0),  # hip_z IS the hip axis height
            (("yaw", "roll", "pitch"), (Z, X, Y)),
            [(0, _CONTINUOUS), (-0.6, 0.6), (-2.0, 1.0)],
            effort=800.0,
        )
        add(l, j)
        add(_Link(f"{side}_thigh", "cylinder", (0.075, s.thigh), (0, 0, -s.thigh / 2)))
        add(j=_Joint(f"{side}_thigh_fixed", f"{side}_hip_pitch_link", f"{side}_thigh",
                     (0, 0, 0), Z, jtype="fixed"))

        add(j=_Joint(f"{side}_knee_pitch_joint", f"{side}_thigh", f"{side}_shank",
                     (0, 0, -s.thigh), Y, lower=0.0, upper=2.4, effort=800.0))
        add(_Link(f"{side}_shank", "cylinder", (0.060, s.shank), (0, 0, -s.shank / 2)))

        l, j = _chain(
            f"{side}_ankle",
            f"{side}_shank",
            (0.0, 0.0, -s.shank),
            (("pitch", "roll"), (Y, X)),
            [(-0.9, 0.6), (-0.5, 0.5)],
            effort=300.0,
        )
        add(l, j)
        foot_len = _LEN["foot_len"] * s.height
        add(_Link(f"{side}_foot", "box", (foot_len, 0.11, s.foot_height),
                  (foot_len * 0.18, 0, -s.foot_height / 2)))
        add(j=_Joint(f"{side}_foot_fixed", f"{side}_ankle_roll_link", f"{side}_foot",
                     (0, 0, 0), Z, jtype="fixed"))

    _assign_masses(spec, links)
    return links, joints


def _hand_links(s: EAtlasSpec, side: str) -> list[_Link]:
    if s.hand == "parallel":
        return [_Link(f"{side}_gripper_{i}", "box", (0.02, 0.02, 0.09),
                      (0, 0, -0.045)) for i in (0, 1)]
    links = []
    for finger, n in (("thumb", 3), ("index", 3), ("middle", 2), ("ring", 2), ("pinky", 2)):
        links += [_Link(f"{side}_{finger}_{k}", "box", (0.015, 0.015, 0.03), (0, 0, -0.015))
                  for k in range(n)]
    return links


def _hand_joints(s: EAtlasSpec, side: str) -> list[_Joint]:
    hand = f"{side}_hand"
    if s.hand == "parallel":
        # One driven jaw, one fixed. A real parallel gripper couples both jaws,
        # which URDF expresses with <mimic>; Isaac Lab's importer handles mimic
        # poorly, and either way the gripper is 1 actuated DoF.
        return [
            _Joint(f"{side}_gripper_0_joint", hand, f"{side}_gripper_0",
                   (0, 0.02, -s.hand_len), Y, lower=0.0, upper=0.05,
                   jtype="prismatic", effort=120.0),
            _Joint(f"{side}_gripper_1_joint", hand, f"{side}_gripper_1",
                   (0, -0.02, -s.hand_len), Y, jtype="fixed"),
        ]
    joints, parent = [], hand
    for finger, n in (("thumb", 3), ("index", 3), ("middle", 2), ("ring", 2), ("pinky", 2)):
        parent = hand
        for k in range(n):
            child = f"{side}_{finger}_{k}"
            joints.append(
                _Joint(f"{side}_{finger}_{k}_joint", parent, child,
                       (0, 0, -s.hand_len if k == 0 else -0.03), Y,
                       lower=-0.2, upper=1.6, effort=15.0)
            )
            parent = child
    return joints


def _assign_masses(spec: EAtlasSpec, links: list[_Link]) -> None:
    """Distribute the published total mass over the real links.

    Winter's fractions are normalised so real links plus the token-mass dummy
    links sum exactly to ``spec.mass``.
    """
    keyed = {
        "pelvis": "pelvis", "torso": "torso", "head": "head",
        "upper_arm": "upper_arm", "forearm": "forearm", "hand": "hand",
        "thigh": "thigh", "shank": "shank", "foot": "foot",
    }

    def fraction_of(link: _Link) -> float:
        if link.geom == "none":
            return 0.0
        for suffix, key in keyed.items():
            if link.name == suffix or link.name.endswith("_" + suffix):
                return _MASS[key]
        return 0.002  # fingers / gripper jaws

    dummies = [l for l in links if l.geom == "none"]
    real = [l for l in links if l.geom != "none"]
    dummy_total = len(dummies) * spec.dummy_link_mass
    raw = {id(l): fraction_of(l) for l in real}
    scale = (spec.mass - dummy_total) / sum(raw.values())

    for l in dummies:
        l.mass = spec.dummy_link_mass
    for l in real:
        l.mass = raw[id(l)] * scale


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------


def inertia_tensor(link: _Link) -> tuple[float, float, float]:
    """Principal moments (ixx, iyy, izz) for the link's primitive geometry."""
    m = link.mass
    if link.geom == "box":
        x, y, z = link.size
        return (m / 12 * (y * y + z * z), m / 12 * (x * x + z * z), m / 12 * (x * x + y * y))
    if link.geom == "cylinder":
        r, h = link.size
        return (m / 12 * (3 * r * r + h * h), m / 12 * (3 * r * r + h * h), m / 2 * r * r)
    e = 1e-5
    return (e, e, e)


# ---------------------------------------------------------------------------
# URDF
# ---------------------------------------------------------------------------


def _primitive(geometry: "ET.Element", link: "_Link") -> None:
    """Convex collision shape for a link. Never a mesh - triangle-mesh
    colliders are the fastest way to make a many-environment scene unusable."""
    if link.geom == "box":
        ET.SubElement(geometry, "box", {"size": " ".join(f"{c:.6f}" for c in link.size)})
    else:
        ET.SubElement(geometry, "cylinder", {
            "radius": f"{link.size[0]:.6f}", "length": f"{link.size[1]:.6f}"})


_HEADER = """UNOFFICIAL APPROXIMATION - NOT a Boston Dynamics asset.

Generated by masonry_rl.robots.eatlas_approx from publicly reported
specifications of the electric Atlas. Not derived from Boston Dynamics CAD,
meshes or any proprietary data. Link shapes are primitives; masses, inertias
and joint limits are estimates. Suitable for workspace, reachability and
balance bring-up only - do not present results from this model as Atlas
results.

height {h:.2f} m | mass {m:.1f} kg | span {s:.2f} m | single-arm reach {r:.2f} m
"""


def build_urdf(spec: EAtlasSpec | None = None, mesh_prefix: str | None = None) -> str:
    """Serialise the model.

    With ``mesh_prefix`` set, ``<visual>`` references the generated OBJ meshes
    while ``<collision>`` keeps the convex primitive. That split is the whole
    point: contact solving stays cheap at hundreds of environments no matter
    how detailed the visual gets.
    """
    spec = spec or EAtlasSpec()
    links, joints = build_model(spec)

    robot = ET.Element("robot", {"name": "eatlas_approx"})
    robot.append(
        ET.Comment(
            _HEADER.format(h=spec.height, m=spec.mass, s=spec.span, r=spec.arm_total)
        )
    )

    def xyz(v) -> str:
        return " ".join(f"{c:.6f}" for c in v)

    for l in links:
        node = ET.SubElement(robot, "link", {"name": l.name})
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": xyz(l.origin), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": f"{l.mass:.6f}"})
        ixx, iyy, izz = inertia_tensor(l)
        ET.SubElement(inertial, "inertia", {
            "ixx": f"{ixx:.8f}", "iyy": f"{iyy:.8f}", "izz": f"{izz:.8f}",
            "ixy": "0", "ixz": "0", "iyz": "0",
        })
        if l.geom == "none":
            continue

        visual = ET.SubElement(node, "visual")
        geo = ET.SubElement(visual, "geometry")
        if mesh_prefix:
            # meshes are authored in the link frame, so no visual origin offset
            ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(geo, "mesh", {"filename": f"{mesh_prefix}{l.name}.obj"})
        else:
            ET.SubElement(visual, "origin", {"xyz": xyz(l.origin), "rpy": "0 0 0"})
            _primitive(geo, l)

        collision = ET.SubElement(node, "collision")
        ET.SubElement(collision, "origin", {"xyz": xyz(l.origin), "rpy": "0 0 0"})
        _primitive(ET.SubElement(collision, "geometry"), l)

    for j in joints:
        node = ET.SubElement(robot, "joint", {"name": j.name, "type": j.jtype})
        ET.SubElement(node, "parent", {"link": j.parent})
        ET.SubElement(node, "child", {"link": j.child})
        ET.SubElement(node, "origin", {"xyz": xyz(j.origin), "rpy": "0 0 0"})
        if j.jtype == "fixed":
            continue
        ET.SubElement(node, "axis", {"xyz": xyz(j.axis)})
        attrs = {"effort": f"{j.effort:.1f}", "velocity": f"{j.velocity:.1f}"}
        if j.jtype != "continuous":
            attrs |= {"lower": f"{j.lower:.4f}", "upper": f"{j.upper:.4f}"}
        ET.SubElement(node, "limit", attrs)

    ET.indent(robot, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"


# ---------------------------------------------------------------------------
# Forward kinematics at zero configuration
# ---------------------------------------------------------------------------


def link_frames(spec: EAtlasSpec | None = None) -> dict[str, tuple[float, float, float]]:
    """World position of every link frame at zero configuration.

    The pelvis is placed so the feet rest on z = 0. Joint origins are pure
    translations, so this is a translation sum.
    """
    spec = spec or EAtlasSpec()
    links, joints = build_model(spec)
    frames = {"pelvis": (0.0, 0.0, spec.hip_z)}
    pending = list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            if j.parent in frames:
                px, py, pz = frames[j.parent]
                frames[j.child] = (px + j.origin[0], py + j.origin[1], pz + j.origin[2])
                pending.remove(j)
                progressed = True
        if not progressed:
            raise RuntimeError(f"disconnected links: {[j.child for j in pending]}")
    return frames


def geometry_report(spec: EAtlasSpec | None = None) -> dict[str, float]:
    """Measured properties of the generated model, for validation."""
    spec = spec or EAtlasSpec()
    links, joints = build_model(spec)
    frames = link_frames(spec)
    by_name = {l.name: l for l in links}

    head = by_name["head"]
    head_top = frames["head"][2] + head.origin[2] + head.size[2] / 2
    foot = by_name["l_foot"]
    foot_bottom = frames["l_foot"][2] + foot.origin[2] - foot.size[2] / 2

    return {
        "height": head_top - foot_bottom,
        "mass": sum(l.mass for l in links),
        "span": spec.shoulder_width + 2 * spec.arm_total,
        "arm_reach": spec.arm_total,
        "dof": sum(1 for j in joints if j.jtype != "fixed"),
        "num_links": len(links),
        "foot_bottom": foot_bottom,
    }


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------

_SPEC = EAtlasSpec()

EATLAS_APPROX_PRESET = RobotPreset(
    name="eatlas_approx",
    usd_path=None,  # produced by scripts/build_eatlas.py then converted in P0
    urdf_path="assets/eatlas_approx.urdf",
    joint_groups={
        "right_arm": ["r_(shoulder|elbow|wrist)_.*"],
        "left_arm": ["l_(shoulder|elbow|wrist)_.*"],
        "torso": ["waist_.*"],
        "legs": ["[lr]_(hip|knee|ankle)_.*"],
        "neck": ["neck_.*"],
        "right_gripper": ["r_(gripper|thumb|index|middle|ring|pinky)_.*"],
        "left_gripper": ["l_(gripper|thumb|index|middle|ring|pinky)_.*"],
    },
    ee_frames={"right": "r_hand", "left": "l_hand"},
    pelvis_body="pelvis",
    foot_bodies=["l_foot", "r_foot"],
    contact_bodies=["l_foot", "r_foot", "r_hand", "l_hand"],
    nominal_pelvis_height=_SPEC.hip_z,
    arm_reach=_SPEC.arm_total,
    default_joint_pos={
        "[lr]_hip_pitch_joint": -0.30,
        "[lr]_knee_pitch_joint": 0.60,
        "[lr]_ankle_pitch_joint": -0.30,
        "[lr]_shoulder_pitch_joint": -0.20,
        "[lr]_elbow_pitch_joint": -0.60,
    },
    actuators={
        "arms": ActuatorGroup(["[lr]_(shoulder|elbow|wrist)_.*"], stiffness=300.0, damping=25.0),
        "torso": ActuatorGroup(["waist_.*"], stiffness=700.0, damping=60.0),
        "legs": ActuatorGroup(["[lr]_(hip|knee|ankle)_.*"], stiffness=1200.0, damping=100.0),
        "neck": ActuatorGroup(["neck_.*"], stiffness=20.0, damping=2.0),
        "grippers": ActuatorGroup(
            ["[lr]_(gripper|thumb|index|middle|ring|pinky)_.*"], stiffness=150.0, damping=8.0
        ),
    },
    gripper_open=0.05,
    gripper_closed=0.0,
    pelvis_height_range=(-0.50, 0.05),
    pelvis_pitch_range=(-0.40, 0.40),
    pelvis_shift_range=(-0.14, 0.14),
)
