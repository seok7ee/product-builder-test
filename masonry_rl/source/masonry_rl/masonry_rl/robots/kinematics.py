"""Forward kinematics for the generated skeleton.

Small enough to be obvious, shared so it cannot drift: the viewer exporter and
the mesh rigger both need to pose the same skeleton, and two copies of a
rotation convention is how a model ends up subtly wrong in one place only.

All joint origins in the generated model are pure translations, so a joint's
local transform is just the rotation about its axis.
"""

from __future__ import annotations

import math

__all__ = ["Vec", "Mat", "IDENTITY", "rot", "matmul", "apply", "add", "sub", "norm",
           "forward_kinematics", "A_POSE"]

Vec = tuple[float, float, float]
Mat = tuple[Vec, Vec, Vec]

IDENTITY: Mat = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rot(axis: Vec, angle: float) -> Mat:
    c, s = math.cos(angle), math.sin(angle)
    if axis[0]:
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    if axis[1]:
        return ((c, 0, s), (0, 1, 0), (-s, 0, c))
    if axis[2]:
        return ((c, -s, 0), (s, c, 0), (0, 0, 1))
    return IDENTITY


def matmul(a: Mat, b: Mat) -> Mat:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def apply(m: Mat, v: Vec) -> Vec:
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore


def transpose(m: Mat) -> Mat:
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))  # type: ignore


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm(v: Vec) -> float:
    return math.sqrt(sum(c * c for c in v))


def forward_kinematics(
    joints, q: dict[str, float], pelvis_z: float
) -> dict[str, tuple[Vec, Mat]]:
    """World pose of every link as ``(position, rotation)``.

    ``q`` maps joint name to angle in radians; absent joints are held at zero.
    """
    frames: dict[str, tuple[Vec, Mat]] = {"pelvis": ((0.0, 0.0, pelvis_z), IDENTITY)}
    pending = list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            if j.parent not in frames:
                continue
            parent_pos, parent_rot = frames[j.parent]
            pos = add(parent_pos, apply(parent_rot, j.origin))
            local = IDENTITY if j.jtype == "fixed" else rot(j.axis, q.get(j.name, 0.0))
            frames[j.child] = (pos, matmul(parent_rot, local))
            pending.remove(j)
            progressed = True
        if not progressed:
            raise RuntimeError(
                f"disconnected kinematic tree: {[j.child for j in pending][:5]}"
            )
    return frames


A_POSE: dict[str, float] = {
    "l_shoulder_roll_joint": math.radians(40.0),
    "r_shoulder_roll_joint": math.radians(-40.0),
}
"""Arms held away from the body.

This is the pose to segment a source mesh in, not the rest pose. With the arms
hanging at the sides, the hands sit within a few centimetres of the thighs and
proximity cannot tell them apart - measured, that single ambiguity accounted
for every kinematically distant misassignment in the round trip. Abducting the
shoulders removes it completely, and it also matches reality: humanoid models
are usually distributed in A- or T-pose, not with the arms flat against the
legs.
"""
