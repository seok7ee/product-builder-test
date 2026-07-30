#!/usr/bin/env python3
"""Export the eatlas_approx model plus posed masonry scenes as JSON for the viewer.

Poses are solved and *verified here*, in Python, rather than in the browser:
the viewer only replays joint angles through forward kinematics. Solving in JS
would mean an unverifiable arm that looks subtly wrong with no way to tell.

    python scripts/export_eatlas_viewer.py --out /tmp/eatlas_model.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "masonry_rl"))

from masonry_rl.robots.eatlas_approx import EAtlasSpec, build_model  # noqa: E402
from masonry_rl.robots.meshes import link_mesh  # noqa: E402
from masonry_rl.tasks.masonry.wall_planner import WallPlanner, WallSpec  # noqa: E402

from masonry_rl.robots.kinematics import (  # noqa: E402
    IDENTITY,
    Mat,
    Vec,
    add,
    apply,
    forward_kinematics,
    matmul,
    norm,
    rot,
    sub,
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Pose solvers
# ---------------------------------------------------------------------------


def solve_legs(spec: EAtlasSpec, pelvis_z: float) -> dict[str, float]:
    """Symmetric squat holding the feet flat and the pelvis over the ankles.

    Thigh and shank are near-equal, so the isoceles solution keeps the shank
    vertical-symmetric and the torso upright without a full IK pass.
    """
    drop = pelvis_z - spec.foot_height
    alpha = math.acos(clamp(drop / (spec.thigh + spec.shank), 0.0, 1.0))
    q = {}
    for side in ("l", "r"):
        q[f"{side}_hip_pitch_joint"] = -alpha
        q[f"{side}_knee_pitch_joint"] = 2 * alpha
        q[f"{side}_ankle_pitch_joint"] = -alpha
    return q


def solve_arm(
    spec: EAtlasSpec, side: str, shoulder: Vec, shoulder_rot: Mat, target: Vec
) -> dict[str, float]:
    """Two-link planar reach, solved in the shoulder's own frame.

    The shoulder chain is pitch(Y) then roll(X), so the arm direction is
    ``R_y(theta) * R_x(phi) * (0,0,-1)`` - composing them the other way round
    is what put the first attempt 300 mm off target. With the elbow bent, the
    quantity that must land on the target direction is the shoulder-to-hand
    *chord*, not the upper-arm axis, so the chord offset enters the solution
    rather than being subtracted afterwards.
    """
    upper, lower = spec.upper_arm, spec.forearm + spec.hand_len

    # work in the shoulder's frame: undo the torso/pelvis rotation
    world_d = sub(target, shoulder)
    inv = tuple(tuple(shoulder_rot[j][i] for j in range(3)) for i in range(3))  # transpose
    d = apply(inv, world_d)  # type: ignore[arg-type]

    distance = norm(d)
    reach = clamp(distance, abs(upper - lower) + 1e-4, (upper + lower) * 0.999)
    dx, dy, dz = (c / distance * reach for c in d)

    bend = math.acos(clamp((reach**2 - upper**2 - lower**2) / (2 * upper * lower), -1.0, 1.0))
    offset = math.atan2(lower * math.sin(bend), upper + lower * math.cos(bend))

    # chord direction in the shoulder frame is (sin(offset), 0, -cos(offset));
    # rotating it by R_y(theta) R_x(phi) must land on (dx, dy, dz) / reach
    s, c = math.sin(offset), math.cos(offset)
    phi = math.asin(clamp(dy / reach / max(c, 1e-6), -1.0, 1.0))
    k = c * math.cos(phi)
    theta = math.atan2(-dz, dx) - math.atan2(k, s)

    return {
        f"{side}_shoulder_pitch_joint": theta,
        f"{side}_shoulder_roll_joint": phi,
        f"{side}_elbow_pitch_joint": -bend,
    }


MAX_WAIST_PITCH = 1.0
"""Waist pitch limit from the generated URDF [rad], about 57 degrees."""


def min_pelvis_height(spec: EAtlasSpec, knee_limit: float = 2.4) -> float:
    """Lowest pelvis the knee limit allows, feet flat."""
    alpha = knee_limit / 2
    return (spec.thigh + spec.shank) * math.cos(alpha) + spec.foot_height


def solve_posture(spec: EAtlasSpec, joints, target: Vec):
    """Find the least extreme squat and torso pitch that puts ``target`` in reach.

    A stationary humanoid with 0.90 m arms cannot touch a floor-level course
    while standing upright - the shoulder is simply too far away. It has to
    squat *and* pitch the torso forward, exactly as a human mason does. This
    searches for the mildest combination that works and reports honestly when
    no combination does.
    """
    usable = spec.arm_total * 0.95  # leave the elbow some room
    floor = min_pelvis_height(spec)
    best = None

    for i in range(61):
        pelvis_z = spec.hip_z - (spec.hip_z - floor) * i / 60
        for k in range(51):
            waist = MAX_WAIST_PITCH * k / 50
            q = solve_legs(spec, pelvis_z) | {"waist_pitch_joint": waist}
            frames = forward_kinematics(joints, q, pelvis_z)
            shoulder, rot_s = frames["r_shoulder_pitch_link"]
            if norm(sub(target, shoulder)) > usable:
                continue
            # prefer standing tall and upright; both terms are normalised
            cost = (spec.hip_z - pelvis_z) / spec.hip_z + waist / MAX_WAIST_PITCH
            if best is None or cost < best[0]:
                best = (cost, pelvis_z, waist, q, frames)
            break  # milder waist angles at this height already failed

    if best is None:
        return None
    _, pelvis_z, waist, q, frames = best
    return pelvis_z, waist, q, frames


def solve_scene(spec: EAtlasSpec, joints, base_courses: int):
    """Pose the robot at a wall with ``base_courses`` already built."""
    wall = WallSpec(base_courses=base_courses)
    planner = WallPlanner(wall, num_envs=1)
    slots, _ = planner.all_slot_poses()
    base, _ = planner.base_brick_poses()

    # work the middle of the lowest course still to be laid
    active = wall.bricks_per_course // 2
    target = tuple(float(c) for c in slots[active])

    posture = solve_posture(spec, joints, target)
    reachable = posture is not None
    if not reachable:
        pelvis_z, waist = min_pelvis_height(spec), MAX_WAIST_PITCH
        q = solve_legs(spec, pelvis_z) | {"waist_pitch_joint": waist}
        frames = forward_kinematics(joints, q, pelvis_z)
    else:
        pelvis_z, waist, q, frames = posture

    for side, aim in (("r", target), ("l", (target[0] - 0.12, target[1] - 0.24, target[2] + 0.08))):
        shoulder, rot_s = frames[f"{side}_shoulder_pitch_link"]
        q |= solve_arm(spec, side, shoulder, rot_s, aim)

    frames = forward_kinematics(joints, q, pelvis_z)
    hand_pos, hand_rot = frames["r_hand"]
    # the hand link frame sits at the wrist; the tool point is hand_len along -z
    tip = add(hand_pos, apply(hand_rot, (0.0, 0.0, -spec.hand_len)))

    return {
        "base_courses": base_courses,
        "pelvis_z": pelvis_z,
        "waist_pitch": waist,
        "work_plane": wall.work_plane_height(),
        "q": q,
        "active_slot": active,
        "reachable": reachable,
        "reach_error": norm(sub(tip, target)),
        "squat_depth": spec.hip_z - pelvis_z,
        "brick": [wall.brick_length, wall.brick_width, wall.brick_height],
        "wall_yaw": wall.yaw,
        "base_bricks": [[float(c) for c in p] for p in base],
        "slots": [[float(c) for c in p] for p in slots],
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _group(name: str) -> str:
    if "gripper" in name or "hand" in name:
        return "hand"
    if any(k in name for k in ("shoulder", "upper_arm", "forearm", "wrist", "elbow")):
        return "arm"
    if "foot" in name or "ankle" in name:
        return "foot"
    if any(k in name for k in ("thigh", "shank", "hip", "knee")):
        return "leg"
    if "head" in name or "neck" in name:
        return "head"
    return "trunk"


def _link_payload(link) -> dict:
    """Ship the generated visual mesh, not the collision primitive.

    Vertices are rounded to a tenth of a millimetre; at this scale that is well
    below anything visible and it roughly halves the payload.
    """
    mesh = link_mesh(link)
    return {
        "name": link.name,
        "group": _group(link.name),
        "mass": round(link.mass, 4),
        "v": [[round(c, 4) for c in v] for v in mesh.vertices],
        "f": [list(f) for f in mesh.faces],
    }


def export(spec: EAtlasSpec) -> dict:
    links, joints = build_model(spec)
    scenes = [solve_scene(spec, joints, n) for n in (12, 6, 3, 0)]
    return {
        "spec": {
            "height": spec.height,
            "mass": spec.mass,
            "span": spec.span,
            "arm_reach": spec.arm_total,
            "hip_z": spec.hip_z,
            "shoulder_z": spec.shoulder_z,
            "hand": spec.hand,
            "dof": sum(1 for j in joints if j.jtype != "fixed"),
        },
        "links": [_link_payload(l) for l in links if l.geom != "none"],
        "joints": [
            {
                "name": j.name,
                "parent": j.parent,
                "child": j.child,
                "origin": list(j.origin),
                "axis": list(j.axis),
                "type": j.jtype,
                "lower": None if j.jtype in ("continuous", "fixed") else round(j.lower, 4),
                "upper": None if j.jtype in ("continuous", "fixed") else round(j.upper, 4),
            }
            for j in joints
        ],
        "scenes": scenes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", choices=("parallel", "five_finger"), default="parallel")
    p.add_argument("--out", default="/tmp/eatlas_model.json")
    args = p.parse_args()

    spec = EAtlasSpec(hand=args.hand)
    data = export(spec)
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))

    faces = sum(len(l["f"]) for l in data["links"])
    print(f"links {len(data['links'])}  joints {len(data['joints'])}  "
          f"dof {data['spec']['dof']}  mesh faces {faces}")
    print("\n  base   work plane   pelvis   squat    waist   reach err  posture")
    for s in data["scenes"]:
        posture = "OK" if s["reachable"] else "OUT OF REACH"
        print(f"  {s['base_courses']:>4}   {s['work_plane']:7.3f} m {s['pelvis_z']:6.3f} m "
              f"{s['squat_depth']:6.3f} m {math.degrees(s['waist_pitch']):6.1f}d "
              f"{s['reach_error'] * 1000:7.1f} mm  {posture}")
    print(f"\n  {args.out}  ({Path(args.out).stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
