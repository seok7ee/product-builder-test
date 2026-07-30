"""Procedural visual meshes for the generated humanoid.

Why generate rather than download: stock Atlas meshes on model marketplaces are
licensed against redistribution, may carry third-party trademark, and - the
practical objection - are single unarticulated shells with no joint structure,
so they cannot drive a URDF without being cut into links by hand.

These meshes are generated from the same ``EAtlasSpec`` that produces the
kinematics, informed only by *published descriptions* of the robot's design
language: a circular head with integrated lights, an "alien rather than human"
aesthetic, a deliberately simplified part count, and mostly fully rotational
joints. That last one is what gives the model its look here - every joint wears
a visible cylindrical actuator housing.

The split matters for physics: these are **visual** meshes only. Collision
geometry stays as the convex primitives in ``eatlas_approx``, because that is
what keeps contact solving cheap and stable at hundreds of parallel
environments. URDF has always kept ``<visual>`` and ``<collision>`` separate;
this uses that separation properly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Mesh", "link_mesh", "to_obj"]


@dataclass
class Mesh:
    """Polygon soup in the link's own frame. Faces are index tuples."""

    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, ...]]

    def extend(self, other: "Mesh") -> None:
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend(tuple(i + offset for i in f) for f in other.faces)

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lo = tuple(min(v[i] for v in self.vertices) for i in range(3))
        hi = tuple(max(v[i] for v in self.vertices) for i in range(3))
        return lo, hi  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------


def _superellipse(sides: int, rx: float, ry: float, power: float = 4.0) -> list[tuple[float, float]]:
    """Rounded-rectangle cross-section. ``power=2`` is an ellipse, higher is boxier."""
    pts = []
    for i in range(sides):
        t = 2 * math.pi * i / sides
        c, s = math.cos(t), math.sin(t)
        pts.append((
            rx * math.copysign(abs(c) ** (2 / power), c),
            ry * math.copysign(abs(s) ** (2 / power), s),
        ))
    return pts


def _loft(rings: list[tuple[float, list[tuple[float, float]]]], close: bool = True) -> Mesh:
    """Skin a stack of equal-sized rings, optionally capping both ends.

    Rings must run in increasing z so the generated faces wind consistently.
    """
    sides = len(rings[0][1])
    verts: list[tuple[float, float, float]] = []
    for z, ring in rings:
        verts.extend((x, y, z) for x, y in ring)

    faces: list[tuple[int, ...]] = []
    for r in range(len(rings) - 1):
        a, b = r * sides, (r + 1) * sides
        for i in range(sides):
            k = (i + 1) % sides
            faces.append((a + i, a + k, b + k, b + i))
    if close:
        faces.append(tuple(range(sides - 1, -1, -1)))
        base = (len(rings) - 1) * sides
        faces.append(tuple(base + i for i in range(sides)))
    return Mesh(verts, faces)


def _segment(
    z_top: float,
    z_bot: float,
    r_prox: tuple[float, float],
    r_dist: tuple[float, float],
    sides: int = 12,
    power: float = 4.0,
    chamfer: float = 0.12,
) -> Mesh:
    """A limb segment: chamfered at both ends, tapering proximal to distal.

    The taper is what stops the model reading as a stack of tubes - real
    actuated limbs are fat at the joint and slim at the midpoint.
    """
    length = z_top - z_bot
    cut = min(chamfer * length, 0.35 * length)
    profile = [
        (z_bot, 0.62, 0.0),
        (z_bot + cut, 1.0, 0.0),
        (z_bot + 0.45 * length, 0.88, 0.45),
        (z_top - cut, 1.0, 1.0),
        (z_top, 0.62, 1.0),
    ]
    rings = []
    for z, scale, blend in profile:
        rx = (r_dist[0] * (1 - blend) + r_prox[0] * blend) * scale
        ry = (r_dist[1] * (1 - blend) + r_prox[1] * blend) * scale
        rings.append((z, _superellipse(sides, rx, ry, power)))
    return _loft(rings)


def _housing(
    center: tuple[float, float, float], axis: str, radius: float, length: float, sides: int = 14
) -> Mesh:
    """Cylindrical actuator housing straddling a joint, aligned to its axis.

    Boston Dynamics describe the production robot as mostly fully rotational
    joints; these barrels are the visual consequence of that and are the single
    biggest thing separating this from a mannequin.
    """
    rings = []
    for t, scale in ((-0.5, 0.86), (-0.42, 1.0), (0.42, 1.0), (0.5, 0.86)):
        section = _superellipse(sides, radius * scale, radius * scale, power=2.0)
        rings.append((t * length, section))
    mesh = _loft(rings)

    cx, cy, cz = center
    if axis == "y":
        mesh.vertices = [(x + cx, z + cy, y + cz) for x, y, z in mesh.vertices]
        mesh.faces = [tuple(reversed(f)) for f in mesh.faces]
    elif axis == "x":
        mesh.vertices = [(z + cx, x + cy, y + cz) for x, y, z in mesh.vertices]
        mesh.faces = [tuple(reversed(f)) for f in mesh.faces]
    else:
        mesh.vertices = [(x + cx, y + cy, z + cz) for x, y, z in mesh.vertices]
    return mesh


def _slab(size: tuple[float, float, float], origin: tuple[float, float, float],
          sides: int = 16, power: float = 6.0, waist: float = 1.0) -> Mesh:
    """A chamfered slab for the trunk and pelvis; ``waist`` pinches the middle."""
    x, y, z = size
    ox, oy, oz = origin
    cut = 0.10 * z
    rings = []
    for t, scale in ((-0.5, 0.70), (-0.5 + cut / z, 1.0), (0.0, waist),
                     (0.5 - cut / z, 1.0), (0.5, 0.70)):
        rings.append((oz + t * z, _superellipse(sides, x / 2 * scale, y / 2 * scale, power)))
    mesh = _loft(rings)
    mesh.vertices = [(vx + ox, vy + oy, vz) for vx, vy, vz in mesh.vertices]
    return mesh


def _head(size: tuple[float, float, float], origin: tuple[float, float, float]) -> Mesh:
    """A disc facing +x with a recessed ring - the robot's most identifiable
    published feature, and the one that signals 'helpful machine, not person'."""
    depth, _, diameter = size[0], size[1], size[2]
    ox, oy, oz = origin
    radius = diameter / 2

    rings = []
    for t, scale in ((-0.5, 0.80), (-0.36, 1.0), (0.30, 1.0), (0.5, 0.90)):
        rings.append((t * depth, _superellipse(20, radius * scale, radius * scale, power=2.0)))
    shell = _loft(rings)
    # lay the disc on its side: local z becomes world x
    shell.vertices = [(z + ox, y + oy, x + oz) for x, y, z in shell.vertices]
    shell.faces = [tuple(reversed(f)) for f in shell.faces]

    lens = []
    for t, scale in ((0.30, 0.62), (0.40, 0.58)):
        lens.append((t * depth, _superellipse(20, radius * scale, radius * scale, power=2.0)))
    ring = _loft(lens)
    ring.vertices = [(z + ox, y + oy, x + oz) for x, y, z in ring.vertices]
    ring.faces = [tuple(reversed(f)) for f in ring.faces]

    shell.extend(ring)
    return shell


# ---------------------------------------------------------------------------
# Per-link recipes
# ---------------------------------------------------------------------------

# joint axis each limb segment's proximal housing straddles
_HOUSING_AXIS = {
    "upper_arm": "y", "forearm": "y", "hand": "y",
    "thigh": "y", "shank": "y", "foot": "y",
}


def link_mesh(link, sides: int = 12) -> Mesh:
    """Visual mesh for one link, in the link's own frame."""
    name, geom, size, origin = link.name, link.geom, tuple(link.size), tuple(link.origin)

    if "head" in name:
        return _head(size, origin)

    if name == "pelvis":
        return _slab(size, origin, sides=16, power=6.0, waist=1.04)

    if name == "torso":
        # narrower at the waist, broad at the shoulder line
        mesh = _slab(size, origin, sides=16, power=5.0, waist=0.88)
        shoulder_r = size[1] * 0.30
        mesh.extend(_housing((0.0, 0.0, origin[2] + size[2] / 2), "y", shoulder_r, size[1] * 1.02))
        return mesh

    if "foot" in name:
        return _slab(size, origin, sides=12, power=8.0, waist=1.0)

    if "gripper" in name or "thumb" in name or "index" in name or "middle" in name \
            or "ring" in name or "pinky" in name:
        return _slab(size, origin, sides=8, power=6.0)

    # limb segments: cylinder links, drawn as tapered shells with a joint barrel
    if geom == "cylinder":
        radius, length = size
        z_top, z_bot = origin[2] + length / 2, origin[2] - length / 2
        key = next((k for k in _HOUSING_AXIS if name.endswith(k)), None)
        prox = (radius * 1.08, radius * 1.18)
        dist = (radius * 0.80, radius * 0.86)
        mesh = _segment(z_top, z_bot, prox, dist, sides=sides)
        mesh.extend(_housing((0.0, 0.0, z_top), _HOUSING_AXIS.get(key or "", "y"),
                             radius * 1.25, radius * 2.5))
        return mesh

    # anything else: chamfered box
    return _slab(size, origin, sides=sides, power=6.0)


# ---------------------------------------------------------------------------
# OBJ output
# ---------------------------------------------------------------------------


def to_obj(mesh: Mesh, name: str) -> str:
    lines = [
        "# generated by masonry_rl.robots.meshes - UNOFFICIAL approximation,",
        "# not a Boston Dynamics asset and not derived from any third-party model",
        f"o {name}",
    ]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.vertices]
    lines += ["f " + " ".join(str(i + 1) for i in face) for face in mesh.faces]
    return "\n".join(lines) + "\n"
