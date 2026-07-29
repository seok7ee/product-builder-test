"""Rig a static humanoid mesh onto the generated skeleton.

A downloaded humanoid model - scan, marketplace asset, sculpt - is one
unarticulated shell. A URDF needs the body cut into links with joint frames
between them. That cutting is the actual work; obtaining the mesh is the easy
part. This module does the cutting automatically:

1. normalise the mesh to the skeleton's scale and orientation
2. assign every triangle to the nearest *bone* - the segment between a link's
   own frame and where its geometry ends
3. re-express each group in its link's local frame and emit one OBJ per link

Then the URDF's ``<visual>`` points at those files while ``<collision>`` keeps
the convex primitives, exactly as with the generated meshes.

Segment in an A-pose, not the rest pose. Measured on a round trip through the
generated model, arms hanging at the sides put the hands within centimetres of
the thighs and proximity cannot separate them - that one ambiguity accounted
for *every* kinematically distant misassignment (12.2% of faces). Abducting the
shoulders removes it entirely. It also matches how humanoid models actually
ship: A- or T-pose, rarely arms-against-legs. Supply the source mesh in a
matching pose and pass the same joint angles.

No third-party geometry ships with this repository. This is a tool that runs
against a file you supply.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .kinematics import A_POSE, Mat, Vec, apply, forward_kinematics, transpose

__all__ = [
    "A_POSE",
    "Bone",
    "posed_frames",
    "Soup",
    "load_obj",
    "normalise",
    "bone_segments",
    "assign_faces",
    "face_adjacency",
    "smooth_labels",
    "split_into_links",
    "RigReport",
]

@dataclass
class Bone:
    """A link's representative capsule: a segment plus the link's girth.

    Assignment scores distance *in units of the bone's own radius*, which is
    the only version of this that works. Two wrong alternatives, both measured:

    * raw segment distance lets a long thin bone beat a short fat one for
      points obviously on the fat one - the limbs eat the trunk (68.8%)
    * subtracting the radius outright makes fat bones win everywhere, so the
      thigh claims the hand hanging beside it (60.5%)

    Dividing by the radius is scale invariant: a point one radius from a thin
    bone ties with a point one radius from a fat one, and each body part keeps
    the geometry that sits within its own girth.
    """

    a: Vec
    b: Vec
    radius: float

    def score(self, p: Vec) -> float:
        """Assignment metric: distance normalised by the bone's girth."""
        return _point_segment_distance(p, self.a, self.b) / max(self.radius, 1e-6)

    def surface_distance(self, p: Vec) -> float:
        """How far the point lies outside the capsule, in metres. Reported to
        the user, because a normalised score means nothing to a human."""
        return max(_point_segment_distance(p, self.a, self.b) - self.radius, 0.0)


@dataclass
class Soup:
    """An unstructured mesh: vertices plus polygon indices."""

    vertices: list[Vec]
    faces: list[tuple[int, ...]]

    @property
    def bounds(self) -> tuple[Vec, Vec]:
        lo = tuple(min(v[i] for v in self.vertices) for i in range(3))
        hi = tuple(max(v[i] for v in self.vertices) for i in range(3))
        return lo, hi  # type: ignore[return-value]

    def centroid(self, face: tuple[int, ...]) -> Vec:
        n = len(face)
        return tuple(sum(self.vertices[i][a] for i in face) / n for a in range(3))  # type: ignore


# ---------------------------------------------------------------------------
# OBJ input
# ---------------------------------------------------------------------------


def load_obj(text: str) -> Soup:
    """Parse the subset of OBJ that carries geometry.

    Handles ``f a b c``, ``f a/vt b/vt``, ``f a/vt/vn ...`` and negative
    (relative) indices. Materials, normals and texture coordinates are ignored -
    they do not survive the split anyway.
    """
    vertices: list[Vec] = []
    faces: list[tuple[int, ...]] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append(tuple(float(c) for c in parts[1:4]))  # type: ignore[arg-type]
        elif parts[0] == "f" and len(parts) >= 4:
            idx = []
            for token in parts[1:]:
                raw = int(token.split("/")[0])
                idx.append(raw - 1 if raw > 0 else len(vertices) + raw)
            faces.append(tuple(idx))
    if not vertices:
        raise ValueError("no vertices found - is this an OBJ file?")
    if not faces:
        raise ValueError("no faces found")
    return Soup(vertices, faces)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise(soup: Soup, target_height: float, up: str = "z", forward: str = "x") -> Soup:
    """Scale and reorient a source mesh into the skeleton's convention.

    Modelling packages commonly export Y-up and in centimetres or inches, so a
    raw import is typically both sideways and one to a hundred times the wrong
    size. Everything here keys off the mesh's own bounding box, so no unit
    metadata is required.

    The skeleton's convention is z up, x forward, feet on z = 0.
    """
    if up not in ("y", "z"):
        raise ValueError("up must be 'y' or 'z'")
    if forward not in ("x", "y", "z", "-x", "-y", "-z"):
        raise ValueError(f"unsupported forward axis: {forward!r}")

    verts = soup.vertices
    if up == "y":
        verts = [(x, -z, y) for x, y, z in verts]  # y-up -> z-up, right handed

    sign = -1.0 if forward.startswith("-") else 1.0
    axis = forward[-1]
    if axis == "y":
        verts = [(sign * y, -sign * x, z) for x, y, z in verts]
    elif axis == "x" and sign < 0:
        verts = [(-x, -y, z) for x, y, z in verts]

    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    height = hi[2] - lo[2]
    if height <= 0:
        raise ValueError("mesh has no vertical extent after reorientation")

    scale = target_height / height
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2
    return Soup(
        [((x - cx) * scale, (y - cy) * scale, (z - lo[2]) * scale) for x, y, z in verts],
        soup.faces,
    )


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _point_segment_distance(p: Vec, a: Vec, b: Vec) -> float:
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    denom = sum(c * c for c in ab)
    t = 0.0 if denom == 0 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / denom))
    closest = [a[i] + t * ab[i] for i in range(3)]
    return math.sqrt(sum((p[i] - closest[i]) ** 2 for i in range(3)))


def posed_frames(spec, joints, q: dict[str, float] | None = None):
    """Link poses for segmentation. Defaults to :data:`A_POSE`."""
    return forward_kinematics(joints, A_POSE if q is None else q, spec.hip_z)


def bone_segments(links, frames: dict[str, tuple[Vec, Mat]]) -> dict[str, Bone]:
    """A representative capsule per link, in world coordinates at the given pose.

    A link's bone runs from its frame origin through its geometry: for a limb
    the full length of the segment, for a compact body a short stub. Nearest
    bone is far more stable than nearest origin, because limb origins all
    cluster at the joints.
    """
    bones: dict[str, Bone] = {}
    for link in links:
        if link.geom == "none" or link.name not in frames:
            continue
        position, rotation = frames[link.name]
        ox, oy, oz = link.origin
        if link.geom == "cylinder":
            half, radius = link.size[1] / 2, link.size[0]
        else:
            half = link.size[2] / 2
            radius = (link.size[0] + link.size[1]) / 4  # mean cross-section half-width
        ends = []
        for sign in (+1, -1):
            local = (ox, oy, oz + sign * half)
            world = apply(rotation, local)
            ends.append(tuple(position[i] + world[i] for i in range(3)))
        bones[link.name] = Bone(ends[0], ends[1], radius)  # type: ignore[arg-type]
    return bones


def face_adjacency(soup: Soup) -> dict[int, set[int]]:
    """Faces that share an edge. Built once, reused by the smoothing pass."""
    by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, face in enumerate(soup.faces):
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])
            by_edge[(min(edge), max(edge))].append(index)

    neighbours: dict[int, set[int]] = defaultdict(set)
    for shared in by_edge.values():
        for a in shared:
            neighbours[a].update(x for x in shared if x != a)
    return neighbours


def smooth_labels(soup: Soup, labels: list[str], iterations: int = 4) -> list[str]:
    """Snap each face to its neighbours' majority label.

    Proximity alone gives speckled boundaries and, worse, hands a whole
    joint-straddling collar to whichever side happens to be a millimetre
    nearer. Body parts are contiguous surfaces, so a face surrounded by forearm
    is forearm regardless of what the distance metric preferred. This is the
    part that also helps a real single-shell download, where every boundary is
    decided by proximity and nothing else.
    """
    neighbours = face_adjacency(soup)
    for _ in range(iterations):
        updated = list(labels)
        changed = 0
        for index in range(len(soup.faces)):
            adjacent = neighbours.get(index)
            if not adjacent:
                continue
            winner, votes = Counter(labels[n] for n in adjacent).most_common(1)[0]
            if winner != labels[index] and votes * 2 > len(adjacent):
                updated[index] = winner
                changed += 1
        labels = updated
        if not changed:
            break
    return labels


@dataclass
class RigReport:
    """What the segmentation actually did. Read this before trusting output."""

    faces_per_link: dict[str, int] = field(default_factory=dict)
    mean_distance: dict[str, float] = field(default_factory=dict)
    total_faces: int = 0

    @property
    def empty_links(self) -> list[str]:
        return sorted(k for k, v in self.faces_per_link.items() if v == 0)

    @property
    def worst_fit(self) -> tuple[str, float]:
        if not self.mean_distance:
            return ("", 0.0)
        return max(self.mean_distance.items(), key=lambda kv: kv[1])

    def summary(self) -> str:
        lines = [f"{self.total_faces} faces over {len(self.faces_per_link)} links"]
        for name, count in sorted(self.faces_per_link.items(), key=lambda kv: -kv[1]):
            d = self.mean_distance.get(name, 0.0)
            lines.append(f"  {name:22s} {count:6d} faces   mean bone distance {d * 1000:6.1f} mm")
        if self.empty_links:
            lines.append(f"  EMPTY: {', '.join(self.empty_links)}")
        return "\n".join(lines)


def assign_faces(
    soup: Soup, bones: dict[str, Bone], smooth: int = 4
) -> tuple[dict[str, list[tuple[int, ...]]], RigReport]:
    """Group faces by bone, then make the grouping spatially contiguous."""
    if not bones:
        raise ValueError("no bones to assign against")

    labels: list[str] = []
    for face in soup.faces:
        c = soup.centroid(face)
        best_name, best_score = "", float("inf")
        for name, bone in bones.items():
            score = bone.score(c)
            if score < best_score:
                best_name, best_score = name, score
        labels.append(best_name)

    if smooth:
        labels = smooth_labels(soup, labels, iterations=smooth)

    groups: dict[str, list[tuple[int, ...]]] = {name: [] for name in bones}
    distances: dict[str, float] = {name: 0.0 for name in bones}
    for face, name in zip(soup.faces, labels):
        groups[name].append(face)
        distances[name] += bones[name].surface_distance(soup.centroid(face))

    report = RigReport(total_faces=len(soup.faces))
    for name, faces in groups.items():
        report.faces_per_link[name] = len(faces)
        report.mean_distance[name] = distances[name] / len(faces) if faces else 0.0
    return groups, report


def split_into_links(
    soup: Soup, links, frames: dict[str, tuple[Vec, Mat]]
) -> tuple[dict[str, Soup], RigReport]:
    """Cut a world-space mesh into per-link meshes in each link's local frame.

    The inverse of the link's world pose is applied, so the result is correct
    for any segmentation pose - not only the rest pose.
    """
    bones = bone_segments(links, frames)
    groups, report = assign_faces(soup, bones)

    out: dict[str, Soup] = {}
    for name, faces in groups.items():
        if not faces:
            continue
        position, rotation = frames[name]
        inverse = transpose(rotation)
        remap: dict[int, int] = {}
        verts: list[Vec] = []
        local_faces: list[tuple[int, ...]] = []
        for face in faces:
            idx = []
            for i in face:
                if i not in remap:
                    world = soup.vertices[i]
                    offset = tuple(world[k] - position[k] for k in range(3))
                    remap[i] = len(verts)
                    verts.append(apply(inverse, offset))  # type: ignore[arg-type]
                idx.append(remap[i])
            local_faces.append(tuple(idx))
        out[name] = Soup(verts, local_faces)
    return out, report
