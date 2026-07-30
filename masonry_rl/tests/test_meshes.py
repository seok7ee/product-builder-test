"""Validation for the procedural visual meshes.

A mesh generator that emits plausible-looking garbage is easy to write and hard
to notice, so these check the properties importers and renderers actually care
about: closed shells, consistent winding, no degenerate faces, and geometry
that still agrees with the collision primitive it stands in for.
"""

from __future__ import annotations

from collections import Counter

import pytest

from masonry_rl.robots.eatlas_approx import EAtlasSpec, build_model
from masonry_rl.robots.meshes import Mesh, link_mesh, to_obj


@pytest.fixture(scope="module")
def links():
    return [l for l in build_model(EAtlasSpec())[0] if l.geom != "none"]


@pytest.fixture(scope="module")
def meshes(links):
    return {l.name: link_mesh(l) for l in links}


def test_every_visual_link_gets_a_mesh(links, meshes):
    assert len(meshes) == len(links)
    for name, mesh in meshes.items():
        assert mesh.vertices, name
        assert mesh.faces, name


def test_faces_reference_valid_vertices(meshes):
    for name, mesh in meshes.items():
        n = len(mesh.vertices)
        for face in mesh.faces:
            assert all(0 <= i < n for i in face), name


def test_no_degenerate_faces(meshes):
    """A face repeating a vertex has zero area and produces a NaN normal."""
    for name, mesh in meshes.items():
        for face in mesh.faces:
            assert len(face) >= 3, name
            assert len(set(face)) == len(face), f"{name}: repeated vertex in {face}"


def test_no_zero_area_faces(meshes):
    for name, mesh in meshes.items():
        for face in mesh.faces:
            a, b, c = (mesh.vertices[i] for i in face[:3])
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            cross = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            assert sum(c * c for c in cross) ** 0.5 > 1e-9, name


def test_shells_are_closed_and_consistently_wound(meshes):
    """Every directed edge must appear exactly once, and its reverse exactly
    once. That is watertightness and consistent winding in one check - and it
    holds for a link made of several disjoint shells, which several are."""
    for name, mesh in meshes.items():
        directed = Counter()
        for face in mesh.faces:
            for i in range(len(face)):
                directed[(face[i], face[(i + 1) % len(face)])] += 1
        for (a, b), count in directed.items():
            assert count == 1, f"{name}: edge {a}->{b} used {count} times"
            assert directed[(b, a)] == 1, f"{name}: edge {a}->{b} has no opposite"


def test_mesh_stays_within_the_collision_primitive(links, meshes):
    """The visual mesh may be prettier than the collider but must not sprawl
    outside it, or contacts fire where nothing appears to touch."""
    for link in links:
        lo, hi = meshes[link.name].bounds
        if link.geom == "box":
            half = [s / 2 for s in link.size]
        else:
            radius, length = link.size
            half = [radius, radius, length / 2]
        for axis in range(3):
            # housings deliberately straddle the joint, so allow a margin
            limit = half[axis] * 1.45 + 0.02
            assert lo[axis] >= link.origin[axis] - limit, f"{link.name} axis {axis}"
            assert hi[axis] <= link.origin[axis] + limit, f"{link.name} axis {axis}"


def test_limb_segments_taper(meshes):
    """Proximal end wider than distal - the thing that stops limbs reading as
    plain tubes."""
    mesh = meshes["l_upper_arm"]
    top = [v for v in mesh.vertices if v[2] > -0.05]
    bottom = [v for v in mesh.vertices if v[2] < -0.30]
    spread = lambda vs: max(abs(v[0]) for v in vs)
    assert spread(top) > spread(bottom)


def test_head_faces_forward(meshes):
    """The circular head is the robot's most identifiable published feature; it
    must be a disc on the x axis, not a sphere."""
    lo, hi = meshes["head"].bounds
    depth = hi[0] - lo[0]
    width = hi[1] - lo[1]
    height = hi[2] - lo[2]
    assert depth < width and depth < height
    assert width == pytest.approx(height, rel=0.25)


def test_torso_is_pinched_at_the_waist(links, meshes):
    """Compare the middle against the full-width ring, not against the ends -
    the ends are chamfered narrow on purpose."""
    torso = next(l for l in links if l.name == "torso")
    mesh = meshes["torso"]
    mid, height = torso.origin[2], torso.size[2]

    at = lambda z: [v for v in mesh.vertices if abs(v[2] - z) < 0.01]
    middle = at(mid)
    full_width = at(mid + 0.4 * height)  # the ring just inside the top chamfer
    assert middle and full_width

    half_width = lambda vs: max(abs(v[1]) for v in vs)
    assert half_width(middle) < half_width(full_width), "waist is not pinched"


def test_obj_output_is_parseable(meshes):
    mesh = meshes["pelvis"]
    text = to_obj(mesh, "pelvis")
    verts = [l for l in text.splitlines() if l.startswith("v ")]
    faces = [l for l in text.splitlines() if l.startswith("f ")]
    assert len(verts) == len(mesh.vertices)
    assert len(faces) == len(mesh.faces)
    # OBJ indices are 1-based and must stay in range
    for line in faces:
        for token in line.split()[1:]:
            assert 1 <= int(token) <= len(mesh.vertices)


def test_obj_carries_the_unofficial_notice(meshes):
    assert "UNOFFICIAL" in to_obj(meshes["head"], "head")


def test_mesh_extend_reindexes_faces():
    a = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    b = Mesh([(0, 0, 1), (1, 0, 1), (0, 1, 1)], [(0, 1, 2)])
    a.extend(b)
    assert len(a.vertices) == 6
    assert a.faces == [(0, 1, 2), (3, 4, 5)]


def test_total_mesh_budget_stays_small(meshes):
    """These get inlined into a viewer payload and imported per environment;
    a generator that quietly emits 100k faces would break both."""
    total = sum(len(m.faces) for m in meshes.values())
    assert total < 4000, f"{total} faces is too heavy for a visual approximation"
