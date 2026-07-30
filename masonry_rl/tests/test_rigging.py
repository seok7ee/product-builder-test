"""Tests for the static-mesh rigger.

The decisive test is the round trip: bake the generated per-link meshes into
one world-space shell, forget the structure, and check the rigger recovers it.
If it cannot re-segment a mesh built from the very skeleton it is fitting, it
has no chance on a downloaded model.
"""

from __future__ import annotations

import math

import pytest

from masonry_rl.robots.eatlas_approx import EAtlasSpec, build_model
from masonry_rl.robots.kinematics import A_POSE, apply
from masonry_rl.robots.meshes import link_mesh, to_obj
from masonry_rl.robots.rigging import (
    Soup,
    assign_faces,
    bone_segments,
    face_adjacency,
    load_obj,
    normalise,
    posed_frames,
    smooth_labels,
    split_into_links,
)


@pytest.fixture(scope="module")
def model():
    """Skeleton posed the way a source mesh should be segmented: A-pose."""
    spec = EAtlasSpec()
    all_links, joints = build_model(spec)
    links = [l for l in all_links if l.geom != "none"]
    return spec, links, joints, posed_frames(spec, joints)


@pytest.fixture(scope="module")
def kin_neighbours(model):
    """Which links are kinematically adjacent, skipping massless dummies.

    Geometry sitting in a joint collar genuinely belongs to either side, so
    parent/child confusion is not an error. Anything further apart is.
    """
    _, links, joints, _ = model
    visual = {l.name for l in links}
    parent = {j.child: j.parent for j in joints}

    def ancestor(name):
        while name in parent:
            name = parent[name]
            if name in visual:
                return name
        return None

    neighbours = {name: set() for name in visual}
    for name in visual:
        up = ancestor(name)
        if up:
            neighbours[name].add(up)
            neighbours[up].add(name)
    return neighbours


@pytest.fixture(scope="module")
def baked(model):
    """The generated model flattened into one world-space soup, with the
    per-face ground truth kept alongside for scoring."""
    _, links, _, frames = model
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    truth: list[str] = []
    for link in links:
        position, rotation = frames[link.name]
        mesh = link_mesh(link)
        offset = len(verts)
        for v in mesh.vertices:
            world = apply(rotation, v)
            verts.append(tuple(position[i] + world[i] for i in range(3)))
        for f in mesh.faces:
            faces.append(tuple(i + offset for i in f))
            truth.append(link.name)
    return Soup(verts, faces), truth


# ---------------------------------------------------------------------------
# OBJ parsing
# ---------------------------------------------------------------------------


def test_load_obj_reads_vertices_and_faces():
    soup = load_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert soup.vertices == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    assert soup.faces == [(0, 1, 2)]


def test_load_obj_handles_texture_and_normal_indices():
    soup = load_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n")
    assert soup.faces == [(0, 1, 2)]


def test_load_obj_handles_negative_indices():
    soup = load_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
    assert soup.faces == [(0, 1, 2)]


def test_load_obj_ignores_comments_normals_and_materials():
    soup = load_obj(
        "# comment\nmtllib x.mtl\nv 0 0 0\nvn 0 0 1\nvt 0 0\n"
        "v 1 0 0\nv 0 1 0\nusemtl m\nf 1 2 3\n"
    )
    assert len(soup.vertices) == 3 and len(soup.faces) == 1


def test_load_obj_rejects_files_without_geometry():
    with pytest.raises(ValueError, match="no vertices"):
        load_obj("# nothing here\n")
    with pytest.raises(ValueError, match="no faces"):
        load_obj("v 0 0 0\n")


def test_obj_round_trip_through_the_generator(model):
    _, links, _, _ = model
    link = next(l for l in links if l.name == "pelvis")
    mesh = link_mesh(link)
    soup = load_obj(to_obj(mesh, "pelvis"))
    assert len(soup.vertices) == len(mesh.vertices)
    assert len(soup.faces) == len(mesh.faces)


# ---------------------------------------------------------------------------
# Normalisation - source meshes arrive sideways and at the wrong scale
# ---------------------------------------------------------------------------


def test_normalise_scales_to_target_height_and_grounds_the_feet():
    soup = Soup([(0, 0, 0), (10, 10, 800), (0, 10, 400)], [(0, 1, 2)])
    out = normalise(soup, target_height=1.9)
    lo, hi = out.bounds
    assert hi[2] - lo[2] == pytest.approx(1.9)
    assert lo[2] == pytest.approx(0.0)


def test_normalise_centres_horizontally():
    soup = Soup([(100, 50, 0), (110, 60, 100), (105, 55, 50)], [(0, 1, 2)])
    lo, hi = normalise(soup, target_height=1.9).bounds
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-9)
    assert (lo[1] + hi[1]) / 2 == pytest.approx(0.0, abs=1e-9)


def test_normalise_converts_y_up_to_z_up():
    """Most modelling packages export Y-up; a raw import lies on its face."""
    soup = Soup([(0, 0, 0), (0, 2, 0), (1, 1, 0)], [(0, 1, 2)])
    out = normalise(soup, target_height=1.9, up="y")
    lo, hi = out.bounds
    assert hi[2] - lo[2] == pytest.approx(1.9)


def test_normalise_rejects_unknown_axes():
    soup = Soup([(0, 0, 0), (0, 0, 1), (1, 0, 0)], [(0, 1, 2)])
    with pytest.raises(ValueError):
        normalise(soup, 1.9, up="q")
    with pytest.raises(ValueError):
        normalise(soup, 1.9, forward="w")


def test_normalise_rejects_a_flat_mesh():
    soup = Soup([(0, 0, 5), (1, 0, 5), (0, 1, 5)], [(0, 1, 2)])
    with pytest.raises(ValueError, match="vertical extent"):
        normalise(soup, 1.9)


# ---------------------------------------------------------------------------
# Bones
# ---------------------------------------------------------------------------


def test_every_visual_link_gets_a_bone(model):
    _, links, _, frames = model
    bones = bone_segments(links, frames)
    assert set(bones) == {l.name for l in links}


def test_limb_bones_span_their_segment(model):
    spec, links, _, frames = model
    bones = bone_segments(links, frames)
    bone = bones["l_upper_arm"]
    assert math.dist(bone.a, bone.b) == pytest.approx(spec.upper_arm, abs=1e-6)
    assert bone.radius > 0


def test_bones_are_positioned_in_the_world(model):
    _, links, _, frames = model
    bones = bone_segments(links, frames)
    foot, head = bones["l_foot"], bones["head"]
    assert max(p[2] for p in (foot.a, foot.b)) < min(p[2] for p in (head.a, head.b))


def test_a_pose_separates_the_hands_from_the_thighs(model):
    """The measured reason segmentation runs in A-pose rather than rest pose."""
    spec, links, joints, _ = model

    def thigh_margin(pose):
        frames = posed_frames(spec, joints, pose)
        bones = bone_segments(links, frames)
        tip = bones["l_hand"].b
        return bones["l_thigh"].score(tip) - bones["l_hand"].score(tip)

    rest = thigh_margin({})
    spread = thigh_margin(A_POSE)
    assert spread > rest, "A-pose should push the hand further from the thigh"
    assert rest < 1.0, "at rest the hand is barely distinguishable from the thigh"


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def _score(model, baked, kin):
    _, links, _, frames = model
    soup, truth = baked
    groups, _ = assign_faces(soup, bone_segments(links, frames))
    owner = {face: name for name, faces in groups.items() for face in faces}

    exact = adjacent = distant = 0
    strays = []
    for face, want in zip(soup.faces, truth):
        got = owner[face]
        if got == want:
            exact += 1
        elif got in kin[want]:
            adjacent += 1
        else:
            distant += 1
            strays.append((want, got))
    n = len(soup.faces)
    return exact / n, adjacent / n, distant / n, strays


def test_no_face_lands_on_an_unrelated_body_part(model, baked, kin_neighbours):
    """The property that matters. Confusing a knee collar between thigh and
    shank is unavoidable; putting a hand on a thigh is a real defect, and it is
    exactly what rest-pose segmentation did before A-pose fixed it."""
    _, _, distant, strays = _score(model, baked, kin_neighbours)
    assert distant == 0.0, f"{distant:.1%} of faces stranded, e.g. {strays[:3]}"


def test_most_faces_land_on_exactly_the_right_link(model, baked, kin_neighbours):
    exact, adjacent, _, _ = _score(model, baked, kin_neighbours)
    assert exact > 0.70, f"only {exact:.1%} exact"
    # the remainder is joint-collar geometry, which is ambiguous by construction
    assert exact + adjacent == pytest.approx(1.0)


def test_split_produces_local_coordinates(model, baked):
    _, links, _, frames = model
    soup, _ = baked
    parts, report = split_into_links(soup, links, frames)

    assert parts, "no links received geometry"
    assert report.total_faces == len(soup.faces)
    # local coordinates sit near the origin; world coordinates would not
    for name, part in parts.items():
        lo, hi = part.bounds
        assert max(abs(c) for c in lo + hi) < 0.9, f"{name} looks like world coordinates"


def test_split_conserves_every_face(model, baked):
    _, links, _, frames = model
    soup, _ = baked
    parts, _ = split_into_links(soup, links, frames)
    assert sum(len(p.faces) for p in parts.values()) == len(soup.faces)


def test_report_flags_links_that_received_nothing(model, baked):
    _, links, _, frames = model
    soup, _ = baked
    _, report = split_into_links(soup, links, frames)
    # the generated model covers every link, so nothing should be starved
    assert report.empty_links == []
    assert "faces over" in report.summary()


def test_smoothing_makes_the_labelling_contiguous(model, baked):
    """Body parts are connected surfaces; a speckled labelling is wrong even
    when each individual face is nearest the bone it was given."""
    _, links, _, frames = model
    soup, _ = baked
    bones = bone_segments(links, frames)

    raw = []
    for face in soup.faces:
        c = soup.centroid(face)
        raw.append(min(bones, key=lambda n: bones[n].score(c)))
    smoothed = smooth_labels(soup, raw)

    neighbours = face_adjacency(soup)
    seams = lambda labels: sum(
        1 for i, adj in neighbours.items() for j in adj if labels[i] != labels[j]
    )
    assert seams(smoothed) <= seams(raw)


def test_report_measures_fit_quality(model, baked):
    """Mean bone distance is how a user tells a good source mesh from a bad
    one - an A-pose source smears across the shoulders and shows up here."""
    _, links, _, frames = model
    soup, _ = baked
    _, report = split_into_links(soup, links, frames)
    worst_link, worst = report.worst_fit
    assert worst < 0.25, f"{worst_link} fits its bone poorly ({worst:.3f} m)"


def test_capsule_radius_keeps_bulky_links_from_being_eaten_by_limbs(model):
    """A point just inside the torso must prefer the torso over the far longer
    thigh bone. Without the radius correction the limbs swallow the trunk."""
    _, links, _, frames = model
    bones = bone_segments(links, frames)
    hip_area = (0.0, 0.05, frames["pelvis"][0][2])
    assert bones["pelvis"].score(hip_area) < bones["l_thigh"].score(hip_area)


def test_assign_faces_needs_bones():
    with pytest.raises(ValueError, match="no bones"):
        assign_faces(Soup([(0, 0, 0)], [(0, 0, 0)]), {})
