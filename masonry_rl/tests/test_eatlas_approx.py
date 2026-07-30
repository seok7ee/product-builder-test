"""Validation for the electric Atlas approximation.

The model is generated, not authored, so these tests are the only thing
standing between a plausible-looking URDF and one that silently fails to
import or falls through the floor.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from masonry_rl.robots.eatlas_approx import (
    EATLAS_APPROX_PRESET,
    EAtlasSpec,
    build_model,
    build_urdf,
    geometry_report,
    inertia_tensor,
    link_frames,
)


@pytest.fixture(params=["parallel", "five_finger"])
def spec(request):
    return EAtlasSpec(hand=request.param)


@pytest.fixture
def urdf(spec):
    return ET.fromstring(build_urdf(spec))


# ---------------------------------------------------------------------------
# Published specs must actually come out the other end
# ---------------------------------------------------------------------------


def test_model_matches_the_published_dimensions(spec):
    r = geometry_report(spec)
    assert r["height"] == pytest.approx(1.90, abs=1e-3)
    assert r["mass"] == pytest.approx(90.0, abs=1e-6)
    assert r["span"] == pytest.approx(2.30, abs=1e-3)


def test_dof_budget(spec):
    expected = 34 if spec.hand == "parallel" else 56
    assert geometry_report(spec)["dof"] == expected


def test_feet_rest_on_the_ground_plane(spec):
    """Off-by-a-few-centimetres here means the robot spawns intersecting the
    floor or dropping into it, and every balance run is poisoned."""
    assert geometry_report(spec)["foot_bottom"] == pytest.approx(0.0, abs=1e-6)


def test_span_and_single_arm_reach_are_different_numbers():
    """The published 2.3 m is fingertip to fingertip. Feeding it to the masonry
    reachability check as single-arm reach would silently allow a wall the
    robot cannot touch."""
    s = EAtlasSpec()
    assert s.arm_total == pytest.approx(0.904, abs=0.01)
    assert s.arm_total < s.span / 2


def test_geometry_scales_with_height():
    tall, short = EAtlasSpec(height=1.90), EAtlasSpec(height=1.50)
    assert geometry_report(tall)["height"] > geometry_report(short)["height"]
    assert short.hip_z / short.height == pytest.approx(tall.hip_z / tall.height)


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        EAtlasSpec(hand="claw")
    with pytest.raises(ValueError):
        EAtlasSpec(height=-1.0)
    with pytest.raises(ValueError):
        EAtlasSpec(span=0.1)  # narrower than the shoulders


# ---------------------------------------------------------------------------
# URDF structure
# ---------------------------------------------------------------------------


def test_urdf_is_well_formed_xml(urdf):
    assert urdf.tag == "robot"
    assert urdf.get("name") == "eatlas_approx"


def test_urdf_carries_the_unofficial_disclaimer(spec):
    text = build_urdf(spec)
    assert "UNOFFICIAL" in text
    assert "NOT a Boston Dynamics asset" in text


def test_kinematic_tree_has_exactly_one_root(urdf):
    links = {l.get("name") for l in urdf.findall("link")}
    children = {j.find("child").get("link") for j in urdf.findall("joint")}
    roots = links - children
    assert roots == {"pelvis"}


def test_every_joint_references_existing_links(urdf):
    links = {l.get("name") for l in urdf.findall("link")}
    for j in urdf.findall("joint"):
        assert j.find("parent").get("link") in links, j.get("name")
        assert j.find("child").get("link") in links, j.get("name")


def test_every_link_is_reachable_from_the_root(urdf):
    edges = [
        (j.find("parent").get("link"), j.find("child").get("link"))
        for j in urdf.findall("joint")
    ]
    seen, frontier = {"pelvis"}, ["pelvis"]
    while frontier:
        node = frontier.pop()
        for parent, child in edges:
            if parent == node and child not in seen:
                seen.add(child)
                frontier.append(child)
    assert seen == {l.get("name") for l in urdf.findall("link")}


def test_no_link_has_two_parents(urdf):
    """A URDF must be a tree; a duplicated child silently creates a cycle."""
    children = [j.find("child").get("link") for j in urdf.findall("joint")]
    assert len(children) == len(set(children))


# ---------------------------------------------------------------------------
# Physical plausibility - importers accept nonsense here and solvers explode later
# ---------------------------------------------------------------------------


def test_all_masses_are_positive(urdf):
    for link in urdf.findall("link"):
        m = float(link.find("inertial/mass").get("value"))
        assert m > 0.0, link.get("name")


def test_inertias_are_positive_and_physically_realisable(spec):
    """Principal moments must be positive and satisfy the triangle inequality,
    or PhysX produces NaNs the moment the body is touched."""
    links, _ = build_model(spec)
    for link in links:
        ixx, iyy, izz = inertia_tensor(link)
        assert min(ixx, iyy, izz) > 0.0, link.name
        assert ixx + iyy >= izz - 1e-12, link.name
        assert iyy + izz >= ixx - 1e-12, link.name
        assert izz + ixx >= iyy - 1e-12, link.name


def test_revolute_joints_have_a_valid_range(urdf):
    for j in urdf.findall("joint"):
        if j.get("type") != "revolute":
            continue
        limit = j.find("limit")
        assert limit is not None, j.get("name")
        assert float(limit.get("lower")) < float(limit.get("upper")), j.get("name")


def test_continuous_joints_declare_no_range(urdf):
    for j in urdf.findall("joint"):
        if j.get("type") != "continuous":
            continue
        limit = j.find("limit")
        assert limit.get("lower") is None and limit.get("upper") is None


def test_published_360_degree_joints_are_continuous(urdf):
    """Boston Dynamics states 360-degree motion at hip, waist and neck."""
    kinds = {j.get("name"): j.get("type") for j in urdf.findall("joint")}
    for name in ("waist_yaw_joint", "neck_yaw_joint", "l_hip_yaw_joint", "r_hip_yaw_joint"):
        assert kinds[name] == "continuous", name


def test_every_movable_joint_has_effort_and_velocity_limits(urdf):
    for j in urdf.findall("joint"):
        if j.get("type") == "fixed":
            continue
        limit = j.find("limit")
        assert float(limit.get("effort")) > 0, j.get("name")
        assert float(limit.get("velocity")) > 0, j.get("name")


def test_arms_are_seven_dof_each(urdf):
    for side in ("l", "r"):
        arm = [
            j for j in urdf.findall("joint")
            if j.get("type") != "fixed"
            and j.get("name").startswith(side + "_")
            and any(k in j.get("name") for k in ("shoulder", "elbow", "wrist"))
        ]
        assert len(arm) == 7, f"{side} arm has {len(arm)} DoF"


def test_legs_are_six_dof_each(urdf):
    for side in ("l", "r"):
        leg = [
            j for j in urdf.findall("joint")
            if j.get("type") != "fixed"
            and j.get("name").startswith(side + "_")
            and any(k in j.get("name") for k in ("hip", "knee", "ankle"))
        ]
        assert len(leg) == 6, f"{side} leg has {len(leg)} DoF"


# ---------------------------------------------------------------------------
# Integration with the rest of the pipeline
# ---------------------------------------------------------------------------


def test_preset_validates_once_an_asset_path_is_set():
    import dataclasses

    preset = dataclasses.replace(EATLAS_APPROX_PRESET, usd_path="assets/eatlas_approx.usd")
    preset.validate()  # must not raise


def test_preset_joint_patterns_match_the_generated_joint_names():
    import re

    _, joints = build_model(EAtlasSpec())
    names = [j.name for j in joints if j.jtype != "fixed"]
    for group in ("right_arm", "left_arm", "torso", "legs"):
        patterns = [re.compile(p) for p in EATLAS_APPROX_PRESET.group(group)]
        assert any(p.fullmatch(n) for n in names for p in patterns), group


def test_preset_covers_every_movable_joint():
    """A joint no group matches gets no actuator and silently goes limp."""
    import re

    _, joints = build_model(EAtlasSpec())
    patterns = [
        re.compile(p) for group in EATLAS_APPROX_PRESET.joint_groups.values() for p in group
    ]
    for j in joints:
        if j.jtype == "fixed":
            continue
        assert any(p.fullmatch(j.name) for p in patterns), f"{j.name} is in no group"


def test_preset_reach_is_single_arm_not_span():
    assert EATLAS_APPROX_PRESET.arm_reach == pytest.approx(EAtlasSpec().arm_total)
    assert EATLAS_APPROX_PRESET.nominal_pelvis_height == pytest.approx(EAtlasSpec().hip_z)


def test_the_default_masonry_wall_is_reachable_by_this_robot():
    from masonry_rl.tasks.masonry.wall_planner import WallPlanner, WallSpec

    planner = WallPlanner(WallSpec(), num_envs=1)
    pos, _ = planner.all_slot_poses()
    worst = pos[:, :2].norm(dim=-1).max().item()
    assert worst <= EATLAS_APPROX_PRESET.arm_reach


def test_forward_kinematics_places_shoulders_at_the_published_height():
    s = EAtlasSpec()
    frames = link_frames(s)
    for side in ("l", "r"):
        z = frames[f"{side}_shoulder_pitch_link"][2]
        assert z == pytest.approx(s.shoulder_z, abs=1e-6)


def test_feet_are_separated_laterally():
    frames = link_frames(EAtlasSpec())
    separation = abs(frames["l_foot"][1] - frames["r_foot"][1])
    assert 0.15 < separation < 0.5, "implausible stance width"


def test_hands_hang_below_the_shoulders_at_zero_configuration():
    frames = link_frames(EAtlasSpec())
    assert frames["l_hand"][2] < frames["l_shoulder_pitch_link"][2]
    assert frames["l_hand"][2] > 0.0, "hands should not be underground"
