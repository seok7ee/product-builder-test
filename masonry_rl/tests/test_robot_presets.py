"""Preset tests. These guard the robot-agnostic property, which is the whole
reason the Atlas asset risk stays contained to P0.
"""

from __future__ import annotations

import re

import pytest

from masonry_rl.robots import ATLAS_DRC_PRESET, G1_PRESET, PRESETS, RobotPreset


ALL = [ATLAS_DRC_PRESET, G1_PRESET]  # eatlas_approx ships a urdf_path, tested separately


@pytest.mark.parametrize("preset", ALL, ids=lambda p: p.name)
def test_presets_declare_everything_the_task_needs(preset):
    # usd_path is filled in during P0, so validate the rest explicitly
    for group in RobotPreset.REQUIRED_GROUPS:
        assert group in preset.joint_groups, f"{preset.name} missing {group}"
    assert "right" in preset.ee_frames
    assert preset.pelvis_body
    assert len(preset.foot_bodies) == 2
    assert preset.nominal_pelvis_height > 0
    assert preset.arm_reach > 0


@pytest.mark.parametrize("preset", ALL, ids=lambda p: p.name)
def test_validate_rejects_a_preset_without_an_asset(preset):
    """Both presets still need their USD path resolved in P0."""
    with pytest.raises(ValueError, match="usd_path or urdf_path"):
        preset.validate()


@pytest.mark.parametrize("preset", ALL, ids=lambda p: p.name)
def test_joint_groups_are_regex_not_fixed_indices(preset):
    """DoF-agnosticism: groups must compile as regex and none may be a bare
    index list. This is what lets a 30-DoF DRC Atlas and a 56-DoF electric
    Atlas share the same task code."""
    for name, patterns in preset.joint_groups.items():
        assert isinstance(patterns, list) and patterns, name
        for pattern in patterns:
            assert isinstance(pattern, str)
            re.compile(pattern)  # raises if malformed


def test_atlas_arm_patterns_match_both_six_and_seven_dof_arms():
    """v3 has 6-DoF arms, v5 has 7 (the extra wry2). One pattern, both models."""
    pattern = re.compile(ATLAS_DRC_PRESET.group("right_arm")[0])
    v3 = ["r_arm_shz", "r_arm_shx", "r_arm_ely", "r_arm_elx", "r_arm_wry", "r_arm_wrx"]
    v5 = v3 + ["r_arm_wry2"]
    assert all(pattern.fullmatch(j) for j in v5)
    assert len([j for j in v5 if pattern.fullmatch(j)]) == 7
    assert len([j for j in v3 if pattern.fullmatch(j)]) == 6
    # and it must not swallow the other arm
    assert not pattern.fullmatch("l_arm_shz")


def test_leg_pattern_matches_both_legs_only():
    pattern = re.compile(ATLAS_DRC_PRESET.group("legs")[0])
    assert pattern.fullmatch("l_leg_kny")
    assert pattern.fullmatch("r_leg_akx")
    assert not pattern.fullmatch("back_bkz")
    assert not pattern.fullmatch("r_arm_shz")


@pytest.mark.parametrize("preset", ALL, ids=lambda p: p.name)
def test_presets_are_bimanual(preset):
    """The mortar nozzle lives on the left arm from P6 onward."""
    assert preset.is_bimanual


@pytest.mark.parametrize("preset", ALL, ids=lambda p: p.name)
def test_pelvis_command_ranges_allow_squatting_down(preset):
    """The lower bound must be negative or the robot can never reach the floor."""
    low, high = preset.pelvis_height_range
    assert low < 0.0 < high or (low < 0.0 and high >= 0.0)
    assert abs(low) > 0.2, "not enough squat range to reach a floor-level course"


def test_group_lookup_raises_a_useful_error():
    with pytest.raises(KeyError, match="third_arm"):
        ATLAS_DRC_PRESET.group("third_arm")


def test_registry_exposes_both_presets():
    assert set(PRESETS) == {"atlas_drc", "eatlas_approx", "unitree_g1"}


def test_simulator_free_import_still_works():
    """The pure modules must import without Isaac Lab, or CI cannot run."""
    from masonry_rl.tasks.masonry import ISAACLAB_AVAILABLE
    from masonry_rl.tasks.masonry.mortar import MortarField  # noqa: F401
    from masonry_rl.tasks.masonry.wall_planner import WallPlanner  # noqa: F401

    assert ISAACLAB_AVAILABLE is False  # no Isaac Sim in this environment
