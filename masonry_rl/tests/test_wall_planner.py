"""Unit tests for the wall planner. No Isaac Sim required."""

from __future__ import annotations

import math

import pytest
import torch

from masonry_rl.tasks.masonry.wall_planner import WallPlanner, WallSpec


def test_derived_geometry_matches_spec_document():
    sp = WallSpec()
    assert sp.course_pitch == pytest.approx(0.067)
    assert sp.column_pitch == pytest.approx(0.200)
    # 4 bricks + 3 joints
    assert sp.wall_length == pytest.approx(0.790)
    assert sp.num_slots == 12


@pytest.mark.parametrize(
    "base_courses, work_plane",
    [(0, 0.010), (3, 0.211), (6, 0.412), (12, 0.814)],
)
def test_work_plane_height_curriculum_table(base_courses, work_plane):
    """The base_courses -> working height table from the plan (section 2)."""
    sp = WallSpec(base_courses=base_courses)
    assert sp.work_plane_height() == pytest.approx(work_plane, abs=1e-9)


def test_top_height_of_three_courses():
    sp = WallSpec(num_courses=3, base_courses=0)
    # bed 10mm + 2 course pitches + one brick height
    assert sp.top_height() == pytest.approx(0.010 + 2 * 0.067 + 0.057)
    assert sp.top_height() == pytest.approx(0.201)


def test_fallback_configuration_builds_courses_four_to_six():
    sp = WallSpec(base_courses=3, num_courses=3)
    assert sp.num_base_bricks == 12
    assert sp.num_slots == 12
    assert sp.total_courses == 6
    assert sp.work_plane_height() == pytest.approx(0.211)
    assert sp.top_height() == pytest.approx(0.402)


def test_slot_ordering_is_bottom_up_left_to_right():
    sp = WallSpec(base_courses=0, yaw=0.0)
    planner = WallPlanner(sp, num_envs=1)
    pos, _ = planner.all_slot_poses()

    # first four slots share a height and increase in x
    z = pos[:, 2]
    assert torch.allclose(z[:4], z[0].expand(4))
    assert torch.all(pos[1:4, 0] > pos[0:3, 0])
    # course 1 sits exactly one course pitch above course 0
    assert (z[4] - z[0]).item() == pytest.approx(sp.course_pitch, abs=1e-6)


def test_running_bond_staggers_odd_courses_by_half_a_column():
    planner = WallPlanner(WallSpec(bond="running", yaw=0.0), num_envs=1)
    pos, _ = planner.all_slot_poses()
    offset = (pos[4, 0] - pos[0, 0]).item()
    assert offset == pytest.approx(0.5 * WallSpec().column_pitch, abs=1e-6)


def test_stack_bond_has_no_stagger():
    planner = WallPlanner(WallSpec(bond="stack", yaw=0.0), num_envs=1)
    pos, _ = planner.all_slot_poses()
    assert (pos[4, 0] - pos[0, 0]).item() == pytest.approx(0.0, abs=1e-6)


def test_base_bricks_sit_below_the_first_slot():
    sp = WallSpec(base_courses=3)
    planner = WallPlanner(sp, num_envs=1)
    base_pos, _ = planner.base_brick_poses()
    slot_pos, _ = planner.all_slot_poses()

    assert base_pos.shape == (12, 3)
    assert base_pos[:, 2].max().item() < slot_pos[:, 2].min().item()


def test_no_base_bricks_when_building_from_the_floor():
    planner = WallPlanner(WallSpec(base_courses=0), num_envs=1)
    base_pos, base_quat = planner.base_brick_poses()
    assert base_pos.shape == (0, 3)
    assert base_quat.shape == (0, 4)


def test_slot_pose_is_batched_per_environment():
    planner = WallPlanner(WallSpec(yaw=0.0), num_envs=4)
    idx = torch.tensor([0, 1, 5, 11])
    pos, quat = planner.slot_pose(idx)
    assert pos.shape == (4, 3)
    assert quat.shape == (4, 4)
    # identity yaw -> unit quaternion (w, x, y, z)
    assert torch.allclose(quat[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_env_origins_offset_every_slot():
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, -5.0, 0.0]])
    planner = WallPlanner(WallSpec(), num_envs=2, env_origins=origins)
    pos, _ = planner.slot_pose(torch.tensor([0, 0]))
    assert torch.allclose(pos[1] - pos[0], torch.tensor([10.0, -5.0, 0.0]))


def test_wall_yaw_rotates_the_layout():
    planner = WallPlanner(WallSpec(yaw=math.pi / 2, origin=(0.0, 0.0, 0.0)), num_envs=1)
    pos, quat = planner.all_slot_poses()
    # with a 90 degree yaw the wall runs along +y, so x stays near zero
    assert abs(pos[3, 0].item()) < 0.1
    assert pos[3, 1].item() > 0.5
    assert quat[0, 3].item() == pytest.approx(math.sin(math.pi / 4))


def test_mortar_cells_lie_on_the_bed_plane_under_the_brick():
    sp = WallSpec(base_courses=0, yaw=0.0)
    planner = WallPlanner(sp, num_envs=2)
    idx = torch.tensor([0, 4])
    cells = planner.slot_cell_positions(idx, num_cells=8)

    assert cells.shape == (2, 8, 3)
    # cell heights equal the bed base of their course
    assert cells[0, :, 2].allclose(torch.zeros(8), atol=1e-6)
    assert cells[1, :, 2].allclose(torch.full((8,), sp.course_pitch), atol=1e-6)
    # cells span the brick length, centered on the slot
    slot_pos, _ = planner.slot_pose(idx)
    spread = cells[0, :, 0].max() - cells[0, :, 0].min()
    assert spread.item() < sp.brick_length
    assert cells[0, :, 0].mean().item() == pytest.approx(slot_pos[0, 0].item(), abs=1e-6)


def test_bed_base_height_matches_cell_height():
    sp = WallSpec(base_courses=2)
    planner = WallPlanner(sp, num_envs=1)
    idx = torch.tensor([0])
    bed = planner.bed_base_height(idx)
    cells = planner.slot_cell_positions(idx, num_cells=8)
    assert bed.item() == pytest.approx(cells[0, 0, 2].item(), abs=1e-6)
    # building on top of 2 pre-laid courses -> bed sits at 2 * pitch
    assert bed.item() == pytest.approx(2 * sp.course_pitch, abs=1e-6)


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        WallSpec(bond="herringbone")
    with pytest.raises(ValueError):
        WallSpec(base_courses=-1)
    with pytest.raises(ValueError):
        WallSpec(num_courses=0)


def test_default_layout_is_within_a_humanoid_arm_reach():
    """The robot is stationary, so every slot must be reachable from one spot.

    A wall running along +x puts its far end ~1.4 m out; the default lays it
    laterally across the frontal plane instead.
    """
    from masonry_rl.robots import ATLAS_DRC_PRESET

    planner = WallPlanner(WallSpec(), num_envs=1)
    pos, _ = planner.all_slot_poses()
    worst = pos[:, :2].norm(dim=-1).max().item()
    assert worst <= ATLAS_DRC_PRESET.arm_reach, f"furthest slot at {worst:.3f} m"


def test_default_wall_is_laid_across_the_frontal_plane():
    planner = WallPlanner(WallSpec(), num_envs=1)
    pos, _ = planner.all_slot_poses()
    # spans in y, stays at a roughly constant standoff in x
    assert (pos[:, 1].max() - pos[:, 1].min()).item() > 0.5
    assert (pos[:, 0].max() - pos[:, 0].min()).item() < 0.2
    assert pos[:, 0].min().item() > 0.0  # in front of the robot
