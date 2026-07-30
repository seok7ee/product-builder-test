"""Unit tests for the reduced-order mortar model. No Isaac Sim required.

These check the *causal structure* the policy has to learn, which is the only
thing this model claims to reproduce.
"""

from __future__ import annotations

import pytest
import torch

from masonry_rl.tasks.masonry.mortar import MortarCfg, MortarField
from masonry_rl.tasks.masonry.wall_planner import WallPlanner, WallSpec


def make_field(num_envs=2, num_slots=12, **cfg_kwargs):
    cfg = MortarCfg(**cfg_kwargs)
    return MortarField(num_envs=num_envs, num_slots=num_slots, cfg=cfg)


def test_starts_empty():
    f = make_field()
    idx = torch.zeros(2, dtype=torch.long)
    assert f.coverage(idx).tolist() == [0.0, 0.0]
    assert f.mean_thickness(idx).tolist() == [0.0, 0.0]
    assert not f.cured.any()


def test_deposit_only_reaches_cells_near_the_nozzle():
    f = make_field(num_envs=1)
    idx = torch.zeros(1, dtype=torch.long)
    # eight cells spread 1 m apart along x, nozzle parked over the first one
    cells = torch.zeros(1, 8, 3)
    cells[0, :, 0] = torch.arange(8, dtype=torch.float32)
    nozzle = torch.tensor([[0.0, 0.0, 0.0]])

    f.deposit(idx, nozzle, cells, rate=torch.ones(1), dt=0.1)

    th = f.thickness[0, 0]
    assert th[0] > 0.0
    assert torch.all(th[1:] == 0.0)


def test_deposit_accumulates_and_is_capped():
    f = make_field(num_envs=1, max_deposit=0.02)
    idx = torch.zeros(1, dtype=torch.long)
    cells = torch.zeros(1, 8, 3)
    nozzle = torch.zeros(1, 3)

    for _ in range(100):
        f.deposit(idx, nozzle, cells, rate=torch.ones(1), dt=0.05)

    assert f.mean_thickness(idx).item() == pytest.approx(0.02)
    assert f.coverage(idx).item() == 1.0


def test_extrusion_rate_scales_the_deposit():
    f = make_field(num_envs=2)
    idx = torch.zeros(2, dtype=torch.long)
    cells = torch.zeros(2, 8, 3)
    nozzle = torch.zeros(2, 3)

    f.deposit(idx, nozzle, cells, rate=torch.tensor([1.0, 0.5]), dt=0.1)
    th = f.mean_thickness(idx)
    assert th[0].item() == pytest.approx(2 * th[1].item())


def test_pressing_compresses_the_bed_to_the_gap_and_squeezes_out():
    f = make_field(num_envs=1)
    idx = torch.zeros(1, dtype=torch.long)
    cells = torch.zeros(1, 8, 3)
    nozzle = torch.zeros(1, 3)
    f.deposit(idx, nozzle, cells, rate=torch.ones(1), dt=0.5)  # 0.03 -> capped
    before = f.mean_thickness(idx).item()
    assert before > 0.010

    joint = f.press(idx, brick_bottom_z=torch.tensor([0.010]), bed_base_z=torch.zeros(1))

    assert joint.item() == pytest.approx(0.010)
    assert f.mean_thickness(idx).item() == pytest.approx(0.010)
    assert f.squeeze_out[0, 0].item() > 0.0


def test_pressing_lightly_leaves_a_thick_joint_and_no_squeeze_out():
    f = make_field(num_envs=1)
    idx = torch.zeros(1, dtype=torch.long)
    cells = torch.zeros(1, 8, 3)
    f.deposit(idx, torch.zeros(1, 3), cells, rate=torch.ones(1), dt=0.2)  # 0.012
    deposited = f.mean_thickness(idx).item()

    joint = f.press(idx, brick_bottom_z=torch.tensor([0.020]), bed_base_z=torch.zeros(1))

    assert joint.item() == pytest.approx(0.020)
    assert not f.joint_ok(joint).item()  # outside the 8-12 mm band
    assert f.mean_thickness(idx).item() == pytest.approx(deposited)
    assert f.squeeze_out[0, 0].item() == pytest.approx(0.0)


def test_joint_acceptance_band():
    f = make_field(num_envs=4)
    joints = torch.tensor([0.007, 0.008, 0.012, 0.013])
    assert f.joint_ok(joints).tolist() == [False, True, True, False]


def test_squeeze_penalty_only_beyond_the_budget():
    f = make_field(num_envs=1, squeeze_budget=1e-5)
    idx = torch.zeros(1, dtype=torch.long)
    assert f.squeeze_penalty(idx).item() == 0.0
    f.squeeze_out[0, 0] = 5e-5
    assert f.squeeze_penalty(idx).item() == pytest.approx(4e-5)


def test_curing_requires_both_stability_and_coverage():
    f = make_field(num_envs=1, cure_steps=3)
    idx = torch.zeros(1, dtype=torch.long)
    stable = torch.ones(1, dtype=torch.bool)

    # no mortar -> never cures no matter how stable
    for _ in range(10):
        assert not f.tick_cure(idx, stable).item()

    f.deposit(idx, torch.zeros(1, 3), torch.zeros(1, 8, 3), torch.ones(1), dt=0.2)
    assert not f.tick_cure(idx, stable).item()
    assert not f.tick_cure(idx, stable).item()
    assert f.tick_cure(idx, stable).item()


def test_disturbance_resets_the_cure_timer():
    f = make_field(num_envs=1, cure_steps=3)
    idx = torch.zeros(1, dtype=torch.long)
    f.deposit(idx, torch.zeros(1, 3), torch.zeros(1, 8, 3), torch.ones(1), dt=0.2)

    f.tick_cure(idx, torch.ones(1, dtype=torch.bool))
    f.tick_cure(idx, torch.ones(1, dtype=torch.bool))
    f.tick_cure(idx, torch.zeros(1, dtype=torch.bool))  # bumped
    assert f.cure_counter[0, 0].item() == 0.0
    assert not f.cured[0, 0].item()


def test_bed_quality_peaks_at_the_target_deposit():
    f = make_field(num_envs=3, target_deposit=0.012)
    idx = torch.zeros(3, dtype=torch.long)
    f.thickness[0, 0] = 0.012  # on target
    f.thickness[1, 0] = 0.030  # over-extruded
    f.thickness[2, 0] = 0.005  # thin but covered

    q = f.bed_quality(idx)
    assert q[0] > q[1]
    assert q[0] > q[2]
    assert q[0].item() == pytest.approx(1.0, abs=1e-6)


def test_partial_coverage_lowers_bed_quality():
    f = make_field(num_envs=2)
    idx = torch.zeros(2, dtype=torch.long)
    f.thickness[0, 0, :] = 0.012
    f.thickness[1, 0, :4] = 0.012  # only half the bed

    assert f.coverage(idx).tolist() == [1.0, 0.5]
    q = f.bed_quality(idx)
    assert q[0] > q[1]


def test_slots_are_independent():
    f = make_field(num_envs=1)
    f.deposit(torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1, 8, 3), torch.ones(1), 0.2)
    assert f.coverage(torch.tensor([0])).item() == 1.0
    assert f.coverage(torch.tensor([1])).item() == 0.0


def test_reset_clears_only_the_selected_environments():
    f = make_field(num_envs=2)
    for e in range(2):
        f.thickness[e, 0] = 0.012
        f.cured[e, 0] = True

    f.reset_idx(torch.tensor([0]))

    assert f.thickness[0].sum().item() == 0.0
    assert not f.cured[0, 0].item()
    assert f.thickness[1, 0].sum().item() > 0.0
    assert f.cured[1, 0].item()


def test_observation_shape_and_content():
    f = make_field(num_envs=2)
    idx = torch.zeros(2, dtype=torch.long)
    obs = f.observation(idx)
    assert obs.shape == (2, MortarCfg().num_cells + 1)
    assert torch.all(obs == 0.0)


def test_integrates_with_the_wall_planner():
    """Nozzle sweeping a real slot's bed should cover it end to end."""
    sp = WallSpec(base_courses=3)
    planner = WallPlanner(sp, num_envs=1)
    f = make_field(num_envs=1, num_slots=sp.num_slots)

    idx = torch.zeros(1, dtype=torch.long)
    cells = planner.slot_cell_positions(idx, num_cells=f.cfg.num_cells)

    # sweep the nozzle along the cells, depositing as it goes
    for c in range(cells.shape[1]):
        f.deposit(idx, cells[:, c, :], cells, rate=torch.ones(1), dt=0.05)

    assert f.coverage(idx).item() == 1.0
    # the bed sits on top of the 3 pre-laid courses
    bed = planner.bed_base_height(idx)
    assert bed.item() == pytest.approx(3 * sp.course_pitch, abs=1e-6)

    joint = f.press(idx, brick_bottom_z=bed + sp.joint_thickness, bed_base_z=bed)
    assert f.joint_ok(joint).all()
