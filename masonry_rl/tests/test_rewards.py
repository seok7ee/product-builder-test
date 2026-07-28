"""Reward tests: pure shaping math plus structural checks on the weight table.

The structural half exists because weights and terms drifted apart once already
(``excess_force`` and ``self_collision`` carried weights with nothing behind
them). That class of bug is invisible in training - the run just underperforms.
"""

from __future__ import annotations

import math

import pytest
import torch

from masonry_rl.tasks.masonry.mdp import rewards, rewards_core as core
from masonry_rl.tasks.masonry.mdp.reward_weights import (
    BUILTIN,
    POSITIVE_TERMS,
    REWARD_WEIGHTS,
    TERM_BINDINGS,
)


# ---------------------------------------------------------------------------
# Shaping kernels
# ---------------------------------------------------------------------------


def test_tanh_kernel_peaks_at_zero_and_decays():
    d = torch.tensor([0.0, 0.05, 0.5, 5.0])
    r = core.tanh_kernel(d, std=0.1)
    assert r[0].item() == pytest.approx(1.0)
    assert torch.all(r[1:] < r[:-1])
    assert r[-1].item() < 0.01


def test_tanh_kernel_rejects_a_nonpositive_std():
    with pytest.raises(ValueError):
        core.tanh_kernel(torch.zeros(1), std=0.0)


def test_reach_reward_is_higher_when_closer():
    ee = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[0.01, 0.0, 0.0], [0.5, 0.0, 0.0]])
    r = core.reach_reward(ee, target)
    assert r[0] > r[1]


# ---------------------------------------------------------------------------
# Alignment - the gate is the important part
# ---------------------------------------------------------------------------


def test_align_reward_is_zero_without_a_grasp():
    """Otherwise the policy farms alignment by waving an empty gripper."""
    pos = torch.zeros(1, 3)
    yaw = torch.zeros(1)
    r = core.align_reward(pos, pos, yaw, yaw, grasped=torch.zeros(1, dtype=torch.bool))
    assert r.item() == 0.0


def test_align_reward_rewards_position_and_yaw():
    grasped = torch.ones(3, dtype=torch.bool)
    brick = torch.zeros(3, 3)
    slot = torch.tensor([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.0, 0.0, 0.0]])
    brick_yaw = torch.tensor([0.0, 0.0, math.pi / 2])
    slot_yaw = torch.zeros(3)

    r = core.align_reward(brick, slot, brick_yaw, slot_yaw, grasped)
    assert r[0] > r[1], "far from the slot should score lower"
    assert r[0] > r[2], "yaw misalignment should score lower"


def test_align_yaw_term_is_bounded():
    grasped = torch.ones(2, dtype=torch.bool)
    pos = torch.zeros(2, 3)
    r = core.align_reward(
        pos, pos, torch.tensor([0.0, math.pi]), torch.zeros(2), grasped, yaw_weight=0.5
    )
    assert r[0].item() == pytest.approx(1.5)  # aligned: 1.0 + 0.5 * 1.0
    assert r[1].item() == pytest.approx(1.0)  # opposed: 1.0 + 0.5 * 0.0


# ---------------------------------------------------------------------------
# Seating force
# ---------------------------------------------------------------------------


def test_force_band_saturates_inside_and_decays_outside():
    f = torch.tensor([0.0, 30.0, 55.0, 80.0, 150.0])
    r = core.force_band(f, lower=30.0, upper=80.0)
    assert r[1].item() == pytest.approx(1.0)
    assert r[2].item() == pytest.approx(1.0)
    assert r[3].item() == pytest.approx(1.0)
    assert r[0] < 0.3 and r[4] < 0.05


def test_force_band_rejects_an_inverted_band():
    with pytest.raises(ValueError):
        core.force_band(torch.zeros(1), lower=80.0, upper=30.0)


def test_excess_over_is_one_sided():
    v = torch.tensor([50.0, 120.0, 200.0])
    assert core.excess_over(v, 120.0).tolist() == [0.0, 0.0, 80.0]


def test_in_band_matches_the_joint_tolerance():
    v = torch.tensor([0.007, 0.008, 0.010, 0.012, 0.013])
    assert core.in_band(v, 0.008, 0.012).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


def _feet(y=0.12):
    return torch.tensor([[[0.0, y], [0.0, -y]]])  # [1, 2, 2]


def test_com_inside_the_support_polygon_has_a_positive_margin():
    m = core.support_margin(torch.zeros(1, 2), _feet(), torch.ones(1, 2, dtype=torch.bool))
    assert m.item() > 0.0


def test_com_outside_the_support_polygon_has_a_negative_margin():
    com = torch.tensor([[0.60, 0.0]])
    m = core.support_margin(com, _feet(), torch.ones(1, 2, dtype=torch.bool))
    assert m.item() < 0.0


def test_lifting_a_foot_shrinks_the_support_polygon():
    com = torch.tensor([[0.0, 0.10]])
    both = core.support_margin(com, _feet(), torch.ones(1, 2, dtype=torch.bool))
    one = core.support_margin(
        com, _feet(), torch.tensor([[True, False]])
    )
    assert one.item() < both.item()


def test_no_foot_contact_is_treated_as_airborne():
    m = core.support_margin(torch.zeros(1, 2), _feet(), torch.zeros(1, 2, dtype=torch.bool))
    assert m.item() == -1.0


def test_com_penalty_is_zero_while_comfortable_and_grows_past_the_edge():
    margin = torch.tensor([0.20, 0.05, 0.0, -0.10])
    p = core.com_margin_penalty(margin, safe_margin=0.05)
    assert p[0].item() == 0.0
    assert p[1].item() == pytest.approx(0.0)
    assert p[2] > 0 and p[3] > p[2], "gradient must keep pointing back to safety"


def test_foot_slip_counts_only_feet_in_contact():
    vel = torch.tensor([[[0.5, 0.0], [0.5, 0.0]]])
    both = core.foot_slip_magnitude(vel, torch.ones(1, 2, dtype=torch.bool))
    one = core.foot_slip_magnitude(vel, torch.tensor([[True, False]]))
    airborne = core.foot_slip_magnitude(vel, torch.zeros(1, 2, dtype=torch.bool))
    assert both.item() == pytest.approx(1.0)
    assert one.item() == pytest.approx(0.5)
    assert airborne.item() == 0.0


def test_fall_detection_is_relative_to_the_robots_own_height():
    """An absolute threshold tuned on Atlas would mark a shorter robot as
    permanently fallen. This is the robot-swap trap."""
    upright = torch.tensor([-1.0])
    atlas_h, g1_h = 0.95, 0.74

    # G1 standing at its own nominal height is not fallen
    assert not core.fall_indicator(torch.tensor([g1_h]), g1_h, upright).item()
    # ...but it would be, if judged against Atlas's nominal height
    assert core.fall_indicator(torch.tensor([g1_h]), atlas_h * 2, upright).item()


def test_fall_detection_triggers_on_collapse_and_on_tilt():
    upright = torch.tensor([-1.0])
    tipped = torch.tensor([-math.cos(1.0)])  # ~57 degrees
    assert core.fall_indicator(torch.tensor([0.50]), 0.95, upright).item()
    assert core.fall_indicator(torch.tensor([0.95]), 0.95, tipped).item()
    assert not core.fall_indicator(torch.tensor([0.95]), 0.95, upright).item()


def test_posture_deviation_is_zero_at_the_nominal_pose():
    q = torch.zeros(2, 6)
    assert core.posture_deviation(q, q).tolist() == [0.0, 0.0]
    q2 = q.clone()
    q2[1, 0] = 1.0
    assert core.posture_deviation(q2, q)[1].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Wall integrity
# ---------------------------------------------------------------------------


def test_collapse_ignores_slots_that_are_not_built_yet():
    ref = torch.zeros(1, 3, 3)
    pos = ref.clone()
    pos[0, 2, 0] = 0.5  # a big drift, but on an unplaced slot
    mask = torch.tensor([[True, True, False]])
    assert not core.collapse_indicator(pos, ref, mask).item()

    mask = torch.tensor([[True, True, True]])
    assert core.collapse_indicator(pos, ref, mask).item()


def test_collapse_respects_the_threshold():
    ref = torch.zeros(1, 2, 3)
    pos = ref.clone()
    pos[0, 0, 0] = 0.005
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert not core.collapse_indicator(pos, ref, mask, threshold=0.01).item()
    assert core.collapse_indicator(pos, ref, mask, threshold=0.001).item()


# ---------------------------------------------------------------------------
# Structural: the weight table and the terms must agree
# ---------------------------------------------------------------------------


def test_every_weight_has_a_binding_and_vice_versa():
    assert set(REWARD_WEIGHTS) == set(TERM_BINDINGS)


def test_every_binding_resolves_to_a_real_function():
    for term, target in TERM_BINDINGS.items():
        if target.startswith(BUILTIN):
            continue  # provided by Isaac Lab, cannot resolve without it
        fn = getattr(rewards, target, None)
        assert callable(fn), f"{term} -> rewards.{target} is missing"


def test_reward_signs_match_their_intent():
    for term, weight in REWARD_WEIGHTS.items():
        if term in POSITIVE_TERMS:
            assert weight > 0, f"{term} should reward progress"
        else:
            assert weight < 0, f"{term} should be a penalty"


def test_release_stable_dominates_the_manipulation_terms():
    """Success is defined by the brick staying put, so nothing else may outrank
    it - otherwise the policy optimises a proxy."""
    others = {k: v for k, v in REWARD_WEIGHTS.items() if k in POSITIVE_TERMS and k != "release_stable"}
    assert REWARD_WEIGHTS["release_stable"] > max(others.values())


def test_falling_outweighs_everything_else():
    """A fall invalidates the episode. If it sits on the same scale as the
    placement rewards, toppling becomes an acceptable price for one more brick.
    """
    fall = abs(REWARD_WEIGHTS["fall"])
    largest_other = max(abs(v) for k, v in REWARD_WEIGHTS.items() if k != "fall")
    assert fall >= 4 * largest_other


def test_no_duplicate_bindings():
    targets = [t for t in TERM_BINDINGS.values() if not t.startswith(BUILTIN)]
    assert len(targets) == len(set(targets))
