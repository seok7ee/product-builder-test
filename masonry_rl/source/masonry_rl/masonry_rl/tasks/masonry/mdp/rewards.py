"""Reward terms: thin adapters over :mod:`rewards_core`.

Each term does two things - fetch state from the environment, then call a pure
function. The shaping logic lives in ``rewards_core`` where it is unit tested;
what remains here is plumbing that genuinely needs a running simulator, marked
with the phase that implements it.

Weights are not here. They live in :mod:`reward_weights`, which the env config
turns into ``RewTerm``s, so there is exactly one place a weight can be edited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import rewards_core as core

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# State access. Everything below this line needs Isaac Sim; everything the
# terms actually compute does not.
# ---------------------------------------------------------------------------


def _ee_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    raise NotImplementedError("P2: right EE frame position from FrameTransformer")


def _target_brick_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    raise NotImplementedError("P2: pose of the brick assigned to the active slot")


def _carried_brick(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(position [E,3], yaw [E], grasped [E] bool)``."""
    raise NotImplementedError("P2: carried brick pose + grasp state")


def _slot_target(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """``(position [E,3], yaw [E])`` from ``planner.slot_pose(active_slot)``."""
    raise NotImplementedError("P2: planner.slot_pose(env.active_slot)")


def _seating_force(env: "ManagerBasedRLEnv") -> torch.Tensor:
    raise NotImplementedError("P2: wrist F/T sensor, normal component [N]")


def _balance_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """``(pelvis_height [E], projected_gravity_z [E])``."""
    raise NotImplementedError("P2: pelvis root state + body-frame gravity")


def _com_and_feet(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(com_xy [E,2], foot_pos_xy [E,F,2], foot_vel_xy [E,F,2], contact [E,F])``."""
    raise NotImplementedError("P2: CoM from articulation, feet from contact sensors")


def _placed_bricks(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pos [E,S,3], reference [E,S,3], placed_mask [E,S])``.

    ``placed_mask`` must exclude the kinematic ``base_courses`` bricks: they
    cannot move, so including them only dilutes the signal.
    """
    raise NotImplementedError("P2: placed brick poses vs their slot targets")


def _joint_state(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """``(joint_pos [E,J], default_joint_pos [E,J])`` over the preset's groups."""
    raise NotImplementedError("P2: articulation joint state for the arm/torso groups")


def _self_collision_count(env: "ManagerBasedRLEnv") -> torch.Tensor:
    raise NotImplementedError("P2: self-collision contact count from the sensor")


# ---------------------------------------------------------------------------
# Manipulation
# ---------------------------------------------------------------------------


def reach_brick(env: "ManagerBasedRLEnv", std: float = 0.10) -> torch.Tensor:
    return core.reach_reward(_ee_pos(env), _target_brick_pos(env), std)


def grasp_brick(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Both fingers in contact *and* the gripper commanded closed."""
    raise NotImplementedError("P2: finger contact sensors AND gripper action")


def lift_brick(env: "ManagerBasedRLEnv", min_height: float = 0.05) -> torch.Tensor:
    raise NotImplementedError("P2: brick height above its spawn height")


def align_to_slot(env: "ManagerBasedRLEnv", std: float = 0.15) -> torch.Tensor:
    brick_pos, brick_yaw, grasped = _carried_brick(env)
    slot_pos, slot_yaw = _slot_target(env)
    return core.align_reward(brick_pos, slot_pos, brick_yaw, slot_yaw, grasped, std)


def seat_force(
    env: "ManagerBasedRLEnv", lower: float = 30.0, upper: float = 80.0
) -> torch.Tensor:
    return core.force_band(_seating_force(env), lower, upper)


def excess_force(env: "ManagerBasedRLEnv", limit: float = 120.0) -> torch.Tensor:
    """Positive magnitude of seating force above ``limit`` [N]."""
    return core.excess_over(_seating_force(env), limit)


# ---------------------------------------------------------------------------
# Mortar. These are real: the model they delegate to runs on CPU and is tested.
# ---------------------------------------------------------------------------


def mortar_bed(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Bed coverage scaled by how close the deposit is to the target thickness."""
    return env.mortar.bed_quality(env.active_slot)


def squeeze_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Positive magnitude of mortar displaced beyond the budget."""
    return env.mortar.squeeze_penalty(env.active_slot)


def joint_thickness(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Finished joint inside the 8-12 mm band.

    Second largest manipulation term, because joint error accumulates upward:
    a thick joint in course 1 tilts every course above it.
    """
    return env.mortar.joint_ok(env.last_joint_thickness).float()


def release_stable(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """THE success term: the brick held position after the gripper opened.

    Delegates to the mortar cure timer, which only advances while the joint is
    both bonded and undisturbed.
    """
    return env.mortar.cured.gather(1, env.active_slot.unsqueeze(1)).squeeze(1).float()


# ---------------------------------------------------------------------------
# Wall integrity
# ---------------------------------------------------------------------------


def collapse(env: "ManagerBasedRLEnv", threshold: float = 0.01) -> torch.Tensor:
    pos, reference, mask = _placed_bricks(env)
    return core.collapse_indicator(pos, reference, mask, threshold).float()


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


def fall(
    env: "ManagerBasedRLEnv", height_fraction: float = 0.60, max_tilt: float = 0.70
) -> torch.Tensor:
    """Fall indicator. Weighted at -200, an order above every other term."""
    height, gravity_z = _balance_state(env)
    nominal = env.cfg.robot.nominal_pelvis_height
    return core.fall_indicator(height, nominal, gravity_z, height_fraction, max_tilt).float()


def com_margin(env: "ManagerBasedRLEnv", safe_margin: float = 0.05) -> torch.Tensor:
    com_xy, foot_pos, _, contact = _com_and_feet(env)
    return core.com_margin_penalty(core.support_margin(com_xy, foot_pos, contact), safe_margin)


def foot_slip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _, _, foot_vel, contact = _com_and_feet(env)
    return core.foot_slip_magnitude(foot_vel, contact)


def posture_deviation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    joint_pos, default = _joint_state(env)
    return core.posture_deviation(joint_pos, default)


def self_collision(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Number of self-collision contacts this step."""
    return _self_collision_count(env).float()
