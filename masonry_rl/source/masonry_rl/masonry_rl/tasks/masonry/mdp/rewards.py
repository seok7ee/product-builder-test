"""Reward terms for the masonry task.

Weights live in ``masonry_env_cfg.py``; this module only defines the shapes of
the signals. Two invariants from the plan drive the design:

1. ``release_stable`` is the dominant *manipulation* term. Success is "the brick
   is still there 30 steps after the gripper opened", not "the brick touched the
   target pose". Rewarding the instant of placement produces walls that fall
   over.
2. ``fall`` is the dominant *balance* term and is an order of magnitude larger
   than anything else. A falling humanoid invalidates the whole episode, so it
   must not be trade-able against one more brick.

STATUS: signatures are final; bodies marked TODO need the Isaac Lab data API
verified against the pinned version during P0/P2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv


# -- shaping helpers --------------------------------------------------------


def _tanh_kernel(distance: torch.Tensor, std: float) -> torch.Tensor:
    """Standard Isaac Lab shaping: 1 at zero distance, decaying with ``std``."""
    return 1.0 - torch.tanh(distance / std)


# -- approach and grasp -----------------------------------------------------


def reach_brick(env: "ManagerBasedRLEnv", std: float = 0.10) -> torch.Tensor:
    """Dense approach toward the brick to be picked up."""
    raise NotImplementedError("P2: distance from right EE frame to target brick")


def grasp_brick(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Both fingers in contact and the gripper commanded closed."""
    raise NotImplementedError("P2: finger contact sensors AND gripper command")


def lift_brick(env: "ManagerBasedRLEnv", min_height: float = 0.05) -> torch.Tensor:
    """Constant bonus once the brick clears its rest height."""
    raise NotImplementedError("P2: brick height above its spawn height")


# -- mortar -----------------------------------------------------------------


def mortar_bed(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Bed coverage scaled by how close the deposit is to the target thickness.

    Thin wrapper over :meth:`MortarField.bed_quality` so the reward manager owns
    only the weight, not the model.
    """
    return env.mortar.bed_quality(env.active_slot)


def squeeze_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty (positive magnitude) for mortar displaced beyond the budget."""
    return env.mortar.squeeze_penalty(env.active_slot)


# -- alignment and seating --------------------------------------------------


def align_to_slot(env: "ManagerBasedRLEnv", std: float = 0.15) -> torch.Tensor:
    """Dense alignment of the carried brick with its target slot pose.

    Position kernel plus a yaw term; only active while the brick is grasped, so
    the policy cannot farm it by hovering an empty gripper over the wall.
    """
    raise NotImplementedError("P2: brick pose vs planner.slot_pose(active_slot)")


def seat_force(
    env: "ManagerBasedRLEnv", lower: float = 30.0, upper: float = 80.0
) -> torch.Tensor:
    """Reward for keeping the seating force inside the target band [N].

    A band rather than a target: too little and the mortar is not compressed,
    too much and it squeezes out. Returns 1 inside the band, decaying outside.
    """
    raise NotImplementedError("P2: wrist force/torque sensor, normal component")


def joint_thickness(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for a finished joint inside the 8-12 mm acceptance band.

    Second largest manipulation term because joint error accumulates upward:
    a thick joint in course 1 tilts every course above it.
    """
    return env.mortar.joint_ok(env.last_joint_thickness).float()


def release_stable(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """THE success term: the brick held position after the gripper opened.

    Delegates to the mortar model's cure timer, which only advances while the
    joint is both bonded and undisturbed.
    """
    return env.mortar.cured.gather(1, env.active_slot.unsqueeze(1)).squeeze(1).float()


# -- wall integrity ---------------------------------------------------------


def collapse(env: "ManagerBasedRLEnv", threshold: float = 0.01) -> torch.Tensor:
    """Penalty (positive magnitude) when an already-placed brick moves.

    Pre-built ``base_courses`` bricks are kinematic and are excluded.
    """
    raise NotImplementedError("P2: displacement of placed bricks vs their slot pose")


# -- balance (bipedal specific) ---------------------------------------------


def fall(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Fall indicator. Weighted at -200: an order above every other term."""
    raise NotImplementedError("P2: pelvis height collapse or torso tilt > 40 deg")


def com_margin(env: "ManagerBasedRLEnv", safe_margin: float = 0.05) -> torch.Tensor:
    """Penalty as the CoM ground projection approaches the support polygon edge.

    Support polygon is the convex hull of the contacting feet, so it shrinks
    automatically when the robot shifts weight.
    """
    raise NotImplementedError("P2: CoM projection vs foot contact hull")


def foot_slip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Horizontal speed of feet that are in contact. Slipping precedes falling."""
    raise NotImplementedError("P2: foot contact mask * planar foot velocity")


def posture_deviation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Deviation from the nominal pose; suppresses contorted solutions."""
    raise NotImplementedError("P2: joint_pos - default_joint_pos, L2 over groups")
