"""Pure reward math. No Isaac Sim, no environment object, fully unit tested.

Same split as the mortar model: the part that can be tested off-GPU is
separated from the part that needs a running simulator. Everything here takes
plain tensors and returns plain tensors, so the shaping logic - which is where
reward bugs actually live - is verified before anything touches Isaac Sim.

:mod:`rewards` holds the thin adapters that fetch state from the environment
and call into this module.
"""

from __future__ import annotations

import torch

__all__ = [
    "tanh_kernel",
    "reach_reward",
    "align_reward",
    "force_band",
    "in_band",
    "support_margin",
    "com_margin_penalty",
    "foot_slip_magnitude",
    "posture_deviation",
    "fall_indicator",
    "collapse_indicator",
    "excess_over",
]


# -- shaping kernels --------------------------------------------------------


def tanh_kernel(distance: torch.Tensor, std: float) -> torch.Tensor:
    """1 at zero distance, decaying with ``std``. Isaac Lab's standard shape."""
    if std <= 0.0:
        raise ValueError("std must be > 0")
    return 1.0 - torch.tanh(distance / std)


def reach_reward(ee_pos: torch.Tensor, target_pos: torch.Tensor, std: float = 0.10) -> torch.Tensor:
    """Dense approach toward a target, ``[E]``."""
    return tanh_kernel(torch.norm(ee_pos - target_pos, dim=-1), std)


def align_reward(
    brick_pos: torch.Tensor,
    slot_pos: torch.Tensor,
    brick_yaw: torch.Tensor,
    slot_yaw: torch.Tensor,
    grasped: torch.Tensor,
    std: float = 0.15,
    yaw_weight: float = 0.5,
) -> torch.Tensor:
    """Position and yaw alignment of the carried brick with its slot, ``[E]``.

    Gated on ``grasped``: without the gate a policy farms this by waving an
    empty gripper over the wall, which is a real and very common failure.
    """
    position = tanh_kernel(torch.norm(brick_pos - slot_pos, dim=-1), std)
    yaw = 0.5 * (torch.cos(brick_yaw - slot_yaw) + 1.0)  # [0, 1], pi-periodic safe
    return grasped.to(position.dtype) * (position + yaw_weight * yaw)


def force_band(
    force: torch.Tensor, lower: float = 30.0, upper: float = 80.0, softness: float = 20.0
) -> torch.Tensor:
    """1 inside ``[lower, upper]``, decaying outside it, ``[E]``.

    A band rather than a target: too little force and the mortar is not
    compressed, too much and it squeezes out. Rewarding a single set-point
    would make the policy chase a number the contact dynamics cannot hold.
    """
    if not lower < upper:
        raise ValueError("lower must be < upper")
    below = (lower - force).clamp(min=0.0)
    above = (force - upper).clamp(min=0.0)
    return torch.exp(-(below + above) / softness)


def in_band(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Float indicator for ``lower <= value <= upper``, ``[E]``."""
    return ((value >= lower) & (value <= upper)).to(value.dtype)


def excess_over(value: torch.Tensor, limit: float) -> torch.Tensor:
    """Positive magnitude above ``limit``, zero below. ``[E]``."""
    return (value - limit).clamp(min=0.0)


# -- balance ----------------------------------------------------------------


def support_margin(
    com_xy: torch.Tensor,
    foot_pos_xy: torch.Tensor,
    foot_contact: torch.Tensor,
    foot_half_extent: tuple[float, float] = (0.13, 0.06),
) -> torch.Tensor:
    """Signed distance from the CoM to the support polygon boundary, ``[E]``.

    Positive inside, negative outside. The polygon is the axis-aligned hull of
    the *contacting* feet grown by the foot half-extents, so it shrinks by
    itself when the robot lifts a foot or shifts weight - no separate
    single/double support case.

    Args:
        com_xy: ``[E, 2]`` ground projection of the centre of mass.
        foot_pos_xy: ``[E, F, 2]`` foot centres.
        foot_contact: ``[E, F]`` bool contact flags.
    """
    mask = foot_contact.bool().unsqueeze(-1)
    half = torch.tensor(foot_half_extent, device=com_xy.device, dtype=com_xy.dtype)

    big = torch.finfo(com_xy.dtype).max / 4
    lo = torch.where(mask, foot_pos_xy, torch.full_like(foot_pos_xy, big)).amin(dim=1) - half
    hi = torch.where(mask, foot_pos_xy, torch.full_like(foot_pos_xy, -big)).amax(dim=1) + half

    margin = torch.minimum(com_xy - lo, hi - com_xy).amin(dim=-1)
    airborne = ~foot_contact.bool().any(dim=1)
    return torch.where(airborne, torch.full_like(margin, -1.0), margin)


def com_margin_penalty(margin: torch.Tensor, safe_margin: float = 0.05) -> torch.Tensor:
    """Positive penalty magnitude as the CoM nears the support boundary, ``[E]``.

    Zero while the margin is comfortable, growing as it shrinks and continuing
    to grow once the CoM leaves the polygon - so the gradient still points back
    toward safety after the robot is already committed to falling.
    """
    return (safe_margin - margin).clamp(min=0.0)


def foot_slip_magnitude(foot_vel_xy: torch.Tensor, foot_contact: torch.Tensor) -> torch.Tensor:
    """Total horizontal speed of feet that are in contact, ``[E]``.

    Slipping reliably precedes falling, and it is observable much earlier.
    """
    speed = torch.norm(foot_vel_xy, dim=-1)
    return (speed * foot_contact.to(speed.dtype)).sum(dim=-1)


def fall_indicator(
    pelvis_height: torch.Tensor,
    nominal_height: float,
    projected_gravity_z: torch.Tensor,
    height_fraction: float = 0.60,
    max_tilt: float = 0.70,
) -> torch.Tensor:
    """Bool ``[E]``: pelvis collapsed, or torso tilted past ``max_tilt`` rad.

    Height is compared as a *fraction of nominal* rather than in metres, so the
    same threshold survives a robot swap. An absolute value tuned on a 1.88 m
    Atlas silently mislabels a 1.32 m G1 as permanently fallen.

    ``projected_gravity_z`` is gravity in the body frame: -1 when upright.
    """
    collapsed = pelvis_height < height_fraction * nominal_height
    tilted = -projected_gravity_z < torch.cos(
        torch.tensor(max_tilt, device=pelvis_height.device, dtype=pelvis_height.dtype)
    )
    return collapsed | tilted


# -- wall integrity ---------------------------------------------------------


def collapse_indicator(
    placed_pos: torch.Tensor,
    reference_pos: torch.Tensor,
    placed_mask: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """Bool ``[E]``: any already-placed brick drifted past ``threshold``.

    ``placed_mask`` excludes slots not yet built and the kinematic
    ``base_courses`` bricks, which cannot move by construction.
    """
    drift = torch.norm(placed_pos - reference_pos, dim=-1)
    return ((drift > threshold) & placed_mask.bool()).any(dim=-1)


def posture_deviation(joint_pos: torch.Tensor, default_joint_pos: torch.Tensor) -> torch.Tensor:
    """L2 deviation from the nominal pose, ``[E]``. Suppresses contorted poses."""
    return torch.norm(joint_pos - default_joint_pos, dim=-1)
