"""Single source of truth for reward term names, weights and bindings.

Weights used to live in a plain dataclass that nothing consumed, so terms and
weights could drift apart silently - and did: ``excess_force`` and
``self_collision`` carried weights with no function behind them. Keeping the
table here, in a module that imports nothing from Isaac Lab, means the
env config builds its ``RewTerm``s from it and the tests can check coverage
without a GPU.

Section 5.3 of the plan. Two entries are load-bearing:

* ``release_stable`` (+30) defines what a successful placement *is*. Success is
  "the brick is still there 30 steps after the gripper opened", not "the brick
  reached the target pose". Reward the instant of placement and you get walls
  that fall over.
* ``fall`` (-200) sits an order of magnitude above everything else. Put it on
  the same scale as the placement rewards and the policy learns that toppling
  is an acceptable price for one more brick.
"""

from __future__ import annotations

__all__ = ["REWARD_WEIGHTS", "TERM_BINDINGS", "BUILTIN", "term_names"]


BUILTIN = "isaaclab:"
"""Prefix marking a term provided by Isaac Lab's own mdp module."""


REWARD_WEIGHTS: dict[str, float] = {
    # -- manipulation ------------------------------------------------------
    "reach": 1.0,
    "grasp": 2.0,
    "lift": 5.0,
    "mortar_bed": 8.0,
    "align": 8.0,
    "seat_force": 6.0,
    "joint_thickness": 15.0,
    "release_stable": 30.0,
    # -- balance -----------------------------------------------------------
    "fall": -200.0,
    "com_margin": -5.0,
    "foot_slip": -2.0,
    "posture_deviation": -0.5,
    # -- wall integrity ----------------------------------------------------
    "collapse": -50.0,
    "squeeze_out": -5.0,
    # -- effort / safety regularisers --------------------------------------
    "excess_force": -0.1,
    "self_collision": -5.0,
    "action_rate": -0.01,
    "joint_vel": -0.01,
}


TERM_BINDINGS: dict[str, str] = {
    "reach": "reach_brick",
    "grasp": "grasp_brick",
    "lift": "lift_brick",
    "mortar_bed": "mortar_bed",
    "align": "align_to_slot",
    "seat_force": "seat_force",
    "joint_thickness": "joint_thickness",
    "release_stable": "release_stable",
    "fall": "fall",
    "com_margin": "com_margin",
    "foot_slip": "foot_slip",
    "posture_deviation": "posture_deviation",
    "collapse": "collapse",
    "squeeze_out": "squeeze_out",
    "excess_force": "excess_force",
    "self_collision": "self_collision",
    # standard regularisers, no reason to reimplement them
    "action_rate": f"{BUILTIN}action_rate_l2",
    "joint_vel": f"{BUILTIN}joint_vel_l2",
}


POSITIVE_TERMS = frozenset(
    {
        "reach",
        "grasp",
        "lift",
        "mortar_bed",
        "align",
        "seat_force",
        "joint_thickness",
        "release_stable",
    }
)
"""Terms that reward progress. Everything else is a penalty and must be
negative - a sign flip here is silent and ruins a whole training run."""


def term_names() -> tuple[str, ...]:
    return tuple(REWARD_WEIGHTS)
