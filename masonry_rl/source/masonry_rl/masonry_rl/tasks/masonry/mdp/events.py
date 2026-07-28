"""Reset and randomisation events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv


def reset_wall(env: "ManagerBasedRLEnv", env_ids: Sequence[int]) -> None:
    """Respawn the wall for the environment's current curriculum level.

    Pre-built ``base_courses`` bricks are placed at
    ``planner.base_brick_poses()`` and set kinematic - they are cured masonry,
    so they neither fall nor cost contact solver time. Remaining bricks are
    parked below the floor and teleported in as their slot comes up.
    """
    raise NotImplementedError("P2: base brick spawn + kinematic flag + mortar reset")


def randomize_pallet(env: "ManagerBasedRLEnv", env_ids: Sequence[int]) -> None:
    """Pallet pose jitter, +/-5 cm and +/-15 deg from L1 onward."""
    raise NotImplementedError("P2: pallet root pose randomisation")


def randomize_physics(env: "ManagerBasedRLEnv", env_ids: Sequence[int]) -> None:
    """Domain randomisation for L7.

    Foot friction has a hard floor of 0.6: below that the robot slips and falls
    for reasons the policy cannot act on, which poisons the balance signal.
    """
    raise NotImplementedError("P2: friction, mass, brick dimensions, sensor noise")
