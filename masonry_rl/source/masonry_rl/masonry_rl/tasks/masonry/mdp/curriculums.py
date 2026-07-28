"""Curriculum terms.

The whole height curriculum is one integer: ``base_courses``, the number of
pre-built courses spawned under the wall. Lowering it raises the balance
difficulty continuously, from "standing at a chest-high wall" down to "deep
squat at floor level".

    base_courses  work plane   posture
    12            0.82 m       standing
     6            0.41 m       moderate bend
     3            0.21 m       shallow squat   <- L5-STOP, the fallback demo
     0            0.01 m       deep squat      <- final target

Two rules that matter:

* the level is **per environment**, not global. Dropping every environment at
  once is a difficulty step change and the policy collapses.
* the fallback is not a separate code path. If L6 never converges, freeze the
  schedule at ``base_courses = 3`` and the same policy, evaluation harness and
  renderer produce the demo unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["BASE_COURSE_SCHEDULE", "base_course_descent", "placement_tolerance"]


BASE_COURSE_SCHEDULE: tuple[int, ...] = (12, 12, 12, 12, 12, 6, 3, 0)
"""Levels L0..L7 from section 5.5 of the plan.

L0-L4 all sit at 12 (manipulation is learned at a comfortable height, then the
pelvis is released). L5 walks 12 -> 6 -> 3. L6 goes to 0. L7 keeps 0 and turns
on domain randomisation.
"""

L5_STOP_INDEX = 6
"""Index into ``BASE_COURSE_SCHEDULE`` holding ``base_courses = 3``.

Set ``freeze_at`` to this to lock in the fallback demo condition.
"""


def base_course_descent(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    success_threshold: float = 0.70,
    window: int = 50,
    freeze_at: int | None = None,
) -> torch.Tensor:
    """Advance each environment's level once its success rate clears the bar.

    Args:
        success_threshold: placement success rate required to promote.
        window: episodes averaged before a promotion is considered.
        freeze_at: optional hard cap on the level index. Set to
            :data:`L5_STOP_INDEX` to stop at 3 pre-built courses.

    Returns:
        Mean level across environments, for logging. **Log this.** A curriculum
        that silently stops advancing is the most common way these runs fail,
        and it is invisible unless the level distribution is on the dashboard.
    """
    raise NotImplementedError(
        "P2: per-env success-rate buffer -> level += 1 -> respawn base courses"
    )


def placement_tolerance(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    start: float = 0.005,
    end: float = 0.003,
) -> torch.Tensor:
    """Tighten the placement tolerance from 5 mm to 3 mm during L7."""
    raise NotImplementedError("P2: interpolate tolerance with the level index")
