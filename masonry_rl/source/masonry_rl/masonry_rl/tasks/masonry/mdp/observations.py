"""Observation terms.

Two groups. ``policy`` sees only what a real robot could estimate; ``critic``
additionally sees privileged state (true brick poses, friction, masses, true
CoM, applied disturbances). Asymmetric actor-critic matters a lot here: the
value function has to explain outcomes that depend on contact state the policy
cannot observe.

STATUS: signatures final, bodies need the Isaac Lab data API verified in P0/P2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv


# -- task state -------------------------------------------------------------

def slot_residual(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Remaining error from the carried brick to its target slot, [E, 9].

    Position error plus a 6-D rotation representation. This is the single most
    important observation: it is what "precision" is measured against.
    """
    raise NotImplementedError("P2: brick pose vs planner.slot_pose(active_slot)")


def brick_in_ee_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target brick pose expressed in the right end-effector frame, [E, 9]."""
    raise NotImplementedError("P2: relative transform, EE frame")


def mortar_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Bed thickness samples plus coverage for the active slot, [E, C+1]."""
    return env.mortar.observation(env.active_slot)


def wall_progress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Course index, slot index and completion fraction, normalised, [E, 6]."""
    raise NotImplementedError("P2: derived from active_slot and WallSpec")


# -- balance state (bipedal specific) ---------------------------------------

def pelvis_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Pelvis height, orientation (6-D) and twist, [E, 13]."""
    raise NotImplementedError("P2: root state of preset.pelvis_body")


def com_support_margin(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """CoM position inside the support polygon, normalised, plus margin, [E, 3].

    Normalised so the observation means the same thing in single and double
    support, and does not change scale when the robot changes stance.
    """
    raise NotImplementedError("P2: CoM projection vs foot contact hull")


def foot_contact_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-foot contact flag and force/torque, [E, 14]."""
    raise NotImplementedError("P2: contact sensor on preset.foot_bodies")


# -- privileged (critic only) -----------------------------------------------

def privileged_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """True brick poses, friction, masses, true CoM, placed-brick drift."""
    raise NotImplementedError("P2: privileged buffer assembly")
