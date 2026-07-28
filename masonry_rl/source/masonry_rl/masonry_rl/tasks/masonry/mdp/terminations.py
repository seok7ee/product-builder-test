"""Termination terms.

Success does not end the episode: the slot index advances and the robot carries
on with the next brick. That is what makes the episode a *wall*, not a brick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv


def fallen(
    env: "ManagerBasedRLEnv", height_fraction: float = 0.60, max_tilt: float = 0.70
) -> torch.Tensor:
    """Pelvis dropped below a fraction of nominal, or torso tilted past ~40 deg.

    The height check is relative to ``preset.nominal_pelvis_height`` so it
    survives a robot swap; an absolute threshold would silently break on G1.
    """
    raise NotImplementedError("P2: pelvis height + gravity projection")


def brick_dropped(env: "ManagerBasedRLEnv", margin: float = 0.05) -> torch.Tensor:
    """Carried brick fell below its bed plane."""
    raise NotImplementedError("P2: brick height vs planner.bed_base_height")


def wall_collapsed(env: "ManagerBasedRLEnv", threshold: float = 0.01) -> torch.Tensor:
    """Any previously placed brick moved more than ``threshold``."""
    raise NotImplementedError("P2: placed brick displacement")


def excessive_force(env: "ManagerBasedRLEnv", limit: float = 200.0) -> torch.Tensor:
    """Wrist normal force above ``limit`` [N]."""
    raise NotImplementedError("P2: wrist F/T sensor")


def seated_without_mortar(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Brick seated on a bed below the minimum coverage.

    Without this the policy discovers that dry-stacking is faster and skips the
    mortar phase entirely.
    """
    return (env.mortar.coverage(env.active_slot) < env.mortar.cfg.min_coverage) & env.just_seated
