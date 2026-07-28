"""Action terms.

Action layout (20-D at full scope, phased in):

    right arm     6   task-space delta, relative IK      P3+
    right gripper 1   binary                             P3+
    pelvis / CoM  4   height, pitch, fore-aft, lateral   P5+
    torso         2   pitch, yaw                         P5+
    left arm      6   nozzle task-space delta            P6+
    extrusion     1   [0, 1]                             P6+

The policy never commands the ~12 leg joints directly. It emits a 4-D pelvis
command that the frozen balance policy tracks. Collapsing the legs from 12
action dimensions to 4 is the single change that makes this learnable on one
GPU; a policy that has to discover balance and millimetre placement at the same
time does neither.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv


class PelvisCommandAction:
    """Bridges the policy's 4-D pelvis command to the frozen balance policy.

    Loads the balance checkpoint produced by ``scripts/train_balance.py``, runs
    it in inference mode, and writes its leg joint targets. The balance policy
    is frozen: no gradient flows back into it from the manipulation objective.
    """

    def __init__(self, cfg: "PelvisCommandActionCfg", env: "ManagerBasedRLEnv") -> None:
        raise NotImplementedError("P5: load frozen balance policy, wire leg targets")

    @property
    def action_dim(self) -> int:
        return 4

    def process_actions(self, actions) -> None:
        """Clip to the preset's pelvis ranges and store as the balance command."""
        raise NotImplementedError("P5")

    def apply_actions(self) -> None:
        """Run the frozen balance policy and set leg joint targets."""
        raise NotImplementedError("P5")


class PelvisCommandActionCfg:
    """Config for :class:`PelvisCommandAction`."""

    class_type = PelvisCommandAction
    asset_name: str = "robot"
    balance_checkpoint: str = MISSING
    """Path to the frozen balance policy from P4."""
    scale: tuple[float, float, float, float] = (0.02, 0.05, 0.02, 0.02)
    """Per-step delta scale: height [m], pitch [rad], fore-aft [m], lateral [m]."""
