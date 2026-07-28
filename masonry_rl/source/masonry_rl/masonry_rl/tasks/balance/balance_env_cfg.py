"""Layer 1: the balance policy, pre-trained and then frozen.

Trained on its own, without bricks or mortar, so it is cheap (4096 envs, ~1 h on
an A6000). Its job is narrow:

    given a pelvis command (height, pitch, fore-aft shift, lateral shift),
    hold it while an external wrench tries to push you over.

The manipulation policy then treats it as part of the plant. Doing this the
other way round - one policy for balance and millimetre placement together -
is what Boston Dynamics does for Atlas, and it needs far more compute than a
single workstation GPU.

The disturbance curriculum matters: the wrenches applied here must cover what
the arm actually does during masonry (a 2.4 kg brick at ~0.7 m reach, plus
30-80 N of seating force), or the manipulation phase will walk straight out of
the balance policy's competence.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...robots import ATLAS_DRC_PRESET, RobotPreset


@configclass
class BalanceEnvCfg:
    robot: RobotPreset = ATLAS_DRC_PRESET

    num_envs: int = 4096
    env_spacing: float = 2.5
    decimation: int = 4
    episode_length_s: float = 20.0

    # command ranges, must cover the masonry task's pelvis action range
    height_range: tuple[float, float] = (-0.45, 0.05)
    pitch_range: tuple[float, float] = (-0.35, 0.35)
    shift_range: tuple[float, float] = (-0.12, 0.12)

    # disturbance curriculum - sized from the masonry task, not guessed
    max_push_force: float = 120.0
    """Covers 80 N of seating force plus margin."""
    max_payload_mass: float = 3.0
    """A brick plus the gripper."""
    payload_offset: float = 0.70
    """Lever arm at full reach [m]."""

    tracking_tolerance: float = 0.02
    """P4 gate: pelvis command tracking error under load [m]."""
