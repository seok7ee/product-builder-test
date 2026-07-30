"""Unitree G1 fallback preset.

Exists so the pipeline is never blocked on the Atlas asset: G1 ships with Isaac
Lab, so P1-P6 can proceed on it and switch to Atlas by changing one config
field. Keeping this preset alive and tested is the cheap insurance described in
section 1 of the plan.
"""

from __future__ import annotations

from .preset import ActuatorGroup, RobotPreset

__all__ = ["G1_PRESET"]

G1_PRESET = RobotPreset(
    name="unitree_g1",
    # Isaac Lab ships this asset; resolve through ISAACLAB_NUCLEUS_DIR in P0.
    usd_path=None,
    joint_groups={
        "right_arm": ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
        "left_arm": ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
        "torso": ["waist_.*"],
        "legs": ["(left|right)_(hip|knee|ankle)_.*"],
        "right_gripper": ["right_(hand|gripper)_.*"],
        "left_gripper": ["left_(hand|gripper)_.*"],
    },
    ee_frames={"right": "right_wrist_yaw_link", "left": "left_wrist_yaw_link"},
    pelvis_body="pelvis",
    foot_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
    contact_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
    nominal_pelvis_height=0.74,
    arm_reach=0.55,
    default_joint_pos={
        "(left|right)_hip_pitch_joint": -0.20,
        "(left|right)_knee_joint": 0.42,
        "(left|right)_ankle_pitch_joint": -0.23,
        "(left|right)_shoulder_pitch_joint": 0.20,
    },
    actuators={
        "arms": ActuatorGroup(
            ["(left|right)_(shoulder|elbow|wrist)_.*"], stiffness=80.0, damping=4.0
        ),
        "torso": ActuatorGroup(["waist_.*"], stiffness=150.0, damping=6.0),
        "legs": ActuatorGroup(
            ["(left|right)_(hip|knee|ankle)_.*"], stiffness=150.0, damping=6.0
        ),
        "grippers": ActuatorGroup(
            ["(left|right)_(hand|gripper)_.*"], stiffness=40.0, damping=2.0
        ),
    },
    gripper_open=0.03,
    gripper_closed=0.0,
    pelvis_height_range=(-0.30, 0.04),
    pelvis_pitch_range=(-0.30, 0.30),
    pelvis_shift_range=(-0.09, 0.09),
)
