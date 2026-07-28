"""Boston Dynamics Atlas preset (DRC generation).

Asset situation as of 2026-07 (see docs/isaaclab-masonry-rl-plan.md section 1):

* The **electric** Atlas has no public URDF/USD. Boston Dynamics trains it in
  Isaac Lab with in-house assets that are not distributed.
* Isaac Lab's own asset registry ships Spot as the only Boston Dynamics robot.
* ``robot_descriptions`` carries exactly two Atlas entries, both DRC-era
  hydraulic: ``atlas_drc_description`` (v3, from Drake) and
  ``atlas_v4_description`` (v4, from roboschool).

So P0 uses the DRC model. Joint groups are regex based, which is what lets the
same environment run on a 30-DoF DRC Atlas today and a 56-DoF electric Atlas
later without touching the task code.

P0 must still: convert URDF -> USD, audit joint limits and inertia tensors,
simplify collision meshes to convex hulls, and attach a gripper (the DRC model
has no hand).
"""

from __future__ import annotations

from .preset import ActuatorGroup, RobotPreset

__all__ = ["ATLAS_DRC_PRESET", "resolve_urdf_path"]


def resolve_urdf_path(variant: str = "drc") -> str:
    """Locate the Atlas URDF via the ``robot_descriptions`` package.

    ``pip install robot_descriptions`` then::

        python -c "from masonry_rl.robots.atlas import resolve_urdf_path; print(resolve_urdf_path())"

    Args:
        variant: ``"drc"`` for Atlas DRC v3 (Drake), ``"v4"`` for Atlas v4
            (roboschool). P0 should download both and keep whichever has the
            better joint count and mesh quality.
    """
    if variant == "drc":
        from robot_descriptions import atlas_drc_description as desc
    elif variant == "v4":
        from robot_descriptions import atlas_v4_description as desc
    else:
        raise ValueError(f"unknown Atlas variant: {variant!r}")
    return desc.URDF_PATH


# Joint name patterns follow the DRC Atlas convention:
#   back_bkz/bky/bkx, {l,r}_arm_shz/shx/ely/elx/wry/wrx/wry2,
#   {l,r}_leg_hpz/hpx/hpy/kny/aky/akx, neck_ry
# The patterns are deliberately loose so v3 (6-DoF arms) and v5 (7-DoF arms)
# both match without edits.
ATLAS_DRC_PRESET = RobotPreset(
    name="atlas_drc",
    usd_path=None,  # set in P0 after URDF -> USD conversion
    urdf_path=None,  # filled by resolve_urdf_path() at conversion time
    joint_groups={
        "right_arm": ["r_arm_.*"],
        "left_arm": ["l_arm_.*"],
        "torso": ["back_bk.*"],
        "legs": ["[lr]_leg_.*"],
        "neck": ["neck_.*"],
        # Added in P0; the DRC model ships without hands.
        "right_gripper": ["r_gripper_.*"],
        "left_gripper": ["l_gripper_.*"],
    },
    ee_frames={"right": "r_hand", "left": "l_hand"},
    pelvis_body="pelvis",
    foot_bodies=["l_foot", "r_foot"],
    contact_bodies=["l_foot", "r_foot", "r_hand", "l_hand"],
    # Measure these in P0 and correct them; the values below are from published
    # specs (1.88 m tall, ~175 kg) and are starting points only.
    nominal_pelvis_height=0.95,
    arm_reach=0.80,
    default_joint_pos={
        "back_bk.*": 0.0,
        "[lr]_leg_hpy": -0.30,
        "[lr]_leg_kny": 0.60,
        "[lr]_leg_aky": -0.30,
        "[lr]_arm_shx": 0.0,
        "[lr]_arm_ely": 1.20,
    },
    actuators={
        # Hydraulic actuation approximated by implicit PD. Retune during P4
        # against the balance policy - these gains are a starting point.
        "arms": ActuatorGroup(["[lr]_arm_.*"], stiffness=400.0, damping=40.0),
        "torso": ActuatorGroup(["back_bk.*"], stiffness=800.0, damping=80.0),
        "legs": ActuatorGroup(["[lr]_leg_.*"], stiffness=1500.0, damping=150.0),
        "neck": ActuatorGroup(["neck_.*"], stiffness=20.0, damping=2.0),
        "grippers": ActuatorGroup(["[lr]_gripper_.*"], stiffness=200.0, damping=10.0),
    },
    gripper_open=0.04,
    gripper_closed=0.0,
    pelvis_height_range=(-0.45, 0.05),  # deep squat to slightly taller
    pelvis_pitch_range=(-0.35, 0.35),
    pelvis_shift_range=(-0.12, 0.12),
)
