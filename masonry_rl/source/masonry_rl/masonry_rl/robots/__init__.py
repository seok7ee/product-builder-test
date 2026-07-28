"""Robot-agnostic presets. See ``preset.RobotPreset``."""

from .preset import ActuatorGroup, RobotPreset
from .atlas import ATLAS_DRC_PRESET, resolve_urdf_path
from .g1 import G1_PRESET

PRESETS = {p.name: p for p in (ATLAS_DRC_PRESET, G1_PRESET)}

__all__ = [
    "ActuatorGroup",
    "RobotPreset",
    "ATLAS_DRC_PRESET",
    "G1_PRESET",
    "PRESETS",
    "resolve_urdf_path",
]
