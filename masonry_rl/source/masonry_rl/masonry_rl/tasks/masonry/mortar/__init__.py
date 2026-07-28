"""Mortar models.

``reduced_model`` (Track A) is used for training and is imported here.
``pbd_demo`` (Track B) is demo-only and must NOT be imported by the training
path - it is loaded explicitly by ``scripts/record_demo.py``.
"""

from .reduced_model import MortarCfg, MortarField

__all__ = ["MortarCfg", "MortarField"]
