"""Masonry task package.

Importing this package registers the gym environments **when Isaac Lab is
available**. When it is not - a CI runner, a laptop, any machine without a GPU -
the import degrades to a no-op so that the simulator-free parts of the package
(:mod:`wall_planner`, :mod:`mortar.reduced_model`, :mod:`masonry_rl.robots`)
stay importable and unit testable.

Keep it that way. Those two modules carry the geometry and the mortar dynamics,
which is exactly the code worth having tests on, and tests that need a GPU do
not get run.

Environment ids follow Isaac Lab's convention
``Isaac-<Task>-<Robot>-<Variant>-v0``. The variant mirrors the roadmap so a
checkpoint's provenance is obvious from its id.
"""

from __future__ import annotations

ISAACLAB_AVAILABLE = True
try:
    import gymnasium as gym

    from . import agents
    from .masonry_env_cfg import MasonryEnvCfg
except ImportError:  # pragma: no cover - exercised only without Isaac Lab
    ISAACLAB_AVAILABLE = False


def register_environments() -> None:
    """Register every masonry environment id. Called on import when possible."""
    _agents = f"{agents.__name__}.rsl_rl_ppo_cfg"
    _cfg = f"{__name__}.masonry_env_cfg:MasonryEnvCfg"
    common = {
        "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
        "disable_env_checker": True,
    }

    # P3: pelvis welded, manipulation only.
    gym.register(
        id="Isaac-Masonry-Wall-Atlas-UpperBody-v0",
        kwargs={
            "env_cfg_entry_point": _cfg,
            "rsl_rl_cfg_entry_point": f"{_agents}:MasonryUpperBodyPPOCfg",
        },
        **common,
    )

    # P5/P6: pelvis released on top of the frozen balance policy, bimanual.
    gym.register(
        id="Isaac-Masonry-Wall-Atlas-WholeBody-v0",
        kwargs={
            "env_cfg_entry_point": _cfg,
            "rsl_rl_cfg_entry_point": f"{_agents}:MasonryWholeBodyPPOCfg",
        },
        **common,
    )

    # Fallback demo condition: frozen at 3 pre-built courses (plan section 11.1).
    # Same environment, same policy, same metrics - only base_courses differs.
    gym.register(
        id="Isaac-Masonry-Wall-Atlas-Base3-v0",
        kwargs={
            "env_cfg_entry_point": _cfg,
            "rsl_rl_cfg_entry_point": f"{_agents}:MasonryWholeBodyPPOCfg",
        },
        **common,
    )


if ISAACLAB_AVAILABLE:
    register_environments()
    __all__ = ["MasonryEnvCfg", "register_environments", "ISAACLAB_AVAILABLE"]
else:
    __all__ = ["ISAACLAB_AVAILABLE"]
