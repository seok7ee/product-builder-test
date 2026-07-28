#!/usr/bin/env python3
"""Train the masonry policy.

    # P3, pelvis welded
    python scripts/train.py --task Isaac-Masonry-Wall-Atlas-UpperBody-v0 \
        --num_envs 2048 --headless

    # P5/P6, on top of the frozen balance policy
    python scripts/train.py --task Isaac-Masonry-Wall-Atlas-WholeBody-v0 \
        --num_envs 512 --headless

Always ``--headless``. Rendering during training costs a large fraction of
throughput and nobody watches it; use ``play.py`` to look at a checkpoint.

Log the curriculum level distribution and the fall rate. A curriculum that
silently stops advancing is the most common way these runs waste a night, and
it is invisible on a reward curve alone.

This is a thin wrapper: Isaac Lab's own
``scripts/reinforcement_learning/rsl_rl/train.py`` already does the work, and
importing ``masonry_rl.tasks.masonry`` is what registers the environment ids.
"""

raise NotImplementedError(
    "P2: copy Isaac Lab's rsl_rl train.py and import masonry_rl.tasks.masonry"
)
