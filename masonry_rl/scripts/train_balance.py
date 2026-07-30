#!/usr/bin/env python3
"""P4: train the layer-1 balance policy, then freeze it.

    python scripts/train_balance.py --num_envs 4096 --headless

No bricks, no mortar, so this is the cheap phase (~1 h on an A6000). The
resulting checkpoint is passed to the masonry env as ``balance_checkpoint`` and
is never fine-tuned afterwards.

Gate before moving on: pelvis command tracking error under a 120 N push and a
3 kg payload at 0.7 m must stay within ``tracking_tolerance``, and the fall rate
must be zero. If the disturbance range here does not cover what the arm does
during masonry, P5 will walk straight out of this policy's competence and the
failure will look like a manipulation problem.
"""

raise NotImplementedError("P4: balance env + PPO runner")
