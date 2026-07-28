#!/usr/bin/env python3
"""Batch evaluation over 100 episodes; writes the metrics table and failure clips.

    python scripts/eval_wall.py --checkpoint ... --episodes 100

Reports section 10 of the plan: placement success, position/yaw RMSE, joint
thickness distribution, course levelness, collapse rate, fall rate, minimum CoM
margin, squat depth range, peak seating force, cycle time, wall completion.

Metric definitions do not change between the floor-level target and the
``base_courses=3`` fallback - completion is always counted over the 12 bricks
the policy places - so the two conditions stay directly comparable.
"""

raise NotImplementedError("P3: evaluation harness")
