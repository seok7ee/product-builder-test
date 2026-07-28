#!/usr/bin/env python3
"""Print the wall layout for a given curriculum level. Runs without Isaac Sim.

Useful before touching the simulator: it shows exactly what the policy will be
asked to build at each ``base_courses`` setting, and flags slots that fall
outside the robot's reach - which is the failure mode that otherwise shows up
as an unexplained success-rate plateau, hours into training.

    python scripts/preview_wall.py --base-courses 3
    python scripts/preview_wall.py --schedule
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "masonry_rl"))

import torch  # noqa: E402

from masonry_rl.robots import PRESETS  # noqa: E402
from masonry_rl.tasks.masonry.wall_planner import WallPlanner, WallSpec  # noqa: E402


def describe(spec: WallSpec, reach: float | None) -> None:
    planner = WallPlanner(spec, num_envs=1)
    slot_pos, _ = planner.all_slot_poses()
    base_pos, _ = planner.base_brick_poses()

    print(f"  work plane      : {spec.work_plane_height() * 1000:7.1f} mm")
    print(f"  finished top    : {spec.top_height() * 1000:7.1f} mm")
    print(f"  pre-built bricks: {base_pos.shape[0]:3d}  ({spec.base_courses} courses)")
    print(f"  bricks to place : {slot_pos.shape[0]:3d}  ({spec.num_courses} courses)")
    print(f"  wall length     : {spec.wall_length * 1000:7.1f} mm")

    if reach is not None:
        planar = slot_pos[:, :2].norm(dim=-1)
        worst = planar.max().item()
        flag = "OK " if worst <= reach else "OUT OF REACH"
        print(f"  furthest slot   : {worst:7.3f} m vs reach {reach:.3f} m  [{flag}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-courses", type=int, default=0)
    p.add_argument("--courses", type=int, default=3)
    p.add_argument("--bricks-per-course", type=int, default=4)
    p.add_argument("--bond", choices=("running", "stack"), default="running")
    p.add_argument("--robot", choices=sorted(PRESETS), default="atlas_drc")
    p.add_argument(
        "--schedule",
        action="store_true",
        help="show every level of the base_courses curriculum instead of one",
    )
    args = p.parse_args()

    reach = PRESETS[args.robot].arm_reach
    levels = (12, 6, 3, 0) if args.schedule else (args.base_courses,)

    torch.set_printoptions(precision=4, sci_mode=False)
    for base in levels:
        spec = WallSpec(
            bricks_per_course=args.bricks_per_course,
            num_courses=args.courses,
            base_courses=base,
            bond=args.bond,
        )
        label = "FALLBACK DEMO" if base == 3 else ("FINAL TARGET" if base == 0 else "")
        print(f"\nbase_courses = {base:2d}   {label}")
        describe(spec, reach)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
