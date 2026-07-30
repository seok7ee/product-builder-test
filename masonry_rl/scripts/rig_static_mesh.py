#!/usr/bin/env python3
"""Rig a downloaded humanoid mesh onto the generated skeleton.

    python scripts/rig_static_mesh.py atlas.obj --up y --out-dir assets/rigged

Takes one unarticulated OBJ, cuts it into per-link meshes, writes them in each
link's local frame, and emits a URDF whose ``<visual>`` points at them. The
``<collision>`` geometry stays as convex primitives - a triangle-mesh collider
is the fastest way to make a many-environment scene unusable.

Nothing is downloaded. Supply your own file, and satisfy yourself that its
licence permits what you are doing with it: marketplace models commonly forbid
redistribution and may carry third-party trademark.

Two things decide whether the result is any good:

* **Pose.** Segment in A-pose (the default). Measured on a round trip, arms
  hanging at the sides put the hands within centimetres of the thighs and every
  stranded face came from that one ambiguity. Supply an A- or T-posed mesh,
  which is how humanoid models usually ship anyway.
* **The report.** Read it. A link with no faces, or a large mean bone distance,
  means the source did not match the skeleton - usually wrong pose, wrong
  scale, or a model with proportions far from the published Atlas figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "masonry_rl"))

from masonry_rl.robots.eatlas_approx import EAtlasSpec, build_model, build_urdf  # noqa: E402
from masonry_rl.robots.kinematics import A_POSE  # noqa: E402
from masonry_rl.robots.meshes import Mesh, to_obj  # noqa: E402
from masonry_rl.robots.rigging import (  # noqa: E402
    load_obj,
    normalise,
    posed_frames,
    split_into_links,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="OBJ file to rig (yours to supply)")
    p.add_argument("--up", choices=("y", "z"), default="z",
                   help="up axis of the source; modelling packages usually export y")
    p.add_argument("--forward", default="x", help="forward axis: x, y, -x, -y")
    p.add_argument("--height", type=float, default=1.90, help="target height [m]")
    p.add_argument("--pose", choices=("a", "rest"), default="a",
                   help="skeleton pose to segment against; match your source")
    p.add_argument("--out-dir", default=str(ROOT / "assets" / "rigged"))
    args = p.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found", file=sys.stderr)
        return 1

    spec = EAtlasSpec(height=args.height)
    all_links, joints = build_model(spec)
    links = [l for l in all_links if l.geom != "none"]
    frames = posed_frames(spec, joints, A_POSE if args.pose == "a" else {})

    soup = load_obj(source.read_text())
    print(f"loaded {len(soup.vertices)} vertices, {len(soup.faces)} faces")
    soup = normalise(soup, target_height=args.height, up=args.up, forward=args.forward)

    parts, report = split_into_links(soup, links, frames)

    out = Path(args.out_dir)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        mesh = Mesh(list(part.vertices), list(part.faces))
        (out / "meshes" / f"{name}.obj").write_text(to_obj(mesh, name))
    (out / "eatlas_rigged.urdf").write_text(build_urdf(spec, mesh_prefix="meshes/"))

    print()
    print(report.summary())
    worst_link, worst = report.worst_fit
    print()
    if report.empty_links:
        print(f"WARNING: no geometry reached {', '.join(report.empty_links)}.")
        print("         Check --up/--forward, and that the source is a full body.")
    if worst > 0.08:
        print(f"WARNING: {worst_link} sits {worst * 1000:.0f} mm off its bone.")
        print(f"         Usually a pose mismatch - try --pose {'rest' if args.pose == 'a' else 'a'}.")
    if not report.empty_links and worst <= 0.08:
        print(f"fit looks reasonable (worst: {worst_link} at {worst * 1000:.0f} mm)")
    print(f"\n  {out / 'eatlas_rigged.urdf'}\n  {out / 'meshes'}/  ({len(parts)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
