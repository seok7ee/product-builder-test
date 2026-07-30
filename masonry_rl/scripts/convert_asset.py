#!/usr/bin/env python3
"""P0: fetch the Atlas URDF and convert it to USD.

    pip install robot_descriptions
    python scripts/convert_asset.py --variant drc --out assets/atlas_drc.usd

Then set ``usd_path`` on the preset. After conversion, audit before training:

* joint limits and inertia tensors (the DRC URDFs are old and have been
  reserialised many times)
* collision meshes -> convex hulls; the visual meshes are far too heavy to use
  as colliders at 512+ environments
* attach a gripper: the DRC model has no hand

Requires Isaac Sim, so this does not run on a CPU-only machine.
"""

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("drc", "v4"), default="drc")
    p.add_argument("--out", required=True)
    p.add_argument("--merge-fixed-joints", action="store_true", default=True)
    args = p.parse_args()

    from masonry_rl.robots.atlas import resolve_urdf_path

    urdf = resolve_urdf_path(args.variant)
    print(f"URDF: {urdf}")
    raise NotImplementedError("P0: drive isaaclab UrdfConverter, write args.out")


if __name__ == "__main__":
    raise SystemExit(main())
