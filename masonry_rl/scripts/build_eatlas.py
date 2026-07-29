#!/usr/bin/env python3
"""Generate the electric Atlas approximation: URDF + an SVG preview.

    python scripts/build_eatlas.py                       # parallel gripper, 34 DoF
    python scripts/build_eatlas.py --hand five_finger    # full 56 DoF
    python scripts/build_eatlas.py --height 1.5 --mass 89

Runs without Isaac Sim - it is stdlib XML and arithmetic.

The model is an UNOFFICIAL approximation from published specs, not a Boston
Dynamics asset. See ``masonry_rl.robots.eatlas_approx`` for what that means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "masonry_rl"))

from masonry_rl.robots.eatlas_approx import (  # noqa: E402
    EAtlasSpec,
    build_model,
    build_urdf,
    geometry_report,
    link_frames,
)

PALETTE = {
    "torso": "#4a5568", "pelvis": "#4a5568", "head": "#2d3748",
    "arm": "#3182ce", "hand": "#2b6cb0",
    "leg": "#38a169", "foot": "#276749",
    "other": "#718096",
}


def _colour(name: str) -> str:
    if "hand" in name or "gripper" in name:
        return PALETTE["hand"]
    if "arm" in name:
        return PALETTE["arm"]
    if "foot" in name:
        return PALETTE["foot"]
    if any(k in name for k in ("thigh", "shank")):
        return PALETTE["leg"]
    for key in ("torso", "pelvis", "head"):
        if key in name:
            return PALETTE[key]
    return PALETTE["other"]


def render_svg(spec: EAtlasSpec, width: int = 760, height: int = 520) -> str:
    """Front and side elevation at zero configuration.

    Frames are pure translations at zero configuration, so every link projects
    to an axis-aligned rectangle - no rotation maths needed for the preview.
    """
    links, _ = build_model(spec)
    frames = link_frames(spec)
    by_name = {l.name: l for l in links}
    report = geometry_report(spec)

    margin, gap = 40, 40
    panel_w = (width - 2 * margin - gap) / 2
    scale = (height - 2 * margin - 30) / spec.height
    ground = height - margin

    def rect(view: str, link, ox: float) -> str:
        fx, fy, fz = frames[link.name]
        cx, cy, cz = link.origin
        if link.geom == "box":
            dx, dy, dz = link.size
        else:  # cylinder along z
            radius, length = link.size
            dx = dy = 2 * radius
            dz = length
        # front view looks along -x -> project (y, z); side view -> (x, z)
        if view == "front":
            u, du = fy + cy, dy
            px = ox + panel_w / 2 - (u + du / 2) * scale  # +y is left
        else:
            u, du = fx + cx, dx
            px = ox + panel_w / 2 + (u - du / 2) * scale
        py = ground - (fz + cz + dz / 2) * scale
        return (
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{du * scale:.1f}" '
            f'height="{dz * scale:.1f}" rx="2" fill="{_colour(link.name)}" '
            f'fill-opacity="0.85" stroke="#1a202c" stroke-width="0.6"/>'
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#f7fafc"/>',
    ]

    order = ["l_", "r_"]  # draw far side first so the near side overlaps
    drawn = sorted(
        (l for l in links if l.geom != "none"),
        key=lambda l: order.index(l.name[:2]) if l.name[:2] in order else 0.5,
        reverse=True,
    )

    for view, ox in (("front", margin), ("side", margin + panel_w + gap)):
        parts.append(
            f'<line x1="{ox:.0f}" y1="{ground:.0f}" x2="{ox + panel_w:.0f}" '
            f'y2="{ground:.0f}" stroke="#a0aec0" stroke-width="1.5"/>'
        )
        for link in drawn:
            parts.append(rect(view, link, ox))
        # joint markers
        for name, (fx, fy, fz) in frames.items():
            if not name.endswith("_link"):
                continue
            u = fy if view == "front" else fx
            px = ox + panel_w / 2 + (-u if view == "front" else u) * scale
            parts.append(
                f'<circle cx="{px:.1f}" cy="{ground - fz * scale:.1f}" r="1.8" '
                f'fill="#e53e3e" fill-opacity="0.8"/>'
            )
        label = "FRONT" if view == "front" else "SIDE"
        parts.append(
            f'<text x="{ox + panel_w / 2:.0f}" y="{margin - 14:.0f}" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="#2d3748">{label}</text>'
        )

    parts.append(
        f'<text x="{margin}" y="{height - 12}" font-size="11" fill="#4a5568">'
        f'eatlas_approx (UNOFFICIAL approximation from published specs) &#183; '
        f'{report["height"]:.2f} m &#183; {report["mass"]:.1f} kg &#183; '
        f'{int(report["dof"])} DoF &#183; span {report["span"]:.2f} m &#183; '
        f'arm reach {report["arm_reach"]:.2f} m</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--height", type=float, default=1.90)
    p.add_argument("--mass", type=float, default=90.0)
    p.add_argument("--span", type=float, default=2.30)
    p.add_argument("--hand", choices=("parallel", "five_finger"), default="parallel")
    p.add_argument("--out-dir", default=str(ROOT / "assets"))
    args = p.parse_args()

    spec = EAtlasSpec(height=args.height, mass=args.mass, span=args.span, hand=args.hand)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    urdf_path = out / "eatlas_approx.urdf"
    svg_path = out / "eatlas_approx.svg"
    urdf_path.write_text(build_urdf(spec))
    svg_path.write_text(render_svg(spec))

    r = geometry_report(spec)
    print("eatlas_approx - UNOFFICIAL approximation, not a Boston Dynamics asset\n")
    print(f"  {'measured':<14}{'target':>10}")
    print(f"  {'-' * 24}")
    print(f"  height  {r['height']:6.3f} m {spec.height:8.2f}")
    print(f"  mass    {r['mass']:6.2f} kg{spec.mass:8.1f}")
    print(f"  span    {r['span']:6.3f} m {spec.span:8.2f}")
    print(f"  DoF     {int(r['dof']):6d}   {'56' if args.hand == 'five_finger' else '34':>8}")
    print(f"\n  single-arm reach {r['arm_reach']:.3f} m   (the 2.3 m figure is span)")
    print(f"  links {int(r['num_links'])}, feet on z={r['foot_bottom']:+.4f}")
    print(f"\n  {urdf_path}\n  {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
