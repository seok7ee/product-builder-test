# Asset notes

Fill this in during P0. It is the record of what was actually obtained, which
matters because the Atlas asset situation is the one thing in this project that
cannot be worked around in code.

## Atlas

- [ ] `atlas_drc_description` (v3, Drake) — URDF path, joint count, mesh quality:
- [ ] `atlas_v4_description` (v4, roboschool) — URDF path, joint count, mesh quality:
- [ ] variant chosen and why:
- [ ] licence terms of the source repository:
- [ ] URDF -> USD conversion command and output path:
- [ ] joint limit / inertia audit findings:
- [ ] collider simplification (convex hull) before/after triangle counts:
- [ ] gripper attached (model, mount frame):
- [ ] measured values replacing the preset's estimates:
      - nominal_pelvis_height (currently 0.95, from published height)
      - arm_reach (currently 0.80, estimated)

## Isaac Lab

- [ ] version tag pinned:
- [ ] manager API fields verified against this version (PhysxCfg field names,
      ArticulationCfg.InitialStateCfg, ImplicitActuatorCfg):
- [ ] measured FPS on the reference humanoid task (corrects the wall-clock
      estimates in the plan, section 8.4):

## Bricks / pallet / nozzle

- [ ] brick.usd (190 x 90 x 57 mm, box collider, friction material):
- [ ] pallet.usd:
- [ ] nozzle.usd (left wrist mount):

---

## Electric Atlas: published-spec conflicts (recorded 2026-07)

No public URDF/USD exists. `masonry_rl.robots.eatlas_approx` generates an
unofficial approximation from published figures. Reporting disagrees, so the
values chosen are recorded here.

| field | value used | conflicting reports |
|---|---|---|
| height | 1.90 m | 1.52 m appears in several aggregator listings |
| mass | 90 kg | 89 kg |
| DoF | 56 | 28 appears in some listings |
| reach | 2.30 m **span** | often quoted as "reach"; single-arm reach is ~0.90 m |
| arms | 7 DoF each, 5-fingered hands | - |
| payload | 30 kg sustained / 50 kg burst | 66 lb / 110 lb, same numbers |
| joint range | 360 deg at hip, waist, neck | - |

The 1.52 m / 89 kg pairing looks like early-2024 reveal reporting; 1.90 m /
90 kg / 56 DoF is what recurs across CES 2026 production-machine coverage.
`EAtlasSpec` is parametric, so the alternative is one flag away:

    python scripts/build_eatlas.py --height 1.52 --mass 89

**The "2.3 m reach" figure is the trap.** It is fingertip to fingertip. Used
as single-arm reach it would pass a wall layout the robot cannot physically
touch, and the failure would show up only as a success rate that never climbs.

- [ ] revisit if Boston Dynamics publishes an official asset or spec sheet
- [ ] replace estimated inertias and joint limits if real data appears
