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
