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

## Why the visual meshes are generated, not sourced

Stock Atlas models on 3D marketplaces were considered and rejected on two
grounds.

**Licence.** 3DModels.org's product licence states you "may not modify, copy,
reproduce, distribute or use the content for any purpose without prior written
consent", and warns that a model "may contain third party copyrights or
trademarks" whose use must be agreed with the owners. Committing such a mesh
into this repository is redistribution, and the trademark position on a
likeness of a real commercial robot is not ours to assume. Other marketplaces
carry comparable terms.

**It would not work anyway.** A stock visual model is one unarticulated shell.
A URDF needs the body cut into ~30 links with joint frames defined between
them, which is manual work in a 3D tool, not something a mesh download saves.

So `masonry_rl.robots.meshes` generates the visual geometry from the same spec
that generates the kinematics, informed only by *published descriptions* of the
design language (circular head with integrated lights, deliberately simplified
part count, mostly fully rotational joints - hence the actuator barrels). No
third-party geometry is involved.

Collision geometry stays as convex primitives. Triangle-mesh colliders at
hundreds of parallel environments are the fastest way to make the scene
unusable, and URDF has always kept `<visual>` and `<collision>` separate.

## Rigging a supplied mesh

`scripts/rig_static_mesh.py` cuts a single unarticulated humanoid OBJ into
per-link meshes on the generated skeleton. Nothing is downloaded; you supply
the file and satisfy yourself its licence permits the use. Outputs land in
`assets/rigged/`, which is gitignored so a derivative of a licensed mesh is not
committed by accident.

Two findings from building it, both measured on a round trip through the
generated model:

**Segment in A-pose, not rest pose.** With the arms hanging at the sides the
hands sit centimetres from the thighs, and proximity cannot separate them. That
single ambiguity produced *every* kinematically distant misassignment - 12.2%
of faces, hands and grippers landing on thighs. Abducting the shoulders 40
degrees removes it completely: distant errors go to zero. It also matches how
humanoid models ship.

**Score distance in units of the bone's own girth.** Three metrics were tried:

| metric | exact | stranded on unrelated parts |
|---|---|---|
| raw segment distance | 68.8% | limbs eat the trunk |
| distance minus radius | 60.5% | fat bones win everywhere |
| distance / radius (kept) | 72.6% | 0% in A-pose |

The residual ~27% is joint-collar geometry assigned to the neighbour across the
joint. That is ambiguous by construction - a collar spanning the knee belongs
to thigh and shank equally - so the test asserts *no face lands on an unrelated
body part* rather than chasing an exact-match percentage.
