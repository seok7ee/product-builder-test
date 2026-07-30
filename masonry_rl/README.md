# masonry_rl

Bricklaying with a stationary bipedal humanoid in Isaac Lab.
Design document: [`../docs/isaaclab-masonry-rl-plan.md`](../docs/isaaclab-masonry-rl-plan.md).

This is a **P2 scaffold**. The layout, interfaces and every number from the plan
are in place; the manager terms that need a running simulator are explicit
`NotImplementedError` markers with the phase that fills them in.

## What already works, without Isaac Sim

Two modules carry the geometry and the mortar dynamics, and both are pure
`torch`. They are unit tested here, on CPU:

| module | what it does |
|---|---|
| `tasks/masonry/wall_planner.py` | slot poses, pre-built course poses, mortar cell positions |
| `tasks/masonry/mortar/reduced_model.py` | Track A mortar: deposit, press, squeeze-out, cure |
| `robots/preset.py` | robot-agnostic joint groups, frames, actuator gains |

```bash
pip install torch pytest
PYTHONPATH=source/masonry_rl python -m pytest tests -q     # 51 tests
python scripts/preview_wall.py --schedule
```

`preview_wall.py` prints the wall at every curriculum level and checks each slot
against the robot's reach. Run it before touching the simulator.

## What needs Isaac Sim

Everything under `tasks/masonry/mdp/`, the env configs, and every script except
`preview_wall.py`. Each raises `NotImplementedError` naming the phase.

## Three design decisions worth knowing before reading the code

**The policy learns one brick, not a wall.** `wall_planner.py` decides slot
order deterministically. The policy only ever sees "put this brick in that
slot". Learning the whole wall end to end does not work: long horizon, contact
rich, and sparse reward all at once.

**The legs are not in the action space.** The policy emits a 4-D pelvis command
(height, pitch, fore-aft, lateral) that a *frozen, separately trained* balance
policy tracks. Collapsing ~12 leg joints into 4 commands is what makes this fit
on one GPU. Boston Dynamics trains Atlas as a single whole-body policy; that
needs far more compute than a workstation.

**`base_courses` is one parameter doing two jobs.** It sets how many courses are
already built (spawned kinematic) under the wall, which fixes the working
height:

| `base_courses` | work plane | posture | role |
|---|---|---|---|
| 12 | 0.81 m | standing | L0-L4, learn manipulation cheaply |
| 6 | 0.41 m | moderate bend | L5 |
| **3** | **0.21 m** | shallow squat | **L5-STOP, fallback demo** |
| 0 | 0.01 m | deep squat | L6, final target |

The curriculum walks it down; if floor level never converges, freezing at 3 *is*
the fallback demo. Same environment, same policy, same metrics, no separate code
path.

## Layout

```
source/masonry_rl/masonry_rl/
  robots/           preset.py, atlas.py, g1.py     # swap robots via config
  tasks/
    balance/        layer 1: pre-trained, frozen
    masonry/        layer 2: the deliverable policy
      wall_planner.py           layer 3: deterministic, not learned
      mortar/reduced_model.py   Track A, training
      mortar/pbd_demo.py        Track B, rendering only (P7)
      mdp/                      observation/reward/termination/curriculum terms
scripts/            preview_wall.py runs anywhere; the rest need Isaac Sim
tests/              CPU only
```

## Asset situation

The **electric** Atlas has no public URDF or USD. Isaac Lab ships Spot as its
only Boston Dynamics robot, and `robot_descriptions` carries only the DRC-era
`atlas_drc` (v3) and `atlas_v4`. So P0 uses the DRC model:

```bash
pip install robot_descriptions
python scripts/convert_asset.py --variant drc --out assets/atlas_drc.usd
```

The DRC model is hydraulic, ~175 kg, and has no hands, so P0 also has to audit
joint limits and inertias, simplify colliders to convex hulls, and attach a
gripper.

Everything the task touches is addressed through `RobotPreset` by regex over
joint names, never by index, so nothing depends on the DoF count. `G1_PRESET` is
kept working and tested as a fallback, and a future 56-DoF electric Atlas would
drop in the same way.

## Training, on a single RTX A6000 48 GB

| phase | envs | wall clock |
|---|---|---|
| P4 balance (no bricks) | 4096 | 1-1.5 h |
| P3 upper body | 2048 | 3-5 h |
| P5 whole body | 512 | 12-21 h |
| P6 three courses, bimanual, DR | 512 | 22-37 h |

Raise the PhysX GPU buffers before the first run (`MasonrySimCfg`). Brick stacks
plus bipedal contact overflow the defaults, and the run dies hours in, on
whichever environment first reaches a tall course.
