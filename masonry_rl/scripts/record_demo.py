#!/usr/bin/env python3
"""P7: render the demo with Track B (PBD particle) mortar.

    python scripts/record_demo.py --checkpoint ... --num_envs 1 --enable_cameras

The trained policy is replayed unchanged; only the mortar representation is
swapped from the reduced-order thickness field to high-viscosity PBD particles
so extrusion, squeeze-out and slump are actually simulated. Particles are far
too expensive for training - this path must never be imported by train.py.
"""

raise NotImplementedError("P7: PBD mortar + camera rig + video capture")
