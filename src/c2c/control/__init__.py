"""Phase 4: closed-loop control of a simulated fly body by the spiking brain model.

    body sensors -> encoder -> sensory neurons -> spiking brain -> descending neurons -> decoder
    -> walking controller -> body

encoder.py   compound-eye readout -> Poisson rates for LPLC2 / LC4
decoder.py   descending-neuron spikes -> walking command
body.py      NeuroMechFly v2 (flygym, MuJoCo) in an arena with a looming ball
loop.py      the loop that runs them in lockstep with the brain (c2c.sim.torch_lif.LIFStepper)
"""

import os as _os
import sys as _sys

# MuJoCo picks its OpenGL backend when it is first imported. On a Linux machine without a display
# (a server, CI) that must be EGL, so set it before anything imports flygym or mujoco.
# Windows and macOS use their native OpenGL and are not affected.
if _sys.platform.startswith("linux") and not _os.environ.get("DISPLAY"):
    _os.environ.setdefault("MUJOCO_GL", "egl")
