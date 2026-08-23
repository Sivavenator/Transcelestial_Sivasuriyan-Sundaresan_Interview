"""Render a sequence of frames from a ground-truth trajectory, and recover
that trajectory back from the noisy frames -- the two operations that turn
`trajectory.py`'s abstract (x, y) motion into something an actual tracking
loop would run.

WHY FLUX IS HELD CONSTANT ACROSS THE SEQUENCE
---------------------------------------------------
The brief separately lists brightness change ("spot's brightness changes
depending on environment") as a real-world condition (§4), and SNR control
is already §2's own topic. Letting flux vary here too would mix the
motion-recovery problem (this section) with the brightness-robustness
problem (§4) in one experiment, making it unclear which effect explains
any given result. Holding flux constant isolates the question this section
actually asks: can the trajectory be recovered from noisy frames, given a
known, fixed SNR.

WHY THE RECOVERY LOOP USES THE PREVIOUS FRAME'S OWN ESTIMATE AS THE NEXT
FRAME'S PRIOR, NOT THE GROUND TRUTH
------------------------------------------------------------------------------
A real tracker never has access to ground truth -- it only has its own
past output. Seeding frame N's window from frame N-1's ESTIMATE (not the
true position) is what a real 1 kHz tracking loop actually does, and is
the only way this experiment can honestly claim its error measurements
reflect a deployable tracker rather than a simulation that cheats by
peeking at the answer.

WHY A FAILED FIT DOESN'T CORRUPT THE RUNNING PRIOR
--------------------------------------------------------
If `gaussian_fit_estimate` fails to converge (`ok=False`) on some frame,
its returned position is not trustworthy. Feeding that untrustworthy
position forward as the next frame's prior would let one bad frame drag
the window off the real spot, potentially cascading into a lost track.
Instead, the running prior only updates on a successful fit; on failure,
the NEXT frame is still seeded from the last known-good position -- a form
of dead reckoning across a single bad frame, and a real, deliberate design
choice (not an oversight) for how this tracker survives isolated failures
rather than propagating them.
"""

from __future__ import annotations

import numpy as np

from .estimators.gaussian_fit import gaussian_fit_estimate
from .simulate import Simulator


def render_sequence(sim: Simulator, x: np.ndarray, y: np.ndarray, flux: float) -> np.ndarray:
    """Render one frame per (x[i], y[i]), returned already converted to
    electrons (see simulate.py's dn_to_electrons) -- the unit system every
    estimator in this project expects."""
    n = len(x)
    frames = np.empty((n, *sim.shape), dtype=np.float64)
    for i in range(n):
        frames[i] = sim.dn_to_electrons(sim.render(float(x[i]), float(y[i]), flux))
    return frames


def recover_trajectory(
    frames: np.ndarray, half_width: int, sigma: float, read_var_e2: float = 0.0
) -> dict:
    """Run the Gaussian fit across a frame sequence with frame-to-frame
    prior gating (see module docstring), returning recovered x/y and a
    per-frame success flag."""
    n = len(frames)
    est_x = np.empty(n, dtype=np.float64)
    est_y = np.empty(n, dtype=np.float64)
    ok = np.empty(n, dtype=bool)

    running_prior = None
    for i in range(n):
        result = gaussian_fit_estimate(frames[i], half_width, sigma, read_var_e2, prior=running_prior)
        est_x[i] = result.x
        est_y[i] = result.y
        ok[i] = result.ok
        if result.ok:
            running_prior = (result.x, result.y)

    return {"x": est_x, "y": est_y, "ok": ok}
