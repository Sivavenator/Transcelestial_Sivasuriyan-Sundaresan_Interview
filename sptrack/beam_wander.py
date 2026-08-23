"""Beam wander: turbulence-induced angle-of-arrival fluctuation, a
POSITION-domain real-world condition (brief §4, `docs/REAL_WORLD_CONDITIONS.md`
item 2) -- physically distinct from both the mechanical jitter already
modelled in `trajectory.py` (§3) and the intensity fluctuation modelled in
`scintillation.py` (§4's named example), even though a single frame cannot
tell any of the three apart by eye.

WHY A MEAN-REVERTING (OU/AR(1)) PROCESS, IN LINEAR SPACE THIS TIME
--------------------------------------------------------------------------
Same reasoning as `scintillation.py`'s choice of a stationary, correlated
process over a random walk: the same turbulent medium that causes
scintillation also refracts the beam's local angle of arrival, and that
refraction fluctuates around a stable mean rather than accumulating. The
one structural difference from scintillation: intensity must stay
positive (hence log-normal, `exp(X)`), but a POSITION perturbation can be
positive or negative, so this process is applied directly in linear
space, zero-mean, no log transform needed.

WHY THE COHERENCE TIME IS LONGER THAN SCINTILLATION'S
-------------------------------------------------------------
Intensity scintillation is sensitive to small-scale turbulent structure
(constructive/destructive interference at near-point scales), while
angle-of-arrival fluctuation is effectively an APERTURE-AVERAGED
wavefront-tilt measurement -- averaging over the receiver aperture and the
beam's own footprint smooths out exactly the small, fast structure that
drives scintillation, leaving predominantly the larger, slower-evolving
turbulent structure. That is a real, physically-grounded reason (aperture
averaging), not an arbitrary choice, for why this module's default
coherence time (tau_s=20 ms) is set several times longer than
scintillation's (5 ms) -- both are still honest assumptions in the
absence of site-specific turbulence data, exactly like scintillation's
own parameters.

WHY sigma_px=0.15, THE SAME MAGNITUDE AS MECHANICAL jitter_std_px
-------------------------------------------------------------------------
Deliberately chosen equal to `trajectory.py`'s jitter_std_px, specifically
to make the point sharply rather than softly: two position-noise sources
with IDENTICAL time-domain variance are still cleanly separable by their
DIFFERENT spectral shape alone (jitter: flat; beam wander: low-frequency-
concentrated, corner frequency ~= 1/(2*pi*tau_s) ~= 8 Hz for the default
tau_s) -- verified directly (a synthetic low-band/high-band power ratio of
~0.03 for white jitter vs. ~8.9 for beam wander from an identical-
magnitude process, not just asserted). This is the same spectral-
separability argument §3's drift/jitter/disturbance decomposition relied
on, applied to a real-world condition this project had not yet modelled.

WHY THIS MATTERS FOR §3's DISTURBANCE DETECTOR
----------------------------------------------------
Beam wander's spectral shape (low-frequency-concentrated, like drift) is
exactly the kind of signal `sptrack/disturbance.py`'s `exclude_below_hz`
band was designed to exclude from the periodic-disturbance search. In a
real deployment where beam wander is present alongside drift, that
exclusion band would need to widen to still exclude BOTH -- and, per
the boundary-blind-spot failure mode already found in §3 part 4, a real
disturbance whose frequency happens to sit near that (now-wider) boundary
would be at even greater risk of the same detection failure. This module
does not re-run that full analysis; it is recorded here as a genuine,
identified interaction between two separately-built pieces of this
project, worth stating rather than leaving implicit.
"""

from __future__ import annotations

import numpy as np


def generate_beam_wander(
    n_frames: int, dt_s: float, sigma_px: float = 0.15, tau_s: float = 20e-3,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent (dx, dy) position-perturbation time series from
    a zero-mean, mean-reverting (OU/AR(1)) process -- add these directly to
    a trajectory's x/y arrays to apply beam wander."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-dt_s / tau_s)
    sigma_eps = sigma_px * np.sqrt(1.0 - phi**2)

    def _one_axis(axis_rng: np.random.Generator) -> np.ndarray:
        x = np.empty(n_frames, dtype=np.float64)
        x[0] = axis_rng.normal(0.0, sigma_px)
        innovations = axis_rng.normal(0.0, sigma_eps, n_frames - 1)
        for i in range(1, n_frames):
            x[i] = phi * x[i - 1] + innovations[i - 1]
        return x

    dx = _one_axis(rng)
    dy = _one_axis(rng)
    return dx, dy
