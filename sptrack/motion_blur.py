"""Motion blur (brief §5): render a spot that moves during the exposure,
and characterize how much this degrades estimator precision -- the last
real-world effect not yet modelled anywhere in this project's simulator.

WHY THIS IS MODELLED AS TEMPORAL SUPERSAMPLING, NOT AN ANALYTIC
CLOSED-FORM BLURRED PSF
------------------------------------------------------------------------------
During one exposure, a spot moving at constant velocity sweeps out a
straight-line path; the recorded image is the TIME-INTEGRAL of the
(otherwise-static) PSF along that path. This is rendered here by summing
many instantaneous renders of `psf.render_spot` at closely-spaced points
along the path, each carrying an equal share of the total flux --
mathematically equivalent to convolving the static PSF with a uniform
("boxcar") kernel along the motion direction, verified directly against
the closed-form result for that convolution: the blurred distribution's
variance along the motion axis should equal sigma^2 + blur_px^2/12 (the
box kernel's own variance). Measured agreement was within the same small
discretization offset already present at zero blur (a known,
already-documented pixel-integration effect, not a flaw in this model).
Reusing `render_spot` this way, rather than deriving a new closed-form
"blurred pixel response," keeps this consistent with the same PSF model
every other estimator and the simulator already use.

WHY 61 SUBSTEPS
--------------------
Checked directly against a 201-substep reference at the largest blur
tested (5 px): 41 substeps left a ~1% peak-value discretization error,
comparable to the noise photon-noise level itself; 61 substeps was chosen
as a reasonable middle ground between accuracy and the added render cost
(this function is called many times per Monte Carlo trial).

WHY THE BLUR MAGNITUDE IS SWEPT AS A FRACTION OF SIGMA, WITH NO CLAIMED
REAL-WORLD VELOCITY
------------------------------------------------------------------------------
An attempt was made to ground a specific blur magnitude in a real
fine-steering-mirror slew-rate spec (Bramall et al., a genuine FSO
tracking-terminal steering mirror, 1.5 mrad/s) converted to pixels via a
sensor plate scale -- but no plate scale (angular size per pixel) exists
anywhere in this project, and the values found while searching for one
(beacon-camera FOV specs, a tracking-CCD's ACHIEVED accuracy rather than
its plate scale) did not combine into one defensible number without
stacking multiple further assumptions on top of each other. Rather than
chain assumption onto assumption, this experiment instead sweeps blur
magnitude directly, expressed as a fraction of the PSF's own sigma
(0 to 3x) -- characterizing HOW estimators degrade with blur severity in
general, without asserting a specific real-world velocity this project
has no solid basis to claim.
"""

from __future__ import annotations

import numpy as np

from .psf import render_spot


def render_motion_blurred_spot(
    shape: tuple[int, int], x0: float, y0: float, flux: float, sigma: float,
    blur_px: float, angle_rad: float = 0.0, n_substeps: int = 61,
) -> np.ndarray:
    """Render a spot blurred by constant-velocity motion of total extent
    ``blur_px`` (start to end of the exposure) along ``angle_rad``,
    centred on (x0, y0) -- i.e. the spot sweeps from
    (x0, y0) - blur_px/2 * (cos, sin) to (x0, y0) + blur_px/2 * (cos, sin).
    """
    if blur_px <= 0:
        return render_spot(shape, x0, y0, flux, sigma)

    offsets = np.linspace(-blur_px / 2.0, blur_px / 2.0, n_substeps)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    img = np.zeros(shape, dtype=np.float64)
    flux_per_step = flux / n_substeps
    for off in offsets:
        img += render_spot(shape, x0 + off * dx, y0 + off * dy, flux_per_step, sigma)
    return img
