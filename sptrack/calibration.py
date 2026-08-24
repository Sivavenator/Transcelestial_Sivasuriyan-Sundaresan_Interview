"""Calibration (brief §5): bias-frame subtraction, flat-field correction,
and lens-distortion correction, plus measuring each one's effect on
position precision -- three standard sensor/optics calibration techniques
applied to this project's already-built noise/optics model.

WHY THESE THREE, AND WHAT EACH ONE TARGETS IN THIS PROJECT SPECIFICALLY
-------------------------------------------------------------------------------
  Bias frame    -> the FIXED hot-pixel defect pattern (`sensor.py`'s
                    hot_mask). Dark current and the black-level pedestal
                    are spatially UNIFORM in this sensor model (same rate
                    everywhere), so they need no per-pixel calibration --
                    `border_median_background`'s scalar estimate already
                    handles a uniform offset correctly. Hot pixels are the
                    one spatially-STRUCTURED fixed additive pattern this
                    sensor model has.

  Flat field    -> the PRNU gain map (`sensor.py`'s prnu_map), whose
                    position-dependent bias was already proven directly
                    when PRNU was first built (a non-uniform gain map
                    biases the centroid differently at different sub-pixel
                    offsets). Flat-fielding estimates that map from bright,
                    uniformly-illuminated calibration frames and divides
                    it back out.

  Lens          -> a real-world effect not modelled anywhere else in this
  distortion       project: a geometric mapping between where a spot
                    TRULY is (an ideal, distortion-free projection) and
                    where it actually lands on the sensor. Uncorrected,
                    this is a purely GEOMETRIC bias (present even with a
                    perfect, noiseless estimator) that grows with distance
                    from the optical centre -- structurally different from
                    every other bias source in this project, which all
                    come from noise or a signal-processing assumption
                    breaking down, not from the optics itself.

WHY N_BIAS=100 AND THE FLAT-FIELD BRIGHTNESS/FRAME-COUNT, AND WHY THEY
ARE DERIVED, NOT ASSUMED
------------------------------------------------------------------------------
  N_bias = 100
      Chosen so the bias map's own residual noise (dominated by read
      noise, reduced by 1/sqrt(N) when averaging N frames) is at most 10%
      of a single frame's read noise (sigma_read_e=5.0) -- so subtracting
      the calibration map back out of a science frame adds negligible
      extra variance of its own. 1/sqrt(N) <= 0.10 => N >= 100.

  Flat-field brightness and frame count
      To reliably MEASURE PRNU (sigma=0.02, already established in
      sensor.py) against photon shot noise, the flat exposure's SNR must
      clear 1/prnu_sigma = 50 with real margin -- a 10x safety margin
      target (SNR=500) requires 500^2 = 250,000 total averaged electrons
      per pixel. A single flat frame is capped at roughly half the
      sensor's headroom (avoiding saturation): ~20,000 electrons at this
      project's default 12-bit/gain=10 configuration, so ceil(250,000 /
      20,000) = 13 frames are averaged to reach the required total.

WHY LENS DISTORTION IS MODELLED AS A SIMPLE ONE-TERM RADIAL (BARREL)
MAPPING, AT -0.1% AT THE FRAME EDGE
------------------------------------------------------------------------------
No real lens exists to calibrate against for this project, so the
magnitude was deliberately sourced from real, cited machine-vision lens
datasheets rather than picked freely: precision/low-distortion machine
vision lenses (Commonlands CIL052: -0.1%; MYUTRON FV series: <0.1%) sit
at roughly -0.1% barrel distortion at the image edge -- chosen over the
~1% "standard lens" tier because this project's camera is a narrow-FOV
precision TRACKING optic, closer to those precision-lens datasheets than
a general wide-angle imaging lens. A single radial term,
``r_distorted = r_true * (1 + k1 * r_norm^2)`` with r normalised by the
frame's half-diagonal, is the standard simplification for a well-
manufactured lens (tangential and higher-order radial terms are
second-order effects on top of this, not modelled here). k1 is set so
the fractional radial displacement at the frame's extreme corner
(r_norm=1) equals the chosen -0.1%: k1 = -0.001, directly, with no
further fitting needed for a single-term model.

WHY THE INVERSE (CORRECTION) MAPPING USES FIXED-POINT ITERATION, NOT A
CLOSED FORM
------------------------------------------------------------------------------
Inverting ``r_d = r*(1 + k1*r^2)`` for r given r_d is a cubic with no
clean closed form in general. Because k1 is small (0.001), the fixed-
point iteration ``r_{n+1} = r_d / (1 + k1 * r_n^2)``, seeded at r_0=r_d,
converges to floating-point precision in only a few iterations --
verified directly in tests/test_calibration.py (round-trip through
distortion then correction recovers the original position to within
1e-9 px), rather than assumed to be "good enough" from the small-k1
argument alone.
"""

from __future__ import annotations

import numpy as np

from .scene import render_background_gradient
from .sensor import add_photon_noise, apply_prnu
from .simulate import Simulator

# Sourced from published machine-vision lens datasheets (Commonlands
# CIL052: -0.1%; MYUTRON FV series: <0.1%), chosen for a narrow-FOV
# precision tracking optic rather than a general wide-angle lens -- see
# module docstring.
DEFAULT_DISTORTION_PCT_AT_EDGE = -0.1


def estimate_bias_frame(sim: Simulator, n_frames: int = 100) -> np.ndarray:
    """Average N zero-flux ("dark") frames to estimate the sensor's fixed
    additive pattern (dominated by the hot-pixel defect map in this
    project's noise model) -- see module docstring for why N=100."""
    accum = np.zeros(sim.shape, dtype=np.float64)
    for _ in range(n_frames):
        accum += sim.dn_to_electrons(sim.render(0.0, 0.0, 0.0))
    return accum / n_frames


def estimate_flat_field(sim: Simulator, flat_level_e: float, n_frames: int) -> np.ndarray:
    """Average N uniformly-illuminated ("flat") frames to estimate the
    PRNU relative-gain map, normalised to mean 1.0. See module docstring
    for how flat_level_e and n_frames were derived."""
    accum = np.zeros(sim.shape, dtype=np.float64)
    for _ in range(n_frames):
        flat_bg = render_background_gradient(sim.shape, flat_level_e, gradient_frac=0.0)
        # Reuses sim's own (private) RNG stream deliberately, so calibration
        # frames stay part of the same reproducible seeded sequence as
        # every other render() call from this Simulator instance.
        e_image = add_photon_noise(apply_prnu(flat_bg, sim.prnu_map), sim._rng)
        accum += e_image
    flat = accum / n_frames
    return flat / flat.mean()


def apply_radial_distortion(
    x: float, y: float, shape: tuple[int, int], distortion_pct_at_edge: float = DEFAULT_DISTORTION_PCT_AT_EDGE
) -> tuple[float, float]:
    """Map a TRUE (distortion-free) position to where it actually lands on
    a sensor with the given radial (barrel/pincushion) distortion."""
    k1 = distortion_pct_at_edge / 100.0
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_max = np.hypot(cx, cy)

    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    if r == 0 or r_max == 0:
        return x, y
    r_norm = r / r_max
    scale = 1.0 + k1 * r_norm**2
    return cx + dx * scale, cy + dy * scale


def correct_radial_distortion(
    x: float, y: float, shape: tuple[int, int], distortion_pct_at_edge: float = DEFAULT_DISTORTION_PCT_AT_EDGE,
    n_iters: int = 10,
) -> tuple[float, float]:
    """Invert apply_radial_distortion: given an OBSERVED (distorted)
    position, recover the true position via fixed-point iteration (see
    module docstring for why this is used instead of a closed form)."""
    k1 = distortion_pct_at_edge / 100.0
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_max = np.hypot(cx, cy)

    dx_d, dy_d = x - cx, y - cy
    r_d = np.hypot(dx_d, dy_d)
    if r_d == 0 or r_max == 0:
        return x, y
    r_d_norm = r_d / r_max

    r_norm = r_d_norm
    for _ in range(n_iters):
        r_norm = r_d_norm / (1.0 + k1 * r_norm**2)

    scale = r_norm / r_d_norm
    return cx + dx_d * scale, cy + dy_d * scale
