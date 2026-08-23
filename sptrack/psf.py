"""Gaussian point-spread function, rendered by exact pixel integration.

THE CONTEXT: RENDERING A SOFT SPOT (LIKE A LIGHT, OR BLUR)
------------------------------------------------------------
When a circular spot of light lands on a sensor, each pixel should receive an
amount of light proportional to how much of the Gaussian "falls" on that
pixel's area -- not a single sampled value.

THE NAIVE APPROACH -- SAMPLING
-------------------------------
The simple way: evaluate the Gaussian at the centre of each pixel and use
that as the pixel's brightness. This is only an approximation -- the
Gaussian can vary rapidly across a pixel's area (especially when the spot is
only a few pixels wide), so the centre value is not representative of the
whole pixel.

THE BETTER APPROACH -- INTEGRATING
-------------------------------------
What we actually want is the *average* intensity across the entire pixel
area: the Gaussian integrated over that pixel's 2-D footprint.

    pixel value = double_integral_over_pixel  G(x, y) dx dy

WHY THE ERROR FUNCTION SAVES US
----------------------------------
A Gaussian has no elementary antiderivative, but its *definite* integral
between two bounds does have a closed form, expressed with the error
function (erf):

    integral_a^b  exp(-x^2) dx   is proportional to   erf(b) - erf(a)

A 2-D Gaussian is separable, G(x, y) = G(x) * G(y), so the 2-D integral over
a rectangular pixel factors into two independent 1-D erf evaluations:

    pixel value = [erf(x2) - erf(x1)] * [erf(y2) - erf(y1)]

where (x1, x2) and (y1, y2) are the pixel's left/right and top/bottom edges.

THE PAYOFF
------------
* Exact, not approximate -- no sampling error at all.
* Fast -- a handful of `erf()` calls per pixel, no supersampling needed.
* Matters most exactly when it matters most: a narrow, sub-pixel-sized spot,
  where centre-sampling would be wildly inaccurate.

Because the Gaussian's integral has a known, computable form via `erf`, we
get the exact pixel brightness analytically instead of guessing from a
single sampled value.

WHY THIS MATTERS FOR US SPECIFICALLY
---------------------------------------
Centre-sampling is wrong by an amount that depends on exactly where the spot
sits within a pixel. That position-dependent error shows up later as "pixel
locking" -- a bias that is worst when the spot sits between two pixel
centres and vanishes when it sits on one. Since we need sub-pixel precision,
we cannot afford a method whose error depends on the very sub-pixel offset
we are trying to measure. Integrating exactly costs one call to `erf` per
pixel edge instead of one call to `exp`, so there is no real cost tradeoff
either.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

_SQRT2 = np.sqrt(2.0)


def pixel_response_1d(pixel_index: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    """Fraction of a unit-area 1-D Gaussian falling inside each pixel.

    ``pixel_index`` are integer pixel coordinates; pixel ``i`` is taken to
    span ``[i - 0.5, i + 0.5]``. Returns an array the same shape as
    ``pixel_index``, summing to ~1 over a window wide enough to enclose the
    spot.
    """
    hi = (pixel_index + 0.5 - centre) / (sigma * _SQRT2)
    lo = (pixel_index - 0.5 - centre) / (sigma * _SQRT2)
    return 0.5 * (erf(hi) - erf(lo))


def render_spot(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    flux: float,
    sigma: float,
) -> np.ndarray:
    """Noise-free mean image of a Gaussian spot.

    ``shape`` is ``(height, width)`` in pixels. ``x0, y0`` is the spot centre
    in pixel coordinates (pixel centres at integers, so ``x0=0`` is the
    centre of the leftmost column). ``flux`` is the total signal (e.g.
    electrons) integrated over an infinite window; ``sigma`` is the PSF
    width in pixels.

    The 2-D Gaussian is separable, so the image is the outer product of two
    1-D pixel responses -- exact, and cheap.
    """
    h, w = shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    px = pixel_response_1d(xs, x0, sigma)
    py = pixel_response_1d(ys, y0, sigma)
    return flux * np.outer(py, px)


def diameter_1e2_to_sigma(diameter_1e2_px: float) -> float:
    """Convert a laser's "diameter at 1/e^2" spec to a Gaussian sigma.

    THE DERIVATION
    ------------------
    Laser beam profiles are conventionally specified using irradiance
    I(r) = I0 * exp(-2 r^2 / w^2), where w is the "beam radius": by
    construction, at r = w the exponent is -2*w^2/w^2 = -2, so
    I(w) = I0 * exp(-2) = I0 / e^2 -- this is exactly what "1/e^2" means.

    Compare that to the standard statistical Gaussian intensity profile in
    terms of sigma: I(r) = I0 * exp(-r^2 / (2 sigma^2)). Setting the two
    exponents equal (since they describe the same physical spot, just in two
    different conventions):

        2 r^2 / w^2  =  r^2 / (2 sigma^2)
        2 / w^2      =  1 / (2 sigma^2)
        w^2          =  4 sigma^2
        w            =  2 sigma
        sigma        =  w / 2

    Since w is the RADIUS at 1/e^2, and the brief specifies a DIAMETER:

        w = diameter_1e2 / 2
        sigma = w / 2 = diameter_1e2 / 4

    With the brief's ~7 px diameter: sigma = 7 / 4 = 1.75 px -- exactly the
    value used as the default sigma throughout this project's tests.
    """
    radius_1e2 = diameter_1e2_px / 2.0
    return radius_1e2 / 2.0


def sample_true_sigma(
    nominal_sigma: float, tolerance_frac: float, rng: np.random.Generator
) -> float:
    """Draw one PSF width representing a specific optical unit's true sigma.

    THE BRIEF: "spot size is ~7 px diameter (1/e^2) but can slightly vary
    depending on optics quality"
    --------------------------------------------------------------------------
    Real optics have manufacturing and assembly tolerances -- lens
    curvature, alignment, focus -- so no two physical units, even of the
    same design, produce EXACTLY the nominal spot size. This is modelled as
    a FIXED property of one simulated optical unit: call this once per
    simulated system (the same way generate_hot_pixel_mask and
    generate_prnu_map are called once per simulated sensor, not per frame),
    not once per frame -- a given unit's optics don't refocus themselves
    between frames.

    ``tolerance_frac`` is the manufacturing tolerance as a fraction of the
    nominal sigma (e.g. 0.1 for +-10%-ish spread). Clipped to a small
    positive floor rather than allowed to go non-positive or unrealistically
    close to zero, since sigma <= 0 has no physical meaning.

    WHY THIS MATTERS BEYOND "MORE REALISM"
    -------------------------------------------
    An estimator that ASSUMES a fixed template sigma (most of the ones in
    this project) is implicitly assuming the design value is the true
    value. If the real optics' true sigma differs even slightly, that
    estimator is fitting the wrong model -- a form of PSF model mismatch.
    This function exists specifically to let later experiments inject that
    mismatch deliberately and measure what it costs, rather than only ever
    testing estimators against PSFs they were built to expect.
    """
    sigma = rng.normal(nominal_sigma, nominal_sigma * tolerance_frac)
    floor = 0.1 * nominal_sigma
    return max(sigma, floor)
