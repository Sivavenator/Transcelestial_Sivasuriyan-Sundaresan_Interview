"""Gaussian point-spread function, rendered by exact pixel integration.

A camera pixel does not sample light at a single point; it integrates
irradiance over its whole area. For a spot with a Gaussian intensity
profile, that integral has a closed form using the error function, so we
compute it exactly rather than approximating it by sampling the Gaussian at
each pixel's centre.

Why this matters: sampling at the centre is wrong by an amount that depends
on exactly where the spot sits within a pixel. That position-dependent error
shows up later as "pixel locking" -- a bias that is worst when the spot sits
between two pixel centres and vanishes when it sits on one. Since we need
sub-pixel precision, we cannot afford a method whose error depends on the
very sub-pixel offset we are trying to measure. Integrating exactly costs
one call to `erf` per pixel edge instead of one call to `exp`, so there is
no real cost tradeoff either.
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
