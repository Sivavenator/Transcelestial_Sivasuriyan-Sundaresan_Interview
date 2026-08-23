"""Windowed intensity-weighted centroid with background subtraction.

THE IDEA: CENTRE OF MASS
-----------------------------
Treat each pixel's (background-subtracted) brightness as a "weight," and
compute the weighted average position -- exactly the physics formula for a
centre of mass:

    x_hat = sum_i(v_i * x_i) / sum_i(v_i)
    y_hat = sum_i(v_i * y_i) / sum_i(v_i)

where v_i is pixel i's background-subtracted value and (x_i, y_i) is its
coordinate. This is the simplest possible position estimator: no model of
the spot's shape, no fitting, no iteration -- just a weighted sum, and it
is exact (unbiased) for a spot whose light distribution is symmetric about
its true centre, sitting in a window free of anything else.

WHY BACKGROUND SUBTRACTION IS NOT OPTIONAL
-----------------------------------------------
Without it, every pixel in the window -- including ones far from the spot,
carrying no real signal -- contributes a roughly EQUAL positive weight (the
background level), because background is present everywhere in the window.
A window is (as close as this project can arrange) symmetric about its own
CENTRE, not about the spot's true position, so weighting every pixel
equally pulls the estimate toward the window's geometric centre rather than
the spot. That is a real, structural bias, not just extra noise -- which is
exactly why `border_median_background` (base.py) exists and every estimator
in this project subtracts it before doing anything else.

WHY NEGATIVE VALUES ARE CLIPPED TO ZERO BY DEFAULT -- A REAL TRADEOFF,
NOT A FREE CLEANUP STEP
------------------------------------------------------------------------------
After subtracting a background estimate, pixels that are truly
background-only will have noisy values scattered around zero -- some
positive, some negative, purely from noise. Clipping the negative ones to
zero is a genuine bias/variance TRADEOFF, not a free correctness fix:

  * Leaving negatives in is unbiased in principle (positive and negative
    noise fluctuations cancel on average), but each negative-weighted
    background pixel still contributes a random weight times its distance
    from the window centre (a "lever arm") to the sum -- and that lever arm
    can be large for a pixel near the window's edge. This inflates the
    estimate's VARIANCE, sometimes substantially, for zero benefit.
  * Clipping to zero removes that variance contribution, at the cost of a
    small RECTIFICATION bias: truncating a two-sided noise distribution to
    one side shifts its mean away from zero, which is exactly the same
    "clipping only ever biases toward the surviving side" mechanism as the
    black-level pedestal discussion in sensor.py's quantization section.

Exposed as a parameter (``clip_negative``) rather than a silent default, so
this tradeoff can be measured directly later rather than assumed.

WHY THE STARTING GUESS IS THE BRIGHTEST PIXEL WHEN NO PRIOR IS GIVEN
--------------------------------------------------------------------------
On the very first frame of a sequence there is no previous position to
start from. The brightest pixel is the cheapest, simplest reasonable guess
-- good enough to place a window that contains the real spot, which is all
it needs to do; the centroid calculation itself does the real work of
finding the sub-pixel position within that window.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Estimate, border_median_background, extract_window, find_brightest_pixel


def centroid_estimate(
    image: np.ndarray,
    half_width: int,
    prior: tuple[float, float] | None = None,
    clip_negative: bool = True,
) -> Estimate:
    """Estimate a spot's position via windowed, background-subtracted
    intensity-weighted centroid.

    ``half_width`` sets the window size (a ``2*half_width + 1`` square).
    ``prior``, if given, is a rough ``(x, y)`` starting guess (e.g. the
    previous frame's estimate); without one, the window is centred on the
    brightest pixel instead.
    """
    if prior is None:
        iy, ix = find_brightest_pixel(image)
        cx, cy = float(ix), float(iy)
    else:
        cx, cy = prior

    window, x0, y0 = extract_window(image, cx, cy, half_width)
    bg = border_median_background(window)
    sub = window - bg
    if clip_negative:
        sub = np.maximum(sub, 0.0)

    total = float(sub.sum())
    if total <= 0 or not math.isfinite(total):
        return Estimate(float("nan"), float("nan"), ok=False)

    h, w = sub.shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    local_x = float((sub.sum(axis=0) * xs).sum() / total)
    local_y = float((sub.sum(axis=1) * ys).sum() / total)

    return Estimate(x=local_x + x0, y=local_y + y0, flux=total, bg=bg)
