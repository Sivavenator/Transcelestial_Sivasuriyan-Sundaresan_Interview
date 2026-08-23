"""Shared utilities every position estimator needs: windowing, an initial
guess, and a background estimate.

WHY THESE LIVE HERE, SHARED, RATHER THAN INSIDE EACH ESTIMATOR
--------------------------------------------------------------------
Every estimator in this project needs the same three things before it can
do its own actual work: a small region of the frame to look at (a window,
not the whole image), a starting guess for where the spot roughly is, and
an estimate of the local background level. Keeping these here means every
estimator gets them identically -- which matters for FAIRNESS when
estimators are compared later (Characterization, §2c): if one estimator
used a bigger window, or a better background estimate, than another, a
difference in their measured precision would be measuring that difference,
not a real difference between the methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    """The result of one position estimate, in the coordinate frame of the
    original (un-windowed) input image -- callers should never need to know
    the window's local coordinates existed at all."""

    x: float
    y: float
    flux: float = float("nan")
    bg: float = float("nan")
    ok: bool = True


def find_brightest_pixel(image: np.ndarray) -> tuple[int, int]:
    """Integer (row, col) of the brightest pixel -- the cheapest possible
    starting guess, used only when no prior position is available (e.g. the
    very first frame of a sequence)."""
    idx = int(np.argmax(image))
    return idx // image.shape[1], idx % image.shape[1]


def extract_window(
    image: np.ndarray, cx: float, cy: float, half_width: int
) -> tuple[np.ndarray, int, int]:
    """Extract a (2*half_width+1)^2 window centred on the nearest pixel to
    (cx, cy), clamped to the image bounds.

    Returns the window and its origin (x0, y0), so a caller can convert a
    position computed in the window's own local coordinates back to the
    original image's coordinates by simply adding the origin back. The
    window is silently smaller near an edge -- callers must use the
    returned origin rather than assuming the window is always centred and
    full-sized, or edge frames will be silently mis-registered.
    """
    h, w = image.shape
    ix, iy = int(round(cx)), int(round(cy))
    x0 = max(ix - half_width, 0)
    x1 = min(ix + half_width + 1, w)
    y0 = max(iy - half_width, 0)
    y1 = min(iy + half_width + 1, h)
    return image[y0:y1, x0:x1], x0, y0


def border_median_background(window: np.ndarray, border_width: int = 2) -> float:
    """Estimate the local background level as the median of the window's
    outer ``border_width`` rows and columns.

    WHY THE BORDER ONLY, NOT THE WHOLE WINDOW
    ----------------------------------------------
    The spot itself sits near the window's centre. Including the centre
    pixels in a background estimate would mean the spot's own signal drags
    the "background" estimate upward, causing systematic OVER-subtraction
    that eats into the spot's real flux -- worst for a bright spot in a
    small window, exactly when the estimate should be easiest.

    WHY MEDIAN, NOT MEAN
    -------------------------
    A single unusually bright border pixel (a hot pixel that happens to
    fall on the border, or just an unlucky noise spike) would drag a MEAN
    background estimate upward, and that error then propagates into every
    pixel's background-subtracted value across the whole window. The median
    is robust to a small number of such outliers -- it takes a majority of
    the border pixels being wrong to move it, not just one.
    """
    if window.size == 0:
        return 0.0
    h, w = window.shape
    mask = np.ones((h, w), dtype=bool)
    if h > 2 * border_width and w > 2 * border_width:
        mask[border_width:-border_width, border_width:-border_width] = False
    return float(np.median(window[mask]))


def planar_background(window: np.ndarray, border_width: int = 2) -> np.ndarray:
    """Estimate a full, per-pixel background PREDICTION by least-squares
    fitting a plane (``a + b*x + c*y``) to the window's border pixels, and
    evaluating it across the whole window -- the mitigation for a strong,
    non-uniform background (solar glare, `docs/REAL_WORLD_CONDITIONS.md`
    item 4), where `border_median_background`'s single scalar estimate
    stops being good enough.

    WHY A SCALAR BACKGROUND ESTIMATE FAILS UNDER A REAL GRADIENT, EVEN
    WHEN THE SCALAR ITSELF IS ACCURATE AT THE WINDOW CENTRE
    ------------------------------------------------------------------------------
    `border_median_background`'s scalar is a genuinely good estimate of the
    background level AT THE WINDOW'S CENTRE (verified directly: under a
    gradient strong enough to vary the background by tens of electrons
    across a window, the median differs from the true centre value by a
    thousandth of an electron -- median-of-border is an accurate,
    essentially unbiased read of the centre value). The problem is what
    happens NEXT: subtracting that one constant number from every pixel in
    the window is only correct where the true background actually equals
    that constant -- the window's centre. Everywhere else, a REAL residual
    background gradient remains in the "background-subtracted" image
    (small near the centre, growing toward the edges), and that residual
    is not just noise -- it is a real, structured signal that the centroid's
    weighted average then responds to, pulling the estimate toward
    whichever side of the window the residual is stronger on. This was
    checked directly, not assumed: at a background varying roughly 30% of
    its mean value across the window, this residual causes a POSITION bias
    of over 0.4 px -- far larger than almost any other systematic bias
    characterised anywhere else in this project.

    WHY A FULL PLANAR PREDICTION FIXES IT
    ------------------------------------------
    Fitting the plane ``a + b*x + c*y`` to the border (least squares) and
    evaluating that SAME plane at every pixel in the window -- not just
    subtracting its value at one point -- removes the gradient's own
    linear structure everywhere in the window simultaneously, not just at
    the centre. Verified directly (`tests/test_planar_background.py` and
    `experiments/exp04e_glare.py`): the resulting centroid bias stays at
    the noise floor (~0.0001 px) across the SAME range of gradient
    strengths that pushed the scalar-median approach's bias past 2.5 px.

    WHY THIS IS A SEPARATE FUNCTION, NOT A REPLACEMENT FOR
    border_median_background
    ------------------------------------------------------------
    A planar fit costs more (a least-squares solve every frame, versus a
    median) and needs enough border pixels to constrain 3 free parameters
    rather than 1 -- overkill when the background genuinely IS flat or
    only mildly graded, which is the common case this project's other
    experiments (§2, §3) deliberately used. This is offered as an
    additional, opt-in tool for the specific glare/strong-gradient
    condition, not a silent replacement for the simpler estimator used
    everywhere else.
    """
    h, w = window.shape
    mask = np.ones((h, w), dtype=bool)
    if h > 2 * border_width and w > 2 * border_width:
        mask[border_width:-border_width, border_width:-border_width] = False

    ys, xs = np.mgrid[0:h, 0:w]
    design = np.column_stack([np.ones(int(mask.sum())), xs[mask], ys[mask]])
    coeffs, *_ = np.linalg.lstsq(design, window[mask], rcond=None)
    a, b, c = coeffs

    full_ys, full_xs = np.mgrid[0:h, 0:w]
    return a + b * full_xs + c * full_ys
