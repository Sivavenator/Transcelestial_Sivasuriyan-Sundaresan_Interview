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
