"""Robust target acquisition: pick the real laser spot among candidate
bright sources by PSF-shape match, not raw brightness alone -- the
mitigation for the background-clutter/false-source condition identified
in `docs/REAL_WORLD_CONDITIONS.md` (item 3).

WHY BRIGHTNESS ALONE IS NOT A SAFE ACQUISITION CRITERION
--------------------------------------------------------------
`estimators/base.py::find_brightest_pixel` -- this project's original
acquisition mechanism, used whenever no prior position is available --
places its search window on the single brightest pixel in the frame with
no other criterion. A real outdoor scene can contain other bright
sources (streetlights, glints, other satellites) that are NOT the laser
spot. Worse, a real non-laser source does not need to have MORE total
flux than the laser spot to win a brightest-PIXEL competition -- it only
needs comparable or higher PEAK brightness, and peak brightness for a
Gaussian-like source scales as flux / (2*pi*sigma^2): a source with more
total flux but spread over a WIDER profile (physically realistic for
almost anything that isn't a collimated, diffraction-limited laser beam)
can still win. Demonstrated directly in `tests/test_acquisition.py`: a
clutter source with over 13x the true spot's total flux, but a much wider
profile, produces a HIGHER peak pixel value and wins outright under
`find_brightest_pixel`.

WHY PSF-SHAPE MATCH IS A ROBUST DISCRIMINATOR HERE
--------------------------------------------------------
The real laser source has one piece of KNOWN structure that generic
clutter does not share by coincidence: it is diffraction-limited, so its
profile matches the assumed PSF width (``sigma``, already known and used
throughout every estimator in this project) closely. This module ranks
every candidate local maximum by how well a small window around it
correlates with an ideal Gaussian template of that SAME assumed sigma
(a Pearson correlation coefficient -- scale-invariant, so it discriminates
by SHAPE alone, not brightness, which is exactly the property clutter
does not reliably share). This is the same idea as the matched filter
(`estimators/matched_filter.py`) applied at ACQUISITION time rather than
for sub-pixel refinement, and reuses the same underlying PSF model
(`psf.render_spot`) rather than a second, independently-defined template.

WHAT THIS DOES NOT SOLVE
------------------------------
This is a coarse, ACQUISITION-time discriminator, not a general clutter
filter. Two limitations stated honestly: (1) a piece of clutter that
happens to be diffraction-limited-shaped too (a genuine point-like light
source, not just a diffuse glint) is not distinguishable by shape alone
-- this project's `docs/REAL_WORLD_CONDITIONS.md` already notes flux/SNR
range and expected VELOCITY as further, unimplemented discriminators for
that harder case; (2) once tracking is locked, the existing prior-gated
window (§3) already provides strong protection on its own -- clutter far
from the current track simply never enters the estimation window -- so
this module's value is concentrated at acquisition and re-acquisition,
not steady-state tracking.
"""

from __future__ import annotations

import numpy as np

from .estimators.base import border_median_background, extract_window
from .psf import render_spot


def psf_shape_score(window: np.ndarray, sigma: float) -> float:
    """Pearson correlation between a (background-subtracted) window and an
    ideal centred Gaussian template of the given sigma -- 1.0 for a
    perfect match, lower for a mismatched profile shape. Scale-invariant:
    does not care how bright the window is, only how well-shaped it is.
    """
    h, w = window.shape
    if h == 0 or w == 0:
        return 0.0
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    template = render_spot((h, w), cx, cy, 1.0, sigma)

    a = window.ravel() - window.mean()
    b = template.ravel() - template.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def find_local_maxima(image: np.ndarray, min_value: float, nms_radius: int) -> list[tuple[int, int, float]]:
    """Local maxima above ``min_value``, non-max-suppressed within
    ``nms_radius`` so one bright blob doesn't produce multiple candidates.
    Returns a list of (x, y, pixel_value).
    """
    h, w = image.shape
    candidates = []
    for y in range(nms_radius, h - nms_radius):
        for x in range(nms_radius, w - nms_radius):
            v = image[y, x]
            if v < min_value:
                continue
            patch = image[y - nms_radius : y + nms_radius + 1, x - nms_radius : x + nms_radius + 1]
            if v == patch.max():
                candidates.append((x, y, float(v)))
    return candidates


def acquire_target(
    image: np.ndarray, half_width: int, sigma: float, min_value: float, nms_radius: int | None = None,
) -> tuple[float, float] | None:
    """Find the best-shape-matched candidate position in ``image``, for use
    as an acquisition-time prior (in place of raw ``find_brightest_pixel``).

    Returns ``(x, y)`` of the best-matching candidate's peak pixel, or
    ``None`` if no candidate clears ``min_value`` at all.
    """
    if nms_radius is None:
        nms_radius = half_width
    candidates = find_local_maxima(image, min_value, nms_radius)
    if not candidates:
        return None

    best_xy = None
    best_score = -np.inf
    for cx, cy, _ in candidates:
        window, wx0, wy0 = extract_window(image, cx, cy, half_width)
        bg = border_median_background(window)
        score = psf_shape_score(window - bg, sigma)
        if score > best_score:
            best_score = score
            best_xy = (float(cx), float(cy))
    return best_xy
