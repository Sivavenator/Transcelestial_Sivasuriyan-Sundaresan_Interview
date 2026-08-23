"""The scene: everything that lands on the sensor besides the tracked spot.

This is deliberately a separate module from ``sensor.py``. The noise chain in
``sensor.py`` models imperfections in the *sensor itself* -- read noise, dark
current, hot pixels. A background gradient is not a sensor flaw; it's real
light entering the system, the same way the tracked spot is. Keeping "what's
in the scene" and "how the sensor mangles it" as separate concerns means each
can be reasoned about (and tested) independently.

WHY THE BACKGROUND ISN'T FLAT
---------------------------------
A real outdoor scene is never a uniform grey level. For a terrestrial
free-space link looking toward the horizon: sky radiance varies with angle
(brighter low toward the horizon, darker looking further up), there can be
stray light from the sun or from artificial sources, and the field of view
may simply contain uneven terrain or structures. Modelling the background as
one constant level per frame would silently assume all of that away.

WHY A LINEAR (PLANAR) GRADIENT, NOT SOMETHING MORE ELABORATE
------------------------------------------------------------------
A full physical sky-radiance model is its own research project. A linear
gradient -- background tilts smoothly across the frame -- is the simplest
model that is still genuinely *non-uniform*, and it captures the dominant,
leading-order effect: over a field of view a few tens of pixels wide (the
relevant scale here, not a wide-field sky camera), any smooth background
variation looks approximately linear locally, the same way any smooth
function looks approximately linear if you zoom in enough. Higher-order
terms (curvature) would be the natural next refinement, not a first one.

THE MODEL
------------
Pixel coordinates are normalised to [-1, 1] across each axis (x_norm=-1 at
the left edge, +1 at the right; same for y), so the gradient's strength is
expressed relative to the frame, not tied to a specific image size. The
background level at a pixel is:

    t = x_norm * cos(angle) + y_norm * sin(angle)
    background(x, y) = mean_level * (1 + (gradient_frac / 2) * t)

``angle`` sets the gradient's direction (0 = pure left-to-right, pi/2 = pure
bottom-to-top, anything else = diagonal). ``gradient_frac`` sets its
strength: for an axis-aligned gradient (angle=0 or pi/2), this produces
exactly ``gradient_frac`` of ``mean_level`` peak-to-peak from edge to edge.
For a diagonal gradient the true corner-to-corner peak-to-peak is up to
sqrt(2) times that (since it's summing the projections along two axes) --
stated explicitly here rather than silently baked into the number, since
"gradient_frac" alone doesn't fully pin down the peak-to-peak spread once the
angle stops being axis-aligned.

Like a spot's flux, this background level is a *mean* -- it's electrons per
pixel before shot noise, and it should be summed with the spot's mean image
and then run through ``sensor.add_photon_noise`` together, once, since
background photons are subject to the exact same Poisson statistics as
signal photons. It is not a separate, independently-drawn noise source the
way dark current is treated -- there's no equivalent "keep it addressable
for calibration" reason to split it out, since a real system has no way to
capture a "background-only, spot-off" frame the way it can capture a
dark frame with the shutter closed.
"""

from __future__ import annotations

import numpy as np


def render_background_gradient(
    shape: tuple[int, int],
    mean_level: float,
    gradient_frac: float = 0.0,
    angle_rad: float = 0.0,
) -> np.ndarray:
    """Render a linear background gradient as a mean-electron-count image.

    ``mean_level`` is the background's average level across the frame, in
    the same units as spot flux (electrons). ``gradient_frac`` is the
    strength of the tilt (0 = perfectly flat background). ``angle_rad`` sets
    its direction; 0 is a left-right tilt, pi/2 is a bottom-top tilt.
    """
    h, w = shape
    x_norm = np.linspace(-1.0, 1.0, w)
    y_norm = np.linspace(-1.0, 1.0, h)
    xx, yy = np.meshgrid(x_norm, y_norm)
    t = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    return mean_level * (1.0 + 0.5 * gradient_frac * t)
