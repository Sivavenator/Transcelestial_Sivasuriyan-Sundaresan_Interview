"""Matched-filter (correlation) peak estimator: a third method, chosen for
a genuinely different reason than the other two -- real-time friendliness.

THE IDEA
------------
Correlate the (background-subtracted) window with a small Gaussian
template, and read the position off where that correlation peaks. This is
not a new idea invented here: the "matched filter" is a classical result
from signal detection theory -- for a known signal shape buried in additive
white noise, correlating with a copy of that shape is the linear filter
that maximises the output signal-to-noise ratio, of ALL possible linear
filters. That makes correlation the right tool for the DETECTION half of
this problem (is there really a spot here, and roughly where); the
sub-pixel interpolation step below is what turns that into a precise
LOCALISATION.

WHY A THIRD METHOD AT ALL, GIVEN THE BRIEF ONLY ASKS FOR TWO
------------------------------------------------------------------
The centroid and the Gaussian fit differ in accuracy (characterised in
experiments/exp01_snr_characterization.py) but not in a way that matters
for real-time deployment: both have a cost that is fixed once configured
(centroid: one pass; the fit: bounded by max_iter). What NEITHER of them
offers is a computation shaped for dedicated hardware. A correlation is a
convolution, and a convolution is the one operation SIMD units, DSPs, and
FPGAs are built to do efficiently and with genuinely fixed, predictable
timing -- unlike an iterative fit, whose actual iteration count (and
therefore actual runtime) varies with the data. This is the real reason to
build a third method: not because it is more accurate, but because it
represents a different point in the accuracy/hardware-friendliness
tradeoff space that the brief's Real-time section (2d) explicitly asks
about.

WHY THE TEMPLATE IS SEPARABLE
----------------------------------
An isotropic 2-D Gaussian factors as G(x, y) = G(x) * G(y), so correlating
with it can be done as two 1-D passes (rows, then columns) instead of one
full 2-D pass. For an 11-tap kernel that is roughly a 5-6x reduction in
work (O(2K) vs O(K^2) per pixel) -- the single most important
implementation detail for making this fast in practice, not just fast in
principle.

WHY CORRELATION, NOT CONVOLUTION -- AND WHY IT DOESN'T MATTER HERE
--------------------------------------------------------------------------
Strictly, a matched filter correlates with a TIME-REVERSED copy of the
template; convolution uses the template as-is. For a symmetric kernel (a
Gaussian centred at zero, as used here), reversing it changes nothing, so
correlation and convolution are identical -- this distinction is worth
knowing, not worth writing separate code for.

THE SUB-PIXEL STEP: WHY A PLAIN PARABOLA IS THE WRONG SHAPE
-----------------------------------------------------------------
Correlating a Gaussian signal with a Gaussian template produces ANOTHER
Gaussian, not a parabola. (Sketch of why: a Gaussian's Fourier transform is
itself a Gaussian; convolution is multiplication in frequency space, and
the product of two Gaussian-shaped spectra is another Gaussian-shaped
spectrum; its inverse transform is a Gaussian again.) Fitting a parabola to
three samples of a true Gaussian peak is only exact when the peak sits
precisely on a sample or precisely halfway between two -- everywhere else
it is systematically wrong, worst at the quarter-sample offsets. That is
the SAME "pixel locking" S-curve bias already characterised for the
centroid (psf.py, exp02-equivalent reasoning), just produced by a
completely different mechanism: curve-SHAPE mismatch here, versus
truncation/weighting there.

THE FIX: FIT THE PARABOLA TO THE LOGARITHM OF THE SAMPLES INSTEAD
-----------------------------------------------------------------------
log(Gaussian) IS exactly a parabola: if y(x) = A * exp(-(x - delta)^2 / (2 w^2)),
then ln(y(x)) = ln(A) - (x - delta)^2 / (2 w^2), a quadratic in x. So fitting
a parabola to the LOG of three samples, rather than the samples themselves,
recovers the true peak position EXACTLY for noiseless Gaussian data. For
three equally-spaced samples at x = -1, 0, +1 with logs (lm, l0, lp), the
fitted parabola's vertex offset from x=0 is (derived by fitting
f(x) = a*x^2 + b*x + c through the three points and solving for -b/2a):

    delta = 0.5 * (lm - lp) / (lm - 2*l0 + lp)

This costs three extra logarithms over the plain parabola and removes
essentially all of the interpolation-shape bias -- a large accuracy win at
negligible extra cost, which is why it is the default here.

WHY THE PLAIN PARABOLA IS KEPT AS A FALLBACK, NOT DELETED
---------------------------------------------------------------
The logarithm needs all three samples to be strictly positive. After
background subtraction, a low-SNR frame can easily have one of the three
correlation samples come out negative from noise alone -- log-parabola
interpolation is then undefined. Falling back to the plain parabola in
that case is the honest choice: still computable, still roughly reasonable
(if biased), rather than failing outright and throwing away a usable
detection.

A GENUINE TENSION, WORTH STATING PLAINLY: DETECTION-OPTIMAL VS
LOCALISATION-OPTIMAL TEMPLATE WIDTH
--------------------------------------------------------------------------
The matched-filter theorem says the OPTIMAL template for detecting a
Gaussian signal has the SAME width as that signal. But once a
sigma_s-width signal is correlated with a sigma_s-width template, the
resulting peak has width sigma_s * sqrt(2) (from the "Gaussian convolved
with Gaussian" combination above, with equal widths) -- WIDER than the
original spot. A wider peak is FLATTER at its centre, meaning it changes
more slowly with sub-pixel position, which is exactly what makes precise
localisation harder (less curvature at the peak carries less information
about exactly where it is). So the template width that is best for
DECIDING a spot is present is not quite the template width that is best
for PINPOINTING it once found -- a real, small tension, exposed here as
the ``template_sigma_scale`` parameter rather than hidden behind a single
hard-coded choice.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import correlate1d

from .base import Estimate, border_median_background, extract_window, find_brightest_pixel


def parabola_offset(y_minus: float, y_zero: float, y_plus: float) -> float:
    """Sub-sample peak offset from three equally-spaced samples, plain
    (linear-amplitude) quadratic interpolation. Returns an offset in
    samples, clipped to [-1, 1] as a sanity bound against a near-flat or
    inverted local shape."""
    denom = y_minus - 2.0 * y_zero + y_plus
    if denom == 0.0:
        return 0.0
    return float(np.clip(0.5 * (y_minus - y_plus) / denom, -1.0, 1.0))


def log_parabola_offset(y_minus: float, y_zero: float, y_plus: float) -> float:
    """Sub-sample peak offset assuming a Gaussian peak shape (exact for
    noiseless Gaussian data -- see the module docstring's derivation).
    Falls back to the plain parabola if any sample isn't strictly
    positive, since the logarithm is then undefined."""
    if y_minus <= 0.0 or y_zero <= 0.0 or y_plus <= 0.0:
        return parabola_offset(y_minus, y_zero, y_plus)
    lm, l0, lp = math.log(y_minus), math.log(y_zero), math.log(y_plus)
    denom = lm - 2.0 * l0 + lp
    if denom == 0.0:
        return 0.0
    return float(np.clip(0.5 * (lm - lp) / denom, -1.0, 1.0))


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """A simple SAMPLED (not pixel-integrated) 1-D Gaussian, normalised to
    sum to 1.

    Deliberately NOT the exact pixel-integrated response from psf.py: that
    exactness mattered there because it was the RENDERING model, and any
    error would show up as position-dependent bias in the very thing being
    measured. Here the kernel's job is different -- a detection filter, not
    a generative model -- and correlating with a slightly-imperfect kernel
    still gives a valid (if marginally sub-optimal) matched filter. The
    sub-pixel interpolation step is what does the real precision work, not
    the kernel's exactness.
    """
    half = max(int(math.ceil(3.0 * sigma)), 1)
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def matched_filter_estimate(
    image: np.ndarray,
    half_width: int,
    sigma: float,
    template_sigma_scale: float = 1.0,
    interp: str = "log",
    prior: tuple[float, float] | None = None,
) -> Estimate:
    """Estimate a spot's position via Gaussian-template correlation plus
    sub-pixel peak interpolation.

    ``template_sigma_scale`` sets the template width relative to ``sigma``
    (1.0 = matched-filter-optimal for detection; see the module docstring
    for the detection-vs-localisation tension this exposes).
    ``interp`` is ``"log"`` (default, exact for noiseless Gaussian data) or
    ``"parabola"`` (the naive, biased alternative -- kept so the difference
    can be measured directly, the same reason the old bias-vs-clipping
    switches exist elsewhere in this project).
    """
    if prior is None:
        iy, ix = find_brightest_pixel(image)
        cx, cy = float(ix), float(iy)
    else:
        cx, cy = prior

    # Correlate on a window 2 px larger than the reporting half_width, so
    # the correlation peak -- which can land anywhere inside the reporting
    # window, including near its edge -- always has valid neighbours on
    # both sides for the 3-point interpolation.
    window, wx0, wy0 = extract_window(image, cx, cy, half_width + 2)
    bg = border_median_background(window)
    sub = window - bg

    kernel = gaussian_kernel_1d(template_sigma_scale * sigma)
    # mode="nearest" replicates the edge value instead of zero-padding --
    # zero-padding would pretend the scene goes dark just past the window,
    # which pulls a near-edge correlation peak artificially inward.
    corr = correlate1d(sub, kernel, axis=1, mode="nearest")
    corr = correlate1d(corr, kernel, axis=0, mode="nearest")

    h, w = corr.shape
    idx = int(np.argmax(corr))
    py, px = idx // w, idx % w
    if not (1 <= px < w - 1 and 1 <= py < h - 1):
        return Estimate(float("nan"), float("nan"), ok=False)

    offset = log_parabola_offset if interp == "log" else parabola_offset
    dx = offset(float(corr[py, px - 1]), float(corr[py, px]), float(corr[py, px + 1]))
    dy = offset(float(corr[py - 1, px]), float(corr[py, px]), float(corr[py + 1, px]))

    return Estimate(
        x=px + dx + wx0,
        y=py + dy + wy0,
        flux=float(np.maximum(sub, 0.0).sum()),
        bg=bg,
        ok=True,
    )
