"""Detect a periodic disturbance's frequency and amplitude from a
RECOVERED (estimator-output, not ground-truth) trajectory -- the last
piece of the brief's core dynamic-tracking ask: "detect the periodic
disturbance's frequency + amplitude, report how close detected values are
to injected values."

WHY A HANN WINDOW BEFORE THE FFT, NOT A PLAIN (RECTANGULAR) ONE
---------------------------------------------------------------------
The recovered trajectory is not periodic within the capture window --
drift (a random walk) does not return to its starting value, so a plain
FFT implicitly assumes the signal repeats and sees a discontinuity at the
wrap-around point. That discontinuity's energy spreads (leaks) across the
ENTIRE spectrum, potentially burying a small disturbance peak under
spurious broadband power. A Hann window tapers both ends of the signal to
zero before the FFT, removing that artificial discontinuity -- standard
practice for detecting a periodic component in a non-periodic background,
at a well-known cost (correctable, see below): it slightly widens each
frequency's spectral peak (a wider "main lobe") in exchange for much lower
leakage everywhere else.

WHY THE LOW-FREQUENCY BAND IS EXCLUDED FROM THE PEAK SEARCH
-----------------------------------------------------------------
Drift's own power is concentrated at the lowest frequencies (its defining
1/f^2 signature, `sptrack/trajectory.py`), and even windowed, some of that
power remains near DC. Searching for "the biggest peak" across the WHOLE
spectrum risks finding drift's own low-frequency power rather than the
actual disturbance. `exclude_below_hz` (default 2.0 Hz) removes that
region from the search entirely, based directly on the spectral
separation already demonstrated in `figures/exp03a_trajectory_diagnostic.png`
(drift power falls off steeply well before 5 Hz; the disturbance sits at
tens of Hz).

WHY AMPLITUDE IS READ FROM THE SINGLE PEAK BIN, NOT SUMMED ACROSS THE
LEAKED LOBE
------------------------------------------------------------------------------
A real sinusoid whose frequency doesn't land exactly on an FFT bin (true
here: at n=4096, dt=1ms, the frequency resolution is 0.244 Hz, and
20.0 Hz sits at fractional bin index 81.92, not an integer) leaks some
energy into neighbouring bins -- the tempting fix is to sum energy across
several bins around the peak to recover it. That was tried and rejected
here: summing (or RSS-combining) several bins' amplitude readings
OVER-estimates the true amplitude by 20-100% in practice, because it
also sums in the window function's own non-zero sidelobes as if they were
independent signal, double-counting energy that a single well-corrected
peak-bin reading already mostly captures. The single peak bin, corrected
for the Hann window's coherent gain (``2 * |X| / sum(window)`` -- the
standard formula for reading a real amplitude off a windowed one-sided
FFT), was verified DIRECTLY against a known-amplitude test tone (both
bin-aligned and, more realistically, not) and recovers the true amplitude
to within 0.4% even off-bin -- see tests/test_disturbance.py. Trusting a
tested, simple formula over an untested "more sophisticated" one that
turned out to be wrong is a deliberate methodological choice, not an
oversight.

WHY FREQUENCY IS REPORTED AT BIN RESOLUTION, NOT SUB-BIN-INTERPOLATED
---------------------------------------------------------------------------
A further refinement (parabolic interpolation across the 3 bins around
the peak, the same idea already used for sub-pixel position in
`estimators/matched_filter.py`) could shrink the frequency error below one
bin width. It was left out here because the plain bin-resolution estimate
already lands within one bin of the true frequency on the default (easy)
scenario (0.244 Hz resolution vs. a 20.02 Hz detected vs. 20.0 Hz true
disturbance) -- the honest, checked answer, not a claim of better
precision than has actually been verified.
"""

from __future__ import annotations

import numpy as np


def detect_disturbance(signal: np.ndarray, dt_s: float, exclude_below_hz: float = 2.0) -> dict:
    """Detect a single dominant periodic component in ``signal`` (a 1-D
    position time series, uniformly sampled at ``dt_s``).

    Returns the detected frequency (Hz), amplitude (same units as
    ``signal``), and the FFT's own frequency resolution (Hz) -- the
    resolution is returned alongside the estimate because it IS the
    estimate's honest frequency uncertainty, not a separate diagnostic.
    """
    n = len(signal)
    window = np.hanning(n)
    windowed = (signal - signal.mean()) * window

    spec = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=dt_s)
    amp_spectrum = 2.0 * np.abs(spec) / window.sum()

    freq_resolution = 1.0 / (n * dt_s)
    searchable = freqs > exclude_below_hz
    candidate_idx = np.where(searchable)[0]
    peak_idx = candidate_idx[np.argmax(amp_spectrum[candidate_idx])]

    return {
        "freq_hz": float(freqs[peak_idx]),
        "amp_px": float(amp_spectrum[peak_idx]),
        "freq_resolution_hz": float(freq_resolution),
    }
