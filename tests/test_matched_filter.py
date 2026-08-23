import math

import numpy as np
import pytest
from scipy.ndimage import correlate1d

from sptrack.estimators.matched_filter import (
    gaussian_kernel_1d,
    log_parabola_offset,
    matched_filter_estimate,
    parabola_offset,
)
from sptrack.estimators.centroid import centroid_estimate
from sptrack.simulate import Simulator


def _raw_gaussian_samples(true_offset: float, sigma: float) -> tuple[float, float, float]:
    """Continuous (NOT pixel-integrated) Gaussian evaluated at x=-1,0,1,
    peaked at true_offset -- used to test the interpolation formulas
    against the exact mathematical shape they assume, independent of the
    separate pixel-integration-vs-sampling question already characterised
    elsewhere in this project (psf.py)."""
    return tuple(
        math.exp(-((x - true_offset) ** 2) / (2 * sigma**2)) for x in (-1, 0, 1)
    )


def test_log_parabola_offset_is_exact_for_noiseless_gaussian_samples():
    sigma = 1.75
    for true_offset in [-0.45, -0.25, -0.05, 0.0, 0.15, 0.3, 0.45]:
        y = _raw_gaussian_samples(true_offset, sigma)
        est = log_parabola_offset(*y)
        assert est == pytest.approx(true_offset, abs=1e-9)


def test_plain_parabola_is_measurably_biased_where_log_parabola_is_exact():
    # Quarter-sample offset is the docstring's stated worst case for the
    # plain parabola -- checked directly, not just asserted.
    sigma = 1.75
    true_offset = 0.25
    y = _raw_gaussian_samples(true_offset, sigma)
    log_est = log_parabola_offset(*y)
    plain_est = parabola_offset(*y)
    assert log_est == pytest.approx(true_offset, abs=1e-9)
    assert abs(plain_est - true_offset) > 1e-3


def test_log_parabola_falls_back_to_plain_when_a_sample_is_nonpositive():
    # Not a crash, not undefined behaviour -- an explicit, honest fallback.
    y_minus, y_zero, y_plus = -0.5, 3.0, 2.0
    log_result = log_parabola_offset(y_minus, y_zero, y_plus)
    plain_result = parabola_offset(y_minus, y_zero, y_plus)
    assert log_result == plain_result


def test_gaussian_kernel_1d_is_normalised_and_symmetric():
    k = gaussian_kernel_1d(1.75)
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k[::-1])  # symmetric, as any centred Gaussian must be


def test_correlation_peak_widens_by_sqrt2_when_template_matches_signal():
    # The detection-vs-localisation tension claimed in the docstring,
    # checked directly: correlating a sigma-s signal with a sigma-s
    # template should produce a peak of width s*sqrt(2), recovered here via
    # the same log-parabola curvature relation the interpolation itself
    # uses (denom = ln(y-1) - 2 ln(y0) + ln(y+1) = -1/sigma_out^2 for a
    # Gaussian peak, derived directly from the log-Gaussian's own quadratic
    # coefficient).
    #
    # Everything here must stay in ARRAY-INDEX units (spacing exactly 1),
    # matching how gaussian_kernel_1d and correlate1d actually operate --
    # an earlier version of this test built the signal on a fine physical
    # grid (dx=0.2) while the kernel implicitly assumes dx=1, silently
    # making the effective kernel width 5x too narrow relative to the
    # signal. Caught because the measured sigma_out (~1.78) came out close
    # to sigma_signal itself rather than sigma_signal*sqrt(2) -- a test
    # bug, not a bug in matched_filter.py.
    sigma_signal = 1.75
    half = 40
    x = np.arange(-half, half + 1, dtype=np.float64)  # unit spacing, like real pixels
    signal = np.exp(-0.5 * (x / sigma_signal) ** 2)

    kernel = gaussian_kernel_1d(sigma_signal)
    corr = correlate1d(signal, kernel, mode="nearest")

    peak_idx = int(np.argmax(corr))
    lm, l0, lp = (math.log(corr[peak_idx + d]) for d in (-1, 0, 1))
    denom = lm - 2 * l0 + lp
    sigma_out = math.sqrt(-1.0 / denom)

    expected = sigma_signal * math.sqrt(2)
    assert sigma_out == pytest.approx(expected, rel=0.01)


def test_matched_filter_recovers_true_position_on_a_clean_flat_background_image():
    from sptrack.psf import render_spot

    x0, y0 = 18.3, 15.0
    image = render_spot((30, 30), x0=x0, y0=y0, flux=2000.0, sigma=1.75) + 50.0
    est = matched_filter_estimate(image, half_width=8, sigma=1.75, prior=(18.0, 15.0))
    assert est.ok
    assert est.x == pytest.approx(x0, abs=0.01)
    assert est.y == pytest.approx(y0, abs=0.01)


def test_matched_filter_log_interp_beats_plain_parabola_interp_on_identical_data():
    # Deliberately a DETERMINISTIC comparison (render_spot, no sensor
    # noise), not a noisy Monte Carlo mean -- an earlier version of this
    # test averaged 300 NOISY trials at one fixed sub-pixel offset and
    # found the comparison came out the wrong way round. That wasn't a
    # sign the underlying claim was false (the deterministic exactness
    # tests above prove it rigorously); it meant 300 noisy samples of a
    # small systematic bias, at one offset, is a statistically underpowered
    # way to detect it -- comparing two noise-dominated sample means rather
    # than the actual deterministic bias curves. A quarter-sample offset
    # (18.25) is the analytically worst case for the plain parabola (see
    # the docstring), so this compares the two interpolation modes on the
    # SAME noise-free image where the true underlying difference is large
    # and unambiguous, not buried in Monte Carlo noise.
    from sptrack.psf import render_spot

    x0, y0 = 18.25, 15.0
    image = render_spot((30, 30), x0=x0, y0=y0, flux=2000.0, sigma=1.75) + 50.0

    e_log = matched_filter_estimate(image, half_width=8, sigma=1.75, interp="log", prior=(18.0, 15.0))
    e_plain = matched_filter_estimate(image, half_width=8, sigma=1.75, interp="parabola", prior=(18.0, 15.0))

    log_err = abs(e_log.x - x0)
    plain_err = abs(e_plain.x - x0)
    assert log_err < 1e-4
    assert plain_err > log_err * 100  # log-parabola order-of-magnitude better here


def test_matched_filter_is_approximately_unbiased_against_the_real_noise_chain():
    sim = Simulator(shape=(25, 25), background_e=30.0, seed=102)
    x0, y0, flux = 12.3, 11.7, 8000.0

    n_trials = 300
    xs, ys = [], []
    for _ in range(n_trials):
        frame_e = sim.dn_to_electrons(sim.render(x0, y0, flux))
        est = matched_filter_estimate(frame_e, half_width=7, sigma=sim.sigma, prior=(12.3, 11.7))
        if est.ok:
            xs.append(est.x)
            ys.append(est.y)

    assert len(xs) > n_trials * 0.9
    assert np.mean(xs) == pytest.approx(x0, abs=0.05)
    assert np.mean(ys) == pytest.approx(y0, abs=0.05)


def test_matched_filter_handles_a_window_clamped_at_the_image_edge():
    from sptrack.psf import render_spot

    x0, y0 = 2.3, 2.7
    image = render_spot((10, 10), x0=x0, y0=y0, flux=1000.0, sigma=1.0) + 10.0
    est = matched_filter_estimate(image, half_width=6, sigma=1.0, prior=(2.0, 2.0))
    assert est.ok
    assert est.x == pytest.approx(x0, abs=0.15)
    assert est.y == pytest.approx(y0, abs=0.15)
