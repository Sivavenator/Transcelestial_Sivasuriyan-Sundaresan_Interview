import numpy as np
import pytest

from sptrack.psf import render_spot
from sptrack.sensor import (
    add_dark_current,
    add_hot_pixels,
    add_photon_noise,
    add_read_noise,
    apply_prnu,
    generate_hot_pixel_mask,
    generate_prnu_map,
)


def test_photon_noise_mean_and_variance_match_poisson_statistics():
    rng = np.random.default_rng(42)

    # lam = 500: high enough that Poisson's relative noise (1/sqrt(lam),
    # ~4.5% here) puts it well into the "looks approximately Gaussian, stable
    # statistics" regime rather than the highly skewed regime of small lam
    # (e.g. lam=5). Also the same order of magnitude as the flux values used
    # in psf.py's own examples (300-1000), so this exercises the noise model
    # in a realistic range rather than an arbitrary extreme.
    lam = 500.0
    mean_image = np.full((10, 10), lam)  # 100 independent pixels

    # n_trials = 5000 is not a round-number guess -- it's chosen backward
    # from a target precision. We want the POOLED variance estimator's
    # standard error to land around ~1 count (0.2% of lam=500), so the test
    # has real power to catch a genuine bug rather than just "looking close."
    # SE(pooled variance) = sqrt((lam + 2*lam^2) / n_total), and
    # n_total = n_trials * 100 pixels. Setting SE = 1 and solving:
    #   n_total ~= lam + 2*lam^2 = 500 + 500,000 = 500,500
    #   n_trials ~= 500,500 / 100 = 5005 ~= 5000
    n_trials = 5000
    samples = np.stack(
        [add_photon_noise(mean_image, rng) for _ in range(n_trials)]
    )

    # Pool across all 100 pixels x 5000 trials rather than checking each
    # pixel's own sample variance separately: a *variance* estimator is
    # itself noisy (its standard error involves the 4th moment, not just
    # lambda), so 100 independent per-pixel checks against a fixed tolerance
    # will occasionally flag one by chance even when nothing is wrong.
    # Pooling to 500,000 draws shrinks that estimator noise until it's no
    # longer a source of test flakiness.
    pooled = samples.ravel()
    pooled_mean = pooled.mean()
    pooled_var = pooled.var()

    # Poisson: Var[N] = E[N] = lambda. Standard error on the pooled mean is
    # sqrt(lambda / n) ~ 0.03; on the pooled variance, sqrt((lambda + 2*lambda^2) / n) ~ 1.0.
    #
    # Tolerances (1.0 and 10.0) are set at roughly 10x those standard errors,
    # not 2-3x: at 10 SE, a false failure from ordinary sampling luck is
    # astronomically unlikely (this test should never be flaky), while a real
    # bug -- e.g. a noise_gain/scaling error of even 10-20% -- would move the
    # empirical variance by 50-100+ counts, still 5-10x past this tolerance.
    # So the margin is wide enough to be robust, not so wide it stops
    # meaning anything.
    assert pooled_mean == pytest.approx(lam, abs=1.0)
    assert pooled_var == pytest.approx(lam, abs=10.0)


def test_photon_noise_is_nonnegative_and_integer_valued():
    rng = np.random.default_rng(0)
    mean_image = np.array([[0.0, 1.0], [10.0, 1000.0]])
    noisy = add_photon_noise(mean_image, rng)

    assert np.all(noisy >= 0)
    assert np.all(noisy == np.round(noisy))  # photon counts are integers


def test_photon_noise_is_reproducible_with_same_seed():
    mean_image = np.full((5, 5), 100.0)
    a = add_photon_noise(mean_image, np.random.default_rng(7))
    b = add_photon_noise(mean_image, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_read_noise_mean_and_variance_match_gaussian_statistics():
    rng = np.random.default_rng(3)

    # sigma_read = 5.0 e-: a plausible real sensor value (typical CMOS/CCD
    # read noise sits roughly in the 1-10 e- range).
    sigma_read = 5.0
    image = np.zeros((10, 10))  # baseline of 0; read noise is additive

    # n_trials = 200 is derived the same way as the photon-noise test: for a
    # Gaussian, Var(sample variance) ~= 2*sigma^4 / n for large n, so
    # SE(pooled variance) = sigma^2 * sqrt(2 / n_total). Targeting an SE of
    # ~1% of sigma_read^2 = 25 (i.e. SE ~= 0.25), with n_total = n_trials * 100:
    #   0.25 = 25 * sqrt(2 / n_total)  =>  n_total = 2 / (0.25/25)^2 = 20,000
    #   n_trials = 20,000 / 100 = 200
    n_trials = 200
    samples = np.stack([add_read_noise(image, sigma_read, rng) for _ in range(n_trials)])
    pooled = samples.ravel()

    # Tolerances at ~10x the derived standard errors, same reasoning as the
    # photon-noise test above: robust against sampling luck, still tight
    # enough to catch a real scaling bug.
    assert pooled.mean() == pytest.approx(0.0, abs=0.35)   # SE(mean) ~= 5/sqrt(20000) ~= 0.035
    assert pooled.var() == pytest.approx(sigma_read**2, abs=2.5)  # SE(var) ~= 0.25


def test_read_noise_magnitude_is_independent_of_signal_brightness():
    # The whole point of read noise, distinguishing it from photon noise: its
    # magnitude does not depend on how bright the underlying pixel is. Apply
    # the same sigma_read to a dark image and a bright one, with two
    # independently-seeded (not shared) generators so the two noise draws are
    # genuinely separate samples, and confirm the empirical noise std comes
    # out the same either way.
    sigma_read = 5.0
    n_trials = 200
    dark_image = np.zeros((10, 10))
    bright_image = np.full((10, 10), 5000.0)

    dark_rng = np.random.default_rng(101)
    bright_rng = np.random.default_rng(102)
    dark_samples = np.stack(
        [add_read_noise(dark_image, sigma_read, dark_rng) for _ in range(n_trials)]
    )
    bright_samples = np.stack(
        [add_read_noise(bright_image, sigma_read, bright_rng) for _ in range(n_trials)]
    )

    dark_std = (dark_samples - dark_image).ravel().std()
    bright_std = (bright_samples - bright_image).ravel().std()

    # Same n_trials/tolerance derivation as above: SE(std) ~= sigma/sqrt(2n) ~= 0.13,
    # so a tolerance of 1.0 is a comfortable ~8x margin.
    assert dark_std == pytest.approx(sigma_read, abs=1.0)
    assert bright_std == pytest.approx(sigma_read, abs=1.0)


def test_read_noise_is_reproducible_with_same_seed():
    image = np.full((5, 5), 100.0)
    a = add_read_noise(image, 5.0, np.random.default_rng(11))
    b = add_read_noise(image, 5.0, np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_dark_current_mean_and_variance_match_poisson_statistics():
    rng = np.random.default_rng(5)

    # dark_rate_e_per_s=500, exposure_s=1.0 gives mean_dark=500 -- deliberately
    # the same lambda as the photon-noise test above, since dark current is
    # the identical Poisson mechanism (just thermally- rather than
    # optically-generated electrons), so the same n_trials=5000 derivation
    # for a ~1-count pooled-variance standard error applies unchanged.
    dark_rate_e_per_s = 500.0
    exposure_s = 1.0
    mean_dark = dark_rate_e_per_s * exposure_s
    image = np.zeros((10, 10))

    n_trials = 5000
    samples = np.stack(
        [add_dark_current(image, dark_rate_e_per_s, exposure_s, rng) for _ in range(n_trials)]
    )
    pooled = samples.ravel()

    assert pooled.mean() == pytest.approx(mean_dark, abs=1.0)
    assert pooled.var() == pytest.approx(mean_dark, abs=10.0)


def test_dark_current_scales_with_exposure_time():
    # Doubling exposure time should double the mean dark charge -- this is
    # the whole reason exposure_s is a parameter rather than folded into a
    # fixed rate.
    rng = np.random.default_rng(6)
    dark_rate_e_per_s = 1000.0
    image = np.zeros((10, 10))
    n_trials = 2000

    short = np.stack(
        [add_dark_current(image, dark_rate_e_per_s, 0.1, rng) for _ in range(n_trials)]
    )
    long = np.stack(
        [add_dark_current(image, dark_rate_e_per_s, 0.2, rng) for _ in range(n_trials)]
    )

    assert long.ravel().mean() == pytest.approx(2 * short.ravel().mean(), rel=0.05)


def test_dark_current_plus_photon_noise_variance_adds_by_poisson_additivity():
    # Validates the docstring's mathematical claim directly: applying photon
    # noise to a signal, then adding independent dark-current Poisson noise,
    # should give a TOTAL variance equal to signal + mean_dark -- because
    # Poisson(a) + Poisson(b) is distributed as Poisson(a+b). If this test
    # failed, the docstring's justification for treating dark current as a
    # separate function (rather than combining means first) would be wrong.
    rng = np.random.default_rng(9)
    signal = 300.0
    dark_rate_e_per_s = 200.0
    exposure_s = 1.0
    mean_dark = dark_rate_e_per_s * exposure_s
    image = np.full((10, 10), signal)

    n_trials = 5000
    totals = np.stack(
        [
            add_dark_current(
                add_photon_noise(image, rng), dark_rate_e_per_s, exposure_s, rng
            )
            for _ in range(n_trials)
        ]
    )
    pooled = totals.ravel()

    expected_mean = signal + mean_dark
    expected_var = signal + mean_dark  # Poisson: Var = mean, for the combined process
    assert pooled.mean() == pytest.approx(expected_mean, abs=1.5)
    assert pooled.var() == pytest.approx(expected_var, abs=15.0)


def test_dark_current_is_reproducible_with_same_seed():
    image = np.full((5, 5), 0.0)
    a = add_dark_current(image, 500.0, 1.0, np.random.default_rng(13))
    b = add_dark_current(image, 500.0, 1.0, np.random.default_rng(13))
    assert np.array_equal(a, b)


def test_hot_pixel_mask_fraction_matches_the_requested_rate():
    rng = np.random.default_rng(21)

    # fraction=0.05 on a 50x50=2500-pixel grid: this is a WAY higher rate
    # than a real sensor's hot-pixel fraction (real defect rates are more
    # like 1e-5 to 1e-3), chosen purely so the test has enough hot pixels to
    # check statistically without needing an enormous grid. The SE on a
    # binomial proportion is sqrt(p(1-p)/n) = sqrt(0.05*0.95/2500) ~= 0.0044;
    # a tolerance of 0.03 is a ~7x margin.
    fraction = 0.05
    mask = generate_hot_pixel_mask((50, 50), fraction, rng)

    empirical_fraction = mask.mean()
    assert empirical_fraction == pytest.approx(fraction, abs=0.03)
    assert mask.dtype == np.bool_


def test_hot_pixel_mask_is_fixed_not_redrawn_per_frame():
    # The defining property that distinguishes hot pixels from every other
    # noise source: the SAME mask, generated once, must be reusable across
    # frames unchanged. Two independent calls with the same seed must
    # produce the identical mask (this is really a reproducibility check,
    # but stated in terms of the physical property it stands in for: this
    # is a fixed sensor defect, not fresh randomness).
    a = generate_hot_pixel_mask((10, 10), 0.1, np.random.default_rng(4))
    b = generate_hot_pixel_mask((10, 10), 0.1, np.random.default_rng(4))
    assert np.array_equal(a, b)


def test_add_hot_pixels_only_changes_masked_pixels():
    # Pixels outside the mask should be EXACTLY unchanged (not "close to"),
    # since add_hot_pixels adds exactly 0.0 there by construction.
    rng = np.random.default_rng(22)
    image = np.full((10, 10), 50.0)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 7] = True  # exactly one hot pixel, at a known location

    result = add_hot_pixels(image, mask, hot_rate_e_per_s=1e6, exposure_s=1.0, rng=rng)

    unmasked = ~mask
    assert np.array_equal(result[unmasked], image[unmasked])
    assert result[3, 7] > image[3, 7]  # the hot pixel itself should have jumped


def test_add_hot_pixels_matches_poisson_statistics_at_the_elevated_rate():
    # Using an all-True mask isolates the elevated-rate Poisson draw from the
    # mask-generation logic (tested separately above), and lets this reuse
    # the exact same lambda=500, n_trials=5000 derivation as the dark-current
    # test -- it's the identical Poisson mechanism, just at a location that's
    # fixed rather than freshly random.
    rng = np.random.default_rng(23)
    hot_rate_e_per_s = 500.0
    exposure_s = 1.0
    mean_hot = hot_rate_e_per_s * exposure_s
    image = np.zeros((10, 10))
    all_hot_mask = np.ones((10, 10), dtype=bool)

    n_trials = 5000
    samples = np.stack(
        [
            add_hot_pixels(image, all_hot_mask, hot_rate_e_per_s, exposure_s, rng)
            for _ in range(n_trials)
        ]
    )
    pooled = samples.ravel()

    assert pooled.mean() == pytest.approx(mean_hot, abs=1.0)
    assert pooled.var() == pytest.approx(mean_hot, abs=10.0)


def test_prnu_map_statistics_match_the_requested_gain_distribution():
    rng = np.random.default_rng(30)

    # sigma_prnu=0.02 (2%): realistic for a consumer sensor. A 100x100 grid
    # (10,000 pixels) gives SE(mean) = sigma/sqrt(n) ~= 0.0002 and
    # SE(std) ~= sigma/sqrt(2n) ~= 0.00014 -- tolerances below are ~10x both.
    sigma_prnu = 0.02
    prnu_map = generate_prnu_map((100, 100), sigma_prnu, rng)

    assert prnu_map.mean() == pytest.approx(1.0, abs=0.002)
    assert prnu_map.std() == pytest.approx(sigma_prnu, abs=0.0015)


def test_prnu_map_is_fixed_with_same_seed():
    a = generate_prnu_map((10, 10), 0.02, np.random.default_rng(31))
    b = generate_prnu_map((10, 10), 0.02, np.random.default_rng(31))
    assert np.array_equal(a, b)


def test_apply_prnu_multiplies_elementwise():
    signal = np.array([[100.0, 200.0], [300.0, 400.0]])
    gain = np.array([[1.02, 0.98], [1.00, 1.05]])
    result = apply_prnu(signal, gain)
    assert np.array_equal(result, signal * gain)


def test_uniform_gain_does_not_shift_the_centroid():
    # A gain map that's exactly 1.0 everywhere (sigma_prnu=0, the "perfectly
    # uniform sensor" case) should leave the centroid EXACTLY where it was --
    # scaling every pixel in a window by the same constant cannot move a
    # weighted average. This is the baseline the position-dependent-bias
    # claim below is contrasted against.
    x0 = 12.3
    spot = render_spot((25, 25), x0=x0, y0=11.7, flux=1000.0, sigma=1.75)
    uniform_gain = np.ones((25, 25))

    adjusted = apply_prnu(spot, uniform_gain)
    h, w = adjusted.shape
    xs = np.arange(w)
    cx = (adjusted.sum(axis=0) * xs).sum() / adjusted.sum()

    assert cx == pytest.approx(x0, abs=1e-9)


def test_prnu_introduces_a_position_dependent_bias():
    # The core physical claim in the docstring: a FIXED, non-uniform gain
    # map biases the centroid by a DIFFERENT amount depending on where the
    # spot's sub-pixel centre sits -- unlike a uniform gain (zero bias,
    # tested above) or random per-frame noise (which would average toward
    # zero, not stay fixed). sigma_prnu=0.05 here is larger than the
    # realistic 0.02 used in the statistics test above, deliberately, so the
    # effect is clearly visible without needing a huge window -- this test
    # demonstrates the mechanism exists, not its realistic magnitude.
    rng = np.random.default_rng(32)
    prnu_map = generate_prnu_map((25, 25), sigma_prnu=0.05, rng=rng)

    def centroid_bias(x0: float) -> float:
        spot = render_spot((25, 25), x0=x0, y0=11.7, flux=1000.0, sigma=1.75)
        adjusted = apply_prnu(spot, prnu_map)
        xs = np.arange(adjusted.shape[1])
        cx = (adjusted.sum(axis=0) * xs).sum() / adjusted.sum()
        return cx - x0

    bias_a = centroid_bias(10.0)   # spot centred on a pixel
    bias_b = centroid_bias(10.5)   # spot centred between two pixels

    # The two biases should differ -- if PRNU only ever produced the same
    # bias regardless of sub-pixel position, it would just be a (harmless,
    # flux-only) calibration offset rather than a position-dependent one.
    assert bias_a != pytest.approx(bias_b, abs=1e-6)
