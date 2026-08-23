import numpy as np
import pytest

from sptrack.sensor import add_photon_noise, add_read_noise


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
