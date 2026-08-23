import numpy as np
import pytest

from sptrack.sensor import add_photon_noise


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
