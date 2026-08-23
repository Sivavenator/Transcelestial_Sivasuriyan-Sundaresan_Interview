import numpy as np
import pytest

from sptrack.sensor import add_photon_noise


def test_photon_noise_mean_and_variance_match_poisson_statistics():
    rng = np.random.default_rng(42)
    lam = 500.0
    mean_image = np.full((10, 10), lam)  # 100 independent pixels

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
    # A tolerance of 1 and 10 respectively is still >>10x the expected noise.
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
