import math

import numpy as np
import pytest

from sptrack.psf import diameter_1e2_to_sigma, sample_true_sigma


def test_diameter_1e2_to_sigma_matches_the_established_constant():
    # 7 px diameter is the brief's stated spot size, and sigma=1.75 has been
    # used as the default throughout this project's tests -- this confirms
    # the two are actually the same number, not a coincidence.
    assert diameter_1e2_to_sigma(7.0) == pytest.approx(1.75, abs=1e-12)


def test_diameter_1e2_to_sigma_satisfies_its_own_definition():
    # Direct check against the physical definition the conversion is built
    # on: at r = diameter/2 (the radius the diameter spec refers to), the
    # continuous Gaussian intensity I(r) = exp(-r^2 / (2*sigma^2)) should
    # equal exp(-2) = 1/e^2 -- that's what "1/e^2" means, by construction.
    for diameter in [4.0, 7.0, 10.0, 15.5]:
        sigma = diameter_1e2_to_sigma(diameter)
        r = diameter / 2.0
        intensity_ratio = math.exp(-(r**2) / (2 * sigma**2))
        assert intensity_ratio == pytest.approx(math.exp(-2), rel=1e-9)


def test_sample_true_sigma_statistics_match_the_requested_tolerance():
    rng = np.random.default_rng(60)
    nominal = 1.75
    tolerance_frac = 0.1

    n = 50_000
    samples = np.array([sample_true_sigma(nominal, tolerance_frac, rng) for _ in range(n)])

    # SE(mean) = sigma/sqrt(n) = (1.75*0.1)/sqrt(50000) ~= 0.00078; tolerance
    # below is ~30x that, consistent with the margin used throughout this
    # project's other statistical tests.
    assert samples.mean() == pytest.approx(nominal, abs=0.03)
    assert samples.std() == pytest.approx(nominal * tolerance_frac, rel=0.05)


def test_sample_true_sigma_never_returns_non_positive_even_at_extreme_tolerance():
    # A large tolerance_frac would routinely draw negative values from the
    # underlying Normal without the floor -- confirms the floor actually
    # engages, not just that it exists in the code.
    rng = np.random.default_rng(61)
    nominal = 1.75
    samples = np.array(
        [sample_true_sigma(nominal, tolerance_frac=5.0, rng=rng) for _ in range(10_000)]
    )
    assert np.all(samples >= 0.1 * nominal)
    # pytest.approx doesn't broadcast elementwise against a NumPy array in an
    # `==` comparison (it compares the whole array as one object), so use
    # np.isclose directly to check the floor value actually appears.
    assert np.any(np.isclose(samples, 0.1 * nominal))
