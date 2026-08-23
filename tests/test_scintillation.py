import numpy as np
import pytest

from sptrack.scintillation import generate_scintillation


def test_output_length_and_reproducibility():
    a = generate_scintillation(500, 1e-3, seed=1)
    b = generate_scintillation(500, 1e-3, seed=1)
    assert len(a) == 500
    assert np.array_equal(a, b)


def test_all_multipliers_positive():
    mult = generate_scintillation(2000, 1e-3, sigma_ln=0.6, seed=2)
    assert np.all(mult > 0)


def test_mean_multiplier_is_unbiased():
    """E[I] = exp(mu + sigma_ln^2/2) should equal 1.0 by construction
    (mu = -sigma_ln^2/2) -- checked over many samples, not assumed."""
    mult = generate_scintillation(200000, 1e-3, sigma_ln=0.4, tau_s=5e-3, seed=3)
    assert mult.mean() == pytest.approx(1.0, abs=0.01)


def test_log_std_matches_requested_sigma_ln():
    mult = generate_scintillation(200000, 1e-3, sigma_ln=0.4, tau_s=5e-3, seed=4)
    assert np.log(mult).std() == pytest.approx(0.4, rel=0.02)


def test_autocorrelation_matches_theoretical_ar1_decay():
    n, dt, tau_s = 100000, 1e-3, 5e-3
    mult = generate_scintillation(n, dt, sigma_ln=0.4, tau_s=tau_s, seed=5)
    log_x = np.log(mult) - np.log(mult).mean()
    phi_theory = np.exp(-dt / tau_s)
    for lag in [1, 5, 10, 20]:
        empirical = np.corrcoef(log_x[:-lag], log_x[lag:])[0, 1]
        assert empirical == pytest.approx(phi_theory**lag, abs=0.03)


def test_shorter_coherence_time_decorrelates_faster():
    n, dt = 50000, 1e-3
    mult_fast = generate_scintillation(n, dt, sigma_ln=0.4, tau_s=1e-3, seed=6)
    mult_slow = generate_scintillation(n, dt, sigma_ln=0.4, tau_s=20e-3, seed=6)
    lag = 5
    ac_fast = np.corrcoef(np.log(mult_fast[:-lag]), np.log(mult_fast[lag:]))[0, 1]
    ac_slow = np.corrcoef(np.log(mult_slow[:-lag]), np.log(mult_slow[lag:]))[0, 1]
    assert ac_fast < ac_slow
