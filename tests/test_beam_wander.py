import numpy as np
import pytest

from sptrack.beam_wander import generate_beam_wander


def test_output_lengths_and_reproducibility():
    dx1, dy1 = generate_beam_wander(500, 1e-3, seed=1)
    dx2, dy2 = generate_beam_wander(500, 1e-3, seed=1)
    assert len(dx1) == 500 and len(dy1) == 500
    assert np.array_equal(dx1, dx2)
    assert np.array_equal(dy1, dy2)


def test_x_and_y_are_independent_draws():
    dx, dy = generate_beam_wander(2000, 1e-3, sigma_px=0.15, seed=2)
    assert not np.array_equal(dx, dy)
    assert abs(np.corrcoef(dx, dy)[0, 1]) < 0.1


def test_std_matches_requested_sigma():
    dx, dy = generate_beam_wander(200000, 1e-3, sigma_px=0.15, tau_s=20e-3, seed=3)
    assert dx.std() == pytest.approx(0.15, rel=0.02)
    assert dy.std() == pytest.approx(0.15, rel=0.02)


def test_spectrally_distinguishable_from_white_noise_despite_equal_variance():
    """The whole point of this module: beam wander and white jitter can
    share identical time-domain variance and still be cleanly separable by
    spectral shape alone."""
    n, dt = 100000, 1e-3
    dx, _ = generate_beam_wander(n, dt, sigma_px=0.15, tau_s=20e-3, seed=4)
    jitter = np.random.default_rng(5).normal(0.0, 0.15, n)
    assert dx.std() == pytest.approx(jitter.std(), rel=0.1)

    freqs = np.fft.rfftfreq(n, d=dt)
    low = (freqs > 0.5) & (freqs < 10)
    high = (freqs >= 100) & (freqs < 400)

    p_wander = np.abs(np.fft.rfft(dx)) ** 2
    p_jitter = np.abs(np.fft.rfft(jitter)) ** 2
    wander_ratio = p_wander[low].sum() / p_wander[high].sum()
    jitter_ratio = p_jitter[low].sum() / p_jitter[high].sum()

    assert wander_ratio > 10 * jitter_ratio


def test_shorter_coherence_time_decorrelates_faster():
    n, dt = 50000, 1e-3
    dx_fast, _ = generate_beam_wander(n, dt, sigma_px=0.15, tau_s=5e-3, seed=6)
    dx_slow, _ = generate_beam_wander(n, dt, sigma_px=0.15, tau_s=50e-3, seed=6)
    lag = 5
    ac_fast = np.corrcoef(dx_fast[:-lag], dx_fast[lag:])[0, 1]
    ac_slow = np.corrcoef(dx_slow[:-lag], dx_slow[lag:])[0, 1]
    assert ac_fast < ac_slow
