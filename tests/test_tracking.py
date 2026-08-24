import numpy as np
import pytest

from sptrack.tracking import (
    AlphaBetaTracker1D,
    KalmanTracker1D,
    filter_sequence,
    sinusoid_response,
)


def test_process_noise_matrix_matches_the_cwna_closed_form():
    dt, q = 1e-3, 4.0
    k = KalmanTracker1D(dt_s=dt, process_psd=q, meas_var_px2=1.0)
    expected = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    assert np.allclose(k._Q, expected)


def test_stationary_target_variance_drops_below_the_measurement_variance():
    """Filtering a stationary target must beat the raw measurement, since
    it is averaging repeated looks at an unchanging quantity."""
    rng = np.random.default_rng(1)
    n, true_pos, meas_std = 4000, 12.34, 0.05
    meas = true_pos + rng.normal(0.0, meas_std, n)
    k = KalmanTracker1D(dt_s=1e-3, process_psd=1e-4, meas_var_px2=meas_std**2)
    out = filter_sequence(k, meas)
    settled = out[500:]
    assert settled.std() < meas_std / 3
    assert settled.mean() == pytest.approx(true_pos, abs=0.01)


def test_tracks_a_constant_velocity_ramp_without_systematic_lag():
    """A constant-velocity model should follow a constant-velocity truth
    with no steady-state position offset, since the model contains it."""
    n, dt, vel = 3000, 1e-3, 20.0
    t = np.arange(n) * dt
    truth = 5.0 + vel * t
    rng = np.random.default_rng(2)
    meas = truth + rng.normal(0.0, 0.02, n)
    k = KalmanTracker1D(dt_s=dt, process_psd=1e-2, meas_var_px2=0.02**2)
    out = filter_sequence(k, meas)
    resid = (out - truth)[500:]
    assert abs(resid.mean()) < 0.005


def test_steady_state_gains_reproduce_the_kalman_output():
    """The alpha-beta filter using the converged gains should match the
    full Kalman recursion once both have settled.

    Two different timescales are involved and only the first is short.
    The Kalman gain itself converges within roughly 1000 frames here
    (alpha reaches 0.007927 against a steady-state 0.007921). The filter
    OUTPUTS take longer to agree, because during the gain transient the
    two filters accumulate different velocity estimates, and at this
    smoothing level beta is about 3e-5, which gives the velocity state a
    memory of tens of thousands of frames. Measured convergence of the
    output difference: 8.6e-3 after index 500, 2.3e-3 after 1000, 3.8e-5
    after 2000. The comparison therefore starts at 2000.
    """
    dt = 1e-3
    k = KalmanTracker1D(dt_s=dt, process_psd=1e-2, meas_var_px2=0.01)
    alpha, beta = k.steady_state_gains()
    assert 0.0 < alpha < 1.0
    assert 0.0 < beta < 1.0

    rng = np.random.default_rng(3)
    meas = 7.0 + np.cumsum(rng.normal(0, 0.01, 3000)) + rng.normal(0, 0.1, 3000)

    k2 = KalmanTracker1D(dt_s=dt, process_psd=1e-2, meas_var_px2=0.01)
    ab = AlphaBetaTracker1D(dt_s=dt, alpha=alpha, beta=beta)
    out_k = filter_sequence(k2, meas)
    out_ab = filter_sequence(ab, meas)
    assert np.allclose(out_k[2000:], out_ab[2000:], atol=2e-4)


def test_heavier_smoothing_gives_more_lag_on_a_tone():
    """The core tradeoff: lowering process noise smooths harder and lags
    more. This must be monotonic or the filter is not behaving."""
    n, dt, freq = 4096, 1e-3, 20.0
    t = np.arange(n) * dt
    truth = 0.3 * np.sin(2 * np.pi * freq * t)
    rng = np.random.default_rng(4)
    meas = truth + rng.normal(0.0, 0.05, n)

    lags = []
    for q in [1e2, 1e1, 1e0]:
        k = KalmanTracker1D(dt_s=dt, process_psd=q, meas_var_px2=0.05**2)
        out = filter_sequence(k, meas)
        r = sinusoid_response(out[500:], truth[500:], freq, dt)
        lags.append(r["lag_ms"])
    assert lags[0] < lags[1] < lags[2]


def test_sinusoid_response_recovers_a_known_shift():
    """Validate the measurement tool itself against a constructed lag."""
    n, dt, freq = 8192, 1e-3, 10.0
    t = np.arange(n) * dt
    truth = np.sin(2 * np.pi * freq * t)
    shift_ms = 5.0
    shifted = np.sin(2 * np.pi * freq * (t - shift_ms / 1000.0))
    r = sinusoid_response(shifted, truth, freq, dt)
    assert r["gain"] == pytest.approx(1.0, abs=0.02)
    assert r["lag_ms"] == pytest.approx(shift_ms, abs=0.2)
