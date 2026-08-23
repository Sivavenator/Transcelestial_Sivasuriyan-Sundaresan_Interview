import numpy as np
import pytest

from sptrack.disturbance import detect_disturbance
from sptrack.sequence import recover_trajectory, render_sequence
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux
from sptrack.trajectory import TrajectoryConfig, generate_trajectory


def test_recovers_a_bin_aligned_clean_tone_almost_exactly():
    n, dt = 4096, 1e-3
    freq_resolution = 1.0 / (n * dt)
    f0 = 20 * freq_resolution  # exactly on a bin
    t = np.arange(n) * dt
    signal = 0.3 * np.sin(2 * np.pi * f0 * t)

    result = detect_disturbance(signal, dt)
    assert result["freq_hz"] == pytest.approx(f0, abs=1e-9)
    assert result["amp_px"] == pytest.approx(0.3, rel=0.01)


def test_recovers_a_non_bin_aligned_clean_tone_within_one_bin_and_1pct_amplitude():
    """20 Hz does NOT land on an exact FFT bin at this resolution -- the
    realistic case, and the one that matters (the trajectory's default
    disturb_freq_hz is 20.0)."""
    n, dt = 4096, 1e-3
    freq_resolution = 1.0 / (n * dt)
    t = np.arange(n) * dt
    signal = 0.3 * np.sin(2 * np.pi * 20.0 * t)

    result = detect_disturbance(signal, dt)
    assert abs(result["freq_hz"] - 20.0) <= freq_resolution
    assert result["amp_px"] == pytest.approx(0.3, rel=0.01)


def test_low_frequency_exclusion_prevents_drift_like_power_from_masquerading_as_the_disturbance():
    """Without exclude_below_hz, a strong low-frequency component would win
    the peak search even though a real (weaker but still detectable)
    higher-frequency tone is present -- exactly what a random-walk drift
    would do to an unguarded peak search."""
    n, dt = 4096, 1e-3
    t = np.arange(n) * dt
    strong_low_freq = 0.4 * np.sin(2 * np.pi * 0.5 * t)  # well below exclude_below_hz
    weak_tone = 0.1 * np.sin(2 * np.pi * 20.0 * t)
    signal = strong_low_freq + weak_tone

    unguarded = detect_disturbance(signal, dt, exclude_below_hz=0.0)
    assert abs(unguarded["freq_hz"] - 0.5) < abs(unguarded["freq_hz"] - 20.0)

    guarded = detect_disturbance(signal, dt, exclude_below_hz=2.0)
    assert abs(guarded["freq_hz"] - 20.0) < 1.0
    assert guarded["amp_px"] == pytest.approx(0.1, rel=0.05)


def test_detects_the_injected_disturbance_on_a_full_noisy_recovered_trajectory():
    """End-to-end: generate the default scenario, render it, recover it
    with the Gaussian fit (never seeing ground truth), then detect the
    disturbance from the RECOVERED trajectory alone -- the actual thing
    the brief asks for, not a shortcut through ground truth."""
    shape = (41, 41)
    half_width = 9
    background_e = 30.0
    sigma_read_e = 5.0
    snr = 50.0

    traj_cfg = TrajectoryConfig(seed=2026, x0=20.3, y0=19.7)
    traj = generate_trajectory(traj_cfg)

    sim = Simulator(
        shape=shape, background_e=background_e, sigma_read_e=sigma_read_e,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2026,
    )
    flux = snr_to_flux(
        snr, sim.sigma, background_e,
        sim.dark_rate_e_per_s * sim.exposure_s, sigma_read_e, sim.gain_e_per_dn,
    )
    frames = render_sequence(sim, traj["x"], traj["y"], flux)
    recovered = recover_trajectory(frames, half_width, sim.sigma, sigma_read_e**2)

    result = detect_disturbance(recovered["x"], traj_cfg.dt_s)
    freq_resolution = 1.0 / (traj_cfg.n_frames * traj_cfg.dt_s)

    assert abs(result["freq_hz"] - traj_cfg.disturb_freq_hz) <= freq_resolution
    assert result["amp_px"] == pytest.approx(traj_cfg.disturb_amp_px, rel=0.05)
