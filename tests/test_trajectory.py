import numpy as np
import pytest

from sptrack.trajectory import TrajectoryConfig, generate_trajectory


def test_output_lengths_match_n_frames():
    cfg = TrajectoryConfig(n_frames=500, seed=1)
    out = generate_trajectory(cfg)
    for key in ("t", "x", "y", "drift_x", "drift_y", "jitter_x", "jitter_y", "disturb_x", "disturb_y"):
        assert len(out[key]) == 500


def test_reproducible_with_seed():
    cfg = TrajectoryConfig(n_frames=200, seed=42)
    out1 = generate_trajectory(cfg)
    out2 = generate_trajectory(cfg)
    assert np.array_equal(out1["x"], out2["x"])
    assert np.array_equal(out1["y"], out2["y"])


def test_different_seeds_differ():
    out1 = generate_trajectory(TrajectoryConfig(n_frames=200, seed=1))
    out2 = generate_trajectory(TrajectoryConfig(n_frames=200, seed=2))
    assert not np.array_equal(out1["x"], out2["x"])


def test_x0_y0_offset_shifts_the_whole_trajectory():
    cfg_a = TrajectoryConfig(n_frames=300, seed=7, x0=0.0, y0=0.0)
    cfg_b = TrajectoryConfig(n_frames=300, seed=7, x0=10.0, y0=-5.0)
    out_a = generate_trajectory(cfg_a)
    out_b = generate_trajectory(cfg_b)
    assert np.allclose(out_b["x"] - out_a["x"], 10.0)
    assert np.allclose(out_b["y"] - out_a["y"], -5.0)


def test_disturbance_confined_to_requested_axis():
    cfg_x = TrajectoryConfig(n_frames=300, seed=3, disturb_axis="x", disturb_amp_px=0.5)
    out_x = generate_trajectory(cfg_x)
    assert np.any(out_x["disturb_x"] != 0.0)
    assert np.all(out_x["disturb_y"] == 0.0)

    cfg_y = TrajectoryConfig(n_frames=300, seed=3, disturb_axis="y", disturb_amp_px=0.5)
    out_y = generate_trajectory(cfg_y)
    assert np.all(out_y["disturb_x"] == 0.0)
    assert np.any(out_y["disturb_y"] != 0.0)

    cfg_both = TrajectoryConfig(n_frames=300, seed=3, disturb_axis="both", disturb_amp_px=0.5)
    out_both = generate_trajectory(cfg_both)
    assert np.any(out_both["disturb_x"] != 0.0)
    assert np.any(out_both["disturb_y"] != 0.0)


def test_disturbance_frequency_and_amplitude_recoverable_from_clean_component():
    """A clean sanity check on the disturbance generator itself, isolated
    from drift/jitter/estimation noise entirely: FFT the noiseless
    disturb_x component directly and confirm the injected frequency and
    amplitude come back out, to the FFT's own bin resolution."""
    cfg = TrajectoryConfig(
        n_frames=4096, dt_s=1e-3, seed=5, disturb_freq_hz=20.0, disturb_amp_px=0.3,
        drift_step_std_px=0.0, jitter_std_px=0.0,
    )
    out = generate_trajectory(cfg)
    n = cfg.n_frames
    spec = np.fft.rfft(out["disturb_x"])
    freqs = np.fft.rfftfreq(n, d=cfg.dt_s)
    peak_idx = np.argmax(np.abs(spec))
    freq_resolution = 1.0 / (n * cfg.dt_s)

    assert abs(freqs[peak_idx] - cfg.disturb_freq_hz) < freq_resolution
    # a real sinusoid's one-sided FFT peak magnitude is n*amplitude/2
    recovered_amp = 2.0 * np.abs(spec[peak_idx]) / n
    assert recovered_amp == pytest.approx(cfg.disturb_amp_px, rel=0.02)


def test_drift_spectrum_is_concentrated_at_low_frequency():
    """A random walk's PSD falls off as 1/f^2 -- verify most of its power
    sits in the lowest decade of resolvable frequencies, not spread
    uniformly like white noise would be."""
    cfg = TrajectoryConfig(n_frames=4096, dt_s=1e-3, seed=11, drift_step_std_px=0.01)
    out = generate_trajectory(cfg)
    n = cfg.n_frames
    spec = np.fft.rfft(out["drift_x"] - out["drift_x"].mean())
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=cfg.dt_s)

    low_band = power[(freqs > 0) & (freqs < 5.0)].sum()
    high_band = power[freqs >= 5.0].sum()
    assert low_band > 10 * high_band


def test_jitter_spectrum_is_approximately_flat_and_matches_requested_std():
    cfg = TrajectoryConfig(n_frames=4096, dt_s=1e-3, seed=13, jitter_std_px=0.15)
    out = generate_trajectory(cfg)
    n = cfg.n_frames
    spec = np.fft.rfft(out["jitter_x"])
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=cfg.dt_s)

    # white noise: low-frequency and high-frequency band power should be
    # comparable (same order of magnitude), unlike the drift case above.
    low_band = power[(freqs > 0) & (freqs < 5.0)].sum()
    high_band = power[(freqs >= 5.0) & (freqs < 10.0)].sum()
    ratio = low_band / high_band
    assert 0.2 < ratio < 5.0

    assert np.std(out["jitter_x"]) == pytest.approx(cfg.jitter_std_px, rel=0.1)


def test_total_excursion_stays_modest_over_the_default_capture_window():
    """Numeric check on the docstring's own claim: with the default
    parameters, the combined trajectory should stay within a few pixels of
    its start over the whole 4096-frame window -- not wander off
    unboundedly, which would make a fixed-size simulation canvas
    impossible to choose sensibly."""
    cfg = TrajectoryConfig(seed=99)
    out = generate_trajectory(cfg)
    max_excursion = max(np.abs(out["x"]).max(), np.abs(out["y"]).max())
    assert max_excursion < 5.0
