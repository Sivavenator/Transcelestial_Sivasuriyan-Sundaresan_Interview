import numpy as np
import pytest

from sptrack.psf import render_spot
from sptrack.sequence import recover_trajectory, render_sequence
from sptrack.simulate import Simulator
from sptrack.trajectory import TrajectoryConfig, generate_trajectory


def _clean_sim(shape=(41, 41), seed=123):
    return Simulator(shape=shape, hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=seed)


def test_render_sequence_shape():
    sim = _clean_sim()
    x = np.array([20.3, 20.5, 20.7])
    y = np.array([19.7, 19.6, 19.5])
    frames = render_sequence(sim, x, y, flux=5000.0)
    assert frames.shape == (3, 41, 41)


def test_render_sequence_reproducible_with_seed():
    x = np.array([20.3, 20.5, 20.7, 20.9])
    y = np.array([19.7, 19.6, 19.5, 19.4])
    frames1 = render_sequence(_clean_sim(seed=77), x, y, flux=5000.0)
    frames2 = render_sequence(_clean_sim(seed=77), x, y, flux=5000.0)
    assert np.array_equal(frames1, frames2)


def test_recover_trajectory_survives_a_degenerate_frame_without_losing_the_prior():
    """A middle frame that fails (all zero, no spot) must not corrupt the
    running prior -- the frame AFTER it should still recover cleanly,
    proving dead-reckoning fell back to the last known-good position
    rather than the failed frame's NaN output."""
    shape = (41, 41)
    sigma = 1.75
    flux = 1.0e6  # very high flux: near-noiseless, isolates the gating logic itself
    frame_a = render_spot(shape, 20.3, 19.7, flux, sigma)
    frame_zero = np.zeros(shape)
    frames = np.stack([frame_a, frame_zero, frame_a])

    result = recover_trajectory(frames, half_width=9, sigma=sigma)

    assert result["ok"][0]
    assert not result["ok"][1]
    assert result["ok"][2]
    assert result["x"][2] == pytest.approx(20.3, abs=0.01)
    assert result["y"][2] == pytest.approx(19.7, abs=0.01)


def test_recover_trajectory_tracks_a_short_moving_sequence_accurately():
    cfg = TrajectoryConfig(n_frames=80, seed=321, x0=20.3, y0=19.7, disturb_amp_px=0.3, disturb_freq_hz=20.0)
    traj = generate_trajectory(cfg)

    sim = _clean_sim(seed=321)
    flux = 20000.0  # comfortably high SNR, so recovery error should be small
    frames = render_sequence(sim, traj["x"], traj["y"], flux)

    result = recover_trajectory(frames, half_width=9, sigma=sim.sigma, read_var_e2=sim.sigma_read_e**2)

    assert np.all(result["ok"])
    err_x = result["x"] - traj["x"]
    err_y = result["y"] - traj["y"]
    assert np.std(err_x) < 0.05
    assert np.std(err_y) < 0.05
    assert abs(np.mean(err_x)) < 0.02
    assert abs(np.mean(err_y)) < 0.02
