import numpy as np
import pytest

from sptrack.calibration import (
    apply_radial_distortion,
    correct_radial_distortion,
    estimate_bias_frame,
    estimate_flat_field,
)
from sptrack.simulate import Simulator


def test_estimate_bias_frame_reveals_hot_pixels():
    sim = Simulator(
        shape=(31, 31), background_e=0.0, sigma_read_e=5.0,
        hot_fraction=0.05, hot_rate_e_per_s=5e4, dark_rate_e_per_s=50.0,
        prnu_sigma=0.0, gradient_frac=0.0, seed=1,
    )
    bias = estimate_bias_frame(sim, n_frames=100)
    hot_mean = bias[sim.hot_mask].mean()
    normal_mean = bias[~sim.hot_mask].mean()
    assert hot_mean > normal_mean + 10  # hot pixels stand out clearly


def test_estimate_bias_frame_residual_noise_shrinks_with_more_frames():
    sim1 = Simulator(shape=(21, 21), background_e=0.0, sigma_read_e=5.0, hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2)
    sim2 = Simulator(shape=(21, 21), background_e=0.0, sigma_read_e=5.0, hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2)
    bias_few = estimate_bias_frame(sim1, n_frames=4)
    bias_many = estimate_bias_frame(sim2, n_frames=100)
    assert bias_many.std() < bias_few.std()


def test_estimate_flat_field_recovers_the_true_prnu_map():
    sim = Simulator(shape=(31, 31), background_e=0.0, sigma_read_e=5.0, hot_fraction=0.0, prnu_sigma=0.02, gradient_frac=0.0, seed=3)
    flat = estimate_flat_field(sim, flat_level_e=20000.0, n_frames=13)
    true_map = sim.prnu_map / sim.prnu_map.mean()
    residual = flat - true_map
    assert residual.std() < 0.005  # well below prnu_sigma=0.02 itself


def test_distortion_is_a_no_op_at_the_image_centre():
    shape = (41, 41)
    cx, cy = 20.0, 20.0
    x, y = apply_radial_distortion(cx, cy, shape)
    assert x == pytest.approx(cx, abs=1e-9)
    assert y == pytest.approx(cy, abs=1e-9)


def test_distortion_displacement_matches_the_configured_percentage_at_the_edge():
    shape = (41, 41)
    cx, cy = 20.0, 20.0
    corner_x, corner_y = 0.0, 0.0  # a frame corner: r_norm = 1.0 by construction
    x, y = apply_radial_distortion(corner_x, corner_y, shape, distortion_pct_at_edge=-0.1)
    r_true = np.hypot(corner_x - cx, corner_y - cy)
    r_distorted = np.hypot(x - cx, y - cy)
    frac = (r_distorted - r_true) / r_true
    assert frac == pytest.approx(-0.001, abs=1e-6)


def test_distortion_and_correction_round_trip_exactly():
    shape = (61, 61)
    for x, y in [(5.3, 7.9), (55.1, 50.2), (30.0, 30.0), (0.5, 60.5)]:
        xd, yd = apply_radial_distortion(x, y, shape)
        xc, yc = correct_radial_distortion(xd, yd, shape)
        assert xc == pytest.approx(x, abs=1e-9)
        assert yc == pytest.approx(y, abs=1e-9)
