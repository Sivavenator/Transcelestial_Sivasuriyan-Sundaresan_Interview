import numpy as np
import pytest

from sptrack.scene import render_background_gradient


def test_zero_gradient_frac_gives_a_flat_background():
    bg = render_background_gradient((10, 10), mean_level=100.0, gradient_frac=0.0)
    assert np.allclose(bg, 100.0)


def test_background_mean_equals_mean_level_regardless_of_gradient():
    # x_norm and y_norm are symmetric about 0 by construction (np.linspace(-1, 1, n)),
    # so the gradient term should average out to exactly 0 over the whole frame,
    # leaving the frame's mean equal to mean_level -- true for any angle/strength.
    for angle in [0.0, np.pi / 4, np.pi / 2, 1.3]:
        bg = render_background_gradient(
            (20, 30), mean_level=100.0, gradient_frac=0.6, angle_rad=angle
        )
        assert bg.mean() == pytest.approx(100.0, abs=1e-9)


def test_axis_aligned_gradient_peak_to_peak_matches_gradient_frac():
    mean_level = 100.0
    gradient_frac = 0.4

    horizontal = render_background_gradient(
        (10, 10), mean_level, gradient_frac, angle_rad=0.0
    )
    # angle=0: pure left-right tilt. Peak-to-peak, edge to edge, should be
    # exactly gradient_frac * mean_level, per the docstring's stated convention.
    peak_to_peak = horizontal[:, -1].mean() - horizontal[:, 0].mean()
    assert peak_to_peak == pytest.approx(gradient_frac * mean_level, rel=1e-6)

    vertical = render_background_gradient(
        (10, 10), mean_level, gradient_frac, angle_rad=np.pi / 2
    )
    peak_to_peak_v = vertical[-1, :].mean() - vertical[0, :].mean()
    assert peak_to_peak_v == pytest.approx(gradient_frac * mean_level, rel=1e-6)


def test_gradient_direction_is_correct():
    # angle=0 (left-right tilt): right edge should be brighter than left edge
    # for a positive gradient_frac.
    bg = render_background_gradient((10, 10), 100.0, gradient_frac=0.5, angle_rad=0.0)
    assert bg[:, -1].mean() > bg[:, 0].mean()

    # angle=pi/2 (bottom-top tilt): top edge (last row) should be brighter
    # than bottom edge (first row).
    bg_v = render_background_gradient((10, 10), 100.0, gradient_frac=0.5, angle_rad=np.pi / 2)
    assert bg_v[-1, :].mean() > bg_v[0, :].mean()
