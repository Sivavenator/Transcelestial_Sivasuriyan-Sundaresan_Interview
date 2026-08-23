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


def test_diagonal_gradient_corner_to_corner_is_sqrt2_times_axis_aligned():
    # Direct empirical check of the docstring's derivation: at angle=45deg,
    # corner-to-corner peak-to-peak should be sqrt(2) times the axis-aligned
    # edge-to-edge peak-to-peak, since t there ranges over [-sqrt(2), sqrt(2)]
    # instead of [-1, 1]. A large odd-sized grid gives corner pixels that sit
    # almost exactly at normalised coordinates (+-1, +-1).
    mean_level = 100.0
    gradient_frac = 0.4
    shape = (101, 101)  # odd size -> corner pixels land exactly on +-1

    diagonal = render_background_gradient(shape, mean_level, gradient_frac, angle_rad=np.pi / 4)
    corner_peak_to_peak = diagonal[-1, -1] - diagonal[0, 0]  # (1,1) minus (-1,-1)

    axis_aligned_peak_to_peak = gradient_frac * mean_level
    assert corner_peak_to_peak == pytest.approx(
        np.sqrt(2) * axis_aligned_peak_to_peak, rel=1e-6
    )
