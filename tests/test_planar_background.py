import numpy as np
import pytest

from sptrack.estimators.base import border_median_background, planar_background


def test_planar_background_matches_a_known_flat_level():
    window = np.full((19, 19), 42.0)
    bg = planar_background(window)
    assert np.allclose(bg, 42.0, atol=1e-6)


def test_planar_background_recovers_a_known_linear_gradient_exactly():
    h, w = 19, 19
    ys, xs = np.mgrid[0:h, 0:w]
    a, b, c = 30.0, 0.7, -0.4
    window = a + b * xs + c * ys
    bg = planar_background(window)
    assert np.allclose(bg, window, atol=1e-6)


def test_planar_background_beats_scalar_median_under_a_real_gradient():
    """The actual claim this function exists to satisfy: a strong linear
    background gradient plus a point source produces a real centroid bias
    when subtracted with a single scalar, and the planar fit removes it."""
    from sptrack.estimators.base import extract_window
    from sptrack.psf import render_spot
    from sptrack.scene import render_background_gradient

    shape = (41, 41)
    sigma = 1.75
    x0, y0 = 20.3, 19.7
    half_width = 9
    gradient_frac = 0.6

    spot = render_spot(shape, x0, y0, 3000.0, sigma)
    bg = render_background_gradient(shape, 30.0, gradient_frac, angle_rad=0.0)
    img = spot + bg
    window, wx0, wy0 = extract_window(img, x0, y0, half_width)
    true_local_x = x0 - wx0

    def _centroid(sub: np.ndarray) -> float:
        sub = np.maximum(sub, 0.0)
        total = sub.sum()
        w_ = sub.shape[1]
        xs = np.arange(w_)
        return float((sub.sum(axis=0) * xs).sum() / total)

    median_err = abs(_centroid(window - border_median_background(window)) - true_local_x)
    planar_err = abs(_centroid(window - planar_background(window)) - true_local_x)

    assert median_err > 0.3  # the real, substantial failure
    assert planar_err < 0.01  # the fix


def test_planar_background_output_shape_matches_input():
    window = np.random.default_rng(1).normal(30, 1, (15, 21))
    bg = planar_background(window)
    assert bg.shape == window.shape
