import numpy as np
import pytest

from sptrack.psf import pixel_response_1d, pixel_response_1d_with_derivative


def test_response_part_matches_pixel_response_1d_exactly():
    idx = np.arange(-5, 6)
    p1, _ = pixel_response_1d_with_derivative(idx, centre=0.3, sigma=1.75)
    p2 = pixel_response_1d(idx, centre=0.3, sigma=1.75)
    assert np.allclose(p1, p2)


def test_analytic_derivative_matches_finite_differences():
    idx = np.arange(-6, 7)
    sigma = 1.75
    h = 1e-6

    for centre in [-1.7, -0.3, 0.0, 0.5, 2.2]:
        _, d_analytic = pixel_response_1d_with_derivative(idx, centre, sigma)
        p_plus = pixel_response_1d(idx, centre + h, sigma)
        p_minus = pixel_response_1d(idx, centre - h, sigma)
        d_numeric = (p_plus - p_minus) / (2 * h)
        assert np.allclose(d_analytic, d_numeric, atol=1e-6), f"mismatch at centre={centre}"


def test_derivative_sign_matches_the_docstring_argument():
    # A pixel to the right of the spot centre should gain brightness as the
    # centre moves right (dP/dc > 0); a pixel to the left should lose it
    # (dP/dc < 0) -- the sign-check argument made in the docstring, checked
    # directly rather than just trusted.
    idx = np.array([10])  # a pixel well to the right of centre=0
    _, d_right = pixel_response_1d_with_derivative(idx, centre=0.0, sigma=1.75)
    assert d_right[0] > 0

    idx = np.array([-10])  # a pixel well to the left
    _, d_left = pixel_response_1d_with_derivative(idx, centre=0.0, sigma=1.75)
    assert d_left[0] < 0
