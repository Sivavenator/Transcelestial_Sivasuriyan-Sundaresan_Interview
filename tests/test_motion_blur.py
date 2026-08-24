import numpy as np
import pytest

from sptrack.motion_blur import render_motion_blurred_spot
from sptrack.psf import render_spot


def test_zero_blur_matches_render_spot_exactly():
    shape, sigma, x0, y0, flux = (41, 41), 1.75, 20.3, 19.7, 5000.0
    blurred = render_motion_blurred_spot(shape, x0, y0, flux, sigma, blur_px=0.0)
    static = render_spot(shape, x0, y0, flux, sigma)
    assert np.array_equal(blurred, static)


def test_flux_is_conserved_across_blur_magnitudes():
    shape, sigma, x0, y0, flux = (41, 41), 1.75, 20.0, 20.0, 100000.0
    for blur_px in [0.5, 1.0, 3.0, 5.0]:
        img = render_motion_blurred_spot(shape, x0, y0, flux, sigma, blur_px)
        assert img.sum() == pytest.approx(flux, rel=1e-6)


def test_centroid_stays_unbiased_regardless_of_blur():
    shape, sigma, x0, y0, flux = (41, 41), 1.75, 20.0, 20.0, 100000.0
    xs = np.arange(shape[1])
    for blur_px in [1.0, 3.0, 5.0]:
        img = render_motion_blurred_spot(shape, x0, y0, flux, sigma, blur_px)
        marg_x = img.sum(axis=0)
        mean_x = (marg_x * xs).sum() / marg_x.sum()
        assert mean_x == pytest.approx(x0, abs=1e-6)


def test_variance_along_motion_axis_matches_the_box_convolution_formula():
    shape, sigma, x0, y0, flux = (41, 41), 1.75, 20.0, 20.0, 100000.0
    xs = np.arange(shape[1])
    zero_blur_var = None
    for blur_px in [0.0, 1.0, 3.0, 5.0]:
        img = render_motion_blurred_spot(shape, x0, y0, flux, sigma, blur_px, n_substeps=81)
        marg_x = img.sum(axis=0)
        mean_x = (marg_x * xs).sum() / marg_x.sum()
        var_x = (marg_x * (xs - mean_x) ** 2).sum() / marg_x.sum()
        if blur_px == 0.0:
            zero_blur_var = var_x
        else:
            expected_increment = blur_px**2 / 12
            measured_increment = var_x - zero_blur_var
            assert measured_increment == pytest.approx(expected_increment, rel=0.05)


def test_blur_does_not_spread_the_perpendicular_axis():
    shape, sigma, x0, y0, flux = (41, 41), 1.75, 20.0, 20.0, 100000.0
    ys = np.arange(shape[0])
    img_no_blur = render_motion_blurred_spot(shape, x0, y0, flux, sigma, 0.0)
    img_blurred = render_motion_blurred_spot(shape, x0, y0, flux, sigma, 5.0, angle_rad=0.0)

    def var_y(img):
        marg_y = img.sum(axis=1)
        mean_y = (marg_y * ys).sum() / marg_y.sum()
        return (marg_y * (ys - mean_y) ** 2).sum() / marg_y.sum()

    assert var_y(img_blurred) == pytest.approx(var_y(img_no_blur), rel=0.02)


def test_angle_rotates_the_blur_direction():
    shape, sigma, x0, y0, flux = (41, 41), 1.75, 20.0, 20.0, 100000.0
    xs = np.arange(shape[1])
    ys = np.arange(shape[0])

    img_horizontal = render_motion_blurred_spot(shape, x0, y0, flux, sigma, 5.0, angle_rad=0.0)
    img_vertical = render_motion_blurred_spot(shape, x0, y0, flux, sigma, 5.0, angle_rad=np.pi / 2)

    def var_x(img):
        marg_x = img.sum(axis=0)
        mean_x = (marg_x * xs).sum() / marg_x.sum()
        return (marg_x * (xs - mean_x) ** 2).sum() / marg_x.sum()

    def var_y(img):
        marg_y = img.sum(axis=1)
        mean_y = (marg_y * ys).sum() / marg_y.sum()
        return (marg_y * (ys - mean_y) ** 2).sum() / marg_y.sum()

    # horizontal blur spreads x, not y; vertical blur spreads y, not x
    assert var_x(img_horizontal) > var_y(img_horizontal) * 1.5
    assert var_y(img_vertical) > var_x(img_vertical) * 1.5
