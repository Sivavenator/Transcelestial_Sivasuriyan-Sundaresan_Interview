import numpy as np
import pytest

from sptrack.acquisition import acquire_target, find_local_maxima, psf_shape_score
from sptrack.estimators.base import find_brightest_pixel
from sptrack.psf import render_spot


def test_psf_shape_score_is_near_perfect_for_a_true_match():
    shape = (19, 19)
    sigma = 1.75
    window = render_spot(shape, 9.0, 9.0, 1000.0, sigma)
    assert psf_shape_score(window, sigma) == pytest.approx(1.0, abs=0.01)


def test_psf_shape_score_is_lower_for_a_much_wider_profile():
    shape = (19, 19)
    true_sigma = 1.75
    wide_window = render_spot(shape, 9.0, 9.0, 1000.0, sigma=5.0)
    assert psf_shape_score(wide_window, true_sigma) < 0.8


def test_find_local_maxima_respects_threshold_and_nms_radius():
    shape = (41, 41)
    img = render_spot(shape, 10.0, 10.0, 5000.0, 1.75) + render_spot(shape, 30.0, 30.0, 5000.0, 1.75)
    candidates = find_local_maxima(img, min_value=50.0, nms_radius=8)
    positions = {(x, y) for x, y, _ in candidates}
    assert (10, 10) in positions
    assert (30, 30) in positions
    assert len(candidates) == 2


def test_acquire_target_correctly_rejects_a_brighter_but_wider_clutter_source():
    """The core claim: find_brightest_pixel is fooled by a wide, bright
    clutter source; acquire_target's shape-matching is not -- both checked
    on the identical image, not asserted separately."""
    shape = (61, 61)
    true_sigma = 1.75
    true_flux = 3000.0
    clutter_flux = 40000.0  # more total flux, but spread over a much wider profile
    clutter_sigma = 5.0

    img = render_spot(shape, 20.3, 19.7, true_flux, true_sigma)
    img = img + render_spot(shape, 45.0, 40.0, clutter_flux, clutter_sigma)
    img = img + 30.0  # flat background

    # confirm the failure mode actually exists on this image before
    # checking the fix -- otherwise this test wouldn't prove anything.
    iy, ix = find_brightest_pixel(img)
    assert (ix, iy) != (20, 20)  # brightest-pixel is fooled

    result = acquire_target(img, half_width=9, sigma=true_sigma, min_value=100.0)
    assert result is not None
    x, y = result
    assert abs(x - 20) <= 1
    assert abs(y - 20) <= 1


def test_acquire_target_returns_none_when_nothing_clears_the_threshold():
    shape = (41, 41)
    img = np.full(shape, 30.0)  # flat background only, no source
    result = acquire_target(img, half_width=9, sigma=1.75, min_value=1000.0)
    assert result is None
