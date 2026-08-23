import numpy as np
import pytest

from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.psf import render_spot
from sptrack.simulate import Simulator


def test_fit_recovers_true_position_on_a_clean_flat_background_image():
    background_e = 50.0
    x0, y0 = 18.3, 15.0
    image = render_spot((30, 30), x0=x0, y0=y0, flux=2000.0, sigma=1.75) + background_e

    est = gaussian_fit_estimate(image, half_width=8, sigma=1.75, prior=(18.0, 15.0))

    assert est.ok
    assert est.x == pytest.approx(x0, abs=1e-3)
    assert est.y == pytest.approx(y0, abs=1e-3)
    assert est.flux == pytest.approx(2000.0, rel=1e-3)
    assert est.bg == pytest.approx(background_e, abs=0.1)


def test_fit_is_approximately_unbiased_against_the_real_noise_chain():
    sim = Simulator(shape=(25, 25), background_e=30.0, seed=88)
    x0, y0, flux = 12.3, 11.7, 8000.0

    n_trials = 300
    xs, ys = [], []
    for _ in range(n_trials):
        frame = sim.render(x0, y0, flux)
        est = gaussian_fit_estimate(
            frame, half_width=7, sigma=sim.sigma, read_var_e2=sim.sigma_read_e**2,
            prior=(12.3, 11.7),
        )
        if est.ok:
            xs.append(est.x)
            ys.append(est.y)

    assert len(xs) > n_trials * 0.9
    assert np.mean(xs) == pytest.approx(x0, abs=0.05)
    assert np.mean(ys) == pytest.approx(y0, abs=0.05)


def test_fit_is_more_precise_than_the_centroid_at_high_snr():
    # The core claim this project makes about the two methods: the fit is
    # statistically more efficient (weights each pixel by its own noise,
    # where the centroid weights every pixel equally once background is
    # subtracted), so it should have LOWER variance across repeated noisy
    # trials of the identical frames -- a real head-to-head, not an
    # assertion, using the same random frames for both estimators.
    sim = Simulator(shape=(25, 25), background_e=30.0, seed=91)
    x0, y0, flux = 12.3, 11.7, 6000.0

    n_trials = 400
    centroid_xs, fit_xs = [], []
    for _ in range(n_trials):
        frame = sim.render(x0, y0, flux)
        c = centroid_estimate(frame, half_width=7, prior=(12.3, 11.7))
        g = gaussian_fit_estimate(
            frame, half_width=7, sigma=sim.sigma, read_var_e2=sim.sigma_read_e**2,
            prior=(12.3, 11.7),
        )
        if c.ok and g.ok:
            centroid_xs.append(c.x)
            fit_xs.append(g.x)

    centroid_std = np.std(centroid_xs)
    fit_std = np.std(fit_xs)
    assert fit_std < centroid_std


def test_fit_handles_a_window_clamped_at_the_image_edge():
    x0, y0 = 2.3, 2.7
    image = render_spot((10, 10), x0=x0, y0=y0, flux=1000.0, sigma=1.0) + 10.0
    est = gaussian_fit_estimate(image, half_width=6, sigma=1.0, prior=(2.0, 2.0))
    assert est.ok
    assert est.x == pytest.approx(x0, abs=0.1)
    assert est.y == pytest.approx(y0, abs=0.1)


def test_fit_fails_gracefully_when_the_seed_centroid_fails():
    # An all-zero image gives the centroid nothing to work with (total flux
    # <= 0), which should propagate as a clean ok=False rather than the fit
    # crashing trying to use a NaN seed.
    image = np.zeros((15, 15))
    est = gaussian_fit_estimate(image, half_width=6, sigma=1.75, prior=(7.0, 7.0))
    assert not est.ok
