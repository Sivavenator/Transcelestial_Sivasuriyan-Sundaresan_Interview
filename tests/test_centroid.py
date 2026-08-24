import numpy as np
import pytest

from sptrack.estimators.base import border_median_background, extract_window, find_brightest_pixel
from sptrack.estimators.centroid import centroid_estimate
from sptrack.psf import render_spot
from sptrack.simulate import Simulator


def test_centroid_recovers_true_position_on_a_clean_flat_background_image():
    background_e = 50.0
    x0, y0 = 18.3, 15.0
    image = render_spot((30, 30), x0=x0, y0=y0, flux=2000.0, sigma=1.75) + background_e

    # prior is close to the TRUE spot position, not an arbitrary window
    # centre, so the window ends up nearly symmetric around the spot: with
    # half_width=8 (~4.6 sigma), both edge margins land comfortably beyond
    # where the PSF's tails matter. (An earlier version of this test used a
    # prior offset from the true position -- (15, 15) with the spot at
    # 18.3 -- which left only a 2.7-sigma margin on the near edge and
    # produced a real, measurable truncation bias of ~0.009 px. Not a bug in
    # the estimator: a badly chosen test setup. Fixed by placing the window
    # where it belongs -- verified numerically before trusting this tolerance.)
    est = centroid_estimate(image, half_width=8, prior=(18.0, 15.0))

    assert est.ok
    assert est.x == pytest.approx(x0, abs=1e-3)
    assert est.y == pytest.approx(y0, abs=1e-3)
    # Not exact, for the same reason established in
    # test_clip_negative_and_no_clip_nearly_agree_...: the PSF's Gaussian
    # tail still reaches the border pixels even at ~4.6 sigma, nudging the
    # median background estimate very slightly above the true pedestal.
    assert est.bg == pytest.approx(background_e, abs=0.01)


def test_background_subtraction_removes_a_real_bias_not_a_cosmetic_one():
    # The claim in centroid.py's docstring, demonstrated directly: without
    # background subtraction, a window's every pixel contributes roughly
    # equal weight from the flat background, pulling the centroid toward
    # the WINDOW's own geometric centre rather than the true spot position.
    #
    # The offset between window centre and true spot (3.3 px) is deliberate
    # -- that's the whole point of this test -- but it means half_width must
    # be large enough that the NEAR edge margin (half_width - offset) still
    # stays comfortably beyond the PSF's tails, or truncation bias would
    # contaminate the "with background subtraction" result too. Verified:
    # half_width=15 gives a near-edge margin of 15 - 3.3 = 11.7 px = 6.7
    # sigma, safely negligible (see the sibling test's comment for what
    # happens when this margin is too tight).
    background_e = 50.0
    x0, y0 = 23.3, 20.0
    window_centre = (20.0, 20.0)  # deliberately NOT the spot's true position
    image = render_spot((40, 40), x0=x0, y0=y0, flux=500.0, sigma=1.75) + background_e

    with_bg_sub = centroid_estimate(image, half_width=15, prior=window_centre)

    # Compute the naive (no background subtraction) centroid directly, on
    # the identical window, for a fair side-by-side comparison.
    window, wx0, wy0 = extract_window(image, *window_centre, half_width=15)
    h, w = window.shape
    xs = np.arange(w, dtype=np.float64)
    naive_x = (window.sum(axis=0) * xs).sum() / window.sum() + wx0

    assert with_bg_sub.ok
    error_with_bg_sub = abs(with_bg_sub.x - x0)
    error_naive = abs(naive_x - x0)
    assert error_with_bg_sub < 0.01
    assert error_naive > 1.0  # pulled meaningfully toward the window centre (20.0)
    assert error_with_bg_sub < error_naive


def test_clip_negative_and_no_clip_nearly_agree_on_a_sensor_noise_free_image():
    # Checked directly rather than assumed: even with NO sensor noise, the
    # background-subtracted window is not perfectly non-negative. The PSF's
    # Gaussian tail still reaches the border at half_width=7 (~4 sigma) --
    # tiny (border values ~20.0 to ~20.07, against a true pedestal of 20.0),
    # but enough to nudge the median background estimate to ~20.012 instead
    # of exactly 20.0, which pushes ~50 of 225 pixels fractionally negative
    # (down to about -0.012) with no sensor noise involved at all. So
    # clip_negative and no-clip are close, not identical, and the tolerance
    # here reflects that real, understood effect rather than an idealised
    # "no noise means no negatives" assumption.
    image = render_spot((25, 25), x0=12.4, y0=11.6, flux=1500.0, sigma=1.75) + 20.0
    est_clipped = centroid_estimate(image, half_width=7, prior=(12, 12), clip_negative=True)
    est_unclipped = centroid_estimate(image, half_width=7, prior=(12, 12), clip_negative=False)
    assert est_clipped.x == pytest.approx(est_unclipped.x, abs=1e-3)
    assert est_clipped.y == pytest.approx(est_unclipped.y, abs=1e-3)


def test_centroid_handles_a_window_clamped_at_the_image_edge():
    # The spot sits close enough to a corner that the requested half_width
    # would extend past the image bounds -- extract_window must clamp
    # rather than error, and the position should still come out sane.
    x0, y0 = 2.3, 2.7
    image = render_spot((10, 10), x0=x0, y0=y0, flux=1000.0, sigma=1.0) + 10.0
    est = centroid_estimate(image, half_width=6, prior=(2.0, 2.0))
    assert est.ok
    assert est.x == pytest.approx(x0, abs=0.1)
    assert est.y == pytest.approx(y0, abs=0.1)


def test_centroid_is_approximately_unbiased_against_the_real_noise_chain():
    # A first, informal check against the full Simulator (Monte Carlo
    # characterisation proper comes later, §2c) -- at a moderate-to-high
    # SNR, averaging many noisy estimates should land close to the true
    # position, confirming the estimator isn't obviously broken against
    # real sensor noise, not just clean synthetic images.
    sim = Simulator(shape=(25, 25), background_e=30.0, seed=77)
    x0, y0, flux = 12.3, 11.7, 8000.0  # high flux -> high SNR, small expected scatter

    n_trials = 300
    xs, ys = [], []
    for _ in range(n_trials):
        frame = sim.render(x0, y0, flux)
        est = centroid_estimate(frame, half_width=7, prior=(12, 12))
        if est.ok:
            xs.append(est.x)
            ys.append(est.y)

    assert len(xs) > n_trials * 0.95  # estimator should succeed almost every trial
    assert np.mean(xs) == pytest.approx(x0, abs=0.05)
    assert np.mean(ys) == pytest.approx(y0, abs=0.05)


def test_find_brightest_pixel_locates_an_isolated_peak():
    image = np.zeros((10, 10))
    image[3, 7] = 100.0
    row, col = find_brightest_pixel(image)
    assert (row, col) == (3, 7)


def test_border_median_background_ignores_a_bright_centre():
    window = np.full((11, 11), 20.0)
    window[4:7, 4:7] = 5000.0  # a bright "spot" in the middle
    bg = border_median_background(window, border_width=2)
    assert bg == pytest.approx(20.0)


def test_clip_negative_introduces_phase_dependent_bias_that_no_clip_does_not():
    """Regression guard for the pixel-locking result (exp06a).

    Clipping negative background-subtracted pixels rectifies downward
    noise excursions. When the spot sits off-centre inside its
    integer-placed window, that residue is spatially lopsided and pulls
    the weighted average toward the window centre, producing a bias that
    varies with sub-pixel phase. Without clipping the same frames show no
    such structure. Phase-dependent bias matters more than a constant
    offset because it does not average away as the spot moves.
    """
    from sptrack.simulate import Simulator
    from sptrack.snr import snr_to_flux

    shape, half_width = (21, 21), 9
    background_e, sigma_read_e = 30.0, 5.0
    sim = Simulator(
        shape=shape, background_e=background_e, sigma_read_e=sigma_read_e,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=4242,
    )
    flux = snr_to_flux(
        100.0, sim.sigma, background_e,
        sim.dark_rate_e_per_s * sim.exposure_s, sigma_read_e, sim.gain_e_per_dn,
    )

    phases = [0.0, 0.2, 0.4, 0.5, 0.7, 0.9]
    clip_bias, noclip_bias = [], []
    for p in phases:
        x0, y0 = 10.0 + p, 10.0
        a, b = [], []
        for _ in range(200):
            frame = sim.dn_to_electrons(sim.render(x0, y0, flux))
            c = centroid_estimate(frame, half_width, prior=(x0, y0), clip_negative=True)
            n = centroid_estimate(frame, half_width, prior=(x0, y0), clip_negative=False)
            if c.ok:
                a.append(c.x - x0)
            if n.ok:
                b.append(n.x - x0)
        clip_bias.append(np.mean(a))
        noclip_bias.append(np.mean(b))

    clip_pp = max(clip_bias) - min(clip_bias)
    noclip_pp = max(noclip_bias) - min(noclip_bias)

    # clipping produces real phase structure, roughly 4 millipixels
    assert clip_pp > 2e-3
    # switching it off removes most of it on the identical noise realisations
    assert noclip_pp < clip_pp / 2
