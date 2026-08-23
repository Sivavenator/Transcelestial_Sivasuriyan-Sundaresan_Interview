import numpy as np
import pytest

from sptrack.crlb import position_crlb
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.simulate import Simulator


def test_crlb_decreases_as_flux_increases():
    # More signal means more information, so the bound must tighten -- a
    # LOWER bound on the achievable variance corresponds to a smaller
    # number here (this function returns the bound's std, so "tighter" is
    # numerically smaller).
    shape = (15, 15)
    stds = [
        position_crlb(shape, x0=7.3, y0=7.7, flux=f, bg=30.0, sigma=1.75, read_var_e2=25.0)[0]
        for f in [500.0, 2000.0, 8000.0, 32000.0]
    ]
    assert stds == sorted(stds, reverse=True)


def test_crlb_is_symmetric_for_a_symmetric_setup():
    # Centred in a square window with an isotropic PSF, the x and y bounds
    # should come out equal by symmetry -- a simple, checkable property.
    std_x, std_y = position_crlb(
        (15, 15), x0=7.0, y0=7.0, flux=5000.0, bg=30.0, sigma=1.75, read_var_e2=25.0
    )
    assert std_x == pytest.approx(std_y, rel=1e-9)


def test_crlb_loosens_with_more_read_noise():
    shape = (15, 15)
    std_low_read = position_crlb(shape, 7.3, 7.7, 5000.0, 30.0, 1.75, read_var_e2=1.0)[0]
    std_high_read = position_crlb(shape, 7.3, 7.7, 5000.0, 30.0, 1.75, read_var_e2=100.0)[0]
    assert std_high_read > std_low_read


def test_crlb_approaches_the_classical_photon_limited_formula():
    # A well-known closed-form approximation exists in the simplest limit:
    # continuous (non-pixelated) sampling, no background, no read noise --
    # sigma_x ~= sigma_PSF / sqrt(N) (e.g. Thompson et al. 2002). At
    # sigma=1.75 px this project's PIXEL-INTEGRATED bound sits ~1.35% above
    # that classical value -- checked directly, not a bug: sweeping sigma
    # confirms the gap shrinks monotonically as sigma grows relative to the
    # fixed 1-pixel pitch (15.1% at sigma=0.5, 1.35% at sigma=1.75, 0.04% at
    # sigma=10), exactly the signature of a genuine pixelation effect (finer
    # sampling relative to the PSF converges to the continuous-sampling
    # limit), matching known results in the localisation-microscopy
    # literature (e.g. Mortensen et al. 2010's pixel-size correction) rather
    # than an implementation error. Tolerance set at 2%, comfortably above
    # the verified ~1.35% gap at this project's actual sigma.
    sigma = 1.75
    flux = 1e7  # very bright, so background/read noise contribute negligibly
    std_x, _ = position_crlb((25, 25), 12.0, 12.0, flux, bg=1e-6, sigma=sigma, read_var_e2=1e-6)
    classical = sigma / np.sqrt(flux)
    assert std_x == pytest.approx(classical, rel=0.02)


def test_gaussian_fit_approaches_the_crlb_at_high_snr():
    # The real validation this module exists for: does the MLE fit actually
    # ATTAIN the bound, the way asymptotic efficiency theory predicts it
    # should at high SNR? Measured directly via Monte Carlo, not assumed.
    sim = Simulator(shape=(21, 21), background_e=30.0, seed=95)
    x0, y0, flux = 10.3, 9.7, 15000.0  # bright -> high SNR

    predicted_std, _ = position_crlb(
        sim.shape, x0, y0, flux, sim.background_e, sim.sigma, sim.sigma_read_e**2
    )

    n_trials = 400
    xs = []
    for _ in range(n_trials):
        frame = sim.render(x0, y0, flux)
        est = gaussian_fit_estimate(
            frame, half_width=10, sigma=sim.sigma, read_var_e2=sim.sigma_read_e**2,
            prior=(x0, y0),
        )
        if est.ok:
            xs.append(est.x)

    empirical_std = np.std(xs)
    efficiency = predicted_std / empirical_std  # ~1.0 means the fit attains the bound

    assert len(xs) > n_trials * 0.9
    # "Attains" in a Monte Carlo sense, not bit-exact: efficiency within
    # +-25% of 1.0 is a meaningful confirmation given 400 trials' sampling
    # noise on the empirical std itself, not a claim of exact equality.
    assert 0.75 < efficiency < 1.25
