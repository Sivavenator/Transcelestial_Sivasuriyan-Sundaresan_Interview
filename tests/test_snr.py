import math

import pytest

from sptrack.snr import flux_to_snr, peak_pixel_fraction, snr_to_flux


def test_peak_pixel_fraction_decreases_as_spot_widens():
    # A wider spot spreads the same total flux more thinly, so the single
    # brightest pixel's share must shrink as sigma grows.
    f_narrow = peak_pixel_fraction(1.0)
    f_mid = peak_pixel_fraction(1.75)
    f_wide = peak_pixel_fraction(3.0)
    assert 0.0 < f_wide < f_mid < f_narrow < 1.0


def test_flux_to_snr_reduces_to_sqrt_peak_when_photon_noise_dominates():
    # With every other noise term negligible, Var[peak] ~= peak, so
    # SNR ~= peak / sqrt(peak) = sqrt(peak) -- the same sqrt(lambda) result
    # derived in sensor.py's relative-noise section. gain_e_per_dn is set
    # tiny so the quantization term (gain^2/12) is negligible too.
    sigma = 1.75
    flux = 20000.0
    snr = flux_to_snr(
        flux, sigma, background_e=0.0, mean_dark_e=0.0, sigma_read_e=0.0,
        gain_e_per_dn=1e-3,
    )
    peak = flux * peak_pixel_fraction(sigma)
    assert snr == pytest.approx(math.sqrt(peak), rel=1e-3)


def test_snr_to_flux_round_trips_through_flux_to_snr():
    # The core correctness check: solving the quadratic for flux, then
    # computing the SNR that flux actually produces, must recover the
    # original target -- across a range of SNR values and noise conditions.
    sigma = 1.75
    cases = [
        dict(background_e=0.0, mean_dark_e=0.0, sigma_read_e=0.0, gain_e_per_dn=1.0),
        dict(background_e=30.0, mean_dark_e=0.5, sigma_read_e=5.0, gain_e_per_dn=10.0),
        dict(background_e=200.0, mean_dark_e=5.0, sigma_read_e=8.0, gain_e_per_dn=15.0),
    ]
    for target_snr in [3.0, 10.0, 50.0, 200.0]:
        for noise_kwargs in cases:
            flux = snr_to_flux(target_snr, sigma, **noise_kwargs)
            recovered_snr = flux_to_snr(flux, sigma, **noise_kwargs)
            assert recovered_snr == pytest.approx(target_snr, rel=1e-6)


def test_snr_to_flux_requires_more_flux_when_background_noise_is_higher():
    # Holding the target SNR fixed, a noisier background should demand more
    # signal to punch through it -- flux must increase with C.
    sigma = 1.75
    flux_low_bg = snr_to_flux(20.0, sigma, background_e=10.0)
    flux_high_bg = snr_to_flux(20.0, sigma, background_e=500.0)
    assert flux_high_bg > flux_low_bg
