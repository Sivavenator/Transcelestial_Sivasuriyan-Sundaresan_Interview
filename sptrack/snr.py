"""SNR control: the dial that ties the PSF, scene, and sensor noise together.

WHY THIS MODULE EXISTS SEPARATELY FROM psf.py, scene.py, sensor.py
--------------------------------------------------------------------
Every noise source has been built and tested in isolation so far. "Expose an
SNR control so you can sweep it" (the brief) is a genuinely different kind
of requirement: it's not one more noise source, it's a single KNOB that,
turned to one number, determines how bright the spot needs to be given
everything else already fixed (background, dark current, read noise,
quantization). That's a cross-cutting concern spanning psf.py (how flux
turns into a peak pixel value) and sensor.py/scene.py (the noise budget at
that peak), so it earns its own module rather than being bolted onto either.

WHAT "SNR" MEANS HERE: PEAK-PIXEL SNR, NOT TOTAL-FLUX SNR
---------------------------------------------------------------
There is more than one reasonable SNR definition. This project uses:

    SNR = peak_pixel_signal / sqrt(peak_pixel_noise_variance)

i.e. the brightest single pixel's signal divided by its own noise standard
deviation -- NOT total flux divided by total noise summed over the whole
spot. Why peak-pixel, specifically: a windowed/weighted position estimator
(any of the methods in this project) concentrates most of its weight on the
brightest few pixels near the peak, so the peak pixel's own SNR is what
actually limits how precisely the position can be found -- far more
directly than a total-flux number that doesn't say anything about how that
flux is distributed. This also matches how SNR is conventionally reported
in point-source imaging (astronomy, laser spot tracking) generally.

THE PEAK PIXEL'S SHARE OF THE TOTAL FLUX
---------------------------------------------
When a spot is centred exactly on a pixel (x0, y0 both integers), that
pixel gets the largest possible share of the total flux -- given by
psf.pixel_response_1d evaluated at its own centre, squared (since the 2-D
PSF is separable, the peak pixel's fraction is peak_x_fraction *
peak_y_fraction, and by symmetry those are the same function evaluated the
same way on each axis):

    peak_frac(sigma) = pixel_response_1d([0], 0, sigma)[0] ** 2

This is a genuine, stated approximation: the ACTUAL peak-pixel fraction for
a spot sitting off-centre (at a non-integer sub-pixel position) is always
slightly LOWER than this canonical value, because being centred exactly on
a pixel is precisely the position that puts the most flux in a single
pixel. Using the on-centre value as the reference for "SNR" gives a
consistent, reproducible number to sweep experiments against, understanding
that the REALISED SNR of a specific rendered frame varies slightly with
sub-pixel phase around this nominal value.

SOLVING FOR FLUX, GIVEN A TARGET SNR
-----------------------------------------
The noise budget at the peak pixel, from everything built so far, is:

    Var[peak] = P + C

where P is the peak pixel's own signal (electrons, contributing its own
photon shot noise) and C collects every noise term that does NOT depend on
P -- background shot noise, dark current shot noise, read noise variance,
and quantization variance:

    C = background_e + mean_dark_e + sigma_read_e**2 + gain_e_per_dn**2 / 12

So SNR = P / sqrt(P + C). Squaring and rearranging into a quadratic in P:

    SNR^2 = P^2 / (P + C)
    SNR^2 * (P + C) = P^2
    P^2 - SNR^2 * P - SNR^2 * C = 0

Solving via the quadratic formula and taking the positive root (P must be a
real electron count):

    P = [SNR^2 + sqrt(SNR^4 + 4 * SNR^2 * C)] / 2
      = [SNR^2 + SNR * sqrt(SNR^2 + 4*C)] / 2   (factoring SNR^2 out of the sqrt)

Then the total flux needed is P divided back by the peak pixel's fraction
of it:

    flux = P / peak_frac(sigma)
"""

from __future__ import annotations

import math

import numpy as np

from .psf import pixel_response_1d


def peak_pixel_fraction(sigma: float) -> float:
    """Fraction of total flux in the peak pixel, spot centred on a pixel.

    The largest possible share any single pixel can get for this sigma --
    see the module docstring for why this is the reference value used to
    define SNR, and why a real frame's realised peak fraction is always
    slightly lower than this once the spot is off-centre.
    """
    p = pixel_response_1d(np.array([0]), 0.0, sigma)[0]
    return float(p * p)


def flux_to_snr(
    flux: float,
    sigma: float,
    background_e: float = 0.0,
    mean_dark_e: float = 0.0,
    sigma_read_e: float = 0.0,
    gain_e_per_dn: float = 1.0,
) -> float:
    """Compute the resulting peak-pixel SNR for a given flux and noise budget."""
    peak = flux * peak_pixel_fraction(sigma)
    noise_var = (
        peak + background_e + mean_dark_e + sigma_read_e**2 + gain_e_per_dn**2 / 12.0
    )
    return peak / math.sqrt(noise_var)


def snr_to_flux(
    target_snr: float,
    sigma: float,
    background_e: float = 0.0,
    mean_dark_e: float = 0.0,
    sigma_read_e: float = 0.0,
    gain_e_per_dn: float = 1.0,
) -> float:
    """Solve for the total flux that produces the given peak-pixel SNR.

    Inverts flux_to_snr by solving the quadratic derived in the module
    docstring: P^2 - SNR^2*P - SNR^2*C = 0, taking the positive root.
    """
    c = background_e + mean_dark_e + sigma_read_e**2 + gain_e_per_dn**2 / 12.0
    snr2 = target_snr * target_snr
    peak = (snr2 + target_snr * math.sqrt(snr2 + 4.0 * c)) / 2.0
    return peak / peak_pixel_fraction(sigma)
