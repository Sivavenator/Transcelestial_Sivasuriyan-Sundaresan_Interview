"""Sensor noise chain: turns a noise-free mean image into a realistic frame.

Each function here adds one physical noise source. They are kept separate
(rather than one big "add_noise" function) because each has a different
physical origin, a different statistical distribution, and a different
place in the signal chain -- and because the brief asks for each to be
individually configurable.

PHOTON (POISSON) NOISE
-----------------------
Light arrives as discrete photons. Even for a perfectly steady, noise-free
source, the *number* of photons a pixel captures in a fixed exposure time is
random -- it follows a Poisson distribution, because photon arrivals are
independent random events at a constant average rate (a Poisson process).

This is not a limitation of the sensor; it is a property of light itself.
No amount of better electronics removes it. It is why "shot noise" is the
fundamental noise floor: even a hypothetically perfect, noiseless sensor
would still show this scatter.

The defining statistical property of a Poisson distribution is that its
variance equals its mean:

    Var[N] = E[N]           for N ~ Poisson(lambda), lambda = E[N]

RELATIVE NOISE: WHY MORE LIGHT MEANS BETTER SNR, NOT JUST MORE BRIGHTNESS
----------------------------------------------------------------------------
The coefficient of variation (CV) is noise expressed as a fraction of the
signal -- standard deviation divided by the mean:

    CV = std / E[N] = sqrt(Var[N]) / E[N] = sqrt(lambda) / lambda = 1 / sqrt(lambda)

Because Var = Mean = lambda, one factor of lambda cancels and the result
collapses to this one clean expression, depending on nothing but lambda
itself.

What that means concretely:

    lambda (avg counts)   absolute noise sqrt(lambda)   relative noise 1/sqrt(lambda)
    1                     1                             100%
    4                     2                              50%
    100                   10                              10%
    10,000                100                              1%

Two things happen at once as lambda grows:
  * absolute noise grows (more photons means more shot noise in raw counts)
  * relative noise shrinks (the signal outpaces the noise)

THE CORE INSIGHT
-------------------
Collecting more signal is a double win: the signal itself increases, AND the
fractional noise decreases. This is why in any photon-counting system, more
light means better SNR, not merely more brightness -- doubling exposure
doesn't just double the signal, it improves SNR by sqrt(2).

Flipped around, SNR itself grows as sqrt(lambda):

    SNR = 1 / CV = sqrt(lambda)

To halve the relative noise (double the SNR) requires 4x as many photons --
a fundamental physical limit set by the statistics of light itself, not an
engineering shortcoming. No algorithm, however clever, can beat this without
extra prior knowledge about the signal. This is why a fainter spot is always
harder to localise precisely, independent of anything else in the system.

GAUSSIAN READ NOISE
----------------------
Reading a pixel's accumulated charge off the sensor and converting it to a
digital number goes through analogue electronics (an amplifier, a
capacitor, an ADC), and each of those stages contributes its own small
random error. By the central limit theorem, summing many small independent
error sources produces something close to Gaussian, so read noise is
modelled as zero-mean Gaussian with a fixed standard deviation:

    N_read ~ Normal(0, sigma_read^2)

The critical difference from photon noise is what it depends on:

    photon noise std  = sqrt(signal)   -- grows with brightness
    read noise std     = sigma_read    -- FIXED, independent of brightness

This is why read noise is the noise floor for a *dark or faint* scene: at
low signal, sqrt(signal) is small and sigma_read dominates; at high signal,
sqrt(signal) eventually outgrows the fixed sigma_read and photon noise takes
over. Total noise combines the two in quadrature (independent sources, so
variances add):

    Var[total] = signal + sigma_read^2
"""

from __future__ import annotations

import numpy as np


def add_photon_noise(mean_image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a Poisson-noisy realisation of a noise-free mean image.

    ``mean_image`` is in electrons (or photons -- Poisson noise is applied
    the same way regardless of unit, since it's a count). Must be
    non-negative; values are used directly as the per-pixel Poisson rate.
    """
    return rng.poisson(mean_image).astype(np.float64)


def add_read_noise(
    image: np.ndarray, sigma_read: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian read noise with a fixed standard deviation.

    Unlike photon noise, ``sigma_read`` does not depend on the image's
    brightness -- it's a property of the sensor's readout electronics, the
    same in a bright pixel and a dark one. Can produce negative values (real
    sensors handle this with a black-level pedestal so raw digital numbers
    stay non-negative; that pedestal is a separate, later concern).
    """
    return image + rng.normal(0.0, sigma_read, size=image.shape)
