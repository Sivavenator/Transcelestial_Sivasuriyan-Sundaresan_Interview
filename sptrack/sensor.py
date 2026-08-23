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

So brighter pixels have *more* absolute noise but *better* relative
precision (SNR = mean / std = mean / sqrt(mean) = sqrt(mean), which grows
with brightness). This one line is the reason a fainter spot is always
harder to localise precisely, independent of anything else in the system.
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
