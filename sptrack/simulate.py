"""Ties every piece of the simulator (psf, scene, sensor, snr) into one
frame generator -- what an estimator actually receives.

WHY THIS MODULE EXISTS
---------------------------
Every noise source in this project has been built and tested in isolation:
photon noise on its own, read noise on its own, PRNU on its own, and so on.
That was deliberate -- each has its own statistical signature, and testing
them separately means a bug in one is caught without ten other effects
confusing the picture. But an estimator never sees any of those in
isolation; it sees ONE frame with all of them baked in together, in the
correct physical order. This module is that assembly step, built once so
every future experiment (estimators, characterization, dynamic tracking)
calls one thing instead of re-chaining eight functions by hand.

THE CORRECT ORDER, AND WHY IT MATTERS
------------------------------------------
1. Render the spot's mean image (psf.render_spot) -- flux distributed
   across pixels by the PSF shape.
2. Add the background gradient (scene.render_background_gradient) -- both
   the spot and the background are real light, so they combine BEFORE any
   noise is applied.
3. Apply PRNU (sensor.apply_prnu) to that combined photo-signal -- PRNU is
   specifically a property of the photon-to-electron conversion pathway,
   so it must be applied to light that actually passed through it. This is
   why it happens here, not at the very end: dark current and hot-pixel
   electrons (added next) never pass through this pathway, so PRNU must
   NOT be applied to them (see sensor.py's PRNU section for the physical
   reasoning).
4. Draw photon noise (sensor.add_photon_noise) on the PRNU-adjusted
   photo-signal.
5. Add dark current (sensor.add_dark_current) -- its own independent
   Poisson draw, added AFTER photon noise on the signal, mathematically
   equivalent to combining means first by Poisson additivity (see
   sensor.py), but kept as a distinct step for the same reasons dark
   current is a separate function at all.
6. Add hot pixels (sensor.add_hot_pixels) at the sensor's fixed defect
   locations.
7. Add read noise (sensor.add_read_noise) -- the sensor's electronics
   noise, added after all the electron-domain noise above.
8. Quantize to DN (sensor.quantize_to_dn) -- the last step, and the only
   one that changes units, with the black-level pedestal that keeps
   negative read-noise excursions from biasing the mean.

WHAT IS FIXED PER SIMULATED UNIT VS. FRESH PER FRAME
----------------------------------------------------------
Three things are properties of one simulated sensor/optics unit and must be
generated ONCE, not redrawn every frame: the hot-pixel defect map, the PRNU
gain map, and (if enabled) the true optical sigma. This class generates
them in ``__init__`` and reuses them across every call to ``render``,
exactly the pattern established when each was built individually.

DEFAULT VALUES, AND WHY EACH ONE WAS CHOSEN
--------------------------------------------------
Every default below is a deliberate choice, not a placeholder. Most reuse a
value already derived and justified earlier in this project, rather than
introducing a fresh unjustified number:

  nominal_diameter_1e2_px = 7.0
      Directly the brief's own spec: "~7 pixels in diameter (1/e^2)".

  sigma_tolerance_frac = 0.0
      Off by default (exact nominal sigma), so a bare Simulator gives a
      deterministic, repeatable PSF unless unit-to-unit variation is
      explicitly requested -- this one parameter has no "typical nonzero"
      value the way a noise level does, so off is the honest default, not
      an arbitrary nonzero guess.

  background_e = 30.0
      Deliberately close to (slightly above) the read-noise/shot-noise
      crossover derived in sensor.py's read-noise section --
      sigma_read^2 = 25 at sigma_read=5.0 -- so the default sits in the
      regime where BOTH read noise and background shot noise matter,
      rather than an extreme where one trivially dominates. Also the same
      value already used consistently across this project's own
      sanity-check visualisations.

  gradient_frac = 0.0, gradient_angle_rad = 0.0
      Flat background by default. NOTE, stated honestly: this is an
      exception to "nothing defaults to off" above -- but a deliberate one,
      not an oversight. Unlike a noise level (which has a physically
      typical nonzero magnitude to default to), a gradient's strength AND
      direction are scene-specific with no single representative value;
      picking any nonzero default would be an arbitrary, unjustified claim
      about a specific deployment scenario. Flat is the only choice that
      isn't secretly asserting something about the world.

  dark_rate_e_per_s = 50.0
      A modest, realistic room-temperature rate for a decent sensor,
      chosen so that at the default exposure_s=1e-3 it gives mean_dark =
      0.05 e- -- genuinely negligible, exactly matching the claim already
      made in sensor.py's docstring ("dark current is often negligible at
      these exposure times... and room temperature"). Non-zero by default
      so it's still exercised (and becomes non-negligible) once exposure
      or temperature scale up in a later experiment.

  exposure_s = 1e-3
      1 ms, directly the brief's ~1 kHz frame rate (1 / 1000 Hz).

  hot_fraction = 1e-4
      Sits in the realistic real-sensor defect-rate range already
      established (1e-5 to 1e-3, ASSUMPTIONS.md) -- unlike the inflated
      0.05 used only in sensor.py's own unit test specifically to get
      enough hot pixels on a small test grid to check statistics against.
      This default is for actual experiments, where realism is the
      priority, not a statistical test needing a large sample.

  hot_rate_e_per_s = 5e4
      Exactly 1000x the default dark_rate_e_per_s (50) -- a clean,
      deliberate multiplier matching sensor.py's own qualitative
      description of a hot pixel: a dark-current rate "orders of
      magnitude above the rest of the sensor."

  sigma_read_e = 5.0
      The same value already derived and placed on the realistic-sensor
      table in sensor.py's read-noise section: solid consumer-CMOS
      mid-range (cheap phone 10-20 e-, consumer CMOS 3-8 e-, scientific
      sCMOS 1-2 e-, EMCCD <1 e-).

  prnu_sigma = 0.02
      2%, the same "realistic for a consumer sensor" value already used in
      sensor.py's own PRNU statistics test -- as opposed to the inflated
      5% used only in that module's position-dependent-bias DEMONSTRATION,
      which needed the effect to be clearly visible on a small test window,
      not to be realistic.

  gain_e_per_dn = 10.0, bit_depth = 12
      Chosen together, consistently with the worked example already used
      in sensor.py's quantization derivation (gain ~= 9.77 e-/DN from a
      40,000 e- full well over a 12-bit/4096-level ADC). 10.0 x 4096 =
      40,960 e- implied full well -- matching that same worked example,
      not a fresh, disconnected number.

  black_level_dn = 100.0
      In DN, sigma_read_e/gain_e_per_dn = 5.0/10.0 = 0.5 DN -- so a 100 DN
      pedestal is a 200-sigma safety margin against the clipping bias
      demonstrated in sensor.py's quantization section, far more than the
      read-noise excursions alone would need. The extra margin matters
      because dark current and background shot noise ALSO contribute to
      the total noise budget the pedestal has to clear, not read noise
      alone. In absolute terms, 100 / 4095 =~ 2.4% of the full 12-bit
      range -- a modest, realistic fraction to spend on headroom rather
      than dynamic range, consistent with common real sensor practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .psf import diameter_1e2_to_sigma, render_spot, sample_true_sigma
from .scene import render_background_gradient
from .sensor import (
    add_dark_current,
    add_hot_pixels,
    add_photon_noise,
    add_read_noise,
    apply_prnu,
    generate_hot_pixel_mask,
    generate_prnu_map,
    quantize_to_dn,
)


@dataclass
class Simulator:
    """A simulated camera: fixed unit properties + a per-frame renderer.

    Every noise LEVEL defaults to a physically modest, non-zero value, so a
    bare ``Simulator(shape=(25, 25))`` still produces a realistic frame --
    silently defaulting noise sources to zero would make it easy to
    accidentally characterise estimators against an unrealistically clean
    simulator. The one deliberate exception is the background gradient
    (``gradient_frac=0.0``): a gradient's strength and direction are
    scene-specific with no single representative default, so flat is the
    only choice that doesn't quietly assert something about a particular
    deployment. See the module docstring's "DEFAULT VALUES, AND WHY" for
    the reasoning behind every other value.
    """

    shape: tuple[int, int]
    nominal_diameter_1e2_px: float = 7.0
    sigma_tolerance_frac: float = 0.0  # 0 = exact nominal sigma, no unit-to-unit variation
    background_e: float = 30.0
    gradient_frac: float = 0.0
    gradient_angle_rad: float = 0.0
    dark_rate_e_per_s: float = 50.0
    exposure_s: float = 1e-3  # 1 ms, matching the brief's 1 kHz frame rate
    hot_fraction: float = 1e-4
    hot_rate_e_per_s: float = 5e4
    sigma_read_e: float = 5.0
    prnu_sigma: float = 0.02
    gain_e_per_dn: float = 10.0
    bit_depth: int = 12
    black_level_dn: float = 100.0
    seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)
    sigma: float = field(init=False)
    hot_mask: np.ndarray = field(init=False, repr=False)
    prnu_map: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        nominal_sigma = diameter_1e2_to_sigma(self.nominal_diameter_1e2_px)
        self.sigma = (
            nominal_sigma
            if self.sigma_tolerance_frac == 0.0
            else sample_true_sigma(nominal_sigma, self.sigma_tolerance_frac, self._rng)
        )
        self.hot_mask = generate_hot_pixel_mask(self.shape, self.hot_fraction, self._rng)
        self.prnu_map = generate_prnu_map(self.shape, self.prnu_sigma, self._rng)

    def render(self, x0: float, y0: float, flux: float) -> np.ndarray:
        """Render one full, realistic DN frame with a spot at (x0, y0)."""
        spot = render_spot(self.shape, x0, y0, flux, self.sigma)
        background = render_background_gradient(
            self.shape, self.background_e, self.gradient_frac, self.gradient_angle_rad
        )
        photo_signal = apply_prnu(spot + background, self.prnu_map)

        e_image = add_photon_noise(photo_signal, self._rng)
        e_image = add_dark_current(
            e_image, self.dark_rate_e_per_s, self.exposure_s, self._rng
        )
        e_image = add_hot_pixels(
            e_image, self.hot_mask, self.hot_rate_e_per_s, self.exposure_s, self._rng
        )
        e_image = add_read_noise(e_image, self.sigma_read_e, self._rng)

        return quantize_to_dn(
            e_image, self.gain_e_per_dn, self.bit_depth, self.black_level_dn
        )

    def dn_to_electrons(self, dn_image: np.ndarray) -> np.ndarray:
        """Invert the quantization step's units, for an estimator that
        wants to work in electrons (undoing gain and pedestal, but NOT
        recoverable: the rounding step and the noise itself are gone)."""
        return (dn_image - self.black_level_dn) * self.gain_e_per_dn
