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
Read noise is the electronic noise added during signal readout -- when the
sensor converts accumulated charge into a digital number. It is present on
EVERY pixel, EVERY frame, regardless of exposure time or light level (unlike
dark current, which at least depends on exposure and temperature). It's
measured in electrons because at this stage we're still counting actual
charge carriers, before analogue-to-digital conversion.

WHERE IT COMES FROM
-----------------------
  * amplifier noise in the pixel's source follower
  * ADC quantisation noise
  * on-chip circuitry interference

Reading a pixel's accumulated charge off the sensor and converting it to a
digital number goes through analogue electronics (an amplifier, a
capacitor, an ADC), and each of those stages contributes its own small
random error. By the central limit theorem, summing many small independent
error sources produces something close to Gaussian -- so unlike photon
noise, this is an additive ELECTRONIC process, not a counting process, and
it does not scale with signal at all. Modelled as zero-mean Gaussian with a
fixed standard deviation:

    N_read ~ Normal(0, sigma_read^2)

The critical difference from photon noise is what it depends on:

    photon noise std  = sqrt(signal)   -- grows with brightness
    read noise std     = sigma_read    -- FIXED, independent of brightness

WHAT sigma_read = 5.0 e- MEANS IN CONTEXT
----------------------------------------------
    sensor type                  typical sigma_read
    cheap phone sensor           10-20 e-
    typical CMOS (consumer)       3-8 e-
    ** this project: 5.0 e- **   right here -- solid mid-range
    scientific sCMOS               1-2 e-
    EMCCD (cooled)                < 1 e-

5.0 e- is a realistic, conservative-but-reasonable value -- nothing exotic,
comfortably within the range of an ordinary consumer CMOS sensor.

HOW IT COMPETES WITH SHOT NOISE: TWO REGIMES
-------------------------------------------------
This is why read noise is the noise floor for a *dark or faint* scene: at
low signal, sqrt(signal) is small and sigma_read dominates; at high signal,
sqrt(signal) eventually outgrows the fixed sigma_read and photon noise takes
over. Total noise combines the two in quadrature (independent sources, so
variances add):

    Var[total] = signal + sigma_read^2
    sigma_total = sqrt(sigma_read^2 + lambda)     where lambda = signal (photon count)

    regime                 condition             dominant noise
    read-noise limited      lambda << sigma_read^2 = 25    dark/dim pixels
    shot-noise limited      lambda >> 25                   bright pixels

With sigma_read = 5.0 e-, the crossover sits around ~25 photons: below that,
read noise dominates and no amount of clever estimation recovers what the
electronics already threw away; above it, Poisson shot noise takes over and
the sensor is behaving as well as physics allows.

THE PRACTICAL IMPLICATION
------------------------------
Read noise sets the noise floor -- the minimum uncertainty per pixel no
matter what. It is why:
  * long exposures beat many short ones (shot noise averages down across
    accumulated signal; read noise is paid once PER READ, so more reads
    means more read-noise contributions adding up)
  * deep-sky astrophotographers obsess over read noise
  * scientific cameras cool the sensor and use low-noise amplifiers

DARK CURRENT
---------------
Even with the shutter closed, a sensor pixel accumulates a small number of
electrons from pure thermal energy: heat randomly kicks electrons across the
semiconductor bandgap, generating electron-hole pairs indistinguishable from
ones a photon would have generated. This has nothing to do with light.

The mean number of dark electrons collected grows linearly with exposure
time and roughly doubles every 6-8 degC (an Arrhenius-type dependence on
temperature) -- both are why it is configurable here rather than a fixed
constant:

    mean_dark = dark_rate_e_per_s * exposure_s

Because dark-electron generation is, like photon detection, a sequence of
independent random events at a constant average rate, it is ALSO a Poisson
process, with the identical defining property:

    Var[N_dark] = E[N_dark] = mean_dark

WHY DARK CURRENT IS DRAWN AS ITS OWN POISSON NOISE, NOT FOLDED INTO THE MEAN
-------------------------------------------------------------------------------
A cleaner-looking pipeline might add ``mean_dark`` directly into the spot's
mean image and apply photon noise once, on the combined total. That would
actually give the identical result, by a basic property of the Poisson
distribution: the sum of two independent Poisson random variables is itself
Poisson with the summed rate,

    Poisson(a) + Poisson(b)   is equal in distribution to   Poisson(a + b)

So drawing dark current as its own independent Poisson noise and adding it
to an already photon-noised signal is mathematically equivalent to combining
the means first and drawing once. We keep it as a separate function anyway,
for the same reason as read noise: the brief asks for each source to be
individually configurable, and a later calibration study (subtracting a
"dark frame") needs dark current to be a distinct, addressable quantity
rather than baked silently into the signal.

Dark current is often negligible at these exposure times (sub-millisecond,
to hit 1 kHz) and room temperature, but grows fast with either a longer
exposure or a hotter sensor -- which is exactly why it needs to be
simulated rather than assumed away.

KEY FACTS, SUMMARISED
-------------------------
  * thermal origin -- silicon lattice vibrations spontaneously free
    electrons, indistinguishable from a true photon hit
  * exponential temperature dependence -- the rate roughly doubles every
    5-8 degC (commonly cited as 6-8 degC), so it escalates fast in a warm
    environment or over a long exposure
  * impact -- reduces usable dynamic range, adds background noise across
    the whole frame, and is the source of the bright "hot spots" in dark
    image regions that hot pixels (below) are the extreme case of

MITIGATION, FOR LATER (NOT YET IMPLEMENTED HERE)
-----------------------------------------------------
Not built into the simulator yet -- these are the real-world techniques a
deployed system would use, worth stating now because they motivate the
calibration work the brief's "go further" section asks about:
  * sensor cooling -- dark current roughly halves for every 6-7 degC
    removed, which is why scientific and astrophotography cameras are
    actively cooled
  * dark-frame subtraction -- capture a reference frame with the same
    exposure time and temperature but no light, then subtract it from every
    science frame, removing the mean dark contribution (not its shot noise,
    which is still random)
  * pixel mapping -- identify permanently misbehaving hot pixels once
    (calibration) and mask/interpolate over them in every subsequent frame,
    rather than trusting the raw reading

HOT PIXELS
-------------
A manufacturing defect can leave a handful of pixels with a dark-current
rate orders of magnitude above the rest of the sensor. This is a genuinely
different kind of noise source from everything above, in one crucial way:

    Every noise source so far is a FRESH random draw every frame.
    A hot pixel is a FIXED spatial defect -- the SAME pixel, every frame.

So modelling it needs two pieces, not one: a defect MAP (which pixels are
hot -- generated once, like a manufacturing property of this particular
sensor, and reused across every frame) and the elevated dark-current draw
itself (still random frame to frame, just at a much higher rate, at exactly
those fixed locations).

WHY THIS MATTERS FOR A CENTROID
-----------------------------------
A plain intensity-weighted centroid has no concept of "this pixel doesn't
belong." A single hot pixel, sitting anywhere in the fitting window, acts as
a large, FIXED lever arm: unlike random noise (which averages toward zero
bias over many frames), a hot pixel pulls the estimate toward its own fixed
location on every single frame, the same direction every time. That is a
systematic bias, not scatter -- and it does not average away no matter how
many frames you collect. This is exactly why real systems need
background/outlier handling (a hot-pixel map from calibration, or a
robust-statistics fit) rather than trusting a naive centroid on raw data.

In practice this is solved by pixel mapping: identify the defective pixels
once, during calibration (they are fixed, so this only has to be done
occasionally), and mask or interpolate over them in every subsequent frame
rather than trusting their raw reading. `generate_hot_pixel_mask` above
produces exactly the kind of map a real calibration step would also
produce -- here it's ground truth (we generated the defect), there it would
be measured.

PIXEL-GAIN NON-UNIFORMITY (PRNU)
-------------------------------------
Every pixel's photodetector has a very slightly different quantum
efficiency -- how many of the photons landing on it actually get converted
to electrons -- because of small manufacturing variations (silicon defects,
micro-lens or coating thickness differences). Point the exact same light at
two different pixels and they report very slightly different counts. Unlike
hot pixels, this is not a rare defect in a handful of pixels: EVERY pixel
has its own small gain, typically within about 1-2% of the sensor's average
for a consumer-grade device.

MULTIPLICATIVE, NOT ADDITIVE -- WHY THAT DISTINCTION MATTERS
------------------------------------------------------------------
Every noise source above this one in the file is additive: it adds a fixed
amount (read noise, PRNU's cousin dark current) or a fixed-rate random draw,
regardless of how bright the pixel already is. PRNU is different -- it's a
GAIN, multiplying whatever signal is already there:

    observed = true_signal * gain_i,     gain_i ~ Normal(1.0, sigma_prnu^2)

A pixel with 2% higher gain reports 2% more electrons whether 10 photons or
10,000 land on it. This means PRNU's absolute effect *scales with signal* --
negligible on a faint pixel, and becomes the dominant error source once
signal is bright enough that shot noise (which only grows as sqrt(signal))
falls behind a gain error that grows linearly with signal.

WHY IT ONLY APPLIES TO PHOTO-GENERATED SIGNAL, NOT DARK CURRENT
---------------------------------------------------------------------
PRNU is specifically a property of the photon-to-electron conversion
pathway (quantum efficiency). Dark current electrons are generated directly
in the silicon by thermal excitation -- they never pass through that
photodetection pathway at all, so a pixel's photon-conversion gain has
nothing to multiply for them. PRNU is applied to the spot's signal and the
background (both are photons entering through the optics), and NOT to dark
current or hot-pixel electrons, which are generated independently of light.

FIXED SPATIAL PATTERN, LIKE HOT PIXELS
-------------------------------------------
Same reasoning as the hot-pixel defect map: this is a manufacturing
property of one particular sensor, so it is generated once and reused every
frame, never redrawn -- and for the same reason, map generation and map
application are kept as two separate functions.

WHY THIS MATTERS MORE THAN IT SOUNDS: A POSITION-DEPENDENT BIAS
---------------------------------------------------------------------
A uniform gain error (every pixel off by the same factor) would just be a
flux-calibration problem -- harmless for position, since scaling every pixel
in a window by the same constant does not move its centroid at all. PRNU is
NOT uniform: it is a different small ripple at every pixel. Under the
spot's PSF, that ripple multiplies different parts of the Gaussian
differently -- and which part of the ripple pattern sits "under" the peak,
versus under the wings, changes depending on exactly where the spot's
sub-pixel centre sits. That makes PRNU a genuinely POSITION-DEPENDENT bias,
in the same family as pixel locking (from psf.py) rather than simple noise:

  * random noise sources (photon, read, dark) average toward zero bias
    over many frames, because they are independent draws every frame
  * PRNU is FIXED, so its effect on any given sub-pixel offset is the
    same, systematic amount every single frame -- it does not average
    away, no matter how many frames are collected

This is why PRNU becomes the error floor that a high-SNR system cannot
average its way past, and why real systems need a flat-field calibration
(measuring each pixel's gain once and dividing it out) rather than relying
on more frames to beat it down.

ONE MORE DETAIL, CHECKED NUMERICALLY RATHER THAN ASSUMED: THE BIAS CURVE IS
SMOOTH BUT NOT PERIODIC ACROSS PIXEL BOUNDARIES
------------------------------------------------------------------------------
Plotting bias vs. sub-pixel position produces a curve that is smooth
(continuous, no jumps) within any local neighbourhood -- that part comes
from the Gaussian PSF, which itself varies smoothly with position. It is
NOT periodic from one pixel to the next, though: comparing the bias at a
given offset to the bias one and two pixels away shows differences of
several millipixels, not zero. This makes physical sense -- as the spot
shifts by exactly one pixel, its rendered footprint shifts by one array
index too, but the FIXED gain map underneath does not shift with it, so
each pixel period lands the identical spot shape against a different,
uncorrelated slice of the map. (At a glance, a wide PSF like sigma=1.75
averages over enough pixels that the curve can still *look* like one smooth
multi-pixel trend rather than obviously different wiggles each period --
which is exactly why this was checked numerically rather than left as a
visual impression.)

A SECOND DETAIL: BOTH SWEEP DIRECTIONS BEHAVE THE SAME WAY, AND A ZERO
CROSSING DOES NOT MEAN "SAFE"
--------------------------------------------------------------------------------
Sweeping the spot vertically (varying y0 at fixed x0) shows the identical
qualitative behaviour as sweeping it horizontally -- a smooth,
non-periodic, generally non-zero bias, confirming this is a property of the
fixed gain map rather than an artefact of one axis. One detail worth being
precise about: the vertical sweep's bias curve happens to cross exactly
zero at one specific position, since it swings from negative to positive
and a continuous curve must pass through zero somewhere in between. That
crossing is a coincidence of this particular random map at that exact
position, not a general cancellation -- a real system has no way to know
where it is without already having measured (calibrated) the gain map, at
which point it would just correct the bias directly rather than search for
a lucky spot. So "sometimes crosses zero" does not weaken the conclusion:
the bias is still present at almost every position, unpredictably, and
still cannot be trusted away by hoping for that one coincidence.

BIT-DEPTH QUANTIZATION
--------------------------
Everything above this point in the file operates in electrons -- a
continuous, real-valued quantity. The very last step in a real sensor's
pipeline is the ADC (analogue-to-digital converter), which reports a
DISCRETE digital number (DN), not the true continuous value. This is the
one noise source in the whole chain that also changes the image's UNITS,
from electrons to DN, via a conversion gain:

    DN = round(electrons / gain_e_per_dn)

with a finite number of representable levels set by the bit depth:

    n_levels = 2 ** bit_depth        (e.g. 12-bit -> 4096 levels, 0..4095)

QUANTIZATION ERROR AS UNIFORM NOISE
----------------------------------------
The ADC only ever outputs whole DN values -- it rounds. That rounding
happens in DN SPACE, not electron space: the natural pipeline is
electrons -> divide by gain -> DN (still a real number at this point) ->
round to the nearest integer -> that rounded integer is the actual
quantized output. Getting this order right matters, because it changes
where the "1" in the variance formula's denominator lives.

Step 1 -- where the 1/12 comes from, in DN^2.
Rounding to the nearest integer DN produces an error
``error = true_value - rounded_value``. If the true value's fractional part
can land anywhere within one DN bin (say, anywhere between n-0.5 and n+0.5,
all of which round to n), that error is uniformly distributed on
[-0.5, +0.5] DN -- a classical result that holds when the signal already
has some other noise mixed in before rounding (here, everything upstream:
photon noise, read noise, dark current), which "dithers" the value enough
that the rounding error behaves like a draw from a uniform distribution
rather than something value-dependent and awkward to model.

The variance of a uniform distribution on [a, b] is a standard result,
Var = (b-a)^2 / 12; here the bin width is b - a = 1 DN, so:

    Var[quantization], in DN^2 = 1^2 / 12 = 1/12

That 12 isn't an arbitrary constant -- it falls straight out of integrating
x^2 over the uniform density on [-0.5, 0.5]:

    integral(-0.5 to 0.5) of x^2 dx  =  [x^3 / 3] from -0.5 to 0.5
                                      =  (1/24) - (-1/24)  =  1/12

Step 2 -- why converting to electrons multiplies by gain^2, not gain.
Since electrons = DN * gain, the rounding error itself also scales by gain:

    error_in_electrons = error_in_DN * gain

And variance scales as the SQUARE of a linear scale factor -- a basic
property of variance, Var[a*X] = a^2 * Var[X] for any constant a. So:

    Var[quantization], in electrons^2 = gain_e_per_dn^2 * Var[quantization in DN]
                                       = gain_e_per_dn^2 * (1/12)
                                       = gain_e_per_dn^2 / 12

A bigger gain means each DN step represents more electrons, so rounding to
the nearest DN throws away more electrons' worth of precision -- which is
exactly what the gain^2 factor captures.

CONCRETE NUMBERS
--------------------
Using a representative gain of ~9.77 e-/DN (a 40,000-electron full well
spread over a 12-bit, 4096-level ADC -- gain = full_well / n_levels):

    Var[quantization], in DN^2        = 1/12                    ~= 0.083 DN^2
    Var[quantization], in electrons^2 = 9.77^2 / 12 ~= 95.5/12   ~= 7.96 e-^2

So the STANDARD DEVIATION of quantization noise, in electrons, is
sqrt(7.96) ~= 2.82 e- -- a real, non-negligible noise floor once compared
against something like sigma_read = 5.0 e- from earlier, and it stacks into
the same running noise budget (variances add for independent sources):

    Var[total], in electrons^2 = signal + mean_dark + sigma_read^2 + gain_e_per_dn^2 / 12

WHY A BLACK-LEVEL PEDESTAL IS NEEDED -- CLOSING THE LOOP FROM add_read_noise
---------------------------------------------------------------------------------
``add_read_noise`` above notes it "can produce negative values... real
sensors handle this with a black-level pedestal... that pedestal is a
separate, later concern." This is that concern, resolved here: without a
pedestal, negative electron counts (from a read-noise excursion below zero
on a dim pixel) would clip to DN=0, and clipping only ever removes the
negative tail -- never the positive one -- which biases the *mean* upward.
Adding a fixed pedestal (e.g. +100 DN) before quantizing shifts the whole
distribution up first, so a typical negative excursion still lands on a
valid, unclipped DN (e.g. 90 instead of being clipped to 0), preserving the
noise's symmetry. The pedestal is later subtracted back out in calibration
(a "bias frame"), which is exactly why bias-frame subtraction exists as a
real calibration step.

SATURATION: THE OTHER CLIP, AT THE TOP END
-----------------------------------------------
The ADC also cannot represent anything above its maximum level
(``2**bit_depth - 1``). Clipping there is what saturation physically is:
once a pixel's true electron count would map to a DN above that ceiling,
all the information about exactly how much brighter it was is gone --
which is why saturation is a hard bias, not recoverable noise, and why
auto-exposure/gain control (a "go further" item) exists specifically to
avoid it.
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


def add_dark_current(
    image: np.ndarray,
    dark_rate_e_per_s: float,
    exposure_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Poisson-distributed dark current electrons to an image.

    ``dark_rate_e_per_s`` is the sensor's dark-current rate (electrons per
    pixel per second, temperature-dependent in reality but a fixed input
    here); ``exposure_s`` is the exposure time. The mean dark charge per
    pixel is ``dark_rate_e_per_s * exposure_s``, drawn as its own Poisson
    noise and added to ``image`` -- mathematically equivalent to adding the
    mean into the signal before a single combined Poisson draw, since
    Poisson(a) + Poisson(b) is distributed as Poisson(a + b).
    """
    mean_dark = dark_rate_e_per_s * exposure_s
    dark_electrons = rng.poisson(mean_dark, size=image.shape).astype(np.float64)
    return image + dark_electrons


def generate_hot_pixel_mask(
    shape: tuple[int, int], fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a FIXED boolean map of which pixels are hot defects.

    This represents a manufacturing property of one particular sensor: call
    it once per simulated sensor and reuse the same mask on every frame --
    do not call it fresh per frame, or the defect would (wrongly) move
    around instead of staying put.
    """
    return rng.random(shape) < fraction


def add_hot_pixels(
    image: np.ndarray,
    hot_mask: np.ndarray,
    hot_rate_e_per_s: float,
    exposure_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add elevated dark current at the fixed locations marked in ``hot_mask``.

    ``hot_rate_e_per_s`` is typically orders of magnitude above a normal
    pixel's dark rate. Still drawn fresh (Poisson) every frame, like ordinary
    dark current -- only the *location* of the defect is fixed, not its
    exact per-frame reading.
    """
    mean_hot = hot_rate_e_per_s * exposure_s
    hot_electrons = rng.poisson(mean_hot, size=image.shape).astype(np.float64)
    return image + np.where(hot_mask, hot_electrons, 0.0)


def generate_prnu_map(
    shape: tuple[int, int], sigma_prnu: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a FIXED per-pixel multiplicative gain map.

    Values are centred on 1.0 (average sensor gain) with standard deviation
    ``sigma_prnu`` (e.g. 0.02 for 2%). Like the hot-pixel mask, this is a
    manufacturing property of one particular sensor -- generate it once and
    reuse the same map every frame, never redraw it per frame.
    """
    return rng.normal(1.0, sigma_prnu, size=shape)


def apply_prnu(photo_signal: np.ndarray, prnu_map: np.ndarray) -> np.ndarray:
    """Apply the fixed per-pixel gain map to photo-generated signal.

    ``photo_signal`` should be the spot plus background -- light that
    actually passed through the photon-to-electron conversion pathway.
    Deliberately does NOT include dark current or hot-pixel electrons, which
    bypass that pathway entirely (see the module docstring).
    """
    return photo_signal * prnu_map


def quantize_to_dn(
    image_e: np.ndarray,
    gain_e_per_dn: float,
    bit_depth: int,
    black_level_dn: float = 0.0,
) -> np.ndarray:
    """Convert electrons to a clipped, integer-valued digital number (DN).

    This is the LAST step in the chain and the only one that changes units:
    input is electrons, output is DN. ``black_level_dn`` is a fixed pedestal
    added before quantizing, so that negative electron excursions (from read
    noise on a dim pixel) don't get asymmetrically clipped at zero and bias
    the mean upward -- see the module docstring. Output is clipped to the
    ADC's representable range ``[0, 2**bit_depth - 1]``; the top clip is
    what saturation physically is.
    """
    max_dn = 2**bit_depth - 1
    dn = np.round(image_e / gain_e_per_dn + black_level_dn)
    return np.clip(dn, 0, max_dn)
