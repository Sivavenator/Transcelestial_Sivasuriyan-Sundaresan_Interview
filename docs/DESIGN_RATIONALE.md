# Design rationale

Every significant concept used in this project, what it means, what it is
doing, and why it was chosen over the alternatives. Written to be
defended in conversation, so each entry states the reasoning from first
principles rather than citing convention, and names what was rejected.

Cross-references: `PROGRESS.md` maps requirements to evidence,
`ASSUMPTIONS.md` records the build history including bugs and wrong
turns, `REAL_WORLD_CONDITIONS.md` covers the deployment analysis.
Numerical results quoted here come from `results/*.json`, produced by the
scripts in `experiments/`.

## Contents

1. [Imaging model](#1-imaging-model)
2. [Noise model](#2-noise-model)
3. [Estimators](#3-estimators)
4. [The optimiser](#4-the-optimiser)
5. [Theoretical bound](#5-theoretical-bound)
6. [Windowing and background](#6-windowing-and-background)
7. [Motion model](#7-motion-model)
8. [Disturbance detection](#8-disturbance-detection)
9. [Temporal filtering, and why it is off](#9-temporal-filtering-and-why-it-is-off)
10. [Exposure control](#10-exposure-control)
11. [Calibration](#11-calibration)
12. [Atmospheric and environmental models](#12-atmospheric-and-environmental-models)
13. [Acquisition](#13-acquisition)
14. [Real-time structure](#14-real-time-structure)
15. [Choices deliberately not made](#15-choices-deliberately-not-made)

---

## 1. Imaging model

### Pixel-integrated PSF, not point-sampled

What it is. A pixel does not sample the optical field at a point. It
collects every photon landing anywhere on its light-sensitive area over
the exposure, so the value it reports is the integral of the incident
intensity across the pixel footprint. `psf.pixel_response_1d` computes
that integral exactly for a Gaussian, using the error function:

```
response(pixel i) = 0.5 * (erf((i + 0.5 - x0) / (sigma * sqrt(2)))
                         - erf((i - 0.5 - x0) / (sigma * sqrt(2))))
```

Why this and not point sampling. Evaluating the Gaussian at pixel centres
is one line shorter and wrong in a specific, measurable way: it makes the
model's response depend on where the spot sits inside a pixel differently
from how the real detector responds, which produces pixel locking, a bias
that varies with sub-pixel phase. `experiments/exp06a_pixel_locking.py`
measures this. On noiseless frames the Gaussian fit, whose forward model
is this same integral, recovers position to solver tolerance at every PSF
width tested, reading 0.000 millipixels of phase-dependent bias. The
centroid and matched filter, which do not carry the integral in their
model, show 27 and 31 millipixels peak-to-peak at sigma=0.4 px.

The cost is one `erf` call per pixel per axis instead of one `exp`. Both
are library calls of similar cost, and the separable structure means the
2-D response is an outer product of two 1-D vectors rather than a full 2-D
evaluation, so the exact version is not meaningfully more expensive.

Verified in `tests/test_psf.py` (flux conservation, centroid recovery to
1e-7 px) and `tests/test_psf_derivative.py` (analytic derivative against
finite differences to 1e-6).

### Spot width from the 1/e^2 diameter

What it is. The brief specifies a spot roughly 7 px in diameter at the
1/e^2 point. Laser beam widths are conventionally quoted as irradiance
`I = I0 * exp(-2 r^2 / w^2)`, where `w` is the 1/e^2 radius. A statistical
Gaussian is written `exp(-r^2 / (2 sigma^2))`. Matching exponents gives
`sigma = w / 2`, and since the quoted 7 px is a diameter, `w = 3.5 px`
and `sigma = 1.75 px`.

Why it matters that this is done carefully. The factor between the two
conventions is 2, so getting it wrong changes sigma by a factor of 2 or 4
depending on the direction of the error. Every precision result in the
project scales with sigma, and the pixel-locking result in particular
depends on where sigma sits relative to the sampling limit: 1.75 px is
comfortably Nyquist-sampled and locking-free, while 0.875 px would sit
near the onset measured in exp06a.

Implemented in `psf.diameter_1e2_to_sigma`, verified against its own
physical definition in `tests/test_psf_sigma.py`.

---

## 2. Noise model

### Why the order of operations is fixed

What it is. `simulate.Simulator.render` applies effects in a specific
order: spot and background combine, then PRNU, then photon noise, then
dark current, then hot pixels, then read noise, then quantisation.

Why that order. It follows the physical signal path, and two steps in
particular are not interchangeable:

- PRNU is applied before photon noise, and only to light that actually
  passed through the photon-to-electron conversion. PRNU is a
  pixel-to-pixel variation in quantum efficiency, so it scales
  photo-generated charge. Dark current and hot pixel charge do not pass
  through that conversion, so PRNU must not scale them. Applying PRNU at
  the end, to the summed image, would incorrectly modulate the dark
  signal.
- Photon noise is drawn on the PRNU-adjusted signal, not before it. The
  Poisson process operates on the actual expected electron count, which
  is what PRNU has already modified.

Read noise is added last in the electron domain because it originates in
the readout electronics, downstream of everything that happens in the
pixel well.

### Poisson for shot noise

What it is. Photon arrivals are independent events at a constant mean
rate, which is the definition of a Poisson process. Variance equals the
mean, so signal-to-noise for a pure photon-limited measurement is
`lambda / sqrt(lambda) = sqrt(lambda)`.

Why not a Gaussian approximation everywhere. At the photon counts this
project reaches in its low-light experiments (`exp05d` goes down to 0.4
photons in the peak pixel) a Gaussian is a poor description: it is
symmetric and admits negative values, while the real distribution is
skewed and non-negative. Using the true Poisson draw keeps the simulator
honest in exactly the regime where the estimators are being stress
tested. The Gaussian approximation is used in one place, the fit's
weighting, and that use is stated explicitly in section 3.

### Quantisation with a black-level pedestal

What it is. The ADC rounds to integer digital numbers. Rounding a
continuous value to a grid of spacing `g` adds a uniformly distributed
error with variance `g^2 / 12`.

Why the pedestal. Without an offset, read noise excursions below zero
electrons clip at the bottom of the ADC range. Clipping a two-sided
distribution on one side moves its mean, so the measured background
becomes biased upward by an amount that depends on the noise level. A
black-level pedestal lifts the whole signal above the clipping point so
both tails survive. This project uses 100 DN against a read noise of
0.5 DN, which is 200 sigma of margin, chosen with headroom because dark
current and background shot noise also contribute. Verified in
`tests/test_sensor.py`, including that the predicted clipping bias
matches `sigma / sqrt(2*pi)` when the pedestal is removed.

### SNR defined at the peak pixel

What it is. `snr.flux_to_snr` defines SNR as the peak pixel's signal
divided by the total noise standard deviation in that pixel.

Why peak-pixel and not total-flux SNR. Total-flux SNR describes how well
the spot could be detected in principle. Peak-pixel SNR describes what
the estimator actually sees in the pixel carrying the most information,
and it is the quantity that determines whether a spot is visible against
the background at all. It also makes the low-photon-count experiment
directly interpretable: SNR=0.1 corresponds to under one photon in the
brightest pixel, which is a statement about the physical situation rather
than about a derived quantity.

---

## 3. Estimators

### Windowed intensity-weighted centroid

What it is. Treat each background-subtracted pixel value as a mass and
compute the centre of mass. No model of the spot shape, no iteration, one
pass over the window.

Why include it. It is the cheapest possible estimator and it is unbiased
for any distribution symmetric about its true centre, which a Gaussian
is. It provides the seed for the fit, and it is the baseline that any
more complex method has to beat to justify itself.

Where it fails, measured. It weights every pixel equally regardless of
that pixel's noise, so background pixels far from the centre inject
variance multiplied by their distance from the centre. `exp06b` shows the
consequence: the centroid has an interior optimum in window size near
2.5 sigma and degrades by 119% at SNR=10 when forced to the project's
default half-width of 9 px, while the fit and matched filter pay under
10%.

### Poisson-weighted 2-D Gaussian fit

What it is. Fit the four parameters (x, y, flux, background) of a
pixel-integrated Gaussian to the window by minimising

```
chi2 = sum_i (data_i - model_i)^2 / var_i,   var_i = model_i + read_var
```

Why weighted and not ordinary least squares. Ordinary least squares
implicitly assumes every pixel has the same noise. That is false here by
construction: photon noise variance equals the signal, so a bright pixel
is genuinely noisier in absolute terms than a faint one. Weighting each
pixel by the inverse of its own predicted variance is the maximum
likelihood solution under Gaussian-approximated Poisson statistics, and
it is what allows this estimator to approach the Cramer-Rao bound.
Measured mean efficiency across the SNR sweep in `exp01`: 0.95 for the
fit against 0.63 for the centroid.

Why the weights use the model, not the data. Setting `var_i` from the
observed pixel value creates a feedback loop: a pixel that happened to
read low is then also trusted more, which biases the fit downward. Using
the current model prediction breaks that loop. Since the model changes
each iteration, the weights are recomputed each iteration, making this
iteratively reweighted least squares.

Why sigma is not fitted. The PSF width is treated as known and calibrated
separately. Fitting it would add a fifth parameter estimated from the
same data, increasing the variance of the position estimate, and real
systems calibrate PSF width once rather than per frame. This also keeps
the comparison against the centroid fair, since the centroid assumes a
fixed window and shape too.

### Matched filter with log-parabola interpolation

What it is. Correlate the window with a Gaussian template, find the peak,
then interpolate sub-pixel position from the peak sample and its two
neighbours.

Why log-parabola rather than plain parabola. A Gaussian's logarithm is
exactly a parabola. Fitting a parabola to three samples of the log of a
Gaussian therefore recovers the true peak position exactly, while fitting
a parabola to the raw samples is only approximate and leaves a bias that
depends on sub-pixel phase. Measured on a deterministic noiseless image
at the worst-case quarter-sample offset: log-parabola error 8e-7 px
against plain-parabola error 0.0075 px, a factor of roughly 10,000.
Verified in `tests/test_matched_filter.py`.

Why include it at all when the fit is more accurate. Fixed cost. The fit
iterates to convergence, so its per-frame time is data-dependent, and
`exp02` measured its worst frame exceeding the 1 ms budget. A correlation
plus a three-point interpolation is a fixed amount of work every frame.
That property, not accuracy, is the argument for it. Its measured mean
efficiency is 0.84, between the fit and the centroid.

A known tension, quantified. Matched filter theory says the
detection-optimal template matches the signal width. But correlating two
Gaussians of equal width produces a peak sqrt(2) times wider than either,
and a wider peak carries less information about its own location. The
detection-optimal template is therefore not the localisation-optimal
template. This shows up as the matched filter's efficiency falling as SNR
rises (0.96 at SNR=3 down to 0.68 at SNR=300) rather than improving, and
`template_sigma_scale` exposes the knob rather than hiding the choice.

---

## 4. The optimiser

This section exists because the choice of solver was raised directly.

### What Gauss-Newton is doing

The fit minimises a sum of squared weighted residuals. Newton's method
would require the full Hessian of that objective, including second
derivatives of the model. Gauss-Newton drops the term involving second
derivatives of the model and approximates the Hessian as `J^T W J`, using
only first derivatives. The approximation is good when residuals are
small, which is the case near a good fit.

### Why Gauss-Newton rather than a general-purpose minimiser

- The problem is a nonlinear least squares problem with an analytic
  Jacobian. `psf.pixel_response_1d_with_derivative` returns the response
  and its position derivative together, sharing the erf evaluations. A
  general minimiser using finite differences would need five model
  evaluations per iteration to build the same Jacobian numerically, and
  would inherit finite-difference truncation error into the step
  direction.
- `J^T W J` is exactly the Fisher information matrix for this problem.
  This has a practical consequence: the same
  matrix the solver already forms each iteration also gives the
  covariance of the estimate at convergence, so uncertainty reporting is
  free. `crlb.py` builds the bound from the same Jacobian and variance
  model, which is why the estimator and the bound cannot silently
  disagree about what problem is being solved.
- The parameter count is four and fixed. There is no need for the
  machinery that makes general optimisers general.

### Why not scipy.optimize.least_squares

It would work and would be fewer lines. It was not used because the
per-frame cost matters here and it carries Python-level call overhead per
residual evaluation, because the analytic Jacobian and its reuse for the
CRLB is a large part of what makes the project internally consistent, and
because a hand-written solver can be ported directly to the embedded
target that a 1 kHz loop implies. The tradeoff is that the solver has to
be tested, and it was: two real bugs in the convergence logic were found
this way, documented in `ASSUMPTIONS.md`.

### Why Levenberg-Marquardt damping

Plain Gauss-Newton can overshoot or diverge when the current guess is far
from the optimum, which is exactly the early-iteration situation. LM adds
`lambda * diag(J^T W J)` to the approximated Hessian. Large lambda makes
the step short and gradient-descent-like, which is safe but slow. Small
lambda recovers full Gauss-Newton speed. The damping is increased when a
step fails to reduce chi2 and decreased when it succeeds, so the solver
is conservative only when it needs to be.

A trust-region cap of one pixel per step is applied on top, because the
linearisation underlying the step is least trustworthy exactly when it
proposes a large jump.

### Why the fit is seeded from the centroid

It is a local optimiser: it refines a starting guess and does not search
globally. The centroid is cheap, already implemented, and typically
accurate to a fraction of a pixel, which puts the fit inside the basin of
attraction. This is also what a real system would do, rather than
starting from an arbitrary guess.

---

## 5. Theoretical bound

### Fisher information and the Cramer-Rao bound

What it is. Fisher information measures how sharply the likelihood
responds to a change in the parameter:

```
I(theta) = sum_i (d mu_i / d theta)^2 / var_i
```

and no unbiased estimator can have variance below `1 / I(theta)`.

What it means physically here. The contribution of a pixel to position
information is the square of how fast its brightness changes with
position, divided by its own noise. The brightest pixel is not the most
informative one, because it sits at the peak where the slope is nearly
zero. The pixels about one sigma out, on the steep flanks, carry the
most. Signal is not information; the derivative of signal with respect to
the parameter is.

Why compute it rather than only measuring performance. It converts "this
estimator is better than that one" into "this estimator is at 95% of what
any unbiased estimator could achieve", which is the difference between a
comparison and a characterisation. It also bounds how much further effort
on estimator design could possibly pay.

Consistency check performed. In the photon-limited case, `mu` scales with
flux, so the numerator scales as flux squared and the denominator as
flux, giving `I` proportional to flux and standard deviation proportional
to `1 / sqrt(flux)`. That reproduces the `sqrt(lambda)` scaling derived
independently from the coefficient-of-variation argument in `sensor.py`.

Known departure from the classical formula. The familiar result
`sigma_x = sigma_PSF / sqrt(N)` assumes continuous sampling. This
project's bound is built from the pixel-integrated response, so it sits
slightly above that: 15.1% at sigma=0.5 px, 1.35% at sigma=1.75 px,
0.04% at sigma=10 px, converging as sampling improves. This matches the
pixel-size correction in the localisation microscopy literature
(Mortensen et al. 2010) and was verified by sweeping sigma rather than
assumed.

---

## 6. Windowing and background

### Why a window at all

Restricting the estimator to a small region around the prior position
bounds compute per frame, and excludes distant clutter from ever entering
the calculation. `exp04d` shows the second point matters: once tracking
is locked, a bright false source outside the window cannot affect the
estimate at all.

### Border median for background

What it is. Estimate the background as the median of the window's outer
rows and columns.

Why the border and not the whole window. The spot occupies the centre.
Including centre pixels would let the spot's own signal raise the
background estimate, causing over-subtraction that scales with spot
brightness.

Why median and not mean. A single hot pixel or noise spike on the border
would move a mean, and that error then propagates into every pixel of the
subtracted window. A median requires a majority of border pixels to be
wrong before it moves.

Where it breaks, and the fix. The scalar assumes the background is
roughly uniform across the window. Under a strong gradient it is not.
`exp04e` measured the consequence: the median remains an accurate
estimate of the background at the window centre, to about 0.002
electrons, but subtracting one constant from a window whose true
background varies leaves a residual gradient, and the centroid responds
to that residual as signal. Measured bias reaches 2.5 px at the strongest
gradient tested. `estimators/base.planar_background` fits a plane to the
border instead and evaluates it per pixel, holding bias at the 0.0001 px
noise floor across the whole sweep.

### Clipping negative values, and its measured cost

What it is. After background subtraction, background-only pixels scatter
around zero. `clip_negative=True` sets the negative ones to zero.

The tradeoff as originally stated. Clipping removes the variance
contribution of negative-weighted pixels at large lever arm, at the cost
of a rectification bias, because truncating a two-sided distribution on
one side moves its mean.

What `exp06a` added. That bias is phase-dependent. Measured at the
operating point: 4.4 millipixels peak-to-peak across one pixel of
sub-pixel phase, against a standard error of 0.25, with a discontinuity
exactly where the integer window origin jumps. Rerunning identical frames
with clipping off flattens it to 0.9. The mechanism is that the rectified
residue is spatially lopsided when the spot sits off-centre in its
integer-placed window, pulling the centroid toward the window centre. A
phase-dependent bias does not average away as the spot moves, so it
appears as a position-dependent offset in a tracking loop rather than as
noise.

---

## 7. Motion model

### Three components with three different spectra

What it is. `trajectory.py` generates slow drift plus white jitter plus
one periodic disturbance.

Why these three, physically. They correspond to three real and distinct
mechanisms: thermal creep and mechanical settling, broadband platform
shake, and a dominant rotating machine.

Why the model for each is what it is:

- Drift is a random walk, a cumulative sum of small independent steps.
  Thermal creep genuinely accumulates: it does not return to where it
  started. A random walk is the direct model of an accumulation, not a
  filter applied to one. Its spectrum falls as 1/f^2, concentrating power
  at low frequency, which is what "slow" means quantitatively.
- Jitter is white Gaussian noise. Many quasi-independent micro-sources
  sum to something approximately Gaussian by the central limit theorem,
  the same argument used for read noise. If each source's correlation
  time is short compared to the 1 ms frame period, the sequence is
  effectively uncorrelated frame to frame.
- The disturbance is a single sinusoid. Rotating machinery couples
  vibration into a narrow band around its rotation rate rather than
  broadband, which is why it appears as a spectral line.

Why the spectral separation is the point, not a side effect. The three
components have deliberately different spectral shapes, and that is what
makes the disturbance separable from the rest of the motion at all. If
jitter were not approximately white, or if drift leaked power into the
disturbance band, the detector in section 8 could not work. Verified
numerically in `tests/test_trajectory.py`.

Parameter sourcing. The disturbance frequency of 20 Hz corresponds to
1200 RPM, inside the published 15 to 40 Hz reaction-wheel fundamental
harmonic band. Jitter magnitude is an engineering assumption: published
micro-vibration data is quoted in microradians, and converting it to
pixels requires a plate scale this project does not have. It is anchored
instead against a measured project quantity, sitting 15 to 20 times above
the fit's own single-frame precision at SNR=50, so that the motion is
resolvable rather than buried in measurement noise.

---

## 8. Disturbance detection

### Hann window before the FFT

What it is. Taper the sequence to zero at both ends before transforming.

Why. The recovered trajectory is not periodic within the capture window,
because drift does not return to its starting value. A plain FFT assumes
periodicity and therefore sees a step discontinuity at the wrap point,
whose energy spreads across the entire spectrum and can bury a small
line. Tapering removes the discontinuity. The cost is a wider main lobe,
which is accepted because the alternative is broadband leakage.

### Excluding the low-frequency band from the peak search

Why. Drift's own power is concentrated at low frequency and survives
windowing. A naive search for the largest peak would find drift rather
than the disturbance. `exclude_below_hz` removes that region, grounded in
the measured drift rolloff.

Its failure mode, measured. This is a fixed threshold, so it has a blind
spot. `exp03d` placed the disturbance near the boundary at an otherwise
easy amplitude and the detector returned the wrong frequency and a badly
wrong amplitude. A more robust design would estimate where drift power
actually falls off in the sequence at hand rather than assuming a fixed
cutoff. This is recorded as a real limitation.

### Single-bin amplitude with coherent-gain correction

What it is. Read amplitude from the peak bin and divide by the window's
coherent gain: `2 * |X| / sum(window)`.

Why not sum energy across the leaked lobe. That was tried first, on the
reasoning that an off-bin tone spreads energy into neighbours. Tested
against a known-amplitude tone it overestimated by 20 to 100%, because it
also sums the window function's own sidelobes as if they were independent
signal. The single corrected bin recovers true amplitude to within 0.4%
even off-bin.

### Noise-floor bias in peak detection

What `exp03d` found. Even with zero true disturbance the detector reports
a nonzero amplitude, 0.033 px in the configuration tested. The reported
value is the maximum over roughly 2000 candidate bins, and the maximum of
pure noise is not zero. This is the same statistical effect as
Rice-distributed noise-floor bias in radar and MRI, not an implementation
error. Practically it means that below roughly the noise floor, amplitude
alone cannot distinguish a weak disturbance from none, and frequency
agreement is the more robust indicator.

---

## 9. Temporal filtering, and why it is off

This section exists because the absence of a Kalman filter was raised
directly. It is implemented in `sptrack/tracking.py` and characterised in
`experiments/exp07_kalman_tracking.py`, so the decision rests on
measurement.

### What a Kalman filter does

It maintains a state estimate and its covariance, propagates them through
a motion model, and blends the prediction with each new measurement in
inverse proportion to their variances. When the model is informative it
reduces variance below what a single measurement can achieve.

### The governing quantity

A prediction only helps if it is competitive with the measurement. The
relevant comparison is how far the target moves unpredictably between
frames against how well one frame can be measured.

Measured for this system: the target moves 215 millipixels per frame,
dominated by white jitter at 150 millipixels, while the Gaussian fit
measures each frame to 8.76 millipixels. The ratio is 24 to 1 in favour
of the measurement. The optimal Kalman gain under those conditions is to
trust the measurement almost completely, which is the same as not
filtering.

### What the sweep found

Process noise was swept across twelve decades at four SNR values. At the
operating point no setting beat the unfiltered result, and as process
noise rises the filter converges back to the raw measurement, which is
the expected limiting behaviour. Improvement against the ratio:

| SNR | motion / noise | best improvement |
|---|---|---|
| 3 | 0.9 | +29% |
| 8 | 2.7 | +1% |
| 20 | 8.3 | 0% |
| 50 | 24.5 | 0% |

The crossover sits where the ratio approaches 1, as the first-principles
argument predicts.

### Two additional reasons specific to this motion

- Jitter is white by construction, so no causal filter can predict it.
  Smoothing it necessarily smooths across real motion.
- A constant-velocity model contains no resonator, so it cannot follow
  the 20 Hz tone without lag. At the smoothest setting tested the tone is
  passed at 0.21 amplitude with 10.7 ms of lag, which is 11 frame
  periods. The tone is the signal the system exists to track, so
  attenuating it is a direct failure.

### Why lag is the cost that matters

The estimate feeds a closed-loop pointing controller. A filter that
averages over the recent past reports where the spot was. Inside a loop
that delay consumes phase margin. Deciding whether a given delay is
acceptable requires the gimbal's bandwidth and phase margin, which this
project does not have, so the estimator exposes the trade rather than
choosing a point on it.

### When it would be switched on

The condition is a ratio, not an SNR, but for this trajectory it
corresponds to low SNR, which section 4 showed fog and deep scintillation
fades produce. A deployed system has a genuine case for enabling
filtering when SNR drops and leaving it off otherwise.

### Where the filter earns its place regardless

Its prediction step, not its smoothing. Section 4 measured 25 dropped
frames from scintillation fades, currently bridged by holding the last
known-good position. A constant-velocity state coasts through such a gap
at the last estimated velocity instead of freezing, which is better while
the target is moving.

### Why alpha-beta as well

For a constant-velocity model with stationary noise the Kalman gain
converges to a constant, after which the covariance recursion recomputes
the same two numbers every frame. The alpha-beta filter uses the
converged gains directly, reducing the update to a few scalar operations.
`steady_state_gains` computes them by running the covariance recursion to
convergence, which requires no data because that recursion does not
depend on the measurements.

---

## 10. Exposure control

### Proportional control on peak DN

What it is. Measure the previous frame's peak digital number, compare it
to a target fraction of the available range, and scale gain by the ratio,
with a bound on how much it can change per frame.

Why proportional and not something more elaborate. The plant is a static
gain with a one-frame delay and no dynamics, so there is nothing for
integral or derivative terms to do. The per-step bound exists because a
real actuator cannot jump instantly and because a single frame's peak is
itself noisy.

Why the correction is applied before the photon draw. Multiplying flux
before Poisson sampling models exposure time, aperture, or attenuation
changes, which genuinely change how many photons arrive and therefore
genuinely improve SNR. Multiplying an already-noisy signal afterwards
models post-hoc analog gain, which scales signal and noise together and
improves nothing. The distinction is stated explicitly to avoid implying
that gain fixes shot noise.

### The measured asymmetry

Recovery from sudden dimming took 9 frames; recovery from sudden
brightening took 31. A saturated reading is clipped, so it reports "still
too bright" without reporting by how much, and the magnitude information
needed for a confident single-step correction is destroyed. An
underexposed reading is never clipped and carries that information
intact. A deployed system should expect slower recovery from clearing
cloud than from arriving cloud.

---

## 11. Calibration

### Bias frame

Averaging dark frames recovers the fixed additive pattern, which in this
sensor model means the hot pixel map. Dark current and the pedestal are
spatially uniform here, so the scalar background estimate already handles
them and they need no per-pixel calibration. Frame count is derived: to
keep the calibration map's own residual noise below 10% of a single
frame's read noise requires `1/sqrt(N) <= 0.1`, so `N >= 100`.

### Flat field

Averaging bright uniform frames recovers the PRNU gain map. Brightness
and frame count are derived rather than chosen: measuring a 2% PRNU
against photon noise requires SNR well above `1/0.02 = 50`, and a 10x
margin means 500, requiring `500^2 = 250,000` total electrons per pixel.
At roughly half the sensor's headroom per frame, that is 13 frames.

### Lens distortion

Modelled as a single radial term with the fractional displacement at the
frame corner set to -0.1%. This magnitude is not derivable from anything
in the project, so it was taken from published machine vision lens
datasheets in the precision class, matching this project's narrow-field
tracking optic rather than a wide-angle imaging lens.

The correction inverts `r_d = r (1 + k1 r^2)` by fixed-point iteration
rather than solving the cubic, because k1 is small and the iteration
converges to floating point precision in a few steps. Verified by
round-trip to 1e-9 px.

Why it is worth correcting at all at 0.1%. The uncorrected geometric
error reaches 0.212 px at the frame edge, which is far above this
project's best single-frame precision of a few millipixels. It is also
purely geometric, present with a perfect noiseless estimator, unlike
every other bias source here.

---

## 12. Atmospheric and environmental models

### Scintillation as a mean-reverting log-normal process

Why log-normal. Weak-turbulence theory gives log-normally distributed
received irradiance, and intensity must remain positive, which a
log-normal guarantees and a Gaussian does not.

Why mean-reverting and not a random walk. Scintillation fluctuates around
a stable long-run mean. Drift accumulates without bound. Using a random
walk here would be the same category of modelling error as using a
mean-reverting process for drift.

Why temporally correlated and not per-frame independent. Turbulent eddies
take real time to cross the beam. The coherence time used, 5 ms, is
comparable to the 1 ms frame period, so consecutive frames fade together.
Modelling it as independent noise would understate its operational
impact, because a multi-frame dropout is a different problem to the same
total noise spread across frames.

Parameter placement. `sigma_ln` of 0.4 and 0.6 give scintillation indices
of 0.174 and 0.433, inside the measured FSO range of 0.083 to 0.71 and
below the 0.75 limit where the log-normal model stops being valid.

### Beam wander

The same mean-reverting structure, but applied linearly rather than in
log space, because a position perturbation can be negative while an
intensity cannot.

Why a longer coherence time than scintillation. Angle-of-arrival
fluctuation is effectively an aperture-averaged wavefront tilt, and
aperture averaging smooths out exactly the small-scale fast structure
that drives intensity scintillation.

Why it is modelled separately from mechanical jitter rather than folded
into it. They are different physical sources, one from the platform and
one from the air, and their variances add rather than merge. They also
have different spectra: `exp04c` shows that two position noise sources
with identical standard deviation of 0.15 px are still separable by
spectral shape by a factor of roughly 250 in low-band to high-band power
ratio, and that they combine in quadrature as independent sources should.

### Fog as a steady sweep, not a process

Fog changes over minutes, not milliseconds. Modelling it as a
within-sequence random process would misrepresent its timescale. It is
instead a steady attenuation level swept across named weather conditions,
which reuses the existing SNR machinery and required no new module.

---

## 13. Acquisition

### Why brightest-pixel acquisition is unsafe

`find_brightest_pixel` picks the single brightest pixel with no other
criterion. A false source does not need more total flux than the laser to
win, only a higher peak, and peak brightness scales as `flux / sigma^2`.
A wider source with considerably more total flux can therefore still win.
`exp04d` demonstrates a clutter source with 13 times the laser's total
flux and a wider profile capturing acquisition.

### Why PSF shape is the right discriminator

The laser is diffraction-limited, so its profile matches the known PSF
width. Generic clutter, being a diffuse reflection or a differently
figured source, usually does not. `acquire_target` ranks candidate local
maxima by Pearson correlation against the assumed PSF template, which is
scale invariant and therefore judges shape rather than brightness.
Measured on the same frame: 0.984 for the true spot against 0.690 for the
clutter.

What it does not solve. A genuinely point-like false source is not
separable by shape. Flux range and velocity consistency are the further
discriminators, and are not implemented.

---

## 14. Real-time structure

### Throughput and latency are different questions

Throughput asks whether the pipeline can sustain one frame per period.
Latency asks how old the answer is when the controller acts on it. Both
matter and they have different answers.

### The structural result

Exposure alone is 1 ms, which is the entire 1 kHz period. Readout and
compute therefore cannot run serially after exposure and still sustain
the rate. They must overlap with the next frame's exposure, which is what
a global shutter sensor with a separate readout node provides. The
correct throughput test is whether readout plus transfer plus compute
fits inside one exposure window, not whether exposure plus readout does.

Under that test all three estimators fit at the median. The Gaussian
fit's measured tail does not: its worst frames push past the window,
which is the one place in the pipeline where the choice of algorithm,
rather than fixed sensor timing, threatens the rate.

### Why worst case rather than mean

A 1 kHz loop must finish every frame, not the average frame. A method
that is fast 999 times in 1000 still misses a deadline once per second.
The reported metric is therefore the high percentile and the observed
maximum.

---

## 15. Choices deliberately not made

Recorded so they are answerable rather than absent.

- Kalman filtering in the main pipeline. Implemented and measured;
  section 9 gives the ratio argument and the crossover.
- Fitting PSF width per frame. Would add a parameter estimated from the
  same data and increase position variance. Real systems calibrate width
  separately.
- Constant-acceleration motion model. Adds a state with nothing to
  estimate against a random walk, and costs variance.
- A general-purpose optimiser. Section 4.
- Sub-bin frequency interpolation in the disturbance detector. The
  bin-resolution estimate already lands within one bin on the easy
  scenario, and claiming better precision than was verified would be
  unsupported.
- PSF model mismatch study. The fit's immunity to pixel locking is a
  model-match result, so it bounds locking due to pixel integration only.
  Bounding locking under model error needs a mismatch study, which has
  not been run. This is the clearest remaining gap in the estimator
  characterisation.
- Sensor saturation as a dedicated study, cosmic ray hits, impulsive
  thruster disturbances, and an angular pointing budget. Identified in
  `REAL_WORLD_CONDITIONS.md` under further considerations, with the
  reasons each would need work this project has not done.
