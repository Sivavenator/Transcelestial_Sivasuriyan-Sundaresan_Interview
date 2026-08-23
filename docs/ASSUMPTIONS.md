# Assumptions

Every deliberate choice made in this project that isn't derived from the
brief, with why it was made. Updated as we build each new piece — nothing
listed here speculatively ahead of the code that actually makes the
assumption.

The brief itself says: *"expect to make and state your own assumptions"* —
this document is where we honour that.

---

## Simulator (`sptrack/psf.py`)

**Pixel ``i`` spans the interval ``[i - 0.5, i + 0.5]``.**
- Why: pixel *centres* sit at integer coordinates, so pixel 0 is centred at
  x=0 and covers x in [-0.5, 0.5]. This is the standard convention in optics
  and astronomy code (and matches how a physical sensor pixel's photosite is
  centred under its own coordinate).
- Consequence: a spot at ``x0 = 0.0`` sits exactly on a pixel centre; a spot
  at ``x0 = 0.5`` sits exactly on a pixel boundary. Both are valid inputs —
  there's no special-casing needed, which is part of why this convention was
  chosen.

**Pixel coordinates start at 0, not 1** (``x0 = 0`` is the centre of the
leftmost/topmost pixel).
- Why: matches NumPy/Python array indexing directly — no off-by-one
  translation needed anywhere else in the code.

**The PSF is a 2-D Gaussian, and it's separable: ``G(x, y) = G(x) · G(y)``.**
- Why: the brief explicitly says "a Gaussian PSF is fine." A real optical PSF
  from a lens with aberrations is not exactly Gaussian and not always
  separable, but a Gaussian is the standard, defensible stand-in, and
  separability turns an expensive 2-D integral into two cheap 1-D ones.
- Consequence: for now, ``render_spot`` uses a single ``sigma`` shared by
  both axes (isotropic). An elongated/astigmatic spot would need separate
  ``sigma_x``/``sigma_y`` — not yet needed, so not yet built.

**``flux`` is the total signal integrated over an infinite window.**
- What it actually is: the total signal from the laser spot, **summed across
  the whole spot** — not any single pixel's brightness, the sum over every
  pixel the spot touches. ``render_spot(..., flux=1000.0, ...)`` means "this
  spot carries 1000 electrons total"; the Gaussian shape then decides how
  that total gets split across pixels (the centre pixel gets the biggest
  share, via ``pixel_response_1d``'s fractions, with less further out).
- Why this unit and not, say, "peak pixel brightness": it's the natural
  physical quantity — in a real system, flux is what ultimately falls out of
  laser power, distance/atmospheric loss, optics collection efficiency,
  exposure time, and sensor quantum efficiency, all collapsed into one
  number: how many electrons landed on the sensor this frame. It also makes
  the renderer's output additive and easy to reason about.
- Why it matters for noise: flux directly sets the photon-noise floor — each
  pixel's own share of the flux becomes *that pixel's* Poisson `lambda`, so
  a bigger total flux means every pixel gets more photons, which (per the CV
  derivation in `sensor.py`) means better relative precision everywhere, not
  just a brighter-looking image.
- Consequence: a *finite* rendered window always captures slightly less than
  the full ``flux`` — the far Gaussian wings fall outside the window. Not yet
  an issue at the window sizes tested (25×25 around ``sigma=1.75``, which
  encloses >99.9% of the flux), but worth remembering once window size
  becomes a tunable parameter.

---

## Noise chain (`sptrack/sensor.py`)

**Randomness is caller-supplied, via `numpy.random.Generator`, never a
module-global seed.**
- Why: the brief's own "strong submission" bar requires Monte Carlo
  characterization and exact reproducibility. A caller-supplied generator
  means every experiment controls its own seed explicitly and a re-run is
  bit-exact, whereas a hidden global RNG state makes runs order-dependent
  and silently unreproducible.

**Noise sources are separate functions, applied one at a time, not one
combined "add all noise" call.**
- Why: each has a different physical origin and the brief asks for each to
  be individually configurable (e.g. sweeping SNR should not require also
  touching dark current). Keeping them separate also means each can be
  unit-tested against its own known statistical signature (e.g. photon
  noise: `Var[N] = E[N]`) in isolation.

**The photon-noise statistical test uses `lambda = 500` and `n_trials =
5000` on a 10x10 grid — neither is a round-number guess.**
- `lambda = 500`: Poisson's *relative* noise is `1/sqrt(lambda)`, about 4.5%
  here — high enough to sit in the "approximately Gaussian, stable
  statistics" regime rather than the highly skewed regime of a small
  `lambda` (e.g. 5), and the same order of magnitude as the flux values used
  in `psf.py`'s own examples (300-1000), so the test exercises the noise
  model in a realistic range, not an arbitrary extreme.
- `n_trials = 5000`: derived backward from a target precision, not chosen
  freehand. We want the *pooled* variance estimator's standard error to land
  around 1 count (0.2% of `lambda`), so the test has real power to catch a
  genuine bug. `SE(pooled variance) = sqrt((lambda + 2*lambda^2) / n_total)`,
  and `n_total = n_trials * 100 pixels`. Setting `SE = 1` and solving:
  `n_total ~= lambda + 2*lambda^2 = 500,500`, so `n_trials ~= 5005 ~= 5000`.
- **Why pool across pixels at all, rather than check each pixel's own sample
  variance:** a *variance* estimator is itself noisy — its standard error
  involves the 4th statistical moment, not just `lambda` — so checking 100
  independent per-pixel variances against one fixed tolerance will
  occasionally flag one by chance alone. This happened during development:
  an earlier per-pixel version of this test failed on a run where nothing
  was actually wrong. Pooling to ~500,000 draws removes that as a source of
  test flakiness.
- **Why the tolerances (1.0 on the mean, 10.0 on the variance) are ~10x
  those standard errors, not 2-3x:** at 10 SE, a false failure from ordinary
  sampling luck is astronomically unlikely, so the test should never be
  flaky — while a real bug (e.g. a scaling error of even 10-20%) would move
  the empirical variance by 50-100+ counts, still 5-10x past the tolerance.
  Wide enough to be robust, not so wide it stops meaning anything.

**`sigma_read = 5.0` electrons — a deliberate mid-range choice, not an
arbitrary round number.**
- Real sensors span a wide range:

  | sensor type | typical σ_read |
  |---|---|
  | cheap phone sensor | 10–20 e⁻ |
  | typical CMOS (consumer) | 3–8 e⁻ |
  | **this project: 5.0 e⁻** | right here — solid mid-range |
  | scientific sCMOS | 1–2 e⁻ |
  | EMCCD (cooled) | < 1 e⁻ |

  5.0 e⁻ sits comfortably inside "typical consumer CMOS" — realistic and
  conservative, nothing exotic in either direction.
- **Why the exact value matters beyond realism — it sets a crossover point.**
  Read noise and photon (shot) noise add in quadrature:
  `sigma_total = sqrt(sigma_read^2 + lambda)`, giving two regimes: read-noise
  limited when `lambda << sigma_read^2 = 25`, shot-noise limited when
  `lambda >> 25`. So `sigma_read = 5.0` puts the crossover around **~25
  photons** — below that, read noise is the floor no matter how good the
  estimator is; above it, the sensor performs as well as photon statistics
  allow. This crossover number will matter directly once SNR sweeps are
  built, since it marks where the dominant noise source (and therefore which
  estimator wins) changes.

**Read noise test uses `n_trials = 200`.**
- Derived the same way as the photon-noise test's
  `n_trials`, just with the Gaussian version of the variance-estimator
  standard-error formula (`Var(sample variance) ~= 2*sigma^4 / n` for large
  `n`, vs Poisson's `(lambda + 2*lambda^2) / n`). Targeting `SE(pooled
  variance) ~= 1%` of `sigma_read^2 = 25` (i.e. `SE ~= 0.25`), with
  `n_total = n_trials * 100 pixels`: `0.25 = 25 * sqrt(2 / n_total)` solves
  to `n_total = 20,000`, so `n_trials = 200`.

**Dark current is drawn as its own independent Poisson noise, not folded
into the mean image before one combined draw.**
- Why this is valid, not just convenient: `Poisson(a) + Poisson(b)` is
  distributed as `Poisson(a + b)` for independent Poisson variables, so the
  two approaches give mathematically identical results. Verified directly
  in `test_dark_current_plus_photon_noise_variance_adds_by_poisson_additivity`
  rather than just asserted.
- Why we still keep it separate despite the equivalence: matches the
  established pattern (individually configurable, individually testable),
  and a later dark-frame calibration study needs dark current to be a
  distinct, addressable quantity.

**Dark-current test uses `dark_rate_e_per_s = 500`, `exposure_s = 1.0`
(mean_dark = 500) — deliberately the same lambda as the photon-noise test.**
- Why: dark current is the *same* Poisson mechanism as photon noise (random
  independent electron-generation events at a constant rate), just thermally
  rather than optically sourced, so the identical `n_trials = 5000`
  derivation for a tight pooled-variance standard error applies unchanged —
  no need to re-derive it. Note this combination is chosen for a clean,
  testable statistic, not because it's a realistic exposure/rate pair; real
  sub-millisecond exposures at room temperature make dark current far
  smaller than this in practice (see the sensor.py docstring).

**Hot pixel defect map is generated once and reused, never redrawn per
frame — unlike every other noise source.**
- Why: this is the one place the simulator models a fixed spatial property
  of a physical sensor (a manufacturing defect) rather than fresh per-frame
  randomness. Redrawing it every frame would be physically wrong — a
  defective pixel doesn't move.
- Consequence for the API: `generate_hot_pixel_mask` and `add_hot_pixels`
  are deliberately two separate functions (map generation vs. per-frame
  application) rather than one combined call, so the caller is forced to
  generate the mask once, outside the per-frame loop, instead of it being
  easy to accidentally call both together every frame.

**Hot pixel test fraction (`0.05`, i.e. 5%) is far higher than a real
sensor's defect rate (more like 1e-5 to 1e-3).**
- Why: chosen purely so the statistical test (checking the empirical
  fraction against the requested one) has enough hot pixels on a modest
  50x50 test grid to be meaningful, without needing an unrealistically large
  grid. Not meant to represent a realistic sensor.

**PRNU is applied as a multiplicative gain to photo-generated signal only
(spot + background), never to dark current or hot-pixel electrons.**
- Why: PRNU is specifically a property of the photon-to-electron conversion
  pathway (quantum efficiency). Dark current and hot-pixel electrons are
  generated directly in the silicon by thermal excitation — they never pass
  through that pathway, so there's nothing for a photon-conversion gain to
  multiply for them. Same fixed-map pattern as hot pixels (`generate_prnu_map`
  / `apply_prnu` kept as two separate functions), for the same reason: it's a
  manufacturing property of this sensor, not fresh per-frame randomness.

**PRNU statistics test uses `sigma_prnu = 0.02` (2%, realistic for a
consumer sensor); the position-dependent-bias test uses `sigma_prnu = 0.05`
(5%, deliberately larger).**
- Why the difference: the statistics test just needs to check the *map
  generation* matches the requested distribution, so a realistic value is
  the right one to test against. The position-dependent-bias test needs the
  *effect* to be clearly visible on a modest 25×25 window without requiring
  an enormous grid to resolve a tiny signal — it demonstrates the mechanism
  exists, not its realistic magnitude. This mirrors the earlier hot-pixel
  fraction choice: use a larger, test-friendly value when the point is
  proving a mechanism, a realistic one when checking a distribution.
- Why this matters more than a simple flux-calibration error: a *uniform*
  gain error would not move a centroid at all (scaling every pixel in a
  window by the same constant leaves a weighted average unchanged — verified
  directly in `test_uniform_gain_does_not_shift_the_centroid`). PRNU is
  non-uniform, so the specific ripple pattern sitting "under" the spot
  changes with the spot's exact sub-pixel position, making it a
  position-dependent bias in the same family as pixel locking (`psf.py`) —
  and, being a *fixed* pattern rather than fresh per-frame noise, it does
  not average away no matter how many frames are collected. Verified
  directly in `test_prnu_introduces_a_position_dependent_bias`: the same
  fixed gain map produces a different bias at x0=10.0 than at x0=10.5.

**Quantization error is treated as uniform noise with variance
`gain^2 / 12`, a classical approximation rather than an exact statement.**
- Why the approximation is valid here: it depends on the signal already
  having some other noise mixed in before rounding (which it does —
  everything upstream: photon, read, dark), so the rounding error's
  fractional part behaves like it's drawn from a uniform distribution
  rather than something value-dependent. Verified directly, not just
  asserted: `test_quantization_error_variance_matches_the_1_over_12_prediction`
  reconstructs 100,000 quantized values and checks the residual variance
  against `gain^2/12` to a tight tolerance.

**A black-level pedestal (`black_level_dn`) exists specifically to prevent
quantization from introducing a bias, not just noise.**
- Why: without a pedestal, negative electron excursions (routine with read
  noise on a dim pixel) clip to DN=0 — and clipping only ever removes the
  negative tail, which can only push the mean UP, never down. This is a
  real, sizeable, measured effect, not a theoretical nicety: in
  `test_black_level_pedestal_removes_the_clipping_bias`, a zero-mean
  Normal(0, 20) signal without a pedestal quantizes to a mean of +7.99
  (matching the predicted `sigma/sqrt(2*pi)` exactly) — with a sufficient
  pedestal, that bias disappears entirely (reconstructed mean back to ~0).
- This resolves something flagged earlier and left open: `add_read_noise`'s
  docstring notes it "can produce negative values... handled with a
  black-level pedestal... that pedestal is a separate, later concern." This
  is that concern, closed.

---

## Scene (`sptrack/scene.py`)

**Background gradient is modelled as a linear (planar) tilt, not a more
elaborate sky-radiance model.**
- Why: a full physical sky-radiance model is its own research project. A
  linear gradient is the simplest model that is still genuinely
  non-uniform, and captures the dominant leading-order effect — any smooth
  background variation looks approximately linear if you zoom into a field
  of view a few tens of pixels wide, the relevant scale here. Curvature
  would be the natural next refinement, not a first one.

**`gradient_frac` is defined relative to normalised coordinates ([-1, 1]
per axis), and its exact peak-to-peak meaning is angle-dependent.**
- For an axis-aligned gradient (angle = 0 or pi/2), `gradient_frac` gives
  *exactly* that fraction of `mean_level` peak-to-peak, edge to edge —
  verified directly in `test_axis_aligned_gradient_peak_to_peak_matches_gradient_frac`.
  For a diagonal angle, the true corner-to-corner peak-to-peak can be up to
  `sqrt(2)` times that, since the gradient is a sum of two axis projections.
  Stated explicitly here rather than left as a silent surprise, since
  "gradient_frac" alone doesn't fully pin down the peak-to-peak spread once
  the angle stops being axis-aligned.

**PLANNED, not yet built: robustness/stress-testing against the background
gradient will deliberately cover the worst-case angle (45°), not randomise
the gradient per trial.**
- Why not randomise: for the core bias-vs-SNR characterization (§2c of the
  brief), the gradient must stay fixed within a run — letting it vary
  randomly trial to trial would confound the thing actually being measured
  (is a bias shift from SNR changing, or from a lucky/unlucky gradient draw
  that trial?). For a dedicated robustness experiment, full randomisation
  also blurs results into one averaged number and can hide exactly which
  condition is worst.
- Why 45° specifically, and why that's *better* than random: we don't have
  to guess which angle is adversarial — the Cauchy-Schwarz derivation above
  already proves 45° is mathematically the worst-case orientation for
  peak-to-peak spread (`|cos(theta)| + |sin(theta)|` is maximised there). A
  stress test built around a derived worst case is more defensible than one
  built around random sampling that might simply miss it. Plan: sweep a
  small set of representative angles (0°, 45°, 90°) rather than either a
  single fixed angle or full randomisation, once estimators exist to test
  against.

**Background is treated as a *mean* contribution, summed with the spot's
mean image and run through one shared `add_photon_noise` call — not drawn
as its own independent noise source, unlike dark current.**
- Why the different treatment from dark current: dark current is kept
  separate specifically so it can be individually addressed later (a real
  system can capture a shutter-closed "dark frame" for calibration).
  Background has no equivalent — there's no way to capture a
  "background-only, spot-off" reference frame in a real deployment, so
  there's no calibration reason to keep it distinct. It belongs with the
  signal from the moment it exists.
- This module lives separately from `sensor.py` for a related but distinct
  reason: it isn't a sensor *imperfection* at all, it's real light entering
  the system, the same conceptual category as the tracked spot itself
  (`psf.py`). "What's in the scene" and "how the sensor mangles it" are kept
  as separate concerns.

---

## SNR control (`sptrack/snr.py`)

**SNR is defined as PEAK-PIXEL SNR, not total-flux SNR.**
- Why: a windowed/weighted position estimator concentrates most of its
  weight on the brightest few pixels near the peak, so the peak pixel's own
  signal-to-noise ratio is what actually limits achievable position
  precision — far more directly than a total-flux number that says nothing
  about how that flux is distributed. Also the conventional definition in
  point-source imaging generally (astronomy, spot tracking).

**Peak-pixel fraction is computed assuming the spot is exactly centred on
a pixel — the canonical, best-case value, not the realised value for any
specific rendered frame.**
- Why: the actual fraction of flux landing in the brightest pixel is always
  slightly lower once the spot sits off-centre (any other sub-pixel
  position spreads more flux into neighbouring pixels). Using the on-centre
  value gives one consistent, reproducible number to define and sweep SNR
  against, understood as a nominal reference rather than a claim about any
  one frame's exact realised SNR.
- Verified `peak_pixel_fraction` behaves sensibly: it strictly decreases as
  the spot widens (`test_peak_pixel_fraction_decreases_as_spot_widens`) —
  a wider spot spreads the same total flux more thinly.

**`snr_to_flux` solves a quadratic rather than using an approximation,
because SNR appears both linearly (numerator) and inside a square root
(denominator, via the signal's own photon-noise contribution).**
- The noise budget is `Var[peak] = P + C`, where `P` is the peak pixel's
  own signal and `C` is every OTHER noise term (background, dark, read,
  quantization — the same `C` used throughout `sensor.py`/`scene.py`).
  `SNR = P / sqrt(P + C)` rearranges to `P^2 - SNR^2*P - SNR^2*C = 0`,
  solved via the quadratic formula, positive root only (P must be real).
- Verified by direct round-trip, not just algebra: `snr_to_flux` followed
  by `flux_to_snr` recovers the original target SNR to 1e-6 relative
  tolerance, across 4 SNR values x 3 noise conditions
  (`test_snr_to_flux_round_trips_through_flux_to_snr`).
- Sanity-checked against the earlier relative-noise derivation
  (`sensor.py`'s CV section): with every non-photon noise term made
  negligible, `flux_to_snr` reduces to `sqrt(peak)` — the same
  `SNR = sqrt(lambda)` result derived from first principles when photon
  noise was first built (`test_flux_to_snr_reduces_to_sqrt_peak_when_photon_noise_dominates`).

---

*(This document will grow as each new part of the simulator — remaining
noise sources, SNR control, dynamic tracking, etc. — introduces its own
assumptions.)*
