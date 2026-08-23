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

**The laser-industry "1/e² diameter" spec is converted to a Gaussian sigma
via `sigma = diameter_1e2 / 4`, derived from matching the laser-optics
irradiance convention to the statistical-Gaussian convention.**
- Why not just accept "sigma=1.75" as a given constant: the brief states the
  spot size as ~7 px diameter *at 1/e²* — a specific laser-optics
  convention, `I(r) = I0*exp(-2r²/w²)`, not the statistical Gaussian
  convention this project's code otherwise uses, `I(r) = I0*exp(-r²/2σ²)`.
  Matching the two exponents gives `σ = w/2`, and since `w` (the laser
  convention's "radius") is half the *diameter*, `σ = diameter/4`. With the
  brief's 7 px: `σ = 1.75` — confirming, rather than assuming, the constant
  used throughout every earlier test in this project.
- Verified against its own physical definition, not just algebra: for every
  diameter tested (not only 7 px), the continuous Gaussian intensity at
  `r = diameter/2` sits exactly on `1/e²`
  (`test_diameter_1e2_to_sigma_satisfies_its_own_definition`).

**Spot-size variation (`sample_true_sigma`) is a FIXED per-optical-unit
property, drawn once, not fresh per-frame noise.**
- Why: real manufacturing/assembly tolerance is a property of one physical
  lens/optics assembly — it doesn't refocus itself between frames. Same
  fixed-once pattern as `generate_hot_pixel_mask` and `generate_prnu_map`.
- Why this matters beyond realism: an estimator that assumes a fixed
  template sigma is implicitly assuming the design value is the true value.
  This function exists specifically so a later experiment can inject PSF
  model mismatch deliberately (a real-world condition from §4 of the brief)
  and measure its cost, rather than only ever testing estimators against
  PSFs they were built to expect.
- The floor (`0.1 * nominal_sigma`) is a physical guard against non-positive
  sigma from an extreme tolerance draw — and the test confirms the floor
  actually engages under an extreme `tolerance_frac=5.0`, not merely that
  the clipping code exists (`test_sample_true_sigma_never_returns_non_positive_even_at_extreme_tolerance`).

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

## Full-chain assembly (`sptrack/simulate.py`)

**PRNU is applied to the spot + background BEFORE photon noise, dark
current, and hot pixels are added — not at the very end of the chain.**
- Why: this is the order the physics actually requires, not an arbitrary
  pipeline choice. PRNU is a property of the photon-to-electron conversion
  pathway, so it can only apply to light that passes through that pathway —
  the spot and background. Dark current and hot-pixel electrons are
  generated directly in the silicon and never pass through it, so applying
  PRNU after adding them would incorrectly scale noise sources it has no
  physical claim on (see `sensor.py`'s PRNU section for the full reasoning;
  this module is where that reasoning actually gets enforced in the
  pipeline order).

**`Simulator` defaults every noise parameter to a modest, non-zero value —
nothing defaults to "off."**
- Why: a bare `Simulator(shape=(25, 25))` should still produce a realistic
  frame. If any source silently defaulted to zero, it would be easy to
  accidentally characterise an estimator against a simulator quieter than
  reality without noticing — the opposite of the brief's "results that are
  quantified and compared" bar.

**The three FIXED per-unit properties (true sigma, hot-pixel mask, PRNU
map) are generated once in `__init__`, not regenerated per `render()`
call.**
- Why: this is a straightforward continuation of a pattern already
  established and tested for each piece individually — a manufacturing
  defect or a lens's true focus doesn't reset itself between frames.
  Verified directly that `Simulator` doesn't violate this
  (`test_fixed_unit_properties_persist_across_renders`): the same mask, map,
  and sigma survive multiple `render()` calls unchanged.

**The end-to-end validation isolates the chain by setting `flux=0`,
`hot_fraction=0`, and `prnu_sigma=0`, rather than trying to predict the
statistics of a full frame with everything enabled.**
- Why: with the spot removed, only background + dark current + read noise +
  quantization remain — a combination this project can still predict
  exactly from formulas already derived and individually verified
  (background/dark: Poisson, `Var=mean`; read noise: fixed `sigma_read^2`;
  quantization: `gain^2/12`). Disabling the two FIXED per-unit effects
  removes unpredictable per-pixel structure that would need its own
  separate accounting rather than adding real coverage. This is the
  narrowest slice of the chain that still proves the pieces compose
  correctly in the right order — a full frame with a spot, PRNU, and hot
  pixels enabled would be a good visual demonstration (and one exists,
  `docs/sanity_check_simulator.png`) but a poor *statistical* test, since
  there'd be no simple closed-form prediction to check it against.
- The variance tolerance in `test_full_chain_statistics_match_the_combined_noise_budget`
  is a pragmatic 5% relative bound, not a derived standard error, stated as
  such in the test's own comment — the combined distribution mixes Poisson,
  Gaussian, and uniform sources, and deriving an exact standard error for
  that mixture's sample variance is not worth the complexity here.

**The worked numbers behind that test, spelled out step by step (why these
specific values, not just what they are).**
- `background_e=1000`, `dark_rate_e_per_s=5000`, `exposure_s=0.01` (giving
  `mean_dark = 50`) are clean, round numbers chosen purely to make every
  downstream number easy to check by hand — not claimed as realistic
  operating conditions (unlike the `Simulator` class defaults, which *are*
  chosen for realism; this test is about verifying arithmetic, not
  simulating a plausible deployment).
- `prnu_sigma=0` was checked to behave sensibly before relying on it: it
  returns an array of exactly `1.0` everywhere, no edge-case randomness or
  degenerate behaviour from a zero-width Normal draw.
- No clipping risk, checked rather than assumed: with `sigma_read_e=5.0`,
  `gain_e_per_dn=10.0`, `bit_depth=16`, the baseline sits around 105 DN with
  negligible noise relative to that baseline — nowhere near either the
  bottom (0) or top (65535) of the ADC's range, so saturation and the
  black-level clipping bias (sensor.py's quantization section) are both
  structurally impossible here, not just unlikely.
- Expected mean, in electrons: `background_e + mean_dark = 1000 + 50 = 1050`.
- Expected variance, in electrons², built up term by term: because
  background shot noise and dark-current shot noise are drawn as two
  *separate* independent Poisson processes in the pipeline (see the Poisson
  additivity reasoning in `sensor.py`'s dark-current section), their
  variances add directly rather than needing to be combined first:
  `1000` (background shot noise) `+ 50` (dark shot noise) `+ 25`
  (`sigma_read_e² = 5.0²`) `+ 8.33` (`gain²/12 = 100/12`, the quantization
  term derived in `sensor.py`) `≈ 1083.33`, giving an expected standard
  deviation of `sqrt(1083.33) ≈ 32.9` electrons.
- In DN terms: at 10 electrons/DN, that's `32.9 / 10 ≈ 3.3` DN of noise —
  fine enough resolution relative to the 105 DN baseline for a clean
  statistical comparison, not so coarse that quantization itself would
  smear out the signal being measured.
- Working out `n_trials` from a target precision, the same method used
  throughout this project (e.g. the photon-noise test's `lambda=500,
  n_trials=5000` derivation): pooling over a 100-pixel window, standard
  error scales as `std / sqrt(n_trials * n_pixels)`. Targeting a tight
  `SE(mean) ≈ 0.5` electrons: `n_total ≈ (32.9 / 0.5)² ≈ 4329`, so
  `n_trials ≈ 44` with 100 pixels. `n_trials = 200` was used instead for a
  comfortable safety margin, giving `SE(mean) ≈ 32.9 / sqrt(20000) ≈ 0.23`
  electrons — tighter than strictly required, at negligible extra runtime
  cost.
- The variance check does not get an equivalently derived standard error,
  and that gap is deliberate, not an oversight: this noise is a *mixture*
  of Poisson (background, dark), Gaussian (read), and uniform
  (quantization) terms, not one clean distribution with a textbook
  variance-of-variance formula. Rather than force a derivation that
  wouldn't really be trustworthy, a generous 5% relative tolerance is used
  instead, with that approximation stated explicitly in the test's own
  comment.

---

## Estimators (`sptrack/estimators/`)

**How the Gaussian fit's implementation was scoped, before any code was
written — the overall shape of the design, not yet the individual choices
detailed below.**
- The brief asks for "a 2D Gaussian least-squares or maximum-likelihood
  fit." The approach settled on going in: a Poisson-weighted Gauss-Newton
  solver, with Levenberg-Marquardt damping for stability, model-based
  variance for the weighting (not the noisy data — see below), computing a
  gradient and an approximate Hessian each iteration, and trust-region-
  style step capping alongside adaptive damping, iterated to convergence.
  This is functionally the maximum-likelihood option the brief offers, via
  the standard route of turning a Poisson likelihood into a weighted
  least-squares problem — not literally scipy's Poisson MLE machinery, but
  the same statistical target.
- Given the time available, a few choices were made deliberately for
  simplicity over generality, stated here rather than left implicit: a
  FIXED 20-iteration cap (not an open-ended convergence search), a
  straightforward step-size convergence check (not a more elaborate
  stopping criterion), and adaptive damping driven only by whether chi²
  improved. These are reasonable, well-tested building blocks for a
  4-parameter, well-conditioned, separable model — not corners cut on
  correctness, but scope kept to what this problem actually needs rather
  than building a general-purpose optimiser.
- Initialization is seeded from the centroid estimator's own output — not
  just its position, but its flux and background estimates too, giving the
  fit a fully-formed starting guess for all four parameters at once. This
  mirrors how real photometry pipelines commonly operate: get a cheap
  estimate first (the centroid), then refine it with something more
  expensive (the fit) — rather than starting the expensive method from
  nothing.
- The docstring's job, deliberately: justify why Poisson-weighted MLE
  fitting is preferable to plain unweighted least squares for this data,
  not just describe the mechanics of how the solver works. The "why" (each
  pixel weighted by its own noise, approaching the Cramér–Rao bound where
  ordinary least squares cannot) is the part worth defending; the Gauss-
  Newton/Levenberg-Marquardt machinery itself is standard numerical
  optimisation, not a novel contribution.
- The individual implementation choices this plan led to — the analytic
  Jacobian, the model-based (not data-based) weighting recomputed every
  iteration, the specific LM damping and trust-region behaviour, and the
  centroid-seeded initialization — are each justified on their own below,
  with the tests that verify them.

**Window and background-estimation utilities (`base.py`) are shared by
every estimator, rather than each estimator implementing its own.**
- Why: this matters for fairness once estimators are compared later (§2c
  Characterization) — if one estimator used a larger window or a better
  background estimate than another, a measured precision difference would
  be measuring THAT difference, not a real difference between the methods.

**Background is estimated from the window's BORDER only (median of the
outer 2 pixels), never the whole window.**
- Why not the whole window: the spot sits near the centre, so including
  centre pixels in a background estimate would let the spot's own signal
  drag the estimate upward, causing systematic over-subtraction.
- Why median, not mean: robust to a single outlier (a hot pixel landing on
  the border, or an unlucky noise spike) — a mean would let one bad pixel's
  error propagate into every pixel's background-subtracted value.

**FINDING (not assumed, discovered while testing): even with zero sensor
noise, `border_median_background` is not perfectly exact — the PSF's
Gaussian tail still reaches the border pixels at ordinary window sizes,
nudging the estimate very slightly above the true flat pedestal.**
- What happened: two tests were first written assuming a background
  estimate and a `clip_negative`/no-clip comparison would be *exact* on a
  noise-free image (tolerances of `1e-9` and `1e-6`). Both failed. Checked
  directly rather than loosening blindly: at `half_width=7` (~4 sigma) on
  `sigma=1.75`, border pixel values ranged up to ~20.07 against a true
  pedestal of 20.0 — small, but enough to pull the median background
  estimate to ~20.012. That in turn pushed ~50 of 225 pixels fractionally
  negative (down to about −0.012) with **no sensor noise involved at all**.
- Why this happens: a Gaussian's tail is never exactly zero at any finite
  distance — "4 sigma" is a *practically* negligible amount of flux, not a
  hard cutoff, and a large enough window will always pick up some residual
  trace of it at the border.
- What changed as a result: the affected tests' tolerances were corrected
  to reflect this real, understood, small effect (`abs=1e-3` to `1e-2`
  scale) instead of an idealised exact match, with the mechanism spelled
  out in each test's comment rather than a bare number.
- A separate test (`test_centroid_recovers_true_position_on_a_clean_flat_background_image`,
  earlier draft) additionally caught a real test-design flaw this same
  session: an *asymmetrically placed* window (centred on a stale prior, not
  near the spot's true position) left only a 2.7-sigma margin on the near
  edge — well inside where truncation bias is measurable (~0.009 px in that
  case). Fixed by placing the window near the true position, as any real
  tracking loop would (using the previous frame's estimate as the prior).
  Not an estimator bug either time — both were test-setup errors, caught by
  checking the actual numbers rather than trusting an assumption about "far
  enough."

**The Gaussian fit uses an ANALYTIC Jacobian (`psf.py::pixel_response_1d_with_derivative`),
not finite differences.**
- Why: the closed form isn't hard to derive (chain rule through `erf`'s own
  derivative), costs one extra `exp` evaluation instead of a second full
  model evaluation per parameter per iteration, and is exact rather than an
  approximation with its own step-size tuning problem.
- Verified against finite differences directly, not just derived on paper:
  `test_analytic_derivative_matches_finite_differences` checks agreement to
  `1e-6` across several centre positions; a separate sign-check test
  confirms the direction (a pixel to the right of the spot gains brightness
  as the centre moves right) matches the reasoning stated in the docstring.

**The fit weights each pixel by the inverse of its own predicted variance
(Poisson-weighted least squares), recomputed every iteration from the
CURRENT model, not the noisy data.**
- Why weighted at all: ordinary least squares treats every pixel as
  equally noisy, which is already known to be wrong from the very first
  noise source built in this project (photon noise variance scales with
  signal) — weighting by predicted variance is what lets this method
  approach the Cramér–Rao bound where the centroid cannot.
- Why the MODEL's prediction, not the DATA, sets the variance: using the
  observed (noisy) value to weight itself creates a feedback loop — a
  pixel that happened to read low also gets treated as if it were expected
  to be low, biasing the fit. The model's current prediction has no such
  circularity.
- Why recomputed every iteration rather than once from the initial
  (centroid) guess: a single fixed set of weights would be a cheaper
  one-step approximation, not the true Poisson MLE fixed point. Since the
  model is recomputed every iteration anyway (needed for the residual),
  recomputing the weights from it costs nothing extra.

**Levenberg-Marquardt damping, not plain Gauss-Newton, and a hard
trust-region cap of 1 px per step regardless of what the damped solve
suggests.**
- Why LM: Gauss-Newton can overshoot or diverge when the current guess is
  still far from the optimum — most likely on the very first iteration,
  before the fit has had a chance to refine anything. LM blends toward
  (safer, slower) gradient descent exactly when a step fails to improve
  chi², and relaxes back toward full Gauss-Newton once it's working.
- Why an additional hard cap on top of that: the local linearisation
  Gauss-Newton relies on is least trustworthy precisely when it wants to
  take a large step — capping at 1 px prevents a single bad iteration from
  ejecting the fit somewhere the model no longer resembles the data at all,
  regardless of what the (possibly ill-conditioned, early-iteration)
  damped solve computes.

**The fit is seeded from the centroid estimator's own output (both
position AND flux/background), not an arbitrary or user-supplied guess.**
- Why: this is a local optimiser, not a global search — it refines a
  starting point, it doesn't find one. The centroid is cheap, already
  built, and normally accurate to a fraction of a pixel, which is a
  realistic, honest starting point (what a real system would actually have
  available), not an artificially generous one.

**sigma is a fixed input to the fit, not a 5th free parameter.**
- Why: mirrors real systems, where PSF width is typically calibrated
  separately rather than re-estimated every single frame, and keeps this
  estimator directly comparable to the centroid (which also assumes a
  fixed shape/window rather than fitting one) — a fair comparison needs
  both methods given the same information, not one given strictly more.
  Fitting sigma too is a natural later extension once PSF-mismatch
  robustness (§4 of the brief) is being studied deliberately.

**Head-to-head precision comparison against the centroid uses the SAME
random frames for both estimators (a paired comparison), not independently
drawn samples for each.**
- Why: removes sampling luck as a confound — any measured difference in
  spread is guaranteed to come from the estimators themselves, not from one
  method happening to see an easier or harder set of noise realisations.
  Result at flux=6000 e-, background=30 e- (`sigma_read_e=5.0`):
  Gaussian fit measured 22% tighter (lower std) than the centroid on 500
  paired trials — the efficiency argument in `gaussian_fit.py`'s docstring,
  actually measured rather than only asserted. A full bias/std-vs-SNR
  sweep, compared against the Cramér–Rao bound, is Characterization's job
  (§2c) — this is a first, informal look at the same question.

---

## Characterization (`sptrack/crlb.py`)

**The CRLB is built from the SAME Jacobian and variance model as
`gaussian_fit.py`'s Gauss-Newton solver, not derived independently.**
- Why: three things need to agree with each other for a characterization
  study to mean anything — the simulator that renders frames, the
  estimator that fits them, and the bound that says how well any estimator
  could do. If the bound came from a different model than the fit uses,
  "attains the bound" or "falls short of the bound" would both be
  comparisons against a fiction. Reusing the exact Jacobian
  (`psf.pixel_response_1d_with_derivative`) and the exact
  Gaussian-approximated-Poisson variance (`mu + read_var`) makes this
  structural, not just consistent by coincidence — the Gauss-Newton
  Hessian a fit computes at every iteration already IS an approximation to
  the Fisher information; `position_crlb` just evaluates that same
  quantity at the true parameters and inverts it.

**FINDING (caught during testing, not assumed): the pixel-integrated CRLB
sits measurably above the classical continuous-sampling formula
(`sigma/sqrt(N)`) at this project's sigma=1.75, and that gap is real
physics, not a bug.**
- What happened: a test asserting agreement with the classical formula to
  1% failed at a measured 1.35% gap. Rather than loosen the tolerance
  blindly, the mechanism was checked: sweeping `sigma` at fixed (huge)
  flux showed the gap shrinks monotonically as `sigma` grows relative to
  the fixed 1-pixel sampling pitch — 15.1% at `sigma=0.5`, 1.35% at
  `sigma=1.75`, 0.04% at `sigma=10`. That is exactly the signature of a
  genuine pixelation effect converging to the continuous-sampling limit as
  relative sampling gets finer, not a coding error — and it matches known
  results in the localisation-microscopy literature (e.g. Mortensen et al.
  2010's pixel-size correction term).
- What changed as a result: the test's tolerance was corrected to 2%
  (comfortably above the verified ~1.35% gap) with the sigma-sweep
  evidence recorded in the test's own comment, and the same finding was
  written into `crlb.py`'s docstring directly — this is a property of the
  bound worth knowing about when using it, not just a note about how one
  test's number was chosen.

**The Gaussian fit's efficiency (CRLB / empirical std) is checked to land
within 25% of 1.0 over 400 Monte Carlo trials, not asserted as exact
attainment.**
- Why not tighter: asymptotic efficiency is a large-sample-size /
  high-SNR *limiting* property, and empirical std from 400 trials carries
  its own sampling noise — demanding near-exact agreement would make the
  test itself unreliable (prone to failing on ordinary statistical
  fluctuation) rather than meaningfully checking the right thing. 25%
  margin is wide enough to be robust while still catching a genuine
  implementation bug (a real error in the Jacobian or weighting scheme
  would show efficiency far outside this band, not just outside it by a
  little).
- What this result actually shows, visualized in
  `docs/sanity_check_crlb.png`: across a 100x flux sweep, the Gaussian
  fit's measured std tracks the CRLB curve closely at every flux tested,
  while the centroid's sits visibly and consistently above it — the
  centroid's gap from the floor does not close as flux increases, meaning
  it is a genuine *efficiency* loss (information discarded by the
  method), not extra noise that would average away with a brighter spot.

---

## Bugs found while building the SNR characterization experiment (§2c)

**Two real bugs in `gaussian_fit_estimate`, both invisible to all 5 tests
that existed for it at the time, caught only when the full SNR sweep
produced impossible results (both estimators reporting bit-identical
output at every SNR point).**

- **Bug 1 — convergence checked on the wrong step.** The `abs(step) <
  tol_px` check ran unconditionally, whether or not that step was about to
  be accepted. When the centroid-seeded starting guess is already close to
  the true optimum (likely at high SNR specifically), the very first
  *proposed* step can be tiny even while making chi² worse — and that tiny
  rejected step was enough to declare "converged," silently returning the
  untouched centroid seed disguised as a fit result. `est.ok` reported
  `True` throughout; nothing about the return value signalled that no
  refinement had actually happened.
- **Bug 2 — the trial step's chi² used a different variance model than
  the comparison baseline.** `new_chi2` computed variance as
  `max(new_mu, 1e-6)`, omitting `+ read_var_e2`, while `old_chi2` correctly
  included it. This meant the two chi² values could never be judged equal
  even when the proposed step shrank to exactly zero under heavy damping —
  an unwinnable comparison that rejected every single iteration, all the
  way to `max_iter`, regardless of how much damping was applied.
- **Why neither bug was caught by the 5 tests written for `gaussian_fit.py`
  alone**: checked directly rather than left a mystery. Bug 1's failure
  mode is SNR-dependent — it only manifests when the very first proposed
  step is already tiny, which is far more likely at the specific high-SNR
  flux level (`50000`) used to debug it than at the moderate flux
  (`6000`) used in the head-to-head precision test. At that lower flux,
  the initial step was large enough to escape the trap and iterate
  genuinely, so `test_fit_is_more_precise_than_the_centroid_at_high_snr`
  measured a real (if possibly understated) effect and passed. Bug 2 was
  present throughout, but with Bug 1 masking it in many trials (returning
  the already-reasonable centroid seed rather than looping to
  `max_iter`), its symptom (silent non-improvement) rarely became visible
  enough to fail a bias/precision assertion — the Monte Carlo bias test
  passed because the wrongly-returned "fit" values were just centroid
  values, and the centroid is *also* independently proven unbiased, so
  averaging them still looked unbiased.
- **What actually surfaced both bugs**: not a unit test, but the full SNR
  sweep experiment producing a result that was qualitatively impossible —
  `centroid_eff` and `fit_eff` identical to 2 decimal places at every one
  of 10 SNR points, which cannot happen from independent noisy Monte Carlo
  samples by chance. That implausibility was the signal to debug, not a
  failed assertion.
- **The fix, verified two ways**: (1) a hand-traced iteration log (printing
  `old_chi2`/`new_chi2`/accept-reject at every step) that showed the exact
  mechanism of both bugs before touching any code, and (2) a new
  regression test, `test_fit_genuinely_iterates_rather_than_silently_returning_the_seed`,
  that checks directly — at the specific high-SNR condition that exposed
  Bug 1 — that the fit's output is never bit-identical to the raw centroid
  seed across 50 trials. This is the test that should have existed from
  the start: not "is the answer plausible" (which both bugs satisfied) but
  "did the optimiser actually run."
- **The corrected result**: Gaussian fit mean efficiency (CRLB / measured
  std) across the full SNR sweep is `0.95`, centroid is `0.63` — the fit
  attains the theoretical floor at essentially every SNR tested; the
  centroid does not, worst in the middle of the range (`0.41` at SNR=8.3).
  The centroid also carries a large low-SNR bias (`-235` millipixels at
  SNR=3) that the fit does not. Full curves in
  `figures/exp01_snr_characterization.png`, raw numbers in
  `results/exp01_snr_characterization.json`.

---

## Third estimator: matched filter (`sptrack/estimators/matched_filter.py`)

**Not part of the original plan — added after being asked directly why it
wasn't considered.**
- The brief only requires two estimators ("windowed centroid... and a 2D
  Gaussian fit"), both of which were already built and characterised. A
  matched-filter/correlation estimator is a legitimate, well-established
  third technique, and it simply hadn't been raised as a "go further"
  candidate before being asked about directly. Re-derived and re-tested
  from scratch here (own docstring reasoning, own log-parabola derivation,
  own tests), not copied from any prior reference implementation.

**Built for a different reason than accuracy: real-time hardware
friendliness.**
- Why this matters as a genuinely separate axis: the centroid and the
  Gaussian fit differ in accuracy, but neither is shaped for dedicated
  hardware — a correlation IS a convolution, the one operation SIMD/DSP/
  FPGA are built to do with fixed, predictable timing, unlike an iterative
  fit whose actual runtime depends on how many iterations the data
  happens to need. This is the real motivation, directly feeding into
  §2d's Real-time comparison next.

**The template is a simple SAMPLED Gaussian, not the pixel-integrated
response `psf.py` uses everywhere else.**
- Why the inconsistency is deliberate, not an oversight: pixel-integration
  exactness mattered in `psf.py` because it was the RENDERING model — any
  error there shows up directly as position-dependent bias in the
  simulated data itself. Here the kernel is a detection filter, not a
  generative model; a slightly-imperfect kernel still gives a valid (if
  marginally sub-optimal) matched filter, and the sub-pixel interpolation
  step is what actually does the precision work.

**Log-parabola interpolation is the default, not the plain parabola —
verified as EXACT for noiseless Gaussian samples, not just theoretically
motivated.**
- The mechanism: correlating a Gaussian with a Gaussian produces another
  Gaussian (not a parabola), so fitting a plain parabola to 3 samples of a
  true Gaussian peak is only exact when the peak sits precisely on a
  sample or precisely halfway between two — the same "pixel locking"
  S-curve bias already characterised for the centroid, arriving through a
  completely different mechanism (curve-shape mismatch, not
  truncation/weighting). Since `log(Gaussian)` IS exactly a parabola,
  fitting the parabola to the LOG of the samples instead recovers the true
  peak exactly.
- Verified two ways, not just derived: (1) `test_log_parabola_offset_is_exact_for_noiseless_gaussian_samples`
  checks agreement to `1e-9` across 7 sub-pixel offsets against raw
  (non-pixel-integrated) Gaussian samples — deliberately raw, to test the
  interpolation formula's own mathematical claim independent of the
  separate pixel-integration question; (2) a full-pipeline deterministic
  comparison at the analytically worst-case quarter-sample offset shows
  log-parabola error ~8e-7 px against plain-parabola error ~0.0075 px —
  about 10,000x worse for the plain version, on identical data.

**FINDING, caught by testing, corrected the same way as similar findings
elsewhere in this project: an early Monte Carlo comparison test came out
backwards, and the fix was to change the TEST's design, not the
conclusion.**
- What happened: a first attempt at proving log-parabola beats the plain
  parabola compared the MEAN of 300 *noisy* trials at one fixed sub-pixel
  offset. It failed — the plain parabola's measured mean bias came out
  *smaller* than the log-parabola's. This did not contradict the
  deterministic exactness proof above; it meant 300 noisy samples of a
  small systematic effect, at a single offset, is a statistically
  underpowered way to detect it — the comparison was dominated by Monte
  Carlo sampling noise on the mean, not by the real underlying bias curve.
  Fixed by comparing on a deterministic, noise-free image instead (the
  same style already used to prove the effect analytically), where the
  true difference is large and unambiguous rather than buried in noise.
- Also caught in the same debugging pass: an initial version of the
  "correlation peak widens by sqrt(2)" test built its test signal on a
  fine PHYSICAL grid (spacing 0.2) while `gaussian_kernel_1d` and
  `correlate1d` operate purely in ARRAY-INDEX units (spacing 1) — silently
  making the effective kernel 5x too narrow relative to the signal. Caught
  because the measured output width came out suspiciously close to the
  input sigma itself rather than `sigma*sqrt(2)`; fixed by keeping the
  test on unit spacing throughout, matching how the real estimator
  actually uses these functions.

**A genuine, stated design tension: the template width that's optimal for
DETECTING a spot is not the width that's optimal for LOCALISING it once
found.**
- The matched-filter theorem says the best template for detection matches
  the signal's own width exactly. But correlating two equal-width
  Gaussians produces a peak `sqrt(2)` WIDER than either — verified
  directly (`test_correlation_peak_widens_by_sqrt2_when_template_matches_signal`,
  agreement to 1%) — and a wider peak is flatter, carrying less
  information about exactly where it sits. This shows up concretely in
  the full characterization sweep: the matched filter's efficiency
  (CRLB/measured std) *falls* as SNR rises (0.96 at SNR=3 down to 0.68 at
  SNR=300) rather than staying flat or improving — once noise stops being
  the limiting factor, the width penalty from matching the detection-
  optimal template becomes the dominant cost, exactly as the tension
  predicts. `template_sigma_scale` exposes this as a tunable knob rather
  than hiding it behind one hard-coded choice.

---

## Real-time characterization (2d)

**Why percentiles, not the mean, decide "fits the budget."** A 1 kHz loop
needs a new estimate every 1 ms — that is a requirement on EVERY frame,
not the average frame. A method that is fast 999 times out of 1000 but
occasionally takes 3 ms still misses a deadline once a second. So the
number that determines whether a method "fits" is a high percentile
(p99, here) of its cost, not the mean — the mean is what a benchmark blog
reports, the tail is what makes a real-time loop miss a frame.

**Why these are honestly labelled as Python wall-clock numbers, not a
production-system claim.** `time.perf_counter()` around each estimator
call measures this project's actual Python implementations under
CPython's interpreter overhead — real numbers, not modelled ones. But
Python's per-call overhead is roughly constant regardless of how much
"real work" a call does, which flatters the expensive method (the fit,
where iteration work is a bigger share of the total) and unfairly
penalises the cheap one (the centroid) relative to what a compiled
implementation would show. The three methods are still validly compared
against EACH OTHER under identical conditions, and the comparison to the
brief's literal 1 kHz/1 ms budget is answered honestly for what it
actually is: a Python number, not a C++/FPGA one.

**Why frames are pre-rendered before timing starts.** The timed region is
only the estimator's own work; timing frame generation together with
estimation would measure the simulator's cost too, which a real system
never pays per-estimate (the sensor delivers a frame already exists).

**Why a warm-up pass runs untimed first.** A handful of untimed calls per
method absorb any first-call cost (e.g. array allocation patterns,
CPU cache warm-up) so the timed measurements reflect steady-state
per-frame cost, not one-time setup.

**FINDING: the Gaussian fit's worst observed frame exceeded the 1 ms
budget, even though its p99 did not.** Two separate runs of
`experiments/exp02_realtime.py` (1000 timed frames each, SNR=50) gave the
fit a p99 of ~884-886 us (under budget) but a max of 1010-1397 us (over
budget) — a genuine measured tail event, not a hypothetical one. This
makes sense structurally: the fit's cost is `max_iter=20` capped
Levenberg-Marquardt, so its per-frame cost is data-dependent (how many
iterations a given noisy frame needs to converge or get capped) rather
than fixed-shape like the centroid's or matched filter's single-pass
work. `max_iter=20` bounds the WORST CASE in principle, but 20 iterations
on an unlucky frame still costs more than the 1 ms budget allows — the
cap prevents unbounded cost, not cost that exceeds the specific real-time
budget in play here. This is exactly the practical, measured version of
the abstract "detection vs. localisation" and "fixed-cost vs.
variable-cost" arguments already made when the matched filter was added
(§5) — now backed by a real timing number rather than only a structural
argument.

**Why the stated tradeoff is about predictability, not the median.** At
the median, cost is a non-issue for any of the three methods — all are
well under 1 ms, so raw speed does not argue against using the most
accurate method (the fit). The real tradeoff, once the tail is measured
honestly, is between the fit's superior accuracy (§2c: efficiency 0.95)
and its lack of a hard cost ceiling, versus the matched filter's slightly
lower accuracy (0.84) bought with a fixed, correlation-shaped cost that
structurally cannot blow the budget the way an iterative solver can. For
a loop that must never miss a deadline, that structural guarantee matters
more than the typical-case number.

---

## Dynamic tracking: ground-truth trajectory (§3, part 1)

Full reasoning lives in `sptrack/trajectory.py`'s module docstring (each
of the three components tied to a specific physical mechanism, with every
default parameter's value justified); this section is a short summary.

**Why three components with deliberately different spectral shapes.** The
brief asks for "slow drift + random jitter + one periodic disturbance."
These map onto three real, distinct physical processes — thermal
creep/mechanical settling (drift), platform shake from many independent
micro-sources (jitter), and a dominant rotating-machinery tone like a fan
or motor (the disturbance) — and, critically, each has a genuinely
different frequency signature: drift is a random walk (power concentrated
at the lowest frequencies, 1/f^2), jitter is modelled as white noise (flat
spectrum, justified the same way read noise's Gaussianity was — many
independent short-correlation-time sources summing via the CLT), and the
disturbance is a single sinusoid (one spectral spike). That separability
is not decorative — it is the entire mechanism that will let a later
disturbance-detection step distinguish "the injected tone" from "the rest
of the motion" at all, and it is verified numerically
(`tests/test_trajectory.py`), not just asserted.

**Why a random walk for drift, not a low-pass filter.** A low-pass filter
would be an engineering choice about smoothing, not a model of the
physical process. Thermal creep and slow mechanical settling genuinely
ARE accumulations of many small physical increments over time — a random
walk (cumulative sum of iid steps) is the direct statistical model of an
accumulation, not a filter imposed on top of one. Its known downside
(unbounded growth) is handled by keeping the per-step size small enough
that, over the durations actually simulated here, the accumulated
excursion stays modest — checked numerically
(`test_total_excursion_stays_modest_over_the_default_capture_window`),
not assumed.

**Why `jitter_std_px = 0.15` is an assumption, but not an unconstrained
one.** The brief gives no real gimbal-vibration spec to derive this
number from, so it is honestly a physically-plausible pick, not a
derivation — that should be stated plainly rather than dressed up as more
precise than it is. What IS a real, checkable constraint is how it
compares to a number this project already measured: at SNR~=50 (the
operating point used throughout §2c/§2d), the Gaussian fit's own
precision is `fit_std ~= 0.007-0.011 px` (`results/exp01_snr_characterization.json`).
`0.15 px` sits 15-20x above that noise floor. This is the right thing to
check because if jitter were smaller than the estimator's own measurement
noise, it would be unmeasurable — indistinguishable from estimation error
rather than real motion — and the entire "recover the trajectory from
noisy frames" exercise would be attempting to detect something the
estimator can't actually see. Anchoring against an already-measured
project quantity, rather than picking an isolated round number, is the
same standard applied to every other assumption in this project (e.g.
`background_e`'s placement relative to the read-noise/shot-noise
crossover in `simulate.py`).

**Why the default disturbance (20 Hz, 0.3 px amplitude) is the EASY case,
not the hard one.** The brief separately requires the scenario be made
deliberately hard (disturbance amplitude near the jitter floor, frequency
near the resolution floor) — see the failure-mode analysis planned later
in this section. Baking "hard" into the default parameters from the start
would make it impossible to first verify the recovery and detection
pipeline works correctly on an easy, unambiguous case before stress-
testing it. The easy case is proven first; the hard variant is built as a
deliberate second configuration once that baseline is established.

---

## Dynamic tracking: frame rendering and recovery (§3, part 2)

Full reasoning lives in `sptrack/sequence.py`'s module docstring and
`experiments/exp03b_trajectory_recovery.py`'s; this section summarises.

**Why the recovery loop is seeded from its own previous estimate, never
ground truth.** A real tracker only ever has its own past output to seed
the next frame's window from — it never has access to the true position.
Seeding from ground truth here would make any measured error artificially
optimistic (cheating), and would not honestly answer "can this be
recovered from noisy frames alone," which is the actual question this
section asks.

**Why a failed fit doesn't corrupt the running prior.** If one frame's fit
doesn't converge, its position is untrustworthy. Feeding that untrustworthy
value forward as the next window's centre risks dragging the window off
the real spot and cascading into a lost track from a single bad frame.
Instead the prior only updates on success; on failure the next frame is
still seeded from the last KNOWN-GOOD position (dead reckoning across one
bad frame). Verified directly, not just designed-in:
`test_recover_trajectory_survives_a_degenerate_frame_without_losing_the_prior`
constructs a 3-frame sequence with a deliberately all-zero middle frame
and confirms the third frame still recovers accurately — proof the second
frame's failure (which returns NaN) never reached `extract_window` as a
window centre, which would otherwise crash.

**Why the canvas is 41x41 with a (20.3, 19.7) start, and why that's
checked, not assumed.** The trajectory's total excursion is bounded under
5 px (already verified in `tests/test_trajectory.py`), and the estimator's
window (`half_width=9`) needs to stay inside the frame across that whole
excursion — 5 + 9 = 14 px of required margin, comfortably under the ~20 px
available from a near-centre start in a 41x41 canvas. `exp03b` doesn't
just trust that arithmetic: it computes the actual minimum observed edge
margin across the real 4096-frame run (9.7 px in the run on record) and
reports it, so a future change to the trajectory's amplitude parameters
that quietly violates the margin would show up as a small or negative
number, not a silent mis-registration.

**Why flux is held constant across the sequence.** Brightness change is
explicitly a real-world condition (§4), separate from motion recovery
(this section). Varying both at once would make it impossible to
attribute a result to one effect or the other; holding flux fixed at a
known SNR (50, matching `exp02_realtime.py` and the jitter-vs-CRLB
argument already on record) isolates the question this section asks.

**Cross-check: motion costs nothing beyond the static-frame precision
floor.** The measured std in the moving-sequence experiment (8.8/8.9
millipixels, x/y) closely matches the single-frame Gaussian-fit precision
already measured at the same SNR=50 in §2c/§2d (~7-11 millipixels). This
is a real, checkable consistency result, not a coincidence to wave away:
it says the frame-to-frame prior-gating scheme adds no meaningful extra
error of its own at this SNR — the tracker's precision while the spot is
moving is (to within measurement noise) the same as its precision on a
static spot.

---

## Dynamic tracking: disturbance detection (§3, part 3)

Full reasoning lives in `sptrack/disturbance.py`'s module docstring; this
section summarises, including a real methodological wrong-turn caught by
testing.

**Why a Hann window, and why the low-frequency band is excluded from the
peak search.** The recovered trajectory is not periodic within the
capture window (drift doesn't return to its start), so a plain FFT sees
an artificial discontinuity at wrap-around that leaks energy across the
whole spectrum. A Hann window removes that. Separately, drift's own power
(1/f^2, concentrated at the lowest frequencies) can still dominate a
naive "biggest peak wins" search even after windowing — `exclude_below_hz`
removes that region from the search, grounded directly in the spectral
separation already demonstrated in `figures/exp03a_trajectory_diagnostic.png`.
`tests/test_disturbance.py::test_low_frequency_exclusion_prevents_drift_like_power_from_masquerading_as_the_disturbance`
proves this matters, not just plausible-sounding: without the exclusion, a
synthetic strong low-frequency component wins the peak search over a real
higher-frequency tone; with it, the real tone wins.

**A wrong turn, caught before it shipped: summing energy across the
leaked lobe overestimates amplitude.** Because the disturbance frequency
doesn't land exactly on an FFT bin (20 Hz at this resolution is bin
81.92, not an integer), some of its energy leaks into neighbouring bins.
The intuitive fix — sum amplitude (or RSS-combine) across a few bins
around the peak to recover that leaked energy — was prototyped and
directly tested against a known-amplitude tone before being trusted: it
overestimated the true amplitude by 20-100%, because it also sums in the
window function's own non-zero sidelobes as though they were independent
signal, double-counting energy a single well-corrected peak-bin reading
already mostly captures. The simpler alternative (single peak bin,
corrected by the Hann window's coherent gain, `2*|X|/sum(window)`) was
tested the same way and recovers the true amplitude to within 0.4% even
off-bin. This is a direct instance of the project's standing rule —
verify a claim numerically before trusting it — catching a genuinely
wrong intuition (more bins = more signal recovered) rather than a
correct one.

**Result, and what it isolates.** On the default (easy) scenario:
frequency detected 20.02 Hz vs. injected 20.00 Hz (19.5 mHz error, inside
the FFT's own 244 mHz bin resolution); amplitude detected 0.3001 px vs.
injected 0.3000 px (+0.04%). Running the identical detector on GROUND
TRUTH instead of the recovered trajectory gives essentially the same
answer (20.02 Hz, 0.3009 px) — confirming the small remaining error is
close to the detection method's own inherent floor (matching the <1%
error already measured on a clean synthetic tone), not something the
Gaussian-fit recovery step (part 2) is adding. That separation matters:
it says a worse result on a harder scenario (planned next) can be
attributed to the SCENARIO being harder, not to some hidden weakness in
the detection method uncovered only now.

---

## Dynamic tracking: the hard scenario and its failure modes (§3, part 4)

Full reasoning lives in `experiments/exp03d_hard_scenario.py`'s module
docstring; this section records the actual measured results (a real
10-trials-per-level Monte Carlo sweep, not a single lucky/unlucky run —
an earlier single-realization prototype was explicitly discarded once the
proper sweep was run, because its numbers weren't reliable enough to
report).

**Why SNR=5 (10x below the easy scenario's 50).** At SNR=50, the fit's
own precision (~0.007-0.011 px) is negligible next to jitter (0.15 px) —
why the easy scenario's recovered-vs-ground-truth detection results came
out nearly identical. At SNR=5, fit_std~=0.132 px becomes comparable to
jitter itself — the deliberate point: estimation noise now genuinely
competes with mechanical jitter as a noise source, not a decoration on
top of an already-solved problem.

**Failure mode 1: amplitude-bias toward a measurable noise floor.**
Sweeping true disturbance amplitude from 0.30 px down through 0.10, 0.05,
0.02, to 0.00 px (SNR=5, freq=2.5 Hz, 10 trials/level, phase randomised
per trial):

| true amp (px) | detected amp (mean +/- std) | freq within 1 bin |
|---|---|---|
| 0.30 | 0.289 +/- 0.013 | 10/10 |
| 0.10 | 0.097 +/- 0.013 | 10/10 |
| 0.05 | 0.057 +/- 0.020 | 8/10 |
| 0.02 | 0.033 +/- 0.007 | 5/10 |
| 0.00 | 0.033 +/- 0.007 | 0/10 (nothing true to match) |

Even with literally zero true disturbance, the detector reports a nonzero
amplitude (0.033 px) — direct proof of the mechanism: the reported value
is always the MAXIMUM over ~2000 candidate frequency bins, and pure noise
alone produces a nonzero maximum. This is the same statistical phenomenon
as Rice/Rayleigh noise-floor bias in radar and MRI peak detection — a
property of any peak-search amplitude estimator, not a bug specific to
this project's `detect_disturbance`. Practically: below roughly the
measured floor, a real disturbance becomes indistinguishable from no
disturbance at all by amplitude alone; frequency agreement (which also
degrades, but more gracefully — 100% -> 100% -> 80% -> 50% within one bin
as amplitude drops) is a more robust "is something really there" signal
at low amplitude than the amplitude reading itself.

**Failure mode 2: a genuinely different failure — the fixed exclusion
threshold has a blind spot.** Separately (SNR=50, easy amplitude 0.3 px —
deliberately the easy scenario's own amplitude, to isolate this as a
frequency-placement failure rather than an amplitude one), placing the
disturbance frequency at or near the fixed `exclude_below_hz=2.0 Hz`
threshold:

| true freq (Hz) | detected freq (Hz) | detected amp (px, true=0.300) |
|---|---|---|
| 1.5 (excluded) | 4.639 | 0.029 |
| 1.9 (excluded) | 2.197 | 0.113 |
| 2.0 (boundary) | 2.197 | 0.200 |

Even with an easily-detectable amplitude, the detector locks onto the
WRONG frequency and badly misreads amplitude once the true frequency sits
close to the exclusion boundary. This is qualitatively different from
failure mode 1: not a graceful, continuous degradation, but a hard,
fixed threshold creating a blind spot exactly where a real disturbance
could plausibly sit. It is a genuine limitation of the current design
(`sptrack/disturbance.py`'s `exclude_below_hz` is a fixed hyperparameter,
not derived from the actual drift realisation in a given sequence) worth
stating honestly rather than hiding: a more robust design would estimate
where drift's power has actually fallen off in THIS sequence (e.g. from
an explicit drift fit/subtraction) rather than assuming a fixed cutoff
works for every possible disturbance frequency.

---

## Real-world conditions: scintillation (§4)

Full analysis of all five identified conditions lives in
`docs/REAL_WORLD_CONDITIONS.md`; full modelling reasoning lives in
`sptrack/scintillation.py`'s module docstring. This section summarises
the one condition that got implemented, not just analysed, per the
brief's own instruction that analysis quality matters more here than
implementation.

**Why a mean-reverting (AR(1)/OU) log-normal process, not a random walk.**
Slow drift (§3) was deliberately modelled as a random walk because
thermal creep genuinely accumulates without bound. Scintillation is
physically different — it fluctuates around a stable long-run mean flux
and never wanders away permanently — so it needed a STATIONARY process,
not a non-stationary one. Using the wrong kind of process here would have
been the same category of mistake already avoided once in this project
(picking a model for its mathematical convenience rather than its actual
physical behaviour).

**Why correlated, not independent per frame.** Turbulent eddies take real
time to cross the beam path; the chosen coherence time (5 ms) is
comparable to, not much faster than, the 1 ms frame period, so several
consecutive frames genuinely fade or peak together. This was checked
directly (autocorrelation at several lags matches AR(1) theory to within
0.03 — `tests/test_scintillation.py`), not just asserted.

**Why sigma_ln=0.4 (module default) but sigma_ln=0.6 for the
demonstration experiment.** No site-specific turbulence data exists for
an actual deployment, so both are honest assumptions grounded in
published FSO-scintillation-index ranges, not derivations. The module's
own default (0.4) represents moderate turbulence and, layered on a
workable baseline SNR, was measured to degrade precision without ever
causing a dropout (0 failed fits at base_snr=20). The demonstration
experiment deliberately used a stronger value (0.6, representing a worse
day of turbulence — still within literature-plausible bounds, the upper
end rather than the middle) specifically because a genuine loss-of-lock
demonstration needed a genuine loss of lock to show, found by direct
experimentation rather than assumed to exist at the moderate setting.

**The actual search that led there, recorded rather than presented as if
the final numbers were obvious from the start.** The first prototype used
the module's own moderate default (sigma_ln=0.4) at a comfortable
base_snr=20, purely to see the real behaviour empirically before writing
anything into the codebase. That run showed the expected precision
degradation during fades (std roughly 2.8x worse in low-flux than
high-flux periods) but zero dropouts — a real result, not a failed
attempt, but not yet the "genuine loss of lock" the real-world-conditions
analysis wanted to demonstrate.

Chasing an actual dropout surfaced a separate, more general fact about
this project's own fit worth recording on its own: `gaussian_fit_estimate`'s
convergence criterion (`gaussian_fit.py`) is based on STEP SIZE, not fit
QUALITY — it declares `ok=True` once successive position updates shrink
below `tol_px`, regardless of how noisy the resulting estimate actually
is. That is why `ok=False` turned out to be rare and specifically tied to
truly degenerate inputs (the centroid seed failing from a non-positive
background-subtracted flux sum, or a linear-algebra failure in the
Gauss-Newton solve) rather than to "low SNR" in general — consistent with
what the §3 hard-scenario sweep had already shown (0 failed fits even at
SNR=5 with no scintillation at all). This matters beyond scintillation:
anywhere this project reports an `ok`/failure rate, it should be read as
"the fit degenerated outright," not "the fit was imprecise" — those are
different things, and conflating them would overstate how often this
estimator visibly signals trouble.

Given that, producing real dropouts meant pushing past "low SNR" alone
into "low SNR combined with a severe enough fade to make the centroid
seed itself fail." A short parameter sweep (base_snr x sigma_ln) found:

| base_snr | sigma_ln | dropouts / 4096 |
|---|---|---|
| 5.0 | 0.4 | 1 |
| 5.0 | 0.6 | 25 |
| 4.0 | 0.6 | 71 |
| 3.0 | 0.6 | 163 |

base_snr=5.0/sigma_ln=0.6 (25 dropouts, ~0.6%) was chosen over the more
dramatic 4.0/0.6 (71, ~1.7%) or 3.0/0.6 (163, ~4.0%) options deliberately:
it is a realistic, visible-but-not-overwhelming dropout rate, and it keeps
base_snr=5.0 consistent with the SAME "low SNR" reference point already
established and justified in §3's own hard scenario — reusing an
already-defended number rather than escalating severity further purely to
make the demonstration more dramatic.

**Result.** With base_snr=5.0 (matching §3's own hard-scenario choice)
and sigma_ln=0.6: overall position-error std is 1.7x worse with
scintillation than steady flux (227 vs 137 millipixels); std during deep
fades (flux multiplier < 0.5, 890 of 4096 frames) is 5.1x worse than
during peaks (multiplier > 1.5, 617 frames) — 391 vs 77 millipixels,
confirming precision tracks the INSTANTANEOUS fade, not just the average
flux, exactly as §2c's SNR-vs-precision relationship predicts. 25 of 4096
frames were a genuine loss of lock (`ok=False`), versus 0 with steady
flux at the same average SNR — real dropouts caused specifically by
scintillation's correlated fades, not just added scatter.

**A genuine, honestly-stated coincidence, not a designed fix.** Those 25
dropouts never derail the recovered trajectory, because §3's dead-
reckoning (`sequence.py::recover_trajectory`, built to survive an
isolated bad frame for an unrelated reason) already has exactly the right
shape of mitigation: a failed frame holds the last known-good position
rather than corrupting the track. This works here because the tested
fades are still short (a handful of frames) relative to the trajectory's
own slower dynamics — worth stating as a lucky structural fit found
after the fact, not a mitigation purpose-built for scintillation.

---

## Real-world conditions, continued: fog/rain and beam wander (§5's "robustness to §4 conditions")

The brief only required scintillation to be simulated (§4); simulating
the remaining conditions from `docs/REAL_WORLD_CONDITIONS.md` was
user-directed, going beyond that explicit ask. Full reasoning lives in
`experiments/exp04b_fog_attenuation.py` and `sptrack/beam_wander.py`'s
own docstrings; summarised here.

**Fog/rain: why a steady sweep, not a time-varying process.**
Scintillation fluctuates WITHIN a single ~4s capture window because its
coherence time (ms) is comparable to the frame period. Fog/rain changes
over minutes to hours — essentially constant across any single capture
window this project simulates. Modelling it as a within-sequence random
process the way scintillation was would misrepresent its real timescale;
the honest framing is a steady attenuation level swept ACROSS conditions
(reusing §2c's own sweep structure), not a new random-process module.

**Fog/rain: dropout rate alone understates the real failure.** At
moderate/dense fog (SNR collapsed to ~0.02/~3e-11), only 43-44% of trials
registered as outright failures (`ok=False`) — but the "successful"
remainder's std explodes to ~2 px, comparable to the whole 21x21
estimation window. These are noise-driven fits to nothing, not
meaningfully imprecise real measurements. This is the SAME mechanism
already noted for scintillation's demonstration parameters
(`gaussian_fit_estimate`'s convergence criterion is step-size-based, not
quality-based) showing up again in a different condition — worth reading
as a general property of this project's `ok` flag, not a fog-specific
quirk.

**Beam wander: why equal-variance-but-different-shape was chosen
deliberately.** `sigma_px=0.15` was set EQUAL to `trajectory.py`'s
`jitter_std_px`, specifically so the demonstration couldn't be dismissed
as "of course they look different, one is bigger" — verified directly
that two position-noise sources with identical std can still be
separated by spectral shape alone (low/high-band power ratio ~0.036 for
white jitter vs. ~9.18 for beam wander, a ~250x difference,
`experiments/exp04c_beam_wander.py`), and that when summed they combine
independently (measured combined std 0.2097 px vs. the quadrature
prediction sqrt(0.15^2+0.15^2)=0.2121 px — almost exact agreement).

**A genuine interaction flagged, not chased down further here.** Beam
wander's spectral shape overlaps with drift's (both low-frequency-
concentrated) — meaning a real deployment with both present would need a
WIDER `exclude_below_hz` in `sptrack/disturbance.py` to keep either from
masquerading as the periodic disturbance, which increases exposure to the
boundary-blind-spot failure mode already found in §3 part 4 if a real
disturbance's frequency happens to sit near that widened boundary.
Recorded as an identified risk connecting two separately-built pieces of
this project, not re-characterized quantitatively here — a reasonable
place to stop rather than open a third nested investigation.

---

*(This document will grow as each new part of the simulator — dynamic
tracking, real-world conditions, etc. — introduces its own assumptions.)*
