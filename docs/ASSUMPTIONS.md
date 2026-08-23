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
- Why: this is the natural unit for "how much light is in the spot," and it
  makes the renderer's output additive and easy to reason about (e.g. flux
  in electrons, straightforwardly convertible from photons via quantum
  efficiency later).
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

---

*(This document will grow as each new part of the simulator — remaining
noise sources, SNR control, dynamic tracking, etc. — introduces its own
assumptions.)*
