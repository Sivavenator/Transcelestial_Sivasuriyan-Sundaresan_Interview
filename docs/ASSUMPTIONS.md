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

*(This document will grow as each new part of the simulator — noise, SNR
control, dynamic tracking, etc. — introduces its own assumptions.)*
