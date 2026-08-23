# Real-World Conditions (brief §4)

The brief is explicit that for this section, analysis quality matters
more than implementation — only the named example (scintillation) was
required to be simulated. This document identifies five conditions a real
outdoor, uncontrolled deployment would face, and for each: how it shows
up in the image, what it does to the position estimate, and how to detect
and handle it. All five were subsequently simulated (beyond the brief's
explicit ask, at the user's direction), each with a "Simulated result"
subsection reporting real, measured numbers — and for the two conditions
with a purely algorithmic fix available (background clutter, solar
glare), a real mitigation was built and verified, not just a demonstrated
failure. See `docs/PROGRESS.md`'s §4/§5 tables and `docs/ASSUMPTIONS.md`
for the full build history; this document is the consolidated reference.

| # | Condition | Simulated? | Mitigation built? |
|---|---|---|---|
| 1 | Atmospheric scintillation (brief's named example) | Yes | Bridged by §3's existing dead-reckoning |
| 2 | Beam wander | Yes | Not separable from jitter without an independent sensor |
| 3 | Background clutter / false sources | Yes | **Yes** — `sptrack/acquisition.py::acquire_target` |
| 4 | Solar glare / non-uniform background | Yes | **Yes** — `sptrack/estimators/base.py::planar_background` |
| 5 | Fog, haze, rain attenuation | Yes | System-level only (no algorithmic fix possible) |

---

## 1. Atmospheric scintillation (named in the brief — simulated below)

**Physical mechanism.** Turbulent, randomly-varying refractive-index
cells along the beam path (thermal micro-convection in the air column)
cause constructive and destructive interference at the receiver,
producing rapid, random fluctuations in received INTENSITY — distinct
from beam wander (item 2), which is a position effect from the same
turbulence. The standard weak-turbulence model (Rytov theory) treats the
received irradiance as log-normally distributed, characterised by a
scintillation index sigma_I^2. Critically, scintillation is not
frame-independent noise: its fluctuations are temporally CORRELATED, with
a coherence time set by the turbulent eddies crossing the beam (typically
single-digit milliseconds for a ground-level horizontal path) — comparable
to, not much faster than, this system's 1 ms frame period. Modelling it as
independent per-frame noise (the way photon shot noise legitimately is)
would understate its real impact: multiple consecutive frames fade
together, not just one.

**What it does to the estimate.** Flux entering the sensor rises and
falls over timescales of several to tens of frames. Since every
estimator's precision scales with SNR (§2c: efficiency measured against
the CRLB, itself proportional to 1/sqrt(flux)), position precision
DEGRADES during a fade and RECOVERS during a peak — not a constant noise
floor but a time-varying one. During a deep enough fade, SNR can drop
low enough that the fit fails to converge entirely (`ok=False`) or, for
the centroid, the background-subtracted flux can go non-positive,
returning a failed estimate outright — a genuine, transient LOSS OF LOCK,
not just added scatter.

**Detect and handle.** Every estimator in this project already returns
`flux` as part of its `Estimate` (`sptrack/estimators/base.py`) — a
per-frame SNR proxy requires no new instrumentation, only watching a
number already being computed. A sustained downward trend flags an
incoming fade before it becomes a dropout. For handling: this project's
existing frame-to-frame prior gating (`sptrack/sequence.py::recover_trajectory`)
already has the right shape of mitigation built in — a failed fit does
not corrupt the running prior; the next frame is still seeded from the
last known-good position (dead reckoning across a bad frame). Section
5's precedent, if pursued further, would be to widen the search window
or increase exposure/gain temporarily once a fade is detected, rather
than only passively surviving it.

**Simulated result** (`sptrack/scintillation.py`, `experiments/exp04a_scintillation.py`).
Modelled as a mean-reverting log-normal (AR(1)) flux multiplier —
stationary (fluctuates around a stable mean, unlike drift's unbounded
random walk) and temporally correlated (5 ms coherence time, comparable
to the 1 ms frame period, verified against AR(1) theory directly). A
direct A/B comparison (identical trajectory and sensor noise,
scintillation toggled, base SNR=5.0, sigma_ln=0.6) found: overall
position-error std 1.7x worse with scintillation (227 vs 137
millipixels); std during deep fades 5.1x worse than during peaks (391 vs
77 millipixels) — precision tracks the INSTANTANEOUS fade, not just
average flux; 25 of 4096 frames were a genuine loss of lock (`ok=False`)
vs. 0 with steady flux at the same average SNR. All 25 dropouts were
bridged by §3's dead-reckoning without derailing the track — a real
coincidence (that mechanism was built to survive one isolated bad frame,
for an unrelated reason), stated honestly rather than claimed as a
purpose-built fix.

---

## 2. Beam wander (angle-of-arrival fluctuation from turbulence)

**Physical mechanism.** The same turbulent cells that cause scintillation
also refract the beam's local angle of arrival, causing the spot's
APPARENT position on the sensor to wander — a real, physical position
fluctuation, distinct from (and in addition to) the mechanical shake
already modelled as `jitter_std_px` in `sptrack/trajectory.py` (§3).
Mechanical jitter comes from the platform; beam wander comes from the air
between the platforms. Both look identical in a single frame — this is
worth being explicit about, since a live reviewer could reasonably ask
"why do you have jitter AND turbulence — aren't they double-counted?" The
honest answer: they are not double-counted in this project — §3's
`trajectory.py` models only mechanical jitter (platform shake); beam
wander is modelled separately (`sptrack/beam_wander.py`) as its own,
independent contributor, summed with jitter rather than merged into one
inflated number.

**What it does to the estimate.** Adds position noise indistinguishable,
frame to frame, from mechanical jitter or estimation noise — the
estimator cannot tell "the spot really moved" from "the estimate is
noisy." Over many frames, though, its spectral signature differs from
white mechanical jitter: turbulence-induced angle-of-arrival noise has
far more LOW-frequency power than a white process (aperture averaging
smooths out the fast structure driving scintillation, leaving
predominantly slower structure), the same kind of character already
modelled for slow drift in §3 — meaning it shows up as an addition to the
drift/low-frequency part of the spectrum, not the flat jitter floor.

**Detect and handle.** Cannot be separated from mechanical jitter using
position data ALONE — no single-camera position measurement can tell
which physical process caused a given wobble. Detection in practice
would correlate against an independent channel: a co-located
accelerometer/IMU on the gimbal platform would isolate the MECHANICAL
component, leaving the residual (position error not explained by the IMU)
attributable to the atmosphere. Handling is the same as for any
zero-mean position noise: the recovery/tracking loop already built in §3
does not assume a specific physical CAUSE for jitter, only its
statistics, so no new estimator logic is required — only a wider
noise budget than mechanical shake alone would suggest.

**Simulated result** (`sptrack/beam_wander.py`, `experiments/exp04c_beam_wander.py`).
Modelled as a zero-mean, mean-reverting (OU/AR(1)) position process —
same mathematical family as scintillation, but linear (not log-normal,
since position can be negative) and with a longer coherence time (20 ms
vs. scintillation's 5 ms, justified by aperture averaging smoothing out
the fast structure). Deliberately set to the SAME std as mechanical
jitter (0.15 px) to make the separability point sharply rather than
softly: verified directly that two position-noise sources with identical
time-domain variance are still ~250x separable by spectral shape alone
(low/high-band power ratio 0.036 for white jitter vs. 9.18 for beam
wander), and that they combine independently in quadrature (measured
combined std 0.2097 px vs. predicted sqrt(0.15²+0.15²)=0.2121 px — almost
exact agreement). A genuine, not-yet-characterized interaction was
flagged: beam wander's low-frequency spectral shape overlaps with
drift's, meaning a real deployment with both present would need a WIDER
exclusion band in `sptrack/disturbance.py`'s peak search — increasing
exposure to the boundary-blind-spot failure mode already found in §3
part 4, if a real disturbance's frequency happens to sit near that
widened boundary.

---

## 3. Background clutter / false bright sources

**Physical mechanism.** A real outdoor scene contains other bright
sources sharing the sensor's field of view or falling within a coarse
acquisition search: streetlights, vehicle headlights, glinting
reflections, or (for a satellite-adjacent link) other satellites,
aircraft, or planets. Unlike every noise source built into `sensor.py`,
this is not a noise process at all — it is real, structured, spatially
localised SIGNAL that happens not to be the laser spot.

**What it does to the estimate.** This project's actual acquisition
mechanism is directly vulnerable: `find_brightest_pixel`
(`sptrack/estimators/base.py`), used whenever no prior position is
available (the very first frame of a sequence, or after a lost lock),
places its search window on the single brightest pixel in the FULL
frame with no other criterion. A bright clutter source anywhere in frame
would win outright over a dim, distant, or attenuated real laser spot —
not a precision loss but a categorical wrong-target lock, with the
estimator then reporting a highly confident, fully "successful"
(`ok=True`) position for the WRONG object. This is a worse failure than
any noise source characterised so far: every noise source in §2/§3
degrades precision or occasionally fails visibly (`ok=False`); clutter
can fail silently and confidently.

**Detect and handle.** Detection requires a criterion beyond "brightest":
the real laser source has known, checkable structure that generic clutter
usually does not share by coincidence — a known approximate PSF width
(`sigma`, already assumed known throughout this project), and, once
tracking is locked, a known approximate VELOCITY (clutter unrelated to
the gimbal's own dynamics will not move consistently with the trajectory
model in §3). Once locked, the existing prior-gating window (§3) already
provides strong ongoing protection on its own: a clutter source appearing
far from the current tracked position, after acquisition, simply falls
outside the tracking window and is never seen by the estimator at all —
clutter is a real risk mainly at acquisition / re-acquisition, not during
steady tracking, which is exactly where the built mitigation below
targets it.

**Simulated result / mitigation implemented** (`sptrack/acquisition.py::acquire_target`,
`experiments/exp04d_clutter.py`). Ranks candidate local maxima by Pearson
correlation against the assumed Gaussian PSF template — scale-invariant,
so it discriminates by SHAPE, not brightness, reusing the same
underlying PSF model as the matched filter (§5) applied at acquisition
time instead of sub-pixel refinement. Proven directly on a constructed
frame: a clutter source with 13x the true spot's total flux but a wider
(sigma=5px vs. the true spot's 1.75px), non-diffraction-limited profile
fools `find_brightest_pixel` outright (picks the clutter), while
`acquire_target` correctly picks the true spot on the IDENTICAL frame
(shape-match score 0.984 for the true spot vs. 0.690 for the clutter).
Worth noting honestly: the first prototype's clutter flux was not
actually bright enough to win on PEAK pixel value despite having more
TOTAL flux (peak brightness falls off with sigma², so a wider source
needs disproportionately more total flux to win) — the failure mode had
to be deliberately strengthened before it was real enough to demonstrate.

---

## 4. Solar glare / strong non-uniform background illumination

**Physical mechanism.** Direct or scattered sunlight near the camera's
field of view raises the background level, often sharply and
NON-uniformly (a glare gradient toward the sun's direction, or a hard
bright/dark boundary at a shadow edge) — far exceeding the mild,
deliberately modest gradient already built in `sptrack/scene.py`
(`gradient_frac` there defaults to a flat background specifically because
"a gradient's strength and direction are scene-specific," per
`simulate.py`'s own docstring — solar glare is exactly the scenario that
default was left open for).

**What it does to the estimate.** Two distinct effects: (1) a much higher
background level directly worsens SNR — the same mechanism as any
background increase in `sptrack/snr.py`'s flux/SNR relationship — reducing
precision uniformly; (2) more seriously, a background level that varies
STRONGLY across the estimation window breaks `border_median_background`'s
(`sptrack/estimators/base.py`) core assumption that the border is
reasonably uniform. A steep real gradient across a 19x19 window would make
the border median a poor estimate of the background directly UNDER the
spot, reintroducing exactly the structural centre-of-window bias that
background subtraction exists to remove (`centroid.py`'s own docstring)
— except now the bias is not fixed and predictable, it depends on which
direction the glare is coming from as the gimbal slews.

**Detect and handle.** Detectable directly from the RAW frame before any
estimation: comparing background estimates from different, well-separated
regions of the frame border (e.g. the four window corners individually,
rather than one pooled median) reveals a gradient — a large discrepancy
between them is a direct, cheap glare/gradient signal, no extra sensing
required. Handling: fit a low-order (planar) background model across the
window instead of a single flat median when a gradient is detected — a
natural, bounded generalisation of the same `border_median_background`
machinery already in place, built and verified below rather than left as
a proposal.

**Simulated result / mitigation implemented** (`sptrack/estimators/base.py::planar_background`,
`experiments/exp04e_glare.py`). Checked directly before assuming
anything: `border_median_background`'s scalar is NOT a bad estimate — it
reads the true background value at the window's centre to ~0.002
electrons even under a strong gradient. The real failure is structural:
subtracting one CONSTANT from a window whose true background genuinely
VARIES leaves a real residual gradient in the "background-subtracted"
image, growing from ~zero at the centre toward the edges, and the
centroid's weighted average responds to that residual as if it were real
signal. Swept gradient strength on identical frames: the scalar-median
approach's bias grows essentially linearly, reaching 2.54 px at the
strongest gradient tested (gradient_frac=3.0, background varying by
300% of its mean) — far larger than almost any other systematic bias
characterised anywhere else in this project, and already substantial
(0.40 px) at a much milder gradient_frac=0.3. The planar-fit mitigation's
bias stays flat at the ~0.0001 px noise floor across the ENTIRE swept
range, because it removes the gradient's linear structure everywhere in
the window at once, not just estimates it accurately at one point.

---

## 5. Fog, haze, and rain attenuation

**Physical mechanism.** Airborne water droplets or particulates
scatter and absorb the beam along its path, attenuating received power —
governed by the Beer-Lambert law (`P_received = P_transmitted *
exp(-alpha * distance)`, attenuation coefficient alpha rising sharply with
fog/rain density). Unlike scintillation's millisecond-scale fluctuation,
this varies slowly (seconds to minutes, tracking weather, not turbulence)
and can be SEVERE — dense fog can attenuate an optical link by tens of dB,
not the few-percent-level effects modelled elsewhere in this project.

**What it does to the estimate.** A direct, large reduction in flux —
mechanically the same SNR-vs-precision relationship as everywhere else in
this project (§2c), but potentially pushing SNR far below anything
characterised so far (the sweep in `exp01_snr_characterization.py` went
down to SNR=3; sustained heavy fog could plausibly push SNR below that
for extended periods, not just a single hard frame). Below some floor,
every estimator built here loses lock outright and stays lost until
conditions improve — this is a genuine OPERATIONAL limit of the system,
not a bug to fix, and worth stating as such rather than implying the
system degrades gracefully forever.

**Detect and handle.** Trivially detectable — flux/SNR (again already
computed) trending down over seconds, not frames, distinguishes this from
a scintillation fade by TIMESCALE alone. Handling at the algorithm level
is limited once flux is genuinely too low (no estimator invents missing
photons); the honest mitigations are system-level: increase transmit
power or receiver exposure/gain if available (§5's auto-exposure item),
widen the acquisition search and accept a slower re-lock once conditions
clear, and — most importantly for a link budget discussion — size the
system's link margin around realistic fog/rain statistics for the
deployment site, which is an optical link-budget decision upstream of
anything this codebase controls.

**Simulated result** (`experiments/exp04b_fog_attenuation.py`). Modelled
as a STEADY attenuation level swept across named weather conditions
(clear/haze/light/moderate/dense fog, published dB/km ranges, 1 km
assumed link) rather than a within-sequence random process — fog changes
over minutes, not milliseconds, so treating it like scintillation's fast
correlated fluctuation would misrepresent its real timescale. Reused
existing SNR/flux machinery directly (no new sptrack module needed).
Found a hard operational CLIFF, not gradual decline: clear air and haze
barely affect lock (SNR 49 and 31), light fog alone pushes SNR down to
2.7 (at the edge of anything characterised in §2c), and moderate/dense
fog collapse SNR to essentially zero. A genuinely important nuance found
along the way: the raw dropout-rate number (43-44% at moderate/dense fog)
UNDERSTATES the real failure — the "successful" (`ok=True`) remainder's
position std explodes to ~2 px, comparable to the whole estimation
window, meaning those are noise-driven fits to nothing, not meaningfully
imprecise real measurements. The same mechanism already noted for
scintillation (`gaussian_fit_estimate`'s convergence criterion is
step-size-based, not fit-quality-based) shows up again here — worth
reading as a general property of this project's `ok` flag, not a
condition-specific quirk.
