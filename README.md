# sub-pixel-tracker

Sub-pixel laser-spot localization for a gimbal-mounted free-space optical
terminal, a Computer Vision Engineer take-home exercise for
Transcelestial Technologies. Two terminals, each with a camera and a
gimbal-mounted laser; the camera must find the incoming laser spot's
position to sub-pixel precision, in real time (~1 kHz), under a range of
noise, motion, and outdoor conditions.

## What this is

A synthetic-frame simulator, three independent position estimators
(windowed centroid, Poisson-weighted 2D Gaussian fit, matched filter),
and a large set of Monte Carlo characterization experiments comparing
them against each other and against the Cramér-Rao lower bound, plus
dynamic tracking (drift + jitter + a periodic disturbance), real-world
condition modelling (scintillation, fog, beam wander, clutter, glare),
and several "go further" extensions (auto-exposure, calibration, motion
blur, low-photon-count, and latency-budget characterization).

## Why it's built this way

The brief's own stated grading criterion is that every decision must be
defensible live. Everything in this repo follows one rule: no numeric
choice ships without either a derivation from something already
established in the project, or (when no derivation is possible, a
physical constant like lens distortion or a sensor's readout time) a
real, cited external source, confirmed rather than guessed. Every design
decision, bug found, and wrong turn taken along the way is recorded in
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md), including the mistakes,
not just the final answers.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

Python 3.10+ (uses `from __future__ import annotations` and modern type
hints throughout). Requires `numpy`, `scipy`, `matplotlib`, `pytest`.

## How to reproduce

Run the test suite (136 tests, covers every `sptrack/` module):

```bash
python -m pytest -q
```

Run any experiment as a module from the repo root (writes a JSON
result to `results/` and a figure with an embedded "what we see / what we
can derive" panel to `figures/`):

```bash
python -m experiments.exp01_snr_characterization
```

There is no single command that reproduces every figure yet, that's
`docs/PROGRESS.md`'s §7 item, not yet built. Until then, each experiment
below is run individually.

| Script | What it produces |
|---|---|
| `exp01_snr_characterization` | Bias/std vs SNR for all 3 estimators, against the CRLB (§2c) |
| `exp02_realtime` | Per-frame compute cost vs the 1 kHz budget (§2d) |
| `exp03a_trajectory_diagnostic` | Ground-truth drift+jitter+disturbance trajectory and its spectrum (§3) |
| `exp03b_trajectory_recovery` | Full-sequence trajectory recovery, error over time (§3) |
| `exp03c_disturbance_detection` | Recovered vs. injected disturbance frequency/amplitude (§3) |
| `exp03d_hard_scenario` | Deliberately hard scenario, two real failure modes (§3) |
| `exp04a_scintillation` | Simulated atmospheric scintillation impact (§4, brief's named example) |
| `exp04b_fog_attenuation` | Weather-attenuation sweep, lock-loss rate (§4) |
| `exp04c_beam_wander` | Turbulence-induced position noise, spectrally separable from jitter (§4) |
| `exp04d_clutter` | Acquisition failure from a false bright source, and its fix (§4) |
| `exp04e_glare` | Background-gradient bias, and the planar-fit fix (§4) |
| `exp05a_auto_exposure` | Auto-exposure/gain control across 5 decades of brightness (§5) |
| `exp05b_calibration` | Bias-frame, flat-field, and lens-distortion calibration (§5) |
| `exp05c_motion_blur` | Intra-frame motion blur robustness sweep (§5) |
| `exp05d_low_photon_count` | Extreme low-photon-count characterization (§5) |
| `exp05e_latency_budget` | Full photon-to-estimate latency/throughput budget (§5) |
| `exp06a_pixel_locking` | Bias vs sub-pixel phase, and PSF sampling margin |
| `exp06b_window_size` | Window half-width sweep against the CRLB |
| `exp07_kalman_tracking` | Kalman and alpha-beta filtering, and when it helps |

## What's tested

136 tests across `tests/`, one file per `sptrack/` module. Every noise
source, every estimator, every calibration/robustness technique has
direct numerical tests, not just "does it run," but checks like: does a
derived formula match a known closed form, does a claimed bug-fix
actually change behaviour, does a proposed mitigation measurably reduce
the effect it targets. Several tests exist specifically because an
initial implementation attempt was wrong and the test is what caught it
(see `docs/ASSUMPTIONS.md` for the specific stories, e.g. two real bugs
in the Gaussian fit's convergence logic, caught not by a unit test but by
an integration experiment producing an impossible result).

## Where it breaks, known limits, not hidden

- Fog/heavy attenuation: below a hard SNR floor, every estimator
  loses lock outright (`exp04b`), an operational limit of the system,
  not a bug to fix.
- The centroid's `ok` flag cannot be trusted at face value at very
  low SNR or photon count: it almost never returns `ok=False`, but its
  "successful" answers become noise-driven and can carry a real,
  non-shrinking bias (`exp05d`, `exp04b`), a formal-success-vs-answer-
  quality gap, found independently twice.
- The matched filter fails at a much HIGHER photon count than the
  Gaussian fit (the opposite of their accuracy ordering at moderate/
  high SNR), its failure test is geometric, not convergence-based
  (`exp05d`).
- The periodic-disturbance detector has a fixed-threshold blind spot:
  a real disturbance whose frequency sits near the drift-exclusion
  boundary is badly misdetected, even at an otherwise easy amplitude
  (`exp03d`).
- The Gaussian fit has no hard cost ceiling: its measured worst-case
  compute time exceeded the 1 ms frame budget in testing, unlike the
  other two estimators (`exp02`).
- Recovering from sudden brightening is far slower than from sudden
  dimming in the auto-exposure controller, a saturated reading
  destroys the magnitude information needed for a confident one-step
  correction (`exp05a`).
- The centroid's default negative-value clipping produces a
  phase-dependent bias of about 4.4 millipixels peak-to-peak, which does
  not average away as the spot moves (`exp06a`).
- The project's fixed window half-width of 9 px is poor for the centroid
  specifically, costing 119% at SNR=10 against its own optimum, so the
  centroid efficiency quoted in `exp01` is pessimistic (`exp06b`).
- Temporal filtering is implemented but off by default. It only reduces
  error once measurement noise approaches per-frame target motion, which
  here means SNR below about 8 (`exp07`).
- Lens distortion, beam wander, and the FSM slew-rate/plate-scale
  numbers used in §5 are sourced from real but imperfect external data
  (cited machine-vision datasheets and one real published FSO
  steering-mirror paper), stated as such, not claimed as precise
  derivations. See `docs/ASSUMPTIONS.md` for exactly which numbers are
  derived vs. sourced vs. deliberately left as round bounds.

## What's next

Per `docs/PROGRESS.md`:
- §6 self-check (this document is part of closing it out)
- §7 deliverables: a one-command reproduction script, and the written
  report itself (this repo + `docs/` currently stand in for it)
- Further real-world conditions identified but deliberately not
  simulated (sensor saturation as a dedicated item, cosmic-ray hits,
  impulsive/thruster-firing disturbances, an angular pointing-precision
  framing) are recorded in
  [`docs/REAL_WORLD_CONDITIONS.md`](docs/REAL_WORLD_CONDITIONS.md)'s
  "Further considerations" section, reserved for the written report.

## Repo map

- `sptrack/`, the simulator, estimators, and analysis modules
- `experiments/`, one script per numbered experiment above; each is
  independently runnable and writes to `results/` and `figures/`
- `tests/`, the test suite, one file per `sptrack/` module
- `docs/DESIGN_RATIONALE.md`, every concept used, what it does, why it
  was chosen over the alternatives, and what was deliberately not done
- `docs/PROGRESS.md`, every brief requirement, its status, and what
  proves it
- `docs/ASSUMPTIONS.md`, every deliberate choice made and why,
  including bugs found and wrong turns taken
- `docs/REAL_WORLD_CONDITIONS.md`, the §4 analysis and simulated results
- `docs/CV_Eng_Assessment_Requirements.md`, this project's own
  bullet-point breakdown of the brief
