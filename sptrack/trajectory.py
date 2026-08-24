"""Ground-truth motion for the dynamic-tracking section (brief §3): a
laser spot's true sub-pixel position over a sequence of frames, built from
three mixed components -- slow drift + random jitter + one periodic
disturbance -- exactly as the brief specifies.

WHY THESE THREE COMPONENTS, PHYSICALLY
-------------------------------------------
The brief's own context (§1) names the real sources: "both platforms
vibrate randomly" and camera/sensor problems include "exposure drift, gain
drift, temperature drift". Three genuinely different physical processes
map onto three genuinely different STATISTICAL signatures:

  Slow drift     <- thermal creep, slow mechanical settling of the gimbal
                     mount. These are cumulative processes: many small,
                     weakly-correlated micro-changes (a joint settling a
                     little more, a bracket expanding a little further)
                     ACCUMULATE over time rather than resetting each frame.
                     A random walk (cumulative sum of small iid Gaussian
                     steps) is the standard model for exactly this kind of
                     accumulation, and it has the right spectral
                     signature for "slow": a random walk's power spectral
                     density falls off as 1/f^2, concentrating almost all
                     of its power at the lowest frequencies the recording
                     window can resolve.

  Random jitter  <- mechanical shake: structural resonances, air currents,
                     minor motor cogging, electrical noise in the gimbal
                     actuators. Many quasi-independent micro-sources
                     summing together -> approximately Gaussian by the
                     central limit theorem (the same argument already used
                     for read noise in sensor.py). If each source's own
                     correlation time is short compared to the 1 ms frame
                     period (a reasonable assumption for the sub-millisecond
                     mechanical events involved), the sequence is well
                     approximated as WHITE -- uncorrelated frame to frame.
                     This is the model used here: iid Gaussian noise added
                     independently to every frame.

  Periodic       <- a single dominant mechanical tone: a cooling fan, a
  disturbance       reaction wheel, motor cogging at a fixed rotation
                     rate. Real rotating machinery generically couples
                     vibration energy into a narrow band around its own
                     rotation frequency (and its harmonics), not
                     broadband -- which is why it shows up as a single
                     spectral SPIKE rather than raising the noise floor
                     everywhere. Modelled here as one sinusoid at a fixed
                     frequency, amplitude, and phase.

WHY THIS MATTERS FOR DETECTION LATER
------------------------------------------
These three components were not chosen just for physical plausibility --
their DIFFERENT spectral shapes (drift: low-frequency-concentrated;
jitter: flat/white; disturbance: a single spike) are exactly what makes
the disturbance's frequency and amplitude separable from the rest of the
motion at all. If jitter were not (approximately) white, or if drift
leaked significant power into the disturbance's frequency band, a later
FFT-based detector could not cleanly isolate the disturbance from the
background noise floor. This separability is verified numerically in
tests/test_trajectory.py, not just asserted here.

WHY A RANDOM WALK RATHER THAN A DESIGNED LOW-PASS FILTER FOR DRIFT
--------------------------------------------------------------------------
An alternative would be to low-pass filter white noise into a smooth
drift curve -- but that is an engineering choice about a FILTER, not a
model of what actually happens physically. Thermal creep and slow
mechanical settling are literally accumulations of many small physical
increments over time; a random walk is the direct, standard statistical
model of an accumulation process (it IS a running sum), not a filter
imposed on top of one. Its downside -- a random walk is unbounded, and
can wander arbitrarily far given enough time -- is why the STEP size is
kept small relative to the sequence length actually used (verified
numerically below and in the tests), so the excursion stays modest (a few
pixels) over the durations this project simulates (a few seconds), rather
than assuming boundedness away.

DEFAULT PARAMETERS, AND WHY EACH ONE WAS CHOSEN
------------------------------------------------------
  dt_s = 1e-3 (1 kHz)
      Directly the brief's own stated loop rate (§1: "~1 kHz").

  n_frames = 4096
      A power of two (FFT-friendly, no zero-padding needed for the
      disturbance-detection step that follows), giving a 4.096 s capture
      window. Long enough for many cycles of a realistic mechanical
      disturbance frequency (tens of Hz -- see below) to be observed, and
      long enough that the FFT's frequency resolution (1 / (N*dt) =
      0.244 Hz here) is fine enough to distinguish the disturbance tone
      from nearby jitter-floor bins.

  drift_step_std_px = 0.01
      Chosen so the random walk's CUMULATIVE std after the full 4096-frame
      window, std_N = drift_step_std_px * sqrt(N) = 0.01 * 64 = 0.64 px,
      stays a modest fraction of a pixel over the whole capture -- "slow"
      in the sense the brief means it: barely moving frame-to-frame
      (0.01 px << jitter_std_px below), but not negligible over the full
      sequence. Verified numerically in the tests rather than assumed.

  jitter_std_px = 0.15
      The dominant FRAME-TO-FRAME mover, consistent with "vibrate
      randomly (shake)" in the brief's context section -- an order of
      magnitude above a single drift step (0.01 px), so jitter, not
      drift, explains most of the apparent motion between any two
      consecutive frames, while drift still wins over the long run because
      it accumulates and jitter's variance does not.

      No spec sheet exists for real gimbal vibration magnitude, so this
      number is honestly an assumption, not a derivation -- but it is not
      an UNCONSTRAINED one. It is anchored against a number this project
      already measured: at SNR~=50 (the operating point used throughout
      2c/2d), the Gaussian fit's own precision is fit_std ~= 0.007-0.011 px
      (results/exp01_snr_characterization.json). 0.15 px sits 15-20x above
      that noise floor. This matters because if jitter were SMALLER than
      the estimator's own measurement noise, it would be unmeasurable --
      indistinguishable from estimation error rather than real motion --
      making "recover the trajectory" meaningless to attempt. 0.15 px
      guarantees jitter is a genuinely resolvable signal against this
      project's own measured precision, not a scale picked in isolation.

  disturb_freq_hz = 20.0, disturb_amp_px = 0.3
      20 Hz is not a free choice. A rotating machine's fundamental
      disturbance tone sits at its rotation rate, so 20 Hz corresponds to
      20 * 60 = 1200 RPM. Published reaction-wheel characterisation puts
      the fundamental harmonic of a spacecraft reaction wheel assembly
      "typically between 15-40 Hz", with a representative assembly at
      800 RPM (13.3 Hz) and operating speeds running up to roughly
      6200 RPM. 1200 RPM / 20 Hz therefore lands inside the documented
      fundamental band for exactly the class of mechanism the brief's own
      context section names (platform vibration from onboard rotating
      machinery), not merely inside the broad 0.1 Hz to 1 kHz
      micro-vibration envelope.

      It is also comfortably below the 500 Hz Nyquist limit at 1 kHz
      sampling, and easily resolved at 4096 frames (bin ~82 of ~2048, far
      from both DC and the Nyquist edge).

      Amplitude 0.3 px is comparable to (double) the jitter std: large
      enough to be a real, recoverable signal for this FIRST, easy
      version of the scenario. The brief's explicit "make it hard"
      requirement (disturbance near the jitter floor, frequency near the
      resolution floor) is deliberately NOT baked into these defaults --
      it is built as a separate, harder configuration once the easy case
      is proven to work correctly (see the dynamic-tracking experiment
      script).

      Source: Characterization of reaction wheel micro-vibrations
      (ISMA 2018); Reaction Wheel Disturbance Modeling, Jitter Analysis,
      and Validation (NASA NTRS 20080039248).

  disturb_axis = "x"
      The disturbance is injected on one axis only, and y carries drift +
      jitter alone. This keeps the frequency-domain detection analysis
      unambiguous (one clean signal to detect, not two summed ones) while
      losing no generality -- the same detection method applies
      identically to y if the disturbance were on that axis instead, or
      to both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryConfig:
    n_frames: int = 4096
    dt_s: float = 1e-3
    x0: float = 0.0
    y0: float = 0.0
    drift_step_std_px: float = 0.01
    jitter_std_px: float = 0.15
    disturb_freq_hz: float = 20.0
    disturb_amp_px: float = 0.3
    disturb_phase_rad: float = 0.0
    disturb_axis: str = "x"  # "x", "y", or "both"
    seed: int | None = None


def generate_trajectory(cfg: TrajectoryConfig) -> dict:
    """Generate ground-truth (x, y) position per frame, plus every
    component separately -- the components are returned individually (not
    just the sum) because the dynamic-tracking experiment needs the
    injected drift/jitter/disturbance values to compare recovered
    estimates against, not just the final trajectory.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_frames
    t = np.arange(n, dtype=np.float64) * cfg.dt_s

    drift_x = np.cumsum(rng.normal(0.0, cfg.drift_step_std_px, n))
    drift_y = np.cumsum(rng.normal(0.0, cfg.drift_step_std_px, n))

    jitter_x = rng.normal(0.0, cfg.jitter_std_px, n)
    jitter_y = rng.normal(0.0, cfg.jitter_std_px, n)

    disturb = cfg.disturb_amp_px * np.sin(
        2.0 * np.pi * cfg.disturb_freq_hz * t + cfg.disturb_phase_rad
    )
    disturb_x = disturb if cfg.disturb_axis in ("x", "both") else np.zeros(n)
    disturb_y = disturb if cfg.disturb_axis in ("y", "both") else np.zeros(n)

    x = cfg.x0 + drift_x + jitter_x + disturb_x
    y = cfg.y0 + drift_y + jitter_y + disturb_y

    return {
        "t": t,
        "x": x,
        "y": y,
        "drift_x": drift_x,
        "drift_y": drift_y,
        "jitter_x": jitter_x,
        "jitter_y": jitter_y,
        "disturb_x": disturb_x,
        "disturb_y": disturb_y,
    }
