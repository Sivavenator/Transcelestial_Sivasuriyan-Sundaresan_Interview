"""Atmospheric scintillation: turbulence-induced, temporally-correlated
fluctuation of received flux -- the brief's named real-world-conditions
example (§4), the one condition in `docs/REAL_WORLD_CONDITIONS.md` that
gets simulated, not just analysed.

WHY LOG-NORMAL, AND WHY MEAN-REVERTING (AN OU/AR(1) PROCESS), NOT A
RANDOM WALK
------------------------------------------------------------------------------
The standard weak-turbulence model for received irradiance (Rytov theory)
treats intensity as log-normally distributed: I = exp(X), X ~ N(mu,
sigma_ln^2). Unlike the RANDOM WALK used for slow drift in
`trajectory.py` -- which is deliberately non-stationary, because thermal
creep genuinely accumulates without bound -- scintillation is a
STATIONARY process: it fluctuates around a stable long-run mean flux,
never wandering permanently away from it. An Ornstein-Uhlenbeck process
(here, its exact discrete-time equivalent, an AR(1) process in log-
intensity) is the right model for exactly that behaviour: mean-reverting,
with a well-defined stationary variance, rather than growing without
bound like a random walk would.

WHY THE FLUCTUATIONS ARE TEMPORALLY CORRELATED, NOT DRAWN INDEPENDENTLY
PER FRAME
-------------------------------------------------------------------------------
Turbulent eddies take real physical time to cross the beam path -- the
correlation ("coherence") time this project uses, tau_s = 5 ms, is
representative of published horizontal-path terrestrial free-space-optical
scintillation measurements, and matters concretely here because it is
comparable to, not much faster than, this system's 1 ms frame period:
consecutive frames are genuinely correlated, so several frames in a row
fade or peak together. Modelling scintillation as independent per-frame
noise (the way photon shot noise legitimately IS independent) would
understate its real operational impact -- a multi-frame dropout looks
very different to a tracking loop than the same total noise spread
independently across frames.

WHY mu = -sigma_ln^2 / 2 (NOT ZERO)
----------------------------------------
For a log-normal variable I = exp(X), E[I] = exp(mu + sigma_ln^2/2). If
mu were 0, the MEAN flux multiplier would be systematically above 1 (a
free amplitude gain that has nothing to do with turbulence). Setting
mu = -sigma_ln^2/2 exactly cancels that so E[I] = 1: scintillation changes
flux's VARIANCE over time without silently changing its long-run average
-- verified numerically in tests/test_scintillation.py, not assumed.

WHY sigma_ln = 0.4 AND tau_s = 5 ms
------------------------------------------
No site-specific turbulence profile (Cn^2) is available for an actual
Transcelestial deployment, so these cannot be derived. They are instead
placed against published FSO measurements and checked to sit inside
them, rather than picked freely:

  sigma_ln = 0.4 gives a scintillation index
  sigma_I^2 = exp(sigma_ln^2) - 1 = 0.174.
  sigma_ln = 0.6 (used only for the stress case in
  experiments/exp04a_scintillation.py) gives 0.433.

  Published context: the log-normal model this module uses is the
  standard one for WEAK turbulence, valid while the scintillation index
  stays below about 0.75, and measured FSO link indices span roughly
  0.083 (32 cm aperture) to 0.71 (5 cm aperture) depending on aperture
  and geometry. Both values used here fall inside that measured span and
  below the log-normal validity limit, so the model and the parameter
  are consistent with each other.

  tau_s = 5 ms sits inside the published coherence-time range for
  atmospheric fading, described as "a few ms" up to "typically around
  10 ms".

Caveat on tau_s: 5 ms puts this process's spectral corner near
1/(2*pi*tau) = 32 Hz, while some FSO sources describe scintillation's
upper frequency limit as "hundreds of Hz". These are different
quantities, the corner where power begins rolling off versus where it
finally dies, so they do not conflict. tau_s is bounded by the published
range, not pinned to a single measurement.

Both are exposed as parameters specifically so a real deployment's site
survey data could replace them without changing this module's structure.

Sources: Free-space optical communication through atmospheric turbulence
channels (Zhu and Kahn); Channel Measurement and Markov Modeling of an
Urban Free-Space Optical Link (JOCN 4(10) 836).
"""

from __future__ import annotations

import numpy as np


def generate_scintillation(
    n_frames: int, dt_s: float, sigma_ln: float = 0.4, tau_s: float = 5e-3,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a per-frame flux multiplier time series from a mean-
    reverting log-normal (AR(1)-in-log-intensity) scintillation model.

    Multiply a Simulator's flux argument by this array's values, frame by
    frame, to apply scintillation. Mean multiplier is 1.0 by construction
    (see module docstring); ``sigma_ln`` sets the fluctuation's magnitude
    and ``tau_s`` its correlation ("coherence") time.
    """
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma_ln**2
    phi = np.exp(-dt_s / tau_s)
    sigma_eps = sigma_ln * np.sqrt(1.0 - phi**2)

    log_intensity = np.empty(n_frames, dtype=np.float64)
    # Drawn from the process's own STATIONARY distribution, not X=mu, so
    # there is no artificial "spin-up" transient at the start of the
    # sequence where variance is too low.
    log_intensity[0] = rng.normal(mu, sigma_ln)
    innovations = rng.normal(0.0, sigma_eps, n_frames - 1)
    for i in range(1, n_frames):
        log_intensity[i] = mu + phi * (log_intensity[i - 1] - mu) + innovations[i - 1]

    return np.exp(log_intensity)
