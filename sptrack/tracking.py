"""Temporal filtering of the per-frame position estimates: a
constant-velocity Kalman filter and its fixed-gain steady-state
equivalent, the alpha-beta filter.

WHAT PROBLEM THIS SOLVES, AND WHAT IT DOES NOT
------------------------------------------------------
Everything upstream of this module estimates position from ONE frame in
isolation. Each estimate carries an independent measurement error whose
size is characterised in exp01. A temporal filter adds the one piece of
information a single-frame estimator cannot use: that the spot was
somewhere specific a millisecond ago, and that real hardware cannot move
arbitrarily far in a millisecond. Combining a prediction from the
previous state with the current measurement produces an estimate with
lower variance than either alone.

The filter cannot improve the information content of a frame. It trades
variance against lag. That trade is the entire engineering content of
this module and it is measured in experiments/exp07_kalman_tracking.py
rather than asserted here.

WHY A CONSTANT-VELOCITY MODEL
------------------------------------
The state is [position, velocity] per axis, propagated as

    x(k+1) = x(k) + v(k)*dt
    v(k+1) = v(k) + (process noise)

Justification from the motion actually present (trajectory.py): the
dominant slow component is drift, modelled as a random walk, and a random
walk in position is exactly what a constant-velocity model with process
noise on velocity is designed to track. Adding an acceleration state
would buy nothing against a random walk and would cost an extra state to
estimate from the same data, so a constant-acceleration model is not
used.

WHY THE PROCESS NOISE USES THE CONTINUOUS WHITE-NOISE-ACCELERATION FORM
-------------------------------------------------------------------------------
Process noise is not a free tuning knob dropped into the covariance. It
is derived by integrating a white acceleration disturbance of power
spectral density q over one sample interval, which gives the standard
result

    Q = q * [[dt^3/3, dt^2/2],
             [dt^2/2, dt   ]]

The off-diagonal terms matter: position and velocity errors accumulated
over the same interval are correlated, and a diagonal Q would assert they
are not. q has physical units of px^2/s^3 and is the single knob that
sets how much the filter trusts its own model versus the measurement.

WHY MEASUREMENT NOISE R IS NOT GUESSED
-----------------------------------------------
R is the variance of the incoming position measurement, and this project
has already measured it: exp01 reports per-estimator standard deviation
against SNR, and crlb.py gives the theoretical floor for the same
configuration. R is therefore set from a known quantity rather than
tuned. A filter given the wrong R is mis-weighted in a way that no amount
of tuning q can fix, so this matters more than it looks.

WHY THE ALPHA-BETA FILTER IS INCLUDED ALONGSIDE
--------------------------------------------------------
For a constant-velocity model with stationary noise, the Kalman gain
converges to a fixed value. Running the covariance recursion every frame
then computes the same two numbers repeatedly. The alpha-beta filter uses
those steady-state gains directly, which removes the matrix algebra from
the per-frame path entirely and reduces the update to a handful of scalar
operations. In a 1 kHz loop on constrained hardware that difference is
the argument for it. Including both allows the cost and the accuracy to
be compared rather than assumed equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanTracker1D:
    """Constant-velocity Kalman filter for one axis.

    ``dt_s`` is the frame interval, ``process_psd`` is q in px^2/s^3, and
    ``meas_var_px2`` is R, the variance of the incoming position
    measurement in px^2.
    """

    dt_s: float
    process_psd: float
    meas_var_px2: float
    initial_pos: float = 0.0
    initial_vel: float = 0.0
    initial_pos_var: float = 1.0
    initial_vel_var: float = 1.0

    state: np.ndarray = field(init=False)
    cov: np.ndarray = field(init=False)
    _F: np.ndarray = field(init=False, repr=False)
    _Q: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dt = self.dt_s
        self.state = np.array([self.initial_pos, self.initial_vel], dtype=np.float64)
        self.cov = np.diag([self.initial_pos_var, self.initial_vel_var]).astype(np.float64)
        self._F = np.array([[1.0, dt], [0.0, 1.0]])
        self._Q = self.process_psd * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
        )

    def predict(self) -> None:
        self.state = self._F @ self.state
        self.cov = self._F @ self.cov @ self._F.T + self._Q

    def update(self, measurement: float) -> float:
        """Fold in one position measurement and return the filtered position.

        The measurement matrix is H = [1, 0], meaning the sensor observes
        position and not velocity. Written out for that specific H the
        update reduces to scalar arithmetic, avoiding the matrix inverse
        and several matrix products that a general implementation would
        perform every frame.
        """
        innovation = measurement - self.state[0]
        innovation_cov = self.cov[0, 0] + self.meas_var_px2
        gain = self.cov[:, 0] / innovation_cov
        self.state = self.state + gain * innovation
        self.cov = self.cov - np.outer(gain, self.cov[0, :])
        return float(self.state[0])

    def step(self, measurement: float) -> float:
        """One predict/update cycle. Returns the filtered position."""
        self.predict()
        return self.update(measurement)

    def steady_state_gains(self, n_burn: int = 5000) -> tuple[float, float]:
        """Run the covariance recursion to convergence and return the
        equivalent alpha-beta gains.

        The covariance recursion depends only on F, Q and R, not on the
        measurements, so it can be run forward without any data to find
        where the gain settles.
        """
        cov = self.cov.copy()
        alpha = beta = 0.0
        for _ in range(n_burn):
            cov = self._F @ cov @ self._F.T + self._Q
            innovation_cov = cov[0, 0] + self.meas_var_px2
            gain = cov[:, 0] / innovation_cov
            cov = cov - np.outer(gain, cov[0, :])
            alpha, beta = float(gain[0]), float(gain[1]) * self.dt_s
        return alpha, beta


@dataclass
class AlphaBetaTracker1D:
    """Fixed-gain steady-state equivalent of the constant-velocity Kalman
    filter. No covariance propagation, two scalar gains."""

    dt_s: float
    alpha: float
    beta: float
    initial_pos: float = 0.0
    initial_vel: float = 0.0

    pos: float = field(init=False)
    vel: float = field(init=False)

    def __post_init__(self) -> None:
        self.pos = float(self.initial_pos)
        self.vel = float(self.initial_vel)

    def step(self, measurement: float) -> float:
        pred_pos = self.pos + self.vel * self.dt_s
        pred_vel = self.vel
        residual = measurement - pred_pos
        self.pos = pred_pos + self.alpha * residual
        self.vel = pred_vel + (self.beta / self.dt_s) * residual
        return self.pos


def filter_sequence(tracker, measurements: np.ndarray) -> np.ndarray:
    """Run a tracker across a measurement sequence, returning the filtered
    series. The tracker is seeded from the first measurement so the
    transient at the start is not counted as tracking error."""
    out = np.empty(len(measurements), dtype=np.float64)
    if len(measurements) == 0:
        return out
    if isinstance(tracker, KalmanTracker1D):
        tracker.state[0] = measurements[0]
    else:
        tracker.pos = float(measurements[0])
    out[0] = measurements[0]
    for i in range(1, len(measurements)):
        out[i] = tracker.step(float(measurements[i]))
    return out


def sinusoid_response(filtered: np.ndarray, truth: np.ndarray, freq_hz: float, dt_s: float) -> dict:
    """Measure a filter's gain and phase lag at one frequency.

    Projects both series onto sin and cos at ``freq_hz`` (a single-bin
    Fourier projection) and compares the resulting complex amplitudes. Gain
    below 1 means the filter attenuated the tone; a negative phase means
    the filtered series lags the truth.
    """
    n = len(truth)
    t = np.arange(n) * dt_s
    ref = np.exp(-2j * np.pi * freq_hz * t)
    a_true = np.sum((truth - truth.mean()) * ref)
    a_filt = np.sum((filtered - filtered.mean()) * ref)
    if abs(a_true) == 0:
        return {"gain": float("nan"), "phase_deg": float("nan"), "lag_ms": float("nan")}
    ratio = a_filt / a_true
    phase_deg = float(np.degrees(np.angle(ratio)))
    lag_ms = -phase_deg / 360.0 / freq_hz * 1000.0
    return {"gain": float(abs(ratio)), "phase_deg": phase_deg, "lag_ms": lag_ms}
