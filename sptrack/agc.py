"""Auto-exposure/gain control (brief §5): a closed-loop controller that
retargets the sensor's effective gain each frame to keep the spot's peak
pixel in a good operating band, so precision stays good across a large
range of scene brightness rather than only near whatever single level a
FIXED exposure/gain setting happened to be tuned for.

WHY THIS MATTERS, EMPIRICALLY, BEFORE ASSUMING IT DOES
------------------------------------------------------------
Checked directly before building anything: this project's Poisson-
weighted Gaussian fit is naturally quite ROBUST to modest saturation
(clipping a handful of the brightest pixels barely moves the fit, since
those pixels are already down-weighted by their own large predicted
variance) -- position error stayed within tens of millipixels even with
over 10% of the estimation window clipped. So the case for AGC here is
NOT "without it the system catastrophically fails" -- it doesn't, within
the ranges tested. The real case is EFFICIENCY across a WIDE brightness
range: a fixed setting is well-tuned for only a narrow band of scene
brightness (too dim outside it wastes achievable SNR; too bright wastes
the sensor's dynamic range once clipping sets in, since additional real
photons above the clip level contribute nothing extra). AGC's job is to
keep RETARGETING the operating point as brightness changes, so precision
stays close to what is actually achievable at each brightness level,
across a range no single fixed setting can cover well.

WHY THIS CONTROLS AN "EFFECTIVE GAIN" MULTIPLIER ON FLUX, NOT SEPARATELY
MODELLING EXPOSURE TIME vs. ANALOG GAIN
------------------------------------------------------------------------------
A real AGC loop can adjust exposure time, analog/digital gain, or an
attenuating element, each with slightly different secondary effects
(exposure time also scales dark current; analog gain also scales read
noise's effective DN contribution). Modelling all of that separately
would require reworking `simulate.py`'s noise chain more deeply than this
feature warrants. This module instead controls one multiplicative
"effective gain" applied directly to the flux passed into
`Simulator.render` -- a legitimate, honestly-simplified proxy for any of
those real mechanisms (all of them ultimately change how many real
photons/electrons the peak pixel accumulates, which is what this
controller actually reacts to), with the caveat stated plainly: dark
current's own scaling with a real exposure-time change is NOT modelled
here (already established elsewhere in this project to be a small
contributor at the default exposure/dark-rate values, so this
simplification does not distort the main result).

WHY A PROPORTIONAL CONTROLLER WITH A BOUNDED PER-STEP CHANGE, NOT AN
INSTANT JUMP TO THE "CORRECT" GAIN
------------------------------------------------------------------------------
A real exposure/gain actuator cannot jump to an arbitrary new setting in
zero time (mechanical or electronic slew limits), and an unbounded
correction would also overreact to a single noisy peak-pixel reading
(photon shot noise on the peak pixel itself makes any single frame's
"ideal gain" estimate noisy). `max_step_ratio` bounds how much the gain
can change in one frame, which was checked directly to still converge
within a handful of frames after a large brightness step (see
`experiments/exp05a_auto_exposure.py`) while damping single-frame noise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoExposureController:
    bit_depth: int
    black_level_dn: float
    target_frac: float = 0.8
    max_step_ratio: float = 4.0
    gain_min: float = 1e-6
    gain_max: float = 1e6
    gain: float = 1.0

    def __post_init__(self) -> None:
        headroom = (2**self.bit_depth - 1) - self.black_level_dn
        self._target_above_pedestal = self.target_frac * headroom

    def update(self, peak_dn: float) -> float:
        """Given the previous frame's peak DN, update and return the new
        effective gain multiplier for the next frame."""
        above_pedestal = max(peak_dn - self.black_level_dn, 1.0)
        ratio = self._target_above_pedestal / above_pedestal
        ratio = min(max(ratio, 1.0 / self.max_step_ratio), self.max_step_ratio)
        self.gain = min(max(self.gain * ratio, self.gain_min), self.gain_max)
        return self.gain
