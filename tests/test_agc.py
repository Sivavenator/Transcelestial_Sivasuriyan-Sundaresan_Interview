import numpy as np
import pytest

from sptrack.agc import AutoExposureController
from sptrack.simulate import Simulator


def _make_sim(seed=1):
    return Simulator(
        shape=(41, 41), background_e=30.0, sigma_read_e=5.0,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=seed,
    )


def test_converges_to_target_band_after_a_large_brightness_jump():
    sim = _make_sim()
    ctrl = AutoExposureController(bit_depth=sim.bit_depth, black_level_dn=sim.black_level_dn, target_frac=0.8)
    x0, y0 = 20.3, 19.7
    true_flux = 1e6  # would saturate outright at gain=1.0

    peak_dns = []
    for _ in range(15):
        dn = sim.render(x0, y0, true_flux * ctrl.gain)
        peak_dns.append(dn.max())
        ctrl.update(dn.max())

    target_dn = sim.black_level_dn + ctrl.target_frac * (2**sim.bit_depth - 1 - sim.black_level_dn)
    # last few frames should be converged, close to the target
    assert abs(np.mean(peak_dns[-5:]) - target_dn) < 200


def test_gain_never_exceeds_configured_bounds():
    ctrl = AutoExposureController(bit_depth=12, black_level_dn=100.0, gain_min=0.01, gain_max=100.0)
    for peak_dn in [1.0, 4095.0, 1.0, 4095.0, 1.0]:
        g = ctrl.update(peak_dn)
        assert ctrl.gain_min <= g <= ctrl.gain_max


def test_per_step_change_never_exceeds_max_step_ratio():
    ctrl = AutoExposureController(bit_depth=12, black_level_dn=100.0, max_step_ratio=3.0)
    prev_gain = ctrl.gain
    new_gain = ctrl.update(4095.0)  # extreme overexposure, wants a big correction
    ratio = prev_gain / new_gain if new_gain < prev_gain else new_gain / prev_gain
    assert ratio <= 3.0 + 1e-9


def test_dim_scene_increases_gain():
    ctrl = AutoExposureController(bit_depth=12, black_level_dn=100.0)
    new_gain = ctrl.update(peak_dn=150.0)  # barely above pedestal, far under target
    assert new_gain > 1.0


def test_bright_scene_decreases_gain():
    ctrl = AutoExposureController(bit_depth=12, black_level_dn=100.0)
    new_gain = ctrl.update(peak_dn=4095.0)  # fully saturated
    assert new_gain < 1.0
