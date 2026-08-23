import numpy as np
import pytest

from sptrack.simulate import Simulator


def test_render_produces_a_valid_dn_frame():
    sim = Simulator(shape=(25, 25), seed=1)
    frame = sim.render(x0=12.0, y0=12.0, flux=2000.0)
    assert frame.shape == (25, 25)
    assert frame.min() >= 0
    assert frame.max() <= 2**sim.bit_depth - 1
    assert np.all(frame == np.round(frame))  # DN values are integer-valued


def test_fixed_unit_properties_persist_across_renders():
    sim = Simulator(shape=(20, 20), hot_fraction=0.05, prnu_sigma=0.02, seed=2)
    hot_mask_before = sim.hot_mask.copy()
    prnu_map_before = sim.prnu_map.copy()
    sigma_before = sim.sigma

    sim.render(x0=10.0, y0=10.0, flux=1000.0)
    sim.render(x0=10.3, y0=9.8, flux=500.0)

    assert np.array_equal(sim.hot_mask, hot_mask_before)
    assert np.array_equal(sim.prnu_map, prnu_map_before)
    assert sim.sigma == sigma_before


def test_render_is_reproducible_with_the_same_seed():
    sim_a = Simulator(shape=(15, 15), seed=42)
    sim_b = Simulator(shape=(15, 15), seed=42)
    frame_a = sim_a.render(x0=7.0, y0=7.0, flux=1500.0)
    frame_b = sim_b.render(x0=7.0, y0=7.0, flux=1500.0)
    assert np.array_equal(frame_a, frame_b)


def test_dn_to_electrons_inverts_the_gain_and_pedestal():
    sim = Simulator(shape=(5, 5), gain_e_per_dn=10.0, black_level_dn=100.0, seed=3)
    dn = np.array([[100.0, 150.0], [90.0, 200.0]])
    electrons = sim.dn_to_electrons(dn)
    expected = (dn - 100.0) * 10.0
    assert np.array_equal(electrons, expected)


def test_full_chain_statistics_match_the_combined_noise_budget():
    # The end-to-end check: every noise source has been verified in
    # isolation, but never all together in the actual order render() uses.
    # flux=0 removes the spot entirely, isolating background + dark current
    # + read noise + quantization on a flat field -- exactly the pieces this
    # test can cleanly predict. hot_fraction=0 and prnu_sigma=0 remove the
    # two FIXED per-unit effects, which are already covered by the test
    # above and would only add unpredictable per-pixel structure here.
    background_e = 1000.0
    dark_rate_e_per_s = 5000.0
    exposure_s = 0.01
    mean_dark = dark_rate_e_per_s * exposure_s  # 50
    sigma_read_e = 5.0
    gain = 10.0

    sim = Simulator(
        shape=(10, 10),
        background_e=background_e,
        gradient_frac=0.0,
        dark_rate_e_per_s=dark_rate_e_per_s,
        exposure_s=exposure_s,
        hot_fraction=0.0,
        prnu_sigma=0.0,
        sigma_read_e=sigma_read_e,
        gain_e_per_dn=gain,
        bit_depth=16,        # far above this range, so saturation never confounds the result
        black_level_dn=0.0,  # mean ~1050 e- is nowhere near 0, so no pedestal needed here
        seed=99,
    )

    n_trials = 200
    frames = np.stack([sim.render(x0=5.0, y0=5.0, flux=0.0) for _ in range(n_trials)])
    electrons = sim.dn_to_electrons(frames)
    pooled = electrons.ravel()

    expected_mean = background_e + mean_dark
    expected_var = background_e + mean_dark + sigma_read_e**2 + gain**2 / 12.0

    # SE(pooled mean) ~= sqrt(expected_var / n_total) ~= sqrt(1083/20000) ~= 0.23;
    # tolerance below is ~10x that. The variance's own standard error has no
    # simple closed form here (it's a mix of Poisson, Gaussian, and uniform
    # sources, not one clean distribution), so a pragmatic 5% relative
    # tolerance is used instead of a derived one -- still tight enough to
    # catch a real ordering or unit-conversion bug in the chain.
    assert pooled.mean() == pytest.approx(expected_mean, abs=2.5)
    assert pooled.var() == pytest.approx(expected_var, rel=0.05)
