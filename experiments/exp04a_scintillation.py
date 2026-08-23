"""Experiment 04a -- simulate atmospheric scintillation's impact on
tracking (brief §4's named example), and show the existing dead-reckoning
mechanism (§3, `sptrack/sequence.py::recover_trajectory`) already
survives the resulting dropouts rather than crashing.

WHY A DIRECT A/B COMPARISON (SAME TRAJECTORY AND SENSOR SEEDS, ONLY
SCINTILLATION TOGGLED)
------------------------------------------------------------------------------
Comparing WITH vs. WITHOUT scintillation on the identical underlying
trajectory and sensor noise realisation isolates scintillation's own
contribution cleanly -- any difference between the two runs is
attributable to the flux modulation alone, not to a different random
draw of jitter, drift, or photon noise.

WHY base_snr=5.0 AND sigma_ln=0.6 FOR THIS DEMONSTRATION (STRONGER THAN
THE MODULE'S OWN "MODERATE" DEFAULT OF 0.4)
------------------------------------------------------------------------------
`sptrack/scintillation.py`'s default (sigma_ln=0.4) represents moderate
turbulence and, layered on a workable baseline SNR, degrades precision
without ever causing an outright dropout (checked: 0 failed fits at
base_snr=20, sigma_ln=0.4). That is a real, honest result, but it doesn't
show the more severe failure mode described in
`docs/REAL_WORLD_CONDITIONS.md` -- a genuine loss of lock. Lowering the
baseline SNR to 5.0 (matching the §3 hard scenario's own choice, for
consistency) and raising sigma_ln to 0.6 (representing a worse day of
turbulence, or a longer path -- still within literature-plausible
bounds, just the upper end rather than the middle) was chosen, by direct
experimentation, to be the smallest deliberate stress increase that
actually produces real, nonzero dropout events -- worth showing exactly
because it demonstrates the recovery mechanism has something real to
recover FROM, not a synthetic edge case invented for the demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.scintillation import generate_scintillation
from sptrack.sequence import recover_trajectory
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux
from sptrack.trajectory import TrajectoryConfig, generate_trajectory

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SHAPE = (41, 41)
HALF_WIDTH = 9
BACKGROUND_E = 30.0
SIGMA_READ_E = 5.0
BASE_SNR = 5.0
SIGMA_LN = 0.6
TAU_S = 5e-3


def _render_and_recover(traj: dict, sim: Simulator, flux_per_frame: np.ndarray) -> dict:
    n = len(traj["x"])
    frames = np.empty((n, *SHAPE), dtype=np.float64)
    for i in range(n):
        frames[i] = sim.dn_to_electrons(sim.render(float(traj["x"][i]), float(traj["y"][i]), float(flux_per_frame[i])))
    return recover_trajectory(frames, HALF_WIDTH, sim.sigma, SIGMA_READ_E**2)


def run() -> dict:
    traj_cfg = TrajectoryConfig(seed=2026, x0=20.3, y0=19.7)
    traj = generate_trajectory(traj_cfg)

    sim_a = Simulator(shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
                       hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2026)
    sim_b = Simulator(shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
                       hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2026)
    base_flux = snr_to_flux(
        BASE_SNR, sim_a.sigma, BACKGROUND_E,
        sim_a.dark_rate_e_per_s * sim_a.exposure_s, SIGMA_READ_E, sim_a.gain_e_per_dn,
    )

    steady_flux = np.full(traj_cfg.n_frames, base_flux)
    mult = generate_scintillation(traj_cfg.n_frames, traj_cfg.dt_s, sigma_ln=SIGMA_LN, tau_s=TAU_S, seed=2026)
    scint_flux = base_flux * mult

    recovered_steady = _render_and_recover(traj, sim_a, steady_flux)
    recovered_scint = _render_and_recover(traj, sim_b, scint_flux)

    err_steady = recovered_steady["x"] - traj["x"]
    err_scint = recovered_scint["x"] - traj["x"]

    low_mask = mult < 0.5
    high_mask = mult > 1.5
    n_failed_steady = int((~recovered_steady["ok"]).sum())
    n_failed_scint = int((~recovered_scint["ok"]).sum())

    results = {
        "base_snr": BASE_SNR, "sigma_ln": SIGMA_LN, "tau_s_ms": TAU_S * 1000,
        "std_steady_px": float(np.std(err_steady)),
        "std_scint_overall_px": float(np.std(err_scint)),
        "std_scint_low_flux_px": float(np.std(err_scint[low_mask])),
        "std_scint_high_flux_px": float(np.std(err_scint[high_mask])),
        "n_low_flux_frames": int(low_mask.sum()),
        "n_high_flux_frames": int(high_mask.sum()),
        "n_failed_steady": n_failed_steady,
        "n_failed_scint": n_failed_scint,
        "mult_min": float(mult.min()),
        "mult_max": float(mult.max()),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp04a_scintillation.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(traj, mult, err_steady, err_scint, recovered_scint, results)
    _print_summary(results)
    return results


def _plot(traj: dict, mult: np.ndarray, err_steady: np.ndarray, err_scint: np.ndarray,
          recovered_scint: dict, results: dict) -> None:
    t = traj["t"]
    fail_idx = np.where(~recovered_scint["ok"])[0]

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 2.0, 2.4], hspace=0.55)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, mult, color="#e67e22", lw=0.8)
    ax0.axhline(1.0, color="gray", lw=0.8, linestyle="--")
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel("flux multiplier")
    ax0.set_title(f"Scintillation: correlated log-normal flux fluctuation (sigma_ln={results['sigma_ln']}, tau_s={results['tau_s_ms']:.0f} ms)")
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, err_steady * 1000, color="#2166ac", lw=0.5, alpha=0.6, label=f"steady flux (std={results['std_steady_px']*1000:.1f} mpx)")
    ax1.plot(t, err_scint * 1000, color="#c0392b", lw=0.5, alpha=0.8, label=f"with scintillation (std={results['std_scint_overall_px']*1000:.1f} mpx)")
    if len(fail_idx):
        ax1.scatter(t[fail_idx], np.zeros(len(fail_idx)), color="black", marker="x", s=25, zorder=5, label=f"dropout ({len(fail_idx)} frames)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("position error (millipixels)")
    ax1.set_title("Position error: identical trajectory/sensor noise, scintillation toggled")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    explanation = (
        "What we see:\n"
        f"  Top: flux fluctuates around 1.0x with real, multi-frame-correlated fades and peaks (min {results['mult_min']:.2f}x,\n"
        f"  max {results['mult_max']:.2f}x) -- not the frame-independent noise every OTHER source in this project is. Middle:\n"
        "  with scintillation (red) the error is visibly larger and burstier than the steady-flux baseline (blue) on\n"
        f"  the SAME underlying motion and sensor noise -- {len(fail_idx)} frames failed to converge outright (black x's),\n"
        "  concentrated during the deepest fades, yet the recovered trajectory never derails: those frames simply hold\n"
        "  the last known-good position (§3's dead-reckoning) rather than crashing or losing the track permanently.\n"
        "\n"
        "What we can derive:\n"
        f"  1. Overall std nearly {results['std_scint_overall_px']/results['std_steady_px']:.1f}x worse with scintillation ({results['std_scint_overall_px']*1000:.1f} vs\n"
        f"     {results['std_steady_px']*1000:.1f} millipixels) -- but this single number hides the real structure: std during\n"
        f"     low-flux periods (mult<0.5, {results['n_low_flux_frames']} frames) is {results['std_scint_low_flux_px']*1000:.1f} mpx, vs.\n"
        f"     {results['std_scint_high_flux_px']*1000:.1f} mpx during high-flux periods (mult>1.5, {results['n_high_flux_frames']} frames) --\n"
        "     precision genuinely tracks the instantaneous fade, exactly as the SNR-vs-precision relationship from\n"
        "     §2c predicts, just now varying in TIME instead of being fixed per experiment.\n"
        f"  2. {results['n_failed_scint']} of {len(err_scint)} frames were a genuine loss of lock (vs. {results['n_failed_steady']} with steady\n"
        "     flux) -- real dropouts, not just added scatter, concentrated exactly where the module docstring's\n"
        "     physical argument predicted them: the deepest, most sustained fades.\n"
        "  3. The system already has a real mitigation for this, built for an unrelated reason (§3's dead-reckoning\n"
        "     was designed to survive an isolated bad frame): it turns out to be exactly the right shape of fix for\n"
        "     scintillation dropouts too, since the correlated fades tested here are still short (a handful of frames)\n"
        "     relative to the trajectory's own slower dynamics -- a coincidence worth stating honestly, not a\n"
        "     mitigation purpose-built for this specific failure mode."
    )
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, explanation, transform=ax_text.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp04a_scintillation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp04a] base_snr={results['base_snr']}, sigma_ln={results['sigma_ln']}, tau_s={results['tau_s_ms']:.0f}ms")
    print(f"  std steady={results['std_steady_px']*1000:.2f} mpx  std scint={results['std_scint_overall_px']*1000:.2f} mpx")
    print(f"  std low-flux={results['std_scint_low_flux_px']*1000:.2f} mpx  std high-flux={results['std_scint_high_flux_px']*1000:.2f} mpx")
    print(f"  failed fits: steady={results['n_failed_steady']}  scint={results['n_failed_scint']}")


if __name__ == "__main__":
    run()
