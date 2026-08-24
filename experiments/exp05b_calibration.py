"""Experiment 05b -- calibration (brief §5): measure the position-precision
effect of bias-frame subtraction, flat-field correction, and lens-
distortion correction, each against the specific defect it targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.calibration import (
    apply_radial_distortion,
    correct_radial_distortion,
    estimate_bias_frame,
    estimate_flat_field,
)
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.simulate import Simulator

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

HALF_WIDTH = 9


def run_bias() -> dict:
    shape = (31, 31)
    x0, y0 = 15.3, 14.7
    # A dim spot (lower SNR, chosen empirically -- see ASSUMPTIONS.md --
    # so the hot pixel's fixed ~50-electron excess is proportionally
    # significant against the spot's own signal, rather than getting lost
    # in it) with the hot pixel close to the spot, inside the window.
    flux = 3000.0
    sim = Simulator(shape=shape, background_e=30.0, sigma_read_e=5.0, hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=10)
    sim.hot_mask = np.zeros(shape, dtype=bool)
    sim.hot_mask[int(y0) - 2, int(x0) + 2] = True

    bias_map = estimate_bias_frame(sim, n_frames=100)

    n_trials = 300
    err_raw, err_corrected = [], []
    for _ in range(n_trials):
        e_image = sim.dn_to_electrons(sim.render(x0, y0, flux))
        est_raw = gaussian_fit_estimate(e_image, HALF_WIDTH, sim.sigma, sim.sigma_read_e**2, prior=(x0, y0))
        est_corrected = gaussian_fit_estimate(e_image - bias_map, HALF_WIDTH, sim.sigma, sim.sigma_read_e**2, prior=(x0, y0))
        if est_raw.ok:
            err_raw.append(est_raw.x - x0)
        if est_corrected.ok:
            err_corrected.append(est_corrected.x - x0)

    return {
        "err_raw": err_raw, "err_corrected": err_corrected,
        "bias_raw_mean": float(np.mean(err_raw)), "bias_raw_std": float(np.std(err_raw)),
        "bias_corrected_mean": float(np.mean(err_corrected)), "bias_corrected_std": float(np.std(err_corrected)),
    }


def run_flat_field() -> dict:
    shape = (31, 31)
    flux = 3000.0
    sim = Simulator(shape=shape, background_e=30.0, sigma_read_e=5.0, hot_fraction=0.0, prnu_sigma=0.02, gradient_frac=0.0, seed=11)
    flat_map = estimate_flat_field(sim, flat_level_e=20000.0, n_frames=13)

    offsets = np.linspace(0.0, 1.0, 11)
    raw_bias, corrected_bias = [], []
    cx, cy = 15.0, 15.0
    for off in offsets:
        x0, y0 = cx + off, cy
        n_trials = 100
        errs_raw, errs_corr = [], []
        for _ in range(n_trials):
            e_image = sim.dn_to_electrons(sim.render(x0, y0, flux))
            est_raw = gaussian_fit_estimate(e_image, HALF_WIDTH, sim.sigma, sim.sigma_read_e**2, prior=(x0, y0))
            est_corr = gaussian_fit_estimate(e_image / flat_map, HALF_WIDTH, sim.sigma, sim.sigma_read_e**2, prior=(x0, y0))
            if est_raw.ok:
                errs_raw.append(est_raw.x - x0)
            if est_corr.ok:
                errs_corr.append(est_corr.x - x0)
        raw_bias.append(float(np.mean(errs_raw)))
        corrected_bias.append(float(np.mean(errs_corr)))

    return {"offsets": offsets.tolist(), "raw_bias": raw_bias, "corrected_bias": corrected_bias}


def run_distortion() -> dict:
    shape = (301, 301)
    x0c, y0c = 150.0, 150.0
    sigma = 1.75
    flux = 1.0e7  # very high SNR, isolates the geometric effect from noise

    from sptrack.psf import render_spot

    r_norms = np.linspace(0.0, 1.0, 11)
    r_max = np.hypot(x0c, y0c)  # the half-diagonal, matching apply_radial_distortion's own normalisation
    raw_err, corrected_err = [], []
    for rn in r_norms:
        # Swept along the DIAGONAL, not a single axis: r_max is defined as
        # the half-diagonal, so a pure-axis sweep would run off the edge
        # of a square canvas well before r_norm=1 (caught directly: an
        # earlier x-axis-only version produced NaNs from r_norm=0.8
        # onward, where the true position had already left the frame).
        true_x = x0c + rn * r_max / np.sqrt(2)
        true_y = y0c + rn * r_max / np.sqrt(2)
        obs_x, obs_y = apply_radial_distortion(true_x, true_y, shape)

        img = render_spot(shape, obs_x, obs_y, flux, sigma) + 30.0
        est = gaussian_fit_estimate(img, HALF_WIDTH, sigma, 0.0, prior=(obs_x, obs_y))

        # radial magnitude of the error, not just the x-component, since
        # the sweep now moves diagonally (both x and y shift together)
        raw_err.append(float(np.hypot(est.x - true_x, est.y - true_y)))
        cx, cy = correct_radial_distortion(est.x, est.y, shape)
        corrected_err.append(float(np.hypot(cx - true_x, cy - true_y)))

    return {"r_norms": r_norms.tolist(), "raw_err": raw_err, "corrected_err": corrected_err}


def run() -> dict:
    bias_results = run_bias()
    flat_results = run_flat_field()
    dist_results = run_distortion()

    results = {"bias": bias_results, "flat_field": flat_results, "distortion": dist_results}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp05b_calibration.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 2.2, 2.2, 3.0], hspace=0.6)

    b = results["bias"]
    ax0 = fig.add_subplot(gs[0])
    ax0.hist(np.array(b["err_raw"]) * 1000, bins=30, alpha=0.6, color="#c0392b", label=f"raw (mean={b['bias_raw_mean']*1000:+.1f} mpx)")
    ax0.hist(np.array(b["err_corrected"]) * 1000, bins=30, alpha=0.6, color="#27ae60", label=f"bias-corrected (mean={b['bias_corrected_mean']*1000:+.1f} mpx)")
    ax0.set_xlabel("position error (millipixels)")
    ax0.set_ylabel("trial count")
    ax0.set_title("Bias-frame subtraction: correcting an in-window hot pixel")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    f = results["flat_field"]
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(f["offsets"], np.array(f["raw_bias"]) * 1000, "o-", color="#c0392b", label="raw (PRNU uncorrected)")
    ax1.plot(f["offsets"], np.array(f["corrected_bias"]) * 1000, "s-", color="#27ae60", label="flat-field corrected")
    ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax1.set_xlabel("sub-pixel offset (px)")
    ax1.set_ylabel("centroid bias (millipixels)")
    ax1.set_title("Flat-field correction: PRNU-induced position-dependent bias")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    d = results["distortion"]
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(d["r_norms"], d["raw_err"], "o-", color="#c0392b", label="raw (uncorrected)")
    ax2.plot(d["r_norms"], d["corrected_err"], "s-", color="#27ae60", label="distortion-corrected")
    ax2.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax2.set_xlabel("radial distance from centre (fraction of half-diagonal)")
    ax2.set_ylabel("position error magnitude (px)")
    ax2.set_title("Lens-distortion correction: geometric error vs. radial distance")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    max_raw_dist_err = max(abs(v) for v in d["raw_err"])
    max_corr_dist_err = max(abs(v) for v in d["corrected_err"])
    explanation = (
        "What we see:\n"
        "  Top: an uncorrected in-window hot pixel drags the fit toward it (raw, red); bias-frame subtraction\n"
        "  (green) removes that pull almost entirely. Middle: PRNU's position-dependent bias (raw, red) oscillates\n"
        "  with sub-pixel offset; flat-field correction (green) flattens it toward zero at every offset tested.\n"
        "  Bottom: uncorrected lens distortion (raw, red) grows with distance from the optical centre -- zero at\n"
        f"  the centre, up to {max_raw_dist_err:.3f} px at the edge; distortion correction (green) removes it almost\n"
        f"  entirely (down to {max_corr_dist_err:.5f} px, limited only by the fixed-point correction's own precision).\n"
        "\n"
        "What we can derive:\n"
        f"  1. Bias: raw mean={b['bias_raw_mean']*1000:+.1f} mpx -> corrected mean={b['bias_corrected_mean']*1000:+.1f} mpx --\n"
        "     a single, deliberately-placed hot pixel inside the window pulls the fit measurably; the calibration\n"
        "     map removes that specific, structured pull without needing to know in advance where a hot pixel is.\n"
        "  2. Flat-field: RMS bias (distance from the true zero, across all tested offsets) shrinks from\n"
        f"     {np.sqrt(np.mean(np.array(f['raw_bias'])**2))*1000:.1f} to {np.sqrt(np.mean(np.array(f['corrected_bias'])**2))*1000:.1f} millipixels after correction --\n"
        "     a {:.1f}x reduction. PRNU's bias doesn't just get smaller at one lucky offset, it flattens across the\n".format(
            np.sqrt(np.mean(np.array(f['raw_bias'])**2)) / max(np.sqrt(np.mean(np.array(f['corrected_bias'])**2)), 1e-12)
        ) +
        "     whole sub-pixel range tested (one offset, 0.7px, still shows a residual bump -- not perfectly flat,\n"
        "     but still a clear net improvement).\n"
        "  3. Distortion is the only one of these three that is a PURE GEOMETRIC effect, present even with a\n"
        "     perfect, noiseless estimator -- unlike every other bias source characterised in this project, which\n"
        "     comes from noise or an approximation breaking down. At -0.1% (this project's chosen precision-lens\n"
        f"     magnitude), the uncorrected effect ({max_raw_dist_err:.3f} px at the frame edge) is still meaningfully\n"
        "     larger than this project's best single-frame estimator precision at high SNR (a few millipixels,\n"
        "     §2c) -- worth correcting even at a magnitude that sounds small on paper."
    )
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    ax3.text(
        0.0, 1.0, explanation, transform=ax3.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp05b_calibration.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    b = results["bias"]
    print(f"\n[exp05b] bias: raw={b['bias_raw_mean']*1000:+.2f}mpx corrected={b['bias_corrected_mean']*1000:+.2f}mpx")
    d = results["distortion"]
    print(f"  distortion: max raw err={max(abs(v) for v in d['raw_err']):.4f}px  max corrected err={max(abs(v) for v in d['corrected_err']):.6f}px")


if __name__ == "__main__":
    run()
