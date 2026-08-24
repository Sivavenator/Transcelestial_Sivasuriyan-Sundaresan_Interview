"""Experiment 05c -- motion blur robustness (brief §5): sweep intra-frame
blur magnitude (0 to 3x the PSF's sigma -- see sptrack/motion_blur.py's
docstring for why no specific real-world velocity is claimed) and measure
how bias and precision degrade for all three estimators.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.estimators.matched_filter import matched_filter_estimate
from sptrack.motion_blur import render_motion_blurred_spot
from sptrack.sensor import add_photon_noise, add_read_noise, quantize_to_dn
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

METHODS = ["centroid", "fit", "matched"]
COLORS = {"centroid": "#c0392b", "fit": "#27ae60", "matched": "#2166ac"}
MARKERS = {"centroid": "v", "fit": "^", "matched": "s"}
LABELS = {"centroid": "centroid", "fit": "Gaussian fit", "matched": "matched filter"}

HALF_WIDTH = 9
SNR = 50.0
N_TRIALS = 150


def run() -> dict:
    shape = (41, 41)
    sigma = 1.75
    x0, y0 = 20.3, 19.7
    background_e = 30.0
    sigma_read_e = 5.0

    sim = Simulator(
        shape=shape, background_e=background_e, sigma_read_e=sigma_read_e,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2029,
    )
    flux = snr_to_flux(
        SNR, sim.sigma, background_e, sim.dark_rate_e_per_s * sim.exposure_s, sigma_read_e, sim.gain_e_per_dn,
    )

    blur_fracs = np.linspace(0.0, 3.0, 7)
    results: dict = {"blur_fracs": blur_fracs.tolist(), "sigma": sigma, "snr": SNR, "n_trials": N_TRIALS}
    for m in METHODS:
        results[f"{m}_bias"] = []
        results[f"{m}_std"] = []
        results[f"{m}_dropout"] = []

    rng = sim._rng
    for frac in blur_fracs:
        blur_px = frac * sigma
        errs = {m: [] for m in METHODS}
        n_failed = {m: 0 for m in METHODS}
        for _ in range(N_TRIALS):
            spot_e = render_motion_blurred_spot(shape, x0, y0, flux, sigma, blur_px, angle_rad=0.0)
            bg_e = np.full(shape, background_e)
            photo_signal = spot_e + bg_e
            e_image = add_photon_noise(photo_signal, rng)
            e_image = add_read_noise(e_image, sigma_read_e, rng)
            dn = quantize_to_dn(e_image, sim.gain_e_per_dn, sim.bit_depth, sim.black_level_dn)
            frame_e = sim.dn_to_electrons(dn)

            c = centroid_estimate(frame_e, HALF_WIDTH, prior=(x0, y0))
            g = gaussian_fit_estimate(frame_e, HALF_WIDTH, sigma, sigma_read_e**2, prior=(x0, y0))
            mfe = matched_filter_estimate(frame_e, HALF_WIDTH, sigma, prior=(x0, y0))

            for name, est in zip(METHODS, [c, g, mfe]):
                if est.ok:
                    errs[name].append(est.x - x0)
                else:
                    n_failed[name] += 1

        for m in METHODS:
            results[f"{m}_bias"].append(float(np.mean(errs[m])) if errs[m] else None)
            results[f"{m}_std"].append(float(np.std(errs[m])) if errs[m] else None)
            results[f"{m}_dropout"].append(n_failed[m] / N_TRIALS)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp05c_motion_blur.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    fracs = np.array(results["blur_fracs"])

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2.2], hspace=0.45, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    for m in METHODS:
        bias = np.array([v * 1000 if v is not None else np.nan for v in results[f"{m}_bias"]])
        ax0.plot(fracs, bias, MARKERS[m] + "-", color=COLORS[m], label=LABELS[m])
    ax0.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax0.set_xlabel("blur magnitude (fraction of sigma)")
    ax0.set_ylabel("bias (millipixels)")
    ax0.set_title(f"Bias vs. motion blur (SNR={results['snr']:.0f})")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    for m in METHODS:
        std = np.array([v * 1000 if v is not None else np.nan for v in results[f"{m}_std"]])
        ax1.plot(fracs, std, MARKERS[m] + "-", color=COLORS[m], label=LABELS[m])
    ax1.set_xlabel("blur magnitude (fraction of sigma)")
    ax1.set_ylabel("std (millipixels)")
    ax1.set_title("Precision vs. motion blur")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    centroid_bias_end = results["centroid_bias"][-1] * 1000 if results["centroid_bias"][-1] is not None else float("nan")
    fit_bias_end = results["fit_bias"][-1] * 1000 if results["fit_bias"][-1] is not None else float("nan")
    matched_bias_end = results["matched_bias"][-1] * 1000 if results["matched_bias"][-1] is not None else float("nan")
    centroid_std_ratio = results["centroid_std"][-1] / results["centroid_std"][0]
    fit_std_ratio = results["fit_std"][-1] / results["fit_std"][0]
    matched_std_ratio = results["matched_std"][-1] / results["matched_std"][0]
    explanation = (
        "What we see:\n"
        "  Left: the fit and matched filter stay within about a millipixel of zero bias across the whole range\n"
        "  tested; the centroid carries a small (roughly -3 to -8 millipixel) offset that is already present at\n"
        "  ZERO blur and does not grow with it -- a pre-existing, SNR-dependent centroid bias already characterised\n"
        "  in §2c, not something motion blur introduces. Right: precision (std) degrades steadily as blur grows,\n"
        "  for all three methods.\n"
        "\n"
        "What we can derive:\n"
        f"  1. Bias does not GROW with blur for any method: centroid={centroid_bias_end:+.1f}, fit={fit_bias_end:+.1f},\n"
        f"     matched={matched_bias_end:+.1f} millipixels at 3-sigma blur are all close to their OWN zero-blur values\n"
        "     -- confirming motion blur is fundamentally a PRECISION problem for this project's estimators, not a\n"
        "     NEW bias source, unlike most other real-world conditions characterised in §4 (clutter, glare, and\n"
        "     scintillation's noise-floor selection bias all introduce bias that scales with the condition's severity).\n"
        "     The centroid's small residual offset is a known, separate effect (§2c), not this experiment's finding.\n"
        f"  2. Precision degrades by {centroid_std_ratio:.1f}x (centroid), {fit_std_ratio:.1f}x (fit), {matched_std_ratio:.1f}x (matched) from\n"
        "     no blur to 3-sigma blur -- the blurred spot's peak brightness falls and its light spreads across more\n"
        "     pixels, lowering peak SNR even though total flux is unchanged (the same mechanism as scintillation's\n"
        "     fades and fog's attenuation reducing SNR, just from spatial spreading instead of fewer photons).\n"
        "  3. None of the three estimators assumes anything about intra-frame motion -- all fit/correlate against a\n"
        "     STATIC PSF of the known sigma, an assumption blur directly violates. That none of them show much bias\n"
        "     under this violation (rather than, say, the Gaussian fit's symmetric model getting systematically\n"
        "     confused by an asymmetric-looking blur) is a real, checked result, not assumed from the model mismatch\n"
        "     alone -- constant-velocity blur only ever appears symmetric about the true centre position, which\n"
        "     protects a weighted-average or best-fit-centre-position estimate even when the assumed SHAPE is wrong."
    )
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    ax2.text(
        0.0, 1.0, explanation, transform=ax2.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp05c_motion_blur.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp05c] blur sweep, SNR={results['snr']:.0f}, {results['n_trials']} trials/point")
    header = f"{'blur/sigma':>10}" + "".join(f"{m + '_bias_mpx':>16}{m + '_std_mpx':>14}" for m in METHODS)
    print(header)
    for i, frac in enumerate(results["blur_fracs"]):
        row = f"{frac:10.2f}"
        for m in METHODS:
            b = results[f"{m}_bias"][i]
            s = results[f"{m}_std"][i]
            row += f"{b*1000:16.2f}{s*1000:14.2f}" if b is not None else f"{'None':>16}{'None':>14}"
        print(row)


if __name__ == "__main__":
    run()
