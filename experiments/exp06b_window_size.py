"""Experiment 06b -- window half-width: the one tuning knob this project
fixed at 9 px everywhere and never measured.

WHAT THE WINDOW DOES
---------------------------
Every estimator here first crops a (2*half_width+1) square around a prior
position and works only inside it. That crop sets what information the
estimator is allowed to use, and it trades two effects against each
other:

  Enlarging it   adds pixels that carry real spot flux, which is
                 information, but also adds background pixels that carry
                 only noise. In a centre-of-mass sum, a noisy pixel far
                 from the centre contributes its noise multiplied by its
                 distance from the centre, so outer pixels inject
                 variance with a long lever arm.

  Shrinking it   removes those noisy outer pixels but starts truncating
                 the spot itself, discarding real signal and clipping the
                 PSF tails asymmetrically once the window is comparable
                 to the spot width.

WHY THE CRLB IS PLOTTED ALONGSIDE, AND WHY IT DOES NOT SHARE THE OPTIMUM
------------------------------------------------------------------------------
`crlb.position_crlb` is evaluated over the same window. Adding pixels can
only add Fisher information, never remove it, so the bound falls
monotonically with window size: for an efficient estimator, bigger is
always better. Any estimator that instead has an interior optimum is
therefore losing information somewhere, and the gap between its curve and
the bound localises where. That contrast is the point of putting both on
one axis.

WHY THREE SNR VALUES
---------------------------
The balance between "extra pixels carry signal" and "extra pixels carry
noise" depends on how bright the spot is relative to the background, so a
single-SNR sweep would report an optimum without showing that it moves.
10, 50 and 200 span the range this project characterises elsewhere.

WHY shape=(31, 31)
-------------------------
The sweep runs to half_width=12, which needs a 25x25 window. The 21x21
canvas used in exp01 would silently clamp the largest windows against the
frame edge (`extract_window` clips rather than failing), turning a window
size study into a study of edge clamping. 31x31 leaves the largest tested
window fully inside the frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.crlb import position_crlb
from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.estimators.matched_filter import matched_filter_estimate
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

METHODS = ["centroid", "fit", "matched"]
COLORS = {"centroid": "#c0392b", "fit": "#27ae60", "matched": "#2166ac"}
MARKERS = {"centroid": "v", "fit": "^", "matched": "s"}
LABELS = {"centroid": "centroid", "fit": "Gaussian fit", "matched": "matched filter"}

SHAPE = (31, 31)
X0, Y0 = 15.3, 14.7
BACKGROUND_E = 30.0
SIGMA_READ_E = 5.0
HALF_WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
SNRS = [10.0, 50.0, 200.0]
N_TRIALS = 300
PROJECT_HALF_WIDTH = 9


def run() -> dict:
    sim = Simulator(
        shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=777,
    )
    results: dict = {
        "half_widths": HALF_WIDTHS, "snrs": SNRS, "n_trials": N_TRIALS,
        "sigma": float(sim.sigma), "project_half_width": PROJECT_HALF_WIDTH,
        "by_snr": {},
    }

    for snr in SNRS:
        flux = snr_to_flux(
            snr, sim.sigma, BACKGROUND_E,
            sim.dark_rate_e_per_s * sim.exposure_s, SIGMA_READ_E, sim.gain_e_per_dn,
        )
        entry: dict = {"flux": float(flux), "crlb": []}
        for m in METHODS:
            entry[f"{m}_std"] = []

        for hw in HALF_WIDTHS:
            side = 2 * hw + 1
            crlb_x, _ = position_crlb(
                (side, side), hw + (X0 - round(X0)), hw + (Y0 - round(Y0)),
                flux, BACKGROUND_E, sim.sigma, SIGMA_READ_E**2,
            )
            entry["crlb"].append(float(crlb_x))

            errs = {m: [] for m in METHODS}
            for _ in range(N_TRIALS):
                frame = sim.dn_to_electrons(sim.render(X0, Y0, flux))
                c = centroid_estimate(frame, hw, prior=(X0, Y0))
                g = gaussian_fit_estimate(frame, hw, sim.sigma, SIGMA_READ_E**2, prior=(X0, Y0))
                mf = matched_filter_estimate(frame, hw, sim.sigma, prior=(X0, Y0))
                for name, est in zip(METHODS, [c, g, mf]):
                    if est.ok:
                        errs[name].append(est.x - X0)
            for m in METHODS:
                a = np.array(errs[m])
                entry[f"{m}_std"].append(float(a.std()) if a.size > 10 else float("nan"))

        for m in METHODS:
            arr = np.array(entry[f"{m}_std"])
            best_i = int(np.nanargmin(arr))
            proj_i = HALF_WIDTHS.index(PROJECT_HALF_WIDTH)
            entry[f"{m}_best_hw"] = HALF_WIDTHS[best_i]
            entry[f"{m}_best_std"] = float(arr[best_i])
            entry[f"{m}_project_std"] = float(arr[proj_i])
            entry[f"{m}_penalty_pct"] = float(100.0 * (arr[proj_i] - arr[best_i]) / arr[best_i])
        results["by_snr"][str(snr)] = entry

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp06b_window_size.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    hws = np.array(results["half_widths"])
    sigma = results["sigma"]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2.7], hspace=0.45, wspace=0.3)

    for i, snr in enumerate(results["snrs"]):
        e = results["by_snr"][str(snr)]
        ax = fig.add_subplot(gs[0, i])
        ax.plot(hws, np.array(e["crlb"]) * 1000, "-", color="black", lw=1.5, label="CRLB over the window")
        for m in METHODS:
            ax.plot(hws, np.array(e[f"{m}_std"]) * 1000, MARKERS[m] + "-", color=COLORS[m], label=LABELS[m], ms=4)
        ax.axvline(results["project_half_width"], color="black", lw=1.0, linestyle=":", label="this project (hw=9)")
        ax.set_xlabel("window half-width (px)")
        ax.set_ylabel("position std (millipixels)")
        ax.set_title(f"SNR = {snr:.0f}")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

    e50 = results["by_snr"]["50.0"]
    lines = [f"  {'':16s}{'best hw':>9}{'best std':>11}{'std at hw=9':>13}{'penalty':>10}"]
    for snr in results["snrs"]:
        e = results["by_snr"][str(snr)]
        lines.append(f"  SNR = {snr:.0f}")
        for m in METHODS:
            lines.append(
                f"    {LABELS[m]:14s}{e[f'{m}_best_hw']:>9d}"
                f"{e[f'{m}_best_std']*1000:>10.1f}{e[f'{m}_project_std']*1000:>13.1f}"
                f"{e[f'{m}_penalty_pct']:>+9.0f}%"
            )

    explanation = (
        "What we see:\n"
        "  The CRLB falls monotonically with window size at every SNR: more pixels can only add information, so\n"
        "  for an efficient estimator a bigger window is never worse. The Gaussian fit and matched filter track\n"
        "  that shape and flatten out. The centroid does not: it has a clear interior minimum and then gets\n"
        "  steadily worse as the window grows, diverging from the bound.\n"
        "\n"
        "What we can derive:\n" + "\n".join(lines) + "\n"
        "  1. The centroid is the only estimator here that needs the window tuned. Its optimum sits near\n"
        f"     hw=4 to 5, which is about 2.3 to 2.9 sigma at sigma={sigma:.2f} px, and it degrades in both directions.\n"
        "     Below that the spot is truncated; above it, each added background pixel contributes noise weighted\n"
        "     by its distance from the centre, so variance grows with window size rather than shrinking.\n"
        "  2. The Gaussian fit and matched filter are close to insensitive above hw=4. The fit weights each pixel\n"
        "     by 1/(model + read variance) and its model predicts almost no signal in the outer pixels, so those\n"
        "     pixels are down-weighted automatically instead of being allowed to inject lever-armed noise. This is\n"
        "     the same weighting that makes the fit efficient in exp01, seen from a different axis.\n"
        f"  3. This project fixed hw=9 everywhere without measuring it. For the fit and matched filter that costs\n"
        f"     little ({e50['fit_penalty_pct']:+.0f}% and {e50['matched_penalty_pct']:+.0f}% at SNR=50). For the centroid it costs {e50['centroid_penalty_pct']:+.0f}% at SNR=50.\n"
        "     The centroid efficiency of 0.63 reported in exp01 is therefore measured at a window size that is\n"
        "     poor for the centroid specifically, and part of that gap is the window choice rather than the\n"
        "     estimator. The three-way ordering in exp01 still holds, but its margin over the centroid is\n"
        "     overstated by this configuration and should be quoted with the window size attached.\n"
        "  4. The optimum moves with SNR, so there is no single correct window. A deployed system that runs across\n"
        "     a wide SNR range would need either an SNR-dependent window or an estimator that does not care, which\n"
        "     is a further argument for the fit or the matched filter over the centroid beyond raw precision."
    )
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.text(
        0.0, 1.0, explanation, transform=ax3.transAxes, fontsize=8.6,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp06b_window_size.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp06b] window sweep, sigma={results['sigma']:.2f}, {results['n_trials']} trials/point")
    for snr in results["snrs"]:
        e = results["by_snr"][str(snr)]
        print(f"  SNR={snr:.0f}")
        for m in METHODS:
            print(f"    {LABELS[m]:15s} best hw={e[f'{m}_best_hw']:2d} "
                  f"({e[f'{m}_best_std']*1000:7.1f} mpx)   hw=9: {e[f'{m}_project_std']*1000:7.1f} mpx "
                  f"({e[f'{m}_penalty_pct']:+.0f}%)")


if __name__ == "__main__":
    run()
