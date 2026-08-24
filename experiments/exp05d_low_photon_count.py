"""Experiment 05d -- very low photon count robustness (brief §5): extend
exp01's SNR characterization far below its previous floor (SNR=3) into
the regime where the peak pixel's signal is a handful of photons or
fewer, and report each method's SUCCESS rate alongside bias/std/CRLB
efficiency -- exp01 never needed to track dropout rate because nothing
failed outright in the range it tested.

WHY NO NEW PHYSICAL PARAMETER WAS NEEDED HERE (UNLIKE LENS DISTORTION OR
MOTION BLUR)
------------------------------------------------------------------------------
"Very low photon count" is not a new physical effect requiring a fresh,
externally-sourced constant -- it is the far end of the SNR axis this
project has already built and fully justified (`sptrack/snr.py`,
`experiments/exp01_snr_characterization.py`). The sweep range here was
extended purely by asking `snr_to_flux` for lower SNR values and reading
off the resulting peak-pixel photon count -- e.g. SNR=3 (exp01's
previous minimum) corresponds to a peak pixel signal of ~29 photons
above background; SNR=0.1 corresponds to under 1 photon. Both numbers
come directly from this project's own established flux/SNR machinery,
not a new assumption.

WHY background_e AND sigma_read_e ARE LEFT AT THEIR DEFAULTS, NOT
LOWERED TO ISOLATE A "PURER" PHOTON-COUNTING REGIME
------------------------------------------------------------------------------
A deep-space, near-zero-background link would show shot-noise-dominated
behaviour more cleanly at these photon counts. Deliberately NOT modelled
here: lowering background_e or sigma_read_e would itself be a fresh,
unjustified physical assumption about a different deployment scenario,
exactly the kind of guess the standing rule exists to prevent. This
experiment instead asks a narrower, fully-answerable question: how do
THIS project's already-established noise budget and estimators behave as
signal drops toward zero, not what a hypothetically cleaner sensor would
do.
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
from sptrack.snr import peak_pixel_fraction, snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

METHODS = ["centroid", "fit", "matched"]
COLORS = {"centroid": "#c0392b", "fit": "#27ae60", "matched": "#2166ac"}
MARKERS = {"centroid": "v", "fit": "^", "matched": "s"}
LABELS = {"centroid": "centroid", "fit": "Gaussian fit", "matched": "matched filter"}


def run(quick: bool = False) -> dict:
    shape = (21, 21)
    x0, y0 = 10.3, 9.7
    half_width = 9
    background_e = 30.0
    sigma_read_e = 5.0

    sim = Simulator(
        shape=shape, background_e=background_e, sigma_read_e=sigma_read_e,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2030,
    )
    peak_frac = peak_pixel_fraction(sim.sigma)

    snr_targets = np.geomspace(3.0, 0.05, 12)
    n_trials = 60 if quick else 200

    results: dict = {"snr": [], "flux": [], "peak_photons": [], "crlb": [], "n_trials": n_trials}
    for m in METHODS:
        results[f"{m}_bias"] = []
        results[f"{m}_std"] = []
        results[f"{m}_success_rate"] = []

    for snr in snr_targets:
        flux = snr_to_flux(
            snr, sim.sigma, background_e, sim.dark_rate_e_per_s * sim.exposure_s, sigma_read_e, sim.gain_e_per_dn,
        )
        crlb_x, _ = position_crlb(shape, x0, y0, flux, background_e, sim.sigma, sigma_read_e**2)

        errs: dict = {m: [] for m in METHODS}
        n_ok = {m: 0 for m in METHODS}
        for _ in range(n_trials):
            frame_e = sim.dn_to_electrons(sim.render(x0, y0, flux))

            c = centroid_estimate(frame_e, half_width, prior=(x0, y0))
            g = gaussian_fit_estimate(frame_e, half_width, sim.sigma, sigma_read_e**2, prior=(x0, y0))
            mfe = matched_filter_estimate(frame_e, half_width, sim.sigma, prior=(x0, y0))

            for name, est in zip(METHODS, [c, g, mfe]):
                if est.ok:
                    n_ok[name] += 1
                    errs[name].append(est.x - x0)

        results["snr"].append(float(snr))
        results["flux"].append(float(flux))
        results["peak_photons"].append(float(flux * peak_frac))
        results["crlb"].append(float(crlb_x))
        for m in METHODS:
            results[f"{m}_success_rate"].append(n_ok[m] / n_trials)
            results[f"{m}_bias"].append(float(np.mean(errs[m])) if errs[m] else None)
            results[f"{m}_std"].append(float(np.std(errs[m])) if errs[m] else None)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp05d_low_photon_count.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    peak_photons = np.array(results["peak_photons"])
    crlb = np.array(results["crlb"])

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2.4], hspace=0.45, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    for m in METHODS:
        rate = np.array(results[f"{m}_success_rate"]) * 100
        ax0.semilogx(peak_photons, rate, MARKERS[m] + "-", color=COLORS[m], label=LABELS[m])
    ax0.set_xlabel("peak-pixel signal (photons above background)")
    ax0.set_ylabel("success rate (%)")
    ax0.set_title("Success rate vs. photon count")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3, which="both")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.loglog(peak_photons, crlb, "-", color="black", lw=1.5, label="CRLB (theoretical floor)")
    for m in METHODS:
        std = np.array([v if v is not None else np.nan for v in results[f"{m}_std"]])
        ax1.loglog(peak_photons, std, MARKERS[m] + "-", color=COLORS[m], label=LABELS[m])
    ax1.set_xlabel("peak-pixel signal (photons above background)")
    ax1.set_ylabel("std (px), successful trials only")
    ax1.set_title("Precision vs. photon count, against the CRLB")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    # find the photon count where each method's success rate crosses 50%,
    # if it ever does within the tested range -- centroid's never does.
    def crossing(m):
        rates = np.array(results[f"{m}_success_rate"])
        below = np.where(rates < 0.5)[0]
        return peak_photons[below[0]] if len(below) else None

    crossings = {m: crossing(m) for m in METHODS}
    fit_str = f"{crossings['fit']:.2f}" if crossings["fit"] is not None else "never (within range tested)"
    matched_str = f"{crossings['matched']:.2f}" if crossings["matched"] is not None else "never (within range tested)"
    centroid_floor = np.nanmean([v for v in results["centroid_std"] if v is not None][-4:]) * 1000
    explanation = (
        "What we see:\n"
        "  Left: the centroid's success rate never drops below 100% anywhere in this sweep, even at 0.4 photons --\n"
        "  but the right panel shows why that is NOT real robustness: its std plateaus around "
        f"{centroid_floor:.0f} millipixels\n"
        "  regardless of photon count, the window's own noise floor -- it always returns SOME position, most of\n"
        "  which are noise-driven, not signal-driven. The fit and matched filter instead fail outright (ok=False)\n"
        "  once photon count drops far enough, and do so at very DIFFERENT photon counts from each other.\n"
        "\n"
        "What we can derive:\n"
        f"  1. 50%-success crossing: fit~={fit_str} photons, matched~={matched_str} photons -- the matched filter\n"
        "     fails at a MUCH higher photon count than the fit, the opposite of the accuracy ordering already\n"
        "     established at moderate-to-high SNR in §2c (fit > matched filter > centroid). This traces to a real,\n"
        "     checked mechanism: the matched filter's failure test (matched_filter.py) is purely GEOMETRIC -- did\n"
        "     the correlation peak land more than 1px from the correlation window's own edge -- while the fit's\n"
        "     failure is CONVERGENCE-based, an unrelated criterion. At low SNR the correlation surface is\n"
        "     noise-dominated, and a noise-dominated peak lands near that window's edge far more often than a\n"
        "     real, centred signal peak would.\n"
        "  2. The centroid's apparent 'success' at every photon count tested is exactly the same failure mode\n"
        "     already found in §4's fog experiment: dropout_rate alone understates real failure when a method's\n"
        "     'ok' flag doesn't check answer QUALITY, only that a formal criterion (here: positive background-\n"
        "     subtracted flux sum) was met. A real system reading only centroid's ok flag would see 100% uptime\n"
        "     while receiving effectively random positions below a few photons.\n"
        "  3. Failure, where it happens, is not a silently wrong answer for the fit or matched filter -- both\n"
        "     return ok=False rather than a confidently wrong position, consistent with every other dropout\n"
        "     characterised in this project (§4's fog/scintillation). The centroid is the one method here where\n"
        "     'ok=True' cannot be trusted at face value at these photon counts."
    )
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    ax2.text(
        0.0, 1.0, explanation, transform=ax2.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp05d_low_photon_count.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp05d] photon-count sweep, {results['n_trials']} trials/point")
    header = f"{'peak_photons':>13}" + "".join(f"{m + '_rate%':>12}" for m in METHODS)
    print(header)
    for i, pp in enumerate(results["peak_photons"]):
        row = f"{pp:13.3f}" + "".join(f"{results[f'{m}_success_rate'][i]*100:12.1f}" for m in METHODS)
        print(row)


if __name__ == "__main__":
    run()
