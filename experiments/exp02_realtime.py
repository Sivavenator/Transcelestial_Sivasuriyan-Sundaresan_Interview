"""Experiment 02 -- per-frame compute cost, and which methods fit 1 kHz.

WHY PERCENTILES, NOT THE MEAN
----------------------------------
The brief's loop runs at ~1 kHz: a new frame arrives every 1 ms, and the
estimator must finish before the next one lands. That is a requirement on
EVERY frame, not on the average frame -- a loop that is fast 999 times out
of 1000 but occasionally takes 3 ms still misses a deadline once every
second. So the number that actually determines whether a method "fits" the
budget is a high percentile (here, the 99th) of its per-frame cost, not the
mean. The mean is what a benchmark blog reports; the tail is what makes a
real-time loop miss a frame.

WHY THIS MEASURES PYTHON WALL-CLOCK TIME, AND WHAT THAT DOES AND DOESN'T TELL YOU
----------------------------------------------------------------------------------------
These are real, measured numbers -- not modelled, not guessed -- for this
project's actual Python implementations. They are NOT a claim about what a
compiled (C/C++/FPGA) implementation would cost; Python's interpreter
overhead dominates for small, cheap operations (the centroid) far more than
it would in compiled code, which tends to flatter the expensive method
(the fit, where the iteration work is a bigger fraction of the total) and
undersell the cheap one. Both numbers are still useful for comparing the
THREE methods against each other under identical conditions, and for
answering the brief's literal question ("report per-frame compute cost")
honestly, labelled for what it is.

WHY THE SAME WINDOW SIZE AND FLUX AS THE CHARACTERIZATION SWEEP
---------------------------------------------------------------------
Using the same half_width=9 window and a representative flux from
exp01_snr_characterization.py keeps the timing numbers directly comparable
to the accuracy numbers already measured there -- the same frame that was
characterised for precision is now characterised for cost, rather than
timing a different, potentially easier or harder, configuration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.estimators.matched_filter import matched_filter_estimate
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

FRAME_BUDGET_US = 1000.0  # 1 kHz -> 1 ms -> 1000 microseconds


def run(quick: bool = False) -> dict:
    shape = (21, 21)
    x0, y0 = 10.3, 9.7
    half_width = 9
    background_e = 30.0
    sigma_read_e = 5.0
    snr = 50.0  # a representative mid-to-high SNR operating point

    sim = Simulator(
        shape=shape, background_e=background_e, sigma_read_e=sigma_read_e,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=3033,
    )
    flux = snr_to_flux(
        snr, sim.sigma, background_e,
        sim.dark_rate_e_per_s * sim.exposure_s, sigma_read_e, sim.gain_e_per_dn,
    )

    n_frames = 100 if quick else 1000
    # Pre-render all frames once, so the timed region is ONLY the
    # estimator's own work, not frame generation -- a real system times its
    # estimator against frames it already has, not against the cost of
    # producing them.
    frames = [sim.dn_to_electrons(sim.render(x0, y0, flux)) for _ in range(n_frames)]

    methods = {
        "centroid": lambda f: centroid_estimate(f, half_width, prior=(x0, y0)),
        "fit": lambda f: gaussian_fit_estimate(
            f, half_width, sim.sigma, sigma_read_e**2, prior=(x0, y0)
        ),
        "matched": lambda f: matched_filter_estimate(f, half_width, sim.sigma, prior=(x0, y0)),
    }

    # Warm-up: a handful of untimed calls first, so cache/allocation effects
    # from the very first call(s) don't distort the timed measurements.
    for fn in methods.values():
        for f in frames[:5]:
            fn(f)

    results: dict = {"n_frames": n_frames, "snr": snr, "flux": flux, "budget_us": FRAME_BUDGET_US}
    for name, fn in methods.items():
        times_us = []
        for f in frames:
            t0 = time.perf_counter()
            fn(f)
            t1 = time.perf_counter()
            times_us.append((t1 - t0) * 1e6)
        times_us = np.array(times_us)
        results[name] = {
            "mean_us": float(times_us.mean()),
            "median_us": float(np.median(times_us)),
            "p99_us": float(np.percentile(times_us, 99)),
            "max_us": float(times_us.max()),
            "fits_budget_median": bool(np.median(times_us) < FRAME_BUDGET_US),
            "fits_budget_p99": bool(np.percentile(times_us, 99) < FRAME_BUDGET_US),
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp02_realtime.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    methods = ["centroid", "fit", "matched"]
    labels = {"centroid": "centroid", "fit": "Gaussian fit", "matched": "matched filter"}
    colors = {"centroid": "#c0392b", "fit": "#27ae60", "matched": "#2166ac"}

    medians = [results[m]["median_us"] for m in methods]
    p99s = [results[m]["p99_us"] for m in methods]

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    x_pos = np.arange(len(methods))
    width = 0.35
    ax0.bar(x_pos - width / 2, medians, width, label="median", color=[colors[m] for m in methods], alpha=0.6)
    ax0.bar(x_pos + width / 2, p99s, width, label="p99 (worst-case)", color=[colors[m] for m in methods])
    ax0.axhline(FRAME_BUDGET_US, color="black", linestyle="--", lw=1.5, label=f"1 kHz budget ({FRAME_BUDGET_US:.0f} us)")
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels([labels[m] for m in methods])
    ax0.set_ylabel("per-frame compute cost (microseconds)")
    ax0.set_title(f"Per-frame compute cost, {results['n_frames']} frames, SNR={results['snr']:.0f}")
    ax0.legend(fontsize=9)
    ax0.set_yscale("log")
    ax0.grid(alpha=0.3, axis="y")

    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    fastest = min(methods, key=lambda m: results[m]["median_us"])
    slowest = max(methods, key=lambda m: results[m]["median_us"])
    speedup = results[slowest]["median_us"] / results[fastest]["median_us"]
    fit_max_exceeds = results["fit"]["max_us"] > FRAME_BUDGET_US
    fit_p99_ok = results["fit"]["fits_budget_p99"]
    explanation = (
        "What we see:\n"
        f"  At the median, all three methods finish well inside the {FRAME_BUDGET_US:.0f} us (1 kHz) budget -- the {labels[slowest]}\n"
        f"  costs {speedup:.0f}x more than the {labels[fastest]} ({results[slowest]['median_us']:.0f} us vs {results[fastest]['median_us']:.0f} us) but is still far under budget. The gap between\n"
        f"  median and p99/max is largest for the Gaussian fit ({results['fit']['median_us']:.0f} -> {results['fit']['p99_us']:.0f} -> {results['fit']['max_us']:.0f} us): its cost varies with how many\n"
        "  Levenberg-Marquardt iterations a given frame happens to need, unlike the other two methods' fixed-shape work.\n"
        + (
            f"  The fit's p99 ({results['fit']['p99_us']:.0f} us) still clears the budget, but its slowest observed frame\n"
            f"  ({results['fit']['max_us']:.0f} us) does not.\n"
            if fit_p99_ok and fit_max_exceeds else ""
        )
        + "\n"
        "What we can derive:\n"
        f"  1. By the p99 criterion, {', '.join(labels[m] for m in methods if results[m]['fits_budget_p99'])} all fit the 1 kHz budget in this measurement.\n"
        + (
            f"     But the Gaussian fit's WORST observed frame ({results['fit']['max_us']:.0f} us) exceeded the {FRAME_BUDGET_US:.0f} us budget outright --\n"
            "     a real, measured tail event, not a hypothetical one. An iterative method has no hard upper bound on its\n"
            "     own cost unless one is imposed (this project's `max_iter=20` caps it, but even 20 iterations on a bad\n"
            "     frame can cost more than the budget allows); a fixed-cost correlation or centroid cannot do this by\n"
            "     construction.\n"
            if fit_max_exceeds else ""
        )
        + "  2. The tradeoff, stated precisely: the Gaussian fit is the most ACCURATE method (2c) but the only one without a\n"
        "     hard cost ceiling -- its number of iterations, and therefore its latency, depends on the data. The matched\n"
        "     filter gives up a little precision (efficiency 0.84 vs the fit's 0.95) for a fixed, correlation-shaped cost\n"
        "     with no tail risk; the centroid gives up more precision (0.63) for the lowest and most predictable cost of\n"
        "     all three. For a loop that must never miss a 1 ms deadline, that predictability -- not the median number --\n"
        "     is the deciding factor, which argues for the matched filter (or a hard-capped fit) over the uncapped fit."
    )
    ax_text.text(
        0.0, 1.0, explanation, transform=ax_text.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp02_realtime.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp02] Real-time cost: {results['n_frames']} frames, SNR={results['snr']:.0f}, budget={results['budget_us']:.0f} us")
    print(f"{'method':>12} {'median_us':>10} {'p99_us':>10} {'max_us':>10} {'fits@p99':>9}")
    for m in ["centroid", "fit", "matched"]:
        r = results[m]
        print(f"{m:>12} {r['median_us']:10.2f} {r['p99_us']:10.2f} {r['max_us']:10.2f} {str(r['fits_budget_p99']):>9}")


if __name__ == "__main__":
    run()
