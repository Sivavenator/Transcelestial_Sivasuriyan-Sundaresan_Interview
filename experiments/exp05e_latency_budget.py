"""Experiment 05e -- latency budget, full photon-to-estimate path (brief
§5): combine exposure, sensor readout/data-transfer, and compute (already
measured in exp02) into one budget, and be explicit about the difference
between LATENCY (one photon's serial journey to an estimate) and
THROUGHPUT (the 1 kHz sustained frame rate the brief actually requires).

WHY READOUT+TRANSFER IS A ROUND, EXPLICITLY-LABELLED BOUND, NOT A
PRECISE DERIVED NUMBER
------------------------------------------------------------------------------
Two real datasheet points exist for a genuinely comparable real camera
(C-BLUE One, a CMOS camera built specifically for laser guide-star
wavefront sensing on Extremely Large Telescopes -- a closely related
application to this project's fine-tracking sensor): a 256x256 ROI reads
out at 1862 fps (537 us/frame), and a 128x128 ROI at 4400 fps (227
us/frame). Extrapolating a precise per-row time from only two rounded
datasheet figures would imply more precision than the source actually
supports. Per the user's explicit direction, this is instead stated as a
single ROUND, CONSERVATIVE bound: READOUT_TRANSFER_BOUND_US = 200 --
comfortably above what either real datapoint would predict for this
project's much smaller window (21-41 rows, well under the 128-256 row
range those figures come from), and explicitly labelled as a rough bound
rather than a derived estimate. Data-transfer time itself is not
separately bounded because it is negligible by comparison: even a
modest machine-vision interface (the same camera uses CoaXPress 2.0 or
10GigE, tens of Gbps class) moves a few-KB ROI in low tens of
microseconds, well inside the 200 us bound already allotted to readout.

WHY LATENCY AND THROUGHPUT ARE REPORTED SEPARATELY, NOT CONFLATED
-------------------------------------------------------------------------
Summing exposure + readout/transfer + compute SERIALLY gives the LATENCY
of one photon's journey to a position estimate -- but a real system does
not need to run these stages serially frame after fread to sustain 1 kHz
THROUGHPUT. Exposure for frame N+1 can begin while frame N is still being
read out, transferred, or processed (pipelining) -- a standard technique,
not a novel claim -- so sustaining 1 kHz only requires each INDIVIDUAL
stage's own duration to fit within the 1 ms period, not the full serial
sum. Reporting only the serial sum against 1 ms, without this
distinction, would either wrongly conclude the system cannot run at
1 kHz at all (if the naive sum exceeds 1 ms, which it does for the
median case here) or wrongly imply zero-latency operation -- neither is
what actually happens in a pipelined real-time system, and conflating
the two would be a real, avoidable analysis error.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

EXPOSURE_US = 1000.0  # sim.exposure_s = 1e-3, fixed by the brief's 1 kHz spec
READOUT_TRANSFER_BOUND_US = 200.0  # round, conservative bound -- see module docstring
FRAME_PERIOD_US = 1000.0  # 1 kHz


def run() -> dict:
    with open(RESULTS_DIR / "exp02_realtime.json") as fh:
        compute = json.load(fh)

    methods = ["centroid", "fit", "matched"]
    results: dict = {
        "exposure_us": EXPOSURE_US,
        "readout_transfer_bound_us": READOUT_TRANSFER_BOUND_US,
        "frame_period_us": FRAME_PERIOD_US,
        "methods": {},
    }
    for m in methods:
        c = compute[m]
        serial_median = EXPOSURE_US + READOUT_TRANSFER_BOUND_US + c["median_us"]
        serial_p99 = EXPOSURE_US + READOUT_TRANSFER_BOUND_US + c["p99_us"]
        serial_max = EXPOSURE_US + READOUT_TRANSFER_BOUND_US + c["max_us"]
        # THROUGHPUT criterion: exposure (1000us) consumes the ENTIRE frame
        # period by construction (sim.exposure_s=1e-3 IS the full 1kHz
        # period, not a fraction of it) -- so readout+compute cannot run
        # serially after exposure at all; they must run in parallel with
        # the NEXT frame's exposure (a standard global-shutter sensor with
        # a separate readout node) for 1kHz to be sustainable at all. The
        # correct throughput check is therefore whether readout+transfer+
        # compute TOGETHER fit inside that next 1000us exposure window --
        # not exposure+readout, and not compute alone.
        pipelined_median = READOUT_TRANSFER_BOUND_US + c["median_us"]
        pipelined_p99 = READOUT_TRANSFER_BOUND_US + c["p99_us"]
        pipelined_max = READOUT_TRANSFER_BOUND_US + c["max_us"]
        results["methods"][m] = {
            "compute_median_us": c["median_us"], "compute_p99_us": c["p99_us"], "compute_max_us": c["max_us"],
            "serial_latency_median_us": serial_median,
            "serial_latency_p99_us": serial_p99,
            "serial_latency_max_us": serial_max,
            "pipelined_readout_plus_compute_median_us": pipelined_median,
            "pipelined_readout_plus_compute_p99_us": pipelined_p99,
            "pipelined_readout_plus_compute_max_us": pipelined_max,
            "fits_throughput_median": pipelined_median < FRAME_PERIOD_US,
            "fits_throughput_p99": pipelined_p99 < FRAME_PERIOD_US,
            "fits_throughput_max": pipelined_max < FRAME_PERIOD_US,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp05e_latency_budget.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    methods = ["centroid", "fit", "matched"]
    labels = {"centroid": "centroid", "fit": "Gaussian fit", "matched": "matched filter"}
    colors = {"centroid": "#c0392b", "fit": "#27ae60", "matched": "#2166ac"}

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 3.0], hspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    x = list(range(len(methods)))
    width = 0.6
    exposure = [results["exposure_us"]] * len(methods)
    readout = [results["readout_transfer_bound_us"]] * len(methods)
    compute_median = [results["methods"][m]["compute_median_us"] for m in methods]

    ax0.bar(x, exposure, width, label="exposure (1 ms, fixed)", color="#7f8c8d")
    ax0.bar(x, readout, width, bottom=exposure, label="readout+transfer (<=200 us bound)", color="#f39c12")
    bottoms = [e + r for e, r in zip(exposure, readout)]
    ax0.bar(x, compute_median, width, bottom=bottoms, label="compute (median, exp02)", color=[colors[m] for m in methods], alpha=0.85)
    ax0.axhline(results["frame_period_us"], color="black", lw=1.5, linestyle="--", label="1 kHz frame period (1000 us)")
    ax0.set_xticks(x)
    ax0.set_xticklabels([labels[m] for m in methods])
    ax0.set_ylabel("serial latency, median case (microseconds)")
    ax0.set_title("Photon-to-estimate SERIAL latency budget (worst-case, no pipelining)")
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(alpha=0.3, axis="y")

    lines = []
    for m in methods:
        r = results["methods"][m]
        lines.append(
            f"  {labels[m]:14s}: compute median={r['compute_median_us']:7.1f}us p99={r['compute_p99_us']:7.1f}us max={r['compute_max_us']:8.1f}us"
            f"  -> serial latency median={r['serial_latency_median_us']:7.1f}us"
        )
    fit_r = results["methods"]["fit"]
    explanation = (
        "What we see:\n"
        "  Exposure (1 ms, fixed by the brief's own 1 kHz spec) alone already equals the full frame period -- before\n"
        "  readout, transfer, or compute are even added. The serial sum for every method exceeds 1 ms.\n"
        "\n"
        "What we can derive:\n" + "\n".join(lines) + "\n"
        "  1. A NAIVE, non-pipelined implementation cannot sustain 1 kHz for ANY estimator, including the cheapest\n"
        "     (centroid) -- exposure alone consumes the entire 1 ms budget, so serial latency exceeds 1 ms\n"
        "     regardless of which algorithm is chosen. This is not an algorithm problem to solve in software.\n"
        "  2. Because exposure alone already consumes the WHOLE 1 ms period (this project's exposure_s=1e-3 IS the\n"
        "     full frame period, not a fraction of it), readout and compute cannot run serially after exposure at\n"
        "     all -- they must overlap with the NEXT frame's exposure (a standard global-shutter sensor with a\n"
        "     separate readout node, not a novel claim) for 1 kHz to be sustainable at all. The correct throughput\n"
        "     check is therefore whether readout+transfer+compute TOGETHER fit inside that next 1 ms exposure\n"
        f"     window: centroid={results['methods']['centroid']['pipelined_readout_plus_compute_median_us']:.0f}us,\n"
        f"     fit={fit_r['pipelined_readout_plus_compute_median_us']:.0f}us, matched=\n"
        f"     {results['methods']['matched']['pipelined_readout_plus_compute_median_us']:.0f}us at the median -- all fit\n"
        f"     comfortably. But the fit's own measured p99 ({fit_r['pipelined_readout_plus_compute_p99_us']:.0f}us) and max\n"
        f"     ({fit_r['pipelined_readout_plus_compute_max_us']:.0f}us) push close to or past the 1000us window on its\n"
        "     worst frames, consistent with exp02's own finding that the fit's tail is a genuine throughput risk --\n"
        "     the other two methods have no such risk even at their own measured maximums.\n"
        "  3. Where the algorithm 'sits' in the full path: compute is the smallest, most controllable piece of the\n"
        "     budget for the centroid and matched filter, but the Gaussian fit's occasional slow frame is the ONE\n"
        "     place in this whole pipeline where the algorithm choice itself, not the fixed sensor timing, is what\n"
        "     could threaten sustained 1 kHz operation -- a concrete, quantified version of the tradeoff already\n"
        "     identified qualitatively in §2d."
    )
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.text(
        0.0, 1.0, explanation, transform=ax1.transAxes, fontsize=8.8,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp05e_latency_budget.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp05e] exposure={results['exposure_us']:.0f}us readout/transfer<={results['readout_transfer_bound_us']:.0f}us")
    for m, r in results["methods"].items():
        print(f"  {m:10s} compute_median={r['compute_median_us']:.1f}us serial_latency_median={r['serial_latency_median_us']:.1f}us")


if __name__ == "__main__":
    run()
