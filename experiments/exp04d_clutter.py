"""Experiment 04d -- background clutter / false bright sources (brief §4,
item 3 in docs/REAL_WORLD_CONDITIONS.md): show the acquisition failure
directly, then show the shape-matched mitigation (sptrack/acquisition.py)
resolving it on the identical frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.acquisition import acquire_target, find_local_maxima, psf_shape_score
from sptrack.estimators.base import border_median_background, extract_window, find_brightest_pixel
from sptrack.psf import render_spot

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run() -> dict:
    shape = (61, 61)
    true_sigma = 1.75
    true_xy = (20.3, 19.7)
    true_flux = 3000.0
    clutter_xy = (45.0, 40.0)
    clutter_flux = 40000.0
    clutter_sigma = 5.0
    half_width = 9

    img = render_spot(shape, *true_xy, true_flux, true_sigma)
    img = img + render_spot(shape, *clutter_xy, clutter_flux, clutter_sigma)
    img = img + 30.0

    iy_bright, ix_bright = find_brightest_pixel(img)
    acquired = acquire_target(img, half_width, true_sigma, min_value=100.0)

    candidates = find_local_maxima(img, min_value=100.0, nms_radius=half_width)
    candidate_scores = []
    for cx, cy, v in candidates:
        window, wx0, wy0 = extract_window(img, cx, cy, half_width)
        bg = border_median_background(window)
        score = psf_shape_score(window - bg, true_sigma)
        candidate_scores.append({"x": cx, "y": cy, "peak_value": v, "shape_score": score})

    results = {
        "true_xy": true_xy, "true_flux": true_flux,
        "clutter_xy": clutter_xy, "clutter_flux": clutter_flux, "clutter_sigma": clutter_sigma,
        "brightest_pixel_pick": [int(ix_bright), int(iy_bright)],
        "shape_matched_pick": list(acquired) if acquired else None,
        "candidates": candidate_scores,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp04d_clutter.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(img, results)
    _print_summary(results)
    return results


def _plot(img: np.ndarray, results: dict) -> None:
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2.2], hspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(img, origin="lower", cmap="inferno")
    plt.colorbar(im, ax=ax0, fraction=0.04, label="electrons")
    tx, ty = results["true_xy"]
    cx, cy = results["clutter_xy"]
    ax0.scatter([tx], [ty], marker="o", s=200, facecolors="none", edgecolors="#2ecc71", linewidths=2, label="true laser spot")
    ax0.scatter([cx], [cy], marker="o", s=200, facecolors="none", edgecolors="#3498db", linewidths=2, label="clutter source")
    bx, by = results["brightest_pixel_pick"]
    ax0.scatter([bx], [by], marker="x", s=150, color="red", linewidths=3, label="find_brightest_pixel picks")
    if results["shape_matched_pick"]:
        sx, sy = results["shape_matched_pick"]
        ax0.scatter([sx], [sy], marker="+", s=250, color="white", linewidths=3, label="acquire_target (shape-matched) picks")
    ax0.set_title("Acquisition on a frame with a brighter, wider clutter source present")
    ax0.legend(fontsize=8, loc="upper left")

    lines = [
        f"  candidate ({c['x']:.0f},{c['y']:.0f}): peak={c['peak_value']:7.1f}  shape_score={c['shape_score']:.4f}"
        for c in results["candidates"]
    ]
    explanation = (
        "What we see:\n"
        f"  The clutter source (blue circle, {results['clutter_flux']:.0f}e- total flux, sigma={results['clutter_sigma']:.1f}px) has\n"
        f"  {results['clutter_flux']/results['true_flux']:.1f}x the true spot's ({results['true_flux']:.0f}e-, sigma=1.75px) total flux, but is spread over a much wider\n"
        "  profile -- its PEAK pixel still exceeds the true spot's, because peak brightness falls off with sigma^2\n"
        "  while total flux does not. find_brightest_pixel (red x) is fooled outright and would confidently report a\n"
        "  fully 'successful' (ok=True) position for the WRONG object.\n"
        "\n"
        "What we can derive:\n" + "\n".join(lines) + "\n"
        "  The true spot's window correlates almost perfectly with the assumed Gaussian PSF template (score near 1.0);\n"
        "  the clutter's much wider profile scores far lower -- a clean, physically-grounded separation using\n"
        "  information find_brightest_pixel never looks at (shape, not brightness). acquire_target (white +) correctly\n"
        "  picks the true spot despite it being the DIMMER-peaked candidate -- the mitigation works on exactly the\n"
        "  failure case it was built for, verified on this same frame rather than a separately-constructed easy case."
    )
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, explanation, transform=ax_text.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp04d_clutter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp04d] brightest_pixel_pick={results['brightest_pixel_pick']}  shape_matched_pick={results['shape_matched_pick']}")
    print(f"  true_xy={results['true_xy']}  clutter_xy={results['clutter_xy']}")


if __name__ == "__main__":
    run()
