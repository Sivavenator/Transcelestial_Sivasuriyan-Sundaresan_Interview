"""Experiment 06a -- pixel locking: does the estimate get pulled toward
pixel centres, and does this project's PSF sampling avoid it?

WHAT PIXEL LOCKING IS
--------------------------
Every other characterisation in this project measures how NOISY an
estimator is. This one measures whether it is systematically WRONG in a
way that depends on where inside a pixel the spot happens to sit. Plot
mean error against true sub-pixel phase and a locking estimator traces an
S-shape: pulled one way just below a pixel centre, the other way just
above. That error does not average away over many frames the way noise
does, because it is a property of the position itself, so it survives
straight into a control loop as a position-dependent offset.

WHY THIS IS MEASURED ON NOISELESS FRAMES FIRST
-----------------------------------------------------
Classical pixel locking is deterministic, not statistical. Rendering the
mean image with no noise and running the estimators on it gives the exact
systematic bias with zero Monte Carlo uncertainty, rather than burying a
1-millipixel effect under a Monte Carlo noise floor.

The noiseless measurement is not sufficient on its own. A Monte Carlo
sweep at the project's operating point is run as well, and it finds a
phase-dependent centroid bias that the noiseless sweep does not predict,
because that bias originates in a nonlinearity (negative-value clipping)
that has no effect without noise present. Both measurements are needed to
characterise the estimator.

WHY SIGMA IS SWEPT, NOT JUST THE PHASE
---------------------------------------------
Pixel locking is fundamentally an UNDERSAMPLING artefact. A spot narrow
enough that a pixel cannot resolve its shape leaves the estimator
guessing between pixel centres. Sweeping only the phase at this project's
fixed sigma=1.75 would show a flat line and prove nothing about whether
the estimators are actually immune or merely untested in the regime where
locking appears. Sweeping sigma as well shows both: where locking really
does appear, and how much margin sigma=1.75 has against it.

SCOPE OF THE GAUSSIAN FIT RESULT
--------------------------------------
The fit's forward model is `psf.pixel_response_1d`, the same erf-based
exact pixel INTEGRAL used to render the frame. Its model matches the
data-generating process exactly, so on a noiseless frame it recovers the
true position to solver tolerance at any sigma, which it does.

This bounds locking due to the pixel-integration approximation only. A
real sensor's PSF is not exactly the assumed Gaussian, and a PSF-mismatch
study would be required to bound locking under model error. That study
has not been run in this project.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.estimators.base import extract_window
from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.estimators.matched_filter import matched_filter_estimate
from sptrack.psf import render_spot
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

METHODS = ["centroid", "fit", "matched"]
COLORS = {"centroid": "#c0392b", "fit": "#27ae60", "matched": "#2166ac"}
MARKERS = {"centroid": "v", "fit": "^", "matched": "s"}
LABELS = {"centroid": "centroid", "fit": "Gaussian fit", "matched": "matched filter"}

SHAPE = (21, 21)
HALF_WIDTH = 9
BACKGROUND_E = 30.0
SIGMA_READ_E = 5.0
NOISELESS_FLUX = 200000.0
PROJECT_SIGMA = 1.75
SIGMAS = [0.4, 0.5, 0.75, 1.0, 1.25, 1.75, 2.5]
N_PHASES = 21
NOISY_SNR = 100.0
NOISY_TRIALS = 400


def _noiseless_bias_curve(sigma: float) -> dict:
    """Exact systematic bias vs sub-pixel phase, no noise at all."""
    phases = np.linspace(0.0, 1.0, N_PHASES)
    out = {m: [] for m in METHODS}
    y0 = 10.0
    for p in phases:
        x0 = 10.0 + p
        img = render_spot(SHAPE, x0, y0, NOISELESS_FLUX, sigma) + BACKGROUND_E
        c = centroid_estimate(img, HALF_WIDTH, prior=(x0, y0))
        g = gaussian_fit_estimate(img, HALF_WIDTH, sigma, 0.0, prior=(x0, y0))
        m = matched_filter_estimate(img, HALF_WIDTH, sigma, prior=(x0, y0))
        for name, est in zip(METHODS, [c, g, m]):
            out[name].append(float(est.x - x0) if est.ok else float("nan"))
    return {"phases": phases.tolist(), **{m: out[m] for m in METHODS}}


def _noisy_bias_curve(sigma: float) -> dict:
    """Monte Carlo bias vs phase at the project's real operating point.

    Also records the window origin at each phase and runs the centroid
    with clipping both on and off, because the first version of this
    experiment found a real phase-dependent centroid bias here that the
    noiseless sweep does not predict, and isolating it needs both.
    """
    sim = Simulator(
        shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=4242,
    )
    flux = snr_to_flux(
        NOISY_SNR, sim.sigma, BACKGROUND_E,
        sim.dark_rate_e_per_s * sim.exposure_s, SIGMA_READ_E, sim.gain_e_per_dn,
    )
    phases = np.linspace(0.0, 1.0, N_PHASES)
    bias = {m: [] for m in METHODS}
    sem = {m: [] for m in METHODS}
    noclip_bias, noclip_sem, window_x0 = [], [], []
    y0 = 10.0
    for p in phases:
        x0 = 10.0 + p
        _, wx0, _ = extract_window(np.zeros(SHAPE), x0, y0, HALF_WIDTH)
        window_x0.append(int(wx0))
        errs = {m: [] for m in METHODS}
        errs_noclip = []
        for _ in range(NOISY_TRIALS):
            frame = sim.dn_to_electrons(sim.render(x0, y0, flux))
            c = centroid_estimate(frame, HALF_WIDTH, prior=(x0, y0))
            cn = centroid_estimate(frame, HALF_WIDTH, prior=(x0, y0), clip_negative=False)
            g = gaussian_fit_estimate(frame, HALF_WIDTH, sim.sigma, SIGMA_READ_E**2, prior=(x0, y0))
            m = matched_filter_estimate(frame, HALF_WIDTH, sim.sigma, prior=(x0, y0))
            for name, est in zip(METHODS, [c, g, m]):
                if est.ok:
                    errs[name].append(est.x - x0)
            if cn.ok:
                errs_noclip.append(cn.x - x0)
        for m in METHODS:
            a = np.array(errs[m])
            bias[m].append(float(a.mean()) if a.size else float("nan"))
            sem[m].append(float(a.std() / np.sqrt(a.size)) if a.size else float("nan"))
        a = np.array(errs_noclip)
        noclip_bias.append(float(a.mean()) if a.size else float("nan"))
        noclip_sem.append(float(a.std() / np.sqrt(a.size)) if a.size else float("nan"))
    return {"phases": phases.tolist(), "flux": float(flux),
            "window_x0": window_x0,
            "centroid_noclip_bias": noclip_bias, "centroid_noclip_sem": noclip_sem,
            **{f"{m}_bias": bias[m] for m in METHODS},
            **{f"{m}_sem": sem[m] for m in METHODS}}


def run() -> dict:
    noiseless = {}
    for sigma in SIGMAS:
        curve = _noiseless_bias_curve(sigma)
        pp = {m: float(np.nanmax(curve[m]) - np.nanmin(curve[m])) for m in METHODS}
        noiseless[str(sigma)] = {"curve": curve, "peak_to_peak": pp}

    noisy = _noisy_bias_curve(PROJECT_SIGMA)

    results = {
        "sigmas": SIGMAS, "n_phases": N_PHASES, "project_sigma": PROJECT_SIGMA,
        "noiseless_flux": NOISELESS_FLUX, "noisy_snr": NOISY_SNR,
        "noisy_trials": NOISY_TRIALS,
        "noiseless": noiseless, "noisy": noisy,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp06a_pixel_locking.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2.7], hspace=0.45, wspace=0.3)

    # Panel 1: S-curves at an undersampled sigma, where locking is real.
    worst_sigma = "0.4"
    ax0 = fig.add_subplot(gs[0, 0])
    curve = results["noiseless"][worst_sigma]["curve"]
    for m in METHODS:
        ax0.plot(curve["phases"], np.array(curve[m]) * 1000, MARKERS[m] + "-", color=COLORS[m], label=LABELS[m], ms=4)
    ax0.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax0.set_xlabel("true sub-pixel phase (px)")
    ax0.set_ylabel("systematic bias (millipixels)")
    ax0.set_title(f"Undersampled: sigma={worst_sigma} px, noiseless")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # Panel 2: locking amplitude vs sigma.
    ax1 = fig.add_subplot(gs[0, 1])
    sigmas = results["sigmas"]
    for m in METHODS:
        pps = [results["noiseless"][str(s)]["peak_to_peak"][m] * 1000 for s in sigmas]
        ax1.semilogy(sigmas, np.maximum(pps, 1e-4), MARKERS[m] + "-", color=COLORS[m], label=LABELS[m], ms=4)
    ax1.axvline(results["project_sigma"], color="black", lw=1.2, linestyle=":", label=f"this project (sigma={results['project_sigma']})")
    ax1.set_xlabel("PSF sigma (px)")
    ax1.set_ylabel("locking amplitude, peak-to-peak (millipixels)")
    ax1.set_title("Locking amplitude vs PSF sampling")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # Panel 3: noisy behaviour at the operating point, with the centroid
    # shown both with and without clipping, because that is what isolates
    # the real mechanism found here.
    ax2 = fig.add_subplot(gs[0, 2])
    noisy = results["noisy"]
    for m in METHODS:
        b = np.array(noisy[f"{m}_bias"]) * 1000
        e = np.array(noisy[f"{m}_sem"]) * 1000
        lab = LABELS[m] + (", clip on (default)" if m == "centroid" else "")
        ax2.errorbar(noisy["phases"], b, yerr=e, fmt=MARKERS[m] + "-", color=COLORS[m], label=lab, ms=4, capsize=2, elinewidth=0.8)
    bn = np.array(noisy["centroid_noclip_bias"]) * 1000
    en = np.array(noisy["centroid_noclip_sem"]) * 1000
    ax2.errorbar(noisy["phases"], bn, yerr=en, fmt="o--", color="#e67e22", label="centroid, clip off", ms=4, capsize=2, elinewidth=0.8)
    wx = np.array(noisy["window_x0"])
    jump = np.where(np.diff(wx) != 0)[0]
    for j in jump:
        ax2.axvline((noisy["phases"][j] + noisy["phases"][j + 1]) / 2, color="black", lw=1.0, linestyle=":")
    ax2.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax2.set_xlabel("true sub-pixel phase (px)")
    ax2.set_ylabel("measured bias (millipixels)")
    ax2.set_title(f"Operating point: sigma={results['project_sigma']}, SNR={results['noisy_snr']:.0f}\n(dotted line: window origin jumps)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    pp04 = results["noiseless"]["0.4"]["peak_to_peak"]
    pp175 = results["noiseless"][str(results["project_sigma"])]["peak_to_peak"]
    cb = np.array(noisy["centroid_bias"]) * 1000
    clip_pp = float(np.nanmax(cb) - np.nanmin(cb))
    noclip_pp = float(np.nanmax(bn) - np.nanmin(bn))
    typ_sem = float(np.nanmedian(noisy["centroid_sem"])) * 1000
    fit_pp = float(np.nanmax(np.array(noisy["fit_bias"]) * 1000) - np.nanmin(np.array(noisy["fit_bias"]) * 1000))

    explanation = (
        "What we see:\n"
        f"  Left: at sigma=0.4 px the spot is badly undersampled and the systematic bias traces the classic\n"
        f"  S-shape, {pp04['centroid']*1000:.0f} millipixels peak-to-peak for the centroid and {pp04['matched']*1000:.0f} for the matched filter.\n"
        "  Middle: that amplitude collapses as sampling improves, and by sigma=1.0 px it is under a tenth of a\n"
        "  millipixel for every method. Right: at the real operating point the fit and matched filter are flat,\n"
        "  but the centroid with its default clipping is NOT, and it breaks at exactly the phase where the\n"
        "  integer window origin jumps.\n"
        "\n"
        "What we can derive:\n"
        f"  1. Classical pixel locking is not a contributor at sigma={results['project_sigma']} px. The noiseless amplitude is\n"
        f"     {pp175['centroid']*1000:.3f} millipixels (centroid), {pp175['fit']*1000:.3f} (fit), {pp175['matched']*1000:.3f} (matched filter), orders of\n"
        "     magnitude below every other bias source characterised in this project. The brief's ~7 px 1/e^2 spot\n"
        "     spec converts to sigma=1.75 px, which is Nyquist-sampled with margin, and the sweep puts the onset\n"
        "     below about sigma=0.75 px. The spot specification is what determines the immunity.\n"
        f"  2. A phase-dependent centroid bias does remain at the operating point: {clip_pp:.1f} millipixels peak-to-peak\n"
        f"     against a typical standard error of {typ_sem:.2f}, roughly 20 standard errors, so it is not noise. The\n"
        "     noiseless sweep does not predict it, so both measurements are required to characterise the estimator.\n"
        f"  3. The mechanism is the clipping, not the pixel grid. Rerunning the identical frames with\n"
        f"     clip_negative=False flattens the curve to {noclip_pp:.1f} millipixels peak-to-peak. Clipping negative\n"
        "     background-subtracted pixels is a nonlinearity that only acts in the presence of noise: it rectifies\n"
        "     downward excursions to zero and leaves a one-sided positive residue. When the spot sits off-centre\n"
        "     inside its integer-placed window that residue is spatially lopsided, dragging the weighted average\n"
        "     toward the window centre. The pull grows as the spot drifts off-centre with phase and resets when\n"
        "     the window re-centres, producing the sawtooth and the discontinuity at the dotted line.\n"
        "  4. centroid.py documents clip_negative as a bias/variance tradeoff: clipping cuts variance at the cost\n"
        "     of a rectification bias. The additional result here is that the bias is phase-dependent, so it does\n"
        "     not average away across a moving spot and appears as a position-dependent offset in a tracking loop.\n"
        f"  5. The Gaussian fit reads {fit_pp:.1f} millipixels peak-to-peak here and 0.000 noiseless at every sigma\n"
        "     tested. This follows from model match: its forward model is the same erf-based exact pixel integral\n"
        "     used to render the frame. A real PSF is never exactly the assumed one, so the result bounds locking\n"
        "     from the pixel-integration approximation only, not locking in general.\n"
        "  6. The centroid rises again at sigma=2.5 px in the middle panel. This is window truncation, a distinct\n"
        "     mechanism: a wider spot spills past the fixed half-width window and its tails are clipped\n"
        "     asymmetrically. It shares the measurement axis but not the cause."
    )
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.text(
        0.0, 1.0, explanation, transform=ax3.transAxes, fontsize=8.8,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp06a_pixel_locking.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print("\n[exp06a] noiseless locking amplitude, peak-to-peak (millipixels)")
    print(f"{'sigma':>7} {'centroid':>10} {'fit':>10} {'matched':>10}")
    for s in results["sigmas"]:
        pp = results["noiseless"][str(s)]["peak_to_peak"]
        print(f"{s:7.2f} {pp['centroid']*1000:10.3f} {pp['fit']*1000:10.3f} {pp['matched']*1000:10.3f}")
    noisy = results["noisy"]
    cb = np.array(noisy["centroid_bias"]) * 1000
    bn = np.array(noisy["centroid_noclip_bias"]) * 1000
    print(f"  noisy at sigma={results['project_sigma']}, SNR={results['noisy_snr']:.0f}:")
    print(f"    centroid clip on  : {np.nanmax(cb)-np.nanmin(cb):.2f} mpx peak-to-peak")
    print(f"    centroid clip off : {np.nanmax(bn)-np.nanmin(bn):.2f} mpx peak-to-peak")
    for m in ["fit", "matched"]:
        b = np.array(noisy[f"{m}_bias"]) * 1000
        print(f"    {m:18s}: {np.nanmax(b)-np.nanmin(b):.2f} mpx peak-to-peak")


if __name__ == "__main__":
    run()
