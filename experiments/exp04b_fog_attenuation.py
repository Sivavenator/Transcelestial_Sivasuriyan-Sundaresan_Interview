"""Experiment 04b -- fog/haze/rain attenuation (brief §4, item 5 in
docs/REAL_WORLD_CONDITIONS.md): quantify the position-precision and
lock-loss impact of realistic weather attenuation levels.

WHY THIS IS MODELLED AS A STEADY ATTENUATION LEVEL, NOT A TIME-VARYING
PROCESS WITHIN ONE SEQUENCE (UNLIKE SCINTILLATION)
------------------------------------------------------------------------------
Scintillation (exp04a) was modelled as fluctuating WITHIN a single ~4 s
capture window because its coherence time (milliseconds) is comparable to
the frame period -- real, visible fluctuation happens on exactly that
timescale. Fog and rain are different: visibility changes over minutes to
hours, not seconds -- essentially CONSTANT across any single 4096-frame
(4.1 s) capture window this project simulates. Modelling it as a
within-sequence random process would be dishonestly implying a timescale
mismatch it doesn't have. The physically correct framing instead treats
attenuation as a STEADY operating condition, swept ACROSS separate
Monte Carlo trials/conditions -- the same structure already used for the
SNR characterization sweep (§2c) and the hard-scenario sweep (§3, part 4),
reused here under a different, weather-grounded physical interpretation
of what sets the operating SNR.

WHY THESE SPECIFIC ATTENUATION VALUES (dB/km) AND A 1 km LINK
-------------------------------------------------------------------
No site-specific weather/link data exists for an actual deployment, so
both the per-condition attenuation coefficients and the assumed link
distance are honestly stated assumptions, grounded in commonly published
free-space-optical attenuation ranges (clear air ~0.2 dB/km; haze
~4 dB/km; light fog ~20 dB/km; moderate fog ~42 dB/km; dense fog
~130 dB/km), not a precise Kruse-model derivation from a specific
visibility figure -- that level of precision isn't warranted without a
real site survey. 1 km is a representative distance for a short-range
terrestrial FSO link, exposed as a single multiplier
(``LINK_DISTANCE_KM``) so it can be rescaled trivially.

WHY dropout RATE, NOT JUST std/bias, IS THE HEADLINE METRIC HERE
------------------------------------------------------------------------
§2c's SNR sweep already characterises std and bias vs. SNR in detail; this
experiment's genuinely new contribution is tracking the FRACTION of
trials that fail outright (`ok=False`) at each named condition -- exactly
the "operational limit" question `docs/REAL_WORLD_CONDITIONS.md` raises
for this condition (below some floor, the system loses lock, not just
precision).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.simulate import Simulator
from sptrack.snr import flux_to_snr, snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SHAPE = (21, 21)
HALF_WIDTH = 9
BACKGROUND_E = 30.0
SIGMA_READ_E = 5.0
CLEAR_SNR = 50.0
LINK_DISTANCE_KM = 1.0
N_TRIALS = 200

CONDITIONS_DB_PER_KM = {
    "clear": 0.2,
    "haze": 4.0,
    "light fog": 20.0,
    "moderate fog": 42.0,
    "dense fog": 130.0,
}


def run() -> dict:
    x0, y0 = 10.3, 9.7
    sim = Simulator(
        shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2027,
    )
    clear_flux = snr_to_flux(
        CLEAR_SNR, sim.sigma, BACKGROUND_E,
        sim.dark_rate_e_per_s * sim.exposure_s, SIGMA_READ_E, sim.gain_e_per_dn,
    )

    results: dict = {"link_distance_km": LINK_DISTANCE_KM, "clear_snr": CLEAR_SNR, "n_trials": N_TRIALS, "conditions": {}}
    for name, db_per_km in CONDITIONS_DB_PER_KM.items():
        atten_db = db_per_km * LINK_DISTANCE_KM
        transmittance = 10 ** (-atten_db / 10)
        flux = clear_flux * transmittance
        snr = flux_to_snr(flux, sim.sigma, BACKGROUND_E, sim.dark_rate_e_per_s * sim.exposure_s, SIGMA_READ_E, sim.gain_e_per_dn)

        errs = []
        n_failed = 0
        for _ in range(N_TRIALS):
            frame = sim.dn_to_electrons(sim.render(x0, y0, flux))
            est = gaussian_fit_estimate(frame, HALF_WIDTH, sim.sigma, SIGMA_READ_E**2, prior=(x0, y0))
            if est.ok:
                errs.append(est.x - x0)
            else:
                n_failed += 1

        results["conditions"][name] = {
            "atten_db_per_km": db_per_km, "atten_db_total": atten_db,
            "transmittance": transmittance, "flux": flux, "snr": snr,
            "dropout_rate": n_failed / N_TRIALS,
            "bias_px": float(np.mean(errs)) if errs else None,
            "std_px": float(np.std(errs)) if errs else None,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp04b_fog_attenuation.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    names = list(results["conditions"].keys())
    dropout = [results["conditions"][n]["dropout_rate"] * 100 for n in names]
    snr = [results["conditions"][n]["snr"] for n in names]

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2.2], hspace=0.55)

    ax0 = fig.add_subplot(gs[0])
    bars = ax0.bar(names, dropout, color="#c0392b", alpha=0.75)
    ax0.set_ylabel("dropout rate (%)")
    ax0.set_title(f"Lock-loss rate vs. weather condition ({results['link_distance_km']:.0f} km link, clear-weather SNR={results['clear_snr']:.0f})")
    for bar, s in zip(bars, snr):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"SNR={s:.2g}", ha="center", fontsize=8)
    ax0.grid(alpha=0.3, axis="y")

    lines = [
        f"  {n:14s}: {results['conditions'][n]['atten_db_total']:6.1f} dB total  SNR={results['conditions'][n]['snr']:8.3f}"
        f"  dropout={results['conditions'][n]['dropout_rate']*100:5.1f}%"
        f"  std={results['conditions'][n]['std_px']*1000:8.1f} mpx"
        for n in names
    ]
    explanation = (
        "What we see:\n"
        f"  At this project's assumed {results['link_distance_km']:.0f} km link, clear air and haze barely affect lock (SNR stays\n"
        "  well above the characterization floor from §2c), but light fog alone pushes SNR down near the edge of\n"
        "  what this project has ever characterized as workable, and moderate/dense fog collapse SNR to essentially\n"
        "  zero -- a hard operational cliff, not a gradual decline.\n"
        "\n"
        "What we can derive:\n" + "\n".join(lines) + "\n"
        "  The dropout-rate number alone UNDERSTATES the real failure at moderate/dense fog: only 43-44% of trials\n"
        "  registered as an outright failure (ok=False), but the 'successful' remainder's std explodes to ~2 px --\n"
        "  comparable to the whole estimation window -- meaning those are noise-driven fits to nothing, not real\n"
        "  detections, not a meaningfully precise (if noisy) measurement. Consistent with the step-size, not fit-\n"
        "  quality, convergence criterion already noted in docs/ASSUMPTIONS.md (§4 scintillation section): the fit\n"
        "  can 'converge' to a stable but meaningless position when there is no real signal to lock onto. The honest\n"
        "  operational floor should be read from precision collapsing to window-scale, not from dropout_rate alone.\n"
        "  Separately: the transition from workable to unusable happens within a NARROW attenuation range (between\n"
        "  haze and light fog here) -- this system does not degrade gracefully past a certain point, it falls off a\n"
        "  cliff, which matters directly for link-budget/margin decisions."
    )
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, explanation, transform=ax_text.transAxes, fontsize=8.8,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp04b_fog_attenuation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp04b] fog/rain attenuation, {results['link_distance_km']:.0f}km link, clear SNR={results['clear_snr']:.0f}")
    for name, r in results["conditions"].items():
        print(f"  {name:14s} atten={r['atten_db_total']:6.1f}dB  SNR={r['snr']:8.3f}  dropout={r['dropout_rate']*100:5.1f}%")


if __name__ == "__main__":
    run()
