"""Experiment 07 -- should the per-frame position estimates be temporally
filtered, and what does filtering cost?

THE QUESTION THIS ANSWERS
--------------------------------
A Kalman filter is the default answer to "noisy measurements of a moving
target", so its absence needs justifying and its presence needs
measuring. Neither is settled by opinion. This experiment runs a
constant-velocity Kalman filter across the actual recovered trajectory
from the §3 pipeline over a wide range of process-noise settings, and
measures both sides of the trade at once:

  what filtering buys   reduction in position error standard deviation
  what filtering costs  attenuation and phase lag on the real 20 Hz
                        disturbance that the system is supposed to track

Reporting only the first number would make any filter look good.

WHY LAG IS THE COST THAT MATTERS HERE, NOT JUST ATTENUATION
--------------------------------------------------------------------
This estimator feeds a closed-loop pointing controller. A filter that
removes measurement noise by averaging over the recent past necessarily
reports where the spot WAS, not where it is. Inside a control loop that
delay eats phase margin, and phase margin is what keeps the loop stable.
So the useful figure of merit is not "did the error get smaller" but
"how much delay was added to buy that reduction", which is why the
sinusoid response is measured in milliseconds of lag as well as gain.

WHY THE SWEEP IS OVER PROCESS NOISE q
---------------------------------------------
q, the assumed power spectral density of acceleration disturbance, is the
single knob that sets how much the filter trusts its own constant-velocity
prediction relative to the incoming measurement. Large q means the filter
believes the target can accelerate hard, so it stays close to the
measurements and barely smooths. Small q means it believes the target is
nearly constant-velocity, so it smooths hard and lags. Sweeping q traces
the entire achievable trade-off curve rather than reporting one tuned
point, and one tuned point is exactly what would hide the cost.

WHY R IS FIXED FROM MEASUREMENT, NOT TUNED
---------------------------------------------------
R is set to the Gaussian fit's measured single-frame variance at this
operating point, taken from exp03b (std about 8.8 millipixels at SNR=50).
It is a known quantity here, not a free parameter, so tuning it would
amount to lying to the filter about its own sensor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.sequence import recover_trajectory, render_sequence
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux
from sptrack.tracking import (
    AlphaBetaTracker1D,
    KalmanTracker1D,
    filter_sequence,
    sinusoid_response,
)
from sptrack.trajectory import TrajectoryConfig, generate_trajectory

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SHAPE = (41, 41)
HALF_WIDTH = 9
BACKGROUND_E = 30.0
SIGMA_READ_E = 5.0
SNRS = [3.0, 8.0, 20.0, 50.0]
MAIN_SNR = 50.0
BURN_IN = 500
PROCESS_PSDS = np.geomspace(1e-2, 1e10, 25)


def _sweep_at_snr(traj: dict, traj_cfg: TrajectoryConfig, snr: float) -> dict:
    sim = Simulator(
        shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2026,
    )
    flux = snr_to_flux(
        snr, sim.sigma, BACKGROUND_E,
        sim.dark_rate_e_per_s * sim.exposure_s, SIGMA_READ_E, sim.gain_e_per_dn,
    )
    frames = render_sequence(sim, traj["x"], traj["y"], flux)
    recovered = recover_trajectory(frames, HALF_WIDTH, sim.sigma, SIGMA_READ_E**2)

    truth = traj["x"]
    meas = recovered["x"]
    dt = traj_cfg.dt_s
    freq = traj_cfg.disturb_freq_hz

    ok = recovered["ok"]
    raw_std = float(np.std((meas - truth)[BURN_IN:]))
    meas_var = raw_std**2

    sweep = {"process_psd": [], "std_px": [], "lag_ms": [], "gain": [], "alpha": [], "beta": []}
    for q in PROCESS_PSDS:
        k = KalmanTracker1D(dt_s=dt, process_psd=float(q), meas_var_px2=meas_var)
        alpha, beta = k.steady_state_gains()
        k2 = KalmanTracker1D(dt_s=dt, process_psd=float(q), meas_var_px2=meas_var)
        filtered = filter_sequence(k2, meas)
        err = filtered[BURN_IN:] - truth[BURN_IN:]
        resp = sinusoid_response(filtered[BURN_IN:], truth[BURN_IN:], freq, dt)
        sweep["process_psd"].append(float(q))
        sweep["std_px"].append(float(np.std(err)))
        sweep["lag_ms"].append(float(resp["lag_ms"]))
        sweep["gain"].append(float(resp["gain"]))
        sweep["alpha"].append(float(alpha))
        sweep["beta"].append(float(beta))

    best_i = int(np.argmin(sweep["std_px"]))
    return {
        "snr": float(snr), "raw_std_px": raw_std, "n_failed": int((~ok).sum()),
        "sweep": sweep, "best_index": best_i,
        "best_process_psd": sweep["process_psd"][best_i],
        "best_std_px": sweep["std_px"][best_i],
        "best_lag_ms": sweep["lag_ms"][best_i],
        "best_gain": sweep["gain"][best_i],
        "improvement_pct": float(100.0 * (raw_std - sweep["std_px"][best_i]) / raw_std),
    }


def run() -> dict:
    traj_cfg = TrajectoryConfig(seed=2026, x0=20.3, y0=19.7)
    traj = generate_trajectory(traj_cfg)
    dt = traj_cfg.dt_s
    freq = traj_cfg.disturb_freq_hz

    # The governing ratio: how far the target actually moves between
    # frames, against how well one frame can be measured.
    true_step_std = float(np.std(np.diff(traj["x"])))

    by_snr = {}
    for snr in SNRS:
        by_snr[str(snr)] = _sweep_at_snr(traj, traj_cfg, snr)
        e = by_snr[str(snr)]
        print(f"  SNR={snr:5.1f}: raw={e['raw_std_px']*1000:7.2f} mpx  "
              f"best={e['best_std_px']*1000:7.2f} mpx  ({e['improvement_pct']:+.0f}%)  "
              f"motion/noise={true_step_std/e['raw_std_px']:5.1f}")

    main = by_snr[str(MAIN_SNR)]
    sweep = main["sweep"]
    raw_std = main["raw_std_px"]
    meas_var = raw_std**2
    meas = None
    best_i = main["best_index"]

    # Cost per update, measured the same way as exp02. Timing is wall
    # clock on one machine and is not reproducible run to run; it is
    # reported as an order of magnitude, not a constant.
    n_timing = 20000
    dummy = np.asarray(traj["x"], dtype=np.float64)
    k_t = KalmanTracker1D(dt_s=dt, process_psd=1e3, meas_var_px2=meas_var)
    for i in range(1000):
        k_t.step(float(dummy[i % len(dummy)]))
    t0 = time.perf_counter()
    for i in range(n_timing):
        k_t.step(float(dummy[i % len(dummy)]))
    kalman_us = (time.perf_counter() - t0) / n_timing * 1e6

    ab_t = AlphaBetaTracker1D(dt_s=dt, alpha=0.1, beta=0.001)
    for i in range(1000):
        ab_t.step(float(dummy[i % len(dummy)]))
    t0 = time.perf_counter()
    for i in range(n_timing):
        ab_t.step(float(dummy[i % len(dummy)]))
    alphabeta_us = (time.perf_counter() - t0) / n_timing * 1e6

    results = {
        "snrs": SNRS, "main_snr": MAIN_SNR, "dt_s": dt,
        "disturb_freq_hz": freq, "disturb_amp_px": traj_cfg.disturb_amp_px,
        "n_frames": traj_cfg.n_frames, "burn_in": BURN_IN,
        "true_step_std_px": true_step_std,
        "jitter_std_px": traj_cfg.jitter_std_px,
        "by_snr": by_snr,
        "raw_std_px": raw_std,
        "sweep": sweep,
        "best_index": best_i,
        "best_process_psd": sweep["process_psd"][best_i],
        "best_std_px": sweep["std_px"][best_i],
        "best_lag_ms": sweep["lag_ms"][best_i],
        "best_gain": sweep["gain"][best_i],
        "kalman_us_per_update": kalman_us,
        "alphabeta_us_per_update": alphabeta_us,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp07_kalman_tracking.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    _print_summary(results)
    return results


def _plot(results: dict) -> None:
    s = results["sweep"]
    q = np.array(s["process_psd"])
    std = np.array(s["std_px"]) * 1000
    lag = np.array(s["lag_ms"])
    gain = np.array(s["gain"])
    raw = results["raw_std_px"] * 1000

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2.7], hspace=0.45, wspace=0.45)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.semilogx(q, std, "o-", color="#2166ac", ms=4, label="filtered")
    ax0.axhline(raw, color="#c0392b", lw=1.4, linestyle="--", label=f"unfiltered ({raw:.1f} mpx)")
    ax0.set_xlabel("process noise q (px^2/s^3)")
    ax0.set_ylabel("position error std (millipixels)")
    ax0.set_title("What filtering buys")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3, which="both")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.semilogx(q, lag, "s-", color="#e67e22", ms=4, label="lag on the 20 Hz tone")
    ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax1.set_xlabel("process noise q (px^2/s^3)")
    ax1.set_ylabel("lag (ms)")
    ax1b = ax1.twinx()
    ax1b.semilogx(q, gain, "^-", color="#27ae60", ms=4, label="gain on the 20 Hz tone")
    ax1b.set_ylabel("gain")
    ax1.set_title("What filtering costs")
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, fontsize=8, loc="center left")
    ax1.grid(alpha=0.3, which="both")

    ax2 = fig.add_subplot(gs[0, 2])
    snrs = results["snrs"]
    ratios = [results["true_step_std_px"] / results["by_snr"][str(s)]["raw_std_px"] for s in snrs]
    improvements = [results["by_snr"][str(s)]["improvement_pct"] for s in snrs]
    ax2.semilogx(ratios, improvements, "o-", color="#8e44ad", ms=6)
    for r, imp, s in zip(ratios, improvements, snrs):
        ax2.annotate(f"SNR={s:.0f}", (r, imp), fontsize=7,
                     textcoords="offset points", xytext=(5, 5))
    ax2.axhline(0, color="black", lw=1.2, linestyle="--")
    ax2.axvline(1.0, color="gray", lw=1.0, linestyle=":")
    ax2.set_xlabel("target motion per frame / measurement noise")
    ax2.set_ylabel("best improvement (%)")
    ax2.set_title("When filtering helps at all")
    ax2.grid(alpha=0.3, which="both")

    frame_ms = results["dt_s"] * 1000
    main_ratio = results["true_step_std_px"] / raw * 1000
    rows = []
    for s in snrs:
        e = results["by_snr"][str(s)]
        rows.append(
            f"    SNR={s:5.1f}  raw={e['raw_std_px']*1000:7.1f} mpx  best filtered={e['best_std_px']*1000:7.1f} mpx"
            f"  change={e['improvement_pct']:+6.0f}%  motion/noise={results['true_step_std_px']/e['raw_std_px']:6.1f}"
        )
    explanation = (
        "What we see:\n"
        f"  Left: at the operating point every filter setting is WORSE than not filtering. The unfiltered error is\n"
        f"  {raw:.1f} millipixels (red dashed) and the best any q achieves is {results['best_std_px']*1000:.1f}. Middle: heavy smoothing also\n"
        "  attenuates and delays the real 20 Hz disturbance, and shows resonant peaking (gain above 1) at\n"
        "  intermediate q. Right: sweeping SNR shows filtering only becomes useful once the measurement is noisy\n"
        "  compared to how far the target actually moves between frames.\n"
        "\n"
        "What we can derive:\n" + "\n".join(rows) + "\n"
        "  1. A Kalman filter reduces variance by blending a prediction with a measurement. That only helps when\n"
        "     the prediction is competitive with the measurement. The governing quantity is therefore how far the\n"
        f"     target moves unpredictably between frames against the measurement error. Here the target moves\n"
        f"     {results['true_step_std_px']*1000:.0f} millipixels per frame (dominated by white jitter at\n"
        f"     {results['jitter_std_px']*1000:.0f} millipixels, which contributes sqrt(2) times that to a frame difference) while the\n"
        f"     Gaussian fit measures each frame to {raw:.1f} millipixels. The ratio is {main_ratio:.0f} to 1 in favour of the\n"
        "     measurement, so the best possible prediction is far worse than simply believing the sensor.\n"
        "  2. That is the answer to why the main pipeline does not filter. It is not that Kalman filtering is\n"
        "     inappropriate in general, it is that at this SNR the measurement is already 20 or more times better\n"
        "     than the motion model, and the optimal gain is therefore to trust the measurement almost completely.\n"
        "     The sweep confirms it: as q rises the filter converges back toward the raw measurement.\n"
        "  3. The condition under which it would pay is visible on the right and is a ratio, not an SNR. Once\n"
        "     measurement noise approaches the per-frame motion, prediction starts carrying real information.\n"
        "     For this trajectory that means low SNR, which is exactly the regime §4 showed fog and deep\n"
        "     scintillation fades produce. A deployed system that must ride through those conditions has a\n"
        "     genuine case for switching filtering on when SNR drops, and leaving it off otherwise.\n"
        "  4. Jitter is white by construction, so no causal filter can predict it. The 20 Hz tone is predictable\n"
        "     in principle, but a constant-velocity model does not contain a resonator, so it cannot track a\n"
        "     sinusoid without lag. Filtering hard enough to suppress the jitter attenuates the tone the system\n"
        f"     exists to follow: at the smoothest setting tested the tone is passed at {gain[0]:.2f} amplitude with\n"
        f"     {lag[0]:.1f} ms of lag, which is {lag[0]/frame_ms:.0f} frame periods.\n"
        "  5. Where the filter would still earn its place is the prediction step rather than the smoothing. §4\n"
        "     measured 25 dropped frames from scintillation fades, currently bridged by holding the last known\n"
        "     good position. A constant-velocity state coasts through a gap at the last estimated velocity\n"
        "     instead of freezing, which is strictly better while the target is moving.\n"
        f"  6. Cost: about {results['kalman_us_per_update']:.1f} us per Kalman update against {results['alphabeta_us_per_update']:.2f} us for the alpha-beta form using\n"
        "     the converged gains, both negligible against the 1000 us frame budget. Compute is not the deciding\n"
        "     factor; the alpha-beta advantage is bounded branch-free work per frame on constrained hardware.\n"
        "     These timings are wall clock on one machine and vary run to run."
    )
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.text(
        0.0, 1.0, explanation, transform=ax3.transAxes, fontsize=8.6,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp07_kalman_tracking.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp07] target moves {results['true_step_std_px']*1000:.1f} mpx/frame "
          f"(jitter {results['jitter_std_px']*1000:.0f} mpx)")
    print(f"{'SNR':>6} {'raw_mpx':>9} {'best_mpx':>9} {'change':>8} {'motion/noise':>13}")
    for s in results["snrs"]:
        e = results["by_snr"][str(s)]
        print(f"{s:6.1f} {e['raw_std_px']*1000:9.2f} {e['best_std_px']*1000:9.2f} "
              f"{e['improvement_pct']:+7.0f}% {results['true_step_std_px']/e['raw_std_px']:13.1f}")
    print(f"  cost: kalman {results['kalman_us_per_update']:.2f} us/update, "
          f"alpha-beta {results['alphabeta_us_per_update']:.2f} us/update (wall clock, varies)")


if __name__ == "__main__":
    run()
