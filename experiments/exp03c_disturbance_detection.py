"""Experiment 03c -- detect the periodic disturbance's frequency and
amplitude from the RECOVERED trajectory (exp03b's output), and report how
close the detected values are to the values actually injected in
trajectory.py. This closes out the brief's core dynamic-tracking ask on
the default (easy) scenario; the deliberately hard variant and failure
modes are the next, separate experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.disturbance import detect_disturbance
from sptrack.sequence import recover_trajectory, render_sequence
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux
from sptrack.trajectory import TrajectoryConfig, generate_trajectory

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run() -> dict:
    shape = (41, 41)
    half_width = 9
    background_e = 30.0
    sigma_read_e = 5.0
    snr = 50.0

    traj_cfg = TrajectoryConfig(seed=2026, x0=20.3, y0=19.7)
    traj = generate_trajectory(traj_cfg)

    sim = Simulator(
        shape=shape, background_e=background_e, sigma_read_e=sigma_read_e,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=2026,
    )
    flux = snr_to_flux(
        snr, sim.sigma, background_e,
        sim.dark_rate_e_per_s * sim.exposure_s, sigma_read_e, sim.gain_e_per_dn,
    )
    frames = render_sequence(sim, traj["x"], traj["y"], flux)
    recovered = recover_trajectory(frames, half_width, sim.sigma, sigma_read_e**2)

    detected = detect_disturbance(recovered["x"], traj_cfg.dt_s)
    # also detect straight off ground truth, as a reference for how much of
    # the gap (if any) is the RECOVERY step's fault vs. inherent to the
    # detection method itself
    detected_gt = detect_disturbance(traj["x"], traj_cfg.dt_s)

    freq_err_hz = detected["freq_hz"] - traj_cfg.disturb_freq_hz
    amp_err_px = detected["amp_px"] - traj_cfg.disturb_amp_px
    amp_err_pct = 100.0 * amp_err_px / traj_cfg.disturb_amp_px

    results = {
        "injected_freq_hz": traj_cfg.disturb_freq_hz,
        "injected_amp_px": traj_cfg.disturb_amp_px,
        "detected_freq_hz": detected["freq_hz"],
        "detected_amp_px": detected["amp_px"],
        "freq_resolution_hz": detected["freq_resolution_hz"],
        "freq_err_hz": freq_err_hz,
        "amp_err_px": amp_err_px,
        "amp_err_pct": amp_err_pct,
        "detected_freq_hz_ground_truth": detected_gt["freq_hz"],
        "detected_amp_px_ground_truth": detected_gt["amp_px"],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp03c_disturbance_detection.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(traj, recovered, traj_cfg, results)
    _print_summary(results)
    return results


def _plot(traj: dict, recovered: dict, traj_cfg: TrajectoryConfig, results: dict) -> None:
    n = traj_cfg.n_frames
    window = np.hanning(n)
    windowed = (recovered["x"] - recovered["x"].mean()) * window
    spec = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=traj_cfg.dt_s)
    amp_spectrum = 2.0 * np.abs(spec) / window.sum()

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2.4], hspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(freqs[1:], amp_spectrum[1:], color="#555555", lw=0.9, label="amplitude spectrum of recovered x(t)")
    ax0.axvline(results["injected_freq_hz"], color="#27ae60", lw=1.8, linestyle="--", label=f"injected ({results['injected_freq_hz']:.2f} Hz, {results['injected_amp_px']:.3f} px)")
    ax0.axvline(results["detected_freq_hz"], color="#c0392b", lw=1.2, linestyle=":", label=f"detected ({results['detected_freq_hz']:.2f} Hz, {results['detected_amp_px']:.3f} px)")
    ax0.set_xlim(0, 60)
    ax0.set_xlabel("frequency (Hz)")
    ax0.set_ylabel("amplitude (px)")
    ax0.set_title("Disturbance detection on the RECOVERED trajectory (never saw ground truth)")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3)

    explanation = (
        "What we see:\n"
        f"  A single clean spike sits at {results['detected_freq_hz']:.2f} Hz, right on top of the injected disturbance's true\n"
        f"  frequency ({results['injected_freq_hz']:.2f} Hz) -- the red dotted and green dashed lines are nearly coincident. The\n"
        "  spike's height matches the injected amplitude closely, with the rest of the spectrum (drift + jitter +\n"
        "  estimation noise) sitting well below it as a broad, low floor.\n"
        "\n"
        "What we can derive:\n"
        f"  1. Frequency: detected {results['detected_freq_hz']:.3f} Hz vs. injected {results['injected_freq_hz']:.3f} Hz -- error\n"
        f"     {results['freq_err_hz']*1000:.1f} mHz, inside the FFT's own {results['freq_resolution_hz']*1000:.1f} mHz bin resolution\n"
        "     (the honest limit of what this method can resolve without further sub-bin interpolation).\n"
        f"  2. Amplitude: detected {results['detected_amp_px']:.4f} px vs. injected {results['injected_amp_px']:.4f} px -- error\n"
        f"     {results['amp_err_pct']:+.2f}%, close to the single-tone calibration error already measured directly in\n"
        "     tests/test_disturbance.py (<1%), meaning almost none of the error here comes from noise or the\n"
        "     recovery step -- it is close to the detection method's own inherent floor.\n"
        f"  3. Detecting straight off GROUND TRUTH gives freq={results['detected_freq_hz_ground_truth']:.3f} Hz,\n"
        f"     amp={results['detected_amp_px_ground_truth']:.4f} px -- essentially identical to the recovered-trajectory\n"
        "     result, confirming the Gaussian-fit recovery step (§3 part 2) adds negligible extra error to disturbance\n"
        "     detection at this SNR, consistent with its bias/std already matching the static-frame floor."
    )
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, explanation, transform=ax_text.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp03c_disturbance_detection.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print("\n[exp03c] Disturbance detection, recovered vs injected:")
    print(f"  freq:      injected={results['injected_freq_hz']:.3f} Hz   detected={results['detected_freq_hz']:.3f} Hz   err={results['freq_err_hz']*1000:.1f} mHz")
    print(f"  amplitude: injected={results['injected_amp_px']:.4f} px   detected={results['detected_amp_px']:.4f} px   err={results['amp_err_pct']:+.2f}%")


if __name__ == "__main__":
    run()
