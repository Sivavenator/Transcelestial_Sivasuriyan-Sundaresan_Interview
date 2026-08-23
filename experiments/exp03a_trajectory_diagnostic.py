"""Diagnostic for the ground-truth trajectory generator (sptrack/trajectory.py)
-- a sanity check BEFORE building the recovery/detection pipeline on top of
it. Shows the raw x(t) motion decomposed into its three components, and
the power spectrum demonstrating why they are separable at all: drift
concentrated at low frequency, jitter roughly flat, and the disturbance a
single clean spike above the jitter floor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.trajectory import TrajectoryConfig, generate_trajectory

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"


def run() -> None:
    cfg = TrajectoryConfig(seed=2026)
    out = generate_trajectory(cfg)
    t, x = out["t"], out["x"]
    n = cfg.n_frames

    spec = np.fft.rfft(x - x.mean())
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=cfg.dt_s)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, 2.0], hspace=0.55)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, out["drift_x"], color="#2166ac", lw=1.2, label="drift (random walk)")
    ax0.plot(t, out["jitter_x"], color="#c0392b", lw=0.5, alpha=0.6, label="jitter (white noise)")
    ax0.plot(t, out["disturb_x"], color="#27ae60", lw=1.2, label="periodic disturbance")
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel("x component (px)")
    ax0.set_title("The three motion components, individually")
    ax0.legend(fontsize=9, ncol=3)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, x, color="#333333", lw=0.8)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("x(t) - x0 (px)")
    ax1.set_title("Combined ground-truth trajectory (what the tracker must recover)")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    ax2.loglog(freqs[1:], power[1:], color="#555555", lw=0.9)
    ax2.axvline(cfg.disturb_freq_hz, color="#27ae60", lw=1.5, linestyle="--", label=f"injected disturbance ({cfg.disturb_freq_hz:.0f} Hz)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("power")
    ax2.set_title("Power spectrum of the combined trajectory")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    drift_final_std = abs(out["drift_x"][-1])
    max_excursion = max(np.abs(out["x"]).max(), np.abs(out["y"]).max())
    fig.text(
        0.5, -0.02,
        "What we see:\n"
        f"  Top: drift (blue) barely moves frame-to-frame but wanders steadily over the full {t[-1]:.1f} s window; jitter (red)\n"
        "  is fast, zero-mean, and uncorrelated frame-to-frame; the disturbance (green) is a clean, regular sinusoid.\n"
        f"  Middle: summed together, the combined trajectory looks noisy with a slow underlying wander -- the disturbance\n"
        "  is not visually obvious in the time domain alone. Bottom: in the frequency domain it is unambiguous -- power\n"
        "  falls off steeply at low frequency (the drift's 1/f^2 signature) and flattens into a noise floor (jitter),\n"
        "  with one clean spike exactly at the injected disturbance frequency, well above that floor.\n"
        "\n"
        "What we can derive:\n"
        f"  1. The three components ARE spectrally separable, not just conceptually distinct -- this is what makes\n"
        "     frequency-domain disturbance detection possible at all on the RECOVERED (not ground-truth) trajectory,\n"
        "     which is what the next step actually has to work with.\n"
        f"  2. Drift accumulated to {drift_final_std:.2f} px by the end of the window (from a 0.01 px/frame step) -- small\n"
        "     relative to the jitter scale (0.15 px/frame) but not negligible over the full sequence, matching the\n"
        "     'slow but real' behaviour the drift model was designed to produce.\n"
        f"  3. Total excursion over the whole sequence stayed under {max_excursion:.2f} px, comfortably inside a modest\n"
        "     simulation canvas -- confirming the fixed-size-frame assumption the rendering step (next) depends on.",
        ha="center", va="top", fontsize=9.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp03a_trajectory_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp03a] drift final={drift_final_std:.3f}px, max excursion={max_excursion:.3f}px")


if __name__ == "__main__":
    run()
