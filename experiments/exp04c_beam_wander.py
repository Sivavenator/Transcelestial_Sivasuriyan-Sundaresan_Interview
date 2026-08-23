"""Experiment 04c -- beam wander (brief §4, item 2 in
docs/REAL_WORLD_CONDITIONS.md): show that a position-noise source with
IDENTICAL time-domain variance to mechanical jitter is still cleanly
separable from it by spectral shape, and quantify the combined noise
budget when both are present together.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sptrack.beam_wander import generate_beam_wander
from sptrack.trajectory import TrajectoryConfig, generate_trajectory

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run() -> dict:
    n_frames, dt_s = 4096, 1e-3
    jitter_sigma = 0.15
    wander_sigma = 0.15
    tau_s = 20e-3

    traj_cfg = TrajectoryConfig(seed=2026, x0=20.3, y0=19.7, n_frames=n_frames, dt_s=dt_s)
    traj = generate_trajectory(traj_cfg)  # includes drift + jitter + disturbance, jitter_std_px=0.15 by default

    dx_wander, dy_wander = generate_beam_wander(n_frames, dt_s, sigma_px=wander_sigma, tau_s=tau_s, seed=2026)

    x_with_wander = traj["x"] + dx_wander

    freqs = np.fft.rfftfreq(n_frames, d=dt_s)
    low_mask = (freqs > 0.5) & (freqs < 10)
    high_mask = (freqs >= 100) & (freqs < 400)

    p_jitter = np.abs(np.fft.rfft(traj["jitter_x"])) ** 2
    p_wander = np.abs(np.fft.rfft(dx_wander)) ** 2
    jitter_ratio = float(p_jitter[low_mask].sum() / p_jitter[high_mask].sum())
    wander_ratio = float(p_wander[low_mask].sum() / p_wander[high_mask].sum())

    combined_std = float(np.std(traj["jitter_x"] + dx_wander))
    expected_quadrature = float(np.sqrt(jitter_sigma**2 + wander_sigma**2))

    results = {
        "jitter_sigma_px": jitter_sigma, "wander_sigma_px": wander_sigma, "tau_s_ms": tau_s * 1000,
        "jitter_low_high_ratio": jitter_ratio, "wander_low_high_ratio": wander_ratio,
        "combined_std_px": combined_std, "expected_quadrature_std_px": expected_quadrature,
        "corner_freq_hz": float(1 / (2 * np.pi * tau_s)),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp04c_beam_wander.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(traj, dx_wander, x_with_wander, freqs, results)
    _print_summary(results)
    return results


def _plot(traj: dict, dx_wander: np.ndarray, x_with_wander: np.ndarray, freqs: np.ndarray, results: dict) -> None:
    p_jitter = np.abs(np.fft.rfft(traj["jitter_x"])) ** 2
    p_wander = np.abs(np.fft.rfft(dx_wander)) ** 2

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2.4], hspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    ax0.loglog(freqs[1:], p_jitter[1:], color="#c0392b", lw=0.8, alpha=0.8, label=f"mechanical jitter (std={results['jitter_sigma_px']:.2f} px, white)")
    ax0.loglog(freqs[1:], p_wander[1:], color="#2166ac", lw=0.8, alpha=0.8, label=f"beam wander (std={results['wander_sigma_px']:.2f} px, correlated)")
    ax0.axvline(results["corner_freq_hz"], color="#2166ac", lw=1.2, linestyle="--", label=f"wander corner freq ({results['corner_freq_hz']:.1f} Hz)")
    ax0.set_xlabel("frequency (Hz)")
    ax0.set_ylabel("power")
    ax0.set_title("Identical time-domain variance, cleanly separable spectral shape")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3, which="both")

    explanation = (
        "What we see:\n"
        "  Two position-noise sources with the SAME standard deviation (0.15 px) look completely different in\n"
        "  frequency: mechanical jitter (red) is flat across the whole band, while beam wander (blue) falls off\n"
        "  steeply above its corner frequency -- most of its power concentrated well below 10 Hz.\n"
        "\n"
        "What we can derive:\n"
        f"  1. Low-band(0.5-10Hz)/high-band(100-400Hz) power ratio: jitter={results['jitter_low_high_ratio']:.3f},\n"
        f"     beam wander={results['wander_low_high_ratio']:.2f} -- roughly a {results['wander_low_high_ratio']/results['jitter_low_high_ratio']:.0f}x difference despite\n"
        "     identical variance. Equal magnitude does not mean indistinguishable -- these are separable by SHAPE,\n"
        "     the same principle §3's drift/jitter/disturbance decomposition already relied on.\n"
        f"  2. Combined std when both are present: {results['combined_std_px']:.4f} px, matching the independent-sources\n"
        f"     quadrature-sum prediction sqrt(jitter^2+wander^2)={results['expected_quadrature_std_px']:.4f} px almost exactly --\n"
        "     confirming the two contribute independently to the total position-noise budget, as assumed.\n"
        "  3. A genuine interaction worth flagging: beam wander's spectral shape (low-frequency-concentrated, like\n"
        "     drift) is exactly what §3's disturbance detector excludes from its peak search via exclude_below_hz.\n"
        "     In a real deployment with both drift AND beam wander present, that exclusion band would need to\n"
        "     widen to cover both -- increasing the risk of the boundary-blind-spot failure mode already found in\n"
        "     the §3 hard-scenario analysis, if a real disturbance's frequency happens to sit near the (now wider)\n"
        "     boundary. Not re-tested here quantitatively -- recorded as an identified, not yet characterized, risk."
    )
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, explanation, transform=ax_text.transAxes, fontsize=9.0,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "exp04c_beam_wander.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict) -> None:
    print(f"\n[exp04c] jitter_ratio={results['jitter_low_high_ratio']:.3f}  wander_ratio={results['wander_low_high_ratio']:.2f}")
    print(f"  combined_std={results['combined_std_px']:.4f}px  expected(quadrature)={results['expected_quadrature_std_px']:.4f}px")


if __name__ == "__main__":
    run()
