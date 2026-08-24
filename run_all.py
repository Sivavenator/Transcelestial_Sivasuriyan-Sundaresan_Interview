"""One command that reproduces every figure and result in this repository.

    python run_all.py              # full run, all experiments at their documented trial counts
    python run_all.py --quick      # reduced statistics where an experiment supports it
    python run_all.py --only exp01 exp06b
    python run_all.py --list

Everything is seeded, so re-running reproduces the same numbers exactly,
with one stated exception: the wall-clock timings in exp02_realtime and
exp05e_latency_budget are Python timings on whatever machine runs them
and are not bit-reproducible across machines. See docs/ASSUMPTIONS.md's
"Panel feedback" section for what was measured about that on this
project's own development machine.

WHY --quick DOES NOT TOUCH MOST EXPERIMENTS
--------------------------------------------------
Three experiments (exp01, exp02, exp05d) accept a quick=True argument
that reduces their own trial count. The other sixteen do not, and this
script does not add it to them. Their trial counts were each chosen for a
stated reason, for example exp01's n_trials=300 is derived in that
module's own docstring from the standard error of a sample standard
deviation, and retrofitting an arbitrary reduction would be changing an
already-justified number without justification, which is exactly what
this project's own standing practice argues against. --quick here means
"use it where it was designed in", not "make everything faster".
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# (module, human description, accepts quick=). Order matches the brief's
# own section numbering, so the run order is also a readable table of
# contents.
EXPERIMENTS: list[tuple[str, str, bool]] = [
    ("exp01_snr_characterization", "Bias and std vs SNR, against the Cramer-Rao bound (§2c)", True),
    ("exp02_realtime", "Per-frame compute cost vs the 1 kHz budget (§2d)", True),
    ("exp03a_trajectory_diagnostic", "Ground-truth drift+jitter+disturbance trajectory (§3)", False),
    ("exp03b_trajectory_recovery", "Full-sequence trajectory recovery (§3)", False),
    ("exp03c_disturbance_detection", "Recovered vs injected disturbance (§3)", False),
    ("exp03d_hard_scenario", "Deliberately hard scenario, two failure modes (§3)", False),
    ("exp04a_scintillation", "Simulated atmospheric scintillation (§4)", False),
    ("exp04b_fog_attenuation", "Weather-attenuation sweep, lock-loss rate (§4)", False),
    ("exp04c_beam_wander", "Turbulence position noise vs mechanical jitter (§4)", False),
    ("exp04d_clutter", "Acquisition failure and its fix (§4)", False),
    ("exp04e_glare", "Background-gradient bias and the planar-fit fix (§4)", False),
    ("exp05a_auto_exposure", "Auto-exposure/gain across 5 decades of brightness (§5)", False),
    ("exp05b_calibration", "Bias, flat-field and lens-distortion calibration (§5)", False),
    ("exp05c_motion_blur", "Intra-frame motion blur robustness (§5)", False),
    ("exp05d_low_photon_count", "Extreme low-photon-count characterisation (§5)", True),
    ("exp05e_latency_budget", "Full photon-to-estimate latency budget (§5)", False),
    ("exp06a_pixel_locking", "Bias vs sub-pixel phase, PSF sampling margin (§6)", False),
    ("exp06b_window_size", "Window half-width sweep against the CRLB (§6)", False),
    ("exp07_kalman_tracking", "Kalman/alpha-beta filtering, and when it helps (§6)", False),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="reduced statistics where supported")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only the named experiment(s), e.g. exp01 or exp01_snr_characterization")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    args = parser.parse_args()

    if args.list:
        for name, desc, quick_ok in EXPERIMENTS:
            flag = " [--quick capable]" if quick_ok else ""
            print(f"  {name:32s} {desc}{flag}")
        return 0

    selected = EXPERIMENTS
    if args.only:
        wanted = args.only
        selected = [e for e in EXPERIMENTS if any(e[0] == w or e[0].startswith(w) for w in wanted)]
        if not selected:
            print(f"no experiments matched --only {args.only}", file=sys.stderr)
            print("run with --list to see valid names", file=sys.stderr)
            return 1

    print(f"running {len(selected)} experiment(s){' (--quick)' if args.quick else ''}\n")

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    failures: list[str] = []
    total_start = time.perf_counter()

    for name, desc, quick_ok in selected:
        print(f"[{name}] {desc}")
        module = importlib.import_module(f"experiments.{name}")
        start = time.perf_counter()
        try:
            if args.quick and quick_ok:
                module.run(quick=True)
            else:
                module.run()
        except Exception:
            print(f"  FAILED\n{traceback.format_exc()}")
            failures.append(name)
            continue
        elapsed = time.perf_counter() - start
        print(f"  done in {elapsed:.1f}s\n")

    total_elapsed = time.perf_counter() - total_start
    print(f"total: {total_elapsed / 60:.1f} minutes")

    if failures:
        print(f"\n{len(failures)} experiment(s) failed: {', '.join(failures)}")
        return 1

    print(f"\nall {len(selected)} experiment(s) completed; results in results/, figures in figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
