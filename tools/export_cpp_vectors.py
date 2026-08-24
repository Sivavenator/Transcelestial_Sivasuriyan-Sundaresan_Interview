"""Export cross-validation vectors for the C++ implementation.

Writes `results/cpp_vectors.csv`: a set of frames together with the
positions the Python estimators produce for them. `cpp/tests/test_against_python.cpp`
reads that file, runs the C++ estimators on the same pixels, and requires
agreement to a stated tolerance.

WHY CROSS-VALIDATION RATHER THAN SEPARATE C++ UNIT TESTS
----------------------------------------------------------------
Independent unit tests on each side would confirm that each
implementation is self-consistent. They would not catch the failure that
actually matters here, which is the two implementations disagreeing: a
transcription error in a Jacobian term, a different convergence rule, or
a different clamping order produces code that passes its own tests and
still returns a different answer from the reference. The Python side is
the characterised one, with 136 tests and every result in `results/`
traceable to it, so it is the reference and the C++ is required to
reproduce it.

WHY THE FRAMES ARE WRITTEN OUT RATHER THAN REGENERATED IN C++
--------------------------------------------------------------------
Regenerating the frames on the C++ side would require reimplementing the
simulator too, including the Poisson and Gaussian draws, and two RNG
implementations will not produce identical streams. Shipping the actual
pixel values removes the simulator from the comparison entirely, so a
failure points at the estimator and nothing else.

WHY THE CASES SPAN SNR AND SUB-PIXEL PHASE
--------------------------------------------------
Sub-pixel phase is swept because it is the axis along which a
transcription error in the pixel-integration term would show up as a
phase-dependent disagreement rather than a constant offset, which is the
same reasoning as `experiments/exp06a_pixel_locking.py`. SNR is swept
because the fit's iteration count depends on it, so a convergence-rule
difference between the implementations appears at some SNR values and
not others.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.simulate import Simulator
from sptrack.snr import snr_to_flux

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SHAPE = (21, 21)
HALF_WIDTH = 9
BACKGROUND_E = 30.0
SIGMA_READ_E = 5.0
SNRS = [5.0, 20.0, 50.0, 200.0]
PHASES = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


def run() -> Path:
    sim = Simulator(
        shape=SHAPE, background_e=BACKGROUND_E, sigma_read_e=SIGMA_READ_E,
        hot_fraction=0.0, prnu_sigma=0.0, gradient_frac=0.0, seed=13579,
    )
    read_var = SIGMA_READ_E**2
    rows = []
    case = 0
    for snr in SNRS:
        flux = snr_to_flux(
            snr, sim.sigma, BACKGROUND_E,
            sim.dark_rate_e_per_s * sim.exposure_s, SIGMA_READ_E, sim.gain_e_per_dn,
        )
        for ph in PHASES:
            x0, y0 = 10.0 + ph, 10.0 - ph / 2.0
            frame = sim.dn_to_electrons(sim.render(x0, y0, flux))

            c = centroid_estimate(frame, HALF_WIDTH, prior=(x0, y0))
            g = gaussian_fit_estimate(frame, HALF_WIDTH, sim.sigma, read_var, prior=(x0, y0))
            if not (c.ok and g.ok):
                continue

            rows.append({
                "case": case,
                "h": SHAPE[0], "w": SHAPE[1],
                "sigma": sim.sigma, "read_var": read_var,
                "half_width": HALF_WIDTH,
                "prior_x": x0, "prior_y": y0,
                "true_x": x0, "true_y": y0,
                "py_cx": c.x, "py_cy": c.y, "py_cflux": c.flux, "py_cbg": c.bg,
                "py_gx": g.x, "py_gy": g.y, "py_gflux": g.flux, "py_gbg": g.bg,
                "pixels": frame.ravel(),
            })
            case += 1

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "cpp_vectors.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "case", "h", "w", "sigma", "read_var", "half_width",
            "prior_x", "prior_y", "true_x", "true_y",
            "py_cx", "py_cy", "py_cflux", "py_cbg",
            "py_gx", "py_gy", "py_gflux", "py_gbg", "pixels",
        ])
        for r in rows:
            # repr() on a numpy scalar emits "np.float64(...)" under
            # numpy 2, which no C++ parser will accept. Force a Python
            # float first so the file contains plain decimal literals.
            def num(v: object) -> str:
                return repr(float(v))  # type: ignore[arg-type]

            wr.writerow([
                r["case"], r["h"], r["w"],
                num(r["sigma"]), num(r["read_var"]), r["half_width"],
                num(r["prior_x"]), num(r["prior_y"]),
                num(r["true_x"]), num(r["true_y"]),
                num(r["py_cx"]), num(r["py_cy"]), num(r["py_cflux"]), num(r["py_cbg"]),
                num(r["py_gx"]), num(r["py_gy"]), num(r["py_gflux"]), num(r["py_gbg"]),
                " ".join(num(v) for v in r["pixels"]),
            ])

    print(f"[export_cpp_vectors] wrote {len(rows)} cases to {out}")
    print(f"  frame {SHAPE[0]}x{SHAPE[1]}, half_width={HALF_WIDTH}, sigma={sim.sigma}")
    print(f"  SNRs {SNRS}, phases {PHASES}")
    return out


if __name__ == "__main__":
    run()
