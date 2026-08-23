# Progress Tracker

Maps every requirement in `CV_Eng_Assessment_Requirements.md` to its build
status and proof of completion. Updated as we go — nothing is marked done
without a specific, checkable thing that proves it (a test, a plot, a
command's output), not just "looks right."

Status legend: `NOT STARTED` / `IN PROGRESS` / `DONE`

---

## 2a. Simulator

| Requirement | Status | Verified by |
|---|---|---|
| Gaussian PSF spot renderer, known sub-pixel position | DONE | `sptrack/psf.py`; `tests/test_psf.py` (flux conserved, centroid matches injected position to 1e-7 px) |
| Photon (Poisson) noise | DONE | `sptrack/sensor.py::add_photon_noise`; `tests/test_sensor.py` (pooled mean/variance match Poisson statistics, reproducible with seed, non-negative integer counts) |
| Gaussian read noise | DONE | `sptrack/sensor.py::add_read_noise`; `tests/test_sensor.py` (pooled mean/variance match Gaussian statistics, noise magnitude independent of signal brightness, reproducible with seed) |
| Dark current | DONE | `sptrack/sensor.py::add_dark_current`; `tests/test_sensor.py` (pooled Poisson statistics, scales linearly with exposure time, additivity with photon noise verified directly, reproducible) |
| Hot pixels | DONE | `sptrack/sensor.py::generate_hot_pixel_mask`, `add_hot_pixels`; `tests/test_sensor.py` (mask fraction correct, mask is fixed not per-frame, unmasked pixels exactly untouched, elevated-rate Poisson statistics) |
| Non-uniform background gradient | DONE | `sptrack/scene.py::render_background_gradient`; `tests/test_scene.py` (flat when frac=0, mean preserved regardless of angle, axis-aligned peak-to-peak matches gradient_frac exactly, direction verified) |
| Pixel-gain non-uniformity (PRNU) | DONE | `sptrack/sensor.py::generate_prnu_map`, `apply_prnu`; `tests/test_sensor.py` (map statistics correct, map fixed with seed, multiplication correct, uniform gain leaves centroid exactly unchanged, non-uniform gain proven to bias differently at different sub-pixel offsets) |
| Bit-depth quantization | DONE | `sptrack/sensor.py::quantize_to_dn`; `tests/test_sensor.py` (basic rounding verified numerically, integer-valued output, saturation clips at top, negative electrons clip at 0 without a pedestal, quantization-error variance matches gain²/12 exactly, pedestal removes the clipping bias — matching the theoretical sigma/sqrt(2*pi) prediction) |
| SNR control (sweepable) | DONE | `sptrack/snr.py::snr_to_flux`, `flux_to_snr`, `peak_pixel_fraction`; `tests/test_snr.py` (peak fraction shrinks with wider spot, reduces to sqrt(peak) in the photon-noise-dominated limit, round-trips exactly across a range of SNR/noise combinations, flux correctly increases with a noisier background) |
| ~7 px diameter spot (1/e²), variable | DONE | `sptrack/psf.py::diameter_1e2_to_sigma`, `sample_true_sigma`; `tests/test_psf_sigma.py` (7px diameter matches the established sigma=1.75 constant, conversion verified against its own physical definition across multiple diameters, per-unit sigma statistics match the requested tolerance, floor prevents non-physical values and is confirmed to actually engage) |

## 2b. Estimators

| Requirement | Status | Verified by |
|---|---|---|
| Method 1: windowed intensity-weighted centroid + background subtraction | DONE | `sptrack/estimators/centroid.py::centroid_estimate`, `sptrack/estimators/base.py`; `tests/test_centroid.py` (recovers true position on a clean image, background subtraction proven to remove a real window-centre bias not just cosmetic, clip/no-clip agree within the understood border-leakage effect, edge-clamped window handled, approximately unbiased against the full Simulator over 300 Monte Carlo trials) |
| Method 2: 2D Gaussian fit (LSQ or MLE) | DONE | `sptrack/estimators/gaussian_fit.py::gaussian_fit_estimate`; `sptrack/psf.py::pixel_response_1d_with_derivative` (analytic Jacobian, verified against finite differences); `tests/test_gaussian_fit.py` (recovers true position/flux/bg on a clean image, approximately unbiased against 300 Monte Carlo trials, handles an edge-clamped window, fails gracefully on a degenerate all-zero image) |
| Explanation of why each behaves as it does | DONE | `tests/test_gaussian_fit.py::test_fit_is_more_precise_than_the_centroid_at_high_snr` — head-to-head on identical noisy frames, Gaussian fit measured 22% tighter (lower std) than the centroid, matching the Poisson-weighting efficiency argument in `gaussian_fit.py`'s docstring; visualized in `docs/sanity_check_gaussian_fit.png` |

## 2c. Characterization

| Requirement | Status | Verified by |
|---|---|---|
| Monte Carlo trials | DONE | `experiments/exp01_snr_characterization.py` — 10 SNR points (log-spaced 3 to 300), 300 trials each, both estimators; results in `results/exp01_snr_characterization.json` |
| Bias vs SNR, per method | DONE | Same experiment; centroid shows a large low-SNR bias (-235 millipixels at SNR=3) shrinking toward zero as SNR rises, Gaussian fit stays within a few millipixels throughout |
| Std dev vs SNR, per method | DONE | Same experiment; std curves for both methods plotted against the CRLB across the full SNR range |
| Error-vs-SNR plots | DONE | `figures/exp01_snr_characterization.png` — bias-vs-SNR and std-vs-SNR (log-log, against CRLB), with embedded explanation panel |
| Theoretical precision floor stated | DONE | `sptrack/crlb.py::position_crlb` — Fisher information built from the same Jacobian/variance model as the Gaussian fit, so the bound and the fit can't silently disagree about what model is being tested; `tests/test_crlb.py` (monotonic in flux/read noise, symmetric for a symmetric setup, converges to the classical continuous-sampling formula as sigma grows relative to the pixel pitch) |
| Comparison to the floor | DONE | `tests/test_crlb.py::test_gaussian_fit_approaches_the_crlb_at_high_snr` (single-point check, 25% tolerance) plus the full sweep in `exp01_snr_characterization.py`: mean efficiency across 10 SNR points is 0.95 for the Gaussian fit vs 0.63 for the centroid |
| Which method wins in which regime, and why | DONE | Gaussian fit wins at every SNR tested (bias and precision both) — no regime favours the centroid on accuracy; the centroid's advantage is per-frame compute cost, not precision (characterised next, §2d). See `figures/exp01_snr_characterization.png`'s embedded analysis and `docs/ASSUMPTIONS.md` |

## 2d. Real-time

| Requirement | Status | Verified by |
|---|---|---|
| Per-frame compute cost reported | NOT STARTED | |
| Which methods fit the 1 kHz / 1 ms budget | NOT STARTED | |
| Stated tradeoff | NOT STARTED | |

## 3. Dynamic Tracking

| Requirement | Status | Verified by |
|---|---|---|
| Frame sequence with moving spot | NOT STARTED | |
| Slow drift component | NOT STARTED | |
| Random jitter component | NOT STARTED | |
| One periodic disturbance component | NOT STARTED | |
| Realism justified | NOT STARTED | |
| Trajectory recovery from noisy frames | NOT STARTED | |
| Position error over sequence, measured | NOT STARTED | |
| Disturbance frequency + amplitude identified | NOT STARTED | |
| Recovered vs injected disturbance, reported | NOT STARTED | |
| Scenario deliberately made challenging | NOT STARTED | |
| Failure modes identified | NOT STARTED | |

## 4. Real-World Conditions

| Requirement | Status | Verified by |
|---|---|---|
| Conditions identified (outdoor/uncontrolled) | NOT STARTED | |
| Each: how it shows up in image | NOT STARTED | |
| Each: what it does to the estimate | NOT STARTED | |
| Each: how to detect + handle | NOT STARTED | |
| Scintillation specifically addressed | NOT STARTED | |

## 5. Go Further (optional)

| Requirement | Status | Verified by |
|---|---|---|
| Auto-exposure/gain control, graceful saturation | NOT STARTED | |
| Robustness to §4 conditions, implemented | NOT STARTED | |
| Calibration (bias/flat-field/lens-distortion) + measured effect | NOT STARTED | |
| Motion blur robustness | NOT STARTED | |
| Very low photon count robustness | NOT STARTED | |
| Precision limit derived from first principles | NOT STARTED | |
| Latency budget, full photon-to-estimate path | NOT STARTED | |

## 6. What a Strong Submission Shows (self-check before submitting)

| Requirement | Status | Verified by |
|---|---|---|
| Results quantified with error bars, not asserted | NOT STARTED | |
| Methods compared to each other AND theory | NOT STARTED | |
| Clear reasoning about noise/bias/limits | NOT STARTED | |
| Documentation another engineer could pick up | NOT STARTED | |
| Honest about assumptions and failure modes | NOT STARTED | |

## 7. Deliverables

| Requirement | Status | Verified by |
|---|---|---|
| Runnable repo, one command reproduces figures + results | NOT STARTED | |
| Written report | NOT STARTED | |
| Dynamic scenario + trajectory + disturbance analysis, reproducible | NOT STARTED | |
