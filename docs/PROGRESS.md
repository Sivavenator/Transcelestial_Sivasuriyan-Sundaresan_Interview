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
| Monte Carlo trials | DONE | `experiments/exp01_snr_characterization.py` — 10 SNR points (log-spaced 3 to 300), 300 trials each, all three estimators; results in `results/exp01_snr_characterization.json` |
| Bias vs SNR, per method | DONE | Same experiment; centroid shows a large low-SNR bias (-235 millipixels at SNR=3) shrinking toward zero as SNR rises, Gaussian fit and matched filter both stay within a few millipixels throughout |
| Std dev vs SNR, per method | DONE | Same experiment; std curves for all three methods plotted against the CRLB across the full SNR range |
| Error-vs-SNR plots | DONE | `figures/exp01_snr_characterization.png` — bias-vs-SNR and std-vs-SNR (log-log, against CRLB), with embedded explanation panel |
| Theoretical precision floor stated | DONE | `sptrack/crlb.py::position_crlb` — Fisher information built from the same Jacobian/variance model as the Gaussian fit, so the bound and the fit can't silently disagree about what model is being tested; `tests/test_crlb.py` (monotonic in flux/read noise, symmetric for a symmetric setup, converges to the classical continuous-sampling formula as sigma grows relative to the pixel pitch) |
| Comparison to the floor | DONE | `tests/test_crlb.py::test_gaussian_fit_approaches_the_crlb_at_high_snr` (single-point check, 25% tolerance) plus the full sweep in `exp01_snr_characterization.py`: mean efficiency across 10 SNR points is 0.95 (Gaussian fit), 0.84 (matched filter), 0.63 (centroid) |
| Which method wins in which regime, and why | DONE | Gaussian fit wins at every SNR tested (bias and precision both); matched filter is the accuracy/cost compromise — most of the fit's precision (log-parabola interpolation removes its curve-shape bias) at a fraction of the compute; centroid is fastest but least accurate everywhere. No regime favours the centroid on accuracy — the tradeoff across all three is speed vs. accuracy, quantified next in §2d. See `figures/exp01_snr_characterization.png`'s embedded analysis and `docs/ASSUMPTIONS.md` |

## 2d. Real-time

| Requirement | Status | Verified by |
|---|---|---|
| Per-frame compute cost reported | DONE | `experiments/exp02_realtime.py` — 1000 timed frames per method (wall-clock, Python), at SNR=50; median/p99/max in `results/exp02_realtime.json` and `figures/exp02_realtime.png` |
| Which methods fit the 1 kHz / 1 ms budget | DONE | All three fit by the p99 (worst-case) criterion: centroid ~40 us, matched filter ~84-95 us, Gaussian fit ~880-890 us p99, all under the 1000 us budget. But the Gaussian fit's slowest *observed* frame exceeded 1000 us in both runs (1010-1397 us) — a real measured tail event, since its iteration count (and therefore cost) is data-dependent and uncapped in practice by anything except `max_iter=20` |
| Stated tradeoff | DONE | Accuracy (2c) vs. cost predictability: the Gaussian fit is most accurate (efficiency 0.95) but has no hard cost ceiling; the matched filter trades a little accuracy (0.84) for fixed correlation-shaped cost with no tail risk; the centroid trades more accuracy (0.63) for the cheapest, most predictable cost. For a loop that must never miss a deadline, predictability — not the median cost — is what should decide the method, which favours the matched filter (or a hard-capped fit) over an uncapped fit. See `figures/exp02_realtime.png`'s embedded analysis |

## 3. Dynamic Tracking

| Requirement | Status | Verified by |
|---|---|---|
| Frame sequence with moving spot | DONE | `sptrack/sequence.py::render_sequence`; `tests/test_sequence.py` (correct shape, reproducible with seed) |
| Slow drift component | DONE | `sptrack/trajectory.py::generate_trajectory` — random-walk model; `tests/test_trajectory.py::test_drift_spectrum_is_concentrated_at_low_frequency` (low-band power >10x high-band power) |
| Random jitter component | DONE | Same module — iid Gaussian per frame; `tests/test_trajectory.py::test_jitter_spectrum_is_approximately_flat_and_matches_requested_std` (flat spectrum, std matches requested value) |
| One periodic disturbance component | DONE | Same module — single sinusoid; `tests/test_trajectory.py::test_disturbance_frequency_and_amplitude_recoverable_from_clean_component` (FFT recovers injected frequency to bin resolution, amplitude to 2%) |
| Realism justified | DONE | `sptrack/trajectory.py` module docstring — each component tied to a specific physical mechanism (thermal creep/settling, mechanical shake, a dominant rotating-machinery tone), with the reasoning for why each statistical model (random walk / white noise / sinusoid) matches that mechanism; spectral separability of the three demonstrated in `figures/exp03a_trajectory_diagnostic.png` |
| Trajectory recovery from noisy frames | DONE | `sptrack/sequence.py::recover_trajectory` — Gaussian fit with frame-to-frame prior gating from the estimator's own last output (never ground truth); `tests/test_sequence.py::test_recover_trajectory_survives_a_degenerate_frame_without_losing_the_prior` (a failed fit doesn't corrupt the running prior — dead reckoning proven, not just claimed), `test_recover_trajectory_tracks_a_short_moving_sequence_accurately` |
| Position error over sequence, measured | DONE | `experiments/exp03b_trajectory_recovery.py` — full 4096-frame default (easy) scenario at SNR=50: 0 failed fits, bias x=-0.00/y=0.11 millipixels, std x=8.8/y=8.9 millipixels, matching the single-frame precision already measured at this SNR in §2c/§2d (motion costs nothing extra beyond the static-frame floor); `figures/exp03b_trajectory_recovery.png` with embedded explanation panel |
| Disturbance frequency + amplitude identified | DONE | `sptrack/disturbance.py::detect_disturbance` — Hann-windowed FFT, low-frequency band excluded from the peak search (keeps drift's own power from masquerading as the disturbance), single-bin amplitude reading with Hann coherent-gain correction (tested directly against a known tone — a "sum across the leaked lobe" alternative was tried and rejected: it overestimated by 20-100%, caught by testing before being used); `tests/test_disturbance.py` (4 tests: bin-aligned tone, non-bin-aligned tone, low-frequency-exclusion actually matters, full noisy recovered trajectory) |
| Recovered vs injected disturbance, reported | DONE | `experiments/exp03c_disturbance_detection.py` — on the default (easy) scenario: freq detected 20.02 Hz vs injected 20.00 Hz (19.5 mHz error, inside the FFT's own 244 mHz bin resolution), amplitude detected 0.3001 px vs injected 0.3000 px (+0.04% error). Also checked against ground truth directly (freq 20.02 Hz, amp 0.3009 px) — nearly identical to the recovered-trajectory result, showing the Gaussian-fit recovery step (part 2) adds negligible extra error to disturbance detection at this SNR; `figures/exp03c_disturbance_detection.png` with embedded explanation panel |
| Scenario deliberately made challenging | DONE | `experiments/exp03d_hard_scenario.py` — SNR dropped 10x to 5.0 (chosen so the fit's own precision, ~0.132 px, becomes comparable to jitter itself, ~0.15 px, rather than negligible as in the easy scenario); disturbance amplitude swept from 0.30 px down to and through the measured noise floor (0.033 px); disturbance frequency (2.5 Hz) placed near the drift-exclusion boundary (2.0 Hz) |
| Failure modes identified | DONE | Two distinct, real (Monte Carlo, 10 trials/level) failure modes found: (1) amplitude-bias — as true amplitude approaches the noise floor, detected amplitude stops tracking truth and flattens toward the floor (0.033 px even at true amp=0.0), a textbook Rice/Rayleigh peak-detection bias, not a bug — frequency reliability degrades in step (100% within one bin at amp=0.30/0.10, 80% at 0.05, 50% at 0.02); (2) exclusion-boundary blind spot — placing the disturbance frequency at/near the fixed `exclude_below_hz=2.0` threshold causes near-total failure (wrong frequency AND badly wrong amplitude) even at an easily-detectable amplitude (0.3px, SNR=50) — a qualitatively different, more severe failure than the graceful amplitude-bias degradation. `figures/exp03d_hard_scenario.png` with embedded explanation panel |

## 4. Real-World Conditions

| Requirement | Status | Verified by |
|---|---|---|
| Conditions identified (outdoor/uncontrolled) | DONE | `docs/REAL_WORLD_CONDITIONS.md` — 5 conditions: atmospheric scintillation, beam wander, background clutter/false sources, solar glare/non-uniform background, fog/haze/rain attenuation |
| Each: how it shows up in image | DONE | Same document, each condition's own "Physical mechanism" section |
| Each: what it does to the estimate | DONE | Same document, each condition's own "What it does to the estimate" section — several tied to specific, real vulnerabilities already in this codebase (e.g. `find_brightest_pixel`'s clutter vulnerability, `border_median_background`'s gradient-breaking assumption), not generic statements |
| Each: how to detect + handle | DONE | Same document, each condition's own "Detect and handle" section |
| Scintillation specifically addressed | DONE | `sptrack/scintillation.py::generate_scintillation` — mean-reverting log-normal (AR(1)) flux-multiplier process, correlation time comparable to the frame period (not independent per-frame noise); `tests/test_scintillation.py` (6 tests: reproducibility, positivity, unbiased mean, correct log-std, autocorrelation matches AR(1) theory, shorter coherence time decorrelates faster). Impact quantified in `experiments/exp04a_scintillation.py`: overall std 1.7x worse with scintillation (227 vs 137 millipixels), std during fades 5.1x worse than during peaks (391 vs 77 mpx), 25/4096 genuine dropout frames (vs 0 with steady flux) — all bridged by §3's existing dead-reckoning mechanism rather than losing the track; `figures/exp04a_scintillation.png` with embedded explanation panel |

## 5. Go Further (optional)

| Requirement | Status | Verified by |
|---|---|---|
| Auto-exposure/gain control, graceful saturation | DONE | `sptrack/agc.py::AutoExposureController` — proportional controller retargeting an effective gain each frame toward a target peak-DN band, bounded per-step change; `tests/test_agc.py` (5 tests: converges after a large brightness jump, gain bounds respected, step-size bounds respected, reacts correctly to dim/bright scenes). `experiments/exp05a_auto_exposure.py`: across a 5-decade brightness sweep, fixed exposure's worst-case std is 133.4 mpx (and fails outright at the top of the range); AGC's worst-case is 3.0 mpx across the entire range. Found and documented a genuine, non-obvious asymmetry: recovering from underexposure took 9 frames, but recovering from saturation took 31 — a saturated (clipped) reading corrupts the feedback signal's magnitude information, so the controller can only tell "still too bright," not by how much, forcing many small conservative corrections instead of one confident one. This also caught a real bug in the experiment itself: an initial fixed 8-iteration settling budget silently left the brightest sweep levels unconverged, comparing AGC's mid-correction state against the fixed baseline's steady state — fixed by adaptive settling (loop until converged, not a fixed count) |
| Robustness to §4 conditions, implemented | DONE | Beyond the brief's explicit ask (only scintillation was required to be simulated, §4) — user directed simulating the remaining conditions too, and building real mitigations, not just demonstrations, for the two where one was tractable. **Fog/rain**: `experiments/exp04b_fog_attenuation.py` — named-weather attenuation (dB/km) swept through existing SNR/flux machinery; found a hard operational cliff between haze and light fog, and that dropout rate alone understates failure at moderate/dense fog (std explodes to ~2px — noise-driven, not real detections). **Beam wander**: `sptrack/beam_wander.py` — mean-reverting (OU/AR(1)) position noise, physically distinct from mechanical jitter; `tests/test_beam_wander.py` (5 tests) proves equal-variance jitter and wander are still ~250x separable by spectral shape; `experiments/exp04c_beam_wander.py` confirms this, the independent-sources quadrature-sum noise budget, and flags a real interaction with §3's disturbance-detector exclusion boundary. **Background clutter**: `sptrack/acquisition.py::acquire_target` — PSF-shape-matched acquisition (Pearson correlation against the assumed Gaussian template, scale-invariant) replacing raw brightest-pixel; `tests/test_acquisition.py` (5 tests) proves a real clutter source (13x the true spot's flux, wider profile) fools `find_brightest_pixel` outright while `acquire_target` correctly picks the true spot on the identical frame; `experiments/exp04d_clutter.py` visualizes both side by side. **Solar glare**: `sptrack/estimators/base.py::planar_background` — least-squares planar background fit replacing a single scalar median; `tests/test_planar_background.py` (4 tests) proves a real gradient-induced bias (0.3-2.5px, growing with gradient strength — far larger than most other systematic biases in this project) is eliminated (stays at the ~0.0001px noise floor across the whole swept range) by fitting and subtracting the FULL planar prediction rather than one constant; `experiments/exp04e_glare.py` sweeps gradient strength and compares both approaches directly. |
| Calibration (bias/flat-field/lens-distortion) + measured effect | DONE | `sptrack/calibration.py` — `estimate_bias_frame` (hot-pixel pattern, N=100 frames derived from read-noise so the map's own residual noise is <10% of a single frame's), `estimate_flat_field` (PRNU gain map, brightness/frame-count derived from prnu_sigma=0.02 for a 10x SNR safety margin), `apply_radial_distortion`/`correct_radial_distortion` (single-term radial model at -0.1% at the frame edge — magnitude sourced from cited machine-vision lens datasheets, confirmed with the user, not guessed: Commonlands CIL052 -0.1%, MYUTRON FV <0.1%); `tests/test_calibration.py` (6 tests: hot pixels detected, residual noise shrinks with more frames, PRNU map recovered to well below its own sigma, distortion no-op at centre, displacement matches the configured %, exact round-trip through distortion+correction). `experiments/exp05b_calibration.py` measures all three: bias-corrected mean drops from +26.1 to -8.7 millipixels; flat-field RMS bias drops 5.1x (20.3 to 4.0 millipixels); distortion correction removes essentially all of a 0.212px edge-of-frame geometric error (down to the fixed-point solve's own ~1e-9px precision) — the only one of the three that is a pure geometric effect, present even with a perfect noiseless estimator |
| Motion blur robustness | DONE | `sptrack/motion_blur.py::render_motion_blurred_spot` — temporal supersampling of `render_spot` along the motion path (61 substeps, checked against the analytic box-convolution variance formula sigma^2+blur^2/12); `tests/test_motion_blur.py` (6 tests: zero-blur matches static render exactly, flux conserved, centroid unbiased, variance formula matches, blur doesn't leak into the perpendicular axis, angle rotates the blur direction correctly). Blur magnitude swept as a fraction of sigma (0-3x), not a claimed real-world velocity — a specific figure was researched (a real FSO fine-steering-mirror slew rate, 1.5 mrad/s, Bramall et al.) but couldn't be converted to pixels without a plate-scale assumption this project has no basis for, confirmed with the user rather than guessed. `experiments/exp05c_motion_blur.py`: bias stays close to each method's own zero-blur value even at 3-sigma blur (motion blur is a precision problem, not a new bias source, since constant-velocity blur is symmetric about the true position); precision degrades 1.3-1.8x from no blur to 3-sigma blur across the three estimators |
| Very low photon count robustness | NOT STARTED | |
| Precision limit derived from first principles | NOT STARTED | |
| Latency budget, full photon-to-estimate path | NOT STARTED | |
| Third estimator (matched filter / correlation peak) — "anything else you think matters" | DONE | `sptrack/estimators/matched_filter.py::matched_filter_estimate`; `tests/test_matched_filter.py` (log-parabola interpolation proven exact for noiseless Gaussian samples, plain parabola proven measurably biased on the same data, correlation peak width matches the sqrt(2) detection/localisation-tension prediction, approximately unbiased against the real noise chain); folded into `experiments/exp01_snr_characterization.py` — mean efficiency 0.84, between the centroid (0.63) and the Gaussian fit (0.95). Built for real-time relevance (fixed-cost convolution vs. variable-cost iteration), not required by the brief |

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
