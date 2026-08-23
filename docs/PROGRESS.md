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
| Gaussian PSF spot renderer, known sub-pixel position | NOT STARTED | |
| Photon (Poisson) noise | NOT STARTED | |
| Gaussian read noise | NOT STARTED | |
| Dark current | NOT STARTED | |
| Hot pixels | NOT STARTED | |
| Non-uniform background gradient | NOT STARTED | |
| Pixel-gain non-uniformity (PRNU) | NOT STARTED | |
| Bit-depth quantization | NOT STARTED | |
| SNR control (sweepable) | NOT STARTED | |
| ~7 px diameter spot (1/e²), variable | NOT STARTED | |

## 2b. Estimators

| Requirement | Status | Verified by |
|---|---|---|
| Method 1: windowed intensity-weighted centroid + background subtraction | NOT STARTED | |
| Method 2: 2D Gaussian fit (LSQ or MLE) | NOT STARTED | |
| Explanation of why each behaves as it does | NOT STARTED | |

## 2c. Characterization

| Requirement | Status | Verified by |
|---|---|---|
| Monte Carlo trials | NOT STARTED | |
| Bias vs SNR, per method | NOT STARTED | |
| Std dev vs SNR, per method | NOT STARTED | |
| Error-vs-SNR plots | NOT STARTED | |
| Theoretical precision floor stated | NOT STARTED | |
| Comparison to the floor | NOT STARTED | |
| Which method wins in which regime, and why | NOT STARTED | |

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
