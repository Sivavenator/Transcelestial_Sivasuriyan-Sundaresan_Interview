# CV Engineer Take-Home, Assessment Requirements

Source: `CV_Engineer_Take_Home (Siva).docx`. This document is our own bullet-point
breakdown of the brief, read and confirmed section by section before any code
was written, so we always know exactly what we're building against.

---

## 1. Context

- Two terminals, each has: a camera + a gimbal-mounted laser
- Both platforms vibrate randomly (shake)
- Goal: each laser stays pointed at the other terminal
- Camera's job: see the incoming laser spot, find its position to sub-pixel precision
- Loop runs at ~1 kHz (1000 times/sec = every 1 ms)
- Camera output → feeds a controller → controller steers the gimbal
- Spot's brightness changes depending on environment (camera settings + conditions)
- Sensor problems to expect: noise, exposure drift, gain drift, temperature drift, background drift
- Must get sub-pixel precision AND real-time 1kHz position, together
- Key line: must defend every decision live, that's the real grading criterion

## 2. Core Task (baseline, required)

### 2a. Simulator

- Build a fake image generator, synthetic camera frames of a laser spot
- Spot shape: Gaussian blob, known sub-pixel position (ground truth)
- Noise types required, ALL of them:
  - Photon noise (Poisson)
  - Gaussian read noise
  - Dark current
  - Hot pixels
  - Non-uniform background gradient
  - Pixel-gain non-uniformity (PRNU)
  - Bit-depth quantization
- SNR control, dial to sweep signal-to-noise
- Spot size: ~7 px diameter at 1/e² point, can vary slightly

### 2b. Estimators

- At least 2 different methods to find position from noisy image
- Example 1: windowed intensity-weighted centroid + background subtraction
- Example 2: 2D Gaussian fit (least-squares or MLE)
- Must explain WHY each behaves the way it does

### 2c. Characterization

- Monte Carlo, many random noisy trials
- Measure per method, per SNR: bias (systematic skew) + standard deviation (scatter)
- Produce error-vs-SNR plots
- State theoretical precision floor, show how close you get to it
- Explain which method wins in which SNR regime, and why

### 2d. Real-time

- Report per-frame compute cost
- State which methods fit inside the 1 kHz (1 ms) budget
- State the tradeoff you'd make

## 3. Dynamic Tracking (build your own)

- Extend simulator: sequence of frames, spot moves
- Motion = 3 components mixed: slow drift + random jitter + one periodic disturbance
- Justify why these are realistic
- You know ground truth (you generated it)
- Recover the trajectory, then:
  - Measure position error over time
  - Detect the periodic disturbance's frequency + amplitude
  - Report how close detected values are to injected values
- Design it to be HARD on purpose: low SNR, disturbance near jitter floor, disturbance frequency near noise floor
- Identify failure modes
- The parameter choices themselves are graded, not just success

## 4. Real-World Conditions

- Real deployment = outdoors, uncontrolled
- Identify conditions that would degrade the system
- For each: how it shows up in image / what it does to the estimate / how to detect + handle it
- Analysis quality > implementation for this section
- Named example: scintillation, simulate its impact, make system robust to it

## 5. Go Further (optional, bonus)

- Nothing here required, shows judgment on where to spend effort
- Auto-exposure/gain control: brightness changes with graceful saturation handling
- Build actual robustness to conditions identified in §4
- Calibration (bias/flat-field/lens-distortion) + measure its effect on precision
- Motion blur robustness, or very low photon counts
- Derive precision limit from first principles (not just measure it)
- Latency budget: full photon → position-estimate path, where your algorithm sits in it
- Anything else you think matters, surprising them is allowed

## 6. What a Strong Submission Shows

- Quantified, not asserted, numbers with error bars
- Compared, methods vs each other AND vs theoretical reference
- Clear reasoning about noise/bias/limits, not just working code
- Documentation someone else could pick up: what/why/how-to-reproduce/what-tested/where-it-breaks/what's-next
- Honest about assumptions and failure modes

## 7. Deliverables

- Runnable repo, one command reproduces all figures + results
- A written report
- Dynamic scenario + recovered trajectory + disturbance analysis, reproducible from repo

## 8. Ground Rules

- No page limit, no time cap
- State your own assumptions; only ask if genuinely blocked
- AI tools allowed, but you must understand and defend everything live
