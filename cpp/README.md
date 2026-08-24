# C++ port

Header-only C++17 port of the centroid and Gaussian-fit estimators, for a
deployment target where a Python process is not appropriate. See
`include/sptrack/spot_estimators.hpp` for the design reasoning and
`docs/DESIGN_RATIONALE.md` section 4 for why Gauss-Newton with an
analytic Jacobian was the algorithm in the first place.

## Status

Not built or run on the development machine used for the rest of this
repository: no C++ compiler, no CMake, and no WSL are present there. The
code is written to compile against a standard C++17 toolchain and its
correctness rests on `cpp/tests/test_against_python.cpp`, which has not
itself been executed. Build and validate it before trusting it, most
directly inside the Ubuntu 24.04 VirtualBox VM already used to validate
this project's Python dependencies.

Two known cross-language pitfalls were found and fixed by inspection
before any build was attempted, not by testing, since testing was not
possible here:

- Python's `round()` is round-half-to-even; `std::lround` rounds half
  away from zero. At an exact `.5` sub-pixel position the two would pick
  different window origins and silently disagree by a pixel.
  `extract_window` uses `std::nearbyint` instead, which defaults to
  round-half-to-even and matches Python. Confirmed directly:
  `round(10.5) == 10` and `round(11.5) == 12` in the Python this project
  runs.
- `numpy`'s `repr()` under numpy 2 emits `np.float64(...)`, which no C++
  numeric parser accepts. `tools/export_cpp_vectors.py` forces a plain
  Python `float` before writing the vector file.

Because these were caught by inspection rather than by running the
comparison, the build and test step below is not optional: it is the
first time the two implementations are actually checked against each
other.

## Build and test

```bash
sudo apt install build-essential cmake
cd sub-pixel-tracker

# Regenerate the cross-validation vectors from the current Python
# reference. Do this again after any change to sptrack/estimators/.
python -m tools.export_cpp_vectors

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

`ctest` runs `test_against_python`, which loads `results/cpp_vectors.csv`
and compares the C++ output to the Python output on the same pixels for
every case. It exits nonzero on any disagreement outside tolerance
(1e-9 px for the centroid, 1e-6 px for the fit), so a passing `ctest` is
the actual claim that the port is correct, not an assumption behind it.

## Benchmark

```bash
./cpp/build/bench
```

Reports median, p99 and maximum per-frame cost for the centroid and the
Gaussian fit, in the same format as
`experiments/exp02_realtime.py`, so the two can be compared directly once
both have real numbers from the same machine. The Python numbers alone
already showed the fit's worst observed frame exceeding the 1 ms budget;
whether the compiled version has the same tail is an open question this
benchmark answers, not one this port assumes an answer to.

## What was ported and what was not

Ported: the centroid and the Gaussian fit, since those are the two
estimators the brief requires and the ones `exp02_realtime.py` measured
as bounded by Python overhead. Not ported: the matched filter, the
temporal filters in `sptrack/tracking.py`, and everything upstream of
per-frame estimation (the simulator, calibration, disturbance detection).
The scope was the part of the pipeline where a compiled implementation
changes the real-time picture, not the whole repository.
