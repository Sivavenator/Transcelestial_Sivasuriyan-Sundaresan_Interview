// Per-frame cost of the C++ estimators, for comparison against the
// Python numbers in results/exp02_realtime.json.
//
// WHY THIS REPORTS PERCENTILES RATHER THAN A MEAN
// -----------------------------------------------
// A 1 kHz loop has to finish every frame, not the average frame. A method
// that is fast 999 times in 1000 and slow once still misses a deadline
// once a second. The mean is what a benchmark headline quotes; the tail
// is what makes a control loop drop a sample. This prints median, p99 and
// maximum for the same reason experiments/exp02_realtime.py does.
//
// WHAT THIS DOES NOT MEASURE
// --------------------------
// Sensor readout and data transfer, which the latency budget in
// experiments/exp05e_latency_budget.py shows dominate the photon-to-
// estimate path. This is the compute stage alone.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "sptrack/spot_estimators.hpp"

namespace {

constexpr int kFrameSize = 21;
constexpr int kHalfWidth = 9;
constexpr double kSigma = 1.75;
constexpr double kReadVar = 25.0;
constexpr double kBackground = 30.0;
constexpr int kWarmup = 200;
constexpr int kFrames = 20000;

double percentile(std::vector<double> v, double frac) {
    if (v.empty()) {
        return 0.0;
    }
    const std::size_t idx = static_cast<std::size_t>(frac * (static_cast<double>(v.size()) - 1.0));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx), v.end());
    return v[idx];
}

// Build a noisy frame with the same structure the Python simulator
// produces. The RNG stream does not need to match Python here: this
// measures cost, not agreement, and agreement is checked separately by
// cpp/tests/test_against_python.cpp.
std::vector<double> make_frame(std::mt19937_64& rng, double flux, double x0, double y0) {
    std::vector<double> frame(static_cast<std::size_t>(kFrameSize) * kFrameSize, 0.0);
    std::normal_distribution<double> read_noise(0.0, std::sqrt(kReadVar));
    for (int row = 0; row < kFrameSize; ++row) {
        const double py = sptrack::detail::pixel_response(row, y0, kSigma);
        for (int col = 0; col < kFrameSize; ++col) {
            const double px = sptrack::detail::pixel_response(col, x0, kSigma);
            const double mean = flux * px * py + kBackground;
            std::poisson_distribution<int> shot(mean);
            frame[static_cast<std::size_t>(row) * kFrameSize + col] =
                static_cast<double>(shot(rng)) + read_noise(rng);
        }
    }
    return frame;
}

}  // namespace

int main() {
    std::mt19937_64 rng(20260824);
    const double flux = 50648.0;  // SNR ~50, matching exp02_realtime.py
    const double x0 = 10.3;
    const double y0 = 9.7;

    std::vector<std::vector<double>> frames;
    frames.reserve(kFrames);
    for (int i = 0; i < kFrames; ++i) {
        frames.push_back(make_frame(rng, flux, x0, y0));
    }

    std::printf("C++ per-frame compute cost, %d frames, %dx%d window half-width %d\n",
                kFrames, kFrameSize, kFrameSize, kHalfWidth);
    std::printf("%-16s %10s %10s %10s\n", "method", "median_us", "p99_us", "max_us");

    volatile double sink = 0.0;

    for (int method = 0; method < 2; ++method) {
        // Untimed warm-up so first-call effects do not enter the numbers.
        for (int i = 0; i < kWarmup; ++i) {
            sptrack::FrameView view{frames[static_cast<std::size_t>(i) % frames.size()].data(),
                                    kFrameSize, kFrameSize};
            const sptrack::Estimate e =
                (method == 0)
                    ? sptrack::centroid_estimate(view, x0, y0, kHalfWidth)
                    : sptrack::gaussian_fit_estimate(view, x0, y0, kHalfWidth, kSigma, kReadVar);
            sink += e.x;
        }

        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(kFrames));
        for (int i = 0; i < kFrames; ++i) {
            sptrack::FrameView view{frames[static_cast<std::size_t>(i)].data(), kFrameSize, kFrameSize};
            const auto start = std::chrono::steady_clock::now();
            const sptrack::Estimate e =
                (method == 0)
                    ? sptrack::centroid_estimate(view, x0, y0, kHalfWidth)
                    : sptrack::gaussian_fit_estimate(view, x0, y0, kHalfWidth, kSigma, kReadVar);
            const auto stop = std::chrono::steady_clock::now();
            sink += e.x;
            times.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
        }

        const char* label = (method == 0) ? "centroid" : "gaussian fit";
        std::printf("%-16s %10.3f %10.3f %10.3f\n", label, percentile(times, 0.50),
                    percentile(times, 0.99), *std::max_element(times.begin(), times.end()));
    }

    if (sink == 12345.6789) {
        std::printf("");  // keep the optimiser from discarding the work
    }
    return 0;
}
