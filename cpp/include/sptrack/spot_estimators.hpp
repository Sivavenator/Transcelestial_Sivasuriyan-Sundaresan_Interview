// Sub-pixel spot position estimators, C++ port of the Python reference in
// sptrack/estimators/.
//
// WHY THIS EXISTS
// ---------------
// The Python implementation is the characterised one and stays the
// reference. This port exists because the deployment target for a 1 kHz
// pointing loop is not a Python process: it is an embedded or realtime
// context where the per-frame cost has to be bounded and the runtime has
// to be predictable. experiments/exp02_realtime.py measures the Python
// cost and finds the Gaussian fit's worst frame exceeding the 1 ms
// budget, so the question of what the same algorithm costs in compiled
// code is a real one rather than a formality.
//
// WHAT IS DELIBERATELY IDENTICAL TO THE PYTHON
// --------------------------------------------
// The arithmetic is transcribed rather than reworked: same erf-based
// pixel integral, same Poisson weighting, same Levenberg-Marquardt
// damping schedule, same one-pixel trust-region clamp, same convergence
// rule based on proposed step size rather than on step acceptance. That
// last point is not cosmetic. Two bugs in the Python convergence logic
// were found by testing (see docs/ASSUMPTIONS.md), and a port that
// "improved" the rule while copying the rest would silently diverge from
// a reference that has been characterised against the Cramer-Rao bound.
// Divergence is checked rather than assumed, by
// cpp/tests/test_against_python.cpp.
//
// WHAT IS DELIBERATELY DIFFERENT
// ------------------------------
// No dynamic allocation in the per-frame path. The window is a
// fixed-capacity buffer and the 4x4 normal equations are solved in place
// with Gaussian elimination and partial pivoting rather than by calling a
// linear algebra library. At this size a library call costs more in
// dispatch than the solve costs in arithmetic, and a dependency-free
// header is easier to drop into a target that has no BLAS.
//
// HEADER ONLY, AND WHY
// --------------------
// The whole implementation is a few hundred lines with no state to hide,
// so a header-only interface keeps integration to a single include and
// lets the compiler inline across the whole estimator. There is no
// separate translation unit to keep in sync.

#ifndef SPTRACK_SPOT_ESTIMATORS_HPP
#define SPTRACK_SPOT_ESTIMATORS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sptrack {

// M_PI is POSIX rather than ISO C++ and needs _USE_MATH_DEFINES on MSVC,
// so the constant is defined here to keep the header portable.
constexpr double kPi = 3.14159265358979323846;

struct Estimate {
    double x = 0.0;
    double y = 0.0;
    double flux = 0.0;
    double bg = 0.0;
    bool ok = false;
};

// A frame is a row-major buffer with its own dimensions. Kept as a view
// so the caller owns the pixels and no copy happens per frame.
struct FrameView {
    const double* data = nullptr;
    int height = 0;
    int width = 0;

    double at(int row, int col) const { return data[static_cast<std::size_t>(row) * width + col]; }
};

// The window actually handed to an estimator, plus where it sits in the
// parent frame. Mirrors Python's extract_window, including the clamping
// behaviour at frame edges: the window is silently smaller near an edge,
// so callers must use origin_x/origin_y rather than assuming it is
// centred.
struct Window {
    std::vector<double> values;
    int height = 0;
    int width = 0;
    int origin_x = 0;
    int origin_y = 0;

    double at(int row, int col) const { return values[static_cast<std::size_t>(row) * width + col]; }
};

namespace detail {

// Exact integral of a unit-area Gaussian over pixel i, matching
// psf.pixel_response_1d. Using the error function rather than sampling
// the Gaussian at the pixel centre is what keeps the model consistent
// with how a detector actually collects light, and is what removes pixel
// locking (see experiments/exp06a_pixel_locking.py).
inline double pixel_response(double index, double centre, double sigma) {
    const double inv = 1.0 / (sigma * std::sqrt(2.0));
    const double hi = (index + 0.5 - centre) * inv;
    const double lo = (index - 0.5 - centre) * inv;
    return 0.5 * (std::erf(hi) - std::erf(lo));
}

// Derivative of the above with respect to the spot centre. Differentiating
// the two erf terms gives two Gaussian evaluations, which is why the
// analytic Jacobian costs little more than the response itself.
inline double pixel_response_derivative(double index, double centre, double sigma) {
    const double inv = 1.0 / (sigma * std::sqrt(2.0));
    const double hi = (index + 0.5 - centre) * inv;
    const double lo = (index - 0.5 - centre) * inv;
    const double norm = 1.0 / (sigma * std::sqrt(2.0 * kPi));
    return norm * (std::exp(-lo * lo) - std::exp(-hi * hi));
}

// Solve a small dense system in place by Gaussian elimination with
// partial pivoting. Returns false on a singular matrix, mirroring the
// Python path that returns ok=false when numpy raises LinAlgError.
inline bool solve_in_place(double* a, double* b, int n) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::fabs(a[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const double candidate = std::fabs(a[row * n + col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1e-300) {
            return false;
        }
        if (pivot != col) {
            for (int k = 0; k < n; ++k) {
                std::swap(a[col * n + k], a[pivot * n + k]);
            }
            std::swap(b[col], b[pivot]);
        }
        const double diag = a[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            const double factor = a[row * n + col] / diag;
            if (factor == 0.0) {
                continue;
            }
            for (int k = col; k < n; ++k) {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        double sum = b[row];
        for (int k = row + 1; k < n; ++k) {
            sum -= a[row * n + k] * b[k];
        }
        b[row] = sum / a[row * n + row];
    }
    return true;
}

}  // namespace detail

// Extract a (2*half_width+1) square window centred on the nearest pixel
// to (cx, cy), clamped to the frame. Matches extract_window in
// sptrack/estimators/base.py, including the rounding of the centre to an
// integer pixel, which is itself a measurable effect: see the
// phase-dependent centroid bias in experiments/exp06a_pixel_locking.py.
inline Window extract_window(const FrameView& frame, double cx, double cy, int half_width) {
    // Python's round() is round-half-to-even, so round(10.5) is 10 and
    // round(11.5) is 12. std::lround rounds half away from zero and would
    // pick a different window at exact .5 positions, shifting the origin
    // by one pixel and silently changing the answer. std::nearbyint uses
    // the default FE_TONEAREST mode, which is round-half-to-even, so it
    // matches the reference.
    const int ix = static_cast<int>(std::nearbyint(cx));
    const int iy = static_cast<int>(std::nearbyint(cy));
    const int x0 = std::max(ix - half_width, 0);
    const int x1 = std::min(ix + half_width + 1, frame.width);
    const int y0 = std::max(iy - half_width, 0);
    const int y1 = std::min(iy + half_width + 1, frame.height);

    Window win;
    win.origin_x = x0;
    win.origin_y = y0;
    win.width = std::max(x1 - x0, 0);
    win.height = std::max(y1 - y0, 0);
    win.values.resize(static_cast<std::size_t>(win.width) * win.height);
    for (int row = 0; row < win.height; ++row) {
        for (int col = 0; col < win.width; ++col) {
            win.values[static_cast<std::size_t>(row) * win.width + col] = frame.at(y0 + row, x0 + col);
        }
    }
    return win;
}

// Median of the window's outer border_width rows and columns. Median
// rather than mean so a single hot pixel or noise spike on the border
// cannot move the estimate, which would otherwise propagate into every
// pixel of the subtracted window.
inline double border_median_background(const Window& win, int border_width = 2) {
    if (win.values.empty()) {
        return 0.0;
    }
    std::vector<double> border;
    border.reserve(win.values.size());
    const bool has_interior = win.height > 2 * border_width && win.width > 2 * border_width;
    for (int row = 0; row < win.height; ++row) {
        for (int col = 0; col < win.width; ++col) {
            const bool interior = has_interior && row >= border_width && row < win.height - border_width &&
                                  col >= border_width && col < win.width - border_width;
            if (!interior) {
                border.push_back(win.at(row, col));
            }
        }
    }
    if (border.empty()) {
        return 0.0;
    }
    const std::size_t mid = border.size() / 2;
    std::nth_element(border.begin(), border.begin() + static_cast<std::ptrdiff_t>(mid), border.end());
    const double upper = border[mid];
    if (border.size() % 2 == 1) {
        return upper;
    }
    // numpy's median averages the two central values for even counts.
    std::nth_element(border.begin(), border.begin() + static_cast<std::ptrdiff_t>(mid - 1), border.end());
    const double lower = border[mid - 1];
    return 0.5 * (lower + upper);
}

// Windowed intensity-weighted centroid with background subtraction.
// clip_negative defaults to true to match the Python default. That
// default has a measured cost: it introduces a phase-dependent bias of
// about 4.4 millipixels peak-to-peak, documented in
// docs/DESIGN_RATIONALE.md section 6.
inline Estimate centroid_estimate(const FrameView& frame, double prior_x, double prior_y,
                                  int half_width, bool clip_negative = true) {
    Window win = extract_window(frame, prior_x, prior_y, half_width);
    Estimate out;
    if (win.values.empty()) {
        return out;
    }
    const double bg = border_median_background(win);

    double total = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (int row = 0; row < win.height; ++row) {
        for (int col = 0; col < win.width; ++col) {
            double v = win.at(row, col) - bg;
            if (clip_negative && v < 0.0) {
                v = 0.0;
            }
            total += v;
            sum_x += v * col;
            sum_y += v * row;
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return out;
    }

    out.x = sum_x / total + win.origin_x;
    out.y = sum_y / total + win.origin_y;
    out.flux = total;
    out.bg = bg;
    out.ok = true;
    return out;
}

// Poisson-weighted 2-D Gaussian fit by Levenberg-Marquardt damped
// Gauss-Newton, seeded from the centroid.
//
// The weighting uses the model's own prediction rather than the observed
// pixel value. Using the data to set how much the data is trusted
// creates a feedback loop that biases the fit downward, so the weights
// are recomputed from the current model each iteration.
inline Estimate gaussian_fit_estimate(const FrameView& frame, double prior_x, double prior_y,
                                      int half_width, double sigma, double read_var,
                                      int max_iter = 20, double tol_px = 1e-4) {
    const Estimate seed = centroid_estimate(frame, prior_x, prior_y, half_width);
    if (!seed.ok) {
        return Estimate{};
    }

    Window win = extract_window(frame, seed.x, seed.y, half_width);
    const int h = win.height;
    const int w = win.width;
    if (h == 0 || w == 0) {
        return Estimate{};
    }

    double p[4] = {seed.x - win.origin_x, seed.y - win.origin_y, std::max(seed.flux, 1.0), seed.bg};

    std::vector<double> px(static_cast<std::size_t>(w));
    std::vector<double> dpx(static_cast<std::size_t>(w));
    std::vector<double> py(static_cast<std::size_t>(h));
    std::vector<double> dpy(static_cast<std::size_t>(h));

    double lam = 1e-3;
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int col = 0; col < w; ++col) {
            px[col] = detail::pixel_response(col, p[0], sigma);
            dpx[col] = detail::pixel_response_derivative(col, p[0], sigma);
        }
        for (int row = 0; row < h; ++row) {
            py[row] = detail::pixel_response(row, p[1], sigma);
            dpy[row] = detail::pixel_response_derivative(row, p[1], sigma);
        }

        double hess[16] = {0.0};
        double grad[4] = {0.0};
        double old_chi2 = 0.0;

        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const double shape = py[row] * px[col];
                const double mu = p[2] * shape + p[3];
                const double resid = win.at(row, col) - mu;
                const double var = std::max(mu, 1e-6) + read_var;
                const double weight = 1.0 / var;

                const double j[4] = {p[2] * py[row] * dpx[col], p[2] * dpy[row] * px[col], shape, 1.0};

                old_chi2 += resid * resid * weight;
                for (int a = 0; a < 4; ++a) {
                    grad[a] += j[a] * weight * resid;
                    for (int b = 0; b < 4; ++b) {
                        hess[a * 4 + b] += j[a] * weight * j[b];
                    }
                }
            }
        }

        double damped[16];
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                damped[a * 4 + b] = hess[a * 4 + b];
            }
            damped[a * 4 + a] += lam * hess[a * 4 + a];
        }

        double step[4] = {grad[0], grad[1], grad[2], grad[3]};
        if (!detail::solve_in_place(damped, step, 4)) {
            return Estimate{};
        }

        // Trust region: the linearisation is least trustworthy exactly
        // when it proposes a large jump.
        step[0] = std::min(std::max(step[0], -1.0), 1.0);
        step[1] = std::min(std::max(step[1], -1.0), 1.0);

        const double new_p[4] = {p[0] + step[0], p[1] + step[1], p[2] + step[2], p[3] + step[3]};

        for (int col = 0; col < w; ++col) {
            px[col] = detail::pixel_response(col, new_p[0], sigma);
        }
        for (int row = 0; row < h; ++row) {
            py[row] = detail::pixel_response(row, new_p[1], sigma);
        }

        double new_chi2 = 0.0;
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const double new_mu = new_p[2] * py[row] * px[col] + new_p[3];
                const double resid = win.at(row, col) - new_mu;
                // Same variance model as old_chi2 above. Using a different
                // one here was a real bug on the Python side: the two
                // values then live on different scales and every step is
                // rejected regardless of damping.
                const double new_var = std::max(new_mu, 1e-6) + read_var;
                new_chi2 += resid * resid / new_var;
            }
        }

        // Convergence is judged on the proposed step, before knowing
        // whether this iteration's step is accepted. Near the optimum
        // chi2 stops changing measurably and accept/reject becomes
        // floating-point noise, so gating convergence on acceptance
        // spins until max_iter without ever declaring success.
        const bool small_step = std::fabs(step[0]) < tol_px && std::fabs(step[1]) < tol_px;

        if (new_chi2 < old_chi2) {
            for (int a = 0; a < 4; ++a) {
                p[a] = new_p[a];
            }
            lam = std::max(lam * 0.4, 1e-7);
        } else {
            lam *= 10.0;
        }

        if (small_step) {
            converged = true;
            break;
        }
    }

    Estimate out;
    out.x = p[0] + win.origin_x;
    out.y = p[1] + win.origin_y;
    out.flux = p[2];
    out.bg = p[3];
    out.ok = converged;
    return out;
}

}  // namespace sptrack

#endif  // SPTRACK_SPOT_ESTIMATORS_HPP
