// Cross-validate the C++ estimators against the Python reference.
//
// Reads results/cpp_vectors.csv, produced by tools/export_cpp_vectors.py.
// Each row carries a full frame plus the positions the Python estimators
// produced for it. This runs the C++ estimators on the same pixels and
// requires agreement.
//
// WHY THE TOLERANCES ARE WHAT THEY ARE
// ------------------------------------
// The centroid is a straight weighted sum over the same pixels in the
// same order, so the only difference between the two implementations is
// floating-point summation order. Agreement should be at the level of
// accumulated rounding, hence 1e-9 px.
//
// The Gaussian fit iterates, and each iteration solves a 4x4 system.
// Python uses LAPACK via numpy.linalg.solve; this uses Gaussian
// elimination with partial pivoting. Those are different algorithms with
// different rounding, so the iterates can differ in the last few bits and
// the difference compounds across iterations. The tolerance is 1e-6 px,
// which is well below the tightest precision this project ever measures
// (a few millipixels) while still being tight enough that a genuine
// algorithmic divergence, a wrong Jacobian term or a different
// convergence rule, would fail loudly rather than hide.
//
// Exit code is 0 on success and 1 on any failure, so ctest and CI treat
// a divergence as a build failure.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sptrack/spot_estimators.hpp"

namespace {

constexpr double kCentroidTolPx = 1e-9;
constexpr double kFitTolPx = 1e-6;

std::vector<std::string> split(const std::string& line, char delim) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, delim)) {
        out.push_back(field);
    }
    return out;
}

struct Case {
    int index = 0;
    int height = 0;
    int width = 0;
    double sigma = 0.0;
    double read_var = 0.0;
    int half_width = 0;
    double prior_x = 0.0;
    double prior_y = 0.0;
    double py_cx = 0.0;
    double py_cy = 0.0;
    double py_gx = 0.0;
    double py_gy = 0.0;
    std::vector<double> pixels;
};

bool report(const char* label, int case_index, const char* axis, double got, double want, double tol,
            int& failures) {
    const double diff = std::fabs(got - want);
    if (!(diff <= tol)) {
        std::printf("  FAIL case %2d  %-14s %s: cpp=%.12f  python=%.12f  diff=%.3e  tol=%.1e\n",
                    case_index, label, axis, got, want, diff, tol);
        ++failures;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "results/cpp_vectors.csv";
    std::ifstream file(path);
    if (!file) {
        std::cerr << "cannot open vector file: " << path << "\n"
                  << "generate it first with:  python -m tools.export_cpp_vectors\n";
        return 1;
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "vector file is empty: " << path << "\n";
        return 1;
    }

    std::vector<Case> cases;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> f = split(line, ',');
        if (f.size() < 19) {
            std::cerr << "malformed row, expected 19 fields, got " << f.size() << "\n";
            return 1;
        }
        Case c;
        c.index = std::atoi(f[0].c_str());
        c.height = std::atoi(f[1].c_str());
        c.width = std::atoi(f[2].c_str());
        c.sigma = std::atof(f[3].c_str());
        c.read_var = std::atof(f[4].c_str());
        c.half_width = std::atoi(f[5].c_str());
        c.prior_x = std::atof(f[6].c_str());
        c.prior_y = std::atof(f[7].c_str());
        c.py_cx = std::atof(f[10].c_str());
        c.py_cy = std::atof(f[11].c_str());
        c.py_gx = std::atof(f[14].c_str());
        c.py_gy = std::atof(f[15].c_str());

        std::istringstream pix(f[18]);
        double v = 0.0;
        while (pix >> v) {
            c.pixels.push_back(v);
        }
        const std::size_t expected = static_cast<std::size_t>(c.height) * c.width;
        if (c.pixels.size() != expected) {
            std::cerr << "case " << c.index << ": expected " << expected << " pixels, got "
                      << c.pixels.size() << "\n";
            return 1;
        }
        cases.push_back(std::move(c));
    }

    if (cases.empty()) {
        std::cerr << "no cases found in " << path << "\n";
        return 1;
    }

    std::printf("cross-validating %zu cases against the Python reference\n", cases.size());
    std::printf("  centroid tolerance %.0e px, Gaussian fit tolerance %.0e px\n\n",
                kCentroidTolPx, kFitTolPx);

    int failures = 0;
    double worst_centroid = 0.0;
    double worst_fit = 0.0;

    for (const Case& c : cases) {
        sptrack::FrameView frame{c.pixels.data(), c.height, c.width};

        const sptrack::Estimate cen =
            sptrack::centroid_estimate(frame, c.prior_x, c.prior_y, c.half_width);
        if (!cen.ok) {
            std::printf("  FAIL case %2d  centroid returned ok=false\n", c.index);
            ++failures;
            continue;
        }
        report("centroid", c.index, "x", cen.x, c.py_cx, kCentroidTolPx, failures);
        report("centroid", c.index, "y", cen.y, c.py_cy, kCentroidTolPx, failures);
        worst_centroid = std::max(worst_centroid, std::fabs(cen.x - c.py_cx));
        worst_centroid = std::max(worst_centroid, std::fabs(cen.y - c.py_cy));

        const sptrack::Estimate fit = sptrack::gaussian_fit_estimate(
            frame, c.prior_x, c.prior_y, c.half_width, c.sigma, c.read_var);
        if (!fit.ok) {
            std::printf("  FAIL case %2d  gaussian fit returned ok=false\n", c.index);
            ++failures;
            continue;
        }
        report("gaussian fit", c.index, "x", fit.x, c.py_gx, kFitTolPx, failures);
        report("gaussian fit", c.index, "y", fit.y, c.py_gy, kFitTolPx, failures);
        worst_fit = std::max(worst_fit, std::fabs(fit.x - c.py_gx));
        worst_fit = std::max(worst_fit, std::fabs(fit.y - c.py_gy));
    }

    std::printf("\nworst disagreement: centroid %.3e px, gaussian fit %.3e px\n",
                worst_centroid, worst_fit);
    if (failures == 0) {
        std::printf("PASS: all %zu cases agree with the Python reference\n", cases.size());
        return 0;
    }
    std::printf("FAIL: %d comparison(s) outside tolerance\n", failures);
    return 1;
}
