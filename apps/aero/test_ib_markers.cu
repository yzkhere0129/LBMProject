/**
 * @file test_ib_markers.cu
 * @brief Phase A unit test: IB marker generation correctness.
 *
 * Tests:
 *   T1. NACA α=0 mirror symmetry: every upper marker (x, +y) has a partner
 *       at (x, -y) (within machine epsilon). Tests both pre-rotation and
 *       post-rotation invariants.
 *   T2. Circle marker count matches 2πR/ds_target ± 1, ds_actual ≈ target.
 *   T3. NACA α=+8° generation: no NaN, total marker count reasonable, dump
 *       to CSV for plotting.
 *   T4. extrudeZ replicates 2D markers across nz layers correctly.
 *
 * No GPU work — host-only test (header-only marker generator).
 */

#include "physics/aero/ib_markers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using lbm::physics::aero::IBMarker;
using lbm::physics::aero::buildNacaMarkers2D;
using lbm::physics::aero::buildCircleMarkers2D;
using lbm::physics::aero::extrudeZ;

namespace {

void dump_csv(const std::vector<IBMarker>& m, const std::string& path) {
    std::ofstream f(path);
    f << "idx,x,y,z,ds,nx,ny,nz\n";
    f.precision(8);
    for (size_t i = 0; i < m.size(); ++i) {
        const auto& a = m[i];
        f << i << "," << a.x << "," << a.y << "," << a.z << "," << a.ds
          << "," << a.nx << "," << a.ny << "," << a.nz << "\n";
    }
    std::cout << "  wrote " << m.size() << " markers → " << path << "\n";
}

bool any_nan(const std::vector<IBMarker>& m) {
    for (const auto& a : m) {
        if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.ds) ||
            !std::isfinite(a.nx) || !std::isfinite(a.ny)) return true;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const std::string out_dir = (argc > 1) ? argv[1] : "output_ib_markers_test";
    std::system(("mkdir -p " + out_dir).c_str());

    int passed = 0, failed = 0;

    // ---------- T1. NACA α=0 mirror symmetry ----------
    {
        std::cout << "\n=== T1. NACA α=0 mirror symmetry ===\n";
        const float chord = 1.0f, dx = 1.0f / 40.0f;
        auto m = buildNacaMarkers2D(chord, 12.0f, /*alpha*/ 0.0f,
                                    /*xLE*/ 0.0f, /*yLE*/ 0.0f, /*ds*/ dx);

        // Find the pair structure: ordering is [LE, upper s↑, TE-interior, lower s↓].
        // For α=0 with yLE=0, every marker at y > 0 should have a mirror at y < 0
        // with the same x. Bin by x and check pairs.
        const float eps_pair = 1e-6f;  // pair-x search tolerance
        const float eps_sym = 1e-12f;  // symmetry tolerance for |y_pair_diff|

        size_t n_paired = 0;
        float max_y_diff = 0.0f, max_x_diff = 0.0f;
        size_t n_zero_y = 0;  // markers ON chord (LE, TE)

        std::vector<bool> matched(m.size(), false);
        for (size_t i = 0; i < m.size(); ++i) {
            if (std::abs(m[i].y) < eps_pair) { ++n_zero_y; matched[i] = true; continue; }
            if (matched[i]) continue;
            float best_d = 1e30f; size_t best_j = 0;
            for (size_t j = 0; j < m.size(); ++j) {
                if (j == i || matched[j]) continue;
                if (std::abs(m[j].y + m[i].y) > eps_pair * 10) continue;
                const float dxij = std::abs(m[j].x - m[i].x);
                if (dxij < best_d) { best_d = dxij; best_j = j; }
            }
            if (best_d < eps_pair) {
                matched[i] = matched[best_j] = true;
                n_paired++;
                max_x_diff = std::max(max_x_diff, std::abs(m[i].x - m[best_j].x));
                max_y_diff = std::max(max_y_diff, std::abs(m[i].y + m[best_j].y));
                // Normal symmetry: nx same, ny opposite
                const float nx_diff = std::abs(m[i].nx - m[best_j].nx);
                const float ny_diff = std::abs(m[i].ny + m[best_j].ny);
                if (nx_diff > 1e-5f || ny_diff > 1e-5f) {
                    std::cout << "  warn: pair " << i << "↔" << best_j
                              << " normal asym nx_diff=" << nx_diff
                              << " ny_diff=" << ny_diff << "\n";
                }
            }
        }
        size_t n_unmatched = 0;
        for (auto b : matched) if (!b) ++n_unmatched;

        std::cout << "  total markers: " << m.size() << "\n";
        std::cout << "  on-chord (y≈0): " << n_zero_y << "\n";
        std::cout << "  paired (mirror): " << n_paired << "\n";
        std::cout << "  unmatched: " << n_unmatched << "\n";
        std::cout << "  max |x_pair_diff| = " << max_x_diff << "\n";
        std::cout << "  max |y_pair_diff| = " << max_y_diff
                  << "  (target < " << eps_sym << ")\n";

        const bool t1 = (n_unmatched == 0) && (max_y_diff < eps_sym);
        std::cout << "  → " << (t1 ? "PASS" : "FAIL") << "\n";
        if (t1) ++passed; else ++failed;

        dump_csv(m, out_dir + "/naca_a0_markers.csv");
    }

    // ---------- T2. Circle marker count + spacing ----------
    {
        std::cout << "\n=== T2. Circle markers (R=0.05 m, ds=0.0025 m) ===\n";
        const float R = 0.05f, ds = 0.0025f;
        auto m = buildCircleMarkers2D(0.0f, 0.0f, R, ds);
        const float L = 2.0f * float(M_PI) * R;
        const int n_expected = int(std::round(L / ds));
        const float ds_actual = L / float(m.size());

        std::cout << "  N actual = " << m.size() << "  (expected ≈ " << n_expected << ")\n";
        std::cout << "  ds actual = " << ds_actual << "  (target = " << ds << ")\n";

        // Check radius preservation: every marker should be at distance R from origin.
        float max_r_err = 0.0f;
        for (const auto& a : m) {
            const float r = std::sqrt(a.x * a.x + a.y * a.y);
            max_r_err = std::max(max_r_err, std::abs(r - R));
        }
        std::cout << "  max |r - R| = " << max_r_err << "\n";

        // Outward normal points radially: (nx, ny) ≈ (x/R, y/R)
        float max_n_err = 0.0f;
        for (const auto& a : m) {
            const float ex = a.x / R, ey = a.y / R;
            const float dn = std::hypot(a.nx - ex, a.ny - ey);
            max_n_err = std::max(max_n_err, dn);
        }
        std::cout << "  max |normal - radial| = " << max_n_err << "\n";

        const bool t2 = std::abs(int(m.size()) - n_expected) <= 1
                     && max_r_err < 1e-6f
                     && max_n_err < 1e-6f;
        std::cout << "  → " << (t2 ? "PASS" : "FAIL") << "\n";
        if (t2) ++passed; else ++failed;

        dump_csv(m, out_dir + "/circle_markers.csv");
    }

    // ---------- T3. NACA α=+8° generation ----------
    {
        std::cout << "\n=== T3. NACA α=+8° (Re=1000 D/c=40 config) ===\n";
        const float chord = 1.0f, dx = 1.0f / 40.0f;
        const float alpha = 8.0f * float(M_PI) / 180.0f;
        const float xLE = 10.0f, yLE = 10.0f;
        auto m = buildNacaMarkers2D(chord, 12.0f, alpha, xLE, yLE, dx);

        std::cout << "  total markers: " << m.size() << "\n";
        std::cout << "  any NaN?       " << (any_nan(m) ? "YES (FAIL)" : "no") << "\n";

        // Confirm rotated geometry: TE world-position should be (xLE + cos(α), yLE + sin(α)).
        const float te_x_expected = xLE + std::cos(alpha) * chord;
        const float te_y_expected = yLE + std::sin(alpha) * chord;
        // Find the marker closest to expected TE position.
        float min_d = 1e30f; size_t i_te = 0;
        for (size_t i = 0; i < m.size(); ++i) {
            const float d = std::hypot(m[i].x - te_x_expected, m[i].y - te_y_expected);
            if (d < min_d) { min_d = d; i_te = i; }
        }
        std::cout << "  expected TE   (" << te_x_expected << ", " << te_y_expected << ")\n";
        std::cout << "  nearest marker (" << m[i_te].x << ", " << m[i_te].y
                  << ")  Δ=" << min_d << "\n";

        const bool t3 = !any_nan(m) && (m.size() > 50) && (min_d < dx);
        std::cout << "  → " << (t3 ? "PASS" : "FAIL") << "\n";
        if (t3) ++passed; else ++failed;

        dump_csv(m, out_dir + "/naca_a8_markers.csv");
    }

    // ---------- T4. extrudeZ ----------
    {
        std::cout << "\n=== T4. extrudeZ replicates per-layer ===\n";
        const float chord = 1.0f, dx = 0.025f;
        auto m2d = buildNacaMarkers2D(chord, 12.0f, 0.0f, 0.0f, 0.0f, dx);
        const int nz = 8;
        auto m3d = extrudeZ(m2d, nz, dx);

        bool t4 = (m3d.size() == m2d.size() * size_t(nz));
        for (int k = 0; k < nz && t4; ++k) {
            for (size_t i = 0; i < m2d.size() && t4; ++i) {
                const auto& a = m3d[k * m2d.size() + i];
                const auto& b = m2d[i];
                if (a.x != b.x || a.y != b.y) t4 = false;
                if (std::abs(a.z - (k + 0.5f) * dx) > 1e-7f) t4 = false;
                if (a.ds != b.ds || a.nx != b.nx || a.ny != b.ny) t4 = false;
            }
        }
        std::cout << "  m2d size = " << m2d.size()
                  << ", m3d size = " << m3d.size()
                  << " (expected " << m2d.size() * size_t(nz) << ")\n";
        std::cout << "  → " << (t4 ? "PASS" : "FAIL") << "\n";
        if (t4) ++passed; else ++failed;
    }

    std::cout << "\n=========================================\n"
              << "Phase A test summary: " << passed << " PASS, " << failed << " FAIL\n";

    return (failed == 0) ? 0 : 1;
}
