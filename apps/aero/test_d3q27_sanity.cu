// Standalone sanity check for D3Q27 lattice tables and equilibrium.
//
// Verifies:
//   1. Weights sum to 1
//   2. opposite[opposite[q]] == q
//   3. c_s² = 1/3 from second moment Σ w_q ex_q²
//   4. Equilibrium zeroth moment: Σ f_eq = ρ
//   5. Equilibrium first moment: Σ c_q f_eq = ρ·u (Galilean covariance check)
//
// Build is part of CMake target test_d3q27_sanity. Run: ./test_d3q27_sanity

#include "core/lattice_d3q27.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace lbm::core;

static int passed = 0, failed = 0;

#define EXPECT(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        ++failed; \
    } else { \
        printf("PASS: " msg "\n", ##__VA_ARGS__); \
        ++passed; \
    } \
} while(0)

int main() {
    printf("=== D3Q27 sanity tests ===\n");

    // 1. Weights sum to 1
    double w_sum = 0.0;
    for (int q = 0; q < D3Q27::Q; ++q) w_sum += D3Q27::h_w_double[q];
    EXPECT(std::abs(w_sum - 1.0) < 1e-10, "Σ w_q = %.12f (expected 1.0)", w_sum);

    // 2. opposite consistency
    bool opp_ok = true;
    for (int q = 0; q < D3Q27::Q; ++q) {
        int oo = D3Q27::h_opposite[D3Q27::h_opposite[q]];
        if (oo != q) { opp_ok = false; printf("  opposite[opposite[%d]] = %d != %d\n", q, oo, q); }
        // also: c_q + c_opp = 0
        int co_x = D3Q27::h_ex[q] + D3Q27::h_ex[D3Q27::h_opposite[q]];
        int co_y = D3Q27::h_ey[q] + D3Q27::h_ey[D3Q27::h_opposite[q]];
        int co_z = D3Q27::h_ez[q] + D3Q27::h_ez[D3Q27::h_opposite[q]];
        if (co_x || co_y || co_z) {
            opp_ok = false;
            printf("  c_%d + c_opp[%d] = (%d, %d, %d) != (0, 0, 0)\n", q, q, co_x, co_y, co_z);
        }
    }
    EXPECT(opp_ok, "opposite[opposite[q]] = q AND c_q + c_opp = 0 for all q");

    // 3. Speed of sound from 2nd moment
    double cs2_xx = 0.0, cs2_yy = 0.0, cs2_zz = 0.0;
    double cs2_xy = 0.0, cs2_yz = 0.0, cs2_xz = 0.0;
    for (int q = 0; q < D3Q27::Q; ++q) {
        const double w = D3Q27::h_w_double[q];
        const int cx = D3Q27::h_ex[q], cy = D3Q27::h_ey[q], cz = D3Q27::h_ez[q];
        cs2_xx += w * cx * cx;
        cs2_yy += w * cy * cy;
        cs2_zz += w * cz * cz;
        cs2_xy += w * cx * cy;
        cs2_yz += w * cy * cz;
        cs2_xz += w * cx * cz;
    }
    EXPECT(std::abs(cs2_xx - 1.0/3.0) < 1e-12, "Σ w_q ex_q² = %.12f (expected 1/3)", cs2_xx);
    EXPECT(std::abs(cs2_yy - 1.0/3.0) < 1e-12, "Σ w_q ey_q² = %.12f (expected 1/3)", cs2_yy);
    EXPECT(std::abs(cs2_zz - 1.0/3.0) < 1e-12, "Σ w_q ez_q² = %.12f (expected 1/3)", cs2_zz);
    EXPECT(std::abs(cs2_xy) < 1e-12, "Σ w_q ex_q ey_q = %.12f (expected 0)", cs2_xy);
    EXPECT(std::abs(cs2_yz) < 1e-12, "Σ w_q ey_q ez_q = %.12f (expected 0)", cs2_yz);
    EXPECT(std::abs(cs2_xz) < 1e-12, "Σ w_q ex_q ez_q = %.12f (expected 0)", cs2_xz);

    // 4. Equilibrium zeroth moment (with non-trivial u)
    const float rho = 1.234f;
    const float ux = 0.1f, uy = 0.05f, uz = -0.07f;
    float feq[27];
    for (int q = 0; q < D3Q27::Q; ++q) {
        feq[q] = D3Q27::computeEquilibrium(q, rho, ux, uy, uz);
    }
    float feq_sum = 0.0f;
    for (int q = 0; q < D3Q27::Q; ++q) feq_sum += feq[q];
    EXPECT(std::abs(feq_sum - rho) < 1e-4, "Σ f_eq = %.6f (expected %.6f)", feq_sum, rho);

    // 5. Equilibrium first moment
    float mx = 0, my = 0, mz = 0;
    for (int q = 0; q < D3Q27::Q; ++q) {
        mx += D3Q27::h_ex[q] * feq[q];
        my += D3Q27::h_ey[q] * feq[q];
        mz += D3Q27::h_ez[q] * feq[q];
    }
    EXPECT(std::abs(mx - rho*ux) < 1e-4,
           "Σ ex_q f_eq = %.6f (expected ρ·ux = %.6f)", mx, rho*ux);
    EXPECT(std::abs(my - rho*uy) < 1e-4,
           "Σ ey_q f_eq = %.6f (expected ρ·uy = %.6f)", my, rho*uy);
    EXPECT(std::abs(mz - rho*uz) < 1e-4,
           "Σ ez_q f_eq = %.6f (expected ρ·uz = %.6f)", mz, rho*uz);

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
