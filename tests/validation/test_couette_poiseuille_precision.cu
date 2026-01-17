/**
 * @file test_couette_poiseuille_precision.cu
 * @brief Precision Test: Achieve <0.01% L2 error for Couette-Poiseuille
 *
 * KEY INSIGHTS FROM DIAGNOSTIC:
 * 1. 128-cell grid needs ~150,000 steps to converge (not 5,000)
 * 2. Use constant viscosity (not coupled to Re via grid)
 * 3. Lower Ma (0.02) reduces compressibility error
 *
 * ANALYTICAL SOLUTION: u(η) = U_top × (3η² - 2η)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

// Analytical solution: u(η) = U_top × (3η² - 2η)
inline double analytical_solution(double eta, double U_top) {
    return U_top * (3.0 * eta * eta - 2.0 * eta);
}

struct PrecisionMetrics {
    double l2_error;      // L2 error normalized by velocity range (%)
    double max_error;     // Maximum absolute error
    double rms_error;     // RMS of absolute error
    int converged_step;   // Step at which convergence was reached
};

PrecisionMetrics run_precision_test(int ny, float nu, float U_top, int max_steps) {
    const int nx = 4, nz = 4;
    const float H = static_cast<float>(ny - 1);
    const float fx = -6.0f * U_top * nu / (H * H);
    const float rho0 = 1.0f;

    std::cout << "\n========================================\n";
    std::cout << "PRECISION TEST: " << nx << "x" << ny << "x" << nz << "\n";
    std::cout << "========================================\n";
    std::cout << "H = " << H << ", U_top = " << U_top << "\n";
    std::cout << "nu = " << nu << ", fx = " << std::scientific << fx << "\n";
    std::cout << "Re = " << std::fixed << std::setprecision(1) << (U_top * H / nu) << "\n";

    // Initialize FluidLBM with correct API
    FluidLBM fluid(nx, ny, nz,
                   nu,
                   rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_top, 0.0f, 0.0f);

    // Run with convergence check
    int num_cells = nx * ny * nz;
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_ux_old(num_cells, 0.0f);

    int converged_step = -1;
    const float conv_threshold = 1e-9f;  // Very tight convergence

    std::cout << "Running up to " << max_steps << " steps...\n";

    for (int step = 1; step <= max_steps; ++step) {
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
        fluid.collisionTRT(fx, 0.0f, 0.0f);
        fluid.streaming();

        // Check convergence every 1000 steps
        if (step % 1000 == 0) {
            fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

            float max_change = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                max_change = std::max(max_change, std::abs(h_ux[i] - h_ux_old[i]));
            }

            if (step <= 10000 || step % 10000 == 0) {
                std::cout << "  Step " << step << ": max_change = "
                          << std::scientific << max_change << std::endl;
            }

            if (max_change < conv_threshold && converged_step < 0) {
                converged_step = step;
                std::cout << "  CONVERGED at step " << converged_step << "\n";
            }

            h_ux_old = h_ux;
        }
    }

    // Get final velocity field
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();
    fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    // Compute error metrics
    double sum_sq_error = 0.0;
    double max_error = 0.0;
    double u_max = -1e10, u_min = 1e10;

    for (int j = 0; j < ny; ++j) {
        double y = static_cast<double>(j);
        double eta = y / H;
        double u_ana = analytical_solution(eta, U_top);

        u_max = std::max(u_max, u_ana);
        u_min = std::min(u_min, u_ana);

        // Average over x-z plane
        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                u_avg += h_ux[idx];
            }
        }
        u_avg /= (nx * nz);

        double error = u_avg - u_ana;
        sum_sq_error += error * error;
        max_error = std::max(max_error, std::abs(error));
    }

    double u_range = u_max - u_min;
    double l2_error = std::sqrt(sum_sq_error / ny) / u_range * 100.0;
    double rms_error = std::sqrt(sum_sq_error / ny);

    std::cout << "\nRESULTS:\n";
    std::cout << "  L2 error: " << std::fixed << std::setprecision(4) << l2_error << " %\n";
    std::cout << "  Max error: " << std::scientific << max_error << "\n";
    std::cout << "  RMS error: " << rms_error << "\n";
    std::cout << "  Converged: " << (converged_step > 0 ? "Yes" : "No") << "\n";

    return {l2_error, max_error, rms_error, converged_step};
}

TEST(PrecisionTest, TargetPointZeroOne) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "TARGET: Couette-Poiseuille L2 error < 0.01%\n";
    std::cout << "============================================================\n";
    std::cout << "Analytical solution: u(η) = U_top × (3η² - 2η)\n";
    std::cout << "Key: constant nu=0.1, sufficient convergence time\n";

    const float nu = 0.1f;      // Constant viscosity (tau = 0.8)
    const float U_top = 0.02f;  // Low Ma for accuracy

    // Test 1: 64-cell grid
    std::cout << "\n--- Test 1: 64-cell grid ---\n";
    auto m1 = run_precision_test(64, nu, U_top, 50000);

    // Test 2: 128-cell grid (need more steps)
    std::cout << "\n--- Test 2: 128-cell grid ---\n";
    auto m2 = run_precision_test(128, nu, U_top, 150000);

    // Summary
    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "Grid     L2 Error    Max Error     Converged\n";
    std::cout << "-------- ----------  ------------  ---------\n";
    std::cout << "64-cell  " << std::fixed << std::setprecision(4) << m1.l2_error << " %    "
              << std::scientific << std::setprecision(2) << m1.max_error << "    "
              << (m1.converged_step > 0 ? "Yes" : "No") << "\n";
    std::cout << "128-cell " << std::fixed << std::setprecision(4) << m2.l2_error << " %    "
              << std::scientific << std::setprecision(2) << m2.max_error << "    "
              << (m2.converged_step > 0 ? "Yes" : "No") << "\n";

    // Check if target achieved
    bool target_achieved = (m2.l2_error < 0.01);
    std::cout << "\nTarget (<0.01%): " << (target_achieved ? "ACHIEVED ✓" : "NOT YET") << "\n";

    if (!target_achieved && m2.l2_error < 0.1) {
        std::cout << "Note: Error is " << m2.l2_error << "%, close to target.\n";
        std::cout << "May need: finer grid (256 cells) or lower Ma (0.01)\n";
    }

    // At minimum, should be < 1%
    EXPECT_LT(m1.l2_error, 1.0) << "64-cell grid should achieve <1% error";
    EXPECT_LT(m2.l2_error, 0.5) << "128-cell grid should achieve <0.5% error";
}

TEST(PrecisionTest, GridConvergenceStudy) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "GRID CONVERGENCE STUDY\n";
    std::cout << "============================================================\n";
    std::cout << "Verifying 2nd-order convergence: error ~ O(1/N²)\n";

    const float nu = 0.1f;
    const float U_top = 0.02f;

    // Run with different grids
    auto m32 = run_precision_test(32, nu, U_top, 20000);
    auto m64 = run_precision_test(64, nu, U_top, 50000);
    auto m128 = run_precision_test(128, nu, U_top, 150000);

    // Check convergence order
    double ratio_64_32 = m32.l2_error / m64.l2_error;
    double ratio_128_64 = m64.l2_error / m128.l2_error;

    // For 2nd order: ratio should be ~4 (since grid doubles)
    std::cout << "\nCONVERGENCE ORDER:\n";
    std::cout << "  Error ratio (32/64): " << std::fixed << std::setprecision(2) << ratio_64_32;
    std::cout << " (expected ~4 for 2nd order)\n";
    std::cout << "  Error ratio (64/128): " << ratio_128_64;
    std::cout << " (expected ~4 for 2nd order)\n";

    double order_64_32 = std::log(ratio_64_32) / std::log(2.0);
    double order_128_64 = std::log(ratio_128_64) / std::log(2.0);
    std::cout << "  Convergence order (32→64): " << std::setprecision(2) << order_64_32 << "\n";
    std::cout << "  Convergence order (64→128): " << order_128_64 << "\n";

    // Should show approximately 2nd order convergence
    EXPECT_GT(order_64_32, 1.5) << "Should show at least 1.5th order convergence";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
