/**
 * @file test_couette_poiseuille_diagnostic.cu
 * @brief Diagnostic Test Suite: Identify Error Sources in Couette-Poiseuille
 *
 * PURPOSE: Find what optimization reduces error from 1.977% to < 1%
 *
 * APPROACH: Concrete testing, not theory. Run actual tests:
 *   1. Double precision computation
 *   2. Grid refinement (64, 128, 256 cells)
 *   3. Lower Mach number (U_top = 0.01 vs 0.05)
 *   4. BGK vs TRT comparison
 *   5. Error distribution analysis
 *
 * BASELINE (from recent run):
 *   Grid: 10x30x10, U_top=0.05, TRT, float
 *   L2 error: 1.16% (improved from 1.977%)
 *
 * TARGET: L2 error < 1.0%
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

// ============================================================================
// ANALYTICAL SOLUTION
// ============================================================================

inline float analytical_couette_poiseuille(float y, float U_top, float fx, float nu, float H) {
    float eta = y / H;
    float u_couette = U_top * eta;
    float u_poiseuille = (fx * H * H) / (2.0f * nu) * eta * (1.0f - eta);
    return u_couette + u_poiseuille;
}

// ============================================================================
// ERROR COMPUTATION
// ============================================================================

struct ErrorMetrics {
    float l2_error;
    float max_error;
    float mean_error;
    float std_error;

    // Error at specific locations
    float error_at_quarter;  // y = H/4
    float error_at_half;      // y = H/2
    float error_at_three_quarter;  // y = 3H/4

    // Wall errors
    float bottom_wall_error;
    float top_wall_error;
};

ErrorMetrics compute_error_metrics(
    const std::vector<float>& u_numerical,
    int nx, int ny, int nz,
    float U_top, float fx, float nu, float H)
{
    ErrorMetrics metrics = {};

    double sum_squared_error = 0.0;
    double sum_error = 0.0;
    float u_min_ana = 0.0f;
    float u_max_ana = 0.0f;
    int count = 0;

    std::vector<float> errors;

    for (int j = 0; j < ny; ++j) {
        float y = static_cast<float>(j);
        float u_analytical = analytical_couette_poiseuille(y, U_top, fx, nu, H);

        u_min_ana = std::min(u_min_ana, u_analytical);
        u_max_ana = std::max(u_max_ana, u_analytical);

        // Average over x-z plane
        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                u_avg += u_numerical[idx];
            }
        }
        u_avg /= (nx * nz);

        float error = u_avg - u_analytical;
        errors.push_back(error);
        sum_squared_error += error * error;
        sum_error += error;
        metrics.max_error = std::max(metrics.max_error, std::abs(error));
        count++;

        // Specific location errors
        if (j == ny / 4) {
            metrics.error_at_quarter = error;
        } else if (j == ny / 2) {
            metrics.error_at_half = error;
        } else if (j == 3 * ny / 4) {
            metrics.error_at_three_quarter = error;
        } else if (j == 0) {
            metrics.bottom_wall_error = error;
        } else if (j == ny - 1) {
            metrics.top_wall_error = error;
        }
    }

    // L2 error normalized by range
    float u_range = u_max_ana - u_min_ana;
    metrics.l2_error = std::sqrt(sum_squared_error / count) / u_range;

    // Mean error
    metrics.mean_error = sum_error / count;

    // Standard deviation
    double sum_sq_dev = 0.0;
    for (float err : errors) {
        double dev = err - metrics.mean_error;
        sum_sq_dev += dev * dev;
    }
    metrics.std_error = std::sqrt(sum_sq_dev / count);

    return metrics;
}

// ============================================================================
// RUN SINGLE TEST
// ============================================================================

ErrorMetrics run_single_test(
    int nx, int ny, int nz,
    float U_top,
    float nu,
    bool use_trt,
    int max_steps,
    const std::string& test_name)
{
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << test_name << "\n";
    std::cout << "========================================\n";
    std::cout << "Grid: " << nx << " x " << ny << " x " << nz << "\n";
    std::cout << "U_top: " << U_top << ", nu: " << nu << "\n";
    std::cout << "Collision: " << (use_trt ? "TRT" : "BGK") << "\n";

    // Setup
    float H = static_cast<float>(ny - 1);
    float fx = -6.0f * U_top * nu / (H * H);
    float rho0 = 1.0f;

    // Initialize solver
    FluidLBM fluid(nx, ny, nz,
                   nu,
                   rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_top, 0.0f, 0.0f);

    // Run simulation
    int num_cells = nx * ny * nz;
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_ux_old(num_cells);

    std::cout << "Running " << max_steps << " steps...\n";

    bool converged = false;
    int conv_step = -1;

    for (int step = 0; step <= max_steps; ++step) {
        if (step > 0) {
            fluid.applyBoundaryConditions(1);
            fluid.computeMacroscopic();

            if (use_trt) {
                fluid.collisionTRT(fx, 0.0f, 0.0f);
            } else {
                fluid.collisionBGK(fx, 0.0f, 0.0f);
            }

            fluid.streaming();
        }

        // Check convergence every 100 steps
        if (step % 100 == 0 && step > 0) {
            fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

            float max_change = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                max_change = std::max(max_change, std::abs(h_ux[i] - h_ux_old[i]));
            }

            if (max_change < 1e-6f && !converged) {
                converged = true;
                conv_step = step;
                std::cout << "Converged at step " << step << "\n";
            }

            h_ux_old = h_ux;
        }
    }

    // Final results
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();
    fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    ErrorMetrics metrics = compute_error_metrics(h_ux, nx, ny, nz, U_top, fx, nu, H);

    std::cout << "\nRESULTS:\n";
    std::cout << "  L2 error: " << std::setprecision(3) << metrics.l2_error * 100.0f << " %\n";
    std::cout << "  Max error: " << std::scientific << metrics.max_error << "\n";
    std::cout << "  Mean error: " << metrics.mean_error << "\n";
    std::cout << "  Std error: " << metrics.std_error << "\n";
    std::cout << "  Converged: " << (converged ? "Yes" : "No");
    if (converged) std::cout << " (step " << conv_step << ")";
    std::cout << "\n";

    return metrics;
}

// ============================================================================
// DIAGNOSTIC TESTS
// ============================================================================

TEST(DiagnosticTests, GridRefinementEffect) {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "DIAGNOSTIC 1: GRID REFINEMENT\n";
    std::cout << "========================================\n";
    std::cout << "Question: Does finer grid reduce error?\n\n";

    float U_top = 0.05f;
    float Re = 10.0f;
    bool use_trt = true;
    int max_steps = 5000;

    // Test 1: Baseline (current)
    int ny_base = 30;
    float H_base = static_cast<float>(ny_base - 1);
    float nu_base = U_top * H_base / Re;
    ErrorMetrics m1 = run_single_test(10, ny_base, 10, U_top, nu_base, use_trt, max_steps,
                                      "Baseline: 10x30x10");

    // Test 2: 2x refinement
    int ny_fine1 = 64;
    float H_fine1 = static_cast<float>(ny_fine1 - 1);
    float nu_fine1 = U_top * H_fine1 / Re;
    ErrorMetrics m2 = run_single_test(4, ny_fine1, 4, U_top, nu_fine1, use_trt, max_steps,
                                      "Fine: 4x64x4 (2x refinement)");

    // Test 3: 4x refinement
    int ny_fine2 = 128;
    float H_fine2 = static_cast<float>(ny_fine2 - 1);
    float nu_fine2 = U_top * H_fine2 / Re;
    ErrorMetrics m3 = run_single_test(4, ny_fine2, 4, U_top, nu_fine2, use_trt, max_steps,
                                      "Very Fine: 4x128x4 (4x refinement)");

    std::cout << "\n========================================\n";
    std::cout << "GRID REFINEMENT SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Baseline (30):  L2 = " << std::setprecision(3) << m1.l2_error * 100.0f << " %\n";
    std::cout << "Fine (64):      L2 = " << m2.l2_error * 100.0f << " %\n";
    std::cout << "Very Fine (128): L2 = " << m3.l2_error * 100.0f << " %\n";

    if (m2.l2_error < 0.01f) {
        std::cout << "\nCONCLUSION: 64-cell grid achieves < 1% error!\n";
    } else if (m3.l2_error < 0.01f) {
        std::cout << "\nCONCLUSION: 128-cell grid achieves < 1% error!\n";
    } else {
        std::cout << "\nCONCLUSION: Grid refinement alone insufficient.\n";
    }
}

TEST(DiagnosticTests, MachNumberEffect) {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "DIAGNOSTIC 2: MACH NUMBER\n";
    std::cout << "========================================\n";
    std::cout << "Question: Does lower Ma reduce error?\n";
    std::cout << "Ma ~ U / cs, where cs = 1/sqrt(3) ~ 0.577\n\n";

    int nx = 10, ny = 30, nz = 10;
    float H = static_cast<float>(ny - 1);
    float Re = 10.0f;
    bool use_trt = true;
    int max_steps = 5000;

    // Test 1: High Ma (baseline)
    float U_high = 0.05f;
    float Ma_high = U_high / 0.577f;
    float nu_high = U_high * H / Re;
    ErrorMetrics m1 = run_single_test(nx, ny, nz, U_high, nu_high, use_trt, max_steps,
                                      "High Ma: U=0.05 (Ma~0.087)");

    // Test 2: Low Ma
    float U_low = 0.01f;
    float Ma_low = U_low / 0.577f;
    float nu_low = U_low * H / Re;
    ErrorMetrics m2 = run_single_test(nx, ny, nz, U_low, nu_low, use_trt, max_steps,
                                      "Low Ma: U=0.01 (Ma~0.017)");

    std::cout << "\n========================================\n";
    std::cout << "MACH NUMBER SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "High Ma (0.087): L2 = " << std::setprecision(3) << m1.l2_error * 100.0f << " %\n";
    std::cout << "Low Ma (0.017):  L2 = " << m2.l2_error * 100.0f << " %\n";

    float reduction = (m1.l2_error - m2.l2_error) / m1.l2_error * 100.0f;
    std::cout << "Error reduction: " << std::setprecision(1) << reduction << " %\n";

    if (m2.l2_error < 0.01f) {
        std::cout << "\nCONCLUSION: Lower Ma achieves < 1% error!\n";
    } else {
        std::cout << "\nCONCLUSION: Ma effect present but insufficient alone.\n";
    }
}

TEST(DiagnosticTests, BGKvsTRT) {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "DIAGNOSTIC 3: BGK vs TRT\n";
    std::cout << "========================================\n";
    std::cout << "Question: Is TRT better than BGK?\n\n";

    int nx = 10, ny = 30, nz = 10;
    float H = static_cast<float>(ny - 1);
    float U_top = 0.05f;
    float Re = 10.0f;
    float nu = U_top * H / Re;
    int max_steps = 5000;

    ErrorMetrics m_bgk = run_single_test(nx, ny, nz, U_top, nu, false, max_steps, "BGK Collision");
    ErrorMetrics m_trt = run_single_test(nx, ny, nz, U_top, nu, true, max_steps, "TRT Collision");

    std::cout << "\n========================================\n";
    std::cout << "BGK vs TRT SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "BGK: L2 = " << std::setprecision(3) << m_bgk.l2_error * 100.0f << " %\n";
    std::cout << "TRT: L2 = " << m_trt.l2_error * 100.0f << " %\n";

    float improvement = (m_bgk.l2_error - m_trt.l2_error) / m_bgk.l2_error * 100.0f;
    std::cout << "TRT improvement: " << std::setprecision(1) << improvement << " %\n";

    if (improvement < 0) {
        std::cout << "\nWARNING: TRT performs worse than BGK!\n";
    }
}

TEST(DiagnosticTests, ErrorDistribution) {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "DIAGNOSTIC 4: ERROR DISTRIBUTION\n";
    std::cout << "========================================\n";
    std::cout << "Question: Where is error largest?\n\n";

    int nx = 10, ny = 30, nz = 10;
    float H = static_cast<float>(ny - 1);
    float U_top = 0.05f;
    float Re = 10.0f;
    float nu = U_top * H / Re;
    float fx = -6.0f * U_top * nu / (H * H);
    float rho0 = 1.0f;

    FluidLBM fluid(nx, ny, nz, nu, rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_top, 0.0f, 0.0f);

    // Run to steady state
    for (int step = 0; step <= 5000; ++step) {
        if (step > 0) {
            fluid.applyBoundaryConditions(1);
            fluid.computeMacroscopic();
            fluid.collisionTRT(fx, 0.0f, 0.0f);
            fluid.streaming();
        }
    }

    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();

    int num_cells = nx * ny * nz;
    std::vector<float> h_ux(num_cells);
    fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    // Analyze error at each y-level
    std::cout << "Error distribution across channel:\n";
    std::cout << "y      eta    u_num      u_ana      error      err%\n";
    std::cout << std::string(60, '-') << "\n";

    float max_error_loc = 0.0f;
    int max_error_j = 0;

    for (int j = 0; j < ny; ++j) {
        float y = static_cast<float>(j);
        float eta = y / H;
        float u_analytical = analytical_couette_poiseuille(y, U_top, fx, nu, H);

        // Average over x-z
        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                u_avg += h_ux[idx];
            }
        }
        u_avg /= (nx * nz);

        float error = u_avg - u_analytical;
        float err_pct = (std::abs(u_analytical) > 1e-10f) ?
                        (error / u_analytical * 100.0f) : 0.0f;

        if (std::abs(error) > max_error_loc) {
            max_error_loc = std::abs(error);
            max_error_j = j;
        }

        // Print every 3rd point to avoid clutter
        if (j % 3 == 0 || j == ny - 1) {
            std::cout << std::setw(4) << j << "  "
                      << std::setprecision(3) << std::setw(5) << eta << "  "
                      << std::scientific << std::setprecision(4) << u_avg << "  "
                      << u_analytical << "  "
                      << error << "  "
                      << std::fixed << std::setprecision(2) << err_pct << "\n";
        }
    }

    std::cout << "\nMaximum error at j=" << max_error_j
              << " (eta=" << std::setprecision(3) << static_cast<float>(max_error_j) / H << ")\n";

    ErrorMetrics metrics = compute_error_metrics(h_ux, nx, ny, nz, U_top, fx, nu, H);

    std::cout << "\nError at specific locations:\n";
    std::cout << "  y=H/4:   " << std::scientific << metrics.error_at_quarter << "\n";
    std::cout << "  y=H/2:   " << metrics.error_at_half << "\n";
    std::cout << "  y=3H/4:  " << metrics.error_at_three_quarter << "\n";
    std::cout << "  Bottom wall: " << metrics.bottom_wall_error << "\n";
    std::cout << "  Top wall:    " << metrics.top_wall_error << "\n";
}

TEST(DiagnosticTests, CombinedOptimization) {
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "DIAGNOSTIC 5: COMBINED OPTIMIZATION\n";
    std::cout << "========================================\n";
    std::cout << "Question: Combining best settings\n\n";

    // Best case: Fine grid + Low Ma + TRT
    int ny = 64;
    float H = static_cast<float>(ny - 1);
    float U_top = 0.01f;  // Low Ma
    float Re = 10.0f;
    float nu = U_top * H / Re;
    bool use_trt = true;
    int max_steps = 10000;

    ErrorMetrics m = run_single_test(4, ny, 4, U_top, nu, use_trt, max_steps,
                                     "BEST: 4x64x4 + Low Ma + TRT");

    std::cout << "\n========================================\n";
    std::cout << "FINAL RESULT\n";
    std::cout << "========================================\n";
    std::cout << "L2 error: " << std::setprecision(3) << m.l2_error * 100.0f << " %\n";

    if (m.l2_error < 0.01f) {
        std::cout << "\nSUCCESS: Achieved < 1% error!\n";
        std::cout << "Recommendation: Use fine grid (64+ cells) + low Ma\n";
    } else {
        std::cout << "\nStill > 1% error. May need:\n";
        std::cout << "  - Even finer grid (128+ cells)\n";
        std::cout << "  - Double precision\n";
        std::cout << "  - Different boundary conditions\n";
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
