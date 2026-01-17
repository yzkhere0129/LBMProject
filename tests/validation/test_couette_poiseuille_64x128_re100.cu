/**
 * @file test_couette_poiseuille_64x128_re100.cu
 * @brief Run Couette-Poiseuille with 64x128 grid at Re=100 to match reference
 *
 * Reference specifications:
 * - Grid: 64 (x) × 128 (y) cells
 * - Re = 100
 * - U_top = 1.0 (normalized)
 * - Body force: f_x = -6/Re = -0.06
 *
 * For LBM:
 * - ν = U_top * H / Re = 1.0 * 127 / 100 = 1.27
 * - f_x = -6 * U_top * ν / H² = -6 * 1.0 * 1.27 / 127² ≈ -0.000472
 * - Analytical: u(η) = U_top × (3η² - 2η) where η = y/H
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

// Analytical solution: u(η) = U_top × (3η² - 2η)
inline double analytical_solution(double eta, double U_top) {
    return U_top * (3.0 * eta * eta - 2.0 * eta);
}

void save_velocity_profile(const std::string& filename,
                           int ny, double H, double U_top,
                           const std::vector<double>& u_sim,
                           const std::vector<double>& u_ana) {
    std::ofstream file(filename);
    file << "# Couette-Poiseuille velocity profile\n";
    file << "# Grid: 64x128, Re=100\n";
    file << "# Columns: y_coord, eta, u_simulation, u_analytical, error, error_percent\n";
    file << std::scientific << std::setprecision(8);

    for (int j = 0; j < ny; ++j) {
        double y = static_cast<double>(j);
        double eta = y / H;
        double error = u_sim[j] - u_ana[j];
        double error_pct = (u_ana[j] != 0.0) ? (error / u_ana[j] * 100.0) : 0.0;

        file << std::setw(16) << y
             << std::setw(16) << eta
             << std::setw(16) << u_sim[j]
             << std::setw(16) << u_ana[j]
             << std::setw(16) << error
             << std::setw(16) << error_pct << "\n";
    }
    file.close();
    std::cout << "Saved velocity profile to: " << filename << "\n";
}

TEST(CouettePoiseuille, Grid64x128_Re100) {
    // Problem parameters matching reference
    const int nx = 64, ny = 128, nz = 4;  // 3D grid: 64x128x4
    const float H = static_cast<float>(ny - 1);  // H = 127
    const float U_top = 1.0f;  // Top wall velocity
    const float Re = 100.0f;   // Reynolds number

    // LBM parameters
    const float nu = U_top * H / Re;  // ν = 1.27
    const float fx = -6.0f * U_top * nu / (H * H);  // Body force
    const float rho0 = 1.0f;

    // Mach number (for stability: Ma < 0.1)
    const float cs = 1.0f / std::sqrt(3.0f);  // Lattice speed of sound
    const float Ma = U_top / cs;

    // Relaxation parameter
    const float tau = 0.5f + 3.0f * nu;

    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "COUETTE-POISEUILLE: 64x128 GRID, Re=100\n";
    std::cout << "========================================================================\n";
    std::cout << "Grid dimensions: " << nx << " × " << ny << " × " << nz << "\n";
    std::cout << "Channel height: H = " << H << " lattice units\n";
    std::cout << "Reynolds number: Re = " << Re << "\n";
    std::cout << "Top wall velocity: U_top = " << U_top << "\n";
    std::cout << "Kinematic viscosity: ν = " << std::fixed << std::setprecision(6) << nu << "\n";
    std::cout << "Body force: f_x = " << std::scientific << std::setprecision(6) << fx << "\n";
    std::cout << "Relaxation time: τ = " << std::fixed << std::setprecision(6) << tau << "\n";
    std::cout << "Mach number: Ma = " << std::setprecision(4) << Ma << " (should be < 0.1)\n";
    std::cout << "Analytical solution: u(η) = U_top × (3η² - 2η), η = y/H\n";
    std::cout << "------------------------------------------------------------------------\n";

    // Check Mach number
    if (Ma >= 0.1f) {
        std::cout << "WARNING: Mach number = " << Ma << " >= 0.1\n";
        std::cout << "Consider reducing U_top or using normalized units with U_top = 0.02\n";
    }

    // Initialize FluidLBM
    FluidLBM fluid(nx, ny, nz,
                   nu,
                   rho0,
                   BoundaryType::PERIODIC,  // x: periodic
                   BoundaryType::WALL,      // y: walls (bottom stationary, top moving)
                   BoundaryType::PERIODIC,  // z: periodic
                   1.0f, 1.0f);

    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_top, 0.0f, 0.0f);

    // Run simulation with convergence monitoring
    const int max_steps = 200000;  // 128-cell grid may need 150k-200k steps
    const int check_interval = 1000;
    const float conv_threshold = 1e-9f;

    int num_cells = nx * ny * nz;
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_ux_old(num_cells, 0.0f);

    int converged_step = -1;

    std::cout << "\nRunning simulation (max " << max_steps << " steps)...\n";
    std::cout << "Checking convergence every " << check_interval << " steps\n";

    for (int step = 1; step <= max_steps; ++step) {
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
        fluid.collisionTRT(fx, 0.0f, 0.0f);
        fluid.streaming();

        // Check convergence
        if (step % check_interval == 0) {
            fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

            float max_change = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                max_change = std::max(max_change, std::abs(h_ux[i] - h_ux_old[i]));
            }

            // Print progress
            if (step <= 10000 || step % 10000 == 0) {
                std::cout << "  Step " << std::setw(6) << step << ": max_change = "
                          << std::scientific << std::setprecision(3) << max_change;
                if (converged_step > 0) {
                    std::cout << " [CONVERGED]";
                }
                std::cout << "\n";
            }

            if (max_change < conv_threshold && converged_step < 0) {
                converged_step = step;
                std::cout << "  >>> CONVERGENCE achieved at step " << converged_step << " <<<\n";
            }

            // Exit early if converged and stable
            if (converged_step > 0 && step > converged_step + 10000) {
                std::cout << "  Stopping: converged and stable for 10000 steps\n";
                break;
            }

            h_ux_old = h_ux;
        }
    }

    // Get final velocity field
    std::cout << "\nFinalizing...\n";
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();
    fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    // Compute velocity profile (average over x-z planes)
    std::vector<double> u_profile(ny);
    std::vector<double> u_analytical(ny);

    for (int j = 0; j < ny; ++j) {
        double y = static_cast<double>(j);
        double eta = y / H;
        u_analytical[j] = analytical_solution(eta, U_top);

        // Average over x-z plane
        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                u_avg += h_ux[idx];
            }
        }
        u_profile[j] = u_avg / (nx * nz);
    }

    // Compute error metrics
    double sum_sq_error = 0.0;
    double sum_abs_error = 0.0;
    double max_error = 0.0;
    double max_error_pct = 0.0;
    double u_max = -1e10, u_min = 1e10;

    for (int j = 0; j < ny; ++j) {
        double u_sim = u_profile[j];
        double u_ana = u_analytical[j];
        double error = u_sim - u_ana;
        double error_pct = (u_ana != 0.0) ? std::abs(error / u_ana * 100.0) : 0.0;

        sum_sq_error += error * error;
        sum_abs_error += std::abs(error);
        max_error = std::max(max_error, std::abs(error));
        max_error_pct = std::max(max_error_pct, error_pct);

        u_max = std::max(u_max, u_ana);
        u_min = std::min(u_min, u_ana);
    }

    double u_range = u_max - u_min;
    double l2_error = std::sqrt(sum_sq_error / ny) / u_range * 100.0;
    double rms_error = std::sqrt(sum_sq_error / ny);
    double mean_abs_error = sum_abs_error / ny;

    // Print results
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "RESULTS\n";
    std::cout << "========================================================================\n";
    std::cout << "Convergence:\n";
    std::cout << "  Converged at step: " << (converged_step > 0 ? std::to_string(converged_step) : "Not converged") << "\n";
    std::cout << "\nError metrics:\n";
    std::cout << "  L2 error (normalized): " << std::fixed << std::setprecision(6) << l2_error << " %\n";
    std::cout << "  RMS error:             " << std::scientific << std::setprecision(6) << rms_error << "\n";
    std::cout << "  Mean absolute error:   " << mean_abs_error << "\n";
    std::cout << "  Max absolute error:    " << max_error << "\n";
    std::cout << "  Max relative error:    " << std::fixed << std::setprecision(4) << max_error_pct << " %\n";
    std::cout << "\nVelocity range:\n";
    std::cout << "  u_min (analytical): " << std::fixed << std::setprecision(8) << u_min << "\n";
    std::cout << "  u_max (analytical): " << u_max << "\n";
    std::cout << "  Range: " << u_max - u_min << "\n";
    std::cout << "========================================================================\n";

    // Sample points for quick verification
    std::cout << "\nSample velocity profile (y, η, u_sim, u_ana, error):\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(8) << "y" << std::setw(10) << "η"
              << std::setw(14) << "u_sim" << std::setw(14) << "u_ana"
              << std::setw(14) << "error" << std::setw(12) << "error(%)\n";

    for (int j : {0, ny/8, ny/4, ny/2, 3*ny/4, 7*ny/8, ny-1}) {
        double y = static_cast<double>(j);
        double eta = y / H;
        double error = u_profile[j] - u_analytical[j];
        double error_pct = (u_analytical[j] != 0.0) ? (error / u_analytical[j] * 100.0) : 0.0;

        std::cout << std::setw(8) << j << std::setw(10) << eta
                  << std::setw(14) << u_profile[j]
                  << std::setw(14) << u_analytical[j]
                  << std::setw(14) << error
                  << std::setw(12) << error_pct << "\n";
    }

    // Save velocity profile to file
    std::string output_file = "velocity_profile_64x128_re100.dat";
    save_velocity_profile(output_file, ny, H, U_top, u_profile, u_analytical);

    // Test assertions
    EXPECT_GT(converged_step, 0) << "Simulation should converge";
    EXPECT_LT(l2_error, 1.0) << "L2 error should be less than 1%";
    EXPECT_LT(max_error, 0.01 * (u_max - u_min)) << "Max error should be less than 1% of velocity range";

    std::cout << "\nTest completed successfully!\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
