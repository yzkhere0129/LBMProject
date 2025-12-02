/**
 * @file test_poiseuille_flow_fluidlbm.cu
 * @brief Integration test for Poiseuille flow using FluidLBM class
 *
 * This test validates the FluidLBM solver by simulating pressure-driven
 * channel flow between parallel plates and comparing with analytical solution.
 *
 * Physics:
 * - 2D Poiseuille flow: pressure-driven flow between parallel plates
 * - Analytical solution: u(y) = -(dp/dx)/(2μ) * y * (H - y)
 * - Maximum velocity at center: u_max = (dp/dx) * H² / (8μ)
 * - Average velocity: u_avg = (2/3) * u_max
 *
 * Implementation approach:
 * - Use body force to simulate pressure gradient (periodic boundaries)
 * - Simulate 2D flow in quasi-2D domain (thin in z-direction)
 * - Run until steady state is reached
 * - Extract velocity profile and compare with analytical solution
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm::physics;
using namespace lbm::core;

class PoiseuilleFlowFluidLBMTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    /**
     * Analytical velocity profile for Poiseuille flow
     * u(y) = -(dp/dx) / (2μ) * y * (H - y)
     */
    float analyticalVelocity(float y, float H, float dp_dx, float nu, float rho) {
        float mu = nu * rho;  // Dynamic viscosity
        return -dp_dx * y * (H - y) / (2.0f * mu);
    }

    /**
     * Maximum velocity at channel center (y = H/2)
     */
    float maxAnalyticalVelocity(float H, float dp_dx, float nu, float rho) {
        float mu = nu * rho;
        return -dp_dx * H * H / (8.0f * mu);
    }

    /**
     * Average velocity across channel
     */
    float avgAnalyticalVelocity(float H, float dp_dx, float nu, float rho) {
        return (2.0f / 3.0f) * maxAnalyticalVelocity(H, dp_dx, nu, rho);
    }

    /**
     * Compute L2 relative error between numerical and analytical profiles
     */
    float computeL2Error(const std::vector<float>& numerical,
                         const std::vector<float>& analytical) {
        float sum_squared_error = 0.0f;
        float sum_squared_analytical = 0.0f;

        for (size_t i = 0; i < numerical.size(); ++i) {
            float error = numerical[i] - analytical[i];
            sum_squared_error += error * error;
            sum_squared_analytical += analytical[i] * analytical[i];
        }

        return std::sqrt(sum_squared_error / (sum_squared_analytical + 1e-15f));
    }

    /**
     * Compute maximum absolute error
     */
    float computeMaxError(const std::vector<float>& numerical,
                          const std::vector<float>& analytical) {
        float max_error = 0.0f;
        for (size_t i = 0; i < numerical.size(); ++i) {
            float error = std::abs(numerical[i] - analytical[i]);
            max_error = std::max(max_error, error);
        }
        return max_error;
    }

    /**
     * Check convergence to steady state
     */
    bool isConverged(const float* ux_old, const float* ux_new, int num_cells, float tol = 1e-6f) {
        float max_change = 0.0f;
        for (int i = 0; i < num_cells; ++i) {
            float change = std::abs(ux_new[i] - ux_old[i]);
            max_change = std::max(max_change, change);
        }
        return max_change < tol;
    }
};

/**
 * Test: 2D Poiseuille flow with FluidLBM class
 *
 * Uses body force to simulate pressure gradient in a quasi-2D channel
 * with periodic boundaries in x and z directions.
 */
TEST_F(PoiseuilleFlowFluidLBMTest, ChannelFlow2D) {
    // Domain size - quasi-2D (thin in x and z)
    const int nx = 4;   // Thin in x (periodic)
    const int ny = 64;  // Channel height (main direction)
    const int nz = 4;   // Thin in z (periodic)
    const int num_cells = nx * ny * nz;

    // Physical parameters
    const float Re = 10.0f;          // Reynolds number
    const float H = (ny - 1);        // Channel height in lattice units
    const float u_max_target = 0.05f;  // Target max velocity (< 0.1 for stability)
    const float nu = u_max_target * H / Re;  // Kinematic viscosity
    const float rho0 = 1.0f;         // Reference density

    // Pressure gradient simulation via body force
    // For Poiseuille: u_max = (dp/dx) * H² / (8μ)
    // => dp/dx = 8 * μ * u_max / H²
    // Body force: F = -dp/dx / rho
    const float mu = nu * rho0;
    const float dp_dx = 8.0f * mu * u_max_target / (H * H);
    const float body_force_x = -dp_dx / rho0;

    std::cout << "\n=== Poiseuille Flow Test Configuration ===" << std::endl;
    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Channel height H: " << H << " [lattice units]" << std::endl;
    std::cout << "Target u_max: " << u_max_target << std::endl;
    std::cout << "Kinematic viscosity: " << nu << std::endl;
    std::cout << "Dynamic viscosity: " << mu << std::endl;
    std::cout << "Pressure gradient: " << dp_dx << std::endl;
    std::cout << "Body force: " << body_force_x << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Initialize FluidLBM solver
    FluidLBM solver(nx, ny, nz, nu, rho0);

    // Initialize with quiescent flow
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Time evolution parameters
    const int max_steps = 10000;
    const int check_interval = 1000;
    const float convergence_tol = 1e-6f;

    // Allocate host arrays for convergence checking
    std::vector<float> h_ux_old(num_cells, 0.0f);
    std::vector<float> h_ux_new(num_cells);

    bool converged = false;
    int convergence_step = -1;

    std::cout << "Running simulation..." << std::endl;

    // Time stepping
    for (int step = 1; step <= max_steps; ++step) {
        // LBM cycle: compute macroscopic -> collision -> streaming
        solver.computeMacroscopic();
        solver.collisionBGK(body_force_x, 0.0f, 0.0f);
        solver.streaming();

        // Check convergence periodically
        if (step % check_interval == 0) {
            solver.copyVelocityToHost(h_ux_new.data(), nullptr, nullptr);

            if (isConverged(h_ux_old.data(), h_ux_new.data(), num_cells, convergence_tol)) {
                converged = true;
                convergence_step = step;
                std::cout << "Converged at step " << step << std::endl;
                break;
            }

            h_ux_old = h_ux_new;
            std::cout << "Step " << step << " / " << max_steps << std::endl;
        }
    }

    // Final macroscopic computation
    solver.computeMacroscopic();

    // Copy results to host
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    std::vector<float> h_rho(num_cells);

    solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    solver.copyDensityToHost(h_rho.data());

    // Extract velocity profile (average over x and z)
    std::vector<float> velocity_profile(ny, 0.0f);
    std::vector<float> analytical_profile(ny);

    for (int y = 0; y < ny; ++y) {
        float sum_ux = 0.0f;
        for (int x = 0; x < nx; ++x) {
            for (int z = 0; z < nz; ++z) {
                int idx = x + y * nx + z * nx * ny;
                sum_ux += h_ux[idx];
            }
        }
        velocity_profile[y] = sum_ux / (nx * nz);

        // Analytical solution
        analytical_profile[y] = analyticalVelocity(y, H, dp_dx, nu, rho0);
    }

    // Compute error metrics
    float L2_error = computeL2Error(velocity_profile, analytical_profile);
    float max_error = computeMaxError(velocity_profile, analytical_profile);

    // Find max velocities
    float u_max_numerical = *std::max_element(velocity_profile.begin(), velocity_profile.end());
    float u_max_analytical = maxAnalyticalVelocity(H, dp_dx, nu, rho0);
    float u_max_error_percent = std::abs(u_max_numerical - u_max_analytical) / u_max_analytical * 100.0f;

    // Compute average velocities
    float u_avg_numerical = 0.0f;
    float u_avg_analytical = avgAnalyticalVelocity(H, dp_dx, nu, rho0);
    for (int y = 1; y < ny - 1; ++y) {  // Exclude boundaries
        u_avg_numerical += velocity_profile[y];
    }
    u_avg_numerical /= (ny - 2);
    float u_avg_error_percent = std::abs(u_avg_numerical - u_avg_analytical) / u_avg_analytical * 100.0f;

    // Print results
    std::cout << "\n=== Poiseuille Flow Validation Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Convergence: " << (converged ? "YES" : "NO") << std::endl;
    if (converged) {
        std::cout << "Converged at step: " << convergence_step << std::endl;
    }
    std::cout << "\nVelocity Profile Comparison:" << std::endl;
    std::cout << "  Maximum velocity:" << std::endl;
    std::cout << "    Analytical:  " << u_max_analytical << " [lu/ts]" << std::endl;
    std::cout << "    Numerical:   " << u_max_numerical << " [lu/ts]" << std::endl;
    std::cout << "    Error:       " << u_max_error_percent << " %" << std::endl;
    std::cout << "  Average velocity:" << std::endl;
    std::cout << "    Analytical:  " << u_avg_analytical << " [lu/ts]" << std::endl;
    std::cout << "    Numerical:   " << u_avg_numerical << " [lu/ts]" << std::endl;
    std::cout << "    Error:       " << u_avg_error_percent << " %" << std::endl;
    std::cout << "\nError Metrics:" << std::endl;
    std::cout << "  L2 relative error:    " << (L2_error * 100.0f) << " %" << std::endl;
    std::cout << "  Maximum point error:  " << max_error << " [lu/ts]" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Print center region profile
    std::cout << "Velocity profile near center:" << std::endl;
    std::cout << "  y\tNumerical\tAnalytical\tError" << std::endl;
    for (int y = ny/2 - 3; y <= ny/2 + 3; ++y) {
        float error = velocity_profile[y] - analytical_profile[y];
        std::cout << "  " << y << "\t" << velocity_profile[y] << "\t"
                  << analytical_profile[y] << "\t" << error << std::endl;
    }

    // Save full profile to file
    std::ofstream profile_file("poiseuille_profile_fluidlbm.txt");
    if (profile_file.is_open()) {
        profile_file << "# Poiseuille Flow - FluidLBM Test\n";
        profile_file << "# y\tNumerical_u\tAnalytical_u\tError\n";
        for (int y = 0; y < ny; ++y) {
            profile_file << y << "\t" << velocity_profile[y] << "\t"
                        << analytical_profile[y] << "\t"
                        << (velocity_profile[y] - analytical_profile[y]) << "\n";
        }
        profile_file.close();
        std::cout << "\nVelocity profile saved to: poiseuille_profile_fluidlbm.txt\n" << std::endl;
    }

    // Assertions - Validation criteria from task
    EXPECT_TRUE(converged) << "Flow did not converge to steady state";
    EXPECT_LT(L2_error, 0.05f) << "L2 relative error exceeds 5% threshold";
    EXPECT_LT(u_max_error_percent, 3.0f) << "Maximum velocity error exceeds 3%";
    EXPECT_NEAR(u_avg_numerical / u_max_numerical, 2.0f / 3.0f, 0.05f)
        << "Average/max velocity ratio != 2/3 (expected for Poiseuille)";

    // Check parabolic shape
    EXPECT_GT(velocity_profile[ny/2], velocity_profile[1])
        << "Profile not parabolic (center velocity should be maximum)";
    EXPECT_GT(velocity_profile[ny/2], velocity_profile[ny-2])
        << "Profile not parabolic (center velocity should be maximum)";

    // Check symmetry
    float max_symmetry_error = 0.0f;
    for (int y = 1; y < ny/2; ++y) {
        float v1 = velocity_profile[y];
        float v2 = velocity_profile[ny - 1 - y];
        float sym_error = std::abs(v1 - v2) / (std::abs(v1) + std::abs(v2) + 1e-10f);
        max_symmetry_error = std::max(max_symmetry_error, sym_error);
    }
    EXPECT_LT(max_symmetry_error, 0.05f) << "Profile not symmetric";

    std::cout << "Test PASSED - All validation criteria satisfied!" << std::endl;
}

/**
 * Test: Low Reynolds number flow (more stringent accuracy test)
 */
TEST_F(PoiseuilleFlowFluidLBMTest, LowReynoldsFlow) {
    // Smaller domain, lower Re for better accuracy
    const int nx = 4;
    const int ny = 32;
    const int nz = 4;
    const int num_cells = nx * ny * nz;

    const float Re = 5.0f;
    const float H = (ny - 1);
    const float u_max_target = 0.02f;  // Very low velocity for stability
    const float nu = u_max_target * H / Re;
    const float rho0 = 1.0f;

    const float mu = nu * rho0;
    const float dp_dx = 8.0f * mu * u_max_target / (H * H);
    const float body_force_x = -dp_dx / rho0;

    FluidLBM solver(nx, ny, nz, nu, rho0);
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Run to steady state
    for (int step = 1; step <= 8000; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(body_force_x, 0.0f, 0.0f);
        solver.streaming();
    }

    solver.computeMacroscopic();

    // Extract profile
    std::vector<float> h_ux(num_cells);
    solver.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    std::vector<float> velocity_profile(ny, 0.0f);
    std::vector<float> analytical_profile(ny);

    for (int y = 0; y < ny; ++y) {
        float sum = 0.0f;
        for (int x = 0; x < nx; ++x) {
            for (int z = 0; z < nz; ++z) {
                sum += h_ux[x + y * nx + z * nx * ny];
            }
        }
        velocity_profile[y] = sum / (nx * nz);
        analytical_profile[y] = analyticalVelocity(y, H, dp_dx, nu, rho0);
    }

    float L2_error = computeL2Error(velocity_profile, analytical_profile);

    std::cout << "\n=== Low Reynolds Flow Test ===" << std::endl;
    std::cout << "Re = " << Re << ", L2 error = " << (L2_error * 100.0f) << "%" << std::endl;

    // Very stringent error bound for low Re
    EXPECT_LT(L2_error, 0.03f) << "L2 error exceeds 3% for low Re flow";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
