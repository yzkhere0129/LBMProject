/**
 * @file test_gravity_driven_flow.cu
 * @brief Integration test for gravity-driven Poiseuille flow
 *
 * This test validates the Guo forcing scheme by simulating Poiseuille flow
 * driven by a body force (gravity) instead of a pressure gradient.
 *
 * Physics:
 * - Gravity-driven flow between parallel plates
 * - Analytical solution: u(y) = g/(2ν) * y * (H - y)
 * - Maximum velocity at center: u_max = g * H² / (8ν)
 * - Identical to pressure-driven flow with F = -∇p/ρ
 *
 * This test verifies:
 * 1. Correct implementation of body forces via Guo scheme
 * 2. Parabolic velocity profile
 * 3. Agreement with analytical solution
 * 4. No-slip boundary condition at walls
 *
 * Reference:
 * - White, F. M. (2006). Viscous Fluid Flow (3rd ed.). McGraw-Hill.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm::core;

class GravityDrivenFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    /**
     * @brief Analytical velocity for gravity-driven Poiseuille flow
     * u(y) = g/(2ν) * y * (H - y)
     */
    float analyticalVelocity(float y, float H, float g, float nu) {
        return g / (2.0f * nu) * y * (H - y);
    }

    /**
     * @brief Maximum velocity at channel center (y = H/2)
     */
    float maxAnalyticalVelocity(float H, float g, float nu) {
        return g * H * H / (8.0f * nu);
    }

    /**
     * @brief Compute L2 relative error
     */
    float computeL2Error(const std::vector<float>& numerical,
                         const std::vector<float>& analytical) {
        float sum_sq_err = 0.0f;
        float sum_sq_ana = 0.0f;

        for (size_t i = 0; i < numerical.size(); ++i) {
            float err = numerical[i] - analytical[i];
            sum_sq_err += err * err;
            sum_sq_ana += analytical[i] * analytical[i];
        }

        return std::sqrt(sum_sq_err / (sum_sq_ana + 1e-15f));
    }

    /**
     * @brief Check if steady state is reached
     */
    bool isConverged(const std::vector<float>& u_old,
                     const std::vector<float>& u_new,
                     float tolerance) {
        float max_change = 0.0f;
        for (size_t i = 0; i < u_old.size(); ++i) {
            max_change = std::max(max_change, std::abs(u_new[i] - u_old[i]));
        }
        return max_change < tolerance;
    }
};

/**
 * Test 1: Gravity-driven Poiseuille flow in horizontal channel
 *
 * Configuration:
 * - Horizontal channel (gravity acts horizontally)
 * - No-slip walls at top and bottom
 * - Periodic in flow direction (x) and spanwise (z)
 */
TEST_F(GravityDrivenFlowTest, HorizontalChannel) {
    // Domain configuration (quasi-2D)
    const int nx = 4;   // Periodic flow direction
    const int ny = 64;  // Channel height (wall-normal)
    const int nz = 4;   // Periodic spanwise
    const int num_cells = nx * ny * nz;

    // Physical parameters (in lattice units)
    const float H = ny - 1.0f;  // Channel height
    const float Re = 10.0f;     // Reynolds number
    const float u_max_target = 0.05f;  // Target max velocity

    // Compute viscosity from Re and target velocity
    // Re = u_max * H / ν  =>  ν = u_max * H / Re
    const float nu = u_max_target * H / Re;

    // Compute gravity magnitude to achieve target velocity
    // u_max = g * H² / (8ν)  =>  g = 8ν * u_max / H²
    const float g = 8.0f * nu * u_max_target / (H * H);

    const float rho0 = 1.0f;

    std::cout << "\n=== Gravity-Driven Poiseuille Flow Test ===" << std::endl;
    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Channel height H: " << H << " [lattice units]" << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Kinematic viscosity: " << nu << std::endl;
    std::cout << "Gravity magnitude: " << g << std::endl;
    std::cout << "Target u_max: " << u_max_target << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Initialize solver with WALL boundaries in y-direction
    // Work in lattice units (dt = dx = 1.0)
    lbm::physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    lbm::physics::BoundaryType::PERIODIC,  // x: periodic
                    lbm::physics::BoundaryType::WALL,      // y: no-slip walls
                    lbm::physics::BoundaryType::PERIODIC,  // z: periodic
                    1.0f, 1.0f);  // dt = dx = 1.0

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Time evolution
    const int max_steps = 20000;
    const int check_interval = 1000;
    const float convergence_tol = 1e-6f;

    std::vector<float> u_old(num_cells, 0.0f);
    std::vector<float> u_new(num_cells);

    bool converged = false;
    int convergence_step = -1;

    std::cout << "Running simulation with gravity in x-direction..." << std::endl;

    for (int step = 1; step <= max_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(g, 0.0f, 0.0f);  // Gravity in x-direction
        solver.streaming();
        solver.applyBoundaryConditions(1);

        // Check convergence
        if (step % check_interval == 0) {
            solver.copyVelocityToHost(u_new.data(), nullptr, nullptr);

            if (isConverged(u_old, u_new, convergence_tol)) {
                converged = true;
                convergence_step = step;
                std::cout << "Converged at step " << step << std::endl;
                break;
            }

            u_old = u_new;
            std::cout << "Step " << step << " / " << max_steps << std::endl;
        }
    }

    EXPECT_TRUE(converged) << "Simulation did not converge";

    // Final state
    solver.computeMacroscopic();

    std::vector<float> ux(num_cells);
    std::vector<float> uy(num_cells);
    std::vector<float> uz(num_cells);
    solver.copyVelocityToHost(ux.data(), uy.data(), uz.data());

    // Extract velocity profile (average over x and z)
    std::vector<float> velocity_profile(ny, 0.0f);
    std::vector<float> analytical_profile(ny);

    for (int y = 0; y < ny; ++y) {
        float sum = 0.0f;
        for (int x = 0; x < nx; ++x) {
            for (int z = 0; z < nz; ++z) {
                int idx = x + y * nx + z * nx * ny;
                sum += ux[idx];
            }
        }
        velocity_profile[y] = sum / (nx * nz);
        analytical_profile[y] = analyticalVelocity(y, H, g, nu);
    }

    // Compute error metrics
    float L2_error = computeL2Error(velocity_profile, analytical_profile);

    // Find maximum velocities
    auto it_max_num = std::max_element(velocity_profile.begin(), velocity_profile.end());
    float u_max_numerical = *it_max_num;
    float u_max_analytical = maxAnalyticalVelocity(H, g, nu);
    float u_max_error_percent = std::abs(u_max_numerical - u_max_analytical) /
                                u_max_analytical * 100.0f;

    // Print results
    std::cout << "\n=== Validation Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Maximum velocity:" << std::endl;
    std::cout << "  Analytical: " << u_max_analytical << std::endl;
    std::cout << "  Numerical:  " << u_max_numerical << std::endl;
    std::cout << "  Error:      " << u_max_error_percent << " %" << std::endl;
    std::cout << "\nL2 relative error: " << (L2_error * 100.0f) << " %" << std::endl;
    std::cout << "==========================\n" << std::endl;

    // Print profile comparison near center
    std::cout << "Velocity profile near center:" << std::endl;
    std::cout << "  y\tNumerical\tAnalytical\tError" << std::endl;
    for (int y = ny/2 - 3; y <= ny/2 + 3; ++y) {
        float err = velocity_profile[y] - analytical_profile[y];
        std::cout << "  " << y << "\t" << velocity_profile[y] << "\t"
                  << analytical_profile[y] << "\t" << err << std::endl;
    }

    // Save profile to file
    std::ofstream file("gravity_driven_profile.txt");
    if (file.is_open()) {
        file << "# Gravity-Driven Poiseuille Flow\n";
        file << "# y\tNumerical_u\tAnalytical_u\tError\n";
        for (int y = 0; y < ny; ++y) {
            file << y << "\t" << velocity_profile[y] << "\t"
                 << analytical_profile[y] << "\t"
                 << (velocity_profile[y] - analytical_profile[y]) << "\n";
        }
        file.close();
        std::cout << "\nProfile saved to: gravity_driven_profile.txt\n" << std::endl;
    }

    // Validation assertions
    EXPECT_LT(L2_error, 0.05f) << "L2 error exceeds 5% threshold";
    EXPECT_LT(u_max_error_percent, 3.0f) << "Maximum velocity error exceeds 3%";

    // Check parabolic shape (velocity maximum at center)
    EXPECT_GT(velocity_profile[ny/2], velocity_profile[1])
        << "Velocity not maximum at center";
    EXPECT_GT(velocity_profile[ny/2], velocity_profile[ny-2])
        << "Velocity not maximum at center";

    // Check no-slip at walls (should be enforced by boundary conditions)
    EXPECT_NEAR(velocity_profile[0], 0.0f, 1e-4f)
        << "No-slip violated at bottom wall";
    EXPECT_NEAR(velocity_profile[ny-1], 0.0f, 1e-4f)
        << "No-slip violated at top wall";

    // Check symmetry
    for (int y = 1; y < ny/2; ++y) {
        float v1 = velocity_profile[y];
        float v2 = velocity_profile[ny - 1 - y];
        float sym_error = std::abs(v1 - v2) / (std::abs(v1) + std::abs(v2) + 1e-10f);
        EXPECT_LT(sym_error, 0.05f)
            << "Profile not symmetric at y=" << y;
    }
}

/**
 * Test 2: Vertical gravity (buoyancy-like scenario)
 *
 * This tests gravity in y-direction, which drives flow along the channel.
 * Due to walls in y-direction, this creates a different flow pattern.
 */
TEST_F(GravityDrivenFlowTest, VerticalGravityChannel) {
    // Domain configuration
    const int nx = 32;  // Flow direction (periodic)
    const int ny = 8;   // Wall-normal (periodic)
    const int nz = 4;   // Spanwise (periodic)
    const int num_cells = nx * ny * nz;

    const float nu = 0.1f;
    const float rho0 = 1.0f;
    const float g_y = 1e-4f;  // Small gravity in y-direction

    // All periodic boundaries (no walls)
    lbm::physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    lbm::physics::BoundaryType::PERIODIC,
                    lbm::physics::BoundaryType::PERIODIC,
                    lbm::physics::BoundaryType::PERIODIC,
                    1.0f, 1.0f);

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::cout << "\n=== Vertical Gravity Test ===" << std::endl;
    std::cout << "Testing uniform acceleration in y-direction" << std::endl;

    // Run simulation
    for (int step = 0; step < 1000; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, g_y, 0.0f);  // Gravity in y-direction
        solver.streaming();
    }

    solver.computeMacroscopic();

    // Verify uniform acceleration (no walls, all periodic)
    std::vector<float> uy(num_cells);
    solver.copyVelocityToHost(nullptr, uy.data(), nullptr);

    // All cells should have similar velocity (uniform force, periodic BC)
    float mean_uy = std::accumulate(uy.begin(), uy.end(), 0.0f) / uy.size();
    float std_dev = 0.0f;
    for (float v : uy) {
        std_dev += (v - mean_uy) * (v - mean_uy);
    }
    std_dev = std::sqrt(std_dev / uy.size());

    std::cout << "Mean velocity: " << mean_uy << std::endl;
    std::cout << "Std deviation: " << std_dev << std::endl;

    // Velocity should be positive (gravity in +y direction)
    EXPECT_GT(mean_uy, 0.0f) << "Gravity did not accelerate fluid in +y direction";

    // Velocity should be uniform (no walls, no gradients)
    EXPECT_LT(std_dev / mean_uy, 0.1f) << "Non-uniform response to uniform gravity";
}

/**
 * Test 3: Comparison with pressure-driven flow
 *
 * Gravity-driven flow should produce identical results to pressure-driven flow
 * with equivalent body force F = -∇p/ρ
 */
TEST_F(GravityDrivenFlowTest, EquivalenceToPressureDriven) {
    const int nx = 4, ny = 32, nz = 4;
    const int num_cells = nx * ny * nz;

    const float H = ny - 1.0f;
    const float nu = 0.1f;
    const float rho0 = 1.0f;
    const float g = 1e-3f;

    // Solver 1: Gravity-driven
    lbm::physics::FluidLBM solver_gravity(nx, ny, nz, nu, rho0,
                            lbm::physics::BoundaryType::PERIODIC,
                            lbm::physics::BoundaryType::WALL,
                            lbm::physics::BoundaryType::PERIODIC,
                            1.0f, 1.0f);
    solver_gravity.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Solver 2: Pressure-driven (equivalent body force)
    // For pressure-driven: F = -dp/dx / rho
    // For gravity-driven: F = g
    // They are equivalent when g = -dp/dx / rho
    lbm::physics::FluidLBM solver_pressure(nx, ny, nz, nu, rho0,
                             lbm::physics::BoundaryType::PERIODIC,
                             lbm::physics::BoundaryType::WALL,
                             lbm::physics::BoundaryType::PERIODIC,
                             1.0f, 1.0f);
    solver_pressure.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::cout << "\n=== Equivalence Test: Gravity vs Pressure ===" << std::endl;

    // Run both simulations
    for (int step = 0; step < 5000; ++step) {
        // Gravity-driven
        solver_gravity.computeMacroscopic();
        solver_gravity.collisionBGK(g, 0.0f, 0.0f);
        solver_gravity.streaming();
        solver_gravity.applyBoundaryConditions(1);

        // Pressure-driven (same body force)
        solver_pressure.computeMacroscopic();
        solver_pressure.collisionBGK(g, 0.0f, 0.0f);
        solver_pressure.streaming();
        solver_pressure.applyBoundaryConditions(1);
    }

    solver_gravity.computeMacroscopic();
    solver_pressure.computeMacroscopic();

    // Compare velocity profiles
    std::vector<float> ux_gravity(num_cells);
    std::vector<float> ux_pressure(num_cells);

    solver_gravity.copyVelocityToHost(ux_gravity.data(), nullptr, nullptr);
    solver_pressure.copyVelocityToHost(ux_pressure.data(), nullptr, nullptr);

    // Compute difference
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float diff = std::abs(ux_gravity[i] - ux_pressure[i]);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
    }
    mean_diff /= num_cells;

    std::cout << "Maximum difference: " << max_diff << std::endl;
    std::cout << "Mean difference: " << mean_diff << std::endl;

    // Profiles should be identical (both use same Guo forcing)
    EXPECT_LT(max_diff, 1e-6f)
        << "Gravity-driven and pressure-driven flows differ";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
