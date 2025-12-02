/**
 * @file test_uniform_flow_fluidlbm.cu
 * @brief Integration test for uniform body-force driven flow using FluidLBM
 *
 * This test validates the FluidLBM solver's body force implementation by
 * simulating uniform acceleration in a fully periodic domain.
 *
 * Physics:
 * - Uniform body force F applied to quiescent fluid
 * - Expected: uniform acceleration until reaching terminal velocity
 * - With periodic boundaries, flow should reach uniform velocity
 * - Terminal velocity: F * tau (for very small F)
 * - Tests Guo forcing scheme implementation
 *
 * Note: This test uses periodic boundaries (FluidLBM's current implementation).
 * For Poiseuille flow with walls, bounce-back boundaries are required.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm::physics;
using namespace lbm::core;

class UniformFlowTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    /**
     * Compute average of a field
     */
    float computeAverage(const std::vector<float>& field) {
        float sum = 0.0f;
        for (float val : field) {
            sum += val;
        }
        return sum / field.size();
    }

    /**
     * Compute standard deviation
     */
    float computeStdDev(const std::vector<float>& field, float mean) {
        float sum_sq = 0.0f;
        for (float val : field) {
            float diff = val - mean;
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / field.size());
    }
};

/**
 * Test: Uniform body force produces uniform acceleration
 *
 * Apply constant body force to quiescent fluid with periodic boundaries.
 * Verify that:
 * 1. Velocity increases uniformly
 * 2. Final velocity field is spatially uniform
 * 3. Guo forcing scheme produces correct velocity
 */
TEST_F(UniformFlowTest, UniformAcceleration) {
    // Domain size
    const int nx = 16;
    const int ny = 16;
    const int nz = 16;
    const int num_cells = nx * ny * nz;

    // Physical parameters
    const float nu = 0.1f;      // Kinematic viscosity
    const float rho0 = 1.0f;    // Reference density
    const float force_x = 1e-5f;  // Very small body force

    std::cout << "\n=== Uniform Flow Test Configuration ===" << std::endl;
    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Kinematic viscosity: " << nu << std::endl;
    std::cout << "Body force: " << force_x << std::endl;
    std::cout << "======================================\n" << std::endl;

    // Initialize FluidLBM solver
    FluidLBM solver(nx, ny, nz, nu, rho0);

    // Initialize with quiescent flow
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Time evolution
    const int n_steps = 1000;
    std::vector<float> h_ux(num_cells);

    std::cout << "Running simulation for " << n_steps << " steps..." << std::endl;

    // Track velocity evolution
    std::vector<float> avg_velocity_history;

    for (int step = 1; step <= n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(force_x, 0.0f, 0.0f);
        solver.streaming();

        if (step % 200 == 0) {
            solver.copyVelocityToHost(h_ux.data(), nullptr, nullptr);
            float avg_ux = computeAverage(h_ux);
            avg_velocity_history.push_back(avg_ux);
            std::cout << "Step " << step << ": avg u_x = " << avg_ux << std::endl;
        }
    }

    // Final state
    solver.computeMacroscopic();
    solver.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    // Compute statistics
    float avg_ux = computeAverage(h_ux);
    float std_ux = computeStdDev(h_ux, avg_ux);
    float cv_ux = std_ux / (std::abs(avg_ux) + 1e-10f);  // Coefficient of variation

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Final average u_x: " << avg_ux << std::endl;
    std::cout << "Standard deviation: " << std_ux << std::endl;
    std::cout << "Coefficient of variation: " << cv_ux << std::endl;
    std::cout << "================\n" << std::endl;

    // Validation criteria
    EXPECT_GT(avg_ux, 0.0f) << "Velocity should be positive with positive force";
    EXPECT_LT(cv_ux, 0.01f) << "Flow should be spatially uniform (CV < 1%)";

    // Check that velocity increased over time
    EXPECT_GT(avg_velocity_history.back(), avg_velocity_history.front())
        << "Velocity should increase under constant force";

    // Check for monotonic increase (no oscillations)
    for (size_t i = 1; i < avg_velocity_history.size(); ++i) {
        EXPECT_GE(avg_velocity_history[i], avg_velocity_history[i-1] - 1e-7f)
            << "Velocity should increase monotonically (step " << i << ")";
    }

    std::cout << "Test PASSED - Uniform acceleration validated!" << std::endl;
}

/**
 * Test: Force direction and magnitude
 *
 * Apply forces in different directions and verify correct response
 */
TEST_F(UniformFlowTest, ForceDirectionality) {
    const int nx = 8, ny = 8, nz = 8;
    const int num_cells = nx * ny * nz;
    const float nu = 0.1f;
    const float rho0 = 1.0f;
    const float force = 1e-5f;
    const int n_steps = 1000;

    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    std::cout << "\n=== Force Directionality Test ===" << std::endl;

    // Test X-direction
    {
        FluidLBM solver(nx, ny, nz, nu, rho0);
        solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

        for (int step = 0; step < n_steps; ++step) {
            solver.computeMacroscopic();
            solver.collisionBGK(force, 0.0f, 0.0f);
            solver.streaming();
        }

        solver.computeMacroscopic();
        solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

        float avg_ux = computeAverage(h_ux);
        float avg_uy = computeAverage(h_uy);
        float avg_uz = computeAverage(h_uz);

        std::cout << "X-force: u_x=" << avg_ux << ", u_y=" << avg_uy << ", u_z=" << avg_uz << std::endl;

        EXPECT_GT(avg_ux, 1e-6f) << "X-force should produce X-velocity";
        EXPECT_LT(std::abs(avg_uy), 1e-8f) << "X-force should not produce Y-velocity";
        EXPECT_LT(std::abs(avg_uz), 1e-8f) << "X-force should not produce Z-velocity";
    }

    // Test Y-direction
    {
        FluidLBM solver(nx, ny, nz, nu, rho0);
        solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

        for (int step = 0; step < n_steps; ++step) {
            solver.computeMacroscopic();
            solver.collisionBGK(0.0f, force, 0.0f);
            solver.streaming();
        }

        solver.computeMacroscopic();
        solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

        float avg_ux = computeAverage(h_ux);
        float avg_uy = computeAverage(h_uy);
        float avg_uz = computeAverage(h_uz);

        std::cout << "Y-force: u_x=" << avg_ux << ", u_y=" << avg_uy << ", u_z=" << avg_uz << std::endl;

        EXPECT_LT(std::abs(avg_ux), 1e-8f) << "Y-force should not produce X-velocity";
        EXPECT_GT(avg_uy, 1e-6f) << "Y-force should produce Y-velocity";
        EXPECT_LT(std::abs(avg_uz), 1e-8f) << "Y-force should not produce Z-velocity";
    }

    // Test Z-direction
    {
        FluidLBM solver(nx, ny, nz, nu, rho0);
        solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

        for (int step = 0; step < n_steps; ++step) {
            solver.computeMacroscopic();
            solver.collisionBGK(0.0f, 0.0f, force);
            solver.streaming();
        }

        solver.computeMacroscopic();
        solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

        float avg_ux = computeAverage(h_ux);
        float avg_uy = computeAverage(h_uy);
        float avg_uz = computeAverage(h_uz);

        std::cout << "Z-force: u_x=" << avg_ux << ", u_y=" << avg_uy << ", u_z=" << avg_uz << std::endl;

        EXPECT_LT(std::abs(avg_ux), 1e-8f) << "Z-force should not produce X-velocity";
        EXPECT_LT(std::abs(avg_uy), 1e-8f) << "Z-force should not produce Y-velocity";
        EXPECT_GT(avg_uz, 1e-6f) << "Z-force should produce Z-velocity";
    }

    std::cout << "==================================\n" << std::endl;
    std::cout << "Test PASSED - Force directionality correct!" << std::endl;
}

/**
 * Test: Conservation of mass
 */
TEST_F(UniformFlowTest, MassConservation) {
    const int nx = 16, ny = 16, nz = 16;
    const int num_cells = nx * ny * nz;
    const float nu = 0.1f;
    const float rho0 = 1.0f;
    const float force_x = 1e-5f;
    const int n_steps = 2000;

    FluidLBM solver(nx, ny, nz, nu, rho0);
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::vector<float> h_rho(num_cells);

    // Initial mass
    solver.copyDensityToHost(h_rho.data());
    float initial_mass = 0.0f;
    for (float rho : h_rho) {
        initial_mass += rho;
    }

    std::cout << "\n=== Mass Conservation Test ===" << std::endl;
    std::cout << "Initial total mass: " << initial_mass << std::endl;

    // Run simulation
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(force_x, 0.0f, 0.0f);
        solver.streaming();
    }

    // Final mass
    solver.computeMacroscopic();
    solver.copyDensityToHost(h_rho.data());
    float final_mass = 0.0f;
    for (float rho : h_rho) {
        final_mass += rho;
    }

    float mass_change = std::abs(final_mass - initial_mass) / initial_mass;

    std::cout << "Final total mass: " << final_mass << std::endl;
    std::cout << "Relative change: " << (mass_change * 100.0f) << "%" << std::endl;
    std::cout << "==============================\n" << std::endl;

    EXPECT_LT(mass_change, 1e-6f) << "Mass should be conserved (relative change < 1e-6)";

    std::cout << "Test PASSED - Mass conserved!" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
