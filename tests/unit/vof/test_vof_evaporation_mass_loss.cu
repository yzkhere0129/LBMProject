/**
 * @file test_vof_evaporation_mass_loss.cu
 * @brief Test VOF-thermal coupling: evaporation mass loss
 *
 * Physics: Evaporation mass flux causes fill level reduction
 * - Formula: df/dt = -J_evap / (ρ * dx)
 * - Integration: df = -J_evap * dt / (ρ * dx)
 * - Units: J_evap [kg/(m²·s)], ρ [kg/m³], dx [m], dt [s]
 * - Result: df [dimensionless]
 *
 * Validates:
 * - Correct mass loss rate
 * - Stability limiter (max 2% per timestep)
 * - Total mass conservation
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace lbm::physics;

class VOFEvaporationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Set uniform evaporation flux on device
     */
    void setUniformEvaporationFlux(float* d_J_evap, float J_value, int num_cells) {
        std::vector<float> h_J(num_cells, J_value);
        cudaMemcpy(d_J_evap, h_J.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }

    /**
     * @brief Set evaporation flux only at top layer
     */
    void setTopLayerEvaporation(float* d_J_evap, float J_value, int nx, int ny, int nz) {
        int num_cells = nx * ny * nz;
        std::vector<float> h_J(num_cells, 0.0f);

        // Set evaporation only at top layer (z = nz-1)
        int top_layer_start = nx * ny * (nz - 1);
        for (int i = top_layer_start; i < num_cells; ++i) {
            h_J[i] = J_value;
        }

        cudaMemcpy(d_J_evap, h_J.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }
};

/**
 * @brief Test 1: Single Timestep Mass Loss
 *
 * Validates formula: df = -J_evap * dt / (ρ * dx)
 * Single step, no limiter active
 */
TEST_F(VOFEvaporationTest, SingleTimestepMassLoss) {
    std::cout << "\n=== VOF Evaporation: Single Timestep ===" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;  // 2 μm
    const int num_cells = nx * ny * nz;

    // Material properties (Ti-6Al-4V)
    const float rho = 4420.0f;  // kg/m³

    // Evaporation flux (moderate, below limiter threshold)
    const float J_evap = 100.0f;  // kg/(m²·s)
    const float dt = 1e-7f;       // 0.1 μs

    // Expected fill level change per cell
    const float df_expected = -J_evap * dt / (rho * dx);

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Material: ρ = " << rho << " kg/m³" << std::endl;
    std::cout << "  Evaporation flux: J = " << J_evap << " kg/(m²·s)" << std::endl;
    std::cout << "  Time step: dt = " << dt * 1e6 << " μs" << std::endl;
    std::cout << "  Expected df: " << df_expected << std::endl;

    // Initialize VOF with uniform fill level
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(1.0f);  // Fully filled

    float mass_initial = vof.computeTotalMass();
    std::cout << "  Initial mass: M0 = " << mass_initial << std::endl;

    // Setup evaporation flux
    float* d_J_evap;
    cudaMalloc(&d_J_evap, num_cells * sizeof(float));
    setUniformEvaporationFlux(d_J_evap, J_evap, num_cells);

    // Apply evaporation for one timestep
    vof.applyEvaporationMassLoss(d_J_evap, rho, dt);

    // Check mass change
    float mass_final = vof.computeTotalMass();
    float mass_change = mass_final - mass_initial;
    float mass_change_per_cell = mass_change / num_cells;

    std::cout << "  Final mass: M = " << mass_final << std::endl;
    std::cout << "  Total mass change: ΔM = " << mass_change << std::endl;
    std::cout << "  Mass change per cell: " << mass_change_per_cell << std::endl;

    // Validation: mass change per cell should match expected df
    float error = std::abs(mass_change_per_cell - df_expected) / std::abs(df_expected);
    std::cout << "  Relative error: " << error * 100.0f << "%" << std::endl;

    EXPECT_LT(error, 0.05f)
        << "Mass loss rate incorrect: error = " << error * 100.0f << "%";

    std::cout << "  ✓ Test passed (df matches analytical formula)" << std::endl;

    cudaFree(d_J_evap);
}

/**
 * @brief Test 2: Stability Limiter
 *
 * Validates that excessive evaporation flux is limited to 2% per timestep
 */
TEST_F(VOFEvaporationTest, StabilityLimiter) {
    std::cout << "\n=== VOF Evaporation: Stability Limiter ===" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const int num_cells = nx * ny * nz;

    const float rho = 4420.0f;
    const float dt = 1e-7f;

    // Extreme evaporation flux (should trigger limiter)
    const float J_evap_extreme = 50000.0f;  // Very high flux
    const float df_unlimited = -J_evap_extreme * dt / (rho * dx);

    std::cout << "  Extreme evaporation flux: J = " << J_evap_extreme << " kg/(m²·s)" << std::endl;
    std::cout << "  Unlimited df: " << df_unlimited << " (> 2% limiter)" << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(1.0f);

    float mass_initial = vof.computeTotalMass();

    // Setup evaporation flux
    float* d_J_evap;
    cudaMalloc(&d_J_evap, num_cells * sizeof(float));
    setUniformEvaporationFlux(d_J_evap, J_evap_extreme, num_cells);

    // Apply evaporation
    vof.applyEvaporationMassLoss(d_J_evap, rho, dt);

    float mass_final = vof.computeTotalMass();
    float mass_change_per_cell = (mass_final - mass_initial) / num_cells;

    std::cout << "  Actual df: " << mass_change_per_cell << std::endl;

    // Expected: limited to -2% per timestep (MAX_DF_PER_STEP = 0.02)
    const float max_df_per_step = 0.02f;
    const float expected_df_limited = -max_df_per_step * 1.0f;  // f_initial = 1.0

    std::cout << "  Expected df (limited): " << expected_df_limited << std::endl;

    // Validation: actual change should be limited
    EXPECT_GT(mass_change_per_cell, expected_df_limited * 1.1f)
        << "Limiter should prevent change < " << expected_df_limited;

    EXPECT_LT(mass_change_per_cell, expected_df_limited * 0.9f)
        << "Limiter should cap change at " << expected_df_limited;

    std::cout << "  ✓ Test passed (limiter active: df capped at 2%)" << std::endl;

    cudaFree(d_J_evap);
}

/**
 * @brief Test 3: Progressive Mass Loss
 *
 * Validates cumulative mass loss over multiple timesteps
 */
TEST_F(VOFEvaporationTest, ProgressiveMassLoss) {
    std::cout << "\n=== VOF Evaporation: Progressive Mass Loss ===" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const int num_cells = nx * ny * nz;

    const float rho = 4420.0f;
    const float J_evap = 200.0f;
    const float dt = 1e-7f;
    const int num_steps = 100;

    // Expected mass loss per step
    const float df_per_step = -J_evap * dt / (rho * dx);
    const float expected_total_df = df_per_step * num_steps;

    std::cout << "  Evaporation flux: J = " << J_evap << " kg/(m²·s)" << std::endl;
    std::cout << "  Steps: " << num_steps << std::endl;
    std::cout << "  Expected df per step: " << df_per_step << std::endl;
    std::cout << "  Expected total df: " << expected_total_df << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(1.0f);

    float mass_initial = vof.computeTotalMass();
    std::cout << "  Initial mass: M0 = " << mass_initial << std::endl;

    // Setup evaporation flux
    float* d_J_evap;
    cudaMalloc(&d_J_evap, num_cells * sizeof(float));
    setUniformEvaporationFlux(d_J_evap, J_evap, num_cells);

    // Apply evaporation for multiple steps
    for (int step = 0; step < num_steps; ++step) {
        vof.applyEvaporationMassLoss(d_J_evap, rho, dt);

        if ((step + 1) % 20 == 0) {
            float mass = vof.computeTotalMass();
            float df = (mass - mass_initial) / num_cells;
            std::cout << "    Step " << (step + 1) << ": df = " << df << std::endl;
        }
    }

    float mass_final = vof.computeTotalMass();
    float total_df = (mass_final - mass_initial) / num_cells;

    std::cout << "  Final mass: M = " << mass_final << std::endl;
    std::cout << "  Actual total df: " << total_df << std::endl;

    // Validation: total mass loss should match cumulative expectation
    float error = std::abs(total_df - expected_total_df) / std::abs(expected_total_df);
    std::cout << "  Relative error: " << error * 100.0f << "%" << std::endl;

    EXPECT_LT(error, 0.10f)
        << "Cumulative mass loss error: " << error * 100.0f << "%";

    std::cout << "  ✓ Test passed (cumulative mass loss correct)" << std::endl;

    cudaFree(d_J_evap);
}

/**
 * @brief Test 4: Top Layer Evaporation Only
 *
 * Validates that evaporation only affects cells with J_evap > 0
 */
TEST_F(VOFEvaporationTest, TopLayerEvaporationOnly) {
    std::cout << "\n=== VOF Evaporation: Top Layer Only ===" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const int num_cells = nx * ny * nz;

    const float rho = 4420.0f;
    const float J_evap = 300.0f;
    const float dt = 1e-7f;

    std::cout << "  Evaporation at top layer (z = nz-1) only" << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(1.0f);

    // Store initial fill level
    std::vector<float> h_fill_initial(num_cells);
    vof.copyFillLevelToHost(h_fill_initial.data());

    // Setup evaporation flux (top layer only)
    float* d_J_evap;
    cudaMalloc(&d_J_evap, num_cells * sizeof(float));
    setTopLayerEvaporation(d_J_evap, J_evap, nx, ny, nz);

    // Apply evaporation
    vof.applyEvaporationMassLoss(d_J_evap, rho, dt);

    // Get final fill level
    std::vector<float> h_fill_final(num_cells);
    vof.copyFillLevelToHost(h_fill_final.data());

    // Check: top layer should change, other layers should not
    int top_layer_start = nx * ny * (nz - 1);
    int top_layer_changed = 0;
    int interior_changed = 0;

    for (int i = 0; i < num_cells; ++i) {
        float df = h_fill_final[i] - h_fill_initial[i];

        if (i >= top_layer_start) {
            // Top layer: should have changed
            if (std::abs(df) > 1e-6f) {
                top_layer_changed++;
            }
        } else {
            // Interior: should not change
            if (std::abs(df) > 1e-6f) {
                interior_changed++;
            }
        }
    }

    std::cout << "  Top layer cells changed: " << top_layer_changed << " / " << (nx * ny) << std::endl;
    std::cout << "  Interior cells changed: " << interior_changed << " / " << (num_cells - nx * ny) << std::endl;

    // Validation: top layer should change, interior should not
    EXPECT_GT(top_layer_changed, nx * ny * 0.9f)
        << "Top layer should show mass loss";

    EXPECT_EQ(interior_changed, 0)
        << "Interior should not change (no evaporation flux)";

    std::cout << "  ✓ Test passed (evaporation localized to top layer)" << std::endl;

    cudaFree(d_J_evap);
}

/**
 * @brief Test 5: Zero Flux → No Change
 *
 * Validates that zero evaporation flux causes no mass change
 */
TEST_F(VOFEvaporationTest, ZeroFluxNoChange) {
    std::cout << "\n=== VOF Evaporation: Zero Flux ===" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const int num_cells = nx * ny * nz;

    const float rho = 4420.0f;
    const float dt = 1e-7f;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(1.0f);

    float mass_initial = vof.computeTotalMass();

    // Setup zero evaporation flux
    float* d_J_evap;
    cudaMalloc(&d_J_evap, num_cells * sizeof(float));
    cudaMemset(d_J_evap, 0, num_cells * sizeof(float));

    // Apply evaporation (should do nothing)
    for (int step = 0; step < 100; ++step) {
        vof.applyEvaporationMassLoss(d_J_evap, rho, dt);
    }

    float mass_final = vof.computeTotalMass();
    float mass_change = std::abs(mass_final - mass_initial);

    std::cout << "  Initial mass: M0 = " << mass_initial << std::endl;
    std::cout << "  Final mass: M = " << mass_final << std::endl;
    std::cout << "  Mass change: |ΔM| = " << mass_change << std::endl;

    // Validation: mass should be unchanged
    EXPECT_LT(mass_change, 1e-6f)
        << "Zero flux should cause no mass change";

    std::cout << "  ✓ Test passed (zero flux → no change)" << std::endl;

    cudaFree(d_J_evap);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
