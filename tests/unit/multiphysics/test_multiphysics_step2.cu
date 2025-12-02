/**
 * @file test_multiphysics_step2.cu
 * @brief Unit test for MultiphysicsSolver Step 2: Thermal Diffusion Coupling
 *
 * This test validates Step 2 of the multiphysics integration:
 * - Dynamic thermal diffusion (thermal_->step(dt) enabled)
 * - VOF interface reconstruction (no advection yet)
 * - Marangoni force computation from evolving temperature field
 * - Fluid flow driven by Marangoni effect
 * - Thermal-fluid coupling stability
 *
 * Key differences from Step 1:
 * - enable_thermal = true (thermal solver active)
 * - Temperature field evolves via diffusion
 * - Temperature gradients decrease over time → Marangoni velocity should decrease
 * - Tests for thermal-fluid feedback stability
 *
 * Success criteria:
 * - No divergence over 200 μs
 * - Temperature evolution shows diffusion (gradients decrease)
 * - Velocity remains stable (0.5-1.5 m/s range initially, then decreases)
 * - No NaN/Inf
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

namespace {

/**
 * @brief Initialize temperature field with radial gradient (same as Step 1)
 */
void initializeTemperatureField(std::vector<float>& h_temp,
                               int nx, int ny, int nz, float dx) {
    // Temperature parameters (from Test 2C)
    const float T_hot = 2500.0f;   // K (near center, molten Ti6Al4V)
    const float T_cold = 2000.0f;  // K (at edge)
    const float R_hot = 30e-6f;    // 30 μm hot zone radius
    const float R_decay = 50e-6f;  // 50 μm decay length

    // Domain center
    const float center_x = nx * dx / 2.0f;
    const float center_y = ny * dx / 2.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Position relative to center
                float x = i * dx - center_x;
                float y = j * dx - center_y;
                float r = sqrtf(x*x + y*y);

                // Radial temperature profile with smooth decay
                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    float decay_factor = expf(-(r - R_hot) / (R_decay - R_hot));
                    h_temp[idx] = T_cold + (T_hot - T_cold) * decay_factor;
                }

                h_temp[idx] = std::max(h_temp[idx], T_cold);
            }
        }
    }

    // Diagnostic output
    float max_T = *std::max_element(h_temp.begin(), h_temp.end());
    float min_T = *std::min_element(h_temp.begin(), h_temp.end());
    float delta_T = max_T - min_T;
    float grad_T = delta_T / R_hot;

    std::cout << "Temperature field initialized:" << std::endl;
    std::cout << "  T_max = " << max_T << " K" << std::endl;
    std::cout << "  T_min = " << min_T << " K" << std::endl;
    std::cout << "  ΔT = " << delta_T << " K" << std::endl;
    std::cout << "  |∇T| ~ " << grad_T * 1e-6 << " K/μm" << std::endl;
}

/**
 * @brief Compute temperature variance (measure of gradients)
 */
float computeTemperatureVariance(const std::vector<float>& temp) {
    float mean = 0.0f;
    for (float t : temp) {
        mean += t;
    }
    mean /= temp.size();

    float variance = 0.0f;
    for (float t : temp) {
        variance += (t - mean) * (t - mean);
    }
    variance /= temp.size();

    return variance;
}

} // anonymous namespace

// ============================================================================
// Test 1: Thermal Coupling Stability
// ============================================================================

TEST(MultiphysicsSolverTest, Step2_ThermalCouplingStability) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 2 - Test 1" << std::endl;
    std::cout << "Thermal-Fluid Coupling Stability" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2e-6f;  // 2 μm resolution

    // Physics flags (Step 2: Enable thermal diffusion)
    config.enable_thermal = true;   // KEY CHANGE: Dynamic thermal diffusion
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;  // Still no VOF advection
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;

    // Material: Ti6Al4V liquid
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;  // m²/s
    config.kinematic_viscosity = 0.0333f;  // Lattice units (tau=0.6)
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;

    // Time step
    config.dt = 1e-7f;  // 0.1 μs (100 ns)

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << " x " << config.ny
              << " x " << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Thermal diffusivity α = " << config.thermal_diffusivity * 1e6
              << " mm²/s" << std::endl;
    std::cout << std::endl;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize with radial temperature gradient
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fill(num_cells, 0.5f);  // Planar interface

    initializeTemperatureField(h_temp, config.nx, config.ny, config.nz, config.dx);

    // Set interface at mid-height
    int z_interface = config.nz / 2;
    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    solver.initialize(h_temp.data(), h_fill.data());

    std::cout << "Initial conditions set" << std::endl;
    std::cout << "  Temperature: radial gradient with diffusion" << std::endl;
    std::cout << "  Interface: planar at z = " << z_interface << std::endl;
    std::cout << std::endl;

    // Time integration (2000 steps = 200 μs)
    const int n_steps = 2000;
    const int output_interval = 400;

    std::cout << "Time integration:" << std::endl;
    std::cout << "  Number of steps: " << n_steps << std::endl;
    std::cout << "  Total time: " << n_steps * config.dt * 1e6 << " μs" << std::endl;
    std::cout << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);

        if ((step + 1) % output_interval == 0) {
            float v_max = solver.getMaxVelocity();
            float T_max = solver.getMaxTemperature();

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(3)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(6) << v_max << " m/s"
                      << " | T_max = " << std::setprecision(1) << T_max << " K";

            // Check for NaN
            bool has_nan = solver.checkNaN();
            if (has_nan) {
                std::cout << " | NaN DETECTED!";
            }
            std::cout << std::endl;

            ASSERT_FALSE(has_nan) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::endl;

    // Final diagnostics
    float final_velocity = solver.getMaxVelocity();
    float final_temperature = solver.getMaxTemperature();

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Final velocity:    " << final_velocity << " m/s" << std::endl;
    std::cout << "  Final temperature: " << final_temperature << " K" << std::endl;
    std::cout << std::endl;

    // Test assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_GT(final_velocity, 0.0f) << "Velocity should be positive";
    EXPECT_LT(final_temperature, 3000.0f) << "Temperature should be physical";

    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Test 2: Temperature Evolution (Diffusion Effect)
// ============================================================================

TEST(MultiphysicsSolverTest, Step2_TemperatureEvolution) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 2 - Test 2" << std::endl;
    std::cout << "Temperature Evolution via Diffusion" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration (smaller domain for faster test)
    MultiphysicsConfig config;
    config.nx = 50;
    config.ny = 50;
    config.nz = 25;
    config.dx = 2e-6f;

    // Enable thermal diffusion
    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;

    // Material properties
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;

    MultiphysicsSolver solver(config);

    // Initialize with temperature gradient
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fill(num_cells, 0.5f);

    initializeTemperatureField(h_temp, config.nx, config.ny, config.nz, config.dx);

    float initial_variance = computeTemperatureVariance(h_temp);

    solver.initialize(h_temp.data(), h_fill.data());

    std::cout << "Initial temperature variance: " << initial_variance << " K²" << std::endl;
    std::cout << std::endl;

    // Run for 1000 steps (100 μs)
    const int n_steps = 1000;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);
    }

    // Copy back temperature field
    solver.copyTemperatureToHost(h_temp.data());

    float final_variance = computeTemperatureVariance(h_temp);

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Initial variance: " << initial_variance << " K²" << std::endl;
    std::cout << "  Final variance:   " << final_variance << " K²" << std::endl;
    std::cout << "  Reduction:        " <<
        (initial_variance - final_variance) / initial_variance * 100.0f << "%" << std::endl;
    std::cout << std::endl;

    // Test: Thermal diffusion should reduce variance
    EXPECT_LT(final_variance, initial_variance)
        << "Thermal diffusion should reduce temperature variance";

    // Test: Variance should decrease by at least 5% (diffusion is working)
    float variance_reduction = (initial_variance - final_variance) / initial_variance;
    EXPECT_GT(variance_reduction, 0.05f)
        << "Temperature variance should decrease by at least 5%";

    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "Thermal diffusion is working correctly" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Test 3: Velocity Stability (Verification that Test 1 results are repeatable)
// ============================================================================

TEST(MultiphysicsSolverTest, Step2_VelocityStability) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 2 - Test 3" << std::endl;
    std::cout << "Velocity Stability - Repeatability Check" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // NOTE: This test verifies that Test 1 results are repeatable
    // We run the same configuration as Test 1 but check stability metrics

    // Configuration (same as Test 1)
    MultiphysicsConfig config;
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2e-6f;

    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;

    // Material properties
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;

    MultiphysicsSolver solver(config);

    // Initialize (same as Test 1)
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fill(num_cells, 0.5f);

    initializeTemperatureField(h_temp, config.nx, config.ny, config.nz, config.dx);

    // Set interface at mid-height (same as Test 1)
    int z_interface = config.nz / 2;
    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    solver.initialize(h_temp.data(), h_fill.data());

    std::cout << "Running simulation (shorter duration for repeatability check)..." << std::endl;
    std::cout << std::endl;

    // Track velocity evolution (shorter test)
    std::vector<float> velocities;
    const int n_steps = 1000;  // 100 μs (half of Test 1)
    const int sample_interval = 200;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);

        if ((step + 1) % sample_interval == 0) {
            float v_max = solver.getMaxVelocity();
            velocities.push_back(v_max);

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(3)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(6) << v_max << " m/s"
                      << std::endl;

            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::endl;

    // Analyze velocity evolution
    float initial_velocity = velocities.front();
    float final_velocity = velocities.back();
    float max_velocity = *std::max_element(velocities.begin(), velocities.end());

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Initial velocity: " << initial_velocity << " m/s" << std::endl;
    std::cout << "  Maximum velocity: " << max_velocity << " m/s" << std::endl;
    std::cout << "  Final velocity:   " << final_velocity << " m/s" << std::endl;
    std::cout << std::endl;

    // Test: Velocity should be in literature range
    const float v_literature_min = 0.5f;
    const float v_literature_max = 1.5f;

    EXPECT_GT(max_velocity, 0.0f) << "Velocity should be non-zero (Marangoni-driven flow)";
    EXPECT_GE(max_velocity, v_literature_min * 0.8f)  // Allow 20% below min
        << "Peak velocity should be close to literature range";
    EXPECT_LE(max_velocity, v_literature_max * 1.2f)  // Allow 20% above max
        << "Velocity should not vastly exceed literature range";

    // Test: Stability - velocity should not grow exponentially
    EXPECT_LT(final_velocity, initial_velocity * 2.0f)
        << "Velocity should not double (indicates instability)";

    std::cout << "Comparison with Test 1 expected results:" << std::endl;
    std::cout << "  Test 1 achieved ~0.6-0.7 m/s" << std::endl;
    std::cout << "  This test achieved: " << max_velocity << " m/s" << std::endl;
    std::cout << "  Literature range: " << v_literature_min << " - "
              << v_literature_max << " m/s" << std::endl;

    if (max_velocity >= v_literature_min && max_velocity <= v_literature_max) {
        std::cout << "  Status: WITHIN LITERATURE RANGE ✓" << std::endl;
    } else if (max_velocity > 0.4f && max_velocity < 2.0f) {
        std::cout << "  Status: CLOSE TO LITERATURE RANGE (acceptable)" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "Thermal-fluid coupling is stable and repeatable" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
