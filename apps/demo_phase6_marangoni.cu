/**
 * @file test_multiphysics_step1.cu
 * @brief Unit test for MultiphysicsSolver Step 1: Static Temperature + Marangoni
 *
 * This test validates Step 1 of the multiphysics integration:
 * - Static temperature field (no thermal diffusion)
 * - VOF interface reconstruction (no advection)
 * - Marangoni force computation
 * - Fluid flow driven by Marangoni effect
 *
 * Success criteria:
 * - Reproduce Test 2C result: surface velocity ~ 0.768 m/s
 * - Acceptable range: 0.5-1.5 m/s (literature values)
 * - No NaN/Inf
 * - Stable for 1000 time steps
 *
 * This test demonstrates that the MultiphysicsSolver correctly integrates
 * the basic Thermal-Marangoni-Fluid coupling without the complexity of
 * dynamic thermal diffusion or VOF advection.
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
 * @brief Initialize static temperature field with radial gradient
 *
 * This matches Test 2C setup: hot center, cold edge
 * Temperature gradient: ~1.67×10⁷ K/m
 */
void initializeTemperatureField(float* d_temperature, int nx, int ny, int nz, float dx) {
    std::vector<float> h_temp(nx * ny * nz);

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

    // Copy to device
    cudaMemcpy(d_temperature, h_temp.data(), nx * ny * nz * sizeof(float),
               cudaMemcpyHostToDevice);

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

} // anonymous namespace

// ============================================================================
// Test: Step 1 - Static Temperature + Marangoni Velocity
// ============================================================================

TEST(MultiphysicsSolverTest, Step1_StaticTemperatureMarangoniVelocity) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 1" << std::endl;
    std::cout << "Static Temperature + Marangoni Velocity" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration (matches Test 2C)
    MultiphysicsConfig config;
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2e-6f;  // 2 μm resolution

    // Physics flags (Step 1)
    config.enable_thermal = false;  // Static temperature
    config.enable_fluid = true;
    config.enable_vof = true;  // Enable interface reconstruction
    config.enable_vof_advection = false;  // NO advection in Step 1
    config.enable_surface_tension = false;  // Not in Step 1
    config.enable_marangoni = true;
    config.enable_laser = false;  // Not in Step 1

    // Material: Ti6Al4V liquid
    // Note: Use LATTICE viscosity for LBM stability (matches Test 2C)
    // tau = 0.6 → nu_lattice = 0.0333
    config.kinematic_viscosity = 0.0333f;  // Lattice units (tau=0.6)
    config.density = 4110.0f;  // kg/m³
    config.dsigma_dT = -0.26e-3f;  // N/(m·K)

    // Time step (match Test 2C)
    config.dt = 1e-7f;  // 0.1 μs (100 ns)

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << " x " << config.ny
              << " x " << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Material: Ti6Al4V liquid" << std::endl;
    std::cout << "    ν = " << config.kinematic_viscosity * 1e6 << " mm²/s" << std::endl;
    std::cout << "    ρ = " << config.density << " kg/m³" << std::endl;
    std::cout << "    dσ/dT = " << config.dsigma_dT * 1e3 << " mN/(m·K)" << std::endl;
    std::cout << std::endl;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize with planar interface at mid-height
    solver.initialize(300.0f, 0.5f);

    // Set static temperature field
    int num_cells = config.nx * config.ny * config.nz;
    float* d_temp_static;
    cudaMalloc(&d_temp_static, num_cells * sizeof(float));

    initializeTemperatureField(d_temp_static, config.nx, config.ny, config.nz, config.dx);
    solver.setStaticTemperature(d_temp_static);

    std::cout << "Initial conditions set" << std::endl;
    std::cout << "  Temperature: static radial gradient" << std::endl;
    std::cout << "  Interface: planar at z = " << 0.5f * config.nz << std::endl;
    std::cout << std::endl;

    // Time integration (match Test 2C)
    const int n_steps = 2000;  // 200 μs total
    const int output_interval = 400;  // Output every 40 μs

    std::cout << "Time integration:" << std::endl;
    std::cout << "  Number of steps: " << n_steps << std::endl;
    std::cout << "  Total time: " << n_steps * config.dt * 1e6 << " μs" << std::endl;
    std::cout << std::endl;

    float max_velocity = 0.0f;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);

        if ((step + 1) % output_interval == 0) {
            float v_max = solver.getMaxVelocity();
            max_velocity = std::max(max_velocity, v_max);

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(3)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(6) << v_max << " m/s";

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
    max_velocity = std::max(max_velocity, final_velocity);

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Maximum velocity: " << max_velocity << " m/s" << std::endl;
    std::cout << "  Final velocity:   " << final_velocity << " m/s" << std::endl;
    std::cout << std::endl;

    // Validation against Test 2C
    const float v_test2c = 0.768f;  // m/s (Test 2C result)
    const float v_literature_min = 0.5f;   // m/s (Khairallah et al.)
    const float v_literature_max = 1.5f;   // m/s (upper bound)

    std::cout << "Comparison with Test 2C:" << std::endl;
    std::cout << "  Test 2C result:    " << v_test2c << " m/s" << std::endl;
    std::cout << "  This test:         " << max_velocity << " m/s" << std::endl;
    std::cout << "  Difference:        " << std::abs(max_velocity - v_test2c)
              << " m/s (" << std::abs(max_velocity - v_test2c) / v_test2c * 100.0f
              << "%)" << std::endl;
    std::cout << std::endl;

    std::cout << "Literature range (Khairallah 2016):" << std::endl;
    std::cout << "  Expected: " << v_literature_min << " - " << v_literature_max
              << " m/s" << std::endl;

    if (max_velocity >= v_literature_min && max_velocity <= v_literature_max) {
        std::cout << "  Status: WITHIN LITERATURE RANGE ✓" << std::endl;
    } else {
        std::cout << "  Status: OUTSIDE LITERATURE RANGE ✗" << std::endl;
    }
    std::cout << std::endl;

    // Test assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_GT(max_velocity, 0.0f) << "Velocity should be positive (Marangoni-driven flow)";

    // CRITICAL: Velocity must be in literature range
    EXPECT_GE(max_velocity, v_literature_min)
        << "Velocity too low (< " << v_literature_min << " m/s)";
    EXPECT_LE(max_velocity, v_literature_max)
        << "Velocity too high (> " << v_literature_max << " m/s)";

    // Verify close to Test 2C result (within 30%)
    float relative_error = std::abs(max_velocity - v_test2c) / v_test2c;
    EXPECT_LT(relative_error, 0.3f)
        << "Velocity differs from Test 2C by more than 30%";

    if (relative_error < 0.1f) {
        std::cout << "EXCELLENT: Matches Test 2C within 10%" << std::endl;
    } else if (relative_error < 0.3f) {
        std::cout << "GOOD: Matches Test 2C within 30%" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;

    cudaFree(d_temp_static);
}

// ============================================================================
// Test: Configuration Validation
// ============================================================================

TEST(MultiphysicsSolverTest, ConfigurationValidation) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Configuration Validation" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MultiphysicsConfig config;

    // Valid configuration should not throw
    EXPECT_NO_THROW({
        MultiphysicsSolver solver(config);
    });

    // Invalid grid dimensions
    config.nx = -1;
    EXPECT_THROW({
        MultiphysicsSolver solver(config);
    }, std::invalid_argument);

    config.nx = 100;
    config.dx = -1e-6f;
    EXPECT_THROW({
        MultiphysicsSolver solver(config);
    }, std::invalid_argument);

    std::cout << "Configuration validation tests passed ✓" << std::endl;
}

// ============================================================================
// Test: Initialization
// ============================================================================

TEST(MultiphysicsSolverTest, Initialization) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Initialization" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MultiphysicsConfig config;
    config.nx = 50;
    config.ny = 50;
    config.nz = 25;

    MultiphysicsSolver solver(config);

    // Initialize with uniform temperature
    float T_init = 300.0f;
    solver.initialize(T_init, 0.5f);

    // Verify temperature field
    float T_max = solver.getMaxTemperature();
    EXPECT_FLOAT_EQ(T_max, T_init) << "Maximum temperature should match initial value";

    // Verify no NaN
    EXPECT_FALSE(solver.checkNaN()) << "No NaN should be present after initialization";

    // Verify initial velocity is zero
    float v_init = solver.getMaxVelocity();
    EXPECT_FLOAT_EQ(v_init, 0.0f) << "Initial velocity should be zero";

    std::cout << "Initialization tests passed ✓" << std::endl;
    std::cout << "  Initial temperature: " << T_max << " K" << std::endl;
    std::cout << "  Initial velocity: " << v_init << " m/s" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
