/**
 * @file test_multiphysics_step3_FIXED.cu
 * @brief FIXED VERSION: Unit test for MultiphysicsSolver Step 3
 *
 * KEY FIX: Initialize with non-uniform temperature to create Marangoni driving force
 *
 * ORIGINAL BUG:
 * - Tests initialized with uniform temperature: solver.initialize(2300.0f, 0.5f)
 * - Result: ∇T = 0 everywhere → F_Marangoni = 0 → no motion
 *
 * FIX APPLIED:
 * - Initialize with spatial temperature gradient (hot center, cold edge)
 * - Result: ∇T ≠ 0 → F_Marangoni ≠ 0 → generates flow
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

// ============================================================================
// Helper Function: Initialize Temperature Gradient
// ============================================================================

/**
 * @brief Create radial temperature gradient (hot center, cold edge)
 *
 * This produces a realistic temperature field for Marangoni convection tests:
 * - Center: T_hot = 2500 K (molten metal)
 * - Edge: T_cold = 2000 K (still liquid, but cooler)
 * - Gradient: Creates strong tangential ∇T at interface
 *
 * @param h_temp Output temperature array (host)
 * @param nx, ny, nz Grid dimensions
 * @param dx Cell size [m]
 */
void initializeRadialTemperatureGradient(
    std::vector<float>& h_temp,
    int nx, int ny, int nz, float dx)
{
    const float T_hot = 2500.0f;   // K (center)
    const float T_cold = 2000.0f;  // K (edge)
    const float R_hot = 15.0f * dx;  // Hot zone radius (15 cells)
    const float R_decay = 25.0f * dx;  // Decay length (25 cells)

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

                // Radial temperature profile with smooth exponential decay
                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    float decay_factor = expf(-(r - R_hot) / (R_decay - R_hot));
                    h_temp[idx] = T_cold + (T_hot - T_cold) * decay_factor;
                }

                // Clamp to minimum temperature
                h_temp[idx] = std::max(h_temp[idx], T_cold);
            }
        }
    }
}

// ============================================================================
// Test 3: Surface Deformation (FIXED)
// ============================================================================

TEST(MultiphysicsSolverTest, Step3_SurfaceDeformation_FIXED) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 3 - Test 3 (FIXED)" << std::endl;
    std::cout << "Surface Deformation with Flow" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;

    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;  // VOF advection enabled
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;

    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;
    config.vof_subcycles = 10;

    MultiphysicsSolver solver(config);

    // ========================================================================
    // FIX APPLIED: Initialize with temperature gradient
    // ========================================================================
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);

    initializeRadialTemperatureGradient(h_temp, config.nx, config.ny, config.nz, config.dx);

    // Copy temperature to device and set in solver
    float* d_temp;
    cudaMalloc(&d_temp, num_cells * sizeof(float));
    cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize solver with gradient temperature (NOT uniform!)
    solver.initialize(2300.0f, 0.5f);  // This sets default, then we override:
    solver.setStaticTemperature(d_temp);

    std::cout << "Temperature field:" << std::endl;
    float T_max = *std::max_element(h_temp.begin(), h_temp.end());
    float T_min = *std::min_element(h_temp.begin(), h_temp.end());
    std::cout << "  T_max = " << T_max << " K (center)" << std::endl;
    std::cout << "  T_min = " << T_min << " K (edge)" << std::endl;
    std::cout << "  ΔT = " << (T_max - T_min) << " K" << std::endl;
    std::cout << "  → Temperature gradient present: ∇T ≠ 0 ✓" << std::endl;
    std::cout << std::endl;
    // ========================================================================

    // Copy initial fill level
    std::vector<float> h_fill_initial(num_cells);
    solver.copyFillLevelToHost(h_fill_initial.data());

    // Run simulation
    const int n_steps = 500;
    std::cout << "Running " << n_steps << " steps..." << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);

        if ((step + 1) % 100 == 0) {
            float v_max = solver.getMaxVelocity();
            std::cout << "  Step " << (step + 1) << ": v_max = " << v_max << " m/s" << std::endl;
        }
    }

    // Copy final fill level
    std::vector<float> h_fill_final(num_cells);
    solver.copyFillLevelToHost(h_fill_final.data());

    // Compute interface change
    float max_change = 0.0f;
    int n_changed = 0;

    for (int i = 0; i < num_cells; ++i) {
        float change = std::abs(h_fill_final[i] - h_fill_initial[i]);
        max_change = std::max(max_change, change);
        if (change > 0.01f) {
            n_changed++;
        }
    }

    float fraction_changed = static_cast<float>(n_changed) / num_cells;

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Max fill level change: " << max_change << std::endl;
    std::cout << "  Cells changed > 1%: " << n_changed << " ("
              << fraction_changed * 100.0f << "%)" << std::endl;
    std::cout << "  Final velocity: " << solver.getMaxVelocity() << " m/s" << std::endl;
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_temp);

    // Test: Surface should deform (change > 0)
    EXPECT_GT(max_change, 0.0f) << "Interface should move with VOF advection and Marangoni flow";
    EXPECT_GT(n_changed, 0) << "Some cells should show interface movement";

    // NEW: Verify velocity is generated
    float final_v = solver.getMaxVelocity();
    EXPECT_GT(final_v, 0.01f) << "Marangoni forces should generate velocity";

    if (max_change > 0.0f && final_v > 0.01f) {
        std::cout << "✓ TEST PASSED - Interface deforms with fluid flow" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED - No interface motion detected" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Test 4: Force Balance (FIXED)
// ============================================================================

TEST(MultiphysicsSolverTest, Step3_ForceBalance_FIXED) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 3 - Test 4 (FIXED)" << std::endl;
    std::cout << "Force Balance: Marangoni + Surface Tension" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;

    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_surface_tension = true;  // Surface tension ON
    config.enable_marangoni = true;
    config.enable_laser = false;

    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.surface_tension_coeff = 1.65f;  // Ti6Al4V liquid-gas
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;
    config.vof_subcycles = 10;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Marangoni:       ENABLED" << std::endl;
    std::cout << "  Surface tension: ENABLED" << std::endl;
    std::cout << "  σ = " << config.surface_tension_coeff << " N/m" << std::endl;
    std::cout << "  dσ/dT = " << config.dsigma_dT * 1e3 << " mN/(m·K)" << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);

    // ========================================================================
    // FIX APPLIED: Initialize with temperature gradient
    // ========================================================================
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);

    initializeRadialTemperatureGradient(h_temp, config.nx, config.ny, config.nz, config.dx);

    // Set temperature field
    float* d_temp;
    cudaMalloc(&d_temp, num_cells * sizeof(float));
    cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    solver.initialize(2300.0f, 0.5f);
    solver.setStaticTemperature(d_temp);

    std::cout << "Temperature field: radial gradient (ΔT = 500 K)" << std::endl;
    std::cout << std::endl;
    // ========================================================================

    // Run simulation
    const int n_steps = 1000;
    const int output_interval = 200;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);

        if ((step + 1) % output_interval == 0) {
            float v_max = solver.getMaxVelocity();
            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(3)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(6) << v_max << " m/s"
                      << std::endl;

            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    float final_velocity = solver.getMaxVelocity();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Final velocity: " << final_velocity << " m/s" << std::endl;
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_temp);

    // Test: Simulation should remain stable with both forces
    EXPECT_FALSE(solver.checkNaN()) << "No NaN with combined forces";
    EXPECT_GT(final_velocity, 0.1f) << "Velocity should be significant with Marangoni forces";
    EXPECT_LT(final_velocity, 5.0f) << "Velocity should be physically reasonable";

    if (final_velocity > 0.1f && final_velocity < 5.0f) {
        std::cout << "✓ TEST PASSED - Marangoni + Surface Tension forces balanced" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED - Velocity out of expected range" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
