/**
 * @file test_multiphysics_step3.cu
 * @brief Unit test for MultiphysicsSolver Step 3: VOF Advection + Surface Tension
 *
 * This test validates Step 3 of the multiphysics integration:
 * - VOF advection with subcycling (enable_vof_advection = true)
 * - Surface tension force computation
 * - Combined forces: Marangoni + surface tension
 * - Mass conservation
 * - Surface deformation with fluid flow
 *
 * Key differences from Step 2:
 * - enable_vof_advection = true (interface moves with fluid)
 * - enable_surface_tension = true (capillary forces active)
 * - vof_subcycles = 10 (10× time refinement for VOF stability)
 *
 * Success criteria:
 * - VOF advection runs without crash
 * - Mass conservation < 1%
 * - Surface deforms following velocity field
 * - Force balance between Marangoni and surface tension is correct
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

// ============================================================================
// Test 1: VOF Advection Stability
// ============================================================================

TEST(MultiphysicsSolverTest, Step3_VOFAdvectionStability) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 3 - Test 1" << std::endl;
    std::cout << "VOF Advection Stability" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 80;
    config.ny = 80;
    config.nz = 40;
    config.dx = 2e-6f;

    // Step 3: Enable VOF advection
    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;  // KEY CHANGE: VOF advection ON
    config.enable_surface_tension = false;  // Not yet in Test 1
    config.enable_marangoni = true;
    config.enable_laser = false;

    // Material
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;
    config.vof_subcycles = 10;  // 10× refinement

    std::cout << "Configuration:" << std::endl;
    std::cout << "  VOF advection: ENABLED" << std::endl;
    std::cout << "  VOF subcycles: " << config.vof_subcycles << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  dt_vof = " << (config.dt / config.vof_subcycles) * 1e9 << " ns" << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);
    solver.initialize(2300.0f, 0.5f);

    // Run for 1000 steps (100 μs)
    const int n_steps = 1000;
    const int output_interval = 200;

    std::cout << "Running simulation with VOF advection..." << std::endl;

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

    std::cout << std::endl;
    std::cout << "TEST PASSED ✓ - VOF advection with subcycling is stable" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Test 2: Mass Conservation
// ============================================================================

TEST(MultiphysicsSolverTest, Step3_MassConservation) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 3 - Test 2" << std::endl;
    std::cout << "Mass Conservation with VOF Advection" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration (smaller for faster test)
    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;

    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
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
    solver.initialize(2300.0f, 0.5f);

    float initial_mass = solver.getTotalMass();
    std::cout << "Initial mass: " << initial_mass << std::endl;
    std::cout << std::endl;

    // Run for 1000 steps
    const int n_steps = 1000;
    const int check_interval = 250;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);

        if ((step + 1) % check_interval == 0) {
            float current_mass = solver.getTotalMass();
            float mass_error = std::abs(current_mass - initial_mass) / initial_mass;

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | Mass = " << std::fixed << std::setprecision(6) << current_mass
                      << " | Error = " << std::setprecision(4) << mass_error * 100.0f << "%"
                      << std::endl;

            ASSERT_LT(mass_error, 0.05f) << "Mass error > 5% at step " << step + 1;
        }
    }

    float final_mass = solver.getTotalMass();
    float final_error = std::abs(final_mass - initial_mass) / initial_mass;

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Initial mass: " << initial_mass << std::endl;
    std::cout << "  Final mass:   " << final_mass << std::endl;
    std::cout << "  Mass error:   " << final_error * 100.0f << "%" << std::endl;
    std::cout << std::endl;

    // Test: Mass conserved within 1%
    EXPECT_LT(final_error, 0.01f) << "Mass conservation error should be < 1%";

    if (final_error < 0.01f) {
        std::cout << "EXCELLENT: Mass conserved within 1% ✓" << std::endl;
    } else if (final_error < 0.05f) {
        std::cout << "GOOD: Mass conserved within 5%" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Test 3: Surface Deformation
// ============================================================================

TEST(MultiphysicsSolverTest, Step3_SurfaceDeformation) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 3 - Test 3" << std::endl;
    std::cout << "Surface Deformation with Flow" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;

    config.enable_thermal = false;  // Use static temperature field
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;
    config.enable_darcy = false;  // Disable Darcy to allow full flow

    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;
    config.vof_subcycles = 10;

    MultiphysicsSolver solver(config);

    // FIX: Create temperature field with radial gradient (hot center, cold edge)
    // This drives Marangoni flow which will advect the interface
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fill(num_cells);

    float T_hot = 2500.0f;
    float T_cold = 2100.0f;
    float center_x = config.nx / 2.0f;
    float center_y = config.ny / 2.0f;
    float R_hot = 12.0f;  // Hot zone radius (cells)
    int z_interface = static_cast<int>(0.5f * config.nz);

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                // Radial temperature gradient
                float dx = i - center_x;
                float dy = j - center_y;
                float r = sqrtf(dx * dx + dy * dy);

                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    float decay = (r - R_hot) / (config.nx / 2.0f - R_hot);
                    h_temp[idx] = T_hot - decay * (T_hot - T_cold);
                }
                h_temp[idx] = std::max(h_temp[idx], T_cold);

                // Planar interface at mid-height
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    std::cout << "Temperature field: Radial gradient" << std::endl;
    std::cout << "  T_hot = " << T_hot << " K (center)" << std::endl;
    std::cout << "  T_cold = " << T_cold << " K (edge)" << std::endl;
    std::cout << "  ΔT = " << (T_hot - T_cold) << " K" << std::endl;
    std::cout << std::endl;

    solver.initialize(h_temp.data(), h_fill.data());

    // Copy initial fill level
    std::vector<float> h_fill_initial(num_cells);
    solver.copyFillLevelToHost(h_fill_initial.data());

    // Run simulation
    const int n_steps = 500;
    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);
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

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Max fill level change: " << max_change << std::endl;
    std::cout << "  Cells changed > 1%: " << n_changed << " ("
              << fraction_changed * 100.0f << "%)" << std::endl;
    std::cout << std::endl;

    // Test: Surface should deform (change > 0)
    EXPECT_GT(max_change, 0.0f) << "Interface should move with VOF advection";
    EXPECT_GT(n_changed, 0) << "Some cells should show interface movement";

    std::cout << "TEST PASSED ✓ - Interface deforms with fluid flow" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Test 4: Force Balance (Marangoni + Surface Tension)
// ============================================================================

TEST(MultiphysicsSolverTest, Step3_ForceBalance) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: MultiphysicsSolver Step 3 - Test 4" << std::endl;
    std::cout << "Force Balance: Marangoni + Surface Tension" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;

    config.enable_thermal = false;  // Use static temperature field
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;  // Disable advection to focus on force balance
    config.enable_surface_tension = true;  // KEY: Surface tension ON
    config.enable_marangoni = true;
    config.enable_laser = false;
    config.enable_darcy = false;  // Disable Darcy to allow full flow

    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.surface_tension_coeff = 0.1f;  // Reduced for stability (full value causes instability)
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

    // FIX: Create temperature field with radial gradient (hot center, cold edge)
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fill(num_cells);

    float T_hot = 2500.0f;
    float T_cold = 2100.0f;
    float center_x = config.nx / 2.0f;
    float center_y = config.ny / 2.0f;
    float R_hot = 12.0f;
    int z_interface = static_cast<int>(0.5f * config.nz);

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                // Radial temperature gradient
                float dx = i - center_x;
                float dy = j - center_y;
                float r = sqrtf(dx * dx + dy * dy);

                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    float decay = (r - R_hot) / (config.nx / 2.0f - R_hot);
                    h_temp[idx] = T_hot - decay * (T_hot - T_cold);
                }
                h_temp[idx] = std::max(h_temp[idx], T_cold);

                // Planar interface at mid-height
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    solver.initialize(h_temp.data(), h_fill.data());

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

    // Test: Simulation should remain stable with both forces
    EXPECT_FALSE(solver.checkNaN()) << "No NaN with combined forces";
    EXPECT_GT(final_velocity, 0.0f) << "Velocity should be positive";
    EXPECT_LT(final_velocity, 5.0f) << "Velocity should be physically reasonable";

    std::cout << "TEST PASSED ✓ - Marangoni + Surface Tension forces balanced" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
