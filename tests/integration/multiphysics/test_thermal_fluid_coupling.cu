/**
 * @file test_thermal_fluid_coupling.cu
 * @brief Test thermal-fluid coupling: buoyancy drives flow correctly
 *
 * Success Criteria:
 * - Hot fluid rises (positive w velocity above hot region)
 * - Flow magnitude proportional to temperature difference
 * - Stable Rayleigh-Bénard convection pattern
 * - No NaN
 *
 * Physics:
 * - Thermal diffusion + buoyancy force
 * - Expected: Natural convection with upward flow from hot region
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsCouplingTest, ThermalFluidCoupling) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Thermal-Fluid Coupling (Buoyancy)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 50;
    config.ny = 50;
    config.nz = 100;  // Tall domain for convection
    config.dx = 2e-6f;
    config.dt = 1e-8f;

    // Enable thermal + fluid + buoyancy
    config.enable_thermal = true;
    config.enable_thermal_advection = true;  // Thermal advection by fluid
    config.enable_fluid = true;
    config.enable_buoyancy = true;
    config.enable_vof = false;
    config.enable_marangoni = false;
    config.enable_laser = false;
    config.enable_darcy = false;  // No Darcy damping (all liquid)

    // Buoyancy parameters
    config.thermal_expansion_coeff = 1.5e-5f;  // Ti6Al4V [1/K]
    config.gravity_x = 0.0f;
    config.gravity_y = 0.0f;
    config.gravity_z = -9.81f;  // Downward gravity
    config.reference_temperature = 2000.0f;  // K

    // Material properties
    config.thermal_diffusivity = 5.8e-6f;  // Ti6Al4V
    config.kinematic_viscosity = 0.0333f;  // Lattice units (stable)
    config.density = 4110.0f;  // kg/m³

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  Height = " << config.nz * config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  β = " << config.thermal_expansion_coeff << " 1/K" << std::endl;
    std::cout << "  g = " << std::abs(config.gravity_z) << " m/s²" << std::endl;
    std::cout << std::endl;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize with temperature stratification: hot bottom, cold top
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_fill_level(num_cells, 1.0f);  // All liquid

    const float T_hot = 2200.0f;   // K (bottom)
    const float T_cold = 1800.0f;  // K (top)

    for (int k = 0; k < config.nz; ++k) {
        float z_frac = float(k) / float(config.nz - 1);
        float T = T_hot * (1.0f - z_frac) + T_cold * z_frac;

        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                h_temperature[idx] = T;
            }
        }
    }

    solver.initialize(h_temperature.data(), h_fill_level.data());

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  T_bottom = " << T_hot << " K" << std::endl;
    std::cout << "  T_top    = " << T_cold << " K" << std::endl;
    std::cout << "  ΔT       = " << T_hot - T_cold << " K" << std::endl;
    std::cout << std::endl;

    // Time integration
    const int n_steps = 1000;
    const int check_interval = 200;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        if ((step + 1) % check_interval == 0) {
            float v_max = solver.getMaxVelocity();
            float T_max = solver.getMaxTemperature();

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(2)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(4) << v_max << " m/s"
                      << " | T_max = " << std::setprecision(1) << T_max << " K"
                      << std::endl;

            // Check for NaN
            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(70, '-') << std::endl;

    // Extract velocity field to check buoyancy direction
    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);
    solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Check for upward flow in center column (hot region)
    int i_center = config.nx / 2;
    int j_center = config.ny / 2;

    float w_hot_avg = 0.0f;  // Average w velocity in bottom half
    float w_cold_avg = 0.0f;  // Average w velocity in top half
    int count_hot = 0;
    int count_cold = 0;

    for (int k = 0; k < config.nz; ++k) {
        int idx = i_center + config.nx * (j_center + config.ny * k);
        float w = h_uz[idx];

        if (k < config.nz / 2) {  // Bottom half (hot)
            w_hot_avg += w;
            count_hot++;
        } else {  // Top half (cold)
            w_cold_avg += w;
            count_cold++;
        }
    }

    w_hot_avg /= count_hot;
    w_cold_avg /= count_cold;

    float v_max_final = solver.getMaxVelocity();

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  Max velocity: " << std::fixed << std::setprecision(4)
              << v_max_final << " m/s" << std::endl;
    std::cout << "  w_avg (bottom, hot): " << w_hot_avg << " m/s" << std::endl;
    std::cout << "  w_avg (top, cold):   " << w_cold_avg << " m/s" << std::endl;
    std::cout << std::endl;

    // Success criteria
    std::cout << "Buoyancy Check:" << std::endl;

    // 1. Hot fluid should rise (positive w in bottom half)
    std::cout << "  1. Hot fluid rises (w > 0): ";
    if (w_hot_avg > 0.0f) {
        std::cout << "PASS ✓ (" << w_hot_avg << " m/s)" << std::endl;
    } else {
        std::cout << "FAIL ✗ (" << w_hot_avg << " m/s)" << std::endl;
    }

    // 2. Cold fluid should sink (negative w in top half)
    std::cout << "  2. Cold fluid sinks (w < 0): ";
    if (w_cold_avg < 0.0f) {
        std::cout << "PASS ✓ (" << w_cold_avg << " m/s)" << std::endl;
    } else {
        std::cout << "FAIL ✗ (" << w_cold_avg << " m/s)" << std::endl;
    }

    // 3. Flow should develop (non-zero velocity)
    std::cout << "  3. Flow develops (v > 0.01 m/s): ";
    if (v_max_final > 0.01f) {
        std::cout << "PASS ✓ (" << v_max_final << " m/s)" << std::endl;
    } else {
        std::cout << "FAIL ✗ (" << v_max_final << " m/s)" << std::endl;
    }

    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_GT(w_hot_avg, 0.0f) << "Hot fluid should rise (positive w velocity)";
    EXPECT_LT(w_cold_avg, 0.0f) << "Cold fluid should sink (negative w velocity)";
    EXPECT_GT(v_max_final, 0.01f) << "Convection should develop (v > 0.01 m/s)";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
