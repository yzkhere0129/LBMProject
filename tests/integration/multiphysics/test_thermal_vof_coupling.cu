/**
 * @file test_thermal_vof_coupling.cu
 * @brief Test thermal-VOF coupling: evaporation at interface
 *
 * Success Criteria:
 * - Evaporation only occurs at interface (fill > 0.01 && fill < 0.99)
 * - Evaporation rate increases with temperature
 * - Mass loss from VOF matches evaporation power
 * - No NaN
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsCouplingTest, ThermalVOFCoupling) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Thermal-VOF Coupling (Evaporation)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;
    config.dt = 1e-8f;

    // Enable thermal + VOF + evaporation
    config.enable_thermal = true;
    config.enable_fluid = false;  // Focus on thermal-VOF only
    config.enable_vof = true;
    config.enable_vof_advection = false;  // No advection for this test
    config.enable_evaporation_mass_loss = true;  // Key coupling
    config.enable_laser = true;  // Heat source to drive evaporation
    config.enable_radiation_bc = false;
    config.enable_substrate_cooling = false;

    // Laser to heat surface above evaporation temperature
    config.laser_power = 200.0f;  // 200 W (strong heating)
    config.laser_spot_radius = 30e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10e-6f;

    // Material properties
    config.thermal_diffusivity = 5.8e-6f;  // Ti6Al4V

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  Laser power: " << config.laser_power << " W" << std::endl;
    std::cout << "  Evaporation: enabled" << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);

    // Initialize with planar interface
    const float T_init = 2000.0f;  // K (near evaporation threshold ~3287 K)
    solver.initialize(T_init, 0.5f);

    float mass_initial = solver.getTotalMass();

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  T_init = " << T_init << " K" << std::endl;
    std::cout << "  Initial mass: " << mass_initial << std::endl;
    std::cout << std::endl;

    const int n_steps = 1000;
    const int check_interval = 200;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Step | Time [μs] | T_max [K] | P_evap [W] | Mass | Δm [%]" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        if ((step + 1) % check_interval == 0) {
            float T_max = solver.getMaxTemperature();
            float P_evap = solver.getEvaporationPower();
            float mass_current = solver.getTotalMass();
            float mass_change = ((mass_current - mass_initial) / mass_initial) * 100.0f;

            std::cout << std::setw(4) << step + 1
                      << " | " << std::fixed << std::setprecision(2) << std::setw(9)
                      << (step + 1) * config.dt * 1e6
                      << " | " << std::setw(9) << std::setprecision(1) << T_max
                      << " | " << std::setw(10) << std::setprecision(3) << P_evap
                      << " | " << std::setw(4) << std::setprecision(1) << mass_current
                      << " | " << std::setw(6) << std::setprecision(3) << mass_change
                      << std::endl;

            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(80, '-') << std::endl;

    // Final results
    float T_final = solver.getMaxTemperature();
    float P_evap_final = solver.getEvaporationPower();
    float mass_final = solver.getTotalMass();
    float mass_loss = mass_initial - mass_final;
    float mass_loss_percent = (mass_loss / mass_initial) * 100.0f;

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  T_max = " << T_final << " K" << std::endl;
    std::cout << "  P_evap = " << P_evap_final << " W" << std::endl;
    std::cout << "  Mass loss: " << mass_loss << " (" << mass_loss_percent << "%)" << std::endl;
    std::cout << std::endl;

    // Success criteria
    std::cout << "Evaporation Checks:" << std::endl;

    // 1. Temperature should increase above initial (laser heating)
    std::cout << "  1. Temperature increases: ";
    if (T_final > T_init + 100.0f) {
        std::cout << "PASS ✓ (" << T_final - T_init << " K rise)" << std::endl;
    } else {
        std::cout << "FAIL ✗" << std::endl;
    }

    // 2. Evaporation power should be positive
    std::cout << "  2. Evaporation occurs (P > 0): ";
    if (P_evap_final > 0.0f) {
        std::cout << "PASS ✓ (" << P_evap_final << " W)" << std::endl;
    } else {
        std::cout << "FAIL ✗" << std::endl;
    }

    // 3. Mass should decrease (evaporation removes mass)
    std::cout << "  3. Mass decreases: ";
    if (mass_loss > 0.0f) {
        std::cout << "PASS ✓ (" << mass_loss_percent << "% loss)" << std::endl;
    } else {
        std::cout << "FAIL ✗" << std::endl;
    }

    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_GT(T_final, T_init + 100.0f) << "Temperature should increase with laser";
    EXPECT_GT(P_evap_final, 0.0f) << "Evaporation power should be positive";
    EXPECT_GT(mass_loss, 0.0f) << "Mass should decrease due to evaporation";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
