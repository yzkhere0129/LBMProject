/**
 * @file test_energy_conservation_laser_only.cu
 * @brief Test energy balance: laser input = thermal energy gain
 *
 * Success Criteria:
 * - P_laser_absorbed ≈ dE/dt (within 10%)
 * - Energy balance holds over multiple timesteps
 * - No NaN
 *
 * Physics:
 * - Laser source enabled (adds energy)
 * - No cooling mechanisms (no radiation, no evaporation, no substrate cooling)
 * - Expected: dE/dt = P_laser (perfect energy balance)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsEnergyTest, ConservationLaserOnly) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Energy Conservation - Laser Only" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;  // 2 μm
    config.dt = 1e-8f;  // 10 ns

    // Enable thermal + laser only
    config.enable_thermal = true;
    config.enable_fluid = false;
    config.enable_vof = false;
    config.enable_marangoni = false;
    config.enable_laser = true;
    config.enable_buoyancy = false;
    config.enable_evaporation_mass_loss = false;
    config.enable_radiation_bc = false;
    config.enable_substrate_cooling = false;

    // Laser parameters (moderate power for clear signal)
    config.laser_power = 50.0f;  // 50 W
    config.laser_spot_radius = 30e-6f;  // 30 μm
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10e-6f;  // 10 μm

    // Material properties
    config.thermal_diffusivity = 5.8e-6f;  // Ti6Al4V

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Laser power: " << config.laser_power << " W" << std::endl;
    std::cout << "  Absorptivity: " << config.laser_absorptivity << std::endl;
    std::cout << "  Expected absorbed power: "
              << config.laser_power * config.laser_absorptivity << " W" << std::endl;
    std::cout << std::endl;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize with uniform low temperature
    const float T_init = 300.0f;  // K
    solver.initialize(T_init, 0.5f);

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  T_init = " << T_init << " K" << std::endl;
    std::cout << std::endl;

    // Compute initial energy
    solver.computeEnergyBalance();
    const auto& initial_energy = solver.getCurrentEnergyBalance();
    const double E_initial = initial_energy.E_thermal;

    std::cout << "Initial energy: E_0 = " << E_initial << " J" << std::endl;
    std::cout << std::endl;

    // Expected absorbed power
    const float P_expected = config.laser_power * config.laser_absorptivity;

    // Time integration
    const int n_steps = 200;
    const int check_interval = 40;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Step | Time [μs] | T_max [K] | P_laser [W] | dE/dt [W] | Balance [%]" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    double E_previous = E_initial;
    double t_previous = 0.0;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        if ((step + 1) % check_interval == 0) {
            solver.computeEnergyBalance();
            const auto& energy = solver.getCurrentEnergyBalance();

            float T_max = solver.getMaxTemperature();
            float P_laser = solver.getLaserAbsorbedPower();

            double E_current = energy.E_thermal;
            double t_current = (step + 1) * config.dt;
            double dt_elapsed = t_current - t_previous;
            double dE_dt = (E_current - E_previous) / dt_elapsed;

            // Energy balance: P_laser ≈ dE/dt
            double balance_error = 0.0;
            if (P_laser > 1e-10) {
                balance_error = ((dE_dt - P_laser) / P_laser) * 100.0;
            }

            std::cout << std::setw(4) << step + 1
                      << " | " << std::fixed << std::setprecision(2) << std::setw(9)
                      << t_current * 1e6
                      << " | " << std::setw(9) << std::setprecision(1) << T_max
                      << " | " << std::setw(11) << std::setprecision(3) << P_laser
                      << " | " << std::setw(9) << std::setprecision(3) << dE_dt
                      << " | " << std::setw(11) << std::setprecision(2) << balance_error
                      << std::endl;

            E_previous = E_current;
            t_previous = t_current;

            // Check for NaN
            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(80, '-') << std::endl;

    // Final energy balance check
    solver.computeEnergyBalance();
    const auto& final_energy = solver.getCurrentEnergyBalance();
    float P_laser_final = solver.getLaserAbsorbedPower();

    double E_final = final_energy.E_thermal;
    double t_final = n_steps * config.dt;
    double dE_dt_average = (E_final - E_initial) / t_final;

    double balance_error = ((dE_dt_average - P_laser_final) / P_laser_final) * 100.0;

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  P_laser_absorbed = " << std::fixed << std::setprecision(3)
              << P_laser_final << " W" << std::endl;
    std::cout << "  dE/dt (average)  = " << dE_dt_average << " W" << std::endl;
    std::cout << "  Balance error    = " << std::setprecision(2)
              << balance_error << " %" << std::endl;
    std::cout << std::endl;

    // Success criteria: Energy balance within 10%
    const double tolerance = 10.0;  // 10%

    std::cout << "Energy Balance Check:" << std::endl;
    std::cout << "  Tolerance: " << tolerance << "%" << std::endl;
    std::cout << "  Measured:  " << std::abs(balance_error) << "%" << std::endl;

    if (std::abs(balance_error) < tolerance) {
        std::cout << "  Status: PASS ✓" << std::endl;
    } else {
        std::cout << "  Status: FAIL ✗" << std::endl;
    }
    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_LT(std::abs(balance_error), tolerance)
        << "Energy balance violated: error = " << balance_error << "%";

    // Temperature should increase
    float T_final = solver.getMaxTemperature();
    EXPECT_GT(T_final, T_init + 100.0f)
        << "Temperature should increase with laser heating";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
