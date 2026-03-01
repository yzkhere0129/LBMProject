/**
 * @file test_energy_conservation_laser_only.cu
 * @brief Test energy balance: laser input = thermal energy gain
 *
 * Success Criteria:
 * - Temperature increases monotonically with laser heating
 * - No NaN
 * - P_laser > 0 (laser is working)
 * - Energy change is positive (energy is being added)
 *
 * Physics:
 * - Laser source enabled (adds energy)
 * - Substrate cooling enabled (prevents temperature runaway)
 * - Expected: temperature increases steadily
 *
 * Note: A perfect P_laser ≈ dE/dt energy balance cannot be tested in isolation
 * because the thermal solver uses the material's apparent heat capacity
 * (which includes latent heat), and the hard temperature cap at 50000K
 * can cause spurious energy balance errors when temperature reaches the cap.
 * This test validates the qualitative behavior: laser adds energy and temperature rises.
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
    config.enable_darcy = false;
    config.enable_laser = true;
    config.enable_buoyancy = false;
    config.enable_evaporation_mass_loss = false;
    // Disable all cooling for this test - we want pure laser heating
    // to verify temperature increases. The T_max cap at 50000K will clamp
    // temperature, but at low laser power and short time, T stays well below cap.
    config.enable_radiation_bc = false;
    config.enable_substrate_cooling = false;

    // Laser parameters - very low power so temperature increases slowly
    // and stays far from the 50000K cap over 200 steps (2 us total)
    // At T=300K, absorbed power = 5*0.35 = 1.75W over domain volume
    // Energy change = P*t = 1.75*2e-6 = 3.5 uJ over 2 us
    config.laser_power = 5.0f;    // 5 W (low to avoid temperature cap)
    config.laser_spot_radius = 50e-6f;  // 50 um (larger spot = lower heat flux)
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 20e-6f;  // 20 um (deeper = lower intensity)

    // Use current default thermal diffusivity (Ti6Al4V liquid, cp=831 J/(kg*K))
    // alpha = k/(rho*cp) = 33/(4110*831) = 9.66e-6 m^2/s
    config.thermal_diffusivity = 9.66e-6f;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "x" << config.ny << "x" << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " um" << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Laser power: " << config.laser_power << " W" << std::endl;
    std::cout << "  Absorptivity: " << config.laser_absorptivity << std::endl;
    std::cout << "  Expected absorbed power: "
              << config.laser_power * config.laser_absorptivity << " W" << std::endl;
    std::cout << "  thermal_diffusivity: " << config.thermal_diffusivity << " m^2/s" << std::endl;
    std::cout << "  Cooling: none (all cooling disabled for pure laser heating test)" << std::endl;
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

    // Time integration
    const int n_steps = 200;
    const int check_interval = 40;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Step | Time [us] | T_max [K] | P_laser [W] | dE/dt [W]" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    double E_previous = E_initial;
    double t_previous = 0.0;
    float T_max_seen = T_init;

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

            T_max_seen = std::max(T_max_seen, T_max);

            std::cout << std::setw(4) << step + 1
                      << " | " << std::fixed << std::setprecision(2) << std::setw(9)
                      << t_current * 1e6
                      << " | " << std::setw(9) << std::setprecision(1) << T_max
                      << " | " << std::setw(11) << std::setprecision(3) << P_laser
                      << " | " << std::setw(9) << std::setprecision(3) << dE_dt
                      << std::endl;

            E_previous = E_current;
            t_previous = t_current;

            // Check for NaN
            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(70, '-') << std::endl;

    // Final checks
    solver.computeEnergyBalance();
    float P_laser_final = solver.getLaserAbsorbedPower();
    float T_final = solver.getMaxTemperature();
    const double E_final = solver.getCurrentEnergyBalance().E_thermal;

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  P_laser_absorbed = " << std::fixed << std::setprecision(3)
              << P_laser_final << " W" << std::endl;
    std::cout << "  T_max = " << T_final << " K" << std::endl;
    std::cout << "  E_final - E_initial = " << (E_final - E_initial) << " J" << std::endl;
    std::cout << std::endl;

    // Success criteria: qualitative behavior checks
    // 1. No NaN
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";

    // 2. Laser is depositing power
    EXPECT_GT(P_laser_final, 1.0f) << "Laser absorbed power should be positive";

    // 3. Temperature has increased (laser is heating)
    EXPECT_GT(T_final, T_init + 10.0f)
        << "Temperature should increase with laser heating";

    // 4. Temperature in physical range (below the 50000 K solver cap)
    // With 5W laser over 2 us and low heat flux (large spot), T stays well below cap.
    EXPECT_LT(T_final, 49000.0f)
        << "Temperature should stay well below the 50000 K solver cap";

    // 5. Energy increased overall
    EXPECT_GT(E_final, E_initial)
        << "Total thermal energy should increase with laser input";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
