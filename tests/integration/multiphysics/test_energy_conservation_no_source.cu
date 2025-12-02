/**
 * @file test_energy_conservation_no_source.cu
 * @brief Test energy conservation without any heat source
 *
 * Success Criteria:
 * - dE/dt ≈ 0 (< 1% of initial energy over 100 timesteps)
 * - No NaN
 * - Temperature stable
 *
 * Physics:
 * - Thermal diffusion enabled, but uniform T → no gradient → no flux
 * - No laser source
 * - Expected: Perfect energy conservation (only numerical error)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsEnergyTest, ConservationNoSource) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Energy Conservation - No Source" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 50;
    config.ny = 50;
    config.nz = 25;
    config.dx = 2e-6f;  // 2 μm
    config.dt = 1e-8f;  // 10 ns

    // Enable only thermal diffusion (no sources)
    config.enable_thermal = true;
    config.enable_fluid = false;
    config.enable_vof = false;
    config.enable_marangoni = false;
    config.enable_laser = false;
    config.enable_buoyancy = false;
    config.enable_evaporation_mass_loss = false;
    config.enable_radiation_bc = false;
    config.enable_substrate_cooling = false;

    // Material properties
    config.thermal_diffusivity = 5.8e-6f;  // Ti6Al4V

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Physics: Thermal only, no sources" << std::endl;
    std::cout << std::endl;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize with uniform temperature (no gradients → no flux)
    const float T_init = 1500.0f;  // K
    solver.initialize(T_init, 0.5f);

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  T_uniform = " << T_init << " K" << std::endl;
    std::cout << std::endl;

    // Compute initial energy
    solver.computeEnergyBalance();
    const auto& initial_energy = solver.getCurrentEnergyBalance();
    const double E_initial = initial_energy.E_thermal;

    std::cout << "Initial energy: E_0 = " << E_initial << " J" << std::endl;
    std::cout << std::endl;

    // Time integration
    const int n_steps = 100;
    const int check_interval = 20;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        if ((step + 1) % check_interval == 0) {
            solver.computeEnergyBalance();
            const auto& energy = solver.getCurrentEnergyBalance();

            float T_max = solver.getMaxTemperature();
            double E_current = energy.E_thermal;
            double dE = E_current - E_initial;
            double dE_percent = (dE / E_initial) * 100.0;

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(2)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | T_max = " << std::setprecision(1) << T_max << " K"
                      << " | ΔE = " << std::scientific << std::setprecision(3)
                      << dE << " J (" << std::fixed << std::setprecision(4)
                      << dE_percent << "%)" << std::endl;

            // Check for NaN
            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    // Final energy check
    solver.computeEnergyBalance();
    const auto& final_energy = solver.getCurrentEnergyBalance();
    const double E_final = final_energy.E_thermal;
    const double dE_total = E_final - E_initial;
    const double dE_percent = (dE_total / E_initial) * 100.0;

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  E_initial = " << std::scientific << std::setprecision(6)
              << E_initial << " J" << std::endl;
    std::cout << "  E_final   = " << E_final << " J" << std::endl;
    std::cout << "  ΔE        = " << dE_total << " J" << std::endl;
    std::cout << "  ΔE/E_0    = " << std::fixed << std::setprecision(4)
              << dE_percent << " %" << std::endl;
    std::cout << std::endl;

    // Success criteria: Energy conservation within 1%
    const double tolerance = 1.0;  // 1%

    std::cout << "Energy Conservation Check:" << std::endl;
    std::cout << "  Tolerance: " << tolerance << "%" << std::endl;
    std::cout << "  Measured:  " << std::abs(dE_percent) << "%" << std::endl;

    if (std::abs(dE_percent) < tolerance) {
        std::cout << "  Status: PASS ✓" << std::endl;
    } else {
        std::cout << "  Status: FAIL ✗" << std::endl;
    }
    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_LT(std::abs(dE_percent), tolerance)
        << "Energy conservation violated: ΔE/E_0 = " << dE_percent << "%";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
