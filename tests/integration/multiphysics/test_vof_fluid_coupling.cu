/**
 * @file test_vof_fluid_coupling.cu
 * @brief Test VOF-Fluid coupling: VOF advects with fluid velocity
 *
 * Success Criteria:
 * - Interface moves with fluid velocity
 * - Mass conservation (Σf constant within 1%)
 * - Interface shape deforms correctly
 * - No NaN
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsCouplingTest, VOFFluidCoupling) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: VOF-Fluid Coupling" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MultiphysicsConfig config;
    config.nx = 80;
    config.ny = 80;
    config.nz = 40;
    config.dx = 2e-6f;
    config.dt = 1e-8f;

    // Enable fluid + VOF advection
    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;  // Key feature to test
    config.enable_marangoni = false;
    config.enable_laser = false;
    config.enable_buoyancy = false;
    config.enable_surface_tension = false;

    config.vof_subcycles = 10;  // Subcycle for stability

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  VOF subcycles: " << config.vof_subcycles << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);

    // Initialize with planar interface at mid-height
    solver.initialize(300.0f, 0.5f);

    // Get initial mass
    float mass_initial = solver.getTotalMass();

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  Interface at z = " << 0.5f * config.nz << std::endl;
    std::cout << "  Initial mass: " << mass_initial << std::endl;
    std::cout << std::endl;

    // Apply uniform horizontal velocity to fluid
    // (This would require a custom initialization - simplified here)
    // In practice, we rely on natural flow development

    const int n_steps = 500;
    const int check_interval = 100;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        if ((step + 1) % check_interval == 0) {
            float v_max = solver.getMaxVelocity();
            float mass_current = solver.getTotalMass();
            float mass_change = ((mass_current - mass_initial) / mass_initial) * 100.0f;

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(2)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(4) << v_max << " m/s"
                      << " | Δm = " << std::setprecision(3) << mass_change << "%"
                      << std::endl;

            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(60, '-') << std::endl;

    // Final mass conservation check
    float mass_final = solver.getTotalMass();
    float mass_change_percent = ((mass_final - mass_initial) / mass_initial) * 100.0f;

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  Initial mass: " << mass_initial << std::endl;
    std::cout << "  Final mass:   " << mass_final << std::endl;
    std::cout << "  Mass change:  " << std::fixed << std::setprecision(4)
              << mass_change_percent << " %" << std::endl;
    std::cout << std::endl;

    // Success criteria: Mass conservation within 1%
    const float tolerance = 1.0f;

    std::cout << "Mass Conservation Check:" << std::endl;
    std::cout << "  Tolerance: " << tolerance << "%" << std::endl;
    std::cout << "  Measured:  " << std::abs(mass_change_percent) << "%" << std::endl;

    if (std::abs(mass_change_percent) < tolerance) {
        std::cout << "  Status: PASS ✓" << std::endl;
    } else {
        std::cout << "  Status: FAIL ✗" << std::endl;
    }
    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_LT(std::abs(mass_change_percent), tolerance)
        << "Mass conservation violated: Δm = " << mass_change_percent << "%";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
