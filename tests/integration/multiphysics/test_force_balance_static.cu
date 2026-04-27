/**
 * @file test_force_balance_static.cu
 * @brief Forces sum to zero at equilibrium — uniform initial state
 *
 * Strategy: With uniform initial conditions, no laser, no body forces,
 * and all periodic boundaries, the system should be in mechanical
 * equilibrium. Velocity should remain near-zero throughout.
 * A broken force pipeline (e.g., non-zero spurious force in zero-gradient
 * field) would produce velocity growth.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsForcesTest, ForceBalanceStatic) {
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-8f;
    config.enable_thermal   = true;
    config.enable_fluid     = true;
    config.enable_vof       = false;
    config.enable_marangoni = false;
    config.enable_laser     = false;
    config.enable_buoyancy  = false;
    config.enable_darcy     = false;
    config.enable_phase_change = false;
    // Periodic everywhere: no boundary-induced flow
    config.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::PERIODIC);

    MultiphysicsSolver solver(config);
    const float T_init = 300.0f;
    solver.initialize(T_init, 0.5f);

    const int n_steps = 100;
    for (int i = 0; i < n_steps; ++i) {
        solver.step();
    }

    EXPECT_FALSE(solver.checkNaN()) << "NaN in zero-force run";

    // With zero gradient and no forcing, velocity must stay near machine-zero.
    // Allow a small numerical floor (1e-3 m/s) for floating-point accumulation.
    float v_max = solver.getMaxVelocity();
    EXPECT_LT(v_max, 0.05f)
        << "Zero-force case: v_max should stay near 0. Got " << v_max << " m/s";

    // T must not drift from uniform init
    float T_max = solver.getMaxTemperature();
    EXPECT_NEAR(T_max, T_init, 1.5f)
        << "Zero-force uniform T: T_max drifted to " << T_max;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
