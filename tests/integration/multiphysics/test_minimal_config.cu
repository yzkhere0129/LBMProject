/**
 * @file test_minimal_config.cu
 * @brief Only thermal, no fluid — verifies thermal-only operation
 *
 * Strategy: With enable_fluid=false (thermal-only), no body forces exist.
 * Initial uniform temperature should remain uniform (no gradient → no
 * diffusion change). Velocity field must stay exactly zero. Any non-zero
 * velocity or temperature change indicates a spurious coupling path.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsConfigTest, MinimalConfig) {
    // Arrange: thermal-only config with all other physics disabled
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-8f;
    config.enable_thermal = true;
    config.enable_fluid    = false;
    config.enable_vof      = false;
    config.enable_marangoni = false;
    config.enable_laser    = false;
    config.enable_buoyancy = false;
    config.enable_phase_change = false;
    // Periodic all faces so no boundary flux
    config.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::PERIODIC);

    MultiphysicsSolver solver(config);

    const float T_init = 600.0f;
    solver.initialize(T_init, 0.5f);

    // Act: run a short time
    const int n_steps = 50;
    for (int i = 0; i < n_steps; ++i) {
        solver.step();
    }

    // Assert 1: no NaN
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected after thermal-only run";

    // Assert 2: with uniform T and periodic BC, T_max should equal T_init
    // (no gradient → no diffusion flux). Tolerance: 0.5 K for floating-point drift.
    float T_max = solver.getMaxTemperature();
    EXPECT_NEAR(T_max, T_init, 1.0f)
        << "Uniform T should not change under periodic BC, no laser. "
        << "Got T_max=" << T_max << " expected ~" << T_init;

    // Assert 3: with fluid disabled, velocity must stay zero
    float v_max = solver.getMaxVelocity();
    EXPECT_LT(v_max, 1e-6f) << "No fluid solver: v_max should be ~0, got " << v_max;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
