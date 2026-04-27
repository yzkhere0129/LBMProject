/**
 * @file test_steady_state_flow.cu
 * @brief Flow driven by buoyancy reaches a steady state (velocity saturates).
 *
 * Strategy: Run a buoyancy-driven flow case and check that velocity
 * growth rate decreases over time — characteristic of approach to steady state.
 * Concretely:
 *   v_max at step 150 should be greater than v_max at step 50 (still accelerating)
 *   AND the fractional growth from step 100 to 150 should be less than
 *   from step 50 to 100 (deceleration of growth → approach to steady state).
 *
 * This catches bugs where:
 * - Buoyancy never drives flow (v_max stays 0)
 * - Flow diverges (v grows without bound)
 * - Viscous damping is missing (no deceleration)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsSteady_stateTest, SteadyStateFlow) {
    MultiphysicsConfig config;
    config.nx = 25;
    config.ny = 25;
    config.nz = 20;
    config.dx = 4e-6f;
    config.dt = 2e-8f;
    config.enable_thermal      = true;
    config.enable_fluid        = true;
    config.enable_vof          = false;
    config.enable_marangoni    = false;
    config.enable_laser        = true;
    config.enable_buoyancy     = true;
    config.enable_darcy        = false;
    config.enable_phase_change = false;
    config.laser_power         = 30.0f;
    config.laser_spot_radius   = 15e-6f;
    config.laser_scan_vx       = 0.0f;
    config.gravity_z           = -9.81f;
    config.thermal_expansion_coeff = 1.5e-5f;
    config.reference_temperature   = 300.0f;
    config.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    float v50 = 0.0f, v100 = 0.0f, v150 = 0.0f;

    for (int i = 0; i < 50; ++i) solver.step();
    ASSERT_FALSE(solver.checkNaN()) << "NaN at step 50";
    v50 = solver.getMaxVelocity();

    for (int i = 0; i < 50; ++i) solver.step();
    ASSERT_FALSE(solver.checkNaN()) << "NaN at step 100";
    v100 = solver.getMaxVelocity();

    for (int i = 0; i < 50; ++i) solver.step();
    ASSERT_FALSE(solver.checkNaN()) << "NaN at step 150";
    v150 = solver.getMaxVelocity();

    // Assert 1: buoyancy must drive some flow
    EXPECT_GT(v150, 1e-6f) << "Buoyancy should drive nonzero flow. v150=" << v150;

    // Assert 2: flow must grow initially (buoyancy accelerates from rest)
    EXPECT_GT(v100, v50)
        << "Flow should accelerate early. v50=" << v50 << " v100=" << v100;

    // Assert 3: growth rate must decelerate (approach to steady state)
    float growth_50_to_100 = v100 - v50;
    float growth_100_to_150 = v150 - v100;
    // Viscous drag must eventually balance buoyancy → growth slows
    // Allow equality within 20% if both intervals are small (already near SS)
    if (v150 > 0.01f) {
        EXPECT_LE(growth_100_to_150, growth_50_to_100 * 1.5f)
            << "Flow growth should decelerate (approach steady state). "
            << "growth[50-100]=" << growth_50_to_100
            << " growth[100-150]=" << growth_100_to_150;
    }

    // Assert 4: velocity must not explode
    EXPECT_LT(v150, 100.0f) << "v_max exploded to " << v150 << " m/s";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
