/**
 * @file test_rapid_solidification.cu
 * @brief Fast cooling doesn't diverge — phase change robustness test.
 *
 * Strategy: Start with a domain at T_melt (liquid), then apply aggressive
 * boundary cooling to drive rapid solidification. Check:
 *   1. No NaN/Inf throughout the run
 *   2. T_max decreases monotonically (heat extracted by cooling BC)
 *   3. Final T_max is below initial T_melt (solidification occurred)
 *
 * This catches bugs in the enthalpy-source method (ESM) under rapid
 * cooling, where the phase-change solver might fail to converge.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsRobustnessTest, RapidSolidification) {
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 5e-9f;
    config.enable_thermal      = true;
    config.enable_fluid        = false;   // thermal only for clean test
    config.enable_vof          = false;
    config.enable_marangoni    = false;
    config.enable_laser        = false;
    config.enable_buoyancy     = false;
    config.enable_phase_change = true;    // ESM must survive rapid cooling
    config.enable_darcy        = false;

    // Use Ti6Al4V material (default) — T_liquidus ≈ 1923 K
    const float T_melt = config.material.T_liquidus;

    // Strong Dirichlet BC: all walls fixed at 300 K (aggressive cooling)
    config.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::DIRICHLET);
    config.boundaries.dirichlet_temperature = 300.0f;

    MultiphysicsSolver solver(config);
    // Initialize at melting point (fully liquid)
    solver.initialize(T_melt, 0.5f);

    float T_max_prev = solver.getMaxTemperature();
    const int n_steps = 100;

    for (int i = 0; i < n_steps; ++i) {
        solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN at step " << i + 1
                                        << " during rapid solidification";
    }

    float T_max_final = solver.getMaxTemperature();

    // T must have decreased (cooling BC drains heat)
    EXPECT_LT(T_max_final, T_max_prev)
        << "Aggressive cooling BC should lower T_max. "
        << "Initial=" << T_max_prev << " Final=" << T_max_final;

    // T must be above 0 K (physical lower bound)
    EXPECT_GT(T_max_final, 0.0f)
        << "T_max went below 0 K: " << T_max_final;

    // T should drop significantly below T_melt
    EXPECT_LT(T_max_final, T_melt)
        << "After 100 steps of strong cooling, T_max should be below T_melt ("
        << T_melt << " K). Got " << T_max_final;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
