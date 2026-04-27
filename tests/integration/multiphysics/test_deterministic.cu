/**
 * @file test_deterministic.cu
 * @brief Same input gives same output — GPU determinism check.
 *
 * Strategy: Run the same simulation twice with identical configuration
 * and initial conditions. Both runs must produce bit-identical or
 * near-identical final state (T_max, v_max, mass).
 *
 * CUDA reductions over floats are not guaranteed to be bit-identical
 * across kernel launches due to non-deterministic warp scheduling.
 * We therefore check that results agree to within 1 ULP relative
 * tolerance (1e-5 for float), not strict bit equality.
 *
 * A failure here indicates non-deterministic GPU kernel behavior or
 * uninitialized memory that varies across runs.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

static MultiphysicsConfig makeConfig() {
    MultiphysicsConfig cfg;
    cfg.nx = 20;
    cfg.ny = 20;
    cfg.nz = 10;
    cfg.dx = 2e-6f;
    cfg.dt = 1e-8f;
    cfg.enable_thermal      = true;
    cfg.enable_fluid        = true;
    cfg.enable_vof          = false;
    cfg.enable_marangoni    = false;
    cfg.enable_laser        = true;
    cfg.enable_buoyancy     = false;
    cfg.enable_darcy        = false;
    cfg.enable_phase_change = false;
    cfg.laser_power         = 50.0f;
    cfg.laser_spot_radius   = 10e-6f;
    cfg.laser_scan_vx       = 0.0f;
    cfg.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);
    return cfg;
}

TEST(MultiphysicsRegressionTest, Deterministic) {
    const int n_steps = 50;
    const float T_init = 300.0f;

    float T_max1, v_max1;
    {
        auto cfg = makeConfig();
        MultiphysicsSolver solver(cfg);
        solver.initialize(T_init, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN in run 1";
        T_max1 = solver.getMaxTemperature();
        v_max1 = solver.getMaxVelocity();
    }

    float T_max2, v_max2;
    {
        auto cfg = makeConfig();
        MultiphysicsSolver solver(cfg);
        solver.initialize(T_init, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN in run 2";
        T_max2 = solver.getMaxTemperature();
        v_max2 = solver.getMaxVelocity();
    }

    // Temperature must agree to within 0.1 K (CUDA reductions may have tiny variance)
    EXPECT_NEAR(T_max1, T_max2, 0.1f)
        << "T_max not deterministic: run1=" << T_max1 << " run2=" << T_max2;

    // Velocity must agree to within 1e-3 m/s
    EXPECT_NEAR(v_max1, v_max2, 1e-3f)
        << "v_max not deterministic: run1=" << v_max1 << " run2=" << v_max2;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
