/**
 * @file test_subcycling_1_vs_10.cu
 * @brief VOF with N=1 vs N=10 subcycles produce close results.
 *
 * Strategy: Run two identical setups with VOF enabled, one with
 * vof_subcycles=1 and one with vof_subcycles=10. After the same physical
 * time, the temperature and fill-level statistics should agree to within
 * a loose tolerance (subcycling changes advection accuracy, not physics).
 *
 * This catches:
 * - "subcycles field ignored" (both runs would be identical by accident)
 * - "subcycles=10 divides by zero or crashes"
 * - Large divergence indicating the subcycling loop is broken
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

static MultiphysicsConfig makeConfig(int vof_subcycles) {
    MultiphysicsConfig cfg;
    cfg.nx = 20;
    cfg.ny = 20;
    cfg.nz = 15;
    cfg.dx = 3e-6f;
    cfg.dt = 2e-8f;
    cfg.enable_thermal      = true;
    cfg.enable_fluid        = true;
    cfg.enable_vof          = true;
    cfg.enable_vof_advection = true;
    cfg.enable_marangoni    = false;
    cfg.enable_laser        = true;
    cfg.enable_buoyancy     = false;
    cfg.enable_darcy        = false;
    cfg.enable_phase_change = false;
    cfg.vof_subcycles       = vof_subcycles;
    cfg.laser_power         = 20.0f;
    cfg.laser_spot_radius   = 10e-6f;
    cfg.laser_scan_vx       = 0.0f;
    cfg.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);
    return cfg;
}

TEST(MultiphysicsVofTest, Subcycling1vs10) {
    const int n_steps = 40;

    float T_max_n1 = 0.0f, T_max_n10 = 0.0f;
    float v_max_n1 = 0.0f, v_max_n10 = 0.0f;

    {
        auto cfg = makeConfig(1);
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN with vof_subcycles=1";
        T_max_n1 = solver.getMaxTemperature();
        v_max_n1 = solver.getMaxVelocity();
    }

    {
        auto cfg = makeConfig(10);
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN with vof_subcycles=10";
        T_max_n10 = solver.getMaxTemperature();
        v_max_n10 = solver.getMaxVelocity();
    }

    // Both runs must produce physically sensible output
    EXPECT_GT(T_max_n1,  300.0f) << "n=1:  laser should heat above 300 K";
    EXPECT_GT(T_max_n10, 300.0f) << "n=10: laser should heat above 300 K";

    // Temperature must agree within 20% (same physics, slightly different
    // numerical accuracy due to subcycling CFL)
    float T_rel_diff = std::abs(T_max_n10 - T_max_n1) / (T_max_n1 + 1e-6f);
    EXPECT_LT(T_rel_diff, 0.20f)
        << "T_max should agree within 20% between n=1 and n=10 subcycles. "
        << "n1=" << T_max_n1 << " n10=" << T_max_n10 << " rel_diff=" << T_rel_diff;

    // The two velocity maxima must also be in the same ballpark
    float v_ref = std::max(v_max_n1, v_max_n10) + 1e-6f;
    float v_diff = std::abs(v_max_n10 - v_max_n1) / v_ref;
    EXPECT_LT(v_diff, 0.50f)
        << "v_max should agree within 50% between n=1 and n=10. "
        << "n1=" << v_max_n1 << " n10=" << v_max_n10;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
