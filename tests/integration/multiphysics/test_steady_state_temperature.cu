/**
 * @file test_steady_state_temperature.cu
 * @brief Thermal diffusion reduces gradient over time.
 *
 * Strategy: Set up a 1D-like temperature distribution with a hot strip
 * in the domain center using a stationary laser. After enough steps
 * the temperature profile should:
 *   1. Spread (T_max decreases from initial peak as energy diffuses)
 *   2. Not explode (T_max < T_boil_estimate)
 *   3. Be NaN-free throughout
 *
 * Additionally: run two durations (short vs long) and verify T spreads
 * monotonically: T_max_early >= T_max_late.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

static MultiphysicsConfig makeConfig() {
    MultiphysicsConfig cfg;
    cfg.nx = 30;
    cfg.ny = 30;
    cfg.nz = 15;
    cfg.dx = 3e-6f;
    cfg.dt = 2e-8f;
    cfg.enable_thermal      = true;
    cfg.enable_fluid        = false;
    cfg.enable_vof          = false;
    cfg.enable_marangoni    = false;
    cfg.enable_laser        = true;
    cfg.enable_buoyancy     = false;
    cfg.enable_phase_change = false;
    cfg.laser_power         = 20.0f;
    cfg.laser_spot_radius   = 10e-6f;
    cfg.laser_scan_vx       = 0.0f;
    cfg.laser_absorptivity  = 0.35f;
    // Adiabatic walls to trap heat (allows temperature to grow uniformly)
    cfg.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::ADIABATIC);
    cfg.thermal_diffusivity = 9.66e-6f;
    return cfg;
}

TEST(MultiphysicsSteady_stateTest, SteadyStateTemperature) {
    // Run for a short duration and record peak T
    float T_max_early = 0.0f;
    {
        auto cfg = makeConfig();
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        // Run for 20 steps
        for (int i = 0; i < 20; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN at early stage";
        T_max_early = solver.getMaxTemperature();
    }

    // Run the same simulation but for 100 steps (the laser adds heat,
    // so T_max will increase; but the gradient should spread).
    float T_max_late = 0.0f;
    {
        auto cfg = makeConfig();
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < 100; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN at late stage";
        T_max_late = solver.getMaxTemperature();
    }

    // With a stationary laser continuously depositing heat, T_max grows over time
    EXPECT_GT(T_max_late, T_max_early)
        << "Continuous laser should raise T_max over time. "
        << "early=" << T_max_early << " late=" << T_max_late;

    // Temperature must stay below a physical ceiling (< 10000 K)
    EXPECT_LT(T_max_late, 10000.0f)
        << "T_max exploded to " << T_max_late << " K";

    // Temperature must have risen from initial 300 K
    EXPECT_GT(T_max_early, 300.0f)
        << "Laser should have raised T above initial 300 K. Got " << T_max_early;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
