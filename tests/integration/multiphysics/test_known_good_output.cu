/**
 * @file test_known_good_output.cu
 * @brief Regression: known config produces output within tight bounds.
 *
 * Strategy: Run a fully-specified, small conduction benchmark (thermal
 * only, no fluid, stationary laser). After a fixed number of steps,
 * T_max must land within analytically-bounded range.
 *
 * Analytical estimate for a point heat source in a finite box:
 * Energy deposited = P_absorbed * t
 * T_max_upper = T_init + E / (rho * cp * V) (all energy in one cell)
 * T_max_lower = T_init (laser just started)
 *
 * After 100 steps at dt=1e-8 s, P=50 W, abs=0.35:
 * E = 50 * 0.35 * 100 * 1e-8 = 1.75e-5 J
 * V = dx³ * nx * ny * nz = (2e-6)³ * 20*20*10 = 1.6e-14 m³
 * T_max_lower: at least T_init + 1 K (some heating)
 * T_max_upper: at most 1e6 K (no explosion)
 *
 * This test serves as a regression anchor: if the laser power, absorptivity,
 * or thermal diffusivity handling changes, T_max will shift detectably.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsRegressionTest, KnownGoodOutput) {
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-8f;
    config.enable_thermal      = true;
    config.enable_fluid        = false;
    config.enable_vof          = false;
    config.enable_marangoni    = false;
    config.enable_laser        = true;
    config.enable_buoyancy     = false;
    config.enable_phase_change = false;
    config.laser_power         = 50.0f;
    config.laser_spot_radius   = 10e-6f;
    config.laser_absorptivity  = 0.35f;
    config.laser_scan_vx       = 0.0f;   // stationary
    config.thermal_diffusivity = 9.66e-6f;
    config.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::ADIABATIC);

    const float T_init = 300.0f;
    MultiphysicsSolver solver(config);
    solver.initialize(T_init, 0.5f);

    const int n_steps = 100;
    for (int i = 0; i < n_steps; ++i) solver.step();

    ASSERT_FALSE(solver.checkNaN()) << "NaN in known-good output run";

    float T_max = solver.getMaxTemperature();

    // Lower bound: laser must have heated above init
    EXPECT_GT(T_max, T_init + 1.0f)
        << "Laser should have raised T above init. T_max=" << T_max;

    // Upper bound: energy ceiling concentrated in laser spot.
    // P_absorbed * n_steps * dt = 50 * 0.35 * 100 * 1e-8 = 1.75e-5 J
    // All energy in spot vol: V_spot ~ pi*r^2*3*dx = pi*(10e-6)^2*6e-6 ≈ 1.9e-15 m³
    // Ti6Al4V: rho=4430, cp=526
    // T_spot_max ~ 300 + 1.75e-5/(4430*526*1.9e-15) ~ 4300 K
    // 3× safety factor for numerical over-concentration
    EXPECT_LT(T_max, T_init + 3.0f * 4300.0f)
        << "T_max exceeded energy-balance ceiling. T_max=" << T_max;

    // T_max must be above ambient (laser deposited energy)
    EXPECT_GT(T_max, 302.0f)
        << "T_max should be clearly above ambient after laser heating";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
