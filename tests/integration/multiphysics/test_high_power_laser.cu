/**
 * @file test_high_power_laser.cu
 * @brief High-power laser (1 kW) on small domain — robustness test.
 *
 * Strategy: Apply 1 kW laser (extreme for this domain size) and verify:
 *   1. Solver remains stable (no NaN)
 *   2. Temperature rises significantly above ambient
 *   3. CFL limiter prevents velocity explosion even under extreme Marangoni
 *
 * This tests the high-end of the operating envelope — parameters that
 * would correspond to deep keyhole mode. The solver should degrade
 * gracefully (clip forces, not crash) rather than diverge.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsRobustnessTest, HighPowerLaser) {
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 15;
    config.dx = 3e-6f;
    config.dt = 1e-8f;
    config.enable_thermal      = true;
    config.enable_fluid        = true;
    config.enable_vof          = false;
    config.enable_marangoni    = true;
    config.enable_laser        = true;
    config.enable_buoyancy     = false;
    config.enable_darcy        = true;    // Darcy needed for partially-liquid domain
    config.enable_phase_change = true;
    // Extreme laser: 1 kW (far above typical 150-200 W LPBF)
    config.laser_power         = 1000.0f;
    config.laser_spot_radius   = 15e-6f;
    config.laser_absorptivity  = 0.35f;
    config.laser_scan_vx       = 0.0f;
    config.dsigma_dT           = -0.26e-3f;
    config.marangoni_csf_multiplier = 1.0f;
    // CFL limiter must be active to survive this
    config.cfl_velocity_target     = 0.15f;
    config.cfl_use_gradual_scaling = true;
    config.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    const int n_steps = 60;
    for (int i = 0; i < n_steps; ++i) {
        solver.step();
        if (solver.checkNaN()) {
            FAIL() << "NaN detected at step " << i + 1
                   << " under 1 kW laser (high-power robustness test)";
        }
    }

    float T_max = solver.getMaxTemperature();
    float v_max = solver.getMaxVelocity();

    // High power laser must raise T significantly
    EXPECT_GT(T_max, 1000.0f)
        << "1 kW laser should raise T well above ambient. T_max=" << T_max;

    // CFL limiter must prevent velocity explosion
    EXPECT_LT(v_max, 500.0f)
        << "CFL limiter should cap v_max even at 1 kW. Got " << v_max;

    // T must not be negative or zero (sanity)
    EXPECT_GT(T_max, 0.0f) << "T_max non-positive: " << T_max;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
