/**
 * @file test_disable_marangoni.cu
 * @brief Disabling Marangoni suppresses thermocapillary flow.
 *
 * Strategy: Run two cases — one WITH Marangoni and one WITHOUT —
 * under an imposed temperature gradient (laser on one side).
 * With Marangoni enabled the temperature gradient drives surface flow;
 * with it disabled no such forcing exists. Therefore:
 *   v_max(marangoni=on) > v_max(marangoni=off)
 *
 * This directly catches a "Marangoni flag ignored" regression.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

static MultiphysicsConfig makeConfig(bool enable_marangoni) {
    MultiphysicsConfig cfg;
    cfg.nx = 30;
    cfg.ny = 30;
    cfg.nz = 15;
    cfg.dx = 2e-6f;
    cfg.dt = 5e-9f;
    cfg.enable_thermal      = true;
    cfg.enable_fluid        = true;
    cfg.enable_vof          = false;
    cfg.enable_marangoni    = enable_marangoni;
    cfg.enable_laser        = true;
    cfg.enable_buoyancy     = false;
    cfg.enable_phase_change = false;
    cfg.enable_darcy        = false;
    // Laser in centre to create a T gradient
    cfg.laser_power         = 50.0f;
    cfg.laser_spot_radius   = 10e-6f;
    cfg.laser_absorptivity  = 0.35f;
    cfg.laser_scan_vx       = 0.0f; // stationary
    // Adiabatic walls so gradient can build
    cfg.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);
    // Marangoni params (only matter when enabled)
    cfg.dsigma_dT = -0.26e-3f;
    cfg.marangoni_csf_multiplier = 1.0f;
    return cfg;
}

TEST(MultiphysicsConfigTest, DisableMarangoni) {
    const int n_steps = 80;

    // Run WITHOUT Marangoni
    {
        auto cfg = makeConfig(false);
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN in Marangoni-disabled run";
    }

    float v_no_marangoni;
    {
        auto cfg = makeConfig(false);
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        v_no_marangoni = solver.getMaxVelocity();
    }

    float v_with_marangoni;
    {
        auto cfg = makeConfig(true);
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN in Marangoni-enabled run";
        v_with_marangoni = solver.getMaxVelocity();
    }

    // Key assertion: Marangoni force must produce measurably higher velocity
    EXPECT_GT(v_with_marangoni, v_no_marangoni)
        << "Marangoni ON should produce higher v_max than Marangoni OFF. "
        << "v_with=" << v_with_marangoni << " v_without=" << v_no_marangoni;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
