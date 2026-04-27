/**
 * @file test_force_magnitude_ordering.cu
 * @brief Stronger buoyancy force → larger steady-state velocity.
 *
 * Strategy: Run two solvers identical except for gravity magnitude:
 *   g1 = 9.81 m/s²  (Earth gravity)
 *   g2 = 49.05 m/s² (5× Earth)
 * With a temperature gradient (laser), buoyancy drives flow.
 * After the same number of steps: v_max(g2) > v_max(g1).
 *
 * A bug that ignores gravity magnitude (e.g., always uses hardcoded g=9.81)
 * would make the two runs identical, failing this ordering check.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

static MultiphysicsConfig makeConfig(float gravity_z) {
    MultiphysicsConfig cfg;
    cfg.nx = 20;
    cfg.ny = 20;
    cfg.nz = 20;
    cfg.dx = 5e-6f;
    cfg.dt = 2e-8f;
    cfg.enable_thermal      = true;
    cfg.enable_fluid        = true;
    cfg.enable_vof          = false;
    cfg.enable_marangoni    = false;
    cfg.enable_laser        = true;
    cfg.enable_buoyancy     = true;
    cfg.enable_darcy        = false;
    cfg.enable_phase_change = false;
    cfg.laser_power         = 30.0f;
    cfg.laser_spot_radius   = 15e-6f;
    cfg.laser_scan_vx       = 0.0f;
    cfg.laser_absorptivity  = 0.35f;
    cfg.gravity_x           = 0.0f;
    cfg.gravity_y           = 0.0f;
    cfg.gravity_z           = gravity_z;
    cfg.thermal_expansion_coeff = 1.5e-5f;
    cfg.reference_temperature   = 300.0f;
    cfg.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);
    return cfg;
}

TEST(MultiphysicsForcesTest, ForceMagnitudeOrdering) {
    const int n_steps = 80;

    float v_g1 = 0.0f;
    {
        auto cfg = makeConfig(-9.81f);
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN with g1=9.81";
        v_g1 = solver.getMaxVelocity();
    }

    float v_g5 = 0.0f;
    {
        auto cfg = makeConfig(-49.05f);   // 5× gravity
        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 0.5f);
        for (int i = 0; i < n_steps; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN with g2=49.05";
        v_g5 = solver.getMaxVelocity();
    }

    // Key assertion: 5× gravity must drive more flow than 1× gravity
    EXPECT_GT(v_g5, v_g1)
        << "5× gravity should produce higher v_max. "
        << "v_g1=" << v_g1 << " v_g5=" << v_g5;

    // Reasonable upper bound on ratio (shouldn't be wildly >5× due to
    // viscous drag, but must be clearly > 1)
    float ratio = v_g5 / (v_g1 + 1e-10f);
    EXPECT_GT(ratio, 1.5f)
        << "5× gravity should produce clearly more flow (ratio > 1.5×). Got " << ratio;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
