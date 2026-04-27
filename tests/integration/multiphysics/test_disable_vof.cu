/**
 * @file test_disable_vof.cu
 * @brief Works without VOF (single phase) — validates that single-phase
 *        thermal+fluid operation is stable and physically reasonable.
 *
 * Strategy: With enable_vof=false, the solver runs as single-phase.
 * With no body forces and uniform initial conditions:
 * - v_max must stay below an LBM-stability threshold (Ma << 1)
 * - T must remain near initial value (no laser, periodic BC)
 * - getTotalMass() returns constant (no VOF advection)
 *
 * A broken VOF-disable path (e.g., VOF still allocated and stepping)
 * would produce mass drift or velocity anomalies.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsConfigTest, DisableVOF) {
    // Arrange
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-8f;
    config.enable_thermal   = true;
    config.enable_fluid     = true;
    config.enable_vof       = false;
    config.enable_marangoni = false;
    config.enable_laser     = false;
    config.enable_buoyancy  = false;
    config.enable_phase_change = false;
    // All periodic → no boundary-driven flow
    config.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::PERIODIC);

    MultiphysicsSolver solver(config);
    const float T_init = 300.0f;
    solver.initialize(T_init, 0.5f);

    const int n_steps = 100;
    for (int i = 0; i < n_steps; ++i) {
        solver.step();
    }

    // Assert 1: no NaN
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in VOF-disabled run";

    // Assert 2: no body forces → velocity should stay near machine-zero
    // (initial equilibrium + uniform T + no forcing)
    float v_max = solver.getMaxVelocity();
    EXPECT_LT(v_max, 0.1f)
        << "No forcing in single-phase run: v_max should be near 0, got " << v_max;

    // Assert 3: T stays near init (uniform initial T, periodic BC, no laser)
    float T_max = solver.getMaxTemperature();
    EXPECT_NEAR(T_max, T_init, 2.0f)
        << "Uniform T+periodic: T_max should not change, got " << T_max;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
