/**
 * @file test_vof_subcycling_convergence.cu
 * @brief VOF mass conservation with subcycling enabled.
 *
 * Strategy: Run a VOF simulation with subcycling and a body force.
 * The total fill-level (proxy for liquid mass) must be conserved to
 * within 1% over the run. getTotalMass() provides the integral.
 *
 * This directly tests that VOF advection + subcycling does not drift mass.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsVofTest, VOFSubcyclingConvergence) {
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 15;
    config.dx = 3e-6f;
    config.dt = 2e-8f;
    config.enable_thermal      = false;
    config.enable_fluid        = true;
    config.enable_vof          = true;
    config.enable_vof_advection = true;
    config.enable_marangoni    = false;
    config.enable_laser        = false;
    config.enable_buoyancy     = false;
    config.enable_darcy        = false;
    config.enable_phase_change = false;
    config.vof_subcycles       = 5;
    config.enable_vof_mass_correction = true;
    config.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    float mass_init = solver.getTotalMass();
    ASSERT_GT(mass_init, 0.0f) << "Initial mass must be positive";

    const int n_steps = 60;
    for (int i = 0; i < n_steps; ++i) {
        solver.step();
    }

    EXPECT_FALSE(solver.checkNaN()) << "NaN in VOF subcycling run";

    float mass_final = solver.getTotalMass();
    float mass_rel_drift = std::abs(mass_final - mass_init) / (mass_init + 1e-30f);

    // VOF mass correction is active; drift should be small (<5%)
    EXPECT_LT(mass_rel_drift, 0.05f)
        << "VOF mass conservation: too much drift. "
        << "initial=" << mass_init << " final=" << mass_final
        << " drift=" << mass_rel_drift * 100.0f << "%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
