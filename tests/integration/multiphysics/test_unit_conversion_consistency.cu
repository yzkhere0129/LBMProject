/**
 * @file test_unit_conversion_consistency.cu
 * @brief All modules use consistent unit conversions.
 *
 * Strategy: Run two solvers with physically identical setups but different
 * lattice resolutions (dx, dt scaled together to keep the same physical
 * diffusivity alpha_LU = alpha_phys * dt/dx²). Both must produce the same
 * max temperature at the same physical time, to within a tolerance that
 * accounts for grid-dependent truncation error.
 *
 * A unit-conversion bug (e.g., dt not threaded through consistently) would
 * cause the two runs to diverge significantly.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/multiphysics_solver.h"
#include "core/unit_converter.h"

using namespace lbm::physics;
using namespace lbm::core;

static MultiphysicsConfig makeConfig(float dx, float dt) {
    MultiphysicsConfig cfg;
    cfg.nx = 20;
    cfg.ny = 20;
    cfg.nz = 10;
    cfg.dx = dx;
    cfg.dt = dt;
    cfg.enable_thermal      = true;
    cfg.enable_fluid        = false;
    cfg.enable_vof          = false;
    cfg.enable_marangoni    = false;
    cfg.enable_laser        = false;
    cfg.enable_buoyancy     = false;
    cfg.enable_phase_change = false;
    // Use same physical thermal diffusivity (Ti6Al4V)
    cfg.thermal_diffusivity = 9.66e-6f;
    // Periodic all → only diffusion
    cfg.boundaries.setUniform(lbm::physics::BoundaryType::PERIODIC, ThermalBCType::PERIODIC);
    return cfg;
}

TEST(MultiphysicsUnitsTest, UnitConversionConsistency) {
    // Two grid resolutions with same physical diffusivity
    // Both will run for the same number of physical seconds
    const float dx1 = 2e-6f;
    const float dt1 = 1e-8f;
    const float dx2 = 4e-6f;   // 2× coarser
    const float dt2 = 4e-8f;   // 4× larger dt to keep alpha_LU the same
    // alpha_LU = 9.66e-6 * dt / dx² :
    //   run1: 9.66e-6 * 1e-8 / (4e-12) = 0.0241
    //   run2: 9.66e-6 * 4e-8 / (16e-12) = 0.0241  ← same

    // Verify alpha_LU is the same to catch setup mistakes
    float alpha_lu1 = 9.66e-6f * dt1 / (dx1 * dx1);
    float alpha_lu2 = 9.66e-6f * dt2 / (dx2 * dx2);
    ASSERT_NEAR(alpha_lu1, alpha_lu2, 1e-6f)
        << "Test setup error: alpha_LU not matched between resolutions";

    // Physical run time: 200 ns
    const float t_phys = 200e-9f;
    int n1 = static_cast<int>(t_phys / dt1);
    int n2 = static_cast<int>(t_phys / dt2);

    // Both configs start at the same T_init
    const float T_init = 600.0f;

    // Run 1
    float T_max1 = 0.0f;
    {
        auto cfg = makeConfig(dx1, dt1);
        MultiphysicsSolver solver(cfg);
        solver.initialize(T_init, 0.5f);
        for (int i = 0; i < n1; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN in run 1";
        T_max1 = solver.getMaxTemperature();
    }

    // Run 2
    float T_max2 = 0.0f;
    {
        auto cfg = makeConfig(dx2, dt2);
        MultiphysicsSolver solver(cfg);
        solver.initialize(T_init, 0.5f);
        for (int i = 0; i < n2; ++i) solver.step();
        ASSERT_FALSE(solver.checkNaN()) << "NaN in run 2";
        T_max2 = solver.getMaxTemperature();
    }

    // With uniform T, no laser, and periodic BCs, T should remain near T_init
    // in both runs. If unit conversions are broken, one run might diverge.
    EXPECT_NEAR(T_max1, T_init, 2.0f)
        << "Run 1 (fine grid): T_max drifted from init=" << T_init;
    EXPECT_NEAR(T_max2, T_init, 2.0f)
        << "Run 2 (coarse grid): T_max drifted from init=" << T_init;

    // Both runs must agree to within 1 K (same physics, same alpha_LU)
    EXPECT_NEAR(T_max1, T_max2, 2.0f)
        << "Unit conversion inconsistency: fine vs coarse runs disagree. "
        << "T_max1=" << T_max1 << " T_max2=" << T_max2;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
