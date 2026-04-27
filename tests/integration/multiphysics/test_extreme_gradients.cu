/**
 * @file test_extreme_gradients.cu
 * @brief Solver survives 10^7 K/m temperature gradients.
 *
 * Strategy: Initialize a steep temperature step (hot half vs cold half)
 * corresponding to ΔT/dx ~ 10^7 K/m. Run for a short time and verify:
 *   1. No NaN/Inf detected
 *   2. T_max stays below a physical ceiling
 *   3. v_max stays bounded (CFL limiter active)
 *
 * This tests robustness of the thermal and Marangoni solvers under
 * the largest gradients seen in LPBF (laser focal spot edge).
 * A broken gradient computation (e.g., divide-by-zero, unclamped index)
 * would produce NaN on the first step.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsStabilityTest, ExtremeGradients) {
    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 5e-9f;
    config.enable_thermal      = true;
    config.enable_fluid        = true;
    config.enable_vof          = false;
    config.enable_marangoni    = true;   // Marangoni uses ∇T — key stress test
    config.enable_laser        = false;
    config.enable_buoyancy     = false;
    config.enable_darcy        = false;
    config.enable_phase_change = false;
    config.dsigma_dT           = -0.26e-3f;
    config.marangoni_csf_multiplier = 1.0f;
    config.boundaries.setUniform(lbm::physics::BoundaryType::WALL, ThermalBCType::ADIABATIC);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    // Build a step-function temperature field: left half 300 K, right half 2000 K
    // ΔT = 1700 K over 1 cell = 1700/2e-6 = 8.5e8 K/m > 10^7 K/m target
    int N = config.nx * config.ny * config.nz;
    std::vector<float> T_host(N, 300.0f);
    std::vector<float> fl_host(N, 1.0f);  // all liquid
    for (int z = 0; z < config.nz; ++z) {
        for (int y = 0; y < config.ny; ++y) {
            for (int x = config.nx / 2; x < config.nx; ++x) {
                int idx = x + config.nx * (y + config.ny * z);
                T_host[idx] = 2000.0f;  // hot half
            }
        }
    }
    solver.initialize(T_host.data(), fl_host.data());

    const int n_steps = 30;
    for (int i = 0; i < n_steps; ++i) {
        solver.step();
        // Check for early NaN
        if (solver.checkNaN()) {
            FAIL() << "NaN detected at step " << i + 1
                   << " under extreme gradient (10^8 K/m)";
        }
    }

    float T_max = solver.getMaxTemperature();
    float v_max = solver.getMaxVelocity();

    // Temperature must stay bounded (diffusion may reduce peak, but no explosion)
    EXPECT_LT(T_max, 1e6f) << "T_max exploded under extreme gradient: " << T_max;
    EXPECT_GT(T_max, 300.0f) << "T_max below initial min: " << T_max;

    // Velocity must stay below LBM stability limit
    EXPECT_LT(v_max, 500.0f)
        << "v_max exceeded physical bound under extreme gradient: " << v_max;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
