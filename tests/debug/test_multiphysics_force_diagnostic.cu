/**
 * @file test_multiphysics_force_diagnostic.cu
 * @brief Diagnostic test to debug MultiphysicsSolver force generation
 *
 * This test checks if forces are being generated and propagated through the pipeline:
 * 1. Temperature gradient → Marangoni force
 * 2. Force → Fluid velocity
 * 3. Velocity → VOF advection
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsForceDiagnostic, TemperatureGradientGeneratesForce) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DIAGNOSTIC: Temperature Gradient → Force" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 40;
    config.ny = 40;
    config.nz = 20;
    config.dx = 2e-6f;

    config.enable_thermal = false;  // Use static temperature
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;  // Don't advect yet
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;
    config.enable_darcy = false;  // Disable Darcy to isolate Marangoni
    config.enable_buoyancy = false;  // Disable buoyancy to isolate Marangoni

    config.material = MaterialDatabase::getTi6Al4V();
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;

    MultiphysicsSolver solver(config);

    // Create temperature field with RADIAL GRADIENT (hot center, cold edge)
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);

    float T_hot = 2500.0f;   // K
    float T_cold = 2000.0f;  // K
    float center_x = config.nx / 2.0f;
    float center_y = config.ny / 2.0f;
    float R_hot = 10.0f;  // cells

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                float dx = i - center_x;
                float dy = j - center_y;
                float r = sqrtf(dx * dx + dy * dy);

                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    // Linear decay
                    float decay = (r - R_hot) / (config.nx / 2.0f - R_hot);
                    h_temp[idx] = T_hot - decay * (T_hot - T_cold);
                }
                h_temp[idx] = std::max(h_temp[idx], T_cold);
            }
        }
    }

    // Initialize with planar interface at mid-height
    std::vector<float> h_fill(num_cells);
    int z_interface = config.nz / 2;

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    solver.initialize(h_temp.data(), h_fill.data());

    std::cout << "Temperature field initialized:" << std::endl;
    std::cout << "  T_hot = " << T_hot << " K (center)" << std::endl;
    std::cout << "  T_cold = " << T_cold << " K (edge)" << std::endl;
    std::cout << "  ΔT = " << (T_hot - T_cold) << " K" << std::endl;
    std::cout << "  Gradient magnitude ~ " << (T_hot - T_cold) / (R_hot * config.dx)
              << " K/m" << std::endl;
    std::cout << std::endl;

    // Run for a few steps
    const int n_steps = 100;

    std::cout << "Running " << n_steps << " steps..." << std::endl;
    std::cout << std::setw(8) << "Step"
              << std::setw(15) << "v_max [m/s]" << std::endl;
    std::cout << std::string(23, '-') << std::endl;

    for (int step = 0; step <= n_steps; ++step) {
        solver.step(config.dt);

        if (step % 20 == 0) {
            float v_max = solver.getMaxVelocity();
            std::cout << std::setw(8) << step
                      << std::setw(15) << std::scientific << std::setprecision(4)
                      << v_max << std::endl;
        }
    }

    std::cout << std::string(23, '-') << std::endl;
    std::cout << std::endl;

    float final_v = solver.getMaxVelocity();

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Final velocity: " << final_v << " m/s" << std::endl;
    std::cout << std::endl;

    // Test: Velocity should develop with temperature gradient
    EXPECT_GT(final_v, 0.0f) << "Velocity should be generated with temperature gradient";
    EXPECT_GT(final_v, 0.01f) << "Velocity should be significant (> 0.01 m/s)";

    if (final_v > 0.01f) {
        std::cout << "PASS: Temperature gradient generates force and velocity" << std::endl;
    } else if (final_v > 0.0f) {
        std::cout << "PARTIAL: Velocity too small (force generation issue?)" << std::endl;
    } else {
        std::cout << "FAIL: NO velocity generated (force pipeline broken)" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;
}

TEST(MultiphysicsForceDiagnostic, UniformTemperatureGeneratesZeroForce) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DIAGNOSTIC: Uniform Temperature → NO Force" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration
    MultiphysicsConfig config;
    config.nx = 40;
    config.ny = 40;
    config.nz = 20;
    config.dx = 2e-6f;

    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;  // Disable buoyancy to isolate Marangoni

    config.material = MaterialDatabase::getTi6Al4V();
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.dt = 1e-7f;

    MultiphysicsSolver solver(config);

    // Uniform temperature (THIS IS THE BUG IN ORIGINAL TESTS)
    solver.initialize(2300.0f, 0.5f);

    std::cout << "Temperature field: UNIFORM 2300 K" << std::endl;
    std::cout << "Expected: NO temperature gradient → NO Marangoni force" << std::endl;
    std::cout << std::endl;

    // Run for a few steps
    const int n_steps = 100;

    for (int step = 0; step < n_steps; ++step) {
        solver.step(config.dt);
    }

    float final_v = solver.getMaxVelocity();

    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Final velocity: " << final_v << " m/s" << std::endl;
    std::cout << std::endl;

    // Test: Velocity should be ZERO (no gradient = no force)
    EXPECT_NEAR(final_v, 0.0f, 1e-6f)
        << "Uniform temperature should produce zero force and zero velocity";

    if (final_v < 1e-6f) {
        std::cout << "PASS: Uniform temperature produces zero force (as expected)" << std::endl;
    } else {
        std::cout << "FAIL: Velocity should be zero with uniform temperature" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
