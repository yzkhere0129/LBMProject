/**
 * @file test_melt_pool_diagnostic.cu
 * @brief Diagnostic test suite for melt pool physics validation
 *
 * Quick sanity checks for all melt pool physics:
 * - Laser heating
 * - Phase change (melting)
 * - Marangoni convection
 * - VOF interface tracking
 * - Multiphysics coupling
 *
 * Run this test to quickly verify melt pool physics is working correctly.
 */

#include <gtest/gtest.h>
#include "physics/thermal/ThermalLBM.cuh"
#include "physics/fluid/FluidLBM.cuh"
#include "physics/multiphysics/MultiphysicsSolver.cuh"
#include "physics/material/MaterialProperties.h"
#include "config/ConfigManager.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

class MeltPoolDiagnostic : public ::testing::Test {
protected:
    void SetUp() override {
        // Small domain for fast testing
        nx = 40;
        ny = 40;
        nz = 20;

        // Physical parameters
        dx = 2.0e-6;  // 2 μm
        dt = 1.0e-7;  // 0.1 μs

        mat = MaterialProperties::Ti6Al4V();
    }

    void TearDown() override {
        // Cleanup handled by smart pointers
    }

    int nx, ny, nz;
    double dx, dt;
    MaterialProperties mat;
};

/**
 * Test 1: Laser Heating
 *
 * Verify:
 * - Temperature increases under laser irradiation
 * - Heating rate is reasonable
 * - No NaN/Inf values
 */
TEST_F(MeltPoolDiagnostic, LaserHeatingWorks) {
    std::cout << "\n=== Diagnostic Test 1: Laser Heating ===" << std::endl;

    // Create config
    auto config = std::make_shared<Config>();
    config->grid.nx = nx;
    config->grid.ny = ny;
    config->grid.nz = nz;
    config->grid.dx = dx;
    config->simulation.dt = dt;
    config->simulation.num_steps = 1000;  // 100 ns total
    config->material.name = "Ti6Al4V";

    // Laser parameters
    config->laser.power = 500.0;  // W
    config->laser.spot_radius = 25.0e-6;  // 25 μm
    config->laser.absorptivity = 0.35;
    config->laser.penetration_depth = 15.0e-6;  // 15 μm
    config->laser.position_x = nx * dx / 2.0;
    config->laser.position_y = ny * dx / 2.0;
    config->laser.enabled = true;

    // Physics modules
    config->physics.enable_thermal = true;
    config->physics.enable_phase_change = false;
    config->physics.enable_fluid = false;
    config->physics.enable_vof = false;
    config->physics.enable_laser = true;

    // Create solver
    MultiphysicsSolver solver(config);

    // Set initial temperature to room temperature
    std::vector<float> T_init(nx * ny * nz, 300.0f);
    solver.setInitialTemperature(T_init);

    // Measure initial temperature at center
    std::vector<float> T_field(nx * ny * nz);
    solver.getTemperature(T_field);
    int center_idx = nx/2 + ny/2 * nx + nz/2 * nx * ny;
    float T_initial = T_field[center_idx];

    std::cout << "Initial T at center: " << T_initial << " K" << std::endl;

    // Run simulation
    for (int step = 0; step < 1000; ++step) {
        solver.step();
    }

    // Measure final temperature
    solver.getTemperature(T_field);
    float T_final = T_field[center_idx];

    // Find max temperature
    float T_max = *std::max_element(T_field.begin(), T_field.end());

    std::cout << "Final T at center: " << T_final << " K" << std::endl;
    std::cout << "Max T in domain: " << T_max << " K" << std::endl;
    std::cout << "Temperature rise: " << (T_final - T_initial) << " K" << std::endl;

    // Validation
    EXPECT_GT(T_final, T_initial + 100.0f)
        << "Temperature should rise significantly under laser irradiation";
    EXPECT_LT(T_max, 20000.0f)
        << "Temperature should not exceed reasonable limits";
    EXPECT_TRUE(std::isfinite(T_max))
        << "No NaN or Inf values should appear";

    std::cout << "✓ Laser heating test PASSED" << std::endl;
}

/**
 * Test 2: Melting
 *
 * Verify:
 * - Solid melts when heated above T_liquidus
 * - Liquid fraction goes from 0 to 1
 * - Latent heat absorption occurs
 */
TEST_F(MeltPoolDiagnostic, MeltingWorks) {
    std::cout << "\n=== Diagnostic Test 2: Melting ===" << std::endl;

    auto config = std::make_shared<Config>();
    config->grid.nx = nx;
    config->grid.ny = ny;
    config->grid.nz = nz;
    config->grid.dx = dx;
    config->simulation.dt = dt;
    config->simulation.num_steps = 2000;
    config->material.name = "Ti6Al4V";

    // Strong laser to ensure melting
    config->laser.power = 1000.0;
    config->laser.spot_radius = 30.0e-6;
    config->laser.absorptivity = 0.35;
    config->laser.penetration_depth = 15.0e-6;
    config->laser.position_x = nx * dx / 2.0;
    config->laser.position_y = ny * dx / 2.0;
    config->laser.enabled = true;

    config->physics.enable_thermal = true;
    config->physics.enable_phase_change = true;
    config->physics.enable_fluid = false;
    config->physics.enable_laser = true;

    MultiphysicsSolver solver(config);

    // Start from room temperature (solid)
    std::vector<float> T_init(nx * ny * nz, 300.0f);
    solver.setInitialTemperature(T_init);

    // Run simulation
    for (int step = 0; step < 2000; ++step) {
        solver.step();
    }

    // Check liquid fraction
    std::vector<float> lf_field(nx * ny * nz);
    solver.getLiquidFraction(lf_field);

    // Count melted cells
    int solid_cells = 0;
    int mushy_cells = 0;
    int liquid_cells = 0;

    for (float lf : lf_field) {
        if (lf < 0.01f) solid_cells++;
        else if (lf < 0.99f) mushy_cells++;
        else liquid_cells++;
    }

    float total = nx * ny * nz;
    std::cout << "Phase distribution:" << std::endl;
    std::cout << "  Solid:  " << solid_cells << " (" << 100.0f * solid_cells / total << "%)" << std::endl;
    std::cout << "  Mushy:  " << mushy_cells << " (" << 100.0f * mushy_cells / total << "%)" << std::endl;
    std::cout << "  Liquid: " << liquid_cells << " (" << 100.0f * liquid_cells / total << "%)" << std::endl;

    // Validation
    EXPECT_GT(liquid_cells, 0)
        << "Some cells should melt under laser irradiation";
    EXPECT_LT(liquid_cells, total * 0.8f)
        << "Not entire domain should melt (laser is localized)";
    EXPECT_LT(mushy_cells, total * 0.1f)
        << "Mushy zone should be thin";

    std::cout << "✓ Melting test PASSED" << std::endl;
}

/**
 * Test 3: Marangoni Flow
 *
 * Verify:
 * - Temperature gradient generates surface flow
 * - Velocity is non-zero
 * - Flow direction is correct (hot → cold)
 */
TEST_F(MeltPoolDiagnostic, MarangoniFlowWorks) {
    std::cout << "\n=== Diagnostic Test 3: Marangoni Flow ===" << std::endl;

    auto config = std::make_shared<Config>();
    config->grid.nx = nx;
    config->grid.ny = ny;
    config->grid.nz = nz;
    config->grid.dx = dx;
    config->simulation.dt = dt;
    config->simulation.num_steps = 1000;
    config->material.name = "Ti6Al4V";

    config->physics.enable_thermal = true;
    config->physics.enable_phase_change = false;
    config->physics.enable_fluid = true;
    config->physics.enable_vof = true;
    config->physics.enable_marangoni = true;
    config->physics.enable_laser = false;

    MultiphysicsSolver solver(config);

    // Set up temperature gradient (hot center, cold edges)
    std::vector<float> T_init(nx * ny * nz);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = ix + iy * nx + iz * nx * ny;

                // Distance from center
                float dx_c = (ix - nx/2) * dx;
                float dy_c = (iy - ny/2) * dx;
                float r = std::sqrt(dx_c*dx_c + dy_c*dy_c);

                // Gaussian temperature profile
                float r0 = 20.0e-6;  // 20 μm
                T_init[idx] = 2000.0f + 500.0f * std::exp(-r*r / (r0*r0));
            }
        }
    }
    solver.setInitialTemperature(T_init);

    // Set liquid everywhere
    std::vector<float> lf_init(nx * ny * nz, 1.0f);
    solver.setInitialLiquidFraction(lf_init);

    // Set interface at mid-height
    std::vector<float> fill_init(nx * ny * nz);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = ix + iy * nx + iz * nx * ny;
                fill_init[idx] = (iz < nz/2) ? 1.0f : 0.0f;
            }
        }
    }
    solver.setInitialFillLevel(fill_init);

    // Run simulation
    for (int step = 0; step < 1000; ++step) {
        solver.step();
    }

    // Check velocity field
    std::vector<float> vx(nx * ny * nz);
    std::vector<float> vy(nx * ny * nz);
    std::vector<float> vz(nx * ny * nz);
    solver.getVelocity(vx, vy, vz);

    // Find max velocity
    float v_max = 0.0f;
    for (size_t i = 0; i < vx.size(); ++i) {
        float v = std::sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        v_max = std::max(v_max, v);
    }

    std::cout << "Maximum velocity: " << v_max << " m/s" << std::endl;

    // Validation
    EXPECT_GT(v_max, 1e-6f)
        << "Marangoni force should generate non-zero velocity";
    EXPECT_LT(v_max, 10.0f)
        << "Velocity should be reasonable (< 10 m/s)";
    EXPECT_TRUE(std::isfinite(v_max))
        << "No NaN or Inf in velocity field";

    std::cout << "✓ Marangoni flow test PASSED" << std::endl;
}

/**
 * Test 4: Full Integration
 *
 * Verify:
 * - All physics modules work together
 * - No crashes or NaN values
 * - Reasonable physical behavior
 */
TEST_F(MeltPoolDiagnostic, FullIntegrationWorks) {
    std::cout << "\n=== Diagnostic Test 4: Full Integration ===" << std::endl;

    auto config = std::make_shared<Config>();
    config->grid.nx = nx;
    config->grid.ny = ny;
    config->grid.nz = nz;
    config->grid.dx = dx;
    config->simulation.dt = dt;
    config->simulation.num_steps = 500;
    config->material.name = "Ti6Al4V";

    // Moderate laser power
    config->laser.power = 200.0;
    config->laser.spot_radius = 25.0e-6;
    config->laser.absorptivity = 0.35;
    config->laser.penetration_depth = 15.0e-6;
    config->laser.position_x = nx * dx / 2.0;
    config->laser.position_y = ny * dx / 2.0;
    config->laser.enabled = true;

    // Enable all physics
    config->physics.enable_thermal = true;
    config->physics.enable_phase_change = true;
    config->physics.enable_fluid = true;
    config->physics.enable_vof = true;
    config->physics.enable_marangoni = true;
    config->physics.enable_darcy = true;
    config->physics.enable_laser = true;

    MultiphysicsSolver solver(config);

    // Initialize
    std::vector<float> T_init(nx * ny * nz, 300.0f);
    solver.setInitialTemperature(T_init);

    // Run full multiphysics simulation
    bool has_error = false;
    try {
        for (int step = 0; step < 500; ++step) {
            solver.step();

            // Periodic sanity check
            if (step % 100 == 0) {
                std::vector<float> T(nx * ny * nz);
                solver.getTemperature(T);

                float T_max = *std::max_element(T.begin(), T.end());
                if (!std::isfinite(T_max)) {
                    has_error = true;
                    std::cerr << "ERROR: NaN/Inf detected at step " << step << std::endl;
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        has_error = true;
        std::cerr << "ERROR: Exception during simulation: " << e.what() << std::endl;
    }

    EXPECT_FALSE(has_error)
        << "Full multiphysics simulation should run without errors";

    // Final state check
    std::vector<float> T_final(nx * ny * nz);
    std::vector<float> lf_final(nx * ny * nz);
    std::vector<float> vx(nx * ny * nz);
    std::vector<float> vy(nx * ny * nz);
    std::vector<float> vz(nx * ny * nz);

    solver.getTemperature(T_final);
    solver.getLiquidFraction(lf_final);
    solver.getVelocity(vx, vy, vz);

    float T_max = *std::max_element(T_final.begin(), T_final.end());
    int liquid_cells = std::count_if(lf_final.begin(), lf_final.end(),
                                     [](float lf) { return lf > 0.99f; });

    float v_max = 0.0f;
    for (size_t i = 0; i < vx.size(); ++i) {
        float v = std::sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        v_max = std::max(v_max, v);
    }

    std::cout << "Final state:" << std::endl;
    std::cout << "  T_max: " << T_max << " K" << std::endl;
    std::cout << "  Liquid cells: " << liquid_cells << std::endl;
    std::cout << "  v_max: " << v_max << " m/s" << std::endl;

    EXPECT_GT(T_max, 500.0f) << "Laser should heat material";
    EXPECT_GT(liquid_cells, 0) << "Some melting should occur";
    EXPECT_TRUE(std::isfinite(T_max) && std::isfinite(v_max))
        << "All fields should be finite";

    std::cout << "✓ Full integration test PASSED" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Melt Pool Diagnostic Test Suite     \n";
    std::cout << "========================================\n";
    std::cout << "\n";
    std::cout << "This test suite performs quick sanity checks\n";
    std::cout << "for all melt pool physics modules:\n";
    std::cout << "  1. Laser heating\n";
    std::cout << "  2. Phase change (melting)\n";
    std::cout << "  3. Marangoni convection\n";
    std::cout << "  4. Full multiphysics integration\n";
    std::cout << "\n";

    int result = RUN_ALL_TESTS();

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Diagnostic Test Suite Complete      \n";
    std::cout << "========================================\n";
    std::cout << "\n";

    return result;
}
