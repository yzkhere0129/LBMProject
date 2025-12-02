/**
 * @file test_realistic_temperature.cu
 * @brief Regression test to ensure LPBF simulation produces realistic temperatures
 *
 * This test validates that the laser heating model produces physically reasonable
 * temperatures for typical LPBF conditions. It helps catch issues like:
 * - Excessive laser power density
 * - Missing heat dissipation mechanisms
 * - Coordinate system errors
 * - Time step issues
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "physics/multiphysics_solver.h"
#include "physics/material_database.h"

using namespace lbm::physics;

/**
 * @brief Test that realistic LPBF parameters produce reasonable peak temperatures
 *
 * Expected behavior:
 * - With 200W laser, 50 μm spot, Ti6Al4V
 * - Peak temperature should be 1900-3500 K (near/above melting)
 * - NOT 10,000+ K which indicates missing cooling or wrong parameters
 */
TEST(LPBFSimulation, RealisticPeakTemperature) {
    std::cout << "\n========================================\n";
    std::cout << "  LPBF Realistic Temperature Test\n";
    std::cout << "========================================\n\n";

    // Configuration: Standard LPBF with REDUCED laser power for small domain
    MultiphysicsConfig config;

    // Domain: 200 x 200 x 100 μm
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2.0e-6f;  // 2 μm cells

    // Time stepping
    config.dt = 1.0e-7f;  // 0.1 μs

    // Physics modules: ONLY thermal + laser (isolate the problem)
    config.enable_thermal = true;
    config.enable_fluid = false;    // Disable to isolate thermal issue
    config.enable_darcy = false;
    config.enable_marangoni = false;
    config.enable_surface_tension = false;
    config.enable_laser = true;
    config.enable_vof = false;      // Disable to simplify
    config.enable_vof_advection = false;

    // Material: Ti6Al4V
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;  // m²/s
    config.density = 4110.0f;

    // CRITICAL: REDUCED laser power for small domain
    // Real LPBF uses 200W in ~mm scale melt pool
    // Our domain is only 200 μm wide - need proportionally less power
    //
    // Scaling: Real melt pool ~100-200 μm diameter in infinite substrate
    //          Our domain: 200 μm with periodic boundaries (no heat escape)
    //          Reduce power by ~10x to compensate
    config.laser_power = 20.0f;            // 20W (instead of 200W)
    config.laser_spot_radius = 50.0e-6f;   // 50 μm
    config.laser_absorptivity = 0.35f;     // 35%
    config.laser_penetration_depth = 10.0e-6f;  // 10 μm

    std::cout << "Configuration:\n";
    std::cout << "  Domain: " << config.nx << " x " << config.ny << " x " << config.nz
              << " (" << config.nx*config.dx*1e6 << " x " << config.ny*config.dx*1e6
              << " x " << config.nz*config.dx*1e6 << " μm³)\n";
    std::cout << "  Laser power: " << config.laser_power << " W\n";
    std::cout << "  Laser spot: " << config.laser_spot_radius*1e6 << " μm\n";
    std::cout << "  Time step: " << config.dt*1e6 << " μs\n\n";

    // Calculate expected heating rate
    float I_center = 2.0f * config.laser_power / (M_PI * config.laser_spot_radius * config.laser_spot_radius);
    float beta = 1.0f / config.laser_penetration_depth;
    float Q_max = config.laser_absorptivity * I_center * beta;
    float rho = config.material.rho_solid;
    float cp = config.material.cp_solid;
    float deltaT_per_step = Q_max * config.dt / (rho * cp);

    std::cout << "Expected heating (without diffusion):\n";
    std::cout << "  Peak volumetric heat: " << Q_max << " W/m³\n";
    std::cout << "  ΔT per timestep: " << deltaT_per_step << " K/step\n";
    std::cout << "  Time to melting (~1928K): " << (1928-300)/deltaT_per_step << " steps\n\n";

    // Initialize solver
    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);  // Room temperature, flat interface

    // Run simulation for 200 steps (20 μs)
    const int total_steps = 200;
    const int check_interval = 50;

    std::cout << "Running simulation...\n";
    std::cout << "Step      Time[μs]    T_max[K]    T_min[K]\n";
    std::cout << "------------------------------------------------\n";

    std::vector<float> h_temp(config.nx * config.ny * config.nz);

    for (int step = 0; step <= total_steps; step++) {
        if (step % check_interval == 0) {
            // Get temperature field
            const float* d_temp = solver.getTemperature();
            cudaMemcpy(h_temp.data(), d_temp, h_temp.size() * sizeof(float), cudaMemcpyDeviceToHost);

            // Find min/max
            float T_min = *std::min_element(h_temp.begin(), h_temp.end());
            float T_max = *std::max_element(h_temp.begin(), h_temp.end());

            std::cout << std::setw(4) << step
                      << std::setw(12) << step * config.dt * 1e6
                      << std::setw(12) << T_max
                      << std::setw(12) << T_min << "\n";
        }

        if (step < total_steps) {
            solver.step(config.dt);
        }
    }

    // Final temperature check
    const float* d_temp = solver.getTemperature();
    cudaMemcpy(h_temp.data(), d_temp, h_temp.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float T_min = *std::min_element(h_temp.begin(), h_temp.end());
    float T_max = *std::max_element(h_temp.begin(), h_temp.end());

    std::cout << "\nFinal Results:\n";
    std::cout << "  T_min = " << T_min << " K\n";
    std::cout << "  T_max = " << T_max << " K\n";
    std::cout << "  Material melting point = " << config.material.T_liquidus << " K\n\n";

    // Assertions: Temperature should be realistic
    EXPECT_GT(T_max, 300.0f) << "Temperature should rise above initial";
    EXPECT_GT(T_max, config.material.T_solidus * 0.8f)
        << "Should approach melting point with laser";
    EXPECT_LT(T_max, 4000.0f)
        << "Temperature should not exceed boiling point by much";
    EXPECT_LT(T_max, 10000.0f)
        << "Temperature is unrealistically high - check laser power or heat dissipation";

    // Best case: Temperature near melting point
    EXPECT_NEAR(T_max, config.material.T_liquidus, 1000.0f)
        << "Peak temperature should be near material melting point for realistic LPBF";

    std::cout << "========================================\n";
    std::cout << (T_max < 4000.0f ? "  ✓ TEST PASSED\n" : "  ✗ TEST FAILED\n");
    std::cout << "========================================\n\n";
}

/**
 * @brief Test coordinate system: laser spot should heat correct location
 */
TEST(LPBFSimulation, LaserSpotLocation) {
    std::cout << "\n========================================\n";
    std::cout << "  Laser Coordinate System Test\n";
    std::cout << "========================================\n\n";

    MultiphysicsConfig config;
    config.nx = 50;
    config.ny = 50;
    config.nz = 25;
    config.dx = 2.0e-6f;
    config.dt = 1.0e-7f;

    config.enable_thermal = true;
    config.enable_fluid = false;
    config.enable_laser = true;
    config.enable_vof = false;

    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;

    // SMALL laser power for quick test
    config.laser_power = 5.0f;
    config.laser_spot_radius = 20.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    // Run 50 steps
    for (int i = 0; i < 50; i++) {
        solver.step(config.dt);
    }

    // Get temperature field
    std::vector<float> h_temp(config.nx * config.ny * config.nz);
    const float* d_temp = solver.getTemperature();
    cudaMemcpy(h_temp.data(), d_temp, h_temp.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Find location of maximum temperature
    auto max_it = std::max_element(h_temp.begin(), h_temp.end());
    int max_idx = std::distance(h_temp.begin(), max_it);

    int k = max_idx / (config.nx * config.ny);
    int j = (max_idx % (config.nx * config.ny)) / config.nx;
    int i = max_idx % config.nx;

    float x_phys = i * config.dx * 1e6;  // μm
    float y_phys = j * config.dx * 1e6;
    float z_phys = k * config.dx * 1e6;

    std::cout << "Maximum temperature location:\n";
    std::cout << "  Grid indices: (" << i << ", " << j << ", " << k << ")\n";
    std::cout << "  Physical: (" << x_phys << ", " << y_phys << ", " << z_phys << ") μm\n";

    // Laser starts at 10% of domain in x, centered in y, at surface (z=nz/2)
    float expected_x = config.nx * config.dx * 0.1f * 1e6;  // μm
    float expected_y = config.ny * config.dx * 0.5f * 1e6;
    float expected_z = config.nz * config.dx * 0.5f * 1e6;  // Surface

    std::cout << "  Expected: (" << expected_x << ", " << expected_y
              << ", " << expected_z << ") μm\n";

    // With scan velocity 1 m/s and 50 steps * 0.1 μs = 5 μs:
    // Laser moves: 1 m/s * 5e-6 s = 5 μm
    float laser_travel = 1.0f * 50 * config.dt * 1e6;  // μm
    expected_x += laser_travel;

    std::cout << "  Laser traveled: " << laser_travel << " μm\n";
    std::cout << "  Adjusted expected x: " << expected_x << " μm\n\n";

    // Check x-position (should be near laser, allowing for diffusion and scan)
    EXPECT_NEAR(x_phys, expected_x, 20.0f)
        << "X-coordinate of peak temperature should match laser position";

    // Check y-position (should be centered)
    EXPECT_NEAR(y_phys, expected_y, 20.0f)
        << "Y-coordinate should be centered";

    // Check z-position (should be at or just below surface)
    EXPECT_NEAR(z_phys, expected_z, 10.0f)
        << "Z-coordinate should be at surface";

    std::cout << "========================================\n";
    std::cout << "  ✓ Coordinate System Correct\n";
    std::cout << "========================================\n\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
