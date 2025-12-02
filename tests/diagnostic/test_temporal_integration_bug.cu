/**
 * @file test_temporal_integration_bug.cu
 * @brief Minimal test to isolate temporal divergence bug
 *
 * CRITICAL BUG DIAGNOSIS (Week 2):
 * This test confirms that temperature predictions depend on timestep size,
 * which violates temporal convergence for time integration.
 *
 * Expected behavior:
 *   - dt=0.2μs, 15 steps (3.0μs total) → should give SAME result as
 *   - dt=0.1μs, 30 steps (3.0μs total) → should give SAME result as
 *   - dt=0.05μs, 60 steps (3.0μs total)
 *
 * Actual behavior (BUG):
 *   - dt=0.2μs → T_max = 4071K
 *   - dt=0.1μs → T_max = 2535K (37% error!)
 *   - dt=0.05μs → T_max = 1661K (59% error!)
 *
 * Root cause hypothesis:
 *   Energy deposition is missing dt scaling somewhere in the chain:
 *   LaserSource → computeLaserHeatSourceKernel → addHeatSource
 */

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace lbm::physics;

// Simple test parameters
constexpr int NX = 20;
constexpr int NY = 20;
constexpr int NZ = 20;
constexpr float DX = 2e-6f;  // 2 μm
constexpr float T_INIT = 300.0f;  // K
constexpr float LASER_POWER = 10.0f;  // W (absorbed)

/**
 * @brief Test thermal integration with different timesteps
 *
 * Setup:
 *   - Single Gaussian heat source (constant position)
 *   - Pure diffusion (no advection, no phase change)
 *   - Adiabatic boundaries
 *   - Run to same total physical time with different timesteps
 *
 * Expected: Final temperature field should be INDEPENDENT of dt
 * Actual (BUG): Temperature scales with dt (bigger dt → higher T)
 */
void testTemporalConvergence(float dt, int num_steps, const char* label) {
    printf("\n");
    printf("========================================\n");
    printf("Test: %s\n", label);
    printf("  dt = %.3e s\n", dt);
    printf("  num_steps = %d\n", num_steps);
    printf("  total_time = %.3e s\n", dt * num_steps);
    printf("========================================\n");

    // Material properties (Ti6Al4V solid)
    MaterialProperties mat;
    mat.name = "Ti6Al4V";
    mat.rho_solid = 4420.0f;
    mat.rho_liquid = 4110.0f;
    mat.cp_solid = 670.0f;
    mat.cp_liquid = 831.0f;
    mat.T_solidus = 1878.0f;
    mat.T_liquidus = 1928.0f;
    mat.L_fusion = 286000.0f;
    mat.L_vaporization = 9830000.0f;
    mat.T_vaporization = 3560.0f;

    // Thermal diffusivity: α = k / (ρ·cp)
    // For Ti6Al4V: k ≈ 7 W/(m·K), α ≈ 2.36e-6 m²/s
    float k_thermal = 7.0f;  // W/(m·K)
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    // Create thermal solver
    ThermalLBM thermal(NX, NY, NZ, mat, alpha, false);  // No phase change for this test
    thermal.initialize(T_INIT);

    // Create laser source at domain center
    LaserSource laser(
        LASER_POWER / 0.2f,  // 50W nominal power with 20% absorptivity = 10W absorbed
        50e-6f,              // 50 μm spot radius
        0.2f,                // 20% absorptivity
        10e-6f               // 10 μm penetration depth
    );

    // Position laser at domain center
    float x_center = NX * DX / 2.0f;
    float y_center = NY * DX / 2.0f;
    laser.setPosition(x_center, y_center, 0.0f);
    laser.setScanVelocity(0.0f, 0.0f);  // Stationary

    // Allocate device memory for heat source
    int num_cells = NX * NY * NZ;
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));

    // Compute volumetric heat source (only once - laser is stationary)
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (NX + threads.x - 1) / threads.x,
        (NY + threads.y - 1) / threads.y,
        (NZ + threads.z - 1) / threads.z
    );

    // CRITICAL: Laser kernel computes heat source Q [W/m³]
    // This is a RATE (energy per time per volume), independent of dt
    computeLaserHeatSourceKernel<<<blocks, threads>>>(
        d_heat_source,
        laser,
        DX, DX, DX,
        NX, NY, NZ
    );
    cudaDeviceSynchronize();

    // Verify total power
    std::vector<float> h_heat_source(num_cells);
    cudaMemcpy(h_heat_source.data(), d_heat_source, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float total_power = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        total_power += h_heat_source[i];
    }
    total_power *= (DX * DX * DX);  // Convert Q [W/m³] to P [W]

    printf("\nLaser diagnostics:\n");
    printf("  Nominal power: %.2f W\n", LASER_POWER / 0.2f);
    printf("  Absorbed power (expected): %.2f W\n", LASER_POWER);
    printf("  Absorbed power (computed): %.2f W\n", total_power);
    printf("  Power error: %.1f%%\n", 100.0f * fabs(total_power - LASER_POWER) / LASER_POWER);

    // Time integration loop
    printf("\nTime integration:\n");
    printf("  Step      Time[μs]   T_max[K]    T_center[K]  Energy[J]\n");
    printf("  ----      --------   --------    -----------  ---------\n");

    for (int step = 0; step < num_steps; ++step) {
        // CRITICAL: addHeatSource(Q, dt) should compute:
        //   dT = (Q [W/m³] * dt [s]) / (ρ * cp)
        //
        // If dt is missing or applied incorrectly, we get temporal divergence
        thermal.addHeatSource(d_heat_source, dt);

        // BGK collision (pure diffusion)
        thermal.collisionBGK(nullptr, nullptr, nullptr);

        // Streaming
        thermal.streaming();

        // Update temperature
        thermal.computeTemperature();

        // Diagnostics every 5 steps
        if ((step + 1) % 5 == 0) {
            std::vector<float> h_temp(num_cells);
            thermal.copyTemperatureToHost(h_temp.data());

            float T_max = *std::max_element(h_temp.begin(), h_temp.end());
            int center_idx = (NZ/2) * NX * NY + (NY/2) * NX + (NX/2);
            float T_center = h_temp[center_idx];

            // Compute total thermal energy
            float total_energy = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                // Sensible energy: E = ρ * cp * (T - T_ref) * V
                float dT = h_temp[i] - T_INIT;
                total_energy += mat.rho_solid * mat.cp_solid * dT * (DX * DX * DX);
            }

            float time_us = (step + 1) * dt * 1e6f;
            printf("  %4d      %8.3f   %8.1f    %8.1f     %.3e\n",
                   step + 1, time_us, T_max, T_center, total_energy);
        }
    }

    // Final diagnostics
    std::vector<float> h_temp_final(num_cells);
    thermal.copyTemperatureToHost(h_temp_final.data());

    float T_max = *std::max_element(h_temp_final.begin(), h_temp_final.end());
    int center_idx = (NZ/2) * NX * NY + (NY/2) * NX + (NX/2);
    float T_center = h_temp_final[center_idx];

    float total_energy = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float dT = h_temp_final[i] - T_INIT;
        total_energy += mat.rho_solid * mat.cp_solid * dT * (DX * DX * DX);
    }

    float total_time = dt * num_steps;
    float expected_energy = LASER_POWER * total_time;  // P * t = E

    printf("\n========================================\n");
    printf("FINAL RESULTS (%s):\n", label);
    printf("  T_max = %.1f K\n", T_max);
    printf("  T_center = %.1f K\n", T_center);
    printf("  Total energy: %.3e J\n", total_energy);
    printf("  Expected energy (P*t): %.3e J\n", expected_energy);
    printf("  Energy error: %.1f%%\n", 100.0f * fabs(total_energy - expected_energy) / expected_energy);
    printf("========================================\n");

    cudaFree(d_heat_source);
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEMPORAL INTEGRATION BUG DIAGNOSTIC TEST                     ║\n");
    printf("║  Week 2 - Timestep Convergence Study                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    printf("\nObjective:\n");
    printf("  Test if final temperature depends on timestep size.\n");
    printf("  All tests run to t = 3.0 μs with different dt.\n");
    printf("\n");
    printf("Expected (correct behavior):\n");
    printf("  T_max should be SAME for all timesteps (±1%%)\n");
    printf("\n");
    printf("Actual (if bug exists):\n");
    printf("  T_max increases with larger dt → temporal divergence\n");
    printf("\n");

    // Test 1: dt = 0.2 μs
    testTemporalConvergence(0.2e-6f, 15, "COARSE (dt=0.2μs)");

    // Test 2: dt = 0.1 μs
    testTemporalConvergence(0.1e-6f, 30, "MEDIUM (dt=0.1μs)");

    // Test 3: dt = 0.05 μs
    testTemporalConvergence(0.05e-6f, 60, "FINE (dt=0.05μs)");

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  ANALYSIS                                                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("If T_max varies by >5%% between tests:\n");
    printf("  → BUG CONFIRMED: Temporal integration error\n");
    printf("  → Root cause: dt scaling missing in energy deposition\n");
    printf("\n");
    printf("If T_max is consistent (±1%%):\n");
    printf("  → NO BUG: Temporal convergence achieved\n");
    printf("\n");

    return 0;
}
