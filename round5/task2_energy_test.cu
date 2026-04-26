/**
 * Round 5 Task 2: Energy Diagnostic Validation (Test C)
 *
 * Setup: VOF interface + Gas Wipe, no heat source, no flow
 * Domain: 20x20x20, f=1 below z=10, f=0 above (sharp interface)
 * Run 5000 steps. Track energy via computeTotalThermalEnergy().
 * Verify: energy residual < 5% after accounting for gas wipe energy.
 *
 * The "100-200% error" from before was because gas wipe was removing
 * energy but the diagnostic didn't track it. This test proves the fix.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "physics/thermal_fdm.h"
#include "physics/material_properties.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

int main() {
    const int NX = 20, NY = 20, NZ = 20;
    const int N = NX * NY * NZ;
    const float dx = 2.5e-6f, dt = 1.0e-8f;
    const float rho = 7900, cp = 700, k_c = 20;
    const float alpha = k_c / (rho * cp);
    const float dV = dx * dx * dx;

    MaterialProperties mat = {};
    mat.rho_solid = rho; mat.rho_liquid = rho;
    mat.cp_solid = cp; mat.cp_liquid = cp;
    mat.k_solid = k_c; mat.k_liquid = k_c;
    mat.T_solidus = 1650; mat.T_liquidus = 1700;
    mat.T_vaporization = 3200;
    mat.L_fusion = 270000; mat.L_vaporization = 6.09e6f;
    mat.molar_mass = 0.05585f;

    ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

    // VOF fill level: f=1 below z=10, f=0 above (sharp interface)
    std::vector<float> h_fill(N, 0.0f);
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
                if (k < 10)
                    h_fill[i + j * NX + k * NX * NY] = 1.0f;

    float* d_fill;
    CUDA_CHECK(cudaMalloc(&d_fill, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fill, h_fill.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Initial temperature: metal at 2000K, gas at 1000K (above T_ambient=600K)
    // This ensures gas wipe will actively cool gas cells
    std::vector<float> h_T(N);
    for (int i = 0; i < N; i++)
        h_T[i] = (h_fill[i] >= 0.01f) ? 2000.0f : 1000.0f;

    thermal.initialize(h_T.data());
    thermal.setVOFFillLevel(d_fill);

    printf("================================================================\n");
    printf("  Task 2: Energy Diagnostic Test C\n");
    printf("  VOF interface + Gas Wipe, no heat source, no flow\n");
    printf("  Domain: %d^3, dx=%.1fμm, dt=%.0fns\n", NX, dx * 1e6f, dt * 1e9f);
    printf("  Metal: %d cells at 2000K, Gas: %d cells at 1000K (T_amb=600K)\n",
           (int)std::count_if(h_fill.begin(), h_fill.end(), [](float f) { return f >= 0.01f; }),
           (int)std::count_if(h_fill.begin(), h_fill.end(), [](float f) { return f < 0.01f; }));
    printf("================================================================\n\n");

    const int total_steps = 5000;
    const int record_interval = 500;

    // Energy tracking
    float E_prev = thermal.computeTotalThermalEnergy(dx);
    double total_gas_wipe_energy = 0.0;
    double total_boiling_cap_energy = 0.0;

    FILE* f_csv = fopen("/home/yzk/LBMProject/round5/data/energy_residual_test_c.csv", "w");
    fprintf(f_csv, "step,t_us,E_thermal_J,dE_J,Q_gas_wipe_J,Q_boil_cap_J,residual_pct\n");

    printf("%-6s %8s %14s %14s %14s %14s %10s\n",
           "Step", "t[μs]", "E_therm[J]", "dE[J]", "Q_gw[J]", "Q_bc[J]", "Res[%]");
    printf("----------------------------------------------------------------------\n");

    for (int step = 1; step <= total_steps; step++) {
        // FDM step: collision + streaming (no heat source, no velocity)
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();

        // No evaporation cooling (no laser, T < T_boil)
        // No boiling cap needed (T < T_boil)

        if (step % record_interval == 0) {
            float E_now = thermal.computeTotalThermalEnergy(dx);
            float dE = E_now - E_prev;

            // Get tracked energy sinks
            double Q_gw = thermal.getGasWipeEnergyRemoved();
            double Q_bc = thermal.getBoilingCapEnergyRemoved();
            total_gas_wipe_energy += Q_gw;
            total_boiling_cap_energy += Q_bc;

            // Energy balance over this interval:
            // No heat source: P_in = 0
            // dE = -Q_gas_wipe - Q_boiling_cap (only energy sinks)
            // Residual = |dE + Q_gw + Q_bc| / |dE| (relative to energy change)
            float expected_dE = -(Q_gw + Q_bc);
            float residual;
            if (fabs(dE) > 1e-12f)
                residual = fabs(dE - expected_dE) / fabs(dE) * 100.0f;
            else if (fabs(Q_gw + Q_bc) > 1e-12f)
                residual = 100.0f; // dE ~0 but sinks are not
            else
                residual = 0.0f;

            float t_us = step * dt * 1e6f;
            printf("%-6d %8.1f %14.6e %14.6e %14.6e %14.6e %10.2f\n",
                   step, t_us, E_now, dE, Q_gw, Q_bc, residual);

            fprintf(f_csv, "%d,%.3f,%.6e,%.6e,%.6e,%.6e,%.2f\n",
                    step, t_us, E_now, dE, Q_gw, Q_bc, residual);

            E_prev = E_now;
        }
    }

    fclose(f_csv);

    // Final summary
    printf("\n================================================================\n");
    printf("  Total gas wipe energy removed: %.6e J\n", total_gas_wipe_energy);
    printf("  Total boiling cap energy removed: %.6e J\n", total_boiling_cap_energy);
    float E_final = thermal.computeTotalThermalEnergy(dx);
    float E_init = 0;
    for (int i = 0; i < N; i++)
        E_init += rho * cp * h_T[i] * dV;
    printf("  E_initial: %.6e J\n", (double)E_init);
    printf("  E_final:   %.6e J\n", (double)E_final);
    printf("  E_change:  %.6e J\n", (double)(E_final - E_init));
    printf("  Energy sinks: %.6e J\n", total_gas_wipe_energy + total_boiling_cap_energy);

    float total_change = E_init - E_final;
    float total_sinks = total_gas_wipe_energy + total_boiling_cap_energy;
    float global_residual = fabs(total_change - total_sinks) / total_change * 100.0f;
    printf("  Global residual: %.2f%%\n", global_residual);
    if (global_residual < 5.0f)
        printf("  PASS (< 5%%)\n");
    else
        printf("  FAIL (>= 5%%)\n");
    printf("================================================================\n");

    cudaFree(d_fill);
    return (global_residual < 5.0f) ? 0 : 1;
}
