/**
 * @file test_enthalpy_diag.cu
 * @brief Standalone energy-budget closure test for computeTotalThermalEnergy
 *
 * Validates that the newly-fixed ThermalFDM::computeTotalThermalEnergy(dx) closes
 * the energy budget to <2% across the 316L solid→mushy→liquid phase transition.
 *
 * Design:
 *   Domain : 16×16×16 cells, dx = 5e-6 m  (80 μm cube)
 *   Material: 316L (MaterialDatabase::get316L())
 *   T_init : 300 K
 *   Power  : 50 W uniform volumetric source (no VOF fill — heat all cells)
 *   Q_vol  : 50 / (16³ × (5e-6)³) ≈ 9.77e13 W/m³
 *   dt     : 5e-7 s   (Fo = 0.081, well within 1/6 limit)
 *   Steps  : 200 → 100 μs total
 *
 * Back-of-envelope (316L, adiabatic walls):
 *   E_in(100 μs) = 5.0e-3 J
 *   E_to_solidus ≈ 2.78e-3 J  → phase change starts at ~55.6 μs
 *   E_mushy      ≈ 9.91e-4 J  → fully liquid at ~75.4 μs
 *   T_final      ≈ 2150 K  (in target window 1900–2200 K)
 *
 * Acceptance: |rel_residual| < 2% at every checkpoint where E_in > 0.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "physics/thermal_fdm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// ============================================================================
// CUDA kernel: fill uniform volumetric heat source array
// ============================================================================
__global__ void fillUniformHeatSourceKernel(float* Q, float Q_vol, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    Q[idx] = Q_vol;
}

// ============================================================================
// Host helper: compute mean / min / max temperature and mean liquid fraction
// ============================================================================
static void computeFieldStats(const std::vector<float>& h_T,
                               const std::vector<float>& h_fl,
                               int num_cells,
                               float& T_mean, float& T_min, float& T_max,
                               float& fl_mean)
{
    T_min = 1e9f; T_max = -1e9f;
    double T_sum = 0.0, fl_sum = 0.0;
    for (int i = 0; i < num_cells; ++i) {
        float T = h_T[i];
        float fl = h_fl[i];
        if (T < T_min) T_min = T;
        if (T > T_max) T_max = T;
        T_sum  += T;
        fl_sum += fl;
    }
    T_mean  = static_cast<float>(T_sum  / num_cells);
    fl_mean = static_cast<float>(fl_sum / num_cells);
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    // -----------------------------------------------------------------------
    // Domain and material
    // -----------------------------------------------------------------------
    const int   NX  = 16;
    const int   NY  = 16;
    const int   NZ  = 16;
    const float dx  = 5.0e-6f;   // 5 μm
    const int   NC  = NX * NY * NZ;
    const float dV  = dx * dx * dx;

    MaterialProperties mat = MaterialDatabase::get316L();

    // Thermal diffusivity (solid-phase, used for dt selection and FDM ctor)
    const float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    // dt = 5e-7 s  →  Fo = alpha*dt/dx² ≈ 0.081  (well below 1/6)
    const float dt    = 5.0e-7f;

    // -----------------------------------------------------------------------
    // Uniform heat source: 50 W total absorbed into all 16³ cells
    // -----------------------------------------------------------------------
    const float P_total = 50.0f;                            // W
    const float Q_vol   = P_total / (NC * dV);             // W/m³

    // -----------------------------------------------------------------------
    // Print setup
    // -----------------------------------------------------------------------
    printf("=== Enthalpy Budget Diagnostic: ThermalFDM + 316L Phase Change ===\n");
    printf("Domain  : %d x %d x %d cells, dx = %.1f um\n", NX, NY, NZ, dx * 1e6f);
    printf("Material: %s\n", mat.name);
    printf("  rho_s=%g kg/m3, cp_s=%g J/(kg·K), k_s=%g W/(m·K)\n",
           mat.rho_solid, mat.cp_solid, mat.k_solid);
    printf("  T_solidus=%.1f K, T_liquidus=%.1f K, L_fusion=%g J/kg\n",
           mat.T_solidus, mat.T_liquidus, mat.L_fusion);
    printf("Power   : %.1f W  (Q_vol = %.4e W/m³)\n", P_total, Q_vol);
    printf("dt      : %.1e s  (Fo = %.4f, limit 1/6 = %.4f)\n",
           dt, alpha * dt / (dx * dx), 1.0f / 6.0f);
    printf("Steps   : 200 = 100 us\n");
    printf("\n");

    // Back-of-envelope trajectory
    const float rho_avg = 0.5f * (mat.rho_solid + mat.rho_liquid);
    const float E_to_solidus = mat.rho_solid * mat.cp_solid
                                * (mat.T_solidus - 300.0f) * NC * dV;
    const float E_mushy      = rho_avg * mat.L_fusion * NC * dV;
    printf("Back-of-envelope:\n");
    printf("  E_in(100us)      = %.4e J\n", P_total * 100e-6f);
    printf("  E_to_solidus     = %.4e J  (phase change starts at t = %.1f us)\n",
           E_to_solidus, E_to_solidus / P_total * 1e6f);
    printf("  E_mushy (latent) = %.4e J  (fully liquid at t = %.1f us)\n",
           E_mushy, (E_to_solidus + E_mushy) / P_total * 1e6f);
    printf("  Expected T_final ~ 2150 K  (target 1900-2200 K)\n");
    printf("\n");

    // -----------------------------------------------------------------------
    // Allocate device heat-source array
    // -----------------------------------------------------------------------
    float* d_Q = nullptr;
    cudaMalloc(&d_Q, NC * sizeof(float));
    {
        int threads = 256;
        int blocks  = (NC + threads - 1) / threads;
        fillUniformHeatSourceKernel<<<blocks, threads>>>(d_Q, Q_vol, NC);
        cudaDeviceSynchronize();
    }

    // -----------------------------------------------------------------------
    // Create ThermalFDM solver (phase change enabled)
    // -----------------------------------------------------------------------
    ThermalFDM thermal(NX, NY, NZ, mat, alpha, /*enable_phase_change=*/true, dt, dx);
    thermal.initialize(300.0f);
    thermal.setSkipTemperatureCap(true);
    // Adiabatic BC: default stencil clamping — no applyFaceThermalBC calls needed.
    // d_vof_fill_ left null → heat source applied to all cells (fill check skipped).

    // -----------------------------------------------------------------------
    // Host buffers for field extraction
    // -----------------------------------------------------------------------
    std::vector<float> h_T(NC), h_fl(NC);

    // -----------------------------------------------------------------------
    // Checkpoint table header
    // -----------------------------------------------------------------------
    printf("%-8s  %-12s  %-12s  %-12s  %-10s  %-8s  %-8s  %-8s  %-8s  %-6s\n",
           "t[us]", "E_in[J]", "E_meas[J]", "residual[J]", "rel[%]",
           "T_mean[K]", "T_min[K]", "T_max[K]", "fl_mean", "PASS");
    printf("%-8s  %-12s  %-12s  %-12s  %-10s  %-8s  %-8s  %-8s  %-8s  %-6s\n",
           "--------", "------------", "------------", "------------", "----------",
           "--------", "--------", "--------", "--------", "------");

    // CSV output
    FILE* fcsv = fopen("enthalpy_diag.csv", "w");
    if (!fcsv) {
        fprintf(stderr, "ERROR: cannot open enthalpy_diag.csv\n");
        cudaFree(d_Q);
        return 1;
    }
    fprintf(fcsv, "t_us,E_in_J,E_meas_J,residual_J,rel_pct,T_mean_K,T_min_K,T_max_K,fl_mean,pass\n");

    // -----------------------------------------------------------------------
    // Time-stepping loop — 200 steps, checkpoint every 20 (= 10 μs)
    // -----------------------------------------------------------------------
    const int TOTAL_STEPS = 200;
    const int CKPT_STEP   = 20;     // every 20 steps = 10 μs
    const float PASS_THRESHOLD = 2.0f;  // %

    int  all_pass    = 1;
    int  n_ckpts     = 0;
    int  n_fail      = 0;

    for (int step = 1; step <= TOTAL_STEPS; ++step) {
        // FDM step: snapshot T → add heat → advDiff → swap → ESM phase change
        // Step order matches ThermalFDM contract:
        //   storePreviousTemperature → addHeatSource → collisionBGK → streaming → computeTemperature
        thermal.storePreviousTemperature();  // required by bisection ESM
        thermal.addHeatSource(d_Q, dt);
        thermal.collisionBGK();   // pure diffusion (no velocity field)
        thermal.streaming();      // buffer swap
        thermal.computeTemperature();  // ESM correction (phase change)
        // Adiabatic BC: automatic via stencil index clamping — no call needed.

        if (step % CKPT_STEP == 0) {
            float t_us  = step * dt * 1e6f;
            float E_in  = P_total * step * dt;

            float E_meas = thermal.computeTotalThermalEnergy(dx);

            float residual = E_meas - E_in;
            float rel_pct  = (E_in > 0.0f) ? (residual / E_in * 100.0f) : 0.0f;
            float rel_abs  = fabsf(rel_pct);

            thermal.copyTemperatureToHost(h_T.data());
            thermal.copyLiquidFractionToHost(h_fl.data());

            float T_mean, T_min, T_max, fl_mean;
            computeFieldStats(h_T, h_fl, NC, T_mean, T_min, T_max, fl_mean);

            int pass = (rel_abs < PASS_THRESHOLD) ? 1 : 0;
            if (!pass) { all_pass = 0; ++n_fail; }
            ++n_ckpts;

            printf("%-8.1f  %-12.5e  %-12.5e  %-+12.5e  %-+10.4f  %-8.1f  %-8.1f  %-8.1f  %-8.4f  %-6s\n",
                   t_us, E_in, E_meas, residual, rel_pct,
                   T_mean, T_min, T_max, fl_mean,
                   pass ? "PASS" : "FAIL");

            fprintf(fcsv, "%.1f,%.8e,%.8e,%.8e,%.6f,%.4f,%.4f,%.4f,%.6f,%s\n",
                    t_us, E_in, E_meas, residual, rel_pct,
                    T_mean, T_min, T_max, fl_mean,
                    pass ? "PASS" : "FAIL");
        }
    }

    fclose(fcsv);
    cudaFree(d_Q);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n=== SUMMARY ===\n");
    printf("Checkpoints : %d\n", n_ckpts);
    printf("Passed      : %d\n", n_ckpts - n_fail);
    printf("Failed      : %d\n", n_fail);
    printf("Threshold   : %.1f%%\n", PASS_THRESHOLD);
    printf("Overall     : %s\n", all_pass ? "PASS" : "FAIL");
    printf("CSV output  : enthalpy_diag.csv\n");

    return all_pass ? 0 : 1;
}
