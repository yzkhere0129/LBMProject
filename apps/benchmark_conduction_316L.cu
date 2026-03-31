/**
 * @file benchmark_conduction_316L.cu
 * @brief Pure Conduction + Phase Change Benchmark — 316L Spot Melting
 *
 * Cross-platform verification against OpenFOAM buoyantPimpleFoam.
 * Reference: benchmark_conduction.json (v1.1, 2026-03-25)
 *
 * Physics:
 *   - Heat conduction (D3Q7 thermal LBM)
 *   - Melting/solidification (ESM: enthalpy source term, Jiaung 2001)
 *   - NO fluid flow (velocity = 0 everywhere)
 *   - NO gas phase, NO ray tracing, NO radiation, NO evaporation
 *
 * Domain (LBM coordinates):
 *   x: [0, 200 um]  NX=100   (JSON x: surface direction 1)
 *   y: [0, 200 um]  NY=100   (JSON z: surface direction 2)
 *   z: [0, 100 um]  NZ=50    (JSON y: depth, z=NZ-1 = surface)
 *
 * Heat source:
 *   Gaussian surface flux converted to volumetric source in top layer (z=NZ-1):
 *     Q_v(x,y) = [2*eta*P / (pi*r0^2)] * exp(-2*r^2/r0^2) / dz   [W/m^3]
 *   Laser ON: t in [0, 50 us],  OFF: t in (50, 100 us]
 *
 * Boundary conditions:
 *   z_max (top):    adiabatic (laser enters as volumetric source)
 *   z_min (bottom): Dirichlet T=300 K
 *   x_min, x_max, y_min, y_max: Dirichlet T=300 K
 *
 * Step ordering (single ESM per step):
 *   collision → streaming → inject_heat → computeTemperature(ESM) → BCs
 *
 * Output: CSV with probe time series and centerline profiles at snapshots
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// ============================================================================
// Heat source kernel: inject + force equilibrium
//
// The standard BGK source injection g_q += w_q*dT is unstable when
// dT/T_local > 0.1 (exponential blowup for omega > 1).
// For concentrated laser sources, dT/step >> T_local at early times.
//
// Fix: compute T_new = (sum g_q) + dT, then SET g_q = w_q * T_new.
// This forces equilibrium at heated cells, preventing non-equilibrium
// amplification. Diffusion is preserved because streaming brings
// non-equilibrium information from neighbor cells at the next step.
//
// Chapman-Enskog note: since g_q is set (not incremented), no
// (1-omega/2) correction is needed — the zeroth moment is exact.
// ============================================================================
__global__ void injectHeatAndEquilibrateKernel(
    float* g,                   // D3Q7 distributions (SoA: g[q*N + idx])
    const float* heat_source,   // Volumetric source [W/m^3]
    float dt_over_rhoCp,        // dt / (rho * cp)
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float Q = heat_source[idx];
    if (Q <= 0.0f) return;

    // Compute current T from distributions
    float T_star = 0.0f;
    for (int q = 0; q < 7; ++q) {
        T_star += g[q * num_cells + idx];
    }

    // Add heat source
    float dT = Q * dt_over_rhoCp;
    float T_new = T_star + dT;

    // Force equilibrium at T_new (eliminates non-eq amplification)
    g[0 * num_cells + idx] = 0.25f  * T_new;
    g[1 * num_cells + idx] = 0.125f * T_new;
    g[2 * num_cells + idx] = 0.125f * T_new;
    g[3 * num_cells + idx] = 0.125f * T_new;
    g[4 * num_cells + idx] = 0.125f * T_new;
    g[5 * num_cells + idx] = 0.125f * T_new;
    g[6 * num_cells + idx] = 0.125f * T_new;
}

// ============================================================================
// Benchmark 316L material — CONSTANT properties from benchmark_conduction.json
// ============================================================================
static MaterialProperties createBenchmark316L() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_Benchmark_Const", sizeof(mat.name) - 1);

    // Constant properties (solid = liquid, no temperature dependence)
    mat.rho_solid  = 7900.0f;   mat.rho_liquid  = 7900.0f;
    mat.cp_solid   = 700.0f;    mat.cp_liquid   = 700.0f;
    mat.k_solid    = 20.0f;     mat.k_liquid    = 20.0f;
    mat.mu_liquid  = 6.0e-3f;   // unused (no fluid)

    // Phase change
    mat.T_solidus      = 1650.0f;   // K
    mat.T_liquidus     = 1700.0f;   // K
    mat.T_vaporization = 3500.0f;   // K (irrelevant — no evaporation)
    mat.L_fusion       = 260000.0f; // J/kg
    mat.L_vaporization = 7.0e6f;    // unused
    mat.molar_mass     = 0.0558f;   // kg/mol (unused)

    // Optical/surface (unused — heat source computed externally)
    mat.surface_tension     = 1.75f;
    mat.dsigma_dT           = -4.3e-4f;
    mat.absorptivity_solid  = 0.35f;
    mat.absorptivity_liquid = 0.35f;
    mat.emissivity          = 0.3f;

    return mat;
}

// ============================================================================
// Probe location in LBM cell indices
// ============================================================================
struct Probe {
    const char* label;
    int ix, iy, iz;  // cell indices
};

int main() {
    auto wall_start = std::chrono::high_resolution_clock::now();

    printf("\n");
    printf("============================================================\n");
    printf("  Pure Conduction + Phase Change Benchmark\n");
    printf("  316L Stainless Steel — Stationary Gaussian Spot\n");
    printf("  Cross-platform verification (LBM vs OpenFOAM)\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Material
    // ==================================================================
    MaterialProperties mat = createBenchmark316L();
    const float rho    = mat.rho_solid;
    const float cp     = mat.cp_solid;
    const float k      = mat.k_solid;
    const float alpha  = k / (rho * cp);  // 3.616e-6 m^2/s
    const float L_f    = mat.L_fusion;
    const float T_sol  = mat.T_solidus;
    const float T_liq  = mat.T_liquidus;
    const float T_init = 300.0f;          // K

    printf("Material: %s\n", mat.name);
    printf("  rho = %.0f kg/m^3 (constant)\n", rho);
    printf("  cp  = %.0f J/(kg*K) (constant)\n", cp);
    printf("  k   = %.1f W/(m*K) (constant)\n", k);
    printf("  alpha = %.4e m^2/s\n", alpha);
    printf("  L_fusion = %.0f J/kg\n", L_f);
    printf("  T_solidus = %.0f K,  T_liquidus = %.0f K  (dT = %.0f K)\n",
           T_sol, T_liq, T_liq - T_sol);
    printf("  C_app(mushy) = %.0f J/(kg*K)\n", cp + L_f / (T_liq - T_sol));

    // ==================================================================
    // Laser parameters
    // ==================================================================
    const float P_laser  = 150.0f;     // W
    const float eta      = 0.35f;      // absorptivity
    const float r0       = 25.0e-6f;   // 1/e^2 beam radius [m]
    const float t_on     = 50.0e-6f;   // laser ON duration [s]
    const float t_total  = 100.0e-6f;  // total simulation time [s]

    const float q0 = 2.0f * eta * P_laser / (static_cast<float>(M_PI) * r0 * r0);
    printf("\nLaser:\n");
    printf("  P = %.0f W, eta = %.2f, r0 = %.0f um\n", P_laser, eta, r0 * 1e6f);
    printf("  q0 (peak absorbed flux) = %.3e W/m^2\n", q0);
    printf("  Schedule: ON [0, %.0f us], OFF (%.0f, %.0f us]\n",
           t_on * 1e6f, t_on * 1e6f, t_total * 1e6f);

    // ==================================================================
    // Domain & LBM discretization
    // ==================================================================
    const int NX = 100;   // x: surface direction 1
    const int NY = 100;   // y: surface direction 2
    const int NZ = 50;    // z: depth (z=NZ-1 = surface)
    const float dx = 2.0e-6f;  // 2 um

    // tau=1.0 (omega=1.0): no overrelaxation, unconditionally stable.
    // At omega=1, collision sets g_q = w_q*T exactly (full relaxation to eq).
    // This is equivalent to explicit FD diffusion, stable for alpha_LU < 1/6.
    const float tau = 1.0f;
    const float alpha_LU = (tau - 0.5f) * D3Q7::CS2;  // 0.025
    const float dt = alpha_LU * dx * dx / alpha;

    const int total_steps = static_cast<int>(t_total / dt + 0.5f);
    const int laser_off_step = static_cast<int>(t_on / dt + 0.5f);

    printf("\nDomain: %d x %d x %d = %d cells  (dx = %.0f um)\n",
           NX, NY, NZ, NX * NY * NZ, dx * 1e6f);
    printf("LBM parameters:\n");
    printf("  tau = %.2f,  omega = %.4f,  alpha_LU = %.4f\n",
           tau, 1.0f / tau, alpha_LU);
    printf("  dt = %.4e s (%.2f ns)\n", dt, dt * 1e9f);
    printf("  Total steps = %d,  Laser OFF at step %d\n", total_steps, laser_off_step);

    // ==================================================================
    // Snapshot times (user request: 25, 50, 60, 75 us)
    // ==================================================================
    struct Snapshot {
        float time;       // [s]
        int step;         // time step number
        const char* tag;
    };
    Snapshot snapshots[] = {
        { 25.0e-6f,  0, "25us"  },
        { 50.0e-6f,  0, "50us"  },
        { 60.0e-6f,  0, "60us"  },
        { 75.0e-6f,  0, "75us"  },
        {100.0e-6f,  0, "100us" },
    };
    const int n_snap = 5;
    for (int i = 0; i < n_snap; i++) {
        snapshots[i].step = static_cast<int>(snapshots[i].time / dt + 0.5f);
    }

    // ==================================================================
    // Probe locations (from benchmark JSON, mapped to LBM cell indices)
    // ==================================================================
    const int cx = NX / 2;  // 50 — center x
    const int cy = NY / 2;  // 50 — center y

    Probe probes[] = {
        { "surface_center",  cx,     cy,     NZ - 1 },
        { "5um_depth",       cx,     cy,     NZ - 1 - 3 },
        { "20um_depth",      cx,     cy,     NZ - 1 - 10 },
        { "50um_depth",      cx,     cy,     NZ - 1 - 25 },
        { "surface_1r0",     cx + 13, cy,    NZ - 1 },
        { "surface_2r0",     cx + 25, cy,    NZ - 1 },
    };
    const int n_probes = 6;

    printf("\nProbe locations (cell indices):\n");
    for (int p = 0; p < n_probes; p++) {
        printf("  %-18s  (%3d, %3d, %3d)\n",
               probes[p].label, probes[p].ix, probes[p].iy, probes[p].iz);
    }

    // ==================================================================
    // Initialize D3Q7 lattice
    // ==================================================================
    if (!D3Q7::isInitialized()) {
        D3Q7::initializeDevice();
    }

    // ==================================================================
    // Create thermal solver with phase change
    // ==================================================================
    // Phase change ON for benchmark
    const bool enable_phase_change = true;
    ThermalLBM thermal(NX, NY, NZ, mat, alpha, enable_phase_change, dt, dx);
    thermal.initialize(T_init);

    // Disable temperature safety cap (no evaporation physics here)
    thermal.setSkipTemperatureCap(true);

    // ==================================================================
    // Prepare Gaussian heat source array (host → device)
    // Only top layer z=NZ-1 has non-zero Q
    // ==================================================================
    const int num_cells = NX * NY * NZ;
    std::vector<float> h_heat_source(num_cells, 0.0f);

    const int iz_top = NZ - 1;
    float total_absorbed_power = 0.0f;
    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
            float x_phys = (static_cast<float>(ix) - cx) * dx;
            float y_phys = (static_cast<float>(iy) - cy) * dx;
            float r2 = x_phys * x_phys + y_phys * y_phys;

            float q_surf = q0 * expf(-2.0f * r2 / (r0 * r0));
            float Q_vol = q_surf / dx;

            int idx = ix + iy * NX + iz_top * NX * NY;
            h_heat_source[idx] = Q_vol;
            total_absorbed_power += q_surf * dx * dx;
        }
    }

    printf("\nHeat source verification:\n");
    printf("  Expected absorbed power: %.2f W  (eta*P = %.2f W)\n",
           total_absorbed_power, eta * P_laser);
    printf("  Peak Q_vol = %.3e W/m^3\n", q0 / dx);
    printf("  Peak dT/step = %.1f K\n", (q0 / dx) * dt / (rho * cp));

    float* d_heat_source = nullptr;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));
    cudaMemcpy(d_heat_source, h_heat_source.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Pre-compute constants for heat injection
    const float dt_over_rhoCp = dt / (rho * cp);

    // CUDA launch config for heat source kernel
    const int blockSize_heat = 256;
    const int gridSize_heat = (num_cells + blockSize_heat - 1) / blockSize_heat;

    // ==================================================================
    // Open output files
    // ==================================================================
    FILE* f_probes = fopen("benchmark_conduction_probes.csv", "w");
    FILE* f_profiles = fopen("benchmark_conduction_profiles.csv", "w");

    if (!f_probes || !f_profiles) {
        fprintf(stderr, "ERROR: Cannot open output files\n");
        return 1;
    }

    fprintf(f_probes, "# Pure Conduction Benchmark: Probe Time Series\n");
    fprintf(f_probes, "# dt=%.6e dx=%.6e tau=%.3f NX=%d NY=%d NZ=%d\n",
            dt, dx, tau, NX, NY, NZ);
    fprintf(f_probes, "step,time_s,time_us,T_max_K");
    for (int p = 0; p < n_probes; p++) {
        fprintf(f_probes, ",T_%s", probes[p].label);
    }
    fprintf(f_probes, "\n");

    fprintf(f_profiles, "# Pure Conduction Benchmark: Centerline Profiles\n");
    fprintf(f_profiles, "# Columns: snapshot_tag, z_index, depth_um, T_K, liquid_fraction\n");
    fprintf(f_profiles, "snapshot,z_idx,depth_um,T_K,fl\n");

    // ==================================================================
    // Host buffers
    // ==================================================================
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fl(num_cells);

    // ==================================================================
    // TIME LOOP
    //
    // Step ordering (single ESM per step):
    //   1. collision (BGK, pure diffusion)
    //   2. streaming (propagate + bounce-back at walls)
    //   3. inject heat source into g (if laser ON) — raw kernel, no ESM
    //   4. computeTemperature (T* = Σg_q + ESM correction)
    //   5. apply Dirichlet BCs (overwrite ESM at boundaries)
    // ==================================================================
    printf("\n%-6s  %8s  %8s  %10s  %10s  %10s  %10s\n",
           "step", "t [us]", "laser", "T_max [K]", "T_surf [K]",
           "depth_sol", "depth_liq");
    printf("------  --------  --------  ----------  ----------  ----------  ----------\n");

    int snap_idx = 0;
    int print_interval = total_steps / 20;
    if (print_interval < 1) print_interval = 1;

    for (int step = 0; step <= total_steps; step++) {
        float t_current = step * dt;
        bool laser_on = (step < laser_off_step);

        // 1. Collision (pure diffusion — no velocity field)
        thermal.collisionBGK();

        // 2. Streaming
        thermal.streaming();

        // 3. Inject heat source directly into distributions (if laser ON)
        if (laser_on) {
            injectHeatAndEquilibrateKernel<<<gridSize_heat, blockSize_heat>>>(
                thermal.getDistributionSrc(), d_heat_source,
                dt_over_rhoCp, num_cells);
            cudaDeviceSynchronize();
        }

        // 4. Compute temperature + ESM phase change correction
        thermal.computeTemperature();

        // 5. Boundary conditions (Dirichlet AFTER ESM)
        //    bc_type: 0=PERIODIC, 1=ADIABATIC, 2=DIRICHLET
        thermal.applyFaceThermalBC(4, 2, dt, dx, T_init);  // z_min: Dirichlet 300K
        thermal.applyFaceThermalBC(0, 2, dt, dx, T_init);  // x_min
        thermal.applyFaceThermalBC(1, 2, dt, dx, T_init);  // x_max
        thermal.applyFaceThermalBC(2, 2, dt, dx, T_init);  // y_min
        thermal.applyFaceThermalBC(3, 2, dt, dx, T_init);  // y_max
        thermal.applyFaceThermalBC(5, 1, dt, dx);           // z_max: adiabatic

        // ============================================================
        // Extract data at print intervals and snapshots
        // ============================================================
        bool is_snapshot = (snap_idx < n_snap && step == snapshots[snap_idx].step);
        bool is_print = (step % print_interval == 0) || is_snapshot || (step == total_steps)
                      ;

        if (is_print || is_snapshot) {
            thermal.copyTemperatureToHost(h_temp.data());

            float T_max = *std::max_element(h_temp.begin(), h_temp.end());

            // Melt pool depth along center column
            float depth_sol_um = 0.0f;
            float depth_liq_um = 0.0f;
            for (int iz = NZ - 1; iz >= 0; iz--) {
                int idx = cx + cy * NX + iz * NX * NY;
                float T = h_temp[idx];
                float depth = static_cast<float>(NZ - 1 - iz) * dx * 1e6f;
                if (T >= T_sol && depth > depth_sol_um) depth_sol_um = depth;
                if (T >= T_liq && depth > depth_liq_um) depth_liq_um = depth;
            }

            int idx_surf = cx + cy * NX + (NZ - 1) * NX * NY;
            float T_surf = h_temp[idx_surf];

            if (is_print) {
                printf("%6d  %8.2f  %8s  %10.1f  %10.1f  %8.1f um  %8.1f um\n",
                       step, t_current * 1e6f, laser_on ? "ON" : "OFF",
                       T_max, T_surf, depth_sol_um, depth_liq_um);
            }

            // Probe time series
            fprintf(f_probes, "%d,%.8e,%.4f,%.2f",
                    step, t_current, t_current * 1e6f, T_max);
            for (int p = 0; p < n_probes; p++) {
                int pidx = probes[p].ix + probes[p].iy * NX + probes[p].iz * NX * NY;
                fprintf(f_probes, ",%.2f", h_temp[pidx]);
            }
            fprintf(f_probes, "\n");
        }

        // ============================================================
        // Snapshot: full centerline profile + melt pool metrics
        // ============================================================
        if (is_snapshot) {
            if (thermal.hasPhaseChange())
                thermal.copyLiquidFractionToHost(h_fl.data());
            else
                std::fill(h_fl.begin(), h_fl.end(), 0.0f);

            printf("\n  >>> SNAPSHOT: %s (step %d, t = %.2f us) <<<\n",
                   snapshots[snap_idx].tag, step, t_current * 1e6f);

            // Centerline z-profile
            for (int iz = 0; iz < NZ; iz++) {
                int idx = cx + cy * NX + iz * NX * NY;
                float depth_um = static_cast<float>(NZ - 1 - iz) * dx * 1e6f;
                fprintf(f_profiles, "%s,%d,%.1f,%.4f,%.6f\n",
                        snapshots[snap_idx].tag, iz, depth_um,
                        h_temp[idx], h_fl[idx]);
            }

            // Melt pool metrics (full domain scan)
            float max_depth_sol = 0.0f, max_depth_liq = 0.0f;
            float max_radius_sol = 0.0f, max_radius_liq = 0.0f;

            for (int iy = 0; iy < NY; iy++) {
                for (int ix = 0; ix < NX; ix++) {
                    float rx = (static_cast<float>(ix) - cx) * dx * 1e6f;
                    float ry = (static_cast<float>(iy) - cy) * dx * 1e6f;
                    float r_um = sqrtf(rx * rx + ry * ry);

                    for (int iz = NZ - 1; iz >= 0; iz--) {
                        int idx = ix + iy * NX + iz * NX * NY;
                        float depth = static_cast<float>(NZ - 1 - iz) * dx * 1e6f;
                        if (h_temp[idx] >= T_sol && depth > max_depth_sol)
                            max_depth_sol = depth;
                        if (h_temp[idx] >= T_liq && depth > max_depth_liq)
                            max_depth_liq = depth;
                    }

                    int idx_top = ix + iy * NX + (NZ - 1) * NX * NY;
                    if (h_temp[idx_top] >= T_sol && r_um > max_radius_sol)
                        max_radius_sol = r_um;
                    if (h_temp[idx_top] >= T_liq && r_um > max_radius_liq)
                        max_radius_liq = r_um;
                }
            }

            printf("  T_max = %.1f K\n", *std::max_element(h_temp.begin(), h_temp.end()));
            printf("  Solidus (T>=1650K): depth = %.1f um, radius = %.1f um\n",
                   max_depth_sol, max_radius_sol);
            printf("  Liquidus (T>=1700K): depth = %.1f um, radius = %.1f um\n",
                   max_depth_liq, max_radius_liq);

            if (t_current > t_on) {
                int idx_5um = cx + cy * NX + (NZ - 1 - 3) * NX * NY;
                printf("  T at 5um depth = %.1f K\n", h_temp[idx_5um]);
            }

            // ============================================================
            // Extract T=1650K (solidus) isotherm contour in the xz mid-plane
            // Coordinate system aligned with OpenFOAM:
            //   X_um = ix * dx_um  (0..198, center at 100)
            //   Z_um = iz * dx_um + Z_OFFSET  (surface at ~149)
            // ============================================================
            const float dx_um = dx * 1e6f;
            const float Z_OFFSET = 51.0f;  // LBM iz=49 → 98+51=149 μm (matches OF surface)
            char contour_fname[128];
            snprintf(contour_fname, sizeof(contour_fname),
                     "lbm_contour_%s.csv", snapshots[snap_idx].tag);
            FILE* f_contour = fopen(contour_fname, "w");
            if (f_contour) {
                fprintf(f_contour, "X_um,Z_um\n");
                // Scan xz-plane at y=cy: find T=1650 crossing by linear interpolation
                // Scan each column ix: find deepest iz where T crosses T_sol
                for (int ix = 0; ix < NX; ix++) {
                    // Z-direction: scan from surface downward
                    for (int iz = NZ - 2; iz >= 0; iz--) {
                        int idx_up  = ix + cy * NX + (iz + 1) * NX * NY;
                        int idx_dn  = ix + cy * NX + iz * NX * NY;
                        float T_up = h_temp[idx_up];
                        float T_dn = h_temp[idx_dn];
                        if ((T_up >= T_sol && T_dn < T_sol) ||
                            (T_up < T_sol && T_dn >= T_sol)) {
                            // Linear interpolation for crossing z-position
                            float frac = (T_sol - T_dn) / (T_up - T_dn);
                            float z_cross = (iz + frac) * dx_um + Z_OFFSET;
                            float x_um = ix * dx_um;
                            fprintf(f_contour, "%.4f,%.4f\n", x_um, z_cross);
                        }
                    }
                    // X-direction: scan each row iz for T crossing along x
                    if (ix < NX - 1) {
                        for (int iz = 0; iz < NZ; iz++) {
                            int idx_l = ix     + cy * NX + iz * NX * NY;
                            int idx_r = (ix+1) + cy * NX + iz * NX * NY;
                            float T_l = h_temp[idx_l];
                            float T_r = h_temp[idx_r];
                            if ((T_l >= T_sol && T_r < T_sol) ||
                                (T_l < T_sol && T_r >= T_sol)) {
                                float frac = (T_sol - T_l) / (T_r - T_l);
                                float x_cross = (ix + frac) * dx_um;
                                float z_um = iz * dx_um + Z_OFFSET;
                                fprintf(f_contour, "%.4f,%.4f\n", x_cross, z_um);
                            }
                        }
                    }
                }
                fclose(f_contour);
                printf("  Contour written: %s\n", contour_fname);
            }

            printf("\n");
            snap_idx++;
        }
    }

    fclose(f_probes);
    fclose(f_profiles);

    cudaFree(d_heat_source);

    auto wall_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(wall_end - wall_start).count();

    printf("\nOutput files:\n");
    printf("  benchmark_conduction_probes.csv   (probe time series)\n");
    printf("  benchmark_conduction_profiles.csv (centerline z-profiles at snapshots)\n");
    printf("\nWall time: %.1f s\n", elapsed);
    printf("Run: python3 scripts/viz/plot_conduction_benchmark.py\n");

    return 0;
}
