/**
 * @file benchmark_spot_melt_316L.cu
 * @brief Spot Melting Benchmark — Conduction + Marangoni Convection + Phase Change
 *
 * Cross-platform verification against OpenFOAM. Same laser/material as the
 * pure-conduction benchmark, but with N-S flow, Marangoni surface stress,
 * and Darcy damping in the mushy/solid zone.
 *
 * Physics ON:
 *   - D3Q19 fluid (EDM collision, TRT stabilization)
 *   - D3Q7 thermal (ESM phase change)
 *   - Marangoni: dσ/dT = +1.0e-4 N/(m·K)  (inward flow → downward jet → deep V pool)
 *   - Darcy mushy-zone damping (Carman-Kozeny)
 *   - Thermal advection (v·∇T coupling)
 *
 * Physics OFF:
 *   - VOF advection (static interface), surface tension (no capillary),
 *   - buoyancy, evaporation, recoil, radiation
 *
 * Domain: 100×100×52  (50 metal + 2 gas-buffer cells at top for VOF interface)
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm;
using namespace lbm::physics;

// ============================================================================
// Benchmark 316L — constant properties (same as conduction benchmark)
// ============================================================================
static MaterialProperties createBenchmark316L() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_Benchmark_Const", sizeof(mat.name) - 1);

    mat.rho_solid  = 7900.0f;   mat.rho_liquid  = 7900.0f;
    mat.cp_solid   = 700.0f;    mat.cp_liquid   = 700.0f;
    mat.k_solid    = 20.0f;     mat.k_liquid    = 20.0f;
    mat.mu_liquid  = 0.005f;    // Pa·s  (benchmark specification)

    mat.T_solidus      = 1650.0f;
    mat.T_liquidus     = 1700.0f;
    mat.T_vaporization = 3200.0f;   // Real 316L boiling point
    mat.L_fusion       = 260000.0f;
    mat.L_vaporization = 7.0e6f;
    mat.molar_mass     = 0.0558f;

    mat.surface_tension     = 1.75f;
    mat.dsigma_dT           = +1.0e-4f;  // POSITIVE: cold→hot inward Marangoni
    mat.absorptivity_solid  = 0.35f;
    mat.absorptivity_liquid = 0.35f;
    mat.emissivity          = 0.3f;

    return mat;
}

int main(int argc, char** argv) {
    auto wall_start = std::chrono::high_resolution_clock::now();

    // Command-line configurable parameters for sweep
    float param_dt = 1.5e-8f;      // default 15 ns
    float param_Cs = 0.10f;        // default Smagorinsky constant
    if (argc >= 2) param_dt = atof(argv[1]);
    if (argc >= 3) param_Cs = atof(argv[2]);

    printf("\n");
    printf("============================================================\n");
    printf("  Spot Melting Benchmark: Conduction + Marangoni Convection\n");
    printf("  316L — positive dσ/dT → inward flow → downward jet\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Material & derived quantities
    // ==================================================================
    MaterialProperties mat = createBenchmark316L();
    const float rho   = mat.rho_solid;
    const float cp    = mat.cp_solid;
    const float k     = mat.k_solid;
    const float mu    = mat.mu_liquid;
    const float alpha = k / (rho * cp);                 // 3.616e-6 m²/s
    const float nu    = mu / rho;                        // 6.329e-7 m²/s

    printf("Material: %s\n", mat.name);
    printf("  ρ=%.0f, cp=%.0f, k=%.1f, μ=%.4f, ν=%.3e\n", rho, cp, k, mu, nu);
    printf("  α=%.4e m²/s,  Pr=ν/α=%.2f\n", alpha, nu / alpha);
    printf("  dσ/dT = %+.1e N/(m·K)  [POSITIVE → inward Marangoni]\n", mat.dsigma_dT);

    // ==================================================================
    // PARAMETER VERIFICATION PRINT (for debugging)
    // ==================================================================
    printf("\n=== PARAMETER VERIFICATION ===\n");
    printf("  dsigma_dT (material):  %+.4e N/(m·K)\n", mat.dsigma_dT);
    printf("  surface_tension:       %.4f N/m\n", mat.surface_tension);
    printf("  T_solidus:             %.0f K\n", mat.T_solidus);
    printf("  T_liquidus:            %.0f K\n", mat.T_liquidus);
    printf("  mu_liquid:             %.4f Pa·s\n", mat.mu_liquid);
    printf("==============================\n");

    // ==================================================================
    // MultiphysicsConfig
    // ==================================================================
    MultiphysicsConfig config;

    // Domain: 100×100×50 (no gas buffer — Inamuro stress BC, no VOF needed)
    const int NX = 100, NY = 100, NZ_METAL = 50, NZ_GAS = 0;
    const int NZ = NZ_METAL;  // 50
    const float dx = 2.0e-6f;

    config.nx = NX;
    config.ny = NY;
    config.nz = NZ;
    config.dx = dx;

    // dt = 15 ns: v_LU < 0.3 at v_phys = 40 m/s.
    // Base τ_f = 0.507 (dangerously low) — Smagorinsky LES dynamically raises
    // τ_eff in high-shear regions via exact algebraic Hou (1996) formula.
    const float dt = param_dt;
    config.dt = dt;

    const float nu_LU = nu * dt / (dx * dx);
    const float tau_fluid = nu_LU / (1.0f / 3.0f) + 0.5f;
    const float Fo = alpha * dt / (dx * dx);

    printf("\nHybrid LBM-FDM + Smagorinsky LES:\n");
    printf("  dx=%.0f μm, dt=%.2f ns\n", dx * 1e6f, dt * 1e9f);
    printf("  Fluid: τ_f_base=%.4f (LES raises in high-shear), ν_LU=%.5f\n", tau_fluid, nu_LU);
    printf("  FDM thermal: Fo=%.4f (limit 0.167)\n", Fo);
    printf("  Ma at 40 m/s = %.3f, at 20 m/s = %.3f\n",
           40.0f * dt / dx, 20.0f * dt / dx);

    config.material = mat;

    // --- Physics flags ---
    config.enable_thermal           = true;
    config.enable_thermal_advection = true;
    config.use_fdm_thermal          = true;   // Hybrid LBM-FDM: FDM for thermal
    config.enable_phase_change      = true;
    config.enable_fluid             = true;
    config.enable_vof               = false;      // No VOF — Inamuro stress BC at wall
    config.enable_vof_advection     = false;
    config.enable_laser             = true;
    config.enable_darcy             = true;
    config.enable_marangoni         = true;       // → routed to stress BC (no VOF)
    config.enable_surface_tension   = false;
    config.enable_buoyancy          = false;
    config.enable_evaporation_mass_loss = false;
    config.enable_recoil_pressure   = false;
    config.enable_radiation_bc      = false;

    // --- Fluid ---
    config.kinematic_viscosity = nu_LU;  // physical viscosity (no artificial increase needed)
    config.density             = rho;
    config.darcy_coefficient   = 1.0e6f;  // Carman-Kozeny (matches OpenFOAM)  // Reduced from 1e6           // Carman-Kozeny constant

    // --- Thermal ---
    config.thermal_diffusivity    = alpha;
    config.ambient_temperature    = 300.0f;
    config.enable_substrate_cooling = false;

    // --- Surface (Marangoni) ---
    config.surface_tension_coeff = mat.surface_tension;
    config.dsigma_dT             = mat.dsigma_dT;   // +1.0e-4

    // Config verification
    printf("\n=== CONFIG VERIFICATION ===\n");
    printf("  enable_marangoni:      %s\n", config.enable_marangoni ? "TRUE" : "FALSE");
    printf("  enable_fluid:          %s\n", config.enable_fluid ? "TRUE" : "FALSE");
    printf("  enable_thermal:        %s\n", config.enable_thermal ? "TRUE" : "FALSE");
    printf("  enable_phase_change:   %s\n", config.enable_phase_change ? "TRUE" : "FALSE");
    printf("  enable_darcy:          %s\n", config.enable_darcy ? "TRUE" : "FALSE");
    printf("  config.dsigma_dT:      %+.4e N/(m·K)\n", config.dsigma_dT);
    printf("  darcy_coefficient:     %.2e\n", config.darcy_coefficient);
    printf("===========================\n");

    // --- Laser: stationary spot, keyhole power ---
    config.laser_power             = 300.0f;       // Phase 3: Higher power for keyhole
    config.laser_spot_radius       = 25.0e-6f;
    config.laser_absorptivity      = 0.35f;
    config.laser_penetration_depth = dx;            // Single top-layer absorption
    config.laser_shutoff_time      = 75.0e-6f;      // OFF after 75 μs (keyhole mode)
    config.laser_start_x           = -1.0f;          // Auto-center
    config.laser_start_y           = -1.0f;
    config.laser_scan_vx           = 0.0f;           // Stationary
    config.laser_scan_vy           = 0.0f;

    // --- Ray tracing OFF ---
    config.ray_tracing.enabled = false;

    // --- CFL ---
    config.cfl_velocity_target     = 0.15f;
    config.cfl_use_gradual_scaling = true;
    config.cfl_force_ramp_factor   = 0.9f;

    // --- VOF ---
    config.vof_subcycles              = 1;
    config.enable_vof_mass_correction = false;

    // --- Boundaries: all walls ---
    config.boundaries.x_min = BoundaryType::WALL;
    config.boundaries.x_max = BoundaryType::WALL;
    config.boundaries.y_min = BoundaryType::WALL;
    config.boundaries.y_max = BoundaryType::WALL;
    config.boundaries.z_min = BoundaryType::WALL;
    config.boundaries.z_max = BoundaryType::WALL;

    // Thermal BCs: ALL ADIABATIC for absolute energy conservation
    // Dirichlet boundaries cause heat loss, violating energy conservation.
    config.boundaries.thermal_x_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_x_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature = 300.0f;

    // ==================================================================
    // Timing
    // ==================================================================
    const float t_total = 100.0e-6f;
    const int total_steps = static_cast<int>(t_total / dt + 0.5f);
    printf("\nSimulation: %.0f μs, %d steps\n", t_total * 1e6f, total_steps);

    // Snapshot times
    struct Snap { float t; const char* tag; };
    Snap snaps[] = {
        {25.0e-6f,  "25us"},
        {50.0e-6f,  "50us"},
        {60.0e-6f,  "60us"},
        {75.0e-6f,  "75us"},
        {100.0e-6f, "100us"},
    };
    const int n_snap = 5;

    // ==================================================================
    // Initialize solver
    // ==================================================================
    printf("\nInitializing MultiphysicsSolver...\n");
    MultiphysicsSolver solver(config);

    const int num_cells = NX * NY * NZ;

    // Temperature: uniform 300K
    std::vector<float> h_temp(num_cells, 300.0f);

    // No VOF — all cells are metal. Surface at z=NZ-1 (top).
    // interface_height = 1.0 maps to z = NZ (clamped to NZ-1 by solver)
    solver.initialize(300.0f, 1.0f);
    solver.setSmagorinskyCs(param_Cs);
    printf("Smagorinsky Cs = %.2f (from cmdline)\n", param_Cs);
    printf("Initialized: %d cells (no VOF, Inamuro stress BC at z=%d)\n",
           num_cells, NZ - 1);

    // ==================================================================
    // Coordinate offset for contour output (match OpenFOAM)
    // LBM iz=0 → Z_um = 0 + Z_OFFSET;  iz=NZ_METAL-1 (surface) → Z_um = 98+Z_OFFSET
    // ==================================================================
    const float dx_um = dx * 1e6f;
    const float Z_OFFSET = 51.0f;  // Aligns LBM iz=49 → Z=149μm (OpenFOAM surface)

    // ==================================================================
    // Host buffers
    // ==================================================================
    std::vector<float> h_T(num_cells);

    // ==================================================================
    // Time loop
    // ==================================================================
    printf("\n%-6s %7s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "T_max[K]", "v_max[m/s]", "D_sol[μm]", "D_liq[μm]");
    printf("--------------------------------------------------------------\n");

    int snap_idx = 0;
    const int print_every = std::max(1, total_steps / 30);

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;

        // Advance physics
        if (step > 0) solver.step();

        // Check snapshot
        bool is_snap = false;
        if (snap_idx < n_snap) {
            int snap_step = static_cast<int>(snaps[snap_idx].t / dt + 0.5f);
            if (step == snap_step) is_snap = true;
        }
        bool do_print = (step % print_every == 0) || is_snap || (step == total_steps);

        if (do_print || is_snap) {
            // Extract temperature to host
            solver.copyTemperatureToHost(h_T.data());
            float T_max = solver.getMaxTemperature();

            // Max velocity (lattice → physical)
            float v_max_LU = solver.getMaxVelocity();
            float v_max_phys = v_max_LU * dx / dt;

            // Melt pool depth at center column
            const int cx = NX / 2, cy = NY / 2;
            float depth_sol = 0.0f, depth_liq = 0.0f;
            for (int iz = NZ_METAL - 1; iz >= 0; iz--) {
                int idx = cx + cy * NX + iz * NX * NY;
                float depth = static_cast<float>(NZ_METAL - 1 - iz) * dx_um;
                if (h_T[idx] >= 1650.0f && depth > depth_sol) depth_sol = depth;
                if (h_T[idx] >= 1700.0f && depth > depth_liq) depth_liq = depth;
            }

            if (do_print) {
                printf("%6d %7.2f %9.0f %9.2f %9.1f %9.1f\n",
                       step, t * 1e6f, T_max, v_max_phys, depth_sol, depth_liq);
            }

        }

        // ==============================================================
        // Snapshot: extract T=1650K contour
        // ==============================================================
        if (is_snap) {
            const char* tag = snaps[snap_idx].tag;
            printf("  >>> SNAPSHOT %s <<<\n", tag);

            // Centerline profile: T and vz along z at (cx, cy)
            // Uses getMaxVelocity to confirm fluid is alive, then dumps
            // raw distribution momentum for vz diagnosis
            {
                const int cx = NX / 2, cy = NY / 2;
                // Read velocity field via distributions (no public API)
                // Instead, compute vz from temperature gradient pattern:
                // The real diagnostic is T(z) — if advection works, T should
                // penetrate deeper than pure conduction
                printf("  Centerline T(z) and depth profile:\n");
                printf("  %6s %8s %8s\n", "z_um", "T[K]", "depth_um");
                for (int iz = NZ - 1; iz >= 0 && iz >= NZ - 30; iz--) {
                    int idx = cx + cy * NX + iz * NX * NY;
                    float z_um = iz * dx_um;
                    float depth = (NZ - 1 - iz) * dx_um;
                    printf("  %6.0f %8.0f %8.0f\n", z_um + Z_OFFSET, h_T[idx], depth);
                }
                // Compute effective Pe at the surface
                float v_max_LU = solver.getMaxVelocity();
                float v_phys = v_max_LU * dx / dt;
                float Pe = v_phys * (20e-6f) / alpha;  // L_char = 20um melt depth
                printf("  v_max = %.2f m/s (%.4f LU), Pe = %.1f\n", v_phys, v_max_LU, Pe);
            }

            // Write contour CSV (same format as conduction benchmark)
            char fname[128];
            snprintf(fname, sizeof(fname), "lbm_marangoni_contour_%s.csv", tag);
            FILE* fc = fopen(fname, "w");
            if (fc) {
                fprintf(fc, "X_um,Z_um\n");
                const int cy = NY / 2;
                const float T_iso = 1650.0f;

                // Z-crossings along each x-column
                for (int ix = 0; ix < NX; ix++) {
                    for (int iz = NZ_METAL - 2; iz >= 0; iz--) {
                        int idx_up = ix + cy * NX + (iz + 1) * NX * NY;
                        int idx_dn = ix + cy * NX + iz * NX * NY;
                        float T_up = h_T[idx_up], T_dn = h_T[idx_dn];
                        if ((T_up >= T_iso) != (T_dn >= T_iso)) {
                            float frac = (T_iso - T_dn) / (T_up - T_dn);
                            fprintf(fc, "%.4f,%.4f\n",
                                    ix * dx_um, (iz + frac) * dx_um + Z_OFFSET);
                        }
                    }
                    // X-crossings along each z-row
                    if (ix < NX - 1) {
                        for (int iz = 0; iz < NZ_METAL; iz++) {
                            int idx_l = ix + cy * NX + iz * NX * NY;
                            int idx_r = (ix+1) + cy * NX + iz * NX * NY;
                            float T_l = h_T[idx_l], T_r = h_T[idx_r];
                            if ((T_l >= T_iso) != (T_r >= T_iso)) {
                                float frac = (T_iso - T_l) / (T_r - T_l);
                                fprintf(fc, "%.4f,%.4f\n",
                                        (ix + frac) * dx_um, iz * dx_um + Z_OFFSET);
                            }
                        }
                    }
                }
                fclose(fc);
                printf("  Contour → %s\n\n", fname);
            }

            // Also write 2D temperature field slice (y=NY/2) for native contour()
            char fname_field[128];
            snprintf(fname_field, sizeof(fname_field), "lbm_temperature_%s.csv", tag);
            FILE* ff = fopen(fname_field, "w");
            if (ff) {
                fprintf(ff, "X_um,Z_um,T_K\n");
                const int cy = NY / 2;
                for (int iz = 0; iz < NZ; iz++) {
                    for (int ix = 0; ix < NX; ix++) {
                        int idx = ix + cy * NX + iz * NX * NY;
                        fprintf(ff, "%.2f,%.2f,%.1f\n",
                                ix * dx_um, iz * dx_um + Z_OFFSET, h_T[idx]);
                    }
                }
                fclose(ff);
                printf("  Field → %s\n", fname_field);
            }

            snap_idx++;
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(wall_end - wall_start).count();
    printf("\nWall time: %.1f s\n", elapsed);

    return 0;
}
// This will be called from the snapshot section - but we need to add it inline.
// Let me just modify the snapshot block to dump profiles.
