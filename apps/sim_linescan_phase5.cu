/**
 * @file sim_linescan_phase5.cu
 * @brief Phase-5 integration validation: Sprint-2 main-branch fixes + F3D emissivity align
 *        Ready to receive Worktree-A (VOF mass-correction redesign) once merged.
 *
 * ============================================================
 * PARAMETER DELTA vs Phase-4 (sim_linescan_phase4.cu)
 * ============================================================
 *
 * | Parameter              | Phase-4          | Phase-5            | Source                  |
 * |------------------------|------------------|--------------------|-------------------------|
 * | emissivity             | 0.28 (mat default) | 0.55             | F3D prepin hflem1=0.55  |
 * | VOF mass correction    | false            | false (CLI toggle) | Phase-2 showed worse; A |
 * |                        |                  |   awaits worktree-A| worktree will enable     |
 * | Ray tracing num_rays   | 2048             | 4096               | Phase-4 uses 2048 (Phase |
 * |                        |                  |                    |  -1 budget cut); restore |
 * |                        |                  |                    |  to Phase-4 production   |
 * | Ray tracing max_bounces| 3                | 5                  | F3D: multi-bounce needed |
 * | Ray tracing max_dda_steps | 800           | 1500               | Phase-4 was limited;     |
 * |                        |                  |                    |  domain 2200 μm / 2 μm   |
 * |                        |                  |                    |  = 1100 cells + slack    |
 * | Output directory       | output_phase4/   | output_phase5/     | —                        |
 *
 * NOTE: ν_LU = 0.065 INTENTIONALLY UNCHANGED.  Phase-3 showed lower viscosity
 *       (τ→0.55) made center Δh WORSE, not better.  Do not touch.
 *
 * NOTE: Domain 1100×75×100 = 8.25 M cells, identical to Phase-4.  Memory budget
 *       at 440 bytes/cell: 8.25e6 × 440 = 3630 MB < 4096 MB.  Safe.
 *
 * ============================================================
 * DEFERRED F3D ALIGNMENT ITEMS (separate tasks, not this PR)
 * ============================================================
 *
 * 1. DENSITY TABLE — F3D prepin rhof(1) table (lines beginning #rhof(1)):
 *    The table shows a DISCONTINUOUS jump at 1723.15 K (approx. T_liquidus):
 *      298.15 K  → 7950 kg/m³
 *      473.15 K  → 7880
 *      ...
 *      1658.15 K → 7269
 *      1723.15 K → 7236  (solid/mushy side)
 *      1723.15 K → 6881  (liquid side — 5% step jump at melting)
 *      1773.15 K → 6842
 *      1873.15 K → 6765
 *    Current API: MaterialProperties exposes scalar rho_solid and rho_liquid only.
 *    Wiring the full T-table + discontinuous jump at T_liquidus requires a new
 *    rho_T_table field plus solver path changes in multiphysics_solver.cu.
 *    Impact estimate: ~1-3 μm improvement to center Δh.
 *    Action: separate task "D — material API extension for ρ(T)".
 *
 * 2. BC OUTFLOW — F3D uses ibct=3 (continuous outflow) on all 6 faces.
 *    LBM currently uses WALL on all faces.  Estimated impact 1-3 μm.
 *    Action: separate task "medium priority BC change".
 *
 * 3. if_vol_corr — F3D vol_corr_time=1e-6 s (1 μs correction cycle).
 *    This is the equivalent of the worktree-A mass-correction redesign.
 *    Activated via CLI_VOF_MASS_CORR=1 once worktree-A merges.
 *
 * ============================================================
 * EXPECTED COMBINED EFFECT ON CENTER Δh (qualitative)
 * ============================================================
 *
 * - Emissivity 0.28→0.55: higher surface energy loss → slightly cooler pool
 *   surface → weaker recoil at tail → SMALL improvement ~1-2 μm (uncertain sign).
 * - Ray tracing 2048→4096 rays / 3→5 bounces: captures more of the multi-bounce
 *   keyhole energy budget.  Should increase effective absorptivity from ~65% toward
 *   70% F3D target.  Pool deeper and hotter.  Effect on track height unclear
 *   (deeper pool can swing either way depending on backflow).
 * - Worktree-A mass correction (once merged): UNKNOWN — that is the purpose of A.
 *   Phase-2 uniform redistribution was harmful (-16→-22 μm).  A redesign may help.
 *
 * ============================================================
 * VALIDATION ORDER WHEN RUN
 * ============================================================
 *
 * 1. Verify no NaN in first 500 steps (regression vs Phase-4 stability).
 * 2. At t=800 μs compare pool D/L/W against Phase-1 baseline (-16 μm center).
 * 3. At t=2000 μs compare center Δh 95%ile against Phase-4 baseline (-22 μm).
 * 4. Target: center Δh > -16 μm (better than Phase-1) as merge bar for A.
 * 5. Compare effective absorptivity printout (target: 65→70%).
 *
 * ============================================================
 * VOF MASS CORRECTION CLI TOGGLE
 * ============================================================
 *
 * Default: OFF (matches Phase-2/3/4 production, avoids broken uniform algorithm).
 * To enable once Worktree-A merges:
 *   Recompile with:  -DPHASE5_VOF_MASS_CORR=1
 * or set environment variable at compile time in CMakeLists:
 *   target_compile_definitions(sim_linescan_phase5 PRIVATE PHASE5_VOF_MASS_CORR=1)
 *
 * CMakeLists registration: DO NOT add yet — leave for human post worktree-A merge.
 * Build pattern (matches existing phase targets):
 *   add_executable(sim_linescan_phase5 apps/sim_linescan_phase5.cu)
 *   set_target_properties(sim_linescan_phase5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 *   target_link_libraries(sim_linescan_phase5 lbm_physics lbm_io CUDA::cudart)
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"
#include "io/field_registry.h"
#include "core/lattice_d3q19.h"

using namespace lbm;
using namespace lbm::physics;

/**
 * @brief Compute melt pool dimensions from liquid fraction field
 *
 * Scans the 3D fl field to find the bounding box of fl > 0.5 cells,
 * masked to metal only (fill_level > 0.5).
 */
struct MeltPoolMetrics {
    float depth_um;
    float length_um;
    float width_um;
};

static MeltPoolMetrics computeMeltPoolDimensions(
    const float* h_lf, const float* h_fl,
    int nx, int ny, int nz, float dx, float interface_z)
{
    int x_min = nx, x_max = -1;
    int y_min = ny, y_max = -1;
    int z_min = nz, z_max = -1;

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                if (h_fl[idx] > 0.5f && h_lf[idx] > 0.5f) {
                    x_min = std::min(x_min, i);
                    x_max = std::max(x_max, i);
                    y_min = std::min(y_min, j);
                    y_max = std::max(y_max, j);
                    z_min = std::min(z_min, k);
                    z_max = std::max(z_max, k);
                }
            }

    MeltPoolMetrics m;
    if (x_max < 0) {
        m.depth_um = m.length_um = m.width_um = 0.0f;
    } else {
        m.length_um = (x_max - x_min + 1) * dx * 1e6f;
        m.width_um  = (y_max - y_min + 1) * dx * 1e6f;
        float z_surface = interface_z;
        m.depth_um  = (z_surface - z_min) * dx * 1e6f;
        if (m.depth_um < 0) m.depth_um = 0;
    }
    return m;
}

int main() {
    auto wall_start = std::chrono::high_resolution_clock::now();

    printf("\n");
    printf("============================================================\n");
    printf("  LPBF Line Scan Phase-5: 316L Integration Validation\n");
    printf("  Sprint-2 fixes + F3D emissivity align + A-merge ready\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Phase-5 VOF mass correction compile-time toggle.
    // OFF by default — prevents the broken Phase-2 uniform redistribution
    // algorithm from running silently.  Define PHASE5_VOF_MASS_CORR=1
    // at compile time to engage once Worktree-A has been merged and
    // validated against the 7-pass / 4-reject criteria in
    // docs/task-A-vof-mass-correction/A_TASK_BRIEF.md.
    // ==================================================================
#ifndef PHASE5_VOF_MASS_CORR
    constexpr bool kVofMassCorr = false;
#else
    constexpr bool kVofMassCorr = (PHASE5_VOF_MASS_CORR != 0);
#endif

    // ==================================================================
    // Configuration
    // ==================================================================
    MultiphysicsConfig config;

    // --- Domain: 1100 × 75 × 100 cells = 2200 × 150 × 200 μm @ dx=2 μm ---
    // Identical to Phase-4 (8.25 M cells).  Memory budget at 440 bytes/cell:
    //   8.25e6 × 440 B = 3630 MB < 4096 MB (4 GB GPU).  Safe.
    //
    // Domain rationale (carried forward from Phase-4 comment):
    //   x: 2200 μm covers 2 ms scan (0.8 m/s × 2 ms = 1600 μm) + 500 μm
    //      pre-scan margin + 100 μm trailing beam tail.
    //   y: 150 μm (75 cells) — tight but sufficient: pool W ≈ 100 μm (LBM),
    //      side ridges at ±60 μm fit within 75 μm half-width.
    //   z: 200 μm (100 cells) — substrate 160 μm (80 cells) + 40 μm gas cap.
    //      Keyhole depth ~80 μm leaves 80 μm buffer to bottom.
    //
    // Alternative considered: 900×95×100 = 8.55M cells for wider transverse
    // (+20 μm each side would help confirm no wall-boundary reflections on
    // Marangoni rolls).  But 8.55M cells × 440 B = 3762 MB — still under
    // budget, barely.  However, the shorter x would clip the laser track at
    // t=1.9 ms (900 cells = 1800 μm, laser reaches 500+0.8×1900=2020 μm >
    // 1800 μm).  Not viable without raising nx.  1100×75×100 is the right
    // choice here.
    config.nx = 1100;
    config.ny = 75;
    config.nz = 100;
    config.dx = 2.0e-6f;
    // Track-A (Round-5 follow-up, 2026-04-26): shrink dt 80→20 ns to push Ma
    // from 0.43 (cap-firing) to 0.11 (well inside BGK-stable envelope).
    // Empirically validated by:
    //   1. TRT/Reg Couette-Poiseuille τ-sweep at τ=0.54 → L2=0.16% PASS
    //   2. Phase-1 1000-step dry run at this dt → 0 NaN, 0 cap triggers,
    //      v_max peak 3.64 m/s (vs cap 6.25 m/s, 73% headroom),
    //      mass drift +0.041%
    // Cost: ~12 hr wall vs original ~80 min. Acceptable per user directive.
    // Resulting τ ≈ 0.54 (was 0.65); TRT (Λ=3/16) keeps it viscous-independent.
    //
    // 2026-04-27 RE-EVALUATION: after worktree A's Track-C iter-4 mass correction
    // landed, Mountain 2 forensics showed v_max in trailing zone is dominantly
    // bounded by mass-balance, not by Mach numerics. The dt/4 protection is
    // architecturally correct but not on the F3D-match critical path. For the
    // F3D-match push (Track-C + emissivity 0.55 + ray bounces=5), use dt=80ns
    // for ~85min wall instead of ~12hr. Revert to dt/4 for higher-power process
    // windows or when Strategy β collision-operator work is on.
    config.dt = 8.0e-8f;

    // --- Material: 316L Mills (aligned to F3D in Sprint-1) ---
    config.material = MaterialDatabase::get316L();
    // Mushy zone from F3D prepin ts1=1674.15, tl1=1697.15 (already set in
    // MaterialDatabase::get316L() but written explicitly here for clarity).
    config.material.T_solidus  = 1674.15f;
    config.material.T_liquidus = 1697.15f;

    // PHASE-5 CHANGE #1: Emissivity 0.28 → 0.55
    // F3D prepin line: hflem1=0.55  (near &xput section)
    // Previous value 0.28 was the 316L database default (slightly above
    // oxidised-solid estimate; F3D uses 0.55 which includes partial oxidation
    // and rougher melt pool surface).  Higher emissivity increases radiative
    // surface loss at high T: at T_boil=3090 K, ΔP_rad ≈ σ_SB·Δε·T⁴ ≈
    // 5.67e-8 × 0.27 × 3090⁴ ≈ 1.3 MW/m² additional cooling.  This is small
    // vs laser irradiance (~2.5 GW/m² peak) but non-zero at the melt pool tail
    // (T~1800 K, ΔP_rad~140 kW/m²).  Expected: slightly shallower, cooler
    // trailing edge → modest improvement in center Δh (estimate: 1-2 μm).
    config.material.emissivity = 0.55f;

    // --- Physics flags (identical to Phase-4) ---
    config.enable_thermal                = true;
    config.enable_thermal_advection      = true;
    config.enable_phase_change           = true;
    // FDM thermal: avoids D3Q7 gas-wipe artefact (thermal_lbm.cu:1186).
    // Column-march V2 and implicit Newton evaporation cooling live here.
    config.use_fdm_thermal               = true;
    config.enable_fluid                  = true;
    config.enable_vof                    = true;
    config.enable_vof_advection          = true;
    config.enable_laser                  = true;
    config.enable_darcy                  = true;
    config.enable_marangoni              = true;
    config.enable_surface_tension        = true;
    config.enable_buoyancy               = true;
    config.enable_evaporation_mass_loss  = true;
    config.enable_recoil_pressure        = true;
    config.enable_radiation_bc           = true;

    // --- Moving laser (identical to Phase-1/4) ---
    const float v_scan = 0.8f;   // 800 mm/s
    config.laser_power             = 150.0f;
    config.laser_spot_radius       = 39.0e-6f;   // F3D dum2=39e-6 m
    config.laser_absorptivity      = 0.40f;       // fallback (unused when Fresnel active)
    config.laser_penetration_depth = 10.0e-6f;
    config.laser_start_x           = 500.0e-6f;   // 500 μm pre-scan margin
    config.laser_start_y           = -1.0f;        // auto-center Y
    config.laser_scan_vx           = v_scan;
    config.laser_scan_vy           = 0.0f;

    // --- Ray-tracing laser ---
    // PHASE-5 CHANGE #2: Restore full ray budget that Phase-4 reduced for
    // cost during night-protocol scheduling.  Phase-4 comment: "first attempt
    // with 4096 rays stalled at 0.12 step/sec".  With Phase-4 FDM + EDM path
    // the per-step budget is available; 4096/5/1500 is the production setting.
    // num_rays 4096  — sufficient angular resolution for 39 μm spot (Δθ≈0.09°)
    // max_bounces 5  — captures ~95% asymptotic Fresnel multi-bounce absorption
    // max_dda_steps 1500 — domain 2200 μm / 2 μm = 1100 cells + 400 cell slack
    config.ray_tracing.enabled           = true;
    config.ray_tracing.use_fresnel       = true;
    config.ray_tracing.fresnel_n_refract = 2.9613f;   // F3D dum8
    config.ray_tracing.fresnel_k_extinct = 4.0133f;   // F3D dum9
    config.ray_tracing.num_rays          = 4096;
    config.ray_tracing.max_bounces       = 5;
    config.ray_tracing.max_dda_steps     = 1500;
    config.ray_tracing.energy_cutoff     = 0.01f;
    config.ray_tracing.absorptivity      = 0.40f;     // unused when use_fresnel=true

    // --- Fluid ---
    // ν_LU = 0.065  INTENTIONALLY UNCHANGED from Phase-3/4.
    // Phase-3 tested τ→0.55 (ν_LU=0.0167) and found center Δh worsened.
    // Physical ν for 316L liquid is ~7.6e-7 m²/s; 0.065 LU is 4.3× artificial.
    // Stability requirement: do not lower until a validated path exists.
    config.kinematic_viscosity     = 0.065f;    // tau ≈ 0.7
    config.density                 = config.material.rho_liquid;

    // --- Darcy ---
    config.darcy_coefficient       = 5.0e4f;

    // --- Thermal ---
    config.thermal_diffusivity     = config.material.getThermalDiffusivity(1700.0f);
    config.ambient_temperature     = 300.0f;
    // emissivity is forwarded from material above (0.55).  The config.emissivity
    // accessor aliases thermal.emissivity; set it explicitly to match material.
    config.emissivity              = config.material.emissivity;  // 0.55

    // --- Surface ---
    config.surface_tension_coeff   = config.material.surface_tension;  // 1.75 N/m
    config.dsigma_dT               = config.material.dsigma_dT;         // -4.3e-4 N/(m·K)
    // csf multiplier stays at 1.0 (S2-A1 at 2.0 made groove deeper).
    config.marangoni_csf_multiplier = 1.0f;
    // recoil_smoothing_width: default 30 K (S2-A2 at 5 K had zero effect; unchanged).

    // --- Buoyancy ---
    config.thermal_expansion_coeff = 1.2e-4f;
    config.gravity_x               = 0.0f;
    config.gravity_y               = 0.0f;
    config.gravity_z               = -9.81f;
    config.reference_temperature   = 0.5f * (config.material.T_solidus + config.material.T_liquidus);

    // --- Substrate cooling ---
    config.enable_substrate_cooling = true;
    config.substrate_h_conv         = 2000.0f;
    config.substrate_temperature    = 300.0f;

    // --- Boundaries (all WALL; ibct=3 outflow deferred — medium priority task) ---
    config.boundaries.x_min = BoundaryType::WALL;
    config.boundaries.x_max = BoundaryType::WALL;
    config.boundaries.y_min = BoundaryType::WALL;
    config.boundaries.y_max = BoundaryType::WALL;
    config.boundaries.z_min = BoundaryType::WALL;
    config.boundaries.z_max = BoundaryType::WALL;

    config.boundaries.thermal_x_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_x_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min = ThermalBCType::CONVECTIVE;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.convective_h     = 2000.0f;
    config.boundaries.convective_T_inf = 300.0f;

    // --- Adaptive CFL ---
    config.cfl_use_adaptive          = true;
    config.cfl_v_target_interface    = 0.15f;
    config.cfl_v_target_bulk         = 0.10f;

    // --- VOF + Track-C iter-4 mass correction (post worktree-A merge, 2026-04-27) ---
    // Worktree A's Track-C iter-4 closed Phase-2 (t=800μs) centerline from
    // -14μm to -6μm (5/6 brief PASS). Phase-5 layers Track-C ON TOP OF
    // emissivity 0.55 + ray bounces=5 (Phase-4 uses bounces=3).
    // The compile-flag kVofMassCorr is now obsolete (Track-C is the production
    // path) but kept for A/B comparison if needed.
    config.vof_subcycles             = 1;
    config.enable_vof_mass_correction          = true;     // Track-C ON
    config.vof_mass_correction_use_flux_weight = true;     // Track-B base
    config.vof_mass_correction_damping         = 0.7f;     // sweet spot
    config.mass_correction_use_track_c         = true;     // geometric gates ON
    config.mass_correction_trailing_margin_lu  = 25.0f;    // 50 μm past laser
    config.mass_correction_z_substrate_lu      = 80.0f;    // matches interface_z=80
    config.mass_correction_z_offset_lu         = 0.0f;     // strict

    // --- Timing: 2 ms = 100000 steps at dt=20 ns (post-Track-A dt shrink) ---
    // VTK every 100 μs → 20 frames + initial. Diag every 4000 steps (80 μs).
    const float t_total  = 2000.0e-6f;
    const int   num_steps  = static_cast<int>(t_total / config.dt);   // 100000
    const int   vtk_every  = static_cast<int>(100.0e-6f / config.dt); // 5000
    const int   diag_every = 4000;

    // Substrate top at z=80 cells (160 μm). Metal [0,160] μm, gas [160,200] μm.
    // 80-cell buffer to keyhole base at ~80 μm depth.
    const float interface_z = 80.0f;  // cells (× dx = 160 μm physical)

    // ==================================================================
    // Print Phase-5 configuration summary
    // ==================================================================
    printf("Phase-5 key changes vs Phase-4:\n");
    printf("  emissivity       : 0.28 (mat default) -> 0.55 (F3D hflem1=0.55)\n");
    printf("  ray_tracing rays : 2048 -> 4096\n");
    printf("  ray_tracing max_bounces: 3 -> 5\n");
    printf("  ray_tracing max_dda_steps: 800 -> 1500\n");
    printf("  VOF mass corr    : %s (compile PHASE5_VOF_MASS_CORR=1 to enable)\n\n",
           kVofMassCorr ? "ON (worktree-A path)" : "OFF (safe default)");

    printf("Domain:  %d x %d x %d cells = %d x %d x %d um\n",
           config.nx, config.ny, config.nz,
           (int)(config.nx * config.dx * 1e6f),
           (int)(config.ny * config.dx * 1e6f),
           (int)(config.nz * config.dx * 1e6f));
    printf("  Memory budget: %.0f M cells x 440 B/cell = %.0f MB (limit 4096 MB)\n",
           (double)(config.nx * config.ny * config.nz) / 1e6,
           (double)(config.nx * config.ny * config.nz) * 440.0 / 1e6);
    printf("dx = %.0f um, dt = %.0f ns\n", config.dx * 1e6f, config.dt * 1e9f);

    printf("\nMaterial: %s\n", config.material.name);
    printf("  T_sol=%.2f K, T_liq=%.2f K, T_boil=%.0f K\n",
           config.material.T_solidus, config.material.T_liquidus,
           config.material.T_vaporization);
    printf("  emissivity = %.2f  dσ/dT = %.1e N/(m·K)\n",
           config.material.emissivity, config.dsigma_dT);

    printf("\nLaser: P=%.0f W, r0=%.0f um, Fresnel n=%.4f k=%.4f\n",
           config.laser_power, config.laser_spot_radius * 1e6f,
           config.ray_tracing.fresnel_n_refract, config.ray_tracing.fresnel_k_extinct);
    printf("  rays=%d  bounces=%d  dda_steps=%d\n",
           config.ray_tracing.num_rays, config.ray_tracing.max_bounces,
           config.ray_tracing.max_dda_steps);
    printf("  v_scan=%.0f mm/s  start_x=%.0f um  track=%.0f um\n",
           v_scan * 1e3f, config.laser_start_x * 1e6f, v_scan * t_total * 1e6f);

    printf("\nFluid: nu_LU=%.3f, tau=%.3f (unchanged from Phase-3/4)\n",
           config.kinematic_viscosity,
           1.0f / (3.0f * config.kinematic_viscosity + 0.5f));
    printf("Darcy: K=%.0e (linear)\n", config.darcy_coefficient);
    printf("Steps: %d (%.0f us)  VTK every %d steps (%.0f us)  diag every %d steps\n\n",
           num_steps, t_total * 1e6f,
           vtk_every, vtk_every * config.dt * 1e6f,
           diag_every);
    fflush(stdout);

    // ==================================================================
    // Initialize
    // ==================================================================
    mkdir("output_phase5", 0755);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.80f);

    const auto& registry = solver.getFieldRegistry();

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_lf(num_cells), h_fl(num_cells);

    float initial_mass = solver.getTotalMass();
    printf("Initial metal volume (sum fill): %.0f cells\n\n", initial_mass);

    // ==================================================================
    // Console table header
    // ==================================================================
    printf("%-6s %7s %7s %7s %7s %7s %7s %9s %8s\n",
           "Step", "t[us]", "T_max", "v_max", "Depth", "Length", "Width",
           "Laser_x", "MassDelta%");
    printf("%-6s %7s %7s %7s %7s %7s %7s %9s %8s\n",
           "", "", "[K]", "[m/s]", "[um]", "[um]", "[um]", "[um]", "");
    printf("-----------------------------------------------------------------------\n");
    fflush(stdout);

    // ==================================================================
    // Time integration
    // ==================================================================
    for (int step = 0; step <= num_steps; ++step) {
        float t       = step * config.dt;
        float laser_x = config.laser_start_x + v_scan * t;

        // --- Diagnostics every diag_every steps ---
        if (step % diag_every == 0) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();

            cudaMemcpy(h_lf.data(),
                       solver.getLiquidFraction(),
                       num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fl.data(),
                       solver.getFillLevel(),
                       num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            MeltPoolMetrics mp = computeMeltPoolDimensions(
                h_lf.data(), h_fl.data(),
                config.nx, config.ny, config.nz, config.dx, interface_z);

            float current_mass = solver.getTotalMass();
            float mass_delta   = (current_mass - initial_mass) / initial_mass * 100.0f;

            printf("%-6d %7.1f %7.0f %7.3f %7.1f %7.1f %7.1f %9.1f %+7.3f%%\n",
                   step, t * 1e6f, T_max, v_max,
                   mp.depth_um, mp.length_um, mp.width_um,
                   laser_x * 1e6f, mass_delta);
            fflush(stdout);

            if (solver.checkNaN()) {
                printf("\n*** FATAL: NaN detected at step %d ***\n", step);
                break;
            }
        }

        // --- VTK output every vtk_every steps ---
        if (step % vtk_every == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "output_phase5/line_scan_%06d", step);
            io::VTKWriter::writeFields(
                std::string(filename), registry, {},
                config.nx, config.ny, config.nz, config.dx);
            printf("  -> VTK: %s.vtk (t=%.0f us, laser_x=%.0f um)\n",
                   filename, t * 1e6f, laser_x * 1e6f);
            fflush(stdout);
        }

        // --- Advance physics ---
        if (step < num_steps) {
            solver.step();
        }
    }

    // ==================================================================
    // Performance summary
    // ==================================================================
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    double wall_minutes = wall_seconds / 60.0;

    long long total_cells   = (long long)config.nx * config.ny * config.nz;
    long long total_updates = total_cells * (long long)num_steps;
    double MLUPS = total_updates / wall_seconds / 1e6;
    double GLUPS = total_updates / wall_seconds / 1e9;

    float final_mass   = solver.getTotalMass();
    float mass_delta_f = (final_mass - initial_mass) / initial_mass * 100.0f;

    printf("\n============================================================\n");
    printf("  Phase-5 Run Complete\n");
    printf("============================================================\n");
    printf("  Sim time:    %.0f us (%d steps)\n", t_total * 1e6f, num_steps);
    printf("  Track: x = %.0f -> %.0f um at %.0f mm/s\n",
           config.laser_start_x * 1e6f,
           (config.laser_start_x + v_scan * t_total) * 1e6f,
           v_scan * 1e3f);
    printf("  T_max: %.0f K\n", solver.getMaxTemperature());
    printf("  v_max: %.3f m/s\n", solver.getMaxVelocity());
    printf("  Mass:  %+.3f%%\n", mass_delta_f);
    printf("  VOF mass corr: %s\n", kVofMassCorr ? "ON" : "OFF");
    printf("\n  === Performance ===\n");
    printf("  Wall clock: %.1f s (%.1f min)\n", wall_seconds, wall_minutes);
    printf("  Grid: %lld cells (%dx%dx%d)\n", total_cells, config.nx, config.ny, config.nz);
    printf("  Total updates: %.2e\n", (double)total_updates);
    printf("  Throughput: %.2f MLUPS (%.4f GLUPS)\n", MLUPS, GLUPS);
    printf("  Steps/sec: %.1f\n", num_steps / wall_seconds);
    printf("\nOutput: output_phase5/line_scan_*.vtk\n");
    printf("\nValidation checkpoints:\n");
    printf("  1. center Dh 95%%ile @ t=2ms vs Phase-4 baseline -22 um\n");
    printf("     Phase-5 target: > -16 um (better than Phase-1)\n");
    printf("  2. Pool D/L/W @ t=800 us vs Phase-1 (D=70, L=113%%, W~73 um)\n");
    printf("  3. Effective absorptivity check: target 65->70%%\n");
    printf("  4. Mass drift target: < 2%% (Phase-1B was -1.66%%)\n");
    printf("============================================================\n\n");

    return 0;
}
