/**
 * @file sim_line_scan_316L.cu
 * @brief LPBF single-track line scan on 316L stainless steel
 *
 * Moving Gaussian laser (P=150W, v_scan=800 mm/s) on flat substrate.
 * Domain: 500 × 150 × 100 μm (250×75×50 cells at dx=2μm)
 *
 * Physics: FluidLBM(TRT+EDM) + ThermalLBM(ESM+VOF mask) + VOF(PLIC)
 *          + Marangoni + surface tension + buoyancy + linear Darcy
 *          + evaporation cooling, recoil OFF
 *
 * Output: VTK every 50 μs, console metrics every 500 steps
 *
 * Usage:
 *   mkdir -p output_phase3 && ./sim_line_scan_316L
 *   ParaView: Open output_phase3/line_scan_*.vtk
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
    float depth_um;   // Max z-extent below surface [μm]
    float length_um;  // Max x-extent [μm]
    float width_um;   // Max y-extent [μm]
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
                // Only count metal cells (fill > 0.5) that are melted (lf > 0.5)
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
        // Depth: distance from interface down to deepest melted cell
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
    printf("  LPBF Single-Track Line Scan: 316L (Production Run)\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Configuration
    // ==================================================================
    MultiphysicsConfig config;

    // --- Domain: 1600 × 75 × 100 μm @ dx=2μm (M16d compromise, 2026-04-25) ---
    // M16/M16b/M16c at 1300×100×120 cells: per-step cost was 18× expected scale
    // (3 sec/step at 15.6M cells vs 0.024 sec/step at 2.25M); root cause is
    // memory traffic / cache pressure beyond the kernel-level optimizations
    // accessible without a profiling pass.
    //
    // Pragmatic compromise: extend ONLY the scan-direction (nx) and ONLY the
    // depth (nz). Keep transverse (ny) tight since Flow3D pool W≈85μm and
    // LBM pool W≈100μm — 150μm transverse is already 75% margin.
    //   x: 1200 → 1600 μm  (covers t=2ms scan: 0+0.8·2000=1600μm + 100 margin)
    //   y: 150 →  150 μm  (unchanged; M13b verified W stays within)
    //   z: 100 →  200 μm  (substrate thickness 2× to keep keyhole ~80μm away from base)
    // Cell count: 800 × 75 × 100 = 6 M (2.7× M13b, ~10 min wall expected).
    // Phase-1 (Night Protocol): 1200×250×200 μm = 600×125×100 cells (7.5M).
    // Wider transverse (125 cells, 250 μm) than production (75 cells, 150 μm)
    // to fully capture side ridges and let mass redistribute back to center.
    // nx=650 (1300 μm): laser at t=800μs reaches 1140 μm; +3·w₀=117 μm beam
    // tail needs to fit, so nx must extend ≥1257 μm — 1300 μm gives 43 μm buffer.
    config.nx = 650;
    config.ny = 125;
    config.nz = 100;
    config.dx = 2.0e-6f;
    config.dt = 8.0e-8f;

    // --- Material ---
    config.material = MaterialDatabase::get316L();
    // PHASE-1 CHANGE #1: shrink mushy zone 200K → 23K (match F3D Mills table).
    // F3D prepin: ts1=1674.15, tl1=1697.15. Previous 1523/1723 over-extended
    // Darcy damping into the trailing-edge liquid pool (still 1700+ K but
    // fl<1 due to 200K window) → liquid stagnated, groove never refilled.
    config.material.T_solidus  = 1674.15f;
    config.material.T_liquidus = 1697.15f;

    // --- Physics ---
    config.enable_thermal           = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change      = true;
    // Sprint-1 (2026-04-25): use FDM thermal — D3Q7 LBM thermal's gas-wipe
    // (thermal_lbm.cu:1186) hard-clamps fill_level<0.01 cells to 600K every
    // step, leaking laser energy from the streaming-pushed phantom field.
    // FDM has neither streaming nor wipe; it is also where R6 column-march V2
    // and implicit Newton evaporation cooling live (much more accurate near
    // T_boil where Anisimov recoil is sensitive).
    config.use_fdm_thermal          = true;
    config.enable_fluid             = true;
    config.enable_vof               = true;
    config.enable_vof_advection     = true;
    config.enable_laser             = true;
    config.enable_darcy             = true;
    config.enable_marangoni         = true;
    config.enable_surface_tension   = true;
    config.enable_buoyancy          = true;
    config.enable_evaporation_mass_loss = true;
    config.enable_recoil_pressure   = true;   // KEYHOLE: recoil pushes surface down
    config.enable_radiation_bc      = true;

    // --- Moving laser ---
    const float v_scan = 0.8f;  // 800 mm/s
    config.laser_power              = 150.0f;
    // PHASE-1 CHANGE #2: spot radius 50 μm → 39 μm (match F3D dum2=39e-6).
    // Peak intensity 1.64× higher → sharper keyhole, narrower Marangoni
    // shear band → smaller side ridges expected.
    config.laser_spot_radius        = 39.0e-6f;
    config.laser_absorptivity       = 0.40f;     // fallback for non-RT path
    config.laser_penetration_depth  = 10.0e-6f;
    config.laser_start_x            = 500.0e-6f;  // Sprint-1: 500 μm pre-scan margin (Flow3D px_min=-497.5μm)
    config.laser_start_y            = -1.0f;      // Auto-center Y
    config.laser_scan_vx            = v_scan;     // 800 mm/s in +x
    config.laser_scan_vy            = 0.0f;

    // --- Ray-tracing laser (Sprint-1: complex-Fresnel for keyhole physics) ---
    config.ray_tracing.enabled            = true;
    config.ray_tracing.use_fresnel        = true;
    config.ray_tracing.fresnel_n_refract  = 2.9613f;   // 316L @ 1064 nm Mills
    config.ray_tracing.fresnel_k_extinct  = 4.0133f;
    // Phase-1 (Night Protocol): reduced ray-tracing cost — first attempt
    // with 4096 rays / 5 bounces / 1500 DDA steps stalled at 0.12 step/sec
    // because the deeper, narrower keyhole (from 39μm spot) trapped rays in
    // many-bounce paths. 2048 rays + 3 bounces still resolve 39μm Gaussian
    // (Δθ ≈ 360°/2048 = 0.18°) and capture the dominant absorption (3
    // bounces ≈ 88% of asymptotic 5-bounce energy at α₀=0.31 metal Fresnel).
    config.ray_tracing.num_rays           = 2048;
    config.ray_tracing.max_bounces        = 3;
    config.ray_tracing.max_dda_steps      = 800;
    config.ray_tracing.energy_cutoff      = 0.02f;
    config.ray_tracing.absorptivity       = 0.40f;      // unused when use_fresnel=true

    // --- Fluid ---
    // PHASE-3 CHANGE: ν_LU 0.065 → 0.0167 (τ 0.7 → 0.55).
    // Removes the 4.3× artificial viscosity inflation. ν_phys becomes
    // 8.35e-7 m²/s, within 1% of 316L's 8.28e-7 (true Mills viscosity).
    // Risk: BGK at τ=0.55 close to instability with strong Marangoni
    // shear. NaN crash possible — protocol says fall back to Phase-1
    // best if so.
    config.kinematic_viscosity      = 0.0167f;
    config.density                  = config.material.rho_liquid;

    // --- Darcy (linear) ---
    config.darcy_coefficient        = 5.0e4f;

    // --- Thermal ---
    config.thermal_diffusivity      = config.material.getThermalDiffusivity(1700.0f);
    config.ambient_temperature      = 300.0f;
    config.emissivity               = config.material.emissivity;

    // --- Surface ---
    config.surface_tension_coeff    = config.material.surface_tension;
    config.dsigma_dT                = config.material.dsigma_dT;
    // S2-A1 (csf=2.0) made groove DEEPER (-24μm vs -20). Marangoni stronger
    // pushes liquid further to edges, where it sinks down instead of rising.
    // Reverted to 1.0. Real fix needs different mechanism.
    config.marangoni_csf_multiplier = 1.0f;
    // S2-A2 tested recoil_smoothing_width=5 (vs default 30K ramp) — ZERO effect
    // on raised track. Reverted to default. Sprint-2 finding: the LBM cannot
    // form central ridge because laser deposition geometry (column-march, no
    // sharp keyhole wall) produces shallow-bowl pool with low surface
    // curvature → weak capillary backflow → groove never refills. Architectural,
    // not parametric.
    // (config.recoil_smoothing_width uses default 30 K from MultiphysicsConfig)

    // --- Buoyancy ---
    config.thermal_expansion_coeff  = 1.2e-4f;
    config.gravity_x = 0.0f;
    config.gravity_y = 0.0f;
    config.gravity_z = -9.81f;
    config.reference_temperature    = 0.5f * (config.material.T_solidus + config.material.T_liquidus);

    // --- Substrate cooling ---
    config.enable_substrate_cooling = true;
    config.substrate_h_conv         = 2000.0f;
    config.substrate_temperature    = 300.0f;

    // --- Boundaries ---
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

    // --- CFL ---
    config.cfl_use_adaptive            = true;
    config.cfl_v_target_interface      = 0.15f;
    config.cfl_v_target_bulk           = 0.10f;

    // --- VOF ---
    config.vof_subcycles               = 1;
    config.enable_vof_mass_correction  = false;

    // --- Timing: Phase-1 SHORT test (Night Protocol) ---
    // 800 μs = 10000 steps. VTK every 100 μs (8 frames + initial).
    const float t_total  = 800.0e-6f;
    const int num_steps  = static_cast<int>(t_total / config.dt);
    const int vtk_every  = static_cast<int>(100.0e-6f / config.dt);  // 100 μs cadence
    const int diag_every = 500;
    // Substrate top at z = 160 μm (k=80). LBM domain z∈[0,200]μm:
    // metal [0,160] (80 cells), gas [160,200] (20 cells). Keyhole 80μm深 → 离底
    // 80μm 缓冲（vs M13b 仅 20μm，避免 punch-through），离顶 20μm OK。
    const float interface_z = 80.0f;  // z=80 cells × dx=2μm = 160 μm

    // ==================================================================
    // Print summary
    // ==================================================================
    printf("Domain:  %d × %d × %d cells = %d × %d × %d μm\n",
           config.nx, config.ny, config.nz,
           (int)(config.nx * config.dx * 1e6f),
           (int)(config.ny * config.dx * 1e6f),
           (int)(config.nz * config.dx * 1e6f));
    printf("dx = %.0f μm, dt = %.0f ns\n", config.dx * 1e6f, config.dt * 1e9f);

    printf("\nMaterial: %s\n", config.material.name);
    printf("  T_solidus=%.0f K, T_liquidus=%.0f K, T_boil=%.0f K\n",
           config.material.T_solidus, config.material.T_liquidus,
           config.material.T_vaporization);
    printf("  dσ/dT = %.1e N/(m·K) [%s]\n",
           config.dsigma_dT, config.dsigma_dT < 0 ? "outward Marangoni" : "inward");

    printf("\nLaser: P=%.0f W, r₀=%.0f μm, η=%.0f%%\n",
           config.laser_power, config.laser_spot_radius * 1e6f,
           config.laser_absorptivity * 100.0f);
    printf("  v_scan = %.0f mm/s (+x direction)\n", v_scan * 1e3f);
    printf("  Start: x=%.0f μm, track=%.0f μm (v×t)\n",
           config.laser_start_x * 1e6f, v_scan * t_total * 1e6f);

    printf("\nFluid: ν_LU=%.3f, τ=%.3f\n",
           config.kinematic_viscosity,
           1.0f / (3.0f * config.kinematic_viscosity + 0.5f));
    printf("Darcy: K=%.0e (linear)\n", config.darcy_coefficient);
    printf("Steps: %d (%.0f μs), VTK every %d steps (%.0f μs)\n\n",
           num_steps, t_total * 1e6f, vtk_every, vtk_every * config.dt * 1e6f);
    fflush(stdout);

    // ==================================================================
    // Initialize
    // ==================================================================
    mkdir("output_phase3", 0755);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.80f);

    const auto& registry = solver.getFieldRegistry();

    // Host buffers for melt pool metrics
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_lf(num_cells), h_fl(num_cells);

    // Initial mass (for conservation tracking)
    float initial_mass = solver.getTotalMass();
    printf("Initial metal volume (Σfill): %.0f\n\n", initial_mass);

    // ==================================================================
    // Console header
    // ==================================================================
    printf("%-6s %7s %7s %7s %7s %7s %7s %9s %8s\n",
           "Step", "t[μs]", "T_max", "v_max", "Depth", "Length", "Width", "Laser_x", "MassΔ%");
    printf("%-6s %7s %7s %7s %7s %7s %7s %9s %8s\n",
           "", "", "[K]", "[m/s]", "[μm]", "[μm]", "[μm]", "[μm]", "");
    printf("----------------------------------------------------------------------\n");
    fflush(stdout);

    // ==================================================================
    // Time integration
    // ==================================================================
    for (int step = 0; step <= num_steps; ++step) {
        float t = step * config.dt;
        float laser_x = config.laser_start_x + v_scan * t;

        // --- Diagnostics ---
        if (step % diag_every == 0) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();

            // Melt pool dimensions
            solver.copyTemperatureToHost(h_lf.data());  // reuse buffer
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
            float mass_delta = (current_mass - initial_mass) / initial_mass * 100.0f;

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

        // --- VTK output ---
        if (step % vtk_every == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "output_phase3/line_scan_%06d", step);
            io::VTKWriter::writeFields(
                std::string(filename), registry, {},
                config.nx, config.ny, config.nz, config.dx);
            printf("  → VTK: %s.vtk (t=%.0f μs, laser_x=%.0f μm)\n",
                   filename, t * 1e6f, laser_x * 1e6f);
            fflush(stdout);
        }

        // --- Step ---
        if (step < num_steps) {
            solver.step();
        }
    }

    // ==================================================================
    // Performance metrics
    // ==================================================================
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    double wall_minutes = wall_seconds / 60.0;

    long long total_cells = (long long)config.nx * config.ny * config.nz;
    long long total_updates = total_cells * (long long)num_steps;
    double GLUPS = total_updates / wall_seconds / 1e9;
    double MLUPS = total_updates / wall_seconds / 1e6;

    // ==================================================================
    // Final summary
    // ==================================================================
    printf("\n============================================================\n");
    printf("  Line Scan Complete\n");
    printf("============================================================\n");
    printf("  Total steps: %d\n", num_steps);
    printf("  Sim time:    %.0f μs\n", t_total * 1e6f);
    printf("  Track: x = %.0f → %.0f μm at %.0f mm/s\n",
           config.laser_start_x * 1e6f,
           (config.laser_start_x + v_scan * t_total) * 1e6f,
           v_scan * 1e3f);
    printf("  T_max: %.0f K\n", solver.getMaxTemperature());
    printf("  v_max: %.3f m/s\n", solver.getMaxVelocity());
    printf("  Mass:  %+.3f%%\n", (solver.getTotalMass() - initial_mass) / initial_mass * 100.0f);
    printf("\n  === Performance ===\n");
    printf("  Wall clock: %.1f s (%.1f min)\n", wall_seconds, wall_minutes);
    printf("  Grid: %lld cells (%dx%dx%d)\n", total_cells, config.nx, config.ny, config.nz);
    printf("  Total updates: %.2e\n", (double)total_updates);
    printf("  Throughput: %.2f MLUPS (%.4f GLUPS)\n", MLUPS, GLUPS);
    printf("  Steps/sec: %.1f\n", num_steps / wall_seconds);
    printf("\nOutput: output_phase3/line_scan_*.vtk\n");
    printf("============================================================\n\n");

    return 0;
}
