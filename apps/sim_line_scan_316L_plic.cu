/**
 * @file sim_line_scan_316L_plic.cu
 * @brief LPBF single-track line scan on 316L — PLIC-VOF variant
 *
 * Near-clone of sim_line_scan_316L.cu with PLIC full-stack enabled.
 * ALL grid / dt / physics parameters are IDENTICAL to the legacy app
 * so the two runs are directly comparable (apples-to-apples).
 *
 * PLIC additions vs legacy:
 *   - cfg.enableFullPLICStack(1.5f) activates all 5 PLIC flags:
 *       laser.plic_aware_column_march    = true
 *       surface.csf_use_plic_delta       = true
 *       surface.marangoni_use_plic_delta = true
 *       surface.recoil_use_plic_delta    = true
 *       surface.evap_use_plic_delta      = true
 *       surface.plic_h_smooth_lu         = 1.5
 *   - VOFSolver normal/curvature methods upgraded automatically in
 *     MultiphysicsSolver::initialize() when any PLIC flag is set.
 *
 * F3D reference (150W / 800 mm/s / r₀=50 μm):
 *   Pool W=73 μm, L=438 μm, D=78 μm, T_max≈4013 K, v_max≈7 m/s
 *   NOTE: F3D uses 70% effective absorptivity; this app uses 40%.
 *         That gap is the dominant source of pool-size under-prediction
 *         and is INDEPENDENT of PLIC vs legacy VOF advection.
 *
 * Output: VTK every 50 μs → output_line_scan_316L_plic/
 * Usage:
 *   mkdir -p output_line_scan_316L_plic && ./build/sim_line_scan_316L_plic
 *   ParaView: Open output_line_scan_316L_plic/line_scan_plic_*.vtk
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
 * @brief Compute melt pool dimensions from liquid fraction field.
 * Identical to sim_line_scan_316L.cu — shared helper.
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
    printf("  [PLIC] LPBF Single-Track Line Scan: 316L (PLIC-VOF Run)\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Configuration  — IDENTICAL to sim_line_scan_316L except for
    // enableFullPLICStack() and output directory.
    // ==================================================================
    MultiphysicsConfig config;

    // --- Domain: 2200 × 200 × 200 μm (1100×100×100 cells at dx=2μm) ---
    config.nx = 1100;
    config.ny = 100;   // match truth-floor
    config.nz = 100;
    config.dx = 2.0e-6f;
    config.dt = 8.0e-8f;   // 80 ns

    // --- Material ---
    config.material = MaterialDatabase::get316L();
    config.material.T_solidus  = 1674.15f;  // F3D Mills table
    config.material.T_liquidus = 1697.15f;

    // --- Physics ---
    config.enable_thermal           = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change      = true;
    config.use_fdm_thermal          = true;  // match truth-floor
    config.enable_fluid             = true;
    config.enable_vof               = true;
    config.enable_vof_advection     = true;
    config.enable_laser             = true;
    config.enable_darcy             = true;
    config.enable_marangoni         = true;
    config.enable_surface_tension   = true;
    config.enable_buoyancy          = true;
    config.enable_evaporation_mass_loss = true;
    config.enable_recoil_pressure   = true;
    config.enable_radiation_bc      = true;

    // --- Moving laser (identical to legacy) ---
    const float v_scan = 0.8f;  // 800 mm/s
    config.laser_power              = 150.0f;
    config.laser_spot_radius        = 39.0e-6f;  // F3D dum2=39e-6
    config.laser_absorptivity       = 0.40f;   // NOTE: F3D uses 0.70; see header
    config.laser_penetration_depth  = 10.0e-6f;
    config.laser_start_x            = 500.0e-6f;  // match truth-floor
    config.laser_start_y            = -1.0f;   // auto-center Y
    config.laser_scan_vx            = v_scan;
    config.laser_scan_vy            = 0.0f;

    // --- Ray-tracing laser ---
    config.ray_tracing.enabled            = true;
    config.ray_tracing.use_fresnel        = true;
    config.ray_tracing.fresnel_n_refract  = 2.9613f;
    config.ray_tracing.fresnel_k_extinct  = 4.0133f;
    config.ray_tracing.num_rays           = 4096;
    config.ray_tracing.max_bounces        = 3;
    config.ray_tracing.max_dda_steps      = 1500;
    config.ray_tracing.energy_cutoff      = 0.01f;
    config.ray_tracing.absorptivity       = 0.40f;

    // --- Fluid ---
    config.kinematic_viscosity      = 0.065f;
    config.density                  = config.material.rho_liquid;

    // --- Darcy ---
    config.darcy_coefficient        = 5.0e4f;

    // --- Thermal ---
    config.thermal_diffusivity      = config.material.getThermalDiffusivity(1700.0f);
    config.ambient_temperature      = 300.0f;
    config.emissivity               = config.material.emissivity;

    // --- Surface ---
    config.surface_tension_coeff    = config.material.surface_tension;
    config.dsigma_dT                = config.material.dsigma_dT;

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
    config.boundaries.thermal_z_max = ThermalBCType::RADIATION;
    config.boundaries.convective_h     = 2000.0f;
    config.boundaries.convective_T_inf = 300.0f;
    config.boundaries.radiation_emissivity = 0.55f;
    config.boundaries.radiation_T_ambient  = 300.0f;

    // --- CFL ---
    config.cfl_use_adaptive            = true;
    config.cfl_v_target_interface      = 0.15f;
    config.cfl_v_target_bulk           = 0.10f;

    // --- VOF ---
    config.vof_subcycles               = 1;
    config.enable_vof_mass_correction  = false;

    // ==================================================================
    // PLIC full-stack  — the ONLY delta vs sim_line_scan_316L.cu
    // Enables all 5 PLIC flags + sets plic_h_smooth_lu = 1.5 cells.
    // MultiphysicsSolver::initialize() auto-upgrades VOFSolver normal
    // reconstruction (HEIGHT_FUNCTION) and curvature (PLIC_DIVERGENCE).
    // ==================================================================
    config.enableFullPLICStack(1.5f);

    printf("[PLIC] Full PLIC stack enabled:\n");
    printf("  plic_aware_column_march    = true\n");
    printf("  csf_use_plic_delta         = true\n");
    printf("  marangoni_use_plic_delta   = true\n");
    printf("  recoil_use_plic_delta      = true\n");
    printf("  evap_use_plic_delta        = true\n");
    printf("  plic_h_smooth_lu           = 1.5\n");
    printf("  VOF normal/curvature method: HEIGHT_FUNCTION / PLIC_DIVERGENCE\n\n");

    // --- Timing: 1.0 ms run (same as legacy) ---
    const float t_total  = 2000.0e-6f;  // match truth-floor
    const int num_steps  = static_cast<int>(t_total / config.dt);
    const int vtk_every  = static_cast<int>(50.0e-6f / config.dt);
    const int diag_every = 1000;
    const float interface_z = 0.80f * config.nz;  // z=40 cells

    // ==================================================================
    // Print summary
    // ==================================================================
    printf("Domain:  %d × %d × %d cells = %d × %d × %d μm\n",
           config.nx, config.ny, config.nz,
           (int)(config.nx * config.dx * 1e6f),
           (int)(config.ny * config.dx * 1e6f),
           (int)(config.nz * config.dx * 1e6f));
    printf("dx = %.0f μm, dt = %.0f ns\n", config.dx * 1e6f, config.dt * 1e9f);
    printf("Material: %s\n", config.material.name);
    printf("Laser: P=%.0f W, r0=%.0f μm, eta=%.0f%%\n",
           config.laser_power, config.laser_spot_radius * 1e6f,
           config.laser_absorptivity * 100.0f);
    printf("  v_scan = %.0f mm/s (+x)\n", v_scan * 1e3f);
    printf("Steps: %d (%.0f μs), VTK every %d steps (%.0f μs)\n\n",
           num_steps, t_total * 1e6f, vtk_every, vtk_every * config.dt * 1e6f);
    fflush(stdout);

    // ==================================================================
    // Initialize
    // ==================================================================
    mkdir("output_line_scan_316L_plic", 0755);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.80f);
    solver.setRegularized(true, 0.5456f);  // match truth-floor

    const auto& registry = solver.getFieldRegistry();

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_lf(num_cells), h_fl(num_cells);

    float initial_mass = solver.getTotalMass();
    printf("[PLIC] Initial metal volume (Σfill): %.0f\n\n", initial_mass);

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
    float T_max_final = 0.0f;
    float v_max_final = 0.0f;

    for (int step = 0; step <= num_steps; ++step) {
        float t = step * config.dt;
        float laser_x = config.laser_start_x + v_scan * t;

        // --- Diagnostics ---
        if (step % diag_every == 0) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();
            T_max_final = T_max;
            v_max_final = v_max;

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

            printf("[PLIC] %-6d %7.1f %7.0f %7.3f %7.1f %7.1f %7.1f %9.1f %+7.3f%%\n",
                   step, t * 1e6f, T_max, v_max,
                   mp.depth_um, mp.length_um, mp.width_um,
                   laser_x * 1e6f, mass_delta);
            fflush(stdout);

            if (solver.checkNaN()) {
                printf("\n[PLIC] *** FATAL: NaN detected at step %d ***\n", step);
                break;
            }
        }

        // --- VTK output ---
        if (step % vtk_every == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "output_line_scan_316L_plic/line_scan_plic_%06d", step);
            io::VTKWriter::writeFields(
                std::string(filename), registry, {},
                config.nx, config.ny, config.nz, config.dx);
            printf("[PLIC]   VTK: %s.vtk (t=%.0f μs)\n", filename, t * 1e6f);
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

    long long total_cells   = (long long)config.nx * config.ny * config.nz;
    long long total_updates = total_cells * (long long)num_steps;
    double MLUPS = total_updates / wall_seconds / 1e6;

    float final_mass  = solver.getTotalMass();
    float mass_loss_pct = (final_mass - initial_mass) / initial_mass * 100.0f;

    // ==================================================================
    // Final summary  (tagged [PLIC] for easy grep)
    // ==================================================================
    printf("\n============================================================\n");
    printf("  [PLIC] Line Scan Complete (PLIC-VOF)\n");
    printf("============================================================\n");
    printf("  Total steps: %d\n", num_steps);
    printf("  Sim time:    %.0f μs\n", t_total * 1e6f);
    printf("  Track:       x = %.0f → %.0f μm at %.0f mm/s\n",
           config.laser_start_x * 1e6f,
           (config.laser_start_x + v_scan * t_total) * 1e6f,
           v_scan * 1e3f);
    printf("\n");
    // Single-line summary for main-process parsing:
    printf("[PLIC] SUMMARY | T_max=%.0f K | v_max=%.3f m/s | mass_loss=%+.3f%% | wall_time=%.1f s (%.1f min)\n",
           T_max_final, v_max_final, mass_loss_pct, wall_seconds, wall_minutes);
    printf("\n");
    printf("  === Performance ===\n");
    printf("  Wall clock: %.1f s (%.1f min)\n", wall_seconds, wall_minutes);
    printf("  Throughput: %.2f MLUPS\n", MLUPS);
    printf("  Steps/sec:  %.1f\n", num_steps / wall_seconds);
    printf("\nOutput: output_line_scan_316L_plic/line_scan_plic_*.vtk\n");
    printf("============================================================\n\n");

    return 0;
}
