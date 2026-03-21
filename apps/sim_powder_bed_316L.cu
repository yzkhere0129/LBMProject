/**
 * @file sim_powder_bed_316L.cu
 * @brief LPBF powder bed fusion simulation — 316L with realistic powder layer
 *
 * Loads pre-generated powder bed (binary fill_level from generate_powder_bed.py)
 * and runs full multiphysics laser scanning simulation.
 *
 * Domain: 1000×300×150 μm (500×150×75 cells at dx=2μm)
 *   z=0-60μm: solid substrate (f=1)
 *   z=60-100μm: powder layer (~45% packing, 299 particles)
 *   z=100-150μm: gas buffer (f=0)
 *
 * Physics: All modules ON (honest model — no T cap, C-C evaporation)
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
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
 * @brief Load binary powder bed fill_level from Python generator
 * Format: [NX:int32][NY:int32][NZ:int32][NX*NY*NZ floats, z-slowest]
 */
static bool loadPowderBed(const char* path, std::vector<float>& fill,
                           int expected_nx, int expected_ny, int expected_nz) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: Cannot open %s\n", path);
        return false;
    }

    int nx, ny, nz;
    fread(&nx, sizeof(int), 1, f);
    fread(&ny, sizeof(int), 1, f);
    fread(&nz, sizeof(int), 1, f);

    if (nx != expected_nx || ny != expected_ny || nz != expected_nz) {
        printf("ERROR: Grid mismatch! File: %dx%dx%d, Expected: %dx%dx%d\n",
               nx, ny, nz, expected_nx, expected_ny, expected_nz);
        fclose(f);
        return false;
    }

    int n = nx * ny * nz;
    // Python stores as (NZ, NY, NX) row-major → z-slowest, x-fastest
    // LBM expects x + nx*(y + ny*z) = x-fastest, z-slowest → same layout!
    std::vector<float> raw(n);
    size_t read = fread(raw.data(), sizeof(float), n, f);
    fclose(f);

    if ((int)read != n) {
        printf("ERROR: Expected %d values, read %zu\n", n, read);
        return false;
    }

    // Convert from Python (k,j,i) to LBM (i + nx*(j + ny*k))
    // Both are x-fastest, z-slowest — direct copy
    fill = raw;

    printf("Loaded powder bed: %dx%dx%d from %s\n", nx, ny, nz, path);

    // Statistics
    int n_solid = 0, n_gas = 0, n_interface = 0;
    for (int i = 0; i < n; ++i) {
        if (fill[i] > 0.99f) n_solid++;
        else if (fill[i] < 0.01f) n_gas++;
        else n_interface++;
    }
    printf("  Solid (f>0.99): %d cells (%.1f%%)\n", n_solid, 100.0f*n_solid/n);
    printf("  Gas (f<0.01):   %d cells (%.1f%%)\n", n_gas, 100.0f*n_gas/n);
    printf("  Interface:      %d cells (%.1f%%)\n", n_interface, 100.0f*n_interface/n);

    return true;
}

int main() {
    auto wall_start = std::chrono::high_resolution_clock::now();

    printf("\n");
    printf("============================================================\n");
    printf("  LPBF Powder Bed Fusion: 316L Stainless Steel\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Configuration
    // ==================================================================
    MultiphysicsConfig config;

    // Domain: 1000×300×130 μm
    //   z=0-60μm:   substrate (30 cells)
    //   z=60-90μm:  powder layer 30μm (15 cells)
    //   z=90-130μm: gas buffer 40μm (20 cells)
    config.nx = 500;
    config.ny = 150;
    config.nz = 65;
    config.dx = 2.0e-6f;
    config.dt = 8.0e-8f;

    config.material = MaterialDatabase::get316L();

    // Full physics (honest model)
    config.enable_thermal           = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change      = true;
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

    // Laser: 316L standard LPBF — P=150W, r₀=50μm, v=800mm/s
    const float v_scan = 0.8f;          // 800 mm/s scan speed
    config.laser_power              = 150.0f;   // [W]
    config.laser_spot_radius        = 35.0e-6f; // [m] 35 μm 1/e² radius (industrial standard)
    config.laser_absorptivity       = 0.35f;    // Base absorptivity (ray tracing adds multi-reflection)
    config.laser_penetration_depth  = 10.0e-6f; // [m] (Beer-Lambert fallback only)
    config.laser_start_x            = 50.0e-6f; // Start 50μm from left wall
    config.laser_start_y            = -1.0f;    // Auto-center Y
    config.laser_scan_vx            = v_scan;
    config.laser_scan_vy            = 0.0f;

    // Ray Tracing — geometric multi-reflection in powder bed
    config.ray_tracing.enabled          = true;
    config.ray_tracing.num_rays         = 2048;
    config.ray_tracing.max_bounces      = 3;
    config.ray_tracing.absorptivity     = 0.35f;   // 316L base absorptivity
    config.ray_tracing.energy_cutoff    = 0.01f;
    config.ray_tracing.max_dda_steps    = 500;

    // Fluid
    config.kinematic_viscosity      = 0.065f;
    config.density                  = config.material.rho_liquid;
    config.darcy_coefficient        = 5.0e4f;

    // Thermal
    config.thermal_diffusivity      = config.material.getThermalDiffusivity(1700.0f);
    config.ambient_temperature      = 600.0f; // Preheated build plate
    config.emissivity               = config.material.emissivity;

    // Surface
    config.surface_tension_coeff    = config.material.surface_tension;
    config.dsigma_dT                = config.material.dsigma_dT;

    // Buoyancy
    config.thermal_expansion_coeff  = 1.2e-4f;
    config.gravity_x = 0.0f;
    config.gravity_y = 0.0f;
    config.gravity_z = -9.81f;
    config.reference_temperature    = 0.5f * (config.material.T_solidus + config.material.T_liquidus);

    // Substrate cooling — DIRICHLET at z=0 (constant 300K heat sink)
    config.enable_substrate_cooling = true;
    config.substrate_h_conv         = 2000.0f;
    config.substrate_temperature    = 600.0f;

    // Boundaries
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
    config.boundaries.thermal_z_min = ThermalBCType::DIRICHLET;  // 300K heat sink
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature = 600.0f; // Preheat from prior layers

    // CFL — scientifically relaxed to allow Marangoni wetting
    // LBM stability limit: 1/√3 ≈ 0.577 LU. Cap at 0.38 gives 34% safety margin.
    config.cfl_use_adaptive            = true;
    config.cfl_v_target_interface      = 0.38f;  // ~9.5 m/s physical
    config.cfl_v_target_bulk           = 0.38f;  // same — no asymmetric throttling

    // VOF
    config.vof_subcycles               = 1;
    config.enable_vof_mass_correction  = true;  // Global mass redistribution each step

    // Timing — full scan across domain
    // Laser travels 900μm at 800mm/s → 1125μs, + 200μs cooldown
    const float t_total  = 1300.0e-6f;  // 1300 μs
    const int num_steps  = static_cast<int>(t_total / config.dt);
    const int vtk_every  = static_cast<int>(50.0e-6f / config.dt);  // VTK every 50μs
    const int diag_every = 1000;

    // ==================================================================
    // Print config
    // ==================================================================
    printf("Domain:  %d×%d×%d cells = %.0f×%.0f×%.0f μm\n",
           config.nx, config.ny, config.nz,
           config.nx * config.dx * 1e6f,
           config.ny * config.dx * 1e6f,
           config.nz * config.dx * 1e6f);
    printf("dx=%.0f μm, dt=%.0f ns\n", config.dx*1e6f, config.dt*1e9f);
    printf("Laser: P=%.0fW, r0=%.0fμm, v=%.0fmm/s\n",
           config.laser_power, config.laser_spot_radius*1e6f, v_scan*1e3f);
    printf("Ray Tracing: %s (%d rays, %d bounces, alpha=%.2f)\n",
           config.ray_tracing.enabled ? "ON" : "OFF",
           config.ray_tracing.num_rays,
           config.ray_tracing.max_bounces,
           config.ray_tracing.absorptivity);
    printf("Bottom BC: DIRICHLET T=%.0f K (heat sink)\n",
           config.boundaries.dirichlet_temperature);
    printf("Steps: %d (%.0f μs)\n\n", num_steps, t_total*1e6f);

    // ==================================================================
    // Load powder bed
    // ==================================================================
    std::vector<float> h_fill;
    const char* powder_path = "output_powder_bed/powder_bed_fill_level.bin";
    if (!loadPowderBed(powder_path, h_fill, config.nx, config.ny, config.nz)) {
        printf("FATAL: Failed to load powder bed\n");
        return 1;
    }

    // ==================================================================
    // Initialize solver with custom fill_level
    // ==================================================================
    mkdir("output_powder_bed_sim", 0755);

    printf("\nInitializing MultiphysicsSolver...\n");
    MultiphysicsSolver solver(config);

    // Initialize with uniform T=300K and the loaded powder bed fill_level
    std::vector<float> h_temp(config.nx * config.ny * config.nz, 600.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    const auto& registry = solver.getFieldRegistry();
    int num_cells = config.nx * config.ny * config.nz;

    float initial_mass = solver.getTotalMass();
    printf("Initial mass (Σfill): %.0f\n\n", initial_mass);

    // ==================================================================
    // Write initial state VTK (t=0)
    // ==================================================================
    {
        io::VTKWriter::writeFields("output_powder_bed_sim/powder_sim_000000",
                                    registry, {},
                                    config.nx, config.ny, config.nz, config.dx);
        printf("→ VTK: output_powder_bed_sim/powder_sim_000000.vtk (t=0, initial state)\n\n");
    }

    // ==================================================================
    // Console header
    // ==================================================================
    printf("%-6s %7s %7s %7s %9s %6s %8s\n",
           "Step", "t[μs]", "T_max", "v_max", "Laser_x", "α_eff", "MassΔ%");
    printf("%-6s %7s %7s %7s %9s %6s %8s\n",
           "", "", "[K]", "[m/s]", "[μm]", "", "");
    printf("------------------------------------------------------------\n");
    fflush(stdout);

    // ==================================================================
    // Time integration
    // ==================================================================
    for (int step = 0; step <= num_steps; ++step) {
        float t = step * config.dt;
        float laser_x = config.laser_start_x + v_scan * t;

        if (step % diag_every == 0) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();
            float mass = solver.getTotalMass();
            float mass_delta = (mass - initial_mass) / initial_mass * 100.0f;
            float alpha_eff = solver.hasRayTracing()
                ? solver.getRayTracingEffectiveAbsorptivity()
                : config.laser_absorptivity;

            printf("%-6d %7.1f %7.0f %7.3f %9.1f %6.3f %+7.3f%%\n",
                   step, t*1e6f, T_max, v_max, laser_x*1e6f, alpha_eff, mass_delta);
            fflush(stdout);

            if (solver.checkNaN()) {
                printf("\n*** FATAL: NaN at step %d ***\n", step);
                break;
            }
        }

        if (step % vtk_every == 0 && step > 0) {
            char fname[256];
            snprintf(fname, sizeof(fname), "output_powder_bed_sim/powder_sim_%06d", step);
            io::VTKWriter::writeFields(std::string(fname), registry, {},
                                        config.nx, config.ny, config.nz, config.dx);
            printf("  → VTK: %s.vtk (t=%.0fμs)\n", fname, t*1e6f);
            fflush(stdout);
        }

        if (step < num_steps) {
            solver.step();
        }
    }

    // ==================================================================
    // Performance
    // ==================================================================
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();
    long long total_updates = (long long)num_cells * num_steps;

    printf("\n============================================================\n");
    printf("  Powder Bed Simulation Complete\n");
    printf("============================================================\n");
    printf("  Steps: %d (%.0f μs)\n", num_steps, t_total*1e6f);
    printf("  T_max: %.0f K\n", solver.getMaxTemperature());
    printf("  v_max: %.3f m/s\n", solver.getMaxVelocity());
    printf("  Mass:  %+.3f%%\n",
           (solver.getTotalMass() - initial_mass) / initial_mass * 100.0f);
    if (solver.hasRayTracing()) {
        printf("  α_eff: %.3f (base=%.3f, boost=%.1fx from multi-reflection)\n",
               solver.getRayTracingEffectiveAbsorptivity(),
               config.ray_tracing.absorptivity,
               solver.getRayTracingEffectiveAbsorptivity() / config.ray_tracing.absorptivity);
        printf("  RT:    dep=%.1fW, input=%.1fW, energy_err=%.2e\n",
               solver.getRayTracingDepositedPower(),
               solver.getRayTracingInputPower(),
               solver.getRayTracingEnergyError());
    }
    printf("  Wall:  %.1f s (%.1f min)\n", wall_s, wall_s/60.0);
    printf("  MLUPS: %.2f\n", total_updates / wall_s / 1e6);
    printf("============================================================\n\n");

    return 0;
}
