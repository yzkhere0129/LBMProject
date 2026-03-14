/**
 * @file sim_spot_melt_316L.cu
 * @brief Stationary laser spot melting on 316L stainless steel substrate
 *
 * Physics:
 *   - Gaussian laser (P=150W, w0=50μm, η=0.4) on flat 316L substrate
 *   - Thermal conduction + convection (ThermalLBM D3Q7)
 *   - Phase change via Enthalpy Source Term (solid→mushy→liquid)
 *   - Melt pool convection (FluidLBM D3Q19 TRT+EDM)
 *   - Marangoni thermocapillary flow (dσ/dT < 0 → outward radial flow)
 *   - Surface tension (CSF model)
 *   - Buoyancy (Boussinesq)
 *   - Darcy mushy-zone damping
 *   - VOF free surface (PLIC geometric advection)
 *   - Recoil pressure: OFF (isolates Marangoni effect)
 *
 * Domain:
 *   150 × 150 × 100 μm  (75 × 75 × 50 cells at dx = 2 μm)
 *   Bottom 80%: solid 316L (f=1, T=300K)
 *   Top 20%: inert gas (f=0)
 *
 * Output:
 *   VTK files every 50 μs with Temperature, Velocity, LiquidFraction, FillLevel
 *   Console: T_max, v_max, melt pool depth/width every 1000 steps
 *
 * Expected behavior:
 *   1. Conduction-dominated melting (0–100 μs): hemispherical isotherm
 *   2. Marangoni onset (100–200 μs): radial surface flow, vortex pair forms
 *   3. Convection-dominated (200+ μs): wide shallow pool (Marangoni flattening)
 *
 * Usage:
 *   mkdir -p output_spot_melt && ./sim_spot_melt_316L
 *   ParaView: File → Open → output_spot_melt/spot_melt_*.vtk
 *     - Color by Temperature, Contour by LiquidFraction=0.5 (melt boundary)
 *     - Glyph or StreamTracer on Velocity for Marangoni vortices
 */

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"
#include "io/field_registry.h"
#include "core/lattice_d3q19.h"

using namespace lbm;
using namespace lbm::physics;

int main() {
    printf("\n");
    printf("============================================================\n");
    printf("  Spot Melting: 316L Stainless Steel (Marangoni Convection)\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Configuration
    // ==================================================================
    MultiphysicsConfig config;

    // --- Domain: 150 × 150 × 100 μm ---
    config.nx = 75;
    config.ny = 75;
    config.nz = 50;
    config.dx = 2.0e-6f;  // 2 μm resolution

    // --- Time step ---
    // dt must satisfy thermal CFL: α·dt/dx² < 1/6 (3D D3Q7)
    // 316L liquid α ≈ 5.6e-6 m²/s → dt < dx²/(6α) = 4e-12/3.36e-5 ≈ 1.19e-7
    // Use dt = 1e-7 s = 100 ns (safe margin)
    config.dt = 1.0e-7f;

    // --- Material ---
    config.material = MaterialDatabase::get316L();

    // --- Physics modules ---
    config.enable_thermal           = true;
    config.enable_thermal_advection = true;   // v·∇T coupling
    config.enable_phase_change      = true;   // Enthalpy Source Term
    config.enable_fluid             = true;   // Navier-Stokes
    config.enable_vof               = true;   // Free surface tracking
    config.enable_vof_advection     = true;   // Interface motion
    config.enable_laser             = true;   // Gaussian heat source
    config.enable_darcy             = true;   // Mushy-zone damping
    config.enable_marangoni         = true;   // Thermocapillary forces
    config.enable_surface_tension   = true;   // Capillary pressure
    config.enable_buoyancy          = true;   // Thermal buoyancy
    config.enable_evaporation_mass_loss = true;  // Evaporation cooling
    config.enable_recoil_pressure   = false;  // OFF for this run
    config.enable_radiation_bc      = true;   // Stefan-Boltzmann

    // --- Laser: stationary spot at domain center ---
    config.laser_power              = 150.0f;       // 150 W
    config.laser_spot_radius        = 50.0e-6f;     // 50 μm (1/e² radius)
    config.laser_absorptivity       = 0.40f;        // 40% absorption
    config.laser_penetration_depth  = 10.0e-6f;     // Beer-Lambert depth
    config.laser_start_x            = -1.0f;        // Auto-center
    config.laser_start_y            = -1.0f;        // Auto-center
    config.laser_scan_vx            = 0.0f;         // Stationary
    config.laser_scan_vy            = 0.0f;

    // --- Fluid: lattice viscosity for tau ≈ 0.6 ---
    // nu_phys(316L liquid) ≈ μ/ρ = 6e-3/6900 ≈ 8.7e-7 m²/s
    // nu_lattice = nu_phys * dt/dx² = 8.7e-7 * 1e-7 / 4e-12 = 0.0218
    // tau = 3*0.0218 + 0.5 = 0.565 (safe with TRT)
    config.kinematic_viscosity      = 0.0218f;      // Lattice units
    config.density                  = config.material.rho_liquid;

    // --- Darcy mushy-zone damping ---
    config.darcy_coefficient        = 1.0e6f;       // Strong enough to freeze mushy zone

    // --- Thermal ---
    config.thermal_diffusivity      = config.material.getThermalDiffusivity(1700.0f);
    config.ambient_temperature      = 300.0f;
    config.emissivity               = config.material.emissivity;

    // --- Surface properties ---
    config.surface_tension_coeff    = config.material.surface_tension;
    config.dsigma_dT                = config.material.dsigma_dT;

    // --- Buoyancy ---
    config.thermal_expansion_coeff  = 1.2e-4f;      // β for liquid steel [1/K]
    config.gravity_x                = 0.0f;
    config.gravity_y                = 0.0f;
    config.gravity_z                = -9.81f;        // Downward
    config.reference_temperature    = 0.5f * (config.material.T_solidus + config.material.T_liquidus);

    // --- Substrate cooling ---
    config.enable_substrate_cooling = true;
    config.substrate_h_conv         = 2000.0f;       // h_conv [W/(m²·K)]
    config.substrate_temperature    = 300.0f;

    // --- Boundary conditions (per-face) ---
    // x/y: WALL + ADIABATIC (large enough domain that heat doesn't reach sides)
    // z_min: WALL + CONVECTIVE (substrate)
    // z_max: WALL + ADIABATIC (gas cap — laser enters volumetrically)
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

    // --- CFL limiting (adaptive, tuned for AM) ---
    config.cfl_use_adaptive            = true;
    config.cfl_v_target_interface      = 0.15f;
    config.cfl_v_target_bulk           = 0.10f;

    // --- VOF ---
    config.vof_subcycles               = 1;
    config.enable_vof_mass_correction  = false;  // PLIC is conservative

    // --- Timing ---
    const float t_total  = 500.0e-6f;   // 500 μs dwell time
    const int num_steps  = static_cast<int>(t_total / config.dt);
    const int vtk_every  = static_cast<int>(50.0e-6f / config.dt);   // VTK every 50 μs
    const int diag_every = 1000;  // Console diagnostics every 1000 steps

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
    printf("  T_solidus = %.0f K, T_liquidus = %.0f K, T_boil = %.0f K\n",
           config.material.T_solidus, config.material.T_liquidus,
           config.material.T_vaporization);
    printf("  σ = %.2f N/m, dσ/dT = %.1e N/(m·K)\n",
           config.surface_tension_coeff, config.dsigma_dT);
    printf("Laser: P = %.0f W, r₀ = %.0f μm, η = %.0f%%\n",
           config.laser_power, config.laser_spot_radius * 1e6f,
           config.laser_absorptivity * 100.0f);
    printf("Fluid: ν_LU = %.4f, τ = %.3f\n",
           config.kinematic_viscosity,
           1.0f / (3.0f * config.kinematic_viscosity + 0.5f));
    printf("Steps: %d (%.0f μs), VTK every %d steps (%.0f μs)\n\n",
           num_steps, t_total * 1e6f, vtk_every, vtk_every * config.dt * 1e6f);
    fflush(stdout);

    // ==================================================================
    // Create output directory
    // ==================================================================
    mkdir("output_spot_melt", 0755);

    // ==================================================================
    // Initialize solver
    // ==================================================================
    printf("Initializing MultiphysicsSolver...\n");
    fflush(stdout);

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.80f);  // T=300K, substrate fills bottom 80%

    printf("  Initialized: T=300K, interface at z=80%% (%.0f μm)\n",
           0.80f * config.nz * config.dx * 1e6f);

    // ==================================================================
    // Field registry for VTK output
    // ==================================================================
    const auto& registry = solver.getFieldRegistry();
    printf("  Registered fields:");
    for (const auto& name : registry.getFieldNames()) {
        printf(" %s", name.c_str());
    }
    printf("\n\n");
    fflush(stdout);

    // ==================================================================
    // Console header
    // ==================================================================
    printf("%-7s %8s %8s %8s %8s %8s %8s\n",
           "Step", "t [μs]", "T_max", "v_max", "Depth", "Width", "Mass");
    printf("%-7s %8s %8s %8s %8s %8s %8s\n",
           "", "", "[K]", "[m/s]", "[μm]", "[μm]", "[Σf]");
    printf("-------------------------------------------------------------\n");
    fflush(stdout);

    // ==================================================================
    // Time integration
    // ==================================================================
    for (int step = 0; step <= num_steps; ++step) {
        float t = step * config.dt;

        // --- Console diagnostics ---
        if (step % diag_every == 0) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();
            float depth = solver.getMeltPoolDepth();
            float mass  = solver.getTotalMass();

            // Estimate melt pool width from temperature field
            // (getMeltPoolDepth gives depth; width is typically 2-3× depth for Marangoni)
            float width = 2.0f * depth;  // Approximate; VTK gives exact shape

            printf("%-7d %8.1f %8.0f %8.3f %8.1f %8.1f %8.0f\n",
                   step, t * 1e6f, T_max, v_max,
                   depth * 1e6f, width * 1e6f, mass);
            fflush(stdout);

            // NaN check
            if (solver.checkNaN()) {
                printf("\n*** FATAL: NaN detected at step %d ***\n", step);
                break;
            }
        }

        // --- VTK output ---
        if (step % vtk_every == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "output_spot_melt/spot_melt_%06d", step);

            io::VTKWriter::writeFields(
                std::string(filename), registry, {},
                config.nx, config.ny, config.nz, config.dx);

            printf("  → VTK: %s.vtk (t=%.0f μs)\n", filename, t * 1e6f);
            fflush(stdout);
        }

        // --- Energy balance (every 100 μs) ---
        if (step > 0 && step % (vtk_every * 2) == 0) {
            solver.printEnergyBalance();
        }

        // --- Step ---
        if (step < num_steps) {
            solver.step();
        }
    }

    // ==================================================================
    // Final summary
    // ==================================================================
    printf("\n============================================================\n");
    printf("  Simulation Complete\n");
    printf("============================================================\n");
    printf("  Total steps: %d\n", num_steps);
    printf("  Total time:  %.0f μs\n", t_total * 1e6f);
    printf("  T_max:       %.0f K\n", solver.getMaxTemperature());
    printf("  v_max:       %.3f m/s\n", solver.getMaxVelocity());
    printf("  Melt depth:  %.1f μm\n", solver.getMeltPoolDepth() * 1e6f);
    printf("  Mass (Σf):   %.0f\n", solver.getTotalMass());
    printf("\nOutput: output_spot_melt/spot_melt_*.vtk\n");
    printf("ParaView instructions:\n");
    printf("  1. Open file series: spot_melt_*.vtk\n");
    printf("  2. Color by Temperature → melt pool shape\n");
    printf("  3. Contour: LiquidFraction = 0.5 → melt boundary\n");
    printf("  4. Glyph filter on Velocity → Marangoni vortices\n");
    printf("  5. StreamTracer → radial outflow pattern\n");
    printf("============================================================\n\n");

    return 0;
}
