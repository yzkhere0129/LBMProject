/**
 * @file diag_flat_plate_scan.cu
 * @brief Flat plate baseline: no powder, pure substrate + laser scan
 *
 * Domain: 80×160×80 (200×400×200 μm), dx=2.5μm
 * Flat metal surface at z=40 (100μm). f=1 for z<40, f=0 for z>=40.
 * Laser: 150W, r=25μm, scan +Y at 1000mm/s from y=100μm.
 * Same physics as production benchmark.
 *
 * Purpose: isolate melt pool behavior on clean flat geometry,
 * without any powder artifacts.
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"
#include "io/field_registry.h"

using namespace lbm;
using namespace lbm::physics;

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_FlatPlate", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700;
    // R6 final alignment to OpenFOAM laserMeltFoam:
    //   T_vap = 3090 K (was 3200K, OpenFOAM boiling-pressure reference)
    //   molar_mass = 0.05593 kg/mol (was 0.0558, weighted avg pure-element value)
    mat.T_vaporization=3090.0f;
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.05593f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=-4.3e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;
    mat.emissivity=0.3f;
    return mat;
}

int main(int argc, char** argv) {
    auto t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;

    const int NX=80, NY=160, NZ=80;
    const float dx=2.5e-6f, dt=1.0e-8f;
    float t_total = 200.0e-6f;
    if (argc >= 2) t_total = atof(argv[1]) * 1e-6f;
    const int z_surface = 40;  // flat surface at z=40 (100μm)
    const int num_cells = NX*NY*NZ;

    // Ablation mode: argv[2] = "M" (Marangoni only), "R" (Recoil only),
    //                           "MR" (both, default), "NONE" (neither)
    const char* abl_mode = (argc >= 3) ? argv[2] : "MR";
    bool enable_marangoni_flag = (std::strcmp(abl_mode, "M") == 0) ||
                                  (std::strcmp(abl_mode, "MR") == 0);
    bool enable_recoil_flag    = (std::strcmp(abl_mode, "R") == 0) ||
                                  (std::strcmp(abl_mode, "MR") == 0);
    printf("Ablation mode: %s  (Marangoni=%s, Recoil=%s)\n",
           abl_mode,
           enable_marangoni_flag ? "ON" : "OFF",
           enable_recoil_flag    ? "ON" : "OFF");

    printf("============================================================\n");
    printf("  Flat Plate Baseline: 316L (no powder)\n");
    printf("============================================================\n");
    printf("Domain: %d×%d×%d (dx=%.1fμm)\n", NX, NY, NZ, dx*1e6f);
    printf("Flat surface at z=%d (%.0fμm)\n", z_surface, z_surface*dx*1e6f);
    printf("Laser: 150W, r=25μm, 1000mm/s along +Y\n");
    printf("RLBM: τ=0.503, mass_loss=1.0, boiling_cap=ON\n\n");

    // Config: identical to production benchmark
    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;

    config.enable_thermal=true; config.enable_thermal_advection=true;
    config.use_fdm_thermal=true; config.enable_phase_change=true;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=true;
    config.enable_laser=true; config.enable_darcy=true;
    config.enable_marangoni=enable_marangoni_flag; config.enable_surface_tension=true;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=true;
    config.enable_recoil_pressure=enable_recoil_flag;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity = nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=1.0f;
    config.recoil_max_pressure=1e8f;
    // R6 Fix 4: repurpose recoil_smoothing_width as temperature ramp [K].
    // Smooth activation from (T_boil - width) to T_boil prevents a single
    // overheated cell from producing full recoil ×3-10 atmospheric pressure.
    // 200K window matches the HKL evaporation-cooling activation scale.
    config.recoil_smoothing_width=200.0f;
    config.marangoni_csf_multiplier=1.0f;
    config.evap_cooling_factor=1.0f;

    config.laser_power=150.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=100.0e-6f;
    config.laser_start_y=100.0e-6f;
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=1.0f;  // 1000 mm/s

    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=0.15f;
    config.cfl_use_gradual_scaling=true;
    config.vof_subcycles=1;
    config.enable_vof_mass_correction=false;

    config.boundaries.x_min=config.boundaries.x_max=BoundaryType::WALL;
    config.boundaries.y_min=config.boundaries.y_max=BoundaryType::WALL;
    config.boundaries.z_min=config.boundaries.z_max=BoundaryType::WALL;
    config.boundaries.thermal_x_min=config.boundaries.thermal_x_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min=config.boundaries.thermal_y_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min=ThermalBCType::DIRICHLET;
    config.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature=300.0f;

    MultiphysicsSolver solver(config);
    solver.setSmagorinskyCs(0.20f);
    solver.setRegularized(true, 0.503f);  // RLBM: stable at τ=0.503
    // R6 Audit 1: Guo-2002 forcing scheme with (1-ω/2) prefactor.
    // At τ=0.51 (ω=1.96), Guo's source factor ≈ 0.02, drastically reducing
    // force injection vs EDM's unconditional equilibrium shift.
    solver.setUseGuoForcing(true);

    // Flat plate: f=1 below surface, smooth interface at z=z_surface
    // The laser kernel requires 0.05 < f < 0.99 to deposit energy,
    // so we need at least one transition cell at the interface.
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++) {
                float z_cell = (k + 0.5f);
                float dist = z_cell - (float)z_surface;
                // Smooth interface: tanh profile over ~1 cell width
                float f = 0.5f * (1.0f - tanhf(2.0f * dist));
                h_fill[i + j*NX + k*NX*NY] = f;
            }

    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    float initial_fill = 0;
    for (auto f : h_fill) initial_fill += f;
    printf("Initial: SumF=%.0f (flat plate, %d solid layers)\n\n", initial_fill, z_surface);

    mkdir("output_flat_plate", 0755);

    int total_steps = (int)(t_total/dt + 0.5f);
    int vtk_interval = (int)(25.0e-6f/dt);
    int print_interval = std::max(1, total_steps/40);

    printf("%-6s %7s %9s %9s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "T_max[K]", "v_max", "Depth", "LiqCells", "SumF", "laser_y");
    printf("--------------------------------------------------------------------------\n");

    FILE* f_ts = fopen("output_flat_plate/timeseries.csv", "w");
    fprintf(f_ts, "step,t_us,T_max,v_max,depth_um,liquid_cells,sum_fill,laser_y_um\n");

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        if (step > 0) solver.step();

        bool do_print = (step % print_interval == 0) || (step == total_steps);
        bool do_vtk = (step % vtk_interval == 0);

        if (do_print || do_vtk) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();
            float laser_y = (100.0e-6f + 1.0f * t) * 1e6f;

            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            float max_depth = 0;
            int liquid_cells = 0;
            float sum_fill = 0;
            for (int kk = 0; kk < NZ; kk++)
                for (int jj = 0; jj < NY; jj++)
                    for (int ii = 0; ii < NX; ii++) {
                        int idx = ii + jj*NX + kk*NX*NY;
                        sum_fill += h_f[idx];
                        if (h_T[idx] >= 1650.0f && h_f[idx] > 0.01f) {
                            liquid_cells++;
                            if (kk < z_surface) {
                                float d = (z_surface-1-kk)*dx*1e6f;
                                if (d > max_depth) max_depth = d;
                            }
                        }
                    }

            if (do_print)
                printf("%6d %7.1f %9.0f %9.1f %7.1fμm %9d %9.0f %7.1fμm\n",
                       step, t*1e6f, T_max, v_max, max_depth,
                       liquid_cells, sum_fill, laser_y);

            fprintf(f_ts, "%d,%.3f,%.1f,%.2f,%.1f,%d,%.1f,%.1f\n",
                    step, t*1e6f, T_max, v_max, max_depth,
                    liquid_cells, sum_fill, laser_y);

            if (do_vtk) {
                char fname[128];
                snprintf(fname, sizeof(fname), "output_flat_plate/flat_%06d", step);
                const auto& reg = solver.getFieldRegistry();
                io::VTKWriter::writeFields(fname, reg, {}, NX, NY, NZ, dx);
            }
        }
    }
    fclose(f_ts);

    printf("\nWall time: %.1f s\n", std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count());
    return 0;
}
