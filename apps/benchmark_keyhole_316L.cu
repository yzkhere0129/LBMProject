/**
 * @file benchmark_keyhole_316L.cu
 * @brief Phase 5: Full LPBF keyhole simulation — moving laser, VOF, recoil, Marangoni
 *
 * All physics ON:
 *   - D3Q19 fluid (TRT+EDM+Smagorinsky LES)
 *   - FDM WENO5 thermal
 *   - VOF free surface (PLIC/TVD advection)
 *   - Recoil pressure (Hertz-Knudsen-Langmuir)
 *   - Evaporation cooling (T_boil=3200K)
 *   - Marangoni (CSF with 4× compensation)
 *   - Darcy mushy zone damping
 *   - Surface tension (CSF curvature)
 *   - Moving Gaussian laser (v_scan=800 mm/s)
 *
 * Domain: 300×150×75 μm (150×75×75 cells at dx=2μm, includes 25-cell gas buffer)
 * Run: 0–200 μs
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
    std::strncpy(mat.name, "316L_Keyhole", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700;
    mat.T_vaporization=3200.0f;  // real 316L boiling point
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=+1e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;
    mat.emissivity=0.3f;
    return mat;
}

int main(int argc, char** argv) {
    auto t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k=20, mu=0.005f;
    const float alpha=k/(rho*cp), nu=mu/rho;

    // Domain: 300×150μm surface, 100μm depth (50 metal + 25 gas)
    const int NX = 150, NY = 75, NZ_METAL = 50, NZ_GAS = 25;
    const int NZ = NZ_METAL + NZ_GAS;
    const float dx = 2.0e-6f;
    const float dt = 1.0e-8f;  // 10 ns — c_s=115 m/s, Ma=0.26 at 30 m/s
    const float t_total = 200.0e-6f;
    const float v_scan = 0.4f;  // 400 mm/s — slower scan for deeper pool + backfill

    printf("============================================================\n");
    printf("  Phase 5: LPBF Keyhole Simulation — 316L, Full Physics\n");
    printf("============================================================\n");
    printf("Domain: %d×%d×%d = %d cells (dx=%.0fμm)\n",
           NX, NY, NZ, NX*NY*NZ, dx*1e6f);
    printf("Metal: %d layers, Gas buffer: %d layers\n", NZ_METAL, NZ_GAS);
    printf("dt=%.0f ns, t_total=%.0f μs\n", dt*1e9f, t_total*1e6f);
    printf("Laser: P=150W, r0=25μm, v_scan=%.0f mm/s\n\n", v_scan*1e3f);

    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;

    // ALL physics ON
    config.enable_thermal           = true;
    config.enable_thermal_advection = true;
    config.use_fdm_thermal          = true;
    config.enable_phase_change      = true;
    config.enable_fluid             = true;
    config.enable_vof               = true;
    config.enable_vof_advection     = true;
    config.enable_laser             = true;
    config.enable_darcy             = true;
    config.enable_marangoni         = true;
    config.enable_surface_tension   = true;
    config.enable_buoyancy          = false;
    config.enable_evaporation_mass_loss = true;
    config.enable_recoil_pressure   = true;
    config.enable_radiation_bc      = false;

    // Fluid
    config.kinematic_viscosity = nu * dt / (dx*dx);
    config.density = rho;
    config.darcy_coefficient = 1.0e6f;

    // Thermal
    config.thermal_diffusivity = alpha;
    config.ambient_temperature = 300.0f;

    // Surface
    config.surface_tension_coeff = mat.surface_tension;
    config.dsigma_dT = mat.dsigma_dT;
    config.recoil_force_multiplier = 1.0f;
    config.recoil_max_pressure = 1e8f;  // Physical: no artificial cap, evaporation self-regulates
    config.marangoni_csf_multiplier = 1.0f;
    config.evap_cooling_factor = 1.0f;

    // Laser: moving source
    config.laser_power = 150.0f;
    config.laser_spot_radius = 25.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = dx;
    config.laser_shutoff_time = -1.0f;  // always on
    config.laser_start_x = 30.0e-6f;   // start 30μm from left
    config.laser_start_y = -1.0f;       // auto center
    config.laser_scan_vx = v_scan;
    config.laser_scan_vy = 0.0f;

    // Ray tracing OFF (use |∂f/∂z| projection instead)
    config.ray_tracing.enabled = false;

    // CFL
    config.cfl_velocity_target = 0.15f;
    config.cfl_use_gradual_scaling = true;
    config.cfl_force_ramp_factor = 0.9f;

    // VOF
    config.vof_subcycles = 1;
    config.enable_vof_mass_correction = false;  // MUST be OFF with evaporation — prevents ghost refill

    // Boundaries: all walls
    config.boundaries.x_min = config.boundaries.x_max = BoundaryType::WALL;
    config.boundaries.y_min = config.boundaries.y_max = BoundaryType::WALL;
    config.boundaries.z_min = config.boundaries.z_max = BoundaryType::WALL;
    config.boundaries.thermal_x_min = config.boundaries.thermal_x_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min = config.boundaries.thermal_y_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min = ThermalBCType::DIRICHLET;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature = 300.0f;

    // Initialize
    MultiphysicsSolver solver(config);
    solver.setSmagorinskyCs(0.20f);

    const int num_cells = NX*NY*NZ;
    std::vector<float> h_temp(num_cells, 300.0f);
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int iz=0; iz<NZ; iz++)
        for (int iy=0; iy<NY; iy++)
            for (int ix=0; ix<NX; ix++) {
                float f = (iz < NZ_METAL) ? 1.0f :
                          (iz == NZ_METAL) ? 0.5f : 0.0f;
                h_fill[ix + iy*NX + iz*NX*NY] = f;
            }
    solver.initialize(h_temp.data(), h_fill.data());

    // Output directory
    mkdir("output_keyhole", 0755);

    // Time loop
    int total_steps = static_cast<int>(t_total / dt + 0.5f);
    int vtk_interval = static_cast<int>(25.0e-6f / dt);  // VTK every 25μs
    int print_interval = std::max(1, total_steps / 40);

    printf("Total steps: %d, VTK every %d steps\n\n", total_steps, vtk_interval);
    printf("%-6s %7s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "T_max[K]", "v_max", "Depth", "laser_x");
    printf("--------------------------------------------------------------\n");

    // Time series output
    FILE* f_ts = fopen("output_keyhole/timeseries.csv", "w");
    fprintf(f_ts, "step,t_us,T_max,v_max_phys,depth_um,laser_x_um\n");

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        if (step > 0) solver.step();

        bool do_print = (step % print_interval == 0) || (step == total_steps);
        bool do_vtk = (step % vtk_interval == 0);

        if (do_print || do_vtk) {
            float T_max = solver.getMaxTemperature();
            float v_max_phys = solver.getMaxVelocity();
            float laser_x = (config.laser_start_x + v_scan * t) * 1e6f;

            // Melt pool depth from temperature
            std::vector<float> h_T(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            float max_depth = 0;
            for (int iz = NZ_METAL-1; iz >= 0; iz--) {
                for (int iy = 0; iy < NY; iy++)
                    for (int ix = 0; ix < NX; ix++) {
                        if (h_T[ix+iy*NX+iz*NX*NY] >= 1650.0f) {
                            float d = (NZ_METAL-1-iz)*dx*1e6f;
                            if (d > max_depth) max_depth = d;
                        }
                    }
            }

            if (do_print) {
                printf("%6d %7.1f %9.0f %9.1f %7.1fμm %7.1fμm\n",
                       step, t*1e6f, T_max, v_max_phys, max_depth, laser_x);
            }

            fprintf(f_ts, "%d,%.3f,%.1f,%.2f,%.1f,%.1f\n",
                    step, t*1e6f, T_max, v_max_phys, max_depth, laser_x);
        }

        if (do_vtk) {
            char fname[128];
            snprintf(fname, sizeof(fname), "output_keyhole/keyhole_%06d", step);
            const auto& registry = solver.getFieldRegistry();
            io::VTKWriter::writeFields(fname, registry, {},
                                       NX, NY, NZ, dx);
        }
    }

    fclose(f_ts);

    auto t1 = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t1 - t0).count();
    printf("\nWall time: %.1f s\n", elapsed);
    printf("VTK output: output_keyhole/\n");
    printf("Time series: output_keyhole/timeseries.csv\n");

    return 0;
}
