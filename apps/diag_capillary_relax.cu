/**
 * @file diag_capillary_relax.cu
 * @brief Test 2: Isothermal Capillary Relaxation
 *
 * Phase 1 (0-100μs): Normal flat plate scan (150W, 1000mm/s) — creates a pit.
 * Phase 2 (100-200μs): Laser OFF, recoil OFF, temperature locked at 2500K
 *   (far above liquidus=1700K → pure liquid, low viscosity).
 *   Observe whether surface tension fills the pit.
 *
 * Diagnostic: if pit fills → solidification froze the flow too early.
 *             if pit persists → surface tension drive is too weak.
 */

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

// Simple kernel to force all metal cells to a fixed temperature
__global__ void forceTemperatureKernel(
    float* __restrict__ T,
    const float* __restrict__ fill_level,
    float T_force, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    if (fill_level != nullptr && fill_level[idx] > 0.01f) {
        T[idx] = T_force;
    }
}

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_Capillary", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700;
    mat.T_vaporization=3200.0f;
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=-4.3e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;
    mat.emissivity=0.3f;
    return mat;
}

int main() {
    auto t0_wall = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;

    const int NX=80, NY=160, NZ=80;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const float t_phase1 = 100.0e-6f;  // Laser ON phase
    const float t_phase2 = 100.0e-6f;  // Isothermal relaxation
    const float t_total  = t_phase1 + t_phase2;
    const float T_lock   = 2500.0f;    // Lock temperature (liquid, low viscosity)
    const int z_surface = 40;
    const int num_cells = NX*NY*NZ;

    printf("============================================================\n");
    printf("  Test 2: Isothermal Capillary Relaxation\n");
    printf("============================================================\n");
    printf("Phase 1: 0-%.0fμs — normal scan (150W, 1000mm/s)\n", t_phase1*1e6f);
    printf("Phase 2: %.0f-%.0fμs — T locked at %.0fK, no laser/recoil\n",
           t_phase1*1e6f, t_total*1e6f, T_lock);
    printf("Purpose: test if surface tension can fill the pit\n\n");

    // Phase 1 config: identical to flat plate scan
    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;

    config.enable_thermal=true; config.enable_thermal_advection=true;
    config.use_fdm_thermal=true; config.enable_phase_change=true;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=true;
    config.enable_laser=true; config.enable_darcy=true;
    config.enable_marangoni=true; config.enable_surface_tension=true;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=true;
    config.enable_recoil_pressure=true;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity = nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=1.0f;
    config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=1.0f;
    config.evap_cooling_factor=1.0f;

    config.laser_power=150.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time = t_phase1;  // Laser OFF at 100μs
    config.laser_start_x=100.0e-6f;
    config.laser_start_y=100.0e-6f;
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=1.0f;

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

    // Flat plate with smooth interface
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int k=0; k<NZ; k++)
        for (int j=0; j<NY; j++)
            for (int i=0; i<NX; i++) {
                float dist = (k + 0.5f) - (float)z_surface;
                h_fill[i+j*NX+k*NX*NY] = 0.5f*(1.0f - tanhf(2.0f*dist));
            }

    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    mkdir("output_capillary", 0755);

    int total_steps = (int)(t_total/dt + 0.5f);
    int phase1_steps = (int)(t_phase1/dt + 0.5f);
    int vtk_interval = (int)(10.0e-6f/dt);  // VTK every 10μs
    int print_interval = std::max(1, total_steps/40);

    printf("%-6s %7s %5s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "Phase", "T_max[K]", "v_max", "LiqCells", "SumF");
    printf("--------------------------------------------------------------\n");

    FILE* f_ts = fopen("output_capillary/timeseries.csv", "w");
    fprintf(f_ts, "step,t_us,phase,T_max,v_max,liquid_cells,sum_fill,z_min_pit_um\n");

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        bool is_phase2 = (step > phase1_steps);

        if (step > 0) {
            solver.step();

            // In phase 2: force temperature to T_lock on all metal cells AFTER step
            // Directly overwrite the thermal solver's device temperature array
            if (is_phase2) {
                std::vector<float> h_T(num_cells), h_f(num_cells);
                solver.copyTemperatureToHost(h_T.data());
                solver.copyFillLevelToHost(h_f.data());
                for (int idx=0; idx<num_cells; idx++) {
                    if (h_f[idx] > 0.01f) h_T[idx] = T_lock;
                    else h_T[idx] = 300.0f;
                }
                // Direct write to thermal solver's active buffer
                float* d_T = const_cast<float*>(solver.getTemperature());
                cudaMemcpy(d_T, h_T.data(), num_cells*sizeof(float),
                           cudaMemcpyHostToDevice);
            }
        }

        bool do_print = (step % print_interval == 0) || (step == total_steps)
                        || (step == phase1_steps) || (step == phase1_steps+1);
        bool do_vtk = (step % vtk_interval == 0);

        if (do_print || do_vtk) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity() * dx / dt;

            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            int liquid_cells = 0;
            float sum_fill = 0;
            // Find z_min of pit (lowest z where f < 0.5 in the center column)
            float z_min_pit = z_surface * dx * 1e6f;
            int x_c = NX/2;
            for (int kk = z_surface; kk >= 0; kk--) {
                // Average over small patch around laser center
                float f_avg = 0; int cnt = 0;
                for (int jj = 38; jj <= 42; jj++)
                    for (int ii = x_c-2; ii <= x_c+2; ii++) {
                        f_avg += h_f[ii+jj*NX+kk*NX*NY]; cnt++;
                    }
                f_avg /= cnt;
                if (f_avg < 0.5f) {
                    z_min_pit = (kk+0.5f)*dx*1e6f;
                }
            }

            for (int idx=0; idx<num_cells; idx++) {
                sum_fill += h_f[idx];
                if (h_T[idx] >= 1650.0f && h_f[idx] > 0.01f) liquid_cells++;
            }

            if (do_print)
                printf("%6d %7.1f %5s %9.0f %9.1f %9d %9.0f  pit_z=%.1fμm\n",
                       step, t*1e6f, is_phase2?"RELAX":"SCAN",
                       T_max, v_max, liquid_cells, sum_fill, z_min_pit);

            fprintf(f_ts, "%d,%.3f,%s,%.1f,%.2f,%d,%.1f,%.2f\n",
                    step, t*1e6f, is_phase2?"relax":"scan",
                    T_max, v_max, liquid_cells, sum_fill, z_min_pit);

            if (do_vtk) {
                char fname[128];
                snprintf(fname, sizeof(fname), "output_capillary/cap_%06d", step);
                const auto& reg = solver.getFieldRegistry();
                io::VTKWriter::writeFields(fname, reg, {}, NX, NY, NZ, dx);
            }
        }
    }
    fclose(f_ts);

    printf("\nWall time: %.1f s\n", std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0_wall).count());
    return 0;
}
