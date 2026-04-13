/**
 * Task 2: CFL OFF vs ON comparison at 150W
 *
 * Run 1: CFL gradual scaling OFF (set cfl_use_gradual_scaling=false)
 *         BUT keep the hard cap at 0.38 LU (safety net)
 * Run 2: CFL ON with production settings (v_target=0.15)
 *
 * Both: 5000 steps, 150W, all physics, flat plate 40×80×40
 * Measure: v_metal (f>0.01), v_gas (f<=0.01), T_max, T_iface
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm;
using namespace lbm::physics;

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L", sizeof(mat.name)-1);
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

struct RunResult {
    float v_metal, v_gas, v_iface;
    float T_max, T_iface;
};

RunResult runTest(const char* label, bool cfl_gradual, float cfl_target, const char* csv_path) {
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;
    const int NX=40, NY=80, NZ=40;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=20, num_cells=NX*NY*NZ;
    const float cs=dx/(dt*sqrtf(3.0f));

    printf("\n================================================================\n");
    printf("  %s\n", label);
    printf("  CFL gradual=%s, v_target=%.2f LU (%.1f m/s)\n",
           cfl_gradual?"ON":"OFF", cfl_target, cfl_target*dx/dt);
    printf("================================================================\n");

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

    config.kinematic_viscosity=nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;
    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=1.0f; config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=1.0f; config.evap_cooling_factor=1.0f;

    config.laser_power=150.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=50.0e-6f; config.laser_start_y=50.0e-6f;
    config.laser_scan_vx=0.0f; config.laser_scan_vy=1.0f;

    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=cfl_target;
    config.cfl_use_gradual_scaling=cfl_gradual;
    config.cfl_use_adaptive=false;
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

    std::vector<float> h_fill(num_cells, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f) - (float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f-tanhf(2.0f*dist));
            }
    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    FILE* csv = fopen(csv_path, "w");
    fprintf(csv, "step,t_us,T_max,T_iface,v_metal,v_gas,v_iface,Ma_metal\n");

    printf("  %6s %7s %8s %8s %9s %9s %9s %8s\n",
           "Step", "t[μs]", "T_max", "T_iface", "v_metal", "v_gas", "v_iface", "Ma_met");

    RunResult last = {};
    for (int step=0; step<=5000; step++) {
        if (step>0) solver.step();

        if (step % 500 == 0) {
            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());
            std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
            solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

            float v_metal=0, v_gas=0, v_iface=0, T_iface=0;
            float T_max = solver.getMaxTemperature();

            for (int idx=0; idx<num_cells; idx++) {
                float f=h_f[idx];
                float vmag = sqrtf(h_vx[idx]*h_vx[idx]+h_vy[idx]*h_vy[idx]+h_vz[idx]*h_vz[idx])*dx/dt;

                if (f <= 0.01f) {
                    if (vmag > v_gas) v_gas = vmag;
                } else {
                    if (vmag > v_metal) v_metal = vmag;
                    if (f < 0.99f) {
                        if (vmag > v_iface) v_iface = vmag;
                        if (h_T[idx] > T_iface) T_iface = h_T[idx];
                    }
                }
            }

            printf("  %6d %7.1f %8.0f %8.0f %9.2f %9.2f %9.2f %8.4f\n",
                   step, step*dt*1e6, T_max, T_iface, v_metal, v_gas, v_iface, v_metal/cs);
            fprintf(csv, "%d,%.3f,%.1f,%.1f,%.3f,%.3f,%.3f,%.5f\n",
                    step, step*dt*1e6, T_max, T_iface, v_metal, v_gas, v_iface, v_metal/cs);

            last = {v_metal, v_gas, v_iface, T_max, T_iface};

            if (std::isnan(T_max) || T_max > 50000) {
                printf("  *** DIVERGED ***\n");
                break;
            }
        }
    }
    fclose(csv);
    return last;
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Run 1: CFL relaxed (v_target=0.30 LU = 75 m/s, 2× production)
    auto r1 = runTest("Run 1: CFL RELAXED (target=0.30 LU = 75 m/s)",
                       true, 0.30f, "round3_audit/data/cfl_relaxed_timeseries.csv");

    // Run 2: CFL ON (production)
    auto r2 = runTest("Run 2: CFL ON (gradual=true, target=0.15 LU)",
                       true, 0.15f, "round3_audit/data/cfl_on_timeseries.csv");

    printf("\n================================================================\n");
    printf("  COMPARISON\n");
    printf("================================================================\n");
    printf("  %-20s %12s %12s\n", "Metric", "CFL OFF", "CFL ON");
    printf("  %-20s %12.2f %12.2f\n", "v_metal [m/s]", r1.v_metal, r2.v_metal);
    printf("  %-20s %12.2f %12.2f\n", "v_gas [m/s]", r1.v_gas, r2.v_gas);
    printf("  %-20s %12.2f %12.2f\n", "v_iface [m/s]", r1.v_iface, r2.v_iface);
    printf("  %-20s %12.0f %12.0f\n", "T_max [K]", r1.T_max, r2.T_max);
    printf("  %-20s %12.0f %12.0f\n", "T_iface [K]", r1.T_iface, r2.T_iface);
    printf("  %-20s %12.4f %12.4f\n", "Ma_metal", r1.v_metal/144.3f, r2.v_metal/144.3f);

    float v_ratio = (r2.v_metal > 0.001f) ? r1.v_metal / r2.v_metal : 0;
    printf("\n  v_metal ratio (OFF/ON): %.3f\n", v_ratio);
    if (fabsf(v_ratio - 1.0f) < 0.05f) {
        printf("  → CFL limiter has NO effect on metal velocity (ratio ≈ 1.0)\n");
    } else if (v_ratio > 1.5f) {
        printf("  → CFL limiter WAS suppressing metal! OFF is %.1f× faster\n", v_ratio);
    } else {
        printf("  → Small difference (%.0f%%), unclear cause\n", (v_ratio-1)*100);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\n  Total time: %.1f s\n", std::chrono::duration<float>(t1-t0).count());
    return 0;
}
