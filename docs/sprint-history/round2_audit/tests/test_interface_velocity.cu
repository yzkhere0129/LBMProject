/**
 * Task 1 supplement: Measure v_max at INTERFACE cells only
 *
 * Key hypothesis: v_max=395 m/s might be in a bulk liquid cell deep in
 * the melt pool. The INTERFACE velocity (where Marangoni acts, where
 * VOF advection happens) may be much lower.
 *
 * This test runs 50W + evap cooling for 5000 steps, then extracts:
 * - v_max_global (all cells)
 * - v_max_interface (0.01 < f < 0.99)
 * - v_max_liquid_surface (f ∈ [0.3, 0.7] — peak interface cells)
 * - T_max_interface (temperature at interface only)
 * - Average Ma at interface
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

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

int main() {
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;
    const int NX=40, NY=80, NZ=40;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=20, num_cells=NX*NY*NZ;
    const float cs=dx/(dt*sqrtf(3.0f));

    printf("================================================================\n");
    printf("  Interface Velocity Analysis\n");
    printf("  50W, Marangoni + evapCool, 5000 steps\n");
    printf("================================================================\n");

    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;
    config.enable_thermal=true; config.enable_thermal_advection=true;
    config.use_fdm_thermal=true; config.enable_phase_change=true;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=true;
    config.enable_laser=true; config.enable_darcy=true;
    config.enable_marangoni=true; config.enable_surface_tension=false;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=false;  // Path B evap cooling
    config.enable_recoil_pressure=false;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity=nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;
    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=0; config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=1.0f; config.evap_cooling_factor=1.0f;

    config.laser_power=50.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=50.0e-6f; config.laser_start_y=50.0e-6f;
    config.laser_scan_vx=0.0f; config.laser_scan_vy=1.0f;

    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=0.30f;
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

    std::vector<float> h_fill(num_cells, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f) - (float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f-tanhf(2.0f*dist));
            }
    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    printf("  %6s %7s %9s %9s %9s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "T_max", "T_iface", "v_global", "v_iface", "v_peak", "Ma_glob", "Ma_iface");

    for (int step=0; step<=5000; step++) {
        if (step>0) solver.step();

        if (step % 1000 == 0) {
            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            // Get velocity (lattice units)
            std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
            solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

            float v_global=0, v_iface=0, v_peak=0;
            float T_max_global=0, T_max_iface=0;
            int n_iface=0, n_peak=0;

            for (int idx=0; idx<num_cells; idx++) {
                float f = h_f[idx];
                float T = h_T[idx];
                float vmag = sqrtf(h_vx[idx]*h_vx[idx]+h_vy[idx]*h_vy[idx]+h_vz[idx]*h_vz[idx]);
                float v_phys = vmag * dx / dt;

                if (v_phys > v_global) v_global = v_phys;
                if (T > T_max_global) T_max_global = T;

                if (f > 0.01f && f < 0.99f) {
                    if (v_phys > v_iface) v_iface = v_phys;
                    if (T > T_max_iface) T_max_iface = T;
                    n_iface++;
                }
                if (f > 0.3f && f < 0.7f) {
                    if (v_phys > v_peak) v_peak = v_phys;
                    n_peak++;
                }
            }

            printf("  %6d %7.1f %9.0f %9.0f %9.1f %9.1f %9.1f %9.3f %9.3f\n",
                   step, step*dt*1e6, T_max_global, T_max_iface,
                   v_global, v_iface, v_peak,
                   v_global/cs, v_iface/cs);
        }
    }

    printf("\n  n_interface=%d (0.01<f<0.99), n_peak=%d (0.3<f<0.7)\n", 0, 0);
    printf("  cs = %.1f m/s\n", cs);
    return 0;
}
