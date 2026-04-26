/**
 * Melt pool state diagnostic: WHY is v_metal only 1.6 m/s?
 *
 * At 50μs into 150W scan: dump the melt pool state:
 * - Distribution of liquid fraction fl
 * - Distribution of velocity in liquid cells (fl > 0.5)
 * - Force magnitude at interface cells
 * - Darcy damping factor at each liquid fraction level
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
    printf("  Melt Pool State Diagnostic: 150W, 50μs\n");
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
    config.enable_evaporation_mass_loss=false;
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

    // Run 5000 steps (50μs)
    for (int step=0; step<5000; step++) solver.step();

    // Dump state
    std::vector<float> h_T(num_cells), h_f(num_cells);
    solver.copyTemperatureToHost(h_T.data());
    solver.copyFillLevelToHost(h_f.data());
    std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
    solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

    // Get liquid fraction from phase change solver
    std::vector<float> h_fl(num_cells, 0.0f);
    const float* d_fl = solver.getTemperature();  // wrong — need liquid fraction
    // Actually: use T to infer fl. For T < T_sol: fl=0, T > T_liq: fl=1
    for (int i=0; i<num_cells; i++) {
        if (h_f[i] < 0.01f) { h_fl[i] = 0; continue; }
        float T = h_T[i];
        if (T < 1650) h_fl[i] = 0;
        else if (T > 1700) h_fl[i] = 1.0f;
        else h_fl[i] = (T - 1650) / 50.0f;
    }

    // Statistics
    printf("\n--- Cell Census ---\n");
    int n_gas=0, n_solid=0, n_mushy=0, n_liquid=0, n_iface=0;
    for (int i=0; i<num_cells; i++) {
        float f = h_f[i], fl = h_fl[i];
        if (f < 0.01f) n_gas++;
        else if (fl < 0.01f) n_solid++;
        else if (fl < 0.99f) n_mushy++;
        else n_liquid++;
        if (f > 0.01f && f < 0.99f) n_iface++;
    }
    printf("Gas: %d, Solid: %d, Mushy: %d, Liquid: %d, Interface: %d\n",
           n_gas, n_solid, n_mushy, n_liquid, n_iface);

    // Velocity distribution in liquid cells
    printf("\n--- Velocity in Liquid Cells (fl > 0.5) ---\n");
    int v_bins[10] = {};  // [0-0.1, 0.1-0.5, 0.5-1, 1-2, 2-5, 5-10, 10-50, 50-100, 100-500, 500+] m/s
    float v_max_liquid = 0;
    int n_liq_cells = 0;
    for (int i=0; i<num_cells; i++) {
        if (h_f[i] < 0.01f || h_fl[i] < 0.5f) continue;
        n_liq_cells++;
        float v = sqrtf(h_vx[i]*h_vx[i]+h_vy[i]*h_vy[i]+h_vz[i]*h_vz[i]) * dx/dt;
        if (v > v_max_liquid) v_max_liquid = v;
        if (v < 0.1) v_bins[0]++;
        else if (v < 0.5) v_bins[1]++;
        else if (v < 1) v_bins[2]++;
        else if (v < 2) v_bins[3]++;
        else if (v < 5) v_bins[4]++;
        else if (v < 10) v_bins[5]++;
        else if (v < 50) v_bins[6]++;
        else if (v < 100) v_bins[7]++;
        else if (v < 500) v_bins[8]++;
        else v_bins[9]++;
    }
    printf("Total liquid cells (fl>0.5): %d\n", n_liq_cells);
    printf("v_max in liquid: %.2f m/s (Ma=%.4f)\n", v_max_liquid, v_max_liquid/cs);
    printf("Distribution:\n");
    const char* labels[] = {"<0.1","0.1-0.5","0.5-1","1-2","2-5","5-10","10-50","50-100","100-500",">500"};
    for (int b=0; b<10; b++) {
        if (v_bins[b] > 0)
            printf("  %10s m/s: %6d cells (%.1f%%)\n",
                   labels[b], v_bins[b], 100.0f*v_bins[b]/n_liq_cells);
    }

    // Darcy distribution
    printf("\n--- Darcy Damping in Metal Cells ---\n");
    printf("  Darcy_coeff = 1e6, K_LU = D × dt\n");
    float K_darcy = 1e6f;
    for (float fl : {0.01f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f}) {
        float eps = 1e-3f;
        float D = K_darcy * (1-fl)*(1-fl) / (fl*fl*fl + eps);
        float K_LU = D * dt;
        float supp = 1.0f / (1.0f + K_LU / 2.0f);
        printf("  fl=%.2f: D=%.2e, K_LU=%.4f, suppression=%.4f\n", fl, D, K_LU, supp);
    }

    // Temperature profile near melt pool center
    printf("\n--- Melt Pool Profile (x=center, y=center) ---\n");
    int ic=NX/2, jc=NY/2;
    printf("  z    z[μm]    T[K]      f      fl     v[m/s]\n");
    for (int kk=NZ-1; kk>=0; kk--) {
        int idx = ic + jc*NX + kk*NX*NY;
        float T = h_T[idx], f = h_f[idx], fl = h_fl[idx];
        float v = sqrtf(h_vx[idx]*h_vx[idx]+h_vy[idx]*h_vy[idx]+h_vz[idx]*h_vz[idx])*dx/dt;
        if (T > 310 || f > 0.01f || kk > z_surface-3)
            printf("  %2d   %5.1f   %6.0f   %5.3f  %5.3f  %6.2f\n",
                   kk, (kk+0.5f)*dx*1e6, T, f, fl, v);
    }

    return 0;
}
