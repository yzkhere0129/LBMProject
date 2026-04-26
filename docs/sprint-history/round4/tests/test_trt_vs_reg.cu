/**
 * Round 4: TRT vs Regularized collision comparison
 *
 * Test 1: Controlled Marangoni (liquid slab + linear T + Marangoni only)
 *   TRT (τ=0.51 clamped) vs REG (τ=0.503 physical)
 *   Expected: REG should have 3.3× higher velocity
 *
 * Test 2: 150W full physics flat plate, 5000 steps
 *   TRT vs REG comparison of melt pool velocity
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm;
using namespace lbm::physics;

static MaterialProperties createMat316L() {
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

void testMarangoni(bool use_reg, const char* label, FILE* csv) {
    const float rho=7900, cp=700, k_c=20, mu=0.005f, nu=mu/rho;
    const float alpha=k_c/(rho*cp);
    const int NX=20, NY=80, NZ=20;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=10, num_cells=NX*NY*NZ;
    const float cs=dx/(dt*sqrtf(3.0f));

    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_ctrl", sizeof(mat.name)-1);
    mat.rho_solid=rho; mat.rho_liquid=rho;
    mat.cp_solid=cp; mat.cp_liquid=cp;
    mat.k_solid=k_c; mat.k_liquid=k_c;
    mat.mu_liquid=mu;
    mat.T_solidus=100; mat.T_liquidus=110;
    mat.T_vaporization=50000.0f;
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=-4.3e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;

    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;
    config.enable_thermal=true; config.enable_thermal_advection=false;
    config.use_fdm_thermal=true; config.enable_phase_change=false;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=false;
    config.enable_laser=false; config.enable_darcy=false;
    config.enable_marangoni=true; config.enable_surface_tension=false;
    config.enable_buoyancy=false; config.enable_evaporation_mass_loss=false;
    config.enable_recoil_pressure=false; config.enable_radiation_bc=false;

    config.kinematic_viscosity=nu*dt/(dx*dx);
    config.density=rho; config.thermal_diffusivity=alpha;
    config.ambient_temperature=300.0f; config.dsigma_dT=mat.dsigma_dT;
    config.marangoni_csf_multiplier=1.0f;
    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=0.30f; config.cfl_use_gradual_scaling=true;
    config.vof_subcycles=1; config.enable_vof_mass_correction=false;

    config.boundaries.x_min=config.boundaries.x_max=BoundaryType::WALL;
    config.boundaries.y_min=config.boundaries.y_max=BoundaryType::PERIODIC;
    config.boundaries.z_min=config.boundaries.z_max=BoundaryType::WALL;
    config.boundaries.thermal_x_min=config.boundaries.thermal_x_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min=config.boundaries.thermal_y_max=ThermalBCType::PERIODIC;
    config.boundaries.thermal_z_min=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature=300.0f;

    MultiphysicsSolver solver(config);
    solver.setSmagorinskyCs(0.0f);  // NO Smagorinsky

    if (use_reg) {
        // Physical τ = 3ν_lat + 0.5 = 3×0.001013 + 0.5 = 0.50304
        float tau_phys = 3.0f * nu * dt / (dx*dx) + 0.5f;
        solver.setRegularized(true, tau_phys);
    }

    std::vector<float> h_fill(num_cells, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f) - (float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f-tanhf(2.0f*dist));
            }

    float T_cold=1000, T_hot=2000;
    std::vector<float> h_temp(num_cells, 300.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                if (h_fill[ii+jj*NX+kk*NX*NY] > 0.01f) {
                    float y = (jj+0.5f)*dx;
                    h_temp[ii+jj*NX+kk*NX*NY] = T_cold + (T_hot-T_cold)*y/(NY*dx);
                }
            }

    solver.initialize(h_temp.data(), h_fill.data());

    float dTdy = (T_hot-T_cold)/(NY*dx);
    float tau_s = fabsf(mat.dsigma_dT) * dTdy;
    float h = z_surface * dx;

    float tau_actual = use_reg ? (3.0f*nu*dt/(dx*dx)+0.5f) : 0.51f;
    float nu_actual = (tau_actual - 0.5f)/3.0f * dx*dx/dt;
    float mu_actual = rho * nu_actual;
    float u_analytical = tau_s * h / (2.0f * mu_actual);

    printf("\n  [%s] τ=%.5f, ν_eff=%.4e, μ_eff=%.5f, u_analytical=%.3f m/s\n",
           label, tau_actual, nu_actual, mu_actual, u_analytical);

    for (int step=0; step<=10000; step++) {
        if (step>0) solver.step();
        if (step % 2000 == 0) {
            std::vector<float> h_vy(num_cells), h_f(num_cells);
            solver.copyVelocityToHost(nullptr, h_vy.data(), nullptr);
            solver.copyFillLevelToHost(h_f.data());

            // Workaround: copyVelocityToHost needs all 3 arrays
            std::vector<float> h_vx(num_cells), h_vz(num_cells);
            solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

            float v_max=0;
            for (int i=0; i<num_cells; i++) {
                if (h_f[i]>0.01f) {
                    float v=fabsf(h_vy[i])*dx/dt;
                    if (v>v_max) v_max=v;
                }
            }
            float ratio = (u_analytical>0) ? v_max/u_analytical : 0;
            printf("  [%s] step=%5d v_max=%.4f m/s ratio=%.4f\n", label, step, v_max, ratio);
            fprintf(csv, "%s,%d,%.6f,%.6f,%.6f\n", label, step, v_max, u_analytical, ratio);
        }
    }
}

void test150W(bool use_reg, const char* label, FILE* csv) {
    MaterialProperties mat = createMat316L();
    const float rho=7900, cp=700, k_c=20, mu=0.005f, nu=mu/rho;
    const float alpha=k_c/(rho*cp);
    const int NX=40, NY=80, NZ=40;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=20, num_cells=NX*NY*NZ;
    const float cs=dx/(dt*sqrtf(3.0f));

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
    config.laser_power=150.0f; config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f; config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=50.0e-6f; config.laser_start_y=50.0e-6f;
    config.laser_scan_vx=0.0f; config.laser_scan_vy=1.0f;
    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=0.30f; config.cfl_use_gradual_scaling=true;
    config.vof_subcycles=1; config.enable_vof_mass_correction=false;

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

    if (use_reg) {
        float tau_phys = 3.0f*nu*dt/(dx*dx)+0.5f;
        solver.setRegularized(true, tau_phys);
    }

    std::vector<float> h_fill(num_cells, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f)-(float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f-tanhf(2.0f*dist));
            }
    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    printf("\n  [%s] 150W flat plate, 5000 steps\n", label);

    for (int step=0; step<=5000; step++) {
        if (step>0) solver.step();
        if (step % 1000 == 0) {
            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());
            std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
            solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

            float v_metal=0, T_max=0;
            for (int i=0; i<num_cells; i++) {
                if (h_f[i]>0.01f) {
                    float v=sqrtf(h_vx[i]*h_vx[i]+h_vy[i]*h_vy[i]+h_vz[i]*h_vz[i])*dx/dt;
                    if (v>v_metal) v_metal=v;
                }
                if (h_T[i]>T_max) T_max=h_T[i];
            }
            float Ma=v_metal/cs;
            printf("  [%s] step=%5d T=%.0f v_metal=%.2f Ma=%.4f\n",
                   label, step, T_max, v_metal, Ma);
            fprintf(csv, "%s,%d,%.1f,%.3f,%.5f\n", label, step, T_max, v_metal, Ma);

            if (std::isnan(T_max)) { printf("  [%s] *** NaN DIVERGED ***\n", label); break; }
        }
    }
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();

    FILE* csv_maran = fopen("round4/data/trt_vs_reg_marangoni.csv", "w");
    fprintf(csv_maran, "mode,step,v_max,u_analytical,ratio\n");

    printf("================================================================\n");
    printf("  Test 1: Controlled Marangoni — TRT vs Regularized\n");
    printf("================================================================\n");
    testMarangoni(false, "TRT", csv_maran);
    testMarangoni(true, "REG", csv_maran);
    fclose(csv_maran);

    FILE* csv_150w = fopen("round4/data/trt_vs_reg_150w.csv", "w");
    fprintf(csv_150w, "mode,step,T_max,v_metal,Ma\n");

    printf("\n================================================================\n");
    printf("  Test 2: 150W Flat Plate — TRT vs Regularized\n");
    printf("================================================================\n");
    test150W(false, "TRT", csv_150w);
    test150W(true, "REG", csv_150w);
    fclose(csv_150w);

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\n================================================================\n");
    printf("  Total time: %.1f s\n", std::chrono::duration<float>(t1-t0).count());
    printf("================================================================\n");
    return 0;
}
