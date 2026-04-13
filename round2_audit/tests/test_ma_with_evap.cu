/**
 * Round 2 Task 1: Ma measurement with evaporation cooling active.
 *
 * Tests R2-4.1 through R2-4.9: varying power and physics with evap cooling ON.
 * Evap cooling ON = enable_evaporation_mass_loss=false triggers Path B (line 1811).
 * For R2-4.8/4.9: enable_evaporation_mass_loss=true uses Path A (line 1504).
 *
 * Each test: 50μs (5000 steps), smaller domain (40×80×40), flat plate.
 * Reports: T_max, v_max, Ma, melt depth, liquid cells, SumF every 1000 steps.
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

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L", sizeof(mat.name) - 1);
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

struct TestConfig {
    const char* id;
    const char* desc;
    float laser_power;
    bool marangoni;
    bool surface_tension;
    bool recoil;
    float recoil_mult;
    bool evap_mass_loss;     // true = Path A (VOF-coupled cooling + mass loss)
    float mass_loss_scale;   // only used if evap_mass_loss=true
    float cfl_target;        // 0.30 = relaxed, 0.15 = production
    int steps;
};

void runTest(const TestConfig& tc, const char* csv_dir) {
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;
    const int NX=40, NY=80, NZ=40;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=20;
    const int num_cells = NX*NY*NZ;
    const float cs_phys = dx / (dt * sqrtf(3.0f));

    printf("\n================================================================\n");
    printf("  %s: %s\n", tc.id, tc.desc);
    printf("  P=%.0fW, Maran=%s, ST=%s, Recoil=%s(×%.1f), EvapML=%s(scale=%.2f), CFL=%.2f\n",
           tc.laser_power,
           tc.marangoni?"ON":"OFF", tc.surface_tension?"ON":"OFF",
           tc.recoil?"ON":"OFF", tc.recoil_mult,
           tc.evap_mass_loss?"ON":"OFF", tc.mass_loss_scale,
           tc.cfl_target);
    printf("================================================================\n");

    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;

    config.enable_thermal=true; config.enable_thermal_advection=true;
    config.use_fdm_thermal=true; config.enable_phase_change=true;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=true;
    config.enable_laser=true;
    config.enable_darcy=true;
    config.enable_marangoni=tc.marangoni;
    config.enable_surface_tension=tc.surface_tension;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=tc.evap_mass_loss;
    config.enable_recoil_pressure=tc.recoil;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity = nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=tc.recoil_mult;
    config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=1.0f;
    config.evap_cooling_factor=1.0f;

    config.laser_power=tc.laser_power;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=50.0e-6f;
    config.laser_start_y=50.0e-6f;
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=1.0f;

    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=tc.cfl_target;
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

    // Flat plate with tanh interface
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f) - (float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f - tanhf(2.0f*dist));
            }
    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    // CSV output
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s/%s_timeseries.csv", csv_dir, tc.id);
    FILE* csv = fopen(csv_path, "w");
    fprintf(csv, "step,t_us,T_max,v_max_phys,Ma,depth_um,liquid_cells,sum_fill\n");

    printf("  %6s %7s %9s %9s %6s %7s %8s %10s\n",
           "Step", "t[μs]", "T_max[K]", "v[m/s]", "Ma", "Depth", "LiqCell", "SumF");
    printf("  %6s %7s %9s %9s %6s %7s %8s %10s\n",
           "------", "-----", "-------", "-------", "----", "-----", "------", "--------");

    bool stable = true;
    for (int step = 0; step <= tc.steps; step++) {
        if (step > 0) solver.step();

        if (step % 1000 == 0 || step == tc.steps) {
            float T_max = solver.getMaxTemperature();
            float v_lat = solver.getMaxVelocity();
            float v_phys = v_lat * dx / dt;
            float Ma = v_phys / cs_phys;

            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            float max_depth = 0;
            int liq = 0;
            float sf = 0;
            for (int kk=0; kk<NZ; kk++)
                for (int jj=0; jj<NY; jj++)
                    for (int ii=0; ii<NX; ii++) {
                        int idx=ii+jj*NX+kk*NX*NY;
                        sf += h_f[idx];
                        if (h_T[idx]>=1650.0f && h_f[idx]>0.01f) {
                            liq++;
                            if (kk<z_surface) {
                                float d=(z_surface-1-kk)*dx*1e6f;
                                if (d>max_depth) max_depth=d;
                            }
                        }
                    }

            printf("  %6d %7.1f %9.0f %9.1f %6.3f %5.1fμm %8d %10.0f\n",
                   step, step*dt*1e6, T_max, v_phys, Ma, max_depth, liq, sf);
            fprintf(csv, "%d,%.3f,%.1f,%.2f,%.4f,%.2f,%d,%.1f\n",
                    step, step*dt*1e6, T_max, v_phys, Ma, max_depth, liq, sf);
            fflush(csv);

            if (std::isnan(T_max) || std::isinf(T_max) || T_max > 50000) {
                printf("  *** DIVERGED at step %d ***\n", step);
                stable = false;
                break;
            }
        }
    }
    fclose(csv);

    float T_final = solver.getMaxTemperature();
    float v_final = solver.getMaxVelocity() * dx / dt;
    float Ma_final = v_final / cs_phys;
    printf("\n  RESULT [%s]: %s — T_max=%.0fK, v_max=%.1f m/s, Ma=%.3f\n\n",
           tc.id, stable?"STABLE":"DIVERGED", T_final, v_final, Ma_final);
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    const char* csv_dir = "round2_audit/data";
    const float cs = 2.5e-6f / (1e-8f * sqrtf(3.0f));

    printf("================================================================\n");
    printf("  Round 2 Task 1: Ma with Evaporation Cooling\n");
    printf("  cs = %.1f m/s, Ma=0.3 threshold = %.1f m/s\n", cs, 0.3f*cs);
    printf("================================================================\n");

    // NOTE: When enable_evaporation_mass_loss=false, Path B provides evap cooling
    // via the fallback kernel (line 1811). Cooling IS active.
    TestConfig tests[] = {
        // id        desc                                            P   Mar  ST   Rec  Rmul EvML  MLS   CFL   steps
        {"R2-4.1", "P=10W, Marangoni+evapCool",                   10, true, false,false,0,  false,1.0f, 0.30f, 5000},
        {"R2-4.2", "P=30W, Marangoni+evapCool",                   30, true, false,false,0,  false,1.0f, 0.30f, 5000},
        {"R2-4.3", "P=50W, Marangoni+evapCool",                   50, true, false,false,0,  false,1.0f, 0.30f, 5000},
        {"R2-4.4", "P=50W, +surfTens+evapCool",                   50, true, true, false,0,  false,1.0f, 0.30f, 5000},
        {"R2-4.5", "P=50W, +recoil×0.1+evapCool",                 50, true, true, true,0.1f,false,1.0f, 0.30f, 5000},
        {"R2-4.6", "P=80W, Maran+ST+evapCool",                    80, true, true, false,0,  false,1.0f, 0.30f, 5000},
        {"R2-4.7", "P=100W, Maran+ST+evapCool",                  100, true, true, false,0,  false,1.0f, 0.30f, 5000},
        {"R2-4.8", "P=150W, all physics, no bandaids (scale=1)",  150, true, true, true,1.0f,true, 1.0f, 0.30f, 5000},
        {"R2-4.9", "P=150W, production config",                   150, true, true, true,1.0f,true, 0.05f,0.15f, 5000},
    };

    for (auto& t : tests) {
        runTest(t, csv_dir);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t1 - t0).count();
    printf("\n================================================================\n");
    printf("  Task 1 Complete. Total time: %.1f s (%.1f min)\n", elapsed, elapsed/60);
    printf("================================================================\n");
    return 0;
}
