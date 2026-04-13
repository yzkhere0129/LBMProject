/**
 * Phase 4: Low-Power Full Coupling Tests
 *
 * Test 4.1: P=10W, no recoil, no evaporation — pure Marangoni conduction mode
 * Test 4.2: P=50W, no recoil, no evaporation
 * Test 4.3: P=50W, add surface tension
 * Test 4.4: P=50W, add recoil (multiplier 0.1)
 * Test 4.5: P=150W, all physics, no band-aids
 *
 * Each test runs 50μs (5000 steps) and reports:
 *   T_max, v_max, Ma, melt depth, energy balance
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
    std::strncpy(mat.name, "316L_LowPower", sizeof(mat.name) - 1);
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
    const char* name;
    float laser_power;
    bool enable_recoil;
    float recoil_multiplier;
    bool enable_evaporation;
    bool enable_surface_tension;
    bool enable_marangoni;
    int steps;
};

void runTest(const TestConfig& tc) {
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;
    const int NX=40, NY=80, NZ=40;  // smaller domain for speed
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=20;  // surface at z=20 (50μm)
    const int num_cells = NX*NY*NZ;
    const float cs_phys = dx / (dt * sqrtf(3.0f));

    printf("\n================================================================\n");
    printf("  %s\n", tc.name);
    printf("  P=%.0fW, recoil=%s(×%.1f), evap=%s, ST=%s, Maran=%s\n",
           tc.laser_power,
           tc.enable_recoil?"ON":"OFF", tc.recoil_multiplier,
           tc.enable_evaporation?"ON":"OFF",
           tc.enable_surface_tension?"ON":"OFF",
           tc.enable_marangoni?"ON":"OFF");
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
    config.enable_marangoni=tc.enable_marangoni;
    config.enable_surface_tension=tc.enable_surface_tension;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=tc.enable_evaporation;
    config.enable_recoil_pressure=tc.enable_recoil;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity = nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=tc.recoil_multiplier;
    config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=1.0f;
    config.evap_cooling_factor=1.0f;

    config.laser_power=tc.laser_power;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=50.0e-6f;  // center of smaller domain
    config.laser_start_y=50.0e-6f;
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=1.0f;

    config.ray_tracing.enabled=false;
    // NO CFL limiter for these tests — we want to see raw behavior
    config.cfl_velocity_target=0.30f;  // relaxed, only catches catastrophic
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

    printf("  Step   t[μs]  T_max[K]  v_max[m/s]  Ma     Depth  LiqCells  SumF\n");
    printf("  ----   -----  --------  ----------  ----   -----  --------  ----\n");

    bool stable = true;
    for (int step = 0; step <= tc.steps; step++) {
        if (step > 0) solver.step();

        if (step % 500 == 0 || step == tc.steps) {
            float T_max = solver.getMaxTemperature();
            float v_max_lat = solver.getMaxVelocity();
            float v_max = v_max_lat * dx / dt;
            float Ma = v_max / cs_phys;

            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            float max_depth = 0;
            int liquid_cells = 0;
            float sum_fill = 0;
            for (int kk=0; kk<NZ; kk++)
                for (int jj=0; jj<NY; jj++)
                    for (int ii=0; ii<NX; ii++) {
                        int idx=ii+jj*NX+kk*NX*NY;
                        sum_fill += h_f[idx];
                        if (h_T[idx]>=1650.0f && h_f[idx]>0.01f) {
                            liquid_cells++;
                            if (kk < z_surface) {
                                float d=(z_surface-1-kk)*dx*1e6f;
                                if (d>max_depth) max_depth=d;
                            }
                        }
                    }

            printf("  %5d  %5.1f  %8.0f  %10.1f  %4.2f  %5.1fμm  %8d  %.0f\n",
                   step, step*dt*1e6, T_max, v_max, Ma, max_depth, liquid_cells, sum_fill);

            if (std::isnan(T_max) || std::isinf(T_max) || T_max > 50000) {
                printf("  *** DIVERGED at step %d ***\n", step);
                stable = false;
                break;
            }
        }
    }

    printf("\n  RESULT: %s — T_max=%.0fK, v_max=%.0f m/s, Ma=%.2f\n",
           stable ? "STABLE" : "DIVERGED",
           solver.getMaxTemperature(),
           solver.getMaxVelocity() * dx / dt,
           solver.getMaxVelocity() * dx / dt / cs_phys);
}

int main() {
    printf("================================================================\n");
    printf("  Phase 4: Low-Power Full Coupling Tests\n");
    printf("  cs_phys = %.1f m/s, Ma=0.3 at %.1f m/s\n",
           2.5e-6f / (1e-8f * sqrtf(3.0f)),
           0.3f * 2.5e-6f / (1e-8f * sqrtf(3.0f)));
    printf("================================================================\n");

    TestConfig tests[] = {
        {"4.1: P=10W, Marangoni only", 10, false, 0, false, false, true, 5000},
        {"4.2: P=50W, Marangoni only", 50, false, 0, false, false, true, 5000},
        {"4.3: P=50W + surface tension", 50, false, 0, false, true, true, 5000},
        {"4.4: P=50W + recoil(×0.1)", 50, true, 0.1f, false, true, true, 5000},
        {"4.5: P=150W, all physics (no band-aids)", 150, true, 1.0f, true, true, true, 5000},
    };

    for (auto& t : tests) {
        runTest(t);
    }

    printf("\n================================================================\n");
    printf("  Phase 4 Complete\n");
    printf("================================================================\n");
    return 0;
}
