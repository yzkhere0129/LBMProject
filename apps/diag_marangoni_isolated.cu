/**
 * @file diag_marangoni_isolated.cu
 * @brief Diagnostic: Frozen T field + fluid-only Marangoni (Test 1 & 3)
 *
 * Test 1: Freeze thermal module. Inject a fixed Gaussian T field.
 *         Run fluid-only with Inamuro stress BC. Check if vz forms deep jet.
 *
 * Test 3: Probes at 3 points, first 100 steps with full coupling.
 */

#include <iostream>
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
    std::strncpy(mat.name, "316L_Diag", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700; mat.T_vaporization=3500;
    mat.L_fusion=260000; mat.L_vaporization=7e6; mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f; mat.dsigma_dT=+1e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f; mat.emissivity=0.3f;
    return mat;
}

int main() {
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k=20, mu=0.005f;
    const float alpha = k/(rho*cp), nu = mu/rho;
    const int NX=100, NY=100, NZ=50;
    const float dx=2e-6f;
    const float tau_th=1.0f, alpha_LU=(tau_th-0.5f)*0.25f;
    const float dt = alpha_LU*dx*dx/alpha;
    const float dx_um = dx*1e6f;

    printf("=== DIAGNOSTIC: Marangoni Isolation + Probes ===\n");
    printf("dx=%.0f um, dt=%.2f ns\n", dx*1e6f, dt*1e9f);

    // ================================================================
    // TEST 3: Full coupling, first 100 steps, 3 probes
    // ================================================================
    printf("\n========== TEST 3: Early-time probes (100 steps) ==========\n");
    {
        MultiphysicsConfig cfg;
        cfg.nx=NX; cfg.ny=NY; cfg.nz=NZ; cfg.dx=dx; cfg.dt=dt;
        cfg.material=mat;
        cfg.enable_thermal=true; cfg.enable_thermal_advection=true;
        cfg.enable_phase_change=true; cfg.enable_fluid=true;
        cfg.enable_vof=false; cfg.enable_laser=true;
        cfg.enable_darcy=true; cfg.enable_marangoni=true;
        cfg.enable_surface_tension=false; cfg.enable_buoyancy=false;
        cfg.enable_evaporation_mass_loss=false; cfg.enable_recoil_pressure=false;
        cfg.kinematic_viscosity = nu*dt/(dx*dx);
        cfg.density=rho; cfg.darcy_coefficient=1e6f;
        cfg.thermal_diffusivity=alpha;
        cfg.surface_tension_coeff=mat.surface_tension;
        cfg.dsigma_dT=mat.dsigma_dT;
        cfg.laser_power=150; cfg.laser_spot_radius=25e-6f;
        cfg.laser_absorptivity=0.35f; cfg.laser_penetration_depth=dx;
        cfg.laser_shutoff_time=50e-6f;
        cfg.laser_start_x=-1; cfg.laser_start_y=-1;
        cfg.laser_scan_vx=0; cfg.laser_scan_vy=0;
        cfg.ray_tracing.enabled=false;
        cfg.cfl_velocity_target=0.15f;
        cfg.boundaries.x_min=cfg.boundaries.x_max=BoundaryType::WALL;
        cfg.boundaries.y_min=cfg.boundaries.y_max=BoundaryType::WALL;
        cfg.boundaries.z_min=cfg.boundaries.z_max=BoundaryType::WALL;
        cfg.boundaries.thermal_x_min=cfg.boundaries.thermal_x_max=ThermalBCType::DIRICHLET;
        cfg.boundaries.thermal_y_min=cfg.boundaries.thermal_y_max=ThermalBCType::DIRICHLET;
        cfg.boundaries.thermal_z_min=ThermalBCType::DIRICHLET;
        cfg.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
        cfg.boundaries.dirichlet_temperature=300.0f;

        MultiphysicsSolver solver(cfg);
        solver.initialize(300.0f, 1.0f);

        const int num_cells = NX*NY*NZ;
        std::vector<float> h_T(num_cells);

        // Probe indices: A=(50,50,49), B=(50,50,44), C=(55,50,49)
        const int pA = 50 + 50*NX + 49*NX*NY;
        const int pB = 50 + 50*NX + 44*NX*NY;
        const int pC = 55 + 50*NX + 49*NX*NY;

        printf("%5s %8s | %8s %10s | %8s %10s | %8s %10s\n",
               "step", "t[ns]",
               "T_A[K]", "dTdx_A",
               "T_B[K]", "T_C[K]",
               "vmax_LU", "vmax_m/s");

        for (int step = 0; step <= 100; step++) {
            if (step > 0) solver.step();

            if (step % 5 == 0 || step <= 10) {
                solver.copyTemperatureToHost(h_T.data());
                float T_A = h_T[pA], T_B = h_T[pB], T_C = h_T[pC];
                // dTdx at probe A: central diff
                float dTdx_A = (h_T[pA+1] - h_T[pA-1]) * 0.5f;
                float v_max_phys = solver.getMaxVelocity();  // m/s
                float v_max_LU = v_max_phys * dt / dx;        // derived lattice-unit view

                printf("%5d %8.1f | %8.0f %10.1f | %8.0f %10.0f | %8.5f %10.3f\n",
                       step, step*dt*1e9f,
                       T_A, dTdx_A,
                       T_B, T_C,
                       v_max_LU, v_max_phys);
            }
        }
    }

    // ================================================================
    // TEST 1: Frozen T field, fluid-only Marangoni
    // ================================================================
    printf("\n========== TEST 1: Frozen T field + fluid only ==========\n");
    {
        // Build a synthetic Gaussian T field mimicking 50us state
        // T_max=10000K at center surface, decaying Gaussian in r and z
        MultiphysicsConfig cfg;
        cfg.nx=NX; cfg.ny=NY; cfg.nz=NZ; cfg.dx=dx; cfg.dt=dt;
        cfg.material=mat;
        cfg.enable_thermal=true;  // needed for liquid_fraction
        cfg.enable_thermal_advection=false;  // NO advection (frozen T)
        cfg.enable_phase_change=true;
        cfg.enable_fluid=true;
        cfg.enable_vof=false; cfg.enable_laser=false;  // NO laser
        cfg.enable_darcy=true; cfg.enable_marangoni=true;
        cfg.enable_surface_tension=false; cfg.enable_buoyancy=false;
        cfg.enable_evaporation_mass_loss=false; cfg.enable_recoil_pressure=false;
        cfg.kinematic_viscosity = nu*dt/(dx*dx);
        cfg.density=rho; cfg.darcy_coefficient=1e6f;
        cfg.thermal_diffusivity=alpha;
        cfg.dsigma_dT=mat.dsigma_dT;
        cfg.cfl_velocity_target=0.15f;
        cfg.boundaries.x_min=cfg.boundaries.x_max=BoundaryType::WALL;
        cfg.boundaries.y_min=cfg.boundaries.y_max=BoundaryType::WALL;
        cfg.boundaries.z_min=cfg.boundaries.z_max=BoundaryType::WALL;
        cfg.boundaries.thermal_x_min=cfg.boundaries.thermal_x_max=ThermalBCType::DIRICHLET;
        cfg.boundaries.thermal_y_min=cfg.boundaries.thermal_y_max=ThermalBCType::DIRICHLET;
        cfg.boundaries.thermal_z_min=ThermalBCType::DIRICHLET;
        cfg.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
        cfg.boundaries.dirichlet_temperature=300.0f;

        MultiphysicsSolver solver(cfg);

        // Create synthetic T field
        const int num_cells = NX*NY*NZ;
        std::vector<float> h_T(num_cells, 300.0f);
        const float T_peak = 10000.0f;
        const float r0 = 25e-6f / dx;  // beam radius in cells
        const float z0 = 15.0f;  // thermal penetration in cells

        for (int iz=0; iz<NZ; iz++)
            for (int iy=0; iy<NY; iy++)
                for (int ix=0; ix<NX; ix++) {
                    float rx = ix - NX/2, ry = iy - NY/2;
                    float rr = sqrtf(rx*rx + ry*ry);
                    float dz = NZ - 1 - iz;  // depth from surface
                    float T = 300.0f + (T_peak - 300.0f)
                              * expf(-2.0f*rr*rr/(r0*r0))
                              * expf(-dz*dz/(z0*z0));
                    h_T[ix + iy*NX + iz*NX*NY] = T;
                }

        // Initialize with synthetic T field + all-metal fill level
        std::vector<float> h_fill(num_cells, 1.0f);  // all metal
        solver.initialize(h_T.data(), h_fill.data());

        printf("Synthetic T field: T_peak=%.0f K at center surface\n", T_peak);
        printf("Running 500 fluid steps with frozen T...\n\n");

        printf("%5s %8s %10s %10s %10s\n",
               "step", "v_max", "v_max_phys", "D_sol[um]", "D_liq[um]");

        std::vector<float> h_T_out(num_cells);
        for (int step = 0; step <= 500; step++) {
            if (step > 0) solver.step();

            // T field evolves by diffusion only (no laser, no advection).
            // The gradient structure persists but weakens over time.

            if (step % 25 == 0) {
                solver.copyTemperatureToHost(h_T_out.data());
                float v_phys = solver.getMaxVelocity();  // m/s
                float v_max_LU = v_phys * dt / dx;       // derived lattice-unit view

                // Melt pool depth from actual (evolved) T
                const int cx=NX/2, cy=NY/2;
                float d_sol=0, d_liq=0;
                for (int iz=NZ-1; iz>=0; iz--) {
                    int idx = cx+cy*NX+iz*NX*NY;
                    float depth = (NZ-1-iz)*dx_um;
                    if (h_T_out[idx]>=1650 && depth>d_sol) d_sol=depth;
                    if (h_T_out[idx]>=1700 && depth>d_liq) d_liq=depth;
                }
                printf("%5d %8.4f %10.2f %10.1f %10.1f\n",
                       step, v_max_LU, v_phys, d_sol, d_liq);
            }
        }
        // (no device buffers to free)
    }

    return 0;
}
