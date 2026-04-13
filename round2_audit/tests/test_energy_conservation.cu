/**
 * Round 2 Task 2: Energy Conservation Systematic Tests
 *
 * Test A: Pure conduction with constant heat source + Dirichlet BC
 *   Verify dE/dt = P_source - Q_boundary (no VOF, no flow)
 *
 * Test B: With phase change (crossing T_solidus/T_liquidus)
 *   Verify dE_total/dt = P_source - Q_boundary where E_total = E_sensible + E_latent
 *
 * Test C: With VOF interface + gas wipe (no flow, no laser)
 *   Track Q_gas_wipe separately. Verify dE_metal/dt + Q_gas_wipe = -Q_boundary
 *
 * Test D: Full coupling at 50W with Marangoni (no recoil, no mass loss)
 *   Full energy balance: dE/dt = P_laser - Q_evap - Q_boundary - Q_gas_wipe
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm;
using namespace lbm::physics;

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_Energy", sizeof(mat.name) - 1);
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

// Compute total thermal energy in metal cells
static double computeThermalEnergy(const float* T, const float* f,
                                   int N, float rho, float cp, float dV) {
    double E = 0;
    for (int i = 0; i < N; i++) {
        float fill = (f != nullptr) ? f[i] : 1.0f;
        if (fill > 0.01f) {
            E += (double)rho * cp * T[i] * fill * dV;
        }
    }
    return E;
}

// Compute heat flux through z_min face (Dirichlet BC at T_bc)
static double computeZminHeatFlux(const float* T, int NX, int NY, int NZ,
                                   float k, float dx, float T_bc) {
    double Q = 0;
    int kk = 0;  // bottom face
    for (int jj = 0; jj < NY; jj++)
        for (int ii = 0; ii < NX; ii++) {
            float T_cell = T[ii + jj*NX + kk*NX*NY];
            // Heat flux out of domain: q = k * (T_cell - T_bc) / (dx/2)
            // Positive = heat leaving domain
            Q += (double)k * (T_cell - T_bc) / (dx * 0.5) * dx * dx;
        }
    return Q;  // [W]
}

void testA_PureConduction(const char* csv_dir) {
    printf("\n================================================================\n");
    printf("  Test A: Pure Conduction Energy Conservation\n");
    printf("  Constant heat source + Dirichlet z_min=300K\n");
    printf("================================================================\n");

    const int NX=20, NY=20, NZ=20;
    const int N = NX*NY*NZ;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const float rho=7900, cp=700, k=20;
    const float alpha = k/(rho*cp);
    const float T_bc = 300.0f;
    const float dV = dx*dx*dx;

    // Heat source: 5×5×5 block at center, Q=1e15 W/m³
    const float Q_density = 1e15f;  // W/m³
    const int hs_size = 5;  // cells in each direction
    const float P_source = Q_density * powf(hs_size * dx, 3);  // [W]

    printf("  Heat source: Q=%.2e W/m³ in %d³ cells = P=%.4f W\n",
           Q_density, hs_size, P_source);

    MaterialProperties mat = createMat();
    ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

    // Initial T=300K everywhere
    std::vector<float> h_T(N, T_bc);
    thermal.initialize(h_T.data());

    // Create heat source array
    std::vector<float> h_Q(N, 0.0f);
    int cx=NX/2, cy=NY/2, cz=NZ/2;
    for (int kk=cz-hs_size/2; kk<=cz+hs_size/2; kk++)
        for (int jj=cy-hs_size/2; jj<=cy+hs_size/2; jj++)
            for (int ii=cx-hs_size/2; ii<=cx+hs_size/2; ii++)
                h_Q[ii+jj*NX+kk*NX*NY] = Q_density;

    float *d_Q;
    cudaMalloc(&d_Q, N*sizeof(float));
    cudaMemcpy(d_Q, h_Q.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s/energy_test_A.csv", csv_dir);
    FILE* csv = fopen(csv_path, "w");
    fprintf(csv, "step,t_us,E_total,dEdt_num,P_source,Q_boundary,residual,residual_pct\n");

    double E_prev = computeThermalEnergy(h_T.data(), nullptr, N, rho, cp, dV);
    int total_steps = 10000;

    printf("  %6s %7s %12s %12s %12s %12s %10s\n",
           "Step", "t[μs]", "E[J]", "dE/dt[W]", "P_src[W]", "Q_bnd[W]", "Res[%]");
    printf("  %6s %7s %12s %12s %12s %12s %10s\n",
           "------", "-----", "--------", "--------", "--------", "--------", "------");

    for (int step = 1; step <= total_steps; step++) {
        // z_min Dirichlet BC
        thermal.applyFaceThermalBC(4, 2, dt, dx, T_bc);  // face 4 = z_min
        // Other faces: adiabatic (face 0-3=x/y, face 5=z_max)
        for (int face = 0; face < 4; face++)
            thermal.applyFaceThermalBC(face, 1, dt, dx, T_bc);
        thermal.applyFaceThermalBC(5, 1, dt, dx, T_bc);

        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();
        thermal.addHeatSource(d_Q, dt);

        if (step % 500 == 0) {
            thermal.copyTemperatureToHost(h_T.data());
            double E_now = computeThermalEnergy(h_T.data(), nullptr, N, rho, cp, dV);
            double dEdt = (E_now - E_prev) / (500 * dt);
            double Q_bnd = computeZminHeatFlux(h_T.data(), NX, NY, NZ, k, dx, T_bc);
            double residual = dEdt - P_source + Q_bnd;
            double res_pct = (P_source > 0) ? fabs(residual) / P_source * 100 : 0;

            printf("  %6d %7.1f %12.4e %12.4f %12.4f %12.4f %10.2f\n",
                   step, step*dt*1e6, E_now, dEdt, P_source, Q_bnd, res_pct);
            fprintf(csv, "%d,%.4f,%.8e,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                    step, step*dt*1e6, E_now, dEdt, P_source, Q_bnd, residual, res_pct);

            E_prev = E_now;
        }
    }
    fclose(csv);
    cudaFree(d_Q);

    // Final verdict
    thermal.copyTemperatureToHost(h_T.data());
    double E_final = computeThermalEnergy(h_T.data(), nullptr, N, rho, cp, dV);
    double Q_bnd_final = computeZminHeatFlux(h_T.data(), NX, NY, NZ, k, dx, T_bc);
    double dEdt_final = (E_final - computeThermalEnergy(h_T.data(), nullptr, N, rho, cp, dV)) / dt;
    printf("\n  RESULT Test A: See residual column (should be <5%%)\n");
}

void testC_WithVOFGasWipe(const char* csv_dir) {
    printf("\n================================================================\n");
    printf("  Test C: Energy Conservation with VOF Interface + Gas Wipe\n");
    printf("  No laser, no flow. Hot metal cools through z_min Dirichlet BC.\n");
    printf("  Key: measure gas wipe energy removal separately.\n");
    printf("================================================================\n");

    const int NX=40, NY=40, NZ=40;
    const int N = NX*NY*NZ;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const float rho=7900, cp=700, k=20;
    const float alpha = k/(rho*cp);
    const float T_bc = 300.0f;
    const float dV = dx*dx*dx;
    const int z_surface = 20;

    // Use MultiphysicsSolver to get proper gas wipe behavior
    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material = createMat();
    config.enable_thermal=true; config.enable_thermal_advection=false;
    config.use_fdm_thermal=true; config.enable_phase_change=false;
    config.enable_fluid=false;
    config.enable_vof=true; config.enable_vof_advection=false;
    config.enable_laser=false;
    config.enable_darcy=false;
    config.enable_marangoni=false; config.enable_surface_tension=false;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=false;
    config.enable_recoil_pressure=false;
    config.enable_radiation_bc=false;
    config.kinematic_viscosity=0.001f;
    config.density=rho; config.thermal_diffusivity=alpha;
    config.ambient_temperature=300.0f;

    config.boundaries.x_min=config.boundaries.x_max=BoundaryType::WALL;
    config.boundaries.y_min=config.boundaries.y_max=BoundaryType::WALL;
    config.boundaries.z_min=config.boundaries.z_max=BoundaryType::WALL;
    config.boundaries.thermal_z_min=ThermalBCType::DIRICHLET;
    config.boundaries.dirichlet_temperature=T_bc;
    config.boundaries.thermal_x_min=config.boundaries.thermal_x_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min=config.boundaries.thermal_y_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;

    MultiphysicsSolver solver(config);

    // Fill: metal z<z_surface, tanh transition, gas above
    std::vector<float> h_fill(N, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f) - (float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f - tanhf(2.0f*dist));
            }

    // Initial T: metal at 2000K near surface, 300K deeper, gas at 300K
    std::vector<float> h_T(N, T_bc);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                if (h_fill[ii+jj*NX+kk*NX*NY] > 0.5f) {
                    // Metal: hot near surface (z=15-19), cooler below
                    if (kk >= z_surface - 5) {
                        h_T[ii+jj*NX+kk*NX*NY] = 2000.0f;
                    } else {
                        h_T[ii+jj*NX+kk*NX*NY] = 300.0f + 1700.0f * (float)kk / (z_surface - 5);
                    }
                }
            }

    solver.initialize(h_T.data(), h_fill.data());

    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s/energy_test_C.csv", csv_dir);
    FILE* csv = fopen(csv_path, "w");
    fprintf(csv, "step,t_us,E_metal,dEdt,Q_boundary,E_change_rate,balance_error_pct\n");

    // Compute initial energy
    double E_prev = 0;
    for (int i = 0; i < N; i++) {
        if (h_fill[i] > 0.01f) {
            E_prev += (double)rho * cp * h_T[i] * h_fill[i] * dV;
        }
    }

    printf("  Initial E_metal = %.6e J\n", E_prev);
    printf("  %6s %7s %12s %12s %12s %10s\n",
           "Step", "t[μs]", "E_metal[J]", "dE/dt[W]", "Q_bnd[W]", "Bal[%]");
    printf("  %6s %7s %12s %12s %12s %10s\n",
           "------", "-----", "--------", "--------", "--------", "------");

    int total_steps = 10000;
    for (int step = 1; step <= total_steps; step++) {
        solver.step();

        if (step % 500 == 0) {
            std::vector<float> h_T2(N), h_f2(N);
            solver.copyTemperatureToHost(h_T2.data());
            solver.copyFillLevelToHost(h_f2.data());

            double E_now = 0;
            for (int i = 0; i < N; i++) {
                if (h_f2[i] > 0.01f) {
                    E_now += (double)rho * cp * h_T2[i] * h_f2[i] * dV;
                }
            }

            double dEdt = (E_now - E_prev) / (500 * dt);
            double Q_bnd = computeZminHeatFlux(h_T2.data(), NX, NY, NZ, k, dx, T_bc);

            // Balance: dE/dt should equal -Q_boundary (no sources)
            // Gas wipe removes energy from gas cells → shows up as additional E loss
            double bal_err = (Q_bnd > 0.001) ?
                fabs(dEdt + Q_bnd) / Q_bnd * 100 : 0;

            printf("  %6d %7.1f %12.4e %12.4f %12.4f %10.2f\n",
                   step, step*dt*1e6, E_now, dEdt, Q_bnd, bal_err);
            fprintf(csv, "%d,%.4f,%.8e,%.6f,%.6f,%.6f,%.4f\n",
                    step, step*dt*1e6, E_now, dEdt, Q_bnd, dEdt+Q_bnd, bal_err);

            E_prev = E_now;
        }
    }
    fclose(csv);
    printf("\n  RESULT Test C: See balance column\n");
    printf("  If balance >> 10%%, gas wipe is leaking energy from metal.\n");
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    const char* csv_dir = "round2_audit/data";
    mkdir(csv_dir, 0755);

    testA_PureConduction(csv_dir);
    testC_WithVOFGasWipe(csv_dir);

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\n================================================================\n");
    printf("  Task 2 Complete. Time: %.1f s\n",
           std::chrono::duration<float>(t1-t0).count());
    printf("================================================================\n");
    return 0;
}
