/**
 * @file benchmark_2d_marangoni_cavity.cu
 * @brief 2D Thermocapillary Cavity Benchmark (Zebib 1988)
 * 
 * With custom Dirichlet BC kernel to maintain temperature gradient.
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>
#include "physics/multiphysics_solver.h"

using namespace lbm;
using namespace lbm::physics;

// ============================================================================
// D3Q7 Lattice (for equilibrium computation)
// ============================================================================
namespace D3Q7Local {
    static constexpr int Q = 7;
    static constexpr float CS2 = 0.25f;  // cs² = 1/4 for D3Q7 thermal
    
    __device__ float computeEquilibrium(int q, float T, float ux, float uy, float uz) {
        // D3Q7 weights
        const float w[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
        // D3Q7 velocities
        const int cx[7] = {0, 1, -1, 0, 0, 0, 0};
        const int cy[7] = {0, 0, 0, 1, -1, 0, 0};
        const int cz[7] = {0, 0, 0, 0, 0, 1, -1};
        
        float cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
        float usq = ux*ux + uy*uy + uz*uz;
        
        return w[q] * T * (1.0f + cu/CS2 + 0.5f*(cu*cu)/(CS2*CS2) - 0.5f*usq/CS2);
    }
}

// ============================================================================
// Zebib Top Surface Temperature BC Kernel
// ============================================================================
__global__ void enforceZebibTopTemperatureBC(
    float* T,
    float* g,
    int nx, int ny, int nz,
    float T_left, float T_right)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= nx || y >= ny) return;
    
    // Target temperature: linear gradient from T_left to T_right
    float target_T = T_left + (T_right - T_left) * (float)x / (float)(nx - 1);
    
    // Apply to top 3 layers (interface region)
    for (int dz = 0; dz < 3; dz++) {
        int z = nz - 1 - dz;
        if (z < 0) continue;
        
        int idx = x + y * nx + z * nx * ny;
        int num_cells = nx * ny * nz;
        
        // Set macroscopic temperature
        T[idx] = target_T;
        
        // Reset distribution functions to equilibrium (u=0)
        for (int q = 0; q < 7; q++) {
            g[q * num_cells + idx] = D3Q7Local::computeEquilibrium(q, target_T, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ============================================================================
// Zebib Benchmark Parameters
// ============================================================================
struct ZebibParams {
    float Re;
    float Ma;
    float Pr;
};

ZebibParams getZebibParams() {
    ZebibParams p;
    p.Re = 1000.0f;
    p.Ma = 1000.0f;
    p.Pr = 0.01f;
    return p;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("\n");
    printf("============================================================\n");
    printf("  2D Thermocapillary Cavity Benchmark (Zebib 1988)\n");
    printf("  Ma = 1000, Re = 1000, Pr = 0.01\n");
    printf("  WITH CUSTOM DIRICHLET BC KERNEL\n");
    printf("============================================================\n\n");

    auto params = getZebibParams();
    
    // ==================================================================
    // ACOUSTIC SCALING
    // ==================================================================
    const float U_LB = 0.05f;
    const int N_H = 100;
    const float DeltaT_LB = 1.0f;
    const float rho_LB = 1.0f;
    const float Re = params.Re;
    const float Pr = params.Pr;
    
    const float nu_LB = U_LB * N_H / Re;
    const float alpha_LB = nu_LB / Pr;
    const float tau_f = 3.0f * nu_LB + 0.5f;
    const float tau_T = 3.0f * alpha_LB + 0.5f;
    const float gamma_LB = U_LB * rho_LB * nu_LB / DeltaT_LB;
    
    printf("Acoustic Scaling:\n");
    printf("  U_LB = %.4f, ν_LB = %.6f, α_LB = %.6f\n", U_LB, nu_LB, alpha_LB);
    printf("  τ_f = %.4f, τ_T = %.4f\n", tau_f, tau_T);
    printf("  γ_LB = %.6f\n", gamma_LB);

    // ==================================================================
    // Domain
    // ==================================================================
    const int NX = N_H, NY = 1, NZ = N_H;
    const float dx = 1.0f, dt = 1.0f;
    const int num_cells = NX * NY * NZ;
    
    // Zebib temperature BC
    const float T_left = 1.0f;   // Hot at x=0
    const float T_right = 0.0f;  // Cold at x=NX-1
    
    // ==================================================================
    // Solver configuration
    // ==================================================================
    MultiphysicsConfig config;
    config.nx = NX;
    config.ny = NY;
    config.nz = NZ;
    config.dx = dx;
    config.dt = dt;
    
    // Physics flags
    config.enable_thermal = true;
    config.enable_thermal_advection = true;
    config.enable_fluid = true;
    config.enable_marangoni = true;
    config.enable_phase_change = false;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_recoil_pressure = false;
    config.enable_buoyancy = false;
    config.enable_surface_tension = false;
    
    // Material (ZERO background temperature)
    config.material.rho_solid = rho_LB;
    config.material.rho_liquid = rho_LB;
    config.material.cp_solid = 1.0f;
    config.material.cp_liquid = 1.0f;
    config.material.k_solid = 1.0f;
    config.material.k_liquid = 1.0f;
    config.material.mu_liquid = nu_LB * rho_LB;
    config.material.T_solidus = 0.5f;  // Normal range
    config.material.T_liquidus = 0.6f;
    config.material.L_fusion = 1.0f;
    config.density = rho_LB;
    
    // Thermal & Marangoni
    config.thermal_diffusivity = alpha_LB;
    config.dsigma_dT = gamma_LB;
    config.surface_tension_coeff = 1.0f;
    
    // Boundary conditions
    config.boundaries.x_min = BoundaryType::WALL;
    config.boundaries.x_max = BoundaryType::WALL;
    config.boundaries.z_min = BoundaryType::WALL;
    config.boundaries.z_max = BoundaryType::WALL;
    
    // ALL ADIABATIC - we'll enforce temperature BC manually
    config.boundaries.thermal_x_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_x_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature = 0.5f;  // Neutral background

    // ==================================================================
    // Initialize with temperature gradient at interface
    // CRITICAL: ALL cells start at 0.5 (no 300K contamination)
    std::vector<float> h_T_init(num_cells, 0.5f);  // ALL 0.5
    std::vector<float> h_fill(num_cells, 1.0f);    // ALL liquid
    
    // Gas layer at top (5 cells from boundary to avoid stencil truncation)
    const int NZ_GAS = 10;  // Larger buffer to ensure interface is interior
    const int iz_interface = NZ - NZ_GAS - 1;  // Interface at z = NZ-11
    
    // Set gas cells
    for (int iz = NZ - NZ_GAS; iz < NZ; iz++) {
        for (int iy = 0; iy < NY; iy++) {
            for (int ix = 0; ix < NX; ix++) {
                h_fill[ix + iy * NX + iz * NX * NY] = 0.0f;  // Gas
            }
        }
    }
    
    // Set interface layer (fill = 0.5)
    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
            h_fill[ix + iy * NX + iz_interface * NX * NY] = 0.5f;
        }
    }
    
    // Temperature gradient: ONLY at interface and nearby cells
    // All other cells stay at 0.5 (clean baseline)
    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
            float x_norm = (float)ix / (float)(NX - 1);
            float T_target = T_left + (T_right - T_left) * x_norm;
            
            // Apply gradient to interface ± 2 cells
            for (int dz = -2; dz <= 2; dz++) {
                int iz = iz_interface + dz;
                if (iz >= 0 && iz < NZ) {
                    h_T_init[ix + iy * NX + iz * NX * NY] = T_target;
                }
            }
        }
    }
    
    printf("Interface setup:\n");
    printf("  iz_interface = %d (NZ=%d, NZ_GAS=%d)\n", iz_interface, NZ, NZ_GAS);
    printf("  fill at interface: %.1f\n", h_fill[NX/2 + iz_interface * NX * NY]);
    printf("  T at interface (x=0): %.2f\n", h_T_init[0 + iz_interface * NX * NY]);
    printf("  T at interface (x=N/2): %.2f\n", h_T_init[NX/2 + iz_interface * NX * NY]);
    printf("  T at interface (x=N-1): %.2f\n", h_T_init[NX-1 + iz_interface * NX * NY]);
    
    // ==================================================================
    // PROBE: Verify input fill_level before solver initialization
    // ==================================================================
    printf("\n=== FILL LEVEL INPUT PROBE ===\n");
    printf("h_fill at interface (iz=%d):\n", iz_interface);
    printf("  fill(x=0,   z=%d) = %.4f\n", iz_interface, h_fill[0 + iz_interface * NX * NY]);
    printf("  fill(x=N/2, z=%d) = %.4f\n", iz_interface, h_fill[NX/2 + iz_interface * NX * NY]);
    printf("  fill(x=N-1, z=%d) = %.4f\n", iz_interface, h_fill[NX-1 + iz_interface * NX * NY]);
    printf("================================\n");
    
    MultiphysicsSolver solver(config);
    solver.initialize(h_T_init.data(), h_fill.data());

    // ==================================================================
    // PROBE: Verify static VOF interface and normals
    // ==================================================================
    printf("\n=== INTERFACE PROBE (After Init) ===\n");
    
    // Check fill_level distribution
    std::vector<float> h_fill_probe(num_cells);
    solver.copyFillLevelToHost(h_fill_probe.data());
    
    printf("Fill level at interface (iz=%d):\n", iz_interface);
    printf("  fill(x=0,   z=%d) = %.4f\n", iz_interface, h_fill_probe[0 + iz_interface * NX * NY]);
    printf("  fill(x=N/2, z=%d) = %.4f\n", iz_interface, h_fill_probe[NX/2 + iz_interface * NX * NY]);
    printf("  fill(x=N-1, z=%d) = %.4f\n", iz_interface, h_fill_probe[NX-1 + iz_interface * NX * NY]);
    
    printf("\nFill level above interface (iz=%d):\n", iz_interface+1);
    printf("  fill(x=N/2, z=%d) = %.4f\n", iz_interface+1, h_fill_probe[NX/2 + (iz_interface+1) * NX * NY]);
    
    printf("\nFill level below interface (iz=%d):\n", iz_interface-1);
    printf("  fill(x=N/2, z=%d) = %.4f\n", iz_interface-1, h_fill_probe[NX/2 + (iz_interface-1) * NX * NY]);
    
    // Check temperature
    std::vector<float> h_T_probe(num_cells);
    solver.copyTemperatureToHost(h_T_probe.data());
    
    printf("\nTemperature at interface:\n");
    printf("  T(x=0,   z=%d) = %.4f\n", iz_interface, h_T_probe[0 + iz_interface * NX * NY]);
    printf("  T(x=N/2, z=%d) = %.4f\n", iz_interface, h_T_probe[NX/2 + iz_interface * NX * NY]);
    printf("  T(x=N-1, z=%d) = %.4f\n", iz_interface, h_T_probe[NX-1 + iz_interface * NX * NY]);
    
    printf("=====================================\n");

    // ==================================================================
    // Simulation - Run to steady state
    // Re=1000: viscous diffusion time ~ N²/ν = 10000/0.005 = 2,000,000 steps
    // Run at least 500,000 steps or until kinetic energy converges
    // ==================================================================
    const int total_steps = 500000;  // Long run for Re=1000
    const int output_interval = 50000;
    const int check_interval = 10000;
    
    std::vector<float> h_ux(num_cells), h_uz(num_cells), h_T(num_cells);
    
    printf("\nStep       t          u_max        x_vortex    z_vortex\n");
    printf("--------  --------  -----------  ----------  ----------\n");
    
    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        
        // HARD OVERRIDE: Enforce temperature BC BEFORE step
        // This ensures Marangoni force sees the correct gradient
        if (step > 0) {
            const float* d_T_const = solver.getTemperature();
            float* d_T = const_cast<float*>(d_T_const);
            
            std::vector<float> h_T_override(num_cells);
            solver.copyTemperatureToHost(h_T_override.data());
            
            // Overwrite top layers with Zebib gradient
            for (int iz = NZ - NZ_GAS - 3; iz < NZ; iz++) {
                for (int iy = 0; iy < NY; iy++) {
                    for (int ix = 0; ix < NX; ix++) {
                        float x_norm = (float)ix / (float)(NX - 1);
                        float T_target = T_left + (T_right - T_left) * x_norm;
                        h_T_override[ix + iy * NX + iz * NX * NY] = T_target;
                    }
                }
            }
            
            cudaMemcpy(d_T, h_T_override.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        // Run simulation
        if (step > 0) {
            solver.step(dt);
        }
        
        // Convergence check every 10000 steps
        if (step > 0 && step % check_interval == 0) {
            solver.copyVelocityToHost(h_ux.data(), nullptr, h_uz.data());
            
            // Compute total kinetic energy
            float KE = 0.0f;
            for (int i = 0; i < num_cells; i++) {
                KE += h_ux[i]*h_ux[i] + h_uz[i]*h_uz[i];
            }
            KE *= 0.5f;
            
            static float prev_KE = 0.0f;
            if (prev_KE > 1e-15f) {
                float dKE = fabsf(KE - prev_KE) / prev_KE;
                if (step % output_interval == 0) {
                    printf("  Step %d: KE=%.6e, dKE=%.2e\n", step, KE, dKE);
                }
                if (dKE < 1e-6f && step > 100000) {
                    printf("\n*** CONVERGED at step %d (dKE < 1e-6) ***\n", step);
                    break;
                }
            }
            prev_KE = KE;
        }
        
        // Output
        if (step % output_interval == 0) {
            solver.copyVelocityToHost(h_ux.data(), nullptr, h_uz.data());
            solver.copyTemperatureToHost(h_T.data());
            
            // Find u_max and vortex
            float u_max = 0.0f;
            float x_vortex = 0.5f, z_vortex = 0.5f;
            float max_vorticity = 0.0f;
            
            for (int iz = 2; iz < NZ-2; iz++) {
                for (int ix = 2; ix < NX-2; ix++) {
                    int idx = ix + iz * NX * NY;
                    float u_mag = sqrtf(h_ux[idx]*h_ux[idx] + h_uz[idx]*h_uz[idx]);
                    if (u_mag > u_max) u_max = u_mag;
                    
                    float dw_dx = (h_ux[idx+1] - h_ux[idx-1]) / (2.0f * dx);
                    float du_dz = (h_uz[idx+NX] - h_uz[idx-NX]) / (2.0f * dx);
                    float vort = fabsf(dw_dx - du_dz);
                    if (vort > max_vorticity) {
                        max_vorticity = vort;
                        x_vortex = (ix + 0.5f) * dx / NX;
                        z_vortex = (iz + 0.5f) * dx / NZ;
                    }
                }
            }
            
            printf("%8d  %8.2f  %11.6f  %10.4f  %10.4f\n",
                   step, t, u_max, x_vortex, z_vortex);
            
            // Temperature probe
            int idx_left = 0 + iz_interface * NX * NY;
            int idx_mid = NX/2 + iz_interface * NX * NY;
            int idx_right = (NX-1) + iz_interface * NX * NY;
            
            printf("  T_probe: (%.4f, %.4f, %.4f)\n",
                   h_T[idx_left], h_T[idx_mid], h_T[idx_right]);
        }
    }

    // ==================================================================
    // Summary
    // ==================================================================
    solver.copyVelocityToHost(h_ux.data(), nullptr, h_uz.data());
    
    float u_max_final = 0.0f;
    for (int i = 0; i < num_cells; i++) {
        float u_mag = sqrtf(h_ux[i]*h_ux[i] + h_uz[i]*h_uz[i]);
        if (u_mag > u_max_final) u_max_final = u_mag;
    }
    
    float u_max_nondim = u_max_final / U_LB;
    
    printf("\n============================================================\n");
    printf("  ZEBIB BENCHMARK RESULTS\n");
    printf("============================================================\n");
    printf("  u_max (lattice):     %.6f\n", u_max_final);
    printf("  u_max (nondim):      %.6f (= u_LB / U_LB)\n", u_max_nondim);
    printf("  u_max (Zebib ref):   0.247\n");
    printf("  Ratio:               %.4f\n", u_max_nondim / 0.247f);
    printf("============================================================\n\n");

    // Write velocity field for visualization
    solver.copyVelocityToHost(h_ux.data(), nullptr, h_uz.data());
    solver.copyTemperatureToHost(h_T.data());
    
    FILE* fv = fopen("marangoni_cavity_velocity.csv", "w");
    if (fv) {
        fprintf(fv, "x,z,ux,uz,u_mag,T\n");
        for (int iz = 0; iz < NZ; iz++) {
            for (int ix = 0; ix < NX; ix++) {
                int idx = ix + iz * NX * NY;
                float x = ix * dx;
                float z = iz * dx;
                float u_mag = sqrtf(h_ux[idx]*h_ux[idx] + h_uz[idx]*h_uz[idx]);
                fprintf(fv, "%.6f,%.6f,%.8f,%.8f,%.8f,%.4f\n",
                        x, z, h_ux[idx], h_uz[idx], u_mag, h_T[idx]);
            }
        }
        fclose(fv);
        printf("Velocity field written to: marangoni_cavity_velocity.csv\n");
    }

    return 0;
}
