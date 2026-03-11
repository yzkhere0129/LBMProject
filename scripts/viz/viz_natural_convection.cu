/**
 * @file viz_natural_convection.cu
 * @brief Visualization program for natural convection in a heated cavity (Ra=1e3)
 *
 * Runs coupled ThermalLBM + FluidLBM to steady state, dumps the z-midplane
 * temperature and velocity fields to CSV for matplotlib isotherms + streamlines.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/lattice_d3q7.h"
#include "core/lattice_d3q19.h"
#include "utils/cuda_check.h"

using lbm::physics::ThermalLBM;
using lbm::physics::FluidLBM;
using lbm::physics::D3Q7;
using lbm::core::D3Q19;

// Force conversion kernel (physical -> lattice)
__global__ void convertForceKernel(float* fx, float* fy, float* fz,
                                    float factor, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    fx[id] *= factor;
    fy[id] *= factor;
    fz[id] *= factor;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    // --- Grid ---
    const int n = 49;
    const int nx = n, ny = n, nz = 3;
    const int num_cells = nx * ny * nz;

    // --- Physical parameters ---
    const float H_physical = 0.01f;  // 1 cm cavity
    const float g_physical = 9.81f;
    const float T_ref = 300.0f;
    const float delta_T = 10.0f;
    const float T_hot = T_ref + delta_T / 2.0f;
    const float T_cold = T_ref - delta_T / 2.0f;
    const float Pr = 0.71f;
    const float Ra = 1e3f;

    // Air properties
    const float nu_air = 1.5e-5f;
    const float alpha_air = nu_air / Pr;

    // Lattice unit conversion
    const float dx_phys = H_physical / static_cast<float>(nx - 1);
    const float nu_lattice = 0.1f;  // Same as passing test (CFL_thermal=0.56 is marginal but works)
    const float dt_phys = nu_lattice * dx_phys * dx_phys / nu_air;
    const float alpha_lat = alpha_air * dt_phys / (dx_phys * dx_phys);

    // Thermal expansion coefficient from Ra
    const float beta = Ra * nu_air * alpha_air /
                       (g_physical * delta_T * H_physical * H_physical * H_physical);

    const float force_conversion = dt_phys * dt_phys / dx_phys;

    printf("Natural Convection Visualization (Ra=1e3)\n");
    printf("  Grid: %d x %d x %d\n", nx, ny, nz);
    printf("  nu_lattice=%.4f, alpha_lattice=%.4f\n", nu_lattice, alpha_lat);
    printf("  beta=%.6e, force_conversion=%.6e\n", beta, force_conversion);
    fflush(stdout);

    // --- Init lattices ---
    if (!D3Q7::isInitialized()) D3Q7::initializeDevice();
    if (!D3Q19::isInitialized()) D3Q19::initializeDevice();

    // --- Create solvers ---
    ThermalLBM thermal(nx, ny, nz, alpha_air, 1.0f, 1.0f, dt_phys, dx_phys);
    thermal.setZPeriodic(true);
    thermal.initialize(T_ref);

    FluidLBM fluid(nx, ny, nz, nu_air, 1.0f,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::PERIODIC,
                   dt_phys, dx_phys);
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // --- Allocate force fields ---
    float* d_fx; float* d_fy; float* d_fz;
    CUDA_CHECK(cudaMalloc(&d_fx, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, num_cells * sizeof(float)));

    // --- Time integration ---
    const int max_steps = 40000;
    const int check_interval = 2000;

    // For convergence check
    std::vector<float> h_ux_old(num_cells, 0.0f);
    std::vector<float> h_ux_new(num_cells, 0.0f);

    printf("Running simulation (%d steps max)...\n", max_steps);
    fflush(stdout);

    for (int step = 0; step < max_steps; ++step) {
        // 1. Thermal step
        thermal.collisionBGK(fluid.getVelocityX(), fluid.getVelocityY(),
                             fluid.getVelocityZ());
        thermal.streaming();
        thermal.computeTemperature();

        // 2. Thermal BCs (Dirichlet left/right, adiabatic top/bottom)
        thermal.applyFaceThermalBC(0, 2, dt_phys, dx_phys, T_hot);   // x_min: hot
        thermal.applyFaceThermalBC(1, 2, dt_phys, dx_phys, T_cold);  // x_max: cold
        thermal.applyFaceThermalBC(2, 1, dt_phys, dx_phys);          // y_min: adiabatic
        thermal.applyFaceThermalBC(3, 1, dt_phys, dx_phys);          // y_max: adiabatic

        // 3. Buoyancy force (physical units)
        CUDA_CHECK(cudaMemset(d_fx, 0, num_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_fy, 0, num_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_fz, 0, num_cells * sizeof(float)));

        fluid.computeBuoyancyForce(thermal.getTemperature(), T_ref, beta,
                                   0.0f, -g_physical, 0.0f,
                                   d_fx, d_fy, d_fz);

        // Convert to lattice units
        int bs = 256;
        int gs = (num_cells + bs - 1) / bs;
        convertForceKernel<<<gs, bs>>>(d_fx, d_fy, d_fz, force_conversion, num_cells);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Fluid step
        fluid.collisionTRT(d_fx, d_fy, d_fz, 3.0f / 16.0f);
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic(d_fx, d_fy, d_fz);

        // 5. Convergence check
        if (step > 0 && step % check_interval == 0) {
            CUDA_CHECK(cudaMemcpy(h_ux_new.data(), fluid.getVelocityX(),
                                  num_cells * sizeof(float), cudaMemcpyDeviceToHost));

            float max_change = 0.0f, max_u = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                max_change = std::max(max_change, std::abs(h_ux_new[i] - h_ux_old[i]));
                max_u = std::max(max_u, std::abs(h_ux_new[i]));
            }
            float rel_change = max_change / (max_u + 1e-12f);

            printf("  step %d: relative change = %.2e\n", step, rel_change);
            fflush(stdout);

            std::copy(h_ux_new.begin(), h_ux_new.end(), h_ux_old.begin());

            if (rel_change < 5e-5f && step > 5000) {
                printf("  CONVERGED at step %d\n", step);
                break;
            }
        }
    }

    // --- Dump z-midplane fields ---
    printf("Dumping data...\n");

    int k_mid = nz / 2;
    std::vector<float> h_temp(num_cells), h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Velocity normalization: lattice -> normalized (u* = u_phys / U0)
    const float U0 = alpha_air / H_physical;
    const float lat_to_phys = dx_phys / dt_phys;

    std::ofstream csv("/home/yzk/LBMProject/scripts/viz/natural_convection_data.csv");
    csv << "ix,iy,x,y,T_K,theta,ux_star,uy_star\n";

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + j * nx + k_mid * nx * ny;
            float x_norm = static_cast<float>(i) / static_cast<float>(nx - 1);
            float y_norm = static_cast<float>(j) / static_cast<float>(ny - 1);

            // Dimensionless temperature theta = (T - T_cold) / (T_hot - T_cold)
            float theta = (h_temp[idx] - T_cold) / (T_hot - T_cold);

            // Normalized velocities
            float ux_star = h_ux[idx] * lat_to_phys / U0;
            float uy_star = h_uy[idx] * lat_to_phys / U0;

            csv << i << "," << j << ","
                << x_norm << "," << y_norm << ","
                << h_temp[idx] << "," << theta << ","
                << ux_star << "," << uy_star << "\n";
        }
    }
    csv.close();

    printf("Data written to scripts/viz/natural_convection_data.csv\n");

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_fz));

    return 0;
}
