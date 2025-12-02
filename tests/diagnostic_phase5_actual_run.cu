/**
 * @file diagnostic_phase5_actual_run.cu
 * @brief Simplified Phase 5 to check actual force values AFTER scaling
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

__global__ void scaleForceArrayKernel(
    float* fx, float* fy, float* fz,
    float scale, int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    fx[id] *= scale;
    fy[id] *= scale;
    fz[id] *= scale;
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << " Diagnostic: Phase 5 Actual Forces\n";
    std::cout << "========================================\n\n";

    // Same parameters as Phase 5
    const int nx = 80, ny = 80, nz = 40;
    const float dx = 2e-6f;
    const float dy = 2e-6f;
    const float dz = 2e-6f;
    const float dt = 5e-10f;
    int num_cells = nx * ny * nz;

    physics::MaterialProperties ti64 = physics::MaterialDatabase::getTi6Al4V();
    float beta_thermal = 9.0e-6f;
    float T_ref = 0.5f * (ti64.T_solidus + ti64.T_liquidus);
    float g = 9.81f;
    float darcy_constant = 1e5f;

    // Initialize solvers
    core::D3Q19::initializeDevice();

    float alpha_thermal = ti64.getThermalDiffusivity(300.0f);
    physics::ThermalLBM thermal(nx, ny, nz, ti64, alpha_thermal, true);
    thermal.initialize(300.0f);

    float nu_lattice = 0.15f;
    physics::FluidLBM fluid(nx, ny, nz, nu_lattice, ti64.rho_liquid);
    fluid.initialize(ti64.rho_liquid, 0.0f, 0.0f, 0.0f);

    // Allocate force arrays
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    // Manually heat a region to create temperature gradient
    std::vector<float> h_temp(num_cells);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                // Hot spot in center
                int dx_val = ix - nx/2;
                int dy_val = iy - ny/2;
                int dz_val = iz - nz/2;
                float dist = sqrtf(dx_val*dx_val + dy_val*dy_val + dz_val*dz_val);
                if (dist < 15.0f) {
                    h_temp[id] = 2500.0f;  // Very hot
                } else {
                    h_temp[id] = 300.0f;   // Cold
                }
            }
        }
    }
    cudaMemcpy(const_cast<float*>(thermal.getTemperature()), h_temp.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Setup:\n";
    std::cout << "  Grid: " << nx << " × " << ny << " × " << nz << "\n";
    std::cout << "  T_ref: " << T_ref << " K\n";
    std::cout << "  Beta: " << beta_thermal << " 1/K\n";
    std::cout << "  Gravity: " << g << " m/s²\n";
    std::cout << "  dt: " << dt << " s\n";
    std::cout << "  dx: " << dx << " m\n\n";

    // Compute buoyancy force (in physical units)
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    fluid.computeBuoyancyForce(
        thermal.getTemperature(), T_ref, beta_thermal,
        0.0f, g, 0.0f,
        d_fx, d_fy, d_fz
    );

    // Check buoyancy forces BEFORE scaling
    std::vector<float> h_fy_before(num_cells);
    cudaMemcpy(h_fy_before.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float fy_min_before = h_fy_before[0], fy_max_before = h_fy_before[0], fy_avg_before = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        fy_min_before = fminf(fy_min_before, h_fy_before[i]);
        fy_max_before = fmaxf(fy_max_before, h_fy_before[i]);
        fy_avg_before += h_fy_before[i];
    }
    fy_avg_before /= num_cells;

    std::cout << "Buoyancy forces BEFORE scaling (physical units [N/m³]):\n";
    std::cout << "  Fy: min=" << std::scientific << fy_min_before
              << " avg=" << fy_avg_before
              << " max=" << fy_max_before << "\n\n";

    // Apply Darcy damping
    fluid.applyDarcyDamping(
        thermal.getLiquidFraction(), darcy_constant,
        d_fx, d_fy, d_fz
    );

    cudaMemcpy(h_fy_before.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    fy_avg_before = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        fy_avg_before += h_fy_before[i];
    }
    fy_avg_before /= num_cells;

    std::cout << "After Darcy damping:\n";
    std::cout << "  Fy avg: " << std::scientific << fy_avg_before << " N/m³\n\n";

    // Apply Phase 5 scaling
    float force_conversion = (dt * dt) / dx;
    float target_force_magnitude = 1e-4f;
    float estimated_phys_force = 200.0f;
    float force_scale = target_force_magnitude / (estimated_phys_force * force_conversion);

    std::cout << "Force scaling parameters:\n";
    std::cout << "  force_conversion (dt²/dx): " << std::scientific << force_conversion << "\n";
    std::cout << "  target_force_magnitude: " << target_force_magnitude << "\n";
    std::cout << "  estimated_phys_force: " << estimated_phys_force << " N/m³\n";
    std::cout << "  force_scale: " << force_scale << "\n\n";

    int block_size = 256;
    int grid_size = (num_cells + block_size - 1) / block_size;
    scaleForceArrayKernel<<<grid_size, block_size>>>(
        d_fx, d_fy, d_fz, force_scale, num_cells
    );
    cudaDeviceSynchronize();

    // Check forces AFTER scaling
    std::vector<float> h_fy_after(num_cells);
    cudaMemcpy(h_fy_after.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float fy_min_after = h_fy_after[0], fy_max_after = h_fy_after[0], fy_avg_after = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        fy_min_after = fminf(fy_min_after, h_fy_after[i]);
        fy_max_after = fmaxf(fy_max_after, h_fy_after[i]);
        fy_avg_after += h_fy_after[i];
    }
    fy_avg_after /= num_cells;

    std::cout << "Forces AFTER scaling (lattice units):\n";
    std::cout << "  Fy: min=" << std::scientific << fy_min_after
              << " avg=" << fy_avg_after
              << " max=" << fy_max_after << "\n\n";

    // Now apply to fluid for 100 steps
    std::cout << "Running 100 fluid steps with these forces...\n";
    for (int step = 0; step < 100; ++step) {
        fluid.computeMacroscopic();
        fluid.collisionBGK(d_fx, d_fy, d_fz);
        fluid.streaming();
    }
    fluid.computeMacroscopic();

    // Check velocities
    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    float ux_max = 0.0f, uy_max = 0.0f, uz_max = 0.0f;
    float uy_avg = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        ux_max = fmaxf(ux_max, fabsf(h_ux[i]));
        uy_max = fmaxf(uy_max, fabsf(h_uy[i]));
        uz_max = fmaxf(uz_max, fabsf(h_uz[i]));
        uy_avg += h_uy[i];
    }
    uy_avg /= num_cells;

    std::cout << "\nVelocities after 100 steps:\n";
    std::cout << "  max |ux|: " << std::scientific << ux_max << "\n";
    std::cout << "  max |uy|: " << std::scientific << uy_max << "\n";
    std::cout << "  max |uz|: " << std::scientific << uz_max << "\n";
    std::cout << "  avg uy: " << std::scientific << uy_avg << "\n\n";

    if (uy_max > 1e-6) {
        std::cout << "✅ SUCCESS: Non-zero velocities detected!\n";
    } else {
        std::cout << "❌ FAILURE: Velocities are still zero\n";
    }

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}
