/**
 * @file test_debug_force_kernel.cu
 * @brief Direct test of the collision kernel with forces
 *
 * This test directly inspects PDF values before and after collision
 * to verify that forces are actually modifying the distribution.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

__global__ void inspectPDFsKernel(const float* f_src, float* pdf_sum, int num_cells) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    float sum = 0.0f;
    for (int q = 0; q < 19; q++) {
        sum += f_src[id + q * num_cells];
    }
    pdf_sum[id] = sum;
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "  Debug: Force Kernel Direct Test\n";
    std::cout << "==============================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;
    float rho0 = 1.0f;

    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    // Initialize at rest
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Create spatially-varying force field
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    // Create large force to make effect visible
    float* h_fx = new float[n_cells];
    for (int i = 0; i < n_cells; i++) {
        h_fx[i] = 0.001f;  // Moderate force in lattice units
    }

    cudaMemcpy(d_fx, h_fx, n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fy, 0, n_cells * sizeof(float));
    cudaMemset(d_fz, 0, n_cells * sizeof(float));

    std::cout << "Test Setup:\n";
    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << "\n";
    std::cout << "  Applied force: fx = 0.001 (lattice units)\n";
    std::cout << "  Boundary: all periodic\n\n";

    // Run ONE collision+streaming cycle and check PDFs
    std::cout << "Running single LBM cycle:\n";
    std::cout << "  1. computeMacroscopic()\n";
    std::cout << "  2. collisionBGK(forces)\n";
    std::cout << "  3. streaming()\n\n";

    solver.computeMacroscopic();
    solver.collisionBGK(d_fx, d_fy, d_fz);
    solver.streaming();
    solver.computeMacroscopic();

    // Copy velocity
    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];

    solver.copyVelocityToHost(h_ux, h_uy, h_uz);

    // Check velocity
    float u_max = 0.0f;
    float u_sum = 0.0f;
    int nonzero_count = 0;

    for (int i = 0; i < n_cells; i++) {
        float u_mag = sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        u_max = std::max(u_max, u_mag);
        u_sum += u_mag;
        if (u_mag > 1e-10f) nonzero_count++;
    }

    float u_avg = u_sum / n_cells;

    std::cout << "Results after 1 time step:\n";
    std::cout << "  Max velocity: " << std::scientific << u_max << "\n";
    std::cout << "  Avg velocity: " << u_avg << "\n";
    std::cout << "  Non-zero cells: " << nonzero_count << " / " << n_cells << "\n\n";

    if (u_max < 1e-10f) {
        std::cout << "  ❌ FAIL: Velocity is ZERO after applying force!\n";
        std::cout << "\n";
        std::cout << "  This confirms the collision kernel is NOT applying forces.\n";
        std::cout << "  Possible causes:\n";
        std::cout << "  1. Force parameters not passed to kernel\n";
        std::cout << "  2. Force term computation is wrong\n";
        std::cout << "  3. Kernel launch configuration error\n";
    } else {
        std::cout << "  ✅ PASS: Force produced non-zero velocity!\n";
        std::cout << "\n";
        std::cout << "  Expected velocity after 1 step with f=0.001:\n";
        std::cout << "  Δu ~ f·Δt ~ 0.001 * 1 = 0.001\n";
        std::cout << "  Actual: " << u_max << "\n";

        if (std::abs(u_max - 0.001f) < 0.0005f) {
            std::cout << "  ✅ Magnitude is correct!\n";
        } else if (u_max < 0.0001f) {
            std::cout << "  ⚠️  Magnitude is too small (force may be weak)\n";
        } else {
            std::cout << "  ⚠️  Magnitude is unexpected\n";
        }
    }

    // Cleanup
    delete[] h_fx;
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "\n==============================================\n";
    return 0;
}
