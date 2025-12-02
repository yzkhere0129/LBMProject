/**
 * @file test_collision_kernel_direct.cu
 * @brief TEST LEVEL 1.1: Direct test of collision kernel with constant force
 *
 * Purpose: Test if collisionBGKKernel applies forces correctly at the lowest level
 *
 * Test Strategy:
 *   - Single cell (1x1x1) or small grid
 *   - Initialize f = equilibrium at rest (rho=1, u=0)
 *   - Set force fx = 0.001, fy = fz = 0
 *   - Call collisionBGKKernel ONCE
 *
 * Expected Result:
 *   - PDFs should change
 *   - Momentum = sum(f_q * e_q) should be > 0 in x-direction
 *
 * If FAIL: Kernel doesn't apply force → BUG FOUND IN KERNEL
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TEST LEVEL 1.1: Collision Kernel Direct\n";
    std::cout << "========================================\n\n";

    // Initialize D3Q19 constants
    core::D3Q19::initializeDevice();

    // Small test domain
    const int nx = 4, ny = 4, nz = 4;
    const int num_cells = nx * ny * nz;

    std::cout << "Test setup:\n";
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << " = " << num_cells << " cells\n";
    std::cout << "  Initial: rho=1.0, u=0.0\n";
    std::cout << "  Applied force: fx=0.001, fy=0, fz=0\n";
    std::cout << "  Omega: 1.0\n\n";

    // Allocate device memory
    size_t f_size = num_cells * core::D3Q19::Q * sizeof(float);
    size_t macro_size = num_cells * sizeof(float);

    float *d_f_src, *d_f_dst;
    float *d_rho, *d_ux, *d_uy, *d_uz;

    cudaMalloc(&d_f_src, f_size);
    cudaMalloc(&d_f_dst, f_size);
    cudaMalloc(&d_rho, macro_size);
    cudaMalloc(&d_ux, macro_size);
    cudaMalloc(&d_uy, macro_size);
    cudaMalloc(&d_uz, macro_size);

    // Initialize host data
    std::vector<float> h_f(num_cells * core::D3Q19::Q);
    std::vector<float> h_rho(num_cells, 1.0f);
    std::vector<float> h_ux(num_cells, 0.0f);
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    // Initialize PDFs to equilibrium at rest
    for (int id = 0; id < num_cells; ++id) {
        for (int q = 0; q < core::D3Q19::Q; ++q) {
            h_f[id + q * num_cells] = core::D3Q19::computeEquilibrium(q, 1.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Copy to device
    cudaMemcpy(d_f_src, h_f.data(), f_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho.data(), macro_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ux, h_ux.data(), macro_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), macro_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), macro_size, cudaMemcpyHostToDevice);

    // Store initial PDFs for comparison
    std::vector<float> h_f_initial = h_f;

    // Apply collision kernel with force
    std::cout << "Applying collision kernel...\n";
    float force_x = 0.001f;
    float force_y = 0.0f;
    float force_z = 0.0f;
    float omega = 1.0f;

    dim3 block(4, 4, 4);
    dim3 grid(1, 1, 1);

    physics::fluidBGKCollisionKernel<<<grid, block>>>(
        d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz,
        force_x, force_y, force_z, omega,
        nx, ny, nz
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy back results
    cudaMemcpy(h_f.data(), d_f_dst, f_size, cudaMemcpyDeviceToHost);

    // Analyze results
    std::cout << "\n=== Analysis ===\n";

    // Check if PDFs changed
    bool pdfs_changed = false;
    float max_change = 0.0f;
    for (size_t i = 0; i < h_f.size(); ++i) {
        float change = std::abs(h_f[i] - h_f_initial[i]);
        max_change = std::max(max_change, change);
        if (change > 1e-10f) {
            pdfs_changed = true;
        }
    }

    std::cout << "1. PDFs changed: " << (pdfs_changed ? "YES" : "NO") << "\n";
    std::cout << "   Max change: " << max_change << "\n";

    // Compute velocities from PDFs using computeMacroscopic kernel
    cudaMemcpy(d_f_src, h_f.data(), f_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_cells + block_size - 1) / block_size;

    physics::computeMacroscopicKernel<<<grid_size, block_size>>>(
        d_f_src, d_rho, d_ux, d_uy, d_uz, num_cells
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_rho.data(), d_rho, macro_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux.data(), d_ux, macro_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), d_uy, macro_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz.data(), d_uz, macro_size, cudaMemcpyDeviceToHost);

    // Compute momentum (momentum = rho * velocity)
    std::cout << "\n2. Velocity analysis (first 4 cells):\n";
    std::cout << "   Cell  |  rho   |   ux      |   uy      |   uz\n";
    std::cout << "   ------|--------|-----------|-----------|----------\n";

    float total_px_final = 0.0f;
    for (int id = 0; id < std::min(4, num_cells); ++id) {
        float px = h_rho[id] * h_ux[id];
        float py = h_rho[id] * h_uy[id];
        float pz = h_rho[id] * h_uz[id];

        total_px_final += px;

        std::cout << "   " << std::setw(4) << id << "  |  "
                  << std::setw(6) << std::fixed << std::setprecision(3) << h_rho[id] << " |  "
                  << std::setw(9) << std::scientific << std::setprecision(2) << h_ux[id] << " |  "
                  << std::setw(9) << h_uy[id] << " |  "
                  << std::setw(9) << h_uz[id] << "\n";
    }

    std::cout << "\n3. Total x-momentum (all cells): " << total_px_final << "\n";

    // Verdict
    std::cout << "\n=== VERDICT ===\n";
    bool test_passed = pdfs_changed && (total_px_final > 1e-8f);

    if (test_passed) {
        std::cout << "PASS: Collision kernel applies force correctly\n";
        std::cout << "  - PDFs changed after collision\n";
        std::cout << "  - Momentum increased in x-direction\n";
    } else {
        std::cout << "FAIL: Collision kernel NOT working\n";
        if (!pdfs_changed) {
            std::cout << "  - PDFs did not change!\n";
            std::cout << "  → BUG: Kernel not executing or force term = 0\n";
        }
        if (total_px_final <= 1e-8f) {
            std::cout << "  - No momentum increase detected\n";
            std::cout << "  → BUG: Force not being applied to PDFs\n";
        }
    }

    // Cleanup
    cudaFree(d_f_src);
    cudaFree(d_f_dst);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    return test_passed ? 0 : 1;
}
