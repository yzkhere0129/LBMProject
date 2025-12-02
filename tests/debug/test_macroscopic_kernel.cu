/**
 * @file test_macroscopic_kernel.cu
 * @brief TEST LEVEL 1.2: Test computeMacroscopic kernel
 *
 * Purpose: Verify that computeMacroscopicKernel extracts velocity correctly
 *
 * Test Strategy:
 *   - Manually set PDFs with known momentum
 *   - Call computeMacroscopicKernel
 *   - Check if extracted velocity matches expected
 *
 * Expected Result:
 *   - If we set f[1] (ex=+1) higher than f[2] (ex=-1), we should get ux > 0
 *
 * If FAIL: Velocity extraction is broken → BUG FOUND IN computeMacroscopicKernel
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
    std::cout << "TEST LEVEL 1.2: Macroscopic Kernel\n";
    std::cout << "========================================\n\n";

    // Initialize D3Q19 constants
    core::D3Q19::initializeDevice();

    const int num_cells = 8;

    std::cout << "Test setup:\n";
    std::cout << "  Cells: " << num_cells << "\n";
    std::cout << "  Strategy: Set PDFs with known momentum\n";
    std::cout << "  Expected: Velocity should match momentum/density\n\n";

    // Allocate device memory
    size_t f_size = num_cells * core::D3Q19::Q * sizeof(float);
    size_t macro_size = num_cells * sizeof(float);

    float *d_f, *d_rho, *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_f, f_size);
    cudaMalloc(&d_rho, macro_size);
    cudaMalloc(&d_ux, macro_size);
    cudaMalloc(&d_uy, macro_size);
    cudaMalloc(&d_uz, macro_size);

    std::cout << "Using D3Q19 lattice (19 velocities)\n\n";

    // Create test cases
    std::vector<float> h_f(num_cells * core::D3Q19::Q, 0.0f);

    std::cout << "Test cases:\n";

    // Case 1: Equilibrium at rest (rho=1, u=0)
    int id = 0;
    for (int q = 0; q < core::D3Q19::Q; ++q) {
        h_f[id + q * num_cells] = core::D3Q19::computeEquilibrium(q, 1.0f, 0.0f, 0.0f, 0.0f);
    }
    std::cout << "  Cell 0: Equilibrium at rest (rho=1, u=0)\n";

    // Case 2: Equilibrium with velocity ux=0.05
    id = 1;
    for (int q = 0; q < core::D3Q19::Q; ++q) {
        h_f[id + q * num_cells] = core::D3Q19::computeEquilibrium(q, 1.0f, 0.05f, 0.0f, 0.0f);
    }
    std::cout << "  Cell 1: Equilibrium with ux=0.05\n";

    // Case 3: Equilibrium with velocity ux=0.1
    id = 2;
    for (int q = 0; q < core::D3Q19::Q; ++q) {
        h_f[id + q * num_cells] = core::D3Q19::computeEquilibrium(q, 1.0f, 0.1f, 0.0f, 0.0f);
    }
    std::cout << "  Cell 2: Equilibrium with ux=0.1\n";

    // Case 4: Equilibrium with velocity uy=0.1 (y-direction)
    id = 3;
    for (int q = 0; q < core::D3Q19::Q; ++q) {
        h_f[id + q * num_cells] = core::D3Q19::computeEquilibrium(q, 1.0f, 0.0f, 0.1f, 0.0f);
    }
    std::cout << "  Cell 3: Equilibrium with uy=0.1\n";

    std::cout << "\n";

    // Copy to device
    cudaMemcpy(d_f, h_f.data(), f_size, cudaMemcpyHostToDevice);

    // Call computeMacroscopicKernel
    std::cout << "Calling computeMacroscopicKernel...\n";
    int block_size = 256;
    int grid_size = (num_cells + block_size - 1) / block_size;

    physics::computeMacroscopicKernel<<<grid_size, block_size>>>(
        d_f, d_rho, d_ux, d_uy, d_uz, num_cells
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy back results
    std::vector<float> h_rho(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    cudaMemcpy(h_rho.data(), d_rho, macro_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux.data(), d_ux, macro_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), d_uy, macro_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz.data(), d_uz, macro_size, cudaMemcpyDeviceToHost);

    // Compute expected values manually
    std::cout << "\n=== Results ===\n";
    std::cout << "Cell |  rho   |   ux   |   uy   |   uz   | Status\n";
    std::cout << "-----|--------|--------|--------|--------|--------\n";

    bool all_passed = true;

    // Expected values based on equilibrium initialization
    float expected_ux[] = {0.0f, 0.05f, 0.1f, 0.0f};
    float expected_uy[] = {0.0f, 0.0f, 0.0f, 0.1f};
    float expected_uz[] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4; ++i) {
        float expected_rho = 1.0f;

        bool rho_ok = std::abs(h_rho[i] - expected_rho) < 1e-5f;
        bool ux_ok = std::abs(h_ux[i] - expected_ux[i]) < 1e-5f;
        bool uy_ok = std::abs(h_uy[i] - expected_uy[i]) < 1e-5f;
        bool uz_ok = std::abs(h_uz[i] - expected_uz[i]) < 1e-5f;

        bool passed = rho_ok && ux_ok && uy_ok && uz_ok;
        all_passed &= passed;

        std::cout << " " << i << "   | "
                  << std::setw(6) << std::fixed << std::setprecision(3) << h_rho[i] << " | "
                  << std::setw(6) << h_ux[i] << " | "
                  << std::setw(6) << h_uy[i] << " | "
                  << std::setw(6) << h_uz[i] << " | "
                  << (passed ? "PASS" : "FAIL") << "\n";

        if (!passed) {
            std::cout << "     Expected: rho=" << expected_rho
                      << " ux=" << expected_ux[i]
                      << " uy=" << expected_uy[i]
                      << " uz=" << expected_uz[i] << "\n";
        }
    }

    // Special check: Cell 1 and 2 should have ux > 0
    bool cell1_positive = h_ux[1] > 0.0f;
    bool cell2_positive = h_ux[2] > 0.0f;
    std::cout << "\nSpecial checks:\n";
    std::cout << "  Cell 1 ux = " << h_ux[1] << " (expect 0.05)\n";
    std::cout << "  Cell 2 ux = " << h_ux[2] << " (expect 0.1)\n";
    std::cout << "  Both positive? " << (cell1_positive && cell2_positive ? "YES" : "NO") << "\n";

    // Verdict
    std::cout << "\n=== VERDICT ===\n";
    if (all_passed && cell1_positive && cell2_positive) {
        std::cout << "PASS: computeMacroscopicKernel works correctly\n";
        std::cout << "  - Density computed correctly\n";
        std::cout << "  - Velocity extracted correctly\n";
        std::cout << "  - Momentum conservation verified\n";
    } else {
        std::cout << "FAIL: computeMacroscopicKernel has issues\n";
        if (!all_passed) {
            std::cout << "  → BUG: Computed values don't match expected\n";
        }
        if (!cell1_positive || !cell2_positive) {
            std::cout << "  → BUG: Failed to detect positive momentum\n";
        }
    }

    // Cleanup
    cudaFree(d_f);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    return (all_passed && cell1_positive && cell2_positive) ? 0 : 1;
}
