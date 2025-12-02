/**
 * @file test_buoyancy_force_check.cu
 * @brief TEST LEVEL 3.1: Verify buoyancy force computation
 *
 * Purpose: Check if computeBuoyancyForce produces non-zero forces
 *
 * Test Strategy:
 *   - Create temperature field with gradient
 *   - Call computeBuoyancyForce
 *   - Check that forces are non-zero where T != T_ref
 *
 * Expected Result:
 *   - Should see non-zero forces in hot regions
 *   - Force magnitude should scale with (T - T_ref)
 *
 * If FAIL: computeBuoyancyForce returns zeros → BUG IN BUOYANCY KERNEL
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TEST LEVEL 3.1: Buoyancy Force Check\n";
    std::cout << "========================================\n\n";

    core::D3Q19::initializeDevice();

    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;

    std::cout << "Test setup:\n";
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << "\n";
    std::cout << "  Temperature: 300K - 400K gradient in y\n";
    std::cout << "  T_ref: 350K\n";
    std::cout << "  Beta: 1e-4 K^-1\n";
    std::cout << "  Gravity: 10 m/s² in +y direction\n";
    std::cout << "  Density: 1000 kg/m³\n\n";

    // Create solver
    float nu = 0.1f;
    float rho0 = 1000.0f;
    physics::FluidLBM solver(nx, ny, nz, nu, rho0);
    solver.initialize();

    // Create temperature field with gradient
    float *d_temperature, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    std::vector<float> h_temperature(num_cells);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                // Linear temperature gradient in y: 300K to 400K
                h_temperature[id] = 300.0f + 100.0f * static_cast<float>(iy) / (ny - 1);
            }
        }
    }

    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    // Compute buoyancy force
    float T_ref = 350.0f;
    float beta = 1e-4f;
    float g = 10.0f;

    std::cout << "Computing buoyancy force...\n";
    solver.computeBuoyancyForce(
        d_temperature, T_ref, beta,
        0.0f, g, 0.0f,  // Gravity in +y
        d_fx, d_fy, d_fz
    );

    // Copy back
    std::vector<float> h_fx(num_cells);
    std::vector<float> h_fy(num_cells);
    std::vector<float> h_fz(num_cells);

    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze results
    std::cout << "\n=== Analysis ===\n";

    // Find min/max forces
    auto minmax_fy = std::minmax_element(h_fy.begin(), h_fy.end());
    float min_fy = *minmax_fy.first;
    float max_fy = *minmax_fy.second;

    std::cout << "Force range in y-direction:\n";
    std::cout << "  min(fy): " << min_fy << " N/m³\n";
    std::cout << "  max(fy): " << max_fy << " N/m³\n";

    // Check first few cells
    std::cout << "\nSample forces (first 5 y-layers, center x-z):\n";
    std::cout << "   y  |   T [K]   |  dT [K]  |   fy [N/m³]  |  Expected  | Match\n";
    std::cout << "------|-----------|----------|--------------|------------|------\n";

    int ix_center = nx / 2;
    int iz_center = nz / 2;
    bool all_correct = true;

    for (int iy = 0; iy < std::min(5, ny); ++iy) {
        int id = ix_center + iy * nx + iz_center * nx * ny;
        float T = h_temperature[id];
        float dT = T - T_ref;
        float fy = h_fy[id];

        // Expected: F = rho0 * beta * (T - T_ref) * g
        float expected_fy = rho0 * beta * dT * g;

        bool match = std::abs(fy - expected_fy) < 1e-3f;
        all_correct &= match;

        std::cout << "  " << std::setw(3) << iy << "  | "
                  << std::setw(9) << std::fixed << std::setprecision(1) << T << " | "
                  << std::setw(8) << std::setprecision(1) << dT << " | "
                  << std::setw(12) << std::setprecision(3) << fy << " | "
                  << std::setw(10) << expected_fy << " | "
                  << (match ? "YES" : "NO") << "\n";
    }

    // Check fx and fz should be zero
    float max_fx = *std::max_element(h_fx.begin(), h_fx.end(),
                                     [](float a, float b) { return std::abs(a) < std::abs(b); });
    float max_fz = *std::max_element(h_fz.begin(), h_fz.end(),
                                     [](float a, float b) { return std::abs(a) < std::abs(b); });

    std::cout << "\nOther components:\n";
    std::cout << "  max|fx|: " << std::abs(max_fx) << " (should be ~0)\n";
    std::cout << "  max|fz|: " << std::abs(max_fz) << " (should be ~0)\n";

    bool fx_zero = std::abs(max_fx) < 1e-5f;
    bool fz_zero = std::abs(max_fz) < 1e-5f;

    // Count non-zero forces
    int nonzero_count = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (std::abs(h_fy[i]) > 1e-6f) nonzero_count++;
    }

    std::cout << "\nNon-zero forces: " << nonzero_count << " / " << num_cells
              << " (" << (100.0f * nonzero_count / num_cells) << "%)\n";

    // Verdict
    std::cout << "\n=== VERDICT ===\n";
    bool test_passed = all_correct && fx_zero && fz_zero && (nonzero_count > 0);

    if (test_passed) {
        std::cout << "PASS: Buoyancy force computed correctly\n";
        std::cout << "  - Forces match analytical formula\n";
        std::cout << "  - Non-zero forces where T != T_ref\n";
        std::cout << "  - Correct directional components\n";
    } else {
        std::cout << "FAIL: Buoyancy force has issues\n";
        if (!all_correct) {
            std::cout << "  → BUG: Force values don't match expected\n";
        }
        if (nonzero_count == 0) {
            std::cout << "  → BUG: All forces are zero!\n";
        }
        if (!fx_zero || !fz_zero) {
            std::cout << "  → BUG: Force in wrong direction\n";
        }
    }

    // Cleanup
    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return test_passed ? 0 : 1;
}
