/**
 * @file test_buoyancy_integration.cu
 * @brief Unit test for buoyancy force computation in isolation
 *
 * Purpose: Validate that computeBuoyancyForce() produces correct force magnitudes
 * and directions before full integration testing.
 *
 * Test Strategy:
 * - Create simple temperature field (cold bottom, hot top)
 * - Compute buoyancy force using FluidLBM::computeBuoyancyForce()
 * - Validate force magnitudes against analytical expectations
 * - Verify force directions (hot should rise, cold should sink)
 *
 * Physics:
 * F_buoyancy = ρ₀ · β · (T - T_ref) · g
 *
 * For Ti6Al4V:
 * - ρ₀ = 4110 kg/m³
 * - β = 1.5e-5 K⁻¹
 * - T_ref = 1923 K (melting point)
 * - g = -9.81 m/s² (downward in z)
 *
 * Expected:
 * - Bottom layer (T=300 K, ΔT=-1623 K): F_z ≈ +1.0e6 N/m³ (upward buoyancy)
 * - Top layer (T=3000 K, ΔT=+1077 K): F_z ≈ -6.5e5 N/m³ (downward)
 */

#include "physics/fluid_lbm.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace lbm::physics;

int main() {
    std::cout << "==========================================================\n";
    std::cout << "  BUOYANCY FORCE INTEGRATION TEST\n";
    std::cout << "==========================================================\n\n";

    // Test parameters
    const int nx = 10, ny = 10, nz = 10;
    const int n = nx * ny * nz;
    const float dx = 2e-6f;  // 2 μm lattice spacing

    // Ti6Al4V parameters
    const float rho0 = 4110.0f;          // Liquid density [kg/m³]
    const float T_ref = 1923.0f;         // Melting point [K]
    const float beta = 1.5e-5f;          // Thermal expansion [1/K]
    const float g_x = 0.0f;
    const float g_y = 0.0f;
    const float g_z = -9.81f;            // Gravity [m/s²] (downward)
    const float nu = 0.0333f;            // Kinematic viscosity (lattice units)

    std::cout << "Test Configuration:\n";
    std::cout << "  Domain: " << nx << " × " << ny << " × " << nz << "\n";
    std::cout << "  dx: " << dx << " m\n";
    std::cout << "  ρ₀: " << rho0 << " kg/m³\n";
    std::cout << "  β: " << beta << " K⁻¹\n";
    std::cout << "  g: (" << g_x << ", " << g_y << ", " << g_z << ") m/s²\n";
    std::cout << "  T_ref: " << T_ref << " K\n\n";

    // Create temperature field: linear gradient from cold bottom to hot top
    std::vector<float> h_temp(n);
    const float T_bottom = 300.0f;       // Ambient temperature
    const float T_top = 3000.0f;         // Above melting point

    for (int k = 0; k < nz; ++k) {
        float T = T_bottom + (T_top - T_bottom) * (float(k) / (nz - 1));
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                h_temp[idx] = T;
            }
        }
    }

    std::cout << "Temperature Field Setup:\n";
    std::cout << "  Bottom (z=0): T = " << T_bottom << " K (ΔT = " << (T_bottom - T_ref) << " K)\n";
    std::cout << "  Top (z=" << nz-1 << "): T = " << T_top << " K (ΔT = " << (T_top - T_ref) << " K)\n\n";

    // Analytical expected forces
    float dT_bottom = T_bottom - T_ref;  // -1623 K
    float dT_top = T_top - T_ref;        // +1077 K
    float F_bottom_expected = rho0 * beta * dT_bottom * g_z;  // Should be positive (upward)
    float F_top_expected = rho0 * beta * dT_top * g_z;        // Should be negative (downward)

    std::cout << "Analytical Expectations:\n";
    std::cout << "  F_z(bottom): " << F_bottom_expected << " N/m³ (cold sinks → upward buoyancy)\n";
    std::cout << "  F_z(top):    " << F_top_expected << " N/m³ (hot rises → downward relative force)\n\n";

    // Initialize FluidLBM
    std::cout << "Initializing FluidLBM...\n";
    FluidLBM fluid(nx, ny, nz, nu, rho0);
    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Allocate device memory for forces
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n * sizeof(float));
    cudaMalloc(&d_fy, n * sizeof(float));
    cudaMalloc(&d_fz, n * sizeof(float));
    cudaMemset(d_fx, 0, n * sizeof(float));
    cudaMemset(d_fy, 0, n * sizeof(float));
    cudaMemset(d_fz, 0, n * sizeof(float));

    // Copy temperature to device
    float* d_temp;
    cudaMalloc(&d_temp, n * sizeof(float));
    cudaMemcpy(d_temp, h_temp.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Compute buoyancy force
    std::cout << "Computing buoyancy forces...\n";
    fluid.computeBuoyancyForce(
        d_temp, T_ref, beta,
        g_x, g_y, g_z,
        d_fx, d_fy, d_fz
    );
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "✗ CUDA Error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Retrieve forces
    std::vector<float> h_fx(n), h_fy(n), h_fz(n);
    cudaMemcpy(h_fx.data(), d_fx, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Sample points for validation
    int idx_bottom = nx/2 + ny/2*nx + 0*nx*ny;      // Center of bottom layer
    int idx_middle = nx/2 + ny/2*nx + (nz/2)*nx*ny; // Center of middle layer
    int idx_top = nx/2 + ny/2*nx + (nz-1)*nx*ny;    // Center of top layer

    float F_bottom_z = h_fz[idx_bottom];
    float F_middle_z = h_fz[idx_middle];
    float F_top_z = h_fz[idx_top];

    std::cout << "\nComputed Forces (z-component):\n";
    std::cout << "  Bottom layer: F_z = " << F_bottom_z << " N/m³\n";
    std::cout << "  Middle layer: F_z = " << F_middle_z << " N/m³\n";
    std::cout << "  Top layer:    F_z = " << F_top_z << " N/m³\n\n";

    // Check for NaN/Inf
    bool has_nan = false;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(h_fx[i]) || std::isnan(h_fy[i]) || std::isnan(h_fz[i]) ||
            std::isinf(h_fx[i]) || std::isinf(h_fy[i]) || std::isinf(h_fz[i])) {
            has_nan = true;
            break;
        }
    }

    if (has_nan) {
        std::cerr << "✗ FAIL: NaN or Inf detected in force field\n";
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_temp);
        return 1;
    }
    std::cout << "✓ PASS: No NaN/Inf values\n";

    // Check horizontal forces should be zero (no horizontal gravity)
    float max_fx = *std::max_element(h_fx.begin(), h_fx.end());
    float max_fy = *std::max_element(h_fy.begin(), h_fy.end());

    if (std::abs(max_fx) > 1e-10 || std::abs(max_fy) > 1e-10) {
        std::cerr << "✗ FAIL: Non-zero horizontal forces (F_x=" << max_fx
                  << ", F_y=" << max_fy << ")\n";
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_temp);
        return 1;
    }
    std::cout << "✓ PASS: Zero horizontal forces (gravity only in z-direction)\n";

    // Validate force magnitudes (within 50% tolerance)
    bool magnitude_pass = true;
    std::cout << "\nValidation (±50% tolerance):\n";

    // Bottom layer check
    float error_bottom = std::abs(F_bottom_z - F_bottom_expected) / std::abs(F_bottom_expected);
    std::cout << "  Bottom: Computed=" << F_bottom_z << " N/m³, Expected=" << F_bottom_expected
              << " N/m³ (error=" << error_bottom*100 << "%)\n";
    if (error_bottom > 0.5) {
        std::cerr << "    ✗ FAIL: Error > 50%\n";
        magnitude_pass = false;
    } else {
        std::cout << "    ✓ PASS\n";
    }

    // Top layer check
    float error_top = std::abs(F_top_z - F_top_expected) / std::abs(F_top_expected);
    std::cout << "  Top: Computed=" << F_top_z << " N/m³, Expected=" << F_top_expected
              << " N/m³ (error=" << error_top*100 << "%)\n";
    if (error_top > 0.5) {
        std::cerr << "    ✗ FAIL: Error > 50%\n";
        magnitude_pass = false;
    } else {
        std::cout << "    ✓ PASS\n";
    }

    if (!magnitude_pass) {
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_temp);
        return 1;
    }

    // Validate force direction (hot rises, cold sinks)
    std::cout << "\nForce Direction Check:\n";
    bool direction_pass = true;

    if (F_bottom_z <= 0.0f) {
        std::cerr << "  ✗ FAIL: Bottom (cold) should have upward buoyancy (F_z > 0)\n";
        direction_pass = false;
    } else {
        std::cout << "  ✓ PASS: Bottom (cold) has upward buoyancy\n";
    }

    if (F_top_z >= 0.0f) {
        std::cerr << "  ✗ FAIL: Top (hot) should have downward relative force (F_z < 0)\n";
        direction_pass = false;
    } else {
        std::cout << "  ✓ PASS: Top (hot) has downward relative force\n";
    }

    if (!direction_pass) {
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_temp);
        return 1;
    }

    // Cleanup
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
    cudaFree(d_temp);

    std::cout << "\n==========================================================\n";
    std::cout << "  ✓ ALL TESTS PASSED\n";
    std::cout << "==========================================================\n";
    std::cout << "\nSummary:\n";
    std::cout << "  - Force computation: Correct\n";
    std::cout << "  - Force magnitudes: Within tolerance\n";
    std::cout << "  - Force directions: Physically correct\n";
    std::cout << "  - No numerical errors: No NaN/Inf\n";
    std::cout << "\nBuoyancy kernel validated. Ready for Stage 3 integration test.\n";

    return 0;
}
