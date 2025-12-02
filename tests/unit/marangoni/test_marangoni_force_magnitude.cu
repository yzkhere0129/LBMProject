/**
 * @file test_marangoni_force_magnitude.cu
 * @brief Unit test: Verify Marangoni force magnitude calculation
 *
 * Test validates that:
 * - F = |dσ/dT| × |∇T_tangential| × |∇f|
 * - Force magnitude matches analytical calculation
 * - Expected magnitude for LPBF conditions: ~10^8-10^9 N/m³
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace lbm::physics;

// Test parameters
constexpr int NX = 10;
constexpr int NY = 10;
constexpr int NZ = 10;
constexpr float DX = 1.0e-6f;  // 1 micron grid spacing
constexpr float DSIGMA_DT = -0.00026f;  // Ti6Al4V: N/(m·K)

// Helper function to allocate and initialize device memory
template<typename T>
T* allocateDeviceArray(int size, T init_value = T{}) {
    T* d_ptr;
    cudaMalloc(&d_ptr, size * sizeof(T));

    T* h_ptr = new T[size];
    for (int i = 0; i < size; i++) {
        h_ptr[i] = init_value;
    }
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    delete[] h_ptr;

    return d_ptr;
}

bool testForceMagnitudeSimple() {
    printf("\n=== Test: Marangoni Force Magnitude (Simple Case) ===\n");

    const int num_cells = NX * NY * NZ;

    // Allocate device arrays
    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    // Setup: Flat horizontal interface with temperature gradient in X
    float* h_temperature = new float[num_cells];
    float* h_fill_level = new float[num_cells];
    float3* h_interface_normal = new float3[num_cells];

    const float T_hot = 2500.0f;   // K
    const float T_cold = 1500.0f;  // K
    const int interface_z = NZ / 2;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                // Linear temperature gradient in X
                float x_frac = float(i) / float(NX - 1);
                h_temperature[idx] = T_hot - (T_hot - T_cold) * x_frac;

                // Sharp interface perpendicular to Z
                if (k < interface_z) {
                    h_fill_level[idx] = 1.0f;  // Liquid
                } else if (k == interface_z) {
                    h_fill_level[idx] = 0.5f;  // Interface
                } else {
                    h_fill_level[idx] = 0.0f;  // Gas
                }

                // Normal points in +Z direction
                h_interface_normal[idx] = make_float3(0.0f, 0.0f, 1.0f);
            }
        }
    }

    // Copy to device
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill_level, h_fill_level, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interface_normal, h_interface_normal, num_cells * sizeof(float3), cudaMemcpyHostToDevice);

    // Create Marangoni effect solver
    MarangoniEffect marangoni(NX, NY, NZ, DSIGMA_DT, DX);

    // Compute Marangoni force
    marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                    d_force_x, d_force_y, d_force_z);

    // Copy results back
    float* h_force_x = new float[num_cells];
    float* h_force_y = new float[num_cells];
    float* h_force_z = new float[num_cells];

    cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_y, d_force_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_z, d_force_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analytical calculation for interface cells
    // Temperature gradient: dT/dx = (T_cold - T_hot) / ((NX-1) * DX)
    float grad_T_x = (T_cold - T_hot) / ((NX - 1) * DX);  // K/m
    printf("Temperature gradient: dT/dx = %.3e K/m\n", grad_T_x);

    // VOF gradient at interface: df/dz ≈ (0 - 1) / (2*DX) = -0.5/DX
    float grad_f_z = 0.5f / DX;  // 1/m (magnitude)
    printf("VOF gradient magnitude: |∇f| = %.3e 1/m\n", grad_f_z);

    // Expected force magnitude: F = |dσ/dT| × |∇T_tangential| × |∇f|
    float expected_force_mag = std::abs(DSIGMA_DT) * std::abs(grad_T_x) * grad_f_z;
    printf("Expected force magnitude: F = %.3e N/m³\n", expected_force_mag);

    // Check interface cells
    bool passed = true;
    int cells_checked = 0;
    float avg_force_mag = 0.0f;

    int k = interface_z;
    for (int j = 1; j < NY - 1; j++) {
        for (int i = 1; i < NX - 1; i++) {
            int idx = i + NX * (j + NY * k);

            float fx = h_force_x[idx];
            float fy = h_force_y[idx];
            float fz = h_force_z[idx];
            float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

            if (h_fill_level[idx] > 0.001f && h_fill_level[idx] < 0.999f) {
                cells_checked++;
                avg_force_mag += f_mag;

                // Check if force magnitude is within reasonable range
                float rel_error = std::abs(f_mag - expected_force_mag) / expected_force_mag;

                if (cells_checked <= 5) {  // Print first few
                    printf("Cell (%d,%d,%d): F_computed = %.3e, F_expected = %.3e, error = %.1f%%\n",
                           i, j, k, f_mag, expected_force_mag, rel_error * 100.0f);
                }

                // Allow 50% tolerance due to numerical gradient approximation
                if (rel_error > 0.5f) {
                    if (cells_checked <= 5) {
                        printf("  WARNING: Large error\n");
                    }
                    passed = false;
                }
            }
        }
    }

    if (cells_checked > 0) {
        avg_force_mag /= cells_checked;
        printf("\nChecked %d interface cells\n", cells_checked);
        printf("Average computed force: %.3e N/m³\n", avg_force_mag);
        printf("Expected force:         %.3e N/m³\n", expected_force_mag);
        printf("Relative error:         %.1f%%\n",
               100.0f * std::abs(avg_force_mag - expected_force_mag) / expected_force_mag);
    }

    // Cleanup
    delete[] h_temperature;
    delete[] h_fill_level;
    delete[] h_interface_normal;
    delete[] h_force_x;
    delete[] h_force_y;
    delete[] h_force_z;

    cudaFree(d_temperature);
    cudaFree(d_fill_level);
    cudaFree(d_interface_normal);
    cudaFree(d_force_x);
    cudaFree(d_force_y);
    cudaFree(d_force_z);

    if (passed && cells_checked > 0) {
        printf("PASSED: Force magnitude matches analytical calculation\n");
    } else if (cells_checked == 0) {
        printf("FAILED: No interface cells found\n");
        passed = false;
    } else {
        printf("FAILED: Force magnitude error too large\n");
    }

    return passed;
}

bool testForceMagnitudeLPBF() {
    printf("\n=== Test: Marangoni Force Magnitude (LPBF Conditions) ===\n");

    const int num_cells = NX * NY * NZ;

    // Allocate device arrays
    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    // Setup: LPBF-like conditions with high temperature gradient
    float* h_temperature = new float[num_cells];
    float* h_fill_level = new float[num_cells];
    float3* h_interface_normal = new float3[num_cells];

    const float T_center = 3000.0f;   // K (above melting, laser spot)
    const float T_edge = 1900.0f;     // K (near melting)
    const int interface_z = NZ / 2;

    // Typical LPBF: ∇T ~ 10^7 K/m
    // With DX = 1 μm, temperature drop over domain: ∇T × L = 10^7 × 10e-6 = 100 K
    // Let's create a radial gradient

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                // Radial temperature profile (hot center, cool edges)
                float cx = (i - NX/2.0f) * DX;
                float cy = (j - NY/2.0f) * DX;
                float r = std::sqrt(cx*cx + cy*cy);
                float r_max = std::sqrt((NX/2.0f * DX)*(NX/2.0f * DX) +
                                       (NY/2.0f * DX)*(NY/2.0f * DX));

                h_temperature[idx] = T_center - (T_center - T_edge) * (r / r_max);

                // Interface perpendicular to Z
                if (k < interface_z) {
                    h_fill_level[idx] = 1.0f;
                } else if (k == interface_z) {
                    h_fill_level[idx] = 0.5f;
                } else {
                    h_fill_level[idx] = 0.0f;
                }

                h_interface_normal[idx] = make_float3(0.0f, 0.0f, 1.0f);
            }
        }
    }

    // Copy to device
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill_level, h_fill_level, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interface_normal, h_interface_normal, num_cells * sizeof(float3), cudaMemcpyHostToDevice);

    // Create Marangoni effect solver
    MarangoniEffect marangoni(NX, NY, NZ, DSIGMA_DT, DX);

    // Compute Marangoni force
    marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                    d_force_x, d_force_y, d_force_z);

    // Copy results back
    float* h_force_x = new float[num_cells];
    float* h_force_y = new float[num_cells];
    float* h_force_z = new float[num_cells];

    cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_y, d_force_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_z, d_force_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze results
    float max_force = 0.0f;
    float avg_force = 0.0f;
    int cells_with_force = 0;

    int k = interface_z;
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int idx = i + NX * (j + NY * k);

            float fx = h_force_x[idx];
            float fy = h_force_y[idx];
            float fz = h_force_z[idx];
            float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

            if (h_fill_level[idx] > 0.001f && h_fill_level[idx] < 0.999f) {
                if (f_mag > 1e-6f) {
                    cells_with_force++;
                    avg_force += f_mag;
                    if (f_mag > max_force) {
                        max_force = f_mag;
                    }
                }
            }
        }
    }

    if (cells_with_force > 0) {
        avg_force /= cells_with_force;
    }

    printf("\nResults for LPBF-like conditions:\n");
    printf("Cells with force: %d\n", cells_with_force);
    printf("Maximum force:    %.3e N/m³\n", max_force);
    printf("Average force:    %.3e N/m³\n", avg_force);

    // For LPBF with ∇T ~ 10^7 K/m:
    // F ~ |dσ/dT| × ∇T × |∇f|
    //   ~ 0.00026 N/(m·K) × 10^7 K/m × 10^6 1/m
    //   ~ 2.6 × 10^9 N/m³
    float grad_T_typical = (T_center - T_edge) / ((NX/2.0f) * DX);
    float grad_f = 0.5f / DX;
    float expected_order = std::abs(DSIGMA_DT) * grad_T_typical * grad_f;

    printf("\nEstimated gradient: %.3e K/m\n", grad_T_typical);
    printf("Expected force magnitude: %.3e N/m³\n", expected_order);

    bool passed = true;

    // Check if force is in reasonable range for LPBF (10^8 - 10^10 N/m³)
    if (max_force < 1.0e8f || max_force > 1.0e10f) {
        printf("WARNING: Force magnitude outside expected LPBF range [10^8, 10^10] N/m³\n");
        passed = false;
    }

    // Cleanup
    delete[] h_temperature;
    delete[] h_fill_level;
    delete[] h_interface_normal;
    delete[] h_force_x;
    delete[] h_force_y;
    delete[] h_force_z;

    cudaFree(d_temperature);
    cudaFree(d_fill_level);
    cudaFree(d_interface_normal);
    cudaFree(d_force_x);
    cudaFree(d_force_y);
    cudaFree(d_force_z);

    if (passed && cells_with_force > 0) {
        printf("PASSED: Force magnitude in expected range for LPBF\n");
    } else {
        printf("FAILED: Force magnitude not in expected range\n");
    }

    return passed;
}

int main() {
    printf("========================================\n");
    printf("Marangoni Force Magnitude Test Suite\n");
    printf("========================================\n");

    bool all_passed = true;

    all_passed &= testForceMagnitudeSimple();
    all_passed &= testForceMagnitudeLPBF();

    printf("\n========================================\n");
    if (all_passed) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
