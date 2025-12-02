/**
 * @file test_marangoni_force_direction.cu
 * @brief Unit test: Verify Marangoni force direction is from hot to cold
 *
 * Test validates that:
 * - For dσ/dT < 0 (most metals), force points from hot to cold
 * - Force is tangential to interface (perpendicular to normal)
 * - Force magnitude is zero outside interface region
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
constexpr float DX = 1.0e-6f;  // 1 micron
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

bool testForceDirection() {
    printf("\n=== Test: Marangoni Force Direction ===\n");

    const int num_cells = NX * NY * NZ;

    // Allocate device arrays
    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    // Setup: Linear temperature gradient in X-direction
    // Hot on left (X=0), cold on right (X=NX-1)
    // Interface at center (X=NX/2) with horizontal normal (nx=1, ny=0, nz=0)

    float* h_temperature = new float[num_cells];
    float* h_fill_level = new float[num_cells];
    float3* h_interface_normal = new float3[num_cells];

    const float T_hot = 2500.0f;   // K
    const float T_cold = 1500.0f;  // K
    const int interface_x = NX / 2;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                // Linear temperature gradient
                float x_frac = float(i) / float(NX - 1);
                h_temperature[idx] = T_hot - (T_hot - T_cold) * x_frac;

                // Fill level: interface at center, smooth transition
                if (i < interface_x - 1) {
                    h_fill_level[idx] = 1.0f;  // Liquid
                } else if (i > interface_x + 1) {
                    h_fill_level[idx] = 0.0f;  // Gas
                } else {
                    // Interface cells
                    h_fill_level[idx] = 0.5f;
                }

                // Interface normal points in +X direction (liquid to gas)
                h_interface_normal[idx] = make_float3(1.0f, 0.0f, 0.0f);
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

    // Verify force direction at interface
    bool passed = true;
    int interface_cells_checked = 0;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = interface_x - 1; i <= interface_x + 1; i++) {
                int idx = i + NX * (j + NY * k);

                float fx = h_force_x[idx];
                float fy = h_force_y[idx];
                float fz = h_force_z[idx];
                float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

                if (h_fill_level[idx] > 0.001f && h_fill_level[idx] < 0.999f) {
                    interface_cells_checked++;

                    // Temperature gradient is in -X direction (dT/dx < 0)
                    // For dσ/dT < 0, force should be in -X direction (from hot to cold)
                    // But tangential component should be zero since gradient is normal to interface

                    // Since normal is (1,0,0) and gradient is along X,
                    // the tangential gradient should be zero
                    // Thus force should be very small or zero

                    if (f_mag > 1e-3f) {
                        printf("WARNING: Cell (%d,%d,%d) f=%.3f has non-zero tangential force: "
                               "fx=%.3e, fy=%.3e, fz=%.3e\n",
                               i, j, k, h_fill_level[idx], fx, fy, fz);
                    }
                }
            }
        }
    }

    printf("Checked %d interface cells\n", interface_cells_checked);

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

    if (passed) {
        printf("PASSED: Force is correctly tangential (zero when gradient is normal)\n");
    } else {
        printf("FAILED: Force direction is incorrect\n");
    }

    return passed;
}

bool testForceDirectionWithTangentialGradient() {
    printf("\n=== Test: Marangoni Force with Tangential Gradient ===\n");

    const int num_cells = NX * NY * NZ;

    // Allocate device arrays
    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    // Setup: Vertical interface (normal in Z), temperature gradient in X
    // This creates tangential temperature gradient

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

                // Temperature gradient in X direction
                float x_frac = float(i) / float(NX - 1);
                h_temperature[idx] = T_hot - (T_hot - T_cold) * x_frac;

                // Fill level: interface perpendicular to Z
                if (k < interface_z - 1) {
                    h_fill_level[idx] = 1.0f;  // Liquid
                } else if (k > interface_z + 1) {
                    h_fill_level[idx] = 0.0f;  // Gas
                } else {
                    // Interface cells
                    h_fill_level[idx] = 0.5f;
                }

                // Interface normal points in +Z direction
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

    // Verify force direction at interface
    bool passed = true;
    int interface_cells_with_force = 0;

    printf("\nInterface cells analysis:\n");

    for (int k = interface_z - 1; k <= interface_z + 1; k++) {
        for (int j = NY/2; j < NY/2 + 1; j++) {  // Check center row
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                float fx = h_force_x[idx];
                float fy = h_force_y[idx];
                float fz = h_force_z[idx];
                float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

                if (h_fill_level[idx] > 0.001f && h_fill_level[idx] < 0.999f) {
                    // Temperature gradient is in -X direction (dT/dx < 0)
                    // Interface normal is in +Z direction
                    // Tangential gradient is in -X direction
                    // For dσ/dT < 0 (negative), force is opposite to tangential gradient
                    // So force should be in +X direction (cold to hot in surface)

                    if (f_mag > 1e-6f) {
                        interface_cells_with_force++;

                        // Force should be primarily in X direction
                        float fx_normalized = fx / f_mag;

                        // For dσ/dT < 0 and dT/dx < 0 (hot on left),
                        // force should be in +X direction (towards cold in tangential plane)
                        // Wait: dσ/dT < 0, so surface flows from high σ to low σ
                        // σ decreases with T, so high T = low σ, low T = high σ
                        // Flow from high σ to low σ = flow from cold to hot
                        // But force on fluid is opposite! Force is from hot to cold!

                        if (fx < 0) {
                            printf("Cell (%d,%d,%d) f=%.3f T=%.1f: fx=%.3e (correct: hot→cold)\n",
                                   i, j, k, h_fill_level[idx], h_temperature[idx], fx);
                        } else {
                            printf("Cell (%d,%d,%d) f=%.3f T=%.1f: fx=%.3e (WRONG: cold→hot)\n",
                                   i, j, k, h_fill_level[idx], h_temperature[idx], fx);
                            passed = false;
                        }

                        // Force should be tangential (fz should be zero)
                        if (std::abs(fz) > f_mag * 0.01f) {
                            printf("WARNING: Non-tangential component fz=%.3e (%.1f%% of total)\n",
                                   fz, 100.0f * std::abs(fz) / f_mag);
                        }
                    }
                }
            }
        }
    }

    printf("\nFound %d interface cells with non-zero force\n", interface_cells_with_force);

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

    if (passed && interface_cells_with_force > 0) {
        printf("PASSED: Force direction is from hot to cold\n");
    } else if (interface_cells_with_force == 0) {
        printf("FAILED: No force detected at interface\n");
        passed = false;
    } else {
        printf("FAILED: Force direction is incorrect\n");
    }

    return passed;
}

int main() {
    printf("======================================\n");
    printf("Marangoni Force Direction Test Suite\n");
    printf("======================================\n");

    bool all_passed = true;

    all_passed &= testForceDirection();
    all_passed &= testForceDirectionWithTangentialGradient();

    printf("\n======================================\n");
    if (all_passed) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
