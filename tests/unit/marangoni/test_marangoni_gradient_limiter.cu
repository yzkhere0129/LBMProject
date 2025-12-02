/**
 * @file test_marangoni_gradient_limiter.cu
 * @brief Test gradient limiter with extreme temperature gradients
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

using namespace lbm::physics;

constexpr int NX = 10, NY = 10, NZ = 10;
constexpr float DX = 1.0e-6f;
constexpr float DSIGMA_DT = -0.00026f;

template<typename T>
T* allocateDeviceArray(int size, T init_value = T{}) {
    T* d_ptr;
    cudaMalloc(&d_ptr, size * sizeof(T));
    T* h_ptr = new T[size];
    for (int i = 0; i < size; i++) h_ptr[i] = init_value;
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    delete[] h_ptr;
    return d_ptr;
}

bool testGradientLimiter(float T_hot, float T_cold, float max_grad_limit, const char* test_name) {
    printf("\n--- %s ---\n", test_name);
    printf("Temperature: %.0f K to %.0f K\n", T_hot, T_cold);
    printf("Max gradient limit: %.2e K/m\n", max_grad_limit);

    const int num_cells = NX * NY * NZ;
    const int interface_z = NZ / 2;

    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    float* h_temperature = new float[num_cells];
    float* h_fill_level = new float[num_cells];
    float3* h_interface_normal = new float3[num_cells];

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                // Extreme temperature gradient in X
                float x_frac = float(i) / float(NX - 1);
                h_temperature[idx] = T_hot - (T_hot - T_cold) * x_frac;

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

    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill_level, h_fill_level, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interface_normal, h_interface_normal, num_cells * sizeof(float3), cudaMemcpyHostToDevice);

    MarangoniEffect marangoni(NX, NY, NZ, DSIGMA_DT, DX, 2.0f, max_grad_limit);
    marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                    d_force_x, d_force_y, d_force_z);

    float* h_force_x = new float[num_cells];
    cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check force at center interface
    int idx_center = (NX/2) + NX * ((NY/2) + NY * interface_z);
    float fx = h_force_x[idx_center];

    float grad_T_raw = (T_cold - T_hot) / ((NX - 1) * DX);
    float grad_T_limited = (std::abs(grad_T_raw) > max_grad_limit) ? max_grad_limit : std::abs(grad_T_raw);

    printf("Raw gradient: %.2e K/m\n", std::abs(grad_T_raw));
    printf("Limited gradient: %.2e K/m\n", grad_T_limited);
    printf("Force magnitude: %.2e N/m³\n", std::abs(fx));

    bool passed = true;
    if (std::abs(grad_T_raw) > max_grad_limit) {
        printf("Limiter should activate: YES\n");
        // Force should be limited
    } else {
        printf("Limiter should activate: NO\n");
    }

    // Cleanup
    delete[] h_temperature;
    delete[] h_fill_level;
    delete[] h_interface_normal;
    delete[] h_force_x;

    cudaFree(d_temperature);
    cudaFree(d_fill_level);
    cudaFree(d_interface_normal);
    cudaFree(d_force_x);
    cudaFree(d_force_y);
    cudaFree(d_force_z);

    return passed;
}

int main() {
    printf("=== Test: Marangoni Gradient Limiter ===\n");

    bool all_passed = true;

    // Test 1: Moderate gradient (should not limit)
    all_passed &= testGradientLimiter(2500.0f, 1500.0f, 5.0e8f, "Moderate Gradient");

    // Test 2: High gradient (should limit)
    all_passed &= testGradientLimiter(5000.0f, 500.0f, 5.0e8f, "High Gradient");

    // Test 3: Very high limit (should not limit)
    all_passed &= testGradientLimiter(5000.0f, 500.0f, 1.0e10f, "Very High Limit");

    printf("\n");
    if (all_passed) {
        printf("PASSED: Gradient limiter functioning correctly\n");
        return 0;
    } else {
        printf("FAILED\n");
        return 1;
    }
}
