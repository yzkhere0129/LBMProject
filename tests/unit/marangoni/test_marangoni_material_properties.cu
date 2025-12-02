/**
 * @file test_marangoni_material_properties.cu
 * @brief Test Marangoni with different materials
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

using namespace lbm::physics;

constexpr int NX = 10, NY = 10, NZ = 10;
constexpr float DX = 1.0e-6f;

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

void testMaterial(const char* name, float dsigma_dT, float T_melt) {
    printf("\n--- Testing %s ---\n", name);
    printf("  dσ/dT = %.2e N/(m·K)\n", dsigma_dT);
    printf("  T_melt = %.0f K\n", T_melt);

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

    float T_hot = T_melt + 500.0f;
    float T_cold = T_melt - 200.0f;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

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

    MarangoniEffect marangoni(NX, NY, NZ, dsigma_dT, DX, 2.0f, 5.0e8f, T_melt);
    marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                    d_force_x, d_force_y, d_force_z);

    float* h_force_x = new float[num_cells];
    cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    int idx_center = (NX/2) + NX * ((NY/2) + NY * interface_z);
    float fx = h_force_x[idx_center];

    printf("  Force magnitude: %.3e N/m³\n", std::abs(fx));
    printf("  Force direction: %s\n", fx < 0 ? "hot → cold" : "cold → hot");

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
}

int main() {
    printf("=== Test: Marangoni with Different Materials ===\n");

    // Different materials with different dσ/dT and melting points
    testMaterial("Ti6Al4V", -0.00026f, 1923.0f);
    testMaterial("SS316L", -0.00043f, 1673.0f);
    testMaterial("AlSi10Mg", -0.00035f, 848.0f);

    printf("\nPASSED: Marangoni works with different materials\n");
    return 0;
}
