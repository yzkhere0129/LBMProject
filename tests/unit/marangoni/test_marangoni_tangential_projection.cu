/**
 * @file test_marangoni_tangential_projection.cu
 * @brief Test that Marangoni force is tangential to interface (F·n ≈ 0)
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

int main() {
    printf("=== Test: Marangoni Tangential Projection ===\n");

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

    const float T_hot = 2500.0f, T_cold = 1500.0f;

    // Create temperature gradient in X, interface normal in Z
    // Force should be in X (tangent to interface)
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

    MarangoniEffect marangoni(NX, NY, NZ, DSIGMA_DT, DX);
    marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                    d_force_x, d_force_y, d_force_z);

    float* h_force_x = new float[num_cells];
    float* h_force_y = new float[num_cells];
    float* h_force_z = new float[num_cells];

    cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_y, d_force_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_z, d_force_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check interface cells for tangential force
    bool passed = true;
    int cells_checked = 0;
    float max_normal_component = 0.0f;

    for (int j = 1; j < NY - 1; j++) {
        for (int i = 1; i < NX - 1; i++) {
            int idx = i + NX * (j + NY * interface_z);

            if (h_fill_level[idx] > 0.001f && h_fill_level[idx] < 0.999f) {
                float fx = h_force_x[idx];
                float fy = h_force_y[idx];
                float fz = h_force_z[idx];
                float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

                if (f_mag > 1e-10f) {
                    float3 n = h_interface_normal[idx];
                    float f_dot_n = fx * n.x + fy * n.y + fz * n.z;
                    float rel_normal = std::abs(f_dot_n) / f_mag;

                    cells_checked++;
                    if (rel_normal > max_normal_component) {
                        max_normal_component = rel_normal;
                    }

                    // Force should be tangential (normal component < 1%)
                    if (rel_normal > 0.01f) {
                        printf("Cell (%d,%d,%d): Normal component = %.1f%% of total\n",
                               i, j, interface_z, rel_normal * 100.0f);
                        passed = false;
                    }
                }
            }
        }
    }

    printf("Checked %d interface cells\n", cells_checked);
    printf("Maximum normal component: %.2f%% of total force\n", max_normal_component * 100.0f);

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
        printf("PASSED: Force is tangential to interface (F·n < 1%%)\n");
        return 0;
    } else {
        printf("FAILED: Force has significant normal component\n");
        return 1;
    }
}
