/**
 * @file test_marangoni_stability.cu
 * @brief Test Marangoni computation stability over many iterations
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

using namespace lbm::physics;

constexpr int NX = 15, NY = 15, NZ = 15;
constexpr float DX = 2.0e-6f;
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
    printf("=== Test: Marangoni Stability ===\n");
    printf("Running 1000 iterations with realistic LPBF conditions...\n\n");

    const int num_cells = NX * NY * NZ;
    const int interface_z = NZ / 2;
    const int num_iterations = 1000;

    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    float* h_temperature = new float[num_cells];
    float* h_fill_level = new float[num_cells];
    float3* h_interface_normal = new float3[num_cells];

    // Setup: radial temperature profile
    const float T_center = 3000.0f, T_edge = 1900.0f;
    float cx = (NX / 2.0f) * DX;
    float cy = (NY / 2.0f) * DX;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                float x = i * DX;
                float y = j * DX;
                float r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                float r_max = std::sqrt(cx * cx + cy * cy);

                h_temperature[idx] = T_center - (T_center - T_edge) * (r / r_max);

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

    float* h_force_x = new float[num_cells];
    float* h_force_y = new float[num_cells];
    float* h_force_z = new float[num_cells];

    bool all_finite = true;
    bool velocity_bounded = true;
    float max_velocity_seen = 0.0f;
    const float rho = 4110.0f;  // kg/m³

    for (int iter = 0; iter < num_iterations; iter++) {
        marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                        d_force_x, d_force_y, d_force_z);

        if (iter % 100 == 0) {
            cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_force_y, d_force_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_force_z, d_force_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float max_force = 0.0f;
            for (int i = 0; i < num_cells; i++) {
                float fx = h_force_x[i];
                float fy = h_force_y[i];
                float fz = h_force_z[i];

                if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(fz)) {
                    printf("ERROR at iteration %d: NaN or Inf detected\n", iter);
                    all_finite = false;
                    break;
                }

                float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);
                if (f_mag > max_force) max_force = f_mag;
            }

            // Estimate velocity: v ~ F*dt/rho (simple Euler)
            float dt = 1e-8f;  // typical timestep
            float max_vel = (max_force * dt) / rho;
            if (max_vel > max_velocity_seen) max_velocity_seen = max_vel;

            printf("Iteration %4d: max_force = %.3e N/m³, max_vel ~ %.3f m/s\n",
                   iter, max_force, max_vel);

            if (max_vel > 10.0f) {  // Unreasonably high for LPBF
                printf("  WARNING: Velocity exceeds reasonable bounds\n");
                velocity_bounded = false;
            }

            if (!all_finite) break;
        }
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

    printf("\n");
    if (all_finite && velocity_bounded) {
        printf("PASSED: No NaN, velocities bounded (max: %.3f m/s)\n", max_velocity_seen);
        return 0;
    } else {
        printf("FAILED: Stability issues detected\n");
        return 1;
    }
}
