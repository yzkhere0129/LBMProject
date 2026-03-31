/**
 * @file test_normals_unit.cu
 * @brief Minimal unit test for VOF interface normal computation
 * 
 * Directly tests reconstructInterfaceKernel with known fill_level input
 * to verify the math works correctly, independent of main loop.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Simple CUDA check macro
#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

// Copy the kernel here for testing
__global__ void testReconstructNormalsKernel(
    const float* fill_level,
    float3* interface_normal,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Central differences with boundary handling
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    int k_m = max(k - 1, 0);
    int k_p = min(k + 1, nz - 1);

    int idx_xm = i_m + nx * (j + ny * k);
    int idx_xp = i_p + nx * (j + ny * k);
    int idx_ym = i + nx * (j_m + ny * k);
    int idx_yp = i + nx * (j_p + ny * k);
    int idx_zm = i + nx * (j + ny * k_m);
    int idx_zp = i + nx * (j + ny * k_p);

    float denom_x = (i > 0 && i < nx - 1) ? 2.0f * dx : dx;
    float denom_y = (j > 0 && j < ny - 1) ? 2.0f * dx : dx;
    float denom_z = (k > 0 && k < nz - 1) ? 2.0f * dx : dx;

    float grad_x = (fill_level[idx_xp] - fill_level[idx_xm]) / denom_x;
    float grad_y = (fill_level[idx_yp] - fill_level[idx_ym]) / denom_y;
    float grad_z = (fill_level[idx_zp] - fill_level[idx_zm]) / denom_z;

    float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

    if (grad_mag > 1e-8f) {
        interface_normal[idx].x = -grad_x / grad_mag;
        interface_normal[idx].y = -grad_y / grad_mag;
        interface_normal[idx].z = -grad_z / grad_mag;
    } else {
        interface_normal[idx].x = 0.0f;
        interface_normal[idx].y = 0.0f;
        interface_normal[idx].z = 0.0f;
    }
}

int main() {
    printf("============================================================\n");
    printf("  VOF Interface Normal Unit Test\n");
    printf("============================================================\n\n");

    // Simple 1D test: 5x1x5 grid with step interface at z=2
    const int NX = 5, NY = 1, NZ = 5;
    const int num_cells = NX * NY * NZ;
    const float dx = 1.0f;

    // Create fill_level: all 1.0 except z=2 (0.5) and z>2 (0.0)
    std::vector<float> h_fill(num_cells);
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + j * NX + k * NX * NY;
                if (k < 2) h_fill[idx] = 1.0f;      // Liquid
                else if (k == 2) h_fill[idx] = 0.5f; // Interface
                else h_fill[idx] = 0.0f;             // Gas
            }
        }
    }

    printf("Input fill_level (center column, x=2):\n");
    for (int k = 0; k < NZ; k++) {
        int idx = 2 + 0 * NX + k * NX * NY;
        printf("  k=%d: fill=%.1f\n", k, h_fill[idx]);
    }

    // Allocate GPU memory
    float* d_fill;
    float3* d_normals;
    CUDA_CHECK(cudaMalloc(&d_fill, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, num_cells * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_fill, h_fill.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((NX+7)/8, (NY+7)/8, (NZ+7)/8);
    testReconstructNormalsKernel<<<gridSize, blockSize>>>(d_fill, d_normals, dx, NX, NY, NZ);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<float3> h_normals(num_cells);
    CUDA_CHECK(cudaMemcpy(h_normals.data(), d_normals, num_cells * sizeof(float3), cudaMemcpyDeviceToHost));

    // Print results at interface
    printf("\nOutput normals (center column, x=2):\n");
    for (int k = 0; k < NZ; k++) {
        int idx = 2 + 0 * NX + k * NX * NY;
        printf("  k=%d: n=(%.4f, %.4f, %.4f), |n|=%.4f\n",
               k, h_normals[idx].x, h_normals[idx].y, h_normals[idx].z,
               sqrtf(h_normals[idx].x*h_normals[idx].x + 
                     h_normals[idx].y*h_normals[idx].y + 
                     h_normals[idx].z*h_normals[idx].z));
    }

    // Expected: at k=2 (interface), n should be (0, 0, 1) or (0, 0, -1)
    // pointing from liquid to gas

    // Cleanup
    cudaFree(d_fill);
    cudaFree(d_normals);

    printf("\n============================================================\n");
    printf("  Test complete\n");
    printf("============================================================\n");

    return 0;
}
