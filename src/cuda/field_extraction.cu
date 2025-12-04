/**
 * @file field_extraction.cu
 * @brief Implementation of efficient field extraction kernels
 */

#include "cuda/field_extraction.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace lbm {
namespace cuda {

/**
 * @brief Fused kernel for extracting multiple fields in one pass
 */
__global__ void extractFieldsForVTKKernel(
    const float* __restrict__ d_temperature,
    const float* __restrict__ d_fill,
    const float* __restrict__ d_ux,
    const float* __restrict__ d_uy,
    const float* __restrict__ d_uz,
    float* __restrict__ d_out_temperature,
    float* __restrict__ d_out_fill,
    float* __restrict__ d_out_phase,
    float* __restrict__ d_out_ux,
    float* __restrict__ d_out_uy,
    float* __restrict__ d_out_uz,
    int num_cells,
    float velocity_conversion)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Single coalesced read per field
    float T = d_temperature[idx];
    float f = d_fill[idx];
    float ux = d_ux[idx];
    float uy = d_uy[idx];
    float uz = d_uz[idx];

    // Compute phase state on-the-fly
    // 0 = vapor (f < 0.1), 1 = interface (0.1 ≤ f ≤ 0.9), 2 = liquid (f > 0.9)
    float phase;
    if (f < 0.1f) {
        phase = 0.0f;
    } else if (f > 0.9f) {
        phase = 2.0f;
    } else {
        phase = 1.0f;
    }

    // Apply velocity conversion if needed
    ux *= velocity_conversion;
    uy *= velocity_conversion;
    uz *= velocity_conversion;

    // Single coalesced write per field
    d_out_temperature[idx] = T;
    d_out_fill[idx] = f;
    d_out_phase[idx] = phase;
    d_out_ux[idx] = ux;
    d_out_uy[idx] = uy;
    d_out_uz[idx] = uz;
}

void extractFieldsForVTK(
    const float* d_temperature,
    const float* d_fill,
    const float* d_ux,
    const float* d_uy,
    const float* d_uz,
    float* d_out_temperature,
    float* d_out_fill,
    float* d_out_phase,
    float* d_out_ux,
    float* d_out_uy,
    float* d_out_uz,
    int num_cells,
    float velocity_conversion)
{
    const int threads_per_block = 256;
    const int num_blocks = (num_cells + threads_per_block - 1) / threads_per_block;

    extractFieldsForVTKKernel<<<num_blocks, threads_per_block>>>(
        d_temperature, d_fill, d_ux, d_uy, d_uz,
        d_out_temperature, d_out_fill, d_out_phase,
        d_out_ux, d_out_uy, d_out_uz,
        num_cells, velocity_conversion
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "extractFieldsForVTK kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

/**
 * @brief Extract single scalar field with transformation
 */
__global__ void extractScalarFieldKernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int num_cells,
    float scale,
    float offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    d_output[idx] = d_input[idx] * scale + offset;
}

void extractScalarField(
    const float* d_input,
    float* d_output,
    int num_cells,
    float scale,
    float offset)
{
    const int threads_per_block = 256;
    const int num_blocks = (num_cells + threads_per_block - 1) / threads_per_block;

    extractScalarFieldKernel<<<num_blocks, threads_per_block>>>(
        d_input, d_output, num_cells, scale, offset
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "extractScalarField kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

/**
 * @brief Compute velocity magnitude
 */
__global__ void extractVelocityMagnitudeKernel(
    const float* __restrict__ d_ux,
    const float* __restrict__ d_uy,
    const float* __restrict__ d_uz,
    float* __restrict__ d_vmag,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float ux = d_ux[idx];
    float uy = d_uy[idx];
    float uz = d_uz[idx];

    d_vmag[idx] = sqrtf(ux*ux + uy*uy + uz*uz);
}

void extractVelocityMagnitude(
    const float* d_ux,
    const float* d_uy,
    const float* d_uz,
    float* d_vmag,
    int num_cells)
{
    const int threads_per_block = 256;
    const int num_blocks = (num_cells + threads_per_block - 1) / threads_per_block;

    extractVelocityMagnitudeKernel<<<num_blocks, threads_per_block>>>(
        d_ux, d_uy, d_uz, d_vmag, num_cells
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "extractVelocityMagnitude kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

/**
 * @brief Extract 2D slice from 3D field
 */
__global__ void extract2DSliceKernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int nx, int ny, int nz,
    int slice_axis,
    int slice_index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int n1, n2, idx_3d, idx_2d;

    if (slice_axis == 0) {  // YZ plane (X = slice_index)
        if (j >= ny || i >= nz) return;
        n1 = ny;
        n2 = nz;
        idx_3d = slice_index + nx * (j + ny * i);
        idx_2d = j + n1 * i;
    }
    else if (slice_axis == 1) {  // XZ plane (Y = slice_index)
        if (j >= nx || i >= nz) return;
        n1 = nx;
        n2 = nz;
        idx_3d = j + nx * (slice_index + ny * i);
        idx_2d = j + n1 * i;
    }
    else {  // XY plane (Z = slice_index)
        if (j >= nx || i >= ny) return;
        n1 = nx;
        n2 = ny;
        idx_3d = j + nx * (i + ny * slice_index);
        idx_2d = j + n1 * i;
    }

    d_output[idx_2d] = d_input[idx_3d];
}

void extract2DSlice(
    const float* d_input,
    float* d_output,
    int nx, int ny, int nz,
    int slice_axis,
    int slice_index)
{
    dim3 blockSize(16, 16);
    dim3 gridSize;

    if (slice_axis == 0) {  // YZ plane
        gridSize = dim3((nz + 15) / 16, (ny + 15) / 16);
    }
    else if (slice_axis == 1) {  // XZ plane
        gridSize = dim3((nz + 15) / 16, (nx + 15) / 16);
    }
    else {  // XY plane
        gridSize = dim3((ny + 15) / 16, (nx + 15) / 16);
    }

    extract2DSliceKernel<<<gridSize, blockSize>>>(
        d_input, d_output, nx, ny, nz, slice_axis, slice_index
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "extract2DSlice kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace lbm
