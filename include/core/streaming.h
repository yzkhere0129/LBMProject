/**
 * @file streaming.h
 * @brief Streaming operations for Lattice Boltzmann Method
 *
 * Implements pull and push streaming schemes for propagating distribution
 * functions to neighboring cells according to their velocity directions.
 *
 * All implementations are inlined here; no separate .cu is needed.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/lattice_d3q19.h"

namespace lbm {
namespace core {

/**
 * @brief Streaming operations for LBM
 */
class Streaming {
public:
    enum Scheme {
        PULL,
        PUSH
    };

    // Boundary type flags
    static constexpr unsigned int BOUNDARY_NONE  = 0x00;
    static constexpr unsigned int BOUNDARY_X_MIN = 0x01;
    static constexpr unsigned int BOUNDARY_X_MAX = 0x02;
    static constexpr unsigned int BOUNDARY_Y_MIN = 0x04;
    static constexpr unsigned int BOUNDARY_Y_MAX = 0x08;
    static constexpr unsigned int BOUNDARY_Z_MIN = 0x10;
    static constexpr unsigned int BOUNDARY_Z_MAX = 0x20;

    __host__ __device__ static bool isAtBoundary(
        int x, int y, int z, int nx, int ny, int nz)
    {
        return (x == 0 || x == nx - 1 ||
                y == 0 || y == ny - 1 ||
                z == 0 || z == nz - 1);
    }

    __host__ __device__ static unsigned int getBoundaryType(
        int x, int y, int z, int nx, int ny, int nz)
    {
        unsigned int t = BOUNDARY_NONE;
        if (x == 0)      t |= BOUNDARY_X_MIN;
        if (x == nx - 1) t |= BOUNDARY_X_MAX;
        if (y == 0)      t |= BOUNDARY_Y_MIN;
        if (y == ny - 1) t |= BOUNDARY_Y_MAX;
        if (z == 0)      t |= BOUNDARY_Z_MIN;
        if (z == nz - 1) t |= BOUNDARY_Z_MAX;
        return t;
    }

    __device__ static float pullStream(
        const float* f_src,
        int x, int y, int z, int q,
        int nx, int ny, int nz, int n_cells)
    {
        int src_x = (x + ex[opposite[q]] + nx) % nx;
        int src_y = (y + ey[opposite[q]] + ny) % ny;
        int src_z = (z + ez[opposite[q]] + nz) % nz;
        int src_id = src_x + src_y * nx + src_z * nx * ny;
        return f_src[src_id + q * n_cells];
    }

    __device__ static void pushStream(
        float f_val,
        float* f_dst,
        int x, int y, int z, int q,
        int nx, int ny, int nz, int n_cells)
    {
        int dst_x = (x + ex[q] + nx) % nx;
        int dst_y = (y + ey[q] + ny) % ny;
        int dst_z = (z + ez[q] + nz) % nz;
        int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        f_dst[dst_id + q * n_cells] = f_val;
    }
};

__global__ inline void pullStreamingKernel(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        int src_x = (idx - ex[q] + nx) % nx;
        int src_y = (idy - ey[q] + ny) % ny;
        int src_z = (idz - ez[q] + nz) % nz;
        int src_id = src_x + src_y * nx + src_z * nx * ny;
        f_dst[id + q * n_cells] = f_src[src_id + q * n_cells];
    }
}

__global__ inline void periodicStreamingKernel(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        int src_x = (idx - ex[q] + nx) % nx;
        int src_y = (idy - ey[q] + ny) % ny;
        int src_z = (idz - ez[q] + nz) % nz;
        int src_id = src_x + src_y * nx + src_z * nx * ny;
        f_dst[id + q * n_cells] = f_src[src_id + q * n_cells];
    }
}

} // namespace core
} // namespace lbm
