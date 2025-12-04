/**
 * @file kernel_config.h
 * @brief Utilities for optimal CUDA kernel launch configuration
 *
 * This file provides helpers to compute optimal block/grid dimensions
 * based on domain geometry and hardware characteristics.
 */

#pragma once

#include <cuda_runtime.h>
#include <utility>

namespace lbm {
namespace cuda {

/**
 * @brief Compute optimal block configuration for 2D or 3D domains
 *
 * Strategy:
 * - For thin domains (Nz ≤ 4): Use 2D-optimized config (16×16×Nz)
 * - For thick domains (Nz > 4): Use balanced 3D config (8×8×8)
 *
 * @param nx Domain size in x-direction
 * @param ny Domain size in y-direction
 * @param nz Domain size in z-direction
 * @return Pair of (blockSize, gridSize)
 */
inline std::pair<dim3, dim3> computeOptimalLaunchConfig(int nx, int ny, int nz) {
    dim3 blockSize;

    // 2D optimization: maximize XY footprint for thin domains
    if (nz <= 4) {
        blockSize = dim3(16, 16, nz);
    }
    // 3D optimization: balanced configuration
    else {
        blockSize = dim3(8, 8, 8);
    }

    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    return {blockSize, gridSize};
}

/**
 * @brief Compute optimal 1D block configuration
 *
 * Use for kernels that flatten 3D indexing into linear loops.
 *
 * @param num_elements Total number of elements
 * @return Pair of (threads_per_block, num_blocks)
 */
inline std::pair<int, int> computeOptimal1DConfig(int num_elements) {
    const int threads_per_block = 256;  // Good balance for most GPUs
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    return {threads_per_block, num_blocks};
}

/**
 * @brief Query device properties for occupancy tuning
 *
 * @param device_id CUDA device ID (default: 0)
 * @return cudaDeviceProp structure
 */
inline cudaDeviceProp getDeviceProperties(int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop;
}

/**
 * @brief Compute theoretical occupancy for given launch config
 *
 * @param blockSize Block dimensions
 * @param dynamic_smem Dynamic shared memory per block [bytes]
 * @param registers_per_thread Registers per thread
 * @return Occupancy ratio [0.0 - 1.0]
 */
inline float computeOccupancy(dim3 blockSize,
                              size_t dynamic_smem = 0,
                              int registers_per_thread = 32) {
    int threads_per_block = blockSize.x * blockSize.y * blockSize.z;

    cudaDeviceProp prop = getDeviceProperties();

    // Max blocks per SM based on thread limit
    int max_blocks_per_sm_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;

    // Max blocks per SM based on block limit
    int max_blocks_per_sm_blocks = prop.maxBlocksPerMultiProcessor;

    // Max blocks per SM based on shared memory
    int max_blocks_per_sm_smem = (dynamic_smem > 0)
        ? prop.sharedMemPerMultiprocessor / dynamic_smem
        : max_blocks_per_sm_blocks;

    // Max blocks per SM based on registers
    int total_registers = threads_per_block * registers_per_thread;
    int max_blocks_per_sm_regs = (total_registers > 0)
        ? prop.regsPerMultiprocessor / total_registers
        : max_blocks_per_sm_blocks;

    // Take minimum (most restrictive resource)
    int max_blocks = std::min({
        max_blocks_per_sm_threads,
        max_blocks_per_sm_blocks,
        max_blocks_per_sm_smem,
        max_blocks_per_sm_regs
    });

    // Compute occupancy
    int active_warps = (threads_per_block / 32) * max_blocks;
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;

    return static_cast<float>(active_warps) / max_warps;
}

/**
 * @brief Print launch configuration summary
 *
 * Useful for debugging and performance tuning.
 *
 * @param blockSize Block dimensions
 * @param gridSize Grid dimensions
 * @param kernel_name Kernel name for logging
 */
inline void printLaunchConfig(dim3 blockSize, dim3 gridSize,
                              const char* kernel_name = "kernel") {
    int threads_per_block = blockSize.x * blockSize.y * blockSize.z;
    int total_threads = threads_per_block * gridSize.x * gridSize.y * gridSize.z;

    printf("Launch config for %s:\n", kernel_name);
    printf("  Block: (%d, %d, %d) = %d threads\n",
           blockSize.x, blockSize.y, blockSize.z, threads_per_block);
    printf("  Grid: (%d, %d, %d)\n",
           gridSize.x, gridSize.y, gridSize.z);
    printf("  Total threads: %d\n", total_threads);
    printf("  Estimated occupancy: %.1f%%\n",
           computeOccupancy(blockSize) * 100.0f);
}

} // namespace cuda
} // namespace lbm
