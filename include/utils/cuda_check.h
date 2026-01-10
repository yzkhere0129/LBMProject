/**
 * @file cuda_check.h
 * @brief CUDA error checking macros for robust error handling
 */

#ifndef LBM_CUDA_CHECK_H
#define LBM_CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdio>

/**
 * @brief Check CUDA API call for errors and throw exception if failed
 *
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

/**
 * @brief Check for kernel launch errors
 *
 * Usage:
 *   myKernel<<<grid, block>>>(...);
 *   CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Kernel launch error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(std::string("Kernel error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#endif // LBM_CUDA_CHECK_H
