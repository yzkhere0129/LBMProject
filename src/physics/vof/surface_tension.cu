/**
 * @file surface_tension.cu
 * @brief Implementation of surface tension using CSF model
 */

#include "physics/surface_tension.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include "utils/cuda_check.h"

namespace lbm {
namespace physics {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief CSF surface tension force kernel
 * @note Computes F = σ * κ * ∇f
 */
__global__ void computeCSFForceKernel(
    const float* fill_level,
    const float* curvature,
    float* force_x,
    float* force_y,
    float* force_z,
    float sigma,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Only compute force at interface cells (where gradient is non-zero)
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        force_x[idx] = 0.0f;
        force_y[idx] = 0.0f;
        force_z[idx] = 0.0f;
        return;
    }

    // Get curvature
    float kappa = curvature[idx];

    // Compute fill level gradient using central differences
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

    float grad_f_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    float grad_f_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    float grad_f_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);

    // CSF force: F = σ * κ * ∇f
    force_x[idx] = sigma * kappa * grad_f_x;
    force_y[idx] = sigma * kappa * grad_f_y;
    force_z[idx] = sigma * kappa * grad_f_z;
}

/**
 * @brief Add CSF force to existing force field
 */
__global__ void addCSFForceKernel(
    const float* fill_level,
    const float* curvature,
    float* force_x,
    float* force_y,
    float* force_z,
    float sigma,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Only compute force at interface cells
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        return;  // Don't modify existing force
    }

    // Get curvature
    float kappa = curvature[idx];

    // Compute fill level gradient
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

    float grad_f_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    float grad_f_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    float grad_f_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);

    // Add CSF force to existing force
    force_x[idx] += sigma * kappa * grad_f_x;
    force_y[idx] += sigma * kappa * grad_f_y;
    force_z[idx] += sigma * kappa * grad_f_z;
}

// ============================================================================
// SurfaceTension Implementation
// ============================================================================

SurfaceTension::SurfaceTension(int nx, int ny, int nz,
                               float surface_tension_coeff, float dx)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      sigma_(surface_tension_coeff), dx_(dx)
{
    // No device memory allocation needed - forces are computed into user-provided arrays
}

SurfaceTension::~SurfaceTension() {
    // No cleanup needed
}

void SurfaceTension::computeCSFForce(const float* fill_level,
                                     const float* curvature,
                                     float* force_x,
                                     float* force_y,
                                     float* force_z) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    computeCSFForceKernel<<<gridSize, blockSize>>>(
        fill_level, curvature, force_x, force_y, force_z,
        sigma_, dx_, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("SurfaceTension: Kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void SurfaceTension::addCSFForce(const float* fill_level,
                                 const float* curvature,
                                 float* force_x,
                                 float* force_y,
                                 float* force_z) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    addCSFForceKernel<<<gridSize, blockSize>>>(
        fill_level, curvature, force_x, force_y, force_z,
        sigma_, dx_, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("SurfaceTension: Kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

} // namespace physics
} // namespace lbm
