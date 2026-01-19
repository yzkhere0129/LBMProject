/**
 * @file surface_tension.cu
 * @brief Implementation of surface tension using CSF model
 */

#include "physics/surface_tension.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
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

    // NaN/Inf protection
    if (isnan(kappa) || isinf(kappa)) {
        force_x[idx] = 0.0f;
        force_y[idx] = 0.0f;
        force_z[idx] = 0.0f;
        return;
    }

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

    // Gradient magnitude check
    float grad_mag = sqrtf(grad_f_x*grad_f_x + grad_f_y*grad_f_y + grad_f_z*grad_f_z);
    if (grad_mag < 1e-12f) {
        force_x[idx] = 0.0f;
        force_y[idx] = 0.0f;
        force_z[idx] = 0.0f;
        return;
    }

    // CSF force: F = σ * κ * ∇f
    float fx = sigma * kappa * grad_f_x;
    float fy = sigma * kappa * grad_f_y;
    float fz = sigma * kappa * grad_f_z;

    // NaN/Inf protection on force components
    if (isnan(fx) || isnan(fy) || isnan(fz) ||
        isinf(fx) || isinf(fy) || isinf(fz)) {
        force_x[idx] = 0.0f;
        force_y[idx] = 0.0f;
        force_z[idx] = 0.0f;
        return;
    }

    // CRITICAL FIX (2026-01-17): Limit force magnitude to prevent numerical instability
    //
    // Physical reasoning: Surface tension acceleration should not exceed reasonable bounds
    // For typical droplet oscillations:
    //   a_max ~ σ·κ/ρ ~ (1.5 N/m) × (1e6 m⁻¹) / (4000 kg/m³) ~ 375 m/s²
    //
    // Setting F_max = 1e12 N/m³ gives a_max = F/ρ ~ 250 m/s² (reasonable)
    // This prevents runaway velocities from numerical curvature spikes
    //
    const float F_MAX = 1e12f;  // N/m³
    float f_mag = sqrtf(fx*fx + fy*fy + fz*fz);

    if (f_mag > F_MAX) {
        float scale = F_MAX / f_mag;
        fx *= scale;
        fy *= scale;
        fz *= scale;
    }

    force_x[idx] = fx;
    force_y[idx] = fy;
    force_z[idx] = fz;
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

    // NaN/Inf protection
    if (isnan(kappa) || isinf(kappa)) {
        return;  // Don't modify existing force
    }

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

    // Gradient magnitude check
    float grad_mag = sqrtf(grad_f_x*grad_f_x + grad_f_y*grad_f_y + grad_f_z*grad_f_z);
    if (grad_mag < 1e-12f) {
        return;  // Don't add zero force
    }

    // CSF force: F = σ * κ * ∇f
    float fx = sigma * kappa * grad_f_x;
    float fy = sigma * kappa * grad_f_y;
    float fz = sigma * kappa * grad_f_z;

    // NaN/Inf protection on force components
    if (isnan(fx) || isnan(fy) || isnan(fz) ||
        isinf(fx) || isinf(fy) || isinf(fz)) {
        return;  // Don't add invalid force
    }

    // CRITICAL FIX (2026-01-17): Limit force magnitude to prevent numerical instability
    const float F_MAX = 1e12f;  // N/m³
    float f_mag = sqrtf(fx*fx + fy*fy + fz*fz);

    if (f_mag > F_MAX) {
        float scale = F_MAX / f_mag;
        fx *= scale;
        fy *= scale;
        fz *= scale;
    }

    // Add limited CSF force to existing force
    force_x[idx] += fx;
    force_y[idx] += fy;
    force_z[idx] += fz;
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

    // DIAGNOSTIC (2026-01-17): Monitor surface tension force magnitude
    static int call_count = 0;
    if (call_count % 500 == 0 && call_count < 5000) {
        // Sample force magnitude from device
        const int sample_size = std::min(10000, num_cells_);
        std::vector<float> h_fx(sample_size), h_fy(sample_size), h_fz(sample_size);

        cudaMemcpy(h_fx.data(), force_x, sample_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fy.data(), force_y, sample_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fz.data(), force_z, sample_size * sizeof(float), cudaMemcpyDeviceToHost);

        float max_force = 0.0f;
        int active_cells = 0;

        for (int i = 0; i < sample_size; ++i) {
            float f_mag = std::sqrt(h_fx[i]*h_fx[i] + h_fy[i]*h_fy[i] + h_fz[i]*h_fz[i]);
            if (f_mag > 1e-6f) {
                max_force = std::max(max_force, f_mag);
                active_cells++;
            }
        }

        if (active_cells > 0) {
            // Typical density for metal (Ti6Al4V)
            const float rho_typical = 4110.0f;  // kg/m³
            float accel = max_force / rho_typical;

            printf("[SURFACE TENSION] Call %d: F_max=%.4e N/m³, a_max=%.4e m/s², active=%d cells\n",
                   call_count, max_force, accel, active_cells);
        }
    }
    call_count++;
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
