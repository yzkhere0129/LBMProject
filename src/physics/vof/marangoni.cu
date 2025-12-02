/**
 * @file marangoni.cu
 * @brief Implementation of Marangoni effect (thermocapillary flow)
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace lbm {
namespace physics {

// ============================================================================
// Physical Constants
// ============================================================================
// Note: MAX_PHYSICAL_GRAD_T is now passed as a parameter to allow configuration

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Marangoni force kernel
 * @note Computes F = (dσ/dT) * ∇_s T * |∇f| [N/m³]
 *       where ∇_s T = (I - n⊗n) · ∇T is tangential gradient
 */
__global__ void computeMarangoniForceKernel(
    const float* temperature,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    float dsigma_dT,
    float dx,
    float h_interface,
    float max_gradient_limit,
    float interface_cutoff_min,
    float interface_cutoff_max,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // FIX BUG 1: Widened interface cutoff from (0.01, 0.99) to configurable range
    // Default: (0.001, 0.999) to capture more of the interface region
    float f = fill_level[idx];
    if (f < interface_cutoff_min || f > interface_cutoff_max) {
        force_x[idx] = 0.0f;
        force_y[idx] = 0.0f;
        force_z[idx] = 0.0f;
        return;
    }

    // Compute temperature gradient using central differences
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

    // Temperature gradient
    float grad_T_x = (temperature[idx_xp] - temperature[idx_xm]) / (2.0f * dx);
    float grad_T_y = (temperature[idx_yp] - temperature[idx_ym]) / (2.0f * dx);
    float grad_T_z = (temperature[idx_zp] - temperature[idx_zm]) / (2.0f * dx);

    // FIX BUG 2: Configurable gradient limiter
    // Physical justification: Maximum gradient from laser heating
    //
    // Derivation from laser parameters:
    // Laser heating creates maximum gradient: ∇T ~ P/(κ·r²)
    // Where:
    //   P = 20 W (laser power)
    //   κ = 35 W/(m·K) (thermal conductivity of Ti6Al4V)
    //   r = 50 μm (laser spot radius)
    //
    // ∇T_max ~ 20 / (35 × (50×10⁻⁶)²) ≈ 2.3×10⁸ K/m
    //
    // Default: 5e8 K/m (2x safety margin for stability and transients)
    // Now configurable to allow higher gradients for high-power LPBF
    float grad_T_mag = sqrtf(grad_T_x * grad_T_x +
                             grad_T_y * grad_T_y +
                             grad_T_z * grad_T_z);

    if (grad_T_mag > max_gradient_limit) {
        float scale = max_gradient_limit / grad_T_mag;
        grad_T_x *= scale;
        grad_T_y *= scale;
        grad_T_z *= scale;
    }

    // Interface normal
    float3 n = interface_normal[idx];

    // Compute tangential temperature gradient: ∇_s T = (I - n⊗n) · ∇T
    // This projects ∇T onto the tangent plane of the interface
    float n_dot_gradT = n.x * grad_T_x + n.y * grad_T_y + n.z * grad_T_z;

    float grad_Ts_x = grad_T_x - n_dot_gradT * n.x;
    float grad_Ts_y = grad_T_y - n_dot_gradT * n.y;
    float grad_Ts_z = grad_T_z - n_dot_gradT * n.z;

    // Compute fill level gradient magnitude (approximates interface delta function)
    float grad_f_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    float grad_f_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    float grad_f_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);

    float grad_f_mag = sqrtf(grad_f_x * grad_f_x +
                             grad_f_y * grad_f_y +
                             grad_f_z * grad_f_z);

    // Marangoni force (CSF formulation): F = (dσ/dT) × ∇_s T × |∇f|
    //
    // Physical derivation:
    //   Surface stress: τ_s = (dσ/dT) × ∇_s T  [N/m²]
    //   CSF converts surface force to volumetric: F = τ_s × |∇f|  [N/m³]
    //   where |∇f| acts as the interface delta function
    //
    // Units: [N/(m·K)] × [K/m] × [1/m] = [N/m³]  ✓
    //
    // Note: NO division by h_phys - |∇f| already provides the delta function
    float coeff = dsigma_dT * grad_f_mag;  // [N/(m³·K)]

    force_x[idx] = coeff * grad_Ts_x;  // [N/m³]
    force_y[idx] = coeff * grad_Ts_y;
    force_z[idx] = coeff * grad_Ts_z;
}

/**
 * @brief Add Marangoni force to existing force field (HYBRID VERSION)
 *
 * CRITICAL FIX for LPBF simulations:
 * Uses fill_level to identify interface cells WITH LIQUID DETECTION.
 * This ensures Marangoni acts at the melt pool surface, not just VOF interface.
 *
 * Logic:
 * 1. Check if cell is at VOF interface (configurable cutoff)
 * 2. OR check if cell has temperature > T_melt (liquid region at top surface)
 * 3. This captures BOTH static VOF interface AND dynamic melt pool surface
 *
 * FIX BUG 1 & BUG 3: Configurable interface cutoffs and material properties
 */
__global__ void addMarangoniForceKernel(
    const float* temperature,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    float dsigma_dT,
    float dx,
    float h_interface,
    float max_gradient_limit,
    float T_melt,
    float interface_cutoff_min,
    float interface_cutoff_max,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // HYBRID INTERFACE DETECTION:
    // 1. Standard VOF interface cells (FIX BUG 1: configurable cutoffs)
    float f = fill_level[idx];
    bool is_vof_interface = (f >= interface_cutoff_min && f <= interface_cutoff_max);

    // 2. Top surface of melt pool (for LPBF with stationary VOF)
    //    Detect by: high temperature + near top + has liquid neighbor above
    //    FIX BUG 3: Use configurable T_melt instead of hardcoded value
    float T = temperature[idx];
    bool is_melt_surface = false;

    if (T > T_melt && k >= nz - 10) {  // Hot region near top surface
        if (k < nz - 1) {
            // Interior: Check for temperature drop above (normal melt pool)
            int k_up = k + 1;
            int idx_up = i + nx * (j + ny * k_up);
            float T_up = temperature[idx_up];
            if (T_up < T - 50.0f) {  // Relaxed from 100K to 50K
                is_melt_surface = true;
            }
        } else {
            // Top boundary (k = nz-1): Laser-heated surface, always apply Marangoni
            // This is critical for LPBF where laser hits the top surface
            is_melt_surface = true;
        }
    }

    // Apply Marangoni force at EITHER interface type
    if (!is_vof_interface && !is_melt_surface) {
        return;  // Don't modify existing force
    }

    // Compute temperature gradient
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

    float grad_T_x = (temperature[idx_xp] - temperature[idx_xm]) / (2.0f * dx);
    float grad_T_y = (temperature[idx_yp] - temperature[idx_ym]) / (2.0f * dx);
    float grad_T_z = (temperature[idx_zp] - temperature[idx_zm]) / (2.0f * dx);

    // FIX BUG 2: Configurable gradient limiter
    float grad_T_mag = sqrtf(grad_T_x * grad_T_x +
                             grad_T_y * grad_T_y +
                             grad_T_z * grad_T_z);

    if (grad_T_mag > max_gradient_limit) {
        float scale = max_gradient_limit / grad_T_mag;
        grad_T_x *= scale;
        grad_T_y *= scale;
        grad_T_z *= scale;
    }

    // Interface normal
    float3 n = interface_normal[idx];

    // Tangential temperature gradient
    float n_dot_gradT = n.x * grad_T_x + n.y * grad_T_y + n.z * grad_T_z;

    float grad_Ts_x = grad_T_x - n_dot_gradT * n.x;
    float grad_Ts_y = grad_T_y - n_dot_gradT * n.y;
    float grad_Ts_z = grad_T_z - n_dot_gradT * n.z;

    // Fill level gradient magnitude
    float grad_f_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    float grad_f_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    float grad_f_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);

    float grad_f_mag = sqrtf(grad_f_x * grad_f_x +
                             grad_f_y * grad_f_y +
                             grad_f_z * grad_f_z);

    // Add Marangoni force (CSF formulation)
    // F = (dσ/dT) × ∇_s T × |∇f|  [N/m³]
    // NO division by h_phys - |∇f| already provides the delta function
    float coeff = dsigma_dT * grad_f_mag;  // [N/(m³·K)]

    force_x[idx] += coeff * grad_Ts_x;  // [N/m³]
    force_y[idx] += coeff * grad_Ts_y;
    force_z[idx] += coeff * grad_Ts_z;
}

// ============================================================================
// MarangoniEffect Implementation
// ============================================================================

MarangoniEffect::MarangoniEffect(int nx, int ny, int nz,
                                 float dsigma_dT,
                                 float dx,
                                 float interface_thickness,
                                 float max_gradient_limit,
                                 float T_melt,
                                 float T_boil,
                                 float interface_cutoff_min,
                                 float interface_cutoff_max)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      dsigma_dT_(dsigma_dT), dx_(dx), h_interface_(interface_thickness),
      max_gradient_limit_(max_gradient_limit), T_melt_(T_melt), T_boil_(T_boil),
      interface_cutoff_min_(interface_cutoff_min), interface_cutoff_max_(interface_cutoff_max)
{
    // No device memory allocation needed
}

MarangoniEffect::~MarangoniEffect() {
    // No cleanup needed
}

void MarangoniEffect::computeMarangoniForce(const float* temperature,
                                            const float* fill_level,
                                            const float3* interface_normal,
                                            float* force_x,
                                            float* force_y,
                                            float* force_z) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    computeMarangoniForceKernel<<<gridSize, blockSize>>>(
        temperature, fill_level, interface_normal,
        force_x, force_y, force_z,
        dsigma_dT_, dx_, h_interface_,
        max_gradient_limit_, interface_cutoff_min_, interface_cutoff_max_,
        nx_, ny_, nz_);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("MarangoniEffect: Kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void MarangoniEffect::addMarangoniForce(const float* temperature,
                                        const float* fill_level,
                                        const float3* interface_normal,
                                        float* force_x,
                                        float* force_y,
                                        float* force_z) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    addMarangoniForceKernel<<<gridSize, blockSize>>>(
        temperature, fill_level, interface_normal,
        force_x, force_y, force_z,
        dsigma_dT_, dx_, h_interface_,
        max_gradient_limit_, T_melt_,
        interface_cutoff_min_, interface_cutoff_max_,
        nx_, ny_, nz_);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("MarangoniEffect: Kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

float MarangoniEffect::computeMarangoniVelocity(float delta_T,
                                                float viscosity) const
{
    // FIX BUG 4: Removed unused length_scale parameter
    // Characteristic Marangoni velocity: v ~ (dσ/dT * ΔT) / μ
    // This is an order-of-magnitude estimate for Marangoni-driven flow
    return std::abs(dsigma_dT_ * delta_T) / viscosity;
}

} // namespace physics
} // namespace lbm
