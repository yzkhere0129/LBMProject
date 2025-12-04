/**
 * @file recoil_pressure.cu
 * @brief Implementation of recoil pressure for evaporating metal surfaces (Anisimov model)
 *
 * Physical model:
 *   P_recoil = C_r * P_sat(T)
 *   P_sat(T) = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]
 *
 * The recoil force pushes the liquid surface downward due to momentum
 * conservation during rapid evaporation. This is the primary driver of
 * keyhole formation in high-power laser welding and LPBF.
 *
 * Force application (CSF format):
 *   F_recoil = -P_recoil * n * |grad(f)|
 *   where n points from liquid into gas (outward from liquid surface)
 *
 * References:
 * - Anisimov, S. I. (1968): Vaporization of metals by laser radiation
 * - Knight, C. J. (1979): AIAA Journal 17(5), 519-523
 * - Khairallah, S. A. et al. (2016): Acta Materialia 108, 36-45
 */

#include "physics/recoil_pressure.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace lbm {
namespace physics {

// ============================================================================
// Physical Constants for Ti6Al4V (Anisimov model)
// ============================================================================

namespace {
    constexpr float P_REF = 101325.0f;      // Reference pressure [Pa] (1 atm)
    constexpr float L_VAP = 8.878e6f;       // Latent heat of vaporization [J/kg]
    constexpr float M_MOLAR = 0.0479f;      // Molar mass [kg/mol]
    constexpr float R_GAS = 8.314f;         // Universal gas constant [J/(mol.K)]
    constexpr float T_BOIL = 3560.0f;       // Boiling temperature [K] (ASM Handbook)

    // Precomputed Clausius-Clapeyron exponent factor: L_vap * M / R
    constexpr float CC_FACTOR = L_VAP * M_MOLAR / R_GAS;  // ~51096 K

    // Temperature threshold for recoil activation (T_boil - 500K)
    // Pre-boiling evaporation occurs due to vapor pressure buildup
    constexpr float T_ACTIVATION = T_BOIL - 500.0f;  // 3060 K
}

// ============================================================================
// Device Functions
// ============================================================================

/**
 * @brief Compute saturation pressure using Clausius-Clapeyron equation
 * @param T Temperature [K]
 * @return Saturation pressure [Pa]
 */
__device__ __forceinline__ float computeSaturationPressure(float T) {
    if (T < T_ACTIVATION) {
        return 0.0f;
    }

    // P_sat = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]
    float exponent = CC_FACTOR * (1.0f / T_BOIL - 1.0f / T);
    return P_REF * expf(exponent);
}

/**
 * @brief Compute recoil pressure from temperature
 * @param T Temperature [K]
 * @param C_r Recoil coefficient (0.54 typical)
 * @param P_max Maximum pressure limiter [Pa]
 * @return Recoil pressure [Pa]
 */
__device__ __forceinline__ float computeRecoilPressure(float T, float C_r, float P_max) {
    float p_sat = computeSaturationPressure(T);
    float P_r = C_r * p_sat;
    return fminf(P_r, P_max);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Compute recoil pressure field from saturation pressure
 *
 * Simple kernel: P_recoil = C_r * p_sat, with limiter
 */
__global__ void computeRecoilPressureKernel(
    const float* __restrict__ saturation_pressure,
    float* __restrict__ recoil_pressure,
    float C_r,
    float P_max,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float p_sat = saturation_pressure[idx];
    float P_r = C_r * p_sat;
    recoil_pressure[idx] = fminf(P_r, P_max);
}

/**
 * @brief Compute recoil force field (volumetric body force)
 *
 * Force direction: INTO the liquid (along -n where n points liquid->gas)
 * Force magnitude: P_recoil * |grad(f)| / h_interface
 *
 * Applied only at interface cells (0.01 < fill_level < 0.99) with T > T_activation
 */
__global__ void computeRecoilForceKernel(
    const float* __restrict__ saturation_pressure,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float C_r,
    float h_interface,
    float dx,
    float P_max,
    float f_threshold,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Initialize force to zero
    force_x[idx] = 0.0f;
    force_y[idx] = 0.0f;
    force_z[idx] = 0.0f;

    // Check interface condition: 0.01 < fill_level < 0.99
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        return;
    }

    // Get saturation pressure (already computed externally, includes T threshold)
    float p_sat = saturation_pressure[idx];
    if (p_sat < 1.0f) {
        return;  // No significant vapor pressure
    }

    // Compute recoil pressure with limiter
    float P_r = fminf(C_r * p_sat, P_max);

    // Compute VOF gradient magnitude using central differences
    int im = max(0, i - 1);
    int ip = min(nx - 1, i + 1);
    int jm = max(0, j - 1);
    int jp = min(ny - 1, j + 1);
    int km = max(0, k - 1);
    int kp = min(nz - 1, k + 1);

    float f_xm = fill_level[im + nx * (j + ny * k)];
    float f_xp = fill_level[ip + nx * (j + ny * k)];
    float f_ym = fill_level[i + nx * (jm + ny * k)];
    float f_yp = fill_level[i + nx * (jp + ny * k)];
    float f_zm = fill_level[i + nx * (j + ny * km)];
    float f_zp = fill_level[i + nx * (j + ny * kp)];

    float grad_f_x = (f_xp - f_xm) / (2.0f * dx);
    float grad_f_y = (f_yp - f_ym) / (2.0f * dx);
    float grad_f_z = (f_zp - f_zm) / (2.0f * dx);

    float grad_f_mag = sqrtf(grad_f_x * grad_f_x +
                             grad_f_y * grad_f_y +
                             grad_f_z * grad_f_z);

    // Check gradient threshold
    if (grad_f_mag < f_threshold) {
        return;
    }

    // Get interface normal (points from liquid into gas)
    float3 n = interface_normal[idx];

    // Recoil force (CSF formulation): F = -P_r * n * |grad(f)|
    // The negative sign makes force point INTO liquid (opposite to vapor direction)
    // |grad(f)| acts as delta function, converting surface pressure to volumetric force
    // Units: [Pa] * [1/m] = [N/m³]
    float coeff = -P_r * grad_f_mag;

    force_x[idx] = coeff * n.x;
    force_y[idx] = coeff * n.y;
    force_z[idx] = coeff * n.z;
}

/**
 * @brief Add recoil force to existing force arrays
 *
 * Same logic as computeRecoilForceKernel, but adds to existing forces
 * instead of overwriting.
 */
__global__ void addRecoilForceKernel(
    const float* __restrict__ saturation_pressure,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float C_r,
    float h_interface,
    float dx,
    float P_max,
    float f_threshold,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Check interface condition
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        return;
    }

    // Get saturation pressure
    float p_sat = saturation_pressure[idx];
    if (p_sat < 1.0f) {
        return;
    }

    // Compute recoil pressure
    float P_r = fminf(C_r * p_sat, P_max);

    // Compute VOF gradient magnitude
    int im = max(0, i - 1);
    int ip = min(nx - 1, i + 1);
    int jm = max(0, j - 1);
    int jp = min(ny - 1, j + 1);
    int km = max(0, k - 1);
    int kp = min(nz - 1, k + 1);

    float f_xm = fill_level[im + nx * (j + ny * k)];
    float f_xp = fill_level[ip + nx * (j + ny * k)];
    float f_ym = fill_level[i + nx * (jm + ny * k)];
    float f_yp = fill_level[i + nx * (jp + ny * k)];
    float f_zm = fill_level[i + nx * (j + ny * km)];
    float f_zp = fill_level[i + nx * (j + ny * kp)];

    float grad_f_x = (f_xp - f_xm) / (2.0f * dx);
    float grad_f_y = (f_yp - f_ym) / (2.0f * dx);
    float grad_f_z = (f_zp - f_zm) / (2.0f * dx);

    float grad_f_mag = sqrtf(grad_f_x * grad_f_x +
                             grad_f_y * grad_f_y +
                             grad_f_z * grad_f_z);

    if (grad_f_mag < f_threshold) {
        return;
    }

    // Get interface normal
    float3 n = interface_normal[idx];

    // Add recoil force (CSF formulation, negative: into liquid)
    // Units: [Pa] * [1/m] = [N/m³]
    float coeff = -P_r * grad_f_mag;

    force_x[idx] += coeff * n.x;
    force_y[idx] += coeff * n.y;
    force_z[idx] += coeff * n.z;
}

/**
 * @brief Combined kernel: compute P_sat from temperature and apply recoil force
 *
 * This fused kernel computes saturation pressure directly from temperature
 * and applies the recoil force in a single pass, avoiding intermediate storage.
 *
 * Activation criteria:
 * - Interface cell: 0.01 < fill_level < 0.99
 * - High temperature: T > T_boil - 500K
 */
__global__ void computeRecoilForceFromTemperatureKernel(
    const float* __restrict__ temperature,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float C_r,
    float h_interface,
    float dx,
    float P_max,
    float f_threshold,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Initialize force to zero
    force_x[idx] = 0.0f;
    force_y[idx] = 0.0f;
    force_z[idx] = 0.0f;

    // Check interface condition
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        return;
    }

    // Check temperature threshold
    float T = temperature[idx];
    if (T < T_ACTIVATION) {
        return;
    }

    // Compute saturation pressure (Clausius-Clapeyron)
    float exponent = CC_FACTOR * (1.0f / T_BOIL - 1.0f / T);
    float p_sat = P_REF * expf(exponent);

    // Compute recoil pressure with limiter
    float P_r = fminf(C_r * p_sat, P_max);

    // Compute VOF gradient magnitude
    int im = max(0, i - 1);
    int ip = min(nx - 1, i + 1);
    int jm = max(0, j - 1);
    int jp = min(ny - 1, j + 1);
    int km = max(0, k - 1);
    int kp = min(nz - 1, k + 1);

    float f_xm = fill_level[im + nx * (j + ny * k)];
    float f_xp = fill_level[ip + nx * (j + ny * k)];
    float f_ym = fill_level[i + nx * (jm + ny * k)];
    float f_yp = fill_level[i + nx * (jp + ny * k)];
    float f_zm = fill_level[i + nx * (j + ny * km)];
    float f_zp = fill_level[i + nx * (j + ny * kp)];

    float grad_f_x = (f_xp - f_xm) / (2.0f * dx);
    float grad_f_y = (f_yp - f_ym) / (2.0f * dx);
    float grad_f_z = (f_zp - f_zm) / (2.0f * dx);

    float grad_f_mag = sqrtf(grad_f_x * grad_f_x +
                             grad_f_y * grad_f_y +
                             grad_f_z * grad_f_z);

    if (grad_f_mag < f_threshold) {
        return;
    }

    // Get interface normal
    float3 n = interface_normal[idx];

    // Recoil force (CSF formulation): F = -P_r * n * |grad(f)|
    // Units: [Pa] * [1/m] = [N/m³]
    float coeff = -P_r * grad_f_mag;

    force_x[idx] = coeff * n.x;
    force_y[idx] = coeff * n.y;
    force_z[idx] = coeff * n.z;
}

/**
 * @brief Add recoil force from temperature to existing force arrays
 */
__global__ void addRecoilForceFromTemperatureKernel(
    const float* __restrict__ temperature,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float C_r,
    float h_interface,
    float dx,
    float P_max,
    float f_threshold,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Check interface condition
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        return;
    }

    // Check temperature threshold
    float T = temperature[idx];
    if (T < T_ACTIVATION) {
        return;
    }

    // Compute saturation pressure
    float exponent = CC_FACTOR * (1.0f / T_BOIL - 1.0f / T);
    float p_sat = P_REF * expf(exponent);

    // Compute recoil pressure
    float P_r = fminf(C_r * p_sat, P_max);

    // Compute VOF gradient magnitude
    int im = max(0, i - 1);
    int ip = min(nx - 1, i + 1);
    int jm = max(0, j - 1);
    int jp = min(ny - 1, j + 1);
    int km = max(0, k - 1);
    int kp = min(nz - 1, k + 1);

    float f_xm = fill_level[im + nx * (j + ny * k)];
    float f_xp = fill_level[ip + nx * (j + ny * k)];
    float f_ym = fill_level[i + nx * (jm + ny * k)];
    float f_yp = fill_level[i + nx * (jp + ny * k)];
    float f_zm = fill_level[i + nx * (j + ny * km)];
    float f_zp = fill_level[i + nx * (j + ny * kp)];

    float grad_f_x = (f_xp - f_xm) / (2.0f * dx);
    float grad_f_y = (f_yp - f_ym) / (2.0f * dx);
    float grad_f_z = (f_zp - f_zm) / (2.0f * dx);

    float grad_f_mag = sqrtf(grad_f_x * grad_f_x +
                             grad_f_y * grad_f_y +
                             grad_f_z * grad_f_z);

    if (grad_f_mag < f_threshold) {
        return;
    }

    // Get interface normal
    float3 n = interface_normal[idx];

    // Add recoil force (CSF formulation)
    // Units: [Pa] * [1/m] = [N/m³]
    float coeff = -P_r * grad_f_mag;

    force_x[idx] += coeff * n.x;
    force_y[idx] += coeff * n.y;
    force_z[idx] += coeff * n.z;
}

// ============================================================================
// RecoilPressure Class Implementation
// ============================================================================

RecoilPressure::RecoilPressure(const RecoilPressureConfig& config, float dx)
    : C_r_(config.coefficient),
      h_interface_(config.smoothing_width),
      dx_(dx),
      P_max_(config.max_pressure),
      f_threshold_(config.fill_level_threshold)
{
}

RecoilPressure::RecoilPressure(float recoil_coefficient,
                               float smoothing_width,
                               float dx)
    : C_r_(recoil_coefficient),
      h_interface_(smoothing_width),
      dx_(dx),
      P_max_(1e8f),         // Default: 100 MPa max
      f_threshold_(0.01f)   // Default: 1% fill level threshold
{
}

void RecoilPressure::computePressureField(
    const float* saturation_pressure,
    float* recoil_pressure,
    int nx, int ny, int nz) const
{
    int num_cells = nx * ny * nz;
    int blockSize = 256;
    int gridSize = (num_cells + blockSize - 1) / blockSize;

    computeRecoilPressureKernel<<<gridSize, blockSize>>>(
        saturation_pressure,
        recoil_pressure,
        C_r_,
        P_max_,
        num_cells);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("RecoilPressure::computePressureField kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}

void RecoilPressure::computeForceField(
    const float* saturation_pressure,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    int nx, int ny, int nz) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    computeRecoilForceKernel<<<gridSize, blockSize>>>(
        saturation_pressure,
        fill_level,
        interface_normal,
        force_x, force_y, force_z,
        C_r_, h_interface_, dx_, P_max_, f_threshold_,
        nx, ny, nz);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("RecoilPressure::computeForceField kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}

void RecoilPressure::addForceField(
    const float* saturation_pressure,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    int nx, int ny, int nz) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    addRecoilForceKernel<<<gridSize, blockSize>>>(
        saturation_pressure,
        fill_level,
        interface_normal,
        force_x, force_y, force_z,
        C_r_, h_interface_, dx_, P_max_, f_threshold_,
        nx, ny, nz);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("RecoilPressure::addForceField kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}

void RecoilPressure::addForceFromTemperature(
    const float* temperature,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    int nx, int ny, int nz) const
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    addRecoilForceFromTemperatureKernel<<<gridSize, blockSize>>>(
        temperature,
        fill_level,
        interface_normal,
        force_x, force_y, force_z,
        C_r_, h_interface_, dx_, P_max_, f_threshold_,
        nx, ny, nz);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("RecoilPressure::addForceFromTemperature kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();
}

} // namespace physics
} // namespace lbm
