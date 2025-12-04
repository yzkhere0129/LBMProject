/**
 * @file energy_balance.cu
 * @brief CUDA implementation of energy balance computation kernels
 */

#include "diagnostics/energy_balance.h"
#include <cuda_runtime.h>
#include <stdio.h>

namespace lbm {
namespace diagnostics {

// ============================================================================
// atomicAdd for double (compute capability < 6.0)
// ============================================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native double atomicAdd available (compute capability >= 6.0)
#else
// Emulated double atomicAdd using CAS (compare-and-swap) for older GPUs
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * @brief Initialize reduction output to zero
 */
__global__ void initializeReductionKernel(double* d_result) {
    *d_result = 0.0;
}

// ============================================================================
// Energy Computation Kernels
// ============================================================================

/**
 * @brief Compute thermal energy per cell and accumulate
 *
 * Thermal energy: E_thermal = ∫ ρ c_p(T) (T - T_ref) dV
 * where c_p(T) depends on liquid fraction for mushy zone
 *
 * BUG FIX (Dec 2, 2025): Use (T - T_ref) for sensible energy
 *
 * The sensible energy must be computed relative to a reference temperature,
 * not absolute temperature. Using absolute T causes energy conservation
 * violations when computing dE/dt.
 *
 * Reference temperature: Typically T_initial or T_solidus
 */
__global__ void computeThermalEnergyKernel(
    const float* T,
    const float* f_liquid,
    float rho,
    float cp_solid,
    float cp_liquid,
    float dx,
    float T_ref,
    int nx, int ny, int nz,
    double* d_result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Temperature [K]
    float temperature = T[idx];

    // Liquid fraction [0-1]
    float fl = f_liquid[idx];

    // Specific heat: linear interpolation based on phase
    float cp = cp_solid * (1.0f - fl) + cp_liquid * fl;

    // Volume element [m³]
    float dV = dx * dx * dx;

    // Thermal energy in this cell: E = ρ c_p (T - T_ref) dV [J]
    // FIXED: Use (T - T_ref) instead of absolute T
    // This prevents artificial energy from constant baseline temperature
    // Note: This is sensible heat only (latent heat computed separately)
    double E_cell = rho * cp * (temperature - T_ref) * dV;

    // Atomic accumulation (global memory)
    atomicAdd(d_result, E_cell);
}

/**
 * @brief Compute kinetic energy per cell and accumulate
 *
 * Kinetic energy: E_kinetic = ∫ 0.5 ρ |u|² dV
 */
__global__ void computeKineticEnergyKernel(
    const float* ux,
    const float* uy,
    const float* uz,
    float rho,
    float dx,
    int nx, int ny, int nz,
    double* d_result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Velocity components [m/s]
    float u = ux[idx];
    float v = uy[idx];
    float w = uz[idx];

    // Velocity magnitude squared [m²/s²]
    float u2 = u*u + v*v + w*w;

    // Volume element [m³]
    float dV = dx * dx * dx;

    // Kinetic energy in this cell: E = 0.5 ρ |u|² dV [J]
    double E_cell = 0.5 * rho * u2 * dV;

    // Atomic accumulation
    atomicAdd(d_result, E_cell);
}

/**
 * @brief Compute latent heat energy per cell and accumulate
 *
 * Latent energy: E_latent = ∫ ρ L_f f_liquid dV
 *
 * This represents the energy stored in phase change:
 * - f_liquid = 0: fully solid, no latent energy stored
 * - f_liquid = 1: fully liquid, maximum latent energy stored
 */
__global__ void computeLatentEnergyKernel(
    const float* f_liquid,
    float rho,
    float L_fusion,
    float dx,
    int nx, int ny, int nz,
    double* d_result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Liquid fraction [0-1]
    float fl = f_liquid[idx];

    // Volume element [m³]
    float dV = dx * dx * dx;

    // Latent energy in this cell: E = ρ L_f f_liquid dV [J]
    double E_cell = rho * L_fusion * fl * dV;

    // Atomic accumulation
    atomicAdd(d_result, E_cell);
}

// ============================================================================
// Host Interface Functions
// ============================================================================

void computeThermalEnergy(
    const float* T,
    const float* f_liquid,
    float rho,
    float cp_solid,
    float cp_liquid,
    float dx,
    float T_ref,
    int nx, int ny, int nz,
    double* d_result)
{
    // Initialize result to zero
    initializeReductionKernel<<<1, 1>>>(d_result);

    // Launch kernel with 3D grid
    dim3 block(8, 8, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );

    computeThermalEnergyKernel<<<grid, block>>>(
        T, f_liquid, rho, cp_solid, cp_liquid, dx, T_ref,
        nx, ny, nz, d_result
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in computeThermalEnergy: %s\n",
                cudaGetErrorString(err));
    }
}

void computeKineticEnergy(
    const float* ux,
    const float* uy,
    const float* uz,
    float rho,
    float dx,
    int nx, int ny, int nz,
    double* d_result)
{
    // Initialize result to zero
    initializeReductionKernel<<<1, 1>>>(d_result);

    // Launch kernel with 3D grid
    dim3 block(8, 8, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );

    computeKineticEnergyKernel<<<grid, block>>>(
        ux, uy, uz, rho, dx,
        nx, ny, nz, d_result
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in computeKineticEnergy: %s\n",
                cudaGetErrorString(err));
    }
}

void computeLatentEnergy(
    const float* f_liquid,
    float rho,
    float L_fusion,
    float dx,
    int nx, int ny, int nz,
    double* d_result)
{
    // Initialize result to zero
    initializeReductionKernel<<<1, 1>>>(d_result);

    // Launch kernel with 3D grid
    dim3 block(8, 8, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );

    computeLatentEnergyKernel<<<grid, block>>>(
        f_liquid, rho, L_fusion, dx,
        nx, ny, nz, d_result
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in computeLatentEnergy: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace diagnostics
} // namespace lbm
