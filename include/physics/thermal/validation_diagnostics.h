#ifndef VALIDATION_DIAGNOSTICS_H
#define VALIDATION_DIAGNOSTICS_H

/**
 * @file validation_diagnostics.h
 * @brief Advanced diagnostic kernels for validation framework
 *
 * Purpose: Provide detailed field-level diagnostics to validate stability fixes:
 *   1. Peclet number field (identifies advection-dominated regions)
 *   2. Advection/Diffusion ratio field (quantifies local transport regime)
 *   3. Distribution function non-negativity check (validates flux limiter)
 *
 * These diagnostics prove whether fixes are "治本" (treating root cause)
 * or "治标" (superficial symptoms).
 *
 * Author: LBM-CUDA Architecture Team
 * Date: 2025-01-19
 */

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// KERNEL 1: Compute Peclet Number Field
// ============================================================================

/**
 * @brief Compute local Peclet number at each grid cell
 *
 * Peclet number: Pe(x,y,z) = |u| * dx / alpha
 *   - Pe >> 1: Advection-dominated (flux limiter should activate)
 *   - Pe ~ 1:  Balanced advection-diffusion
 *   - Pe << 1: Diffusion-dominated (flux limiter unnecessary)
 *
 * @param ux, uy, uz  Velocity components [m/s]
 * @param pe_field    Output: Peclet number at each cell (scalar field)
 * @param dx          Grid spacing [m]
 * @param alpha       Thermal diffusivity [m²/s]
 * @param nx, ny, nz  Grid dimensions
 */
__global__ void computePecletNumberField(
    const float* ux, const float* uy, const float* uz,
    float* pe_field,
    float dx, float alpha,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Compute velocity magnitude
    float u_mag = sqrtf(ux[idx] * ux[idx] +
                        uy[idx] * uy[idx] +
                        uz[idx] * uz[idx]);

    // Compute local Peclet number
    // Pe = u * L / alpha, where L = dx (characteristic length)
    float pe = (u_mag * dx) / alpha;

    pe_field[idx] = pe;
}

// ============================================================================
// KERNEL 2: Compute Advection/Diffusion Ratio Field
// ============================================================================

/**
 * @brief Compute ratio of advection to diffusion at each grid cell
 *
 * Ratio = |u·∇T| / |α·∇²T|
 *   - Ratio >> 1: Advection dominates (high risk of oscillations)
 *   - Ratio ~ 1:  Balanced transport
 *   - Ratio << 1: Diffusion dominates (stable, but may be over-dissipative)
 *
 * This is a LOCAL diagnostic (computed at each cell) to identify
 * problem regions spatially.
 *
 * @param T           Temperature field [K]
 * @param ux, uy, uz  Velocity components [m/s]
 * @param ratio       Output: Advection/Diffusion ratio at each cell
 * @param alpha       Thermal diffusivity [m²/s]
 * @param dx          Grid spacing [m]
 * @param nx, ny, nz  Grid dimensions
 */
__global__ void computeAdvectionDiffusionRatio(
    const float* T,
    const float* ux, const float* uy, const float* uz,
    float* ratio,
    float alpha, float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Interior cells only (need neighbors for gradients)
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1 || k <= 0 || k >= nz-1) {
        int idx = i + nx * (j + ny * k);
        ratio[idx] = 0.0f;
        return;
    }

    int idx = i + nx * (j + ny * k);

    // ========================================================================
    // Compute temperature gradients (central differences)
    // ========================================================================
    float dT_dx = (T[idx + 1] - T[idx - 1]) / (2.0f * dx);
    float dT_dy = (T[idx + nx] - T[idx - nx]) / (2.0f * dx);
    float dT_dz = (T[idx + nx*ny] - T[idx - nx*ny]) / (2.0f * dx);

    // ========================================================================
    // Compute advection term: |u·∇T|
    // ========================================================================
    float advection = fabsf(ux[idx] * dT_dx +
                            uy[idx] * dT_dy +
                            uz[idx] * dT_dz);

    // ========================================================================
    // Compute diffusion term: |α·∇²T|
    // ========================================================================
    // Laplacian using 7-point stencil:
    // ∇²T ≈ (T[i+1] + T[i-1] + T[j+1] + T[j-1] + T[k+1] + T[k-1] - 6*T[i,j,k]) / dx²
    float laplacian = (T[idx + 1] + T[idx - 1] +
                       T[idx + nx] + T[idx - nx] +
                       T[idx + nx*ny] + T[idx - nx*ny] -
                       6.0f * T[idx]) / (dx * dx);

    float diffusion = fabsf(alpha * laplacian);

    // ========================================================================
    // Compute ratio (with small epsilon to avoid division by zero)
    // ========================================================================
    float eps = 1e-10f;
    ratio[idx] = advection / (diffusion + eps);
}

// ============================================================================
// KERNEL 3: Check Distribution Function Non-Negativity
// ============================================================================

/**
 * @brief Check if thermal distribution functions g[i] are non-negative
 *
 * Theory: In LBM, distribution functions must satisfy g[i] >= 0 for all i.
 * Negative populations indicate numerical instability and non-physical states.
 *
 * The flux limiter's primary goal is to prevent negative populations.
 * This kernel verifies success.
 *
 * @param g           Thermal distribution functions [Q directions × grid cells]
 * @param min_g       Output: Minimum g value at each cell (should be >= 0)
 * @param violation_count  Output: Number of cells with min(g) < 0
 * @param Q           Number of lattice directions (e.g., 19 for D3Q19)
 * @param num_cells   Total number of grid cells
 */
__global__ void checkDistributionNonNegativity(
    const float* g,
    float* min_g,
    int* violation_count,
    int Q,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_cells) return;

    // Find minimum g across all directions at this cell
    float local_min = g[0 * num_cells + idx];  // Start with q=0

    for (int q = 1; q < Q; q++) {
        float g_q = g[q * num_cells + idx];
        if (g_q < local_min) {
            local_min = g_q;
        }
    }

    min_g[idx] = local_min;

    // Check for violation (with small tolerance for floating-point errors)
    const float tolerance = -1e-8f;
    if (local_min < tolerance) {
        atomicAdd(violation_count, 1);
    }
}

// ============================================================================
// KERNEL 4: Compute Global Minimum (Reduction)
// ============================================================================

/**
 * @brief Find global minimum of distribution functions
 *
 * This is a two-stage reduction:
 *   1. Each block finds its local minimum
 *   2. Host reduces block minima to global minimum
 *
 * @param g           Distribution functions
 * @param global_min  Output: Global minimum g value
 * @param Q           Number of lattice directions
 * @param num_cells   Total grid cells
 */
__global__ void findGlobalMinimum(
    const float* g,
    float* global_min,
    int Q,
    int num_cells)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory with large value
    sdata[tid] = 1e30f;

    // Each thread finds local minimum across all Q directions
    if (idx < num_cells) {
        float local_min = g[0 * num_cells + idx];
        for (int q = 1; q < Q; q++) {
            float g_q = g[q * num_cells + idx];
            if (g_q < local_min) local_min = g_q;
        }
        sdata[tid] = local_min;
    }

    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // First thread writes block result
    if (tid == 0) {
        atomicMin((int*)global_min, __float_as_int(sdata[0]));
    }
}

// ============================================================================
// HOST-SIDE WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Host wrapper to compute Peclet field and output to VTK
 */
inline void computeAndOutputPecletField(
    const float* d_ux, const float* d_uy, const float* d_uz,
    float* d_pe_field,
    float dx, float alpha,
    int nx, int ny, int nz)
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    computePecletNumberField<<<gridSize, blockSize>>>(
        d_ux, d_uy, d_uz,
        d_pe_field,
        dx, alpha,
        nx, ny, nz);

    cudaDeviceSynchronize();
}

/**
 * @brief Host wrapper to compute advection/diffusion ratio field
 */
inline void computeAndOutputAdvDiffRatio(
    const float* d_T,
    const float* d_ux, const float* d_uy, const float* d_uz,
    float* d_ratio,
    float alpha, float dx,
    int nx, int ny, int nz)
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    computeAdvectionDiffusionRatio<<<gridSize, blockSize>>>(
        d_T, d_ux, d_uy, d_uz,
        d_ratio,
        alpha, dx,
        nx, ny, nz);

    cudaDeviceSynchronize();
}

/**
 * @brief Host wrapper to check non-negativity and report violations
 *
 * @return Number of cells with negative distributions
 */
inline int checkAndReportNonNegativity(
    const float* d_g,
    float* d_min_g,
    int Q,
    int num_cells,
    float& global_min_g)
{
    // Allocate violation counter
    int* d_violation_count;
    cudaMalloc(&d_violation_count, sizeof(int));
    cudaMemset(d_violation_count, 0, sizeof(int));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_cells + blockSize - 1) / blockSize;

    checkDistributionNonNegativity<<<gridSize, blockSize>>>(
        d_g, d_min_g, d_violation_count,
        Q, num_cells);

    // Get violation count
    int h_violation_count;
    cudaMemcpy(&h_violation_count, d_violation_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Find global minimum
    float* d_global_min;
    cudaMalloc(&d_global_min, sizeof(float));
    float init_val = 1e30f;
    cudaMemcpy(d_global_min, &init_val, sizeof(float), cudaMemcpyHostToDevice);

    findGlobalMinimum<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_g, d_global_min, Q, num_cells);

    cudaMemcpy(&global_min_g, d_global_min, sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_violation_count);
    cudaFree(d_global_min);

    return h_violation_count;
}

// ============================================================================
// DIAGNOSTIC REPORTING
// ============================================================================

/**
 * @brief Print comprehensive validation diagnostics report
 */
inline void printValidationDiagnostics(
    int violation_count,
    float min_g_global,
    float max_peclet,
    float max_adv_diff_ratio,
    int timestep)
{
    printf("\n");
    printf("========================================================================\n");
    printf("  VALIDATION DIAGNOSTICS - Timestep %d\n", timestep);
    printf("========================================================================\n");
    printf("\n");

    // Non-negativity check
    printf("Distribution Function Non-Negativity:\n");
    printf("  Global min(g):       %.6e\n", min_g_global);
    printf("  Violation count:     %d cells\n", violation_count);
    if (violation_count == 0 && min_g_global >= 0.0f) {
        printf("  Status:              ✓ PASS (all g >= 0)\n");
    } else {
        printf("  Status:              ✗ FAIL (negative populations detected)\n");
    }
    printf("\n");

    // Peclet number
    printf("Peclet Number Analysis:\n");
    printf("  Max Peclet:          %.2f\n", max_peclet);
    if (max_peclet > 10.0f) {
        printf("  Regime:              ADVECTION-DOMINATED (flux limiter needed)\n");
    } else if (max_peclet > 1.0f) {
        printf("  Regime:              BALANCED\n");
    } else {
        printf("  Regime:              DIFFUSION-DOMINATED\n");
    }
    printf("\n");

    // Advection/Diffusion ratio
    printf("Advection/Diffusion Ratio:\n");
    printf("  Max ratio:           %.2f\n", max_adv_diff_ratio);
    if (max_adv_diff_ratio > 10.0f) {
        printf("  Status:              High-gradient regions detected\n");
        printf("                       → Flux limiter should be active\n");
    } else {
        printf("  Status:              Gradients under control\n");
    }
    printf("\n");

    printf("========================================================================\n");
    printf("\n");
}

#endif // VALIDATION_DIAGNOSTICS_H
