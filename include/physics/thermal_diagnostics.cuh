/**
 * @file thermal_diagnostics.cuh
 * @brief CUDA kernels for thermal field diagnostics and debugging
 */

#ifndef THERMAL_DIAGNOSTICS_CUH
#define THERMAL_DIAGNOSTICS_CUH

#include <cuda_runtime.h>

/**
 * @brief Extract vertical temperature profile at (x,y) position
 *
 * Extracts temperature values along a vertical line (z-direction)
 * at a specified (x,y) position. Useful for debugging thermal
 * boundary conditions and identifying temperature inversions.
 *
 * Usage:
 *   int threads = 256;
 *   int blocks = (nz + threads - 1) / threads;
 *   extractTemperatureProfile<<<blocks, threads>>>(
 *       d_temperature, d_T_profile, nx, ny, nz, nx/2, ny/2);
 *
 * @param temperature Temperature field [K] (size: nx × ny × nz)
 * @param T_profile Output: temperature vs z [K] (size: nz)
 * @param nx, ny, nz Grid dimensions
 * @param x_center X-coordinate for vertical slice
 * @param y_center Y-coordinate for vertical slice
 */
__global__ void extractTemperatureProfile(
    const float* temperature,
    float* T_profile,
    int nx, int ny, int nz,
    int x_center, int y_center)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= nz) return;

    // Compute linear index: idx = x + nx*(y + ny*z)
    int idx = x_center + nx * (y_center + ny * z);
    T_profile[z] = temperature[idx];
}

/**
 * @brief Extract 2D temperature slice at constant z
 *
 * Extracts a horizontal temperature slice at a given z-level.
 * Useful for visualizing laser spot and thermal gradients.
 *
 * @param temperature Temperature field [K] (size: nx × ny × nz)
 * @param T_slice Output: temperature on xy-plane [K] (size: nx × ny)
 * @param nx, ny, nz Grid dimensions
 * @param z_level Z-coordinate for horizontal slice
 */
__global__ void extractTemperatureSlice(
    const float* temperature,
    float* T_slice,
    int nx, int ny, int nz,
    int z_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // 3D index at z_level
    int idx_3d = i + nx * (j + ny * z_level);
    // 2D index in output
    int idx_2d = i + nx * j;

    T_slice[idx_2d] = temperature[idx_3d];
}

/**
 * @brief Compute temperature statistics in a region
 *
 * Computes min, max, and average temperature in a specified region.
 * Uses shared memory reduction for efficiency.
 *
 * @param temperature Temperature field [K]
 * @param stats Output: [min, max, avg] (size: 3)
 * @param nx, ny, nz Grid dimensions
 * @param x0, y0, z0 Region start coordinates
 * @param x1, y1, z1 Region end coordinates (exclusive)
 */
__global__ void computeTemperatureStats(
    const float* temperature,
    float* stats,  // [min, max, sum]
    int* count,
    int nx, int ny, int nz,
    int x0, int y0, int z0,
    int x1, int y1, int z1)
{
    extern __shared__ float sdata[];
    float* s_min = sdata;
    float* s_max = sdata + blockDim.x;
    float* s_sum = sdata + 2 * blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute 3D coordinates
    int total_cells = (x1 - x0) * (y1 - y0) * (z1 - z0);
    if (idx >= total_cells) {
        s_min[tid] = 1e30f;
        s_max[tid] = -1e30f;
        s_sum[tid] = 0.0f;
    } else {
        int local_x = idx % (x1 - x0);
        int local_y = (idx / (x1 - x0)) % (y1 - y0);
        int local_z = idx / ((x1 - x0) * (y1 - y0));

        int x = x0 + local_x;
        int y = y0 + local_y;
        int z = z0 + local_z;

        int global_idx = x + nx * (y + ny * z);
        float T = temperature[global_idx];

        s_min[tid] = T;
        s_max[tid] = T;
        s_sum[tid] = T;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicMin((int*)&stats[0], __float_as_int(s_min[0]));
        atomicMax((int*)&stats[1], __float_as_int(s_max[0]));
        atomicAdd(&stats[2], s_sum[0]);
        atomicAdd(count, blockDim.x < total_cells ? blockDim.x : total_cells - blockIdx.x * blockDim.x);
    }
}

/**
 * @brief Check for temperature inversion at surface
 *
 * Compares surface temperature (z=nz-1) with subsurface (z=nz-2, nz-3).
 * Returns number of cells where T_surface < T_subsurface (inversion).
 *
 * @param temperature Temperature field [K]
 * @param inversion_count Output: number of inversion cells
 * @param nx, ny, nz Grid dimensions
 */
__global__ void detectTemperatureInversion(
    const float* temperature,
    int* inversion_count,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Surface cell (z = nz-1)
    int idx_surface = i + nx * (j + ny * (nz - 1));
    float T_surface = temperature[idx_surface];

    // Subsurface cell (z = nz-3, skip nz-2 which might be transitional)
    if (nz < 4) return;  // Need at least 4 layers
    int idx_subsurface = i + nx * (j + ny * (nz - 3));
    float T_subsurface = temperature[idx_subsurface];

    // Check for inversion
    if (T_surface < T_subsurface) {
        atomicAdd(inversion_count, 1);
    }
}

/**
 * @brief Extract laser heat source profile (for validation)
 *
 * @param heat_source Volumetric heat source [W/m³]
 * @param Q_profile Output: heat source vs z [W/m³]
 * @param nx, ny, nz Grid dimensions
 * @param x_center, y_center Position for vertical slice
 */
__global__ void extractHeatSourceProfile(
    const float* heat_source,
    float* Q_profile,
    int nx, int ny, int nz,
    int x_center, int y_center)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= nz) return;

    int idx = x_center + nx * (y_center + ny * z);
    Q_profile[z] = heat_source[idx];
}

/**
 * @brief Compute radiation cooling power at surface
 *
 * Computes Stefan-Boltzmann radiation heat loss at each surface cell.
 * For validation and debugging of radiation BC.
 *
 * @param temperature Temperature field [K]
 * @param radiation_flux Output: radiation heat flux [W/m²] (size: nx × ny)
 * @param nx, ny, nz Grid dimensions
 * @param epsilon Emissivity [0-1]
 * @param T_ambient Ambient temperature [K]
 */
__global__ void computeRadiationFlux(
    const float* temperature,
    float* radiation_flux,
    int nx, int ny, int nz,
    float epsilon,
    float T_ambient)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Surface temperature (z = nz-1)
    int idx_3d = i + nx * (j + ny * (nz - 1));
    float T_surf = temperature[idx_3d];

    // Stefan-Boltzmann radiation law
    const float sigma = 5.67e-8f;  // W/(m²·K⁴)
    float q_rad = epsilon * sigma * (powf(T_surf, 4.0f) - powf(T_ambient, 4.0f));

    // Output (2D index)
    int idx_2d = i + nx * j;
    radiation_flux[idx_2d] = q_rad;
}

#endif // THERMAL_DIAGNOSTICS_CUH
