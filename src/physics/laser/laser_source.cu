/**
 * @file laser_source.cu
 * @brief CUDA implementation of laser heat source kernels
 */

#include "physics/laser_source.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

/**
 * @brief Compute laser heat source distribution on GPU
 *
 * Each thread computes the heat source for one grid cell.
 * The heat source follows the Beer-Lambert law with Gaussian surface profile.
 */
__global__ void computeLaserHeatSourceKernel(
    float* heat_source,
    const LaserSource laser,
    float dx, float dy, float dz,
    int nx, int ny, int nz)
{
    // Global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (i >= nx || j >= ny || k >= nz) return;

    // Calculate physical position
    float x = i * dx;
    float y = j * dy;
    float z = k * dz;

    // Compute volumetric heat source
    float q = laser.computeVolumetricHeatSource(x, y, z);

    // Store result
    int idx = k * nx * ny + j * nx + i;
    heat_source[idx] = q;
}

/**
 * @brief Add heat source to thermal field
 *
 * Updates the equilibrium distribution (f0) to account for heat source.
 * In LBM thermal model, heat source appears as a source term in f0.
 *
 * Temperature change: ΔT = Q * dt / (ρ * cp)
 * Where Q is volumetric heat source [W/m³]
 */
__global__ void addHeatSourceToThermalFieldKernel(
    float* g_distributions,
    const float* heat_source,
    float dt,
    float rho, float cp,
    int nx, int ny, int nz)
{
    // Global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = k * nx * ny + j * nx + i;

    // Get heat source at this cell
    float Q = heat_source[idx];

    // Convert to temperature increment
    // ΔT = Q * dt / (ρ * cp)
    float deltaT = Q * dt / (rho * cp);

    // In D3Q7 thermal model, temperature is sum of all distributions
    // We add the temperature change to the equilibrium (f0) distribution
    // Since w0 = 1/4 for D3Q7, we need to scale appropriately

    // For D3Q7: f0 = w0 * T = (1/4) * T
    // So we add (1/4) * deltaT to f0
    float delta_f0 = 0.25f * deltaT;

    // Index for f0 (q=0) in the distribution array
    // Assuming memory layout: [q0, q1, q2, q3, q4, q5, q6] for each cell
    int f0_idx = idx * 7;  // First component (q=0)

    // Update the equilibrium distribution
    g_distributions[f0_idx] += delta_f0;
}

/**
 * @brief Compute total laser energy in domain (host function)
 *
 * This function is called from host to verify energy conservation.
 * It sums up all volumetric heat sources and multiplies by cell volume.
 */
float computeTotalLaserEnergy(
    const float* heat_source,
    float dx, float dy, float dz,
    int nx, int ny, int nz)
{
    size_t total_cells = nx * ny * nz;
    float dV = dx * dy * dz;

    // Use CUB for efficient reduction on GPU
    size_t temp_storage_bytes = 0;
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));

    // Get required temporary storage size
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes,
                           heat_source, d_sum, total_cells);

    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           heat_source, d_sum, total_cells);

    // Copy result back to host
    float total_heat_rate;
    cudaMemcpy(&total_heat_rate, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Free temporary storage
    cudaFree(d_temp_storage);
    cudaFree(d_sum);

    // Convert to total power [W]
    return total_heat_rate * dV;
}

/**
 * @brief Alternative implementation without CUB (for compatibility)
 *
 * Simple reduction kernel for summing heat source values
 */
__global__ void sumReductionKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Load data to shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/**
 * @brief Alternative total energy computation without CUB
 */
float computeTotalLaserEnergyNoCUB(
    const float* heat_source,
    float dx, float dy, float dz,
    int nx, int ny, int nz)
{
    size_t total_cells = nx * ny * nz;
    float dV = dx * dy * dz;

    // Configure reduction
    const int threads = 256;
    const int blocks = (total_cells + threads - 1) / threads;

    // Allocate memory for partial sums
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));

    // First reduction
    sumReductionKernel<<<blocks, threads, threads * sizeof(float)>>>(
        heat_source, d_partial_sums, total_cells);

    // If we have multiple blocks, need another reduction
    if (blocks > 1) {
        float* d_final_sum;
        cudaMalloc(&d_final_sum, sizeof(float));

        sumReductionKernel<<<1, blocks, blocks * sizeof(float)>>>(
            d_partial_sums, d_final_sum, blocks);

        float total_heat_rate;
        cudaMemcpy(&total_heat_rate, d_final_sum, sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_final_sum);
        cudaFree(d_partial_sums);

        return total_heat_rate * dV;
    } else {
        float total_heat_rate;
        cudaMemcpy(&total_heat_rate, d_partial_sums, sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_partial_sums);

        return total_heat_rate * dV;
    }
}

/**
 * @brief Optimized kernel using shared memory for laser parameters
 *
 * This version loads laser parameters into shared memory once per block
 * to reduce global memory accesses.
 */
__global__ void computeLaserHeatSourceOptimizedKernel(
    float* heat_source,
    const LaserSource laser,
    float dx, float dy, float dz,
    int nx, int ny, int nz)
{
    // For now, just use the laser parameter directly from global memory
    // Shared memory optimization can be added later if needed
    const LaserSource& s_laser = laser;

    // Global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (i >= nx || j >= ny || k >= nz) return;

    // Calculate physical position
    float x = i * dx;
    float y = j * dy;
    float z = k * dz;

    // Early exit if far from laser center (optimization)
    float dx_laser = x - s_laser.x0;
    float dy_laser = y - s_laser.y0;
    float r2 = dx_laser * dx_laser + dy_laser * dy_laser;

    // Skip if more than 5 spot radii away
    if (r2 > 25.0f * s_laser.spot_radius * s_laser.spot_radius) {
        int idx = k * nx * ny + j * nx + i;
        heat_source[idx] = 0.0f;
        return;
    }

    // Compute volumetric heat source
    float q = s_laser.computeVolumetricHeatSource(x, y, z);

    // Store result
    int idx = k * nx * ny + j * nx + i;
    heat_source[idx] = q;
}

/**
 * @brief Apply multiple laser sources simultaneously
 *
 * Useful for multi-beam or multi-track simulations
 */
__global__ void computeMultipleLasersKernel(
    float* heat_source,
    const LaserSource* lasers,
    int num_lasers,
    float dx, float dy, float dz,
    int nx, int ny, int nz)
{
    // Global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (i >= nx || j >= ny || k >= nz) return;

    // Calculate physical position
    float x = i * dx;
    float y = j * dy;
    float z = k * dz;

    // Sum contributions from all lasers
    float total_q = 0.0f;
    for (int laser_id = 0; laser_id < num_lasers; ++laser_id) {
        total_q += lasers[laser_id].computeVolumetricHeatSource(x, y, z);
    }

    // Store result
    int idx = k * nx * ny + j * nx + i;
    heat_source[idx] = total_q;
}

/**
 * @brief Update thermal field with adaptive time stepping
 *
 * Adjusts the time step based on the maximum heat source to maintain stability
 */
__global__ void addHeatSourceAdaptiveKernel(
    float* g_distributions,
    const float* heat_source,
    float base_dt,
    float rho, float cp,
    float max_delta_T,  // Maximum allowed temperature change per step
    int nx, int ny, int nz)
{
    // Global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = k * nx * ny + j * nx + i;

    // Get heat source at this cell
    float Q = heat_source[idx];

    // Calculate potential temperature change
    float potential_deltaT = Q * base_dt / (rho * cp);

    // Adaptive time step (local)
    float dt = base_dt;
    if (fabsf(potential_deltaT) > max_delta_T) {
        dt = max_delta_T * (rho * cp) / fabsf(Q);
    }

    // Actual temperature increment
    float deltaT = Q * dt / (rho * cp);

    // Update distribution
    float delta_f0 = 0.25f * deltaT;
    int f0_idx = idx * 7;
    g_distributions[f0_idx] += delta_f0;
}