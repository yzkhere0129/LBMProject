/**
 * @file vof_solver.cu
 * @brief Implementation of Volume of Fluid (VOF) solver
 */

#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace lbm {
namespace physics {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief First-order upwind advection kernel (donor-cell scheme)
 * @note Stable but diffusive - suitable for VOF advection
 */
__global__ void advectFillLevelUpwindKernel(
    const float* fill_level,
    float* fill_level_new,
    const float* ux,
    const float* uy,
    const float* uz,
    float dt,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Get local velocity
    float u = ux[idx];
    float v = uy[idx];
    float w = uz[idx];

    // Upwind scheme: choose upstream direction based on velocity sign
    // x, y: periodic boundary (allows lateral flow)
    int i_up = (u > 0.0f) ? (i > 0 ? i - 1 : nx - 1) : (i < nx - 1 ? i + 1 : 0);
    int j_up = (v > 0.0f) ? (j > 0 ? j - 1 : ny - 1) : (j < ny - 1 ? j + 1 : 0);

    // z: outflow boundary (allows material to leave domain)
    // - At interior: standard upwind
    // - At top boundary with upward flow: use interior cell to allow outflow
    // - At bottom boundary with downward flow: use interior cell to allow outflow
    int k_up;
    bool at_top_outflow = (k == nz - 1) && (w > 0.0f);
    bool at_bottom_outflow = (k == 0) && (w < 0.0f);

    if (w > 0.0f) {
        // Upward velocity: upstream is below (k-1)
        if (k > 0) {
            k_up = k - 1;
        } else {
            k_up = k;  // Bottom boundary: zero gradient
        }
    } else {
        // Downward velocity: upstream is above (k+1)
        if (k < nz - 1) {
            k_up = k + 1;
        } else {
            k_up = k;  // Top boundary: zero gradient
        }
    }

    int idx_x = i_up + nx * (j + ny * k);
    int idx_y = i + nx * (j_up + ny * k);
    int idx_z = i + nx * (j + ny * k_up);

    // Compute gradients using upwind differences
    // Upwind scheme: df/dt + u * df/dx = 0
    // For u > 0: upstream is at i-1, so df/dx = (f[i] - f[i-1]) / dx
    // For u < 0: upstream is at i+1, so df/dx = (f[i+1] - f[i]) / dx
    //
    // With idx_x pointing to upstream cell:
    // - u > 0: idx_x = i-1, so df/dx should be (f[i] - f[idx_x]) / dx  -- POSITIVE if f increases with x
    // - u < 0: idx_x = i+1, so df/dx should be (f[idx_x] - f[i]) / dx  -- using downstream-upstream
    //
    // BUG FIX: The original code used (f[idx] - f[idx_x]) for both cases.
    // This is correct for u > 0 but WRONG for u < 0.
    // The correct upwind derivative is always (f_downstream - f_upstream) / dx
    // where downstream is in the direction of velocity and upstream is opposite.
    //
    // Simpler formulation: dfdt = u * (f[idx] - f[idx_up]) / dx for u > 0
    //                      dfdt = u * (f[idx_down] - f[idx]) / dx for u < 0
    //                    = u * (f[i+1] - f[i]) / dx for u < 0
    //                    = u * (f[idx_x] - f[idx]) / dx  (since idx_x = i+1 when u < 0)
    //
    // So the fix is: use (f[idx_x] - f[idx]) / dx when u < 0
    float dfdt_x, dfdt_y, dfdt_z;

    if (u >= 0.0f) {
        dfdt_x = u * (fill_level[idx] - fill_level[idx_x]) / dx;
    } else {
        dfdt_x = u * (fill_level[idx_x] - fill_level[idx]) / dx;
    }

    if (v >= 0.0f) {
        dfdt_y = v * (fill_level[idx] - fill_level[idx_y]) / dx;
    } else {
        dfdt_y = v * (fill_level[idx_y] - fill_level[idx]) / dx;
    }

    // Special handling for z-direction to allow proper outflow at boundaries
    if (w >= 0.0f) {
        if (at_top_outflow) {
            // At top boundary with upward flow: use one-sided gradient to allow outflow
            // Extrapolate: assume boundary gradient equals interior gradient
            int k_interior = nz - 2;
            int idx_interior = i + nx * (j + ny * k_interior);
            dfdt_z = w * (fill_level[idx] - fill_level[idx_interior]) / dx;
        } else {
            dfdt_z = w * (fill_level[idx] - fill_level[idx_z]) / dx;
        }
    } else {
        if (at_bottom_outflow) {
            // At bottom boundary with downward flow: use one-sided gradient to allow outflow
            int k_interior = 1;
            int idx_interior = i + nx * (j + ny * k_interior);
            dfdt_z = w * (fill_level[idx_interior] - fill_level[idx]) / dx;
        } else {
            dfdt_z = w * (fill_level[idx_z] - fill_level[idx]) / dx;
        }
    }

    // Forward Euler time integration
    float f_new = fill_level[idx] - dt * (dfdt_x + dfdt_y + dfdt_z);

    // Flush tiny values to zero (prevent denormalized float underflow)
    if (f_new < 1e-6f) f_new = 0.0f;

    // Clamp to [0, 1] to maintain physical bounds
    fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}

/**
 * @brief Interface reconstruction using central differences
 * @note Computes interface normal n = -∇f / |∇f|
 */
__global__ void reconstructInterfaceKernel(
    const float* fill_level,
    float3* interface_normal,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Central differences with boundary handling
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

    // Compute fill level gradient
    float grad_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    float grad_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    float grad_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);

    // Compute gradient magnitude
    float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

    // Interface normal points from liquid to gas: n = -∇f / |∇f|
    if (grad_mag > 1e-8f) {
        interface_normal[idx].x = -grad_x / grad_mag;
        interface_normal[idx].y = -grad_y / grad_mag;
        interface_normal[idx].z = -grad_z / grad_mag;
    } else {
        // Zero gradient (bulk liquid or gas)
        interface_normal[idx].x = 0.0f;
        interface_normal[idx].y = 0.0f;
        interface_normal[idx].z = 0.0f;
    }
}

/**
 * @brief Curvature computation: κ = ∇·n
 * @note Uses finite differences on interface normals
 */
__global__ void computeCurvatureKernel(
    const float* fill_level,
    const float3* interface_normal,
    float* curvature,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Only compute curvature at interface cells
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        curvature[idx] = 0.0f;
        return;
    }

    // Central differences with boundary handling
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

    // Compute divergence of normal: κ = ∇·n
    float dnx_dx = (interface_normal[idx_xp].x - interface_normal[idx_xm].x) / (2.0f * dx);
    float dny_dy = (interface_normal[idx_yp].y - interface_normal[idx_ym].y) / (2.0f * dx);
    float dnz_dz = (interface_normal[idx_zp].z - interface_normal[idx_zm].z) / (2.0f * dx);

    curvature[idx] = dnx_dx + dny_dy + dnz_dz;
}

/**
 * @brief Cell type conversion based on fill level
 */
__global__ void convertCellsKernel(
    const float* fill_level,
    uint8_t* cell_flags,
    float eps,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];

    if (f < eps) {
        cell_flags[idx] = static_cast<uint8_t>(CellFlag::GAS);
    } else if (f > 1.0f - eps) {
        cell_flags[idx] = static_cast<uint8_t>(CellFlag::LIQUID);
    } else {
        cell_flags[idx] = static_cast<uint8_t>(CellFlag::INTERFACE);
    }
}

/**
 * @brief Contact angle boundary condition
 * @note Modifies interface normal at walls: n_wall = n - (n·n_w)·n_w + cos(θ)·n_w
 */
__global__ void applyContactAngleBoundaryKernel(
    float3* interface_normal,
    const uint8_t* cell_flags,
    float contact_angle,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    // Only apply to boundary cells
    if (i != 0 && i != nx - 1 && j != 0 && j != ny - 1 && k != 0 && k != nz - 1) {
        return;
    }

    int idx = i + nx * (j + ny * k);

    // Check if this is an interface cell
    if (cell_flags[idx] != static_cast<uint8_t>(CellFlag::INTERFACE)) {
        return;
    }

    // Wall normal (pointing inward)
    float3 n_wall = make_float3(0.0f, 0.0f, 0.0f);
    if (i == 0) n_wall.x = 1.0f;
    if (i == nx - 1) n_wall.x = -1.0f;
    if (j == 0) n_wall.y = 1.0f;
    if (j == ny - 1) n_wall.y = -1.0f;
    if (k == 0) n_wall.z = 1.0f;
    if (k == nz - 1) n_wall.z = -1.0f;

    // Current interface normal
    float3 n = interface_normal[idx];

    // Compute contact angle adjustment
    float cos_theta = cosf(contact_angle * 3.14159265f / 180.0f);

    // Project normal onto wall plane and add contact angle component
    float n_dot_nwall = n.x * n_wall.x + n.y * n_wall.y + n.z * n_wall.z;

    interface_normal[idx].x = n.x - n_dot_nwall * n_wall.x + cos_theta * n_wall.x;
    interface_normal[idx].y = n.y - n_dot_nwall * n_wall.y + cos_theta * n_wall.y;
    interface_normal[idx].z = n.z - n_dot_nwall * n_wall.z + cos_theta * n_wall.z;

    // Renormalize
    float norm = sqrtf(interface_normal[idx].x * interface_normal[idx].x +
                       interface_normal[idx].y * interface_normal[idx].y +
                       interface_normal[idx].z * interface_normal[idx].z);
    if (norm > 1e-8f) {
        interface_normal[idx].x /= norm;
        interface_normal[idx].y /= norm;
        interface_normal[idx].z /= norm;
    }
}

/**
 * @brief Initialize spherical droplet
 */
__global__ void initializeDropletKernel(
    float* fill_level,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Distance from droplet center
    float dx = static_cast<float>(i) - center_x;
    float dy = static_cast<float>(j) - center_y;
    float dz = static_cast<float>(k) - center_z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    // Smooth interface using tanh profile (width ~ 2 grid cells)
    float interface_width = 2.0f;
    fill_level[idx] = 0.5f * (1.0f - tanhf((dist - radius) / interface_width));
}

/**
 * @brief Mass reduction kernel for conservation check
 */
__global__ void computeMassReductionKernel(
    const float* fill_level,
    float* partial_sums,
    int num_cells)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < num_cells) ? fill_level[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// VOFSolver Implementation
// ============================================================================

VOFSolver::VOFSolver(int nx, int ny, int nz, float dx)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz), dx_(dx),
      d_fill_level_(nullptr), d_cell_flags_(nullptr),
      d_interface_normal_(nullptr), d_curvature_(nullptr),
      d_fill_level_tmp_(nullptr)
{
    allocateMemory();
}

VOFSolver::~VOFSolver() {
    freeMemory();
}

void VOFSolver::allocateMemory() {
    // Clear any previous CUDA errors before allocation
    cudaGetLastError();

    cudaError_t err;

    err = cudaMalloc(&d_fill_level_, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_fill_level: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_cell_flags_, num_cells_ * sizeof(uint8_t));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_cell_flags: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_interface_normal_, num_cells_ * sizeof(float3));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_interface_normal: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_curvature_, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_curvature: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_fill_level_tmp_, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_fill_level_tmp: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void VOFSolver::freeMemory() {
    if (d_fill_level_) cudaFree(d_fill_level_);
    if (d_cell_flags_) cudaFree(d_cell_flags_);
    if (d_interface_normal_) cudaFree(d_interface_normal_);
    if (d_curvature_) cudaFree(d_curvature_);
    if (d_fill_level_tmp_) cudaFree(d_fill_level_tmp_);
}

void VOFSolver::initialize(const float* fill_level) {
    cudaMemcpy(d_fill_level_, fill_level, num_cells_ * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize cell flags based on fill level
    convertCells();

    // Compute initial interface normals and curvature
    reconstructInterface();
    computeCurvature();
}

void VOFSolver::initialize(float uniform_fill) {
    std::vector<float> h_fill(num_cells_, uniform_fill);
    initialize(h_fill.data());
}

void VOFSolver::initializeDroplet(float center_x, float center_y,
                                   float center_z, float radius) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    initializeDropletKernel<<<gridSize, blockSize>>>(
        d_fill_level_, center_x, center_y, center_z, radius, nx_, ny_, nz_);

    cudaDeviceSynchronize();

    // Update cell flags and interface properties
    convertCells();
    reconstructInterface();
    computeCurvature();
}

void VOFSolver::advectFillLevel(const float* velocity_x,
                                 const float* velocity_y,
                                 const float* velocity_z,
                                 float dt) {
    // Check VOF CFL condition before advection
    // CFL = v_max * dt / dx should be < 0.5 for stability
    // Sample from TOP LAYER (z = nz-1) where Marangoni flow is active
    const int top_layer_size = nx_ * ny_;
    const int top_layer_offset = (nz_ - 1) * nx_ * ny_;  // Start of top layer
    const int sample_size = std::min(top_layer_size, num_cells_ - top_layer_offset);

    std::vector<float> h_ux(sample_size), h_uy(sample_size), h_uz(sample_size);
    cudaMemcpy(h_ux.data(), velocity_x + top_layer_offset, sample_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), velocity_y + top_layer_offset, sample_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz.data(), velocity_z + top_layer_offset, sample_size * sizeof(float), cudaMemcpyDeviceToHost);

    float v_max = 0.0f;
    for (int i = 0; i < sample_size; ++i) {
        float v_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        v_max = std::max(v_max, v_mag);
    }

    float vof_cfl = v_max * dt / dx_;
    if (vof_cfl > 0.5f) {
        printf("WARNING: VOF CFL violation: %.3f > 0.5 (v_max=%.2e m/s, dt=%.2e s, dx=%.2e m)\n",
               vof_cfl, v_max, dt, dx_);
    }

    // Diagnostic: print VOF advection info periodically
    static int call_count = 0;
    static float prev_mass = -1.0f;
    if (call_count % 500 == 0 && call_count < 5000) {
        // Compute current mass
        float mass = computeTotalMass();
        float mass_change = (prev_mass > 0) ? (mass - prev_mass) : 0.0f;
        printf("[VOF ADVECT] Call %d: v_max=%.4f m/s (%.2f mm/s), CFL=%.6f, mass=%.1f (delta=%.3f)\n",
               call_count, v_max, v_max * 1000, vof_cfl, mass, mass_change);
        prev_mass = mass;
    }
    call_count++;

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    advectFillLevelUpwindKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_fill_level_tmp_, velocity_x, velocity_y, velocity_z,
        dt, dx_, nx_, ny_, nz_);

    // IMPORTANT: Synchronize BEFORE swapping buffers to ensure kernel completion
    cudaDeviceSynchronize();

    // Swap buffers (safe now that kernel has completed)
    float* tmp = d_fill_level_;
    d_fill_level_ = d_fill_level_tmp_;
    d_fill_level_tmp_ = tmp;
}

void VOFSolver::reconstructInterface() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    reconstructInterfaceKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_normal_, dx_, nx_, ny_, nz_);

    cudaDeviceSynchronize();
}

void VOFSolver::computeCurvature() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    computeCurvatureKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_normal_, d_curvature_, dx_, nx_, ny_, nz_);

    cudaDeviceSynchronize();
}

void VOFSolver::convertCells() {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    float eps = 0.01f;  // Threshold for interface detection

    convertCellsKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_cell_flags_, eps, num_cells_);

    cudaDeviceSynchronize();
}

void VOFSolver::applyBoundaryConditions(int boundary_type, float contact_angle) {
    if (boundary_type == 0) {
        // Periodic boundaries - no action needed
        return;
    }

    if (boundary_type == 1) {
        // Apply contact angle at walls
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                      (ny_ + blockSize.y - 1) / blockSize.y,
                      (nz_ + blockSize.z - 1) / blockSize.z);

        applyContactAngleBoundaryKernel<<<gridSize, blockSize>>>(
            d_interface_normal_, d_cell_flags_, contact_angle, nx_, ny_, nz_);

        cudaDeviceSynchronize();
    }
}

void VOFSolver::copyFillLevelToHost(float* host_fill) const {
    cudaMemcpy(host_fill, d_fill_level_, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void VOFSolver::copyCellFlagsToHost(uint8_t* host_flags) const {
    cudaMemcpy(host_flags, d_cell_flags_, num_cells_ * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
}

void VOFSolver::copyCurvatureToHost(float* host_curvature) const {
    cudaMemcpy(host_curvature, d_curvature_, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost);
}

float VOFSolver::computeTotalMass() const {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Allocate partial sums
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, gridSize * sizeof(float));

    // First reduction: compute partial sums
    computeMassReductionKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_fill_level_, d_partial_sums, num_cells_);

    // Copy partial sums to host and finish reduction on CPU
    std::vector<float> h_partial_sums(gridSize);
    cudaMemcpy(h_partial_sums.data(), d_partial_sums, gridSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_partial_sums);

    // Final reduction on CPU
    float total_mass = 0.0f;
    for (float sum : h_partial_sums) {
        total_mass += sum;
    }

    return total_mass;
}

// ============================================================================
// Evaporation Mass Loss (VOF-Thermal Coupling)
// ============================================================================

/**
 * @brief Apply evaporation mass loss to fill level field
 *
 * Physics:
 *   dm/dt = -J_evap * A_interface    [kg/s]
 *   df/dt = -J_evap / (rho * dx)     [1/s]
 *   df = -J_evap * dt / (rho * dx)   [dimensionless]
 *
 * Where:
 *   J_evap: Evaporation mass flux [kg/(m^2*s)]
 *   rho: Material density [kg/m^3]
 *   dx: Grid spacing [m]
 *   dt: Time step [s]
 *
 * Stability:
 *   - Only applies to cells with material (f > 0)
 *   - Limited to max 10% reduction per timestep
 *   - Clamps result to [0, 1]
 */
__global__ void applyEvaporationMassLossKernel(
    float* fill_level,
    const float* J_evap,
    float rho,
    float dx,
    float dt,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    float f = fill_level[idx];
    float J = J_evap[idx];

    // Skip cells with no material or no evaporation
    if (f <= 0.0f || J <= 0.0f) {
        return;
    }

    // Compute fill level change
    // df = -J_evap * dt / (rho * dx)
    float df = -J * dt / (rho * dx);

    // ============================================================
    // Stability limiter: max 2% reduction per timestep
    // ============================================================
    // CRITICAL FIX (2025-11-27): Reduced from 10% to 2% to prevent
    // excessive mass loss at extreme temperatures (>40,000 K).
    //
    // This prevents numerical instability from large evaporation
    // rates causing fill_level to go negative in a single step.
    //
    // Physical justification: In real AM processes, the maximum
    // evaporation rate is limited by heat transport to the surface.
    // This limiter ensures the numerical scheme remains stable
    // while maintaining physical fidelity.
    // ============================================================
    const float MAX_DF_PER_STEP = 0.02f;  // Max 2% change per step

    if (df < -MAX_DF_PER_STEP * f) {
        df = -MAX_DF_PER_STEP * f;
    }

    // Apply change and clamp to [0, 1]
    float f_new = f + df;

    // Flush tiny values to zero (prevent denormalized float underflow)
    if (f_new < 1e-6f) f_new = 0.0f;

    fill_level[idx] = fmaxf(0.0f, fminf(1.0f, f_new));
}

void VOFSolver::applyEvaporationMassLoss(const float* J_evap, float rho, float dt) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    applyEvaporationMassLossKernel<<<gridSize, blockSize>>>(
        d_fill_level_, J_evap, rho, dx_, dt, nx_, ny_, nz_
    );
    cudaDeviceSynchronize();

    // Diagnostic: Print evaporation mass loss info periodically
    static int evap_call_count = 0;
    if (evap_call_count % 500 == 0 && evap_call_count < 5000) {
        // Sample J_evap to check if evaporation is active
        int top_layer_start = (nz_ - 1) * nx_ * ny_;
        int sample_size = std::min(nx_ * ny_, 10000);
        std::vector<float> h_J(sample_size);
        cudaMemcpy(h_J.data(), J_evap + top_layer_start, sample_size * sizeof(float), cudaMemcpyDeviceToHost);

        float max_J = 0.0f;
        int active_cells = 0;
        for (int i = 0; i < sample_size; ++i) {
            if (h_J[i] > 0.0f) {
                active_cells++;
                max_J = std::max(max_J, h_J[i]);
            }
        }

        if (active_cells > 0) {
            printf("[VOF EVAP] Call %d: active_cells=%d, max_J=%.4e kg/(m^2*s), df_max=%.6f\n",
                   evap_call_count, active_cells, max_J, max_J * dt / (rho * dx_));
        }
    }
    evap_call_count++;
}

// ============================================================================
// Solidification Shrinkage (VOF-Thermal Coupling)
// ============================================================================

/**
 * @brief Apply solidification shrinkage mass source to fill level
 *
 * Physics:
 *   Solidification shrinkage occurs when liquid metal transforms to solid,
 *   causing volume contraction due to higher solid density.
 *
 *   Volume change: dV/V = -beta * dfl
 *   where beta = (rho_solid - rho_liquid) / rho_solid = 1 - rho_l/rho_s
 *         (typically ~0.07 for metals)
 *
 *   VOF fill level change (dimensionless):
 *   df = -beta * (dfl/dt) * dt = -beta * dfl
 *
 *   Solidifying: dfl/dt < 0 --> df > 0? NO!
 *   Actually: dfl < 0 (liquid decreasing) --> df < 0 (volume shrinks)
 *   So: df = beta * dfl_dt * dt (positive beta, negative rate = negative df)
 *
 * CRITICAL CONSTRAINTS (Bug Fix 2024-11):
 *   1. Only apply at VOF INTERFACE cells (0.01 < f < 0.99)
 *      - Internal bulk cells should not have their fill_level modified
 *      - Shrinkage manifests as surface depression, not internal voids
 *   2. Only apply during SOLIDIFICATION (dfl/dt < 0)
 *      - Melting expansion is handled differently (material addition)
 *   3. Correct dimensionless formula: df = beta * dfl_dt * dt
 *      - Previous formula had spurious /dx term causing grid-dependent behavior
 *   4. Conservative limiter: max 1% change per step
 *      - Prevents numerical instability at sharp solidification fronts
 *
 * Stability:
 *   - Only applies to interface cells (0.01 < f < 0.99)
 *   - Only during solidification (rate < 0)
 *   - Limited to max 1% reduction per timestep
 *   - Clamps result to [0, 1]
 */
__global__ void applySolidificationShrinkageKernel(
    float* fill_level,
    const float* dfl_dt,
    float beta,
    float dx,      // Not used in corrected formula, kept for API compatibility
    float dt,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];
    float rate = dfl_dt[idx];

    // ========================================================================
    // CONSTRAINT 1: Only apply at VOF interface cells
    // ========================================================================
    // Shrinkage can only manifest where there's a free surface.
    // Internal cells (f ~ 1.0) represent bulk material - their density changes
    // but the VOF fill_level should remain 1.0 (still fully filled with metal).
    // Gas cells (f ~ 0.0) have no material to shrink.
    // Only interface cells (0.01 < f < 0.99) can show volume reduction.
    // ========================================================================
    if (f <= 0.01f || f >= 0.99f) return;

    // ========================================================================
    // CONSTRAINT 2: Only apply during solidification (rate < 0)
    // ========================================================================
    // When dfl/dt < 0: liquid is becoming solid, volume shrinks
    // When dfl/dt > 0: solid is melting, but we don't add material here
    //                  (melting expansion would require mass source from substrate)
    // ========================================================================
    if (rate >= 0.0f) return;

    // Skip if no significant phase change happening
    if (fabsf(rate) < 1e-10f) return;

    // ========================================================================
    // CORRECTED FORMULA (dimensionless)
    // ========================================================================
    // df = beta * dfl_dt * dt
    //
    // Derivation:
    //   dV/V = -beta * dfl  (volume change due to solidification)
    //   For a cell: df_VOF = dV/V = -beta * dfl
    //             = -beta * (dfl/dt) * dt
    //             = beta * |rate| * dt  (since rate < 0 for solidification)
    //
    // Sign: rate < 0 (solidifying), beta > 0, dt > 0
    //       df = beta * rate * dt < 0 (fill level decreases = shrinkage)
    //
    // NOTE: Previous formula had /dx which is dimensionally incorrect:
    //       df = rate * beta * dt / dx  [1/s * - * s / m = 1/m] WRONG!
    //       Correct: df = rate * beta * dt  [1/s * - * s = -] CORRECT!
    // ========================================================================
    float df = beta * rate * dt;

    // ========================================================================
    // CONSERVATIVE LIMITER: max 1% reduction per step
    // ========================================================================
    // More conservative than before (was 5%) to prevent:
    // - Excessive dimpling at sharp solidification fronts
    // - Numerical oscillations near mushy zone boundaries
    // - Unrealistic rapid surface depression
    // ========================================================================
    const float MAX_DF_FRACTION = 0.01f;  // Max 1% change per step
    float max_reduction = MAX_DF_FRACTION * f;

    // df is already negative for solidification, so we limit its magnitude
    if (df < -max_reduction) {
        df = -max_reduction;
    }

    // Apply change and clamp
    float f_new = f + df;

    // Flush tiny values to zero (prevent denormalized float underflow)
    if (f_new < 1e-6f) f_new = 0.0f;

    fill_level[idx] = fmaxf(0.0f, fminf(1.0f, f_new));
}

void VOFSolver::applySolidificationShrinkage(const float* dfl_dt, float beta, float dx, float dt) {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    applySolidificationShrinkageKernel<<<blocks, threads>>>(
        d_fill_level_, dfl_dt, beta, dx, dt, num_cells_);

    cudaDeviceSynchronize();
}

} // namespace physics
} // namespace lbm
