/**
 * @file phase_change.cu
 * @brief Implementation of phase change solver using enthalpy method
 */

#include "physics/phase_change.h"
#include "physics/material_properties.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include "utils/cuda_check.h"

// Local __device__ variable for material properties in this translation unit.
//
// Why __device__ instead of __constant__:
// With CUDA separable compilation (CUDA_SEPARABLE_COMPILATION ON), each .cu
// file is compiled as relocatable device code (-rdc=true).  In this mode,
// cudaMemcpyToSymbol() fails with "invalid device symbol" for __constant__
// variables because the symbol registration happens at device-link time, not
// at compile time — the runtime cannot locate the symbol in the module table.
// cudaGetSymbolAddress() + cudaMemcpy() works correctly in RDC mode.
//
// Performance impact: __device__ global memory vs __constant__ cache.
// MaterialProperties is read once per kernel invocation for initialization
// of rho_ref/cp_ref locals, so the extra latency is negligible.
__device__ lbm::physics::MaterialProperties d_phase_material;

// Helper: upload material properties to d_phase_material on the device.
// Must be file-scope (not inside a namespace) so it can reference the
// file-scope d_phase_material symbol via cudaGetSymbolAddress.
static void uploadPhaseMaterial(const lbm::physics::MaterialProperties& mat) {
    lbm::physics::MaterialProperties* sym_ptr;
    cudaError_t err = cudaGetSymbolAddress(
        reinterpret_cast<void**>(&sym_ptr), d_phase_material);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetSymbolAddress(d_phase_material): ") +
            cudaGetErrorString(err));
    }
    err = cudaMemcpy(sym_ptr, &mat, sizeof(lbm::physics::MaterialProperties),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy to d_phase_material: ") +
            cudaGetErrorString(err));
    }
}

namespace lbm {
namespace physics {

//==============================================================================
// CUDA Kernels
//==============================================================================

/**
 * @brief Compute enthalpy from temperature
 *
 * H = ρ_ref·cp_ref·T + fl(T)·ρ_ref·L_fusion
 *
 * We use reference density (solid) for consistency in Newton solver
 *
 * Units check:
 *   [J/m³] = [kg/m³]·[J/(kg·K)]·[K] + [1]·[kg/m³]·[J/kg]
 *   [J/m³] = [J/m³] + [J/m³] ✓
 */
__global__ void computeEnthalpyFromTemperatureKernel(
    const float* temperature,
    float* enthalpy,
    float* liquid_fraction,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];

    // Compute liquid fraction
    float fl = d_phase_material.liquidFraction(T);

    // Use solid density and cp as reference for enthalpy calculation
    // This ensures consistency with the Newton solver
    float rho_ref = d_phase_material.rho_solid;
    float cp_ref = d_phase_material.cp_solid;

    // Compute total enthalpy: H = ρ_ref·cp_ref·T + fl·ρ_ref·L_fusion
    // Units: [J/m³] = [kg/m³]·[J/(kg·K)]·[K] + [kg/m³]·[J/kg]
    float H = rho_ref * cp_ref * T + fl * rho_ref * d_phase_material.L_fusion;

    enthalpy[idx] = H;
    liquid_fraction[idx] = fl;
}

/**
 * @brief Solve for temperature from enthalpy using Newton-Raphson
 *
 * We need to solve: H = ρ_ref·cp_ref·T + fl(T)·ρ_ref·L_fusion
 * for T given H.
 *
 * Define: f(T) = ρ_ref·cp_ref·T + fl(T)·ρ_ref·L_fusion - H = 0
 *
 * Newton iteration: T_new = T_old - f(T_old)/f'(T_old)
 *
 * where f'(T) = ρ_ref·cp_ref + (dfl/dT)·ρ_ref·L_fusion
 *       dfl/dT = 1/ΔT_melt in mushy zone, 0 otherwise
 *
 * IMPORTANT: We use solid properties as reference for consistency
 */
__global__ void solveTemperatureFromEnthalpyKernel(
    const float* enthalpy,
    float* temperature,
    float* liquid_fraction,
    int* converged,
    float tolerance,
    int max_iterations,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float H_target = enthalpy[idx];
    float T = temperature[idx];  // Use current T as initial guess

    // Use solid properties as reference (consistent with H calculation)
    float rho_ref = d_phase_material.rho_solid;
    float cp_ref = d_phase_material.cp_solid;

    // Determine which phase we're likely in based on enthalpy
    // H_solidus = rho*cp*T_solidus
    // H_liquidus = rho*cp*T_liquidus + rho*L
    float H_solidus = rho_ref * cp_ref * d_phase_material.T_solidus;
    float H_liquidus = rho_ref * cp_ref * d_phase_material.T_liquidus + rho_ref * d_phase_material.L_fusion;

    // Better initial guess if current T is far off
    if (H_target < H_solidus) {
        // Solid phase: H = rho*cp*T => T = H/(rho*cp)
        T = H_target / (rho_ref * cp_ref);
    } else if (H_target > H_liquidus) {
        // Liquid phase: H = rho*cp*T + rho*L => T = (H - rho*L)/(rho*cp)
        T = (H_target - rho_ref * d_phase_material.L_fusion) / (rho_ref * cp_ref);
    } else {
        // Mushy zone: use bisection for initial guess
        T = (d_phase_material.T_solidus + d_phase_material.T_liquidus) / 2.0f;
    }

    // Newton-Raphson iteration
    bool conv = false;
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute current liquid fraction
        float fl = d_phase_material.liquidFraction(T);

        // Compute f(T) = ρ_ref·cp_ref·T + fl·ρ_ref·L - H
        float H_current = rho_ref * cp_ref * T + fl * rho_ref * d_phase_material.L_fusion;
        float f = H_current - H_target;

        // Check convergence
        if (fabsf(f) < tolerance * rho_ref * cp_ref) {
            conv = true;
            break;
        }

        // Compute derivative f'(T) = ρ_ref·cp_ref + (dfl/dT)·ρ_ref·L
        float dfl_dT = 0.0f;
        if (d_phase_material.isMushy(T)) {
            // In mushy zone: dfl/dT = 1/(T_liquidus - T_solidus)
            dfl_dT = 1.0f / (d_phase_material.T_liquidus - d_phase_material.T_solidus);
        }
        float df_dT = rho_ref * cp_ref + dfl_dT * rho_ref * d_phase_material.L_fusion;

        // Newton update
        float dT = f / df_dT;
        T -= dT;

        // CRITICAL FIX: Clamp to prevent thermal runaway
        // T_MAX should be limited to 1.2 * T_boil to prevent temperatures
        // far above the boiling point where the physics model breaks down
        float T_MAX_SAFE = fminf(T_MAX, 1.2f * d_phase_material.T_vaporization);
        T = fmaxf(T_MIN, fminf(T_MAX_SAFE, T));
    }

    // CRITICAL FIX: Bisection fallback if Newton-Raphson failed
    if (!conv) {
        // Use bisection method as robust fallback
        float T_MAX_SAFE = fminf(T_MAX, 1.2f * d_phase_material.T_vaporization);
        float T_low = T_MIN;
        float T_high = T_MAX_SAFE;

        for (int iter = 0; iter < max_iterations; ++iter) {
            T = (T_low + T_high) / 2.0f;
            float fl = d_phase_material.liquidFraction(T);
            float H_current = rho_ref * cp_ref * T + fl * rho_ref * d_phase_material.L_fusion;
            float f = H_current - H_target;

            if (fabsf(f) < tolerance * rho_ref * cp_ref) {
                conv = true;
                break;
            }

            if (f > 0.0f) {
                T_high = T;  // H too high, reduce T
            } else {
                T_low = T;   // H too low, increase T
            }
        }
    }

    // Update outputs
    temperature[idx] = T;
    liquid_fraction[idx] = d_phase_material.liquidFraction(T);
    converged[idx] = conv ? 1 : 0;
}

/**
 * @brief Update liquid fraction from temperature
 */
__global__ void updateLiquidFractionKernel(
    const float* temperature,
    float* liquid_fraction,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    liquid_fraction[idx] = d_phase_material.liquidFraction(T);
}

/**
 * @brief Add enthalpy change
 */
__global__ void addEnthalpyChangeKernel(
    float* enthalpy,
    const float* dH,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    enthalpy[idx] += dH[idx];
}

/**
 * @brief Compute liquid fraction rate of change
 */
__global__ void computeLiquidFractionRateKernel(
    const float* fl_curr,
    const float* fl_prev,
    float* dfl_dt,
    float dt,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    dfl_dt[idx] = (fl_curr[idx] - fl_prev[idx]) / dt;
}

/**
 * @brief Store current liquid fraction for next step
 */
__global__ void storeLiquidFractionKernel(
    const float* fl_curr,
    float* fl_prev,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    fl_prev[idx] = fl_curr[idx];
}

/**
 * @brief Compute total energy (reduction)
 *
 * Each block computes partial sum, then host reduces
 */
__global__ void computeTotalEnergyKernel(
    const float* enthalpy,
    float* partial_sums,
    int num_cells,
    float cell_volume)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float sum = (idx < num_cells) ? enthalpy[idx] * cell_volume : 0.0f;
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

//==============================================================================
// PhaseChangeSolver Implementation
//==============================================================================

PhaseChangeSolver::PhaseChangeSolver(int nx, int ny, int nz,
                                     const MaterialProperties& material)
    : nx_(nx), ny_(ny), nz_(nz),
      num_cells_(nx * ny * nz),
      material_(material),
      dx_(1.0f),
      d_enthalpy(nullptr),
      d_liquid_fraction(nullptr),
      d_liquid_fraction_prev_(nullptr),
      d_dfl_dt_(nullptr)
{
    allocateMemory();
}

PhaseChangeSolver::~PhaseChangeSolver() {
    freeMemory();
}

PhaseChangeSolver::PhaseChangeSolver(PhaseChangeSolver&& other) noexcept
    : nx_(other.nx_), ny_(other.ny_), nz_(other.nz_),
      num_cells_(other.num_cells_),
      material_(other.material_),
      dx_(other.dx_),
      d_enthalpy(other.d_enthalpy),
      d_liquid_fraction(other.d_liquid_fraction),
      d_liquid_fraction_prev_(other.d_liquid_fraction_prev_),
      d_dfl_dt_(other.d_dfl_dt_)
{
    other.d_enthalpy = nullptr;
    other.d_liquid_fraction = nullptr;
    other.d_liquid_fraction_prev_ = nullptr;
    other.d_dfl_dt_ = nullptr;
}

PhaseChangeSolver& PhaseChangeSolver::operator=(PhaseChangeSolver&& other) noexcept {
    if (this != &other) {
        freeMemory();
        nx_ = other.nx_;
        ny_ = other.ny_;
        nz_ = other.nz_;
        num_cells_ = other.num_cells_;
        material_ = other.material_;
        dx_ = other.dx_;
        d_enthalpy = other.d_enthalpy;
        d_liquid_fraction = other.d_liquid_fraction;
        d_liquid_fraction_prev_ = other.d_liquid_fraction_prev_;
        d_dfl_dt_ = other.d_dfl_dt_;
        other.d_enthalpy = nullptr;
        other.d_liquid_fraction = nullptr;
        other.d_liquid_fraction_prev_ = nullptr;
        other.d_dfl_dt_ = nullptr;
    }
    return *this;
}

void PhaseChangeSolver::allocateMemory() {
    size_t size = num_cells_ * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_enthalpy, size));
    CUDA_CHECK(cudaMalloc(&d_liquid_fraction, size));
    CUDA_CHECK(cudaMalloc(&d_liquid_fraction_prev_, size));
    CUDA_CHECK(cudaMalloc(&d_dfl_dt_, size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_enthalpy, 0, size));
    CUDA_CHECK(cudaMemset(d_liquid_fraction, 0, size));
    CUDA_CHECK(cudaMemset(d_liquid_fraction_prev_, 0, size));
    CUDA_CHECK(cudaMemset(d_dfl_dt_, 0, size));
}

void PhaseChangeSolver::freeMemory() {
    if (d_enthalpy) { CUDA_CHECK(cudaFree(d_enthalpy)); d_enthalpy = nullptr; }
    if (d_liquid_fraction) { CUDA_CHECK(cudaFree(d_liquid_fraction)); d_liquid_fraction = nullptr; }
    if (d_liquid_fraction_prev_) { CUDA_CHECK(cudaFree(d_liquid_fraction_prev_)); d_liquid_fraction_prev_ = nullptr; }
    if (d_dfl_dt_) { CUDA_CHECK(cudaFree(d_dfl_dt_)); d_dfl_dt_ = nullptr; }
}

void PhaseChangeSolver::initializeFromTemperature(const float* temperature) {
    // Upload material properties to device
    uploadPhaseMaterial(material_);

    // Compute initial enthalpy
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    computeEnthalpyFromTemperatureKernel<<<blocks, threads>>>(
        temperature, d_enthalpy, d_liquid_fraction, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Store initial liquid fraction as "previous" for first time step
    CUDA_CHECK(cudaMemcpy(d_liquid_fraction_prev_, d_liquid_fraction,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));
}

void PhaseChangeSolver::updateEnthalpyFromTemperature(const float* temperature) {
    // Ensure material is uploaded to device
    uploadPhaseMaterial(material_);

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    computeEnthalpyFromTemperatureKernel<<<blocks, threads>>>(
        temperature, d_enthalpy, d_liquid_fraction, num_cells_);
    CUDA_CHECK_KERNEL();
}

int PhaseChangeSolver::updateTemperatureFromEnthalpy(float* temperature,
                                                      float tolerance,
                                                      int max_iterations) {
    // Ensure material is uploaded to device
    uploadPhaseMaterial(material_);

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    // Allocate convergence flag array
    int* d_converged;
    CUDA_CHECK(cudaMalloc(&d_converged, num_cells_ * sizeof(int)));

    solveTemperatureFromEnthalpyKernel<<<blocks, threads>>>(
        d_enthalpy, temperature, d_liquid_fraction, d_converged,
        tolerance, max_iterations, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Count how many cells actually converged
    int* h_converged = new int[num_cells_];
    CUDA_CHECK(cudaMemcpy(h_converged, d_converged, num_cells_ * sizeof(int),
               cudaMemcpyDeviceToHost));

    int total_converged = 0;
    for (int i = 0; i < num_cells_; ++i) {
        total_converged += h_converged[i];
    }

    delete[] h_converged;
    CUDA_CHECK(cudaFree(d_converged));

    return total_converged;
}

void PhaseChangeSolver::updateLiquidFraction(const float* temperature) {
    // Ensure material is uploaded to device before kernel reads d_phase_material
    uploadPhaseMaterial(material_);

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    updateLiquidFractionKernel<<<blocks, threads>>>(
        temperature, d_liquid_fraction, num_cells_);
    CUDA_CHECK_KERNEL();
}

void PhaseChangeSolver::addEnthalpyChange(const float* dH) {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    addEnthalpyChangeKernel<<<blocks, threads>>>(
        d_enthalpy, dH, num_cells_);
    CUDA_CHECK_KERNEL();
}

void PhaseChangeSolver::copyEnthalpyToHost(float* host_enthalpy) const {
    CUDA_CHECK(cudaMemcpy(host_enthalpy, d_enthalpy, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost));
}

void PhaseChangeSolver::copyLiquidFractionToHost(float* host_fl) const {
    CUDA_CHECK(cudaMemcpy(host_fl, d_liquid_fraction, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost));
}

float PhaseChangeSolver::computeTotalEnergy() const {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    // Allocate partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, blocks * sizeof(float)));

    float cell_volume = dx_ * dx_ * dx_;

    computeTotalEnergyKernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_enthalpy, d_partial_sums, num_cells_, cell_volume);
    CUDA_CHECK_KERNEL();

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[blocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float),
               cudaMemcpyDeviceToHost));

    float total_energy = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        total_energy += h_partial_sums[i];
    }

    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return total_energy;
}

void PhaseChangeSolver::computeLiquidFractionRate(float dt) {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    computeLiquidFractionRateKernel<<<blocks, threads>>>(
        d_liquid_fraction, d_liquid_fraction_prev_, d_dfl_dt_, dt, num_cells_);
    CUDA_CHECK_KERNEL();
}

void PhaseChangeSolver::storePreviousLiquidFraction() {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    storeLiquidFractionKernel<<<blocks, threads>>>(
        d_liquid_fraction, d_liquid_fraction_prev_, num_cells_);
    CUDA_CHECK_KERNEL();
}

} // namespace physics
} // namespace lbm
