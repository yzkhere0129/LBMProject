/**
 * @file thermal_lbm.cu
 * @brief Implementation of Thermal Lattice Boltzmann Method solver
 */

#include "physics/thermal_lbm.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace lbm {
namespace physics {

// External reference to D3Q7 lattice constants (defined in lattice_d3q7.cu)
extern __constant__ int tex[7];
extern __constant__ int tey[7];
extern __constant__ int tez[7];

// Forward declaration of kernel
__global__ void initializeEquilibriumKernel(float* g_src, const float* temperature, int num_cells);

// Constructor (deprecated - for backward compatibility)
ThermalLBM::ThermalLBM(int nx, int ny, int nz, float thermal_diffusivity,
                       float density, float specific_heat,
                       float dt, float dx)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      dt_(dt), dx_(dx),
      thermal_diff_physical_(thermal_diffusivity), rho_(density), cp_(specific_heat),
      emissivity_(0.35f),  // Default emissivity
      phase_solver_(nullptr), has_material_(false) {

    // Initialize D3Q7 lattice if not already done
    if (!D3Q7::isInitialized()) {
        D3Q7::initializeDevice();
    }

    // ============================================================================
    // CRITICAL FIX: Convert thermal diffusivity to lattice units
    // ============================================================================
    // LBM requires dimensionless diffusivity in lattice units
    // Formula: alpha_lattice = alpha_physical * dt / (dx²)
    //
    // Physical: alpha ~ 5.8e-6 m²/s (Ti-6Al-4V)
    // dt: e.g., 1e-7 s (0.1 μs)
    // dx: e.g., 2e-6 m (2 μm)
    //
    // Example: alpha_lattice = 5.8e-6 * 1e-7 / (2e-6)² = 0.145 (dimensionless)
    // ============================================================================

    thermal_diff_lattice_ = thermal_diffusivity * dt / (dx * dx);

    // Compute tau from LATTICE diffusivity
    // For D3Q7: alpha = cs^2 * (tau_T - 0.5)
    // Therefore: tau_T = alpha / cs^2 + 0.5
    tau_T_ = thermal_diff_lattice_ / D3Q7::CS2 + 0.5f;
    omega_T_ = 1.0f / tau_T_;

    // ============================================================
    // BGK Stability for High-Peclet Advection-Diffusion
    // ============================================================
    // Reference: "Stability limits of single relaxation-time
    //            advection-diffusion LBM" (Int. J. Mod. Phys. C 2017)
    //
    // Theory:
    //   - Standard BGK: ω < 2.0 (von Neumann stability)
    //   - High-Pe flows: ω < 1.5 (advection stability)
    //   - Current Pe ≈ 10 >> 2.0 → requires ω ≤ 1.5
    //
    // Trade-off:
    //   - Lower ω → more implicit diffusion (over-diffusive)
    //   - Higher ω → less stable (under-relaxed)
    //   - ω = 1.45 is a compromise for Pe ~ 10
    //
    // Note: This reduction is temporary. Proper solution is MRT
    //       collision operator (decouples diffusion and advection).
    // ============================================================

    // Fix: Only clamp near true instability (omega >= 1.9), not at omega >= 1.5
    // This preserves physical diffusivity while maintaining stability
    if (omega_T_ >= 1.95f) {
        std::cerr << "WARNING: omega_T = " << omega_T_
                  << " critically unstable! Clamping to 1.85.\n";
        omega_T_ = 1.85f;
        tau_T_ = 1.0f / omega_T_;
    } else if (omega_T_ >= 1.9f) {
        std::cout << "INFO: omega_T = " << omega_T_
                  << " is high. Reducing to 1.85 for stability.\n";
        omega_T_ = 1.85f;
        tau_T_ = 1.0f / omega_T_;
    }
    // omega in [1.5, 1.9) now allowed - preserves physical diffusivity

    std::cout << "ThermalLBM initialized:\n"
              << "  Domain: " << nx_ << " x " << ny_ << " x " << nz_ << "\n"
              << "  dt = " << dt_ << " s\n"
              << "  dx = " << dx_ << " m\n"
              << "  alpha_physical = " << thermal_diff_physical_ << " m²/s\n"
              << "  alpha_lattice = " << thermal_diff_lattice_ << " (dimensionless)\n"
              << "  tau = " << tau_T_ << "\n"
              << "  omega = " << omega_T_ << "\n"
              << "  CFL_thermal = " << (thermal_diff_lattice_ / D3Q7::CS2) << " (should be < 0.5)\n"
              << "  Density: " << rho_ << " kg/m³\n"
              << "  Specific heat: " << cp_ << " J/(kg·K)" << std::endl;

    // Allocate device memory
    allocateMemory();
}

// Constructor with phase change support
ThermalLBM::ThermalLBM(int nx, int ny, int nz,
                       const MaterialProperties& material,
                       float thermal_diffusivity,
                       bool enable_phase_change,
                       float dt, float dx)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      dt_(dt), dx_(dx),
      thermal_diff_physical_(thermal_diffusivity),
      rho_(material.rho_solid), cp_(material.cp_solid),
      emissivity_(0.35f),  // Default emissivity (will be overridden by config)
      material_(material), has_material_(true),
      phase_solver_(nullptr) {

    // Initialize D3Q7 lattice if not already done
    if (!D3Q7::isInitialized()) {
        D3Q7::initializeDevice();
    }

    // ============================================================================
    // CRITICAL FIX: Convert thermal diffusivity to lattice units
    // ============================================================================
    // LBM requires dimensionless diffusivity in lattice units
    // Formula: alpha_lattice = alpha_physical * dt / (dx²)
    //
    // Physical: alpha ~ 5.8e-6 m²/s (Ti-6Al-4V)
    // dt: e.g., 1e-7 s (0.1 μs)
    // dx: e.g., 2e-6 m (2 μm)
    //
    // Example: alpha_lattice = 5.8e-6 * 1e-7 / (2e-6)² = 0.145 (dimensionless)
    // ============================================================================

    thermal_diff_lattice_ = thermal_diffusivity * dt / (dx * dx);

    // Compute tau from LATTICE diffusivity
    // For D3Q7: alpha = cs^2 * (tau_T - 0.5)
    // Therefore: tau_T = alpha / cs^2 + 0.5
    tau_T_ = thermal_diff_lattice_ / D3Q7::CS2 + 0.5f;
    omega_T_ = 1.0f / tau_T_;

    // ============================================================
    // BGK Stability for High-Peclet Advection-Diffusion
    // ============================================================
    // Reference: "Stability limits of single relaxation-time
    //            advection-diffusion LBM" (Int. J. Mod. Phys. C 2017)
    //
    // Theory:
    //   - Standard BGK: ω < 2.0 (von Neumann stability)
    //   - High-Pe flows: ω < 1.5 (advection stability)
    //   - Current Pe ≈ 10 >> 2.0 → requires ω ≤ 1.5
    //
    // Trade-off:
    //   - Lower ω → more implicit diffusion (over-diffusive)
    //   - Higher ω → less stable (under-relaxed)
    //   - ω = 1.45 is a compromise for Pe ~ 10
    //
    // Note: This reduction is temporary. Proper solution is MRT
    //       collision operator (decouples diffusion and advection).
    // ============================================================

    // Fix: Only clamp near true instability (omega >= 1.9), not at omega >= 1.5
    // This preserves physical diffusivity while maintaining stability
    if (omega_T_ >= 1.95f) {
        std::cerr << "WARNING: omega_T = " << omega_T_
                  << " critically unstable! Clamping to 1.85.\n";
        omega_T_ = 1.85f;
        tau_T_ = 1.0f / omega_T_;
    } else if (omega_T_ >= 1.9f) {
        std::cout << "INFO: omega_T = " << omega_T_
                  << " is high. Reducing to 1.85 for stability.\n";
        omega_T_ = 1.85f;
        tau_T_ = 1.0f / omega_T_;
    }
    // omega in [1.5, 1.9) now allowed - preserves physical diffusivity

    std::cout << "ThermalLBM initialized with phase change:\n"
              << "  Domain: " << nx_ << " x " << ny_ << " x " << nz_ << "\n"
              << "  Material: " << material_.name << "\n"
              << "  dt = " << dt_ << " s\n"
              << "  dx = " << dx_ << " m\n"
              << "  alpha_physical = " << thermal_diff_physical_ << " m²/s\n"
              << "  alpha_lattice = " << thermal_diff_lattice_ << " (dimensionless)\n"
              << "  tau = " << tau_T_ << "\n"
              << "  omega = " << omega_T_ << "\n"
              << "  CFL_thermal = " << (thermal_diff_lattice_ / D3Q7::CS2) << " (should be < 0.5)\n"
              << "  Density: " << rho_ << " kg/m³\n"
              << "  Specific heat: " << cp_ << " J/(kg·K)\n"
              << "  Melting point: " << material_.T_liquidus << " K\n"
              << "  Phase change: " << (enable_phase_change ? "ENABLED" : "DISABLED") << std::endl;

    // Allocate device memory
    allocateMemory();

    // Create phase change solver if enabled
    if (enable_phase_change) {
        phase_solver_ = new PhaseChangeSolver(nx_, ny_, nz_, material_);
        std::cout << "  Phase change solver created\n";
    }
}

// Destructor
ThermalLBM::~ThermalLBM() {
    freeMemory();
    if (phase_solver_) {
        delete phase_solver_;
    }
}

// Allocate device memory
void ThermalLBM::allocateMemory() {
    cudaError_t err;
    size_t size_dist = num_cells_ * D3Q7::Q * sizeof(float);
    size_t size_scalar = num_cells_ * sizeof(float);

    // Allocate distribution functions
    err = cudaMalloc(&d_g_src, size_dist);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate d_g_src: " +
                               std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_g_dst, size_dist);
    if (err != cudaSuccess) {
        cudaFree(d_g_src);
        throw std::runtime_error("Failed to allocate d_g_dst: " +
                               std::string(cudaGetErrorString(err)));
    }

    // Allocate temperature field
    err = cudaMalloc(&d_temperature, size_scalar);
    if (err != cudaSuccess) {
        cudaFree(d_g_src);
        cudaFree(d_g_dst);
        throw std::runtime_error("Failed to allocate d_temperature: " +
                               std::string(cudaGetErrorString(err)));
    }

    // Initialize to zero
    cudaMemset(d_g_src, 0, size_dist);
    cudaMemset(d_g_dst, 0, size_dist);
    cudaMemset(d_temperature, 0, size_scalar);
}

// Free device memory
void ThermalLBM::freeMemory() {
    if (d_g_src) cudaFree(d_g_src);
    if (d_g_dst) cudaFree(d_g_dst);
    if (d_temperature) cudaFree(d_temperature);
}

// Swap source and destination distributions
void ThermalLBM::swapDistributions() {
    float* temp = d_g_src;
    d_g_src = d_g_dst;
    d_g_dst = temp;
}

// Initialize with uniform temperature
void ThermalLBM::initialize(float initial_temp) {
    // Set uniform temperature
    cudaMemset(d_temperature, 0, num_cells_ * sizeof(float));

    // Create uniform temperature array
    float* h_temp = new float[num_cells_];
    for (int i = 0; i < num_cells_; ++i) {
        h_temp[i] = initial_temp;
    }

    // Copy to device
    cudaMemcpy(d_temperature, h_temp, num_cells_ * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_temp;

    // Initialize distribution functions to equilibrium
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Simple kernel launch without lambda
    initializeEquilibriumKernel<<<gridSize, blockSize>>>(d_g_src, d_temperature, num_cells_);

    cudaDeviceSynchronize();

    // Initialize phase change solver if enabled
    if (phase_solver_) {
        phase_solver_->initializeFromTemperature(d_temperature);
    }
}

// Initialize with custom temperature field
void ThermalLBM::initialize(const float* temp_field) {
    // Copy temperature field to device
    cudaMemcpy(d_temperature, temp_field, num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize distribution functions to equilibrium
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Simple kernel launch without lambda
    initializeEquilibriumKernel<<<gridSize, blockSize>>>(d_g_src, d_temperature, num_cells_);

    cudaDeviceSynchronize();

    // Initialize phase change solver if enabled
    if (phase_solver_) {
        phase_solver_->initializeFromTemperature(d_temperature);
    }
}

// Perform BGK collision
void ThermalLBM::collisionBGK(const float* ux, const float* uy, const float* uz) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    thermalBGKCollisionKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature, ux, uy, uz, omega_T_, nx_, ny_, nz_);

    cudaDeviceSynchronize();
}

// Perform streaming
void ThermalLBM::streaming() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    // ADIABATIC BOUNDARIES: Initialize g_dst to zero
    // Bounce-back will write reflected distributions
    cudaMemset(d_g_dst, 0, num_cells_ * D3Q7::Q * sizeof(float));

    // Stream all distributions with adiabatic bounce-back at boundaries
    thermalStreamingKernel<<<gridSize, blockSize>>>(
        d_g_src, d_g_dst, nx_, ny_, nz_, emissivity_);

    cudaDeviceSynchronize();
    swapDistributions();
}

// Apply boundary conditions
void ThermalLBM::applyBoundaryConditions(int boundary_type, float boundary_value) {
    if (boundary_type == 0) {
        // Periodic boundaries - handled automatically in streaming
        return;
    } else if (boundary_type == 1) {
        // Constant temperature boundaries
        dim3 blockSize(256);

        // Apply to all 6 faces
        for (int face = 0; face < 6; ++face) {
            int face_size;
            if (face < 2) face_size = ny_ * nz_;       // x-faces
            else if (face < 4) face_size = nx_ * nz_;  // y-faces
            else face_size = nx_ * ny_;                // z-faces

            dim3 gridSize((face_size + blockSize.x - 1) / blockSize.x);

            applyConstantTemperatureBoundary<<<gridSize, blockSize>>>(
                d_g_src, d_temperature, boundary_value, nx_, ny_, nz_, face);
        }
        cudaDeviceSynchronize();

    } else if (boundary_type == 2) {
        // Adiabatic boundaries
        dim3 blockSize(256);

        // Apply to all 6 faces
        for (int face = 0; face < 6; ++face) {
            int face_size;
            if (face < 2) face_size = ny_ * nz_;       // x-faces
            else if (face < 4) face_size = nx_ * nz_;  // y-faces
            else face_size = nx_ * ny_;                // z-faces

            dim3 gridSize((face_size + blockSize.x - 1) / blockSize.x);

            applyAdiabaticBoundary<<<gridSize, blockSize>>>(
                d_g_src, nx_, ny_, nz_, face);
        }
        cudaDeviceSynchronize();
    }
}

// Add heat source
void ThermalLBM::addHeatSource(const float* heat_source, float dt) {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Use material properties if available, otherwise fallback to hardcoded values
    if (has_material_) {
        addHeatSourceKernel<<<gridSize, blockSize>>>(
            d_g_src, heat_source, d_temperature, dt, omega_T_, material_, num_cells_);
    } else {
        // Backward compatibility: create temporary material with fixed properties
        MaterialProperties temp_mat;
        temp_mat.rho_solid = rho_;
        temp_mat.rho_liquid = rho_;
        temp_mat.cp_solid = cp_;
        temp_mat.cp_liquid = cp_;
        temp_mat.T_solidus = 0.0f;
        temp_mat.T_liquidus = 0.0f;
        addHeatSourceKernel<<<gridSize, blockSize>>>(
            d_g_src, heat_source, d_temperature, dt, omega_T_, temp_mat, num_cells_);
    }

    cudaDeviceSynchronize();

    // CRITICAL: Update temperature field after adding heat
    // The collision step uses d_temperature, so we must recompute it
    computeTemperature();
}

// Apply radiation boundary condition
void ThermalLBM::applyRadiationBC(float dt, float dx, float epsilon, float T_ambient) {
    // Launch 2D grid for top surface (nx × ny)
    dim3 blockSize(16, 16);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y);

    // Use material properties if available, otherwise fallback to hardcoded values
    if (has_material_) {
        applyRadiationBoundaryCondition<<<gridSize, blockSize>>>(
            d_g_src, d_temperature, nx_, ny_, nz_,
            dx, dt, epsilon, material_, T_ambient);
    } else {
        // Backward compatibility: create temporary material with fixed properties
        MaterialProperties temp_mat;
        temp_mat.rho_solid = rho_;
        temp_mat.rho_liquid = rho_;
        temp_mat.cp_solid = cp_;
        temp_mat.cp_liquid = cp_;
        temp_mat.T_solidus = 0.0f;
        temp_mat.T_liquidus = 0.0f;
        applyRadiationBoundaryCondition<<<gridSize, blockSize>>>(
            d_g_src, d_temperature, nx_, ny_, nz_,
            dx, dt, epsilon, temp_mat, T_ambient);
    }

    cudaDeviceSynchronize();

    // Update temperature field after radiation cooling
    computeTemperature();
}

// Apply substrate cooling boundary condition
void ThermalLBM::applySubstrateCoolingBC(float dt, float dx, float h_conv, float T_substrate) {
    // Use material properties if available, otherwise fallback to class defaults
    // (matches pattern in addHeatSource for backward compatibility)
    float rho = has_material_ ? material_.rho_solid : rho_;
    float cp = has_material_ ? material_.cp_solid : cp_;

    // Launch 2D grid for bottom surface (nx × ny)
    dim3 blockSize(16, 16);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y);

    applySubstrateCoolingKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature,
        nx_, ny_, nz_,
        dx, dt, h_conv, T_substrate,
        rho, cp
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Substrate BC kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    // Update temperature field after substrate cooling
    computeTemperature();
}

// Compute temperature from distribution
void ThermalLBM::computeTemperature() {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    computeTemperatureKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature, num_cells_);

    cudaDeviceSynchronize();

    // Update liquid fraction if phase change is enabled
    if (phase_solver_) {
        phase_solver_->updateLiquidFraction(d_temperature);
    }
}

// Copy temperature to host
void ThermalLBM::copyTemperatureToHost(float* host_temp) const {
    cudaMemcpy(host_temp, d_temperature, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);
}

// Compute thermal relaxation time
float ThermalLBM::computeThermalTau(float alpha, float dx, float dt) {
    // Convert physical diffusivity to lattice units
    float alpha_lattice = alpha * dt / (dx * dx);
    return alpha_lattice / D3Q7::CS2 + 0.5f;
}

// Get liquid fraction field (non-const)
float* ThermalLBM::getLiquidFraction() {
    if (!phase_solver_) {
        return nullptr;
    }
    return phase_solver_->getLiquidFraction();
}

// Get liquid fraction field (const)
const float* ThermalLBM::getLiquidFraction() const {
    if (!phase_solver_) {
        return nullptr;
    }
    return phase_solver_->getLiquidFraction();
}

// Copy liquid fraction to host
void ThermalLBM::copyLiquidFractionToHost(float* host_fl) const {
    if (!phase_solver_) {
        throw std::runtime_error("Phase change is not enabled in this ThermalLBM instance");
    }
    phase_solver_->copyLiquidFractionToHost(host_fl);
}

// ============= CUDA Kernels =============

// Helper kernel for initialization
__global__ void initializeEquilibriumKernel(float* g_src, const float* temperature, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    // At rest: g_eq = w_i * T
    for (int q = 0; q < D3Q7::Q; ++q) {
        g_src[idx * D3Q7::Q + q] = D3Q7::computeThermalEquilibrium(q, T, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void thermalBGKCollisionKernel(
    float* g_src,
    const float* temperature,
    const float* ux,
    const float* uy,
    const float* uz,
    float omega_T,
    int nx, int ny, int nz) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    float T = temperature[idx];

    // Get velocity (use zero if not provided)
    float vel_x = ux ? ux[idx] : 0.0f;
    float vel_y = uy ? uy[idx] : 0.0f;
    float vel_z = uz ? uz[idx] : 0.0f;

    // BGK collision for each direction
    for (int q = 0; q < D3Q7::Q; ++q) {
        int dist_idx = idx * D3Q7::Q + q;
        float g_eq = D3Q7::computeThermalEquilibrium(q, T, vel_x, vel_y, vel_z);

        // BGK collision: g_new = g - omega * (g - g_eq)
        g_src[dist_idx] = g_src[dist_idx] - omega_T * (g_src[dist_idx] - g_eq);
    }
}

__global__ void thermalStreamingKernel(
    const float* g_src,
    float* g_dst,
    int nx, int ny, int nz,
    float emissivity) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;

    // ============================================================================
    // BUG FIX 2025-11-18: Removed incorrect pseudo-radiation implementation
    // ============================================================================
    // PROBLEM IDENTIFIED:
    //   Previous code computed "reflection_coeff = 1 - q_rad/q_cond" and applied
    //   it in the streaming kernel. This violates LBM theory and physics:
    //
    //   1. PHYSICAL ERROR: Radiation and conduction are independent mechanisms.
    //      Their ratio (q_rad/q_cond) has no physical basis for reflection.
    //
    //   2. LBM THEORY ERROR: Streaming should only propagate distributions.
    //      Boundary conditions must be applied separately (operator splitting).
    //
    //   3. NUMERICAL ERROR: Clamping to [0.7, 0.98] was arbitrary and backwards:
    //      - High T should → strong radiation → LOW reflection (→ 0.0)
    //      - Low T should → weak radiation → HIGH reflection (→ 1.0)
    //      - Code did the opposite
    //
    //   4. DOUBLE IMPLEMENTATION CONFLICT:
    //      - Explicit radiation BC (applyRadiationBoundaryCondition) exists
    //      - Pseudo-radiation in streaming weakened the true radiation effect
    //
    // SOLUTION:
    //   Use standard LBM streaming with bounce-back for boundaries.
    //   Radiation BC is properly handled by applyRadiationBoundaryCondition().
    //
    // REFERENCE:
    //   - Standard LBM: streaming propagates f_i(x + e_i*dt) = f_i_post(x)
    //   - Adiabatic BC: full bounce-back (reflection_coeff = 1.0)
    //   - Radiation BC: applied via source term, NOT via reflection coefficient
    // ============================================================================

    // STANDARD LBM BOUNDARY CONDITIONS:
    // - X, Y boundaries: Adiabatic (full bounce-back for zero flux)
    // - Z-top (z=nz-1): Adiabatic bounce-back (radiation applied separately)
    // - Z-bottom (z=0): Adiabatic (substrate)

    for (int q = 0; q < D3Q7::Q; ++q) {
        int cx, cy, cz;
#ifdef __CUDA_ARCH__
        cx = tex[q];
        cy = tey[q];
        cz = tez[q];
#else
        cx = 0; cy = 0; cz = 0;
#endif

        // Calculate target position
        int nx_target = x + cx;
        int ny_target = y + cy;
        int nz_target = z + cz;

        // Check if target is inside domain
        if (nx_target >= 0 && nx_target < nx &&
            ny_target >= 0 && ny_target < ny &&
            nz_target >= 0 && nz_target < nz) {

            // Normal streaming: target is inside domain
            int target_idx = nx_target + ny_target * nx + nz_target * nx * ny;
            g_dst[target_idx * D3Q7::Q + q] = g_src[idx * D3Q7::Q + q];
        } else {
            // Boundary: apply standard bounce-back
            // Find opposite direction for bounce-back
            int q_opposite = 0;
            if (q == 1) q_opposite = 2;      // +x -> -x
            else if (q == 2) q_opposite = 1; // -x -> +x
            else if (q == 3) q_opposite = 4; // +y -> -y
            else if (q == 4) q_opposite = 3; // -y -> +y
            else if (q == 5) q_opposite = 6; // +z -> -z
            else if (q == 6) q_opposite = 5; // -z -> +z

            // CORRECTED: Standard adiabatic bounce-back for ALL boundaries
            // Radiation is applied separately in applyRadiationBoundaryCondition()
            // This ensures clean separation of concerns per LBM theory
            g_dst[idx * D3Q7::Q + q_opposite] = g_src[idx * D3Q7::Q + q];
        }
    }
}

__global__ void computeTemperatureKernel(
    const float* g,
    float* temperature,
    int num_cells) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Sum all distribution functions
    float T = 0.0f;
    for (int q = 0; q < D3Q7::Q; ++q) {
        T += g[idx * D3Q7::Q + q];
    }

    // ============================================================
    // Physical temperature bounds for Ti6Al4V
    // ============================================================
    // Lower bound: Ambient temperature (no active sub-ambient cooling modeled)
    // Upper bound: 2× vaporization temperature
    //
    // Physical justification:
    //   T_ambient = 300 K (room temperature)
    //   T_melt = 1923 K (solidus)
    //   T_boil = 3560 K (vaporization)
    //
    // At T > 7000 K, all material is vaporized (gas phase).
    // Further heating is non-physical in condensed-phase model.
    //
    // BUG FIX (2025-11-21): Changed T_MIN from 0.0 to 300.0
    // Temperature cannot go below ambient without active cooling
    // below ambient (not modeled). Sub-ambient temperatures indicate
    // numerical errors from incorrect energy extraction.
    // ============================================================

    constexpr float T_MIN = 300.0f;   // Cannot go below ambient temperature
    constexpr float T_MAX = 7000.0f;  // 2× T_vaporization for safety margin

    T = fmaxf(T, T_MIN);
    T = fminf(T, T_MAX);

    temperature[idx] = T;
}

__global__ void applyConstantTemperatureBoundary(
    float* g,
    float* temperature,
    float T_boundary,
    int nx, int ny, int nz,
    int boundary_face) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int x, y, z;

    // Determine position based on boundary face
    switch (boundary_face) {
        case 0: // x_min
            if (tid >= ny * nz) return;
            x = 0;
            y = tid % ny;
            z = tid / ny;
            break;
        case 1: // x_max
            if (tid >= ny * nz) return;
            x = nx - 1;
            y = tid % ny;
            z = tid / ny;
            break;
        case 2: // y_min
            if (tid >= nx * nz) return;
            y = 0;
            x = tid % nx;
            z = tid / nx;
            break;
        case 3: // y_max
            if (tid >= nx * nz) return;
            y = ny - 1;
            x = tid % nx;
            z = tid / nx;
            break;
        case 4: // z_min
            if (tid >= nx * ny) return;
            z = 0;
            x = tid % nx;
            y = tid / nx;
            break;
        case 5: // z_max
            if (tid >= nx * ny) return;
            z = nz - 1;
            x = tid % nx;
            y = tid / nx;
            break;
        default:
            return;
    }

    int idx = x + y * nx + z * nx * ny;

    // Set temperature
    temperature[idx] = T_boundary;

    // Set equilibrium distribution for boundary temperature
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[idx * D3Q7::Q + q] = D3Q7::computeThermalEquilibrium(q, T_boundary, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void applyAdiabaticBoundary(
    float* g,
    int nx, int ny, int nz,
    int boundary_face) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int x, y, z;

    // Determine position based on boundary face
    switch (boundary_face) {
        case 0: // x_min
            if (tid >= ny * nz) return;
            x = 0;
            y = tid % ny;
            z = tid / ny;
            break;
        case 1: // x_max
            if (tid >= ny * nz) return;
            x = nx - 1;
            y = tid % ny;
            z = tid / ny;
            break;
        case 2: // y_min
            if (tid >= nx * nz) return;
            y = 0;
            x = tid % nx;
            z = tid / nx;
            break;
        case 3: // y_max
            if (tid >= nx * nz) return;
            y = ny - 1;
            x = tid % nx;
            z = tid / nx;
            break;
        case 4: // z_min
            if (tid >= nx * ny) return;
            z = 0;
            x = tid % nx;
            y = tid / nx;
            break;
        case 5: // z_max
            if (tid >= nx * ny) return;
            z = nz - 1;
            x = tid % nx;
            y = tid / nx;
            break;
        default:
            return;
    }

    int idx = x + y * nx + z * nx * ny;

    // For adiabatic boundary, copy distribution from neighboring interior cell
    // This implements zero-flux condition
    int interior_x = x, interior_y = y, interior_z = z;

    if (boundary_face == 0) interior_x = 1;
    else if (boundary_face == 1) interior_x = nx - 2;
    else if (boundary_face == 2) interior_y = 1;
    else if (boundary_face == 3) interior_y = ny - 2;
    else if (boundary_face == 4) interior_z = 1;
    else if (boundary_face == 5) interior_z = nz - 2;

    int interior_idx = interior_x + interior_y * nx + interior_z * nx * ny;

    // Copy distribution from interior
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[idx * D3Q7::Q + q] = g[interior_idx * D3Q7::Q + q];
    }
}

__global__ void addHeatSourceKernel(
    float* g,
    const float* heat_source,
    const float* temperature,
    float dt,
    float omega_T,
    MaterialProperties material,
    int num_cells) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Get local temperature
    float T = temperature[idx];

    // Get temperature-dependent properties
    float rho = material.getDensity(T);
    float cp = material.getSpecificHeat(T);

    // Heat source Q is in [W/m³] = [J/(m³·s)]
    // Convert to temperature increase:
    // Q * dt gives [J/m³] (energy density)
    // dT = Q * dt / (rho * cp)  [K]
    //
    // Where:
    //   Q: volumetric heat source [W/m³]
    //   dt: time step [s]
    //   rho: density [kg/m³] - temperature-dependent
    //   cp: specific heat [J/(kg·K)] - temperature-dependent
    float Q = heat_source[idx];
    float dT = (Q * dt) / (rho * cp);

    // ============================================================================
    // LBM SOURCE TERM CORRECTION - REMOVED (Bug Fix)
    // ============================================================================
    // PREVIOUS CODE (BUGGY):
    //   float source_correction = 1.0f / (1.0f - 0.5f * omega_T);  // ≈ 3.636 for ω=1.45
    //
    // ISSUE: The Chapman-Enskog correction from Guo et al. (2002) applies to
    // FORCING TERMS (momentum/velocity sources), NOT scalar transport (temperature).
    //
    // For scalar advection-diffusion equation, the source term is added directly
    // without correction. The collision operator already handles relaxation properly.
    //
    // ENERGY CONSERVATION BUG:
    //   - With correction: Energy deposited = P_laser * 3.636 (violates conservation!)
    //   - Without correction: Energy deposited = P_laser (correct!)
    //
    // REFERENCES:
    //   - Guo et al. (2002) PRE 65, 046308: Force correction for momentum equation
    //   - Li et al. (2013) PRE 87, 053301: No correction needed for scalar sources
    //
    // FIX: Remove correction factor (set to 1.0)
    // ============================================================================
    float source_correction = 1.0f;  // FIX: No correction for thermal sources

    // D3Q7 weights: w0 = 1/4, w1-6 = 1/8
    const float weights[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};

    // Add temperature increase equally to all distributions
    // This preserves the temperature increase while maintaining isotropy
    // Apply correction factor to ensure correct energy deposition
    for (int q = 0; q < 7; ++q) {
        g[idx * 7 + q] += weights[q] * dT * source_correction;
    }
}

/**
 * @brief Apply radiation and evaporation boundary conditions to top surface
 *
 * Implements combined thermal boundary conditions:
 *
 * 1. Stefan-Boltzmann radiation heat loss:
 *    q_rad = ε·σ·(T_surface⁴ - T_ambient⁴)  [W/m²]
 *
 * 2. Hertz-Knudsen-Langmuir evaporation cooling (when T > T_boil):
 *    J_evap = α_evap · P_sat(T) / sqrt(2π·R·T/M)  [kg/(m²·s)]
 *    P_sat(T) = P_ref · exp[(L_vap·M/R) · (1/T_boil - 1/T)]  [Pa]
 *    q_evap = J_evap · L_vap  [W/m²]
 *
 * Total cooling heat flux:
 *   q_total = q_rad + q_evap  [W/m²]
 *
 * Temperature decrease:
 *   ΔT = -q_total · dt / (ρ·c_p·dx)  [K]
 *
 * Physics constants for Ti6Al4V:
 *   - Evaporation coefficient α_evap = 0.82
 *   - Molar mass M = 0.0479 kg/mol
 *   - Gas constant R = 8.314 J/(mol·K)
 *   - Reference pressure P_ref = 101,325 Pa (at T_boil)
 *   - Latent heat L_vap from material.L_vaporization
 *   - Boiling point T_boil from material.T_vaporization
 *
 * Numerical stability:
 *   - Explicit scheme with adaptive limiter based on temperature regime
 *   - Evaporation only active when T > T_boil to avoid unphysical cooling
 *   - Combined cooling limited to 10-25% per timestep depending on regime
 */
__global__ void applyRadiationBoundaryCondition(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float epsilon,
    MaterialProperties material,
    float T_ambient) {

    // Top surface only (z = nz-1)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    int z = nz - 1;  // Top surface
    int idx = x + y * nx + z * nx * ny;

    float T_surf = temperature[idx];

    // Get temperature-dependent properties
    float rho = material.getDensity(T_surf);
    float cp = material.getSpecificHeat(T_surf);

    // ============================================================================
    // RADIATION COOLING: Stefan-Boltzmann law
    // ============================================================================
    const float sigma = 5.67e-8f;  // Stefan-Boltzmann constant [W/(m²·K⁴)]
    float q_rad = epsilon * sigma * (powf(T_surf, 4.0f) - powf(T_ambient, 4.0f));

    // ============================================================================
    // EVAPORATION COOLING: Hertz-Knudsen-Langmuir model (only when T > T_boil)
    // ============================================================================
    float q_evap = 0.0f;  // Evaporative heat flux [W/m²]

    // Physical constants for Ti6Al4V evaporation
    // CRITICAL FIX (2025-11-27): Reduced from 0.82 to 0.18 to prevent excessive evaporation
    const float alpha_evap = 0.18f;           // Evaporation coefficient (dimensionless)
    const float M_molar = 0.0479f;            // Molar mass [kg/mol]
    const float R_gas = 8.314f;               // Universal gas constant [J/(mol·K)]
    const float P_ref = 101325.0f;            // Reference pressure at boiling point [Pa]
    const float PI = 3.14159265359f;

    // Get material-specific properties
    float T_boil = material.T_vaporization;   // Boiling temperature [K]
    float L_vap = material.L_vaporization;    // Latent heat of vaporization [J/kg]

    // Only apply evaporation cooling when surface temperature exceeds boiling point
    if (T_surf > T_boil) {
        // Clausius-Clapeyron equation for vapor pressure
        // P_sat(T) = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]
        // OVERFLOW PROTECTION: Cap temperature and exponent
        float T_capped = fminf(T_surf, 2.0f * T_boil);
        float exponent = (L_vap * M_molar / R_gas) * (1.0f / T_boil - 1.0f / T_capped);
        exponent = fminf(exponent, 20.0f);  // Prevent exp overflow
        float P_sat = P_ref * expf(exponent);
        P_sat = fminf(P_sat, 10.0f * P_ref);  // Cap to 10 atm

        // Hertz-Knudsen-Langmuir evaporation mass flux
        // J_evap = α_evap * P_sat / sqrt(2π * R * T / M)  [kg/(m²·s)]
        float denominator = sqrtf(2.0f * PI * R_gas * T_capped / M_molar);  // [m/s]
        float J_evap = alpha_evap * P_sat / denominator;  // [kg/(m²·s)]

        // Evaporative cooling heat flux
        // q_evap = J_evap * L_vap  [W/m²]
        q_evap = J_evap * L_vap;
    }

    // ============================================================================
    // COMBINED COOLING: Total heat flux and temperature change
    // ============================================================================
    float q_total = q_rad + q_evap;

    // Convert to temperature decrease
    // q_total [W/m²] → energy loss per unit volume [W/m³] = q_total / dx
    // Temperature change: ΔT = -(q_total/dx) · dt / (ρ·c_p)
    // Using temperature-dependent ρ and cp for accurate energy conservation
    float dT = -(q_total / dx) * dt / (rho * cp);

    // Adaptive stability limiter based on temperature regime
    // Reduced from original values to prevent excessive cooling
    // - Low T (< 5kK): 15% max cooling (was 25%)
    // - Medium T (5-15kK): 12% max cooling (was 15%)
    // - High T (> 15kK): 10% max cooling (unchanged)
    float max_cooling;
    if (T_surf < 5000.0f) {
        max_cooling = -0.15f * T_surf;  // 15% for low-temperature regime (reduced from 25%)
    } else if (T_surf < 15000.0f) {
        max_cooling = -0.12f * T_surf;  // 12% for medium-temperature regime (reduced from 15%)
    } else {
        max_cooling = -0.10f * T_surf;  // 10% for high-temperature regime
    }

    if (dT < max_cooling) {
        dT = max_cooling;
    }

    // Apply temperature change to all distributions (maintains isotropy)
    const float weights[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[idx * D3Q7::Q + q] += weights[q] * dT;
    }
}

/**
 * @brief Apply substrate cooling boundary condition to entire substrate volume
 *
 * Implements convective heat transfer to water-cooled substrate:
 *   q_conv = h_conv * (T_cell - T_substrate)  [W/m²]
 *
 * Physics:
 *   - Water-cooled substrate: h_conv ≈ 50,000 W/(m²·K) (calibrated value)
 *   - Typical substrate temperature: T_substrate = 300 K
 *   - Heat flux is proportional to temperature difference (Newton's law of cooling)
 *
 * Implementation (Plan B):
 *   - Apply to entire substrate volume (z=0 to z=70 μm, k=0..7)
 *   - Extends cooling beyond powder-substrate interface to improve heat extraction
 *   - Heat loss rate per volume: q_conv / dx [W/m³]
 *   - Temperature change: dT = -q_loss * dt / (ρ*cp) [K]
 *   - Negative sign: heat is leaving the domain
 */
__global__ void applySubstrateCoolingKernel(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float h_conv,
    float T_substrate,
    float rho,
    float cp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Apply substrate cooling to entire substrate volume (k=0 to k=substrate_depth)
    // This models the effective heat extraction through the substrate
    // The cooling strength decays exponentially with distance from bottom
    // Extended to 60% of domain height to capture heat from melt pool region
    const int substrate_depth = min(60, (nz * 3) / 5);  // ~120 μm or 60% of domain

    for (int k = 0; k < substrate_depth; ++k) {
        int idx = i + nx * (j + ny * k);

        float T_cell = temperature[idx];

        // No cooling if cell is already at or below substrate temperature
        if (T_cell <= T_substrate) {
            continue;
        }

        // Decay factor: cooling strength decreases with distance from substrate
        // Models the finite thermal resistance through the solid
        // At k=0: full h_conv, at k=substrate_depth: ~37% of h_conv
        float decay = expf(-1.0f * k / (float)substrate_depth);
        float h_effective = h_conv * decay;

        // Convective heat flux [W/m²]
        float q_conv = h_effective * (T_cell - T_substrate);

        // Heat loss rate per volume [W/m³]
        float heat_rate = q_conv / dx;

        // Temperature change from substrate cooling
        float dT = -heat_rate * dt / (rho * cp);

        // CFL-type stability limiter (prevent unphysical cooling)
        // Never cool more than 10% of temperature difference per timestep
        float max_cooling = -0.10f * (T_cell - T_substrate);
        if (dT < max_cooling) {
            dT = max_cooling;
        }

        // Apply to all distribution functions (isotropic cooling)
        const float weights[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
        for (int q = 0; q < D3Q7::Q; ++q) {
            g[idx * D3Q7::Q + q] += weights[q] * dT;
        }
    }
}

// ============================================================================
// Energy Diagnostics Kernels
// ============================================================================

/**
 * @brief Compute evaporation power at surface cells
 *
 * BUG FIX (Nov 21, 2025): Removed fill_level filter to match BC kernel behavior
 *
 * The applyRadiationBoundaryCondition() kernel applies evaporation cooling
 * to ALL top surface cells regardless of fill_level. The diagnostic kernel
 * must match this behavior to report the correct power.
 *
 * Previous bug: Only counted evaporation for interface cells (0.1 < f < 0.9),
 * but BC kernel applied evaporation to ALL cells. Result: P_evap = 0 W.
 */
__global__ void computeEvaporationPowerKernel(
    const float* temperature,
    const float* fill_level,
    float* power_out,
    MaterialProperties material,
    float dx,
    int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    // Top surface only
    int z = nz - 1;
    int idx = x + y * nx + z * nx * ny;

    float T = temperature[idx];
    // NOTE: fill_level is no longer used for filtering (matches BC kernel)
    // The BC kernel (applyRadiationBoundaryCondition) applies evaporation to
    // ALL top surface cells when T > T_boil, regardless of fill_level.
    // float f = fill_level[idx];

    // Check if above boiling point (matches BC kernel logic at line 1075)
    if (T < material.T_vaporization) {
        power_out[idx] = 0.0f;
        return;
    }

    // Compute evaporation mass flux using Hertz-Knudsen equation
    const float alpha_evap = 0.82f;  // Evaporation coefficient
    const float M = 0.0479f;  // kg/mol (Ti6Al4V)
    const float R = 8.314f;   // J/(mol·K)
    const float P_ref = 101325.0f;  // Pa
    const float PI = 3.14159265f;

    // Clausius-Clapeyron: P_sat = P_ref * exp[(L_v*M/R) * (1/T_boil - 1/T)]
    float T_boil = material.T_vaporization;
    float L_vap = material.L_vaporization;
    // OVERFLOW PROTECTION: Cap temperature and exponent
    float T_capped = fminf(T, 2.0f * T_boil);
    float exponent = (L_vap * M / R) * (1.0f / T_boil - 1.0f / T_capped);
    exponent = fminf(exponent, 20.0f);  // Prevent exp overflow
    float P_sat = P_ref * expf(exponent);
    P_sat = fminf(P_sat, 10.0f * P_ref);  // Cap to 10 atm

    // Mass flux [kg/(m²·s)]
    // Hertz-Knudsen: J_evap = alpha * P_sat * sqrt(M / (2π*R*T))
    // M is in kg/mol, R in J/(mol·K), T in K
    float sqrt_term = sqrtf(2.0f * PI * R * T_capped / M);  // sqrt(2πRT/M) [J/kg]^0.5 = [m/s]
    float J_evap = alpha_evap * P_sat / sqrt_term;   // [Pa] / [m/s] = [kg/(m²·s)]

    // Heat flux [W/m²]
    float q_evap = J_evap * L_vap;  // [kg/(m²·s)] * [J/kg] = [W/m²]

    // Cell surface area [m²]
    float A = dx * dx;

    // Power [W]
    power_out[idx] = q_evap * A;
}

/**
 * @brief Compute radiation power at surface cells
 *
 * BUG FIX (Nov 21, 2025): Removed fill_level filter to match BC kernel behavior
 *
 * Same fix as computeEvaporationPowerKernel - the BC kernel applies radiation
 * to ALL top surface cells regardless of fill_level.
 */
__global__ void computeRadiationPowerKernel(
    const float* temperature,
    const float* fill_level,
    float* power_out,
    float epsilon,
    float T_ambient,
    float dx,
    int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    // Top surface only
    int z = nz - 1;
    int idx = x + y * nx + z * nx * ny;

    float T = temperature[idx];
    // NOTE: fill_level is no longer used for filtering (matches BC kernel)
    // float f = fill_level[idx];

    // Stefan-Boltzmann radiation
    const float sigma = 5.67e-8f;  // W/(m²·K⁴)
    float T4 = T * T * T * T;
    float Tamb4 = T_ambient * T_ambient * T_ambient * T_ambient;
    float q_rad = epsilon * sigma * (T4 - Tamb4);

    // Cell surface area [m²]
    float A = dx * dx;

    // Power [W]
    power_out[idx] = q_rad * A;
}

/**
 * @brief Compute total thermal energy in domain
 *
 * CRITICAL FIX (Nov 19, 2025): Use (T - T_ref) for sensible energy
 *
 * The sensible energy must be computed relative to a reference temperature,
 * not absolute temperature. Using absolute T causes artificial energy
 * creation/destruction when temperature-dependent properties (ρ, cp) change.
 *
 * Reference temperature: T_solidus (natural reference for phase change materials)
 */
__global__ void computeThermalEnergyKernel(
    const float* temperature,
    const float* liquid_fraction,
    float* energy_out,
    MaterialProperties material,
    float dx,
    float T_ref,  // Reference temperature for sensible energy [K]
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    float f_l = liquid_fraction ? liquid_fraction[idx] : 1.0f;  // Assume liquid if not provided

    // Get temperature-dependent properties
    float rho = material.getDensity(T);
    float cp = material.getSpecificHeat(T);

    // Cell volume
    float V = dx * dx * dx;

    // Sensible energy: E = ρ * c_p * (T - T_ref) * V
    // FIXED: Use (T - T_ref) instead of absolute T
    // This prevents artificial energy from constant baseline temperature
    float E_sensible = rho * cp * (T - T_ref) * V;

    // Latent energy: E_latent = f_l * ρ * L_f * V
    // (energy stored in liquid phase above solid reference)
    float E_latent = f_l * rho * material.L_fusion * V;

    // Total energy
    energy_out[idx] = E_sensible + E_latent;
}

/**
 * @brief Compute substrate cooling power across entire substrate volume
 *
 * UPDATED: Matches new BC kernel with exponential decay over substrate_depth layers
 */
__global__ void computeSubstratePowerKernel(
    const float* temperature,
    float* power_out,
    float h_conv,
    float T_substrate,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Loop over substrate layers (matches BC kernel: k=0 to substrate_depth)
    const int substrate_depth = min(60, (nz * 3) / 5);
    float total_power = 0.0f;

    for (int k = 0; k < substrate_depth; ++k) {
        int idx = i + nx * (j + ny * k);
        float T = temperature[idx];

        // Match BC logic - no heat transfer if T <= T_substrate
        if (T <= T_substrate) {
            continue;
        }

        // Decay factor matching the BC kernel
        float decay = expf(-1.0f * k / (float)substrate_depth);
        float h_effective = h_conv * decay;

        // Convective heat flux [W/m²] (positive = heat leaving domain)
        float q_conv = h_effective * (T - T_substrate);

        // Cell surface area [m²]
        float A = dx * dx;

        // Power [W] for this layer (positive = heat loss)
        float layer_power = q_conv * A;
        total_power += layer_power;
    }

    // Store total power for this (i,j) column
    int idx_out = i + nx * j;
    power_out[idx_out] = total_power;
}

// ============================================================================
// Energy Diagnostics Methods (Host)
// ============================================================================

float ThermalLBM::computeEvaporationPower(const float* fill_level, float dx) const {
    if (!has_material_) {
        std::cerr << "WARNING: computeEvaporationPower called without material properties\n";
        return 0.0f;
    }

    int nx = nx_;
    int ny = ny_;
    int nz = nz_;

    // Allocate device memory for per-cell power and initialize to zero
    // CRITICAL: Kernel only writes to top surface cells, but we sum all cells
    float* d_power;
    cudaMalloc(&d_power, num_cells_ * sizeof(float));
    cudaMemset(d_power, 0, num_cells_ * sizeof(float));  // Initialize to 0!

    // Compute power for each surface cell
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    computeEvaporationPowerKernel<<<blocks, threads>>>(
        d_temperature, fill_level, d_power,
        material_, dx, nx, ny, nz
    );
    cudaDeviceSynchronize();

    // Sum all powers
    std::vector<float> h_power(num_cells_);
    cudaMemcpy(h_power.data(), d_power, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float total_power = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        total_power += h_power[i];
    }

    cudaFree(d_power);

    return total_power;
}

float ThermalLBM::computeRadiationPower(const float* fill_level, float dx,
                                         float epsilon, float T_ambient) const {
    int nx = nx_;
    int ny = ny_;
    int nz = nz_;

    // Allocate device memory for per-cell power and initialize to zero
    // CRITICAL: Kernel only writes to top surface cells, but we sum all cells
    float* d_power;
    cudaMalloc(&d_power, num_cells_ * sizeof(float));
    cudaMemset(d_power, 0, num_cells_ * sizeof(float));  // Initialize to 0!

    // Compute power for each surface cell
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    computeRadiationPowerKernel<<<blocks, threads>>>(
        d_temperature, fill_level, d_power,
        epsilon, T_ambient, dx, nx, ny, nz
    );
    cudaDeviceSynchronize();

    // Sum all powers
    std::vector<float> h_power(num_cells_);
    cudaMemcpy(h_power.data(), d_power, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float total_power = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        total_power += h_power[i];
    }

    cudaFree(d_power);

    return total_power;
}

float ThermalLBM::computeTotalThermalEnergy(float dx) const {
    if (!has_material_) {
        std::cerr << "WARNING: computeTotalThermalEnergy called without material properties\n";
        return 0.0f;
    }

    // Get liquid fraction (or nullptr if phase change disabled)
    const float* d_liquid_fraction = phase_solver_ ? phase_solver_->getLiquidFraction() : nullptr;

    // Allocate device memory for per-cell energy
    float* d_energy;
    cudaMalloc(&d_energy, num_cells_ * sizeof(float));

    // Compute energy for each cell
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    // Use T_solidus as reference temperature (natural reference for phase change)
    // This is the temperature at which solid begins to melt
    float T_ref = material_.T_solidus;

    computeThermalEnergyKernel<<<blocks, threads>>>(
        d_temperature, d_liquid_fraction, d_energy,
        material_, dx, T_ref, num_cells_
    );
    cudaDeviceSynchronize();

    // Sum all energies
    std::vector<float> h_energy(num_cells_);
    cudaMemcpy(h_energy.data(), d_energy, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float total_energy = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        total_energy += h_energy[i];
    }

    cudaFree(d_energy);

    return total_energy;
}

float ThermalLBM::computeSubstratePower(float dx, float h_conv, float T_substrate) const {
    if (!has_material_) {
        std::cerr << "WARNING: computeSubstratePower called without material properties\n";
        return 0.0f;
    }

    int nx = nx_;
    int ny = ny_;
    int nz = nz_;

    // Allocate device memory for per-column power (nx * ny entries)
    int num_columns = nx * ny;
    float* d_power;
    cudaMalloc(&d_power, num_columns * sizeof(float));
    cudaMemset(d_power, 0, num_columns * sizeof(float));

    // Compute power for each (i,j) column
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    computeSubstratePowerKernel<<<blocks, threads>>>(
        d_temperature, d_power,
        h_conv, T_substrate, dx,
        nx, ny, nz
    );
    cudaDeviceSynchronize();

    // Sum all column powers
    std::vector<float> h_power(num_columns);
    cudaMemcpy(h_power.data(), d_power, num_columns * sizeof(float), cudaMemcpyDeviceToHost);

    float total_power = 0.0f;
    for (int i = 0; i < num_columns; ++i) {
        total_power += h_power[i];
    }

    cudaFree(d_power);

    return total_power;
}

// ============================================================================
// Evaporation Mass Flux Computation (for VOF mass coupling)
// ============================================================================

/**
 * @brief Compute evaporation mass flux at surface cells
 *
 * Uses Hertz-Knudsen-Langmuir model (same as in applyRadiationBoundaryCondition):
 *   J_evap = alpha_evap * P_sat(T) / sqrt(2*pi*R*T/M)  [kg/(m^2*s)]
 *
 * Where:
 *   P_sat(T) = P_ref * exp[(L_vap*M/R) * (1/T_boil - 1/T)]  [Pa]
 *
 * This flux is used by VOF solver to remove material:
 *   df/dt = -J_evap / (rho * dx)
 */
__global__ void computeEvaporationMassFluxKernel(
    const float* temperature,
    const float* fill_level,  // NEW: VOF fill level for interface detection
    float* J_evap,
    MaterialProperties material,
    int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // NEW: iterate all z

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;

    // FIX: Only evaporate at VOF interface cells (0.01 < f < 0.99)
    // This ensures evaporation only happens at the actual free surface,
    // not at a fixed z=nz-1 position
    float f = fill_level[idx];
    if (f <= 0.01f || f >= 0.99f) {
        J_evap[idx] = 0.0f;
        return;
    }

    float T = temperature[idx];

    // Physical constants for Ti6Al4V evaporation
    // CRITICAL FIX (2025-11-27): Reduced from 0.82 to 0.18 to prevent excessive evaporation
    const float alpha_evap = 0.18f;           // Evaporation coefficient (dimensionless)
    const float M_molar = 0.0479f;            // Molar mass [kg/mol]
    const float R_gas = 8.314f;               // Universal gas constant [J/(mol·K)]
    const float P_ref = 101325.0f;            // Reference pressure at boiling point [Pa]
    const float PI = 3.14159265359f;

    // Get material-specific properties
    float T_boil = material.T_vaporization;   // Boiling temperature [K]
    float L_vap = material.L_vaporization;    // Latent heat of vaporization [J/kg]

    // Only compute mass flux when T > T_boil
    if (T <= T_boil) {
        J_evap[idx] = 0.0f;
        return;
    }

    // Clausius-Clapeyron equation for vapor pressure
    // OVERFLOW PROTECTION: Cap temperature and exponent to prevent inf
    float T_capped = fminf(T, 2.0f * T_boil);  // Cap at 2x boiling point
    float exponent = (L_vap * M_molar / R_gas) * (1.0f / T_boil - 1.0f / T_capped);
    exponent = fminf(exponent, 20.0f);  // Cap exponent to prevent exp overflow (exp(20) ~ 5e8)
    float P_sat = P_ref * expf(exponent);

    // Sanity check: cap P_sat to reasonable physical limit (10x atmospheric)
    P_sat = fminf(P_sat, 10.0f * P_ref);

    // Hertz-Knudsen-Langmuir evaporation mass flux [kg/(m^2·s)]
    float denominator = sqrtf(2.0f * PI * R_gas * T_capped / M_molar);
    J_evap[idx] = alpha_evap * P_sat / denominator;
}

void ThermalLBM::computeEvaporationMassFlux(float* d_J_evap, const float* fill_level) const {
    if (!has_material_) {
        std::cerr << "WARNING: computeEvaporationMassFlux called without material properties\n";
        cudaMemset(d_J_evap, 0, num_cells_ * sizeof(float));
        return;
    }

    // Initialize to zero
    cudaMemset(d_J_evap, 0, num_cells_ * sizeof(float));

    // Launch kernel for ALL cells (not just top surface)
    // FIX: Use 3D grid to cover entire domain, kernel will filter to interface cells
    dim3 threads(8, 8, 4);
    dim3 blocks((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 3) / 4);

    computeEvaporationMassFluxKernel<<<blocks, threads>>>(
        d_temperature, fill_level, d_J_evap, material_, nx_, ny_, nz_
    );
    cudaDeviceSynchronize();
}

} // namespace physics
} // namespace lbm