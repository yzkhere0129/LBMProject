/**
 * @file thermal_lbm.cu
 * @brief Implementation of Thermal Lattice Boltzmann Method solver
 */

#include "physics/thermal_lbm.h"
#include "utils/cuda_check.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>

namespace lbm {
namespace physics {

// External reference to D3Q7 lattice constants (defined in lattice_d3q7.cu)
extern __device__ int tex[7];
extern __device__ int tey[7];
extern __device__ int tez[7];

// Forward declaration of kernels
__global__ void initializeEquilibriumKernel(float* g_src, const float* temperature, int num_cells);
__global__ void thermalStreamingKernel(const float* g_src, float* g_dst,
                                       int nx, int ny, int nz,
                                       float emissivity, bool z_periodic);
__global__ void applyPhaseChangeCorrectionKernel(float* g, float* temperature,
                                                   const float* fl_curr, const float* fl_prev,
                                                   MaterialProperties material, int num_cells,
                                                   int nx, int ny, int nz);
__global__ void enthalpySourceTermKernel(float* g, float* temperature,
                                          float* liquid_fraction,
                                          const float* liquid_fraction_prev,
                                          const float* fill_level,
                                          MaterialProperties material, int num_cells);
__global__ void computeTemperatureKernel(const float* g, float* temperature,
                                          int num_cells, const float* fill_level,
                                          float T_boil_clamp);
__global__ void thermalBGKCollisionKernel(float* g_src, const float* temperature,
                                           const float* ux, const float* uy, const float* uz,
                                           float omega_T, int nx, int ny, int nz,
                                           MaterialProperties material, float dt, float dx,
                                           bool use_apparent_cp, const float* fill_level);

// Constructor (deprecated - for backward compatibility)
ThermalLBM::ThermalLBM(int nx, int ny, int nz, float thermal_diffusivity,
                       float density, float specific_heat,
                       float dt, float dx)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      dt_(dt), dx_(dx),
      thermal_diff_physical_(thermal_diffusivity), rho_(density), cp_(specific_heat),
      emissivity_(0.35f),  // Default emissivity
      T_initial_(300.0f),  // Default ambient temperature
      z_periodic_(false),
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
      T_initial_(300.0f),  // Default ambient temperature (will be set by initialize())
      z_periodic_(false),
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
    CUDA_CHECK(cudaMemset(d_g_src, 0, size_dist));
    CUDA_CHECK(cudaMemset(d_g_dst, 0, size_dist));
    CUDA_CHECK(cudaMemset(d_temperature, 0, size_scalar));
}

// Free device memory
void ThermalLBM::freeMemory() {
    if (d_g_src) cudaFree(d_g_src);
    if (d_g_dst) cudaFree(d_g_dst);
    if (d_temperature) cudaFree(d_temperature);
    if (d_cap_energy_removed_) cudaFree(d_cap_energy_removed_);
}

// Swap source and destination distributions
void ThermalLBM::swapDistributions() {
    float* temp = d_g_src;
    d_g_src = d_g_dst;
    d_g_dst = temp;
}

// Initialize with uniform temperature
void ThermalLBM::initialize(float initial_temp) {
    // Store initial temperature for energy reference
    T_initial_ = initial_temp;

    // Set uniform temperature
    CUDA_CHECK(cudaMemset(d_temperature, 0, num_cells_ * sizeof(float)));

    // Create uniform temperature array
    float* h_temp = new float[num_cells_];
    for (int i = 0; i < num_cells_; ++i) {
        h_temp[i] = initial_temp;
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_temperature, h_temp, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_temp;

    // Initialize distribution functions to equilibrium
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Simple kernel launch without lambda
    initializeEquilibriumKernel<<<gridSize, blockSize>>>(d_g_src, d_temperature, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize phase change solver if enabled
    if (phase_solver_) {
        phase_solver_->initializeFromTemperature(d_temperature);
    }
}

// Initialize with custom temperature field
void ThermalLBM::initialize(const float* temp_field) {
    // Compute average temperature for energy reference
    float avg_temp = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        avg_temp += temp_field[i];
    }
    T_initial_ = avg_temp / num_cells_;

    // Copy temperature field to device
    CUDA_CHECK(cudaMemcpy(d_temperature, temp_field, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize distribution functions to equilibrium
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Simple kernel launch without lambda
    initializeEquilibriumKernel<<<gridSize, blockSize>>>(d_g_src, d_temperature, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

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

    // Disable apparent Cp when enthalpy source term (ESM) is active.
    // ESM handles latent heat via post-collision correction; using apparent Cp
    // on top would double-count the latent heat effect.
    bool use_apparent_cp = false;

    thermalBGKCollisionKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature, ux, uy, uz, omega_T_, nx_, ny_, nz_,
        material_, dt_, dx_, use_apparent_cp, d_vof_fill_level_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Perform streaming
void ThermalLBM::streaming() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    // ADIABATIC BOUNDARIES: Initialize g_dst to zero
    // Bounce-back will write reflected distributions
    CUDA_CHECK(cudaMemset(d_g_dst, 0, num_cells_ * D3Q7::Q * sizeof(float)));

    // Stream all distributions with boundary treatment
    thermalStreamingKernel<<<gridSize, blockSize>>>(
        d_g_src, d_g_dst, nx_, ny_, nz_, emissivity_, z_periodic_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
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
            CUDA_CHECK_KERNEL();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

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
            CUDA_CHECK_KERNEL();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
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
        CUDA_CHECK_KERNEL();
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
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

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
        CUDA_CHECK_KERNEL();
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
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

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
    CUDA_CHECK_KERNEL();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Substrate BC kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update temperature field after substrate cooling
    computeTemperature();
}

// ============================================================================
// Per-face thermal boundary condition dispatch
// ============================================================================

void ThermalLBM::applyFaceThermalBC(int face, int bc_type,
                                     float dt, float dx,
                                     float dirichlet_T,
                                     float h_conv,
                                     float T_inf,
                                     float emissivity_val,
                                     float T_ambient) {
    // bc_type mapping from ThermalBCType enum:
    //   0 = PERIODIC   -> no-op (handled in streaming)
    //   1 = ADIABATIC  -> zero-flux
    //   2 = DIRICHLET  -> fixed temperature
    //   3 = CONVECTIVE -> Newton cooling
    //   4 = RADIATION  -> Stefan-Boltzmann

    if (bc_type == 0) {
        // PERIODIC: nothing to do
        return;
    }

    // Compute face size for kernel launch
    int face_size;
    if (face < 2) face_size = ny_ * nz_;       // x-faces
    else if (face < 4) face_size = nx_ * nz_;   // y-faces
    else face_size = nx_ * ny_;                  // z-faces

    dim3 blockSize(256);
    dim3 gridSize((face_size + blockSize.x - 1) / blockSize.x);

    if (bc_type == 1) {
        // ADIABATIC: zero-flux boundary
        applyAdiabaticBoundary<<<gridSize, blockSize>>>(
            d_g_src, nx_, ny_, nz_, face);
        CUDA_CHECK_KERNEL();

    } else if (bc_type == 2) {
        // DIRICHLET: fixed temperature
        applyConstantTemperatureBoundary<<<gridSize, blockSize>>>(
            d_g_src, d_temperature, dirichlet_T, nx_, ny_, nz_, face);
        CUDA_CHECK_KERNEL();

    } else if (bc_type == 3) {
        // CONVECTIVE: Newton's law of cooling
        float rho = has_material_ ? material_.rho_solid : rho_;
        float cp = has_material_ ? material_.cp_solid : cp_;

        applyConvectiveBCKernel<<<gridSize, blockSize>>>(
            d_g_src, d_temperature,
            nx_, ny_, nz_,
            dx, dt, h_conv, T_inf,
            rho, cp, face);
        CUDA_CHECK_KERNEL();

    } else if (bc_type == 4) {
        // RADIATION: Stefan-Boltzmann
        if (has_material_) {
            applyRadiationBCFaceKernel<<<gridSize, blockSize>>>(
                d_g_src, d_temperature,
                nx_, ny_, nz_,
                dx, dt, emissivity_val, material_, T_ambient, face);
        } else {
            MaterialProperties temp_mat;
            temp_mat.rho_solid = rho_;
            temp_mat.rho_liquid = rho_;
            temp_mat.cp_solid = cp_;
            temp_mat.cp_liquid = cp_;
            temp_mat.T_solidus = 0.0f;
            temp_mat.T_liquidus = 0.0f;
            applyRadiationBCFaceKernel<<<gridSize, blockSize>>>(
                d_g_src, d_temperature,
                nx_, ny_, nz_,
                dx, dt, emissivity_val, temp_mat, T_ambient, face);
        }
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// EVAPORATION COOLING KERNEL - Apply latent heat removal at VOF interface
// ============================================================================
// This kernel applies evaporative cooling at interface cells where evaporation
// mass flux J_evap > 0. The cooling removes latent heat Q = J_evap * L_vap.
//
// Physics: When liquid evaporates, it absorbs latent heat from the surface,
// providing natural temperature capping near the boiling point.
//
// This is CRITICAL for laser melting simulations where temperatures can
// exceed the boiling point without proper evaporation cooling.
// ============================================================================
__global__ void applyEvaporationCoolingKernel(
    float* g,
    const float* temperature,
    const float* J_evap,
    const float* fill_level,
    int nx, int ny, int nz,
    float dx, float dt,
    MaterialProperties material)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;

    // CRITICAL FIX (2026-01-25): Apply cooling to ALL liquid cells above T_boil
    // Previously only cooled interface cells, but hottest cells are often
    // fully liquid (f=1) below the interface due to Beer-Lambert absorption
    //
    // Physics: While evaporation mass loss only occurs at interface,
    // heat conduction from the hot liquid to the interface causes
    // effective cooling of the entire high-temperature region.
    // We model this as "effective evaporative cooling" for all cells > T_boil
    float f = fill_level[idx];
    if (f <= 0.01f) return;  // Skip gas cells (f~0)

    float T = temperature[idx];
    float T_boil = material.T_vaporization;
    float L_vap = material.L_vaporization;

    // Skip if no evaporation flux at this cell OR if temperature below boiling
    float J = J_evap[idx];

    // For non-interface cells (f >= 0.99), compute effective evaporation
    // based on temperature excess above boiling point
    if (f >= 0.99f) {
        if (T <= T_boil) return;

        // Clausius-Clapeyron evaporation rate (Anisimov 1995)
        // NO artificial caps on T, P_sat, or exponent — let physics self-regulate
        const float alpha_evap = 0.18f;
        const float M_molar = material.molar_mass;
        const float R_gas = 8.314f;
        const float P_ref = 101325.0f;
        const float PI = 3.14159265359f;

        float exponent = (L_vap * M_molar / R_gas) * (1.0f / T_boil - 1.0f / T);
        exponent = fminf(exponent, 50.0f);  // Only prevent exp overflow (float32 limit)
        float P_sat = P_ref * expf(exponent);

        float denominator = sqrtf(2.0f * PI * R_gas * T / M_molar);
        if (denominator > 1e-10f) {
            J = alpha_evap * P_sat / denominator;
        }
    } else {
        if (J <= 0.0f) return;
    }

    float rho = material.getDensity(T);
    float cp = material.getSpecificHeat(T);
    float q_evap = J * L_vap;  // [W/m²]

    float rho_cp = rho * cp;
    if (rho_cp < 1e-6f) return;

    float dT = -(q_evap / dx) * dt / rho_cp;

    // ONLY physical limiter: don't cool below T_boil (can't evaporate below boiling)
    float T_after = T + dT;
    if (T_after < T_boil) {
        dT = T_boil - T;  // Clamp to exactly T_boil
    }

    // Apply temperature change to all distributions (maintains isotropy)
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    int num_cells = nx * ny * nz;
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
    }
}

// ============================================================================
// TEMPERATURE CAP KERNEL - Hard cap for numerical stability
// ============================================================================
// This kernel applies a hard temperature cap to ALL liquid cells above T_boil.
// This is a safety measure to prevent numerical runaway when evaporative cooling
// is insufficient to limit temperature.
//
// Physical justification: In reality, strong surface evaporation self-limits
// temperature to near T_boil. Numerical heat input faster than evaporative
// cooling can remove leads to unrealistic temperatures.
// ============================================================================
__global__ void applyTemperatureCapKernel(
    float* g,
    const float* temperature,
    const float* fill_level,
    float* energy_removed,   // [out] per-cell energy removed [K] (will be summed on host)
    int nx, int ny, int nz,
    MaterialProperties material)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;

    // Only apply to liquid/interface cells (f > 0.01)
    float f = fill_level[idx];
    if (f <= 0.01f) {
        energy_removed[idx] = 0.0f;
        return;
    }

    float T = temperature[idx];
    float T_boil = material.T_vaporization;

    // Hard temperature cap: T_max = T_boil - 100K (e.g., 2990K for steel)
    // Physically: evaporation self-limits temperature near T_boil.
    // Energy removed by this cap represents effective evaporative cooling.
    const float T_max_allowed = T_boil - 100.0f;

    if (T > T_max_allowed) {
        float dT = T_max_allowed - T;  // negative

        // Track energy removed: dE = rho * cp * |dT| * dx^3 per cell
        // Store |dT| here; host multiplies by rho*cp*dx^3 to get Watts
        energy_removed[idx] = -dT;  // positive value = energy removed [K]

        // Apply temperature correction to all distributions
        const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
        int num_cells = nx * ny * nz;
        for (int q = 0; q < D3Q7::Q; ++q) {
            g[q * num_cells + idx] += weights[q] * dT;
        }
    } else {
        energy_removed[idx] = 0.0f;
    }
}

void ThermalLBM::applyEvaporationCooling(const float* J_evap, const float* fill_level, float dt, float dx) {
    if (!has_material_) {
        std::cerr << "WARNING: applyEvaporationCooling called without material properties\n";
        return;
    }

    // Launch 3D kernel for entire domain
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    applyEvaporationCoolingKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature, J_evap, fill_level,
        nx_, ny_, nz_, dx, dt, material_
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate energy tracking buffer (lazy init, persistent)
    if (!d_cap_energy_removed_) {
        CUDA_CHECK(cudaMalloc(&d_cap_energy_removed_, num_cells_ * sizeof(float)));
    }

    // Temperature cap REMOVED — evaporation cooling via Clausius-Clapeyron
    // is the sole temperature regulator. The exponential growth of P_sat
    // with T creates a natural thermostat: at T >> T_boil, evaporation
    // removes energy far faster than the laser can add it, driving T
    // back toward T_boil. No artificial intervention needed.
    if (d_cap_energy_removed_) {
        CUDA_CHECK(cudaMemset(d_cap_energy_removed_, 0, num_cells_ * sizeof(float)));
    }

    // Update temperature field
    computeTemperature();
}

// ============================================================================
// STANDALONE TEMPERATURE SAFETY CAP
// ============================================================================
// Clamps temperature at T_vaporization for ALL cells, without requiring
// VOF fill_level. This is the physics-based temperature limit: in reality,
// strong surface evaporation prevents temperatures significantly exceeding
// the boiling point. When VOF/evaporation coupling is disabled, this cap
// provides the equivalent constraint.
// ============================================================================
__global__ void applyTemperatureSafetyCapKernel(
    float* g,
    const float* temperature,
    int num_cells,
    float T_max_allowed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    if (T <= T_max_allowed) return;

    float dT = T_max_allowed - T;

    // Apply temperature correction to all distributions (maintains isotropy)
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
    }
}

void ThermalLBM::applyTemperatureSafetyCap() {
    if (!has_material_) return;  // No material = no boiling point info

    float T_boil = material_.T_vaporization;
    if (T_boil <= 0.0f) return;  // No boiling point defined

    // Safety margin above T_boil. In keyhole mode (skip_temperature_cap_),
    // allow much higher temperatures for deep keyhole dynamics.
    float T_max_allowed = skip_temperature_cap_ ?
        T_boil + 2000.0f :  // Keyhole: allow up to ~5000K
        T_boil + 100.0f;    // Conduction: tight cap

    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    applyTemperatureSafetyCapKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature, num_cells_, T_max_allowed
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update temperature field after capping
    computeTemperature();
}

// Compute temperature from distribution
void ThermalLBM::computeTemperature() {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    computeTemperatureKernel<<<gridSize, blockSize>>>(
        d_g_src, d_temperature, num_cells_,
        d_vof_fill_level_, material_.T_vaporization);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Enthalpy Source Term Method (Jiaung 2001)
    // ========================================================================
    // After collision+streaming gives T* = Σg_q, enforce enthalpy conservation:
    //   1. Compute total enthalpy: H = cp·T* + fl_old·L
    //   2. Decode (T_new, fl_new) from H via direct mushy-zone formula
    //   3. Correct distributions: g_q += w_q · (T_new - T*)
    //
    // This ensures latent heat acts as a "thermal brake": in the mushy zone,
    // heat goes into increasing fl rather than raising T.
    // ========================================================================
    if (phase_solver_) {
        // Save fl from previous step (used as fl_old in ESM)
        phase_solver_->storePreviousLiquidFraction();

        // Apply enthalpy source term: corrects T, fl, and g simultaneously
        // VOF fill_level masks gas cells (no phase change in inert atmosphere)
        enthalpySourceTermKernel<<<gridSize, blockSize>>>(
            d_g_src,
            d_temperature,
            phase_solver_->getLiquidFraction(),
            phase_solver_->getPreviousLiquidFraction(),
            d_vof_fill_level_,   // nullptr if no VOF → no masking (backward compat)
            material_,
            num_cells_
        );
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Copy temperature to host
void ThermalLBM::copyTemperatureToHost(float* host_temp) const {
    CUDA_CHECK(cudaMemcpy(host_temp, d_temperature, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
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
// Layout: SoA — g_src[q * num_cells + idx]
__global__ void initializeEquilibriumKernel(float* g_src, const float* temperature, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    // At rest: g_eq = w_i * T
    for (int q = 0; q < D3Q7::Q; ++q) {
        g_src[q * num_cells + idx] = D3Q7::computeThermalEquilibrium(q, T, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void thermalBGKCollisionKernel(
    float* g_src,
    const float* temperature,
    const float* ux,
    const float* uy,
    const float* uz,
    float omega_T,
    int nx, int ny, int nz,
    MaterialProperties material,
    float dt,
    float dx,
    bool use_apparent_cp,
    const float* fill_level) {  // nullable: nullptr → no gas isolation

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    float T = temperature[idx];

    // ========================================================================
    // GAS-PHASE THERMAL ISOLATION (Fix 1)
    // Pure gas cells (f < 0.05) get omega → 0 (no thermal relaxation).
    // This blocks heat conduction into gas while preserving streaming
    // (which is needed for bounce-back BCs at domain walls).
    // ========================================================================
    bool is_gas = false;
    if (fill_level != nullptr) {
        float f = fill_level[idx];
        if (f < 0.05f) {
            is_gas = true;
        }
    }

    // Get velocity (use zero if not provided)
    float vel_x = ux ? ux[idx] : 0.0f;
    float vel_y = uy ? uy[idx] : 0.0f;
    float vel_z = uz ? uz[idx] : 0.0f;

    // ============================================================================
    // LATENT HEAT CORRECTION: Apparent Heat Capacity Method
    // ============================================================================
    // When phase change is enabled, the thermal diffusivity must account for
    // latent heat absorption/release in the mushy zone:
    //
    //   alpha_eff = k / (rho * c_eff)
    //
    // where c_eff = c_p + L_f * (df_l/dT) is the apparent heat capacity.
    //
    // In the mushy zone (T_solidus < T < T_liquidus):
    //   df_l/dT = 1 / (T_liquidus - T_solidus)
    //   c_eff = c_p + L_f / (T_liquidus - T_solidus)
    //
    // This naturally captures latent heat without post-step corrections.
    //
    // Reference: Voller & Prakash (1987), "A fixed grid numerical modelling
    //            methodology for convection-diffusion mushy region phase-change
    //            problems", Int. J. Heat Mass Transfer, 30(8), 1709-1719.
    // ============================================================================

    float omega_T_local = omega_T;  // Default: use global omega_T

    if (use_apparent_cp) {
        // Compute effective thermal diffusivity using apparent heat capacity
        float rho = material.getDensity(T);
        float k = material.getThermalConductivity(T);
        float c_eff = material.getApparentHeatCapacity(T);

        // Thermal diffusivity: alpha_eff = k / (rho * c_eff)
        float alpha_eff = k / (rho * c_eff);

        // Convert to lattice units
        float alpha_lattice = alpha_eff * dt / (dx * dx);

        // Compute tau from lattice diffusivity: tau = alpha / cs^2 + 0.5
        float tau_T_local = alpha_lattice / D3Q7::CS2 + 0.5f;

        // Clamp for stability (same as global initialization)
        if (tau_T_local < 0.51f) {
            tau_T_local = 0.51f;  // Minimum stable tau
        } else if (tau_T_local > 1.0f / 0.1f) {
            tau_T_local = 1.0f / 0.1f;  // Maximum tau (omega_min = 0.1)
        }

        omega_T_local = 1.0f / tau_T_local;
    }

    // Gas cells: kill thermal relaxation → zero diffusivity
    if (is_gas) {
        omega_T_local = 0.0f;
    }

    // BGK collision for each direction
    int num_cells = nx * ny * nz;
    for (int q = 0; q < D3Q7::Q; ++q) {
        int dist_idx = q * num_cells + idx;
        float g_eq = D3Q7::computeThermalEquilibrium(q, T, vel_x, vel_y, vel_z);

        // BGK collision: g_new = g - omega * (g - g_eq)
        g_src[dist_idx] = g_src[dist_idx] - omega_T_local * (g_src[dist_idx] - g_eq);
    }
}

__global__ void thermalStreamingKernel(
    const float* g_src,
    float* g_dst,
    int nx, int ny, int nz,
    float emissivity,
    bool z_periodic) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = x + y * nx + z * nx * ny;
    int num_cells = nx * ny * nz;

    for (int q = 0; q < D3Q7::Q; ++q) {
        int cx, cy, cz;
#ifdef __CUDA_ARCH__
        cx = tex[q];
        cy = tey[q];
        cz = tez[q];
#else
        cx = 0; cy = 0; cz = 0;
#endif

        int nx_target = x + cx;
        int ny_target = y + cy;
        int nz_target = z + cz;

        // Z-periodic wrapping (for quasi-2D simulations)
        if (z_periodic) {
            nz_target = (nz_target + nz) % nz;
        }

        // Check if target is inside domain
        if (nx_target >= 0 && nx_target < nx &&
            ny_target >= 0 && ny_target < ny &&
            nz_target >= 0 && nz_target < nz) {

            int target_idx = nx_target + ny_target * nx + nz_target * nx * ny;
            g_dst[q * num_cells + target_idx] = g_src[q * num_cells + idx];
        } else {
            // Bounce-back for x and y boundaries (adiabatic)
            int q_opposite = 0;
            if (q == 1) q_opposite = 2;      // +x -> -x
            else if (q == 2) q_opposite = 1; // -x -> +x
            else if (q == 3) q_opposite = 4; // +y -> -y
            else if (q == 4) q_opposite = 3; // -y -> +y
            else if (q == 5) q_opposite = 6; // +z -> -z
            else if (q == 6) q_opposite = 5; // -z -> +z

            g_dst[q_opposite * num_cells + idx] = g_src[q * num_cells + idx];
        }
    }
}

__global__ void computeTemperatureKernel(
    const float* g,
    float* temperature,
    int num_cells,
    const float* fill_level,  // nullable: nullptr → no gas clamp
    float T_boil_clamp) {     // Gas-phase temperature ceiling (e.g. T_boil)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Sum all distribution functions
    float T = 0.0f;
    for (int q = 0; q < D3Q7::Q; ++q) {
        T += g[q * num_cells + idx];
    }

    constexpr float T_MIN = 0.0f;
    constexpr float T_MAX = 50000.0f;

    T = fmaxf(T, T_MIN);
    T = fminf(T, T_MAX);

    // ========================================================================
    // GAS-PHASE TEMPERATURE CLAMP (Fix 2)
    // Pure gas cells (f < 0.05) are clamped to T_boil. Any residual heat
    // from numerical diffusion or streaming is removed. The distributions
    // are also reset to equilibrium at the clamped temperature to prevent
    // spurious ∇T at the metal/gas boundary.
    // ========================================================================
    if (fill_level != nullptr) {
        float f = fill_level[idx];
        if (f < 0.05f && T > T_boil_clamp) {
            T = T_boil_clamp;
            // Reset distributions to equilibrium at clamped T
            const float w[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
            for (int q = 0; q < 7; ++q) {
                // g_eq(q, T, u=0) = w_q * T for stationary gas
                ((float*)g)[q * num_cells + idx] = w[q] * T;
            }
        }
    }

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
    int num_cells = nx * ny * nz;

    // Set temperature
    temperature[idx] = T_boundary;

    // Set equilibrium distribution for boundary temperature
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] = D3Q7::computeThermalEquilibrium(q, T_boundary, 0.0f, 0.0f, 0.0f);
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
    int num_cells = nx * ny * nz;

    // Copy distribution from interior
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] = g[q * num_cells + interior_idx];
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

    // ============================================================================
    // CRITICAL: Zero-division protection (BUG FIX 2026-01-04)
    // ============================================================================
    // Protect against division by zero when rho*cp = 0 (corrupted properties)
    // ============================================================================
    float rho_cp_product = rho * cp;
    float dT;
    if (rho_cp_product < 1e-6f) {
        // Invalid material properties - skip heat source to prevent NaN/Inf
        dT = 0.0f;
    } else {
        dT = (Q * dt) / rho_cp_product;
    }

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
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};

    // Add temperature increase equally to all distributions
    // This preserves the temperature increase while maintaining isotropy
    // Apply correction factor to ensure correct energy deposition
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT * source_correction;
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
    // EVAPORATION COOLING: Hertz-Knudsen-Langmuir model
    // ============================================================================
    // PHYSICS NOTE (2025-12-02): Evaporation activation threshold
    // - Pre-boiling evaporation occurs due to vapor pressure buildup
    // - Aligned with recoil pressure activation at T > T_boil - 500K
    // - This ensures physical consistency between recoil force and cooling
    // ============================================================================
    float q_evap = 0.0f;  // Evaporative heat flux [W/m²]

    // Physical constants for Ti6Al4V evaporation
    // CRITICAL FIX (2025-11-27): Reduced from 0.82 to 0.18 to prevent excessive evaporation
    // Calibrated alpha_evap = 0.18 (Anisimov 1995)
    const float alpha_evap = 0.18f;
    const float M_molar = material.molar_mass; // Molar mass [kg/mol]
    const float R_gas = 8.314f;               // Universal gas constant [J/(mol·K)]
    const float P_ref = 101325.0f;            // Reference pressure at boiling point [Pa]
    const float PI = 3.14159265359f;

    // Get material-specific properties
    float T_boil = material.T_vaporization;   // Boiling temperature [K]
    float L_vap = material.L_vaporization;    // Latent heat of vaporization [J/kg]

    // Evaporation activation threshold (aligned with recoil pressure)
    // Pre-boiling evaporation occurs in low-pressure environments and near surfaces
    const float T_evap_threshold = T_boil - 500.0f;  // K

    // Apply evaporation cooling when surface temperature exceeds threshold
    if (T_surf > T_evap_threshold) {
        // Clausius-Clapeyron equation for vapor pressure
        // P_sat(T) = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]
        // OVERFLOW PROTECTION: Cap temperature and ensure minimum value
        // BUG FIX 2026-01-04: Ensure T_capped >= 1K to prevent division by zero
        float T_capped = fminf(T_surf, 2.0f * T_boil);
        T_capped = fmaxf(T_capped, 1.0f);  // Minimum 1K to prevent division by zero
        float exponent = (L_vap * M_molar / R_gas) * (1.0f / T_boil - 1.0f / T_capped);
        exponent = fminf(exponent, 20.0f);  // Prevent exp overflow
        float P_sat = P_ref * expf(exponent);
        P_sat = fminf(P_sat, 10.0f * P_ref);  // Cap to 10 atm

        // Hertz-Knudsen-Langmuir evaporation mass flux
        // J_evap = α_evap * P_sat / sqrt(2π * R * T / M)  [kg/(m²·s)]
        float denominator = sqrtf(2.0f * PI * R_gas * T_capped / M_molar);  // [m/s]

        // ============================================================================
        // CRITICAL: Zero-division protection (BUG FIX 2026-01-04)
        // ============================================================================
        // If T_capped = 0 (corrupted temperature), denominator = 0 → J_evap = Inf
        // This prevents NaN/Inf propagation through the simulation
        // Minimum threshold: 1e-10 m/s (physically impossible, indicates bad data)
        // ============================================================================
        if (denominator < 1e-10f) {
            // No evaporation at zero temperature (physically correct)
            q_evap = 0.0f;
        } else {
            float J_evap = alpha_evap * P_sat / denominator;  // [kg/(m²·s)]

            // Evaporative cooling heat flux
            // q_evap = J_evap * L_vap  [W/m²]
            q_evap = J_evap * L_vap;
        }
    }

    // ============================================================================
    // COMBINED COOLING: Total heat flux and temperature change
    // ============================================================================
    float q_total = q_rad + q_evap;

    // Convert to temperature decrease
    // q_total [W/m²] → energy loss per unit volume [W/m³] = q_total / dx
    // Temperature change: ΔT = -(q_total/dx) · dt / (ρ·c_p)
    // Using temperature-dependent ρ and cp for accurate energy conservation

    // ============================================================================
    // CRITICAL: Zero-division protection (BUG FIX 2026-01-04)
    // ============================================================================
    // Protect against division by zero when rho*cp = 0 (corrupted properties)
    // This can occur if material properties are uninitialized or if temperature
    // is invalid, leading to getDensity()/getSpecificHeat() returning zero
    // ============================================================================
    float rho_cp_product = rho * cp;
    float dT;
    if (rho_cp_product < 1e-6f) {
        // Invalid material properties - skip cooling to prevent NaN/Inf
        dT = 0.0f;
    } else {
        dT = -(q_total / dx) * dt / rho_cp_product;
    }

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
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    int num_cells = nx * ny * nz;
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
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

    // Apply convective BC only at bottom surface (k=0)
    // This is the correct physics: heat flux is a surface phenomenon
    // Heat from the interior reaches the boundary through conduction (LBM handles this)
    const int k = 0;
    int idx = i + nx * (j + ny * k);

    float T_cell = temperature[idx];

    // No cooling if cell is already at or below substrate temperature
    if (T_cell <= T_substrate) {
        return;
    }

    // Convective heat flux at surface [W/m²]
    float q_conv = h_conv * (T_cell - T_substrate);

    // Convert surface flux to volumetric heat rate [W/m³]
    // For a surface BC, the flux enters through one face (area = dx²)
    // and affects the cell volume (V = dx³)
    // Therefore: heat_rate = q_conv * A / V = q_conv * dx² / dx³ = q_conv / dx
    float heat_rate = q_conv / dx;

    // Temperature change from substrate cooling
    // ============================================================================
    // CRITICAL: Zero-division protection (BUG FIX 2026-01-04)
    // ============================================================================
    // Protect against division by zero when rho*cp = 0 (corrupted properties)
    // ============================================================================
    float rho_cp_product = rho * cp;
    float dT;
    if (rho_cp_product < 1e-6f) {
        // Invalid material properties - skip cooling to prevent NaN/Inf
        dT = 0.0f;
    } else {
        dT = -heat_rate * dt / rho_cp_product;
    }

    // CFL-type stability limiter (prevent unphysical cooling)
    // Never cool more than 10% of temperature difference per timestep
    float max_cooling = -0.10f * (T_cell - T_substrate);
    if (dT < max_cooling) {
        dT = max_cooling;
    }

    // Apply to all distribution functions (isotropic cooling)
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    int num_cells = nx * ny * nz;
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
    }
}

// ============================================================================
// Generalized per-face convective BC kernel
// ============================================================================

/**
 * @brief Helper: compute linear index and check bounds for a given face
 * @return -1 if out of bounds, otherwise the linear cell index
 */
__device__ static int faceIndexHelper(int tid, int face,
                                       int nx, int ny, int nz,
                                       int& x, int& y, int& z) {
    switch (face) {
        case 0: // x_min
            if (tid >= ny * nz) return -1;
            x = 0; y = tid % ny; z = tid / ny;
            break;
        case 1: // x_max
            if (tid >= ny * nz) return -1;
            x = nx - 1; y = tid % ny; z = tid / ny;
            break;
        case 2: // y_min
            if (tid >= nx * nz) return -1;
            y = 0; x = tid % nx; z = tid / nx;
            break;
        case 3: // y_max
            if (tid >= nx * nz) return -1;
            y = ny - 1; x = tid % nx; z = tid / nx;
            break;
        case 4: // z_min
            if (tid >= nx * ny) return -1;
            z = 0; x = tid % nx; y = tid / nx;
            break;
        case 5: // z_max
            if (tid >= nx * ny) return -1;
            z = nz - 1; x = tid % nx; y = tid / nx;
            break;
        default:
            return -1;
    }
    return x + nx * (y + ny * z);
}

__global__ void applyConvectiveBCKernel(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float h_conv,
    float T_inf,
    float rho,
    float cp,
    int boundary_face)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x, y, z;
    int idx = faceIndexHelper(tid, boundary_face, nx, ny, nz, x, y, z);
    if (idx < 0) return;

    float T_cell = temperature[idx];

    // No cooling if cell is already at or below far-field temperature
    if (T_cell <= T_inf) return;

    // Convective heat flux at surface [W/m^2]
    float q_conv = h_conv * (T_cell - T_inf);

    // Convert surface flux to volumetric heat rate [W/m^3]
    float heat_rate = q_conv / dx;

    // Temperature change
    float rho_cp = rho * cp;
    float dT;
    if (rho_cp < 1e-6f) {
        dT = 0.0f;
    } else {
        dT = -heat_rate * dt / rho_cp;
    }

    // Stability limiter: never cool more than 10% of temp difference per step
    float max_cooling = -0.10f * (T_cell - T_inf);
    if (dT < max_cooling) {
        dT = max_cooling;
    }

    // Apply to all distribution functions (isotropic cooling)
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    int num_cells = nx * ny * nz;
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
    }
}

// ============================================================================
// Generalized per-face radiation BC kernel
// ============================================================================

__global__ void applyRadiationBCFaceKernel(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float epsilon,
    MaterialProperties material,
    float T_ambient,
    int boundary_face)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x, y, z;
    int idx = faceIndexHelper(tid, boundary_face, nx, ny, nz, x, y, z);
    if (idx < 0) return;

    float T_surf = temperature[idx];

    // Get temperature-dependent properties
    float rho = material.getDensity(T_surf);
    float cp = material.getSpecificHeat(T_surf);

    // Stefan-Boltzmann radiation: q = eps * sigma * (T^4 - T_amb^4)
    const float sigma = 5.67e-8f;
    float q_rad = epsilon * sigma * (powf(T_surf, 4.0f) - powf(T_ambient, 4.0f));

    // Convert surface flux to volumetric heat rate [W/m^3]
    float heat_rate = q_rad / dx;

    // Temperature change
    float rho_cp = rho * cp;
    float dT;
    if (rho_cp < 1e-6f) {
        dT = 0.0f;
    } else {
        dT = -heat_rate * dt / rho_cp;
    }

    // Stability limiter
    float max_cooling = -0.10f * fmaxf(T_surf - T_ambient, 0.0f);
    if (dT < max_cooling) {
        dT = max_cooling;
    }

    // Apply to all distribution functions
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    int num_cells = nx * ny * nz;
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
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
    // ALL top surface cells when T > T_evap_threshold, regardless of fill_level.
    // float f = fill_level[idx];

    // Check if above evaporation threshold (matches BC kernel logic)
    // FIX (2025-12-02): Changed from T_vaporization to T_vaporization - 500K
    float T_evap_threshold = material.T_vaporization - 500.0f;
    if (T < T_evap_threshold) {
        power_out[idx] = 0.0f;
        return;
    }

    // Compute evaporation mass flux using Hertz-Knudsen equation
    // Calibrated alpha_evap = 0.18 (Anisimov 1995)
    const float alpha_evap = 0.18f;
    const float M = material.molar_mass;  // kg/mol
    const float R = 8.314f;   // J/(mol·K)
    const float P_ref = 101325.0f;  // Pa
    const float PI = 3.14159265f;

    // Clausius-Clapeyron: P_sat = P_ref * exp[(L_v*M/R) * (1/T_boil - 1/T)]
    float T_boil = material.T_vaporization;
    float L_vap = material.L_vaporization;
    // OVERFLOW PROTECTION: Cap temperature and ensure minimum value
    // BUG FIX 2026-01-04: Ensure T_capped >= 1K to prevent division by zero
    float T_capped = fminf(T, 2.0f * T_boil);
    T_capped = fmaxf(T_capped, 1.0f);  // Minimum 1K to prevent division by zero
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

    // Compute power only at bottom surface (k=0) to match BC kernel
    const int k = 0;
    int idx = i + nx * (j + ny * k);
    float T = temperature[idx];

    // Match BC logic - no heat transfer if T <= T_substrate
    if (T <= T_substrate) {
        power_out[i + nx * j] = 0.0f;
        return;
    }

    // Convective heat flux [W/m²] (positive = heat leaving domain)
    float q_conv = h_conv * (T - T_substrate);

    // Cell surface area [m²]
    float A = dx * dx;

    // Power [W] for this cell (positive = heat loss)
    float cell_power = q_conv * A;

    // Store power for this (i,j) cell
    int idx_out = i + nx * j;
    power_out[idx_out] = cell_power;
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
    CUDA_CHECK(cudaMalloc(&d_power, num_cells_ * sizeof(float)));
    cudaMemset(d_power, 0, num_cells_ * sizeof(float));  // Initialize to 0!

    // Compute power for each surface cell
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    computeEvaporationPowerKernel<<<blocks, threads>>>(
        d_temperature, fill_level, d_power,
        material_, dx, nx, ny, nz
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum all powers
    std::vector<float> h_power(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_power.data(), d_power, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaMalloc(&d_power, num_cells_ * sizeof(float)));
    cudaMemset(d_power, 0, num_cells_ * sizeof(float));  // Initialize to 0!

    // Compute power for each surface cell
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    computeRadiationPowerKernel<<<blocks, threads>>>(
        d_temperature, fill_level, d_power,
        epsilon, T_ambient, dx, nx, ny, nz
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum all powers
    std::vector<float> h_power(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_power.data(), d_power, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    float total_power = 0.0f;
    int non_zero_count = 0;
    float max_cell_power = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        total_power += h_power[i];
        if (h_power[i] > 0.0f) {
            non_zero_count++;
            max_cell_power = std::max(max_cell_power, h_power[i]);
        }
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
    CUDA_CHECK(cudaMalloc(&d_energy, num_cells_ * sizeof(float)));

    // Compute energy for each cell
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    // ============================================================================
    // CRITICAL FIX (2025-12-02): Use initial temperature as reference
    // ============================================================================
    // BUG: Previously used T_solidus (1878K) as reference, but initial temperature
    // is typically 300K (ambient). This created artificial energy baseline shifts:
    //   E(T=300K) = rho*cp*(300K - 1878K)*V = NEGATIVE baseline
    // This caused 34% error in energy conservation tests.
    //
    // FIX: Use T_initial as reference (the actual starting temperature).
    // This ensures E_initial ≈ 0, so dE directly reflects energy added/removed.
    //
    // Physics rationale:
    //   - Energy is always measured relative to a reference state
    //   - For conservation tracking, reference should be the initial state
    //   - This makes dE = E_final - E_initial = actual energy change
    //   - For phase change materials, latent energy is still correctly tracked
    //     via the f_l * rho * L_fusion term (independent of T_ref choice)
    // ============================================================================
    float T_ref = T_initial_;

    computeThermalEnergyKernel<<<blocks, threads>>>(
        d_temperature, d_liquid_fraction, d_energy,
        material_, dx, T_ref, num_cells_
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum all energies
    std::vector<float> h_energy(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    float total_energy = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        total_energy += h_energy[i];
    }

    cudaFree(d_energy);

    return total_energy;
}

float ThermalLBM::computeSubstratePower(float dx, float h_conv, float T_substrate) const {
    // BUG FIX (2026-01-26): Remove incorrect has_material_ check
    // Substrate power calculation only needs T, dx, h_conv, T_substrate
    // No material properties required (unlike evaporation/radiation which need L_vap, etc.)

    int nx = nx_;
    int ny = ny_;
    int nz = nz_;

    // Allocate device memory for per-column power (nx * ny entries)
    int num_columns = nx * ny;
    float* d_power;
    CUDA_CHECK(cudaMalloc(&d_power, num_columns * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_power, 0, num_columns * sizeof(float)));

    // Compute power for each (i,j) column
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    computeSubstratePowerKernel<<<blocks, threads>>>(
        d_temperature, d_power,
        h_conv, T_substrate, dx,
        nx, ny, nz
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum all column powers
    std::vector<float> h_power(num_columns);
    CUDA_CHECK(cudaMemcpy(h_power.data(), d_power, num_columns * sizeof(float), cudaMemcpyDeviceToHost));

    float total_power = 0.0f;
    int non_zero_count = 0;
    float max_cell_power = 0.0f;
    for (int i = 0; i < num_columns; ++i) {
        total_power += h_power[i];
        if (h_power[i] > 0.0f) {
            non_zero_count++;
            max_cell_power = std::max(max_cell_power, h_power[i]);
        }
    }

    cudaFree(d_power);

    return total_power;
}

float ThermalLBM::computeCapPower(float dx, float dt) const {
    if (!d_cap_energy_removed_) return 0.0f;
    if (!has_material_) return 0.0f;

    // d_cap_energy_removed_[i] contains |ΔT| removed at cell i [K]
    // Power = Σ (ρ · cp · |ΔT| · dx³) / dt  [W]
    std::vector<float> h_dT(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_dT.data(), d_cap_energy_removed_,
                          num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    // Use liquid-phase properties (cap only fires in liquid/mush cells)
    float rho = material_.rho_liquid;
    float cp = material_.cp_liquid;
    float cell_vol = dx * dx * dx;

    double total_energy = 0.0;  // Use double for summation accuracy
    for (int i = 0; i < num_cells_; ++i) {
        total_energy += static_cast<double>(h_dT[i]);
    }

    return static_cast<float>(total_energy * rho * cp * cell_vol / dt);
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
    // Calibrated alpha_evap = 0.18 (Anisimov 1995)
    const float alpha_evap = 0.18f;
    const float M_molar = material.molar_mass; // Molar mass [kg/mol]
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
    // OVERFLOW PROTECTION: Cap temperature and ensure minimum value
    // BUG FIX 2026-01-04: Ensure T_capped >= 1K to prevent division by zero
    float T_capped = fminf(T, 2.0f * T_boil);  // Cap at 2x boiling point
    T_capped = fmaxf(T_capped, 1.0f);  // Minimum 1K to prevent division by zero
    float exponent = (L_vap * M_molar / R_gas) * (1.0f / T_boil - 1.0f / T_capped);
    exponent = fminf(exponent, 20.0f);  // Cap exponent to prevent exp overflow (exp(20) ~ 5e8)
    float P_sat = P_ref * expf(exponent);

    // Sanity check: cap P_sat to reasonable physical limit (10x atmospheric)
    P_sat = fminf(P_sat, 10.0f * P_ref);

    // Hertz-Knudsen-Langmuir evaporation mass flux [kg/(m^2·s)]
    float denominator = sqrtf(2.0f * PI * R_gas * T_capped / M_molar);

    // ============================================================================
    // CRITICAL: Zero-division protection (BUG FIX 2026-01-04)
    // ============================================================================
    // If T_capped = 0 (corrupted temperature), denominator = 0 → J_evap = Inf
    // This prevents NaN/Inf propagation through the simulation
    // ============================================================================
    if (denominator < 1e-10f) {
        // No evaporation at zero temperature (physically correct)
        J_evap[idx] = 0.0f;
    } else {
        J_evap[idx] = alpha_evap * P_sat / denominator;
    }
}

void ThermalLBM::computeEvaporationMassFlux(float* d_J_evap, const float* fill_level) const {
    if (!has_material_) {
        std::cerr << "WARNING: computeEvaporationMassFlux called without material properties\n";
        CUDA_CHECK(cudaMemset(d_J_evap, 0, num_cells_ * sizeof(float)));
        return;
    }

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_J_evap, 0, num_cells_ * sizeof(float)));

    // Launch kernel for ALL cells (not just top surface)
    // FIX: Use 3D grid to cover entire domain, kernel will filter to interface cells
    dim3 threads(8, 8, 4);
    dim3 blocks((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 3) / 4);

    computeEvaporationMassFluxKernel<<<blocks, threads>>>(
        d_temperature, fill_level, d_J_evap, material_, nx_, ny_, nz_
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Enthalpy Source Term kernel for phase change (Jiaung 2001)
 *
 * After LBM collision+streaming gives T* = Σg_q, this kernel enforces
 * strict enthalpy conservation by redistributing energy between sensible
 * heat (temperature) and latent heat (liquid fraction).
 *
 * Algorithm:
 *   1. Compute total specific enthalpy: H = cp·T* + fl_old·L
 *   2. Decode (T_new, fl_new) from H:
 *      - Solid:  H < H_sol → T = H/cp, fl = 0
 *      - Liquid: H > H_liq → T = (H-L)/cp, fl = 1
 *      - Mushy:  fl = (H - H_sol) / (cp·ΔT + L),  T = T_sol + fl·ΔT
 *   3. Correct: ΔT = T_new - T*, update g_q += w_q · ΔT
 *
 * The mushy-zone formula is derived from H = cp·T + fl·L with fl = (T-T_sol)/ΔT.
 * Computing fl first (not T) avoids catastrophic cancellation when L/ΔT >> cp.
 *
 * Reference: Jiaung, Ho & Lan (2001), "Lattice Boltzmann method for the heat
 *            conduction problem with phase change", Numer. Heat Transfer B, 39(2).
 */
__global__ void enthalpySourceTermKernel(
    float* g,
    float* temperature,
    float* liquid_fraction,
    const float* liquid_fraction_prev,
    const float* fill_level,      // VOF field (nullptr = no masking)
    MaterialProperties material,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // ========================================================================
    // VOF MASK: Phase change only occurs in metal, not in gas.
    //
    // Without this, hot gas above T_liquidus gets fl=1 and absorbs the full
    // latent heat of fusion — a 260 kJ/kg energy black hole that artificially
    // cools the melt pool and corrupts material properties in the gas phase.
    //
    // For interface cells (0 < fill < 1), scale latent heat by the metal
    // fraction: only the metal portion of the cell undergoes phase change.
    // ========================================================================
    float metal_fraction = 1.0f;
    if (fill_level != nullptr) {
        float f = fill_level[idx];
        if (f < 0.01f) {
            // Pure gas cell: no phase change, fl = 0
            liquid_fraction[idx] = 0.0f;
            return;
        }
        // Interface cell: scale latent heat by metal fraction
        metal_fraction = fminf(f, 1.0f);
    }

    float T_star = temperature[idx];
    float fl_old = liquid_fraction_prev[idx];

    // Use solid-phase reference properties (constant cp for enthalpy consistency)
    float cp = material.cp_solid;
    float T_sol = material.T_solidus;
    float T_liq = material.T_liquidus;
    float L = material.L_fusion * metal_fraction;  // Scale latent heat by metal fraction
    float dT_melt = T_liq - T_sol;

    // Guard: skip if no phase change parameters
    if (dT_melt < 1e-8f || material.L_fusion < 1e-6f) return;

    // Total specific enthalpy [J/kg]: sensible + latent
    float H = cp * T_star + fl_old * L;

    // Enthalpy bounds for phase detection
    float H_solidus = cp * T_sol;
    float H_liquidus = cp * T_liq + L;

    float T_new, fl_new;

    if (H <= H_solidus) {
        // Pure solid: all energy is sensible
        T_new = H / cp;
        fl_new = 0.0f;
    } else if (H >= H_liquidus) {
        // Pure liquid: subtract latent heat to get temperature
        T_new = (H - L) / cp;
        fl_new = 1.0f;
    } else {
        // Mushy zone: compute fl directly for numerical stability
        fl_new = (H - H_solidus) / (cp * dT_melt + L);
        fl_new = fmaxf(0.0f, fminf(1.0f, fl_new));
        T_new = T_sol + fl_new * dT_melt;
    }

    // Source term correction
    float dT = T_new - T_star;
    temperature[idx] = T_new;
    liquid_fraction[idx] = fl_new;

    // Update distribution functions to maintain consistency with corrected T
    const float weights[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    for (int q = 0; q < 7; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
    }
}

/**
 * @brief CUDA kernel to apply latent heat correction for phase change (LEGACY)
 *
 * This kernel implements the source term method for phase change:
 * ΔT = -L/(ρ·cp) · Δfl
 *
 * Where:
 * - L is latent heat of fusion [J/kg]
 * - ρ is density [kg/m³]
 * - cp is specific heat [J/(kg·K)]
 * - Δfl is change in liquid fraction
 *
 * Physical interpretation:
 * - During melting (Δfl > 0): Temperature decreases because energy goes into latent heat
 * - During solidification (Δfl < 0): Temperature increases from latent heat release
 *
 * @param g Distribution functions (will be modified)
 * @param temperature Temperature field (will be modified)
 * @param fl_curr Current liquid fraction
 * @param fl_prev Previous liquid fraction
 * @param material Material properties (includes L_fusion)
 * @param num_cells Number of cells
 */
__global__ void applyPhaseChangeCorrectionKernel(
    float* g,
    float* temperature,
    const float* fl_curr,
    const float* fl_prev,
    MaterialProperties material,
    int num_cells,
    int nx,
    int ny,
    int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // CRITICAL: Skip ALL boundary cells
    // The boundary conditions enforce T=T_boundary, and we should not
    // modify them with phase change correction
    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    // Compute change in liquid fraction
    float dfl = fl_curr[idx] - fl_prev[idx];

    // Skip if no phase change occurred
    if (fabsf(dfl) < 1e-6f) return;

    float T = temperature[idx];

    // Get temperature-dependent properties
    float rho = material.getDensity(T);
    float cp = material.getSpecificHeat(T);

    // Compute temperature correction due to latent heat
    // Positive dfl (melting): absorbs energy → temperature decrease
    // Negative dfl (solidification): releases energy → temperature increase
    //
    // CRITICAL: For mushy zone phase change, the correction should account
    // for the fact that the temperature change already occurred in the LBM step.
    // The correction represents the additional temperature drop needed to account
    // for latent heat absorption that wasn't in the LBM diffusion equation.

    // ============================================================================
    // CRITICAL: Zero-division protection (BUG FIX 2026-01-04)
    // ============================================================================
    // Protect against division by zero when cp = 0 (corrupted properties)
    // ============================================================================
    float dT;
    if (cp < 1e-6f) {
        // Invalid specific heat - skip phase change correction to prevent NaN/Inf
        dT = 0.0f;
    } else {
        dT = -(material.L_fusion / cp) * dfl;
    }

    // Apply correction to temperature
    temperature[idx] += dT;

    // Clamp to physical bounds
    temperature[idx] = fmaxf(T_MIN, fminf(T_MAX, temperature[idx]));

    // Update distribution functions to maintain equilibrium at new temperature
    // D3Q7 weights: w0 = 1/4, w1-6 = 1/8
    const float weights[D3Q7::Q] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    for (int q = 0; q < D3Q7::Q; ++q) {
        g[q * num_cells + idx] += weights[q] * dT;
    }
}

void ThermalLBM::applyPhaseChangeCorrection(float dt) {
    if (!phase_solver_) {
        return;  // Phase change not enabled
    }

    // Store current liquid fraction before update
    phase_solver_->storePreviousLiquidFraction();

    // Update liquid fraction based on current temperature
    phase_solver_->updateLiquidFraction(d_temperature);

    // Apply latent heat correction
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Get material with proper constant memory setup
    if (has_material_) {
        applyPhaseChangeCorrectionKernel<<<gridSize, blockSize>>>(
            d_g_src,
            d_temperature,
            phase_solver_->getLiquidFraction(),
            phase_solver_->getPreviousLiquidFraction(),
            material_,
            num_cells_,
            nx_,
            ny_,
            nz_
        );
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Recompute liquid fraction with corrected temperature
    phase_solver_->updateLiquidFraction(d_temperature);
}

} // namespace physics
} // namespace lbm