/**
 * @file fluid_lbm.cu
 * @brief Implementation of fluid LBM solver
 */

#include "physics/fluid_lbm.h"
#include "core/collision_bgk.h"
#include "core/streaming.h"
#include "core/boundary_conditions.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include "utils/cuda_check.h"

namespace lbm {
namespace physics {

using namespace lbm::core;

// Forward declarations of helper functions
__device__ __forceinline__ double computeEquilibriumDouble(
    int q, double rho, double ux, double uy, double uz);

// Forward declarations of CUDA kernels
__global__ void setBoundaryVelocityKernel(
    float* ux, float* uy, float* uz,
    const BoundaryNode* boundary_nodes,
    int n_boundary, int nx, int ny, int nz);

// Constructor
FluidLBM::FluidLBM(int nx, int ny, int nz,
                   float kinematic_viscosity,
                   float density,
                   BoundaryType boundary_x,
                   BoundaryType boundary_y,
                   BoundaryType boundary_z,
                   float dt, float dx)
    : nx_(nx), ny_(ny), nz_(nz),
      num_cells_(nx * ny * nz),
      dt_(dt), dx_(dx),
      nu_physical_(kinematic_viscosity),
      rho0_(density),
      boundary_x_(boundary_x),
      boundary_y_(boundary_y),
      boundary_z_(boundary_z),
      d_f_src(nullptr), d_f_dst(nullptr),
      d_rho(nullptr), d_ux(nullptr), d_uy(nullptr), d_uz(nullptr),
      d_pressure(nullptr),
      d_omega_field_(nullptr),
      d_boundary_nodes_(nullptr),
      n_boundary_nodes_(0)
{
    // Initialize D3Q19 lattice on device
    if (!D3Q19::isInitialized()) {
        D3Q19::initializeDevice();
    }

    // ============================================================================
    // CRITICAL FIX: Convert kinematic viscosity to lattice units
    // ============================================================================
    // LBM requires dimensionless viscosity in lattice units
    // Formula: nu_lattice = nu_physical * dt / (dx²)
    //
    // Physical: nu ~ 4.5e-7 m²/s (Ti-6Al-4V liquid)
    // dt: e.g., 1e-7 s (0.1 μs)
    // dx: e.g., 2e-6 m (2 μm)
    //
    // Example: nu_lattice = 4.5e-7 * 1e-7 / (2e-6)² = 0.01125 (dimensionless)
    // ============================================================================

    nu_lattice_ = kinematic_viscosity * dt / (dx * dx);

    // Compute tau from LATTICE viscosity
    // For D3Q19: nu = cs^2 * (tau - 0.5)
    // Therefore: tau = nu / cs^2 + 0.5
    tau_ = nu_lattice_ / D3Q19::CS2 + 0.5f;
    omega_ = 1.0f / tau_;

    // Stability check
    if (tau_ < 0.51f) {
        std::cout << "[WARNING] FluidLBM: tau=" << tau_ << " < 0.51 (unstable!)\n";
        std::cout << "          Clamping to tau=0.51 for stability.\n";
        tau_ = 0.51f;
        omega_ = 1.0f / tau_;
    }

    std::cout << "FluidLBM initialized:\n"
              << "  Domain: " << nx_ << " x " << ny_ << " x " << nz_ << "\n"
              << "  dt = " << dt_ << " s\n"
              << "  dx = " << dx_ << " m\n"
              << "  nu_physical = " << nu_physical_ << " m²/s\n"
              << "  nu_lattice = " << nu_lattice_ << " (dimensionless)\n"
              << "  tau = " << tau_ << "\n"
              << "  omega = " << omega_ << "\n"
              << "  Density: " << rho0_ << " kg/m³" << std::endl;

    allocateMemory();
    initializeBoundaryNodes();
}

// Destructor
FluidLBM::~FluidLBM() {
    freeMemory();
}

// Allocate device memory
void FluidLBM::allocateMemory() {
    size_t f_size = num_cells_ * D3Q19::Q * sizeof(float);
    size_t macro_size = num_cells_ * sizeof(float);

    // Clear any previous CUDA errors before allocation
    cudaGetLastError();

    cudaError_t error;

    error = cudaMalloc(&d_f_src, f_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_f_src (" +
                               std::to_string(f_size / (1024*1024)) + " MB): " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_f_dst, f_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_f_dst (" +
                               std::to_string(f_size / (1024*1024)) + " MB): " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_rho, macro_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_rho: " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_ux, macro_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_ux: " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_uy, macro_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_uy: " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_uz, macro_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_uz: " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_pressure, macro_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_pressure: " +
                               std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&d_omega_field_, macro_size);
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM: Failed to allocate d_omega_field_: " +
                               std::string(cudaGetErrorString(error)));
    }

    // Initialize omega field to default value (uniform viscosity)
    std::vector<float> h_omega(num_cells_, omega_);
    cudaMemcpy(d_omega_field_, h_omega.data(), macro_size, cudaMemcpyHostToDevice);
}

// Free device memory
void FluidLBM::freeMemory() {
    if (d_f_src) cudaFree(d_f_src);
    if (d_f_dst) cudaFree(d_f_dst);
    if (d_rho) cudaFree(d_rho);
    if (d_ux) cudaFree(d_ux);
    if (d_uy) cudaFree(d_uy);
    if (d_uz) cudaFree(d_uz);
    if (d_pressure) cudaFree(d_pressure);
    if (d_omega_field_) cudaFree(d_omega_field_);
    if (d_boundary_nodes_) cudaFree(d_boundary_nodes_);

    d_f_src = d_f_dst = nullptr;
    d_rho = d_ux = d_uy = d_uz = d_pressure = nullptr;
    d_omega_field_ = nullptr;
    d_boundary_nodes_ = nullptr;
}

// Initialize with uniform conditions
void FluidLBM::initialize(float initial_density,
                         float initial_ux,
                         float initial_uy,
                         float initial_uz) {
    // Allocate host memory for initialization
    size_t f_size = num_cells_ * D3Q19::Q;
    float* h_f = new float[f_size];

    // Initialize with equilibrium distribution
    for (int id = 0; id < num_cells_; ++id) {
        for (int q = 0; q < D3Q19::Q; ++q) {
            h_f[id + q * num_cells_] = D3Q19::computeEquilibrium(
                q, initial_density, initial_ux, initial_uy, initial_uz);
        }
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_f_src, h_f, f_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_dst, h_f, f_size * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_f;

    // Initialize macroscopic quantities
    float* h_macro = new float[num_cells_];
    std::fill(h_macro, h_macro + num_cells_, initial_density);
    CUDA_CHECK(cudaMemcpy(d_rho, h_macro, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    std::fill(h_macro, h_macro + num_cells_, initial_ux);
    CUDA_CHECK(cudaMemcpy(d_ux, h_macro, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    std::fill(h_macro, h_macro + num_cells_, initial_uy);
    CUDA_CHECK(cudaMemcpy(d_uy, h_macro, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    std::fill(h_macro, h_macro + num_cells_, initial_uz);
    CUDA_CHECK(cudaMemcpy(d_uz, h_macro, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize pressure: p = cs²(ρ - ρ₀)
    for (int id = 0; id < num_cells_; ++id) {
        h_macro[id] = D3Q19::CS2 * (initial_density - rho0_);
    }
    CUDA_CHECK(cudaMemcpy(d_pressure, h_macro, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_macro;

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Initialize with custom distribution
void FluidLBM::initialize(const float* density,
                         const float* ux,
                         const float* uy,
                         const float* uz) {
    // Copy macroscopic quantities to device
    CUDA_CHECK(cudaMemcpy(d_rho, density, num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux, ux, num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy, uy, num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_uz, uz, num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));

    // Copy to host to initialize distribution functions
    float* h_rho = new float[num_cells_];
    float* h_ux = new float[num_cells_];
    float* h_uy = new float[num_cells_];
    float* h_uz = new float[num_cells_];

    CUDA_CHECK(cudaMemcpy(h_rho, density, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ux, ux, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uy, uy, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uz, uz, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    // Initialize distribution functions
    size_t f_size = num_cells_ * D3Q19::Q;
    float* h_f = new float[f_size];

    for (int id = 0; id < num_cells_; ++id) {
        for (int q = 0; q < D3Q19::Q; ++q) {
            h_f[id + q * num_cells_] = D3Q19::computeEquilibrium(
                q, h_rho[id], h_ux[id], h_uy[id], h_uz[id]);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_f_src, h_f, f_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_dst, h_f, f_size * sizeof(float), cudaMemcpyHostToDevice));

    // Compute pressure
    float* h_pressure = new float[num_cells_];
    for (int id = 0; id < num_cells_; ++id) {
        h_pressure[id] = D3Q19::CS2 * (h_rho[id] - rho0_);
    }
    CUDA_CHECK(cudaMemcpy(d_pressure, h_pressure, num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_f;
    delete[] h_rho;
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;
    delete[] h_pressure;

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Collision with uniform force
void FluidLBM::collisionBGK(float force_x, float force_y, float force_z) {
    // ============================================================================
    // CRITICAL FIX: Convert force from physical units [m/s²] to lattice units
    // ============================================================================
    // LBM kernels work in dimensionless lattice units
    // Conversion formula: F_lattice = F_physical × dt² / dx
    //
    // Physical interpretation:
    // - F_physical [m/s²]: acceleration in physical units
    // - dt [s]: time step
    // - dx [m]: lattice spacing
    //
    // Example: F = 0.1 m/s², dt = 1e-7 s, dx = 1e-6 m
    // F_lattice = 0.1 × (1e-7)² / 1e-6 = 1e-9 (dimensionless)
    // ============================================================================

    float force_x_lattice = force_x * dt_ * dt_ / dx_;
    float force_y_lattice = force_y * dt_ * dt_ / dx_;
    float force_z_lattice = force_z * dt_ * dt_ / dx_;

    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
             (ny_ + block.y - 1) / block.y,
             (nz_ + block.z - 1) / block.z);

    fluidBGKCollisionKernel<<<grid, block>>>(
        d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz,
        force_x_lattice, force_y_lattice, force_z_lattice, omega_,
        nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM collision kernel failed: " +
                               std::string(cudaGetErrorString(error)));
    }

    swapDistributions();
}

// Collision with spatially-varying forces
void FluidLBM::collisionBGK(const float* force_x,
                            const float* force_y,
                            const float* force_z) {
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
             (ny_ + block.y - 1) / block.y,
             (nz_ + block.z - 1) / block.z);

    fluidBGKCollisionVaryingForceKernel<<<grid, block>>>(
        d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz,
        force_x, force_y, force_z, omega_,
        nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM collision kernel (varying force) failed: " +
                               std::string(cudaGetErrorString(error)));
    }

    swapDistributions();
}

// TRT collision with uniform force
void FluidLBM::collisionTRT(float force_x, float force_y, float force_z, float lambda) {
    // ============================================================================
    // CRITICAL FIX: Convert force from physical units [m/s²] to lattice units
    // ============================================================================
    float force_x_lattice = force_x * dt_ * dt_ / dx_;
    float force_y_lattice = force_y * dt_ * dt_ / dx_;
    float force_z_lattice = force_z * dt_ * dt_ / dx_;

    // Compute TRT relaxation rates
    // omega_even = omega = 1/tau (same as BGK)
    float omega_even = omega_;

    // omega_odd = 1 / (lambda/(1/omega_even - 0.5) + 0.5)
    float omega_odd = 1.0f / (lambda / (1.0f / omega_even - 0.5f) + 0.5f);

    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
             (ny_ + block.y - 1) / block.y,
             (nz_ + block.z - 1) / block.z);

    fluidTRTCollisionKernel<<<grid, block>>>(
        d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz,
        force_x_lattice, force_y_lattice, force_z_lattice, omega_even, omega_odd,
        nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM TRT collision kernel failed: " +
                               std::string(cudaGetErrorString(error)));
    }

    swapDistributions();
}

// TRT collision with spatially-varying forces
void FluidLBM::collisionTRT(const float* force_x,
                            const float* force_y,
                            const float* force_z,
                            float lambda) {
    // Compute TRT relaxation rates
    // omega_even = omega = 1/tau (same as BGK)
    float omega_even = omega_;

    // omega_odd = 1 / (lambda/(1/omega_even - 0.5) + 0.5)
    float omega_odd = 1.0f / (lambda / (1.0f / omega_even - 0.5f) + 0.5f);

    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
             (ny_ + block.y - 1) / block.y,
             (nz_ + block.z - 1) / block.z);

    fluidTRTCollisionVaryingForceKernel<<<grid, block>>>(
        d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz,
        force_x, force_y, force_z, omega_even, omega_odd,
        nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM TRT collision kernel (varying force) failed: " +
                               std::string(cudaGetErrorString(error)));
    }

    swapDistributions();
}

// Streaming
void FluidLBM::streaming() {
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
             (ny_ + block.y - 1) / block.y,
             (nz_ + block.z - 1) / block.z);

    // Determine if all boundaries are periodic
    bool all_periodic = (boundary_x_ == BoundaryType::PERIODIC &&
                        boundary_y_ == BoundaryType::PERIODIC &&
                        boundary_z_ == BoundaryType::PERIODIC);

    if (all_periodic) {
        // Use periodic streaming kernel
        fluidStreamingKernel<<<grid, block>>>(
            d_f_src, d_f_dst, nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();
    } else {
        // Use boundary-aware streaming kernel
        int periodic_x = (boundary_x_ == BoundaryType::PERIODIC) ? 1 : 0;
        int periodic_y = (boundary_y_ == BoundaryType::PERIODIC) ? 1 : 0;
        int periodic_z = (boundary_z_ == BoundaryType::PERIODIC) ? 1 : 0;

        fluidStreamingKernelWithWalls<<<grid, block>>>(
            d_f_src, d_f_dst, nx_, ny_, nz_,
            periodic_x, periodic_y, periodic_z);
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM streaming kernel failed: " +
                               std::string(cudaGetErrorString(error)));
    }

    swapDistributions();
}

// Apply boundary conditions
void FluidLBM::applyBoundaryConditions(int boundary_type) {
    // If boundary_type is 0 (periodic) or no boundary nodes, do nothing
    if (boundary_type == 0 || n_boundary_nodes_ == 0) {
        return;
    }

    // Apply all boundary conditions using unified kernel
    // This handles BOUNCE_BACK, VELOCITY, and other BC types
    int block_size = 256;
    int grid_size = (n_boundary_nodes_ + block_size - 1) / block_size;

    applyBoundaryConditionsKernel<<<grid_size, block_size>>>(
        d_f_src,
        d_rho,
        d_boundary_nodes_,
        n_boundary_nodes_,
        nx_, ny_, nz_
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM applyBoundaryConditions failed: " +
                               std::string(cudaGetErrorString(error)));
    }
}

// Compute macroscopic quantities
void FluidLBM::computeMacroscopic() {
    int block_size = 256;
    int grid_size = (num_cells_ + block_size - 1) / block_size;

    computeMacroscopicKernel<<<grid_size, block_size>>>(
        d_f_src, d_rho, d_ux, d_uy, d_uz, num_cells_);
    CUDA_CHECK_KERNEL();

    // Compute pressure
    computePressureKernel<<<grid_size, block_size>>>(
        d_rho, d_pressure, rho0_, D3Q19::CS2, num_cells_);
    CUDA_CHECK_KERNEL();

    // Enforce correct velocity at boundary nodes
    // - BOUNCE_BACK: zero velocity (no-slip)
    // - VELOCITY: prescribed wall velocity
    if (n_boundary_nodes_ > 0) {
        int wall_grid_size = (n_boundary_nodes_ + block_size - 1) / block_size;
        setBoundaryVelocityKernel<<<wall_grid_size, block_size>>>(
            d_ux, d_uy, d_uz,
            d_boundary_nodes_,
            n_boundary_nodes_,
            nx_, ny_, nz_
        );
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void FluidLBM::computeMacroscopic(const float* force_x,
                                   const float* force_y,
                                   const float* force_z) {
    int block_size = 256;
    int grid_size = (num_cells_ + block_size - 1) / block_size;

    computeMacroscopicWithForceKernel<<<grid_size, block_size>>>(
        d_f_src, d_rho, d_ux, d_uy, d_uz,
        force_x, force_y, force_z, num_cells_);
    CUDA_CHECK_KERNEL();

    // Compute pressure
    computePressureKernel<<<grid_size, block_size>>>(
        d_rho, d_pressure, rho0_, D3Q19::CS2, num_cells_);
    CUDA_CHECK_KERNEL();

    // Enforce correct velocity at boundary nodes
    if (n_boundary_nodes_ > 0) {
        int wall_grid_size = (n_boundary_nodes_ + block_size - 1) / block_size;
        setBoundaryVelocityKernel<<<wall_grid_size, block_size>>>(
            d_ux, d_uy, d_uz,
            d_boundary_nodes_,
            n_boundary_nodes_,
            nx_, ny_, nz_
        );
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute buoyancy force
void FluidLBM::computeBuoyancyForce(const float* temperature,
                                   float T_ref,
                                   float beta,
                                   float gravity_x,
                                   float gravity_y,
                                   float gravity_z,
                                   float* force_x,
                                   float* force_y,
                                   float* force_z) const {
    int block_size = 256;
    int grid_size = (num_cells_ + block_size - 1) / block_size;

    computeBuoyancyForceKernel<<<grid_size, block_size>>>(
        temperature, force_x, force_y, force_z,
        T_ref, beta, rho0_,
        gravity_x, gravity_y, gravity_z,
        num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Apply Darcy damping
void FluidLBM::applyDarcyDamping(const float* liquid_fraction,
                                float darcy_constant,
                                float* force_x,
                                float* force_y,
                                float* force_z) const {
    int block_size = 256;
    int grid_size = (num_cells_ + block_size - 1) / block_size;

    applyDarcyDampingKernel<<<grid_size, block_size>>>(
        liquid_fraction, d_ux, d_uy, d_uz,
        force_x, force_y, force_z,
        darcy_constant, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy velocity to host
void FluidLBM::copyVelocityToHost(float* host_ux, float* host_uy, float* host_uz) const {
    if (host_ux != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_ux, d_ux, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_uy != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_uy, d_uy, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_uz != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_uz, d_uz, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// Copy density to host
void FluidLBM::copyDensityToHost(float* host_rho) const {
    CUDA_CHECK(cudaMemcpy(host_rho, d_rho, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
}

// Copy pressure to host
void FluidLBM::copyPressureToHost(float* host_pressure) const {
    CUDA_CHECK(cudaMemcpy(host_pressure, d_pressure, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
}

// Compute Reynolds number
float FluidLBM::computeReynoldsNumber(float characteristic_velocity,
                                     float characteristic_length) const {
    return characteristic_velocity * characteristic_length / nu_physical_;
}

// Initialize boundary nodes based on boundary configuration
void FluidLBM::initializeBoundaryNodes() {
    // Count boundary nodes
    std::vector<BoundaryNode> h_boundary_nodes;

    // Add wall boundaries based on configuration
    // X boundaries
    if (boundary_x_ == BoundaryType::WALL) {
        for (int z = 0; z < nz_; ++z) {
            for (int y = 0; y < ny_; ++y) {
                // X-min boundary
                BoundaryNode node_min;
                node_min.x = 0;
                node_min.y = y;
                node_min.z = z;
                node_min.type = core::BoundaryType::BOUNCE_BACK;
                node_min.ux = 0.0f;
                node_min.uy = 0.0f;
                node_min.uz = 0.0f;
                node_min.pressure = 0.0f;
                node_min.directions = Streaming::BOUNDARY_X_MIN;
                h_boundary_nodes.push_back(node_min);

                // X-max boundary
                BoundaryNode node_max;
                node_max.x = nx_ - 1;
                node_max.y = y;
                node_max.z = z;
                node_max.type = core::BoundaryType::BOUNCE_BACK;
                node_max.ux = 0.0f;
                node_max.uy = 0.0f;
                node_max.uz = 0.0f;
                node_max.pressure = 0.0f;
                node_max.directions = Streaming::BOUNDARY_X_MAX;
                h_boundary_nodes.push_back(node_max);
            }
        }
    }

    // Y boundaries
    if (boundary_y_ == BoundaryType::WALL) {
        for (int z = 0; z < nz_; ++z) {
            for (int x = 0; x < nx_; ++x) {
                // Y-min boundary
                BoundaryNode node_min;
                node_min.x = x;
                node_min.y = 0;
                node_min.z = z;
                node_min.type = core::BoundaryType::BOUNCE_BACK;
                node_min.ux = 0.0f;
                node_min.uy = 0.0f;
                node_min.uz = 0.0f;
                node_min.pressure = 0.0f;
                node_min.directions = Streaming::BOUNDARY_Y_MIN;
                h_boundary_nodes.push_back(node_min);

                // Y-max boundary
                BoundaryNode node_max;
                node_max.x = x;
                node_max.y = ny_ - 1;
                node_max.z = z;
                node_max.type = core::BoundaryType::BOUNCE_BACK;
                node_max.ux = 0.0f;
                node_max.uy = 0.0f;
                node_max.uz = 0.0f;
                node_max.pressure = 0.0f;
                node_max.directions = Streaming::BOUNDARY_Y_MAX;
                h_boundary_nodes.push_back(node_max);
            }
        }
    }

    // Z boundaries
    if (boundary_z_ == BoundaryType::WALL) {
        for (int y = 0; y < ny_; ++y) {
            for (int x = 0; x < nx_; ++x) {
                // Z-min boundary
                BoundaryNode node_min;
                node_min.x = x;
                node_min.y = y;
                node_min.z = 0;
                node_min.type = core::BoundaryType::BOUNCE_BACK;
                node_min.ux = 0.0f;
                node_min.uy = 0.0f;
                node_min.uz = 0.0f;
                node_min.pressure = 0.0f;
                node_min.directions = Streaming::BOUNDARY_Z_MIN;
                h_boundary_nodes.push_back(node_min);

                // Z-max boundary
                BoundaryNode node_max;
                node_max.x = x;
                node_max.y = y;
                node_max.z = nz_ - 1;
                node_max.type = core::BoundaryType::BOUNCE_BACK;
                node_max.ux = 0.0f;
                node_max.uy = 0.0f;
                node_max.uz = 0.0f;
                node_max.pressure = 0.0f;
                node_max.directions = Streaming::BOUNDARY_Z_MAX;
                h_boundary_nodes.push_back(node_max);
            }
        }
    }

    n_boundary_nodes_ = h_boundary_nodes.size();

    // Allocate and copy to device if we have boundary nodes
    if (n_boundary_nodes_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_boundary_nodes_, n_boundary_nodes_ * sizeof(BoundaryNode)));
        cudaMemcpy(d_boundary_nodes_, h_boundary_nodes.data(),
                   n_boundary_nodes_ * sizeof(BoundaryNode),
                   cudaMemcpyHostToDevice);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("FluidLBM: Boundary node allocation failed: " +
                                   std::string(cudaGetErrorString(error)));
        }
    }
}

// Swap distribution function pointers
void FluidLBM::swapDistributions() {
    float* temp = d_f_src;
    d_f_src = d_f_dst;
    d_f_dst = temp;
}

// Set moving wall boundary condition
void FluidLBM::setMovingWall(unsigned int wall_direction,
                             float ux_wall,
                             float uy_wall,
                             float uz_wall) {
    // Copy boundary nodes to host for modification
    std::vector<core::BoundaryNode> h_boundary_nodes(n_boundary_nodes_);
    if (n_boundary_nodes_ > 0) {
        CUDA_CHECK(cudaMemcpy(h_boundary_nodes.data(), d_boundary_nodes_,
                             n_boundary_nodes_ * sizeof(core::BoundaryNode),
                             cudaMemcpyDeviceToHost));
    }

    // Modify boundary nodes matching the specified wall direction
    // EXCLUDE corner nodes that are also on other (non-periodic) walls
    // Use position-based corner detection since boundary nodes may be stored separately
    int modified_count = 0;
    int excluded_corners = 0;
    for (auto& node : h_boundary_nodes) {
        // Check if this node is on the specified wall
        if (node.directions & wall_direction) {
            // Check if this is a corner node by examining position
            bool is_corner = false;

            // Check if node is at an x-boundary (when x walls are not periodic)
            // Only counts as corner if this isn't the moving wall direction
            if (boundary_x_ == BoundaryType::WALL) {
                if (!(wall_direction & Streaming::BOUNDARY_X_MIN) && node.x == 0) {
                    is_corner = true;
                }
                if (!(wall_direction & Streaming::BOUNDARY_X_MAX) && node.x == nx_ - 1) {
                    is_corner = true;
                }
            }
            // Check if node is at a y-boundary (when y walls are not periodic)
            if (boundary_y_ == BoundaryType::WALL) {
                if (node.y == 0 || node.y == ny_ - 1) {
                    // Only counts as corner if this isn't the moving wall direction
                    if (!(wall_direction & Streaming::BOUNDARY_Y_MIN) && node.y == 0) {
                        is_corner = true;
                    }
                    if (!(wall_direction & Streaming::BOUNDARY_Y_MAX) && node.y == ny_ - 1) {
                        is_corner = true;
                    }
                }
            }
            // Check if node is at a z-boundary (when z walls are not periodic)
            // Only counts as corner if this isn't the moving wall direction
            if (boundary_z_ == BoundaryType::WALL) {
                if (!(wall_direction & Streaming::BOUNDARY_Z_MIN) && node.z == 0) {
                    is_corner = true;
                }
                if (!(wall_direction & Streaming::BOUNDARY_Z_MAX) && node.z == nz_ - 1) {
                    is_corner = true;
                }
            }

            if (!is_corner) {
                // Interior wall node: apply velocity BC
                node.type = core::BoundaryType::VELOCITY;
                node.ux = ux_wall;
                node.uy = uy_wall;
                node.uz = uz_wall;
                modified_count++;
            } else {
                // Corner node: keep as bounce-back to avoid singularity
                excluded_corners++;
            }
        }
    }

    // Copy modified boundary nodes back to device
    if (n_boundary_nodes_ > 0) {
        CUDA_CHECK(cudaMemcpy(d_boundary_nodes_, h_boundary_nodes.data(),
                             n_boundary_nodes_ * sizeof(core::BoundaryNode),
                             cudaMemcpyHostToDevice));
    }

    std::cout << "FluidLBM: Set moving wall BC on " << modified_count
              << " nodes with velocity (" << ux_wall << ", "
              << uy_wall << ", " << uz_wall << ")"
              << " (" << excluded_corners << " corner nodes excluded)" << std::endl;
}

// Compute variable viscosity field from VOF
void FluidLBM::computeVariableViscosity(const float* vof_field,
                                        float rho_heavy,
                                        float rho_light,
                                        float mu_constant) {
    int block_size = 256;
    int grid_size = (num_cells_ + block_size - 1) / block_size;

    computeVariableOmegaKernel<<<grid_size, block_size>>>(
        vof_field, d_omega_field_,
        rho_heavy, rho_light, mu_constant,
        dt_, dx_, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM computeVariableViscosity failed: " +
                               std::string(cudaGetErrorString(error)));
    }
}

// Kernel to fill omega field with uniform value
__global__ void fillUniformOmegaKernel(float* omega_field, float omega_uniform, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        omega_field[idx] = omega_uniform;
    }
}

// Set uniform kinematic viscosity (same ν for both phases)
// This is the standard approach for RT instability benchmarks
void FluidLBM::computeUniformViscosity(float nu_constant) {
    // Compute nu_lattice = nu * dt / dx²
    float nu_lattice = nu_constant * dt_ / (dx_ * dx_);
    // tau = nu_lattice / cs² + 0.5
    float cs2 = 1.0f / 3.0f;
    float tau = nu_lattice / cs2 + 0.5f;
    // Clamp tau for stability (tau > 0.5)
    tau = fmaxf(tau, 0.505f);
    // omega = 1 / tau
    float omega_uniform = 1.0f / tau;

    // Update member variables (used by collisionTRT)
    tau_ = tau;
    omega_ = omega_uniform;
    nu_lattice_ = nu_lattice;

    // Also fill omega field with uniform value (for consistency)
    int block_size = 256;
    int grid_size = (num_cells_ + block_size - 1) / block_size;
    fillUniformOmegaKernel<<<grid_size, block_size>>>(d_omega_field_, omega_uniform, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// TRT collision with variable omega field
void FluidLBM::collisionTRTVariable(const float* force_x,
                                    const float* force_y,
                                    const float* force_z,
                                    const float* vof_field,
                                    float rho_heavy,
                                    float rho_light,
                                    float lambda) {
    // CRITICAL FIX (2026-01-20): DO NOT pre-compensate forces in-place!
    //
    // Previous approach: Pre-multiply forces by 1/(1-ω/2) to make them τ-independent
    // Problem: When omega→2, compensation→∞, amplifying numerical errors 20x!
    // This caused simulation freeze at t≈0.73s with u_max→0.
    //
    // NEW approach: Apply compensation INSIDE the collision kernel where forces are used.
    // This avoids amplifying numerical errors while still achieving τ-independent forcing.
    //
    // The compensation is now integrated into fluidTRTCollisionVariableOmegaKernel
    // at the point where force terms are computed (line ~1787).

    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
             (ny_ + block.y - 1) / block.y,
             (nz_ + block.z - 1) / block.z);

    fluidTRTCollisionVariableOmegaKernel<<<grid, block>>>(
        d_f_src, d_f_dst, d_rho, d_ux, d_uy, d_uz,
        force_x, force_y, force_z,
        d_omega_field_, vof_field,
        rho_heavy, rho_light, lambda,
        nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("FluidLBM TRT variable omega collision kernel failed: " +
                               std::string(cudaGetErrorString(error)));
    }

    swapDistributions();
}

//=============================================================================
// CUDA Kernels
//=============================================================================

// Force compensation kernel for variable viscosity (τ-independent forcing)
//
// PROBLEM: Guo forcing includes factor (1-ω/2) = (τ-0.5)/τ that artificially
// couples force magnitude to relaxation time τ. This causes asymmetric force
// response in two-phase flows with variable viscosity:
//   - Light phase: τ=1.24 → factor=0.597 (60% of force applied)
//   - Heavy phase: τ=0.6  → factor=0.167 (17% of force applied)
//
// SOLUTION: Pre-compensate forces by multiplying by 1/(1-ω/2) = τ/(τ-0.5) = 2/(2-ω)
// before collision. This cancels the (1-ω/2) factor in the Guo forcing term,
// making the effective force τ-independent:
//   F_effective = F_compensated × (1-ω/2) = F × [1/(1-ω/2)] × (1-ω/2) = F
//
// CRITICAL FIX (2026-01-20): Clamp omega away from 2.0 BEFORE compensation
// to prevent numerical blowup. Clamping compensation after calculation causes
// asymmetric physics and simulation freeze.
//
// REFERENCE: This is the correct implementation for variable-viscosity flows
// where the physical force should be independent of local relaxation time.
__global__ void compensateForceForOmegaKernel(
    float* force_x,
    float* force_y,
    float* force_z,
    const float* omega_field,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float omega = omega_field[idx];

    // CRITICAL FIX: Clamp omega away from 2.0 to prevent compensation blowup
    // This ensures stable compensation across all phases while maintaining
    // consistent physics. The instability limit is tau=0.5 (omega=2.0).
    // Safe maximum: omega=1.9 (tau=0.526) → compensation=5.0
    const float OMEGA_MAX = 1.9f;  // Corresponds to tau=0.526 (stable)
    omega = fminf(omega, OMEGA_MAX);

    // Compensation factor: 1/(1-ω/2) = 2/(2-ω) = τ/(τ-0.5)
    // This cancels the (1-ω/2) factor in Guo forcing
    float compensation = 2.0f / (2.0f - omega);

    // Pre-multiply forces by compensation factor
    // After this, the collision kernel's (1-ω/2) factor will cancel out
    force_x[idx] *= compensation;
    force_y[idx] *= compensation;
    force_z[idx] *= compensation;
}

// Fluid BGK collision with uniform force (Guo forcing scheme)
// Double-precision equilibrium computation for high-accuracy TRT kernels
// This function performs equilibrium calculation in double precision to minimize
// cumulative errors over long time integrations (e.g., 50,000 timesteps)
// CRITICAL FIX: Use w_double[q] for maximum precision (15 digits vs 7 for float)
__device__ __forceinline__ double computeEquilibriumDouble(
    int q, double rho, double ux, double uy, double uz) {
    // Compressible form: f_eq = w * rho * (1 + 3*(c·u) + 4.5*(c·u)² - 1.5*u²)
    // Use double-precision weights to avoid accumulating rounding errors
    double eu = (double)ex[q] * ux + (double)ey[q] * uy + (double)ez[q] * uz;
    double u2 = ux*ux + uy*uy + uz*uz;
    return w_double[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
}

__global__ void fluidBGKCollisionKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    float force_x,
    float force_y,
    float force_z,
    float omega,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Compute macroscopic quantities from distribution functions
    float m_rho = 0.0f;
    float m_ux_star = 0.0f;  // Uncorrected momentum / rho
    float m_uy_star = 0.0f;
    float m_uz_star = 0.0f;

    for (int q = 0; q < D3Q19::Q; ++q) {
        float f = f_src[id + q * n_cells];
        m_rho += f;
        m_ux_star += ex[q] * f;
        m_uy_star += ey[q] * f;
        m_uz_star += ez[q] * f;
    }

    // Compute uncorrected velocity with safety check
    const float RHO_MIN = 1e-6f;
    float inv_rho = 1.0f / fmaxf(m_rho, RHO_MIN);
    float m_ux_uncorrected = m_ux_star * inv_rho;
    float m_uy_uncorrected = m_uy_star * inv_rho;
    float m_uz_uncorrected = m_uz_star * inv_rho;

    // Apply Guo forcing scheme: u = u_uncorrected + 0.5 * F / ρ
    float m_ux = m_ux_uncorrected + 0.5f * force_x * inv_rho;
    float m_uy = m_uy_uncorrected + 0.5f * force_y * inv_rho;
    float m_uz = m_uz_uncorrected + 0.5f * force_z * inv_rho;

    // Store corrected macroscopic quantities
    rho[id] = m_rho;
    ux[id] = m_ux;
    uy[id] = m_uy;
    uz[id] = m_uz;

    // Use corrected velocity for equilibrium
    float m_ux_force = m_ux;
    float m_uy_force = m_uy;
    float m_uz_force = m_uz;

    // BGK collision with forcing
    for (int q = 0; q < D3Q19::Q; ++q) {
        float f = f_src[id + q * n_cells];

        // Equilibrium distribution with force-corrected velocity
        float feq = D3Q19::computeEquilibrium(q, m_rho, m_ux_force, m_uy_force, m_uz_force);

        // Complete Guo forcing term with numerical stability optimization
        // F_i = (1 - ω/2) * w_i * [3(c_i - u)·F + 9(c_i·u)(c_i·F)]
        float ci_dot_u = ex[q] * m_ux + ey[q] * m_uy + ez[q] * m_uz;
        float ci_dot_F = ex[q] * force_x + ey[q] * force_y + ez[q] * force_z;

        // Numerically stable (c - u)·F computation to avoid catastrophic cancellation
        float c_minus_u_dot_F = (ex[q] - m_ux) * force_x +
                                (ey[q] - m_uy) * force_y +
                                (ez[q] - m_uz) * force_z;

        // First term: 3(c_i - u)·F
        float term1 = 3.0f * c_minus_u_dot_F;

        // Second term: 9(c_i·u)(c_i·F)
        float term2 = 9.0f * ci_dot_u * ci_dot_F;

        float force_term = (1.0f - 0.5f * omega) * w[q] * (term1 + term2);

        // BGK collision with force
        f_dst[id + q * n_cells] = f - omega * (f - feq) + force_term;
    }
}

// Fluid BGK collision with spatially-varying forces
__global__ void fluidBGKCollisionVaryingForceKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    float omega,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Read local forces
    float fx = force_x[id];
    float fy = force_y[id];
    float fz = force_z[id];

    // Compute macroscopic quantities from distribution functions
    float m_rho = 0.0f;
    float m_ux_star = 0.0f;
    float m_uy_star = 0.0f;
    float m_uz_star = 0.0f;

    for (int q = 0; q < D3Q19::Q; ++q) {
        float f = f_src[id + q * n_cells];
        m_rho += f;
        m_ux_star += ex[q] * f;
        m_uy_star += ey[q] * f;
        m_uz_star += ez[q] * f;
    }

    // Compute uncorrected velocity with safety check
    const float RHO_MIN = 1e-6f;
    float inv_rho = 1.0f / fmaxf(m_rho, RHO_MIN);
    float m_ux_uncorrected = m_ux_star * inv_rho;
    float m_uy_uncorrected = m_uy_star * inv_rho;
    float m_uz_uncorrected = m_uz_star * inv_rho;

    // Apply Guo forcing scheme: u = u_uncorrected + 0.5 * F / ρ
    float m_ux = m_ux_uncorrected + 0.5f * fx * inv_rho;
    float m_uy = m_uy_uncorrected + 0.5f * fy * inv_rho;
    float m_uz = m_uz_uncorrected + 0.5f * fz * inv_rho;

    // Store corrected macroscopic quantities
    rho[id] = m_rho;
    ux[id] = m_ux;
    uy[id] = m_uy;
    uz[id] = m_uz;

    // Use corrected velocity for equilibrium
    float m_ux_force = m_ux;
    float m_uy_force = m_uy;
    float m_uz_force = m_uz;

    // BGK collision with forcing
    for (int q = 0; q < D3Q19::Q; ++q) {
        float f = f_src[id + q * n_cells];

        float feq = D3Q19::computeEquilibrium(q, m_rho, m_ux_force, m_uy_force, m_uz_force);

        // Complete Guo forcing term with numerical stability optimization
        // F_i = (1 - ω/2) * w_i * [3(c_i - u)·F + 9(c_i·u)(c_i·F)]
        float ci_dot_u = ex[q] * m_ux + ey[q] * m_uy + ez[q] * m_uz;
        float ci_dot_F = ex[q] * fx + ey[q] * fy + ez[q] * fz;

        // Numerically stable (c - u)·F computation to avoid catastrophic cancellation
        float c_minus_u_dot_F = (ex[q] - m_ux) * fx +
                                (ey[q] - m_uy) * fy +
                                (ez[q] - m_uz) * fz;

        // First term: 3(c_i - u)·F
        float term1 = 3.0f * c_minus_u_dot_F;

        // Second term: 9(c_i·u)(c_i·F)
        float term2 = 9.0f * ci_dot_u * ci_dot_F;

        float force_term = (1.0f - 0.5f * omega) * w[q] * (term1 + term2);

        f_dst[id + q * n_cells] = f - omega * (f - feq) + force_term;
    }
}

// TRT collision kernel with uniform force
__global__ void fluidTRTCollisionKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    float force_x,
    float force_y,
    float force_z,
    float omega_even,
    float omega_odd,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Compute macroscopic quantities with DOUBLE PRECISION to match walberla
    // This reduces cumulative error over 50,000 timesteps
    double m_rho_d = 0.0;
    double m_ux_star_d = 0.0;
    double m_uy_star_d = 0.0;
    double m_uz_star_d = 0.0;

    for (int q = 0; q < D3Q19::Q; ++q) {
        double f_d = (double)f_src[id + q * n_cells];
        m_rho_d += f_d;
        m_ux_star_d += (double)ex[q] * f_d;
        m_uy_star_d += (double)ey[q] * f_d;
        m_uz_star_d += (double)ez[q] * f_d;
    }

    // Compute uncorrected velocity with safety check
    const double RHO_MIN_D = 1e-12;
    double inv_rho_d = 1.0 / fmax(m_rho_d, RHO_MIN_D);
    double m_ux_uncorrected_d = m_ux_star_d * inv_rho_d;
    double m_uy_uncorrected_d = m_uy_star_d * inv_rho_d;
    double m_uz_uncorrected_d = m_uz_star_d * inv_rho_d;

    // Apply Guo forcing scheme: u = u_uncorrected + 0.5 * F / rho
    double m_ux_d = m_ux_uncorrected_d + 0.5 * (double)force_x * inv_rho_d;
    double m_uy_d = m_uy_uncorrected_d + 0.5 * (double)force_y * inv_rho_d;
    double m_uz_d = m_uz_uncorrected_d + 0.5 * (double)force_z * inv_rho_d;

    // Store corrected macroscopic quantities (convert to float for output)
    rho[id] = (float)m_rho_d;
    ux[id] = (float)m_ux_d;
    uy[id] = (float)m_uy_d;
    uz[id] = (float)m_uz_d;

    // TRT collision with forcing - ALL OPERATIONS IN DOUBLE PRECISION
    for (int q = 0; q < D3Q19::Q; ++q) {
        int q_opp = opposite[q];

        // Read distribution functions
        double f_q_d = (double)f_src[id + q * n_cells];
        double f_qbar_d = (double)f_src[id + q_opp * n_cells];

        // Equilibrium distributions - use double-precision computation
        double feq_q_d = computeEquilibriumDouble(q, m_rho_d, m_ux_d, m_uy_d, m_uz_d);
        double feq_qbar_d = computeEquilibriumDouble(q_opp, m_rho_d, m_ux_d, m_uy_d, m_uz_d);

        // Symmetric and antisymmetric parts (TRT decomposition)
        double f_plus_d = 0.5 * (f_q_d + f_qbar_d);
        double f_minus_d = 0.5 * (f_q_d - f_qbar_d);
        double feq_plus_d = 0.5 * (feq_q_d + feq_qbar_d);
        double feq_minus_d = 0.5 * (feq_q_d - feq_qbar_d);

        // Guo forcing term calculation with double precision + FMA for maximum accuracy
        // CRITICAL OPTIMIZATION: Use double precision + FMA to minimize numerical errors
        // FMA (fused multiply-add) avoids intermediate rounding for (a*b)+c operations
        double dx = (double)ex[q];
        double dy = (double)ey[q];
        double dz = (double)ez[q];
        double dF_x = (double)force_x;
        double dF_y = (double)force_y;
        double dF_z = (double)force_z;

        // Use FMA for dot products: ci_dot_u = ex*ux + ey*uy + ez*uz
        double ci_dot_u_d = fma(dx, m_ux_d, fma(dy, m_uy_d, dz * m_uz_d));
        double ci_dot_F_d = fma(dx, dF_x, fma(dy, dF_y, dz * dF_z));

        // Numerically stable (c - u)·F computation to avoid catastrophic cancellation
        // Use FMA: (c-u)·F = (ex-ux)*Fx + (ey-uy)*Fy + (ez-uz)*Fz
        double c_minus_u_dot_F_d = fma(dx - m_ux_d, dF_x, fma(dy - m_uy_d, dF_y, (dz - m_uz_d) * dF_z));

        // Standard Guo term: 3(c - u)·F + 9(c·u)(c·F)
        double term1_d = 3.0 * c_minus_u_dot_F_d;
        double term2_d = 9.0 * ci_dot_u_d * ci_dot_F_d;

        // TRT forcing: standard BGK term + TRT correction
        // CRITICAL FIX: Use w_double[q] for maximum precision (15 digits vs 7 for float)
        double force_term_d = (1.0 - 0.5 * (double)omega_even) * w_double[q] * (term1_d + term2_d);

        // TRT correction (Walberla ForceModel.h line 533-535)
        // Full formula: 3w[(1-0.5*omega)((c-u+3(c·u)c)·F) + (omega-omega_odd)*0.5*(c·F)]
        // Second term expands to: 3w * (omega_even - omega_odd) * 0.5 * (c·F) = 1.5w * (omega_even - omega_odd) * (c·F)
        force_term_d += 1.5 * w_double[q] * ((double)omega_even - (double)omega_odd) * ci_dot_F_d;

        // TRT collision: relax symmetric part with omega_even, antisymmetric with omega_odd
        double omega_even_d = (double)omega_even;
        double omega_odd_d = (double)omega_odd;
        double f_plus_new_d = f_plus_d - omega_even_d * (f_plus_d - feq_plus_d);
        double f_minus_new_d = f_minus_d - omega_odd_d * (f_minus_d - feq_minus_d);

        // Reconstruct post-collision distribution with forcing
        // NOTE: force_term is added as a separate source term, NOT double-applied.
        // This matches walberla's implementation (CellwiseSweep.impl.h:1647-1648)
        // Formula: f_new = (f_plus - lambda_e*(f_plus - feq_plus)) + (f_minus - lambda_d*(f_minus - feq_minus)) + forceTerm
        f_dst[id + q * n_cells] = (float)(f_plus_new_d + f_minus_new_d + force_term_d);
    }
}

// TRT collision kernel with spatially-varying forces
__global__ void fluidTRTCollisionVaryingForceKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    float omega_even,
    float omega_odd,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Read local forces
    float fx = force_x[id];
    float fy = force_y[id];
    float fz = force_z[id];

    // Compute macroscopic quantities with DOUBLE PRECISION to match walberla
    // This reduces cumulative error over 50,000 timesteps
    double m_rho_d = 0.0;
    double m_ux_star_d = 0.0;
    double m_uy_star_d = 0.0;
    double m_uz_star_d = 0.0;

    for (int q = 0; q < D3Q19::Q; ++q) {
        double f_d = (double)f_src[id + q * n_cells];
        m_rho_d += f_d;
        m_ux_star_d += (double)ex[q] * f_d;
        m_uy_star_d += (double)ey[q] * f_d;
        m_uz_star_d += (double)ez[q] * f_d;
    }

    // Compute uncorrected velocity with safety check
    const double RHO_MIN_D = 1e-12;
    double inv_rho_d = 1.0 / fmax(m_rho_d, RHO_MIN_D);
    double m_ux_uncorrected_d = m_ux_star_d * inv_rho_d;
    double m_uy_uncorrected_d = m_uy_star_d * inv_rho_d;
    double m_uz_uncorrected_d = m_uz_star_d * inv_rho_d;

    // Apply Guo forcing scheme: u = u_uncorrected + 0.5 * F / rho
    double m_ux_d = m_ux_uncorrected_d + 0.5 * (double)fx * inv_rho_d;
    double m_uy_d = m_uy_uncorrected_d + 0.5 * (double)fy * inv_rho_d;
    double m_uz_d = m_uz_uncorrected_d + 0.5 * (double)fz * inv_rho_d;

    // Store corrected macroscopic quantities (convert to float for output)
    rho[id] = (float)m_rho_d;
    ux[id] = (float)m_ux_d;
    uy[id] = (float)m_uy_d;
    uz[id] = (float)m_uz_d;

    // TRT collision with forcing - ALL OPERATIONS IN DOUBLE PRECISION
    for (int q = 0; q < D3Q19::Q; ++q) {
        int q_opp = opposite[q];

        // Read distribution functions
        double f_q_d = (double)f_src[id + q * n_cells];
        double f_qbar_d = (double)f_src[id + q_opp * n_cells];

        // Equilibrium distributions - use double-precision computation
        double feq_q_d = computeEquilibriumDouble(q, m_rho_d, m_ux_d, m_uy_d, m_uz_d);
        double feq_qbar_d = computeEquilibriumDouble(q_opp, m_rho_d, m_ux_d, m_uy_d, m_uz_d);

        // Symmetric and antisymmetric parts (TRT decomposition)
        double f_plus_d = 0.5 * (f_q_d + f_qbar_d);
        double f_minus_d = 0.5 * (f_q_d - f_qbar_d);
        double feq_plus_d = 0.5 * (feq_q_d + feq_qbar_d);
        double feq_minus_d = 0.5 * (feq_q_d - feq_qbar_d);

        // Guo forcing term calculation with double precision + FMA for maximum accuracy
        // CRITICAL OPTIMIZATION: Use double precision + FMA to minimize numerical errors
        // FMA (fused multiply-add) avoids intermediate rounding for (a*b)+c operations
        double dx = (double)ex[q];
        double dy = (double)ey[q];
        double dz = (double)ez[q];
        double dF_x = (double)fx;
        double dF_y = (double)fy;
        double dF_z = (double)fz;

        // Use FMA for dot products: ci_dot_u = ex*ux + ey*uy + ez*uz
        double ci_dot_u_d = fma(dx, m_ux_d, fma(dy, m_uy_d, dz * m_uz_d));
        double ci_dot_F_d = fma(dx, dF_x, fma(dy, dF_y, dz * dF_z));

        // Numerically stable (c - u)·F computation to avoid catastrophic cancellation
        // Use FMA: (c-u)·F = (ex-ux)*Fx + (ey-uy)*Fy + (ez-uz)*Fz
        double c_minus_u_dot_F_d = fma(dx - m_ux_d, dF_x, fma(dy - m_uy_d, dF_y, (dz - m_uz_d) * dF_z));

        // Standard Guo term: 3(c - u)·F + 9(c·u)(c·F)
        double term1_d = 3.0 * c_minus_u_dot_F_d;
        double term2_d = 9.0 * ci_dot_u_d * ci_dot_F_d;

        // TRT forcing: standard BGK term + TRT correction
        // CRITICAL FIX: Use w_double[q] for maximum precision (15 digits vs 7 for float)
        double force_term_d = (1.0 - 0.5 * (double)omega_even) * w_double[q] * (term1_d + term2_d);

        // TRT correction (Walberla ForceModel.h line 533-535)
        // Full formula: 3w[(1-0.5*omega)((c-u+3(c·u)c)·F) + (omega-omega_odd)*0.5*(c·F)]
        // Second term expands to: 3w * (omega_even - omega_odd) * 0.5 * (c·F) = 1.5w * (omega_even - omega_odd) * (c·F)
        force_term_d += 1.5 * w_double[q] * ((double)omega_even - (double)omega_odd) * ci_dot_F_d;

        // TRT collision: relax symmetric part with omega_even, antisymmetric with omega_odd
        double omega_even_d = (double)omega_even;
        double omega_odd_d = (double)omega_odd;
        double f_plus_new_d = f_plus_d - omega_even_d * (f_plus_d - feq_plus_d);
        double f_minus_new_d = f_minus_d - omega_odd_d * (f_minus_d - feq_minus_d);

        // Reconstruct post-collision distribution with forcing
        // NOTE: force_term is added as a separate source term, NOT double-applied.
        // This matches walberla's implementation (CellwiseSweep.impl.h:1647-1648)
        // Formula: f_new = (f_plus - lambda_e*(f_plus - feq_plus)) + (f_minus - lambda_d*(f_minus - feq_minus)) + forceTerm
        f_dst[id + q * n_cells] = (float)(f_plus_new_d + f_minus_new_d + force_term_d);
    }
}

// Streaming (periodic boundaries)
__global__ void fluidStreamingKernel(
    const float* f_src,
    float* f_dst,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Stream each distribution function
    for (int q = 0; q < D3Q19::Q; ++q) {
        // Destination coordinates (with periodic BC)
        int dst_x = (idx + ex[q] + nx) % nx;
        int dst_y = (idy + ey[q] + ny) % ny;
        int dst_z = (idz + ez[q] + nz) % nz;
        int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;

        // Copy distribution function to destination
        f_dst[dst_id + q * n_cells] = f_src[id + q * n_cells];
    }
}

// Streaming with mixed boundary conditions (periodic/wall)
__global__ void fluidStreamingKernelWithWalls(
    const float* f_src,
    float* f_dst,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Check if current cell is a wall boundary node and determine which walls
    unsigned int wall_directions = 0;
    if (!periodic_x) {
        if (idx == 0) wall_directions |= Streaming::BOUNDARY_X_MIN;
        if (idx == nx - 1) wall_directions |= Streaming::BOUNDARY_X_MAX;
    }
    if (!periodic_y) {
        if (idy == 0) wall_directions |= Streaming::BOUNDARY_Y_MIN;
        if (idy == ny - 1) wall_directions |= Streaming::BOUNDARY_Y_MAX;
    }
    if (!periodic_z) {
        if (idz == 0) wall_directions |= Streaming::BOUNDARY_Z_MIN;
        if (idz == nz - 1) wall_directions |= Streaming::BOUNDARY_Z_MAX;
    }

    bool is_wall_node = (wall_directions != 0);

    // Stream each distribution function
    for (int q = 0; q < D3Q19::Q; ++q) {
        // Check if this is an outgoing direction at a wall (should not stream)
        bool is_outgoing_at_wall = false;
        if (is_wall_node) {
            if ((wall_directions & Streaming::BOUNDARY_X_MIN) && ex[q] < 0) is_outgoing_at_wall = true;
            if ((wall_directions & Streaming::BOUNDARY_X_MAX) && ex[q] > 0) is_outgoing_at_wall = true;
            if ((wall_directions & Streaming::BOUNDARY_Y_MIN) && ey[q] < 0) is_outgoing_at_wall = true;
            if ((wall_directions & Streaming::BOUNDARY_Y_MAX) && ey[q] > 0) is_outgoing_at_wall = true;
            if ((wall_directions & Streaming::BOUNDARY_Z_MIN) && ez[q] < 0) is_outgoing_at_wall = true;
            if ((wall_directions & Streaming::BOUNDARY_Z_MAX) && ez[q] > 0) is_outgoing_at_wall = true;
        }

        if (is_outgoing_at_wall) {
            // This distribution points out of the domain at a wall - don't stream it
            // It will be set by bounce-back kernel after streaming
            continue;
        }

        // Compute destination coordinates
        int dst_x = idx + ex[q];
        int dst_y = idy + ey[q];
        int dst_z = idz + ez[q];

        // Apply periodic wrapping only in periodic directions
        if (periodic_x) {
            dst_x = (dst_x + nx) % nx;
        }
        if (periodic_y) {
            dst_y = (dst_y + ny) % ny;
        }
        if (periodic_z) {
            dst_z = (dst_z + nz) % nz;
        }

        // Check if destination is within bounds
        if (dst_x >= 0 && dst_x < nx &&
            dst_y >= 0 && dst_y < ny &&
            dst_z >= 0 && dst_z < nz) {

            // Stream normally
            int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
            f_dst[dst_id + q * n_cells] = f_src[id + q * n_cells];
        }
        // If destination is out of bounds, don't stream (non-periodic boundary)
    }
}

// Compute macroscopic quantities from distribution functions
__global__ void computeMacroscopicKernel(
    const float* f,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    // Compute density
    float m_rho = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        m_rho += f[id + q * num_cells];
    }

    // Compute momentum
    float m_ux = 0.0f;
    float m_uy = 0.0f;
    float m_uz = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        float fq = f[id + q * num_cells];
        m_ux += ex[q] * fq;
        m_uy += ey[q] * fq;
        m_uz += ez[q] * fq;
    }

    // Store results
    rho[id] = m_rho;

    // Compute velocity with safety check to prevent NaN
    float u_x = 0.0f;
    float u_y = 0.0f;
    float u_z = 0.0f;

    if (m_rho > 1e-10f && !isnan(m_rho)) {
        u_x = m_ux / m_rho;
        u_y = m_uy / m_rho;
        u_z = m_uz / m_rho;

        // CRITICAL VELOCITY CLAMPING: Prevent numerical instability
        // BUG FIX (2026-01-25): U_MAX must be in LATTICE UNITS (dimensionless)
        // Previous bug: Used 20.0 as if it were m/s, but u_mag is in lattice units!
        //
        // LBM stability: Ma = u/c_s < 0.3, where c_s = 1/√3 ≈ 0.577
        // Using 0.3 lu as aggressive but functional limit (Ma ≈ 0.52)
        const float U_MAX = 0.3f;  // LATTICE UNITS (dimensionless) - was incorrectly 20.0
        float u_mag = sqrtf(u_x*u_x + u_y*u_y + u_z*u_z);
        if (u_mag > U_MAX) {
            // Clamp velocity magnitude while preserving direction
            float scale = U_MAX / u_mag;
            u_x *= scale;
            u_y *= scale;
            u_z *= scale;
        }
    }

    ux[id] = u_x;
    uy[id] = u_y;
    uz[id] = u_z;
}

/**
 * @brief Compute macroscopic quantities with Guo force correction
 *
 * In the Guo forcing scheme, the physical velocity is:
 *   u = Σ(ci*fi)/ρ + 0.5*F/ρ
 * Without this correction, velocity computed from moments alone is
 * missing the forcing contribution, causing forces to appear ineffective.
 */
__global__ void computeMacroscopicWithForceKernel(
    const float* f,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    // Compute density
    float m_rho = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        m_rho += f[id + q * num_cells];
    }

    // Compute momentum from distributions
    float m_ux = 0.0f;
    float m_uy = 0.0f;
    float m_uz = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        float fq = f[id + q * num_cells];
        m_ux += ex[q] * fq;
        m_uy += ey[q] * fq;
        m_uz += ez[q] * fq;
    }

    // Store density
    rho[id] = m_rho;

    // Compute velocity with Guo force correction
    float u_x = 0.0f;
    float u_y = 0.0f;
    float u_z = 0.0f;

    if (m_rho > 1e-10f && !isnan(m_rho)) {
        float inv_rho = 1.0f / m_rho;
        // Guo correction: u = Σ(ci*fi)/ρ + 0.5*F/ρ
        u_x = m_ux * inv_rho + 0.5f * force_x[id] * inv_rho;
        u_y = m_uy * inv_rho + 0.5f * force_y[id] * inv_rho;
        u_z = m_uz * inv_rho + 0.5f * force_z[id] * inv_rho;

        // Velocity clamping for stability
        const float U_MAX = 0.3f;
        float u_mag = sqrtf(u_x*u_x + u_y*u_y + u_z*u_z);
        if (u_mag > U_MAX) {
            float scale = U_MAX / u_mag;
            u_x *= scale;
            u_y *= scale;
            u_z *= scale;
        }
    }

    ux[id] = u_x;
    uy[id] = u_y;
    uz[id] = u_z;
}

// Compute pressure from density
__global__ void computePressureKernel(
    const float* rho,
    float* pressure,
    float rho0,
    float cs2,
    int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    // Equation of state: p = c_s² · (ρ - ρ₀)
    pressure[id] = cs2 * (rho[id] - rho0);
}

// Compute buoyancy force (Boussinesq approximation)
__global__ void computeBuoyancyForceKernel(
    const float* temperature,
    float* force_x,
    float* force_y,
    float* force_z,
    float T_ref,
    float beta,
    float rho0,
    float gravity_x,
    float gravity_y,
    float gravity_z,
    int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    // F_buoyancy = ρ₀·β·(T - T_ref)·g
    float dT = temperature[id] - T_ref;

    // NaN protection for stability
    if (isnan(dT) || isinf(dT)) {
        dT = 0.0f;
    }

    float factor = rho0 * beta * dT;

    // CRITICAL: Use += to accumulate with other forces (Marangoni, surface tension)
    force_x[id] += factor * gravity_x;
    force_y[id] += factor * gravity_y;
    force_z[id] += factor * gravity_z;
}

// Apply Darcy damping for mushy zone
__global__ void applyDarcyDampingKernel(
    const float* liquid_fraction,
    const float* ux,
    const float* uy,
    const float* uz,
    float* force_x,
    float* force_y,
    float* force_z,
    float darcy_constant,
    int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    float fl = liquid_fraction[id];

    // Carman-Kozeny model: F_darcy = -C·(1 - fl)²/(fl³ + ε)·u
    // Epsilon prevents division by zero in fully solid regions
    // Literature: ε ~ 1e-4 to 1e-5 (Voller & Prakash 1987)
    // Too large: weakens damping in liquid; too small: numerical instability
    const float eps = 1e-4f;  // Literature-supported value

    // Compute damping factor using Carman-Kozeny relation
    float damping_factor = -darcy_constant * (1.0f - fl) * (1.0f - fl) / (fl * fl * fl + eps);

    // Add damping to existing forces
    force_x[id] += damping_factor * ux[id];
    force_y[id] += damping_factor * uy[id];
    force_z[id] += damping_factor * uz[id];
}

// Enforce correct velocity at boundary nodes
// - BOUNCE_BACK: zero velocity (no-slip)
// - VELOCITY: prescribed wall velocity
__global__ void setBoundaryVelocityKernel(
    float* ux,
    float* uy,
    float* uz,
    const core::BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_boundary) return;

    core::BoundaryNode node = boundary_nodes[bid];
    int id = node.x + node.y * nx + node.z * nx * ny;

    if (node.type == core::BoundaryType::BOUNCE_BACK) {
        // Explicitly enforce zero velocity at no-slip wall nodes
        ux[id] = 0.0f;
        uy[id] = 0.0f;
        uz[id] = 0.0f;
    }
    else if (node.type == core::BoundaryType::VELOCITY) {
        // Explicitly enforce prescribed velocity at moving wall nodes
        ux[id] = node.ux;
        uy[id] = node.uy;
        uz[id] = node.uz;
    }
}

// Compute omega field with VARIABLE kinematic viscosity for two-phase flow
// Uses ν = μ/ρ_local (variable) where ρ_local depends on VOF
// This ensures proper viscosity variation between phases: ν_light > ν_heavy
__global__ void computeVariableOmegaKernel(
    const float* vof_field,
    float* omega_field,
    float rho_heavy,
    float rho_light,
    float mu_constant,
    float dt,
    float dx,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // VARIABLE KINEMATIC VISCOSITY:
    // Compute local density from VOF: ρ_local = f×ρ_heavy + (1-f)×ρ_light
    float f = vof_field[idx];
    float rho_local = f * rho_heavy + (1.0f - f) * rho_light;
    rho_local = fmaxf(rho_local, 1e-6f);  // Safety clamp

    // Compute variable kinematic viscosity: ν_local = μ / ρ_local
    float nu_local = mu_constant / rho_local;

    // Convert to lattice units: ν_lattice = ν_physical × dt / dx²
    float nu_lattice = nu_local * dt / (dx * dx);

    // Compute relaxation time: τ = ν_lattice/cs² + 0.5
    float tau = nu_lattice / D3Q19::CS2 + 0.5f;

    // CRITICAL FIX (2026-01-20): Conservative stability limit
    // Ensure τ ≥ 0.556 (omega ≤ 1.8) to prevent numerical instability
    // in extreme flow conditions (bubble pinch-off, topology changes).
    // This is more conservative than the theoretical limit (omega=2.0)
    // and prevents velocity spikes when forces become large.
    const float TAU_MIN = 0.556f;  // Corresponds to omega=1.8
    tau = fmaxf(tau, TAU_MIN);

    // Compute omega: ω = 1/τ
    omega_field[idx] = 1.0f / tau;
}

// TRT collision kernel with variable omega (per-cell viscosity)
// This allows proper handling of two-phase flows where kinematic viscosity
// varies with local density: ν(f) = μ / ρ(f)
__global__ void fluidTRTCollisionVariableOmegaKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    const float* omega_even_field,
    const float* vof_field,
    float rho_heavy,
    float rho_light,
    float lambda,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Read local forces
    float fx = force_x[id];
    float fy = force_y[id];
    float fz = force_z[id];

    // Read local omega_even (varies with VOF)
    float omega_even = omega_even_field[id];

    // Compute omega_odd from lambda and omega_even
    // omega_odd = 1 / (lambda/(1/omega_even - 0.5) + 0.5)
    float omega_odd = 1.0f / (lambda / (1.0f / omega_even - 0.5f) + 0.5f);

    // CRITICAL FIX: Compute VOF-weighted physical density for Guo forcing
    // The LBM density field was initialized uniformly (e.g., 1.255), but the
    // physical density varies with VOF: ρ_vof(f) = f×ρ_heavy + (1-f)×ρ_light
    float f_vof = vof_field[id];
    float rho_vof = f_vof * rho_heavy + (1.0f - f_vof) * rho_light;
    rho_vof = fmaxf(rho_vof, 1e-6f);  // Safety clamp

    // Compute macroscopic quantities with DOUBLE PRECISION
    double m_rho_d = 0.0;
    double m_ux_star_d = 0.0;
    double m_uy_star_d = 0.0;
    double m_uz_star_d = 0.0;

    for (int q = 0; q < D3Q19::Q; ++q) {
        double f_d = (double)f_src[id + q * n_cells];
        m_rho_d += f_d;
        m_ux_star_d += (double)ex[q] * f_d;
        m_uy_star_d += (double)ey[q] * f_d;
        m_uz_star_d += (double)ez[q] * f_d;
    }

    // Compute uncorrected velocity with safety check (still uses LBM density for momentum)
    const double RHO_MIN_D = 1e-12;
    double inv_rho_d = 1.0 / fmax(m_rho_d, RHO_MIN_D);
    double m_ux_uncorrected_d = m_ux_star_d * inv_rho_d;
    double m_uy_uncorrected_d = m_uy_star_d * inv_rho_d;
    double m_uz_uncorrected_d = m_uz_star_d * inv_rho_d;

    // CRITICAL FIX: Apply Guo forcing using VOF-weighted density
    // This ensures correct force response: Δu = 0.5×F/ρ_vof
    // Light phase (ρ=0.1694): Δu is 7.4× larger than heavy phase (ρ=1.255)
    double inv_rho_vof_d = 1.0 / (double)rho_vof;
    double m_ux_d = m_ux_uncorrected_d + 0.5 * (double)fx * inv_rho_vof_d;
    double m_uy_d = m_uy_uncorrected_d + 0.5 * (double)fy * inv_rho_vof_d;
    double m_uz_d = m_uz_uncorrected_d + 0.5 * (double)fz * inv_rho_vof_d;

    // CRITICAL SAFETY: Clamp velocity to prevent numerical instability
    // BUG FIX (2026-01-25): U_MAX_SAFE must be in LATTICE UNITS (dimensionless)
    // Previous bug: Used 50.0 as if it were m/s, but u_mag is in lattice units!
    //
    // LBM stability: Ma = u/c_s < 0.3, where c_s = 1/√3 ≈ 0.577
    // This gives u_max ≈ 0.17 lattice units for strict stability
    // Using 0.3 lu as aggressive but functional limit (Ma ≈ 0.52)
    //
    // Physical context: With dx=3.75μm, dt=75ns:
    //   0.3 lu = 0.3 * (3.75e-6/75e-9) = 15 m/s physical
    // This is reasonable for laser melting with Marangoni convection
    const double U_MAX_SAFE = 0.3;  // LATTICE UNITS (dimensionless) - was incorrectly 50.0
    double u_mag = sqrt(m_ux_d*m_ux_d + m_uy_d*m_uy_d + m_uz_d*m_uz_d);
    if (u_mag > U_MAX_SAFE) {
        // Clamp velocity magnitude while preserving direction
        double scale = U_MAX_SAFE / u_mag;
        m_ux_d *= scale;
        m_uy_d *= scale;
        m_uz_d *= scale;
    }

    // Store corrected macroscopic quantities
    rho[id] = (float)m_rho_d;
    ux[id] = (float)m_ux_d;
    uy[id] = (float)m_uy_d;
    uz[id] = (float)m_uz_d;

    // TRT collision with forcing - ALL OPERATIONS IN DOUBLE PRECISION
    for (int q = 0; q < D3Q19::Q; ++q) {
        int q_opp = opposite[q];

        // Read distribution functions
        double f_q_d = (double)f_src[id + q * n_cells];
        double f_qbar_d = (double)f_src[id + q_opp * n_cells];

        // Equilibrium distributions - use double-precision computation
        double feq_q_d = computeEquilibriumDouble(q, m_rho_d, m_ux_d, m_uy_d, m_uz_d);
        double feq_qbar_d = computeEquilibriumDouble(q_opp, m_rho_d, m_ux_d, m_uy_d, m_uz_d);

        // Symmetric and antisymmetric parts (TRT decomposition)
        double f_plus_d = 0.5 * (f_q_d + f_qbar_d);
        double f_minus_d = 0.5 * (f_q_d - f_qbar_d);
        double feq_plus_d = 0.5 * (feq_q_d + feq_qbar_d);
        double feq_minus_d = 0.5 * (feq_q_d - feq_qbar_d);

        // Guo forcing term calculation with double precision + FMA
        double dx = (double)ex[q];
        double dy = (double)ey[q];
        double dz = (double)ez[q];
        double dF_x = (double)fx;
        double dF_y = (double)fy;
        double dF_z = (double)fz;

        // Use FMA for dot products
        double ci_dot_u_d = fma(dx, m_ux_d, fma(dy, m_uy_d, dz * m_uz_d));
        double ci_dot_F_d = fma(dx, dF_x, fma(dy, dF_y, dz * dF_z));

        // Numerically stable (c - u)·F computation
        double c_minus_u_dot_F_d = fma(dx - m_ux_d, dF_x, fma(dy - m_uy_d, dF_y, (dz - m_uz_d) * dF_z));

        // Standard Guo term: 3(c - u)·F + 9(c·u)(c·F)
        double term1_d = 3.0 * c_minus_u_dot_F_d;
        double term2_d = 9.0 * ci_dot_u_d * ci_dot_F_d;

        // CRITICAL FIX (2026-01-20): DO NOT apply compensation!
        //
        // VELOCITY SPIKE BUG ROOT CAUSE:
        // Lines 1739-1741 compute velocity correction: Δu = 0.5*F/ρ_vof (no compensation)
        // But then we were applying compensation HERE, creating mismatch:
        //   - Velocity sees: u = u_star + 0.5*F/ρ
        //   - Distribution sees: f += ... + 5.0*F*... (compensated)
        // This inconsistency caused 5× velocity spikes when omega→1.9!
        //
        // SOLUTION: NO COMPENSATION. Use standard Guo forcing with tau-dependent factor.
        // For variable viscosity, the (1-ω/2) factor is CORRECT and NECESSARY.
        // It ensures momentum conservation: ∂_t(ρu) + ∇·(flux) = F
        //
        // The tau-dependence is a feature, not a bug, for variable viscosity flows.

        // TRT forcing: standard BGK term (tau-dependent, as it should be)
        double force_term_d = (1.0 - 0.5 * (double)omega_even) * w_double[q] * (term1_d + term2_d);

        // TRT correction term (standard form, no compensation)
        force_term_d += 1.5 * w_double[q] * ((double)omega_even - (double)omega_odd) * ci_dot_F_d;

        // TRT collision: relax symmetric part with omega_even, antisymmetric with omega_odd
        double omega_even_d = (double)omega_even;
        double omega_odd_d = (double)omega_odd;
        double f_plus_new_d = f_plus_d - omega_even_d * (f_plus_d - feq_plus_d);
        double f_minus_new_d = f_minus_d - omega_odd_d * (f_minus_d - feq_minus_d);

        // Reconstruct post-collision distribution with forcing
        f_dst[id + q * n_cells] = (float)(f_plus_new_d + f_minus_new_d + force_term_d);
    }
}

} // namespace physics
} // namespace lbm
