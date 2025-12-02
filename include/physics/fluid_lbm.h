/**
 * @file fluid_lbm.h
 * @brief Fluid Lattice Boltzmann Method solver for incompressible Navier-Stokes
 *
 * This file implements the fluid flow solver using the D3Q19 lattice for
 * simulating incompressible fluid dynamics with:
 * - BGK collision operator with adjustable viscosity
 * - Body force implementation (Guo forcing scheme)
 * - Buoyancy-driven flow (Boussinesq approximation)
 * - Coupling with thermal solver for convective heat transfer
 * - Phase-dependent flow properties (solid/mushy/liquid)
 *
 * Physical equations solved:
 * - Incompressible Navier-Stokes: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + F
 * - Buoyancy force: F_buoyancy = ρ₀·β·(T - T_ref)·g
 * - Continuity: ∇·u = 0
 *
 * Reference:
 * - Guo, Z., Zheng, C., & Shi, B. (2002). Discrete lattice effects on the
 *   forcing term in the lattice Boltzmann method. Physical Review E, 65(4), 046308.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/lattice_d3q19.h"
#include "core/boundary_conditions.h"
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

/**
 * @brief Boundary type enumeration for FluidLBM
 */
enum class BoundaryType {
    PERIODIC = 0,  ///< Periodic boundary (no action needed)
    WALL = 1       ///< No-slip wall (bounce-back)
};

/**
 * @brief Fluid LBM solver using D3Q19 lattice
 *
 * This class implements the incompressible Navier-Stokes solver using LBM.
 * It supports:
 * - Variable viscosity (through relaxation time)
 * - Body forces (gravity, buoyancy, Darcy damping)
 * - Coupling with thermal solver
 * - Phase-dependent properties
 * - Configurable boundary conditions (periodic or no-slip walls)
 */
class FluidLBM {
public:
    /**
     * @brief Constructor with boundary configuration
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param kinematic_viscosity Kinematic viscosity ν [m²/s]
     * @param density Reference density ρ₀ [kg/m³]
     * @param boundary_x Boundary type in x-direction (default: periodic)
     * @param boundary_y Boundary type in y-direction (default: periodic)
     * @param boundary_z Boundary type in z-direction (default: periodic)
     * @param dt Time step [s] (required for lattice unit conversion)
     * @param dx Lattice spacing [m] (required for lattice unit conversion)
     */
    FluidLBM(int nx, int ny, int nz,
             float kinematic_viscosity,
             float density = 1.0f,
             BoundaryType boundary_x = BoundaryType::PERIODIC,
             BoundaryType boundary_y = BoundaryType::PERIODIC,
             BoundaryType boundary_z = BoundaryType::PERIODIC,
             float dt = 1.0e-7f,
             float dx = 2.0e-6f);

    /**
     * @brief Destructor
     */
    ~FluidLBM();

    /**
     * @brief Initialize flow field
     * @param initial_density Uniform initial density
     * @param initial_ux Initial velocity x-component
     * @param initial_uy Initial velocity y-component
     * @param initial_uz Initial velocity z-component
     */
    void initialize(float initial_density = 1.0f,
                   float initial_ux = 0.0f,
                   float initial_uy = 0.0f,
                   float initial_uz = 0.0f);

    /**
     * @brief Initialize flow field with custom velocity distribution
     * @param density Device array of density values
     * @param ux Device array of velocity x-component
     * @param uy Device array of velocity y-component
     * @param uz Device array of velocity z-component
     */
    void initialize(const float* density,
                   const float* ux,
                   const float* uy,
                   const float* uz);

    /**
     * @brief Perform BGK collision step
     * @param force_x Body force x-component [m/s²]
     * @param force_y Body force y-component [m/s²]
     * @param force_z Body force z-component [m/s²]
     */
    void collisionBGK(float force_x = 0.0f,
                     float force_y = 0.0f,
                     float force_z = 0.0f);

    /**
     * @brief Perform BGK collision with spatially-varying forces
     * @param force_x Device array of force x-component [m/s²]
     * @param force_y Device array of force y-component [m/s²]
     * @param force_z Device array of force z-component [m/s²]
     */
    void collisionBGK(const float* force_x,
                     const float* force_y,
                     const float* force_z);

    /**
     * @brief Perform streaming step
     */
    void streaming();

    /**
     * @brief Apply boundary conditions
     * @param boundary_type Type of boundary (0=periodic, 1=no-slip, 2=free-slip)
     */
    void applyBoundaryConditions(int boundary_type);

    /**
     * @brief Compute macroscopic quantities (density, velocity, pressure)
     */
    void computeMacroscopic();

    /**
     * @brief Compute buoyancy force using Boussinesq approximation
     * @param temperature Device array of temperature field [K]
     * @param T_ref Reference temperature [K]
     * @param beta Thermal expansion coefficient [1/K]
     * @param gravity_x Gravity vector x-component [m/s²]
     * @param gravity_y Gravity vector y-component [m/s²]
     * @param gravity_z Gravity vector z-component [m/s²]
     * @param force_x Output: buoyancy force x-component (device array)
     * @param force_y Output: buoyancy force y-component (device array)
     * @param force_z Output: buoyancy force z-component (device array)
     */
    void computeBuoyancyForce(const float* temperature,
                             float T_ref,
                             float beta,
                             float gravity_x,
                             float gravity_y,
                             float gravity_z,
                             float* force_x,
                             float* force_y,
                             float* force_z) const;

    /**
     * @brief Compute Darcy damping force for mushy zone
     * @param liquid_fraction Device array of liquid fraction (0-1)
     * @param darcy_constant Darcy constant C [kg/(m³·s)]
     * @param force_x Input/Output: force x-component (modified in place)
     * @param force_y Input/Output: force y-component (modified in place)
     * @param force_z Input/Output: force z-component (modified in place)
     */
    void applyDarcyDamping(const float* liquid_fraction,
                          float darcy_constant,
                          float* force_x,
                          float* force_y,
                          float* force_z) const;

    /**
     * @brief Get density field (device pointer)
     * @return Device pointer to density array
     */
    float* getDensity() { return d_rho; }
    const float* getDensity() const { return d_rho; }

    /**
     * @brief Get velocity field (device pointers)
     */
    float* getVelocityX() { return d_ux; }
    const float* getVelocityX() const { return d_ux; }

    float* getVelocityY() { return d_uy; }
    const float* getVelocityY() const { return d_uy; }

    float* getVelocityZ() { return d_uz; }
    const float* getVelocityZ() const { return d_uz; }

    /**
     * @brief Get pressure field (device pointer)
     * @return Device pointer to pressure array [Pa]
     */
    float* getPressure() { return d_pressure; }
    const float* getPressure() const { return d_pressure; }

    /**
     * @brief Copy velocity to host
     * @param host_ux Host array for velocity x-component (must be pre-allocated)
     * @param host_uy Host array for velocity y-component (must be pre-allocated)
     * @param host_uz Host array for velocity z-component (must be pre-allocated)
     */
    void copyVelocityToHost(float* host_ux, float* host_uy, float* host_uz) const;

    /**
     * @brief Copy density to host
     * @param host_rho Host array (must be pre-allocated)
     */
    void copyDensityToHost(float* host_rho) const;

    /**
     * @brief Copy pressure to host
     * @param host_pressure Host array (must be pre-allocated)
     */
    void copyPressureToHost(float* host_pressure) const;

    /**
     * @brief Get domain dimensions
     */
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

    /**
     * @brief Get flow parameters
     */
    float getOmega() const { return omega_; }
    float getTau() const { return tau_; }
    float getViscosity() const { return nu_physical_; }
    float getReferenceDensity() const { return rho0_; }

    /**
     * @brief Compute Reynolds number
     * @param characteristic_velocity Characteristic velocity [m/s]
     * @param characteristic_length Characteristic length [m]
     * @return Reynolds number
     */
    float computeReynoldsNumber(float characteristic_velocity,
                               float characteristic_length) const;

private:
    // Domain dimensions
    int nx_, ny_, nz_;
    int num_cells_;

    // Lattice parameters
    float dt_;              ///< Time step [s]
    float dx_;              ///< Lattice spacing [m]

    // Flow parameters
    float nu_physical_;     ///< Physical kinematic viscosity [m²/s]
    float nu_lattice_;      ///< Lattice kinematic viscosity [dimensionless]
    float rho0_;            ///< Reference density [kg/m³]
    float omega_;           ///< BGK relaxation parameter (1/tau)
    float tau_;             ///< BGK relaxation time

    // Boundary configuration
    BoundaryType boundary_x_;  ///< Boundary type in x-direction
    BoundaryType boundary_y_;  ///< Boundary type in y-direction
    BoundaryType boundary_z_;  ///< Boundary type in z-direction

    // Device memory for distribution functions (SoA layout)
    float* d_f_src;     ///< Source distribution (before streaming)
    float* d_f_dst;     ///< Destination distribution (after streaming)

    // Device memory for macroscopic quantities
    float* d_rho;       ///< Density field
    float* d_ux;        ///< Velocity field x-component
    float* d_uy;        ///< Velocity field y-component
    float* d_uz;        ///< Velocity field z-component
    float* d_pressure;  ///< Pressure field [Pa]

    // Boundary node management
    core::BoundaryNode* d_boundary_nodes_;  ///< Device array of boundary nodes
    int n_boundary_nodes_;                   ///< Number of boundary nodes

    // Utility functions
    void allocateMemory();
    void freeMemory();
    void swapDistributions();
    void initializeBoundaryNodes();  ///< Initialize boundary node list
};

// CUDA kernels for fluid LBM

/**
 * @brief CUDA kernel for fluid BGK collision with uniform body force
 */
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
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for fluid BGK collision with spatially-varying forces
 */
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
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for streaming distribution functions (periodic boundaries)
 */
__global__ void fluidStreamingKernel(
    const float* f_src,
    float* f_dst,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for streaming with mixed boundary conditions
 */
__global__ void fluidStreamingKernelWithWalls(
    const float* f_src,
    float* f_dst,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z);

/**
 * @brief CUDA kernel for computing macroscopic quantities
 */
__global__ void computeMacroscopicKernel(
    const float* f,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    int num_cells);

/**
 * @brief CUDA kernel for computing pressure from density
 * p = c_s² · (ρ - ρ₀)
 */
__global__ void computePressureKernel(
    const float* rho,
    float* pressure,
    float rho0,
    float cs2,
    int num_cells);

/**
 * @brief CUDA kernel for computing buoyancy force (Boussinesq approximation)
 * F = ρ₀·β·(T - T_ref)·g
 */
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
    int num_cells);

/**
 * @brief CUDA kernel for applying Darcy damping in mushy zone
 * F_darcy = -C·(1 - fl)²/(fl³ + ε)·u
 */
__global__ void applyDarcyDampingKernel(
    const float* liquid_fraction,
    const float* ux,
    const float* uy,
    const float* uz,
    float* force_x,
    float* force_y,
    float* force_z,
    float darcy_constant,
    int num_cells);

} // namespace physics
} // namespace lbm
