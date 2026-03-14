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
     * @brief Perform BGK collision with spatially-varying forces (Guo scheme)
     * @param force_x Device array of force x-component [lattice units]
     * @param force_y Device array of force y-component [lattice units]
     * @param force_z Device array of force z-component [lattice units]
     */
    void collisionBGK(const float* force_x,
                     const float* force_y,
                     const float* force_z);

    /**
     * @brief Perform BGK collision with EDM (Exact Difference Method) forcing
     *
     * Replaces Guo source term with equilibrium shift:
     *   f_new = f - ω(f - f_eq(ρ, u_bare)) + [f_eq(ρ, u_bare+Δu) - f_eq(ρ, u_bare)]
     *
     * where u_bare = m/(ρ+K/2) (semi-implicit Darcy) and Δu = F/ρ.
     * The EDM shift lives in equilibrium subspace — no distribution anisotropy
     * accumulation, eliminating Guo/Darcy velocity shocks.
     *
     * Reference: Kupershtokh et al., Comput. Math. Appl. 58:862-872 (2009)
     *
     * @param force_x Device array of non-Darcy force x-component [lattice units]
     * @param force_y Device array of non-Darcy force y-component [lattice units]
     * @param force_z Device array of non-Darcy force z-component [lattice units]
     * @param darcy_coeff Device array of Darcy K per cell [lattice units]
     */
    void collisionBGKwithEDM(const float* force_x,
                              const float* force_y,
                              const float* force_z,
                              const float* darcy_coeff);

    /**
     * @brief Enable TRT mode for subsequent collisionBGKwithEDM calls
     *
     * Computes omega_minus from the magic parameter Λ:
     *   tau_minus = 0.5 + Λ / (tau - 0.5)
     *   omega_minus = 1 / tau_minus
     *
     * With Λ = 3/16 (default), checkerboard instability is suppressed at low
     * tau while the physical (even) relaxation rate is unchanged. Call once
     * after construction; collisionBGKwithEDM will automatically dispatch to
     * fluidTRTCollisionEDMKernel when omega_minus > 0.
     *
     * To revert to BGK+EDM, call setTRT(0): omega_minus_ is set to 0.
     *
     * @param magic_parameter Λ = (τ+ - 0.5)(τ- - 0.5), default 3/16
     */
    void setTRT(float magic_parameter = 3.0f / 16.0f);

    /**
     * @brief Perform TRT collision step with uniform force
     *
     * Two-Relaxation-Time (TRT) collision operator improves accuracy
     * by separately relaxing symmetric and antisymmetric distribution
     * parts with different rates. Uses magic parameter Λ for optimal
     * wall accuracy (default: Λ = 3/16).
     *
     * Reference: Ginzburg, I., et al. (2008). Two-relaxation-time
     * Lattice Boltzmann scheme: About parametrization, velocity,
     * pressure and mixed boundary conditions. Commun. Comput. Phys.
     *
     * @param force_x Body force x-component [m/s²]
     * @param force_y Body force y-component [m/s²]
     * @param force_z Body force z-component [m/s²]
     * @param lambda Magic parameter (default: 3/16 for optimal walls)
     */
    void collisionTRT(float force_x = 0.0f,
                     float force_y = 0.0f,
                     float force_z = 0.0f,
                     float lambda = 3.0f / 16.0f);

    /**
     * @brief Perform TRT collision with spatially-varying forces
     * @param force_x Device array of force x-component [m/s²]
     * @param force_y Device array of force y-component [m/s²]
     * @param force_z Device array of force z-component [m/s²]
     * @param lambda Magic parameter (default: 3/16 for optimal walls)
     */
    void collisionTRT(const float* force_x,
                     const float* force_y,
                     const float* force_z,
                     float lambda = 3.0f / 16.0f);

    /**
     * @brief Set UNIFORM kinematic viscosity (constant ν for all cells)
     *
     * Standard for RT instability benchmarks. Both phases have same ν,
     * giving symmetric viscous damping for bubble and spike.
     *
     * @param nu_constant Kinematic viscosity [m²/s] (same for both phases)
     */
    void computeUniformViscosity(float nu_constant);

    /**
     * @brief Compute variable viscosity field from VOF for two-phase flow
     *
     * WARNING: This gives asymmetric damping! Light phase has much higher ν.
     * For symmetric RT benchmarks, use computeUniformViscosity() instead.
     *
     * For constant dynamic viscosity μ but variable density ρ(f):
     * - ν(f) = μ / ρ(f) where ρ(f) = f×ρ_heavy + (1-f)×ρ_light
     * - τ(f) = ν_lattice(f)/cs² + 0.5
     * - ω(f) = 1/τ(f)
     *
     * @param vof_field VOF fill level field (0=light, 1=heavy)
     * @param rho_heavy Heavy phase density [kg/m³]
     * @param rho_light Light phase density [kg/m³]
     * @param mu_constant Dynamic viscosity [Pa·s] (constant for both phases)
     */
    void computeVariableViscosity(const float* vof_field,
                                  float rho_heavy,
                                  float rho_light,
                                  float mu_constant);

    /**
     * @brief Compute variable viscosity with per-phase dynamic viscosity
     *
     * For two-phase flows with DIFFERENT dynamic viscosities:
     * - μ(f) = f×μ_heavy + (1-f)×μ_light  (arithmetic interpolation)
     * - ν(f) = μ(f) / ρ(f)
     * - τ(f) = ν_lattice(f)/cs² + 0.5
     * - ω(f) = 1/τ(f)
     *
     * @param vof_field VOF fill level field (0=light, 1=heavy)
     * @param rho_heavy Heavy phase density [kg/m³]
     * @param rho_light Light phase density [kg/m³]
     * @param mu_heavy Dynamic viscosity of heavy phase [Pa·s]
     * @param mu_light Dynamic viscosity of light phase [Pa·s]
     */
    void computeVariableViscosity(const float* vof_field,
                                  float rho_heavy,
                                  float rho_light,
                                  float mu_heavy,
                                  float mu_light);

    /**
     * @brief Perform TRT collision with variable viscosity (variable omega field)
     *
     * Uses per-cell relaxation parameter omega[i] computed by computeVariableViscosity().
     * Essential for two-phase flows where kinematic viscosity varies with local density:
     * ν(f) = μ / ρ(f).
     *
     * CRITICAL: For correct dynamics, Guo forcing must use VOF-weighted physical density
     * ρ_vof(f) = f×ρ_heavy + (1-f)×ρ_light, not the LBM density field.
     *
     * @param force_x Device array of force x-component [m/s²]
     * @param force_y Device array of force y-component [m/s²]
     * @param force_z Device array of force z-component [m/s²]
     * @param vof_field VOF fill level field (0=light, 1=heavy)
     * @param rho_heavy Heavy phase density [kg/m³]
     * @param rho_light Light phase density [kg/m³]
     * @param lambda Magic parameter (default: 3/16 for optimal walls)
     */
    void collisionTRTVariable(const float* force_x,
                             const float* force_y,
                             const float* force_z,
                             const float* vof_field,
                             float rho_heavy,
                             float rho_light,
                             float lambda = 3.0f / 16.0f);

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
     * @brief Compute macroscopic quantities with Guo force correction
     *
     * In the Guo forcing scheme, the physical velocity is:
     *   u = Σ(ci*fi)/ρ + 0.5*F/ρ
     * This overload applies the force correction to get the true velocity.
     *
     * @param force_x Device array of x-force (lattice units)
     * @param force_y Device array of y-force (lattice units)
     * @param force_z Device array of z-force (lattice units)
     */
    void computeMacroscopic(const float* force_x, const float* force_y, const float* force_z);

    /**
     * @brief Compute macroscopic quantities with semi-implicit Darcy damping
     *
     * Semi-implicit treatment of Darcy drag in the Guo velocity definition:
     *   u = [Σ(ci·fi) + 0.5·F_other] / (ρ + 0.5·K_darcy)
     *
     * When K_darcy → ∞ (solid), velocity → 0 smoothly. No NaN or oscillation.
     * When K_darcy = 0 (liquid), reduces to standard Guo correction.
     *
     * Reference: Voller & Prakash (1987), Brent et al. (1988)
     *
     * @param force_x Device array of x-force (lattice units, excluding Darcy)
     * @param force_y Device array of y-force (lattice units, excluding Darcy)
     * @param force_z Device array of z-force (lattice units, excluding Darcy)
     * @param darcy_coeff Device array of Darcy coefficient K per cell (lattice units)
     *                    K = C·(1-fl)²/(fl³+ε)·ρ_phys·dt, or 0 for liquid cells
     */
    void computeMacroscopic(const float* force_x, const float* force_y,
                            const float* force_z, const float* darcy_coeff);

    /**
     * @brief Compute macroscopic quantities for EDM scheme with semi-implicit Darcy
     *
     * In EDM, forcing is handled by the equilibrium shift in collision, not by
     * the +0.5*F/ρ Guo correction in macroscopic computation. The output
     * velocity includes F/(2ρ) for second-order accurate physical velocity.
     *
     * @param force_x Device array of non-Darcy force [lattice units]
     * @param force_y Device array of non-Darcy force [lattice units]
     * @param force_z Device array of non-Darcy force [lattice units]
     * @param darcy_coeff Device array of Darcy K per cell [lattice units]
     */
    void computeMacroscopicEDM(const float* force_x, const float* force_y,
                                const float* force_z, const float* darcy_coeff);

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
     * @brief Get distribution function arrays (device pointers)
     * Needed for custom boundary conditions (e.g., free-surface stress BC).
     */
    float* getDistributionSrc() { return d_f_src; }
    float* getDistributionDst() { return d_f_dst; }

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

    /**
     * @brief Set moving wall boundary condition
     *
     * Replaces existing wall boundaries with moving wall (velocity) boundaries.
     * Uses Zou-He velocity boundary condition to enforce wall velocity.
     *
     * Implementation note: This modifies the boundary node list to change
     * wall boundaries (BOUNCE_BACK) to velocity boundaries (VELOCITY) with
     * prescribed wall motion.
     *
     * Reference: Zou, Q., & He, X. (1997). On pressure and velocity boundary
     * conditions for the lattice Boltzmann BGK model. Physics of Fluids, 9(6), 1591-1598.
     *
     * @param wall_direction Direction of the wall (use Streaming::BOUNDARY_* flags)
     *                      Example: Streaming::BOUNDARY_Y_MAX for top wall (y=ny-1)
     * @param ux_wall Wall velocity x-component [m/s]
     * @param uy_wall Wall velocity y-component [m/s]
     * @param uz_wall Wall velocity z-component [m/s]
     *
     * Example usage for lid-driven cavity (moving top wall):
     *   fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, 0.1, 0.0, 0.0);
     */
    void setMovingWall(unsigned int wall_direction,
                      float ux_wall,
                      float uy_wall,
                      float uz_wall);

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
    float omega_minus_;     ///< TRT anti-symmetric relaxation (0 = BGK mode)

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

    // Variable viscosity support for two-phase flow
    float* d_omega_field_;  ///< Per-cell relaxation parameter for variable viscosity

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
 * @brief CUDA kernel for fluid TRT collision with uniform body force
 */
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
    float omega_e,
    float omega_o,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for fluid TRT collision with spatially-varying forces
 */
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
    float omega_e,
    float omega_o,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for computing variable omega field from VOF
 */
__global__ void computeVariableOmegaKernel(
    const float* vof_field,
    float* omega_field,
    float rho_heavy,
    float rho_light,
    float mu_constant,
    float dt,
    float dx,
    int num_cells);

/**
 * @brief CUDA kernel for TRT collision with variable omega (per-cell viscosity)
 */
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
 * @brief CUDA kernel for computing macroscopic quantities with Guo force correction
 * u = Σ(ci*fi)/ρ + 0.5*F/ρ
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
    int num_cells);

/**
 * @brief CUDA kernel for semi-implicit Darcy treatment in macroscopic velocity
 *
 * u = [Σ(ci·fi) + 0.5·F_other] / (ρ + 0.5·K_darcy)
 *
 * This avoids the catastrophic explicit Darcy force that causes NaN when
 * K → ∞ in solid regions. Instead, velocity smoothly → 0.
 */
__global__ void computeMacroscopicSemiImplicitDarcyKernel(
    const float* f,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    const float* darcy_coeff,
    int num_cells);

/**
 * @brief CUDA kernel for BGK collision with EDM (Exact Difference Method)
 *
 * f_new = f - ω(f - f_eq(ρ, u_bare)) + [f_eq(ρ, u_bare+Δu) - f_eq(ρ, u_bare)]
 * where u_bare = m/(ρ+K/2), Δu = F/ρ
 */
__global__ void fluidBGKCollisionEDMKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    const float* darcy_coeff,
    float omega,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for TRT collision with EDM (Exact Difference Method) forcing
 *
 * TRT extension of BGK+EDM: symmetric non-equilibrium relaxed with omega+,
 * anti-symmetric non-equilibrium relaxed with omega_minus. EDM shift added
 * identically to BGK+EDM (lives in equilibrium subspace, unaffected by split).
 *
 * Eliminates checkerboard instability at low tau without touching the physical
 * (even) relaxation rate that controls viscosity.
 *
 * @param omega     ω+ — even/symmetric relaxation (= 1/tau, controls viscosity)
 * @param omega_minus ω- — odd/anti-symmetric relaxation (from magic parameter Λ)
 */
__global__ void fluidTRTCollisionEDMKernel(
    const float* f_src,
    float* f_dst,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    const float* darcy_coeff,
    float omega,
    float omega_minus,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for semi-implicit Darcy macroscopic with EDM scheme
 *
 * u_bare = m/(ρ+K/2), u_phys = u_bare + F/(2ρ) for output only.
 * No Guo +0.5*F correction — EDM handles forcing in collision.
 */
__global__ void computeMacroscopicSemiImplicitDarcyEDMKernel(
    const float* f,
    float* rho,
    float* ux,
    float* uy,
    float* uz,
    const float* force_x,
    const float* force_y,
    const float* force_z,
    const float* darcy_coeff,
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

/**
 * @brief CUDA kernel for compensating forces to make them τ-independent
 *
 * Guo forcing includes factor (1-ω/2) that couples force magnitude to relaxation
 * time τ. This kernel pre-compensates forces by multiplying by 1/(1-ω/2) = 2/(2-ω)
 * to cancel this dependency, ensuring forces have the same physical effect in both
 * light and heavy phases regardless of local viscosity.
 *
 * @param force_x Force x-component (modified in place)
 * @param force_y Force y-component (modified in place)
 * @param force_z Force z-component (modified in place)
 * @param omega_field Per-cell relaxation parameter
 * @param num_cells Number of cells
 */
__global__ void compensateForceForOmegaKernel(
    float* force_x,
    float* force_y,
    float* force_z,
    const float* omega_field,
    int num_cells);

} // namespace physics
} // namespace lbm
