/**
 * @file vof_solver.h
 * @brief Volume of Fluid (VOF) solver for free surface tracking
 *
 * This file implements the VOF method for tracking free surfaces and interface
 * deformation in multiphase flows. The implementation follows the approach used
 * in walberla (Koerner et al. 2005, Thuerey 2007) with LBM-based advection.
 *
 * Key features:
 * - Fill level field (0 = gas, 1 = liquid)
 * - Cell flag system (LIQUID, GAS, INTERFACE, OBSTACLE)
 * - Interface reconstruction (PLIC or height function)
 * - Curvature computation for surface tension
 * - Mass-conservative advection
 *
 * Physical model:
 * - Advection: ∂f/∂t + ∇·(f·u) = 0
 * - Interface: reconstructed from fill level gradient
 * - Curvature: κ = ∇·n where n is interface normal
 *
 * References:
 * - Koerner, C., Thies, M., Hofmann, T., Thuerey, N., & Rude, U. (2005).
 *   Lattice Boltzmann model for free surface flow for modeling foaming.
 *   Journal of Statistical Physics, 121(1), 179-196.
 * - Thuerey, N. (2007). A single-phase free-surface lattice Boltzmann method.
 *   Ph.D. thesis, University of Erlangen-Nuremberg.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lbm {
namespace physics {

/**
 * @brief Cell flag types for VOF solver
 */
enum class CellFlag : uint8_t {
    GAS = 0,        ///< Pure gas cell (f = 0)
    LIQUID = 1,     ///< Pure liquid cell (f = 1)
    INTERFACE = 2,  ///< Interface cell (0 < f < 1)
    OBSTACLE = 3    ///< Solid obstacle cell
};

/**
 * @brief VOF advection scheme selection
 */
enum class VOFAdvectionScheme : uint8_t {
    UPWIND = 0,     ///< First-order upwind (most diffusive, most stable)
    TVD = 1         ///< TVD with flux limiter (2nd-order in smooth regions)
};

/**
 * @brief TVD flux limiter types
 */
enum class TVDLimiter : uint8_t {
    MINMOD = 0,     ///< Most diffusive, most stable
    VAN_LEER = 1,   ///< Balanced accuracy and stability (recommended)
    SUPERBEE = 2,   ///< Least diffusive, most compressive
    MC = 3          ///< Monotonized Central, good for smooth flows
};

/**
 * @brief VOF solver for free surface tracking
 *
 * This class implements the Volume of Fluid method for tracking interfaces
 * between liquid and gas phases. The solver uses:
 * - Fill level field f (0-1) representing liquid volume fraction
 * - Cell flags to distinguish different regions
 * - Geometric interface reconstruction
 * - Curvature computation for surface tension
 *
 * The solver is designed to integrate with FluidLBM for coupled flow simulation.
 */
class VOFSolver {
public:
    /**
     * @brief Boundary type enumeration for VOF solver
     */
    enum class BoundaryType {
        PERIODIC = 0,  // Periodic (wrapping)
        WALL = 1       // Wall (zero-flux / no-penetration)
    };

    /**
     * @brief Constructor with boundary configuration
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param dx Lattice spacing [m]
     * @param bc_x Boundary condition in x-direction (default: PERIODIC)
     * @param bc_y Boundary condition in y-direction (default: PERIODIC)
     * @param bc_z Boundary condition in z-direction (default: PERIODIC)
     */
    VOFSolver(int nx, int ny, int nz, float dx = 1.0f,
              BoundaryType bc_x = BoundaryType::PERIODIC,
              BoundaryType bc_y = BoundaryType::PERIODIC,
              BoundaryType bc_z = BoundaryType::PERIODIC);

    /**
     * @brief Destructor
     */
    ~VOFSolver();

    /**
     * @brief Initialize fill level field
     * @param fill_level Host array of initial fill level values (size nx*ny*nz)
     *                   Use 1.0 for liquid region, 0.0 for gas region
     */
    void initialize(const float* fill_level);

    /**
     * @brief Initialize with uniform fill level
     * @param uniform_fill Uniform fill level value (0.0 to 1.0)
     */
    void initialize(float uniform_fill = 1.0f);

    /**
     * @brief Initialize with spherical droplet
     * @param center_x Droplet center x-coordinate
     * @param center_y Droplet center y-coordinate
     * @param center_z Droplet center z-coordinate
     * @param radius Droplet radius [lattice units]
     */
    void initializeDroplet(float center_x, float center_y, float center_z, float radius);

    /**
     * @brief Advect fill level field using velocity field
     * @param velocity_x Device array of velocity x-component [m/s]
     * @param velocity_y Device array of velocity y-component [m/s]
     * @param velocity_z Device array of velocity z-component [m/s]
     * @param dt Time step [s]
     * @note Uses first-order upwind/donor-cell scheme for stability
     */
    void advectFillLevel(const float* velocity_x,
                         const float* velocity_y,
                         const float* velocity_z,
                         float dt);

    /**
     * @brief Reconstruct interface from fill level field
     * @note Computes interface normal vectors from fill level gradients
     */
    void reconstructInterface();

    /**
     * @brief Compute interface curvature
     * @note Uses height function method or finite difference on normals
     */
    void computeCurvature();

    /**
     * @brief Convert cells between interface, liquid, and gas
     * @note Updates cell flags based on fill level:
     *       f = 0 → GAS, f = 1 → LIQUID, 0 < f < 1 → INTERFACE
     */
    void convertCells();

    /**
     * @brief Apply boundary conditions
     * @param boundary_type Type of boundary (0=periodic, 1=wall with contact angle)
     * @param contact_angle Contact angle for wall boundaries [degrees]
     */
    void applyBoundaryConditions(int boundary_type, float contact_angle = 90.0f);

    /**
     * @brief Get fill level field (device pointer)
     * @return Device pointer to fill level array (0-1)
     */
    float* getFillLevel() { return d_fill_level_; }
    const float* getFillLevel() const { return d_fill_level_; }

    /**
     * @brief Get cell flags (device pointer)
     * @return Device pointer to cell flag array
     */
    uint8_t* getCellFlags() { return d_cell_flags_; }
    const uint8_t* getCellFlags() const { return d_cell_flags_; }

    /**
     * @brief Get interface normal vectors (device pointer)
     * @return Device pointer to interface normal array (float3)
     */
    float3* getInterfaceNormals() { return d_interface_normal_; }
    const float3* getInterfaceNormals() const { return d_interface_normal_; }

    /**
     * @brief Get interface curvature (device pointer)
     * @return Device pointer to curvature array [1/m]
     */
    float* getCurvature() { return d_curvature_; }
    const float* getCurvature() const { return d_curvature_; }

    /**
     * @brief Copy fill level to host
     * @param host_fill Host array (must be pre-allocated, size nx*ny*nz)
     */
    void copyFillLevelToHost(float* host_fill) const;

    /**
     * @brief Copy cell flags to host
     * @param host_flags Host array (must be pre-allocated, size nx*ny*nz)
     */
    void copyCellFlagsToHost(uint8_t* host_flags) const;

    /**
     * @brief Copy curvature to host
     * @param host_curvature Host array (must be pre-allocated, size nx*ny*nz)
     */
    void copyCurvatureToHost(float* host_curvature) const;

    /**
     * @brief Compute total liquid mass for mass conservation check
     * @return Total liquid mass Σf_i
     */
    float computeTotalMass() const;

    /**
     * @brief Apply evaporation mass loss to fill level
     * @param J_evap Device array of evaporation mass flux [kg/(m^2*s)]
     * @param rho Material density [kg/m^3]
     * @param dt Time step [s]
     * @note Formula: df/dt = -J_evap / (rho * dx)
     * @note Only applies to interface cells (0.01 < f < 0.99) or cells with f > 0
     * @note Includes stability limiter to prevent df > max_df_per_step
     */
    void applyEvaporationMassLoss(const float* J_evap, float rho, float dt);

    /**
     * @brief Apply solidification shrinkage to fill level
     * @param dfl_dt Liquid fraction rate of change [1/s] (device pointer)
     * @param beta Shrinkage factor = 1 - rho_liquid/rho_solid
     * @param dx Grid spacing [m]
     * @param dt Time step [s]
     */
    void applySolidificationShrinkage(const float* dfl_dt, float beta, float dx, float dt);

    /**
     * @brief Get domain dimensions
     */
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

    /**
     * @brief Get lattice spacing
     */
    float getDx() const { return dx_; }

    /**
     * @brief Set VOF advection scheme
     * @param scheme UPWIND or TVD
     * @note Default is UPWIND for stability. Use TVD for better mass conservation.
     */
    void setAdvectionScheme(VOFAdvectionScheme scheme) { advection_scheme_ = scheme; }

    /**
     * @brief Get current advection scheme
     */
    VOFAdvectionScheme getAdvectionScheme() const { return advection_scheme_; }

    /**
     * @brief Set TVD flux limiter type
     * @param limiter MINMOD, VAN_LEER, SUPERBEE, or MC
     * @note Only applies when advection_scheme = TVD
     * @note Recommended: VAN_LEER for general use, SUPERBEE for sharper interfaces
     */
    void setTVDLimiter(TVDLimiter limiter) { tvd_limiter_ = limiter; }

    /**
     * @brief Get current TVD limiter
     */
    TVDLimiter getTVDLimiter() const { return tvd_limiter_; }

private:
    // Domain dimensions
    int nx_, ny_, nz_;
    int num_cells_;
    float dx_;  ///< Lattice spacing [m]

    // Boundary conditions
    BoundaryType bc_x_, bc_y_, bc_z_;

    // Advection scheme settings
    VOFAdvectionScheme advection_scheme_;  ///< Current advection scheme (default: UPWIND)
    TVDLimiter tvd_limiter_;               ///< TVD flux limiter type (default: VAN_LEER)

    // Device memory for VOF fields
    float* d_fill_level_;           ///< Fill level field (0-1)
    uint8_t* d_cell_flags_;         ///< Cell flag field (GAS/LIQUID/INTERFACE/OBSTACLE)
    float3* d_interface_normal_;    ///< Interface normal vectors
    float* d_curvature_;            ///< Interface curvature [1/m]

    // Temporary storage for advection
    float* d_fill_level_tmp_;       ///< Temporary fill level for advection

    // Utility functions
    void allocateMemory();
    void freeMemory();
};

// CUDA kernels for VOF solver

/**
 * @brief CUDA kernel for first-order upwind advection of fill level
 * @note Uses donor-cell scheme: stable but diffusive
 * @param bc_x Boundary condition in x (0=periodic, 1=wall)
 * @param bc_y Boundary condition in y (0=periodic, 1=wall)
 * @param bc_z Boundary condition in z (0=periodic, 1=wall)
 */
__global__ void advectFillLevelUpwindKernel(
    const float* fill_level,
    float* fill_level_new,
    const float* ux,
    const float* uy,
    const float* uz,
    float dt,
    float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z);

/**
 * @brief CUDA kernel for TVD advection with flux limiter
 * @note Second-order accurate in smooth regions, first-order near discontinuities
 * @param fill_level Input fill level field [0-1]
 * @param fill_level_new Output fill level field [0-1]
 * @param ux, uy, uz Velocity components [m/s]
 * @param dt Time step [s]
 * @param dx Grid spacing [m]
 * @param nx, ny, nz Grid dimensions
 * @param bc_x, bc_y, bc_z Boundary conditions (0=periodic, 1=wall)
 * @param limiter_type TVD limiter (0=minmod, 1=van Leer, 2=superbee, 3=MC)
 * @note Maintains conservative flux formulation for mass conservation
 * @note CFL condition: |u|dt/dx < 0.5 (same as upwind)
 * @note TVD property ensures no spurious oscillations
 */
__global__ void advectFillLevelTVDKernel(
    const float* fill_level,
    float* fill_level_new,
    const float* ux,
    const float* uy,
    const float* uz,
    float dt,
    float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z,
    int limiter_type);

/**
 * @brief CUDA kernel for interface reconstruction
 * @note Computes interface normals from fill level gradients
 */
__global__ void reconstructInterfaceKernel(
    const float* fill_level,
    float3* interface_normal,
    float dx,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for curvature computation using height function method
 * @note More accurate than finite difference on normals
 */
__global__ void computeCurvatureKernel(
    const float* fill_level,
    const float3* interface_normal,
    float* curvature,
    float dx,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for cell type conversion
 * @note Updates cell flags based on fill level thresholds
 */
__global__ void convertCellsKernel(
    const float* fill_level,
    uint8_t* cell_flags,
    float eps,
    int num_cells);

/**
 * @brief CUDA kernel for contact angle boundary condition
 * @note Modifies interface normal at walls to match contact angle
 */
__global__ void applyContactAngleBoundaryKernel(
    float3* interface_normal,
    const uint8_t* cell_flags,
    float contact_angle,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for initializing spherical droplet
 */
__global__ void initializeDropletKernel(
    float* fill_level,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for mass summation (reduction)
 * @note Computes Σf_i for mass conservation check
 */
__global__ void computeMassReductionKernel(
    const float* fill_level,
    float* partial_sums,
    int num_cells);

/**
 * @brief CUDA kernel for applying evaporation mass loss
 * @param fill_level Fill level field [0-1] (modified in-place)
 * @param J_evap Evaporation mass flux [kg/(m^2*s)]
 * @param rho Material density [kg/m^3]
 * @param dx Lattice spacing [m]
 * @param dt Time step [s]
 * @param nx, ny, nz Grid dimensions
 * @note df = -J_evap * dt / (rho * dx)
 * @note Limited to max 10% reduction per timestep for stability
 */
__global__ void applyEvaporationMassLossKernel(
    float* fill_level,
    const float* J_evap,
    float rho,
    float dx,
    float dt,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for applying solidification shrinkage mass source
 * @param fill_level Fill level field [0-1] (modified in-place)
 * @param dfl_dt Liquid fraction rate of change [1/s]
 * @param beta Shrinkage factor = 1 - rho_liquid/rho_solid
 * @param dx Grid spacing [m] (kept for API compatibility, not used in formula)
 * @param dt Time step [s]
 * @param num_cells Total number of cells
 * @note CORRECTED: df = beta * (df_l/dt) * dt  (dimensionless, no /dx)
 * @note Only applied at interface cells (0.01 < f < 0.99) during solidification (rate < 0)
 * @note Solidifying: df_l/dt < 0 --> df < 0 --> volume shrinks
 */
__global__ void applySolidificationShrinkageKernel(
    float* fill_level,
    const float* dfl_dt,
    float beta,
    float dx,
    float dt,
    int num_cells);

/**
 * @brief CUDA kernel for Olsson-Kreiss interface compression
 * @param fill_level Output compressed fill level field [0-1]
 * @param fill_level_old Input fill level field after advection [0-1]
 * @param ux Velocity field x-component [m/s] (device pointer)
 * @param uy Velocity field y-component [m/s] (device pointer)
 * @param uz Velocity field z-component [m/s] (device pointer)
 * @param dx Lattice spacing [m]
 * @param dt Time step [s]
 * @param C_compress Compression coefficient (typically 0.5)
 * @param nx, ny, nz Grid dimensions
 *
 * @note Implements: ∂φ/∂t = ∇·(ε·φ·(1-φ)·n) where ε = C * |u|_max * dx
 * @note Only acts on interface cells (0.01 < φ < 0.99)
 * @note Counteracts numerical diffusion from upwind advection
 * @note Preserves mass through conservative divergence formulation
 *
 * References:
 *   - Olsson & Kreiss (2005). A conservative level set method for two phase flow.
 *     Journal of Computational Physics, 210(1), 225-246.
 */
__global__ void applyInterfaceCompressionKernel(
    float* fill_level,
    const float* fill_level_old,
    const float* ux,
    const float* uy,
    const float* uz,
    float dx,
    float dt,
    float C_compress,
    int nx, int ny, int nz);

} // namespace physics
} // namespace lbm
