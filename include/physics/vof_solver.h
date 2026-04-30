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
#include "utils/cuda_memory.h"
#include "physics/interface_geometry.h"

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
    TVD    = 1,     ///< TVD with flux limiter (2nd-order in smooth regions)
    PLIC   = 2      ///< Geometric PLIC (Piecewise Linear Interface Calculation)
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
 * @brief PLIC normal reconstruction algorithm.
 *
 * Affects the kernel that VOFSolver uses to estimate the per-cell unit
 * interface normal n̂ inside `recomputePLICReconstruction()` and inside
 * the Strang-split sweeps in `advectFillLevelPLIC()`. The choice trades
 * implementation simplicity / robustness against accuracy on curved
 * interfaces.
 *
 *   - YOUNGS: Parker-Youngs 3×3×3 weighted central differences. Default.
 *     Exact for axis-aligned planar interfaces (round-off only). Angular
 *     error on a sphere of radius R is O(h/R) — the algorithm is first
 *     order on curved surfaces. Robust at boundaries and for under-resolved
 *     features. Cost: ~30 reads per cell.
 *
 *   - HEIGHT_FUNCTION: 7-point column heights along the Youngs-determined
 *     dominant axis, with central differences for the two non-dominant
 *     normal components. Angular error on a sphere is O((h/R)²) — second
 *     order on curved surfaces. Falls back to Youngs in cells where the
 *     column does not bracket a clean liquid→gas transition. Required to
 *     hit the roadmap §3 Phase 1 acceptance gate of ε<1e-3 on a curved
 *     interface. Cost: ~70 reads per cell.
 *
 * Reference for HEIGHT_FUNCTION:
 *   Cummins, Francois & Kothe (2005). Estimating curvature from volume
 *   fractions. Computers & Structures 83, 425-434. (height-function
 *   kernel; PLIC normals derived from the same column-height construct.)
 */
enum class NormalReconstructionMethod : uint8_t {
    YOUNGS          = 0,
    HEIGHT_FUNCTION = 1
};

/**
 * @brief Curvature reconstruction algorithm.
 *
 * Phase 3a of the PLIC upgrade. The default LEGACY_HF path is the existing
 * height-function-on-fill kernel, which differentiates the smoothed fill
 * level twice — robust but smeared and prone to ~10% κ errors on a
 * resolved sphere. PLIC_DIVERGENCE computes κ = -∇·n̂ via central
 * differences on the per-cell unit normal stored in plic_nx_/ny_/nz_; with
 * HEIGHT_FUNCTION normals it gives the standard O((h/R)²) Cummins-Francois-
 * Kothe accuracy, which is required to pair with sharp surface deltas in
 * Phase 3b without amplifying κ noise.
 */
enum class CurvatureMethod : uint8_t {
    LEGACY_HF       = 0,   ///< height-function on fill_level (current behaviour)
    PLIC_DIVERGENCE = 1    ///< -∇·n̂ on the PLIC unit-normal field
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
     * @brief Enforce global mass conservation by scaling fill levels
     * @param target_mass Target total mass to conserve
     * @note Scales all fill levels uniformly: f_new = f_old * (target_mass / current_mass)
     * @note Should be called after advection to correct accumulated mass errors
     * @note Only applies correction if mass error > 0.1% to avoid unnecessary rescaling
     */
    void enforceGlobalMassConservation(float target_mass);

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
     * @brief Phase 4a PLIC-aware evaporation: df = -J · δ_h(d) · dt / ρ.
     *
     * Sharp-delta replacement for applyEvaporationMassLoss that uses the
     * cached PLIC plane geometry. Mass loss is concentrated in the cosine-
     * kernel band around the PLIC interface (default 3 cells). Bulk cells
     * (deep in the metal where T may be high but no surface is exposed)
     * receive no mass loss — fixing a known artefact of the legacy kernel
     * which removed mass anywhere f > 0 ∧ J > 0.
     *
     * Caller does NOT need to call recomputePLICReconstruction() first;
     * this method does it internally if the cache is dirty.
     */
    void applyEvaporationMassLossPLIC(const float* J_evap, float rho,
                                       float dt, float h_smooth_lu = 1.5f);

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

    /**
     * @brief Enable/disable mass conservation correction
     * @param enable True to enable global mass correction after advection
     * @param damping Damping factor [0.1-1.0] for mass redistribution (default: 0.7)
     * @note Recommended: enable=true for long simulations, damping=0.5-0.8
     * @note Cost: ~5% overhead, benefit: <1% mass error (vs 5-20% without)
     */
    void setMassConservationCorrection(bool enable, float damping = 0.7f) {
        mass_correction_enabled_ = enable;
        mass_correction_damping_ = damping;
    }

    /**
     * @brief Set reference mass for conservation tracking
     * @param mass_ref Reference mass (typically computed at t=0)
     * @note Call this after initialization to establish baseline
     */
    void setReferenceMass(float mass_ref) { mass_reference_ = mass_ref; }

    /**
     * @brief Enable or disable Olsson-Kreiss interface compression
     * @param enabled True to enable compression, false to disable (default: false)
     * @param coefficient Compression coefficient C in ε = C·|u|_max·dx (default: 0.10)
     * @note Compression sharpens diffuse interfaces but can cause artifacts at concave
     *       corners (e.g., Zalesak disk slot). Disable for pure advection benchmarks.
     */
    void setInterfaceCompression(bool enabled, float coefficient = 0.10f);

    /**
     * @brief Get reference mass
     */
    float getReferenceMass() const { return mass_reference_; }

    // ========================================================================
    // PLIC Interface Geometry API (Phase 1 of PLIC upgrade)
    // ========================================================================
    //
    // Phase 1 makes the per-cell PLIC reconstruction (Youngs unit normal +
    // signed alpha) accessible to downstream physics modules so they can
    // replace the smeared `|∇f|` kernel-based delta with a sharp interface.
    //
    // Storage convention (see include/physics/interface_geometry.h):
    //   Plane n·X = alpha_signed in cell-corner unit-cube frame [0,1]^3.
    //   Liquid lives where n·X < alpha_signed. n̂ is unit, points toward gas.
    //
    // Lifecycle:
    //   1. Any call that modifies fill_level (initialize, advect, evap,
    //      shrinkage, mass correction) sets plic_dirty_ = true and the cached
    //      arrays may no longer match d_fill_level_.
    //   2. recomputePLICReconstruction() runs the Youngs + alpha kernels and
    //      clears the dirty flag. If already clean, returns immediately.
    //   3. getInterfaceGeometry() returns a view marked plic_ready=true iff
    //      the cache is current. Callers MUST consult plic_ready before
    //      reading d_alpha (the underlying buffer may be uninitialized on a
    //      brand-new VOFSolver that has never run PLIC advection).

    /**
     * @brief (Re)compute Youngs normal + alpha from the current fill_level.
     * @note No-op if the cached reconstruction is already consistent with the
     *       current fill (plic_dirty_ == false). Lazy-allocates the PLIC
     *       buffers on first use.
     * @note Cost: 2 kernel launches (~5–20 μs for 100^3 grid). Idempotent.
     */
    void recomputePLICReconstruction();

    /**
     * @brief Get a read-only view of the interface geometry.
     * @return InterfaceGeometryView with d_normal_x/y/z, d_alpha (nullable),
     *         d_fill, plic_ready, and grid dimensions.
     * @note `view.plic_ready == false` iff PLIC reconstruction is stale or
     *       has never been run. In that case, `view.d_alpha` and the normals
     *       must NOT be dereferenced; downstream kernels should fall back to
     *       the legacy `|∇f|`-based path.
     */
    InterfaceGeometryView getInterfaceGeometry() const;

    /**
     * @brief True iff cached PLIC reconstruction matches current fill_level.
     */
    bool isPLICReady() const { return !plic_dirty_; }

    /**
     * @brief Select the normal-reconstruction algorithm used by PLIC paths.
     * @param method YOUNGS (default, O(h/R) on curves) or HEIGHT_FUNCTION
     *               (O((h/R)²) on curves; fallback to Youngs in degenerate cells).
     * @note Marks the cache dirty so the next reconstruction uses the new method.
     */
    void setNormalReconstructionMethod(NormalReconstructionMethod method) {
        normal_method_ = method;
        plicMarkDirty();
    }

    /**
     * @brief Get the active normal-reconstruction algorithm.
     */
    NormalReconstructionMethod getNormalReconstructionMethod() const {
        return normal_method_;
    }

    /**
     * @brief Select the curvature-reconstruction algorithm.
     * @param method LEGACY_HF (default, height-function on fill_level) or
     *               PLIC_DIVERGENCE (-∇·n̂ on PLIC normals).
     */
    void setCurvatureMethod(CurvatureMethod method) {
        curvature_method_ = method;
        // PLIC_DIVERGENCE uses the cached PLIC normal field; ensure it is
        // current next time computeCurvature() runs.
        plicMarkDirty();
    }

    CurvatureMethod getCurvatureMethod() const { return curvature_method_; }

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

    // Mass conservation correction settings
    bool mass_correction_enabled_;         ///< Enable global mass correction (default: false)
    float mass_correction_damping_;        ///< Damping factor for redistribution (default: 0.7)
    float mass_reference_;                 ///< Reference mass for conservation tracking

    // Interface compression settings
    bool interface_compression_enabled_ = false;  ///< Enable Olsson-Kreiss compression (default: OFF)
    float C_compress_coeff_ = 0.10f;              ///< Compression coefficient when enabled

    // Device memory for VOF fields
    float* d_fill_level_;           ///< Fill level field (0-1)
    uint8_t* d_cell_flags_;         ///< Cell flag field (GAS/LIQUID/INTERFACE/OBSTACLE)
    float3* d_interface_normal_;    ///< Interface normal vectors
    float* d_curvature_;            ///< Interface curvature [1/m]

    // Temporary storage for advection
    float* d_fill_level_tmp_;       ///< Temporary fill level for advection

    // ---- PLIC geometric advection buffers (lazy-allocated) ----
    lbm::utils::CudaBuffer<float> plic_nx_;
    lbm::utils::CudaBuffer<float> plic_ny_;
    lbm::utils::CudaBuffer<float> plic_nz_;
    lbm::utils::CudaBuffer<float> plic_alpha_;
    lbm::utils::CudaBuffer<float> plic_flux_;        // reusable per-direction face flux
    lbm::utils::CudaBuffer<float> plic_face_vel_;    // reusable per-direction face velocity

    // 3-way symmetric Strang rotation: cycle through 6 permutations of {x,y,z}
    //   even step (0,2,4): forward order (XYZ, YZX, ZXY)
    //   odd  step (1,3,5): reverse order (ZYX, XZY, YXZ)
    // Counter wraps mod 6 so all three axes spend equal time as the "last sweep".
    // This eliminates the persistent z-bias that XY-only swap leaves untreated.
    int plic_strang_phase_ = 0;        // 0..5

    // Persistent per-instance counters (replace function-level statics — H1 fix).
    // Static counters caused two VOFSolver instances to share state and race on
    // d_block_max reallocation in the parent advectFillLevel().
    int plic_call_count_ = 0;          // diagnostic print cadence for PLIC clamp
    int plic_substep_call_count_ = 0;  // diagnostic print cadence for CFL substepping

    // Per-instance reduction buffers for CFL/v_max computation.
    // Replace the function-level static d_block_max in advectFillLevel() (H1 fix).
    lbm::utils::CudaBuffer<float> reduction_block_max_;

    // Active normal-reconstruction algorithm (default YOUNGS for backward
    // compatibility — HEIGHT_FUNCTION must be opted-in by tests / Phase 3).
    NormalReconstructionMethod normal_method_ = NormalReconstructionMethod::YOUNGS;

    // Active curvature algorithm (Phase 3a — default keeps legacy behaviour).
    CurvatureMethod curvature_method_ = CurvatureMethod::LEGACY_HF;

    // PLIC reconstruction freshness flag.
    // - Set to false on every fill_level write (initialize, advect, evap, etc.).
    // - Set to true at the end of recomputePLICReconstruction() and after the
    //   final post-advection reconstruction in advectFillLevelPLIC().
    // External callers should call recomputePLICReconstruction() before
    // reading plic_nx_/ny_/nz_/alpha_ via the public InterfaceGeometryView.
    bool plic_dirty_ = true;

    // Utility functions
    void allocateMemory();
    void freeMemory();
    void advectFillLevelPLIC(const float* d_ux, const float* d_uy, const float* d_uz, float dt);
    void plicAllocateIfNeeded();
    void plicMarkDirty() { plic_dirty_ = true; }
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
 * @brief CUDA kernel for global mass conservation correction
 * @param fill_level Fill level field [0-1] (modified in-place)
 * @param scale_factor Multiplicative factor = target_mass / current_mass
 * @param num_cells Total number of cells
 * @note Applies uniform scaling: f_new = f_old * scale_factor
 * @note Clamps result to [0, 1] to maintain physical bounds
 */
__global__ void enforceGlobalMassConservationKernel(
    float* fill_level,
    float scale_factor,
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
