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

    // F-04 (code-audit pass 1, 2026-04-27): rule of five — own raw cudaMalloc'd
    // device pointers, so default copy/assign would shallow-copy them and
    // double-free at destruction.
    VOFSolver(const VOFSolver&)            = delete;
    VOFSolver& operator=(const VOFSolver&) = delete;

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
     * @brief Enforce global mass conservation
     * @param target_mass Target total mass Σf to conserve.
     * @param d_vz Optional device pointer to vertical velocity field [m/s].
     *             When non-null, switches to A1 v_z-weighted additive
     *             correction: redistribute mass deficit preferentially to
     *             interface cells with upward flow (capillary back-flow zone)
     *             and away from over-deposited stagnant regions.
     *             When null (default), falls back to the legacy uniform
     *             multiplicative scaling for backward compatibility.
     *
     * Algorithm A1 (when d_vz != nullptr):
     *   w_i = max(sign(Δm) * v_z[i], 0)  for interface cells (0<f<1), else 0
     *   W = Σ w_i
     *   f_new[i] = clamp(f[i] + (Δm/W) * w_i, 0, 1)
     *
     * Falls back to uniform additive correction over interface cells if
     * W ≈ 0 (no cells with the right flow direction).
     *
     * @note Only applies correction if relative mass error > 0.1%.
     */
    void enforceGlobalMassConservation(float target_mass,
                                       const float* d_vz = nullptr);

    /**
     * @brief Track-B public entry: w = max(sign(Δm)·(-∇f·v), 0).
     * @param target_mass Target Σf to conserve.
     * @param d_vx, d_vy, d_vz Device velocity ptrs [m/s].
     *
     * Computes the inward-flux weight inline from 6-neighbour fill_level
     * gradients. Falls back to uniform additive over interface cells if W ≈ 0.
     * Used by unit tests that need to drive the Track-B kernels with synthetic
     * fields (matching Track-A's 2-arg testing pattern).
     */
    void enforceGlobalMassConservation(float target_mass,
                                       const float* d_vx,
                                       const float* d_vy,
                                       const float* d_vz);

    /**
     * @brief Track-C public entry: Track-B + geometric gate arguments.
     *
     * Intended for unit tests that need to exercise the Track-C gates with
     * synthetic inputs, without going through advectFillLevel().  Production
     * code uses the member setters (setMassCorrectionLaserX /
     * setMassCorrectionZSubstrate) and lets advectFillLevel() pick them up.
     *
     * @param target_mass  Target Σf to conserve.
     * @param d_vx/vy/vz   Device velocity arrays [m/s].
     * @param laser_x_lu   Current laser x in lattice units; negative = gate off.
     * @param trailing_margin_lu  Exclusion half-width [lu] behind laser front.
     *                    Cells with i > laser_x_lu - margin are skipped.
     *                    Typical: 25 lu (= 50 μm at dx=2 μm).
     * @param z_substrate_lu  Substrate top index [lu]; negative = gate off.
     * @param z_offset_lu  Allowance above substrate before exclusion [lu].
     *                    Typical: 2 lu.
     *
     * Falls back to uniform additive over interface cells when W ≈ 0.
     */
    void enforceMassConservationFlux(float target_mass,
                                     const float* d_vx,
                                     const float* d_vy,
                                     const float* d_vz,
                                     float laser_x_lu        = -1.0f,
                                     float trailing_margin_lu = 25.0f,
                                     float z_substrate_lu    = -1.0f,
                                     float z_offset_lu       =  2.0f);

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
     * @brief Toggle Track-B (inline-∇f flux weight) on the mass-correction helper.
     * @param use true → w = max(-∇f·v, 0) using inline central-diff gradient
     *            false → w = max(v_z, 0) Track-A (default, legacy)
     * @note When enabled, the TVD and PLIC advection paths call the new
     *       applyMassCorrectionInline(vx,vy,vz) overload instead of the
     *       single-vz one. Both forms guard on mass_correction_enabled_.
     */
    void setMassCorrectionUseFluxWeight(bool use) {
        mass_correction_use_flux_weight_ = use;
    }
    bool getMassCorrectionUseFluxWeight() const {
        return mass_correction_use_flux_weight_;
    }

    /**
     * @brief Track-C Gate 1 setter: update the laser x position used by the
     *        trailing-band exclusion mask inside the flux-weight kernels.
     *
     * Call once per simulation step, BEFORE advectFillLevel(), so the new
     * position is picked up in applyMassCorrectionInline().
     *
     * @param laser_x_lu  Current laser x in lattice units.
     *                    Pass a negative value to disable Gate 1.
     * @param margin_lu   Exclusion half-width [lu]; cells with
     *                    i > laser_x_lu - margin_lu are skipped.
     *                    Default 25 lu (= 50 μm at dx=2 μm).
     */
    void setMassCorrectionLaserX(float laser_x_lu, float margin_lu = 25.0f) {
        mass_correction_laser_x_lu_       = laser_x_lu;
        mass_correction_trailing_margin_lu_ = margin_lu;
    }

    /**
     * @brief Track-C Gate 2 setter: set substrate top index for the z-floor gate.
     *
     * Call once after initialization (the substrate height is fixed for the
     * duration of a single-layer simulation).
     *
     * @param z_substrate_lu  Substrate top cell index in lattice units.
     *                        Pass a negative value to disable Gate 2.
     * @param z_offset_lu     Extra cells of allowance above substrate before
     *                        exclusion fires.  Default 2 lu.
     */
    /// Note: z_offset_lu=0 → strict (cells above substrate top fully excluded).
    /// z_offset_lu=2 → tolerant (allow 2 cells of growth before exclusion).
    /// For F3D match try 0 first; relax if W collapses too often.
    void setMassCorrectionZSubstrate(float z_substrate_lu, float z_offset_lu = 2.0f) {
        mass_correction_z_substrate_lu_ = z_substrate_lu;
        mass_correction_z_offset_lu_    = z_offset_lu;
    }

    float getMassCorrectionLaserX()      const { return mass_correction_laser_x_lu_; }
    float getMassCorrectionZSubstrate()  const { return mass_correction_z_substrate_lu_; }

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
    bool mass_correction_use_flux_weight_ = false; ///< Track-B/C (inline-∇f flux); false = Track-A (v_z)

    // Track-C geometric gate parameters (defaults: all gates disabled).
    // Updated each step by MultiphysicsSolver via setMassCorrectionLaserX().
    float mass_correction_laser_x_lu_        = -1.0f; ///< Gate 1: laser x [lu]; <0 = off
    float mass_correction_trailing_margin_lu_ = 25.0f; ///< Gate 1: exclusion half-width [lu]
    float mass_correction_z_substrate_lu_    = -1.0f; ///< Gate 2: substrate top [lu]; <0 = off
    float mass_correction_z_offset_lu_       =  2.0f; ///< Gate 2: allowance above substrate [lu]

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

    // Cached scratch buffers (lazy-allocated, reused across calls).
    // Bug-3 fix (2026-04-26): previously cudaMalloc'd inside computeTotalMass()
    // and the correction path on every invocation — 3× per advection step.
    //
    // INVARIANT: every kernel that uses these buffers MUST launch with the same
    // block size (currently 256). The lazy realloc only checks gridSize against
    // a stored `_size_`, so changing block size between launches without changing
    // num_cells_ would silently underrun the buffer. Search "blockSize = 256"
    // in vof_solver.cu — all uses agree. (B3 hazard noted 2026-04-27.)
    mutable float* d_mass_partial_sums_ = nullptr;
    mutable int d_mass_partial_sums_size_ = 0;
    int* d_interface_partial_counts_ = nullptr;
    int d_interface_partial_counts_size_ = 0;

    // ---- PLIC geometric advection buffers (lazy-allocated) ----
    lbm::utils::CudaBuffer<float> plic_nx_;
    lbm::utils::CudaBuffer<float> plic_ny_;
    lbm::utils::CudaBuffer<float> plic_nz_;
    lbm::utils::CudaBuffer<float> plic_alpha_;
    lbm::utils::CudaBuffer<float> plic_flux_;        // reusable per-direction face flux
    lbm::utils::CudaBuffer<float> plic_face_vel_;    // reusable per-direction face velocity
    bool plic_strang_x_first_ = true;

    // Utility functions
    void allocateMemory();
    void freeMemory();
    void advectFillLevelPLIC(const float* d_ux, const float* d_uy, const float* d_uz, float dt);
    void plicAllocateIfNeeded();

    /// A1 (Track-A) helper: post-advection global-mass correction with
    /// w = max(sign(Δm)·v_z, 0) weight. Single-velocity, fast.
    void applyMassCorrectionInline(const float* d_vz);

    /// B1 (Track-B) helper: post-advection global-mass correction with
    /// w = max(sign(Δm)·(-∇f·v), 0) weight. ∇f is computed inline via
    /// central differences from 6 neighbour fill_levels — no normal field
    /// needed. Better at distinguishing capillary back-fill (toward groove)
    /// from recoil-driven outward jet (away from liquid surface).
    /// Falls back to uniform-additive when W ≈ 0.
    void applyMassCorrectionInline(const float* d_vx, const float* d_vy,
                                    const float* d_vz);

    int mass_correction_call_count_ = 0;
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
