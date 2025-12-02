/**
 * @file force_accumulator.h
 * @brief Robust force accumulation pipeline for multiphysics simulations
 *
 * This class provides a clean, debuggable interface for accumulating multiple
 * force contributions (buoyancy, Darcy damping, surface tension, Marangoni,
 * recoil pressure) with proper unit tracking and diagnostics.
 *
 * Design principles:
 * - Unit consistency: All forces accumulated in physical units [N/m³]
 * - Order independence: Each force adds independently (no in-place modifications)
 * - Clear separation: Unit conversion and CFL limiting are separate stages
 * - Debuggable: Track individual force magnitudes for diagnostics
 * - Testable: Each force component can be tested in isolation
 *
 * Pipeline stages:
 * 1. reset() - Zero all force arrays
 * 2. addBuoyancyForce(...) - Add thermal buoyancy
 * 3. addDarcyDamping(...) - Add velocity-dependent Darcy damping
 * 4. addSurfaceTensionForce(...) - Add curvature-driven capillary forces
 * 5. addMarangoniForce(...) - Add thermocapillary forces
 * 6. addRecoilPressureForce(...) - Add evaporation recoil pressure (optional)
 * 7. convertToLatticeUnits(dx, dt) - Convert [N/m³] → lattice units
 * 8. applyCFLLimiting(...) - Limit forces to maintain numerical stability
 *
 * @note This class replaces the fragile force computation pipeline in
 *       MultiphysicsSolver::computeTotalForce()
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Force accumulator for multiphysics simulations
 *
 * Provides a robust pipeline for accumulating and processing forces from
 * multiple physics modules with proper unit tracking and diagnostics.
 */
class ForceAccumulator {
public:
    /**
     * @brief Constructor
     * @param nx, ny, nz Grid dimensions
     */
    ForceAccumulator(int nx, int ny, int nz);

    /**
     * @brief Destructor - frees device memory
     */
    ~ForceAccumulator();

    // Disable copy and move (device memory management)
    ForceAccumulator(const ForceAccumulator&) = delete;
    ForceAccumulator& operator=(const ForceAccumulator&) = delete;
    ForceAccumulator(ForceAccumulator&&) = delete;
    ForceAccumulator& operator=(ForceAccumulator&&) = delete;

    /**
     * @brief Reset all forces to zero (call at start of each timestep)
     */
    void reset();

    /**
     * @brief Add buoyancy force (Boussinesq approximation)
     * @param temperature Temperature field [K]
     * @param T_ref Reference temperature [K]
     * @param beta Thermal expansion coefficient [1/K]
     * @param rho Density [kg/m³]
     * @param gx, gy, gz Gravity vector [m/s²]
     * @param liquid_fraction Optional liquid fraction field (0=solid, 1=liquid)
     *                        If provided, buoyancy only applied to liquid regions
     * @note Adds F_buoyancy = ρ₀ · β · (T - T_ref) · g [N/m³]
     */
    void addBuoyancyForce(const float* temperature, float T_ref, float beta,
                          float rho, float gx, float gy, float gz,
                          const float* liquid_fraction = nullptr);

    /**
     * @brief Add Darcy damping for mushy/solid regions
     * @param liquid_fraction Liquid fraction field (0=solid, 1=liquid)
     * @param vx, vy, vz Velocity components [lattice units]
     * @param darcy_coeff Darcy coefficient C [dimensionless]
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     * @note Adds F_darcy = -C · (1 - f_l)² / (f_l³ + ε) · ρ · v [N/m³]
     * @note This is velocity-dependent damping, not a true force
     * @note Requires velocity in lattice units for consistency with LBM solver
     */
    void addDarcyDamping(const float* liquid_fraction, const float* vx,
                         const float* vy, const float* vz, float darcy_coeff,
                         float dx, float dt, float rho);

    /**
     * @brief Add surface tension force (CSF model)
     * @param curvature Interface curvature κ [1/m]
     * @param fill_level VOF fill level (0=gas, 1=liquid)
     * @param sigma Surface tension coefficient [N/m]
     * @note Adds F_st = σ · κ · ∇f [N/m³]
     */
    void addSurfaceTensionForce(const float* curvature, const float* fill_level,
                                float sigma, int nx, int ny, int nz, float dx);

    /**
     * @brief Add Marangoni force (thermocapillary effect)
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0=gas, 1=liquid)
     * @param normals Interface normal vectors (float3)
     * @param dsigma_dT Temperature coefficient of surface tension [N/(m·K)]
     * @param nx, ny, nz Grid dimensions
     * @param dx Lattice spacing [m]
     * @note Adds F_m = (dσ/dT · ∇T) × n [N/m³]
     */
    void addMarangoniForce(const float* temperature, const float* fill_level,
                           const float3* normals, float dsigma_dT,
                           int nx, int ny, int nz, float dx);

    /**
     * @brief Add recoil pressure force (evaporation-driven)
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0=gas, 1=liquid)
     * @param normals Interface normal vectors (float3)
     * @param T_boil Boiling temperature [K]
     * @param L_v Latent heat of vaporization [J/kg]
     * @param M Molar mass [kg/mol]
     * @param P_atm Atmospheric pressure [Pa]
     * @param C_r Recoil coefficient (dimensionless, ~0.54)
     * @param smoothing_width Interface smoothing width [cells]
     * @param max_pressure Maximum allowed pressure [Pa]
     * @param nx, ny, nz Grid dimensions
     * @note Adds F_recoil = P_recoil · n [N/m³]
     * @note P_recoil = C_r · P_sat(T) where P_sat uses Clausius-Clapeyron
     */
    void addRecoilPressureForce(const float* temperature, const float* fill_level,
                                const float3* normals, float T_boil, float L_v,
                                float M, float P_atm, float C_r,
                                float smoothing_width, float max_pressure,
                                int nx, int ny, int nz);

    /**
     * @brief Convert accumulated forces from physical units to lattice units
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     * @param rho Density [kg/m³]
     * @note Conversion: F_lattice = F_physical · (dt² / (dx · ρ)) [dimensionless]
     * @note Must be called after all forces are added, before CFL limiting
     */
    void convertToLatticeUnits(float dx, float dt, float rho);

    /**
     * @brief Apply CFL-based force limiting for numerical stability
     * @param vx, vy, vz Current velocity field [lattice units]
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     * @param v_target Target maximum lattice velocity (typically 0.1-0.15)
     * @param ramp_factor Gradual ramp factor (0.8-0.9, where limiting starts)
     * @note Limits forces to prevent CFL number from exceeding safe bounds
     * @note Uses gradual scaling to avoid discontinuous force jumps
     */
    void applyCFLLimiting(const float* vx, const float* vy, const float* vz,
                          float dx, float dt, float v_target, float ramp_factor);

    /**
     * @brief Apply adaptive CFL limiting with region-based velocity targets
     * @param vx, vy, vz Current velocity field [lattice units]
     * @param fill_level VOF fill level (0=gas, 1=liquid)
     * @param liquid_fraction Phase field (0=solid, 1=liquid)
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     * @param v_target_interface Target velocity for interface cells
     * @param v_target_bulk Target velocity for bulk liquid cells
     * @param interface_lo Lower fill level threshold for interface
     * @param interface_hi Upper fill level threshold for interface
     * @param recoil_boost_factor Extra velocity allowance for recoil forces
     * @param ramp_factor Gradual ramp factor
     * @note More sophisticated than basic CFL limiting: allows different
     *       velocity limits in different regions (interface vs bulk vs solid)
     */
    void applyCFLLimitingAdaptive(const float* vx, const float* vy, const float* vz,
                                  const float* fill_level, const float* liquid_fraction,
                                  float dx, float dt, float v_target_interface,
                                  float v_target_bulk, float interface_lo,
                                  float interface_hi, float recoil_boost_factor,
                                  float ramp_factor);

    /**
     * @brief Get force arrays (const access for solver)
     */
    const float* getFx() const { return d_fx_; }
    const float* getFy() const { return d_fy_; }
    const float* getFz() const { return d_fz_; }

    /**
     * @brief Get force arrays (mutable access for testing)
     */
    float* getFx() { return d_fx_; }
    float* getFy() { return d_fy_; }
    float* getFz() { return d_fz_; }

    /**
     * @brief Get maximum force magnitude in domain
     * @return Max |F| (units depend on whether conversion has been applied)
     */
    float getMaxForceMagnitude() const;

    /**
     * @brief Get breakdown of force contributions
     * @param[out] buoyancy_mag Maximum buoyancy force magnitude
     * @param[out] darcy_mag Maximum Darcy damping magnitude
     * @param[out] surface_tension_mag Maximum surface tension force magnitude
     * @param[out] marangoni_mag Maximum Marangoni force magnitude
     * @param[out] recoil_mag Maximum recoil pressure force magnitude
     * @note Call after each force addition to track individual contributions
     * @note Values are in physical units [N/m³] before conversion
     */
    void getForceBreakdown(float& buoyancy_mag, float& darcy_mag,
                           float& surface_tension_mag, float& marangoni_mag,
                           float& recoil_mag) const;

    /**
     * @brief Print force breakdown to console (for debugging)
     * @note Useful for diagnosing which force is causing instability
     */
    void printForceBreakdown() const;

private:
    // Grid dimensions
    int nx_, ny_, nz_;
    int num_cells_;

    // Device memory: Total force arrays
    float* d_fx_;  // X-component of force
    float* d_fy_;  // Y-component of force
    float* d_fz_;  // Z-component of force

    // Diagnostic tracking: Maximum magnitude of each force type
    // Updated after each addXXXForce() call for debugging
    float buoyancy_mag_ = 0.0f;
    float darcy_mag_ = 0.0f;
    float surface_tension_mag_ = 0.0f;
    float marangoni_mag_ = 0.0f;
    float recoil_mag_ = 0.0f;

    // Internal helpers
    void allocateMemory();
    void freeMemory();
};

} // namespace physics
} // namespace lbm
