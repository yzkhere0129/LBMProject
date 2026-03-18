/**
 * @file unit_converter.h
 * @brief Centralized unit conversion utilities for LBM-CUDA framework
 *
 * This class provides a single source of truth for all unit conversions between
 * lattice units (dimensionless) and physical units (SI).
 *
 * CRITICAL SAFETY: Unit conversion errors are silent and catastrophic in LBM.
 * This utility centralizes all conversions to prevent scattered, error-prone
 * conversion code throughout the codebase.
 *
 * LBM Unit System:
 * - Lattice spacing: dx_lattice = 1 (dimensionless)
 * - Time step: dt_lattice = 1 (dimensionless)
 * - Density: rho_lattice ≈ 1 (normalized in incompressible LBM)
 *
 * Physical Unit System:
 * - Length: meters [m]
 * - Time: seconds [s]
 * - Mass: kilograms [kg]
 * - Force: Newtons [N]
 * - Pressure: Pascals [Pa]
 *
 * Conversion Formulas:
 * - Velocity: v_phys = v_lattice * (dx / dt)
 * - Force (volumetric): F_lattice = F_phys * (dt² / (dx * rho_phys))
 * - Pressure: p_phys = p_lattice * (rho_phys * dx² / dt²)
 * - Diffusivity: alpha_lattice = alpha_phys * (dt / dx²)
 * - Viscosity: nu_lattice = nu_phys * (dt / dx²)
 *
 * References:
 * - Krüger et al. (2017): "The Lattice Boltzmann Method", Chapter 3
 * - Timm Krüger's LBM unit conversion guide
 */

#pragma once

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define HOST_DEVICE __host__ __device__
    #define CUDA_ARCH_CHECK __CUDA_ARCH__
#else
    #define HOST_DEVICE
    #define CUDA_ARCH_CHECK 0
#endif

#include <stdexcept>
#include <string>

namespace lbm {
namespace core {

/**
 * @brief Unit converter for LBM simulations
 *
 * This class handles all conversions between lattice units (used internally
 * by LBM solvers) and physical units (SI units for real-world quantities).
 *
 * All conversion functions are marked HOST_DEVICE to support both
 * CPU and GPU execution.
 */
class UnitConverter {
public:
    /**
     * @brief Constructor
     * @param dx Physical grid spacing [m]
     * @param dt Physical time step [s]
     * @param rho_phys Reference density [kg/m³] (default: 1.0, for normalization)
     *
     * @note For incompressible LBM, rho_phys is the reference density used for
     *       pressure and force conversions. Typically set to liquid density.
     */
    HOST_DEVICE
    UnitConverter(float dx, float dt, float rho_phys = 1.0f)
        : dx_(dx), dt_(dt), rho_phys_(rho_phys)
    {
        #if !CUDA_ARCH_CHECK
        // Validation only on host side (exceptions not available in device code)
        if (dx <= 0.0f || dt <= 0.0f || rho_phys <= 0.0f) {
            throw std::invalid_argument("UnitConverter: dx, dt, and rho_phys must be positive");
        }
        #endif
    }

    // ========================================================================
    // Velocity Conversions
    // ========================================================================

    /**
     * @brief Convert velocity from lattice units to physical units
     * @param v_lattice Velocity in lattice units [dimensionless]
     * @return Velocity in physical units [m/s]
     *
     * Formula: v_phys = v_lattice * (dx / dt)
     *
     * Derivation:
     * - Lattice velocity: distance / time = (1 cell) / (1 timestep)
     * - Physical velocity: (dx meters) / (dt seconds) = dx/dt [m/s]
     */
    HOST_DEVICE
    inline float velocityToPhysical(float v_lattice) const {
        return v_lattice * (dx_ / dt_);
    }

    /**
     * @brief Convert velocity from physical units to lattice units
     * @param v_phys Velocity in physical units [m/s]
     * @return Velocity in lattice units [dimensionless]
     *
     * Formula: v_lattice = v_phys * (dt / dx)
     */
    HOST_DEVICE
    inline float velocityToLattice(float v_phys) const {
        return v_phys * (dt_ / dx_);
    }

    // ========================================================================
    // Force Conversions (Volumetric Force Density)
    // ========================================================================

    /**
     * @brief Convert volumetric force from physical units to lattice units
     * @param F_phys Volumetric force density [N/m³] = [kg/(m²·s²)]
     * @return Force in lattice units [dimensionless]
     *
     * Formula: F_lattice = F_phys * (dt² / (dx * rho_phys))
     *
     * Derivation:
     * - Physical acceleration: a_phys = F_phys / rho_phys [m/s²]
     * - Velocity change: Δv_phys = a_phys * dt = (F_phys / rho_phys) * dt [m/s]
     * - Lattice velocity change: Δv_lattice = Δv_phys * (dt / dx)
     * - Substituting: Δv_lattice = (F_phys / rho_phys) * dt * (dt / dx)
     * - In LBM with Guo forcing: Δv_lattice = F_lattice (with rho_lattice ≈ 1)
     * - Therefore: F_lattice = F_phys * (dt² / (dx * rho_phys))
     *
     * Units check:
     * [N/m³] * [s²] / ([m] * [kg/m³]) = [kg/(m·s²)] * [s²] / ([m] * [kg/m³])
     *                                   = dimensionless ✓
     */
    HOST_DEVICE
    inline float forceToLattice(float F_phys) const {
        return F_phys * (dt_ * dt_) / (dx_ * rho_phys_);
    }

    /**
     * @brief Convert volumetric force from lattice units to physical units
     * @param F_lattice Force in lattice units [dimensionless]
     * @return Volumetric force density [N/m³]
     *
     * Formula: F_phys = F_lattice * (dx * rho_phys) / dt²
     */
    HOST_DEVICE
    inline float forceToPhysical(float F_lattice) const {
        return F_lattice * (dx_ * rho_phys_) / (dt_ * dt_);
    }

    // ========================================================================
    // Pressure Conversions
    // ========================================================================

    /**
     * @brief Convert pressure from lattice units to physical units
     * @param p_lattice Pressure in lattice units [dimensionless]
     * @return Pressure in physical units [Pa]
     *
     * Formula: p_phys = p_lattice * (rho_phys * dx² / dt²)
     *
     * Derivation:
     * - In LBM: p_lattice = c_s² * rho_lattice (with c_s = 1/√3 in lattice units)
     * - Physical pressure: p_phys = c_s² * rho_phys * (dx/dt)²
     * - Simplifying: p_phys = p_lattice * (rho_phys * dx² / dt²)
     */
    HOST_DEVICE
    inline float pressureToPhysical(float p_lattice) const {
        return p_lattice * (rho_phys_ * dx_ * dx_) / (dt_ * dt_);
    }

    /**
     * @brief Convert pressure from physical units to lattice units
     * @param p_phys Pressure in physical units [Pa]
     * @return Pressure in lattice units [dimensionless]
     *
     * Formula: p_lattice = p_phys * (dt² / (rho_phys * dx²))
     */
    HOST_DEVICE
    inline float pressureToLattice(float p_phys) const {
        return p_phys * (dt_ * dt_) / (rho_phys_ * dx_ * dx_);
    }

    // ========================================================================
    // Diffusivity Conversions (Thermal, Mass, etc.)
    // ========================================================================

    /**
     * @brief Convert diffusivity from physical units to lattice units
     * @param alpha_phys Diffusivity [m²/s]
     * @return Diffusivity in lattice units [dimensionless]
     *
     * Formula: alpha_lattice = alpha_phys * (dt / dx²)
     *
     * Derivation:
     * - Diffusion equation: ∂T/∂t = α ∇²T
     * - Dimensionless: [1/time] = [diffusivity] / [length²]
     * - Lattice diffusivity: alpha_lattice = alpha_phys * (dt / dx²)
     *
     * This is used for thermal diffusivity, mass diffusivity, etc.
     */
    HOST_DEVICE
    inline float diffusivityToLattice(float alpha_phys) const {
        return alpha_phys * dt_ / (dx_ * dx_);
    }

    /**
     * @brief Convert diffusivity from lattice units to physical units
     * @param alpha_lattice Diffusivity in lattice units [dimensionless]
     * @return Diffusivity [m²/s]
     *
     * Formula: alpha_phys = alpha_lattice * (dx² / dt)
     */
    HOST_DEVICE
    inline float diffusivityToPhysical(float alpha_lattice) const {
        return alpha_lattice * (dx_ * dx_) / dt_;
    }

    // ========================================================================
    // Viscosity Conversions (Kinematic)
    // ========================================================================

    /**
     * @brief Convert kinematic viscosity from physical units to lattice units
     * @param nu_phys Kinematic viscosity [m²/s]
     * @return Kinematic viscosity in lattice units [dimensionless]
     *
     * Formula: nu_lattice = nu_phys * (dt / dx²)
     *
     * Note: Kinematic viscosity has the same dimensions as diffusivity,
     *       so the conversion is identical.
     *
     * LBM relation: nu_lattice = (tau - 0.5) / 3, where tau is relaxation time
     */
    HOST_DEVICE
    inline float viscosityToLattice(float nu_phys) const {
        return diffusivityToLattice(nu_phys);
    }

    /**
     * @brief Convert kinematic viscosity from lattice units to physical units
     * @param nu_lattice Kinematic viscosity in lattice units [dimensionless]
     * @return Kinematic viscosity [m²/s]
     *
     * Formula: nu_phys = nu_lattice * (dx² / dt)
     */
    HOST_DEVICE
    inline float viscosityToPhysical(float nu_lattice) const {
        return diffusivityToPhysical(nu_lattice);
    }

    // ========================================================================
    // Time Conversions
    // ========================================================================

    /**
     * @brief Convert time from lattice timesteps to physical units
     * @param timesteps Number of lattice timesteps [dimensionless]
     * @return Time in physical units [s]
     *
     * Formula: t_phys = timesteps * dt
     */
    HOST_DEVICE
    inline float timeToPhysical(int timesteps) const {
        return static_cast<float>(timesteps) * dt_;
    }

    /**
     * @brief Convert time from physical units to lattice timesteps
     * @param t_phys Time in physical units [s]
     * @return Number of lattice timesteps [dimensionless]
     *
     * Formula: timesteps = t_phys / dt
     */
    HOST_DEVICE
    inline int timeToLattice(float t_phys) const {
        return static_cast<int>(t_phys / dt_ + 0.5f);  // Round to nearest
    }

    // ========================================================================
    // Getters
    // ========================================================================

    HOST_DEVICE
    inline float getDx() const { return dx_; }

    HOST_DEVICE
    inline float getDt() const { return dt_; }

    HOST_DEVICE
    inline float getRhoPhys() const { return rho_phys_; }

private:
    float dx_;       ///< Physical grid spacing [m]
    float dt_;       ///< Physical time step [s]
    float rho_phys_; ///< Reference density [kg/m³]
};

} // namespace core
} // namespace lbm
