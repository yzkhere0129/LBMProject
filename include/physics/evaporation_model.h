/**
 * @file evaporation_model.h
 * @brief Evaporation mass flux computation using Hertz-Knudsen equation
 *
 * This module computes evaporative mass flux at metal-vapor interfaces
 * using the Hertz-Knudsen model from kinetic theory of gases.
 *
 * Physical models:
 *
 * 1. Clausius-Clapeyron equation for saturation pressure:
 *    p_sat(T) = p_ref * exp(L_v * M / R * (1/T_ref - 1/T))
 *
 * 2. Hertz-Knudsen equation for mass flux:
 *    m_dot = (1 - beta_r) * p_sat(T) * sqrt(M / (2*pi*R*T))
 *
 * where:
 *    - beta_r: recondensation (sticking) coefficient (~0.82 for metals)
 *    - L_v: latent heat of vaporization [J/kg]
 *    - M: molar mass [kg/mol]
 *    - R: universal gas constant = 8.314 J/(mol.K)
 *    - T_ref, p_ref: reference point (typically boiling point)
 *
 * References:
 * - Anisimov, S. I. (1968): Vaporization of metals by laser radiation
 * - Knight, C. J. (1979): Theoretical modeling of rapid surface vaporization
 * - Tan, W. et al. (2013): Multi-scale modeling of solidification and microstructure
 */

#pragma once

#include <cuda_runtime.h>
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

/**
 * @brief Physical constants for evaporation calculations
 */
namespace EvaporationConstants {
    constexpr float R_GAS = 8.314f;           ///< Universal gas constant [J/(mol.K)]
    constexpr float PI = 3.14159265358979f;
    constexpr float TWO_PI = 6.28318530717959f;

    // Molar masses for common metals [kg/mol]
    constexpr float M_TI = 0.04788f;          ///< Titanium
    constexpr float M_FE = 0.05585f;          ///< Iron
    constexpr float M_AL = 0.02698f;          ///< Aluminum
    constexpr float M_NI = 0.05869f;          ///< Nickel
}

/**
 * @brief Evaporation model configuration
 */
struct EvaporationConfig {
    float sticking_coefficient = 0.18f;   ///< CRITICAL FIX (2025-11-27): Reduced from 0.82 to 0.18 to prevent excessive evaporation
    float ambient_pressure = 101325.0f;   ///< Ambient pressure [Pa]
    float min_temperature = 2000.0f;      ///< Min T for evaporation [K]
    float max_mass_flux = 1000.0f;        ///< Numerical limiter [kg/(m2.s)]

    EvaporationConfig() = default;
};

/**
 * @brief Evaporation model for metal vapor generation
 *
 * This class computes evaporative mass flux using the Hertz-Knudsen
 * equation with Clausius-Clapeyron saturation pressure.
 *
 * Typical values for Ti6Al4V at T_boil = 3560 K:
 *   - p_sat = 101325 Pa (1 atm at boiling point)
 *   - m_dot ~ 10-100 kg/(m2.s) (depending on T)
 *   - Q_evap ~ m_dot * L_v ~ 1e8 - 1e9 W/m2
 */
class EvaporationModel {
public:
    /**
     * @brief Constructor
     * @param material Material properties (for L_vaporization, T_boil, etc.)
     * @param config Evaporation configuration
     */
    EvaporationModel(const MaterialProperties& material,
                     const EvaporationConfig& config = EvaporationConfig());

    /**
     * @brief Constructor with direct parameters
     * @param T_boil Boiling temperature [K]
     * @param L_vaporization Latent heat of vaporization [J/kg]
     * @param molar_mass Molar mass [kg/mol]
     * @param sticking_coeff Recondensation coefficient [0-1]
     * @param ambient_pressure Ambient pressure [Pa]
     */
    EvaporationModel(float T_boil,
                     float L_vaporization,
                     float molar_mass,
                     float sticking_coeff = 0.82f,
                     float ambient_pressure = 101325.0f);

    /**
     * @brief Destructor
     */
    ~EvaporationModel() = default;

    // ========================================================================
    // Device-callable computation methods
    // ========================================================================

    /**
     * @brief Compute saturation pressure at temperature T
     * @param T Temperature [K]
     * @return Saturation pressure [Pa]
     *
     * Uses Clausius-Clapeyron:
     *   p_sat(T) = p_ref * exp(L_v * M / R * (1/T_ref - 1/T))
     */
    __host__ __device__ float computeSaturationPressure(float T) const;

    /**
     * @brief Compute evaporation mass flux at temperature T
     * @param T Temperature [K]
     * @return Mass flux [kg/(m2.s)] (positive = evaporation)
     *
     * Uses Hertz-Knudsen:
     *   m_dot = (1 - beta_r) * p_sat(T) * sqrt(M / (2*pi*R*T))
     */
    __host__ __device__ float computeMassFlux(float T) const;

    /**
     * @brief Compute evaporative cooling flux
     * @param T Temperature [K]
     * @return Heat flux [W/m2] (positive = energy leaving surface)
     *
     * Q = m_dot * L_vaporization
     */
    __host__ __device__ float computeHeatFlux(float T) const;

    // ========================================================================
    // Field computation methods (CUDA kernel wrappers)
    // ========================================================================

    /**
     * @brief Compute saturation pressure field
     * @param temperature Temperature field [K]
     * @param saturation_pressure Output: p_sat field [Pa]
     * @param nx, ny, nz Grid dimensions
     */
    void computeSaturationPressureField(
        const float* temperature,
        float* saturation_pressure,
        int nx, int ny, int nz) const;

    /**
     * @brief Compute evaporation mass flux field
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param mass_flux Output: mass flux field [kg/(m2.s)]
     * @param nx, ny, nz Grid dimensions
     *
     * Mass flux is only computed at interface cells (0 < f < 1)
     */
    void computeMassFluxField(
        const float* temperature,
        const float* fill_level,
        float* mass_flux,
        int nx, int ny, int nz) const;

    /**
     * @brief Compute both saturation pressure and mass flux
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param saturation_pressure Output: p_sat field [Pa]
     * @param mass_flux Output: mass flux field [kg/(m2.s)]
     * @param nx, ny, nz Grid dimensions
     */
    void computeFields(
        const float* temperature,
        const float* fill_level,
        float* saturation_pressure,
        float* mass_flux,
        int nx, int ny, int nz) const;

    // ========================================================================
    // Parameter access and modification
    // ========================================================================

    float getStickingCoefficient() const { return beta_r_; }
    void setStickingCoefficient(float beta) { beta_r_ = beta; }

    float getBoilingTemperature() const { return T_boil_; }
    float getLatentHeat() const { return L_vap_; }
    float getMolarMass() const { return M_molar_; }
    float getAmbientPressure() const { return p_ambient_; }

    /**
     * @brief Get reference parameters for diagnostics
     * @return p_ref used in Clausius-Clapeyron (equals ambient pressure)
     */
    float getReferencePressure() const { return p_ambient_; }
    float getReferenceTemperature() const { return T_boil_; }

private:
    // Material parameters
    float T_boil_;          ///< Boiling temperature [K]
    float L_vap_;           ///< Latent heat of vaporization [J/kg]
    float M_molar_;         ///< Molar mass [kg/mol]

    // Model parameters
    float beta_r_;          ///< Recondensation (sticking) coefficient [0-1]
    float p_ambient_;       ///< Ambient/reference pressure [Pa]
    float T_min_;           ///< Minimum temperature for evaporation [K]
    float m_dot_max_;       ///< Maximum mass flux limiter [kg/(m2.s)]

    // Precomputed constants for performance
    float clausius_exponent_;  ///< L_vap * M_molar / R_gas
    float hertz_prefactor_;    ///< sqrt(M_molar / (2*pi*R_gas))
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief CUDA kernel for saturation pressure computation
 */
__global__ void computeSaturationPressureKernel(
    const float* __restrict__ temperature,
    float* __restrict__ saturation_pressure,
    float T_boil,
    float p_ambient,
    float clausius_exponent,
    int num_cells);

/**
 * @brief CUDA kernel for evaporation mass flux computation
 *
 * Only computes at interface cells where 0 < fill_level < 1
 */
__global__ void computeEvaporationMassFluxKernel(
    const float* __restrict__ temperature,
    const float* __restrict__ fill_level,
    float* __restrict__ mass_flux,
    float* __restrict__ saturation_pressure,
    float T_boil,
    float L_vap,
    float M_molar,
    float p_ambient,
    float beta_r,
    float T_min,
    float m_dot_max,
    float clausius_exponent,
    float hertz_prefactor,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for combined p_sat and m_dot computation
 *
 * Fused kernel for better performance (single pass over data)
 */
__global__ void computeEvaporationFieldsKernel(
    const float* __restrict__ temperature,
    const float* __restrict__ fill_level,
    float* __restrict__ saturation_pressure,
    float* __restrict__ mass_flux,
    float T_boil,
    float L_vap,
    float M_molar,
    float p_ambient,
    float beta_r,
    float T_min,
    float m_dot_max,
    float clausius_exponent,
    float hertz_prefactor,
    int nx, int ny, int nz);

// ============================================================================
// Inline device function implementations
// ============================================================================

__host__ __device__ inline float EvaporationModel::computeSaturationPressure(float T) const {
    if (T < T_min_) {
        return 0.0f;  // No significant vapor pressure below T_min
    }

    // Clausius-Clapeyron: p_sat = p_ref * exp(L*M/R * (1/T_ref - 1/T))
    float exponent = clausius_exponent_ * (1.0f / T_boil_ - 1.0f / T);
    float p_sat = p_ambient_ * expf(exponent);

    return p_sat;
}

__host__ __device__ inline float EvaporationModel::computeMassFlux(float T) const {
    if (T < T_min_) {
        return 0.0f;
    }

    float p_sat = computeSaturationPressure(T);

    // Hertz-Knudsen: m_dot = (1 - beta_r) * p_sat * sqrt(M / (2*pi*R*T))
    float m_dot = (1.0f - beta_r_) * p_sat * hertz_prefactor_ / sqrtf(T);

    // Apply limiter for numerical stability
    return fminf(m_dot, m_dot_max_);
}

__host__ __device__ inline float EvaporationModel::computeHeatFlux(float T) const {
    return computeMassFlux(T) * L_vap_;
}

} // namespace physics
} // namespace lbm
