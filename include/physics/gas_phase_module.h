/**
 * @file gas_phase_module.h
 * @brief Gas phase management for multiphase LPBF simulations
 *
 * This module handles all gas-phase physics including:
 * - Evaporation/condensation at gas-liquid interface
 * - Recoil pressure from evaporating metal vapor
 * - Optional explicit gas flow computation
 * - Gas-phase thermal effects
 *
 * Architecture:
 * - Modular: Evaporation, recoil pressure as independent sub-components
 * - Configurable: IMPLICIT (efficient) or EXPLICIT (high-fidelity) modes
 * - Extensible: Easy to add condensation, vapor deposition in future
 *
 * Physical models:
 * - Evaporation: Hertz-Knudsen equation with Clausius-Clapeyron p_sat
 * - Recoil pressure: P_r = 0.54 * p_sat (Knight 1979)
 * - Mass source: -m_dot / (rho_liquid * dx) for VOF advection
 *
 * References:
 * - Khairallah et al. (2016): Keyhole physics in LPBF
 * - Anisimov & Khokhlov (1995): Gas dynamics of laser ablation
 * - Knight (1979): Theoretical modeling of rapid surface vaporization
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

// Forward declarations
class EvaporationModel;
class RecoilPressure;
class GasFlowSolver;

/**
 * @brief Gas phase simulation mode
 */
enum class GasMode {
    IMPLICIT,  ///< Gas implicitly handled by VOF (default, efficient)
    EXPLICIT   ///< Full gas phase LBM solver (high-fidelity)
};

/**
 * @brief Configuration for gas phase module
 */
struct GasPhaseConfig {
    // Mode selection
    GasMode mode = GasMode::IMPLICIT;

    // Gas properties (Argon at 1 atm default)
    float gas_density = 1.2f;               ///< Ambient gas density [kg/m3]
    float gas_viscosity = 2.2e-5f;          ///< Dynamic viscosity [Pa.s]
    float gas_thermal_conductivity = 0.018f;///< k_gas [W/(m.K)]
    float gas_specific_heat = 520.0f;       ///< cp_gas [J/(kg.K)]

    // Evaporation parameters
    bool enable_evaporation = true;
    float sticking_coefficient = 0.82f;     ///< beta_r: fraction that recondenses
    float ambient_pressure = 101325.0f;     ///< P_ambient [Pa]

    // Recoil pressure parameters
    bool enable_recoil_pressure = true;
    float recoil_coefficient = 0.54f;       ///< C_r: P_recoil = C_r * p_sat
    float recoil_smoothing_width = 2.0f;    ///< Interface smoothing [cells]

    // Explicit mode only
    bool enable_gas_thermal = false;        ///< Enable gas-phase thermal solver
    int gas_lbm_subcycles = 1;              ///< Gas LBM subcycles

    // Numerical parameters
    float evap_mass_limit = 0.1f;           ///< Max mass fraction change/step
    float min_evap_temperature = 2000.0f;   ///< Minimum T for evaporation [K]

    GasPhaseConfig() = default;
};

/**
 * @brief Main gas phase module
 *
 * Orchestrates all gas-related physics:
 * - In IMPLICIT mode: Computes evaporation and recoil as source terms
 * - In EXPLICIT mode: Runs full gas-phase LBM solver
 *
 * Integration with MultiphysicsSolver:
 * - Called after thermal solve (needs T field)
 * - Before VOF advection (provides mass source)
 * - Before fluid solve (provides recoil force)
 */
class GasPhaseModule {
public:
    /**
     * @brief Constructor
     * @param nx, ny, nz Grid dimensions
     * @param config Gas phase configuration
     * @param material Metal material properties (for evaporation)
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     */
    GasPhaseModule(int nx, int ny, int nz,
                   const GasPhaseConfig& config,
                   const MaterialProperties& material,
                   float dx, float dt);

    /**
     * @brief Destructor
     */
    ~GasPhaseModule();

    /**
     * @brief Initialize gas phase fields
     * @param fill_level VOF fill level (1=liquid, 0=gas)
     * @param temperature Initial temperature field
     */
    void initialize(const float* fill_level, const float* temperature);

    // ========================================================================
    // Core computation methods
    // ========================================================================

    /**
     * @brief Compute evaporation mass flux at interface
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param interface_normal Surface normal vectors
     * @param d_mass_flux Output: mass flux [kg/(m2.s)]
     *
     * Uses Hertz-Knudsen equation:
     *   m_dot = (1 - beta_r) * p_sat(T) * sqrt(M / (2*pi*R*T))
     */
    void computeEvaporationMassFlux(
        const float* temperature,
        const float* fill_level,
        const float3* interface_normal,
        float* d_mass_flux) const;

    /**
     * @brief Compute evaporation (convenience: uses internal buffer)
     */
    void computeEvaporationMassFlux(
        const float* temperature,
        const float* fill_level,
        const float3* interface_normal);

    /**
     * @brief Compute recoil pressure force
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param interface_normal Surface normal vectors
     * @param force_x, force_y, force_z Output: recoil force [N/m3]
     *
     * Recoil pressure: P_r = C_r * p_sat(T)
     * Force: F = P_r * n * |grad(f)| / h_interface
     */
    void computeRecoilForce(
        const float* temperature,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z) const;

    /**
     * @brief Add recoil force to existing force field
     *
     * Convenience method for integration with computeTotalForce()
     */
    void addRecoilForce(
        const float* temperature,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z) const;

    /**
     * @brief Compute evaporative cooling heat sink
     * @param d_mass_flux Evaporation mass flux [kg/(m2.s)]
     * @param d_heat_sink Output: volumetric heat sink [W/m3]
     *
     * Q_evap = m_dot * L_vaporization (negative = cooling)
     */
    void computeEvaporativeCooling(
        const float* d_mass_flux,
        float* d_heat_sink) const;

    /**
     * @brief Compute evaporative cooling (convenience: uses internal buffers)
     */
    void computeEvaporativeCooling();

    /**
     * @brief Get evaporation mass source for VOF advection
     * @return Device pointer to mass source term [kg/(m3.s)]
     *
     * For VOF equation: df/dt + div(f*u) = -m_dot / (rho_liquid * dx)
     */
    const float* getMassSource() const { return d_mass_source_; }

    /**
     * @brief Get evaporation mass flux field
     * @return Device pointer to mass flux [kg/(m2.s)]
     */
    const float* getMassFlux() const { return d_mass_flux_; }

    /**
     * @brief Get evaporative cooling heat sink field
     * @return Device pointer to heat sink [W/m3]
     */
    const float* getHeatSink() const { return d_heat_sink_; }

    /**
     * @brief Get saturation pressure field
     * @return Device pointer to p_sat(T) [Pa]
     */
    const float* getSaturationPressure() const { return d_saturation_pressure_; }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /**
     * @brief Get total evaporated mass for conservation check
     * @return Total mass evaporated since initialization [kg]
     */
    float getTotalEvaporatedMass() const;

    /**
     * @brief Get total evaporation power for energy balance
     * @return Evaporation power [W]
     */
    float getEvaporationPower() const;

    /**
     * @brief Get maximum recoil pressure for diagnostics
     * @return Max P_recoil [Pa]
     */
    float getMaxRecoilPressure() const;

    /**
     * @brief Get maximum evaporation mass flux
     * @return Max m_dot [kg/(m2.s)]
     */
    float getMaxMassFlux() const;

    // ========================================================================
    // Explicit mode methods (GasMode::EXPLICIT only)
    // ========================================================================

    /**
     * @brief Perform gas phase LBM step (EXPLICIT mode only)
     * @param liquid_velocity_x, y, z Liquid velocity at interface
     * @param dt Time step
     */
    void stepGasFlow(const float* liquid_velocity_x,
                     const float* liquid_velocity_y,
                     const float* liquid_velocity_z,
                     float dt);

    /**
     * @brief Get gas velocity field (EXPLICIT mode only)
     * @return Device pointer to gas velocity component (nullptr in IMPLICIT)
     */
    const float* getGasVelocityX() const;
    const float* getGasVelocityY() const;
    const float* getGasVelocityZ() const;

    // ========================================================================
    // Configuration access
    // ========================================================================

    const GasPhaseConfig& getConfig() const { return config_; }
    GasMode getMode() const { return config_.mode; }
    bool isExplicitMode() const { return config_.mode == GasMode::EXPLICIT; }
    bool isEvaporationEnabled() const { return config_.enable_evaporation; }
    bool isRecoilPressureEnabled() const { return config_.enable_recoil_pressure; }

    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

private:
    // Configuration
    GasPhaseConfig config_;
    MaterialProperties material_;
    int nx_, ny_, nz_;
    int num_cells_;
    float dx_, dt_;

    // Sub-modules
    std::unique_ptr<EvaporationModel> evaporation_;
    std::unique_ptr<RecoilPressure> recoil_;
    std::unique_ptr<GasFlowSolver> gas_flow_;  // Only for EXPLICIT mode

    // Device memory
    float* d_mass_flux_;            ///< Evaporation mass flux [kg/(m2.s)]
    float* d_mass_source_;          ///< VOF mass source term [kg/(m3.s)]
    float* d_heat_sink_;            ///< Evaporative cooling [W/m3]
    float* d_saturation_pressure_;  ///< p_sat(T) [Pa]

    // Cumulative tracking
    mutable float total_evaporated_mass_;
    mutable float cumulative_evap_energy_;

    // Internal methods
    void allocateMemory();
    void freeMemory();
    void computeMassSourceFromFlux();  ///< Convert flux to VOF source
};

} // namespace physics
} // namespace lbm
