/**
 * @file thermal_lbm.h
 * @brief Thermal Lattice Boltzmann Method solver
 *
 * This file implements the double distribution function approach for
 * solving coupled fluid flow and heat transfer problems.
 *
 * Reference:
 * - He, X., Chen, S., & Doolen, G. D. (1998). A novel thermal model for
 *   the lattice Boltzmann method in incompressible limit. Journal of
 *   Computational Physics, 146(1), 282-300.
 */

#pragma once

#include <cuda_runtime.h>
#include "physics/lattice_d3q7.h"
#include "physics/phase_change.h"
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

/**
 * @brief Thermal LBM solver using D3Q7 lattice
 *
 * This class implements the thermal LBM solver for heat transfer problems.
 * It uses the D3Q7 lattice for temperature field evolution and can be
 * coupled with the fluid solver (D3Q19) for convection-diffusion problems.
 */
class ThermalLBM {
public:
    /**
     * @brief Constructor (deprecated - for backward compatibility)
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param thermal_diffusivity Thermal diffusivity (α = k/(ρ*c_p)) [m²/s, physical units]
     * @param density Material density (ρ) [kg/m³]
     * @param specific_heat Material specific heat (cp) [J/(kg·K)]
     * @param dt Time step [s] (required for lattice unit conversion)
     * @param dx Lattice spacing [m] (required for lattice unit conversion)
     */
    ThermalLBM(int nx, int ny, int nz, float thermal_diffusivity,
               float density = 8000.0f, float specific_heat = 500.0f,
               float dt = 1.0e-7f, float dx = 2.0e-6f);

    /**
     * @brief Constructor with phase change support
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param material Material properties (includes phase change parameters)
     * @param thermal_diffusivity Thermal diffusivity (α = k/(ρ*c_p)) [m²/s, physical units]
     * @param enable_phase_change Enable phase change tracking
     * @param dt Time step [s] (required for lattice unit conversion)
     * @param dx Lattice spacing [m] (required for lattice unit conversion)
     */
    ThermalLBM(int nx, int ny, int nz,
               const MaterialProperties& material,
               float thermal_diffusivity,
               bool enable_phase_change = true,
               float dt = 1.0e-7f, float dx = 2.0e-6f);

    /**
     * @brief Destructor
     */
    ~ThermalLBM();

    /**
     * @brief Initialize temperature field
     * @param initial_temp Uniform initial temperature
     */
    void initialize(float initial_temp);

    /**
     * @brief Initialize temperature field with custom distribution
     * @param temp_field Host array of temperature values (size nx*ny*nz)
     */
    void initialize(const float* temp_field);

    /**
     * @brief Perform BGK collision step for thermal distribution
     * @param ux Velocity field x-component (can be nullptr for pure diffusion)
     * @param uy Velocity field y-component (can be nullptr for pure diffusion)
     * @param uz Velocity field z-component (can be nullptr for pure diffusion)
     */
    void collisionBGK(const float* ux = nullptr,
                      const float* uy = nullptr,
                      const float* uz = nullptr);

    /**
     * @brief Perform streaming step for thermal distribution
     */
    void streaming();

    /**
     * @brief Apply boundary conditions
     * @param boundary_type Type of boundary (0=periodic, 1=constant T, 2=adiabatic)
     * @param boundary_value Temperature value for constant T boundaries
     */
    void applyBoundaryConditions(int boundary_type, float boundary_value = 0.0f);

    /**
     * @brief Apply thermal boundary condition to a single face
     * @param face Face index (0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max)
     * @param bc_type Type of boundary condition (PERIODIC, ADIABATIC, DIRICHLET, CONVECTIVE, RADIATION)
     * @param dt Time step [s]
     * @param dx Grid spacing [m]
     * @param dirichlet_T Temperature for DIRICHLET faces [K]
     * @param h_conv Convective coefficient for CONVECTIVE faces [W/(m^2*K)]
     * @param T_inf Far-field temperature for CONVECTIVE faces [K]
     * @param emissivity Emissivity for RADIATION faces [0-1]
     * @param T_ambient Ambient temperature for RADIATION faces [K]
     */
    void applyFaceThermalBC(int face, int bc_type,
                            float dt, float dx,
                            float dirichlet_T = 300.0f,
                            float h_conv = 1000.0f,
                            float T_inf = 300.0f,
                            float emissivity = 0.3f,
                            float T_ambient = 300.0f);

    /**
     * @brief Add heat source term
     * @param heat_source Device array of heat source values (W/m³)
     * @param dt Time step
     */
    void addHeatSource(const float* heat_source, float dt);

    /**
     * @brief Apply radiation boundary condition
     * @param dt Time step
     * @param dx Grid spacing in physical units
     * @param epsilon Emissivity (0-1)
     * @param T_ambient Ambient temperature [K]
     */
    void applyRadiationBC(float dt, float dx, float epsilon = 0.35f, float T_ambient = 300.0f);

    /**
     * @brief Apply substrate cooling boundary condition at bottom surface (z=0)
     * @param dt Time step [s]
     * @param dx Grid spacing [m]
     * @param h_conv Convective heat transfer coefficient [W/(m²·K)]
     * @param T_substrate Substrate temperature [K]
     * @note Implements convective BC: q = h_conv * (T_cell - T_substrate)
     */
    void applySubstrateCoolingBC(float dt, float dx, float h_conv, float T_substrate);

    /**
     * @brief Apply evaporation cooling at VOF interface cells
     * @param J_evap Evaporation mass flux field [kg/(m²·s)]
     * @param fill_level VOF fill level field (0-1)
     * @param dt Time step [s]
     * @param dx Grid spacing [m]
     * @note Removes latent heat Q = J_evap * L_vap from interface cells
     * @note This provides temperature capping at boiling point through physics
     */
    void applyEvaporationCooling(const float* J_evap, const float* fill_level, float dt, float dx,
                                 float cooling_factor = 1.0f);

    /**
     * @brief Compute total substrate cooling power
     * @param dx Grid spacing [m]
     * @param h_conv Convective heat transfer coefficient [W/(m²·K)]
     * @param T_substrate Substrate temperature [K]
     * @return Total substrate cooling power [W]
     * @note Computes P_substrate = Σ(q_conv * A_cell) at bottom surface (z=0)
     */
    float computeSubstratePower(float dx, float h_conv, float T_substrate) const;

    /**
     * @brief Compute power removed by the temperature cap [W]
     *
     * The hard temperature cap (T_boil - 100K) removes thermal energy that
     * represents effective evaporative cooling. This method sums the energy
     * removed and converts to power: P = Σ(ρ·cp·ΔT·dx³) / dt
     *
     * @param dx Grid spacing [m]
     * @param dt Time step [s]
     * @return Power removed by cap this step [W]
     */
    float computeCapPower(float dx, float dt) const;

    /**
     * @brief Apply physics-based temperature safety cap
     *
     * Clamps temperature at T_vaporization for all cells. This prevents
     * unphysical temperature runaway in laser melting simulations when
     * evaporative cooling (which requires VOF) is not available.
     *
     * Physical justification: In reality, strong surface evaporation
     * self-limits temperature to near T_boil. When VOF is disabled,
     * this explicit cap provides the same physics constraint.
     *
     * @note This is a no-op if material properties are not set
     * @note Does NOT require VOF -- can be called in thermal-only mode
     */
    void applyTemperatureSafetyCap();

    /**
     * @brief Compute temperature field from distribution functions
     */
    void computeTemperature();

    /**
     * @brief Get temperature field (device pointer)
     * @return Device pointer to temperature array
     */
    float* getTemperature() { return d_temperature; }

    /**
     * @brief Get temperature field (const device pointer)
     * @return Const device pointer to temperature array
     */
    const float* getTemperature() const { return d_temperature; }

    /// Get current (post-streaming) D3Q7 distribution pointer
    float* getDistributionSrc() { return d_g_src; }

    /**
     * @brief Copy temperature to host
     * @param host_temp Host array to store temperature (must be pre-allocated)
     */
    void copyTemperatureToHost(float* host_temp) const;

    /**
     * @brief Compute thermal relaxation time from diffusivity
     * @param alpha Thermal diffusivity
     * @param dx Lattice spacing
     * @param dt Time step
     * @return Thermal relaxation time tau_T
     */
    static float computeThermalTau(float alpha, float dx = 1.0f, float dt = 1.0f);

    /**
     * @brief Get domain dimensions
     */
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

    /**
     * @brief Get thermal relaxation time
     */
    float getThermalTau() const { return tau_T_; }

    /**
     * @brief Set surface emissivity for radiation boundary condition
     * @param eps Emissivity value (0-1)
     */
    void setEmissivity(float eps) { emissivity_ = eps; }

    /**
     * @brief Enable periodic boundary in z-direction (for quasi-2D simulations)
     */
    void setZPeriodic(bool enable) { z_periodic_ = enable; }

    /**
     * @brief Get liquid fraction field (device pointer)
     * @return Device pointer to liquid fraction array (nullptr if phase change disabled)
     */
    float* getLiquidFraction();

    /**
     * @brief Get liquid fraction field (const device pointer)
     * @return Const device pointer to liquid fraction array (nullptr if phase change disabled)
     */
    const float* getLiquidFraction() const;

    /**
     * @brief Copy liquid fraction to host
     * @param host_fl Host array (must be pre-allocated)
     */
    void copyLiquidFractionToHost(float* host_fl) const;

    /**
     * @brief Check if phase change is enabled
     */
    bool hasPhaseChange() const { return phase_solver_ != nullptr; }

    /**
     * @brief Get phase change solver (for solidification shrinkage coupling)
     * @return Pointer to phase change solver (nullptr if disabled)
     */
    PhaseChangeSolver* getPhaseChangeSolver() { return phase_solver_; }

    /**
     * @brief Apply latent heat correction for phase change
     * @param dt Time step [s]
     *
     * This method implements the source term approach for phase change:
     * 1. Store current liquid fraction
     * 2. Update liquid fraction based on new temperature
     * 3. Compute ΔT = -L/(ρ·cp) · Δfl (latent heat sink/source)
     * 4. Apply correction to temperature field
     *
     * Must be called after computeTemperature() in each time step.
     */
    void applyPhaseChangeCorrection(float dt);

    // ========================================================================
    // Energy Diagnostics (for energy conservation verification)
    // ========================================================================

    /**
     * @brief Compute total evaporation power at all surface cells
     * @param fill_level VOF fill level field (to identify surface cells)
     * @param dx Grid spacing [m]
     * @return Total evaporation power [W]
     * @note This computes P_evap = Σ(m_dot * L_v) at all cells with T > T_boil
     */
    float computeEvaporationPower(const float* fill_level, float dx) const;

    /**
     * @brief Compute total radiation power at all surface cells
     * @param fill_level VOF fill level field (to identify surface cells)
     * @param dx Grid spacing [m]
     * @param epsilon Emissivity [0-1]
     * @param T_ambient Ambient temperature [K]
     * @return Total radiation power [W]
     * @note This computes P_rad = Σ(ε*σ*A*(T^4 - T_amb^4)) at all surface cells
     */
    float computeRadiationPower(const float* fill_level, float dx,
                                float epsilon, float T_ambient) const;

    /**
     * @brief Compute total thermal energy stored in the domain
     * @param dx Grid spacing [m]
     * @return Total internal energy [J]
     * @note E = Σ(ρ * c_p * T * V) + Σ(f_l * ρ * L_f * V) (sensible + latent)
     */
    float computeTotalThermalEnergy(float dx) const;

    /**
     * @brief Get material properties (for energy diagnostics)
     */
    const MaterialProperties& getMaterialProperties() const { return material_; }

    /**
     * @brief Get density [kg/m³]
     */
    float getDensity() const { return rho_; }

    /**
     * @brief Get specific heat [J/(kg·K)]
     */
    float getSpecificHeat() const { return cp_; }

    // ========================================================================
    // Evaporation Mass Flux (for VOF mass coupling)
    // ========================================================================

    /**
     * @brief Compute evaporation mass flux field at VOF interface
     * @param d_J_evap Output: device array for mass flux [kg/(m^2*s)]
     * @param fill_level VOF fill level field for interface detection (device pointer)
     * @note Uses Hertz-Knudsen-Langmuir model: J = alpha * P_sat / sqrt(2*pi*R*T/M)
     * @note Only interface cells (0.01 < f < 0.99) with T > T_boil have non-zero flux
     * @note FIX: Now evaporates at actual interface, not fixed z=nz-1
     */
    void computeEvaporationMassFlux(float* d_J_evap, const float* fill_level) const;

private:
    // Domain dimensions
    int nx_, ny_, nz_;
    int num_cells_;

    // Lattice parameters
    float dt_;              ///< Time step [s]
    float dx_;              ///< Lattice spacing [m]

    // Thermal parameters
    float tau_T_;           ///< Thermal relaxation time
    float omega_T_;         ///< Thermal relaxation frequency (1/tau_T)
    float thermal_diff_physical_;    ///< Thermal diffusivity [m²/s, physical units]
    float thermal_diff_lattice_;     ///< Thermal diffusivity [dimensionless, lattice units]
    float rho_;             ///< Material density [kg/m³]
    float cp_;              ///< Specific heat capacity [J/(kg·K)]
    float emissivity_;      ///< Surface emissivity (0-1)
    float T_initial_;       ///< Initial temperature for energy reference [K]
    bool z_periodic_;       ///< Use periodic BC in z-direction (for quasi-2D)

    // Phase change support (optional)
    PhaseChangeSolver* phase_solver_;  ///< Phase change solver (nullptr if disabled)
    MaterialProperties material_;       ///< Material properties (only used if phase change enabled)
    bool has_material_;                 ///< Flag indicating if material properties are set

    // Device memory for thermal distribution functions
    // Layout: SoA (Structure of Arrays) — index as g[q * num_cells + cell_idx]
    // This matches the D3Q19 fluid solver layout and enables coalesced GPU access:
    // consecutive threads read g[q * num_cells + 0], g[q * num_cells + 1], ...
    float* d_g_src;         ///< Source distribution (before streaming)
    float* d_g_dst;         ///< Destination distribution (after streaming)
    float* d_temperature;   ///< Temperature field
    float* d_cap_energy_removed_ = nullptr;  ///< Per-cell energy removed by T cap [K]
    const float* d_vof_fill_level_ = nullptr;  ///< VOF field for ESM gas masking (external, not owned)

public:
    /// Set VOF fill_level pointer for ESM gas masking (called by MultiphysicsSolver)
    void setVOFFillLevel(const float* fill_level) { d_vof_fill_level_ = fill_level; }

    /// Disable the hard temperature cap at T_boil-100K for keyhole mode.
    /// When recoil pressure is active, evaporation physics naturally limits T.
    bool skip_temperature_cap_ = false;
    void setSkipTemperatureCap(bool skip) { skip_temperature_cap_ = skip; }

private:
    // Utility functions
    void allocateMemory();
    void freeMemory();
    void swapDistributions();
};

// CUDA kernels for thermal LBM
//
// Distribution function memory layout: SoA — g[q * num_cells + cell_idx]
// where num_cells = nx * ny * nz and cell_idx = x + y*nx + z*nx*ny.
// This enables coalesced memory access: threads in a warp read
// g[q * num_cells + idx], g[q * num_cells + idx+1], ... (stride-1).

/**
 * @brief Enthalpy Source Term kernel for phase change (Jiaung 2001)
 *
 * After T* = Σg_q, enforces enthalpy conservation:
 *   H = cp·T* + fl_old·L → decode (T_new, fl_new) → correct g
 *
 * @param g Distribution functions (SoA layout, modified in place)
 * @param temperature Temperature field (modified: T* → T_new)
 * @param liquid_fraction Current liquid fraction (modified: → fl_new)
 * @param liquid_fraction_prev Previous step's liquid fraction (fl_old)
 * @param material Material properties (cp_solid, T_solidus, T_liquidus, L_fusion)
 * @param num_cells Total number of cells
 */
__global__ void enthalpySourceTermKernel(
    float* g,
    float* temperature,
    float* liquid_fraction,
    const float* liquid_fraction_prev,
    MaterialProperties material,
    int num_cells);

/**
 * @brief CUDA kernel for thermal BGK collision with apparent heat capacity
 * @param g_src Distribution functions (SoA layout: g_src[q * num_cells + idx])
 * @param temperature Temperature field
 * @param ux, uy, uz Velocity components (can be nullptr)
 * @param omega_T Base thermal relaxation frequency
 * @param nx, ny, nz Grid dimensions
 * @param material Material properties (for apparent heat capacity)
 * @param dt Time step [s]
 * @param dx Grid spacing [m]
 * @param use_apparent_cp If true, compute omega_T per-cell using apparent heat capacity
 */
__global__ void thermalBGKCollisionKernel(
    float* g_src,
    const float* temperature,
    const float* ux,
    const float* uy,
    const float* uz,
    float omega_T,
    int nx, int ny, int nz,
    MaterialProperties material,
    float dt,
    float dx,
    bool use_apparent_cp);

/**
 * @brief CUDA kernel for thermal streaming
 */
__global__ void thermalStreamingKernel(
    const float* g_src,
    float* g_dst,
    int nx, int ny, int nz,
    float emissivity);

/**
 * @brief CUDA kernel for computing temperature from distribution
 */
__global__ void computeTemperatureKernel(
    const float* g,
    float* temperature,
    int num_cells);

/**
 * @brief CUDA kernel for applying constant temperature boundary
 */
__global__ void applyConstantTemperatureBoundary(
    float* g,
    float* temperature,
    float T_boundary,
    int nx, int ny, int nz,
    int boundary_face); // 0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max

/**
 * @brief CUDA kernel for applying adiabatic (zero-flux) boundary
 */
__global__ void applyAdiabaticBoundary(
    float* g,
    int nx, int ny, int nz,
    int boundary_face);

/**
 * @brief CUDA kernel for adding heat source
 */
__global__ void addHeatSourceKernel(
    float* g,
    const float* heat_source,
    const float* temperature,
    float dt,
    float omega_T,
    MaterialProperties material,
    int num_cells);

/**
 * @brief CUDA kernel for applying radiation boundary condition
 */
__global__ void applyRadiationBoundaryCondition(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float epsilon,
    MaterialProperties material,
    float T_ambient);

/**
 * @brief CUDA kernel for applying substrate cooling boundary condition
 * @param g Temperature distribution functions [device]
 * @param temperature Current temperature field [device]
 * @param nx, ny, nz Grid dimensions
 * @param dx Lattice spacing [m]
 * @param dt Time step [s]
 * @param h_conv Convective heat transfer coefficient [W/(m²·K)]
 * @param T_substrate Substrate temperature [K]
 * @param rho Density [kg/m³]
 * @param cp Specific heat [J/(kg·K)]
 */
__global__ void applySubstrateCoolingKernel(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float h_conv,
    float T_substrate,
    float rho,
    float cp);

/**
 * @brief CUDA kernel for convective cooling on an arbitrary face
 * @param g Temperature distribution functions [device]
 * @param temperature Current temperature field [device]
 * @param nx, ny, nz Grid dimensions
 * @param dx Lattice spacing [m]
 * @param dt Time step [s]
 * @param h_conv Convective heat transfer coefficient [W/(m^2*K)]
 * @param T_inf Far-field temperature [K]
 * @param rho Density [kg/m^3]
 * @param cp Specific heat [J/(kg*K)]
 * @param boundary_face Face index (0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max)
 */
__global__ void applyConvectiveBCKernel(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float h_conv,
    float T_inf,
    float rho,
    float cp,
    int boundary_face);

/**
 * @brief CUDA kernel for radiation cooling on an arbitrary face
 * @param g Temperature distribution functions [device]
 * @param temperature Current temperature field [device]
 * @param nx, ny, nz Grid dimensions
 * @param dx Lattice spacing [m]
 * @param dt Time step [s]
 * @param epsilon Emissivity [0-1]
 * @param material Material properties (for rho, cp)
 * @param T_ambient Ambient temperature [K]
 * @param boundary_face Face index (0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max)
 */
__global__ void applyRadiationBCFaceKernel(
    float* g,
    const float* temperature,
    int nx, int ny, int nz,
    float dx,
    float dt,
    float epsilon,
    MaterialProperties material,
    float T_ambient,
    int boundary_face);

/**
 * @brief CUDA kernel for computing evaporation mass flux at VOF interface
 * @param temperature Temperature field [K]
 * @param fill_level VOF fill level field for interface detection
 * @param J_evap Output: evaporation mass flux [kg/(m^2*s)]
 * @param material Material properties (T_vaporization, L_vaporization)
 * @param nx, ny, nz Grid dimensions
 * @note FIX: Now evaporates at interface cells (0.01 < f < 0.99), not fixed z=nz-1
 */
__global__ void computeEvaporationMassFluxKernel(
    const float* temperature,
    const float* fill_level,
    float* J_evap,
    MaterialProperties material,
    int nx, int ny, int nz);

} // namespace physics
} // namespace lbm