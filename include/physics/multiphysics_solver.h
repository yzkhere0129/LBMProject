/**
 * @file multiphysics_solver.h
 * @brief Multiphysics solver for coupled thermal-fluid-VOF-Marangoni simulation
 *
 * This file implements the full coupling of multiple physics modules for
 * metal additive manufacturing simulations (LPBF, DED, LIFT).
 *
 * Architecture:
 * - Modular: Each physics module (thermal, fluid, vof, marangoni) is independent
 * - Testable: Each coupling step can be tested separately
 * - Extensible: Easy to add new physics (e.g., recoil pressure in Phase 7)
 * - Configurable: Enable/disable physics via configuration flags
 * - Observable: Provides diagnostic outputs for validation
 *
 * Integration Steps:
 * - Step 1: Static temperature + Marangoni-driven fluid
 * - Step 2: Dynamic thermal diffusion + Marangoni-driven fluid
 * - Step 3: Add VOF advection with subcycling
 * - Step 4: Add laser heat source
 *
 * Physical coupling:
 * - Thermal → Marangoni: Temperature gradient drives surface forces
 * - Marangoni → Fluid: Surface forces accelerate liquid
 * - Fluid → VOF: Velocity field advects interface
 * - VOF → All: Interface reconstruction provides normals and curvature
 * - Laser → Thermal: Volumetric heat source
 * - SurfaceTension → Fluid: Curvature-driven forces
 *
 * References:
 * - Panwisawas et al. (2017): Mesoscale modelling of selective laser melting
 * - Khairallah et al. (2016): Laser powder-bed fusion physics
 * - walberla free surface framework: apps/showcases/FreeSurface/
 */

#pragma once

#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <vector>

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/vof_solver.h"
#include "physics/marangoni.h"
#include "physics/surface_tension.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "physics/recoil_pressure.h"
#include "physics/force_accumulator.h"
#include "diagnostics/energy_balance.h"
#include "core/unit_converter.h"

namespace lbm {
namespace physics {

/**
 * @brief Configuration for MultiphysicsSolver
 *
 * This structure holds all configuration parameters for the multiphysics
 * simulation, including physics flags, numerical parameters, and material
 * properties.
 */
struct MultiphysicsConfig {
    // Domain parameters
    int nx, ny, nz;           ///< Grid dimensions
    float dx;                 ///< Lattice spacing [m]

    // Physics enable flags
    bool enable_thermal;      ///< Enable thermal diffusion
    bool enable_thermal_advection; ///< Enable thermal advection (v·∇T coupling, requires enable_thermal=true)
    bool enable_phase_change; ///< Enable phase change (melting/solidification)
    bool enable_fluid;        ///< Enable fluid flow
    bool enable_vof;          ///< Enable VOF interface reconstruction
    bool enable_vof_advection; ///< Enable VOF advection (requires enable_vof=true)
    bool enable_surface_tension;  ///< Enable surface tension forces
    bool enable_marangoni;    ///< Enable Marangoni effect
    bool enable_laser;        ///< Enable laser heat source
    bool enable_darcy;        ///< Enable Darcy damping for mushy/solid zones
    bool enable_buoyancy;     ///< Enable buoyancy force (Boussinesq approximation)
    bool enable_evaporation_mass_loss;  ///< Enable evaporation mass loss from VOF (requires thermal + VOF)
    bool enable_recoil_pressure;  ///< Enable recoil pressure for keyhole formation (requires thermal + VOF)
    bool enable_solidification_shrinkage;  ///< Enable solidification volume shrinkage

    // Recoil pressure parameters
    float recoil_coefficient;     ///< C_r coefficient (0.54 typical, Knight 1979)
    float recoil_smoothing_width; ///< Interface smoothing width [cells]
    float recoil_max_pressure;    ///< Numerical pressure limiter [Pa]

    // CFL limiter parameters (force limiting for numerical stability)
    float cfl_limit;              ///< Maximum CFL number (default: 0.5, LBM stability requires < 0.577)
    float cfl_velocity_target;    ///< Target maximum lattice velocity (default: 0.1, safe LBM range)
    bool cfl_use_gradual_scaling; ///< Use gradual scaling instead of hard cutoff (default: true)
    float cfl_force_ramp_factor;  ///< Gradual force ramp-up factor (default: 0.8, lower = more gradual)

    // Adaptive CFL parameters for keyhole simulation (region-based limiting)
    bool cfl_use_adaptive;             ///< Enable adaptive region-based CFL limiting (default: false)
    float cfl_v_target_interface;      ///< Target velocity for interface cells [lattice] (0.5 = 10 m/s)
    float cfl_v_target_bulk;           ///< Target velocity for bulk liquid cells [lattice] (0.3 = 6 m/s)
    float cfl_interface_threshold_lo;  ///< Fill level lower bound for interface detection (default: 0.01)
    float cfl_interface_threshold_hi;  ///< Fill level upper bound for interface detection (default: 0.99)
    float cfl_recoil_boost_factor;     ///< Extra velocity allowance for z-dominant forces (default: 1.5)

    // Numerical parameters
    int vof_subcycles;        ///< Number of VOF subcycles per time step
    float dt;                 ///< Time step [s]

    // Material properties
    MaterialProperties material;  ///< Material properties (Ti6Al4V, etc.)

    // Thermal properties
    float thermal_diffusivity;    ///< Thermal diffusivity α [m²/s]

    // Fluid properties
    float kinematic_viscosity;    ///< Kinematic viscosity ν [m²/s]
    float density;                ///< Liquid density [kg/m³]
    float darcy_coefficient;      ///< Darcy damping constant C [dimensionless]

    // Surface properties
    float surface_tension_coeff;  ///< Surface tension σ [N/m]
    float dsigma_dT;              ///< Temperature coefficient dσ/dT [N/(m·K)]

    // Buoyancy properties (Boussinesq approximation)
    float thermal_expansion_coeff; ///< Thermal expansion coefficient β [1/K]
    float gravity_x;               ///< Gravity vector x-component [m/s²]
    float gravity_y;               ///< Gravity vector y-component [m/s²]
    float gravity_z;               ///< Gravity vector z-component [m/s²]
    float reference_temperature;   ///< Reference temperature T_ref for buoyancy [K]

    // Laser properties (optional)
    float laser_power;            ///< Laser power [W]
    float laser_spot_radius;      ///< Beam radius [m]
    float laser_absorptivity;     ///< Absorptivity [0-1]
    float laser_penetration_depth;///< Penetration depth [m]
    float laser_shutoff_time;     ///< Time to turn off laser [s] (negative = never)

    // Laser scanning parameters
    float laser_start_x;          ///< Initial laser X position [m] (negative = auto center)
    float laser_start_y;          ///< Initial laser Y position [m] (negative = auto center)
    float laser_scan_vx;          ///< Scan velocity X [m/s]
    float laser_scan_vy;          ///< Scan velocity Y [m/s]

    // Boundary conditions
    int boundary_type;            ///< Boundary type (0=periodic, 1=wall)

    // Radiation boundary condition (v4 fix for thermal runaway)
    bool enable_radiation_bc;     ///< Enable Stefan-Boltzmann radiation BC
    float emissivity;             ///< Surface emissivity [0-1] (0.3 for Ti6Al4V)
    float ambient_temperature;    ///< Ambient temperature [K] (typically 300 K)

    // Substrate cooling boundary condition (Week 1 Tuesday: energy balance fix)
    bool enable_substrate_cooling;  ///< Enable convective cooling at bottom surface (z=0)
    float substrate_h_conv;         ///< Convective heat transfer coefficient [W/(m²·K)]
    float substrate_temperature;    ///< Substrate temperature [K]

    /**
     * @brief Default constructor with Ti6Al4V LPBF parameters
     * @note kinematic_viscosity is in LATTICE UNITS for LBM stability
     *       Physical viscosity: ν_phys = 1.217e-6 m²/s
     *       Lattice viscosity: ν_lattice = 0.0333 (tau=0.6)
     */
    MultiphysicsConfig()
        : nx(100), ny(100), nz(50),
          dx(1e-6f),
          enable_thermal(false),  // Disabled by default (Step 1)
          enable_thermal_advection(false),  // v5: Thermal-fluid coupling (disabled by default for backward compatibility)
          enable_fluid(true),
          enable_vof(true),
          enable_vof_advection(false),  // Disabled by default (Step 1)
          enable_surface_tension(false),  // Add in Step 3
          enable_marangoni(true),
          enable_laser(false),    // Add in Step 4
          enable_darcy(true),     // Enable Darcy damping by default
          enable_buoyancy(true),  // Enable buoyancy for realistic LPBF
          enable_evaporation_mass_loss(true),  // Enable evaporation mass loss from VOF
          enable_recoil_pressure(false),  // Disabled by default (enable for keyhole simulations)
          enable_solidification_shrinkage(true),  // Enable by default
          recoil_coefficient(0.54f),      // Anisimov/Knight coefficient
          recoil_smoothing_width(2.0f),   // 2 cells interface smoothing
          recoil_max_pressure(1e8f),      // 100 MPa limiter
          cfl_limit(0.6f),                // LBM stability limit (relaxed for keyhole, < 0.577 traditional)
          cfl_velocity_target(0.15f),     // Default: 3 m/s @ dx=2um,dt=0.1us (safe for Marangoni)
          cfl_use_gradual_scaling(true),  // Gradual scaling for smoother deformation
          cfl_force_ramp_factor(0.9f),    // Start limiting at 90% of target (was 0.8)
          cfl_use_adaptive(false),        // Adaptive region-based CFL (for keyhole simulations)
          cfl_v_target_interface(0.5f),   // Interface: 0.5 lattice = ~10 m/s (strong recoil allowed)
          cfl_v_target_bulk(0.3f),        // Bulk liquid: 0.3 lattice = ~6 m/s (moderate Marangoni)
          cfl_interface_threshold_lo(0.01f),  // Interface detection: fill > 0.01
          cfl_interface_threshold_hi(0.99f),  // Interface detection: fill < 0.99
          cfl_recoil_boost_factor(1.5f),  // 50% extra allowance for z-dominant (recoil) forces
          enable_phase_change(false),  // Add in Phase 2
          vof_subcycles(10),
          dt(1e-9f),  // 1 ns
          material(MaterialDatabase::getTi6Al4V()),  // CRITICAL FIX: Initialize material properties
          thermal_diffusivity(5.8e-6f),  // Ti6Al4V liquid
          kinematic_viscosity(0.0333f),  // LATTICE UNITS (tau=0.6, stable)
          density(4110.0f),  // Ti6Al4V liquid
          darcy_coefficient(1e7f),  // Darcy damping constant
          surface_tension_coeff(1.65f),  // Ti6Al4V liquid-gas
          dsigma_dT(-0.26e-3f),  // Ti6Al4V
          thermal_expansion_coeff(1.5e-5f),  // Ti6Al4V typical value [1/K]
          gravity_x(0.0f),                   // No horizontal gravity
          gravity_y(0.0f),                   // No horizontal gravity
          gravity_z(-9.81f),                 // Standard gravity (downward)
          reference_temperature(1923.0f),    // Ti6Al4V melting point [K]
          laser_power(200.0f),
          laser_spot_radius(50e-6f),
          laser_absorptivity(0.35f),
          laser_penetration_depth(10e-6f),
          laser_shutoff_time(-1.0f),  // Negative = laser always on
          laser_start_x(-1.0f),  // Negative = auto (domain center)
          laser_start_y(-1.0f),  // Negative = auto (domain center)
          laser_scan_vx(0.36f),  // Default scan velocity (moderate LPBF speed)
          laser_scan_vy(0.0f),   // No Y scanning by default
          boundary_type(0),
          enable_radiation_bc(false),      // v4 fix: disabled by default
          emissivity(0.3f),                // Ti6Al4V typical emissivity
          ambient_temperature(300.0f),     // Room temperature
          enable_substrate_cooling(true),  // Week 1 Tuesday: enable substrate BC by default
          substrate_h_conv(1000.0f),       // Water-cooled substrate [W/(m²·K)]
          substrate_temperature(300.0f)    // Substrate at room temperature [K]
        {}
};

/**
 * @brief Multiphysics solver for metal AM simulations
 *
 * This class orchestrates the coupling of multiple physics solvers:
 * - ThermalLBM: Heat diffusion and convection
 * - FluidLBM: Incompressible Navier-Stokes
 * - VOFSolver: Free surface tracking
 * - MarangoniEffect: Thermocapillary forces
 * - SurfaceTension: Capillary forces
 * - LaserSource: Volumetric heating
 *
 * The solver uses a sequential coupling approach with subcycling for
 * numerical stability (VOF requires smaller time steps than LBM).
 */
class MultiphysicsSolver {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    MultiphysicsSolver(const MultiphysicsConfig& config);

    /**
     * @brief Destructor
     */
    ~MultiphysicsSolver();

    /**
     * @brief Initialize all physics modules
     * @param initial_temperature Uniform initial temperature [K]
     * @param interface_height Interface height as fraction of nz (0-1)
     */
    void initialize(float initial_temperature = 300.0f,
                   float interface_height = 0.5f);

    /**
     * @brief Initialize with custom temperature field
     * @param temperature_field Host array of temperature values
     * @param fill_level_field Host array of fill level values (0-1)
     */
    void initialize(const float* temperature_field,
                   const float* fill_level_field);

    /**
     * @brief Perform one time step
     * @param dt Time step [s] (if 0, uses config.dt)
     *
     * Integration sequence:
     * 1. Laser heat source (if enabled)
     * 2. Thermal diffusion (if enabled)
     * 3. VOF advection with subcycling (if enabled)
     * 4. Interface reconstruction
     * 5. Force computation (Marangoni + surface tension)
     * 6. Fluid flow
     */
    void step(float dt = 0.0f);

    /**
     * @brief Set static temperature field (for Step 1 testing)
     * @param temperature_field Device array of temperature values
     * @note This bypasses thermal solver and sets temperature directly
     */
    void setStaticTemperature(const float* temperature_field);

    /**
     * @brief Set static liquid fraction field (for testing without phase change)
     * @param liquid_fraction_field Device array of liquid fraction values (0=solid, 1=liquid)
     * @note This is used when thermal solver has phase change disabled
     */
    void setStaticLiquidFraction(const float* liquid_fraction_field);

    // ========================================================================
    // Diagnostic outputs
    // ========================================================================

    /**
     * @brief Get maximum velocity magnitude
     * @return Max |u| [m/s]
     */
    float getMaxVelocity() const;

    /**
     * @brief Get maximum temperature
     * @return Max T [K]
     */
    float getMaxTemperature() const;

    /**
     * @brief Get melt pool depth (distance from interface to T < T_liquidus)
     * @return Depth [m]
     */
    float getMeltPoolDepth() const;

    /**
     * @brief Get surface protrusion (max z-displacement of interface)
     * @return Protrusion height [m]
     */
    float getSurfaceProtrusion() const;

    /**
     * @brief Get total liquid mass for conservation check
     * @return Total mass Σf_i
     */
    float getTotalMass() const;

    /**
     * @brief Check for NaN or Inf in all fields
     * @return True if any NaN/Inf detected
     */
    bool checkNaN() const;

    // ========================================================================
    // Energy Conservation Diagnostics
    // ========================================================================

    /**
     * @brief Compute absorbed laser power (actual power entering domain)
     * @return P_laser_absorbed [W]
     * @note This integrates the volumetric heat source over the domain
     */
    float getLaserAbsorbedPower() const;

    /**
     * @brief Compute total evaporation cooling power
     * @return P_evap [W]
     * @note Sum of evaporation power at all surface cells
     */
    float getEvaporationPower() const;

    /**
     * @brief Compute total radiation cooling power
     * @return P_rad [W]
     * @note Sum of Stefan-Boltzmann radiation at all surface cells
     */
    float getRadiationPower() const;

    /**
     * @brief Compute total substrate cooling power
     * @return P_substrate [W]
     * @note Sum of convective cooling power at bottom surface (z=0)
     */
    float getSubstratePower() const;

    /**
     * @brief Compute rate of change of internal thermal energy
     * @return dE/dt [W]
     * @note Computed as (E_current - E_previous) / dt
     */
    float getThermalEnergyChangeRate() const;

    /**
     * @brief Print energy balance diagnostic
     * @note Prints: P_laser, P_evap, P_rad, P_cond, dE/dt, energy balance error
     * @note Called automatically every diagnostic_interval_ timesteps
     */
    void printEnergyBalance();

    /**
     * @brief Compute comprehensive energy balance (Week 3 P1)
     * @note Computes E_thermal, E_kinetic, E_latent, and all power terms
     * @note Updates internal energy balance tracker
     */
    void computeEnergyBalance();

    /**
     * @brief Write energy balance time series to file
     * @param filename Output file path
     * @note Writes accumulated energy history to ASCII file for post-processing
     */
    void writeEnergyBalanceHistory(const std::string& filename) const;

    /**
     * @brief Get current energy balance snapshot
     * @return Current energy balance data
     */
    const diagnostics::EnergyBalance& getCurrentEnergyBalance() const;

    // ========================================================================
    // Data access (for visualization and testing)
    // ========================================================================

    const float* getTemperature() const;
    const float* getLiquidFraction() const;  ///< Get liquid fraction field from thermal solver
    const float* getVelocityX() const;
    const float* getVelocityY() const;
    const float* getVelocityZ() const;
    const float* getFillLevel() const;
    float* getFillLevelMutable();  ///< Mutable access for powder bed initialization
    const float* getCurvature() const;
    const float* getPressure() const;

    /**
     * @brief Set fill level from host array (for powder bed initialization)
     * @param h_fill_level Host array of fill level values [size: nx*ny*nz]
     */
    void setFillLevel(const float* h_fill_level);

    /**
     * @brief Copy velocity to host
     */
    void copyVelocityToHost(float* ux, float* uy, float* uz) const;

    /**
     * @brief Copy temperature to host
     */
    void copyTemperatureToHost(float* temperature) const;

    /**
     * @brief Copy fill level to host
     */
    void copyFillLevelToHost(float* fill_level) const;

    // ========================================================================
    // Configuration access
    // ========================================================================

    const MultiphysicsConfig& getConfig() const { return config_; }
    int getNx() const { return config_.nx; }
    int getNy() const { return config_.ny; }
    int getNz() const { return config_.nz; }
    float getDx() const { return config_.dx; }

private:
    // Configuration
    MultiphysicsConfig config_;

    // Unit converter for all lattice <-> physical conversions
    core::UnitConverter unit_converter_;

    // Physics modules (using smart pointers for RAII)
    std::unique_ptr<ThermalLBM> thermal_;
    std::unique_ptr<FluidLBM> fluid_;
    std::unique_ptr<VOFSolver> vof_;
    std::unique_ptr<SurfaceTension> surface_tension_;
    std::unique_ptr<MarangoniEffect> marangoni_;
    std::unique_ptr<LaserSource> laser_;
    std::unique_ptr<RecoilPressure> recoil_pressure_;

    // Force accumulation pipeline (replaces fragile scattered force computation)
    std::unique_ptr<ForceAccumulator> force_accumulator_;

    // Device memory for recoil pressure computation
    float* d_saturation_pressure_;  ///< Saturation pressure field [Pa]

    // Device memory for force accumulation
    float* d_force_x_;
    float* d_force_y_;
    float* d_force_z_;

    // Device memory for temperature (if static)
    float* d_temperature_static_;

    // Device memory for liquid fraction (if static, when phase change disabled)
    float* d_liquid_fraction_static_;

    // Device memory for VOF advection (physical unit velocity)
    // VOF expects velocity in [m/s], but LBM outputs lattice units
    // These buffers store v_physical = v_lattice * (dx / dt)
    float* d_velocity_physical_x_;
    float* d_velocity_physical_y_;
    float* d_velocity_physical_z_;

    // Device memory for evaporation mass flux (VOF-thermal coupling)
    // J_evap [kg/(m^2*s)] computed from thermal solver, applied to VOF
    float* d_evap_mass_flux_;

    // Current simulation time
    float current_time_;

    // Interface z position (in lattice units) - where fill_level = 0.5
    // Used for laser surface detection (where Beer-Lambert absorption starts)
    float interface_z_;

    // Internal methods

    /**
     * @brief Allocate device memory for force fields
     */
    void allocateMemory();

    /**
     * @brief Free device memory
     */
    void freeMemory();

    /**
     * @brief Compute total force using ForceAccumulator pipeline
     * @note Replaces the old fragile computeTotalForce(float*, float*, float*) implementation
     *       New pipeline: reset → add forces → convert units → apply CFL
     */
    void computeTotalForce();

    /**
     * @brief Apply laser heat source
     * @param dt Time step [s]
     */
    void applyLaserSource(float dt);

    /**
     * @brief Perform thermal step
     * @param dt Time step [s]
     */
    void thermalStep(float dt);

    /**
     * @brief Perform VOF advection with subcycling
     * @param dt Time step [s]
     */
    void vofStep(float dt);

    /**
     * @brief Perform fluid step
     * @param dt Time step [s]
     */
    void fluidStep(float dt);

    /**
     * @brief Check mass conservation
     * @return Mass change relative to initial mass
     */
    float checkMassConservation() const;

    // Initial mass for conservation check
    float initial_mass_;

    // Energy tracking for diagnostics
    mutable float previous_thermal_energy_;  ///< Previous timestep's thermal energy [J]
    mutable float previous_time_;            ///< Previous timestep's time [s]

    // Week 1 Monday: Energy balance diagnostics
    int current_step_;                       ///< Current simulation step counter
    int diagnostic_interval_;                ///< Print energy balance every N steps (adaptive)

    // Week 3 Day 1: Time-averaged dE/dt for noise reduction (Bug 3 fix)
    static constexpr int ENERGY_HISTORY_SIZE = 10;  ///< Number of points for moving average
    mutable std::vector<float> energy_history_;      ///< Energy history for time averaging
    mutable std::vector<float> time_history_;        ///< Time history for time averaging

    // Week 3 P1: Comprehensive energy balance tracking
    diagnostics::EnergyBalanceTracker energy_tracker_;  ///< Energy balance time series tracker
    double* d_energy_temp_;                              ///< Device memory for energy reduction
    int energy_output_interval_;                         ///< Output energy balance every N steps
    static constexpr int default_energy_interval_ = 10;  ///< Default: every 10 steps

    /**
     * @brief Compute total heat flux through domain boundaries (lateral conduction)
     * @return P_cond [W] - total power leaving through boundaries
     * @note For periodic BC: P_cond = 0
     * @note For adiabatic BC: P_cond should be ≈ 0 (numerical error only)
     * @note For isothermal BC: P_cond = Σ(-k * ∇T * A) at boundaries
     */
    float computeBoundaryHeatFlux() const;
};

} // namespace physics
} // namespace lbm
