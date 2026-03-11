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
#include "io/field_registry.h"

namespace lbm {
namespace physics {

/**
 * @brief Thermal boundary condition types for per-face specification
 */
enum class ThermalBCType {
    PERIODIC,     ///< Periodic (handled automatically in streaming)
    ADIABATIC,    ///< Zero heat flux (dT/dn = 0)
    DIRICHLET,    ///< Fixed temperature
    CONVECTIVE,   ///< Newton's law of cooling: q = h*(T - T_inf)
    RADIATION     ///< Stefan-Boltzmann: q = eps*sigma*(T^4 - T_amb^4)
};

/**
 * @brief Per-face boundary condition specification
 *
 * Supports different boundary types on each of the 6 domain faces.
 * This replaces the old single `boundary_type` integer with a richer
 * configuration that LPBF simulations need (e.g., no-slip substrate
 * at z=0, periodic sides, open top).
 *
 * For axes where min and max have different fluid BC types, the axis
 * is treated as WALL (the more restrictive choice) for the streaming
 * kernel, which already handles per-axis periodicity.
 */
struct FaceBoundaryConfig {
    // Fluid boundary types per face
    BoundaryType x_min = BoundaryType::PERIODIC;
    BoundaryType x_max = BoundaryType::PERIODIC;
    BoundaryType y_min = BoundaryType::PERIODIC;
    BoundaryType y_max = BoundaryType::PERIODIC;
    BoundaryType z_min = BoundaryType::WALL;      ///< Default: substrate (no-slip)
    BoundaryType z_max = BoundaryType::PERIODIC;   ///< Default: open top

    // Thermal boundary types per face
    ThermalBCType thermal_x_min = ThermalBCType::PERIODIC;
    ThermalBCType thermal_x_max = ThermalBCType::PERIODIC;
    ThermalBCType thermal_y_min = ThermalBCType::PERIODIC;
    ThermalBCType thermal_y_max = ThermalBCType::PERIODIC;
    ThermalBCType thermal_z_min = ThermalBCType::CONVECTIVE;  ///< Substrate cooling
    ThermalBCType thermal_z_max = ThermalBCType::ADIABATIC;   ///< Open top

    // Thermal BC parameters (apply to faces with matching type)
    float dirichlet_temperature = 300.0f;  ///< For DIRICHLET faces [K]
    float convective_h = 1000.0f;          ///< For CONVECTIVE faces [W/(m^2*K)]
    float convective_T_inf = 300.0f;       ///< For CONVECTIVE faces [K]
    float radiation_emissivity = 0.3f;     ///< For RADIATION faces
    float radiation_T_ambient = 300.0f;    ///< For RADIATION faces [K]

    /// Axis is periodic only if BOTH faces are periodic
    bool isPeriodicX() const { return x_min == BoundaryType::PERIODIC && x_max == BoundaryType::PERIODIC; }
    bool isPeriodicY() const { return y_min == BoundaryType::PERIODIC && y_max == BoundaryType::PERIODIC; }
    bool isPeriodicZ() const { return z_min == BoundaryType::PERIODIC && z_max == BoundaryType::PERIODIC; }

    /// Derive per-axis fluid BC: WALL if either face is WALL, else PERIODIC
    BoundaryType fluidBCX() const { return isPeriodicX() ? BoundaryType::PERIODIC : BoundaryType::WALL; }
    BoundaryType fluidBCY() const { return isPeriodicY() ? BoundaryType::PERIODIC : BoundaryType::WALL; }
    BoundaryType fluidBCZ() const { return isPeriodicZ() ? BoundaryType::PERIODIC : BoundaryType::WALL; }

    /// Derive per-axis VOF BC
    VOFSolver::BoundaryType vofBCX() const {
        return isPeriodicX() ? VOFSolver::BoundaryType::PERIODIC : VOFSolver::BoundaryType::WALL;
    }
    VOFSolver::BoundaryType vofBCY() const {
        return isPeriodicY() ? VOFSolver::BoundaryType::PERIODIC : VOFSolver::BoundaryType::WALL;
    }
    VOFSolver::BoundaryType vofBCZ() const {
        return isPeriodicZ() ? VOFSolver::BoundaryType::PERIODIC : VOFSolver::BoundaryType::WALL;
    }

    /// Check if any face has a given thermal BC type
    bool hasAnyThermalBC(ThermalBCType type) const {
        return thermal_x_min == type || thermal_x_max == type ||
               thermal_y_min == type || thermal_y_max == type ||
               thermal_z_min == type || thermal_z_max == type;
    }

    /// Get thermal BC for a specific face (0=x_min, 1=x_max, 2=y_min, 3=y_max, 4=z_min, 5=z_max)
    ThermalBCType thermalBCForFace(int face) const {
        switch (face) {
            case 0: return thermal_x_min;
            case 1: return thermal_x_max;
            case 2: return thermal_y_min;
            case 3: return thermal_y_max;
            case 4: return thermal_z_min;
            case 5: return thermal_z_max;
            default: return ThermalBCType::PERIODIC;
        }
    }

    /**
     * @brief Set all faces to uniform boundary types (convenience)
     * @param fluid_bc Fluid boundary type for all 6 faces
     * @param thermal_bc Thermal boundary type for all 6 faces
     */
    void setUniform(BoundaryType fluid_bc, ThermalBCType thermal_bc) {
        x_min = x_max = y_min = y_max = z_min = z_max = fluid_bc;
        thermal_x_min = thermal_x_max = thermal_y_min = thermal_y_max =
            thermal_z_min = thermal_z_max = thermal_bc;
    }

    /**
     * @brief Initialize from legacy boundary_type integer
     * @param bt 0=all periodic, 1=all walls with Dirichlet thermal, 2=all walls with adiabatic thermal
     */
    static FaceBoundaryConfig fromLegacy(int bt) {
        FaceBoundaryConfig cfg;
        if (bt == 0) {
            cfg.setUniform(BoundaryType::PERIODIC, ThermalBCType::PERIODIC);
        } else if (bt == 1) {
            cfg.setUniform(BoundaryType::WALL, ThermalBCType::DIRICHLET);
        } else if (bt == 2) {
            cfg.setUniform(BoundaryType::WALL, ThermalBCType::ADIABATIC);
        }
        return cfg;
    }
};

/**
 * @brief Configuration for phase change solver (Newton/bisection iteration)
 */
struct PhaseChangeConfig {
    float newton_tolerance = 0.01f;  ///< Newton solver temperature tolerance [K]
    int max_iterations = 50;         ///< Max Newton+bisection iterations
};

/**
 * @brief Configuration for MultiphysicsSolver
 *
 * Parameters are grouped into named sub-structs for maintainability.
 * Direct field access is preserved via forwarding members so existing
 * code that does  config.nx, config.enable_thermal, etc. still compiles.
 *
 * UNIT CONVENTION:
 * - All dimensional parameters are in SI physical units EXCEPT:
 *   - fluid.kinematic_viscosity: in LATTICE units (historical)
 *     TODO: Convert to physical units [m^2/s] in future refactor
 * - The solver uses UnitConverter internally to convert to lattice units.
 */
struct MultiphysicsConfig {
    // ------------------------------------------------------------------ //
    // Sub-config structs (nested types)
    // ------------------------------------------------------------------ //

    struct DomainConfig {
        int nx = 100;      ///< Grid dimension X
        int ny = 100;      ///< Grid dimension Y
        int nz = 50;       ///< Grid dimension Z
        float dx = 1e-6f;  ///< Lattice spacing [m]
    };

    struct PhysicsFlags {
        bool enable_thermal                 = false; ///< Enable thermal diffusion
        bool enable_thermal_advection       = false; ///< Enable v·∇T coupling (requires thermal+fluid)
        bool enable_phase_change            = false; ///< Enable melting/solidification
        bool enable_fluid                   = true;  ///< Enable fluid flow
        bool enable_vof                     = true;  ///< Enable VOF interface reconstruction
        bool enable_vof_advection           = false; ///< Enable VOF advection (requires vof+fluid)
        bool enable_surface_tension         = false; ///< Enable surface tension forces
        bool enable_marangoni               = false; ///< Enable Marangoni (thermocapillary) effect
        bool enable_laser                   = false; ///< Enable laser heat source
        bool enable_darcy                   = false; ///< Enable Darcy damping for mushy/solid zones
        bool enable_buoyancy                = false; ///< Enable Boussinesq buoyancy
        bool enable_evaporation_mass_loss   = false; ///< Enable evaporation VOF mass loss
        bool enable_recoil_pressure         = false; ///< Enable evaporation recoil pressure
        bool enable_solidification_shrinkage = false; ///< Enable solidification volume shrinkage
    };

    struct NumericalConfig {
        float dt                     = 1e-9f;   ///< Time step [s]
        int   vof_subcycles          = 10;      ///< VOF subcycles per LBM step
        bool  enable_vof_mass_correction = true; ///< Global VOF mass correction
        float cfl_limit              = 0.5f;    ///< Advisory only -- used in validate() warning, not in actual CFL limiter.
                                                   ///< Actual limiting uses cfl_velocity_target and cfl_force_ramp_factor.
        float cfl_velocity_target    = 0.15f;   ///< Target max lattice velocity
        bool  cfl_use_gradual_scaling = true;   ///< Gradual vs hard force cutoff
        float cfl_force_ramp_factor  = 0.9f;    ///< Ramp onset fraction of v_target
        bool  cfl_use_adaptive       = false;   ///< Region-based adaptive CFL
        float cfl_v_target_interface = 0.5f;    ///< Interface cell velocity target [lattice]
        float cfl_v_target_bulk      = 0.3f;    ///< Bulk liquid velocity target [lattice]
        float cfl_interface_threshold_lo = 0.01f; ///< Interface lower fill bound
        float cfl_interface_threshold_hi = 0.99f; ///< Interface upper fill bound
        float cfl_recoil_boost_factor    = 1.5f;  ///< Extra allowance for z-dominant forces
    };

    struct FluidConfig {
        /// WARNING: LATTICE UNITS (not m²/s) — see UNIT CONVENTION above
        float kinematic_viscosity = 0.0333f; ///< Kinematic viscosity (tau=0.6)
        float density             = 4110.0f; ///< Liquid density [kg/m³]
        float darcy_coefficient   = 1e7f;    ///< Darcy damping constant
    };

    struct ThermalConfig {
        float thermal_diffusivity    = 9.66e-6f; ///< α [m²/s] Ti6Al4V liquid: k/(rho*cp) = 33/(4110*831)
        bool  enable_radiation_bc    = false;   ///< Stefan-Boltzmann radiation BC
        float emissivity             = 0.3f;    ///< Surface emissivity [0-1]
        float ambient_temperature    = 300.0f;  ///< Ambient temperature [K]
        bool  enable_substrate_cooling = true;  ///< Convective bottom BC
        float substrate_h_conv       = 1000.0f; ///< h_conv [W/(m²·K)]
        float substrate_temperature  = 300.0f;  ///< Substrate T [K]
    };

    struct SurfaceConfig {
        float surface_tension_coeff  = 1.65f;    ///< σ [N/m]
        float dsigma_dT              = -0.26e-3f; ///< dσ/dT [N/(m·K)]
        float recoil_coefficient     = 0.54f;    ///< C_r (Knight 1979)
        float recoil_smoothing_width = 2.0f;     ///< Interface smoothing [cells]
        float recoil_max_pressure    = 1e8f;     ///< Numerical pressure cap [Pa]
        float molar_mass             = 0.0476f;  ///< Molar mass [kg/mol] (Ti6Al4V default)
    };

    struct BuoyancyConfig {
        float thermal_expansion_coeff = 1.5e-5f; ///< β [1/K]
        float gravity_x               = 0.0f;    ///< g_x [m/s²]
        float gravity_y               = 0.0f;    ///< g_y [m/s²]
        float gravity_z               = -9.81f;  ///< g_z [m/s²]
        float reference_temperature   = 1923.0f; ///< T_ref for buoyancy [K]
    };

    struct LaserConfig {
        float power             = 200.0f;   ///< Laser power [W]
        float spot_radius       = 50e-6f;   ///< Beam radius [m]
        float absorptivity      = 0.35f;    ///< Absorptivity [0-1]
        float penetration_depth = 10e-6f;   ///< Beer-Lambert depth [m]
        float shutoff_time      = -1.0f;    ///< Shutoff time [s] (<0 = always on)
        float start_x           = -1.0f;    ///< Initial X [m] (<0 = auto center)
        float start_y           = -1.0f;    ///< Initial Y [m] (<0 = auto center)
        float scan_vx           = 0.36f;    ///< Scan velocity X [m/s]
        float scan_vy           = 0.0f;     ///< Scan velocity Y [m/s]
    };

    // ------------------------------------------------------------------ //
    // Sub-config instances
    // ------------------------------------------------------------------ //
    DomainConfig   domain;
    PhysicsFlags   physics;
    NumericalConfig numerics;
    FluidConfig    fluid;
    ThermalConfig  thermal;
    SurfaceConfig  surface;
    BuoyancyConfig buoyancy;
    LaserConfig    laser;
    MaterialProperties material;
    PhaseChangeConfig phase_change;
    FaceBoundaryConfig boundaries;  ///< Per-face boundary conditions (preferred)
    int boundary_type = 0; ///< Deprecated: 0=periodic, 1=Dirichlet, 2=adiabatic (use boundaries instead)

    // ------------------------------------------------------------------ //
    // Backward-compatible flat accessors
    //
    // These reference members let existing code of the form
    //   config.nx = 64;  config.enable_thermal = true;  etc.
    // continue to compile without modification.
    // ------------------------------------------------------------------ //

    // Domain
    int&   nx  = domain.nx;
    int&   ny  = domain.ny;
    int&   nz  = domain.nz;
    float& dx  = domain.dx;

    // Numerical
    float& dt                       = numerics.dt;
    int&   vof_subcycles            = numerics.vof_subcycles;
    bool&  enable_vof_mass_correction = numerics.enable_vof_mass_correction;
    float& cfl_limit                = numerics.cfl_limit;
    float& cfl_velocity_target      = numerics.cfl_velocity_target;
    bool&  cfl_use_gradual_scaling  = numerics.cfl_use_gradual_scaling;
    float& cfl_force_ramp_factor    = numerics.cfl_force_ramp_factor;
    bool&  cfl_use_adaptive         = numerics.cfl_use_adaptive;
    float& cfl_v_target_interface   = numerics.cfl_v_target_interface;
    float& cfl_v_target_bulk        = numerics.cfl_v_target_bulk;
    float& cfl_interface_threshold_lo = numerics.cfl_interface_threshold_lo;
    float& cfl_interface_threshold_hi = numerics.cfl_interface_threshold_hi;
    float& cfl_recoil_boost_factor  = numerics.cfl_recoil_boost_factor;

    // Physics flags
    bool& enable_thermal                  = physics.enable_thermal;
    bool& enable_thermal_advection        = physics.enable_thermal_advection;
    bool& enable_phase_change             = physics.enable_phase_change;
    bool& enable_fluid                    = physics.enable_fluid;
    bool& enable_vof                      = physics.enable_vof;
    bool& enable_vof_advection            = physics.enable_vof_advection;
    bool& enable_surface_tension          = physics.enable_surface_tension;
    bool& enable_marangoni                = physics.enable_marangoni;
    bool& enable_laser                    = physics.enable_laser;
    bool& enable_darcy                    = physics.enable_darcy;
    bool& enable_buoyancy                 = physics.enable_buoyancy;
    bool& enable_evaporation_mass_loss    = physics.enable_evaporation_mass_loss;
    bool& enable_recoil_pressure          = physics.enable_recoil_pressure;
    bool& enable_solidification_shrinkage = physics.enable_solidification_shrinkage;

    // Fluid
    float& kinematic_viscosity = fluid.kinematic_viscosity;
    float& density             = fluid.density;
    float& darcy_coefficient   = fluid.darcy_coefficient;

    // Thermal
    float& thermal_diffusivity     = thermal.thermal_diffusivity;
    bool&  enable_radiation_bc     = thermal.enable_radiation_bc;
    float& emissivity              = thermal.emissivity;
    float& ambient_temperature     = thermal.ambient_temperature;
    bool&  enable_substrate_cooling = thermal.enable_substrate_cooling;
    float& substrate_h_conv        = thermal.substrate_h_conv;
    float& substrate_temperature   = thermal.substrate_temperature;

    // Surface / recoil
    float& surface_tension_coeff  = surface.surface_tension_coeff;
    float& dsigma_dT              = surface.dsigma_dT;
    float& recoil_coefficient     = surface.recoil_coefficient;
    float& recoil_smoothing_width = surface.recoil_smoothing_width;
    float& recoil_max_pressure    = surface.recoil_max_pressure;

    // Buoyancy
    float& thermal_expansion_coeff = buoyancy.thermal_expansion_coeff;
    float& gravity_x               = buoyancy.gravity_x;
    float& gravity_y               = buoyancy.gravity_y;
    float& gravity_z               = buoyancy.gravity_z;
    float& reference_temperature   = buoyancy.reference_temperature;

    // Laser
    float& laser_power             = laser.power;
    float& laser_spot_radius       = laser.spot_radius;
    float& laser_absorptivity      = laser.absorptivity;
    float& laser_penetration_depth = laser.penetration_depth;
    float& laser_shutoff_time      = laser.shutoff_time;
    float& laser_start_x           = laser.start_x;
    float& laser_start_y           = laser.start_y;
    float& laser_scan_vx           = laser.scan_vx;
    float& laser_scan_vy           = laser.scan_vy;

    // ------------------------------------------------------------------ //
    // Constructor / copy / move
    // ------------------------------------------------------------------ //

    /**
     * @brief Default constructor with Ti6Al4V LPBF parameters.
     * @note kinematic_viscosity is in LATTICE UNITS (tau=0.6, stable).
     *       Physical viscosity: ν_phys = 1.217e-6 m²/s
     */
    MultiphysicsConfig()
        : domain{}, physics{}, numerics{}, fluid{}, thermal{},
          surface{}, buoyancy{}, laser{},
          material(MaterialDatabase::getTi6Al4V())
    {}

    // Reference members make the implicit copy constructor unusable, so we
    // define explicit copy that copies the value sub-structs and then rebinds.
    MultiphysicsConfig(const MultiphysicsConfig& o)
        : domain(o.domain), physics(o.physics), numerics(o.numerics),
          fluid(o.fluid), thermal(o.thermal), surface(o.surface),
          buoyancy(o.buoyancy), laser(o.laser),
          material(o.material), boundaries(o.boundaries),
          boundary_type(o.boundary_type),
          // reference members auto-bind to this object's sub-structs via the
          // in-class initializer (member declarations above)
          nx(domain.nx), ny(domain.ny), nz(domain.nz), dx(domain.dx),
          dt(numerics.dt), vof_subcycles(numerics.vof_subcycles),
          enable_vof_mass_correction(numerics.enable_vof_mass_correction),
          cfl_limit(numerics.cfl_limit),
          cfl_velocity_target(numerics.cfl_velocity_target),
          cfl_use_gradual_scaling(numerics.cfl_use_gradual_scaling),
          cfl_force_ramp_factor(numerics.cfl_force_ramp_factor),
          cfl_use_adaptive(numerics.cfl_use_adaptive),
          cfl_v_target_interface(numerics.cfl_v_target_interface),
          cfl_v_target_bulk(numerics.cfl_v_target_bulk),
          cfl_interface_threshold_lo(numerics.cfl_interface_threshold_lo),
          cfl_interface_threshold_hi(numerics.cfl_interface_threshold_hi),
          cfl_recoil_boost_factor(numerics.cfl_recoil_boost_factor),
          enable_thermal(physics.enable_thermal),
          enable_thermal_advection(physics.enable_thermal_advection),
          enable_phase_change(physics.enable_phase_change),
          enable_fluid(physics.enable_fluid),
          enable_vof(physics.enable_vof),
          enable_vof_advection(physics.enable_vof_advection),
          enable_surface_tension(physics.enable_surface_tension),
          enable_marangoni(physics.enable_marangoni),
          enable_laser(physics.enable_laser),
          enable_darcy(physics.enable_darcy),
          enable_buoyancy(physics.enable_buoyancy),
          enable_evaporation_mass_loss(physics.enable_evaporation_mass_loss),
          enable_recoil_pressure(physics.enable_recoil_pressure),
          enable_solidification_shrinkage(physics.enable_solidification_shrinkage),
          kinematic_viscosity(fluid.kinematic_viscosity),
          density(fluid.density),
          darcy_coefficient(fluid.darcy_coefficient),
          thermal_diffusivity(thermal.thermal_diffusivity),
          enable_radiation_bc(thermal.enable_radiation_bc),
          emissivity(thermal.emissivity),
          ambient_temperature(thermal.ambient_temperature),
          enable_substrate_cooling(thermal.enable_substrate_cooling),
          substrate_h_conv(thermal.substrate_h_conv),
          substrate_temperature(thermal.substrate_temperature),
          surface_tension_coeff(surface.surface_tension_coeff),
          dsigma_dT(surface.dsigma_dT),
          recoil_coefficient(surface.recoil_coefficient),
          recoil_smoothing_width(surface.recoil_smoothing_width),
          recoil_max_pressure(surface.recoil_max_pressure),
          thermal_expansion_coeff(buoyancy.thermal_expansion_coeff),
          gravity_x(buoyancy.gravity_x),
          gravity_y(buoyancy.gravity_y),
          gravity_z(buoyancy.gravity_z),
          reference_temperature(buoyancy.reference_temperature),
          laser_power(laser.power),
          laser_spot_radius(laser.spot_radius),
          laser_absorptivity(laser.absorptivity),
          laser_penetration_depth(laser.penetration_depth),
          laser_shutoff_time(laser.shutoff_time),
          laser_start_x(laser.start_x),
          laser_start_y(laser.start_y),
          laser_scan_vx(laser.scan_vx),
          laser_scan_vy(laser.scan_vy)
    {}

    MultiphysicsConfig& operator=(const MultiphysicsConfig& o) {
        if (this == &o) return *this;
        domain        = o.domain;
        physics       = o.physics;
        numerics      = o.numerics;
        fluid         = o.fluid;
        thermal       = o.thermal;
        surface       = o.surface;
        buoyancy      = o.buoyancy;
        laser         = o.laser;
        material      = o.material;
        boundaries    = o.boundaries;
        boundary_type = o.boundary_type;
        // Reference members already point to our sub-structs; no rebind needed.
        return *this;
    }

    // ------------------------------------------------------------------ //
    // Convenience getters (for code that prefers method syntax)
    // ------------------------------------------------------------------ //
    int   getNx() const { return domain.nx; }
    int   getNy() const { return domain.ny; }
    int   getNz() const { return domain.nz; }
    float getDx() const { return domain.dx; }
    float getDt() const { return numerics.dt; }

    /**
     * @brief Validate configuration for LBM stability, physics consistency,
     *        and parameter ranges.
     * @throws std::runtime_error for fatal misconfigurations.
     * Prints warnings to stderr for non-fatal issues.
     */
    void validate() const;
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

    /// Validate configuration and print warnings for non-fatal issues
    void validate() const;

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

    /// Get the field registry (populated after initialize())
    const io::FieldRegistry& getFieldRegistry() const { return field_registry_; }

private:
    // Configuration
    MultiphysicsConfig config_;

    // Field registry for configurable VTK output
    io::FieldRegistry field_registry_;

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
     * @brief Register available output fields in field_registry_
     * @note Called at the end of initialize() after all sub-solvers are ready
     */
    void registerOutputFields();

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
    float initial_temperature_;              ///< Initial temperature for energy reference [K]
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
    mutable float time_last_computed_;                   ///< Time when energy balance was last computed [s]

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
