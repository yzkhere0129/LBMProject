# Gas Phase Module Architecture Design Document

## LBM-CUDA LPBF Platform - Phase 7 Extension

**Document Version:** 1.0
**Date:** 2025-11-21
**Status:** Design Specification
**Author:** LBM-CUDA Architecture Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Requirements Analysis](#2-requirements-analysis)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Structure Design](#4-module-structure-design)
5. [Coupling Interface Design](#5-coupling-interface-design)
6. [Execution Sequence](#6-execution-sequence)
7. [File Organization](#7-file-organization)
8. [Configuration Extension](#8-configuration-extension)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Validation Strategy](#10-validation-strategy)

---

## 1. Executive Summary

This document specifies the software architecture for integrating a Gas Phase module into the existing LBM-CUDA LPBF simulation platform. The gas phase is essential for accurate simulation of metal additive manufacturing processes, particularly for modeling:

- **Evaporation-induced recoil pressure** (primary driver of keyhole formation in LPBF)
- **Gas-phase heat transfer** (convective cooling at the free surface)
- **Mass conservation** at gas-liquid interfaces during phase change
- **Realistic interface dynamics** with proper gas-liquid momentum exchange

### Design Philosophy

Following the project's CLAUDE.md principles:
- **Incremental progress**: Build on existing VOF/FluidLBM infrastructure
- **Learn from existing code**: Match patterns established by ThermalLBM, MarangoniEffect
- **Pragmatic over dogmatic**: Single-fluid approach for efficiency, optional two-fluid for accuracy
- **Clear intent**: Explicit interfaces between gas and liquid solvers

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1 | Gas phase flow field computation | P0 | Core requirement |
| FR-2 | Gas-liquid interface tracking | P0 | Extends existing VOF |
| FR-3 | Evaporative mass flux computation | P0 | Hertz-Knudsen model |
| FR-4 | Recoil pressure computation | P0 | Critical for keyhole physics |
| FR-5 | Gas-liquid momentum exchange | P1 | Interface momentum balance |
| FR-6 | Evaporative cooling | P1 | Energy sink at interface |
| FR-7 | Gas-side convective heat transfer | P2 | Secondary thermal effect |
| FR-8 | Gas-solid interface handling | P2 | Vapor deposition |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Memory overhead | < 30% increase over current |
| NFR-2 | Performance impact | < 50% slowdown for single-fluid mode |
| NFR-3 | Modularity | Enable/disable via config flag |
| NFR-4 | Backward compatibility | Existing tests must pass unchanged |

### 2.3 Physical Models Required

```
Evaporation Mass Flux (Hertz-Knudsen):
    m_dot = (1 - beta_r) * p_sat(T) * sqrt(M / (2*pi*R*T))

Clausius-Clapeyron (Saturation Pressure):
    p_sat(T) = p_ref * exp(L_v * M / R * (1/T_ref - 1/T))

Recoil Pressure:
    P_recoil = 0.54 * p_sat(T)

Evaporative Cooling:
    Q_evap = m_dot * L_vaporization
```

---

## 3. Architecture Overview

### 3.1 Current Architecture

```
MultiphysicsSolver
    |
    +-- ThermalLBM (D3Q7)         : Heat diffusion + advection
    |       |
    |       +-- PhaseChangeSolver : Melting/solidification
    |
    +-- FluidLBM (D3Q19)          : Liquid flow
    |       |
    |       +-- computeBuoyancyForce()
    |       +-- applyDarcyDamping()
    |
    +-- VOFSolver                  : Gas-liquid interface
    |       |
    |       +-- fill_level (0=gas, 1=liquid)
    |       +-- interface_normal
    |       +-- curvature
    |
    +-- SurfaceTension             : Capillary forces (CSF)
    |
    +-- MarangoniEffect            : Thermocapillary forces
    |
    +-- LaserSource                : Volumetric heating
```

### 3.2 Proposed Architecture with Gas Phase

```
MultiphysicsSolver
    |
    +-- ThermalLBM (D3Q7)
    |       |
    |       +-- PhaseChangeSolver
    |
    +-- FluidLBM (D3Q19)           : Liquid phase solver
    |
    +-- GasPhaseModule [NEW]       : Gas phase management
    |       |
    |       +-- GasFlowSolver      : Optional separate gas LBM (D3Q19)
    |       +-- EvaporationModel   : Mass flux computation
    |       +-- RecoilPressure     : Pressure force at interface
    |       +-- GasThermal         : Optional gas thermal (D3Q7)
    |
    +-- VOFSolver                  : Extended for 3-phase
    |       |
    |       +-- fill_level_liquid  : Liquid fraction
    |       +-- fill_level_gas     : Gas fraction (1 - f_liquid for 2-phase)
    |       +-- interface_type     : GAS_LIQUID, GAS_SOLID, LIQUID_SOLID
    |
    +-- InterfaceCoupling [NEW]    : Manages all interface exchanges
    |       |
    |       +-- computeMassTransfer()
    |       +-- computeMomentumTransfer()
    |       +-- computeHeatTransfer()
    |
    +-- SurfaceTension
    +-- MarangoniEffect
    +-- LaserSource
```

### 3.3 Design Decision: Single-Fluid vs Two-Fluid Approach

**Recommended: Hybrid Approach**

| Mode | Description | Use Case |
|------|-------------|----------|
| `GasMode::IMPLICIT` | Gas treated implicitly via VOF | Default, efficient |
| `GasMode::EXPLICIT` | Separate gas phase LBM solver | High-fidelity, expensive |

The implicit mode uses the existing single-fluid LBM with:
- Recoil pressure applied as surface force (like Marangoni)
- Gas density/viscosity in VOF gas cells
- No separate gas velocity solve

The explicit mode enables:
- Full Navier-Stokes in gas phase
- Gas-liquid velocity continuity at interface
- Vapor jet dynamics and powder entrainment

---

## 4. Module Structure Design

### 4.1 GasPhaseModule Class

```cpp
/**
 * @file gas_phase_module.h
 * @brief Gas phase management for multiphase LPBF simulations
 *
 * This module handles all gas-phase physics including:
 * - Evaporation/condensation at gas-liquid interface
 * - Recoil pressure from evaporating metal vapor
 * - Optional explicit gas flow computation
 * - Gas-phase thermal effects
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "physics/gas_flow_solver.h"
#include "physics/evaporation_model.h"
#include "physics/recoil_pressure.h"

namespace lbm {
namespace physics {

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

    // Gas properties
    float gas_density = 1.2f;           ///< Ambient gas density [kg/m3] (Argon at 1 atm)
    float gas_viscosity = 2.2e-5f;      ///< Dynamic viscosity [Pa.s]
    float gas_thermal_conductivity = 0.018f;  ///< k_gas [W/(m.K)]
    float gas_specific_heat = 520.0f;   ///< cp_gas [J/(kg.K)]

    // Evaporation parameters
    bool enable_evaporation = true;
    float sticking_coefficient = 0.82f; ///< beta_r: fraction of vapor that recondenses
    float ambient_pressure = 101325.0f; ///< P_ambient [Pa]

    // Recoil pressure parameters
    bool enable_recoil_pressure = true;
    float recoil_coefficient = 0.54f;   ///< Empirical coefficient (0.54-0.56)

    // Explicit mode only
    bool enable_gas_thermal = false;    ///< Enable gas-phase thermal solver
    int gas_lbm_subcycles = 1;          ///< Gas LBM subcycles (gas has lower tau)

    // Numerical parameters
    float evap_mass_limit = 0.1f;       ///< Max mass fraction change per timestep
    float recoil_smoothing_width = 2.0f; ///< Interface smoothing [lattice units]

    GasPhaseConfig() = default;
};

/**
 * @brief Main gas phase module
 *
 * Orchestrates all gas-related physics:
 * - In IMPLICIT mode: Computes evaporation and recoil as source terms
 * - In EXPLICIT mode: Runs full gas-phase LBM solver
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
     */
    void computeEvaporationMassFlux(
        const float* temperature,
        const float* fill_level,
        const float3* interface_normal,
        float* d_mass_flux) const;

    /**
     * @brief Compute recoil pressure force
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param interface_normal Surface normal vectors
     * @param force_x, force_y, force_z Output: recoil force [N/m3]
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
     */
    void computeEvaporativeCooling(
        const float* d_mass_flux,
        float* d_heat_sink) const;

    /**
     * @brief Get evaporation mass source for VOF advection
     * @return Device pointer to mass source term [kg/(m3.s)]
     */
    const float* getMassSource() const { return d_mass_source_; }

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
    float* d_mass_flux_;        ///< Evaporation mass flux [kg/(m2.s)]
    float* d_mass_source_;      ///< VOF mass source term [kg/(m3.s)]
    float* d_heat_sink_;        ///< Evaporative cooling [W/m3]
    float* d_saturation_pressure_; ///< P_sat(T) [Pa]

    // Cumulative tracking
    mutable float total_evaporated_mass_;

    void allocateMemory();
    void freeMemory();
};

} // namespace physics
} // namespace lbm
```

### 4.2 EvaporationModel Class

```cpp
/**
 * @file evaporation_model.h
 * @brief Evaporation mass flux computation using Hertz-Knudsen equation
 */

#pragma once

#include <cuda_runtime.h>
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

/**
 * @brief Evaporation model for metal vapor generation
 *
 * Implements the Hertz-Knudsen model for evaporative mass flux:
 *   m_dot = (1 - beta_r) * p_sat(T) * sqrt(M / (2*pi*R*T))
 *
 * where beta_r is the recondensation coefficient (sticking coefficient).
 *
 * The saturation pressure is computed via Clausius-Clapeyron:
 *   p_sat(T) = p_ref * exp(L_v * M / R * (1/T_ref - 1/T))
 */
class EvaporationModel {
public:
    /**
     * @brief Constructor
     * @param material Material properties (for L_vaporization, etc.)
     * @param sticking_coeff Recondensation coefficient (0.82 typical for metals)
     * @param ambient_pressure Ambient pressure [Pa]
     */
    EvaporationModel(const MaterialProperties& material,
                     float sticking_coeff = 0.82f,
                     float ambient_pressure = 101325.0f);

    /**
     * @brief Compute saturation pressure at temperature T
     * @param T Temperature [K]
     * @return Saturation pressure [Pa]
     */
    __host__ __device__ float computeSaturationPressure(float T) const;

    /**
     * @brief Compute evaporation mass flux at temperature T
     * @param T Temperature [K]
     * @return Mass flux [kg/(m2.s)] (positive = evaporation)
     */
    __host__ __device__ float computeMassFlux(float T) const;

    /**
     * @brief Compute evaporation mass flux field (CUDA kernel wrapper)
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param mass_flux Output mass flux field [kg/(m2.s)]
     * @param nx, ny, nz Grid dimensions
     */
    void computeMassFluxField(
        const float* temperature,
        const float* fill_level,
        float* mass_flux,
        int nx, int ny, int nz) const;

    // Parameter access
    float getStickingCoefficient() const { return sticking_coeff_; }
    void setStickingCoefficient(float beta) { sticking_coeff_ = beta; }

private:
    MaterialProperties material_;
    float sticking_coeff_;      ///< Recondensation coefficient
    float p_ambient_;           ///< Ambient pressure [Pa]
    float M_molar_;             ///< Molar mass [kg/mol]
    float R_gas_;               ///< Gas constant [J/(mol.K)]
};

// CUDA kernels
__global__ void computeEvaporationMassFluxKernel(
    const float* temperature,
    const float* fill_level,
    float* mass_flux,
    float* saturation_pressure,
    EvaporationModel model,
    float dx,
    int nx, int ny, int nz);

} // namespace physics
} // namespace lbm
```

### 4.3 RecoilPressure Class

```cpp
/**
 * @file recoil_pressure.h
 * @brief Recoil pressure computation for evaporating metal surfaces
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Recoil pressure model
 *
 * Computes the recoil pressure from metal vapor evaporation:
 *   P_recoil = C_r * p_sat(T)
 *
 * where C_r is typically 0.54 (from kinetic theory of gases).
 *
 * The recoil pressure acts normal to the surface, directed into the liquid,
 * and is a primary driver of keyhole formation in high-power LPBF.
 */
class RecoilPressure {
public:
    /**
     * @brief Constructor
     * @param recoil_coefficient C_r coefficient (0.54 typical)
     * @param smoothing_width Interface smoothing width [lattice units]
     */
    RecoilPressure(float recoil_coefficient = 0.54f,
                   float smoothing_width = 2.0f);

    /**
     * @brief Compute recoil pressure from saturation pressure
     * @param p_sat Saturation pressure [Pa]
     * @return Recoil pressure [Pa]
     */
    __host__ __device__ float computePressure(float p_sat) const {
        return recoil_coeff_ * p_sat;
    }

    /**
     * @brief Compute recoil force field (volumetric force)
     * @param saturation_pressure P_sat(T) field [Pa]
     * @param fill_level VOF fill level (0-1)
     * @param interface_normal Surface normals
     * @param force_x, force_y, force_z Output forces [N/m3]
     * @param dx Lattice spacing [m]
     * @param nx, ny, nz Grid dimensions
     */
    void computeForceField(
        const float* saturation_pressure,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z,
        float dx,
        int nx, int ny, int nz) const;

    /**
     * @brief Add recoil force to existing force arrays
     */
    void addForceField(
        const float* saturation_pressure,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z,
        float dx,
        int nx, int ny, int nz) const;

    // Parameter access
    float getRecoilCoefficient() const { return recoil_coeff_; }
    void setRecoilCoefficient(float C_r) { recoil_coeff_ = C_r; }

private:
    float recoil_coeff_;        ///< C_r coefficient
    float smoothing_width_;     ///< Interface smoothing [lattice units]
};

// CUDA kernels
__global__ void computeRecoilForceKernel(
    const float* saturation_pressure,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    float recoil_coeff,
    float smoothing_width,
    float dx,
    int nx, int ny, int nz);

__global__ void addRecoilForceKernel(
    const float* saturation_pressure,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    float recoil_coeff,
    float smoothing_width,
    float dx,
    int nx, int ny, int nz);

} // namespace physics
} // namespace lbm
```

### 4.4 GasFlowSolver Class (Explicit Mode Only)

```cpp
/**
 * @file gas_flow_solver.h
 * @brief LBM solver for explicit gas phase flow (optional high-fidelity mode)
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Gas phase LBM solver (D3Q19)
 *
 * Full Navier-Stokes solver for the gas phase. Only instantiated when
 * GasMode::EXPLICIT is selected. Uses D3Q19 lattice like FluidLBM.
 *
 * Key differences from FluidLBM:
 * - Much lower density (1.2 kg/m3 vs 4420 kg/m3 for liquid Ti)
 * - Lower viscosity, smaller relaxation time
 * - May require subcycling relative to liquid phase
 * - Interface coupling with liquid velocity
 */
class GasFlowSolver {
public:
    /**
     * @brief Constructor
     * @param nx, ny, nz Grid dimensions
     * @param gas_density Gas density [kg/m3]
     * @param gas_viscosity Dynamic viscosity [Pa.s]
     * @param dt Time step [s]
     * @param dx Lattice spacing [m]
     */
    GasFlowSolver(int nx, int ny, int nz,
                  float gas_density,
                  float gas_viscosity,
                  float dt, float dx);

    ~GasFlowSolver();

    /**
     * @brief Initialize gas flow field
     * @param fill_level VOF fill level (0=gas region active)
     */
    void initialize(const float* fill_level);

    /**
     * @brief Perform gas LBM step
     * @param fill_level VOF fill level (determines active region)
     * @param liquid_ux, liquid_uy, liquid_uz Liquid velocity at interface
     * @param mass_source Evaporation mass source [kg/(m3.s)]
     */
    void step(const float* fill_level,
              const float* liquid_ux,
              const float* liquid_uy,
              const float* liquid_uz,
              const float* mass_source);

    // Velocity field access
    const float* getVelocityX() const { return d_gas_ux_; }
    const float* getVelocityY() const { return d_gas_uy_; }
    const float* getVelocityZ() const { return d_gas_uz_; }
    const float* getDensity() const { return d_gas_rho_; }

private:
    int nx_, ny_, nz_, num_cells_;
    float rho_gas_, mu_gas_;
    float tau_, omega_;
    float dt_, dx_;

    // Distribution functions
    float* d_g_src_;    // Gas distribution source
    float* d_g_dst_;    // Gas distribution destination

    // Macroscopic fields
    float* d_gas_rho_;
    float* d_gas_ux_;
    float* d_gas_uy_;
    float* d_gas_uz_;

    void allocateMemory();
    void freeMemory();
};

} // namespace physics
} // namespace lbm
```

---

## 5. Coupling Interface Design

### 5.1 Interface Data Exchange

```cpp
/**
 * @file interface_coupling.h
 * @brief Manages data exchange at phase interfaces
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Interface coupling manager
 *
 * Handles all data exchange between phases at interfaces:
 * - Gas-liquid: evaporation, recoil, momentum transfer
 * - Liquid-solid: Darcy damping (existing)
 * - Gas-solid: vapor deposition (future)
 */
class InterfaceCoupling {
public:
    InterfaceCoupling(int nx, int ny, int nz, float dx, float dt);
    ~InterfaceCoupling();

    // ========================================================================
    // Gas-Liquid Interface
    // ========================================================================

    /**
     * @brief Compute mass transfer at gas-liquid interface
     * @param evap_mass_flux Evaporation mass flux [kg/(m2.s)]
     * @param fill_level VOF fill level
     * @param vof_mass_source Output: mass source for VOF [kg/(m3.s)]
     */
    void computeGasLiquidMassTransfer(
        const float* evap_mass_flux,
        const float* fill_level,
        float* vof_mass_source);

    /**
     * @brief Apply interface velocity continuity (EXPLICIT gas mode)
     * @param liquid_ux, uy, uz Liquid velocity
     * @param gas_ux, uy, uz Gas velocity (modified at interface)
     * @param fill_level VOF fill level
     */
    void applyVelocityContinuity(
        const float* liquid_ux, const float* liquid_uy, const float* liquid_uz,
        float* gas_ux, float* gas_uy, float* gas_uz,
        const float* fill_level);

    /**
     * @brief Compute heat transfer at interface
     * @param liquid_T Liquid temperature
     * @param gas_T Gas temperature (or ambient)
     * @param fill_level VOF fill level
     * @param heat_flux Output: interface heat flux [W/m2]
     */
    void computeInterfaceHeatTransfer(
        const float* liquid_T,
        const float* gas_T,
        const float* fill_level,
        float* heat_flux);

private:
    int nx_, ny_, nz_, num_cells_;
    float dx_, dt_;

    // Interface cell identification
    uint8_t* d_interface_mask_;  // 1 = interface cell
};

} // namespace physics
} // namespace lbm
```

### 5.2 Data Flow Diagram

```
                        +------------------+
                        |  LaserSource     |
                        |  Q_laser(x,y,z)  |
                        +--------+---------+
                                 |
                                 v
+----------------+      +------------------+      +------------------+
|  ThermalLBM    |<---->|  Temperature     |----->| EvaporationModel |
|  (D3Q7)        |      |  T(x,y,z)        |      | m_dot(T)         |
+-------+--------+      +------------------+      +--------+---------+
        |                        ^                         |
        |                        |                         v
        |               +--------+--------+       +------------------+
        |               | MaterialProps   |       | RecoilPressure   |
        |               | p_sat(T), L_v   |       | P_r = 0.54*p_sat |
        |               +-----------------+       +--------+---------+
        |                                                  |
        v                                                  v
+-------+--------+      +------------------+      +------------------+
|  FluidLBM      |<-----|  Force Fields    |<-----| GasPhaseModule   |
|  (D3Q19)       |      |  F_total         |      | (coordinates)    |
+-------+--------+      +------------------+      +------------------+
        |                        ^
        |                        |
        v                        |
+-------+--------+      +--------+---------+
|  VOFSolver     |----->| SurfaceTension   |
|  fill_level    |      | + Marangoni      |
+----------------+      +------------------+
```

### 5.3 Coupling Equations

**Mass Conservation at Interface:**
```
d(fill_level)/dt + div(f * u) = -m_dot / (rho_liquid * dx)
```

**Momentum Balance at Interface:**
```
F_total = F_marangoni + F_surface_tension + F_recoil + F_buoyancy
F_recoil = P_recoil * n * |grad(f)| / h_interface
```

**Energy Balance at Interface:**
```
Q_total = Q_laser - Q_radiation - Q_evaporation - Q_convection
Q_evaporation = m_dot * L_vaporization
```

---

## 6. Execution Sequence

### 6.1 Modified MultiphysicsSolver::step()

```cpp
void MultiphysicsSolver::step(float dt) {
    if (dt == 0.0f) dt = config_.dt;

    // ========================================================================
    // Phase 1: Heat Sources (before thermal solve)
    // ========================================================================

    // 1.1 Laser heat source
    if (config_.enable_laser && laser_) {
        applyLaserSource(dt);
    }

    // ========================================================================
    // Phase 2: Thermal Evolution
    // ========================================================================

    // 2.1 Thermal diffusion and advection
    if (config_.enable_thermal && thermal_) {
        thermalStep(dt);
    }

    // ========================================================================
    // Phase 3: Gas Phase Computations [NEW]
    // ========================================================================

    if (config_.enable_gas_phase && gas_phase_) {
        // 3.1 Compute evaporation mass flux from T field
        gas_phase_->computeEvaporationMassFlux(
            getTemperature(),
            vof_->getFillLevel(),
            vof_->getInterfaceNormals(),
            d_evap_mass_flux_
        );

        // 3.2 Compute evaporative cooling (heat sink)
        if (config_.enable_evaporative_cooling) {
            gas_phase_->computeEvaporativeCooling(
                d_evap_mass_flux_,
                d_evap_heat_sink_
            );
            // Apply to thermal solver
            thermal_->addHeatSource(d_evap_heat_sink_, dt);  // negative = sink
        }

        // 3.3 Explicit gas flow (if enabled)
        if (gas_phase_->isExplicitMode()) {
            gas_phase_->stepGasFlow(
                fluid_->getVelocityX(),
                fluid_->getVelocityY(),
                fluid_->getVelocityZ(),
                dt
            );
        }
    }

    // ========================================================================
    // Phase 4: VOF Interface Management
    // ========================================================================

    if (vof_) {
        if (config_.enable_vof_advection) {
            // 4.1 VOF advection with mass source
            if (config_.enable_gas_phase && gas_phase_) {
                vofStepWithMassSource(dt, gas_phase_->getMassSource());
            } else {
                vofStep(dt);
            }
        } else {
            vof_->reconstructInterface();
        }

        // 4.2 Compute curvature for surface tension
        if (config_.enable_surface_tension) {
            vof_->computeCurvature();
        }
    }

    // ========================================================================
    // Phase 5: Fluid Flow
    // ========================================================================

    if (config_.enable_fluid && fluid_) {
        fluidStep(dt);  // Modified to include recoil force
    }

    // ========================================================================
    // Phase 6: Update and Diagnostics
    // ========================================================================

    current_time_ += dt;
    current_step_++;

    if (current_step_ % diagnostic_interval_ == 0) {
        printEnergyBalance();
    }
}
```

### 6.2 Modified computeTotalForce()

```cpp
void MultiphysicsSolver::computeTotalForce(float* d_fx, float* d_fy, float* d_fz) {
    int num_cells = config_.nx * config_.ny * config_.nz;
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    // Zero forces
    zeroForceKernel<<<blocks, threads>>>(d_fx, d_fy, d_fz, num_cells);
    cudaDeviceSynchronize();

    const float* temperature = getTemperature();
    const float* fill_level = vof_ ? vof_->getFillLevel() : nullptr;
    const float3* normals = vof_ ? vof_->getInterfaceNormals() : nullptr;

    // 1. Marangoni force (existing)
    if (config_.enable_marangoni && marangoni_ && fill_level && normals) {
        marangoni_->addMarangoniForce(temperature, fill_level, normals,
                                       d_fx, d_fy, d_fz);
    }

    // 2. Surface tension force (existing)
    if (config_.enable_surface_tension && surface_tension_ && vof_) {
        surface_tension_->addCSFForce(fill_level, vof_->getCurvature(),
                                       d_fx, d_fy, d_fz);
    }

    // 3. Recoil pressure force [NEW]
    if (config_.enable_recoil_pressure && gas_phase_ && fill_level && normals) {
        gas_phase_->addRecoilForce(temperature, fill_level, normals,
                                    d_fx, d_fy, d_fz);
    }

    // 4. Buoyancy force (existing)
    if (config_.enable_buoyancy && fluid_ && temperature) {
        fluid_->computeBuoyancyForce(temperature,
                                      config_.reference_temperature,
                                      config_.thermal_expansion_coeff,
                                      config_.gravity_x, config_.gravity_y, config_.gravity_z,
                                      d_fx, d_fy, d_fz);
    }

    // 5. Darcy damping (existing)
    if (config_.enable_darcy && fluid_) {
        const float* liquid_fraction = getLiquidFraction();
        if (liquid_fraction) {
            fluid_->applyDarcyDamping(liquid_fraction, config_.darcy_coefficient,
                                       d_fx, d_fy, d_fz);
        }
    }
}
```

### 6.3 Execution Order Summary

| Step | Component | Input | Output |
|------|-----------|-------|--------|
| 1 | LaserSource | position, time | Q_laser(x,y,z) |
| 2 | ThermalLBM | Q_laser, u, BCs | T(x,y,z) |
| 3.1 | EvaporationModel | T, f, n | m_dot |
| 3.2 | EvaporativeCooling | m_dot | Q_evap (sink) |
| 3.3 | GasFlowSolver* | u_liquid, m_dot | u_gas |
| 4.1 | VOFSolver | u, m_source | f(x,y,z) |
| 4.2 | VOF::curvature | f | kappa |
| 5.1 | computeTotalForce | T, f, n, kappa | F_total |
| 5.2 | FluidLBM | F_total | u(x,y,z) |

*Only in EXPLICIT gas mode

---

## 7. File Organization

### 7.1 Directory Structure

```
LBMProject/
|-- include/
|   |-- physics/
|   |   |-- gas_phase_module.h       [NEW] Main gas phase interface
|   |   |-- evaporation_model.h      [NEW] Hertz-Knudsen evaporation
|   |   |-- recoil_pressure.h        [NEW] Recoil pressure model
|   |   |-- gas_flow_solver.h        [NEW] Explicit gas LBM
|   |   |-- interface_coupling.h     [NEW] Interface data exchange
|   |   |
|   |   |-- multiphysics_solver.h    [MODIFY] Add gas phase member
|   |   |-- vof_solver.h             [MODIFY] Add mass source support
|   |   |-- ... (existing)
|   |
|-- src/
|   |-- physics/
|   |   |-- gas/                     [NEW DIRECTORY]
|   |   |   |-- gas_phase_module.cu
|   |   |   |-- evaporation_model.cu
|   |   |   |-- recoil_pressure.cu
|   |   |   |-- gas_flow_solver.cu
|   |   |   |-- interface_coupling.cu
|   |   |
|   |   |-- multiphysics/
|   |   |   |-- multiphysics_solver.cu  [MODIFY]
|   |   |
|   |   |-- vof/
|   |   |   |-- vof_solver.cu           [MODIFY]
|   |
|-- tests/
|   |-- unit/
|   |   |-- gas/                     [NEW DIRECTORY]
|   |   |   |-- test_evaporation_model.cu
|   |   |   |-- test_recoil_pressure.cu
|   |   |   |-- test_gas_phase_module.cu
|   |
|   |-- integration/
|   |   |-- test_evaporation_mass_conservation.cu  [NEW]
|   |   |-- test_keyhole_formation.cu              [NEW]
|   |   |-- test_recoil_driven_flow.cu             [NEW]
|   |
|-- configs/
|   |-- gas_phase_config.json        [NEW] Example configuration
|   |-- lpbf_with_gas.yaml           [NEW] Full LPBF config with gas
```

### 7.2 Naming Conventions

Following existing codebase patterns:

| Element | Convention | Example |
|---------|------------|---------|
| Header files | lowercase_with_underscores.h | `gas_phase_module.h` |
| Source files | lowercase_with_underscores.cu | `evaporation_model.cu` |
| Classes | PascalCase | `GasPhaseModule` |
| Methods | camelCase | `computeEvaporationMassFlux()` |
| CUDA kernels | camelCase + Kernel suffix | `computeRecoilForceKernel()` |
| Config structs | PascalCase + Config suffix | `GasPhaseConfig` |
| Device pointers | d_ prefix | `d_mass_flux_` |
| Constants | UPPER_CASE | `MAX_EVAP_RATE` |

---

## 8. Configuration Extension

### 8.1 Extended MultiphysicsConfig

```cpp
struct MultiphysicsConfig {
    // ... existing fields ...

    // ========================================================================
    // Gas Phase Configuration [NEW]
    // ========================================================================

    bool enable_gas_phase = false;          ///< Master switch for gas phase
    bool enable_evaporation = true;         ///< Enable evaporative mass loss
    bool enable_recoil_pressure = true;     ///< Enable recoil pressure force
    bool enable_evaporative_cooling = true; ///< Enable evaporative heat sink

    GasMode gas_mode = GasMode::IMPLICIT;   ///< IMPLICIT or EXPLICIT gas solver

    // Gas properties (Argon default)
    float gas_density = 1.2f;               ///< Gas density [kg/m3]
    float gas_viscosity = 2.2e-5f;          ///< Dynamic viscosity [Pa.s]
    float gas_thermal_conductivity = 0.018f;///< k_gas [W/(m.K)]
    float gas_specific_heat = 520.0f;       ///< cp_gas [J/(kg.K)]

    // Evaporation parameters
    float sticking_coefficient = 0.82f;     ///< Recondensation coefficient
    float ambient_pressure = 101325.0f;     ///< Ambient pressure [Pa]

    // Recoil pressure parameters
    float recoil_coefficient = 0.54f;       ///< P_recoil = C_r * p_sat
    float recoil_smoothing_width = 2.0f;    ///< Interface smoothing [cells]

    // Explicit gas mode parameters
    bool enable_gas_thermal = false;        ///< Gas-phase thermal solver
    int gas_lbm_subcycles = 1;              ///< Subcycles for gas LBM

    // ========================================================================
    // Constructor update
    // ========================================================================
    MultiphysicsConfig()
        : // ... existing initializers ...
          enable_gas_phase(false),
          enable_evaporation(true),
          enable_recoil_pressure(true),
          enable_evaporative_cooling(true),
          gas_mode(GasMode::IMPLICIT),
          gas_density(1.2f),
          gas_viscosity(2.2e-5f),
          gas_thermal_conductivity(0.018f),
          gas_specific_heat(520.0f),
          sticking_coefficient(0.82f),
          ambient_pressure(101325.0f),
          recoil_coefficient(0.54f),
          recoil_smoothing_width(2.0f),
          enable_gas_thermal(false),
          gas_lbm_subcycles(1)
    {}
};
```

### 8.2 JSON Configuration Example

```json
{
  "simulation": {
    "name": "LPBF_Ti6Al4V_with_gas",
    "grid": { "nx": 200, "ny": 200, "nz": 100 },
    "dx": 2e-6,
    "dt": 1e-8,
    "total_time": 100e-6
  },

  "material": "Ti6Al4V",

  "physics": {
    "thermal": {
      "enabled": true,
      "advection": true,
      "phase_change": true,
      "radiation_bc": true,
      "substrate_cooling": true
    },

    "fluid": {
      "enabled": true,
      "buoyancy": true,
      "darcy_damping": true
    },

    "vof": {
      "enabled": true,
      "advection": true,
      "surface_tension": true,
      "marangoni": true
    },

    "gas_phase": {
      "enabled": true,
      "mode": "IMPLICIT",
      "evaporation": {
        "enabled": true,
        "sticking_coefficient": 0.82,
        "cooling_enabled": true
      },
      "recoil_pressure": {
        "enabled": true,
        "coefficient": 0.54,
        "smoothing_width": 2.0
      },
      "properties": {
        "density": 1.2,
        "viscosity": 2.2e-5,
        "thermal_conductivity": 0.018,
        "specific_heat": 520.0
      }
    },

    "laser": {
      "enabled": true,
      "power": 200,
      "spot_radius": 50e-6,
      "absorptivity": 0.35,
      "penetration_depth": 10e-6,
      "scan_velocity": [0.5, 0.0]
    }
  }
}
```

---

## 9. Implementation Roadmap

### 9.1 Phase 7a: Core Evaporation (2-3 weeks)

**Goal:** Basic evaporation mass flux and recoil pressure

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| EvaporationModel class | evaporation_model.h/cu | 3 days | MaterialProperties |
| RecoilPressure class | recoil_pressure.h/cu | 2 days | EvaporationModel |
| GasPhaseConfig struct | gas_phase_module.h | 1 day | - |
| MultiphysicsConfig extension | multiphysics_solver.h | 1 day | - |
| Recoil force in computeTotalForce() | multiphysics_solver.cu | 2 days | RecoilPressure |
| Unit tests | test_evaporation_model.cu | 2 days | EvaporationModel |

**Validation:**
- Clausius-Clapeyron p_sat(T) vs analytical
- Hertz-Knudsen m_dot(T) vs literature values
- Recoil pressure magnitude vs keyhole threshold

### 9.2 Phase 7b: Evaporative Cooling & Mass Source (1-2 weeks)

**Goal:** Complete evaporation physics with energy and mass coupling

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| Evaporative cooling computation | gas_phase_module.cu | 2 days | EvaporationModel |
| Heat sink in thermalStep() | multiphysics_solver.cu | 1 day | ThermalLBM |
| Mass source in vofStep() | vof_solver.cu, multiphysics.cu | 3 days | VOFSolver |
| Energy balance update | energy_balance.cu | 1 day | diagnostics |
| Integration tests | test_evaporation_mass_conservation.cu | 2 days | All above |

**Validation:**
- Energy balance: P_laser = P_rad + P_evap + dE/dt
- Mass conservation: d(mass)/dt = -integral(m_dot)
- Temperature saturation near T_boil

### 9.3 Phase 7c: GasPhaseModule Integration (1-2 weeks)

**Goal:** Unified gas phase module with IMPLICIT mode

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| GasPhaseModule class | gas_phase_module.h/cu | 3 days | All above |
| Integration in MultiphysicsSolver | multiphysics_solver.h/cu | 2 days | GasPhaseModule |
| Diagnostic output | multiphysics_solver.cu | 1 day | - |
| Config loading | config_loader.cu (new or extend) | 2 days | - |
| Full system test | test_keyhole_formation.cu | 2 days | All above |

**Validation:**
- Keyhole formation at high power
- Keyhole depth scaling with power
- Stable simulation at various powers

### 9.4 Phase 7d: Explicit Gas Mode (Optional, 3-4 weeks)

**Goal:** Full gas-phase LBM for high-fidelity simulations

| Task | Files | Effort | Dependencies |
|------|-------|--------|--------------|
| GasFlowSolver class | gas_flow_solver.h/cu | 5 days | FluidLBM pattern |
| InterfaceCoupling class | interface_coupling.h/cu | 3 days | VOFSolver |
| Velocity continuity at interface | interface_coupling.cu | 2 days | - |
| Gas thermal solver | gas_thermal.h/cu | 3 days | ThermalLBM pattern |
| Two-way coupling | multiphysics_solver.cu | 3 days | All above |
| Validation suite | test_vapor_jet.cu | 3 days | All above |

**Validation:**
- Vapor jet velocity
- Powder entrainment (qualitative)
- Gas convection effects on melt pool

---

## 10. Validation Strategy

### 10.1 Unit Test Cases

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| UT-E1 | Saturation pressure | p_sat(T) within 5% of reference data |
| UT-E2 | Mass flux vs T | m_dot monotonically increases with T |
| UT-E3 | Recoil force direction | F_recoil points into liquid (along -n) |
| UT-R1 | Recoil magnitude | P_r = 0.54 * p_sat within 1% |
| UT-R2 | Force smoothing | Force localized to interface |

### 10.2 Integration Test Cases

| Test | Description | Acceptance Criteria |
|------|-------------|---------------------|
| IT-M1 | Mass conservation | Total mass error < 1% over 1000 steps |
| IT-E1 | Energy balance | Balance error < 10% |
| IT-K1 | Keyhole threshold | Keyhole forms above critical power |
| IT-K2 | Keyhole depth | Depth increases with power |
| IT-F1 | Flow stability | No divergence after 10000 steps |

### 10.3 Reference Data

**Ti6Al4V Evaporation Properties:**

| Property | Value | Source |
|----------|-------|--------|
| T_boil | 3560 K | [Khairallah 2016] |
| L_vaporization | 8.9e6 J/kg | [Metals Handbook] |
| p_sat at T_boil | 101325 Pa | Definition |
| beta_r (sticking) | 0.82 | [Anisimov 1995] |
| C_r (recoil) | 0.54 | [Knight 1979] |

**Expected Behaviors:**

1. **Onset of evaporation:** T > 2800 K (significant p_sat)
2. **Keyhole threshold:** P_laser > 150-200 W (for 50um spot)
3. **Recoil pressure:** 1-100 kPa range at keyhole surface
4. **Evaporation rate:** 0.1-10 kg/(m2.s) at melt pool surface

---

## Appendix A: CUDA Kernel Signatures

```cpp
// Evaporation mass flux computation
__global__ void computeEvaporationMassFluxKernel(
    const float* __restrict__ temperature,
    const float* __restrict__ fill_level,
    float* __restrict__ mass_flux,
    float* __restrict__ saturation_pressure,
    float T_boil,
    float L_vaporization,
    float M_molar,
    float sticking_coeff,
    float dx,
    int nx, int ny, int nz);

// Recoil force computation
__global__ void computeRecoilForceKernel(
    const float* __restrict__ saturation_pressure,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float recoil_coeff,
    float smoothing_width,
    float dx,
    int nx, int ny, int nz);

// Evaporative cooling heat sink
__global__ void computeEvaporativeCoolingKernel(
    const float* __restrict__ mass_flux,
    float* __restrict__ heat_sink,
    float L_vaporization,
    float dx,
    int nx, int ny, int nz);

// VOF advection with mass source
__global__ void advectFillLevelWithMassSourceKernel(
    const float* __restrict__ fill_level,
    float* __restrict__ fill_level_new,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    const float* __restrict__ mass_source,
    float rho_liquid,
    float dt,
    float dx,
    int nx, int ny, int nz);
```

---

## Appendix B: Memory Layout

All fields use Structure-of-Arrays (SoA) layout consistent with existing codebase:

```
Device Memory Map (IMPLICIT mode, 100x100x50 grid):
+------------------+------------+---------------------------+
| Field            | Size       | Notes                     |
+------------------+------------+---------------------------+
| d_mass_flux_     | 2 MB       | Evaporation rate          |
| d_mass_source_   | 2 MB       | VOF source term           |
| d_heat_sink_     | 2 MB       | Evaporative cooling       |
| d_saturation_p_  | 2 MB       | p_sat(T) cached           |
+------------------+------------+---------------------------+
| Total new memory | ~8 MB      | ~16% of current ~50 MB    |
+------------------+------------+---------------------------+

Device Memory Map (EXPLICIT mode, adds):
+------------------+------------+---------------------------+
| d_g_gas_src      | 38 MB      | Gas dist. (19 x 2 MB)     |
| d_g_gas_dst      | 38 MB      | Gas dist. (19 x 2 MB)     |
| d_gas_rho        | 2 MB       | Gas density               |
| d_gas_ux/uy/uz   | 6 MB       | Gas velocity              |
+------------------+------------+---------------------------+
| Total EXPLICIT   | ~92 MB     | Nearly doubles memory     |
+------------------+------------+---------------------------+
```

---

## Appendix C: Performance Considerations

### C.1 Kernel Optimization Guidelines

1. **Coalesced memory access:** All field arrays indexed by `i + nx*(j + ny*k)`
2. **Minimize atomic operations:** Use per-block reductions
3. **Register pressure:** Limit local variables in mass flux kernel
4. **Occupancy:** Target 50%+ occupancy for interface kernels

### C.2 Expected Performance Impact

| Mode | Overhead | Notes |
|------|----------|-------|
| Gas disabled | 0% | Baseline |
| IMPLICIT, no evap cooling | 5-10% | Mass flux + recoil only |
| IMPLICIT, full | 15-25% | With evap cooling |
| EXPLICIT | 80-120% | Full gas LBM |

### C.3 Optimization Opportunities

1. **Fuse kernels:** Combine mass flux + recoil computation
2. **Skip empty regions:** Only compute at interface cells
3. **Async transfers:** Overlap energy balance computation with main loop
4. **Multi-GPU:** Gas and liquid solvers on separate GPUs (future)

---

*Document End*
