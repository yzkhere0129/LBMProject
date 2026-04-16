/**
 * @file thermal_solver_interface.h
 * @brief Abstract interface for thermal solvers (LBM or FDM)
 *
 * Allows MultiphysicsSolver to use either ThermalLBM (D3Q7) or ThermalFDM
 * (explicit finite difference) without changing any call-site code.
 *
 * Design: Pure virtual interface. No data members, no implementation.
 * Derived classes own all GPU memory and implement all physics.
 */

#pragma once

#include "physics/material_properties.h"

namespace lbm {
namespace physics {

// Forward declaration
class PhaseChangeSolver;

class IThermalSolver {
public:
    virtual ~IThermalSolver() = default;

    // ================================================================
    // Initialization
    // ================================================================
    virtual void initialize(float initial_temp) = 0;
    virtual void initialize(const float* temp_field) = 0;

    // ================================================================
    // Per-step update (called by MultiphysicsSolver::thermalStep)
    //
    // Call sequence preserved from ThermalLBM:
    //   1. addHeatSource (laser)
    //   2. applyFaceThermalBC (pre-step BCs)
    //   3. collisionBGK (LBM: collision; FDM: advection+diffusion)
    //   4. streaming    (LBM: propagate; FDM: buffer swap)
    //   5. computeTemperature (LBM: sum g + ESM; FDM: ESM only)
    //   6. applyFaceThermalBC (post-step Dirichlet re-apply)
    // ================================================================
    virtual void collisionBGK(const float* ux = nullptr,
                              const float* uy = nullptr,
                              const float* uz = nullptr) = 0;
    virtual void streaming() = 0;
    virtual void computeTemperature() = 0;

    virtual void addHeatSource(const float* heat_source, float dt) = 0;

    virtual void applyBoundaryConditions(int boundary_type,
                                         float boundary_value = 0.0f) = 0;

    virtual void applyFaceThermalBC(int face, int bc_type,
                                    float dt, float dx,
                                    float dirichlet_T = 300.0f,
                                    float h_conv = 1000.0f,
                                    float T_inf = 300.0f,
                                    float emissivity = 0.3f,
                                    float T_ambient = 300.0f) = 0;

    virtual void applyRadiationBC(float dt, float dx,
                                  float epsilon = 0.35f,
                                  float T_ambient = 300.0f) = 0;

    virtual void applySubstrateCoolingBC(float dt, float dx,
                                         float h_conv, float T_substrate) = 0;

    virtual void applyEvaporationCooling(const float* J_evap,
                                         const float* fill_level,
                                         float dt, float dx,
                                         float cooling_factor = 1.0f) = 0;

    virtual void applyTemperatureSafetyCap() = 0;
    /// Hard cap at an arbitrary ceiling (failsafe — does not use material T_vaporization)
    virtual void applyTemperatureFailsafeCap(float T_max) { applyTemperatureSafetyCap(); (void)T_max; }

    // Sub-surface boiling cap (FDM only; no-op for LBM)
    virtual void applySubsurfaceBoilingCap(float T_boil, float overshoot_K = 50.0f) { (void)T_boil; (void)overshoot_K; }

    // Energy tracking for gas wipe and boiling cap [J], resets counter each call
    virtual double getGasWipeEnergyRemoved() { return 0.0; }
    virtual double getBoilingCapEnergyRemoved() { return 0.0; }

    // Metal-only thermal energy (fill >= 0.01) for diagnostic isolation
    virtual double computeMetalThermalEnergy(float dx) const { return 0.0; }

    /**
     * @brief Snapshot current T field for bisection-based ESM phase change inversion.
     *
     * Must be called at the top of each step BEFORE any T-modifying operation
     * (laser, heat source, diffusion, cooling). Bisection ESM reconstructs the
     * exact delivered energy as ρ(T_old)·cp(T_old)·(T*−T_old) and inverts the
     * true H(T) to recover T_new honoring latent heat.
     *
     * Default: no-op (solvers without bisection ESM ignore this).
     */
    virtual void storePreviousTemperature() {}

    // ================================================================
    // Field access (device pointers)
    // ================================================================
    virtual float* getTemperature() = 0;
    virtual const float* getTemperature() const = 0;
    virtual float* getLiquidFraction() = 0;
    virtual const float* getLiquidFraction() const = 0;

    // D3Q7 distribution access (LBM only; FDM returns nullptr)
    virtual float* getDistributionSrc() { return nullptr; }

    // ================================================================
    // Phase change
    // ================================================================
    virtual bool hasPhaseChange() const = 0;
    virtual PhaseChangeSolver* getPhaseChangeSolver() = 0;

    // ================================================================
    // Configuration
    // ================================================================
    virtual void setEmissivity(float eps) = 0;
    virtual void setZPeriodic(bool enable) = 0;
    virtual void setVOFFillLevel(const float* fill_level) = 0;
    virtual void setSkipTemperatureCap(bool skip) = 0;

    // ================================================================
    // Diagnostics
    // ================================================================
    virtual void copyTemperatureToHost(float* host_temp) const = 0;
    virtual void copyLiquidFractionToHost(float* host_fl) const = 0;

    virtual float computeTotalThermalEnergy(float dx) const = 0;
    virtual float computeEvaporationPower(const float* fill_level,
                                          float dx) const = 0;
    virtual float computeRadiationPower(const float* fill_level, float dx,
                                        float epsilon, float T_ambient) const = 0;
    virtual float computeSubstratePower(float dx, float h_conv,
                                        float T_substrate) const = 0;
    virtual float computeCapPower(float dx, float dt) const = 0;
    virtual void computeEvaporationMassFlux(float* d_J_evap,
                                            const float* fill_level) const = 0;

    // ================================================================
    // Dimension & material accessors
    // ================================================================
    virtual int getNx() const = 0;
    virtual int getNy() const = 0;
    virtual int getNz() const = 0;
    virtual float getThermalTau() const = 0;
    virtual float getDensity() const = 0;
    virtual float getSpecificHeat() const = 0;
    virtual const MaterialProperties& getMaterialProperties() const = 0;
};

} // namespace physics
} // namespace lbm
