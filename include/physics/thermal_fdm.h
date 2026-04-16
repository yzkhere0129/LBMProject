/**
 * @file thermal_fdm.h
 * @brief Explicit FDM thermal solver — drop-in replacement for ThermalLBM
 *
 * Solves: dT/dt + u·nabla(T) = alpha·laplacian(T) + S_laser + S_phase
 *
 * Discretization:
 *   - Advection: first-order upwind (branch-free, TVD upgrade possible)
 *   - Diffusion: second-order central difference (7-point stencil)
 *   - Time: explicit Euler (with optional subcycling for CFL)
 *   - Phase change: Enthalpy Source Method (Jiaung 2001), same as LBM ESM
 *
 * Stability: 6·Fo + sum|Cd| <= 1, where Fo = alpha·dt/dx², Cd = u·dt/dx
 *
 * Memory: 2×N floats (T ping-pong) + N floats (liquid fraction) = 3N
 *         vs D3Q7 LBM: 14N + N + N = 16N. Savings: 5.3×
 *
 * Key advantage over D3Q7: No Mach number constraint on the advection term.
 * The velocity field from D3Q19 FluidLBM enters as a coefficient in the
 * FDM upwind scheme, not through an equilibrium distribution function.
 * This eliminates the low-Pr deadlock that plagues D3Q7 for metals.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "physics/thermal_solver_interface.h"
#include "physics/phase_change.h"
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

class ThermalFDM : public IThermalSolver {
public:
    ThermalFDM(int nx, int ny, int nz,
               const MaterialProperties& material,
               float thermal_diffusivity,
               bool enable_phase_change,
               float dt, float dx);
    ~ThermalFDM() override;

    // Non-copyable, movable
    ThermalFDM(const ThermalFDM&) = delete;
    ThermalFDM& operator=(const ThermalFDM&) = delete;
    ThermalFDM(ThermalFDM&&) noexcept;
    ThermalFDM& operator=(ThermalFDM&&) = delete;

    // ================================================================
    // IThermalSolver implementation
    // ================================================================
    void initialize(float initial_temp) override;
    void initialize(const float* temp_field) override;

    // collisionBGK → FDM advection + diffusion (writes to d_T_new_)
    void collisionBGK(const float* ux = nullptr,
                      const float* uy = nullptr,
                      const float* uz = nullptr) override;

    // streaming → buffer swap (d_T_ ↔ d_T_new_)
    void streaming() override;

    // computeTemperature → ESM phase change correction on d_T_
    void computeTemperature() override;

    void addHeatSource(const float* heat_source, float dt) override;

    void applyBoundaryConditions(int boundary_type,
                                 float boundary_value = 0.0f) override;
    void applyFaceThermalBC(int face, int bc_type,
                            float dt, float dx,
                            float dirichlet_T = 300.0f,
                            float h_conv = 1000.0f,
                            float T_inf = 300.0f,
                            float emissivity = 0.3f,
                            float T_ambient = 300.0f) override;
    void applyRadiationBC(float dt, float dx,
                          float epsilon = 0.35f,
                          float T_ambient = 300.0f) override;
    void applySubstrateCoolingBC(float dt, float dx,
                                 float h_conv, float T_substrate) override;
    void applyEvaporationCooling(const float* J_evap,
                                 const float* fill_level,
                                 float dt, float dx,
                                 float cooling_factor = 1.0f) override;
    void applyTemperatureSafetyCap() override;
    void applyTemperatureFailsafeCap(float T_max);

    float* getTemperature() override { return d_T_; }
    const float* getTemperature() const override { return d_T_; }
    float* getLiquidFraction() override;
    const float* getLiquidFraction() const override;

    bool hasPhaseChange() const override { return phase_solver_ != nullptr; }
    PhaseChangeSolver* getPhaseChangeSolver() override { return phase_solver_; }

    void setEmissivity(float eps) override { emissivity_ = eps; }
    void setZPeriodic(bool enable) override { z_periodic_ = enable; }
    void setSubcycleCount(int n) { n_subcycle_ = n; }
    void setVOFFillLevel(const float* fill) override { d_vof_fill_ = fill; }
    void setSkipTemperatureCap(bool skip) override { skip_T_cap_ = skip; }

    void copyTemperatureToHost(float* h_T) const override;
    void copyLiquidFractionToHost(float* h_fl) const override;

    float computeTotalThermalEnergy(float dx) const override;
    float computeEvaporationPower(const float* fill, float dx) const override;
    float computeRadiationPower(const float* fill, float dx,
                                float eps, float T_amb) const override;
    float computeSubstratePower(float dx, float h, float T_sub) const override;
    float computeCapPower(float dx, float dt) const override;
    void computeEvaporationMassFlux(float* d_J, const float* fill) const override;

    int getNx() const override { return nx_; }
    int getNy() const override { return ny_; }
    int getNz() const override { return nz_; }
    float getThermalTau() const override { return 0.0f; } // N/A for FDM
    float getDensity() const override { return material_.rho_solid; }
    float getSpecificHeat() const override { return material_.cp_solid; }
    const MaterialProperties& getMaterialProperties() const override { return material_; }

private:
    int nx_, ny_, nz_, num_cells_;
    float dt_, dx_;
    float alpha_phys_;
    float emissivity_ = 0.3f;
    bool z_periodic_ = false;
    bool skip_T_cap_ = false;
    int n_subcycle_ = 1;

    MaterialProperties material_;
    PhaseChangeSolver* phase_solver_ = nullptr;

    // Device memory (ping-pong T buffers)
    float* d_T_     = nullptr;  // current temperature
    float* d_T_new_ = nullptr;  // next temperature
    const float* d_vof_fill_ = nullptr;

    // Gas wipe protection mask: 1 = near interface (protected), 0 = far field (wipe)
    uint8_t* d_gas_wipe_mask_ = nullptr;

    // Energy tracking for gas wipe and subsurface cap (fixed-point, scale 1e12)
    unsigned long long* d_gas_wipe_energy_raw_ = nullptr;
    unsigned long long* d_boiling_cap_energy_raw_ = nullptr;

    void allocateMemory();
    void freeMemory();
    void swapBuffers() { float* tmp = d_T_; d_T_ = d_T_new_; d_T_new_ = tmp; }
    void computeGasWipeProtectionMask(int protection_layers = 5);

public:
    /// Apply sub-surface boiling cap: T_cap = T_boil + overshoot_K for f>=0.99 cells
    void applySubsurfaceBoilingCap(float T_boil, float overshoot_K = 50.0f);

    /// Get energy removed by gas wipe this step [J], resets counter
    double getGasWipeEnergyRemoved();

    /// Get energy removed by sub-surface boiling cap this step [J], resets counter
    double getBoilingCapEnergyRemoved();

    /// Compute thermal energy of metal cells only (fill >= 0.01) [J]
    double computeMetalThermalEnergy(float dx) const override;
};

} // namespace physics
} // namespace lbm
