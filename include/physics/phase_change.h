/**
 * @file phase_change.h
 * @brief Phase change solver using enthalpy method for melting/solidification
 *
 * This module implements the enthalpy-based phase change model for metal
 * additive manufacturing simulations. It handles:
 * - Enthalpy-temperature coupling with latent heat
 * - Liquid fraction evolution in mushy zone
 * - Phase-dependent property updates
 * - Energy conservation
 *
 * Physical basis:
 * Total enthalpy: H = ρ·cp·T + fl·ρ·L_fusion
 * where fl(T) is the liquid fraction:
 *   fl = 0                           if T < T_solidus
 *   fl = (T - T_solidus)/ΔT_melt     if T_solidus ≤ T ≤ T_liquidus
 *   fl = 1                           if T > T_liquidus
 *
 * The enthalpy method solves:
 *   ∂H/∂t = ∇·(k∇T) + Q
 *
 * Requiring iterative solution for T from H since fl = fl(T) is nonlinear.
 */

#pragma once

#include <cuda_runtime.h>
#include "physics/material_properties.h"

namespace lbm {
namespace physics {

/**
 * @brief Phase change solver class
 *
 * This class manages the enthalpy-temperature coupling and phase transitions.
 * It integrates with the thermal LBM solver to handle melting and solidification.
 */
class PhaseChangeSolver {
public:
    /**
     * @brief Constructor
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param material Material properties
     */
    PhaseChangeSolver(int nx, int ny, int nz, const MaterialProperties& material);

    /**
     * @brief Destructor
     */
    ~PhaseChangeSolver();

    /**
     * @brief Initialize enthalpy field from temperature field
     * @param temperature Device array of temperature values [K]
     */
    void initializeFromTemperature(const float* temperature);

    /**
     * @brief Update enthalpy field based on temperature changes
     * @param temperature Device array of temperature values [K]
     *
     * This computes: H = ρ·cp·T + fl(T)·ρ·L_fusion
     */
    void updateEnthalpyFromTemperature(const float* temperature);

    /**
     * @brief Solve for temperature from enthalpy field (iterative)
     * @param temperature Device array to store computed temperature [K]
     * @param tolerance Convergence tolerance [K]
     * @param max_iterations Maximum Newton iterations
     * @return Number of iterations used
     *
     * This solves: T = T(H) using Newton-Raphson method since fl(T) is nonlinear
     */
    int updateTemperatureFromEnthalpy(float* temperature,
                                      float tolerance = 0.01f,
                                      int max_iterations = 10);

    /**
     * @brief Update liquid fraction field based on current temperature
     * @param temperature Device array of temperature values [K]
     */
    void updateLiquidFraction(const float* temperature);

    /**
     * @brief Add enthalpy change (from heat source or diffusion)
     * @param dH Device array of enthalpy change [J/m³]
     *
     * This directly modifies the enthalpy field: H_new = H_old + dH
     */
    void addEnthalpyChange(const float* dH);

    /**
     * @brief Get enthalpy field (device pointer)
     * @return Device pointer to enthalpy array [J/m³]
     */
    float* getEnthalpy() { return d_enthalpy; }

    /**
     * @brief Get enthalpy field (const device pointer)
     * @return Const device pointer to enthalpy array [J/m³]
     */
    const float* getEnthalpy() const { return d_enthalpy; }

    /**
     * @brief Get liquid fraction field (device pointer)
     * @return Device pointer to liquid fraction array (0-1)
     */
    float* getLiquidFraction() { return d_liquid_fraction; }

    /**
     * @brief Get liquid fraction field (const device pointer)
     * @return Const device pointer to liquid fraction array (0-1)
     */
    const float* getLiquidFraction() const { return d_liquid_fraction; }

    /**
     * @brief Store current liquid fraction for next step's rate calculation
     */
    void storePreviousLiquidFraction();

    /**
     * @brief Compute liquid fraction rate of change
     * @param dt Time step [s]
     */
    void computeLiquidFractionRate(float dt);

    /**
     * @brief Get pointer to liquid fraction rate field (device memory)
     * @return Device pointer to dfl/dt array [1/s]
     */
    float* getLiquidFractionRate() { return d_dfl_dt_; }

    /**
     * @brief Get pointer to liquid fraction rate field (const device memory)
     * @return Const device pointer to dfl/dt array [1/s]
     */
    const float* getLiquidFractionRate() const { return d_dfl_dt_; }

    /**
     * @brief Get pointer to previous liquid fraction field (device memory)
     * @return Device pointer to previous fl array
     */
    float* getPreviousLiquidFraction() { return d_liquid_fraction_prev_; }

    /**
     * @brief Get pointer to previous liquid fraction field (const device memory)
     * @return Const device pointer to previous fl array
     */
    const float* getPreviousLiquidFraction() const { return d_liquid_fraction_prev_; }

    /**
     * @brief Copy enthalpy to host
     * @param host_enthalpy Host array (must be pre-allocated)
     */
    void copyEnthalpyToHost(float* host_enthalpy) const;

    /**
     * @brief Copy liquid fraction to host
     * @param host_fl Host array (must be pre-allocated)
     */
    void copyLiquidFractionToHost(float* host_fl) const;

    /**
     * @brief Compute total energy in the system
     * @return Total energy [J]
     */
    float computeTotalEnergy() const;

    /**
     * @brief Get domain dimensions
     */
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

private:
    // Domain dimensions
    int nx_, ny_, nz_;
    int num_cells_;

    // Material properties
    MaterialProperties material_;

    // Device memory for phase change fields
    float* d_enthalpy;          ///< Total enthalpy field H = ρ·cp·T + fl·ρ·L [J/m³]
    float* d_liquid_fraction;   ///< Liquid fraction field (0-1)
    float* d_liquid_fraction_prev_;  ///< Previous time step liquid fraction
    float* d_dfl_dt_;                ///< Liquid fraction rate of change [1/s]

    // Utility functions
    void allocateMemory();
    void freeMemory();
};

// CUDA kernels for phase change

/**
 * @brief CUDA kernel to compute enthalpy from temperature
 *
 * H = ρ·cp·T + fl(T)·ρ·L_fusion
 *
 * @param temperature Input temperature field [K]
 * @param enthalpy Output enthalpy field [J/m³]
 * @param liquid_fraction Output liquid fraction field (0-1)
 * @param num_cells Total number of cells
 */
__global__ void computeEnthalpyFromTemperatureKernel(
    const float* temperature,
    float* enthalpy,
    float* liquid_fraction,
    int num_cells);

/**
 * @brief CUDA kernel to solve for temperature from enthalpy (Newton-Raphson)
 *
 * Given H, solve for T using:
 * f(T) = ρ·cp·T + fl(T)·ρ·L_fusion - H = 0
 *
 * Newton iteration: T_new = T_old - f(T_old)/f'(T_old)
 *
 * where f'(T) = ρ·cp + dfl/dT·ρ·L_fusion
 *       dfl/dT = 1/(T_liquidus - T_solidus) in mushy zone
 *
 * @param enthalpy Input enthalpy field [J/m³]
 * @param temperature Input/output temperature field [K]
 * @param liquid_fraction Output liquid fraction field (0-1)
 * @param converged Output convergence flag per cell
 * @param tolerance Convergence tolerance [K]
 * @param max_iterations Maximum iterations
 * @param num_cells Total number of cells
 */
__global__ void solveTemperatureFromEnthalpyKernel(
    const float* enthalpy,
    float* temperature,
    float* liquid_fraction,
    int* converged,
    float tolerance,
    int max_iterations,
    int num_cells);

/**
 * @brief CUDA kernel to update liquid fraction from temperature
 *
 * fl(T) = 0                           if T < T_solidus
 * fl(T) = (T - T_solidus)/ΔT_melt     if T_solidus ≤ T ≤ T_liquidus
 * fl(T) = 1                           if T > T_liquidus
 *
 * @param temperature Input temperature field [K]
 * @param liquid_fraction Output liquid fraction field (0-1)
 * @param num_cells Total number of cells
 */
__global__ void updateLiquidFractionKernel(
    const float* temperature,
    float* liquid_fraction,
    int num_cells);

/**
 * @brief CUDA kernel to add enthalpy change
 *
 * H_new = H_old + dH
 *
 * @param enthalpy Input/output enthalpy field [J/m³]
 * @param dH Enthalpy change [J/m³]
 * @param num_cells Total number of cells
 */
__global__ void addEnthalpyChangeKernel(
    float* enthalpy,
    const float* dH,
    int num_cells);

/**
 * @brief CUDA kernel to compute liquid fraction rate of change
 *
 * dfl_dt = (fl_curr - fl_prev) / dt
 *
 * @param fl_curr Current liquid fraction field
 * @param fl_prev Previous time step liquid fraction field
 * @param dfl_dt Output rate of change field [1/s]
 * @param dt Time step [s]
 * @param num_cells Total number of cells
 */
__global__ void computeLiquidFractionRateKernel(
    const float* fl_curr,
    const float* fl_prev,
    float* dfl_dt,
    float dt,
    int num_cells);

/**
 * @brief CUDA kernel to store current liquid fraction for next step
 *
 * @param fl_curr Current liquid fraction field
 * @param fl_prev Output previous liquid fraction field
 * @param num_cells Total number of cells
 */
__global__ void storeLiquidFractionKernel(
    const float* fl_curr,
    float* fl_prev,
    int num_cells);

/**
 * @brief CUDA kernel to compute total energy (reduction helper)
 *
 * @param enthalpy Enthalpy field [J/m³]
 * @param partial_sums Output array of partial sums per block
 * @param num_cells Total number of cells
 * @param cell_volume Volume of each cell [m³]
 */
__global__ void computeTotalEnergyKernel(
    const float* enthalpy,
    float* partial_sums,
    int num_cells,
    float cell_volume);

} // namespace physics
} // namespace lbm
