/**
 * @file energy_balance.h
 * @brief Comprehensive energy balance tracking for multiphysics simulations
 *
 * This module provides real-time energy conservation diagnostics including:
 * - Thermal, kinetic, and latent energy tracking
 * - Power term monitoring (laser, evaporation, radiation, substrate)
 * - Term-by-term energy balance validation
 * - Time series data collection for post-processing
 *
 * Physical Energy Balance Equation:
 * dE/dt = P_laser - P_evap - P_rad - P_substrate - P_convection
 *
 * Where:
 * - E_total = E_thermal + E_kinetic + E_latent
 * - E_thermal = ∫ ρ c_p T dV (sensible heat)
 * - E_kinetic = ∫ 0.5 ρ |u|² dV (fluid motion)
 * - E_latent = ∫ ρ L_f f_liquid dV (phase change)
 *
 * Week 3 P1: Essential for validating energy conservation in LPBF simulations
 */

#ifndef ENERGY_BALANCE_H
#define ENERGY_BALANCE_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

namespace lbm {
namespace diagnostics {

/**
 * @brief Energy balance snapshot at a single timestep
 *
 * This structure holds all energy state variables and power terms
 * needed to verify energy conservation.
 */
struct EnergyBalance {
    // ========================================================================
    // State energies [J]
    // ========================================================================
    double E_thermal;        ///< Thermal energy: ∫ ρ c_p T dV
    double E_kinetic;        ///< Kinetic energy: ∫ 0.5 ρ |u|² dV
    double E_latent;         ///< Latent heat: ∫ ρ L_f f_liquid dV
    double E_total;          ///< Total energy: E_thermal + E_kinetic + E_latent

    // ========================================================================
    // Power terms [W]
    // ========================================================================
    double P_laser;          ///< Laser absorbed power (input)
    double P_evaporation;    ///< Evaporative cooling (output)
    double P_radiation;      ///< Radiation loss (output)
    double P_substrate;      ///< Substrate cooling (output)
    double P_convection;     ///< Fluid convection through boundaries (typically small)

    // ========================================================================
    // Conservation check
    // ========================================================================
    double dE_dt_computed;   ///< dE/dt from state change: (E_total - E_total_prev) / dt
    double dE_dt_balance;    ///< dE/dt from power balance: P_in - P_out
    double error_absolute;   ///< Absolute error: |dE_dt_computed - dE_dt_balance| [W]
    double error_percent;    ///< Relative error: 100 × error / |dE_dt_balance| [%]

    // ========================================================================
    // Timestep info
    // ========================================================================
    double time;             ///< Physical time [s]
    int step;                ///< Time step number

    /**
     * @brief Initialize to zero
     */
    __host__ __device__ void reset() {
        E_thermal = 0.0;
        E_kinetic = 0.0;
        E_latent = 0.0;
        E_total = 0.0;

        P_laser = 0.0;
        P_evaporation = 0.0;
        P_radiation = 0.0;
        P_substrate = 0.0;
        P_convection = 0.0;

        dE_dt_computed = 0.0;
        dE_dt_balance = 0.0;
        error_absolute = 0.0;
        error_percent = 0.0;

        time = 0.0;
        step = 0;
    }

    /**
     * @brief Compute total energy from components
     */
    void updateTotal() {
        E_total = E_thermal + E_kinetic + E_latent;
    }

    /**
     * @brief Compute energy balance error
     * @param E_total_prev Previous timestep total energy [J]
     * @param dt Time step [s]
     */
    void computeError(double E_total_prev, double dt) {
        // Computed rate from state change
        if (dt > 0.0) {
            dE_dt_computed = (E_total - E_total_prev) / dt;
        } else {
            dE_dt_computed = 0.0;
        }

        // Expected rate from power balance
        dE_dt_balance = P_laser - P_evaporation - P_radiation - P_substrate - P_convection;

        // Error metrics
        error_absolute = std::abs(dE_dt_computed - dE_dt_balance);

        // Relative error (avoid division by zero)
        if (std::abs(dE_dt_balance) > 1e-10) {
            error_percent = 100.0 * error_absolute / std::abs(dE_dt_balance);
        } else if (error_absolute > 1e-6) {
            error_percent = 100.0;  // Large absolute error with near-zero balance
        } else {
            error_percent = 0.0;    // Both negligible
        }
    }

    /**
     * @brief Print summary to stdout
     */
    void print() const {
        printf("[ENERGY] t=%.2e s, step=%d\n", time, step);
        printf("  State energies [J]:\n");
        printf("    E_thermal  = %12.4e  (sensible heat)\n", E_thermal);
        printf("    E_kinetic  = %12.4e  (fluid motion)\n", E_kinetic);
        printf("    E_latent   = %12.4e  (phase change)\n", E_latent);
        printf("    E_total    = %12.4e\n", E_total);
        printf("  Power terms [W]:\n");
        printf("    P_laser    = %10.2f  (input)\n", P_laser);
        printf("    P_evap     = %10.2f  (output)\n", P_evaporation);
        printf("    P_rad      = %10.2f  (output)\n", P_radiation);
        printf("    P_sub      = %10.2f  (output)\n", P_substrate);
        printf("  Balance:\n");
        printf("    dE/dt (computed) = %10.2f W\n", dE_dt_computed);
        printf("    dE/dt (balance)  = %10.2f W\n", dE_dt_balance);
        printf("    Error            = %10.2f%% ", error_percent);

        if (error_percent < 5.0) {
            printf("✓ PASS\n");
        } else if (error_percent < 10.0) {
            printf("⚠ WARNING\n");
        } else {
            printf("✗ FAIL\n");
        }
    }
};

/**
 * @brief Energy balance tracker with time series storage
 *
 * This class manages energy balance computation and stores historical
 * data for post-processing and visualization.
 */
class EnergyBalanceTracker {
public:
    /**
     * @brief Constructor
     */
    EnergyBalanceTracker() : current_(), E_total_prev_(0.0) {
        current_.reset();
    }

    /**
     * @brief Get current energy balance snapshot
     */
    const EnergyBalance& getCurrent() const {
        return current_;
    }

    /**
     * @brief Update current snapshot (without storing in history)
     * @param balance New energy balance data
     * @param dt Time step [s]
     */
    void update(const EnergyBalance& balance, double dt) {
        current_ = balance;
        current_.computeError(E_total_prev_, dt);
        E_total_prev_ = current_.E_total;
    }

    /**
     * @brief Record current snapshot to history
     */
    void record() {
        history_.push_back(current_);
    }

    /**
     * @brief Get full history
     */
    const std::vector<EnergyBalance>& getHistory() const {
        return history_;
    }

    /**
     * @brief Clear history (keep current snapshot)
     */
    void clearHistory() {
        history_.clear();
    }

    /**
     * @brief Write history to ASCII file
     * @param filename Output file path
     */
    void writeToFile(const std::string& filename) const {
        std::ofstream f(filename);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open " + filename);
        }

        // Header
        f << "# Energy Balance Time Series\n";
        f << "# Columns:\n";
        f << "#  1: Time [s]\n";
        f << "#  2: Step [-]\n";
        f << "#  3: E_thermal [J]\n";
        f << "#  4: E_kinetic [J]\n";
        f << "#  5: E_latent [J]\n";
        f << "#  6: E_total [J]\n";
        f << "#  7: P_laser [W]\n";
        f << "#  8: P_evap [W]\n";
        f << "#  9: P_rad [W]\n";
        f << "# 10: P_substrate [W]\n";
        f << "# 11: dE/dt_computed [W]\n";
        f << "# 12: dE/dt_balance [W]\n";
        f << "# 13: Error [%]\n";
        f << "#\n";

        // Data
        f.precision(6);
        f << std::scientific;

        for (const auto& e : history_) {
            f << e.time << " "
              << e.step << " "
              << e.E_thermal << " "
              << e.E_kinetic << " "
              << e.E_latent << " "
              << e.E_total << " "
              << e.P_laser << " "
              << e.P_evaporation << " "
              << e.P_radiation << " "
              << e.P_substrate << " "
              << e.dE_dt_computed << " "
              << e.dE_dt_balance << " "
              << e.error_percent << "\n";
        }

        f.close();
    }

private:
    EnergyBalance current_;           ///< Current timestep data
    std::vector<EnergyBalance> history_;  ///< Time series
    double E_total_prev_;             ///< Previous E_total for dE/dt
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

/**
 * @brief Compute total thermal energy
 * @param T Temperature field [K]
 * @param f_liquid Liquid fraction field [0-1]
 * @param rho Density field [kg/m³] (or constant for incompressible)
 * @param cp_solid Solid specific heat [J/(kg·K)]
 * @param cp_liquid Liquid specific heat [J/(kg·K)]
 * @param dx Lattice spacing [m]
 * @param nx, ny, nz Grid dimensions
 * @param d_result Output: thermal energy [J] (device memory, single double)
 */
void computeThermalEnergy(
    const float* T,
    const float* f_liquid,
    float rho,
    float cp_solid,
    float cp_liquid,
    float dx,
    int nx, int ny, int nz,
    double* d_result);

/**
 * @brief Compute total kinetic energy
 * @param ux, uy, uz Velocity components [m/s]
 * @param rho Density [kg/m³]
 * @param dx Lattice spacing [m]
 * @param nx, ny, nz Grid dimensions
 * @param d_result Output: kinetic energy [J]
 */
void computeKineticEnergy(
    const float* ux,
    const float* uy,
    const float* uz,
    float rho,
    float dx,
    int nx, int ny, int nz,
    double* d_result);

/**
 * @brief Compute total latent heat energy
 * @param f_liquid Liquid fraction field [0-1]
 * @param rho Density [kg/m³]
 * @param L_fusion Latent heat of fusion [J/kg]
 * @param dx Lattice spacing [m]
 * @param nx, ny, nz Grid dimensions
 * @param d_result Output: latent energy [J]
 */
void computeLatentEnergy(
    const float* f_liquid,
    float rho,
    float L_fusion,
    float dx,
    int nx, int ny, int nz,
    double* d_result);

} // namespace diagnostics
} // namespace lbm

#endif // ENERGY_BALANCE_H
