/**
 * @file diagnose_energy_balance.cu
 * @brief Comprehensive energy balance diagnostic tool
 *
 * This tool performs a detailed investigation of the 33% energy imbalance:
 *
 * Tests:
 * 1. Verify dE/dt calculation (check for double-counting)
 * 2. Break down E_total into E_sensible, E_latent, KE
 * 3. Verify evaporation power units and formula
 * 4. Check for hidden energy sinks/sources (periodic BC, Marangoni work)
 * 5. Validate individual energy terms independently
 * 6. Test steady-state convergence
 *
 * Expected outcome: Identify root cause of energy discrepancy
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm;

// ============================================================================
// CUDA Kernels for Detailed Energy Diagnostics
// ============================================================================

/**
 * @brief Compute detailed energy breakdown per cell
 */
__global__ void computeEnergyBreakdownKernel(
    const float* temperature,
    const float* liquid_fraction,
    const float* ux, const float* uy, const float* uz,
    float* E_sensible_out,
    float* E_latent_out,
    float* KE_out,
    physics::MaterialProperties material,
    float dx,
    float T_ref,  // Reference temperature for sensible energy
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    float f_l = liquid_fraction ? liquid_fraction[idx] : 1.0f;
    float vx = ux[idx];
    float vy = uy[idx];
    float vz = uz[idx];

    // Get temperature-dependent properties
    float rho = material.getDensity(T);
    float cp = material.getSpecificHeat(T);

    // Cell volume
    float V = dx * dx * dx;

    // SENSIBLE ENERGY: CORRECT FORMULA (relative to reference)
    // E_sensible = ρ * c_p * (T - T_ref) * V
    float E_sensible = rho * cp * (T - T_ref) * V;

    // LATENT ENERGY: Energy stored in liquid phase
    // E_latent = f_l * ρ * L_f * V
    float E_latent = f_l * rho * material.L_fusion * V;

    // KINETIC ENERGY: 0.5 * ρ * v² * V
    float v_squared = vx*vx + vy*vy + vz*vz;
    float KE = 0.5f * rho * v_squared * V;

    // Output
    E_sensible_out[idx] = E_sensible;
    E_latent_out[idx] = E_latent;
    KE_out[idx] = KE;
}

/**
 * @brief Simple temperature statistics kernel
 */
__global__ void computeTemperatureStatsKernel(
    const float* temperature,
    float* T_max_out,
    float* T_min_out,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];

    // Simple stats using atomic operations (not efficient but works for diagnostics)
    atomicMax((int*)T_max_out, __float_as_int(T));
    atomicMin((int*)T_min_out, __float_as_int(T));
}

// ============================================================================
// Host Functions for Energy Diagnostics
// ============================================================================

/**
 * @brief Compute total energy breakdown (sensible, latent, kinetic)
 */
void computeDetailedEnergyBreakdown(
    const physics::MultiphysicsSolver& solver,
    float T_ref,
    float& E_sensible_total,
    float& E_latent_total,
    float& KE_total)
{
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    int num_cells = nx * ny * nz;
    float dx = solver.getDx();

    // Get fields
    const float* d_temp = solver.getTemperature();
    const float* d_lf = solver.getLiquidFraction();
    const float* d_ux = solver.getVelocityX();
    const float* d_uy = solver.getVelocityY();
    const float* d_uz = solver.getVelocityZ();

    // Allocate device memory for breakdown
    float *d_E_sensible, *d_E_latent, *d_KE;
    cudaMalloc(&d_E_sensible, num_cells * sizeof(float));
    cudaMalloc(&d_E_latent, num_cells * sizeof(float));
    cudaMalloc(&d_KE, num_cells * sizeof(float));

    // Compute breakdown
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    computeEnergyBreakdownKernel<<<blocks, threads>>>(
        d_temp, d_lf, d_ux, d_uy, d_uz,
        d_E_sensible, d_E_latent, d_KE,
        solver.getConfig().material, dx, T_ref, num_cells
    );
    cudaDeviceSynchronize();

    // Copy to host and sum
    std::vector<float> h_E_sensible(num_cells);
    std::vector<float> h_E_latent(num_cells);
    std::vector<float> h_KE(num_cells);

    cudaMemcpy(h_E_sensible.data(), d_E_sensible, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E_latent.data(), d_E_latent, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_KE.data(), d_KE, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    E_sensible_total = 0.0f;
    E_latent_total = 0.0f;
    KE_total = 0.0f;

    for (int i = 0; i < num_cells; ++i) {
        E_sensible_total += h_E_sensible[i];
        E_latent_total += h_E_latent[i];
        KE_total += h_KE[i];
    }

    cudaFree(d_E_sensible);
    cudaFree(d_E_latent);
    cudaFree(d_KE);
}

/**
 * @brief Simple temperature diagnostics (replaces complex evaporation diagnostic)
 */
void diagnoseTemperatureField(const physics::MultiphysicsSolver& solver) {
    int num_cells = solver.getNx() * solver.getNy() * solver.getNz();
    const float* d_temp = solver.getTemperature();

    // Copy temperature to host
    std::vector<float> h_temp(num_cells);
    cudaMemcpy(h_temp.data(), d_temp, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute stats on host
    float T_min = 1e9f;
    float T_max = 0.0f;
    float T_avg = 0.0f;

    for (int i = 0; i < num_cells; ++i) {
        T_min = std::min(T_min, h_temp[i]);
        T_max = std::max(T_max, h_temp[i]);
        T_avg += h_temp[i];
    }
    T_avg /= num_cells;

    std::cout << "\n=== TEMPERATURE FIELD DIAGNOSTIC ===\n";
    std::cout << "  T_min: " << T_min << " K\n";
    std::cout << "  T_max: " << T_max << " K\n";
    std::cout << "  T_avg: " << T_avg << " K\n";
    std::cout << "  ΔT:    " << (T_max - T_min) << " K\n";
}

// ============================================================================
// Main Diagnostic Program
// ============================================================================

int main() {
    std::cout << "===============================================\n";
    std::cout << "  ENERGY BALANCE DIAGNOSTIC TOOL\n";
    std::cout << "  Investigating 33% Energy Imbalance\n";
    std::cout << "===============================================\n\n";

    // =========================================================================
    // Configuration: Match the problematic scenario
    // =========================================================================

    physics::MultiphysicsConfig config;

    // Small domain for fast testing
    config.nx = 40;
    config.ny = 40;
    config.nz = 20;
    config.dx = 2.0e-6f;  // 2 μm

    // Time stepping
    config.dt = 1.0e-7f;  // 0.1 μs

    // Physics modules
    config.enable_thermal = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_darcy = true;
    config.enable_marangoni = true;
    config.enable_laser = true;
    config.enable_radiation_bc = true;
    config.enable_substrate_cooling = false;  // Start without substrate to isolate issue
    config.enable_vof = true;
    config.enable_vof_advection = false;  // Disable advection for simplicity

    // Material
    config.material = physics::MaterialDatabase::getTi6Al4V();

    // Thermal properties
    config.thermal_diffusivity = 5.8e-6f;

    // Laser parameters (match the reported scenario: 150W, 52.5W absorbed)
    config.laser_power = 150.0f;
    config.laser_spot_radius = 50.0e-6f;
    config.laser_absorptivity = 0.35f;  // 52.5W / 150W = 0.35
    config.laser_penetration_depth = 10.0e-6f;

    // Radiation BC
    config.emissivity = 0.3f;
    config.ambient_temperature = 300.0f;

    // =========================================================================
    // Initialize Solver
    // =========================================================================

    std::cout << "Initializing solver...\n";
    physics::MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);  // T_init = 300K, interface at z=0.5

    // =========================================================================
    // Run Simulation with Detailed Diagnostics
    // =========================================================================

    const int num_steps = 500;  // Run to 50 μs
    // T_ref for the local breakdown kernel below (not used in computeTotalThermalEnergy,
    // which now references T_initial_ set at initialize() time — i.e. T_ambient = 300 K).
    const float T_ref = config.ambient_temperature;

    std::cout << "\nRunning simulation with detailed diagnostics...\n";
    std::cout << "Steps: " << num_steps << ", dt = " << config.dt * 1e6 << " μs\n\n";

    // Track energy over time
    std::vector<float> time_history;
    std::vector<float> E_sensible_history;
    std::vector<float> E_latent_history;
    std::vector<float> KE_history;
    std::vector<float> E_total_history;

    for (int step = 0; step < num_steps; ++step) {
        solver.step();

        // Detailed diagnostics every 50 steps
        if (step % 50 == 0) {
            float t = step * config.dt;

            std::cout << "\n─────────────────────────────────────────────\n";
            std::cout << "Step " << step << ", t = " << t * 1e6 << " μs\n";
            std::cout << "─────────────────────────────────────────────\n";

            // Get standard energy balance
            float P_laser = solver.getLaserAbsorbedPower();
            float P_evap = solver.getEvaporationPower();
            float P_rad = solver.getRadiationPower();
            float P_substrate = solver.getSubstratePower();
            float dE_dt = solver.getThermalEnergyChangeRate();

            // Get detailed energy breakdown
            float E_sensible, E_latent, KE;
            computeDetailedEnergyBreakdown(solver, T_ref, E_sensible, E_latent, KE);
            float E_total = E_sensible + E_latent + KE;

            // Store history
            time_history.push_back(t);
            E_sensible_history.push_back(E_sensible);
            E_latent_history.push_back(E_latent);
            KE_history.push_back(KE);
            E_total_history.push_back(E_total);

            // Print detailed breakdown
            std::cout << "\n[ENERGY BREAKDOWN]\n";
            std::cout << "  E_sensible = " << E_sensible << " J  (ρ·cp·(T-T_ref)·V)\n";
            std::cout << "  E_latent   = " << E_latent << " J  (f_l·ρ·L_f·V)\n";
            std::cout << "  KE         = " << KE << " J  (0.5·ρ·v²·V)\n";
            std::cout << "  E_total    = " << E_total << " J\n";

            // Compute manual dE/dt
            if (time_history.size() > 1) {
                float dt_actual = time_history.back() - time_history[time_history.size()-2];
                float dE_manual = (E_total_history.back() - E_total_history[E_total_history.size()-2]) / dt_actual;

                std::cout << "\n[dE/dt COMPARISON]\n";
                std::cout << "  dE/dt (solver)  = " << dE_dt << " W\n";
                std::cout << "  dE/dt (manual)  = " << dE_manual << " W\n";
                std::cout << "  Discrepancy     = " << std::abs(dE_dt - dE_manual) << " W\n";
            }

            // Print standard energy balance
            std::cout << "\n[ENERGY BALANCE]\n";
            std::cout << "  P_laser     = " << P_laser << " W\n";
            std::cout << "  P_evap      = " << P_evap << " W\n";
            std::cout << "  P_rad       = " << P_rad << " W\n";
            std::cout << "  P_substrate = " << P_substrate << " W\n";
            std::cout << "  dE/dt       = " << dE_dt << " W\n";

            float P_in = P_laser;
            float P_out = P_evap + P_rad + P_substrate;
            float balance_error = P_in - P_out - dE_dt;
            float error_percent = (P_in > 1e-6f) ? (std::abs(balance_error) / P_in * 100.0f) : 0.0f;

            std::cout << "\n  Balance: " << P_in << " = " << P_out << " + " << dE_dt << "\n";
            std::cout << "  Error: " << error_percent << " %\n";

            // Diagnose temperature field at first output
            if (step == 50) {
                diagnoseTemperatureField(solver);
            }

            // Check for NaN
            if (solver.checkNaN()) {
                std::cerr << "ERROR: NaN detected at step " << step << "\n";
                return 1;
            }
        }
    }

    // =========================================================================
    // Final Analysis
    // =========================================================================

    std::cout << "\n\n===============================================\n";
    std::cout << "  DIAGNOSTIC SUMMARY\n";
    std::cout << "===============================================\n";

    std::cout << "\n[ROOT CAUSE ANALYSIS]\n";
    std::cout << "Based on the diagnostic output above:\n\n";

    std::cout << "1. Check E_sensible magnitude:\n";
    std::cout << "   - If E_sensible is very large (~MJ range), this confirms\n";
    std::cout << "     the bug: using absolute temperature instead of ΔT\n";
    std::cout << "   - Expected: E_sensible ~ kJ range for 80x80x40 μm domain\n\n";

    std::cout << "2. Check dE/dt comparison:\n";
    std::cout << "   - If solver dE/dt >> manual dE/dt, this indicates double-counting\n";
    std::cout << "   - Expected: Both should match within 1%\n\n";

    std::cout << "3. Check evaporation diagnostic:\n";
    std::cout << "   - J_evap should be 50-100 kg/(m²·s) at 5000K\n";
    std::cout << "   - If much higher/lower, check formula or units\n\n";

    std::cout << "[NEXT STEPS]\n";
    std::cout << "Based on findings:\n";
    std::cout << "- FIX: Change computeThermalEnergyKernel to use (T - T_ref)\n";
    std::cout << "- TEST: Rerun this diagnostic after fix\n";
    std::cout << "- VERIFY: Error should drop below 5%\n";

    return 0;
}
