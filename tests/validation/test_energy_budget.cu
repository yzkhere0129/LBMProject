/**
 * @file test_energy_budget.cu
 * @brief Rigorous energy conservation test to identify energy loss mechanisms
 *
 * This test implements a strict energy budget analysis:
 *
 * EXPECTED BEHAVIOR:
 * - Laser input: P * eta = 200 * 0.35 = 70 W
 * - Over 50 μs: E_total = 70 * 50e-6 = 3.5 mJ
 * - With ADIABATIC boundaries: E_stored should equal E_input
 *
 * ENERGY BALANCE:
 *   E_input = ∫∫∫ Q(x,y,z) * dt dV  [J] - Total energy deposited by laser
 *   E_stored = ∫∫∫ ρ * cp * (T - T0) dV  [J] - Total sensible energy in domain
 *   E_balance = E_input - E_stored  [J] - Energy discrepancy
 *
 * If E_balance > 5% of E_input, there is a BUG.
 *
 * DIAGNOSTIC OUTPUTS:
 * 1. Total energy input vs time
 * 2. Total energy stored vs time
 * 3. Energy balance error (%)
 * 4. Peak temperature vs time
 * 5. Spatial distribution of energy storage
 *
 * POSSIBLE LOSS MECHANISMS:
 * - Numerical diffusion in LBM
 * - Boundary leakage (if BC not truly adiabatic)
 * - Source term error (incorrect Chapman-Enskog correction)
 * - Temperature clamping (if max_T is being hit)
 * - Roundoff errors in accumulation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "physics/laser_source.h"
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            FAIL() << "CUDA error: " << cudaGetErrorString(err); \
        } \
    } while(0)

/**
 * @brief Helper function to compute total thermal energy in domain
 *
 * E = ∫∫∫ ρ * cp * (T - T0) dV
 *
 * This is the sensible energy stored above the initial temperature.
 */
float computeTotalEnergy(const std::vector<float>& temperature,
                         const MaterialProperties& material,
                         float T0, float dx, int nx, int ny, int nz) {
    float E_total = 0.0f;
    float dV = dx * dx * dx;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = k * nx * ny + j * nx + i;
                float T = temperature[idx];
                float rho = material.getDensity(T);
                float cp = material.getSpecificHeat(T);

                // Sensible energy above T0
                float dT = T - T0;
                E_total += rho * cp * dT * dV;
            }
        }
    }

    return E_total;
}

/**
 * @brief Helper function to compute total heat source power
 *
 * P = ∫∫∫ Q(x,y,z) dV  [W]
 */
float computeTotalPower(const std::vector<float>& heat_source,
                        float dx, int nx, int ny, int nz) {
    float P_total = 0.0f;
    float dV = dx * dx * dx;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = k * nx * ny + j * nx + i;
                P_total += heat_source[idx] * dV;
            }
        }
    }

    return P_total;
}

/**
 * @brief Test 1: Energy budget with adiabatic boundaries
 *
 * This is the PRIMARY diagnostic test. With adiabatic boundaries,
 * ALL laser energy should be stored in the domain.
 */
TEST(EnergyBudgetTest, AdiabaticEnergyConservation) {
    std::cout << "\n=== ENERGY BUDGET TEST: Adiabatic Boundaries ===\n";
    std::cout << "Expected: E_input = E_stored (100% conservation)\n\n";

    // ========================================================================
    // SETUP: Small domain with fine resolution
    // ========================================================================
    const int nx = 50;
    const int ny = 50;
    const int nz = 50;
    const float dx = 2e-6f;  // 2 μm → 100 μm × 100 μm × 100 μm domain

    std::cout << "Domain setup:\n";
    std::cout << "  Grid: " << nx << " × " << ny << " × " << nz << "\n";
    std::cout << "  Resolution: " << dx * 1e6 << " μm\n";
    std::cout << "  Domain size: " << nx * dx * 1e6 << " × "
              << ny * dx * 1e6 << " × " << nz * dx * 1e6 << " μm³\n\n";

    // ========================================================================
    // LASER PARAMETERS (matching validation tests)
    // ========================================================================
    const float P = 200.0f;        // W
    const float w0 = 50e-6f;       // 50 μm spot radius
    const float eta = 0.35f;       // absorptivity
    const float delta = 10e-6f;    // 10 μm penetration depth

    const float P_absorbed = P * eta;  // 70 W expected

    std::cout << "Laser parameters:\n";
    std::cout << "  Power: " << P << " W\n";
    std::cout << "  Spot radius: " << w0 * 1e6 << " μm\n";
    std::cout << "  Absorptivity: " << eta << "\n";
    std::cout << "  Penetration depth: " << delta * 1e6 << " μm\n";
    std::cout << "  Expected absorbed power: " << P_absorbed << " W\n\n";

    LaserSource laser(P, w0, eta, delta);
    laser.setPosition(nx * dx / 2.0f, ny * dx / 2.0f, 0.0f);

    // ========================================================================
    // MATERIAL AND THERMAL SOLVER
    // ========================================================================
    MaterialProperties material = MaterialDatabase::getTi6Al4V();

    const float dt = 100e-9f;  // 100 ns timestep
    const float alpha = 5.8e-6f;  // m²/s (Ti6Al4V)
    const float T0 = 300.0f;  // K (initial temperature)

    std::cout << "Thermal parameters:\n";
    std::cout << "  Material: " << material.name << "\n";
    std::cout << "  Density: " << material.rho_solid << " kg/m³\n";
    std::cout << "  Specific heat: " << material.cp_solid << " J/(kg·K)\n";
    std::cout << "  Thermal diffusivity: " << alpha << " m²/s\n";
    std::cout << "  Time step: " << dt * 1e9 << " ns\n";
    std::cout << "  Initial temperature: " << T0 << " K\n\n";

    ThermalLBM thermal(nx, ny, nz, material, alpha, false, dt, dx);
    thermal.initialize(T0);

    // ========================================================================
    // COMPUTE HEAT SOURCE DISTRIBUTION
    // ========================================================================
    float* d_heat_source;
    CUDA_CHECK(cudaMalloc(&d_heat_source, nx * ny * nz * sizeof(float)));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
        d_heat_source, laser, dx, dx, dx, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify total power
    std::vector<float> h_heat_source(nx * ny * nz);
    CUDA_CHECK(cudaMemcpy(h_heat_source.data(), d_heat_source,
                          nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

    float P_integrated = computeTotalPower(h_heat_source, dx, nx, ny, nz);

    std::cout << "Heat source verification:\n";
    std::cout << "  Integrated power: " << P_integrated << " W\n";
    std::cout << "  Expected power: " << P_absorbed << " W\n";
    std::cout << "  Error: " << std::abs(P_integrated - P_absorbed) / P_absorbed * 100 << " %\n\n";

    EXPECT_NEAR(P_integrated, P_absorbed, 0.05 * P_absorbed)
        << "Heat source integration should match P * η within 5%";

    // ========================================================================
    // TIME INTEGRATION WITH ENERGY TRACKING
    // ========================================================================
    const float t_total = 50e-6f;  // 50 μs
    const int num_steps = static_cast<int>(t_total / dt);
    const int output_interval = 50;  // Output every 5 μs

    std::cout << "Time integration:\n";
    std::cout << "  Total time: " << t_total * 1e6 << " μs\n";
    std::cout << "  Time steps: " << num_steps << "\n";
    std::cout << "  Output interval: " << output_interval << " steps ("
              << output_interval * dt * 1e6 << " μs)\n\n";

    std::cout << "Time (μs) | Max T (K) | E_input (mJ) | E_stored (mJ) | Balance (%) | Error\n";
    std::cout << std::string(85, '-') << "\n";

    // Energy accumulators
    float E_input_cumulative = 0.0f;  // Total energy deposited

    // Store results for analysis
    std::vector<float> time_history;
    std::vector<float> E_input_history;
    std::vector<float> E_stored_history;
    std::vector<float> E_balance_history;
    std::vector<float> max_T_history;

    for (int step = 0; step <= num_steps; ++step) {
        float t = step * dt;

        // Add heat source (energy deposition)
        thermal.addHeatSource(d_heat_source, dt);

        // Accumulate input energy
        E_input_cumulative += P_integrated * dt;

        // LBM collision and streaming
        thermal.collisionBGK(nullptr, nullptr, nullptr);

        // Apply ADIABATIC boundary conditions (no energy loss)
        thermal.applyBoundaryConditions(2);  // 2 = adiabatic

        thermal.streaming();

        // Compute temperature
        thermal.computeTemperature();

        // Diagnostics
        if (step % output_interval == 0) {
            // Get temperature field
            std::vector<float> temp(nx * ny * nz);
            thermal.copyTemperatureToHost(temp.data());

            // Compute stored energy
            float E_stored = computeTotalEnergy(temp, material, T0, dx, nx, ny, nz);

            // Energy balance
            float E_balance = E_input_cumulative - E_stored;
            float balance_percent = (E_input_cumulative > 0)
                ? (E_balance / E_input_cumulative) * 100.0f : 0.0f;

            // Peak temperature
            float max_T = *std::max_element(temp.begin(), temp.end());

            // Store history
            time_history.push_back(t * 1e6);  // μs
            E_input_history.push_back(E_input_cumulative * 1e3);  // mJ
            E_stored_history.push_back(E_stored * 1e3);  // mJ
            E_balance_history.push_back(balance_percent);
            max_T_history.push_back(max_T);

            // Print status
            std::string status = (std::abs(balance_percent) < 5.0f) ? "OK" : "FAIL";
            std::cout << std::setw(9) << std::fixed << std::setprecision(2) << t * 1e6
                      << " | " << std::setw(9) << std::setprecision(1) << max_T
                      << " | " << std::setw(12) << std::setprecision(4) << E_input_cumulative * 1e3
                      << " | " << std::setw(13) << std::setprecision(4) << E_stored * 1e3
                      << " | " << std::setw(11) << std::setprecision(2) << balance_percent
                      << " | " << status << "\n";
        }
    }

    std::cout << std::string(85, '-') << "\n\n";

    // ========================================================================
    // FINAL ENERGY BALANCE ANALYSIS
    // ========================================================================
    float E_input_final = E_input_cumulative;

    std::vector<float> temp_final(nx * ny * nz);
    thermal.copyTemperatureToHost(temp_final.data());
    float E_stored_final = computeTotalEnergy(temp_final, material, T0, dx, nx, ny, nz);

    float E_balance_final = E_input_final - E_stored_final;
    float balance_percent_final = (E_balance_final / E_input_final) * 100.0f;

    std::cout << "FINAL ENERGY BALANCE:\n";
    std::cout << "  E_input (theoretical): " << P_absorbed * t_total * 1e3 << " mJ\n";
    std::cout << "  E_input (integrated):  " << E_input_final * 1e3 << " mJ\n";
    std::cout << "  E_stored:              " << E_stored_final * 1e3 << " mJ\n";
    std::cout << "  E_balance (lost):      " << E_balance_final * 1e3 << " mJ\n";
    std::cout << "  Balance error:         " << balance_percent_final << " %\n\n";

    // ========================================================================
    // SPATIAL ENERGY DISTRIBUTION ANALYSIS
    // ========================================================================
    std::cout << "SPATIAL ENERGY DISTRIBUTION:\n";

    // Compute energy per layer
    std::vector<float> energy_per_layer(nz, 0.0f);
    float dV = dx * dx * dx;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = k * nx * ny + j * nx + i;
                float T = temp_final[idx];
                float rho = material.getDensity(T);
                float cp = material.getSpecificHeat(T);
                energy_per_layer[k] += rho * cp * (T - T0) * dV;
            }
        }
    }

    std::cout << "  Layer (μm) | Energy (mJ) | Fraction (%)\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    for (int k = 0; k < std::min(nz, 20); ++k) {
        float z = k * dx;
        float fraction = (E_stored_final > 0)
            ? (energy_per_layer[k] / E_stored_final) * 100.0f : 0.0f;

        if (k % 5 == 0) {  // Print every 5 layers
            std::cout << "  " << std::setw(10) << std::fixed << std::setprecision(1) << z * 1e6
                      << " | " << std::setw(11) << std::setprecision(4) << energy_per_layer[k] * 1e3
                      << " | " << std::setw(11) << std::setprecision(2) << fraction << "\n";
        }
    }
    std::cout << "\n";

    // ========================================================================
    // WRITE DETAILED RESULTS TO FILE
    // ========================================================================
    std::ofstream outfile("/home/yzk/LBMProject/benchmark/ENERGY_BUDGET_ANALYSIS.md");

    outfile << "# Energy Budget Analysis Report\n\n";
    outfile << "## Test Configuration\n\n";
    outfile << "**Domain:** " << nx << " × " << ny << " × " << nz << " cells  \n";
    outfile << "**Resolution:** " << dx * 1e6 << " μm  \n";
    outfile << "**Physical size:** " << nx * dx * 1e6 << " × "
            << ny * dx * 1e6 << " × " << nz * dx * 1e6 << " μm³  \n\n";

    outfile << "**Laser parameters:**  \n";
    outfile << "- Power: " << P << " W  \n";
    outfile << "- Spot radius: " << w0 * 1e6 << " μm  \n";
    outfile << "- Absorptivity: " << eta << "  \n";
    outfile << "- Penetration depth: " << delta * 1e6 << " μm  \n";
    outfile << "- Expected absorbed power: " << P_absorbed << " W  \n\n";

    outfile << "**Material:** " << material.name << "  \n";
    outfile << "- Density: " << material.rho_solid << " kg/m³  \n";
    outfile << "- Specific heat: " << material.cp_solid << " J/(kg·K)  \n";
    outfile << "- Initial temperature: " << T0 << " K  \n\n";

    outfile << "**Time integration:**  \n";
    outfile << "- Time step: " << dt * 1e9 << " ns  \n";
    outfile << "- Total time: " << t_total * 1e6 << " μs  \n";
    outfile << "- Number of steps: " << num_steps << "  \n\n";

    outfile << "## Energy Balance Results\n\n";
    outfile << "| Parameter | Value |\n";
    outfile << "|-----------|-------|\n";
    outfile << "| E_input (theoretical) | " << P_absorbed * t_total * 1e3 << " mJ |\n";
    outfile << "| E_input (integrated) | " << E_input_final * 1e3 << " mJ |\n";
    outfile << "| E_stored | " << E_stored_final * 1e3 << " mJ |\n";
    outfile << "| E_balance (lost) | " << E_balance_final * 1e3 << " mJ |\n";
    outfile << "| Balance error | " << std::abs(balance_percent_final) << " % |\n\n";

    outfile << "### Verdict\n\n";
    if (std::abs(balance_percent_final) < 5.0f) {
        outfile << "**PASS**: Energy is conserved within 5% tolerance.  \n";
        outfile << "The LBM thermal solver is correctly conserving energy.\n\n";
    } else {
        outfile << "**FAIL**: Energy balance error exceeds 5% threshold.  \n";
        outfile << "**Energy is being LOST somewhere in the simulation!**\n\n";

        outfile << "### Possible Loss Mechanisms\n\n";
        outfile << "1. **Boundary leakage**: Adiabatic BC may not be truly adiabatic  \n";
        outfile << "2. **Source term error**: Chapman-Enskog correction may be incorrect  \n";
        outfile << "3. **Numerical diffusion**: LBM collision operator may be over-dissipative  \n";
        outfile << "4. **Temperature clamping**: Max temperature limit may be artificially capping T  \n";
        outfile << "5. **Accumulation error**: Roundoff in energy integration  \n\n";

        outfile << "### Debugging Steps\n\n";
        outfile << "1. Check `thermalStreamingKernel` for proper bounce-back implementation  \n";
        outfile << "2. Verify `addHeatSourceKernel` source term correction factor  \n";
        outfile << "3. Check `computeTemperatureKernel` for clamping logic  \n";
        outfile << "4. Verify omega_T value (should be < 1.9 for stability)  \n";
        outfile << "5. Compare with finite-difference solution for same problem  \n\n";
    }

    outfile << "## Time History\n\n";
    outfile << "| Time (μs) | E_input (mJ) | E_stored (mJ) | Balance (%) | Max T (K) |\n";
    outfile << "|-----------|--------------|---------------|-------------|----------|\n";

    for (size_t i = 0; i < time_history.size(); ++i) {
        outfile << "| " << std::fixed << std::setprecision(2) << time_history[i]
                << " | " << std::setprecision(4) << E_input_history[i]
                << " | " << std::setprecision(4) << E_stored_history[i]
                << " | " << std::setprecision(2) << E_balance_history[i]
                << " | " << std::setprecision(1) << max_T_history[i] << " |\n";
    }

    outfile << "\n## Spatial Energy Distribution\n\n";
    outfile << "Energy stored per layer (depth from surface):\n\n";
    outfile << "| Depth (μm) | Energy (mJ) | Fraction (%) |\n";
    outfile << "|------------|-------------|-------------|\n";

    for (int k = 0; k < nz; ++k) {
        float z = k * dx;
        float fraction = (E_stored_final > 0)
            ? (energy_per_layer[k] / E_stored_final) * 100.0f : 0.0f;

        outfile << "| " << std::fixed << std::setprecision(1) << z * 1e6
                << " | " << std::setprecision(4) << energy_per_layer[k] * 1e3
                << " | " << std::setprecision(2) << fraction << " |\n";
    }

    outfile << "\n## Recommendations\n\n";

    if (std::abs(balance_percent_final) >= 5.0f) {
        outfile << "Energy conservation is VIOLATED. Immediate action required:\n\n";
        outfile << "1. **Critical**: Identify and fix the energy loss mechanism  \n";
        outfile << "2. **Critical**: Verify boundary conditions are truly adiabatic  \n";
        outfile << "3. **Critical**: Check source term implementation  \n";
        outfile << "4. Re-run validation tests after fixes  \n\n";
    } else {
        outfile << "Energy conservation is VERIFIED within tolerance.\n\n";
        outfile << "The thermal solver is working correctly. Any temperature discrepancies\n";
        outfile << "with walberla must be due to:\n";
        outfile << "- Different laser source models  \n";
        outfile << "- Different boundary conditions  \n";
        outfile << "- Different material properties  \n";
        outfile << "- Different numerical diffusion characteristics  \n\n";
    }

    outfile.close();

    std::cout << "Detailed results written to: /home/yzk/LBMProject/benchmark/ENERGY_BUDGET_ANALYSIS.md\n\n";

    // ========================================================================
    // ASSERT FINAL VERDICT
    // ========================================================================
    EXPECT_LT(std::abs(balance_percent_final), 5.0f)
        << "Energy balance error must be < 5%. Found: " << balance_percent_final << " %\n"
        << "Energy is being LOST! Check ENERGY_BUDGET_ANALYSIS.md for details.";

    // Cleanup
    CUDA_CHECK(cudaFree(d_heat_source));
}

/**
 * @brief Test 2: Energy conservation check with shorter time scale
 *
 * Run for only 5 μs to check if energy loss is time-dependent
 */
TEST(EnergyBudgetTest, ShortTimeEnergyConservation) {
    std::cout << "\n=== SHORT TIME ENERGY CONSERVATION (5 μs) ===\n";

    // Same setup as Test 1, but shorter duration
    const int nx = 50, ny = 50, nz = 50;
    const float dx = 2e-6f;
    const float dt = 100e-9f;
    const float t_total = 5e-6f;  // 5 μs
    const float T0 = 300.0f;

    const float P = 200.0f;
    const float w0 = 50e-6f;
    const float eta = 0.35f;
    const float delta = 10e-6f;
    const float P_absorbed = P * eta;

    LaserSource laser(P, w0, eta, delta);
    laser.setPosition(nx * dx / 2.0f, ny * dx / 2.0f, 0.0f);

    MaterialProperties material = MaterialDatabase::getTi6Al4V();
    const float alpha = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false, dt, dx);
    thermal.initialize(T0);

    // Heat source
    float* d_heat_source;
    CUDA_CHECK(cudaMalloc(&d_heat_source, nx * ny * nz * sizeof(float)));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
        d_heat_source, laser, dx, dx, dx, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time integration
    const int num_steps = static_cast<int>(t_total / dt);
    float E_input_cumulative = 0.0f;

    for (int step = 0; step <= num_steps; ++step) {
        thermal.addHeatSource(d_heat_source, dt);
        E_input_cumulative += P_absorbed * dt;

        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.applyBoundaryConditions(2);  // Adiabatic
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Final energy balance
    std::vector<float> temp_final(nx * ny * nz);
    thermal.copyTemperatureToHost(temp_final.data());
    float E_stored_final = computeTotalEnergy(temp_final, material, T0, dx, nx, ny, nz);

    float balance_percent = ((E_input_cumulative - E_stored_final) / E_input_cumulative) * 100.0f;

    std::cout << "Results after " << t_total * 1e6 << " μs:\n";
    std::cout << "  E_input:  " << E_input_cumulative * 1e3 << " mJ\n";
    std::cout << "  E_stored: " << E_stored_final * 1e3 << " mJ\n";
    std::cout << "  Balance:  " << balance_percent << " %\n\n";

    EXPECT_LT(std::abs(balance_percent), 5.0f)
        << "Energy balance error at 5 μs should be < 5%";

    CUDA_CHECK(cudaFree(d_heat_source));
}

/**
 * @brief Test 3: Power input rate verification
 *
 * Verify that dE/dt = P_absorbed over short intervals
 */
TEST(EnergyBudgetTest, PowerInputRate) {
    std::cout << "\n=== POWER INPUT RATE VERIFICATION ===\n";

    const int nx = 40, ny = 40, nz = 40;
    const float dx = 2.5e-6f;
    const float dt = 100e-9f;
    const float T0 = 300.0f;

    const float P = 200.0f;
    const float w0 = 50e-6f;
    const float eta = 0.35f;
    const float delta = 10e-6f;
    const float P_absorbed = P * eta;

    LaserSource laser(P, w0, eta, delta);
    laser.setPosition(nx * dx / 2.0f, ny * dx / 2.0f, 0.0f);

    MaterialProperties material = MaterialDatabase::getTi6Al4V();
    const float alpha = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false, dt, dx);
    thermal.initialize(T0);

    float* d_heat_source;
    CUDA_CHECK(cudaMalloc(&d_heat_source, nx * ny * nz * sizeof(float)));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
        d_heat_source, laser, dx, dx, dx, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Measuring dE/dt over 1 μs intervals:\n";
    std::cout << "Interval | dE/dt (W) | Expected (W) | Error (%)\n";
    std::cout << std::string(60, '-') << "\n";

    const int interval_steps = 10;  // 1 μs intervals
    const int num_intervals = 10;

    float E_prev = 0.0f;

    for (int interval = 0; interval < num_intervals; ++interval) {
        // Advance simulation
        for (int step = 0; step < interval_steps; ++step) {
            thermal.addHeatSource(d_heat_source, dt);
            thermal.collisionBGK(nullptr, nullptr, nullptr);
            thermal.applyBoundaryConditions(2);
            thermal.streaming();
            thermal.computeTemperature();
        }

        // Measure energy
        std::vector<float> temp(nx * ny * nz);
        thermal.copyTemperatureToHost(temp.data());
        float E_current = computeTotalEnergy(temp, material, T0, dx, nx, ny, nz);

        float dE = E_current - E_prev;
        float dt_interval = interval_steps * dt;
        float power_measured = dE / dt_interval;
        float error_percent = std::abs(power_measured - P_absorbed) / P_absorbed * 100.0f;

        std::cout << std::setw(8) << interval + 1
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2) << power_measured
                  << " | " << std::setw(12) << P_absorbed
                  << " | " << std::setw(8) << std::setprecision(1) << error_percent << "\n";

        E_prev = E_current;

        // Each interval should have dE/dt ≈ P_absorbed
        EXPECT_NEAR(power_measured, P_absorbed, 0.1 * P_absorbed)
            << "Power input rate should match P_absorbed at interval " << interval + 1;
    }

    std::cout << "\n";
    CUDA_CHECK(cudaFree(d_heat_source));
}
