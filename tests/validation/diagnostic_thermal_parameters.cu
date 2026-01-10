/**
 * @file diagnostic_thermal_parameters.cu
 * @brief Diagnostic test to identify thermal parameter discrepancies
 *
 * This test performs a systematic comparison of thermal parameters between
 * our implementation and the walberla reference to identify why our peak
 * temperature (3,544.8 K) is much lower than walberla's (17,500 K).
 *
 * The test checks:
 * 1. Thermal diffusivity calculation
 * 2. Lattice Boltzmann parameters (tau, omega)
 * 3. Energy deposition from laser
 * 4. Temperature field evolution
 * 5. Boundary condition effects
 *
 * Author: Testing and Debugging Specialist
 * Date: 2025-12-22
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"
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
 * @brief Test 1: Verify thermal diffusivity calculation
 */
TEST(ThermalDiagnostics, ThermalDiffusivityCalculation) {
    std::cout << "\n========================================\n";
    std::cout << "TEST 1: THERMAL DIFFUSIVITY CALCULATION\n";
    std::cout << "========================================\n\n";

    // Ti6Al4V properties (should match walberla)
    float k = 6.7f;        // W/(m·K) - thermal conductivity
    float rho = 4420.0f;   // kg/m³ - density
    float cp = 610.0f;     // J/(kg·K) - specific heat

    float alpha_physical = k / (rho * cp);  // m²/s

    std::cout << "Material: Ti6Al4V\n";
    std::cout << "-----------------\n";
    std::cout << "Thermal conductivity k:  " << k << " W/(m·K)\n";
    std::cout << "Density rho:             " << rho << " kg/m³\n";
    std::cout << "Specific heat cp:        " << cp << " J/(kg·K)\n";
    std::cout << "\n";
    std::cout << "Calculated alpha:        " << alpha_physical << " m²/s\n";
    std::cout << "Expected alpha:          2.88e-06 m²/s (from test output)\n";
    std::cout << "\n";

    float expected_alpha = 2.88e-6f;
    float relative_error = std::abs(alpha_physical - expected_alpha) / expected_alpha;

    std::cout << "Relative error:          " << (relative_error * 100) << "%\n";
    std::cout << "\n";

    if (relative_error > 0.01f) {
        std::cout << "WARNING: Thermal diffusivity mismatch!\n";
        std::cout << "  This will affect temperature evolution.\n";
    } else {
        std::cout << "PASS: Thermal diffusivity is correct.\n";
    }

    EXPECT_NEAR(alpha_physical, expected_alpha, expected_alpha * 0.01f);
}

/**
 * @brief Test 2: Verify lattice Boltzmann parameters
 */
TEST(ThermalDiagnostics, LatticeBoltzmannParameters) {
    std::cout << "\n========================================\n";
    std::cout << "TEST 2: LATTICE BOLTZMANN PARAMETERS\n";
    std::cout << "========================================\n\n";

    // Grid parameters
    float dx = 3.75e-6f;   // m
    float dt = 5.0e-8f;    // s

    // Physical properties
    float alpha_physical = 2.88e-6f;  // m²/s

    // LBM parameters
    float alpha_lattice = alpha_physical * dt / (dx * dx);
    float tau = 0.5f + alpha_lattice;
    float omega = 1.0f / tau;

    std::cout << "Grid parameters:\n";
    std::cout << "----------------\n";
    std::cout << "dx = " << dx << " m\n";
    std::cout << "dt = " << dt << " s\n";
    std::cout << "\n";

    std::cout << "LBM parameters:\n";
    std::cout << "----------------\n";
    std::cout << "alpha_lattice:  " << alpha_lattice << " (dimensionless)\n";
    std::cout << "tau:            " << tau << "\n";
    std::cout << "omega:          " << omega << "\n";
    std::cout << "\n";

    std::cout << "Expected values (from test output):\n";
    std::cout << "------------------------------------\n";
    std::cout << "alpha_lattice:  0.01024\n";
    std::cout << "tau:            0.54096\n";
    std::cout << "omega:          1.84857\n";
    std::cout << "\n";

    float expected_alpha_lattice = 0.01024f;
    float expected_tau = 0.54096f;
    float expected_omega = 1.84857f;

    std::cout << "Comparison:\n";
    std::cout << "-----------\n";
    std::cout << "alpha_lattice error: " << std::abs(alpha_lattice - expected_alpha_lattice) << "\n";
    std::cout << "tau error:           " << std::abs(tau - expected_tau) << "\n";
    std::cout << "omega error:         " << std::abs(omega - expected_omega) << "\n";
    std::cout << "\n";

    if (std::abs(alpha_lattice - expected_alpha_lattice) > 0.001f) {
        std::cout << "WARNING: alpha_lattice mismatch!\n";
    } else {
        std::cout << "PASS: LBM parameters are correct.\n";
    }

    EXPECT_NEAR(alpha_lattice, expected_alpha_lattice, 0.001f);
    EXPECT_NEAR(tau, expected_tau, 0.001f);
    EXPECT_NEAR(omega, expected_omega, 0.01f);
}

/**
 * @brief Test 3: Verify laser energy deposition
 */
TEST(ThermalDiagnostics, LaserEnergyDeposition) {
    std::cout << "\n========================================\n";
    std::cout << "TEST 3: LASER ENERGY DEPOSITION\n";
    std::cout << "========================================\n\n";

    // Laser parameters
    float P_laser = 200.0f;      // W
    float r0 = 50.0e-6f;         // m (spot radius)
    float absorptivity = 0.35f;
    float penetration = 50.0e-6f; // m

    // Grid parameters
    float dx = 3.75e-6f;  // m
    float dt = 5.0e-8f;   // s

    // Absorbed power
    float P_absorbed = P_laser * absorptivity;

    std::cout << "Laser parameters:\n";
    std::cout << "-----------------\n";
    std::cout << "Power P:         " << P_laser << " W\n";
    std::cout << "Spot radius r0:  " << r0 * 1e6 << " μm\n";
    std::cout << "Absorptivity:    " << absorptivity << "\n";
    std::cout << "Penetration:     " << penetration * 1e6 << " μm\n";
    std::cout << "\n";
    std::cout << "Absorbed power:  " << P_absorbed << " W\n";
    std::cout << "\n";

    // Energy per timestep
    float E_per_step = P_absorbed * dt;

    std::cout << "Energy per timestep:\n";
    std::cout << "--------------------\n";
    std::cout << "E_step = P * dt = " << E_per_step << " J\n";
    std::cout << "                = " << E_per_step * 1e6 << " μJ\n";
    std::cout << "\n";

    // Gaussian distribution - peak intensity at center
    // I(r) = (2*P) / (π * r0²) * exp(-2*r²/r0²)
    float I_peak = (2.0f * P_absorbed) / (M_PI * r0 * r0);

    std::cout << "Peak intensity at center:\n";
    std::cout << "-------------------------\n";
    std::cout << "I_peak = " << I_peak << " W/m²\n";
    std::cout << "       = " << I_peak * 1e-9 << " GW/m²\n";
    std::cout << "\n";

    // Volume of one cell
    float cell_volume = dx * dx * dx;
    float cell_mass = 4420.0f * cell_volume;  // kg

    // Energy deposited in center cell per timestep
    float area_cell = dx * dx;
    float E_cell = I_peak * area_cell * dt * exp(-2.0f * penetration / dx);

    std::cout << "Energy in center cell:\n";
    std::cout << "----------------------\n";
    std::cout << "Cell volume:     " << cell_volume * 1e18 << " μm³\n";
    std::cout << "Cell mass:       " << cell_mass * 1e9 << " μg\n";
    std::cout << "E_cell per step: " << E_cell << " J\n";
    std::cout << "\n";

    // Temperature rise in center cell
    float cp = 610.0f;  // J/(kg·K)
    float dT_cell = E_cell / (cell_mass * cp);

    std::cout << "Temperature rise per timestep:\n";
    std::cout << "------------------------------\n";
    std::cout << "dT = E / (m * cp) = " << dT_cell << " K\n";
    std::cout << "\n";

    // After 1000 timesteps (50 μs)
    int n_steps = 1000;
    float dT_total = dT_cell * n_steps;

    std::cout << "After " << n_steps << " timesteps (" << (n_steps * dt * 1e6) << " μs):\n";
    std::cout << "--------------------\n";
    std::cout << "Total dT (no diffusion): " << dT_total << " K\n";
    std::cout << "Final T (from 300 K):    " << (300.0f + dT_total) << " K\n";
    std::cout << "\n";

    std::cout << "NOTE: This is an upper bound. Actual temperature will be lower due to:\n";
    std::cout << "  1. Thermal diffusion spreading heat to neighboring cells\n";
    std::cout << "  2. Heat loss from boundaries\n";
    std::cout << "  3. Latent heat absorption during melting\n";
    std::cout << "\n";

    // Compare with observed peak temperature
    float T_observed = 3544.8f;
    float diffusion_factor = T_observed / (300.0f + dT_total);

    std::cout << "Observed peak temperature: " << T_observed << " K\n";
    std::cout << "Effective diffusion factor: " << diffusion_factor << "\n";
    std::cout << "  (fraction of heat remaining in center cell)\n";
    std::cout << "\n";
}

/**
 * @brief Test 4: Compare with walberla reference
 */
TEST(ThermalDiagnostics, WalberlaComparison) {
    std::cout << "\n========================================\n";
    std::cout << "TEST 4: WALBERLA COMPARISON\n";
    std::cout << "========================================\n\n";

    // Our results
    float our_peak = 3544.8f;      // K
    float our_time = 50.0f;        // μs
    float our_depth = 60.0f;       // μm

    // Walberla reference
    float walberla_peak = 17500.0f;  // K
    float walberla_time = 50.0f;     // μs (assumed)
    float walberla_depth = 60.0f;    // μm (assumed)

    std::cout << "Comparison at t = 50 μs:\n";
    std::cout << "------------------------\n";
    std::cout << std::setw(25) << "Parameter" << std::setw(15) << "Our Code"
              << std::setw(15) << "walberla" << std::setw(15) << "Ratio\n";
    std::cout << std::string(70, '-') << "\n";

    float temp_ratio = our_peak / walberla_peak;
    float depth_ratio = our_depth / walberla_depth;

    std::cout << std::setw(25) << "Peak temperature (K)" << std::setw(15) << our_peak
              << std::setw(15) << walberla_peak << std::setw(15) << temp_ratio << "\n";
    std::cout << std::setw(25) << "Melt pool depth (μm)" << std::setw(15) << our_depth
              << std::setw(15) << walberla_depth << std::setw(15) << depth_ratio << "\n";
    std::cout << "\n";

    std::cout << "Analysis:\n";
    std::cout << "---------\n";
    std::cout << "Temperature ratio: " << (temp_ratio * 100) << "% of walberla\n";
    std::cout << "Temperature deficit: " << (walberla_peak - our_peak) << " K\n";
    std::cout << "\n";

    if (temp_ratio < 0.3f) {
        std::cout << "CRITICAL: Temperature is much lower than walberla!\n";
        std::cout << "\n";
        std::cout << "Possible root causes:\n";
        std::cout << "1. Laser power or absorption coefficient different\n";
        std::cout << "2. Thermal diffusivity too high (heat spreading too fast)\n";
        std::cout << "3. Specific heat or thermal mass too high\n";
        std::cout << "4. Boundary conditions removing too much heat\n";
        std::cout << "5. Time integration or collision operator issues\n";
        std::cout << "6. Temperature clamping or limiting\n";
        std::cout << "\n";
        std::cout << "Recommended actions:\n";
        std::cout << "1. Get exact walberla parameters (k, cp, rho, alpha)\n";
        std::cout << "2. Compare laser source term implementation\n";
        std::cout << "3. Check boundary condition implementation\n";
        std::cout << "4. Verify collision operator matches walberla\n";
        std::cout << "\n";
    }
}

/**
 * @brief Test 5: Energy balance diagnostic
 */
TEST(ThermalDiagnostics, EnergyBalance) {
    std::cout << "\n========================================\n";
    std::cout << "TEST 5: ENERGY BALANCE DIAGNOSTIC\n";
    std::cout << "========================================\n\n";

    // Simulation parameters
    float P_laser = 200.0f;        // W
    float absorptivity = 0.35f;
    float P_absorbed = P_laser * absorptivity;  // W

    float laser_on_time = 50e-6f;  // s
    float E_input = P_absorbed * laser_on_time;  // J

    // Domain
    float dx = 3.75e-6f;  // m
    int nx = 40, ny = 80, nz = 80;
    float volume = (nx * dx) * (ny * dx) * (nz * dx);  // m³

    // Material
    float rho = 4420.0f;  // kg/m³
    float cp = 610.0f;    // J/(kg·K)
    float mass = rho * volume;  // kg

    // Temperature rise
    float T_initial = 300.0f;   // K
    float T_final = 3544.8f;    // K (observed)
    float dT = T_final - T_initial;

    // Energy stored
    float E_stored = mass * cp * dT;  // J

    std::cout << "Energy input:\n";
    std::cout << "-------------\n";
    std::cout << "Laser power:      " << P_laser << " W\n";
    std::cout << "Absorptivity:     " << absorptivity << "\n";
    std::cout << "Absorbed power:   " << P_absorbed << " W\n";
    std::cout << "Laser on time:    " << (laser_on_time * 1e6) << " μs\n";
    std::cout << "Total energy in:  " << E_input << " J\n";
    std::cout << "\n";

    std::cout << "Energy storage:\n";
    std::cout << "---------------\n";
    std::cout << "Domain volume:    " << (volume * 1e9) << " mm³\n";
    std::cout << "Total mass:       " << (mass * 1e6) << " mg\n";
    std::cout << "Temperature rise: " << dT << " K\n";
    std::cout << "Energy stored:    " << E_stored << " J\n";
    std::cout << "\n";

    std::cout << "Energy balance:\n";
    std::cout << "---------------\n";
    std::cout << "E_stored / E_input = " << (E_stored / E_input) << "\n";
    std::cout << "\n";

    if (E_stored > E_input * 2) {
        std::cout << "ERROR: Energy stored exceeds energy input by >2×!\n";
        std::cout << "This indicates:\n";
        std::cout << "  - Laser source term is too strong, OR\n";
        std::cout << "  - Temperature calculation has errors\n";
    } else if (E_stored > E_input) {
        std::cout << "WARNING: Energy stored exceeds energy input.\n";
        std::cout << "This is physically impossible without external heat sources.\n";
        std::cout << "Possible causes:\n";
        std::cout << "  - Laser source term overestimates energy deposition\n";
        std::cout << "  - Spatial integration of laser profile is incorrect\n";
    } else {
        std::cout << "PASS: Energy balance is reasonable.\n";
        std::cout << "Energy losses account for " << ((1 - E_stored / E_input) * 100) << "% of input.\n";
    }
    std::cout << "\n";

    // What temperature would we get if all energy was retained?
    float dT_ideal = E_input / (mass * cp);
    float T_ideal = T_initial + dT_ideal;

    std::cout << "Ideal case (no heat loss):\n";
    std::cout << "--------------------------\n";
    std::cout << "dT_ideal:  " << dT_ideal << " K\n";
    std::cout << "T_ideal:   " << T_ideal << " K\n";
    std::cout << "Efficiency: " << (dT / dT_ideal * 100) << "%\n";
    std::cout << "\n";

    std::cout << "For walberla peak of 17500 K:\n";
    std::cout << "-----------------------------\n";
    float dT_walberla = 17500.0f - 300.0f;
    float E_walberla = mass * cp * dT_walberla;
    std::cout << "Required energy: " << E_walberla << " J\n";
    std::cout << "Input energy:    " << E_input << " J\n";
    std::cout << "Energy multiplier needed: " << (E_walberla / E_input) << "×\n";
    std::cout << "\n";
    std::cout << "This suggests walberla either:\n";
    std::cout << "  1. Has much higher laser power/absorption\n";
    std::cout << "  2. Has lower thermal mass (smaller effective volume heated)\n";
    std::cout << "  3. Has concentrated energy deposition\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
