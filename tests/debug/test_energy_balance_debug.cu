/**
 * @file test_energy_balance_debug.cu
 * @brief Debug test for energy balance computation
 *
 * This test isolates the energy balance computation to diagnose
 * why dE/dt != P_laser (34% error observed).
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "diagnostics/energy_balance.h"
#include "physics/material_properties.h"

using namespace lbm::diagnostics;

TEST(EnergyBalanceDebug, SimpleHeatUp) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG: Energy Balance Computation" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Simple 10x10x10 domain
    const int nx = 10;
    const int ny = 10;
    const int nz = 10;
    const int num_cells = nx * ny * nz;

    // Physical parameters
    const float dx = 2e-6f;  // 2 μm
    const float dt = 1e-8f;  // 10 ns

    // Ti6Al4V properties
    const float rho = 4420.0f;  // Solid density [kg/m³]
    const float cp = 610.0f;    // Specific heat [J/(kg·K)]
    const float T_ref = 300.0f; // Reference temperature [K]

    // Create temperature field: uniform at 400 K (100 K above reference)
    std::vector<float> h_T(num_cells, 400.0f);

    // Create dummy liquid fraction (all solid)
    std::vector<float> h_fl(num_cells, 0.0f);

    // Allocate device memory
    float* d_T = nullptr;
    float* d_fl = nullptr;
    double* d_energy = nullptr;

    cudaMalloc(&d_T, num_cells * sizeof(float));
    cudaMalloc(&d_fl, num_cells * sizeof(float));
    cudaMalloc(&d_energy, sizeof(double));

    // Copy to device
    cudaMemcpy(d_T, h_T.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fl, h_fl.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Compute energy using diagnostics kernel
    computeThermalEnergy(
        d_T, d_fl,
        rho,
        cp, cp,  // cp_solid, cp_liquid (same for this test)
        dx,
        T_ref,
        nx, ny, nz,
        d_energy
    );

    // Copy result back
    double E_computed = 0.0;
    cudaMemcpy(&E_computed, d_energy, sizeof(double), cudaMemcpyDeviceToHost);

    // Compute expected energy manually
    float dV = dx * dx * dx;
    double E_expected = rho * cp * (400.0f - T_ref) * dV * num_cells;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << " = " << num_cells << " cells" << std::endl;
    std::cout << "  dx = " << dx * 1e6 << " μm" << std::endl;
    std::cout << "  dV = " << dV * 1e18 << " μm³" << std::endl;
    std::cout << "  T_uniform = 400 K" << std::endl;
    std::cout << "  T_ref = " << T_ref << " K" << std::endl;
    std::cout << "  dT = " << 400.0f - T_ref << " K" << std::endl;
    std::cout << "  rho = " << rho << " kg/m³" << std::endl;
    std::cout << "  cp = " << cp << " J/(kg·K)" << std::endl;
    std::cout << std::endl;

    std::cout << "Energy Calculation:" << std::endl;
    std::cout << "  E_expected = rho * cp * dT * dV * N" << std::endl;
    std::cout << "             = " << rho << " * " << cp << " * " << (400.0f - T_ref)
              << " * " << dV << " * " << num_cells << std::endl;
    std::cout << "             = " << std::scientific << std::setprecision(6) << E_expected << " J" << std::endl;
    std::cout << "  E_computed = " << E_computed << " J" << std::endl;
    std::cout << std::endl;

    double error = std::abs(E_computed - E_expected) / E_expected * 100.0;
    std::cout << "Error: " << std::fixed << std::setprecision(3) << error << " %" << std::endl;

    // Check if they match within 0.1%
    EXPECT_LT(error, 0.1) << "Energy computation error too large";

    // Now simulate heating: add 10 W for 10 timesteps
    std::cout << "\n--- Simulating Heating ---" << std::endl;

    const float P_input = 10.0f;  // 10 W input power
    const int n_steps = 10;

    // Volume of domain
    double total_volume = dV * num_cells;

    // Energy added per timestep
    double dE_per_step = P_input * dt;

    // Temperature increase per timestep (uniform heating)
    float dT_per_step = dE_per_step / (rho * cp * total_volume);

    std::cout << "  P_input = " << P_input << " W" << std::endl;
    std::cout << "  dt = " << dt * 1e9 << " ns" << std::endl;
    std::cout << "  dE/step = " << std::scientific << dE_per_step << " J" << std::endl;
    std::cout << "  dT/step = " << std::fixed << std::setprecision(6) << dT_per_step << " K" << std::endl;
    std::cout << std::endl;

    double E_prev = E_computed;
    double t_prev = 0.0;

    for (int step = 0; step < n_steps; ++step) {
        // Heat up all cells
        for (int i = 0; i < num_cells; ++i) {
            h_T[i] += dT_per_step;
        }

        // Copy to device and recompute energy
        cudaMemcpy(d_T, h_T.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

        computeThermalEnergy(
            d_T, d_fl,
            rho,
            cp, cp,
            dx,
            T_ref,
            nx, ny, nz,
            d_energy
        );

        double E_current = 0.0;
        cudaMemcpy(&E_current, d_energy, sizeof(double), cudaMemcpyDeviceToHost);

        double t_current = (step + 1) * dt;
        double dt_elapsed = t_current - t_prev;
        double dE_dt_measured = (E_current - E_prev) / dt_elapsed;

        std::cout << "Step " << std::setw(2) << step + 1
                  << ": T=" << std::fixed << std::setprecision(2) << h_T[0] << " K"
                  << ", E=" << std::scientific << std::setprecision(4) << E_current << " J"
                  << ", dE/dt=" << std::fixed << std::setprecision(3) << dE_dt_measured << " W"
                  << " (expected " << P_input << " W)" << std::endl;

        E_prev = E_current;
        t_prev = t_current;
    }

    std::cout << "\n========================================" << std::endl;

    // Cleanup
    cudaFree(d_T);
    cudaFree(d_fl);
    cudaFree(d_energy);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
