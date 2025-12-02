/**
 * @file test_unit_conversion.cu
 * @brief Test proper unit conversion for forces in LBM
 *
 * LBM operates in lattice units where dx = dt = 1.
 * Physical forces must be converted to lattice units.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "==============================================\n";
    std::cout << "  Unit Conversion Test\n";
    std::cout << "==============================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Physical parameters (same as main simulation)
    const float dx_phys = 2e-6f;  // [m]
    const float dt_phys = 5e-10f;  // [s]
    const float rho_phys = 4420.0f;  // [kg/m³]

    const int nx = 20, ny = 20, nz = 20;
    const int n_cells = nx * ny * nz;

    std::cout << "Physical Parameters:\n";
    std::cout << "  dx = " << dx_phys * 1e6 << " um\n";
    std::cout << "  dt = " << dt_phys * 1e9 << " ns\n";
    std::cout << "  rho = " << rho_phys << " kg/m³\n\n";

    // LBM parameters
    float nu_lattice = 0.15f;

    physics::FluidLBM solver(nx, ny, nz, nu_lattice, 1.0f,  // Note: rho0 = 1.0 in lattice units!
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    solver.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // Physical buoyancy force calculation
    float T_hot = 2400.0f;  // [K]
    float T_ref = 1900.5f;  // [K]
    float beta = 9.0e-6f;   // [1/K]
    float g_phys = 9.81f;   // [m/s²]

    float F_phys = rho_phys * beta * (T_hot - T_ref) * g_phys;  // [N/m³] = [kg/(m²·s²)]

    std::cout << "Buoyancy Force (Physical Units):\n";
    std::cout << "  F_phys = ρ·β·ΔT·g = " << F_phys << " N/m³\n";
    std::cout << "  F_phys = " << F_phys << " kg/(m²·s²)\n\n";

    // Convert to lattice units
    // In LBM: F_lattice has units of [1/lattice_time²]
    // Conversion: F_lattice = F_phys * (dt²/dx) / rho_lattice
    // Since rho_lattice = 1.0 in LBM:
    float F_lattice = F_phys * (dt_phys * dt_phys / dx_phys) / 1.0f;

    std::cout << "Force Conversion to Lattice Units:\n";
    std::cout << "  Conversion factor: dt²/dx = " << (dt_phys * dt_phys / dx_phys) << "\n";
    std::cout << "  F_lattice = F_phys * (dt²/dx) / rho_lattice\n";
    std::cout << "  F_lattice = " << F_phys << " * " << (dt_phys * dt_phys / dx_phys) << " / 1.0\n";
    std::cout << "  F_lattice = " << F_lattice << " (lattice units)\n\n";

    if (F_lattice > 1e-3f) {
        std::cout << "  ⚠️  WARNING: F_lattice = " << F_lattice << " is TOO LARGE!\n";
        std::cout << "  LBM forces should typically be << 1e-3 for stability.\n";
        std::cout << "  This will cause numerical instability.\n\n";
    } else {
        std::cout << "  ✅ F_lattice = " << F_lattice << " is reasonable for LBM.\n\n";
    }

    // Test 1: Apply force WITHOUT conversion (physical units directly)
    std::cout << "Test 1: Force in PHYSICAL units (WRONG!)\n";
    std::cout << "-----------------------------------------\n";

    solver.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    // Set force to physical value (this is WRONG!)
    float* h_fx = new float[n_cells];
    for (int i = 0; i < n_cells; i++) h_fx[i] = F_phys;
    cudaMemcpy(d_fx, h_fx, n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fy, 0, n_cells * sizeof(float));
    cudaMemset(d_fz, 0, n_cells * sizeof(float));

    bool stable1 = true;
    for (int step = 0; step < 100 && stable1; step++) {
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();

        if (step % 20 == 0) {
            float* h_ux = new float[n_cells];
            solver.copyVelocityToHost(h_ux, nullptr, nullptr);

            float u_max = 0.0f;
            int nan_count = 0;
            for (int i = 0; i < n_cells; i++) {
                if (std::isnan(h_ux[i]) || std::isinf(h_ux[i])) {
                    nan_count++;
                }
                u_max = std::max(u_max, std::abs(h_ux[i]));
            }

            std::cout << "  Step " << std::setw(3) << step << ": u_max = " << std::scientific << u_max;
            if (nan_count > 0) {
                std::cout << " (NaN/Inf detected!)";
                stable1 = false;
            }
            std::cout << "\n";

            delete[] h_ux;
        }
    }

    if (!stable1) {
        std::cout << "  ❌ UNSTABLE: Simulation diverged with physical units!\n\n";
    } else {
        std::cout << "  ⚠️  Simulation survived but may be inaccurate.\n\n";
    }

    // Test 2: Apply force WITH proper conversion (lattice units)
    std::cout << "Test 2: Force in LATTICE units (CORRECT!)\n";
    std::cout << "------------------------------------------\n";

    solver.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // Set force to lattice value (this is CORRECT!)
    for (int i = 0; i < n_cells; i++) h_fx[i] = F_lattice;
    cudaMemcpy(d_fx, h_fx, n_cells * sizeof(float), cudaMemcpyHostToDevice);

    bool stable2 = true;
    for (int step = 0; step < 100 && stable2; step++) {
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();

        if (step % 20 == 0) {
            float* h_ux = new float[n_cells];
            solver.copyVelocityToHost(h_ux, nullptr, nullptr);

            float u_max = 0.0f;
            int nan_count = 0;
            for (int i = 0; i < n_cells; i++) {
                if (std::isnan(h_ux[i]) || std::isinf(h_ux[i])) {
                    nan_count++;
                }
                u_max = std::max(u_max, std::abs(h_ux[i]));
            }

            std::cout << "  Step " << std::setw(3) << step << ": u_max = " << std::scientific << u_max;
            if (nan_count > 0) {
                std::cout << " (NaN/Inf detected!)";
                stable2 = false;
            }
            std::cout << "\n";

            delete[] h_ux;
        }
    }

    if (stable2) {
        std::cout << "  ✅ STABLE: Simulation converged with lattice units!\n\n";
    } else {
        std::cout << "  ❌ UNSTABLE: Even lattice units caused instability.\n\n";
    }

    // Cleanup
    delete[] h_fx;
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "==============================================\n";
    std::cout << "Conclusion:\n";
    std::cout << "  Forces MUST be converted from physical to\n";
    std::cout << "  lattice units using: F_latt = F_phys * dt²/dx\n";
    std::cout << "==============================================\n";

    return 0;
}
