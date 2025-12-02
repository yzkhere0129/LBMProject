/**
 * @file test0_unit_conversion.cu
 * @brief Test 0: Unit Conversion Verification
 *
 * PURPOSE: Determine the exact conversion factor needed for force scaling
 *
 * This test applies forces with different scaling factors to find the
 * correct lattice-to-physical unit conversion.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  TEST 0: Unit Conversion Verification\n";
    std::cout << "  Determines correct force scaling factor\n";
    std::cout << "================================================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Test parameters
    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;      // Lattice viscosity
    float rho0 = 4000.0f;  // Density [kg/m³]

    // Physical parameters
    const float dt_physical = 1e-9f;  // 1 nanosecond
    const float dx_physical = 1e-6f;  // 1 micrometer

    std::cout << "Physical parameters:\n";
    std::cout << "  dt = " << dt_physical << " s\n";
    std::cout << "  dx = " << dx_physical << " m\n";
    std::cout << "  rho = " << rho0 << " kg/m³\n";
    std::cout << "  nu_lattice = " << nu << "\n\n";

    // Expected conversion factors
    float conv_dt2_dx = dt_physical * dt_physical / dx_physical;
    float conv_dt_dx2 = dt_physical / (dx_physical * dx_physical);

    std::cout << "Possible conversion factors:\n";
    std::cout << "  dt²/dx = " << conv_dt2_dx << " (dimensionless force)\n";
    std::cout << "  dt/dx² = " << conv_dt_dx2 << " (for acceleration)\n\n";

    // Test different scaling factors
    float scaling_factors[] = {
        1.0f,                  // No scaling (current bug)
        1e4f,                  // Empirical from Test 1
        1e6f,
        1e8f,
        1e12f,                 // dt²/dx
        1e15f,                 // dt/dx²
        1e-3f / rho0,         // dt/rho
    };

    const int n_factors = sizeof(scaling_factors) / sizeof(float);

    std::cout << "Testing " << n_factors << " different scaling factors:\n";
    std::cout << "----------------------------------------------------\n\n";

    // Physical force to apply
    const float F_physical = 1e-5f;  // m/s² (same as Test 1)
    const int n_steps = 50;

    for (int i = 0; i < n_factors; i++) {
        float scale = scaling_factors[i];
        float F_lattice = F_physical * scale;

        // Create fresh solver for each test
        physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                        physics::BoundaryType::PERIODIC,
                        physics::BoundaryType::PERIODIC,
                        physics::BoundaryType::PERIODIC);

        solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

        // Run simulation
        for (int step = 0; step < n_steps; step++) {
            solver.computeMacroscopic();
            solver.collisionBGK(F_lattice, 0.0f, 0.0f);
            solver.streaming();
        }

        solver.computeMacroscopic();

        // Check final velocity
        float* h_ux = new float[n_cells];
        solver.copyVelocityToHost(h_ux, nullptr, nullptr);

        float v_max = 0.0f;
        float v_avg = 0.0f;
        for (int j = 0; j < n_cells; j++) {
            v_max = std::max(v_max, std::abs(h_ux[j]));
            v_avg += std::abs(h_ux[j]);
        }
        v_avg /= n_cells;

        delete[] h_ux;

        // Check for NaN or instability
        bool stable = !std::isnan(v_max) && !std::isinf(v_max) && v_max < 1e10f;

        std::cout << "Scale = " << std::scientific << std::setprecision(2) << scale
                  << " | F_lattice = " << F_lattice
                  << " | v_max = " << v_max << " m/s";

        if (!stable) {
            std::cout << " [UNSTABLE]";
        } else if (v_max > 1e-3f && v_max < 1.0f) {
            std::cout << " [GOOD]";
        } else if (v_max < 1e-6f) {
            std::cout << " [TOO SMALL]";
        } else if (v_max > 1.0f) {
            std::cout << " [TOO LARGE]";
        }

        std::cout << "\n";
    }

    std::cout << "\n";

    // Analytical estimate
    std::cout << "================================================================\n";
    std::cout << "ANALYTICAL ESTIMATE:\n";
    std::cout << "================================================================\n\n";

    std::cout << "For LBM, the force term in BGK collision is:\n";
    std::cout << "  F_LBM = (1 - omega/2) * F_lattice * dt_lattice\n\n";

    std::cout << "Where dt_lattice = 1 (in LBM, one time step)\n";
    std::cout << "And omega = 1/tau, with tau = nu/cs² + 0.5\n\n";

    float tau = nu / core::D3Q19::CS2 + 0.5f;
    float omega = 1.0f / tau;

    std::cout << "With our parameters:\n";
    std::cout << "  nu = " << nu << "\n";
    std::cout << "  cs² = " << core::D3Q19::CS2 << "\n";
    std::cout << "  tau = " << tau << "\n";
    std::cout << "  omega = " << omega << "\n\n";

    std::cout << "For a physical force F_phys [m/s²], after N time steps:\n";
    std::cout << "  v ≈ F_phys * N * dt_phys / rho (if no viscosity)\n";
    std::cout << "  v ≈ " << F_physical << " * " << n_steps << " * " << dt_physical
              << " / " << rho0 << "\n";
    std::cout << "  v ≈ " << F_physical * n_steps * dt_physical / rho0 << " m/s\n\n";

    std::cout << "In LBM units, the same velocity should be:\n";
    std::cout << "  v_lattice = v_phys * dt_phys / dx_phys\n";
    std::cout << "  v_lattice = " << F_physical * n_steps * dt_physical / rho0
              << " * " << dt_physical / dx_physical << "\n";
    std::cout << "  v_lattice = " << F_physical * n_steps * dt_physical * dt_physical / (rho0 * dx_physical)
              << " (dimensionless)\n\n";

    std::cout << "Therefore, the force conversion should be:\n";
    std::cout << "  F_lattice = F_phys * (dt_phys / rho_phys)\n";
    std::cout << "  OR\n";
    std::cout << "  F_lattice = F_phys / rho_phys  (if dt_lattice = 1)\n\n";

    float recommended_scale = 1.0f / rho0;
    std::cout << "RECOMMENDED SCALING FACTOR: " << recommended_scale << "\n";
    std::cout << "  (This converts force from [m/s²] to [lattice units])\n\n";

    std::cout << "================================================================\n";
    std::cout << "CONCLUSION:\n";
    std::cout << "================================================================\n\n";

    std::cout << "The force scaling factor depends on the interpretation:\n\n";

    std::cout << "IF forces are accelerations [m/s²]:\n";
    std::cout << "  F_lattice = F_phys / rho_phys\n";
    std::cout << "  Scale = 1/rho = " << 1.0f/rho0 << "\n\n";

    std::cout << "IF forces are force densities [N/m³]:\n";
    std::cout << "  F_lattice = F_phys * dt² / (rho * dx)\n";
    std::cout << "  Scale = dt²/(rho*dx) = " << dt_physical*dt_physical/(rho0*dx_physical) << "\n\n";

    std::cout << "Check the units in your FluidLBM implementation!\n";

    return 0;
}
