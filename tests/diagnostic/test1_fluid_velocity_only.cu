/**
 * @file test1_fluid_velocity_only.cu
 * @brief Test 1: Bare FluidLBM Velocity Computation
 *
 * PURPOSE: Does FluidLBM compute ANY velocity without multiphysics?
 *
 * TEST CASE:
 * - Initialize 10x10x10 FluidLBM
 * - Apply constant force F_x = 1e-5 everywhere
 * - Run 100 steps
 * - Check: Is max(|ux|) > 0?
 *
 * EXPECTED: Should see v_x ~ F*tau/rho ~ 1e-5 * 100us / 4000 ~ 0.001 m/s
 * IF ZERO: FluidLBM::step() not computing velocity from forces
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
    std::cout << "  TEST 1: Bare FluidLBM Velocity Computation\n";
    std::cout << "  Isolates: FluidLBM velocity computation from forces\n";
    std::cout << "================================================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Test parameters
    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;      // Lattice viscosity (tau ~ 0.95)
    float rho0 = 4000.0f;  // Approximate liquid density [kg/m³]

    std::cout << "Setup:\n";
    std::cout << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Viscosity: " << nu << " (lattice units)\n";
    std::cout << "  Density: " << rho0 << " kg/m³\n";
    std::cout << "  Boundaries: All periodic (no walls)\n\n";

    // Create solver with periodic boundaries
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    // Initialize at rest
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::cout << "Test: Apply uniform force F_x = 1e-5 m/s²\n";
    std::cout << "----------------------------------------------------\n\n";

    // Constant force in x-direction
    const float F_x = 1e-5f;  // m/s² (small force)
    const float F_y = 0.0f;
    const float F_z = 0.0f;

    // Expected velocity after 100 steps
    // For small forces: v ~ F * t / rho (approx, ignoring viscosity)
    // But in LBM with BGK: velocity builds up gradually
    // After equilibration: F = viscous_drag, so steady state velocity
    // For this test, we just check if velocity > 0

    const float dt_lattice = 1.0f;  // Lattice time step
    const int n_steps = 100;
    const float expected_v_order = F_x * n_steps * dt_lattice / rho0;

    std::cout << "Expected velocity order of magnitude: " << expected_v_order << " m/s\n";
    std::cout << "  (simplified estimate: v ~ F*t/rho)\n\n";

    std::cout << std::setw(8) << "Step"
              << std::setw(15) << "v_max [m/s]"
              << std::setw(15) << "v_avg [m/s]"
              << "\n";
    std::cout << std::string(38, '-') << "\n";

    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];

    for (int step = 0; step <= n_steps; step++) {
        // FluidLBM solution step
        solver.computeMacroscopic();
        solver.collisionBGK(F_x, F_y, F_z);  // Uniform force
        solver.streaming();

        // Output every 10 steps
        if (step % 10 == 0) {
            solver.copyVelocityToHost(h_ux, h_uy, h_uz);

            float v_max = 0.0f, v_sum = 0.0f;
            for (int i = 0; i < n_cells; i++) {
                float v_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                v_max = std::max(v_max, v_mag);
                v_sum += v_mag;
            }
            float v_avg = v_sum / n_cells;

            std::cout << std::setw(8) << step
                      << std::setw(15) << std::scientific << std::setprecision(4)
                      << v_max
                      << std::setw(15) << v_avg
                      << "\n";
        }
    }

    std::cout << "\n";

    // Final analysis
    solver.copyVelocityToHost(h_ux, h_uy, h_uz);

    float v_max = 0.0f;
    float v_x_max = 0.0f;
    int zero_count = 0;

    for (int i = 0; i < n_cells; i++) {
        float v_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        v_max = std::max(v_max, v_mag);
        v_x_max = std::max(v_x_max, std::abs(h_ux[i]));

        if (v_mag < 1e-10f) zero_count++;
    }

    std::cout << "================================================================\n";
    std::cout << "RESULTS:\n";
    std::cout << "  Max |v|: " << v_max << " m/s\n";
    std::cout << "  Max |v_x|: " << v_x_max << " m/s\n";
    std::cout << "  Cells with zero velocity: " << zero_count << " / " << n_cells << "\n\n";

    // Diagnosis
    std::cout << "DIAGNOSIS:\n";
    bool passed = false;

    if (v_max < 1e-10f) {
        std::cout << "  [FAIL] Velocity is ZERO after applying constant force!\n";
        std::cout << "  ROOT CAUSE: FluidLBM is NOT computing velocity from forces.\n";
        std::cout << "\n";
        std::cout << "  Possible issues:\n";
        std::cout << "  1. FluidLBM::collisionBGK() not applying forces to distribution\n";
        std::cout << "  2. FluidLBM::computeMacroscopic() not computing velocity correctly\n";
        std::cout << "  3. Force term in BGK collision kernel is broken\n";
    } else if (v_x_max < 1e-8f) {
        std::cout << "  [FAIL] Velocity is very small (< 1e-8 m/s)\n";
        std::cout << "  ROOT CAUSE: Force is being applied but with wrong magnitude.\n";
        std::cout << "\n";
        std::cout << "  Possible issues:\n";
        std::cout << "  1. Force scaling/unit conversion error\n";
        std::cout << "  2. Excessive damping in collision operator\n";
    } else {
        std::cout << "  [PASS] FluidLBM correctly computes velocity from forces!\n";
        std::cout << "  Velocity magnitude is reasonable.\n";
        std::cout << "\n";
        std::cout << "  CONCLUSION: The basic FluidLBM force->velocity pipeline works.\n";
        std::cout << "  If v5 test has zero velocity, the problem is in:\n";
        std::cout << "    - Buoyancy force computation\n";
        std::cout << "    - Darcy damping\n";
        std::cout << "    - Multiphysics coupling\n";
        passed = true;
    }

    // Cleanup
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;

    std::cout << "\n================================================================\n";
    std::cout << "  Test 1 Complete: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "================================================================\n";

    return passed ? 0 : 1;
}
