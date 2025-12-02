/**
 * @file diagnostic_force_propagation.cu
 * @brief Diagnostic test to verify force propagation to velocity
 *
 * This test isolates the suspected architectural bug:
 * - Initialize fluid at rest
 * - Apply uniform force for N steps
 * - Check if computeMacroscopic() placement affects velocity
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm::physics;
using namespace lbm::core;

int main() {
    std::cout << "\n========================================\n";
    std::cout << " Diagnostic: Force Propagation Test\n";
    std::cout << "========================================\n\n";

    // Small grid for fast testing
    const int nx = 8, ny = 8, nz = 8;
    int num_cells = nx * ny * nz;

    float nu = 0.1f;
    float rho0 = 1.0f;
    float force_x = 1e-4f;

    // Initialize D3Q19
    D3Q19::initializeDevice();

    // ====================================================================
    // Test 1: WRONG order (as in Phase 5) - computeMacroscopic BEFORE collision
    // ====================================================================
    std::cout << "Test 1: computeMacroscopic() BEFORE collision (WRONG)\n";
    std::cout << "Order: computeMacro → collision → streaming (repeat)\n";
    std::cout << std::string(60, '-') << "\n";

    FluidLBM solver1(nx, ny, nz, nu, rho0);
    solver1.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::vector<float> ux1(num_cells);

    // Run 10 steps with WRONG order
    for (int step = 0; step < 10; ++step) {
        solver1.computeMacroscopic();  // ❌ BEFORE collision
        solver1.collisionBGK(force_x, 0.0f, 0.0f);
        solver1.streaming();

        // Check velocity at each step
        solver1.copyVelocityToHost(ux1.data(), nullptr, nullptr);
        float avg_ux = 0.0f;
        for (int i = 0; i < num_cells; ++i) {
            avg_ux += ux1[i];
        }
        avg_ux /= num_cells;

        std::cout << "  Step " << std::setw(2) << step
                  << ": avg_ux = " << std::scientific << std::setprecision(6) << avg_ux << "\n";
    }

    std::cout << "\n";

    // ====================================================================
    // Test 2: CORRECT order - computeMacroscopic AFTER streaming
    // ====================================================================
    std::cout << "Test 2: computeMacroscopic() AFTER streaming (CORRECT)\n";
    std::cout << "Order: collision → streaming → computeMacro (repeat)\n";
    std::cout << std::string(60, '-') << "\n";

    FluidLBM solver2(nx, ny, nz, nu, rho0);
    solver2.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::vector<float> ux2(num_cells);

    // Run 10 steps with CORRECT order
    for (int step = 0; step < 10; ++step) {
        solver2.collisionBGK(force_x, 0.0f, 0.0f);
        solver2.streaming();
        solver2.computeMacroscopic();  // ✅ AFTER streaming

        // Check velocity at each step
        solver2.copyVelocityToHost(ux2.data(), nullptr, nullptr);
        float avg_ux = 0.0f;
        for (int i = 0; i < num_cells; ++i) {
            avg_ux += ux2[i];
        }
        avg_ux /= num_cells;

        std::cout << "  Step " << std::setw(2) << step
                  << ": avg_ux = " << std::scientific << std::setprecision(6) << avg_ux << "\n";
    }

    std::cout << "\n";

    // ====================================================================
    // Test 3: AS IN UNIT TEST - computeMacro before, but final call after loop
    // ====================================================================
    std::cout << "Test 3: Unit test pattern (computeMacro before + final after)\n";
    std::cout << "Order: (computeMacro → collision → streaming) x N, then final computeMacro\n";
    std::cout << std::string(60, '-') << "\n";

    FluidLBM solver3(nx, ny, nz, nu, rho0);
    solver3.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::vector<float> ux3(num_cells);

    // Run 10 steps like the unit test
    for (int step = 0; step < 10; ++step) {
        solver3.computeMacroscopic();  // Before collision
        solver3.collisionBGK(force_x, 0.0f, 0.0f);
        solver3.streaming();

        // DON'T check velocity here - it's stale
    }

    solver3.computeMacroscopic();  // ✅ Final call to get updated velocities

    solver3.copyVelocityToHost(ux3.data(), nullptr, nullptr);
    float avg_ux3 = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        avg_ux3 += ux3[i];
    }
    avg_ux3 /= num_cells;

    std::cout << "  After 10 steps + final computeMacro: avg_ux = "
              << std::scientific << std::setprecision(6) << avg_ux3 << "\n";

    std::cout << "\n";

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n========================================\n";
    std::cout << " Summary\n";
    std::cout << "========================================\n\n";

    // Get final velocities from Test 1 (wrong order)
    solver1.copyVelocityToHost(ux1.data(), nullptr, nullptr);
    float final_ux1 = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        final_ux1 += ux1[i];
    }
    final_ux1 /= num_cells;

    // Get final velocities from Test 2 (correct order)
    solver2.copyVelocityToHost(ux2.data(), nullptr, nullptr);
    float final_ux2 = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        final_ux2 += ux2[i];
    }
    final_ux2 /= num_cells;

    std::cout << "Test 1 (WRONG order):    final avg_ux = "
              << std::scientific << std::setprecision(6) << final_ux1
              << " " << (fabs(final_ux1) < 1e-10 ? "⚠️  ZERO!" : "✓") << "\n";

    std::cout << "Test 2 (CORRECT order):  final avg_ux = "
              << std::scientific << std::setprecision(6) << final_ux2
              << " " << (fabs(final_ux2) > 1e-6 ? "✓" : "⚠️  ZERO!") << "\n";

    std::cout << "Test 3 (UNIT TEST way):  final avg_ux = "
              << std::scientific << std::setprecision(6) << avg_ux3
              << " " << (fabs(avg_ux3) > 1e-6 ? "✓" : "⚠️  ZERO!") << "\n";

    std::cout << "\n";

    if (fabs(final_ux1) < 1e-10 && fabs(final_ux2) > 1e-6) {
        std::cout << "✅ DIAGNOSIS CONFIRMED: computeMacroscopic() placement is the bug!\n";
        std::cout << "   - When called BEFORE collision: velocities never update (ZERO)\n";
        std::cout << "   - When called AFTER streaming: velocities accumulate correctly\n";
    } else {
        std::cout << "❌ Unexpected result - bug is NOT the ordering issue\n";
    }

    std::cout << "\n";

    return 0;
}
