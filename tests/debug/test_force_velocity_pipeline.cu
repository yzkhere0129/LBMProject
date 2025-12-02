/**
 * @file test_force_velocity_pipeline.cu
 * @brief TEST LEVEL 2: Test complete force → velocity pipeline
 *
 * Purpose: Test entire pipeline from force application to velocity
 *
 * Test Strategy:
 *   - Initialize FluidLBM solver
 *   - Apply constant force
 *   - Run: collision → streaming → computeMacroscopic
 *   - Repeat multiple times
 *   - Check velocity grows linearly: v ≈ F*t
 *
 * Expected Result:
 *   - Velocity should increase over time
 *   - Growth should be roughly linear
 *
 * If FAIL: Somewhere in pipeline velocity is reset or not accumulated
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TEST LEVEL 2: Force→Velocity Pipeline\n";
    std::cout << "========================================\n\n";

    // Initialize D3Q19
    core::D3Q19::initializeDevice();

    // Small domain for testing
    const int nx = 8, ny = 8, nz = 8;
    const int num_cells = nx * ny * nz;

    std::cout << "Test setup:\n";
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << "\n";
    std::cout << "  Viscosity: 0.1\n";
    std::cout << "  Density: 1.0\n";
    std::cout << "  Applied force: fx=1e-4, fy=0, fz=0\n";
    std::cout << "  Boundary: All periodic\n\n";

    // Create solver
    float nu = 0.1f;
    float rho0 = 1.0f;
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                             physics::BoundaryType::PERIODIC,
                             physics::BoundaryType::PERIODIC,
                             physics::BoundaryType::PERIODIC);

    // Initialize at rest
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Test parameters
    float force_x = 1e-4f;
    float force_y = 0.0f;
    float force_z = 0.0f;
    const int n_steps = 200;
    const int check_interval = 20;

    std::cout << "Running " << n_steps << " steps with constant force...\n";
    std::cout << "Expected: Velocity should increase linearly\n\n";

    std::cout << "Step  |  Max |ux|  |  Avg ux  |  Expected  | Status\n";
    std::cout << "------|----------|----------|------------|--------\n";

    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    bool velocity_increased = false;
    bool test_passed = true;
    float last_avg_ux = 0.0f;

    for (int step = 0; step <= n_steps; ++step) {
        // LBM cycle
        solver.computeMacroscopic();
        solver.collisionBGK(force_x, force_y, force_z);
        solver.streaming();

        // Check velocity periodically
        if (step % check_interval == 0) {
            solver.computeMacroscopic();
            solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            // Compute statistics
            float max_ux = 0.0f;
            float avg_ux = 0.0f;
            float max_uy = 0.0f;
            float max_uz = 0.0f;

            for (int i = 0; i < num_cells; ++i) {
                float ux_mag = std::abs(h_ux[i]);
                float uy_mag = std::abs(h_uy[i]);
                float uz_mag = std::abs(h_uz[i]);

                max_ux = std::max(max_ux, ux_mag);
                max_uy = std::max(max_uy, uy_mag);
                max_uz = std::max(max_uz, uz_mag);
                avg_ux += h_ux[i];
            }
            avg_ux /= num_cells;

            // Expected velocity: v ≈ F/ρ * t (ignoring viscous effects for short times)
            // In LBM with forcing: du/dt ≈ F/ρ
            float expected_ux = force_x * step / rho0;

            // Check if velocity is growing
            bool growing = (avg_ux > last_avg_ux) || (step == 0);
            if (avg_ux > 1e-8f) velocity_increased = true;

            std::string status = "OK";
            if (step > 0 && !growing) {
                status = "FAIL";
                test_passed = false;
            }

            std::cout << std::setw(5) << step << " | "
                      << std::setw(8) << std::scientific << std::setprecision(2) << max_ux << " | "
                      << std::setw(8) << avg_ux << " | "
                      << std::setw(10) << expected_ux << " | "
                      << status << "\n";

            last_avg_ux = avg_ux;

            // Check for NaN
            if (std::isnan(avg_ux) || std::isnan(max_ux)) {
                std::cout << "\nERROR: NaN detected at step " << step << "\n";
                test_passed = false;
                break;
            }
        }
    }

    // Final velocity check
    solver.computeMacroscopic();
    solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    float final_avg_ux = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        final_avg_ux += h_ux[i];
    }
    final_avg_ux /= num_cells;

    std::cout << "\n=== Analysis ===\n";
    std::cout << "Final average ux: " << final_avg_ux << "\n";
    std::cout << "Velocity increased: " << (velocity_increased ? "YES" : "NO") << "\n";

    // Check uniformity (for periodic BC, flow should be uniform)
    float max_deviation = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float dev = std::abs(h_ux[i] - final_avg_ux);
        max_deviation = std::max(max_deviation, dev);
    }
    std::cout << "Max deviation from average: " << max_deviation << "\n";
    bool uniform = (max_deviation < 1e-3f);
    std::cout << "Flow uniform: " << (uniform ? "YES" : "NO") << "\n";

    // Verdict
    std::cout << "\n=== VERDICT ===\n";
    test_passed = test_passed && velocity_increased && uniform;

    if (test_passed) {
        std::cout << "PASS: Force→Velocity pipeline works\n";
        std::cout << "  - Velocity increases over time\n";
        std::cout << "  - Growth is monotonic\n";
        std::cout << "  - Flow is uniform (periodic BC)\n";
    } else {
        std::cout << "FAIL: Pipeline has issues\n";
        if (!velocity_increased) {
            std::cout << "  → BUG: Velocity not increasing (force not applied or velocity reset)\n";
        }
        if (!uniform) {
            std::cout << "  → WARNING: Flow not uniform (boundary issue?)\n";
        }
    }

    return test_passed ? 0 : 1;
}
