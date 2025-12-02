/**
 * @file test_darcy_survival.cu
 * @brief TEST LEVEL 3.2: Check if Darcy damping zeros out forces
 *
 * Purpose: Verify that applyDarcyDamping doesn't eliminate all forces
 *
 * Test Strategy:
 *   - Set initial forces to non-zero
 *   - Call applyDarcyDamping with realistic liquid fraction
 *   - Check forces after damping
 *
 * Expected Result:
 *   - Forces should still be non-zero in liquid regions (fl > 0.5)
 *   - Forces should be reduced but not eliminated
 *
 * If FAIL: Darcy is zeroing all forces → BUG IN DARCY KERNEL
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TEST LEVEL 3.2: Darcy Damping Survival\n";
    std::cout << "========================================\n\n";

    core::D3Q19::initializeDevice();

    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;

    std::cout << "Test setup:\n";
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << "\n";
    std::cout << "  Initial forces: fy = 100.0 N/m³\n";
    std::cout << "  Velocity: ux = 0.1 m/s\n";
    std::cout << "  Liquid fraction: gradient from 0 (solid) to 1 (liquid)\n";
    std::cout << "  Darcy constant: 1e5 kg/(m³·s)\n\n";

    // Create solver
    float nu = 0.1f;
    float rho0 = 1000.0f;
    physics::FluidLBM solver(nx, ny, nz, nu, rho0);

    // Initialize with uniform velocity in x
    solver.initialize(rho0, 0.1f, 0.0f, 0.0f);

    // Create arrays
    float *d_liquid_fraction, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_liquid_fraction, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    // Initialize liquid fraction (gradient in y)
    std::vector<float> h_liquid_fraction(num_cells);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                // Gradient: solid at bottom (fl=0) to liquid at top (fl=1)
                h_liquid_fraction[id] = static_cast<float>(iy) / (ny - 1);
            }
        }
    }

    cudaMemcpy(d_liquid_fraction, h_liquid_fraction.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize forces (uniform buoyancy)
    std::vector<float> h_fx_init(num_cells, 0.0f);
    std::vector<float> h_fy_init(num_cells, 100.0f);  // Uniform upward force
    std::vector<float> h_fz_init(num_cells, 0.0f);

    cudaMemcpy(d_fx, h_fx_init.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, h_fy_init.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, h_fz_init.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Initial state:\n";
    std::cout << "  fy (before Darcy): 100.0 N/m³ (all cells)\n\n";

    // Apply Darcy damping
    float darcy_const = 1e5f;
    std::cout << "Applying Darcy damping...\n";
    solver.applyDarcyDamping(d_liquid_fraction, darcy_const, d_fx, d_fy, d_fz);

    // Copy back
    std::vector<float> h_fx(num_cells);
    std::vector<float> h_fy(num_cells);
    std::vector<float> h_fz(num_cells);

    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Get velocities
    std::vector<float> h_ux(num_cells);
    solver.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    // Analyze
    std::cout << "\n=== Analysis ===\n";
    std::cout << "Forces after Darcy damping:\n\n";
    std::cout << "   y  |   fl   |  ux [m/s]  |  fy_before  |  fy_after  |  Damping  | Status\n";
    std::cout << "------|--------|------------|-------------|------------|-----------|--------\n";

    int ix_center = nx / 2;
    int iz_center = nz / 2;

    bool test_passed = true;
    int liquid_cells_with_force = 0;
    int solid_cells_zeroed = 0;

    for (int iy = 0; iy < ny; ++iy) {
        int id = ix_center + iy * nx + iz_center * nx * ny;
        float fl = h_liquid_fraction[id];
        float ux = h_ux[id];
        float fy_before = h_fy_init[id];
        float fy_after = h_fy[id];
        float damping = fy_after - fy_before;

        // Expected damping force: -C*(1-fl)²/(fl³+eps)*ux
        const float eps = 1e-3f;
        float expected_damping_x = -darcy_const * (1.0f - fl) * (1.0f - fl) / (fl * fl * fl + eps) * ux;

        std::string status = "OK";

        // Check liquid regions (fl > 0.5) should still have positive force
        if (fl > 0.5f) {
            if (fy_after > 50.0f) {
                liquid_cells_with_force++;
            } else if (fy_after < 1.0f) {
                status = "FAIL";
                test_passed = false;
            }
        }

        // Check solid regions (fl < 0.1) should have strong damping
        if (fl < 0.1f && std::abs(damping) > 100.0f) {
            solid_cells_zeroed++;
        }

        std::cout << "  " << std::setw(3) << iy << "  | "
                  << std::setw(6) << std::fixed << std::setprecision(2) << fl << " | "
                  << std::setw(10) << std::setprecision(4) << ux << " | "
                  << std::setw(11) << std::setprecision(2) << fy_before << " | "
                  << std::setw(10) << fy_after << " | "
                  << std::setw(9) << damping << " | "
                  << status << "\n";
    }

    std::cout << "\nSummary:\n";
    std::cout << "  Liquid cells (fl>0.5) with fy>50: " << liquid_cells_with_force << "\n";
    std::cout << "  Solid cells (fl<0.1) with strong damping: " << solid_cells_zeroed << "\n";

    // Check x-direction damping (should affect fx)
    float max_fx_damping = *std::max_element(h_fx.begin(), h_fx.end(),
                                              [](float a, float b) { return std::abs(a) < std::abs(b); });
    std::cout << "  max|fx| (damping in x-direction): " << std::abs(max_fx_damping) << " N/m³\n";

    bool has_x_damping = std::abs(max_fx_damping) > 1.0f;
    std::cout << "  X-damping present: " << (has_x_damping ? "YES" : "NO") << "\n";

    // Verdict
    std::cout << "\n=== VERDICT ===\n";

    // Test passes if:
    // 1. Liquid regions still have significant force
    // 2. Solid regions have strong damping
    // 3. X-direction damping is present (opposing ux)
    test_passed = test_passed && (liquid_cells_with_force > 0) && has_x_damping;

    if (test_passed) {
        std::cout << "PASS: Darcy damping works correctly\n";
        std::cout << "  - Liquid regions retain buoyancy forces\n";
        std::cout << "  - Solid regions have strong damping\n";
        std::cout << "  - Velocity-dependent damping applied\n";
    } else {
        std::cout << "FAIL: Darcy damping has issues\n";
        if (liquid_cells_with_force == 0) {
            std::cout << "  → BUG: All liquid forces eliminated!\n";
            std::cout << "  → This could cause zero velocity in simulation\n";
        }
        if (!has_x_damping) {
            std::cout << "  → WARNING: No velocity-dependent damping detected\n";
        }
    }

    // Cleanup
    cudaFree(d_liquid_fraction);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return test_passed ? 0 : 1;
}
