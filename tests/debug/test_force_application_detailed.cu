/**
 * @file test_force_application_detailed.cu
 * @brief Detailed diagnostic test for force application in FluidLBM
 *
 * This test creates a minimal setup to verify that:
 * 1. Forces are correctly applied in collision
 * 2. Velocities increase when forces are applied
 * 3. The magnitude of velocity increase is physically reasonable
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
    std::cout << "  Force Application Diagnostic Test\n";
    std::cout << "==============================================\n\n";

    // Initialize device
    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Small domain for easy analysis
    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;  // Same as in the main simulation
    float rho0 = 1.0f;

    std::cout << "Setup:\n";
    std::cout << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Viscosity: " << nu << "\n";
    std::cout << "  Density: " << rho0 << "\n\n";

    // Create solver with PERIODIC boundaries (no walls to complicate things)
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    // Initialize with zero velocity
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::cout << "Initial state: fluid at rest (u=0)\n\n";

    // Test 1: Apply uniform constant force
    std::cout << "Test 1: Uniform constant force\n";
    std::cout << "-------------------------------\n";

    float force_x = 0.01f;  // Moderate force in x-direction
    float force_y = 0.0f;
    float force_z = 0.0f;

    std::cout << "Applied force: fx = " << force_x << " m/s²\n";
    std::cout << "Running 1000 time steps...\n\n";

    std::cout << std::setw(8) << "Step"
              << std::setw(15) << "ux_avg"
              << std::setw(15) << "ux_max"
              << std::setw(15) << "ux_min"
              << "\n";
    std::cout << std::string(53, '-') << "\n";

    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];

    for (int step = 0; step <= 1000; step++) {
        // Standard LBM cycle
        solver.computeMacroscopic();
        solver.collisionBGK(force_x, force_y, force_z);
        solver.streaming();

        // Output every 100 steps
        if (step % 100 == 0) {
            solver.copyVelocityToHost(h_ux, h_uy, h_uz);

            float ux_sum = 0.0f, ux_max = h_ux[0], ux_min = h_ux[0];
            for (int i = 0; i < n_cells; i++) {
                ux_sum += h_ux[i];
                ux_max = std::max(ux_max, h_ux[i]);
                ux_min = std::min(ux_min, h_ux[i]);
            }
            float ux_avg = ux_sum / n_cells;

            std::cout << std::setw(8) << step
                      << std::setw(15) << std::scientific << std::setprecision(6)
                      << ux_avg
                      << std::setw(15) << ux_max
                      << std::setw(15) << ux_min
                      << "\n";
        }
    }

    std::cout << "\n";

    // Final analysis
    solver.copyVelocityToHost(h_ux, h_uy, h_uz);

    float ux_sum = 0.0f, ux_max = h_ux[0], ux_min = h_ux[0];
    int zero_count = 0;

    for (int i = 0; i < n_cells; i++) {
        ux_sum += h_ux[i];
        ux_max = std::max(ux_max, h_ux[i]);
        ux_min = std::min(ux_min, h_ux[i]);

        if (std::abs(h_ux[i]) < 1e-10f) zero_count++;
    }

    float ux_avg = ux_sum / n_cells;

    std::cout << "Final Results:\n";
    std::cout << "  Average ux: " << ux_avg << " m/s\n";
    std::cout << "  Max ux: " << ux_max << " m/s\n";
    std::cout << "  Min ux: " << ux_min << " m/s\n";
    std::cout << "  Cells with zero velocity: " << zero_count << " / " << n_cells << "\n\n";

    // Expected behavior analysis
    std::cout << "Analysis:\n";
    if (ux_avg < 1e-10f && zero_count == n_cells) {
        std::cout << "  ❌ FAIL: All velocities are ZERO!\n";
        std::cout << "  This confirms the force is NOT being applied correctly.\n\n";
        std::cout << "  Possible causes:\n";
        std::cout << "  1. Force term missing in collision kernel\n";
        std::cout << "  2. Force term incorrectly computed\n";
        std::cout << "  3. Force correction missing in computeMacroscopic\n";
    } else if (ux_avg < 1e-6f) {
        std::cout << "  ⚠️  WARNING: Velocity is very small (< 1e-6 m/s)\n";
        std::cout << "  Force may be applied incorrectly or scaled down too much.\n";
    } else if (ux_avg > 0.0f && ux_avg < 0.1f) {
        std::cout << "  ✅ PASS: Velocity increased as expected!\n";
        std::cout << "  Force is being applied correctly.\n";

        // Check uniformity
        float ux_std = 0.0f;
        for (int i = 0; i < n_cells; i++) {
            float diff = h_ux[i] - ux_avg;
            ux_std += diff * diff;
        }
        ux_std = std::sqrt(ux_std / n_cells);

        std::cout << "  Standard deviation: " << ux_std << "\n";
        if (ux_std / ux_avg < 0.01f) {
            std::cout << "  ✅ Flow is uniform (good!)\n";
        } else {
            std::cout << "  ⚠️  Flow is non-uniform (may indicate BC issues)\n";
        }
    } else {
        std::cout << "  ⚠️  WARNING: Velocity too large (> 0.1 m/s)\n";
        std::cout << "  Possible numerical instability.\n";
    }

    std::cout << "\n";

    // Test 2: Spatially-varying forces
    std::cout << "Test 2: Spatially-varying forces\n";
    std::cout << "---------------------------------\n";

    // Reset solver
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Create spatially-varying force field
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    float* h_fx = new float[n_cells];
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;
                // Force increases linearly in x-direction
                h_fx[id] = 0.001f * (float)ix / (nx - 1);
            }
        }
    }

    cudaMemcpy(d_fx, h_fx, n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fy, 0, n_cells * sizeof(float));
    cudaMemset(d_fz, 0, n_cells * sizeof(float));

    std::cout << "Force field: fx varies from 0 to " << h_fx[n_cells-1] << " m/s²\n";
    std::cout << "Running 1000 time steps...\n\n";

    for (int step = 0; step < 1000; step++) {
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();
    }

    solver.copyVelocityToHost(h_ux, h_uy, h_uz);

    // Check velocity gradient
    float ux_left = 0.0f, ux_right = 0.0f;
    int count_left = 0, count_right = 0;

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            int id_left = 0 + iy * nx + iz * nx * ny;  // ix = 0
            int id_right = (nx-1) + iy * nx + iz * nx * ny;  // ix = nx-1

            ux_left += h_ux[id_left];
            ux_right += h_ux[id_right];
            count_left++;
            count_right++;
        }
    }

    ux_left /= count_left;
    ux_right /= count_right;

    std::cout << "Results:\n";
    std::cout << "  Average ux at x=0: " << ux_left << " m/s\n";
    std::cout << "  Average ux at x=max: " << ux_right << " m/s\n";
    std::cout << "  Velocity gradient: " << (ux_right - ux_left) << " m/s\n\n";

    if (ux_right > ux_left + 1e-6f) {
        std::cout << "  ✅ PASS: Spatially-varying forces work correctly!\n";
    } else if (std::abs(ux_right) < 1e-10f && std::abs(ux_left) < 1e-10f) {
        std::cout << "  ❌ FAIL: Both velocities are ZERO!\n";
        std::cout << "  Spatially-varying forces are NOT working.\n";
    } else {
        std::cout << "  ⚠️  WARNING: Expected ux_right > ux_left, but gradient is too small.\n";
    }

    // Cleanup
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;
    delete[] h_fx;
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "\n==============================================\n";
    std::cout << "  Diagnostic Test Complete\n";
    std::cout << "==============================================\n";

    return 0;
}
