/**
 * @file test_no_boundaries.cu
 * @brief Test force application WITHOUT boundary conditions
 *
 * This test checks if applyBoundaryConditions is causing the zero-velocity bug.
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
    std::cout << "  Test: Force Application WITHOUT BC\n";
    std::cout << "==============================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    const int nx = 20, ny = 20, nz = 20;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;
    float rho0 = 4420.0f;

    // Create solver with WALL boundaries
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::WALL,
                    physics::BoundaryType::WALL,
                    physics::BoundaryType::PERIODIC);

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Create constant force
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    float force_val = 100.0f;  // Large force for testing
    float* h_force_x = new float[n_cells];
    float* h_force_y = new float[n_cells];
    float* h_force_z = new float[n_cells];

    for (int i = 0; i < n_cells; i++) {
        h_force_x[i] = force_val;
        h_force_y[i] = 0.0f;
        h_force_z[i] = 0.0f;
    }

    cudaMemcpy(d_fx, h_force_x, n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, h_force_y, n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, h_force_z, n_cells * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Applied force: fx = " << force_val << " m/s²\n\n";

    float* h_ux = new float[n_cells];

    // Test 1: WITH boundary conditions
    std::cout << "Test 1: WITH applyBoundaryConditions(1)\n";
    std::cout << "----------------------------------------\n";

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    for (int step = 0; step < 100; step++) {
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();
        solver.applyBoundaryConditions(1);  // Apply BC
    }

    solver.copyVelocityToHost(h_ux, nullptr, nullptr);

    float u_max_with_bc = 0.0f;
    int zero_count_with_bc = 0;
    for (int i = 0; i < n_cells; i++) {
        u_max_with_bc = std::max(u_max_with_bc, std::abs(h_ux[i]));
        if (std::abs(h_ux[i]) < 1e-10f) zero_count_with_bc++;
    }

    std::cout << "  Max velocity: " << u_max_with_bc << " m/s\n";
    std::cout << "  Zero cells: " << zero_count_with_bc << " / " << n_cells << "\n\n";

    // Test 2: WITHOUT boundary conditions
    std::cout << "Test 2: WITHOUT applyBoundaryConditions()\n";
    std::cout << "-----------------------------------------\n";

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    for (int step = 0; step < 100; step++) {
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();
        // NO boundary conditions!
    }

    solver.copyVelocityToHost(h_ux, nullptr, nullptr);

    float u_max_no_bc = 0.0f;
    int zero_count_no_bc = 0;
    for (int i = 0; i < n_cells; i++) {
        u_max_no_bc = std::max(u_max_no_bc, std::abs(h_ux[i]));
        if (std::abs(h_ux[i]) < 1e-10f) zero_count_no_bc++;
    }

    std::cout << "  Max velocity: " << u_max_no_bc << " m/s\n";
    std::cout << "  Zero cells: " << zero_count_no_bc << " / " << n_cells << "\n\n";

    // Analysis
    std::cout << "Analysis:\n";
    if (u_max_with_bc < 1e-6f && u_max_no_bc > 1e-3f) {
        std::cout << "  ❌ BUG CONFIRMED: applyBoundaryConditions() is zeroing ALL velocities!\n";
        std::cout << "  The boundary condition implementation is broken.\n";
    } else if (u_max_with_bc < 1e-6f && u_max_no_bc < 1e-6f) {
        std::cout << "  ❌ BUG: Forces not being applied even without BC.\n";
        std::cout << "  Problem is in collision or streaming kernel.\n";
    } else {
        std::cout << "  ✅ Both tests show non-zero velocity.\n";
        std::cout << "  Bug may be elsewhere in the simulation setup.\n";
    }

    delete[] h_ux;
    delete[] h_force_x;
    delete[] h_force_y;
    delete[] h_force_z;
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}
