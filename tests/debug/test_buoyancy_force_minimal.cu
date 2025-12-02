/**
 * @file test_buoyancy_force_minimal.cu
 * @brief Minimal test to verify buoyancy force produces non-zero velocity
 *
 * This test creates the EXACT same conditions as visualize_laser_melting_with_flow
 * but in a minimal domain to quickly diagnose the issue.
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
    std::cout << "  Buoyancy Force Minimal Test\n";
    std::cout << "  (Reproducing main simulation conditions)\n";
    std::cout << "==============================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Same parameters as main simulation
    const int nx = 20, ny = 20, nz = 20;  // Smaller for speed
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;  // Lattice viscosity
    float rho0 = 4420.0f;  // Ti6Al4V density [kg/m³]

    std::cout << "Setup (same as main simulation):\n";
    std::cout << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Viscosity (lattice): " << nu << "\n";
    std::cout << "  Density: " << rho0 << " kg/m³\n";
    std::cout << "  Boundaries: x/y walls, z periodic\n\n";

    // Create solver with WALL boundaries (same as main sim)
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::WALL,      // x: walls
                    physics::BoundaryType::WALL,      // y: walls
                    physics::BoundaryType::PERIODIC); // z: periodic

    // Initialize at rest
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::cout << "Test: Apply buoyancy force (hot liquid should rise)\n";
    std::cout << "----------------------------------------------------\n\n";

    // Create temperature field with hot region in center
    float *d_temperature, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_temperature, n_cells * sizeof(float));
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    float* h_temperature = new float[n_cells];

    // Temperature distribution: cold everywhere except center
    float T_cold = 1900.0f;  // Below melting
    float T_hot = 2400.0f;   // Above melting (like in main sim)

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;

                // Hot region in center (radius ~5 cells)
                float dx = ix - nx/2;
                float dy = iy - ny/2;
                float dz = iz - nz/2;
                float r = sqrt(dx*dx + dy*dy + dz*dz);

                if (r < 5.0f) {
                    h_temperature[id] = T_hot;
                } else {
                    h_temperature[id] = T_cold;
                }
            }
        }
    }

    cudaMemcpy(d_temperature, h_temperature, n_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Buoyancy parameters (same as main simulation)
    float T_ref = 1900.5f;  // Reference temperature [K]
    float beta = 9.0e-6f;   // Thermal expansion coefficient [1/K]
    float g = 9.81f;        // Gravity [m/s²]

    std::cout << "Buoyancy parameters:\n";
    std::cout << "  T_ref = " << T_ref << " K\n";
    std::cout << "  T_hot = " << T_hot << " K (center)\n";
    std::cout << "  T_cold = " << T_cold << " K (exterior)\n";
    std::cout << "  ΔT = " << (T_hot - T_ref) << " K\n";
    std::cout << "  β = " << beta << " 1/K\n";
    std::cout << "  g = " << g << " m/s²\n";
    std::cout << "  ρ₀ = " << rho0 << " kg/m³\n\n";

    // Expected buoyancy force magnitude
    float F_buoy_expected = rho0 * beta * (T_hot - T_ref) * g;
    std::cout << "Expected buoyancy force in hot region:\n";
    std::cout << "  F = ρ₀·β·(T-T_ref)·g = " << F_buoy_expected << " N/m³\n";
    std::cout << "  (or " << F_buoy_expected << " m/s² in lattice units)\n\n";

    // Run simulation
    std::cout << "Running 1000 time steps...\n\n";

    std::cout << std::setw(8) << "Step"
              << std::setw(15) << "u_max [m/s]"
              << std::setw(15) << "u_avg [m/s]"
              << std::setw(15) << "F_max [m/s²]"
              << "\n";
    std::cout << std::string(53, '-') << "\n";

    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];
    float* h_fx = new float[n_cells];
    float* h_fy = new float[n_cells];

    for (int step = 0; step <= 1000; step++) {
        // Compute buoyancy force (same as main simulation)
        solver.computeBuoyancyForce(
            d_temperature, T_ref, beta,
            0.0f, g, 0.0f,  // Gravity in +y direction
            d_fx, d_fy, d_fz
        );

        // NO Darcy damping (assume fully liquid for this test)

        // Solve Navier-Stokes (same order as main simulation)
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();
        solver.applyBoundaryConditions(1);  // Apply wall BC

        // Output every 100 steps
        if (step % 100 == 0) {
            solver.copyVelocityToHost(h_ux, h_uy, h_uz);
            cudaMemcpy(h_fx, d_fx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy, d_fy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float u_max = 0.0f, u_sum = 0.0f, f_max = 0.0f;
            for (int i = 0; i < n_cells; i++) {
                float u_mag = sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                float f_mag = sqrt(h_fx[i]*h_fx[i] + h_fy[i]*h_fy[i]);
                u_max = std::max(u_max, u_mag);
                u_sum += u_mag;
                f_max = std::max(f_max, f_mag);
            }
            float u_avg = u_sum / n_cells;

            std::cout << std::setw(8) << step
                      << std::setw(15) << std::scientific << std::setprecision(4)
                      << u_max
                      << std::setw(15) << u_avg
                      << std::setw(15) << f_max
                      << "\n";
        }
    }

    std::cout << "\n";

    // Final analysis
    solver.copyVelocityToHost(h_ux, h_uy, h_uz);
    cudaMemcpy(h_fx, d_fx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy, d_fy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float u_max = 0.0f;
    int zero_count = 0;
    float f_max = 0.0f;

    for (int i = 0; i < n_cells; i++) {
        float u_mag = sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        float f_mag = sqrt(h_fx[i]*h_fx[i] + h_fy[i]*h_fy[i]);

        u_max = std::max(u_max, u_mag);
        f_max = std::max(f_max, f_mag);

        if (u_mag < 1e-10f) zero_count++;
    }

    std::cout << "Final Results:\n";
    std::cout << "  Max velocity: " << u_max << " m/s\n";
    std::cout << "  Max force: " << f_max << " m/s²\n";
    std::cout << "  Expected force: " << F_buoy_expected << " m/s²\n";
    std::cout << "  Cells with zero velocity: " << zero_count << " / " << n_cells << "\n\n";

    std::cout << "Analysis:\n";
    if (f_max < 1e-6f) {
        std::cout << "  ❌ CRITICAL: Buoyancy force is ZERO!\n";
        std::cout << "  Problem in computeBuoyancyForce() implementation.\n";
    } else if (u_max < 1e-10f) {
        std::cout << "  ❌ FAIL: Force is computed but velocity is ZERO!\n";
        std::cout << "  Force is NOT being applied in collision step.\n";
        std::cout << "\n";
        std::cout << "  Possible causes:\n";
        std::cout << "  1. Force arrays not passed correctly to collisionBGK\n";
        std::cout << "  2. Boundary conditions overwriting interior velocities\n";
        std::cout << "  3. Bug in fluidBGKCollisionVaryingForceKernel\n";
    } else if (u_max < 1e-6f) {
        std::cout << "  ⚠️  WARNING: Velocity is very small (<1e-6 m/s)\n";
        std::cout << "  Force application may be weak or scaled incorrectly.\n";
    } else {
        std::cout << "  ✅ PASS: Buoyancy force produces non-zero velocity!\n";
        std::cout << "  System is working correctly.\n";
    }

    // Cleanup
    delete[] h_temperature;
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;
    delete[] h_fx;
    delete[] h_fy;

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "\n==============================================\n";
    std::cout << "  Test Complete\n";
    std::cout << "==============================================\n";

    return 0;
}
