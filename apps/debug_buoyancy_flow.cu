// Debug program to check buoyancy-driven flow
// This program tests if the fluid solver can respond to buoyancy forces

#include <iostream>
#include <cmath>
#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"
#include "physics/material_properties.h"

int main() {
    std::cout << "=== Buoyancy Flow Debug Test ===\n\n";

    // Small domain for quick testing
    int nx = 40, ny = 40, nz = 20;
    int num_cells = nx * ny * nz;
    float dx = 2.0e-6f;  // 2 um

    // LBM parameters
    float nu_lattice = 0.15f;
    float rho0 = 1.0f;

    // Physical parameters
    float beta_thermal = 9.0e-6f;  // Thermal expansion
    float g = 9.81f * 1e-3f;  // Scaled gravity
    float T_ref = 1900.0f;  // Reference temperature

    std::cout << "Test Setup:\n";
    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << "\n";
    std::cout << "  Kinematic viscosity: " << nu_lattice << "\n";
    std::cout << "  Beta: " << beta_thermal << " K⁻¹\n";
    std::cout << "  Gravity: " << g << " m/s²\n";
    std::cout << "  T_ref: " << T_ref << " K\n\n";

    // Initialize lattice
    core::D3Q19::initializeDevice();

    // Create fluid solver with periodic boundaries for simplicity
    physics::FluidLBM fluid(nx, ny, nz, nu_lattice, rho0,
                           physics::BoundaryType::PERIODIC,
                           physics::BoundaryType::PERIODIC,
                           physics::BoundaryType::PERIODIC);

    // Initialize with zero velocity
    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Create temperature field: hot bottom, cold top
    float* h_temperature = new float[num_cells];
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                // Linear temperature profile
                float z_frac = float(iz) / float(nz - 1);
                h_temperature[id] = T_ref - 200.0f + 400.0f * (1.0f - z_frac);  // Hot bottom, cold top
            }
        }
    }

    // Copy temperature to device
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate force arrays
    float* d_fx;
    float* d_fy;
    float* d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    std::cout << "Temperature field created (hot bottom, cold top)\n";
    std::cout << "  T_bottom ≈ " << h_temperature[0] << " K\n";
    std::cout << "  T_top ≈ " << h_temperature[num_cells - 1] << " K\n\n";

    std::cout << "Running simulation with buoyancy force...\n";
    std::cout << "Expected: Hot fluid rises (positive z velocity)\n\n";

    // Run for a few hundred steps
    int n_steps = 1000;
    for (int step = 0; step <= n_steps; ++step) {
        // Compute buoyancy force (gravity in +z direction)
        fluid.computeBuoyancyForce(
            d_temperature, T_ref, beta_thermal,
            0.0f, 0.0f, g,  // Gravity in +z direction
            d_fx, d_fy, d_fz
        );

        // Fluid step
        fluid.computeMacroscopic();
        fluid.collisionBGK(d_fx, d_fy, d_fz);
        fluid.streaming();
        fluid.applyBoundaryConditions(1);

        // Check velocity every 200 steps
        if (step % 200 == 0) {
            float* h_ux = new float[num_cells];
            float* h_uy = new float[num_cells];
            float* h_uz = new float[num_cells];

            fluid.copyVelocityToHost(h_ux, h_uy, h_uz);

            // Compute statistics
            float max_ux = 0.0f, max_uy = 0.0f, max_uz = 0.0f;
            float avg_uz = 0.0f;
            int valid_count = 0;

            for (int i = 0; i < num_cells; ++i) {
                if (!std::isnan(h_ux[i]) && !std::isnan(h_uy[i]) && !std::isnan(h_uz[i])) {
                    max_ux = std::max(max_ux, std::abs(h_ux[i]));
                    max_uy = std::max(max_uy, std::abs(h_uy[i]));
                    max_uz = std::max(max_uz, std::abs(h_uz[i]));
                    avg_uz += h_uz[i];
                    valid_count++;
                }
            }

            if (valid_count > 0) {
                avg_uz /= valid_count;
            }

            std::cout << "Step " << step << ":\n";
            std::cout << "  Max |ux|: " << max_ux << " m/s\n";
            std::cout << "  Max |uy|: " << max_uy << " m/s\n";
            std::cout << "  Max |uz|: " << max_uz << " m/s\n";
            std::cout << "  Avg uz: " << avg_uz << " m/s\n";

            // Check bottom center velocity (should be upward)
            int center_bottom = nx/2 + (ny/2) * nx + 0 * nx * ny;
            std::cout << "  Bottom center uz: " << h_uz[center_bottom] << " m/s\n\n";

            delete[] h_ux;
            delete[] h_uy;
            delete[] h_uz;
        }
    }

    std::cout << "\n=== Test Complete ===\n";
    std::cout << "If velocities remained zero, there's a problem with:\n";
    std::cout << "  1. Buoyancy force computation\n";
    std::cout << "  2. Force application in collision\n";
    std::cout << "  3. Boundary conditions blocking flow\n";

    // Cleanup
    delete[] h_temperature;
    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}
