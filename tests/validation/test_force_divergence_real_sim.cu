/**
 * @file test_force_divergence_real_sim.cu
 * @brief Diagnostic tool to analyze force divergence in real LPBF simulation
 *
 * This program runs the realistic LPBF simulation for a few steps and
 * computes the divergence of each force component to identify which
 * force is creating compressibility.
 */

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace lbm::physics;

// Compute divergence using central differences
float computeMaxDivergence(const std::vector<float>& fx,
                          const std::vector<float>& fy,
                          const std::vector<float>& fz,
                          int nx, int ny, int nz,
                          float dx,
                          const std::string& label)
{
    float max_div = 0.0f;
    int count_high = 0;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = i + nx * (j + ny * k);

                // Central differences
                float dfx_dx = (fx[idx + 1] - fx[idx - 1]) / (2.0f * dx);
                float dfy_dy = (fy[idx + nx] - fy[idx - nx]) / (2.0f * dx);
                float dfz_dz = (fz[idx + nx*ny] - fz[idx - nx*ny]) / (2.0f * dx);

                float div = dfx_dx + dfy_dy + dfz_dz;

                if (std::abs(div) > max_div) {
                    max_div = std::abs(div);
                }
                if (std::abs(div) > 0.001f) {
                    count_high++;
                }
            }
        }
    }

    int total = (nx - 2) * (ny - 2) * (nz - 2);
    std::cout << label << ":\n";
    std::cout << "  Max |∇·F|: " << max_div << "\n";
    std::cout << "  Cells with |∇·F| > 0.001: " << count_high
              << " (" << 100.0f * count_high / total << "%)\n";

    return max_div;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "FORCE DIVERGENCE DIAGNOSTIC\n";
    std::cout << "========================================\n\n";

    // Domain setup (same as realistic simulation)
    const int nx = 100;
    const int ny = 100;
    const int nz = 50;
    const int num_cells = nx * ny * nz;
    const float dx = 2e-6f; // 2 μm

    // Create multiphysics solver
    MultiphysicsConfig config;
    config.nx = nx;
    config.ny = ny;
    config.nz = nz;
    config.dx = dx;
    config.dt = 0.1e-6f; // 0.1 μs

    // Material properties (Ti6Al4V)
    config.material = MaterialProperties::Ti6Al4V();
    config.density = config.material.density_liquid;
    config.kinematic_viscosity = config.material.viscosity_liquid / config.material.density_liquid;
    config.thermal_diffusivity = config.material.thermal_conductivity_liquid /
                                (config.material.density_liquid * config.material.specific_heat_liquid);

    // Enable physics
    config.enable_thermal = false; // Disable thermal for now
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_marangoni = true;
    config.enable_darcy = true;
    config.enable_surface_tension = false;
    config.enable_laser = false;

    // Marangoni parameters
    config.dsigma_dT = -0.26e-3f; // N/(m·K) for Ti6Al4V

    // Darcy damping
    config.darcy_coefficient = 1e8f; // kg/(m³·s)

    MultiphysicsSolver solver(config);

    // Initialize with interface at z = 0.5
    std::vector<float> h_temperature(num_cells, 300.0f);

    // Create temperature gradient to generate Marangoni force
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Hot center, cold edges (to create Marangoni flow)
                float x_norm = (i - nx/2.0f) / (nx/2.0f);
                float y_norm = (j - ny/2.0f) / (ny/2.0f);
                float r2 = x_norm*x_norm + y_norm*y_norm;
                h_temperature[idx] = 300.0f + 1700.0f * expf(-5.0f * r2);
            }
        }
    }

    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    solver.setStaticTemperature(d_temperature);
    solver.initialize(d_temperature, 0.5f);

    std::cout << "Simulation initialized\n";
    std::cout << "Running 10 steps...\n\n";

    // Run a few steps
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

    std::cout << "Simulation completed\n\n";

    // Get velocity field
    const float* d_ux = solver.getVelocityX();
    const float* d_uy = solver.getVelocityY();
    const float* d_uz = solver.getVelocityZ();

    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    cudaMemcpy(h_ux.data(), d_ux, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), d_uy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz.data(), d_uz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute velocity divergence
    std::cout << "========================================\n";
    std::cout << "VELOCITY DIVERGENCE\n";
    std::cout << "========================================\n";
    float max_vel_div = computeMaxDivergence(h_ux, h_uy, h_uz, nx, ny, nz, dx, "Velocity");
    std::cout << "\n";

    // Now check force divergence
    // We need to recompute forces to analyze them individually

    std::cout << "========================================\n";
    std::cout << "FORCE DIVERGENCE (INDIVIDUAL COMPONENTS)\n";
    std::cout << "========================================\n\n";

    std::cout << "Note: Need to add code to extract individual force components\n";
    std::cout << "      from MultiphysicsSolver for detailed analysis.\n\n";

    std::cout << "========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Maximum velocity divergence: " << max_vel_div << "\n";
    std::cout << "\n";

    if (max_vel_div > 1e-3f) {
        std::cout << "RESULT: FAIL ✗\n";
        std::cout << "Velocity field has large divergence!\n";
        std::cout << "This will cause VOF mass conservation errors.\n";
        return 1;
    } else {
        std::cout << "RESULT: PASS ✓\n";
        std::cout << "Velocity field is approximately divergence-free.\n";
        return 0;
    }
}
