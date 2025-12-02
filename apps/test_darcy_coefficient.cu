/**
 * @file test_darcy_coefficient.cu
 * @brief Test different Darcy coefficients to find optimal value
 *
 * This test runs short simulations with different darcy_coefficient values
 * to determine which gives acceptable divergence while maintaining solid stability.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"

using namespace lbm;

void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

void runTest(float darcy_coeff, const std::string& output_dir) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Testing darcy_coefficient = " << darcy_coeff << "\n";
    std::cout << "========================================\n\n";

    physics::MultiphysicsConfig config;

    // Domain (smaller for speed)
    config.nx = 50;
    config.ny = 50;
    config.nz = 25;
    config.dx = 2.0e-6f;
    config.dt = 1.0e-7f;

    // Physics modules
    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_darcy = true;
    config.enable_marangoni = true;
    config.enable_surface_tension = false;
    config.enable_laser = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;

    // Material
    config.material = physics::MaterialDatabase::getTi6Al4V();

    // Thermal
    config.thermal_diffusivity = 5.8e-6f;

    // Fluid
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;

    // DARCY COEFFICIENT - THIS IS WHAT WE'RE TESTING
    config.darcy_coefficient = darcy_coeff;

    // Surface tension
    config.surface_tension_coeff = 1.65f;
    config.dsigma_dT = -0.26e-3f;

    // Laser
    config.laser_power = 200.0f;
    config.laser_spot_radius = 50.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;

    // Boundaries
    config.boundary_type = 0;  // Periodic

    // Simulation parameters
    const int num_steps = 500;  // Just 500 steps for quick test
    const int output_interval = 100;

    // Initialize solver
    std::cout << "Initializing solver...\n";
    physics::MultiphysicsSolver solver(config);

    const float T_initial = 300.0f;
    const float interface_height = 0.5f;
    solver.initialize(T_initial, interface_height);

    // Set initial liquid fraction (all solid)
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_liquid_fraction(num_cells, 0.0f);

    float* d_lf;
    cudaMalloc(&d_lf, num_cells * sizeof(float));
    cudaMemcpy(d_lf, h_liquid_fraction.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticLiquidFraction(d_lf);
    cudaFree(d_lf);

    // Create output directory
    createDirectory(output_dir);

    // Allocate output arrays
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_phase(num_cells);

    std::cout << "Running simulation...\n";
    std::cout << "Step    T_max[K]   v_max[mm/s]\n";
    std::cout << "------------------------------\n";

    // Time integration
    for (int step = 0; step <= num_steps; ++step) {
        if (step % output_interval == 0) {
            // Get data
            const float* d_T = solver.getTemperature();
            const float* d_vx = solver.getVelocityX();
            const float* d_vy = solver.getVelocityY();
            const float* d_vz = solver.getVelocityZ();
            const float* d_f = solver.getFillLevel();

            cudaMemcpy(h_temperature.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ux.data(), d_vx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_vy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_vz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fill.data(), d_f, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            // Stats
            float T_max = *std::max_element(h_temperature.begin(), h_temperature.end());
            float v_max = 0.0f;
            for (size_t i = 0; i < num_cells; ++i) {
                float v = sqrtf(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                v_max = std::max(v_max, v);
            }

            std::cout << std::setw(4) << step
                      << std::setw(12) << std::fixed << std::setprecision(1) << T_max
                      << std::setw(14) << std::fixed << std::setprecision(3) << v_max * 1e3
                      << "\n";

            // Compute phase
            const float T_solidus = config.material.T_solidus;
            const float T_liquidus = config.material.T_liquidus;

            for (size_t i = 0; i < num_cells; ++i) {
                float T = h_temperature[i];
                if (T < T_solidus) h_phase[i] = 0.0f;
                else if (T > T_liquidus) h_phase[i] = 2.0f;
                else h_phase[i] = 1.0f;
            }

            // Write VTK
            std::string filename = io::VTKWriter::getTimeSeriesFilename(
                output_dir + "/test", step);

            io::VTKWriter::writeStructuredGridWithVectors(
                filename,
                h_temperature.data(),
                h_fill.data(),
                h_phase.data(),
                h_fill.data(),  // use fill as VOF fill_level
                h_ux.data(), h_uy.data(), h_uz.data(),
                config.nx, config.ny, config.nz,
                config.dx, config.dx, config.dx
            );
        }

        if (step < num_steps) {
            solver.step(config.dt);
        }
    }

    std::cout << "\nTest complete. Output: " << output_dir << "/\n";
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Darcy Coefficient Optimization Test              \n";
    std::cout << "====================================================\n";
    std::cout << "\nGoal: Find darcy_coefficient that minimizes divergence\n";
    std::cout << "      while keeping solid regions stable.\n";

    // Test different coefficients
    std::vector<float> coefficients = {
        1.0e7f,  // Current value (known to cause large divergence)
        1.0e6f,  // 10x reduction
        1.0e5f,  // 100x reduction
        5.0e5f,  // Middle value
    };

    for (float coeff : coefficients) {
        std::string output_dir = "darcy_test_" + std::to_string(int(coeff));
        runTest(coeff, output_dir);
    }

    std::cout << "\n";
    std::cout << "====================================================\n";
    std::cout << "All tests complete!                                \n";
    std::cout << "====================================================\n";
    std::cout << "\nNext step: Run check_velocity_divergence.py on each output:\n";
    std::cout << "  python3 scripts/check_velocity_divergence.py darcy_test_1000000/test_000500.vtk\n";
    std::cout << "  python3 scripts/check_velocity_divergence.py darcy_test_100000/test_000500.vtk\n";
    std::cout << "  ...\n";

    return 0;
}
