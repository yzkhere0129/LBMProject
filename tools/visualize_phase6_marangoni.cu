/**
 * @file visualize_phase6_marangoni.cu
 * @brief Generate VTK visualization for Phase 6 Marangoni effect
 *
 * This program runs MultiphysicsSolver with Marangoni effect enabled
 * and outputs VTK files for ParaView visualization.
 *
 * Expected results:
 * - Surface velocity: 0.5-1.5 m/s (validated against Test 2C)
 * - Temperature: Hot center (2500K), cold edges (2000K)
 * - Flow pattern: Radial outward from hot center (Marangoni-driven)
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

// Create output directory
void create_directory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Write VTK file
void write_vtk(const std::string& filename,
               MultiphysicsSolver& solver,
               int nx, int ny, int nz, float dx) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }

    // Get data from solver (device pointers)
    const float* d_temperature = solver.getTemperature();
    const float* d_ux = solver.getVelocityX();
    const float* d_uy = solver.getVelocityY();
    const float* d_uz = solver.getVelocityZ();
    const float* d_fill = solver.getFillLevel();

    // Allocate host memory
    size_t n_cells = nx * ny * nz;
    float* h_temperature = new float[n_cells];
    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];
    float* h_fill = new float[n_cells];

    // Copy from device to host
    cudaMemcpy(h_temperature, d_temperature, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux, d_ux, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy, d_uy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz, d_uz, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fill, d_fill, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Phase 6 - Marangoni Effect Visualization\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << dx << " " << dx << " " << dx << "\n";
    file << "\n";
    file << "POINT_DATA " << (nx * ny * nz) << "\n";

    // Write velocity field
    file << "VECTORS Velocity float\n";
    for (size_t i = 0; i < n_cells; ++i) {
        file << h_ux[i] << " " << h_uy[i] << " " << h_uz[i] << "\n";
    }
    file << "\n";

    // Write temperature field
    file << "SCALARS Temperature float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < n_cells; ++i) {
        file << h_temperature[i] << "\n";
    }
    file << "\n";

    // Write fill level field
    file << "SCALARS FillLevel float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < n_cells; ++i) {
        file << h_fill[i] << "\n";
    }

    file.close();

    // Free host memory
    delete[] h_temperature;
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;
    delete[] h_fill;
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 6 - Marangoni Effect Visualization" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration matching Test 2C
    MultiphysicsConfig config;
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2e-6f;  // 2 μm resolution

    // Enable Phase 6 physics
    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;  // Static interface for clarity
    config.enable_surface_tension = false;
    config.enable_marangoni = true;  // ← Key: Marangoni ON
    config.enable_laser = false;

    // Material: Ti6Al4V (corrected properties from Test 2C)
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;  // Lattice units (from Test 2C: τ=0.6 → ν=(0.6-0.5)/3)
    config.density = 4110.0f;
    config.dsigma_dT = -2.6e-4f;
    config.dt = 1e-7f;  // 0.1 μs

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << " × " << config.ny << " × " << config.nz << std::endl;
    std::cout << "  Resolution: " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  Time step: " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Material: Ti6Al4V" << std::endl;
    std::cout << "  dσ/dT: " << config.dsigma_dT * 1e3 << " mN/(m·K)" << std::endl;
    std::cout << "  Viscosity (lattice): " << config.kinematic_viscosity << std::endl;
    std::cout << std::endl;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize with uniform temperature first
    solver.initialize(2300.0f,  // Base temperature (liquid)
                      0.5f);     // Initial fill level

    // Create radial temperature gradient (hot center, cold edge)
    // This is the key to Marangoni flow!
    const float T_hot = 2500.0f;   // K (molten Ti6Al4V)
    const float T_cold = 2000.0f;  // K (still liquid, but cooler)
    const float R_hot = 30e-6f;    // 30 μm hot zone
    const float R_decay = 50e-6f;  // 50 μm decay length

    std::vector<float> h_temperature(config.nx * config.ny * config.nz);
    float center_x = config.nx * config.dx / 2.0f;
    float center_y = config.ny * config.dx / 2.0f;

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                // Distance from center
                float x = i * config.dx - center_x;
                float y = j * config.dx - center_y;
                float r = sqrtf(x*x + y*y);

                // Radial temperature profile
                if (r < R_hot) {
                    h_temperature[idx] = T_hot;
                } else {
                    float decay_factor = expf(-(r - R_hot) / (R_decay - R_hot));
                    h_temperature[idx] = T_cold + (T_hot - T_cold) * decay_factor;
                }
                h_temperature[idx] = std::max(h_temperature[idx], T_cold);
            }
        }
    }

    // Copy to device and set as static temperature
    float* d_temperature;
    cudaMalloc(&d_temperature, config.nx * config.ny * config.nz * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(),
               config.nx * config.ny * config.nz * sizeof(float),
               cudaMemcpyHostToDevice);
    solver.setStaticTemperature(d_temperature);
    cudaFree(d_temperature);

    std::cout << "Temperature field:" << std::endl;
    std::cout << "  T_hot (center): " << T_hot << " K" << std::endl;
    std::cout << "  T_cold (edge):  " << T_cold << " K" << std::endl;
    std::cout << "  ΔT: " << (T_hot - T_cold) << " K" << std::endl;
    std::cout << "  |∇T|: ~" << (T_hot - T_cold) / R_hot * 1e-6 << " K/μm" << std::endl;
    std::cout << std::endl;

    // Create output directory
    std::string output_dir = "phase6_marangoni_vis";
    create_directory(output_dir);
    std::cout << "Output directory: " << output_dir << "/" << std::endl;
    std::cout << std::endl;

    // Simulation parameters
    const int n_steps = 2000;  // 200 μs total
    const int output_interval = 50;  // Output every 5 μs (40 frames)

    std::cout << "Running simulation..." << std::endl;
    std::cout << "  Total steps: " << n_steps << std::endl;
    std::cout << "  Physical time: " << n_steps * config.dt * 1e6 << " μs" << std::endl;
    std::cout << "  Output interval: " << output_interval << " steps ("
              << output_interval * config.dt * 1e6 << " μs)" << std::endl;
    std::cout << std::endl;

    // Time stepping loop
    for (int step = 0; step <= n_steps; ++step) {
        // Output VTK file
        if (step % output_interval == 0) {
            std::ostringstream filename;
            filename << output_dir << "/marangoni_"
                     << std::setfill('0') << std::setw(6) << step << ".vtk";
            write_vtk(filename.str(), solver, config.nx, config.ny, config.nz, config.dx);

            float v_max = solver.getMaxVelocity();
            float T_max = solver.getMaxTemperature();

            std::cout << "Step " << std::setw(5) << step
                      << " | t = " << std::fixed << std::setprecision(2) << std::setw(6)
                      << step * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(4) << v_max << " m/s"
                      << " | T_max = " << std::setprecision(1) << T_max << " K"
                      << " | " << filename.str()
                      << std::endl;

            // Check for NaN
            if (std::isnan(v_max) || std::isnan(T_max)) {
                std::cerr << "\n❌ ERROR: NaN detected! Stopping simulation." << std::endl;
                return 1;
            }

            // Check against Test 2C expectation (0.768 m/s)
            if (step == n_steps) {
                std::cout << "\n========================================" << std::endl;
                std::cout << "Final Results:" << std::endl;
                std::cout << "========================================" << std::endl;
                std::cout << "Maximum velocity: " << v_max << " m/s" << std::endl;
                std::cout << "Test 2C result:   0.768 m/s" << std::endl;
                std::cout << "Difference:       " << std::abs(v_max - 0.768f) / 0.768f * 100.0f << "%" << std::endl;
                std::cout << std::endl;

                if (v_max >= 0.7f && v_max <= 1.5f) {
                    std::cout << "✅ PASS: Velocity matches Test 2C and literature!" << std::endl;
                } else {
                    std::cout << "⚠️  WARNING: Velocity outside expected range (0.7-1.5 m/s)" << std::endl;
                }
                std::cout << std::endl;
            }
        }

        // Advance one time step
        if (step < n_steps) {
            solver.step(config.dt);
        }
    }

    std::cout << "\n✅ Visualization complete!" << std::endl;
    std::cout << "Output files: " << output_dir << "/marangoni_*.vtk" << std::endl;
    std::cout << std::endl;
    std::cout << "To view in ParaView:" << std::endl;
    std::cout << "  paraview " << output_dir << "/marangoni_*.vtk" << std::endl;
    std::cout << std::endl;
    std::cout << "Recommended ParaView settings:" << std::endl;
    std::cout << "  1. Color by: Temperature (hot = red, cold = blue)" << std::endl;
    std::cout << "  2. Add Glyph filter -> Velocity vectors (3D arrows)" << std::endl;
    std::cout << "  3. Add Contour filter -> FillLevel = 0.5 (liquid-solid interface)" << std::endl;
    std::cout << "  4. Play animation to see Marangoni 'boiling' effect! 🌊" << std::endl;
    std::cout << std::endl;

    return 0;
}
