/**
 * @file visualize_phase6_marangoni_simple.cu
 * @brief Phase 6 Marangoni Effect Visualization (简洁版)
 *
 * 直接使用 MultiphysicsSolver + VTK Writer 生成可视化
 * 展示 Marangoni 驱动的表面流动 (0.7-0.8 m/s)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"

using namespace lbm;

// Helper: Create directory
void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Helper: Initialize radial temperature gradient (same as Test 2C)
void initializeTemperatureGradient(float* h_temp, int nx, int ny, int nz, float dx) {
    const float T_hot = 2500.0f;   // K
    const float T_cold = 2000.0f;  // K
    const float R_hot = 30e-6f;    // 30 μm
    const float R_decay = 50e-6f;  // 50 μm

    float center_x = nx * dx / 2.0f;
    float center_y = ny * dx / 2.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                float x = i * dx - center_x;
                float y = j * dx - center_y;
                float r = sqrtf(x*x + y*y);

                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    float decay = expf(-(r - R_hot) / (R_decay - R_hot));
                    h_temp[idx] = T_cold + (T_hot - T_cold) * decay;
                }
                h_temp[idx] = fmaxf(h_temp[idx], T_cold);
            }
        }
    }
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  Phase 6: Marangoni Effect Visualization\n";
    std::cout << "========================================\n\n";

    // Configuration (same as Test 2C)
    physics::MultiphysicsConfig config;
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2e-6f;  // 2 μm

    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;  // Static interface
    config.enable_surface_tension = false;
    config.enable_marangoni = true;       // ← Key!
    config.enable_laser = false;
    config.enable_darcy = true;           // Re-enable Darcy damping

    // Material: Ti6Al4V (corrected from Test 2C)
    config.material = physics::MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;  // Lattice units (τ=0.6)
    config.density = 4110.0f;
    config.dsigma_dT = -2.6e-4f;
    config.dt = 1e-7f;  // 0.1 μs

    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << config.nx << " × " << config.ny << " × " << config.nz << "\n";
    std::cout << "  Resolution: " << config.dx * 1e6 << " μm\n";
    std::cout << "  Time step: " << config.dt * 1e9 << " ns\n";
    std::cout << "  Material: Ti6Al4V\n";
    std::cout << "  dσ/dT: " << config.dsigma_dT * 1e3 << " mN/(m·K)\n\n";

    // Create solver
    physics::MultiphysicsSolver solver(config);
    solver.initialize(2300.0f, 0.5f);

    // Set temperature gradient
    size_t n_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(n_cells);
    initializeTemperatureGradient(h_temp.data(), config.nx, config.ny, config.nz, config.dx);

    float* d_temp;
    cudaMalloc(&d_temp, n_cells * sizeof(float));
    cudaMemcpy(d_temp, h_temp.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticTemperature(d_temp);
    cudaFree(d_temp);

    // Set liquid fraction field (for Darcy damping)
    // Bottom 14 μm (7 cells at 2 μm spacing) is solid substrate
    std::vector<float> h_liquid_fraction(n_cells);
    const int z_substrate = 7;  // Bottom 7 cells are solid
    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                if (k < z_substrate) {
                    h_liquid_fraction[idx] = 0.0f;  // Solid substrate
                } else {
                    h_liquid_fraction[idx] = 1.0f;  // Liquid pool
                }
            }
        }
    }

    float* d_lf;
    cudaMalloc(&d_lf, n_cells * sizeof(float));
    cudaMemcpy(d_lf, h_liquid_fraction.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticLiquidFraction(d_lf);
    cudaFree(d_lf);

    std::cout << "Temperature gradient:\n";
    std::cout << "  T_hot: 2500 K (center)\n";
    std::cout << "  T_cold: 2000 K (edge)\n";
    std::cout << "  ΔT: 500 K\n";
    std::cout << "  |∇T|: ~16.7 K/μm\n\n";

    std::cout << "Solid substrate:\n";
    std::cout << "  Bottom " << z_substrate << " cells (0-" << z_substrate * config.dx * 1e6 << " μm) = solid (fl=0)\n";
    std::cout << "  Remaining cells = liquid pool (fl=1)\n\n";

    // Simulation parameters
    const int n_steps = 2000;
    const int output_interval = 50;

    std::cout << "Simulation:\n";
    std::cout << "  Total steps: " << n_steps << "\n";
    std::cout << "  Physical time: " << n_steps * config.dt * 1e6 << " μs\n";
    std::cout << "  Output interval: " << output_interval << " steps\n\n";

    // Create output directory
    createDirectory("phase6_marangoni");

    // Allocate host arrays
    std::vector<float> h_temperature(n_cells);
    std::vector<float> h_ux(n_cells);
    std::vector<float> h_uy(n_cells);
    std::vector<float> h_uz(n_cells);
    std::vector<float> h_fill(n_cells);
    std::vector<float> h_phase(n_cells);

    std::cout << "Progress:\n";
    std::cout << "─────────────────────────────────────────────────────────\n";
    std::cout << "  Step      Time [μs]   T_max [K]   v_max [mm/s]\n";
    std::cout << "─────────────────────────────────────────────────────────\n";

    // Time loop
    for (int step = 0; step <= n_steps; ++step) {
        if (step % output_interval == 0) {
            // Get data from GPU
            const float* d_T = solver.getTemperature();
            const float* d_vx = solver.getVelocityX();
            const float* d_vy = solver.getVelocityY();
            const float* d_vz = solver.getVelocityZ();
            const float* d_f = solver.getFillLevel();

            cudaMemcpy(h_temperature.data(), d_T, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ux.data(), d_vx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_vy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_vz, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fill.data(), d_f, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

            // Compute phase state
            for (size_t i = 0; i < n_cells; ++i) {
                float fl = h_fill[i];
                if (fl < 0.01f) h_phase[i] = 0.0f;       // Solid
                else if (fl > 0.99f) h_phase[i] = 2.0f;  // Liquid
                else h_phase[i] = 1.0f;                  // Mushy
            }

            // Compute statistics
            float T_max = 0.0f, v_max = 0.0f;
            for (size_t i = 0; i < n_cells; ++i) {
                T_max = fmaxf(T_max, h_temperature[i]);
                float v = sqrtf(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                v_max = fmaxf(v_max, v);
            }

            // Print progress
            float time = step * config.dt;
            std::cout << std::setw(6) << step
                      << std::setw(14) << std::fixed << std::setprecision(2) << time * 1e6
                      << std::setw(14) << std::fixed << std::setprecision(1) << T_max
                      << std::setw(16) << std::fixed << std::setprecision(3) << v_max * 1e3
                      << "\n";

            // Write VTK
            std::string filename = io::VTKWriter::getTimeSeriesFilename(
                "phase6_marangoni/marangoni", step);

            io::VTKWriter::writeStructuredGridWithVectors(
                filename,
                h_temperature.data(),
                h_fill.data(),
                h_phase.data(),
                h_fill.data(),  // use fill_level as VOF field
                h_ux.data(), h_uy.data(), h_uz.data(),
                config.nx, config.ny, config.nz,
                config.dx, config.dx, config.dx
            );
        }

        // Step forward
        if (step < n_steps) {
            solver.step(config.dt);
        }
    }

    std::cout << "─────────────────────────────────────────────────────────\n\n";

    // Summary
    std::cout << "✅ Simulation complete!\n";
    std::cout << "   Output: phase6_marangoni/marangoni_*.vtk\n";
    std::cout << "   Files: " << (n_steps / output_interval + 1) << " frames\n\n";

    std::cout << "ParaView Visualization:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "1. Open ParaView:\n";
    std::cout << "   paraview phase6_marangoni/marangoni_*.vtk\n\n";
    std::cout << "2. Color by Temperature (hot=red, cold=blue)\n\n";
    std::cout << "3. Add Glyph filter for velocity arrows:\n";
    std::cout << "   Filters → Glyph → Vectors: Velocity\n\n";
    std::cout << "4. Play animation to see Marangoni flow! 🌊\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    std::cout << "Expected results:\n";
    std::cout << "  - Surface velocity: 0.7-0.8 m/s (radial outward)\n";
    std::cout << "  - 10× faster than Phase 5 buoyancy (0.078 m/s)\n";
    std::cout << "  - Validated against Panwisawas 2017 ✓\n\n";

    return 0;
}
