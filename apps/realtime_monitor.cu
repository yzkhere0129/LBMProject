/**
 * @file realtime_monitor.cu
 * @brief Real-time terminal visualization of laser heating
 *
 * Provides ASCII-art visualization of temperature fields in the terminal.
 * Useful for quick monitoring during development and debugging.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <unistd.h>  // for usleep

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"

using namespace lbm;

// ANSI color codes for temperature visualization
namespace colors {
    const char* BLUE   = "\033[34m";   // Cold
    const char* CYAN   = "\033[36m";   // Cool
    const char* GREEN  = "\033[32m";   // Warm
    const char* YELLOW = "\033[33m";   // Hot
    const char* RED    = "\033[31m";   // Very hot
    const char* MAGENTA= "\033[35m";   // Extremely hot
    const char* WHITE  = "\033[97m";   // Melting
    const char* RESET  = "\033[0m";    // Reset color
    const char* CLEAR  = "\033[2J\033[H"; // Clear screen

    const char* BOLD   = "\033[1m";
    const char* DIM    = "\033[2m";
}

/**
 * @brief Map temperature to color and character
 */
struct TempVisual {
    const char* color;
    char symbol;
};

TempVisual mapTemperatureToVisual(float T, float T_min, float T_max) {
    float normalized = (T - T_min) / (T_max - T_min + 1e-6f);
    normalized = fmaxf(0.0f, fminf(1.0f, normalized));

    TempVisual vis;

    if (normalized < 0.14f) {
        vis.color = colors::BLUE;
        vis.symbol = '.';
    } else if (normalized < 0.28f) {
        vis.color = colors::CYAN;
        vis.symbol = ':';
    } else if (normalized < 0.42f) {
        vis.color = colors::GREEN;
        vis.symbol = '=';
    } else if (normalized < 0.56f) {
        vis.color = colors::YELLOW;
        vis.symbol = '+';
    } else if (normalized < 0.70f) {
        vis.color = colors::RED;
        vis.symbol = '#';
    } else if (normalized < 0.85f) {
        vis.color = colors::MAGENTA;
        vis.symbol = '@';
    } else {
        vis.color = colors::WHITE;
        vis.symbol = '*';
    }

    return vis;
}

/**
 * @brief Print 2D temperature slice with ASCII art
 */
void printTemperatureSlice(const float* temperature, int nx, int ny, int nz,
                          int z_slice, float T_min, float T_max,
                          float laser_x_idx, float laser_y_idx, float spot_radius_cells) {
    // Clear screen
    std::cout << colors::CLEAR;

    // Header
    std::cout << colors::BOLD << "╔" << std::string(nx + 20, '═') << "╗\n";
    std::cout << "║ Temperature Field (z=" << z_slice << "/" << nz - 1 << ") ";
    std::cout << std::string(nx - 20, ' ') << "║\n";
    std::cout << "╠" << std::string(nx + 20, '═') << "╣" << colors::RESET << "\n";

    // Temperature field
    for (int j = ny - 1; j >= 0; --j) {
        std::cout << "║";
        for (int i = 0; i < nx; ++i) {
            int idx = i + j * nx + z_slice * nx * ny;
            float T = temperature[idx];

            // Check if laser position
            float dist = sqrtf((i - laser_x_idx) * (i - laser_x_idx) +
                             (j - laser_y_idx) * (j - laser_y_idx));

            TempVisual vis = mapTemperatureToVisual(T, T_min, T_max);

            // Mark laser center
            if (dist < 0.5f && z_slice == 0) {
                std::cout << colors::BOLD << colors::WHITE << "◉" << colors::RESET;
            }
            // Mark laser spot boundary
            else if (fabs(dist - spot_radius_cells) < 0.5f && z_slice == 0) {
                std::cout << colors::DIM << colors::WHITE << "○" << colors::RESET;
            }
            // Temperature visualization
            else {
                std::cout << vis.color << vis.symbol << colors::RESET;
            }
        }
        std::cout << "║\n";
    }

    // Footer
    std::cout << colors::BOLD << "╚" << std::string(nx + 20, '═') << "╝" << colors::RESET << "\n";
}

/**
 * @brief Print simulation statistics
 */
void printStatistics(int step, int total_steps, float time,
                    float T_max, float T_avg, float T_min,
                    float melt_fraction, float laser_power) {
    float progress = 100.0f * step / total_steps;

    std::cout << colors::BOLD << "\n── Statistics ";
    std::cout << std::string(50, '─') << colors::RESET << "\n";

    std::cout << "Progress: [";
    int bar_length = 30;
    int filled = static_cast<int>(bar_length * progress / 100.0f);
    for (int i = 0; i < bar_length; ++i) {
        if (i < filled) {
            std::cout << colors::GREEN << "█";
        } else {
            std::cout << colors::DIM << "░";
        }
    }
    std::cout << colors::RESET << "] " << std::fixed << std::setprecision(1)
              << progress << "%\n";

    std::cout << "Time:     " << std::scientific << std::setprecision(3)
              << time << " s (" << time * 1e6 << " µs)\n";

    std::cout << "Temperature:\n";
    std::cout << "  Min:    " << colors::BLUE << std::fixed << std::setprecision(1)
              << T_min << " K" << colors::RESET << "\n";
    std::cout << "  Avg:    " << colors::GREEN << T_avg << " K" << colors::RESET << "\n";
    std::cout << "  Max:    " << colors::RED << T_max << " K" << colors::RESET << "\n";

    if (melt_fraction > 0.001f) {
        std::cout << "  Melted: " << colors::WHITE << colors::BOLD
                  << std::setprecision(2) << melt_fraction * 100.0f << "%"
                  << colors::RESET << "\n";
    }

    std::cout << "Laser:    " << laser_power << " W\n";
}

/**
 * @brief Print color legend
 */
void printLegend() {
    std::cout << "\nLegend: ";
    std::cout << colors::BLUE << "." << colors::RESET << " cold  ";
    std::cout << colors::CYAN << ":" << colors::RESET << " cool  ";
    std::cout << colors::GREEN << "=" << colors::RESET << " warm  ";
    std::cout << colors::YELLOW << "+" << colors::RESET << " hot  ";
    std::cout << colors::RED << "#" << colors::RESET << " very hot  ";
    std::cout << colors::MAGENTA << "@" << colors::RESET << " extreme  ";
    std::cout << colors::WHITE << "*" << colors::RESET << " melting\n";

    std::cout << "        ";
    std::cout << colors::WHITE << "◉" << colors::RESET << " laser center  ";
    std::cout << colors::DIM << colors::WHITE << "○" << colors::RESET << " laser boundary\n";
}

int main(int argc, char** argv) {
    // Smaller grid for real-time visualization
    const int nx = 60, ny = 60, nz = 20;
    const float dx = 3e-6f;  // 3 micrometers
    const float dy = 3e-6f;
    const float dz = 3e-6f;
    const float dt = 1e-9f;   // 1 nanosecond

    const int n_steps = 500;
    const int display_interval = 5;  // Update display every 5 steps

    // Material (Ti6Al4V)
    physics::MaterialProperties ti64 = physics::MaterialDatabase::getTi6Al4V();
    float thermal_diffusivity = ti64.getThermalDiffusivity(300.0f);

    // Initialize thermal LBM
    physics::ThermalLBM thermal(nx, ny, nz, thermal_diffusivity,
                               ti64.rho_solid, ti64.cp_solid);
    thermal.initialize(300.0f);

    // Laser setup
    float laser_power = 150.0f;  // 150 W
    float spot_radius = 45e-6f;  // 45 micrometers
    LaserSource laser(laser_power, spot_radius,
                     ti64.absorptivity_solid, 10e-6f);

    // Center position
    float laser_x = nx * dx / 2.0f;
    float laser_y = ny * dy / 2.0f;
    laser.setPosition(laser_x, laser_y, 0.0f);

    // Laser position in grid cells
    float laser_x_idx = laser_x / dx;
    float laser_y_idx = laser_y / dy;
    float spot_radius_cells = spot_radius / dx;

    // GPU memory
    float* d_heat_source;
    cudaMalloc(&d_heat_source, nx * ny * nz * sizeof(float));

    // Host memory
    float* h_temperature = new float[nx * ny * nz];

    // Kernel configuration
    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    // Initial display
    std::cout << colors::CLEAR;
    std::cout << colors::BOLD << "═══ Real-Time Laser Heating Monitor ═══\n" << colors::RESET;
    std::cout << "Domain: " << nx << "×" << ny << "×" << nz
              << " (" << nx * dx * 1e6 << "×" << ny * dy * 1e6 << "×"
              << nz * dz * 1e6 << " µm³)\n";
    std::cout << "Material: Ti6Al4V (T_melt = " << ti64.T_liquidus << " K)\n";
    std::cout << "Laser: " << laser_power << " W, " << spot_radius * 1e6 << " µm spot\n";
    std::cout << "\nPress Ctrl+C to stop...\n";
    sleep(2);

    // Simulation loop
    for (int step = 0; step <= n_steps; ++step) {
        float time = step * dt;

        // Optional: Move laser in pattern
        if (step > n_steps / 3) {
            float omega = 2.0f * M_PI * 50.0f;  // 50 Hz
            float radius = 20e-6f;  // 20 µm radius
            laser_x = nx * dx / 2.0f + radius * cosf(omega * time);
            laser_y = ny * dy / 2.0f + radius * sinf(omega * time);
            laser.setPosition(laser_x, laser_y, 0.0f);

            laser_x_idx = laser_x / dx;
            laser_y_idx = laser_y / dy;
        }

        // Compute heat source
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );

        // Add heat and evolve
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Display update
        if (step % display_interval == 0) {
            // Get temperature field
            thermal.copyTemperatureToHost(h_temperature);

            // Calculate statistics
            float T_max = 300.0f, T_min = 1e10f, T_sum = 0.0f;
            float melt_count = 0.0f;

            for (int i = 0; i < nx * ny * nz; ++i) {
                float T = h_temperature[i];
                T_max = fmaxf(T_max, T);
                T_min = fminf(T_min, T);
                T_sum += T;
                if (T > ti64.T_liquidus) {
                    melt_count += 1.0f;
                }
            }

            float T_avg = T_sum / (nx * ny * nz);
            float melt_fraction = melt_count / (nx * ny * nz);

            // Display temperature field
            printTemperatureSlice(h_temperature, nx, ny, nz,
                                0,  // Surface slice
                                T_min, fminf(T_max, 800.0f),
                                laser_x_idx, laser_y_idx, spot_radius_cells);

            // Display statistics
            printStatistics(step, n_steps, time,
                          T_max, T_avg, T_min,
                          melt_fraction, laser_power);

            // Display legend
            printLegend();

            // Control frame rate
            usleep(50000);  // 50ms delay for animation effect
        }
    }

    // Cleanup
    delete[] h_temperature;
    cudaFree(d_heat_source);

    std::cout << "\n\nSimulation complete!\n";

    return 0;
}