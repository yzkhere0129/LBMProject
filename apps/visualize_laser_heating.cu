/**
 * @file visualize_laser_heating.cu
 * @brief Laser heating visualization demonstration program
 *
 * This program simulates laser heating of metal and outputs VTK files
 * for visualization in ParaView. It demonstrates:
 * - Temperature field evolution
 * - Laser spot movement
 * - Heat diffusion in metal
 *
 * Usage:
 *   ./visualize_laser_heating
 *   Then open generated .vtk files in ParaView
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"

using namespace lbm;
using namespace lbm::physics;

/**
 * @brief Print simulation progress and statistics
 */
void printProgress(int step, int total_steps, float time,
                  float T_max, float T_avg, float T_min) {
    float progress = 100.0f * step / total_steps;
    std::cout << "\r[" << std::setw(3) << static_cast<int>(progress) << "%] "
              << "Step " << step << "/" << total_steps
              << " | t=" << std::scientific << std::setprecision(3) << time << "s"
              << " | T: [" << std::fixed << std::setprecision(1)
              << T_min << ", " << T_avg << ", " << T_max << "] K"
              << std::flush;
}

int main(int argc, char** argv) {
    std::cout << "\n========================================\n";
    std::cout << "   Laser Heating Visualization Demo    \n";
    std::cout << "========================================\n\n";

    // ========== Simulation Parameters ==========
    const int nx = 100, ny = 100, nz = 50;  // Grid size
    const float dx = 2e-6f;  // 2 micrometers grid spacing
    const float dy = 2e-6f;
    const float dz = 2e-6f;
    const float dt = 5e-10f;  // 0.5 nanosecond time step

    const int n_steps = 16000;       // Total simulation steps (longer for melting)
    const int output_interval = 200; // Output every 200 steps

    // Domain size in physical units
    float Lx = nx * dx;
    float Ly = ny * dy;
    float Lz = nz * dz;

    std::cout << "Simulation Setup:\n";
    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Domain: " << Lx * 1e6 << " x " << Ly * 1e6 << " x "
              << Lz * 1e6 << " micrometers\n";
    std::cout << "  Grid spacing: " << dx * 1e6 << " um\n";
    std::cout << "  Time step: " << dt * 1e9 << " ns\n";
    std::cout << "  Total time: " << n_steps * dt * 1e6 << " microseconds\n";
    std::cout << "  Output interval: every " << output_interval << " steps\n\n";

    // ========== Material Setup (Ti6Al4V) ==========
    physics::MaterialProperties ti64 = physics::MaterialDatabase::getTi6Al4V();

    std::cout << "Material: Ti6Al4V\n";
    std::cout << "  Density: " << ti64.rho_solid << " kg/m^3\n";
    std::cout << "  Melting point: " << ti64.T_liquidus << " K\n";
    std::cout << "  Thermal conductivity: " << ti64.k_solid << " W/(m·K)\n";
    std::cout << "  Thermal diffusivity: "
              << ti64.getThermalDiffusivity(300.0f) * 1e6 << " mm^2/s\n";
    std::cout << "  Laser absorptivity: " << ti64.absorptivity_solid << "\n\n";

    // ========== Initialize Thermal LBM with Phase Change ==========
    float thermal_diffusivity = ti64.getThermalDiffusivity(300.0f);

    // Use new constructor with phase change support
    // CRITICAL FIX: Pass dt and dx for proper tau scaling
    physics::ThermalLBM thermal(nx, ny, nz, ti64, thermal_diffusivity, true, dt, dx);

    // Set initial temperature (room temperature)
    float T_initial = 300.0f;  // 300 K
    thermal.initialize(T_initial);

    std::cout << "Thermal LBM initialized with phase change support\n";
    std::cout << "  Initial temperature: " << T_initial << " K\n";
    std::cout << "  Phase change enabled: " << (thermal.hasPhaseChange() ? "Yes" : "No") << "\n\n";

    // ========== Laser Setup ==========
    // 可调参数 - 根据需要修改这些值
    float laser_power = 1500.0f;       // 激光功率 [W] (1500W for larger melt pool - easier to visualize)
    float spot_radius = 50e-6f;        // 光斑半径 [m] (50微米)
    float absorptivity = ti64.absorptivity_solid;
    float penetration_depth = 15e-6f;  // 穿透深度 [m] (15微米)

    LaserSource laser(laser_power, spot_radius,
                     absorptivity, penetration_depth);

    // Initial laser position (center of top surface)
    float laser_x = Lx / 2.0f;
    float laser_y = Ly / 2.0f;
    float laser_z = 0.0f;  // Top surface
    laser.setPosition(laser_x, laser_y, laser_z);

    std::cout << "Laser Parameters:\n";
    std::cout << "  Power: " << laser_power << " W\n";
    std::cout << "  Spot radius: " << spot_radius * 1e6 << " um\n";
    std::cout << "  Penetration depth: " << penetration_depth * 1e6 << " um\n";
    std::cout << "  Initial position: ("
              << laser_x * 1e6 << ", " << laser_y * 1e6 << ", "
              << laser_z * 1e6 << ") um\n\n";

    // ========== Allocate GPU Memory ==========
    size_t field_size = nx * ny * nz * sizeof(float);
    float* d_heat_source;
    cudaMalloc(&d_heat_source, field_size);
    cudaMemset(d_heat_source, 0, field_size);

    // Host memory for temperature, liquid fraction, and phase state
    float* h_temperature = new float[nx * ny * nz];
    float* h_liquid_fraction = new float[nx * ny * nz];
    float* h_phase_state = new float[nx * ny * nz];

    // ========== Create Output Directory ==========
    system("mkdir -p visualization_output");
    std::cout << "Output directory: visualization_output/\n\n";

    // ========== Time Evolution Loop ==========
    std::cout << "Starting simulation...\n";
    std::cout << "Progress:\n";

    for (int step = 0; step <= n_steps; ++step) {
        float time = step * dt;

        // 激光运动模式选择 - 可以在这里切换不同模式

        // 模式1: 静止激光（观察热扩散）- 当前使用
        // 激光保持在中心位置不动
        // (无需额外代码，激光已在初始化时设置在中心)

        /* 模式2: 圆周运动（取消注释下面代码使用）
        if (step > 0) {
            float omega = 2.0f * M_PI * 100.0f;  // 100 Hz rotation
            float radius = 30e-6f;  // 30 um radius
            laser_x = Lx / 2.0f + radius * cosf(omega * time);
            laser_y = Ly / 2.0f + radius * sinf(omega * time);
            laser.setPosition(laser_x, laser_y, laser_z);
        }
        */

        /* 模式3: 直线扫描（取消注释下面代码使用）
        if (step > 0) {
            float scan_speed = 0.05f;  // 0.05 m/s
            laser_x = 20e-6f + scan_speed * time;
            laser_y = Ly / 2.0f;
            if (laser_x > Lx - 20e-6f) laser_x = 20e-6f;  // 边界重置
            laser.setPosition(laser_x, laser_y, laser_z);
        }
        */

        // Compute laser heat source
        dim3 block(8, 8, 8);
        dim3 grid((nx + block.x - 1) / block.x,
                 (ny + block.y - 1) / block.y,
                 (nz + block.z - 1) / block.z);

        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        // Add heat source to thermal field
        thermal.addHeatSource(d_heat_source, dt);

        // Evolve thermal field one time step (collision + streaming)
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Output VTK files at specified intervals
        if (step % output_interval == 0) {
            // Copy temperature and liquid fraction to host
            thermal.copyTemperatureToHost(h_temperature);
            thermal.copyLiquidFractionToHost(h_liquid_fraction);

            // Calculate statistics
            float T_max = h_temperature[0];
            float T_min = h_temperature[0];
            float T_sum = 0.0f;
            int n_melting = 0;

            // Compute phase state field (0=solid, 1=mushy, 2=liquid)
            for (int i = 0; i < nx * ny * nz; ++i) {
                float T = h_temperature[i];
                float fl = h_liquid_fraction[i];

                T_max = fmaxf(T_max, T);
                T_min = fminf(T_min, T);
                T_sum += T;

                // Count melting cells
                if (fl > 0.01f) n_melting++;

                // Determine phase state
                if (fl < 0.01f) {
                    h_phase_state[i] = 0.0f;  // Solid
                } else if (fl > 0.99f) {
                    h_phase_state[i] = 2.0f;  // Liquid
                } else {
                    h_phase_state[i] = 1.0f;  // Mushy
                }
            }
            float T_avg = T_sum / (nx * ny * nz);
            float melt_pct = 100.0f * n_melting / (nx * ny * nz);

            // Show progress with melting info
            std::cout << "\r[" << std::setw(3) << static_cast<int>(100.0f * step / n_steps) << "%] "
                      << "Step " << step << "/" << n_steps
                      << " | t=" << std::scientific << std::setprecision(3) << time << "s"
                      << " | T: [" << std::fixed << std::setprecision(1)
                      << T_min << ", " << T_avg << ", " << T_max << "] K"
                      << " | Melting: " << n_melting << " cells ("
                      << std::fixed << std::setprecision(2) << melt_pct << "%)"
                      << std::flush;

            // Generate filename
            std::string filename = io::VTKWriter::getTimeSeriesFilename(
                "visualization_output/laser_heating", step
            );

            // Write full 3D snapshot with all fields
            io::VTKWriter::writeStructuredGrid(
                filename,
                h_temperature, h_liquid_fraction, h_phase_state,
                nx, ny, nz, dx, dy, dz,
                "Temperature", "LiquidFraction", "PhaseState"
            );

            // Also write a 2D slice at the surface
            std::string slice_filename = io::VTKWriter::getTimeSeriesFilename(
                "visualization_output/surface_temp", step
            );
            io::VTKWriter::write2DSlice(
                slice_filename, h_temperature,
                nx, ny, nz,
                2, 0,  // z-slice at index 0 (surface)
                dx, dy, dz,
                "Temperature"
            );
        }
    }

    std::cout << "\n\n";

    // ========== Cleanup ==========
    delete[] h_temperature;
    delete[] h_liquid_fraction;
    delete[] h_phase_state;
    cudaFree(d_heat_source);

    // ========== Final Output Summary ==========
    int n_files = (n_steps / output_interval) + 1;

    std::cout << "\n========================================\n";
    std::cout << "         Simulation Complete!           \n";
    std::cout << "========================================\n\n";

    std::cout << "Generated " << n_files << " VTK file pairs:\n";
    std::cout << "  - laser_heating_*.vtk (3D volume data with phase change)\n";
    std::cout << "  - surface_temp_*.vtk (2D surface slices)\n\n";

    std::cout << "VTK files include the following fields:\n";
    std::cout << "  - Temperature: Temperature field in Kelvin\n";
    std::cout << "  - LiquidFraction: Liquid fraction (0-1)\n";
    std::cout << "  - PhaseState: Phase indicator (0=solid, 1=mushy, 2=liquid)\n\n";

    std::cout << "To visualize the results:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "1. Install ParaView:\n";
    std::cout << "   wget -O ParaView.tar.gz \"https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.11&type=binary&os=Linux&downloadFile=ParaView-5.11.2-MPI-Linux-Python3.9-x86_64.tar.gz\"\n";
    std::cout << "   tar -xzf ParaView.tar.gz\n\n";

    std::cout << "2. Open ParaView:\n";
    std::cout << "   ./ParaView-*/bin/paraview\n\n";

    std::cout << "3. Load the data:\n";
    std::cout << "   File → Open → visualization_output/laser_heating_*.vtk\n";
    std::cout << "   Click 'Apply'\n\n";

    std::cout << "4. Visualize different fields:\n";
    std::cout << "   - Color by 'Temperature' to see hot spot\n";
    std::cout << "   - Color by 'LiquidFraction' to see melting region (0-1)\n";
    std::cout << "   - Color by 'PhaseState' to see solid/mushy/liquid zones\n";
    std::cout << "   - Add slice filter: Filters → Common → Slice\n";
    std::cout << "   - Click play button ▶ to animate\n\n";

    std::cout << "5. Tips:\n";
    std::cout << "   - Use threshold filter to isolate melting regions (LiquidFraction > 0.01)\n";
    std::cout << "   - Adjust color range for better visualization\n";
    std::cout << "   - Try different color maps (Rainbow, Cool to Warm, etc.)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    return 0;
}