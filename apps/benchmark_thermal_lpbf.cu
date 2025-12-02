/**
 * @file benchmark_thermal_lpbf.cu
 * @brief Performance benchmark for thermal LPBF simulation
 *
 * This application measures MLUPS (Million Lattice Updates Per Second)
 * and accuracy metrics for comparison with waLBerla and other frameworks.
 *
 * Benchmark Modes:
 *   Mode A: Pure thermal diffusion (fair comparison with FD solvers)
 *   Mode B: Thermal + phase change (LBM feature)
 *   Mode C: Full multiphysics (CUDA showcase)
 *
 * Usage:
 *   ./benchmark_thermal_lpbf [options]
 *   --mode <A|B|C>     Benchmark mode (default: A)
 *   --size <S|M|L>     Domain size: Small/Medium/Large (default: M)
 *   --steps <N>        Number of time steps (default: 10000)
 *   --output           Enable VTK output (default: off for benchmarking)
 *   --warmup <N>       Warmup steps (default: 100)
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"
#include "utils/benchmark_timer.h"

using namespace lbm;

// ============================================================================
// Configuration structures
// ============================================================================

struct BenchmarkConfig {
    // Domain
    int nx, ny, nz;
    float dx;       // Grid spacing [m]

    // Time
    float dt;       // Time step [s]
    int num_steps;
    int warmup_steps;

    // Physics mode
    char mode;      // 'A', 'B', or 'C'
    bool enable_phase_change;
    bool enable_laser_scan;
    bool enable_fluid;

    // Laser
    float laser_power;
    float laser_radius;
    float laser_absorptivity;

    // Material (Ti-6Al-4V)
    float density;
    float specific_heat;
    float conductivity;
    float T_melt;
    float T_ambient;

    // Output
    bool output_vtk;
    int output_interval;
    std::string output_dir;

    // Probe points (in physical coordinates [m])
    struct Probe {
        float x, y, z;
        std::string name;
    };
    std::vector<Probe> probes;

    BenchmarkConfig() {
        // Default: Medium size, Mode A
        nx = 200; ny = 200; nz = 100;
        dx = 2.0e-6f;  // 2 um

        dt = 1.0e-7f;  // 0.1 us
        num_steps = 10000;
        warmup_steps = 100;

        mode = 'A';
        enable_phase_change = false;
        enable_laser_scan = false;
        enable_fluid = false;

        laser_power = 200.0f;
        laser_radius = 50.0e-6f;
        laser_absorptivity = 0.35f;

        // Ti-6Al-4V
        density = 4430.0f;
        specific_heat = 526.0f;
        conductivity = 6.7f;
        T_melt = 1923.0f;
        T_ambient = 300.0f;

        output_vtk = false;
        output_interval = 100;
        output_dir = "benchmark_output";

        // Default probe points
        float cx = nx * dx / 2.0f;  // Center x
        float cy = ny * dx / 2.0f;  // Center y
        float surface_z = nz * dx * 0.8f;  // Surface at 80% height

        probes = {
            {cx, cy, surface_z, "center_surface"},
            {cx + 50e-6f, cy, surface_z, "offset_50um"},
            {cx, cy, surface_z - 20e-6f, "depth_20um"},
            {cx, cy, surface_z - 40e-6f, "depth_40um"}
        };
    }

    void setSize(char size) {
        switch (size) {
            case 'S':
                nx = 100; ny = 100; nz = 50;
                break;
            case 'M':
                nx = 200; ny = 200; nz = 100;
                break;
            case 'L':
                nx = 400; ny = 400; nz = 200;
                break;
            default:
                std::cerr << "Unknown size: " << size << ", using Medium\n";
        }
        // Update probe positions for new domain size
        float cx = nx * dx / 2.0f;
        float cy = ny * dx / 2.0f;
        float surface_z = nz * dx * 0.8f;
        probes[0] = {cx, cy, surface_z, "center_surface"};
        probes[1] = {cx + 50e-6f, cy, surface_z, "offset_50um"};
        probes[2] = {cx, cy, surface_z - 20e-6f, "depth_20um"};
        probes[3] = {cx, cy, surface_z - 40e-6f, "depth_40um"};
    }

    void setMode(char m) {
        mode = m;
        switch (mode) {
            case 'A':  // Pure thermal
                enable_phase_change = false;
                enable_laser_scan = false;
                enable_fluid = false;
                break;
            case 'B':  // Thermal + phase change
                enable_phase_change = true;
                enable_laser_scan = true;
                enable_fluid = false;
                break;
            case 'C':  // Full multiphysics
                enable_phase_change = true;
                enable_laser_scan = true;
                enable_fluid = true;
                break;
        }
    }

    void print() const {
        std::cout << "\n============ Benchmark Configuration ============\n";
        std::cout << "Mode: " << mode << "\n";
        std::cout << "Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
        std::cout << "Physical size: " << (nx * dx * 1e6) << " x "
                  << (ny * dx * 1e6) << " x " << (nz * dx * 1e6) << " um\n";
        std::cout << "Grid spacing: " << (dx * 1e6) << " um\n";
        std::cout << "Time step: " << (dt * 1e9) << " ns\n";
        std::cout << "Total steps: " << num_steps << " (warmup: " << warmup_steps << ")\n";
        std::cout << "Total time: " << (num_steps * dt * 1e6) << " us\n";
        std::cout << "\nPhysics:\n";
        std::cout << "  Phase change: " << (enable_phase_change ? "YES" : "NO") << "\n";
        std::cout << "  Laser scan: " << (enable_laser_scan ? "YES" : "NO") << "\n";
        std::cout << "  Fluid flow: " << (enable_fluid ? "YES" : "NO") << "\n";
        std::cout << "\nLaser:\n";
        std::cout << "  Power: " << laser_power << " W\n";
        std::cout << "  Radius: " << (laser_radius * 1e6) << " um\n";
        std::cout << "  Absorptivity: " << laser_absorptivity << "\n";
        std::cout << "\nMaterial: Ti-6Al-4V\n";
        std::cout << "  Density: " << density << " kg/m^3\n";
        std::cout << "  Specific heat: " << specific_heat << " J/(kg*K)\n";
        std::cout << "  Conductivity: " << conductivity << " W/(m*K)\n";
        std::cout << "  Melting point: " << T_melt << " K\n";
        std::cout << "================================================\n\n";
    }
};

// ============================================================================
// Helper functions
// ============================================================================

void parseArgs(int argc, char** argv, BenchmarkConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            config.setMode(argv[++i][0]);
        } else if (arg == "--size" && i + 1 < argc) {
            config.setSize(argv[++i][0]);
        } else if (arg == "--steps" && i + 1 < argc) {
            config.num_steps = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup_steps = std::stoi(argv[++i]);
        } else if (arg == "--output") {
            config.output_vtk = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_thermal_lpbf [options]\n";
            std::cout << "  --mode <A|B|C>   Benchmark mode\n";
            std::cout << "                   A: Pure thermal (default)\n";
            std::cout << "                   B: Thermal + phase change\n";
            std::cout << "                   C: Full multiphysics\n";
            std::cout << "  --size <S|M|L>   Domain size (default: M)\n";
            std::cout << "                   S: 100x100x50\n";
            std::cout << "                   M: 200x200x100\n";
            std::cout << "                   L: 400x400x200\n";
            std::cout << "  --steps <N>      Time steps (default: 10000)\n";
            std::cout << "  --warmup <N>     Warmup steps (default: 100)\n";
            std::cout << "  --output         Enable VTK output\n";
            exit(0);
        }
    }
}

void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Get temperature at a probe point (interpolated)
float getProbeTemperature(const float* h_temperature, int nx, int ny, int nz,
                          float dx, float px, float py, float pz) {
    // Convert physical coordinates to grid indices
    int ix = static_cast<int>(px / dx);
    int iy = static_cast<int>(py / dx);
    int iz = static_cast<int>(pz / dx);

    // Clamp to valid range
    ix = std::max(0, std::min(nx - 1, ix));
    iy = std::max(0, std::min(ny - 1, iy));
    iz = std::max(0, std::min(nz - 1, iz));

    return h_temperature[ix + iy * nx + iz * nx * ny];
}

// Compute melt pool dimensions
void computeMeltPoolDimensions(const float* h_temperature,
                                int nx, int ny, int nz,
                                float dx, float T_melt,
                                float& width_um, float& depth_um) {
    // Find center of domain
    int cx = nx / 2;
    int cy = ny / 2;

    // Find melt pool width (along x at surface)
    int surface_z = static_cast<int>(nz * 0.8f);
    int min_x = cx, max_x = cx;
    for (int x = 0; x < nx; x++) {
        int idx = x + cy * nx + surface_z * nx * ny;
        if (h_temperature[idx] > T_melt) {
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
        }
    }
    width_um = (max_x - min_x) * dx * 1e6f;

    // Find melt pool depth (along z at center)
    int max_depth_z = surface_z;
    for (int z = surface_z; z >= 0; z--) {
        int idx = cx + cy * nx + z * nx * ny;
        if (h_temperature[idx] > T_melt) {
            max_depth_z = z;
        } else {
            break;
        }
    }
    depth_um = (surface_z - max_depth_z) * dx * 1e6f;
}

// ============================================================================
// Mode A: Pure Thermal Benchmark (using ThermalLBM directly)
// ============================================================================

utils::BenchmarkResults runModeA(const BenchmarkConfig& config) {
    std::cout << "Running Mode A: Pure Thermal Benchmark\n\n";

    utils::BenchmarkResults results;
    results.nx = config.nx;
    results.ny = config.ny;
    results.nz = config.nz;
    results.num_steps = config.num_steps;
    results.total_cells = static_cast<size_t>(config.nx) * config.ny * config.nz;

    utils::TimerPool timers;

    // Material properties
    physics::MaterialProperties material = physics::MaterialDatabase::getTi6Al4V();
    float thermal_diffusivity = config.conductivity / (config.density * config.specific_heat);

    // Initialize thermal LBM
    physics::ThermalLBM thermal(config.nx, config.ny, config.nz,
                                material, thermal_diffusivity,
                                false,  // No phase change for Mode A
                                config.dt, config.dx);
    thermal.initialize(config.T_ambient);

    // Laser setup (stationary at center)
    LaserSource laser(config.laser_power, config.laser_radius,
                      config.laser_absorptivity, config.laser_radius);  // penetration = radius
    float laser_x = config.nx * config.dx / 2.0f;
    float laser_y = config.ny * config.dx / 2.0f;
    laser.setPosition(laser_x, laser_y, 0.0f);

    // Allocate heat source field
    int num_cells = config.nx * config.ny * config.nz;
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));
    cudaMemset(d_heat_source, 0, num_cells * sizeof(float));

    // Host arrays for output
    std::vector<float> h_temperature(num_cells);

    // Record initial memory
    results.peak_memory_bytes = utils::getGPUMemoryUsage();
    results.allocated_fields = 4;  // g_src, g_dst, temperature, heat_source

    // VTK output setup
    if (config.output_vtk) {
        createDirectory(config.output_dir);
    }

    // CUDA grid for laser kernel
    dim3 block(8, 8, 8);
    dim3 grid((config.nx + block.x - 1) / block.x,
              (config.ny + block.y - 1) / block.y,
              (config.nz + block.z - 1) / block.z);

    // ========== Warmup Phase ==========
    std::cout << "Warmup: " << config.warmup_steps << " steps...\n";
    for (int step = 0; step < config.warmup_steps; step++) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, config.dx, config.dx, config.dx,
            config.nx, config.ny, config.nz
        );
        thermal.addHeatSource(d_heat_source, config.dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();
    }
    cudaDeviceSynchronize();

    // ========== Benchmark Phase ==========
    std::cout << "Benchmark: " << config.num_steps << " steps...\n";

    utils::WallTimer total_timer;
    total_timer.start();

    for (int step = 0; step < config.num_steps; step++) {
        // Laser heat source
        {
            auto t = timers.scoped("Laser");
            computeLaserHeatSourceKernel<<<grid, block>>>(
                d_heat_source, laser, config.dx, config.dx, config.dx,
                config.nx, config.ny, config.nz
            );
            cudaDeviceSynchronize();
        }

        // Add heat source to thermal
        thermal.addHeatSource(d_heat_source, config.dt);

        // Thermal LBM steps
        {
            auto t = timers.scoped("Collision");
            thermal.collisionBGK();
            cudaDeviceSynchronize();
        }

        {
            auto t = timers.scoped("Streaming");
            thermal.streaming();
            cudaDeviceSynchronize();
        }

        {
            auto t = timers.scoped("ComputeT");
            thermal.computeTemperature();
            cudaDeviceSynchronize();
        }

        // Record metrics at intervals
        if (step % config.output_interval == 0) {
            auto t = timers.scoped("Metrics");

            thermal.copyTemperatureToHost(h_temperature.data());

            // Find T_max
            float T_max = config.T_ambient;
            float T_min = 1e10f;
            for (int i = 0; i < num_cells; i++) {
                T_max = std::max(T_max, h_temperature[i]);
                T_min = std::min(T_min, h_temperature[i]);
            }

            float time_us = (config.warmup_steps + step) * config.dt * 1e6f;
            results.time_history_us.push_back(time_us);
            results.tmax_history.push_back(T_max);

            // Probe temperatures
            if (results.probe_history.empty()) {
                results.probe_history.resize(config.probes.size());
            }
            for (size_t i = 0; i < config.probes.size(); i++) {
                float T = getProbeTemperature(h_temperature.data(),
                                               config.nx, config.ny, config.nz,
                                               config.dx,
                                               config.probes[i].x,
                                               config.probes[i].y,
                                               config.probes[i].z);
                results.probe_history[i].push_back(T);
            }

            // Progress output
            if (step % (config.output_interval * 10) == 0) {
                std::cout << "  Step " << step << "/" << config.num_steps
                          << " | T_max = " << std::fixed << std::setprecision(1)
                          << T_max << " K\n";
            }

            // VTK output
            if (config.output_vtk) {
                auto t2 = timers.scoped("VTK");
                std::vector<float> dummy(num_cells, 0.0f);
                std::string filename = config.output_dir + "/benchmark_" +
                                       std::to_string(step) + ".vtk";
                io::VTKWriter::writeStructuredGrid(
                    filename,
                    h_temperature.data(), dummy.data(), dummy.data(),
                    config.nx, config.ny, config.nz,
                    config.dx, config.dx, config.dx
                );
            }

            results.max_temperature = T_max;
            results.min_temperature = T_min;
        }
    }

    cudaDeviceSynchronize();
    total_timer.stop();

    // ========== Final metrics ==========
    thermal.copyTemperatureToHost(h_temperature.data());

    // Melt pool dimensions
    computeMeltPoolDimensions(h_temperature.data(),
                               config.nx, config.ny, config.nz,
                               config.dx, config.T_melt,
                               results.melt_pool_width_um,
                               results.melt_pool_depth_um);

    // Timing
    results.total_time_sec = total_timer.elapsedSec();
    results.kernel_time_sec = (timers.get("Collision").total_ms +
                               timers.get("Streaming").total_ms +
                               timers.get("ComputeT").total_ms +
                               timers.get("Laser").total_ms) / 1000.0;
    results.io_time_sec = (timers.get("VTK").total_ms +
                           timers.get("Metrics").total_ms) / 1000.0;

    results.computeMLUPS();

    // Estimate bandwidth (D3Q7: 7 floats per cell, read + write)
    double bytes_per_step = results.total_cells * 7 * sizeof(float) * 2;
    results.bandwidth_gb_s = (bytes_per_step * config.num_steps) /
                              (results.kernel_time_sec * 1e9);

    // Energy balance (simplified)
    results.energy_balance_error_pct = 0.0f;  // TODO: compute properly

    // Cleanup
    cudaFree(d_heat_source);

    // Print timing breakdown
    timers.print();

    return results;
}

// ============================================================================
// Mode B: Thermal + Phase Change (using MultiphysicsSolver)
// ============================================================================

utils::BenchmarkResults runModeB(const BenchmarkConfig& config) {
    std::cout << "Running Mode B: Thermal + Phase Change Benchmark\n\n";

    utils::BenchmarkResults results;
    results.nx = config.nx;
    results.ny = config.ny;
    results.nz = config.nz;
    results.num_steps = config.num_steps;
    results.total_cells = static_cast<size_t>(config.nx) * config.ny * config.nz;

    utils::TimerPool timers;

    // Configure multiphysics solver
    physics::MultiphysicsConfig mpc;
    mpc.nx = config.nx;
    mpc.ny = config.ny;
    mpc.nz = config.nz;
    mpc.dx = config.dx;
    mpc.dt = config.dt;

    // Enable only thermal + phase change
    mpc.enable_thermal = true;
    mpc.enable_phase_change = true;
    mpc.enable_laser = true;
    mpc.enable_fluid = false;
    mpc.enable_vof = false;
    mpc.enable_marangoni = false;

    // Laser parameters
    mpc.laser_power = config.laser_power;
    mpc.laser_spot_radius = config.laser_radius;
    mpc.laser_absorptivity = config.laser_absorptivity;
    mpc.laser_penetration_depth = config.laser_radius;

    // Scanning laser
    if (config.enable_laser_scan) {
        mpc.laser_start_x = config.nx * config.dx * 0.2f;
        mpc.laser_start_y = config.ny * config.dx / 2.0f;
        mpc.laser_scan_vx = 0.5f;  // 0.5 m/s scan speed
    } else {
        mpc.laser_start_x = config.nx * config.dx / 2.0f;
        mpc.laser_start_y = config.ny * config.dx / 2.0f;
        mpc.laser_scan_vx = 0.0f;
    }

    mpc.material = physics::MaterialDatabase::getTi6Al4V();

    physics::MultiphysicsSolver solver(mpc);
    solver.initialize(config.T_ambient, 0.8f);  // Interface at 80%

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);

    results.peak_memory_bytes = utils::getGPUMemoryUsage();
    results.allocated_fields = 8;

    // Warmup
    std::cout << "Warmup: " << config.warmup_steps << " steps...\n";
    for (int step = 0; step < config.warmup_steps; step++) {
        solver.step(config.dt);
    }
    cudaDeviceSynchronize();

    // Benchmark
    std::cout << "Benchmark: " << config.num_steps << " steps...\n";

    utils::WallTimer total_timer;
    total_timer.start();

    for (int step = 0; step < config.num_steps; step++) {
        {
            auto t = timers.scoped("Solver");
            solver.step(config.dt);
            cudaDeviceSynchronize();
        }

        if (step % config.output_interval == 0) {
            auto t = timers.scoped("Metrics");

            solver.copyTemperatureToHost(h_temperature.data());

            float T_max = config.T_ambient;
            for (int i = 0; i < num_cells; i++) {
                T_max = std::max(T_max, h_temperature[i]);
            }

            float time_us = (config.warmup_steps + step) * config.dt * 1e6f;
            results.time_history_us.push_back(time_us);
            results.tmax_history.push_back(T_max);

            if (step % (config.output_interval * 10) == 0) {
                std::cout << "  Step " << step << "/" << config.num_steps
                          << " | T_max = " << std::fixed << std::setprecision(1)
                          << T_max << " K\n";
            }

            results.max_temperature = T_max;
        }
    }

    cudaDeviceSynchronize();
    total_timer.stop();

    // Final metrics
    solver.copyTemperatureToHost(h_temperature.data());
    computeMeltPoolDimensions(h_temperature.data(),
                               config.nx, config.ny, config.nz,
                               config.dx, config.T_melt,
                               results.melt_pool_width_um,
                               results.melt_pool_depth_um);

    results.total_time_sec = total_timer.elapsedSec();
    results.kernel_time_sec = timers.get("Solver").total_ms / 1000.0;
    results.io_time_sec = timers.get("Metrics").total_ms / 1000.0;
    results.computeMLUPS();

    timers.print();

    return results;
}

// ============================================================================
// Mode C: Full Multiphysics
// ============================================================================

utils::BenchmarkResults runModeC(const BenchmarkConfig& config) {
    std::cout << "Running Mode C: Full Multiphysics Benchmark\n\n";

    utils::BenchmarkResults results;
    results.nx = config.nx;
    results.ny = config.ny;
    results.nz = config.nz;
    results.num_steps = config.num_steps;
    results.total_cells = static_cast<size_t>(config.nx) * config.ny * config.nz;

    utils::TimerPool timers;

    // Full multiphysics configuration
    physics::MultiphysicsConfig mpc;
    mpc.nx = config.nx;
    mpc.ny = config.ny;
    mpc.nz = config.nz;
    mpc.dx = config.dx;
    mpc.dt = config.dt;

    mpc.enable_thermal = true;
    mpc.enable_phase_change = true;
    mpc.enable_laser = true;
    mpc.enable_fluid = true;
    mpc.enable_vof = true;
    mpc.enable_marangoni = true;
    mpc.enable_darcy = true;

    mpc.laser_power = config.laser_power;
    mpc.laser_spot_radius = config.laser_radius;
    mpc.laser_absorptivity = config.laser_absorptivity;
    mpc.laser_start_x = config.nx * config.dx * 0.2f;
    mpc.laser_start_y = config.ny * config.dx / 2.0f;
    mpc.laser_scan_vx = 0.5f;

    mpc.material = physics::MaterialDatabase::getTi6Al4V();

    physics::MultiphysicsSolver solver(mpc);
    solver.initialize(config.T_ambient, 0.8f);

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);

    results.peak_memory_bytes = utils::getGPUMemoryUsage();
    results.allocated_fields = 20;  // Many fields for full physics

    // Warmup
    std::cout << "Warmup: " << config.warmup_steps << " steps...\n";
    for (int step = 0; step < config.warmup_steps; step++) {
        solver.step(config.dt);
    }
    cudaDeviceSynchronize();

    // Benchmark
    std::cout << "Benchmark: " << config.num_steps << " steps...\n";

    utils::WallTimer total_timer;
    total_timer.start();

    for (int step = 0; step < config.num_steps; step++) {
        {
            auto t = timers.scoped("Solver");
            solver.step(config.dt);
            cudaDeviceSynchronize();
        }

        if (step % config.output_interval == 0) {
            auto t = timers.scoped("Metrics");

            solver.copyTemperatureToHost(h_temperature.data());

            float T_max = config.T_ambient;
            for (int i = 0; i < num_cells; i++) {
                T_max = std::max(T_max, h_temperature[i]);
            }

            float v_max = solver.getMaxVelocity();

            float time_us = (config.warmup_steps + step) * config.dt * 1e6f;
            results.time_history_us.push_back(time_us);
            results.tmax_history.push_back(T_max);

            if (step % (config.output_interval * 10) == 0) {
                std::cout << "  Step " << step << "/" << config.num_steps
                          << " | T_max = " << std::fixed << std::setprecision(1)
                          << T_max << " K"
                          << " | v_max = " << std::scientific << std::setprecision(2)
                          << v_max << " m/s\n";
            }

            results.max_temperature = T_max;
        }
    }

    cudaDeviceSynchronize();
    total_timer.stop();

    // Final metrics
    solver.copyTemperatureToHost(h_temperature.data());
    computeMeltPoolDimensions(h_temperature.data(),
                               config.nx, config.ny, config.nz,
                               config.dx, config.T_melt,
                               results.melt_pool_width_um,
                               results.melt_pool_depth_um);

    results.total_time_sec = total_timer.elapsedSec();
    results.kernel_time_sec = timers.get("Solver").total_ms / 1000.0;
    results.io_time_sec = timers.get("Metrics").total_ms / 1000.0;
    results.computeMLUPS();

    timers.print();

    return results;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "================================================\n";
    std::cout << "    LBMProject Thermal LPBF Benchmark           \n";
    std::cout << "================================================\n";

    // Print GPU info
    utils::printGPUInfo();

    // Parse configuration
    BenchmarkConfig config;
    parseArgs(argc, argv, config);
    config.print();

    // Run benchmark based on mode
    utils::BenchmarkResults results;
    switch (config.mode) {
        case 'A':
            results = runModeA(config);
            break;
        case 'B':
            results = runModeB(config);
            break;
        case 'C':
            results = runModeC(config);
            break;
        default:
            std::cerr << "Unknown mode: " << config.mode << "\n";
            return 1;
    }

    // Print results
    results.print();

    // Write CSV output
    std::string csv_filename = "benchmark_results_mode" +
                                std::string(1, config.mode) + ".csv";
    results.writeCSV(csv_filename);
    std::cout << "\nResults written to: " << csv_filename << "\n";

    std::cout << "\n================================================\n";
    std::cout << "    Benchmark Complete                          \n";
    std::cout << "================================================\n\n";

    return 0;
}
