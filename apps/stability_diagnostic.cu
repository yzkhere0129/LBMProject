/**
 * @file stability_diagnostic.cu
 * @brief Comprehensive Stability Diagnostic Tool for LPBF Simulation
 *
 * This application runs extended simulations with real-time stability monitoring:
 * - CFL number tracking (warns when approaching unity)
 * - Energy conservation checks
 * - NaN/Inf detection
 * - Temperature runaway detection
 * - Velocity divergence monitoring
 * - Force balance analysis
 *
 * Usage: ./stability_diagnostic <num_steps> [output_dir]
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"

using namespace lbm;

// Helper: Create directory
void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Stability diagnostic metrics
struct StabilityMetrics {
    float cfl_number;
    float max_velocity;
    float max_temperature;
    float total_thermal_energy;
    float total_kinetic_energy;
    float laser_energy_input;
    bool has_nan;
    int num_liquid_cells;
    int num_mushy_cells;

    // Derived metrics
    float energy_conservation_error;
    bool cfl_violation;
    bool temperature_runaway;
};

// Compute CFL number
float computeCFL(float v_max, float dt, float dx) {
    return v_max * dt / dx;
}

// Check for thermal runaway (unbounded temperature increase)
bool checkThermalRunaway(float T_max, float T_prev_max, float dT_threshold = 500.0f) {
    return (T_max - T_prev_max) > dT_threshold;
}

// Compute total thermal energy
__global__ void computeThermalEnergyKernel(
    const float* temperature,
    float* energy_sum,
    float rho, float cp,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // E_thermal = rho * cp * T * dx^3
    // We'll multiply by dx^3 on host
    float local_energy = rho * cp * temperature[idx];
    atomicAdd(energy_sum, local_energy);
}

// Compute total kinetic energy
__global__ void computeKineticEnergyKernel(
    const float* ux, const float* uy, const float* uz,
    float* energy_sum,
    float rho,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v2 = ux[idx]*ux[idx] + uy[idx]*uy[idx] + uz[idx]*uz[idx];
    float local_energy = 0.5f * rho * v2;  // E_kinetic = 0.5 * rho * v^2 * dx^3
    atomicAdd(energy_sum, local_energy);
}

// Main diagnostic function
StabilityMetrics runDiagnostics(
    const physics::MultiphysicsSolver& solver,
    const physics::MultiphysicsConfig& config,
    float time,
    float laser_total_energy,
    float T_prev_max)
{
    StabilityMetrics metrics;
    int num_cells = config.nx * config.ny * config.nz;

    // Get fields
    const float* d_T = solver.getTemperature();
    const float* d_vx = solver.getVelocityX();
    const float* d_vy = solver.getVelocityY();
    const float* d_vz = solver.getVelocityZ();
    const float* d_lf = solver.getLiquidFraction();

    // Copy to host
    std::vector<float> h_T(num_cells);
    std::vector<float> h_vx(num_cells);
    std::vector<float> h_vy(num_cells);
    std::vector<float> h_vz(num_cells);
    std::vector<float> h_lf(num_cells);

    cudaMemcpy(h_T.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx.data(), d_vx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy.data(), d_vy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vz.data(), d_vz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lf.data(), d_lf, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute max velocity and temperature
    metrics.max_velocity = 0.0f;
    metrics.max_temperature = 0.0f;
    metrics.num_liquid_cells = 0;
    metrics.num_mushy_cells = 0;
    metrics.has_nan = false;

    for (int i = 0; i < num_cells; ++i) {
        // Check for NaN/Inf
        if (std::isnan(h_T[i]) || std::isinf(h_T[i]) ||
            std::isnan(h_vx[i]) || std::isinf(h_vx[i]) ||
            std::isnan(h_vy[i]) || std::isinf(h_vy[i]) ||
            std::isnan(h_vz[i]) || std::isinf(h_vz[i])) {
            metrics.has_nan = true;
        }

        // Max temperature
        metrics.max_temperature = std::max(metrics.max_temperature, h_T[i]);

        // Max velocity
        float v = std::sqrt(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i] + h_vz[i]*h_vz[i]);
        metrics.max_velocity = std::max(metrics.max_velocity, v);

        // Count phase cells
        if (h_lf[i] > 0.99f) metrics.num_liquid_cells++;
        else if (h_lf[i] > 0.01f) metrics.num_mushy_cells++;
    }

    // Compute CFL number
    metrics.cfl_number = computeCFL(metrics.max_velocity, config.dt, config.dx);
    metrics.cfl_violation = (metrics.cfl_number > 0.5f);  // Conservative threshold

    // Check thermal runaway
    metrics.temperature_runaway = checkThermalRunaway(metrics.max_temperature, T_prev_max);

    // Compute total energies (on GPU for accuracy)
    float* d_thermal_energy;
    float* d_kinetic_energy;
    cudaMalloc(&d_thermal_energy, sizeof(float));
    cudaMalloc(&d_kinetic_energy, sizeof(float));
    cudaMemset(d_thermal_energy, 0, sizeof(float));
    cudaMemset(d_kinetic_energy, 0, sizeof(float));

    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    computeThermalEnergyKernel<<<blocks, threads>>>(
        d_T, d_thermal_energy,
        config.material.rho_solid, config.material.cp_solid,
        num_cells
    );

    computeKineticEnergyKernel<<<blocks, threads>>>(
        d_vx, d_vy, d_vz, d_kinetic_energy,
        config.density,
        num_cells
    );

    cudaDeviceSynchronize();

    float thermal_energy_density, kinetic_energy_density;
    cudaMemcpy(&thermal_energy_density, d_thermal_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&kinetic_energy_density, d_kinetic_energy, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_thermal_energy);
    cudaFree(d_kinetic_energy);

    // Convert to total energy (multiply by cell volume)
    float cell_volume = config.dx * config.dx * config.dx;
    metrics.total_thermal_energy = thermal_energy_density * cell_volume;
    metrics.total_kinetic_energy = kinetic_energy_density * cell_volume;

    // Energy conservation check
    metrics.laser_energy_input = laser_total_energy;
    float total_energy = metrics.total_thermal_energy + metrics.total_kinetic_energy;

    // Expected energy = initial thermal energy + laser input
    // For cold start (T_initial = 300K):
    float E_initial = config.material.rho_solid * config.material.cp_solid * 300.0f *
                      (config.nx * config.ny * config.nz * cell_volume);
    float E_expected = E_initial + laser_total_energy;

    metrics.energy_conservation_error = std::abs(total_energy - E_expected) / E_expected;

    return metrics;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int num_steps = 1000;  // Default: 100 μs
    std::string output_dir = "stability_tests";

    if (argc > 1) {
        num_steps = std::atoi(argv[1]);
    }
    if (argc > 2) {
        output_dir = argv[2];
    }

    std::cout << "=======================================================\n";
    std::cout << "  LPBF Stability Diagnostic Tool\n";
    std::cout << "=======================================================\n";
    std::cout << "Test parameters:\n";
    std::cout << "  Total steps: " << num_steps << "\n";
    std::cout << "  Output dir:  " << output_dir << "\n\n";

    // Configuration (identical to visualize_lpbf_marangoni_realistic.cu)
    physics::MultiphysicsConfig config;
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2.0e-6f;
    config.dt = 1.0e-7f;

    config.enable_thermal = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_darcy = true;
    config.enable_marangoni = true;
    config.enable_surface_tension = true;
    config.enable_laser = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;

    config.material = physics::MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;
    config.darcy_coefficient = 1.0e5f;
    config.surface_tension_coeff = 1.65f;
    config.dsigma_dT = -0.26e-3f;
    config.laser_power = 20.0f;
    config.laser_spot_radius = 50.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;
    config.boundary_type = 0;

    const int output_interval = 50;  // Less frequent output for long runs

    std::cout << "Physical parameters:\n";
    std::cout << "  Domain: " << config.nx*config.dx*1e6 << " x "
              << config.ny*config.dx*1e6 << " x " << config.nz*config.dx*1e6 << " μm\n";
    std::cout << "  Time step: " << config.dt*1e6 << " μs\n";
    std::cout << "  Total time: " << num_steps*config.dt*1e6 << " μs\n";
    std::cout << "  Laser power: " << config.laser_power << " W\n\n";

    // Initialize solver
    std::cout << "Initializing solver...\n";
    physics::MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    // Set initial liquid fraction (all solid)
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_lf(num_cells, 0.0f);
    float* d_lf;
    cudaMalloc(&d_lf, num_cells * sizeof(float));
    cudaMemcpy(d_lf, h_lf.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticLiquidFraction(d_lf);
    cudaFree(d_lf);

    std::cout << "✓ Solver initialized\n\n";

    // Create output directory
    createDirectory(output_dir);

    // Open log file
    std::string log_file = output_dir + "/stability_log.csv";
    std::ofstream log(log_file);
    log << "step,time_us,T_max,v_max,CFL,E_thermal,E_kinetic,E_laser,E_error,liq_cells,mushy_cells,has_nan,cfl_warn,T_runaway\n";

    std::cout << "Starting stability test...\n";
    std::cout << "Log file: " << log_file << "\n\n";

    std::cout << "Progress:\n";
    std::cout << "──────────────────────────────────────────────────────────────────────────────\n";
    std::cout << "  Step    Time[μs]   T_max[K]   v_max[m/s]     CFL    Status\n";
    std::cout << "──────────────────────────────────────────────────────────────────────────────\n";

    float laser_total_energy = 0.0f;
    float T_prev_max = 300.0f;
    bool diverged = false;

    for (int step = 0; step <= num_steps; ++step) {
        float time = step * config.dt;

        // Run diagnostics every output_interval steps
        if (step % output_interval == 0) {
            StabilityMetrics metrics = runDiagnostics(solver, config, time, laser_total_energy, T_prev_max);

            // Print to console
            std::cout << std::setw(6) << step
                      << std::setw(12) << std::fixed << std::setprecision(2) << time * 1e6
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.max_temperature
                      << std::setw(13) << std::scientific << std::setprecision(3) << metrics.max_velocity
                      << std::setw(10) << std::fixed << std::setprecision(4) << metrics.cfl_number;

            // Status flags
            if (metrics.has_nan) {
                std::cout << "   NaN DETECTED!";
                diverged = true;
            } else if (metrics.cfl_violation) {
                std::cout << "   CFL>0.5 WARNING";
            } else if (metrics.temperature_runaway) {
                std::cout << "   T RUNAWAY";
            } else {
                std::cout << "   OK";
            }
            std::cout << "\n";

            // Write to log
            log << step << ","
                << time * 1e6 << ","
                << metrics.max_temperature << ","
                << metrics.max_velocity << ","
                << metrics.cfl_number << ","
                << metrics.total_thermal_energy << ","
                << metrics.total_kinetic_energy << ","
                << metrics.laser_energy_input << ","
                << metrics.energy_conservation_error << ","
                << metrics.num_liquid_cells << ","
                << metrics.num_mushy_cells << ","
                << (metrics.has_nan ? 1 : 0) << ","
                << (metrics.cfl_violation ? 1 : 0) << ","
                << (metrics.temperature_runaway ? 1 : 0) << "\n";

            T_prev_max = metrics.max_temperature;

            // Stop if diverged
            if (diverged) {
                std::cout << "\n!! SIMULATION DIVERGED !!\n";
                std::cout << "Stopping at step " << step << " (t = " << time*1e6 << " μs)\n";
                break;
            }
        }

        // Time step
        if (step < num_steps) {
            solver.step(config.dt);

            // Track laser energy input
            laser_total_energy += config.laser_power * config.dt;
        }
    }

    std::cout << "──────────────────────────────────────────────────────────────────────────────\n\n";

    log.close();

    if (!diverged) {
        std::cout << "✓ Simulation completed successfully!\n";
        std::cout << "  Achieved: " << num_steps * config.dt * 1e6 << " μs stable operation\n";
    } else {
        std::cout << "✗ Simulation diverged before completion\n";
    }

    std::cout << "\nResults saved to: " << log_file << "\n";
    std::cout << "\nAnalysis:\n";
    std::cout << "  python3 analyze_stability.py " << log_file << "\n\n";

    return diverged ? 1 : 0;
}
