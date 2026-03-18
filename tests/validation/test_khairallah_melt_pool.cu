/**
 * @file test_khairallah_melt_pool.cu
 * @brief Khairallah 2016 Melt Pool Size Benchmark Test
 *
 * This validation test replicates the laser powder-bed fusion simulations from:
 * Khairallah, S. A., Anderson, A. T., Rubenchik, A., & King, W. E. (2016).
 * "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow
 * and formation mechanisms of pores, spatter, and denudation zones."
 * Acta Materialia, 108, 36-45.
 *
 * KEY EXPERIMENTAL PARAMETERS (from paper):
 * - Material: 316L stainless steel
 * - Laser power: 195 W
 * - Scan speeds: 0.8, 1.0, 1.2, 1.5 m/s (we test 1.0 m/s)
 * - Spot diameter: ~50-100 μm (we use 80 μm FWHM)
 * - Powder layer: 30 μm
 *
 * VALIDATION METRICS (from paper, Table 1 and Figures 3-5):
 * - Melt pool width: 100-150 μm (varies with scan speed)
 * - Melt pool depth: 50-100 μm (conduction mode)
 * - Keyhole depth: 150-250 μm (keyhole mode at higher powers)
 * - Denudation zone width: ~100 μm
 * - Peak temperatures: 2500-3500 K (approaching boiling)
 *
 * SIMULATION CONFIGURATION:
 * - Domain: 600×400×200 μm (300×200×100 cells)
 * - Grid spacing: dx = 2.0 μm
 * - Timestep: dt = 100 ns (thermal CFL stability)
 * - Scan length: 400 μm (along X-axis)
 * - Scan speed: 1.0 m/s
 * - Total simulation time: 400 μs (complete scan + solidification)
 *
 * PHYSICS ENABLED:
 * - Thermal diffusion + advection
 * - Phase change (melting/solidification)
 * - Incompressible Navier-Stokes flow
 * - VOF interface tracking
 * - Marangoni convection (thermocapillary)
 * - Surface tension
 * - Recoil pressure (if power > 300 W)
 * - Evaporative mass loss
 *
 * SUCCESS CRITERIA:
 * - Melt pool width: 80-200 μm (±50% of literature range)
 * - Melt pool depth: 30-150 μm (±50% of literature range)
 * - Peak temperature: 2000-4000 K (realistic range)
 * - Steady-state melt pool formation (after initial transient)
 * - No NaN or instability
 *
 * References:
 * - Khairallah et al. (2016) Acta Materialia 108:36-45
 * - King et al. (2015) J. Mater. Process. Technol. 214:2915-2925
 * - Körner et al. (2011) Modelling Simul. Mater. Sci. Eng. 19:064001
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"

using namespace lbm::physics;
using namespace lbm::io;

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            FAIL() << "CUDA error: " << cudaGetErrorString(err); \
        } \
    } while(0)

/**
 * @brief Helper function to create output directory
 */
bool createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        return mkdir(path.c_str(), 0755) == 0;
    } else if (info.st_mode & S_IFDIR) {
        // Directory exists
        return true;
    }
    return false;
}

/**
 * @brief Melt pool metrics computed from simulation
 */
struct MeltPoolMetrics {
    float width;          // Melt pool width [m]
    float depth;          // Melt pool depth [m]
    float length;         // Melt pool length [m]
    float peak_temp;      // Peak temperature [K]
    float surface_temp;   // Peak surface temperature [K]
    float liquid_volume;  // Total liquid volume [m³]
    float max_velocity;   // Maximum velocity magnitude [m/s]
};

/**
 * @brief Compute melt pool width (transverse dimension, Y-direction)
 *
 * Method: Find the width of the T > T_liquidus region at the interface
 */
float computeMeltPoolWidth(const MultiphysicsSolver& solver, float scan_position_x) {
    const auto& config = solver.getConfig();
    int nx = config.nx;
    int ny = config.ny;
    int nz = config.nz;
    float dx = config.dx;
    float T_liquidus = config.material.T_liquidus;

    // Get temperature and fill level fields
    std::vector<float> temperature(nx * ny * nz);
    std::vector<float> fill_level(nx * ny * nz);
    solver.copyTemperatureToHost(temperature.data());
    solver.copyFillLevelToHost(fill_level.data());

    // Find average interface height
    float avg_interface_z = 0.0f;
    int interface_count = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_level[idx];
                if (f > 0.3f && f < 0.7f) {
                    avg_interface_z += k;
                    interface_count++;
                }
            }
        }
    }

    if (interface_count > 0) {
        avg_interface_z /= interface_count;
    } else {
        avg_interface_z = nz / 2;
    }

    int k_interface = static_cast<int>(avg_interface_z);

    // Find scan position index
    int i_scan = static_cast<int>(scan_position_x / dx);
    i_scan = std::max(0, std::min(nx - 1, i_scan));

    // Scan along Y-axis at scan position to find melt pool width
    std::vector<bool> molten_y(ny, false);

    for (int j = 0; j < ny; ++j) {
        int idx = i_scan + nx * (j + ny * k_interface);
        float T = temperature[idx];
        float f = fill_level[idx];

        // Consider molten if T > T_liquidus and liquid (f > 0.5)
        if (T > T_liquidus && f > 0.5f) {
            molten_y[j] = true;
        }
    }

    // Find contiguous molten region
    int width_start = -1;
    int width_end = -1;

    for (int j = 0; j < ny; ++j) {
        if (molten_y[j]) {
            if (width_start == -1) width_start = j;
            width_end = j;
        }
    }

    if (width_start >= 0 && width_end >= width_start) {
        return (width_end - width_start + 1) * dx;
    }

    return 0.0f;
}

/**
 * @brief Compute melt pool depth (vertical dimension, Z-direction)
 *
 * Method: Find maximum depth where T > T_liquidus below the interface
 */
float computeMeltPoolDepth(const MultiphysicsSolver& solver, float scan_position_x) {
    const auto& config = solver.getConfig();
    int nx = config.nx;
    int ny = config.ny;
    int nz = config.nz;
    float dx = config.dx;
    float T_liquidus = config.material.T_liquidus;

    std::vector<float> temperature(nx * ny * nz);
    std::vector<float> fill_level(nx * ny * nz);
    solver.copyTemperatureToHost(temperature.data());
    solver.copyFillLevelToHost(fill_level.data());

    // Find interface height
    float avg_interface_z = 0.0f;
    int interface_count = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_level[idx];
                if (f > 0.3f && f < 0.7f) {
                    avg_interface_z += k;
                    interface_count++;
                }
            }
        }
    }

    if (interface_count > 0) {
        avg_interface_z /= interface_count;
    } else {
        avg_interface_z = nz / 2;
    }

    // Find scan position
    int i_scan = static_cast<int>(scan_position_x / dx);
    i_scan = std::max(0, std::min(nx - 1, i_scan));

    // Find maximum depth at scan position
    float max_depth = 0.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            int idx = i_scan + nx * (j + ny * k);
            float T = temperature[idx];
            float f = fill_level[idx];

            if (T > T_liquidus && f > 0.5f && k < avg_interface_z) {
                float depth = (avg_interface_z - k) * dx;
                max_depth = std::max(max_depth, depth);
            }
        }
    }

    return max_depth;
}

/**
 * @brief Compute melt pool length (along scan direction, X-direction)
 *
 * Method: Find length of molten region along X-axis
 */
float computeMeltPoolLength(const MultiphysicsSolver& solver) {
    const auto& config = solver.getConfig();
    int nx = config.nx;
    int ny = config.ny;
    int nz = config.nz;
    float dx = config.dx;
    float T_liquidus = config.material.T_liquidus;

    std::vector<float> temperature(nx * ny * nz);
    std::vector<float> fill_level(nx * ny * nz);
    solver.copyTemperatureToHost(temperature.data());
    solver.copyFillLevelToHost(fill_level.data());

    // Find interface height
    float avg_interface_z = 0.0f;
    int interface_count = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_level[idx];
                if (f > 0.3f && f < 0.7f) {
                    avg_interface_z += k;
                    interface_count++;
                }
            }
        }
    }

    if (interface_count > 0) {
        avg_interface_z /= interface_count;
    } else {
        avg_interface_z = nz / 2;
    }

    int k_interface = static_cast<int>(avg_interface_z);
    int j_center = ny / 2;

    // Find molten region along X
    std::vector<bool> molten_x(nx, false);

    for (int i = 0; i < nx; ++i) {
        int idx = i + nx * (j_center + ny * k_interface);
        float T = temperature[idx];
        float f = fill_level[idx];

        if (T > T_liquidus && f > 0.5f) {
            molten_x[i] = true;
        }
    }

    // Find contiguous region
    int length_start = -1;
    int length_end = -1;

    for (int i = 0; i < nx; ++i) {
        if (molten_x[i]) {
            if (length_start == -1) length_start = i;
            length_end = i;
        }
    }

    if (length_start >= 0 && length_end >= length_start) {
        return (length_end - length_start + 1) * dx;
    }

    return 0.0f;
}

/**
 * @brief Compute all melt pool metrics
 */
MeltPoolMetrics computeMeltPoolMetrics(const MultiphysicsSolver& solver, float scan_position_x) {
    MeltPoolMetrics metrics;

    metrics.width = computeMeltPoolWidth(solver, scan_position_x);
    metrics.depth = computeMeltPoolDepth(solver, scan_position_x);
    metrics.length = computeMeltPoolLength(solver);
    metrics.peak_temp = solver.getMaxTemperature();
    metrics.max_velocity = solver.getMaxVelocity();

    // Compute surface temperature (maximum at interface)
    const auto& config = solver.getConfig();
    int nx = config.nx;
    int ny = config.ny;
    int nz = config.nz;

    std::vector<float> temperature(nx * ny * nz);
    std::vector<float> fill_level(nx * ny * nz);
    solver.copyTemperatureToHost(temperature.data());
    solver.copyFillLevelToHost(fill_level.data());

    metrics.surface_temp = 0.0f;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_level[idx];
                if (f > 0.3f && f < 0.7f) {
                    metrics.surface_temp = std::max(metrics.surface_temp, temperature[idx]);
                }
            }
        }
    }

    // Compute liquid volume
    float T_liquidus = config.material.T_liquidus;
    float cell_volume = config.dx * config.dx * config.dx;
    metrics.liquid_volume = 0.0f;

    for (int idx = 0; idx < nx * ny * nz; ++idx) {
        if (temperature[idx] > T_liquidus && fill_level[idx] > 0.5f) {
            metrics.liquid_volume += cell_volume;
        }
    }

    return metrics;
}

/**
 * @brief Main test: Khairallah 2016 melt pool size benchmark
 */
TEST(KhairallahBenchmark, MeltPoolSize_316L_195W_1mps) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "KHAIRALLAH 2016 MELT POOL SIZE BENCHMARK\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "Reference: Khairallah et al. (2016) Acta Materialia 108:36-45\n";
    std::cout << "Test case: 316L stainless steel, P=195W, v=1.0 m/s\n\n";

    // ========================================================================
    // DOMAIN SETUP
    // ========================================================================

    const int nx = 300;  // 600 μm / 2 μm
    const int ny = 200;  // 400 μm / 2 μm
    const int nz = 100;  // 200 μm / 2 μm
    const float dx = 2.0e-6f;  // 2 μm grid spacing

    const float domain_x = nx * dx;  // 600 μm
    const float domain_y = ny * dx;  // 400 μm
    const float domain_z = nz * dx;  // 200 μm

    std::cout << "=== Domain Configuration ===\n";
    std::cout << "Grid: " << nx << " × " << ny << " × " << nz << " cells\n";
    std::cout << "Grid spacing: " << dx * 1e6 << " μm\n";
    std::cout << "Domain size: " << domain_x * 1e6 << " × " << domain_y * 1e6
              << " × " << domain_z * 1e6 << " μm³\n\n";

    // ========================================================================
    // MATERIAL: 316L STAINLESS STEEL
    // ========================================================================

    MaterialProperties material = MaterialDatabase::get316L();

    std::cout << "=== Material: 316L Stainless Steel ===\n";
    std::cout << "  ρ_solid  = " << material.rho_solid << " kg/m³\n";
    std::cout << "  ρ_liquid = " << material.rho_liquid << " kg/m³\n";
    std::cout << "  cp_solid = " << material.cp_solid << " J/(kg·K)\n";
    std::cout << "  k_solid  = " << material.k_solid << " W/(m·K)\n";
    std::cout << "  T_solidus = " << material.T_solidus << " K\n";
    std::cout << "  T_liquidus = " << material.T_liquidus << " K (melting point)\n";
    std::cout << "  T_boiling = " << material.T_vaporization << " K\n";
    std::cout << "  L_fusion = " << material.L_fusion * 1e-3 << " kJ/kg\n";
    std::cout << "  σ = " << material.surface_tension << " N/m\n";
    std::cout << "  dσ/dT = " << material.dsigma_dT * 1e3 << " mN/(m·K)\n\n";

    // ========================================================================
    // LASER PARAMETERS (Khairallah 2016)
    // ========================================================================

    const float laser_power = 195.0f;  // W (from paper)
    const float laser_spot_radius = 40e-6f;  // 40 μm (80 μm FWHM diameter)
    const float laser_absorptivity = 0.38f;  // 316L typical value
    const float laser_penetration_depth = 10e-6f;  // 10 μm Beer-Lambert depth
    const float scan_speed = 1.0f;  // m/s

    std::cout << "=== Laser Parameters ===\n";
    std::cout << "  Power: " << laser_power << " W\n";
    std::cout << "  Spot radius (1/e²): " << laser_spot_radius * 1e6 << " μm\n";
    std::cout << "  FWHM diameter: " << laser_spot_radius * 2.0 * 1.177 << " μm\n";
    std::cout << "  Absorptivity: " << laser_absorptivity << "\n";
    std::cout << "  Penetration depth: " << laser_penetration_depth * 1e6 << " μm\n";
    std::cout << "  Scan speed: " << scan_speed << " m/s\n";
    std::cout << "  Energy density: " << laser_power / (scan_speed * 2 * laser_spot_radius) * 1e-6
              << " J/mm²\n\n";

    // ========================================================================
    // SIMULATION PARAMETERS
    // ========================================================================

    // Thermal diffusivity for 316L
    float alpha_thermal = material.k_solid / (material.rho_solid * material.cp_solid);
    std::cout << "Thermal diffusivity: " << alpha_thermal * 1e6 << " mm²/s\n";

    // Timestep selection (CFL stability)
    const float dt = 100e-9f;  // 100 ns
    const float scan_length = 400e-6f;  // 400 μm scan length
    const float scan_time = scan_length / scan_speed;  // 400 μs
    const float total_time = scan_time + 200e-6f;  // Scan + 200 μs solidification
    const int total_steps = static_cast<int>(total_time / dt);
    const int output_interval = 100;  // Every 10 μs

    std::cout << "\n=== Simulation Parameters ===\n";
    std::cout << "  Timestep: " << dt * 1e9 << " ns\n";
    std::cout << "  Scan length: " << scan_length * 1e6 << " μm\n";
    std::cout << "  Scan time: " << scan_time * 1e6 << " μs\n";
    std::cout << "  Total time: " << total_time * 1e6 << " μs\n";
    std::cout << "  Total steps: " << total_steps << "\n";
    std::cout << "  Output interval: " << output_interval * dt * 1e6 << " μs\n";

    // Verify thermal CFL stability
    float omega = 1.0f / (0.5f + 3.0f * alpha_thermal * dt / (dx * dx));
    std::cout << "  LBM omega = " << omega << " (should be < 1.9)\n";
    ASSERT_LT(omega, 1.9f) << "Thermal solver unstable (omega too large)";
    ASSERT_GT(omega, 0.5f) << "Thermal solver invalid (omega too small)";
    std::cout << "\n";

    // ========================================================================
    // MULTIPHYSICS CONFIGURATION
    // ========================================================================

    MultiphysicsConfig config;
    config.nx = nx;
    config.ny = ny;
    config.nz = nz;
    config.dx = dx;
    config.dt = dt;
    config.material = material;

    // Enable full multiphysics
    config.enable_thermal = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_surface_tension = true;
    config.enable_marangoni = true;
    config.enable_laser = true;
    config.enable_darcy = true;
    config.enable_buoyancy = true;
    config.enable_evaporation_mass_loss = true;
    config.enable_recoil_pressure = false;  // Conduction mode (not keyhole)
    config.enable_solidification_shrinkage = true;

    // Thermal properties
    config.thermal_diffusivity = alpha_thermal;

    // Fluid properties
    config.kinematic_viscosity = 0.0333f;  // Lattice units (tau=0.6)
    config.density = material.rho_liquid;
    config.darcy_coefficient = 1e7f;

    // Surface properties
    config.surface_tension_coeff = material.surface_tension;
    config.dsigma_dT = material.dsigma_dT;

    // Buoyancy
    config.thermal_expansion_coeff = 1.5e-5f;
    config.gravity_x = 0.0f;
    config.gravity_y = 0.0f;
    config.gravity_z = -9.81f;
    config.reference_temperature = material.T_liquidus;

    // Laser configuration (MOVING BEAM)
    config.laser_power = laser_power;
    config.laser_spot_radius = laser_spot_radius;
    config.laser_absorptivity = laser_absorptivity;
    config.laser_penetration_depth = laser_penetration_depth;
    config.laser_shutoff_time = scan_time;  // Turn off after scan complete
    config.laser_start_x = 100e-6f;  // Start 100 μm from left edge
    config.laser_start_y = domain_y / 2.0f;  // Center in Y
    config.laser_scan_vx = scan_speed;  // 1.0 m/s in X
    config.laser_scan_vy = 0.0f;

    // Boundary conditions — per-face specification
    // X/Y: wall + adiabatic (zero heat flux, finite domain)
    // Z_min: wall + convective (water-cooled build plate)
    // Z_max: wall + adiabatic (gas phase above free surface)
    config.boundaries.x_min = BoundaryType::WALL;
    config.boundaries.x_max = BoundaryType::WALL;
    config.boundaries.y_min = BoundaryType::WALL;
    config.boundaries.y_max = BoundaryType::WALL;
    config.boundaries.z_min = BoundaryType::WALL;
    config.boundaries.z_max = BoundaryType::WALL;

    config.boundaries.thermal_x_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_x_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min = ThermalBCType::CONVECTIVE;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;

    config.boundaries.convective_h = 2000.0f;      // W/(m²·K) water-cooled build plate
    config.boundaries.convective_T_inf = 300.0f;    // K ambient

    config.enable_radiation_bc = false;
    config.enable_substrate_cooling = true;
    config.substrate_h_conv = 2000.0f;      // For energy balance diagnostic
    config.substrate_temperature = 300.0f;   // For energy balance diagnostic

    // VOF subcycling
    config.vof_subcycles = 10;

    // CFL limiting
    config.cfl_limit = 0.6f;
    config.cfl_velocity_target = 0.15f;

    // ========================================================================
    // INITIALIZE SOLVER
    // ========================================================================

    std::cout << "Initializing MultiphysicsSolver...\n";
    MultiphysicsSolver solver(config);

    // Initialize: substrate at 300K, powder bed at 300K, flat interface at z=100μm
    const float T_initial = 300.0f;
    const float interface_height = 0.5f;  // Middle of domain
    solver.initialize(T_initial, interface_height);

    std::cout << "Initialization complete.\n\n";

    // ========================================================================
    // OUTPUT DIRECTORY
    // ========================================================================

    std::string output_dir = "/home/yzk/LBMProject/tests/validation/output_khairallah_melt_pool";
    if (!createDirectory(output_dir)) {
        std::cerr << "Warning: Could not create output directory: " << output_dir << "\n";
    }

    // ========================================================================
    // TIME INTEGRATION
    // ========================================================================

    std::cout << "Starting simulation...\n";
    std::cout << std::string(80, '=') << "\n";

    VTKWriter vtk_writer;

    // Time series storage
    std::vector<float> time_history;
    std::vector<float> width_history;
    std::vector<float> depth_history;
    std::vector<float> length_history;
    std::vector<float> peak_temp_history;
    std::vector<float> surface_temp_history;
    std::vector<float> max_velocity_history;
    std::vector<float> laser_position_history;

    // Track steady-state metrics (last 100 μs of scan)
    std::vector<float> steady_state_widths;
    std::vector<float> steady_state_depths;

    for (int step = 0; step <= total_steps; ++step) {
        float current_time = step * dt;

        // Time step
        solver.step(dt);

        // Check for NaN
        if (solver.checkNaN()) {
            FAIL() << "NaN detected at step " << step << " (t = "
                   << current_time * 1e6 << " μs)";
        }

        // Diagnostics and output
        if (step % output_interval == 0) {
            // Compute laser position
            float laser_x = 100e-6f + scan_speed * std::min(current_time, scan_time);

            // Compute melt pool metrics
            MeltPoolMetrics metrics = computeMeltPoolMetrics(solver, laser_x);

            // Store history
            time_history.push_back(current_time);
            width_history.push_back(metrics.width);
            depth_history.push_back(metrics.depth);
            length_history.push_back(metrics.length);
            peak_temp_history.push_back(metrics.peak_temp);
            surface_temp_history.push_back(metrics.surface_temp);
            max_velocity_history.push_back(metrics.max_velocity);
            laser_position_history.push_back(laser_x);

            // Track steady-state (after 200 μs, before laser shutoff)
            if (current_time > 200e-6f && current_time < scan_time) {
                steady_state_widths.push_back(metrics.width);
                steady_state_depths.push_back(metrics.depth);
            }

            // Print progress
            std::cout << "t = " << std::setw(7) << std::fixed << std::setprecision(1)
                      << current_time * 1e6 << " μs"
                      << "  |  Laser X = " << std::setw(6) << std::setprecision(1)
                      << laser_x * 1e6 << " μm"
                      << "  |  Width = " << std::setw(6) << std::setprecision(1)
                      << metrics.width * 1e6 << " μm"
                      << "  |  Depth = " << std::setw(6) << std::setprecision(1)
                      << metrics.depth * 1e6 << " μm"
                      << "  |  T_max = " << std::setw(7) << std::setprecision(0)
                      << metrics.peak_temp << " K"
                      << "  |  v_max = " << std::setw(5) << std::setprecision(2)
                      << metrics.max_velocity << " m/s\n";

            // Write VTK output (every 50 μs)
            if (step % (output_interval * 5) == 0) {
                std::ostringstream filename;
                filename << output_dir << "/khairallah_t"
                         << std::setw(6) << std::setfill('0') << step;

                // Get all field data
                std::vector<float> h_temp(nx * ny * nz);
                std::vector<float> h_fl(nx * ny * nz);
                std::vector<float> h_fill(nx * ny * nz);
                std::vector<float> h_vx(nx * ny * nz);
                std::vector<float> h_vy(nx * ny * nz);
                std::vector<float> h_vz(nx * ny * nz);

                solver.copyTemperatureToHost(h_temp.data());
                solver.copyFillLevelToHost(h_fill.data());
                solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

                // Copy liquid fraction from device
                CUDA_CHECK(cudaMemcpy(h_fl.data(), solver.getLiquidFraction(),
                                     nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

                // Create phase state field (0=solid, 1=mushy, 2=liquid)
                std::vector<float> h_phase(nx * ny * nz);
                for (int idx = 0; idx < nx * ny * nz; ++idx) {
                    float fl = h_fl[idx];
                    if (fl < 0.01f) {
                        h_phase[idx] = 0.0f;  // Solid
                    } else if (fl > 0.99f) {
                        h_phase[idx] = 2.0f;  // Liquid
                    } else {
                        h_phase[idx] = 1.0f;  // Mushy
                    }
                }

                VTKWriter::writeStructuredGridWithVectors(
                    filename.str(),
                    h_temp.data(),
                    h_fl.data(),
                    h_phase.data(),
                    h_fill.data(),
                    h_vx.data(),
                    h_vy.data(),
                    h_vz.data(),
                    nx, ny, nz, dx, dx, dx);
            }
        }
    }

    std::cout << std::string(80, '=') << "\n";
    std::cout << "Simulation complete.\n\n";

    // ========================================================================
    // SAVE TIME SERIES DATA
    // ========================================================================

    std::string csv_file = output_dir + "/melt_pool_metrics.csv";
    std::ofstream csv(csv_file);
    csv << "time_us,laser_x_um,width_um,depth_um,length_um,peak_temp_K,surface_temp_K,max_velocity_mps\n";

    for (size_t i = 0; i < time_history.size(); ++i) {
        csv << time_history[i] * 1e6 << ","
            << laser_position_history[i] * 1e6 << ","
            << width_history[i] * 1e6 << ","
            << depth_history[i] * 1e6 << ","
            << length_history[i] * 1e6 << ","
            << peak_temp_history[i] << ","
            << surface_temp_history[i] << ","
            << max_velocity_history[i] << "\n";
    }
    csv.close();
    std::cout << "Time series data saved to: " << csv_file << "\n\n";

    // ========================================================================
    // COMPUTE STEADY-STATE STATISTICS
    // ========================================================================

    if (steady_state_widths.empty()) {
        std::cout << "WARNING: No steady-state data collected (simulation too short?)\n";
    } else {
        // Compute average and standard deviation
        float avg_width = 0.0f, avg_depth = 0.0f;
        for (size_t i = 0; i < steady_state_widths.size(); ++i) {
            avg_width += steady_state_widths[i];
            avg_depth += steady_state_depths[i];
        }
        avg_width /= steady_state_widths.size();
        avg_depth /= steady_state_depths.size();

        float std_width = 0.0f, std_depth = 0.0f;
        for (size_t i = 0; i < steady_state_widths.size(); ++i) {
            std_width += (steady_state_widths[i] - avg_width) * (steady_state_widths[i] - avg_width);
            std_depth += (steady_state_depths[i] - avg_depth) * (steady_state_depths[i] - avg_depth);
        }
        std_width = std::sqrt(std_width / steady_state_widths.size());
        std_depth = std::sqrt(std_depth / steady_state_depths.size());

        // Find maximum values
        float max_width = *std::max_element(width_history.begin(), width_history.end());
        float max_depth = *std::max_element(depth_history.begin(), depth_history.end());
        float max_temp = *std::max_element(peak_temp_history.begin(), peak_temp_history.end());
        float max_velocity = *std::max_element(max_velocity_history.begin(), max_velocity_history.end());

        // ========================================================================
        // VALIDATION AGAINST KHAIRALLAH 2016
        // ========================================================================

        std::cout << std::string(80, '=') << "\n";
        std::cout << "VALIDATION RESULTS\n";
        std::cout << std::string(80, '=') << "\n\n";

        std::cout << "=== Steady-State Melt Pool Dimensions ===\n";
        std::cout << "  Width:  " << avg_width * 1e6 << " ± " << std_width * 1e6 << " μm"
                  << "  (max: " << max_width * 1e6 << " μm)\n";
        std::cout << "  Depth:  " << avg_depth * 1e6 << " ± " << std_depth * 1e6 << " μm"
                  << "  (max: " << max_depth * 1e6 << " μm)\n";
        std::cout << "  Peak Temperature: " << max_temp << " K\n";
        std::cout << "  Max Velocity: " << max_velocity << " m/s\n\n";

        std::cout << "=== Literature Values (Khairallah 2016) ===\n";
        std::cout << "  Width:  100-150 μm (conduction mode)\n";
        std::cout << "  Depth:  50-100 μm (conduction mode)\n";
        std::cout << "  Peak Temperature: 2500-3500 K\n";
        std::cout << "  Marangoni Velocity: 0.5-2.0 m/s\n\n";

        // Validation criteria (±50% tolerance due to parameter uncertainties)
        const float width_min = 80e-6f;   // 80 μm
        const float width_max = 200e-6f;  // 200 μm
        const float depth_min = 30e-6f;   // 30 μm
        const float depth_max = 150e-6f;  // 150 μm
        const float temp_min = 2000.0f;   // 2000 K
        const float temp_max = 4000.0f;   // 4000 K

        bool width_valid = (avg_width >= width_min && avg_width <= width_max);
        bool depth_valid = (avg_depth >= depth_min && avg_depth <= depth_max);
        bool temp_valid = (max_temp >= temp_min && max_temp <= temp_max);

        std::cout << "=== Validation Status ===\n";
        std::cout << "  Melt pool width: " << (width_valid ? "PASS" : "FAIL")
                  << " (" << avg_width * 1e6 << " μm in range "
                  << width_min * 1e6 << "-" << width_max * 1e6 << " μm)\n";
        std::cout << "  Melt pool depth: " << (depth_valid ? "PASS" : "FAIL")
                  << " (" << avg_depth * 1e6 << " μm in range "
                  << depth_min * 1e6 << "-" << depth_max * 1e6 << " μm)\n";
        std::cout << "  Peak temperature: " << (temp_valid ? "PASS" : "FAIL")
                  << " (" << max_temp << " K in range "
                  << temp_min << "-" << temp_max << " K)\n\n";

        // GTest assertions
        EXPECT_GE(avg_width, width_min) << "Melt pool width too small";
        EXPECT_LE(avg_width, width_max) << "Melt pool width too large";
        EXPECT_GE(avg_depth, depth_min) << "Melt pool depth too small";
        EXPECT_LE(avg_depth, depth_max) << "Melt pool depth too large";
        EXPECT_GE(max_temp, temp_min) << "Peak temperature too low";
        EXPECT_LE(max_temp, temp_max) << "Peak temperature too high";

        // Overall pass/fail
        bool overall_pass = width_valid && depth_valid && temp_valid;
        std::cout << "=== OVERALL RESULT: " << (overall_pass ? "PASS" : "FAIL") << " ===\n\n";

        if (overall_pass) {
            std::cout << "SUCCESS: Melt pool dimensions match Khairallah 2016 benchmark.\n";
        } else {
            std::cout << "PARTIAL SUCCESS: Some metrics outside expected range.\n";
            std::cout << "This may indicate:\n";
            std::cout << "  - Parameter calibration needed (absorptivity, surface tension, etc.)\n";
            std::cout << "  - Grid resolution effects\n";
            std::cout << "  - Physical model differences (powder vs. solid substrate)\n";
        }
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Test complete. Output files in: " << output_dir << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

// Additional test cases (optional, for comprehensive validation)

/**
 * @brief Test case: Lower scan speed (0.8 m/s) - wider, deeper melt pool expected
 */
TEST(KhairallahBenchmark, DISABLED_MeltPoolSize_316L_195W_0p8mps) {
    // Similar setup to main test, but with v = 0.8 m/s
    // Expected: Wider and deeper melt pool due to higher energy density
    GTEST_SKIP() << "Extended test case - enable manually for full benchmark suite";
}

/**
 * @brief Test case: Higher scan speed (1.5 m/s) - shallower melt pool expected
 */
TEST(KhairallahBenchmark, DISABLED_MeltPoolSize_316L_195W_1p5mps) {
    // Similar setup to main test, but with v = 1.5 m/s
    // Expected: Shallower melt pool due to lower energy density
    GTEST_SKIP() << "Extended test case - enable manually for full benchmark suite";
}
