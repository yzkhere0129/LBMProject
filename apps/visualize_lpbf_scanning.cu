/**
 * @file visualize_lpbf_marangoni_realistic.cu
 * @brief Realistic LPBF Simulation with Marangoni Convection
 *
 * Starting conditions:
 * - Cold solid metal at room temperature (300K)
 * - Laser heats the surface progressively
 * - Material melts when T > T_liquidus
 * - Marangoni forces drive flow in liquid pool
 * - Darcy damping keeps solid stationary
 *
 * Physics modules enabled:
 * - Thermal: Laser heating + diffusion
 * - Phase change: Solid <-> liquid transitions
 * - Fluid: LBM flow with buoyancy
 * - Marangoni: Surface tension gradient forces
 * - Darcy: Solid region damping
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"
#include "config/lpbf_config_loader.h"  // NEW: Config file support

using namespace lbm;

// Helper: Create directory
void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "  LPBF Laser Scanning Simulation   \n";
    std::cout << "==============================================\n\n";

    // =========================================================================
    // CONFIGURATION: Realistic LPBF Parameters
    // =========================================================================
    // NEW: Support config file loading via --config flag
    // Without --config: Uses embedded defaults (backward compatible)
    // With --config <file>: Loads parameters from file

    physics::MultiphysicsConfig config;

    // =========================================================================
    // DOMAIN SIZE - OPTIMIZED FOR KEYHOLE SIMULATION
    // =========================================================================
    // Design principles:
    // - XY: >= 5x laser diameter (100μm) to avoid boundary effects
    // - Z gas: >= keyhole depth (~100μm) + buffer space
    // - Z metal: substrate (14μm) + melt pool depth (~80μm)
    // KEYHOLE SIMULATION: Larger domain for better visualization
    // - XY: 4x laser diameter to capture full melt pool and solidification
    // - Z: 80% metal, 20% gas for keyhole formation
    config.nx = 200;  // 400 μm (4x laser diameter) - EXPANDED
    config.ny = 150;  // 300 μm (3x laser diameter) - EXPANDED
    config.nz = 100;  // 200 μm total height - EXPANDED
    config.dx = 2.0e-6f;  // 2 μm cell size

    // Time stepping
    config.dt = 1.0e-7f;  // 0.1 μs time step

    // Physics modules: ENABLE MARANGONI FOR FLOW
    config.enable_thermal = true;           // ✓ Thermal diffusion
    config.enable_phase_change = true;      // ✓ PHASE CHANGE (melting/solidification)
    config.enable_fluid = true;             // ✓ Fluid flow
    config.enable_darcy = true;             // Re-enabled for realistic damping
    config.enable_marangoni = true;         // ✓ ENABLED - drives melt pool flow
    config.enable_surface_tension = true;   // Enabled for interface smoothing
    config.enable_laser = true;             // ✓ LASER HEATING (KEY!)
    config.enable_vof = true;               // ✓ Free surface tracking
    config.enable_vof_advection = true;     // PHASE 2: Enable interface advection
    config.enable_recoil_pressure = true;   // ✓ RECOIL PRESSURE (for keyhole mode at P > 300W)

    // Recoil pressure parameters (Ti6Al4V Anisimov model)
    // At T > 5000K, P_recoil can exceed 300 MPa - need higher limit for keyhole
    config.recoil_coefficient = 0.54f;      // Knight (1979) coefficient
    config.recoil_smoothing_width = 2.0f;   // Temperature ramp window [K] (2 K ≈ hard threshold)
    config.recoil_max_pressure = 1e8f;      // 100 MPa max (balanced: keyhole + stability)

    // Material: Ti6Al4V
    config.material = physics::MaterialDatabase::getTi6Al4V();

    // Thermal properties
    config.thermal_diffusivity = 5.8e-6f;  // Ti6Al4V liquid (m²/s)

    // Fluid properties
    config.kinematic_viscosity = 0.0333f;  // LATTICE UNITS (tau=0.6, stable)
    config.density = 4110.0f;              // Ti6Al4V liquid (kg/m³)

    // Darcy damping coefficient (tuned for Ti6Al4V mushy zone)
    // Literature values:
    //   - Khairallah et al. 2016: C ~ 1e5 to 1e7 for LPBF Ti6Al4V
    //   - Carman-Kozeny (λ₂=1μm): C ~ 2.4e4
    // Historical success (Nov 17, 00:49): 1e5 yielded stable 600 μs run with 6.9 mm/s peak
    config.darcy_coefficient = 1.0e5f;  // Proven working value

    // =========================================================================
    // CFL LIMITER TUNING FOR KEYHOLE FORMATION (Revised Nov 22)
    // =========================================================================
    // Problem analysis:
    //   - v_target=0.3 (6 m/s) caused 95.7% force reduction
    //   - Keyhole needs v_phys = 5-20 m/s for recoil-driven depression
    //   - LBM stability: Ma < 1 (v_lattice < 0.577), but Ma < 0.3 preferred
    //
    // Three CFL modes available (choose ONE):
    //
    // MODE 1: ADAPTIVE REGION-BASED (RECOMMENDED for keyhole)
    //   - Different velocity limits for interface/bulk/solid/gas
    //   - Interface: v_target=0.5 (10 m/s) * recoil_boost=1.5 = 15 m/s max
    //   - Bulk liquid: v_target=0.3 (6 m/s) for Marangoni
    //   - Solid: v=0 (Darcy handles this)
    //
    // MODE 2: GRADUAL SCALING (simpler, uniform limit)
    //   - Single v_target for all cells
    //   - Good for moderate deformation studies
    //
    // MODE 3: HARD CFL LIMIT (most conservative)
    //   - Traditional CFL cutoff
    //   - Best stability, least deformation
    //
    // -------------------------------------------------------------------------
    // CURRENT SELECTION: MODE 1 - ADAPTIVE REGION-BASED
    // -------------------------------------------------------------------------
    config.cfl_use_adaptive = true;              // Enable adaptive region-based limiting
    config.cfl_v_target_interface = 0.55f;       // Interface: 11 m/s (recoil pressure)
    config.cfl_v_target_bulk = 0.35f;            // Bulk: 7 m/s (Marangoni convection)
    config.cfl_recoil_boost_factor = 2.0f;       // 2x boost for z-dominant forces (keyhole)
    config.cfl_interface_threshold_lo = 0.01f;   // Interface: fill > 0.01
    config.cfl_interface_threshold_hi = 0.99f;   // Interface: fill < 0.99
    config.cfl_force_ramp_factor = 0.95f;        // Start limiting at 95% of target
    //
    // Alternative: MODE 2 - GRADUAL SCALING (uncomment to use)
    // config.cfl_use_adaptive = false;
    // config.cfl_use_gradual_scaling = true;
    // config.cfl_velocity_target = 0.5f;        // 10 m/s uniform limit
    // config.cfl_force_ramp_factor = 0.98f;     // Very late ramp
    //
    // Alternative: MODE 3 - HARD CFL LIMIT (uncomment to use)
    // config.cfl_use_adaptive = false;
    // config.cfl_use_gradual_scaling = false;
    // config.cfl_limit = 0.6f;                  // Direct CFL cap

    // Surface tension properties
    config.surface_tension_coeff = 1.65f;          // N/m at T_melt
    config.dsigma_dT = -0.26e-3f;                  // N/(m·K)

    // Laser parameters
    // HIGH POWER MODE: 300W for keyhole simulation with recoil pressure
    // Recoil pressure activates at T > 3033K (T_boil - 500K for Ti6Al4V)
    // At 300W, expect T > 3500K (boiling) within ~10 μs
    config.laser_power = 300.0f;                   // 300W for keyhole mode
    config.laser_spot_radius = 50.0e-6f;           // 50 μm
    config.laser_absorptivity = 0.35f;             // 35% absorption
    config.laser_penetration_depth = 10.0e-6f;     // 10 μm
    // Laser shutoff time: Calculated from scan distance / velocity
    // Scan distance = 260 μm (from 80 μm to 340 μm), velocity = 0.5 m/s
    // Shutoff time = 260e-6 / 0.5 = 520 μs
    config.laser_shutoff_time = 520.0e-6f;  // Laser OFF at 520 μs (covers 65% of domain)

    // =========================================================================
    // LASER SCANNING CONFIGURATION (Optimized Nov 22)
    // =========================================================================
    // Design principle: Maximize domain utilization while maintaining buffers
    //
    // Domain X = 400 μm:
    //   - Front buffer: 80 μm (2x laser radius, thermal diffusion zone)
    //   - Scan track:   260 μm (65% utilization, main observation region)
    //   - Rear buffer:  60 μm (1.2x laser radius, solidification zone)
    //
    // Scan time = 260 μm / 0.5 m/s = 520 μs
    // Total sim = 520 μs (scan) + 180 μs (cooling) = 700 μs
    config.laser_start_x = 80.0e-6f;               // Start 80 μm from left edge (buffer = 2x spot radius)
    config.laser_start_y = 150.0e-6f;              // Y center (300 μm / 2)
    config.laser_scan_vx = 0.5f;                   // Scan at 0.5 m/s (realistic LPBF)
    config.laser_scan_vy = 0.0f;                   // Straight line scan

    // Boundary conditions
    config.boundary_type = 0;  // Periodic boundaries (FUTURE: add adiabatic option)

    // Thermal boundary conditions (散热参数优化 - Nov 23)
    // LPBF water-cooled substrate: h_conv = 5000-50000 W/(m²·K)
    // Previous value 100 W/(m²·K) was natural convection level (too low for LPBF)
    // New value 10000 W/(m²·K) represents typical water-cooled baseplate
    config.enable_radiation_bc = true;       // Enable Stefan-Boltzmann radiation
    config.enable_substrate_cooling = true;  // Keep substrate cooling
    config.substrate_h_conv = 10000.0f;      // Water-cooled substrate (was 100, natural convection)
    config.material.emissivity = 0.25f;      // Ti6Al4V typical emissivity

    // =========================================================================
    // SIMULATION CONTROL (Optimized Nov 22)
    // =========================================================================
    // Time budget:
    //   - Laser ON:  0 - 520 μs (scanning phase)
    //   - Laser OFF: 520 - 700 μs (cooling/solidification observation)
    //   - Total: 700 μs = 7000 steps @ dt=0.1 μs
    //
    // Output strategy:
    //   - interval=50 steps (5 μs per frame) → 140 frames total
    //   - VTK file size ~20 MB/frame → ~2.8 GB total
    int num_steps = 7000;  // Default: 700 μs (520 μs scan + 180 μs cooling)
    int output_interval = 50;   // Output every 50 steps (5 μs per frame)
    std::string output_dir = "lpbf_realistic";  // Output directory

    // =========================================================================
    // LOAD CONFIGURATION FILE (if provided)
    // =========================================================================
    // NEW: Parse command-line arguments and load config file
    // This happens AFTER setting defaults, so any missing parameters in
    // config file will use the hardcoded defaults above (backward compatible)

    std::string config_file;
    config::ConfigMetadata metadata;

    // Parse command-line arguments
    if (!config::parseCommandLineArgs(argc, argv, config_file, config, num_steps, output_dir)) {
        // Help was requested, exit gracefully
        return 0;
    }

    // Load config file if specified
    if (!config_file.empty()) {
        std::cout << "Loading configuration from: " << config_file << "\n\n";
        if (!config::loadLPBFConfig(config_file, config, num_steps, output_interval, output_dir, &metadata)) {
            std::cerr << "ERROR: Failed to load configuration file\n";
            std::cerr << "Falling back to embedded defaults\n\n";
        } else {
            std::cout << "Configuration loaded successfully\n\n";
        }
    }

    // =========================================================================
    // PRINT CONFIGURATION SUMMARY
    // =========================================================================

    if (!metadata.name.empty()) {
        // Print detailed config summary if loaded from file
        config::printConfigSummary(config, num_steps, output_interval, output_dir, &metadata);
    } else {
        // Print minimal summary for default configuration (backward compatible output)
        std::cout << "Configuration:\n";
        std::cout << "  Domain: " << config.nx << " x " << config.ny << " x " << config.nz << " cells\n";
        std::cout << "  Physical size: " << config.nx * config.dx * 1e6 << " x "
                  << config.ny * config.dx * 1e6 << " x " << config.nz * config.dx * 1e6 << " μm\n";
        std::cout << "  Time step: " << config.dt * 1e6 << " μs\n";
        std::cout << "  Total simulation time: " << num_steps * config.dt * 1e6 << " μs\n";
        std::cout << "  Laser power: " << config.laser_power << " W\n";
        std::cout << "  Laser spot: " << config.laser_spot_radius * 1e6 << " μm\n";
        std::cout << "\n";
    }

    // =========================================================================
    // INITIALIZE SOLVER
    // =========================================================================

    std::cout << "Initializing MultiphysicsSolver...\n";
    physics::MultiphysicsSolver solver(config);

    // Initialize with room temperature and flat interface at TOP
    // CRITICAL FIX: Interface must be at TOP where laser creates temperature gradients
    // CORRECTED: VOF fill_level convention:
    //   - fill_level=1 (metal) for z < z_interface
    //   - fill_level=0 (gas) for z > z_interface
    // For LPBF: Metal must be at TOP where laser hits, thin gas layer above
    // Keyhole forms DOWNWARD into metal, not into pre-existing gas
    //
    // interface_height = 0.8 means:
    //   - Metal: z = 0 to 100 (200 μm) ← laser heats this surface
    //   - Gas:   z = 100 to 125 (50 μm) ← atmosphere above
    const float T_initial = 300.0f;  // K (room temperature)
    const float interface_height = 0.8f;  // Metal at bottom 80%, gas at top 20%
    solver.initialize(T_initial, interface_height);

    std::cout << "✓ Solver initialized\n\n";

    // =========================================================================
    // SET INITIAL CONDITIONS: COLD SOLID METAL
    // =========================================================================

    std::cout << "Setting initial conditions (cold solid metal)...\n";

    const int num_cells = config.nx * config.ny * config.nz;

    // Initial liquid fraction: ZERO (all solid)
    // The liquid fraction will increase as laser heats the material
    std::vector<float> h_liquid_fraction(num_cells, 0.0f);

    // Exception: Bottom 7 cells (14 μm) are solid substrate that never melts
    const int z_substrate = 7;  // Bottom 7 layers
    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                if (k < z_substrate) {
                    h_liquid_fraction[idx] = 0.0f;  // Substrate (always solid)
                } else {
                    h_liquid_fraction[idx] = 0.0f;  // Initially solid, will melt under laser
                }
            }
        }
    }

    float* d_lf;
    cudaMalloc(&d_lf, num_cells * sizeof(float));
    cudaMemcpy(d_lf, h_liquid_fraction.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticLiquidFraction(d_lf);
    cudaFree(d_lf);

    std::cout << "✓ Initial conditions set:\n";
    std::cout << "  - Temperature: " << T_initial << " K (room temperature)\n";
    std::cout << "  - Liquid fraction: 0.0 (all solid)\n";
    std::cout << "  - Substrate: Bottom " << z_substrate << " layers (" << z_substrate * config.dx * 1e6 << " μm)\n";
    std::cout << "\n";

    // =========================================================================
    // TIME INTEGRATION LOOP
    // =========================================================================

    std::cout << "Starting time integration...\n";
    std::cout << "Output directory: " << output_dir << "/\n\n";

    // Create output directory
    createDirectory(output_dir);

    // Allocate host arrays for visualization
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    std::vector<float> h_fill(num_cells);           // VOF fill fraction
    std::vector<float> h_liquid_frac(num_cells);    // TRUE liquid fraction from phase change solver
    std::vector<float> h_phase(num_cells);

    std::cout << "Progress:\n";
    std::cout << "─────────────────────────────────────────────────────────\n";
    std::cout << "  Step      Time [μs]   T_max [K]   v_max [mm/s]\n";
    std::cout << "─────────────────────────────────────────────────────────\n";

    // Simulation loop
    for (int step = 0; step <= num_steps; ++step) {
        // VTK output and progress reporting
        if (step % output_interval == 0) {
            // Get data from GPU
            const float* d_T = solver.getTemperature();
            const float* d_vx = solver.getVelocityX();
            const float* d_vy = solver.getVelocityY();
            const float* d_vz = solver.getVelocityZ();
            const float* d_f = solver.getFillLevel();
            const float* d_lf = solver.getLiquidFraction();  // TRUE liquid fraction from phase change solver

            cudaMemcpy(h_temperature.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ux.data(), d_vx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_vy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_vz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fill.data(), d_f, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_liquid_frac.data(), d_lf, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            // Compute phase state from temperature (simplified)
            const float T_solidus = config.material.T_solidus;
            const float T_liquidus = config.material.T_liquidus;

            for (size_t i = 0; i < num_cells; ++i) {
                float T = h_temperature[i];
                if (T < T_solidus) {
                    h_phase[i] = 0.0f;  // Solid
                } else if (T > T_liquidus) {
                    h_phase[i] = 2.0f;  // Liquid
                } else {
                    h_phase[i] = 1.0f;  // Mushy
                }
            }

            // Compute statistics
            float T_max = 0.0f, T_min = 1e10f, v_max = 0.0f;
            int num_liquid = 0, num_solid = 0, num_mushy = 0;

            for (size_t i = 0; i < num_cells; ++i) {
                T_max = std::max(T_max, h_temperature[i]);
                T_min = std::min(T_min, h_temperature[i]);
                float v = sqrtf(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                v_max = std::max(v_max, v);

                if (h_phase[i] < 0.5f) num_solid++;
                else if (h_phase[i] > 1.5f) num_liquid++;
                else num_mushy++;
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
                output_dir + "/lpbf", step);

            io::VTKWriter::writeStructuredGridWithVectors(
                filename,
                h_temperature.data(),
                h_liquid_frac.data(),  // USE TRUE liquid fraction from phase change solver
                h_phase.data(),
                h_fill.data(),         // VOF fill level for free surface tracking
                h_ux.data(), h_uy.data(), h_uz.data(),
                config.nx, config.ny, config.nz,
                config.dx, config.dx, config.dx
            );
        }

        // Step forward (skip on last iteration)
        if (step < num_steps) {
            solver.step(config.dt);
        }
    }

    std::cout << "─────────────────────────────────────────────────────────\n\n";

    // =========================================================================
    // FINAL STATISTICS
    // =========================================================================

    std::cout << "✓ Simulation complete!\n";
    std::cout << "  Output files: " << output_dir << "/lpbf_*.vtk\n";
    std::cout << "  Files: " << (num_steps / output_interval + 1) << " frames\n\n";

    std::cout << "Final statistics:\n";

    // Get final fields
    const float* d_temp_final = solver.getTemperature();
    std::vector<float> h_temp_final(num_cells);
    cudaMemcpy(h_temp_final.data(), d_temp_final, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute statistics
    float T_min = h_temp_final[0], T_max = h_temp_final[0];
    int num_liquid = 0, num_solid = 0, num_mushy = 0;

    const float T_solidus = config.material.T_solidus;
    const float T_liquidus = config.material.T_liquidus;

    for (int i = 0; i < num_cells; ++i) {
        T_min = std::min(T_min, h_temp_final[i]);
        T_max = std::max(T_max, h_temp_final[i]);

        if (h_temp_final[i] < T_solidus) num_solid++;
        else if (h_temp_final[i] > T_liquidus) num_liquid++;
        else num_mushy++;
    }

    std::cout << "  Temperature range: " << T_min << " - " << T_max << " K\n";
    std::cout << "  Phase distribution:\n";
    std::cout << "    Solid:  " << num_solid << " cells (" << 100.0f * num_solid / num_cells << "%)\n";
    std::cout << "    Mushy:  " << num_mushy << " cells (" << 100.0f * num_mushy / num_cells << "%)\n";
    std::cout << "    Liquid: " << num_liquid << " cells (" << 100.0f * num_liquid / num_cells << "%)\n";

    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  Simulation completed successfully!         \n";
    std::cout << "==============================================\n";

    std::cout << "\nParaView Visualization:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "1. Open ParaView:\n";
    std::cout << "   paraview " << output_dir << "/lpbf_*.vtk\n\n";
    std::cout << "2. Color by Temperature to see laser heating\n\n";
    std::cout << "3. Add Glyph filter for velocity arrows:\n";
    std::cout << "   Filters → Glyph → Vectors: Velocity\n\n";
    std::cout << "4. Play animation to see melting and Marangoni flow!\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    return 0;
}
