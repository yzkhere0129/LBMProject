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

using namespace lbm;

// Helper: Create directory
void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "  Realistic LPBF with Marangoni Convection   \n";
    std::cout << "==============================================\n\n";

    // =========================================================================
    // CONFIGURATION: Realistic LPBF Parameters
    // =========================================================================

    physics::MultiphysicsConfig config;

    // Domain size (200 x 200 x 100 μm)
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
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

    // Material: Ti6Al4V
    config.material = physics::MaterialDatabase::getTi6Al4V();

    // Thermal properties
    config.thermal_diffusivity = 5.8e-6f;  // Ti6Al4V liquid (m²/s)

    // Fluid properties
    config.kinematic_viscosity = 0.0333f;  // LATTICE UNITS (tau=0.6, stable)
    config.density = 4110.0f;              // Ti6Al4V liquid (kg/m³)

    // Darcy damping (AGGRESSIVELY REDUCED to fix divergence issue)
    // Original: 1.0e7 caused ∇·F_darcy = 5e14 N/m^4 >> ∇·F_marangoni
    // Reduced by 100x to test if divergence improves
    config.darcy_coefficient = 1.0e5f;  // Reduced from 1e7

    // Surface tension properties
    config.surface_tension_coeff = 1.65f;          // N/m at T_melt
    config.dsigma_dT = -0.26e-3f;                  // N/(m·K)

    // Laser parameters
    // CRITICAL FIX: Reduced laser power for small domain (200x200x100 μm)
    // Real LPBF uses 200W but domain is mm-scale with infinite heat sink
    // Our periodic domain has no heat escape - reduce power by 10x
    config.laser_power = 20.0f;                    // 20W (reduced from 200W)
    config.laser_spot_radius = 50.0e-6f;           // 50 μm
    config.laser_absorptivity = 0.35f;             // 35% absorption
    config.laser_penetration_depth = 10.0e-6f;     // 10 μm
    config.laser_shutoff_time = 50.0e-6f;          // Turn off laser at 50 μs

    // Boundary conditions
    config.boundary_type = 0;  // Periodic boundaries (FUTURE: add adiabatic option)

    // Simulation control
    const int num_steps = 1000;       // Total time steps (100 μs simulation) - STABLE REGIME
    const int output_interval = 25;   // Output every 25 steps (more frequent for flow analysis)

    std::cout << "Configuration:\n";
    std::cout << "  Domain: " << config.nx << " x " << config.ny << " x " << config.nz << " cells\n";
    std::cout << "  Physical size: " << config.nx * config.dx * 1e6 << " x "
              << config.ny * config.dx * 1e6 << " x " << config.nz * config.dx * 1e6 << " μm\n";
    std::cout << "  Time step: " << config.dt * 1e6 << " μs\n";
    std::cout << "  Total simulation time: " << num_steps * config.dt * 1e6 << " μs\n";
    std::cout << "  Laser power: " << config.laser_power << " W\n";
    std::cout << "  Laser spot: " << config.laser_spot_radius * 1e6 << " μm\n";
    std::cout << "\n";

    // =========================================================================
    // INITIALIZE SOLVER
    // =========================================================================

    std::cout << "Initializing MultiphysicsSolver...\n";
    physics::MultiphysicsSolver solver(config);

    // Initialize with room temperature and flat interface at 50% height
    const float T_initial = 300.0f;  // K (room temperature - KEY DIFFERENCE!)
    const float interface_height = 0.5f;  // Middle of domain
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
    std::cout << "Output directory: lpbf_realistic/\n\n";

    // Create output directory
    createDirectory("lpbf_realistic");

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
                "lpbf_realistic/lpbf", step);

            io::VTKWriter::writeStructuredGridWithVectors(
                filename,
                h_temperature.data(),
                h_liquid_frac.data(),  // TRUE liquid fraction from phase change solver
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
    std::cout << "  Output files: lpbf_realistic/lpbf_*.vtk\n";
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
    std::cout << "   paraview lpbf_realistic/lpbf_*.vtk\n\n";
    std::cout << "2. Color by Temperature to see laser heating\n\n";
    std::cout << "3. Add Glyph filter for velocity arrows:\n";
    std::cout << "   Filters → Glyph → Vectors: Velocity\n\n";
    std::cout << "4. Play animation to see melting and Marangoni flow!\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    return 0;
}
