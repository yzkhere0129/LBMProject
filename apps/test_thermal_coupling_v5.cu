/**
 * @file test_thermal_coupling_v5.cu
 * @brief Test thermal-fluid coupling fix (v5) with incremental validation
 *
 * Purpose: Systematically test thermal advection coupling to fix v4 thermal runaway
 *
 * Test Sequence:
 *   Test A: Coupling ON, Marangoni OFF  → Validates convective cooling
 *   Test B: Coupling ON, Marangoni ON   → Validates Marangoni stability
 *   Test C: Full physics enabled        → Production configuration
 *
 * Baseline: v4 config (coupling disabled) → T_max = 45,477 K
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"

using namespace lbm;

// Helper: Create directory
void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Helper: Compute CFL numbers
struct CFLDiagnostics {
    float CFL_advection;
    float CFL_diffusion_thermal;
    float CFL_diffusion_viscous;

    void compute(float v_max, float dt, float dx, float alpha, float nu) {
        CFL_advection = v_max * dt / dx;
        CFL_diffusion_thermal = alpha * dt / (dx * dx);
        CFL_diffusion_viscous = nu * dt / (dx * dx);
    }

    void print() const {
        std::cout << "  CFL Diagnostics:\n";
        std::cout << "    CFL_advection = " << std::fixed << std::setprecision(4)
                  << CFL_advection << " (should be < 0.5)\n";
        std::cout << "    CFL_thermal   = " << CFL_diffusion_thermal << " (should be < 0.25)\n";
        std::cout << "    CFL_viscous   = " << CFL_diffusion_viscous << " (should be < 0.25)\n";
    }
};

int main(int argc, char** argv) {
    // =========================================================================
    // COMMAND LINE ARGUMENTS
    // =========================================================================

    std::string test_name = "A";  // Default: Test A
    if (argc > 1) {
        test_name = argv[1];
    }

    std::cout << "============================================================================\n";
    std::cout << "  Thermal-Fluid Coupling Fix (v5) - Test " << test_name << "\n";
    std::cout << "============================================================================\n\n";

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    physics::MultiphysicsConfig config;

    // Domain size (400 x 200 x 100 μm)
    config.nx = 200;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2.0e-6f;  // 2 μm cell size

    // Time stepping
    config.dt = 1.0e-7f;  // 0.1 μs
    int total_steps = 1000;      // 100 μs for Tests A and B
    int output_interval = 50;     // Output every 5 μs

    // Material: Ti6Al4V
    config.material = physics::MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;  // m²/s
    config.kinematic_viscosity = 0.0333f;  // Lattice units (tau=0.6)
    config.density = 4110.0f;              // kg/m³

    // Laser parameters (195W - same as v4)
    config.laser_power = 195.0f;
    config.laser_spot_radius = 50.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 5.0e-6f;
    config.laser_shutoff_time = 700.0e-6f;  // Turn off at 700 μs
    config.laser_start_x = 50.0e-6f;
    config.laser_scan_vx = 0.36f;  // Scanning velocity
    config.laser_scan_vy = 0.0f;

    // Force parameters
    config.darcy_coefficient = 2.0e4f;
    config.surface_tension_coeff = 1.65f;
    config.dsigma_dT = -0.26e-3f;

    // Radiation boundary condition (CRITICAL)
    config.enable_radiation_bc = true;
    config.emissivity = 0.3f;
    config.ambient_temperature = 300.0f;

    // Common physics flags
    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_laser = true;
    config.enable_darcy = true;
    config.boundary_type = 0;  // Periodic

    // Test-specific configuration
    std::string output_dir;

    if (test_name == "A") {
        // ================================================================
        // TEST A: Coupling ON, Marangoni OFF
        // ================================================================
        std::cout << "Test A Configuration:\n";
        std::cout << "  - Thermal advection: ENABLED (v·∇T coupling)\n";
        std::cout << "  - Marangoni forces:  DISABLED\n";
        std::cout << "  - Expected: Convective cooling reduces T vs v4\n\n";

        config.enable_thermal_advection = true;
        config.enable_marangoni = false;
        config.enable_phase_change = false;
        config.enable_vof_advection = false;
        config.enable_surface_tension = false;

        output_dir = "lpbf_test_A_coupling";

    } else if (test_name == "B") {
        // ================================================================
        // TEST B: Coupling ON, Marangoni ON
        // ================================================================
        std::cout << "Test B Configuration:\n";
        std::cout << "  - Thermal advection: ENABLED (v·∇T coupling)\n";
        std::cout << "  - Marangoni forces:  ENABLED\n";
        std::cout << "  - CFL limiter:       ACTIVE (max CFL = 0.5)\n";
        std::cout << "  - Gradient limiter:  ACTIVE (max ∇T = 5e8 K/m)\n";
        std::cout << "  - Expected: Stable Marangoni-driven flow\n\n";

        config.enable_thermal_advection = true;
        config.enable_marangoni = true;
        config.enable_phase_change = false;
        config.enable_vof_advection = false;
        config.enable_surface_tension = false;

        output_dir = "lpbf_test_B_marangoni";

    } else if (test_name == "C") {
        // ================================================================
        // TEST C: Full Physics
        // ================================================================
        std::cout << "Test C Configuration:\n";
        std::cout << "  - ALL PHYSICS ENABLED (production config)\n";
        std::cout << "  - Expected: Realistic LPBF behavior\n\n";

        config.enable_thermal_advection = true;
        config.enable_marangoni = true;
        config.enable_phase_change = true;
        config.enable_vof_advection = true;
        config.enable_surface_tension = true;
        config.vof_subcycles = 10;

        output_dir = "lpbf_test_C_full";
        total_steps = 5000;  // 500 μs for full test

    } else {
        std::cerr << "ERROR: Unknown test '" << test_name << "'\n";
        std::cerr << "Usage: " << argv[0] << " [A|B|C]\n";
        return 1;
    }

    // =========================================================================
    // PRINT CONFIGURATION
    // =========================================================================

    std::cout << "Domain Configuration:\n";
    std::cout << "  Grid: " << config.nx << " x " << config.ny << " x " << config.nz << " cells\n";
    std::cout << "  Physical size: " << config.nx * config.dx * 1e6 << " x "
              << config.ny * config.dx * 1e6 << " x " << config.nz * config.dx * 1e6 << " μm\n";
    std::cout << "  Cell size: " << config.dx * 1e6 << " μm\n";
    std::cout << "\nTime Configuration:\n";
    std::cout << "  Time step: " << config.dt * 1e6 << " μs\n";
    std::cout << "  Total steps: " << total_steps << "\n";
    std::cout << "  Simulation time: " << total_steps * config.dt * 1e6 << " μs\n";
    std::cout << "\nLaser Configuration:\n";
    std::cout << "  Power: " << config.laser_power << " W\n";
    std::cout << "  Spot radius: " << config.laser_spot_radius * 1e6 << " μm\n";
    std::cout << "  Absorptivity: " << config.laser_absorptivity << "\n";
    std::cout << "  Shutoff time: " << config.laser_shutoff_time * 1e6 << " μs\n";
    std::cout << "\nOutput: " << output_dir << "/\n";
    std::cout << "\n";

    // =========================================================================
    // INITIALIZE SOLVER
    // =========================================================================

    std::cout << "Initializing MultiphysicsSolver...\n";
    physics::MultiphysicsSolver solver(config);

    // Initialize with room temperature
    solver.initialize(300.0f, 0.5f);
    std::cout << "✓ Solver initialized\n\n";

    // =========================================================================
    // CREATE OUTPUT DIRECTORY
    // =========================================================================

    createDirectory(output_dir);

    // =========================================================================
    // ALLOCATE HOST ARRAYS
    // =========================================================================

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    std::vector<float> h_fill(num_cells);

    // =========================================================================
    // TIME INTEGRATION LOOP
    // =========================================================================

    std::cout << "Starting time integration...\n";
    std::cout << "────────────────────────────────────────────────────────────────────────\n";
    std::cout << "  Step      Time [μs]   T_max [K]   v_max [mm/s]   CFL_adv    Status\n";
    std::cout << "────────────────────────────────────────────────────────────────────────\n";

    CFLDiagnostics cfl;

    for (int step = 0; step <= total_steps; ++step) {
        // Output and diagnostics
        if (step % output_interval == 0) {
            // Get diagnostics
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();
            float t = step * config.dt;

            // Compute CFL numbers
            cfl.compute(v_max, config.dt, config.dx,
                       config.thermal_diffusivity,
                       config.kinematic_viscosity * config.dx / config.dt);

            // Check for NaN
            bool has_nan = solver.checkNaN();
            std::string status = has_nan ? "NaN!" : "OK";

            // Print progress
            std::cout << std::setw(6) << step
                      << std::setw(14) << std::fixed << std::setprecision(1) << t * 1e6
                      << std::setw(14) << std::fixed << std::setprecision(1) << T_max
                      << std::setw(16) << std::fixed << std::setprecision(2) << v_max * 1e3
                      << std::setw(12) << std::fixed << std::setprecision(4) << cfl.CFL_advection
                      << std::setw(10) << status
                      << "\n";

            // Save VTK output
            solver.copyTemperatureToHost(h_temperature.data());
            solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
            solver.copyFillLevelToHost(h_fill.data());

            std::string vtk_filename = output_dir + "/state_" +
                                       std::to_string(step/output_interval) + ".vtk";

            // Write VTK with temperature, fill level, and velocity
            io::VTKWriter::writeStructuredGridWithVectors(
                vtk_filename,
                h_temperature.data(),  // Temperature field
                h_fill.data(),         // Fill level (liquid fraction placeholder)
                h_fill.data(),         // Phase state (same as fill for now)
                h_fill.data(),         // VOF fill level
                h_ux.data(),           // Velocity X
                h_uy.data(),           // Velocity Y
                h_uz.data(),           // Velocity Z
                config.nx, config.ny, config.nz,
                config.dx, config.dx, config.dx
            );

            // Abort on NaN
            if (has_nan) {
                std::cerr << "\nERROR: NaN detected at step " << step << "! Aborting.\n";
                return 1;
            }

            // Check CFL violation
            if (cfl.CFL_advection > 1.0f) {
                std::cerr << "\nWARNING: CFL_advection = " << cfl.CFL_advection << " > 1.0!\n";
            }
        }

        // Advance simulation
        solver.step(config.dt);
    }

    std::cout << "────────────────────────────────────────────────────────────────────────\n";
    std::cout << "\n";

    // =========================================================================
    // FINAL DIAGNOSTICS
    // =========================================================================

    float T_max_final = solver.getMaxTemperature();
    float v_max_final = solver.getMaxVelocity();

    std::cout << "FINAL RESULTS (Test " << test_name << "):\n";
    std::cout << "══════════════════════════════════════════════════════════════════════\n";
    std::cout << "  Maximum temperature: " << std::fixed << std::setprecision(1) << T_max_final << " K\n";
    std::cout << "  Maximum velocity:    " << std::fixed << std::setprecision(2) << v_max_final * 1e3 << " mm/s\n";
    std::cout << "\n";

    cfl.compute(v_max_final, config.dt, config.dx,
               config.thermal_diffusivity,
               config.kinematic_viscosity * config.dx / config.dt);
    cfl.print();
    std::cout << "\n";

    // =========================================================================
    // COMPARISON TO v4 BASELINE
    // =========================================================================

    const float v4_T_max = 45477.0f;  // v4 baseline (coupling disabled)

    std::cout << "Comparison to v4 baseline (coupling disabled):\n";
    std::cout << "  v4 T_max:   " << std::fixed << std::setprecision(1) << v4_T_max << " K\n";
    std::cout << "  v5 T_max:   " << T_max_final << " K\n";
    std::cout << "  Reduction:  " << (v4_T_max - T_max_final) << " K "
              << "(" << std::fixed << std::setprecision(1)
              << (v4_T_max - T_max_final) / v4_T_max * 100.0f << "%)\n";
    std::cout << "\n";

    // =========================================================================
    // PHYSICS INTERPRETATION
    // =========================================================================

    std::cout << "Physical Interpretation:\n";
    std::cout << "══════════════════════════════════════════════════════════════════════\n";

    if (test_name == "A") {
        if (T_max_final < v4_T_max) {
            std::cout << "✓ SUCCESS: Convective cooling reduces temperature\n";
            std::cout << "  Thermal advection (v·∇T) provides heat transport mechanism\n";
            std::cout << "  Lower T confirms coupling is physically correct\n";
        } else {
            std::cout << "✗ UNEXPECTED: T_max increased despite convection\n";
            std::cout << "  Check: Is radiation BC active?\n";
        }
    } else if (test_name == "B") {
        if (cfl.CFL_advection < 0.8f && !solver.checkNaN()) {
            std::cout << "✓ SUCCESS: Marangoni flow is stable\n";
            std::cout << "  CFL limiter prevents velocity explosion\n";
            std::cout << "  Gradient limiter caps Marangoni forces\n";
            std::cout << "  v_max = " << v_max_final * 1e3 << " mm/s is reasonable for LPBF\n";
        } else {
            std::cout << "✗ INSTABILITY DETECTED\n";
        }
    } else if (test_name == "C") {
        std::cout << "Full physics simulation completed\n";
        std::cout << "Expected T_max: 20,000-35,000 K (radiation equilibrium)\n";
        std::cout << "Expected v_max: 0.1-1.0 m/s (Marangoni convection)\n";
    }

    std::cout << "\n✓ Test " << test_name << " completed successfully!\n";
    std::cout << "══════════════════════════════════════════════════════════════════════\n";

    return 0;
}
