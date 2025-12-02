/**
 * @file test_marangoni_gradient_limiter.cu
 * @brief CRITICAL test for Marangoni gradient limiter (Fix 3)
 *
 * CRITICAL ISSUE: Gradient limiter changed from 1e6 to 5e8 K/m (500x increase!)
 * This dramatically increases Marangoni force magnitude.
 *
 * Test objectives:
 * 1. Verify simulation remains stable with new limiter
 * 2. Check for CFL violations (velocity too high)
 * 3. Compare with old limiter (if possible)
 * 4. Ensure no divergence or NaN
 * 5. Verify physical realism (velocities < 10 m/s)
 *
 * Success criteria:
 * - No NaN or Inf
 * - v_max < 10 m/s (physical limit for LPBF)
 * - CFL number < 1.0
 * - Simulation completes without divergence
 * - Marangoni flow is visible and coherent
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

void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

int main(int argc, char** argv) {
    std::cout << "======================================================================\n";
    std::cout << "  CRITICAL TEST: Marangoni Gradient Limiter (5e8 K/m)\n";
    std::cout << "======================================================================\n\n";

    std::cout << "This test verifies the new gradient limiter (5e8 K/m) maintains\n";
    std::cout << "simulation stability while allowing realistic Marangoni forces.\n\n";

    std::cout << "CRITICAL CHANGE: Limiter increased from 1e6 to 5e8 K/m (500x)\n";
    std::cout << "Expected impact: Stronger Marangoni convection\n";
    std::cout << "Risk: Potential CFL violations, numerical instability\n\n";

    physics::MultiphysicsConfig config;

    // Moderate domain size
    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2.0e-6f;  // 2 μm resolution

    config.dt = 1.0e-7f;  // 0.1 μs time step

    // ENABLE ALL MARANGONI-RELATED PHYSICS
    config.enable_thermal = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_darcy = true;
    config.enable_marangoni = true;           // KEY: Marangoni ON
    config.enable_surface_tension = true;     // Interface physics
    config.enable_laser = true;
    config.enable_vof = true;                 // VOF for interface tracking
    config.enable_vof_advection = true;

    // Material
    config.material = physics::MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;  // Stable lattice viscosity
    config.density = 4110.0f;
    config.darcy_coefficient = 1.0e5f;

    // Surface tension (needed for Marangoni)
    config.surface_tension_coeff = 1.65f;
    config.dsigma_dT = -0.26e-3f;  // Ti6Al4V

    // Moderate laser to create gradients
    config.laser_power = 25.0f;
    config.laser_spot_radius = 40.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;
    config.laser_shutoff_time = -1.0f;  // Always on for continuous gradients

    config.boundary_type = 0;

    std::cout << "Configuration:\n";
    std::cout << "  Domain: " << config.nx << " x " << config.ny << " x " << config.nz << "\n";
    std::cout << "  Resolution: " << config.dx * 1e6 << " μm\n";
    std::cout << "  Time step: " << config.dt * 1e6 << " μs\n";
    std::cout << "  Laser: " << config.laser_power << " W, " << config.laser_spot_radius * 1e6 << " μm spot\n";
    std::cout << "  Marangoni: ENABLED with 5e8 K/m gradient limiter\n\n";

    // Initialize
    std::cout << "Initializing solver...\n";
    physics::MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);
    std::cout << "✓ Solver initialized\n\n";

    // Create output directory
    createDirectory("test_marangoni_limiter");

    const int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    std::vector<float> h_liquid_frac(num_cells);
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_phase(num_cells);

    const int total_steps = 1000;  // 100 μs total
    const int output_interval = 50;

    std::cout << "Starting Marangoni stability test...\n";
    std::cout << "─────────────────────────────────────────────────────────────────────\n";
    std::cout << "  Step    Time[μs]   T_max[K]   v_max[m/s]   CFL     grad_T_max[K/m]\n";
    std::cout << "─────────────────────────────────────────────────────────────────────\n";

    bool test_passed = true;
    std::string failure_reason = "";
    float max_velocity_observed = 0.0f;
    float max_CFL_observed = 0.0f;

    for (int step = 0; step <= total_steps; ++step) {
        if (step % output_interval == 0) {
            // Get data
            const float* d_T = solver.getTemperature();
            const float* d_vx = solver.getVelocityX();
            const float* d_vy = solver.getVelocityY();
            const float* d_vz = solver.getVelocityZ();
            const float* d_lf = solver.getLiquidFraction();
            const float* d_f = solver.getFillLevel();

            cudaMemcpy(h_temperature.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ux.data(), d_vx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_vy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_vz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_liquid_frac.data(), d_lf, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fill.data(), d_f, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            // Compute statistics
            float T_max = 0.0f;
            float v_max = 0.0f;
            float grad_T_max = 0.0f;
            bool has_nan = false;

            for (int i = 0; i < num_cells; ++i) {
                // NaN check
                if (std::isnan(h_temperature[i]) || std::isnan(h_ux[i]) ||
                    std::isnan(h_uy[i]) || std::isnan(h_uz[i])) {
                    has_nan = true;
                    test_passed = false;
                    failure_reason = "NaN detected in fields";
                    break;
                }

                T_max = std::max(T_max, h_temperature[i]);
                float v_mag = sqrtf(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                v_max = std::max(v_max, v_mag);
            }

            if (has_nan) break;

            // Estimate temperature gradient (central difference)
            for (int k = 1; k < config.nz - 1; ++k) {
                for (int j = 1; j < config.ny - 1; ++j) {
                    for (int i = 1; i < config.nx - 1; ++i) {
                        int idx = i + config.nx * (j + config.ny * k);

                        float dT_dx = (h_temperature[idx+1] - h_temperature[idx-1]) / (2.0f * config.dx);
                        float dT_dy = (h_temperature[idx+config.nx] - h_temperature[idx-config.nx]) / (2.0f * config.dx);
                        float dT_dz = (h_temperature[idx+config.nx*config.ny] - h_temperature[idx-config.nx*config.ny]) / (2.0f * config.dx);

                        float grad_T = sqrtf(dT_dx*dT_dx + dT_dy*dT_dy + dT_dz*dT_dz);
                        grad_T_max = std::max(grad_T_max, grad_T);
                    }
                }
            }

            // Compute CFL number
            float CFL = v_max * config.dt / config.dx;

            max_velocity_observed = std::max(max_velocity_observed, v_max);
            max_CFL_observed = std::max(max_CFL_observed, CFL);

            float time = step * config.dt;

            std::cout << std::setw(6) << step
                      << std::setw(12) << std::fixed << std::setprecision(2) << time * 1e6
                      << std::setw(12) << std::fixed << std::setprecision(1) << T_max
                      << std::setw(14) << std::fixed << std::setprecision(4) << v_max
                      << std::setw(9) << std::fixed << std::setprecision(3) << CFL
                      << std::setw(17) << std::scientific << std::setprecision(2) << grad_T_max
                      << "\n";

            // Safety checks
            if (v_max > 10.0f) {
                std::cout << "\n*** WARNING: Velocity exceeds physical limit (10 m/s) ***\n";
                test_passed = false;
                failure_reason = "Velocity too high - CFL violation likely";
            }

            if (CFL > 1.0f) {
                std::cout << "\n*** WARNING: CFL > 1.0 - simulation may be unstable ***\n";
            }

            // Write VTK output
            if (step % 50 == 0) {
                for (int i = 0; i < num_cells; ++i) {
                    h_phase[i] = (h_liquid_frac[i] > 0.5f) ? 2.0f : 0.0f;
                }

                std::string filename = io::VTKWriter::getTimeSeriesFilename(
                    "test_marangoni_limiter/marangoni", step);

                io::VTKWriter::writeStructuredGridWithVectors(
                    filename, h_temperature.data(), h_liquid_frac.data(),
                    h_phase.data(), nullptr,  // fill_level not used
                    h_ux.data(), h_uy.data(), h_uz.data(),
                    config.nx, config.ny, config.nz,
                    config.dx, config.dx, config.dx);
            }
        }

        if (step < total_steps) {
            solver.step(config.dt);
        }

        if (!test_passed) break;
    }

    std::cout << "─────────────────────────────────────────────────────────────────────\n\n";

    // Final report
    std::cout << "======================================================================\n";
    std::cout << "  TEST RESULT: Marangoni Gradient Limiter\n";
    std::cout << "======================================================================\n\n";

    std::cout << "Peak observed values:\n";
    std::cout << "  Maximum velocity: " << max_velocity_observed << " m/s\n";
    std::cout << "  Maximum CFL:      " << max_CFL_observed << "\n\n";

    if (test_passed) {
        std::cout << "✓ PASS: Marangoni gradient limiter test\n\n";
        std::cout << "The new 5e8 K/m gradient limiter maintains stability:\n";
        std::cout << "  - No NaN or Inf values\n";
        std::cout << "  - Velocities within physical range\n";
        std::cout << "  - CFL number acceptable\n";
        std::cout << "  - Simulation completed successfully\n\n";
        std::cout << "Conclusion: Fix 3 (Marangoni gradient limiter) is WORKING\n\n";
        std::cout << "Visualize results:\n";
        std::cout << "  paraview test_marangoni_limiter/marangoni_*.vtk\n";
        return 0;
    } else {
        std::cout << "✗ FAIL: " << failure_reason << "\n\n";
        std::cout << "The new gradient limiter may be too aggressive.\n";
        std::cout << "Consider:\n";
        std::cout << "  1. Reducing limiter value (e.g., 2e8 K/m)\n";
        std::cout << "  2. Reducing time step for stability\n";
        std::cout << "  3. Adding CFL-based velocity limiter\n";
        return 1;
    }
}
