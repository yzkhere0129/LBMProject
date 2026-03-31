/**
 * @file benchmark_1d_stefan.cu
 * @brief 1D Stefan Problem Benchmark - Pure Phase Change Heat Conduction
 *
 * Tests: ThermalLBM + ESM phase change (enthalpy-porosity method)
 * Validates against analytical solution: x_i(t) = 2*lambda*sqrt(alpha*t)
 *
 * Setup:
 * - 100x1x1 domain (1D heat conduction)
 * - Initial: all solid at T0 < T_solidus
 * - BC: x=0 wall at T_wall > T_liquidus (Dirichlet)
 * - Physics: thermal conduction + ESM phase change only
 * - No fluid, no VOF, no Marangoni, no recoil
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>
#include "config/lpbf_config_loader.h"
#include "physics/material_properties.h"
#include "physics/multiphysics_solver.h"

using namespace lbm;
using namespace lbm::physics;

// ============================================================================
// Material: Simple phase change material (for analytical comparison)
// ============================================================================
MaterialProperties createStefanMaterial() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "Stefan_Test", sizeof(mat.name) - 1);

    // Simple constant properties
    mat.rho_solid  = 1.0f;      // Normalized density
    mat.rho_liquid = 1.0f;
    mat.cp_solid   = 1.0f;      // Normalized specific heat
    mat.cp_liquid  = 1.0f;
    mat.k_solid    = 1.0f;      // Normalized thermal conductivity
    mat.k_liquid   = 1.0f;
    mat.mu_liquid  = 0.01f;     // Not used (no fluid)

    // Phase change temperatures
    mat.T_solidus      = 0.5f;  // Normalized melting point
    mat.T_liquidus     = 0.5f;  // Sharp phase change (mushy zone = 0)
    mat.T_vaporization = 2.0f;  // Not used
    mat.L_fusion       = 1.0f;  // Normalized latent heat

    // Stefan number: St = Cp*(T_wall - T_melt) / L = 1*(1.0-0.5)/1.0 = 0.5
    // With T_wall = 1.0, T_melt = 0.5, L = 1.0

    return mat;
}

int main() {
    printf("\n");
    printf("============================================================\n");
    printf("  1D Stefan Problem Benchmark (Phase Change Validation)\n");
    printf("  Analytical: x_i(t) = 2*lambda*sqrt(alpha*t)\n");
    printf("============================================================\n\n");

    // ==================================================================
    // Domain configuration (1D: 100x1x1)
    // ==================================================================
    const int NX = 100, NY = 1, NZ = 1;
    const float dx = 1.0f;              // Normalized lattice spacing
    const float alpha = 1.0f;           // Normalized thermal diffusivity
    const float dt = dx * dx / (2.0f * alpha * 3.0f);  // CFL-safe dt
    const float T_wall = 1.0f;          // Wall temperature at x=0
    const float T_init = 0.0f;          // Initial temperature (solid)
    const float T_melt = 0.5f;          // Melting temperature

    // Stefan number
    MaterialProperties mat = createStefanMaterial();
    float St = mat.cp_solid * (T_wall - T_solidus()) / mat.L_fusion;

    // Compute lambda from transcendental equation: lambda*exp(lambda^2)*erf(lambda) = St/sqrt(pi)
    // Use Newton-Raphson iteration
    float lambda = std::sqrt(St / 2.0f);  // Initial guess
    for (int iter = 0; iter < 100; iter++) {
        float lambda2 = lambda * lambda;
        float exp_l2 = std::exp(lambda2);
        float erf_l = std::erf(lambda);
        float f = lambda * exp_l2 * erf_l - St / std::sqrt(M_PI);
        float df = exp_l2 * erf_l + lambda * exp_l2 * (2.0f * lambda * erf_l + 2.0f / std::sqrt(M_PI) * std::exp(-lambda2));
        if (std::abs(df) < 1e-12f) break;
        lambda -= f / df;
        if (std::abs(f) < 1e-10f) break;
    }

    printf("Configuration:\n");
    printf("  Domain: %dx%dx%d (1D heat conduction)\n", NX, NY, NZ);
    printf("  dx = %.4f, dt = %.6f\n", dx, dt);
    printf("  alpha = %.4f\n", alpha);
    printf("  T_wall = %.2f, T_init = %.2f, T_melt = %.2f\n", T_wall, T_init, T_melt);
    printf("  Stefan number St = %.4f\n", St);
    printf("  lambda (analytical) = %.6f\n", lambda);
    printf("  x_i(t) = 2*%.6f*sqrt(%.4f*t) = %.6f*sqrt(t)\n", lambda, alpha, 2.0f * lambda * std::sqrt(alpha));

    // ==================================================================
    // Solver configuration
    // ==================================================================
    MultiphysicsConfig config;
    config.nx = NX;
    config.ny = NY;
    config.nz = NZ;
    config.dx = dx;
    config.dt = dt;

    // Physics flags: ONLY thermal + phase change
    config.enable_thermal           = true;
    config.enable_phase_change      = true;
    config.enable_thermal_advection = false;  // No velocity coupling
    config.enable_fluid             = false;  // No fluid
    config.enable_vof               = false;  // No VOF
    config.enable_vof_advection     = false;
    config.enable_laser             = false;  // No laser
    config.enable_darcy             = false;  // No Darcy
    config.enable_marangoni         = false;  // No Marangoni
    config.enable_recoil_pressure   = false;  // No recoil
    config.enable_buoyancy          = false;
    config.enable_surface_tension   = false;

    // Thermal parameters
    config.thermal_diffusivity = alpha;
    config.density = mat.rho_solid;
    config.specific_heat = mat.cp_solid;
    config.material = mat;

    // Boundary conditions: Dirichlet at x=0, adiabatic elsewhere
    config.boundaries.thermal_x_min = ThermalBCType::DIRICHLET;
    config.boundaries.thermal_x_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_max = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min = ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature = T_wall;

    // Phase change: sharp interface (T_solidus = T_liquidus)
    config.enable_phase_change = true;

    // ==================================================================
    // Initialize solver
    // ==================================================================
    int num_cells = NX * NY * NZ;
    std::vector<float> h_T_init(num_cells, T_init);
    std::vector<float> h_fill(num_cells, 1.0f);  // All metal

    MultiphysicsSolver solver(config);
    solver.initialize(h_T_init.data(), h_fill.data());

    // ==================================================================
    // Simulation parameters
    // ==================================================================
    const float t_total = 100.0f * dt * 100;  // ~100 time units
    const int total_steps = static_cast<int>(t_total / dt);
    const int output_interval = 10;

    printf("\nSimulation:\n");
    printf("  Total steps: %d\n", total_steps);
    printf("  Output interval: %d steps\n", output_interval);

    // ==================================================================
    // Main loop with interface tracking
    // ==================================================================
    std::vector<float> h_T(num_cells);
    std::vector<float> h_fl(num_cells);

    printf("\nStep     t        x_i(LBM)  x_i(Analytical)  Error(%%)\n");
    printf("-----  --------  ---------  ---------------  --------\n");

    std::vector<float> time_data;
    std::vector<float> xi_lbm_data;
    std::vector<float> xi_analytical_data;

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;

        // Run simulation
        if (step > 0) {
            solver.step(dt);
        }

        // Output at intervals
        if (step % output_interval == 0) {
            // Copy temperature and liquid fraction to host
            solver.copyTemperatureToHost(h_T.data());
            solver.copyLiquidFractionToHost(h_fl.data());

            // Find interface position (fl = 0.5)
            float xi_lbm = 0.0f;
            for (int ix = 0; ix < NX - 1; ix++) {
                float fl_left = h_fl[ix];
                float fl_right = h_fl[ix + 1];
                if ((fl_left <= 0.5f && fl_right > 0.5f) || (fl_left >= 0.5f && fl_right < 0.5f)) {
                    // Linear interpolation
                    float frac = (0.5f - fl_left) / (fl_right - fl_left);
                    xi_lbm = (ix + frac) * dx;
                    break;
                }
            }

            // Analytical solution: x_i = 2*lambda*sqrt(alpha*t)
            float xi_analytical = 2.0f * lambda * std::sqrt(alpha * t);

            // Error
            float error = (xi_analytical > 1e-6f) ? 
                std::abs(xi_lbm - xi_analytical) / xi_analytical * 100.0f : 0.0f;

            printf("%5d  %8.4f  %9.4f  %15.4f  %8.2f\n",
                   step, t, xi_lbm, xi_analytical, error);

            time_data.push_back(t);
            xi_lbm_data.push_back(xi_lbm);
            xi_analytical_data.push_back(xi_analytical);
        }
    }

    // ==================================================================
    // Write results for Python validation
    // ==================================================================
    std::string csv_file = "benchmark_stefan_1d_results.csv";
    FILE* fp = fopen(csv_file.c_str(), "w");
    if (fp) {
        fprintf(fp, "t,xi_lbm,xi_analytical\n");
        for (size_t i = 0; i < time_data.size(); i++) {
            fprintf(fp, "%.6f,%.6f,%.6f\n", time_data[i], xi_lbm_data[i], xi_analytical_data[i]);
        }
        fclose(fp);
        printf("\nResults saved to: %s\n", csv_file.c_str());
    }

    // Final error summary
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int count = 0;
    for (size_t i = 0; i < time_data.size(); i++) {
        if (xi_analytical_data[i] > 1e-6f) {
            float error = std::abs(xi_lbm_data[i] - xi_analytical_data[i]) / xi_analytical_data[i] * 100.0f;
            max_error = std::max(max_error, error);
            avg_error += error;
            count++;
        }
    }
    avg_error /= count;

    printf("\n============================================================\n");
    printf("  VALIDATION SUMMARY\n");
    printf("============================================================\n");
    printf("  Max error:  %.2f %%\n", max_error);
    printf("  Avg error:  %.2f %%\n", avg_error);
    printf("  Status:     %s\n", (max_error < 5.0f) ? "PASS ✓" : "FAIL ✗");
    printf("============================================================\n\n");

    return 0;
}
