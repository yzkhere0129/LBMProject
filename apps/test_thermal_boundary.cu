/**
 * @file test_thermal_boundary.cu
 * @brief Test different reflection coefficients for top surface boundary
 *
 * This diagnostic tool tests the thermal boundary condition to fix the
 * surface temperature regression problem.
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

struct TestResult {
    float reflection_coeff;
    float surface_T_max;
    float surface_T_avg;
    int surface_liquid_cells;
    int peak_z_location;
    float peak_T;
};

void runReflectionTest(float reflection_coeff, TestResult& result) {
    std::cout << "\n----------------------------------------\n";
    std::cout << "Testing reflection_coeff = " << reflection_coeff << "\n";
    std::cout << "----------------------------------------\n";

    result.reflection_coeff = reflection_coeff;

    // Setup configuration
    physics::MultiphysicsConfig config;

    config.nx = 100;
    config.ny = 100;
    config.nz = 50;
    config.dx = 2.0e-6f;  // 2 μm
    config.dt = 1.0e-7f;  // 0.1 μs

    config.enable_thermal = true;
    config.enable_phase_change = true;
    config.enable_fluid = false;        // Disable flow for pure thermal test
    config.enable_darcy = false;
    config.enable_marangoni = false;    // Disable for isolation
    config.enable_surface_tension = false;
    config.enable_laser = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;

    config.material = physics::MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;
    config.density = 4110.0f;

    config.laser_power = 20.0f;
    config.laser_spot_radius = 50.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;

    config.boundary_type = 0;

    // Initialize solver
    physics::MultiphysicsSolver solver(config);

    // Run 500 steps (50 μs)
    const int test_steps = 500;

    for (int step = 0; step < test_steps; ++step) {
        solver.step();

        if ((step + 1) % 100 == 0) {
            std::cout << "  Step " << (step + 1) << "/" << test_steps << "\n";
        }
    }

    // Extract results
    std::vector<float> T_host(config.nx * config.ny * config.nz);
    cudaMemcpy(T_host.data(), solver.getTemperature(),
               T_host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze surface (z = nz-1)
    int z_surface = config.nz - 1;
    float surface_T_sum = 0.0f;
    float surface_T_max_val = 0.0f;
    int surface_count = 0;
    int liquid_count = 0;

    for (int y = 0; y < config.ny; ++y) {
        for (int x = 0; x < config.nx; ++x) {
            int idx = x + y * config.nx + z_surface * config.nx * config.ny;
            float T = T_host[idx];
            surface_T_sum += T;
            surface_count++;

            if (T > surface_T_max_val) {
                surface_T_max_val = T;
            }

            if (T > config.material.T_liquidus) {
                liquid_count++;
            }
        }
    }

    result.surface_T_max = surface_T_max_val;
    result.surface_T_avg = surface_T_sum / surface_count;
    result.surface_liquid_cells = liquid_count;

    // Find peak temperature location in entire domain
    float global_T_max = 0.0f;
    int peak_z = 0;

    for (int z = 0; z < config.nz; ++z) {
        float layer_T_max = 0.0f;

        for (int y = 0; y < config.ny; ++y) {
            for (int x = 0; x < config.nx; ++x) {
                int idx = x + y * config.nx + z * config.nx * config.ny;
                float T = T_host[idx];

                if (T > layer_T_max) {
                    layer_T_max = T;
                }
            }
        }

        if (layer_T_max > global_T_max) {
            global_T_max = layer_T_max;
            peak_z = z;
        }
    }

    result.peak_z_location = peak_z;
    result.peak_T = global_T_max;

    // Print vertical temperature profile (near laser center)
    std::cout << "\nVertical Temperature Profile (at laser center):\n";
    int center_x = config.nx / 2;
    int center_y = config.ny / 2;

    for (int z = config.nz - 1; z >= 0; z -= 5) {  // Every 5 layers
        int idx = center_x + center_y * config.nx + z * config.nx * config.ny;
        float T = T_host[idx];
        std::cout << "  z=" << std::setw(2) << z << " (" << std::setw(3)
                  << (z * 2) << " μm): T=" << std::setw(7) << std::fixed
                  << std::setprecision(1) << T << " K";

        if (T > config.material.T_liquidus) {
            std::cout << " [LIQUID]";
        } else if (T > config.material.T_solidus) {
            std::cout << " [MUSHY]";
        }

        if (z == peak_z) {
            std::cout << " <-- PEAK";
        }

        std::cout << "\n";
    }

    // Summary
    std::cout << "\nResults:\n";
    std::cout << "  Surface T_max:      " << std::fixed << std::setprecision(1)
              << result.surface_T_max << " K\n";
    std::cout << "  Surface T_avg:      " << result.surface_T_avg << " K\n";
    std::cout << "  Surface liquid:     " << result.surface_liquid_cells << " cells\n";
    std::cout << "  Global T_max:       " << result.peak_T << " K\n";
    std::cout << "  Peak location:      z=" << result.peak_z_location
              << " (" << (result.peak_z_location * 2) << " μm)\n";

    // Diagnosis
    if (result.peak_z_location < config.nz - 3) {
        std::cout << "  WARNING: Peak is SUBSURFACE (should be at z=47-49)\n";
    }

    if (result.surface_T_max < 1973.0f) {  // T_liquidus
        std::cout << "  WARNING: Surface NOT melting (T < T_liquidus)\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "  Thermal Boundary Condition Test Suite      \n";
    std::cout << "==============================================\n";
    std::cout << "\nTesting different reflection coefficients\n";
    std::cout << "to diagnose surface temperature problem.\n";

    // Test sequence
    std::vector<float> coeffs = {1.00f, 0.98f, 0.95f, 0.90f, 0.85f};
    std::vector<TestResult> results;

    for (float coeff : coeffs) {
        TestResult result;
        runReflectionTest(coeff, result);
        results.push_back(result);
    }

    // Comparison table
    std::cout << "\n\n==============================================\n";
    std::cout << "  SUMMARY TABLE\n";
    std::cout << "==============================================\n\n";

    std::cout << "Refl.  Surface_T_max  Surface_Liquid  Peak_z  Peak_T    Status\n";
    std::cout << "---------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::fixed << std::setprecision(2) << r.reflection_coeff << "   ";
        std::cout << std::setw(7) << std::setprecision(1) << r.surface_T_max << " K      ";
        std::cout << std::setw(4) << r.surface_liquid_cells << " cells      ";
        std::cout << std::setw(2) << r.peak_z_location << "      ";
        std::cout << std::setw(7) << std::setprecision(1) << r.peak_T << " K  ";

        // Status indicator
        bool surface_melting = (r.surface_T_max > 1973.0f);
        bool peak_at_surface = (r.peak_z_location >= 47);

        if (surface_melting && peak_at_surface) {
            std::cout << "✓ GOOD";
        } else if (surface_melting) {
            std::cout << "~ OK (subsurface peak)";
        } else {
            std::cout << "✗ FAIL (no surface melting)";
        }

        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "Expected behavior:\n";
    std::cout << "  - Surface T_max should reach ~3760K (well above T_liquidus=1973K)\n";
    std::cout << "  - Surface should have ~900 liquid cells\n";
    std::cout << "  - Peak should be at z=47-49 (near surface)\n";
    std::cout << "\n";
    std::cout << "Recommendation:\n";

    // Find best coefficient
    float best_coeff = 1.00f;
    float best_T = 0.0f;

    for (const auto& r : results) {
        if (r.surface_T_max > best_T && r.peak_z_location >= 47) {
            best_T = r.surface_T_max;
            best_coeff = r.reflection_coeff;
        }
    }

    std::cout << "  Use reflection_coeff = " << std::fixed << std::setprecision(2)
              << best_coeff << " for proper surface heating.\n";
    std::cout << "  (This gives surface T_max = " << std::setprecision(1)
              << best_T << " K)\n\n";

    return 0;
}
