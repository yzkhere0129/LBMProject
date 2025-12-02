/**
 * @file test_laser_shutoff.cu
 * @brief Test laser shutoff time configuration (Fix 2)
 *
 * Test cases:
 * 1. shutoff_time = -1.0 (laser always on)
 * 2. shutoff_time = 0.0 (laser never turns on - edge case)
 * 3. shutoff_time = 25.0e-6 (normal mid-simulation shutoff)
 *
 * Success criteria:
 * - Laser behavior matches configuration
 * - Temperature rises when laser on, stabilizes when off
 * - No crashes or unexpected behavior
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

// Test function for different shutoff times
bool testShutoffTime(float shutoff_time, const std::string& test_name) {
    std::cout << "\n======================================================================\n";
    std::cout << "  TEST: " << test_name << "\n";
    std::cout << "  Shutoff time: " << shutoff_time << " s\n";
    std::cout << "======================================================================\n\n";

    physics::MultiphysicsConfig config;

    // Small fast domain
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 3.0e-6f;

    config.dt = 1.0e-7f;  // 0.1 μs time step

    // Physics
    config.enable_thermal = true;
    config.enable_phase_change = true;
    config.enable_fluid = false;  // Disable for speed
    config.enable_darcy = false;
    config.enable_marangoni = false;
    config.enable_surface_tension = false;
    config.enable_laser = true;  // LASER MUST BE ON
    config.enable_vof = false;
    config.enable_vof_advection = false;

    // Material
    config.material = physics::MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.density = 4110.0f;

    // Laser parameters
    config.laser_power = 25.0f;
    config.laser_spot_radius = 40.0e-6f;
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;
    config.laser_shutoff_time = shutoff_time;  // TEST PARAMETER

    config.boundary_type = 0;

    std::cout << "Initializing solver...\n";
    physics::MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);
    std::cout << "✓ Initialized\n\n";

    const int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);

    const int total_steps = 500;  // 50 μs total
    const int output_interval = 100;

    std::cout << "Running simulation (50 μs total)...\n";
    std::cout << "─────────────────────────────────────────────\n";
    std::cout << "  Step    Time[μs]   T_max[K]   dT/dt[K/μs]\n";
    std::cout << "─────────────────────────────────────────────\n";

    float prev_T_max = 300.0f;
    bool heating_phase_ok = false;
    bool cooling_phase_ok = false;

    for (int step = 0; step <= total_steps; ++step) {
        if (step % output_interval == 0) {
            const float* d_T = solver.getTemperature();
            cudaMemcpy(h_temperature.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float T_max = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                T_max = std::max(T_max, h_temperature[i]);
            }

            float time = step * config.dt;
            float dT_dt = (T_max - prev_T_max) / (output_interval * config.dt * 1e-6);  // K/μs

            std::cout << std::setw(6) << step
                      << std::setw(12) << std::fixed << std::setprecision(2) << time * 1e6
                      << std::setw(12) << std::fixed << std::setprecision(1) << T_max
                      << std::setw(15) << std::fixed << std::setprecision(1) << dT_dt
                      << "\n";

            // Verify behavior
            if (shutoff_time < 0.0f) {
                // Laser always on - should keep heating
                if (step > 0 && dT_dt > 10.0f) {
                    heating_phase_ok = true;
                }
            } else if (shutoff_time == 0.0f) {
                // Laser never on - temperature should stay constant
                if (step > 0 && fabs(dT_dt) < 5.0f) {
                    cooling_phase_ok = true;
                }
            } else {
                // Normal shutoff
                if (time < shutoff_time && step > 0 && dT_dt > 10.0f) {
                    heating_phase_ok = true;
                }
                if (time > shutoff_time && step > output_interval && dT_dt < -5.0f) {
                    cooling_phase_ok = true;
                }
            }

            prev_T_max = T_max;
        }

        if (step < total_steps) {
            solver.step(config.dt);
        }
    }

    std::cout << "─────────────────────────────────────────────\n\n";

    // Evaluate test result
    bool test_passed = false;
    std::string status_msg;

    if (shutoff_time < 0.0f) {
        test_passed = heating_phase_ok;
        status_msg = heating_phase_ok ?
            "✓ Laser stayed on throughout simulation" :
            "✗ Laser did not heat as expected";
    } else if (shutoff_time == 0.0f) {
        test_passed = cooling_phase_ok;
        status_msg = cooling_phase_ok ?
            "✓ Laser remained off (no heating observed)" :
            "✗ Unexpected heating occurred";
    } else {
        test_passed = (heating_phase_ok && cooling_phase_ok);
        if (test_passed) {
            status_msg = "✓ Laser turned on then off as configured";
        } else if (!heating_phase_ok) {
            status_msg = "✗ No heating detected before shutoff time";
        } else {
            status_msg = "✗ No cooling detected after shutoff time";
        }
    }

    std::cout << "Result: " << status_msg << "\n";
    return test_passed;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================\n";
    std::cout << "  LASER SHUTOFF TIME CONFIGURATION TEST SUITE\n";
    std::cout << "======================================================================\n";
    std::cout << "\nTesting Fix 2: Configurable laser shutoff time\n\n";

    int tests_passed = 0;
    int tests_total = 0;

    // Test 1: Laser always on
    tests_total++;
    if (testShutoffTime(-1.0f, "Laser Always On (shutoff = -1)")) {
        tests_passed++;
    }

    // Test 2: Laser never on (edge case)
    tests_total++;
    if (testShutoffTime(0.0f, "Laser Immediate Shutoff (shutoff = 0)")) {
        tests_passed++;
    }

    // Test 3: Normal shutoff at 25 μs
    tests_total++;
    if (testShutoffTime(25.0e-6f, "Normal Shutoff at 25 μs")) {
        tests_passed++;
    }

    // Final summary
    std::cout << "\n======================================================================\n";
    std::cout << "  FINAL TEST SUMMARY\n";
    std::cout << "======================================================================\n\n";
    std::cout << "Tests passed: " << tests_passed << " / " << tests_total << "\n\n";

    if (tests_passed == tests_total) {
        std::cout << "✓ ALL TESTS PASSED\n";
        std::cout << "Fix 2 (laser shutoff configuration) is working correctly.\n\n";
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED\n";
        std::cout << "Review the laser shutoff implementation.\n\n";
        return 1;
    }
}
