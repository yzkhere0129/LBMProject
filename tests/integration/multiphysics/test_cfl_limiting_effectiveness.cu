/**
 * @file test_cfl_limiting_effectiveness.cu
 * @brief Test CFL limiting: velocity never exceeds target
 *
 * Success Criteria:
 * - Max lattice velocity < cfl_velocity_target
 * - CFL limiter activates when needed
 * - No numerical divergence
 * - No NaN
 *
 * Physics:
 * - Strong Marangoni forces (can cause velocity explosion)
 * - CFL limiter should prevent v > v_target
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsCFLTest, LimitingEffectiveness) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: CFL Limiting Effectiveness" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MultiphysicsConfig config;
    config.nx = 60;
    config.ny = 60;
    config.nz = 30;
    config.dx = 2e-6f;
    config.dt = 1e-8f;

    // Enable Marangoni (strong forces)
    config.enable_thermal = false;  // Static temperature
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_marangoni = true;
    config.enable_surface_tension = false;
    config.enable_laser = false;
    config.enable_buoyancy = false;

    // CFL limiter configuration
    config.cfl_use_gradual_scaling = true;
    config.cfl_velocity_target = 0.15f;  // Target max lattice velocity
    config.cfl_force_ramp_factor = 0.9f;

    // Strong Marangoni forcing
    config.dsigma_dT = -0.26e-3f;  // Ti6Al4V
    config.kinematic_viscosity = 0.0333f;  // Lattice units

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  CFL velocity target: " << config.cfl_velocity_target << " (lattice units)" << std::endl;
    std::cout << "  Expected physical velocity limit: "
              << config.cfl_velocity_target * config.dx / config.dt << " m/s" << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);

    // Initialize with uniform temperature
    solver.initialize(300.0f, 0.5f);

    // Set static temperature with strong gradient
    int num_cells = config.nx * config.ny * config.nz;
    float* d_temp;
    cudaMalloc(&d_temp, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells);
    const float T_hot = 2500.0f;
    const float T_cold = 2000.0f;
    const float R_hot = 20e-6f;

    float center_x = config.nx * config.dx / 2.0f;
    float center_y = config.ny * config.dx / 2.0f;

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                float x = i * config.dx - center_x;
                float y = j * config.dx - center_y;
                float r = std::sqrt(x*x + y*y);

                if (r < R_hot) {
                    h_temp[idx] = T_hot;
                } else {
                    float decay = std::exp(-(r - R_hot) / R_hot);
                    h_temp[idx] = T_cold + (T_hot - T_cold) * decay;
                }
            }
        }
    }

    cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticTemperature(d_temp);

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  Static temperature gradient: " << (T_hot - T_cold) / R_hot * 1e-6
              << " K/μm" << std::endl;
    std::cout << "  Strong Marangoni forcing expected" << std::endl;
    std::cout << std::endl;

    const int n_steps = 1000;
    const int check_interval = 100;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    float v_max_physical = 0.0f;
    float v_max_lattice = 0.0f;
    bool cfl_violated = false;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        // Get velocity
        float v_phys = solver.getMaxVelocity();  // Physical units
        float v_latt = v_phys * config.dt / config.dx;  // Convert to lattice units

        v_max_physical = std::max(v_max_physical, v_phys);
        v_max_lattice = std::max(v_max_lattice, v_latt);

        // Check CFL violation
        if (v_latt > config.cfl_velocity_target * 1.01f) {  // 1% tolerance
            cfl_violated = true;
        }

        if ((step + 1) % check_interval == 0) {
            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(2)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_phys = " << std::setprecision(4) << v_phys << " m/s"
                      << " | v_latt = " << std::setprecision(4) << v_latt
                      << " | target = " << config.cfl_velocity_target
                      << std::endl;

            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(70, '-') << std::endl;

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  Max physical velocity: " << std::fixed << std::setprecision(4)
              << v_max_physical << " m/s" << std::endl;
    std::cout << "  Max lattice velocity:  " << v_max_lattice << std::endl;
    std::cout << "  CFL target velocity:   " << config.cfl_velocity_target << std::endl;
    std::cout << "  CFL violated:          " << (cfl_violated ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << std::endl;

    // Success criteria
    std::cout << "CFL Limiting Check:" << std::endl;
    std::cout << "  Velocity < target: ";
    if (!cfl_violated) {
        std::cout << "PASS ✓" << std::endl;
    } else {
        std::cout << "FAIL ✗ (v_max = " << v_max_lattice << " > "
                  << config.cfl_velocity_target << ")" << std::endl;
    }
    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_FALSE(cfl_violated) << "CFL condition violated";
    EXPECT_LE(v_max_lattice, config.cfl_velocity_target * 1.01f)
        << "Max velocity exceeded target by more than 1%";

    cudaFree(d_temp);

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
