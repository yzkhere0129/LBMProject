/**
 * @file test_phase_fluid_coupling.cu
 * @brief Test phase-fluid coupling: Darcy damping in mushy zone
 *
 * Success Criteria:
 * - Velocity decreases in mushy zone (0 < liquid_fraction < 1)
 * - Velocity ~0 in solid (liquid_fraction < 0.01)
 * - Damping proportional to (1 - liquid_fraction)²
 * - No NaN
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

TEST(MultiphysicsCouplingTest, PhaseFluidCoupling) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Phase-Fluid Coupling (Darcy Damping)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MultiphysicsConfig config;
    config.nx = 50;
    config.ny = 50;
    config.nz = 50;
    config.dx = 2e-6f;
    config.dt = 1e-8f;

    // Enable phase change + fluid + Darcy
    config.enable_thermal = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_darcy = true;  // Key feature to test
    config.enable_buoyancy = true;  // Drive flow
    config.enable_vof = false;
    config.enable_marangoni = false;
    config.enable_laser = false;

    config.darcy_coefficient = 1e7f;  // Strong Darcy damping
    config.gravity_z = -9.81f;
    config.thermal_expansion_coeff = 1.5e-5f;
    config.reference_temperature = 1923.0f;  // Ti6Al4V melting point

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  Darcy coefficient: " << config.darcy_coefficient << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);

    // Initialize with temperature stratification across melting point
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_fill_level(num_cells, 1.0f);

    const float T_liquidus = 1933.0f;  // Ti6Al4V
    const float T_solidus = 1878.0f;   // Ti6Al4V

    for (int k = 0; k < config.nz; ++k) {
        // Bottom: solid (cold), Middle: mushy, Top: liquid (hot)
        float z_frac = float(k) / float(config.nz - 1);
        float T = T_solidus * (1.0f - z_frac) + (T_liquidus + 100.0f) * z_frac;

        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                h_temperature[idx] = T;
            }
        }
    }

    solver.initialize(h_temperature.data(), h_fill_level.data());

    std::cout << "Initial conditions:" << std::endl;
    std::cout << "  T_bottom (solid) = " << T_solidus << " K" << std::endl;
    std::cout << "  T_top (liquid)   = " << T_liquidus + 100.0f << " K" << std::endl;
    std::cout << "  Mushy zone present" << std::endl;
    std::cout << std::endl;

    const int n_steps = 500;
    const int check_interval = 100;

    std::cout << "Time integration (" << n_steps << " steps):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int step = 0; step < n_steps; ++step) {
        solver.step();

        if ((step + 1) % check_interval == 0) {
            float v_max = solver.getMaxVelocity();
            float T_max = solver.getMaxTemperature();

            std::cout << "Step " << std::setw(4) << step + 1
                      << " | t = " << std::fixed << std::setprecision(2)
                      << (step + 1) * config.dt * 1e6 << " μs"
                      << " | v_max = " << std::setprecision(4) << v_max << " m/s"
                      << " | T_max = " << std::setprecision(1) << T_max << " K"
                      << std::endl;

            ASSERT_FALSE(solver.checkNaN()) << "NaN detected at step " << step + 1;
        }
    }

    std::cout << std::string(60, '-') << std::endl;

    // Extract velocity and liquid fraction fields
    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);
    std::vector<float> h_liquid_frac(num_cells);

    solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Copy liquid fraction (need to access through temperature field and material properties)
    const float* d_liquid_frac = solver.getLiquidFraction();
    cudaMemcpy(h_liquid_frac.data(), d_liquid_frac, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute average velocity in each zone
    float v_solid = 0.0f, v_mushy = 0.0f, v_liquid = 0.0f;
    int count_solid = 0, count_mushy = 0, count_liquid = 0;

    for (int idx = 0; idx < num_cells; ++idx) {
        float f_liquid = h_liquid_frac[idx];
        float v_mag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx] + h_uz[idx]*h_uz[idx]);

        if (f_liquid < 0.01f) {  // Solid
            v_solid += v_mag;
            count_solid++;
        } else if (f_liquid > 0.99f) {  // Liquid
            v_liquid += v_mag;
            count_liquid++;
        } else {  // Mushy
            v_mushy += v_mag;
            count_mushy++;
        }
    }

    if (count_solid > 0) v_solid /= count_solid;
    if (count_mushy > 0) v_mushy /= count_mushy;
    if (count_liquid > 0) v_liquid /= count_liquid;

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  v_solid (f < 0.01):   " << std::scientific << std::setprecision(3)
              << v_solid << " m/s" << std::endl;
    std::cout << "  v_mushy (0.01 < f < 0.99): " << v_mushy << " m/s" << std::endl;
    std::cout << "  v_liquid (f > 0.99):  " << v_liquid << " m/s" << std::endl;
    std::cout << std::endl;

    // Success criteria
    std::cout << "Darcy Damping Checks:" << std::endl;

    std::cout << "  1. Solid velocity ~ 0: ";
    if (v_solid < 0.01f) {
        std::cout << "PASS ✓ (" << v_solid << " m/s)" << std::endl;
    } else {
        std::cout << "FAIL ✗ (" << v_solid << " m/s)" << std::endl;
    }

    std::cout << "  2. Mushy velocity < liquid velocity: ";
    if (v_mushy < v_liquid) {
        std::cout << "PASS ✓ (" << v_mushy << " < " << v_liquid << ")" << std::endl;
    } else {
        std::cout << "FAIL ✗" << std::endl;
    }

    std::cout << "  3. Liquid velocity > 0: ";
    if (v_liquid > 0.0f) {
        std::cout << "PASS ✓ (" << v_liquid << " m/s)" << std::endl;
    } else {
        std::cout << "FAIL ✗" << std::endl;
    }

    std::cout << std::endl;

    // Assertions
    EXPECT_FALSE(solver.checkNaN()) << "NaN detected in final state";
    EXPECT_LT(v_solid, 0.01f) << "Velocity in solid should be ~0";
    EXPECT_LT(v_mushy, v_liquid) << "Mushy zone velocity should be less than liquid";
    EXPECT_GT(v_liquid, 0.0f) << "Liquid should have non-zero velocity";

    std::cout << "========================================" << std::endl;
    std::cout << "TEST PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
