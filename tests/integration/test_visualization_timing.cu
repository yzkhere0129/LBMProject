/**
 * @file test_visualization_timing.cu
 * @brief Test that visualization app runs long enough to observe melting
 *
 * This test verifies that the simulation duration is sufficient to observe
 * phase change and liquid fraction in the output. It compares the actual
 * behavior of the visualization app parameters vs the integration test parameters.
 *
 * ROOT CAUSE: Visualization app runs 10,000 steps but melting starts AT step 10,000
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"

using namespace lbm;

class VisualizationTimingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // EXACT parameters from visualize_laser_heating.cu
        nx = 100;
        ny = 100;
        nz = 50;
        num_cells = nx * ny * nz;

        dx = 2e-6f;
        dy = 2e-6f;
        dz = 2e-6f;
        dt = 5e-10f;

        material = physics::MaterialDatabase::getTi6Al4V();

        // EXACT laser parameters from visualization app
        laser_power = 400.0f;
        spot_radius = 50e-6f;
        penetration_depth = 15e-6f;

        T_initial = 300.0f;

        size_t field_size = num_cells * sizeof(float);
        cudaMalloc(&d_heat_source, field_size);
        cudaMemset(d_heat_source, 0, field_size);

        system("mkdir -p test_output");
    }

    void TearDown() override {
        if (d_heat_source) cudaFree(d_heat_source);
    }

    int nx, ny, nz, num_cells;
    float dx, dy, dz, dt;
    physics::MaterialProperties material;
    float laser_power, spot_radius, penetration_depth;
    float T_initial;
    float* d_heat_source;
};

/**
 * Test 1: Verify that 10,000 steps is INSUFFICIENT to observe melting
 *
 * This test replicates the EXACT conditions in visualize_laser_heating.cu
 * with n_steps = 10000 and demonstrates that liquid fraction is still ZERO
 * at the end of the simulation.
 */
TEST_F(VisualizationTimingTest, TenThousandStepsInsufficientForMelting) {
    std::cout << "\n=== TEST 1: Verify 10,000 Steps is Insufficient ===\n";
    std::cout << "This replicates the visualization app bug where n_steps = 10000\n";
    std::cout << "Expected result: liquid fraction = 0 throughout simulation\n\n";

    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true, dt, dx);
    thermal.initialize(T_initial);

    ASSERT_TRUE(thermal.hasPhaseChange()) << "Phase change must be enabled";

    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // CRITICAL: Use EXACT visualization app parameter
    const int VIZ_APP_STEPS = 10000;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    float max_temp = T_initial;
    float max_liquid_fraction = 0.0f;
    int first_melting_step = -1;

    std::cout << "Running simulation for " << VIZ_APP_STEPS << " steps...\n";

    for (int step = 0; step <= VIZ_APP_STEPS; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        if (step % 1000 == 0) {
            float* h_temp = new float[num_cells];
            float* h_fl = new float[num_cells];
            thermal.copyTemperatureToHost(h_temp);
            thermal.copyLiquidFractionToHost(h_fl);

            float T_max = h_temp[0];
            float fl_max = 0.0f;
            int n_melting = 0;

            for (int i = 0; i < num_cells; ++i) {
                T_max = fmaxf(T_max, h_temp[i]);
                fl_max = fmaxf(fl_max, h_fl[i]);
                if (h_fl[i] > 0.01f) n_melting++;
            }

            max_temp = fmaxf(max_temp, T_max);
            max_liquid_fraction = fmaxf(max_liquid_fraction, fl_max);

            if (first_melting_step < 0 && fl_max > 0.01f) {
                first_melting_step = step;
            }

            std::cout << "  Step " << step << ": T_max = " << std::fixed
                      << std::setprecision(1) << T_max << " K, "
                      << "fl_max = " << std::setprecision(5) << fl_max
                      << ", melting cells: " << n_melting << "\n";

            delete[] h_temp;
            delete[] h_fl;
        }
    }

    std::cout << "\n--- RESULTS for n_steps = " << VIZ_APP_STEPS << " ---\n";
    std::cout << "Maximum temperature reached: " << max_temp << " K\n";
    std::cout << "Melting point (T_liquidus): " << material.T_liquidus << " K\n";
    std::cout << "Maximum liquid fraction: " << max_liquid_fraction << "\n";
    std::cout << "First melting observed at step: " <<
        (first_melting_step >= 0 ? std::to_string(first_melting_step) : "NEVER") << "\n";

    // CRITICAL ASSERTION: With 10,000 steps, liquid fraction should be ZERO or VERY LOW
    // This demonstrates the bug in the visualization app
    std::cout << "\n*** CRITICAL FINDING ***\n";
    if (max_liquid_fraction < 0.1f) {
        std::cout << "BUG CONFIRMED: Liquid fraction is " << max_liquid_fraction
                  << " after " << VIZ_APP_STEPS << " steps!\n";
        std::cout << "The visualization app stops BEFORE significant melting occurs.\n";
    } else {
        std::cout << "UNEXPECTED: Melting occurred at step " << first_melting_step << "\n";
        std::cout << "The visualization app MAY capture some melting.\n";
    }

    // Document the bug: liquid fraction is essentially zero with current viz app settings
    EXPECT_LT(max_liquid_fraction, 0.1f)
        << "With 10,000 steps, liquid fraction should be near zero (BUG CONFIRMATION)";
}

/**
 * Test 2: Verify that 12,000 steps IS SUFFICIENT to observe melting
 *
 * This test uses the integration test parameter of 12,000 steps and
 * demonstrates that liquid fraction DOES appear with sufficient time.
 */
TEST_F(VisualizationTimingTest, TwelveThousandStepsSufficientForMelting) {
    std::cout << "\n=== TEST 2: Verify 12,000 Steps is Sufficient ===\n";
    std::cout << "This uses the integration test parameter n_steps = 12000\n";
    std::cout << "Expected result: liquid fraction > 0.5 observed\n\n";

    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true, dt, dx);
    thermal.initialize(T_initial);

    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // CRITICAL: Use integration test parameter
    const int TEST_STEPS = 12000;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    float max_liquid_fraction = 0.0f;
    int first_melting_step = -1;

    std::cout << "Running simulation for " << TEST_STEPS << " steps...\n";

    for (int step = 0; step <= TEST_STEPS; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        if (step % 1000 == 0) {
            float* h_fl = new float[num_cells];
            thermal.copyLiquidFractionToHost(h_fl);

            float fl_max = 0.0f;
            int n_melting = 0;

            for (int i = 0; i < num_cells; ++i) {
                fl_max = fmaxf(fl_max, h_fl[i]);
                if (h_fl[i] > 0.01f) n_melting++;
            }

            max_liquid_fraction = fmaxf(max_liquid_fraction, fl_max);

            if (first_melting_step < 0 && fl_max > 0.01f) {
                first_melting_step = step;
            }

            std::cout << "  Step " << step << ": fl_max = " << std::setprecision(5)
                      << fl_max << ", melting cells: " << n_melting << "\n";

            delete[] h_fl;
        }
    }

    std::cout << "\n--- RESULTS for n_steps = " << TEST_STEPS << " ---\n";
    std::cout << "Maximum liquid fraction: " << max_liquid_fraction << "\n";
    std::cout << "First melting observed at step: " << first_melting_step << "\n";

    std::cout << "\n*** SUCCESS ***\n";
    std::cout << "With " << TEST_STEPS << " steps, significant melting is observed!\n";
    std::cout << "First melting at step " << first_melting_step
              << ", maximum fl = " << max_liquid_fraction << "\n";

    // Verify that melting DOES occur with sufficient time
    EXPECT_GT(max_liquid_fraction, 0.5f)
        << "With 12,000 steps, liquid fraction should exceed 0.5";

    EXPECT_GE(first_melting_step, 0)
        << "Melting should be observed during simulation";

    EXPECT_LE(first_melting_step, 11000)
        << "Melting should start before end of simulation";
}

/**
 * Test 3: Find the EXACT step when melting begins
 *
 * This test runs with fine-grained monitoring to identify the precise
 * time step when liquid fraction first becomes non-zero.
 */
TEST_F(VisualizationTimingTest, FindExactMeltingStartTime) {
    std::cout << "\n=== TEST 3: Find Exact Melting Start Time ===\n";
    std::cout << "Monitor every 100 steps to find when fl first becomes > 0\n\n";

    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true, dt, dx);
    thermal.initialize(T_initial);

    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    const int MAX_STEPS = 15000;
    const int CHECK_INTERVAL = 100;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    int first_melting_step = -1;
    bool melting_detected = false;

    std::cout << "Running simulation with fine-grained monitoring...\n";

    for (int step = 0; step <= MAX_STEPS; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        if (step % CHECK_INTERVAL == 0) {
            float* h_fl = new float[num_cells];
            thermal.copyLiquidFractionToHost(h_fl);

            float fl_max = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                fl_max = fmaxf(fl_max, h_fl[i]);
            }

            if (!melting_detected && fl_max > 0.001f) {
                first_melting_step = step;
                melting_detected = true;
                std::cout << "\n*** MELTING DETECTED at step " << step << " ***\n";
                std::cout << "    Liquid fraction = " << fl_max << "\n\n";
            }

            if (step % 1000 == 0) {
                std::cout << "  Step " << step << ": fl_max = " << fl_max << "\n";
            }

            delete[] h_fl;

            // Stop after detecting melting and running a bit longer
            if (melting_detected && step > first_melting_step + 2000) {
                break;
            }
        }
    }

    std::cout << "\n--- TIMING ANALYSIS ---\n";
    std::cout << "First melting detected at step: " << first_melting_step << "\n";
    std::cout << "Visualization app stops at step: 10000\n";
    std::cout << "Integration test runs to step: 12000\n\n";

    if (first_melting_step > 10000) {
        std::cout << "*** ROOT CAUSE CONFIRMED ***\n";
        std::cout << "Melting starts at step " << first_melting_step
                  << ", AFTER visualization app ends!\n";
        std::cout << "Recommendation: Set n_steps >= " << (first_melting_step + 2000)
                  << " to observe melting\n";
    } else {
        std::cout << "Melting starts BEFORE visualization app ends.\n";
        std::cout << "Check other potential issues (data copy, VTK write, etc.)\n";
    }

    EXPECT_GT(first_melting_step, 0) << "Melting should be detected";
}

/**
 * Test 4: Recommended n_steps for visualization app
 *
 * This test determines the optimal n_steps value to ensure liquid fraction
 * reaches at least 0.8 (nearly fully melted in center region).
 */
TEST_F(VisualizationTimingTest, RecommendedStepsForVisualization) {
    std::cout << "\n=== TEST 4: Determine Recommended n_steps ===\n";
    std::cout << "Find minimum steps needed for fl_max >= 0.8\n\n";

    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true, dt, dx);
    thermal.initialize(T_initial);

    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    const int MAX_STEPS = 20000;
    const float TARGET_FL = 0.8f;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    int recommended_steps = -1;

    std::cout << "Running simulation to find when fl_max >= " << TARGET_FL << "...\n";

    for (int step = 0; step <= MAX_STEPS; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        if (step % 500 == 0) {
            float* h_fl = new float[num_cells];
            thermal.copyLiquidFractionToHost(h_fl);

            float fl_max = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                fl_max = fmaxf(fl_max, h_fl[i]);
            }

            if (step % 1000 == 0) {
                std::cout << "  Step " << step << ": fl_max = "
                          << std::setprecision(3) << fl_max << "\n";
            }

            if (recommended_steps < 0 && fl_max >= TARGET_FL) {
                recommended_steps = step;
                std::cout << "\n*** TARGET REACHED at step " << step << " ***\n";
                std::cout << "    fl_max = " << fl_max << "\n";
                delete[] h_fl;
                break;
            }

            delete[] h_fl;
        }
    }

    std::cout << "\n--- RECOMMENDATION ---\n";
    if (recommended_steps > 0) {
        std::cout << "Minimum n_steps for fl >= " << TARGET_FL << ": "
                  << recommended_steps << "\n";
        std::cout << "Recommended n_steps (with safety margin): "
                  << (recommended_steps + 2000) << "\n\n";
        std::cout << "SUGGESTED FIX for visualize_laser_heating.cu:\n";
        std::cout << "  Change line 54 from:\n";
        std::cout << "    const int n_steps = 10000;\n";
        std::cout << "  To:\n";
        std::cout << "    const int n_steps = " << (recommended_steps + 2000) << ";\n";
    } else {
        std::cout << "Target not reached within " << MAX_STEPS << " steps.\n";
        std::cout << "Consider increasing laser power or reducing domain size.\n";
    }

    EXPECT_GT(recommended_steps, 0) << "Should reach target liquid fraction";
    EXPECT_GT(recommended_steps, 10000)
        << "Recommended steps should exceed current visualization app setting";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
