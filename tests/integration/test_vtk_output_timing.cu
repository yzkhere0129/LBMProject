/**
 * @file test_vtk_output_timing.cu
 * @brief Test VTK output at critical time steps around melting onset
 *
 * This test verifies that VTK files written at steps near melting onset
 * correctly capture the liquid fraction values.
 *
 * KEY FINDINGS:
 * - Melting starts at step 9600
 * - By step 10000, fl_max = 0.995 (nearly fully melted in center)
 * - Visualization app outputs every 200 steps
 * - Last output is at step 10000
 *
 * This test checks if the VTK output at step 10000 captures the liquid fraction.
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

class VTKOutputTimingTest : public ::testing::Test {
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
 * Test: Replicate EXACT visualization app behavior
 *
 * Run to step 10000 with output every 200 steps (matching viz app).
 * Verify that VTK file at step 10000 contains non-zero liquid fraction.
 */
TEST_F(VTKOutputTimingTest, ReplicateVisualizationAppExactly) {
    std::cout << "\n=== Replicate Visualization App Exactly ===\n";
    std::cout << "Run parameters:\n";
    std::cout << "  n_steps = 10000\n";
    std::cout << "  output_interval = 200\n";
    std::cout << "  Expected: VTK at steps 0, 200, 400, ..., 9800, 10000\n\n";

    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);
    thermal.initialize(T_initial);

    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    const int n_steps = 10000;
    const int output_interval = 200;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    float* h_temperature = new float[num_cells];
    float* h_liquid_fraction = new float[num_cells];
    float* h_phase_state = new float[num_cells];

    std::cout << "Running simulation...\n";

    for (int step = 0; step <= n_steps; ++step) {
        // Compute laser heat source
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        // Add heat and evolve
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Output VTK files at specified intervals
        if (step % output_interval == 0) {
            // Copy temperature and liquid fraction to host
            thermal.copyTemperatureToHost(h_temperature);
            thermal.copyLiquidFractionToHost(h_liquid_fraction);

            // Calculate statistics
            float T_max = h_temperature[0];
            float fl_max = 0.0f;
            int n_melting = 0;

            // Compute phase state field
            for (int i = 0; i < num_cells; ++i) {
                float T = h_temperature[i];
                float fl = h_liquid_fraction[i];

                T_max = fmaxf(T_max, T);
                fl_max = fmaxf(fl_max, fl);

                if (fl > 0.01f) n_melting++;

                // Determine phase state
                if (fl < 0.01f) {
                    h_phase_state[i] = 0.0f;  // Solid
                } else if (fl > 0.99f) {
                    h_phase_state[i] = 2.0f;  // Liquid
                } else {
                    h_phase_state[i] = 1.0f;  // Mushy
                }
            }

            std::cout << "Step " << step << ": T_max = " << T_max
                      << " K, fl_max = " << fl_max
                      << ", melting cells: " << n_melting << "\n";

            // Write VTK file
            std::string filename = io::VTKWriter::getTimeSeriesFilename(
                "test_output/vtk_timing_test", step
            );

            io::VTKWriter::writeStructuredGrid(
                filename, h_temperature, h_liquid_fraction, h_phase_state,
                nx, ny, nz, dx, dy, dz,
                "Temperature", "LiquidFraction", "PhaseState"
            );
        }
    }

    // Now read back the VTK file at step 10000 and verify liquid fraction
    std::string final_vtk = "test_output/vtk_timing_test_010000.vtk";
    std::ifstream file(final_vtk);
    ASSERT_TRUE(file.is_open()) << "VTK file at step 10000 should exist";

    // Search for LiquidFraction field
    std::string line;
    bool found_field = false;
    while (std::getline(file, line)) {
        if (line.find("SCALARS LiquidFraction") != std::string::npos) {
            found_field = true;
            std::getline(file, line);  // Skip LOOKUP_TABLE
            break;
        }
    }

    ASSERT_TRUE(found_field) << "LiquidFraction field should exist in VTK";

    // Read liquid fraction values
    std::vector<float> fl_values;
    float value;
    while (file >> value && fl_values.size() < num_cells) {
        fl_values.push_back(value);
    }
    file.close();

    ASSERT_EQ(fl_values.size(), num_cells) << "Should read all liquid fraction values";

    // Analyze the values
    float fl_max_vtk = 0.0f;
    int n_nonzero = 0;
    for (float fl : fl_values) {
        fl_max_vtk = fmaxf(fl_max_vtk, fl);
        if (fl > 0.01f) n_nonzero++;
    }

    std::cout << "\n=== VTK File Analysis (step 10000) ===\n";
    std::cout << "Maximum liquid fraction in VTK: " << fl_max_vtk << "\n";
    std::cout << "Cells with fl > 0.01: " << n_nonzero << "\n";

    // Final assertions
    std::cout << "\n=== CRITICAL TEST ===\n";
    if (fl_max_vtk > 0.5f) {
        std::cout << "SUCCESS: VTK file DOES contain liquid fraction data!\n";
        std::cout << "The visualization app SHOULD show melting at step 10000.\n";
        std::cout << "\nPOSSIBLE ISSUE: User may be looking at wrong timestep in ParaView.\n";
        std::cout << "RECOMMENDATION: Check that ParaView is displaying the LAST timestep (step 10000).\n";
    } else if (fl_max_vtk > 0.01f) {
        std::cout << "PARTIAL: VTK file contains some liquid fraction, but it's small.\n";
        std::cout << "The visualization app captures the ONSET of melting.\n";
        std::cout << "\nRECOMMENDATION: Increase n_steps to 12000 for clearer visualization.\n";
    } else {
        std::cout << "BUG CONFIRMED: VTK file contains NO liquid fraction data!\n";
        std::cout << "This indicates a problem with the data pipeline.\n";
    }

    delete[] h_temperature;
    delete[] h_liquid_fraction;
    delete[] h_phase_state;

    EXPECT_GT(fl_max_vtk, 0.5f)
        << "VTK file at step 10000 should contain significant liquid fraction (>0.5)";
}

/**
 * Test: Check output at step 9800 vs 10000
 *
 * The last two output steps to understand the transition.
 */
TEST_F(VTKOutputTimingTest, CompareSteps9800And10000) {
    std::cout << "\n=== Compare Output at Steps 9800 and 10000 ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);
    thermal.initialize(T_initial);

    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    float* h_fl = new float[num_cells];
    float fl_9800 = 0.0f;
    float fl_10000 = 0.0f;

    for (int step = 0; step <= 10000; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        if (step == 9800 || step == 10000) {
            thermal.copyLiquidFractionToHost(h_fl);

            float fl_max = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                fl_max = fmaxf(fl_max, h_fl[i]);
            }

            if (step == 9800) {
                fl_9800 = fl_max;
                std::cout << "Step 9800 (last regular output): fl_max = " << fl_max << "\n";
            } else {
                fl_10000 = fl_max;
                std::cout << "Step 10000 (final output): fl_max = " << fl_max << "\n";
            }
        }
    }

    std::cout << "\nDelta between outputs: " << (fl_10000 - fl_9800) << "\n";
    std::cout << "\nConclusion:\n";
    if (fl_9800 < 0.01f && fl_10000 > 0.5f) {
        std::cout << "  Melting happens BETWEEN step 9800 and 10000.\n";
        std::cout << "  The final VTK output SHOULD capture significant melting.\n";
    }

    delete[] h_fl;

    EXPECT_GT(fl_10000, 0.5f) << "Step 10000 should have significant melting";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
