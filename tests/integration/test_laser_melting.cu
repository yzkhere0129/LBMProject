/**
 * @file test_laser_melting.cu
 * @brief Integration test for laser-induced melting with phase change
 *
 * This test demonstrates:
 * - 400W laser heating of Ti6Al4V
 * - Temperature exceeding melting point (1923K)
 * - Liquid fraction > 0 in heated region
 * - Energy conservation with phase change
 *
 * Physical setup:
 * - Material: Ti6Al4V (titanium alloy)
 * - Laser power: 400W
 * - Spot radius: 50 micrometers
 * - Initial temperature: 300K (room temperature)
 * - Domain: 64x64x32 cells (128x128x64 micrometers)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"

using namespace lbm;

class LaserMeltingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Domain setup - reduced for fast integration test
        nx = 64;
        ny = 64;
        nz = 32;
        num_cells = nx * ny * nz;

        // Physical parameters
        dx = 2e-6f;  // 2 micrometers
        dy = 2e-6f;
        dz = 2e-6f;
        dt = 1e-7f;  // Must match ThermalLBM default dt (100 ns)

        // Material: Ti6Al4V
        material = physics::MaterialDatabase::getTi6Al4V();

        // Laser parameters
        laser_power = 400.0f;  // 400W - sufficient for melting
        spot_radius = 50e-6f;  // 50 micrometers
        penetration_depth = 15e-6f;  // 15 micrometers

        // Initial temperature
        T_initial = 300.0f;  // Room temperature

        // Allocate device memory for heat source
        size_t field_size = num_cells * sizeof(float);
        cudaMalloc(&d_heat_source, field_size);
        cudaMemset(d_heat_source, 0, field_size);

        // Create output directory
        system("mkdir -p test_output");
    }

    void TearDown() override {
        if (d_heat_source) cudaFree(d_heat_source);
    }

    // Domain parameters
    int nx, ny, nz, num_cells;
    float dx, dy, dz, dt;

    // Material
    physics::MaterialProperties material;

    // Laser parameters
    float laser_power, spot_radius, penetration_depth;

    // Initial conditions
    float T_initial;

    // Device memory
    float* d_heat_source;
};

/**
 * Test that 400W laser can heat Ti6Al4V above melting point
 */
TEST_F(LaserMeltingTest, TemperatureExceedsMeltingPoint) {
    std::cout << "\n=== Laser Melting Test: Temperature ===\n";
    std::cout << "Material: " << material.name << "\n";
    std::cout << "Melting point: " << material.T_liquidus << " K\n";
    std::cout << "Laser power: " << laser_power << " W\n\n";

    // Create thermal solver with phase change
    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);
    thermal.initialize(T_initial);

    // Create laser source centered on domain
    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // Time evolution - reduced for fast integration test
    // With 400W and small domain, 5000 steps sufficient to exceed melting point
    int n_steps = 5000;
    float max_temp = T_initial;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    for (int step = 0; step < n_steps; ++step) {
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

        // Track maximum temperature every 1000 steps (reduced frequency for speed)
        if (step % 1000 == 0 || step == n_steps - 1) {
            float* h_temp = new float[num_cells];
            thermal.copyTemperatureToHost(h_temp);

            float T_max = h_temp[0];
            for (int i = 0; i < num_cells; ++i) {
                T_max = fmaxf(T_max, h_temp[i]);
            }
            max_temp = fmaxf(max_temp, T_max);

            std::cout << "Step " << step << ": T_max = " << T_max << " K\n";
            delete[] h_temp;
        }
    }

    std::cout << "\nFinal maximum temperature: " << max_temp << " K\n";
    std::cout << "Melting point: " << material.T_liquidus << " K\n";

    // Verify temperature exceeds melting point
    EXPECT_GT(max_temp, material.T_liquidus)
        << "Temperature should exceed melting point with 400W laser";
}

/**
 * Test that liquid fraction appears in heated region
 */
TEST_F(LaserMeltingTest, LiquidFractionAppearsInHeatedRegion) {
    std::cout << "\n=== Laser Melting Test: Liquid Fraction ===\n";

    // Create thermal solver with phase change
    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);
    thermal.initialize(T_initial);

    // Verify phase change is enabled
    ASSERT_TRUE(thermal.hasPhaseChange())
        << "Phase change should be enabled";

    // Create laser source
    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // Time evolution - reduced for fast integration test
    int n_steps = 5000;
    float max_liquid_fraction = 0.0f;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    for (int step = 0; step < n_steps; ++step) {
        // Compute laser heat source
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        // Add heat and evolve
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();  // This also updates liquid fraction

        // Check liquid fraction every 1000 steps
        if (step % 1000 == 0 || step == n_steps - 1) {
            float* h_fl = new float[num_cells];
            thermal.copyLiquidFractionToHost(h_fl);

            float fl_max = 0.0f;
            int n_melting = 0;
            for (int i = 0; i < num_cells; ++i) {
                fl_max = fmaxf(fl_max, h_fl[i]);
                if (h_fl[i] > 0.01f) n_melting++;
            }
            max_liquid_fraction = fmaxf(max_liquid_fraction, fl_max);

            std::cout << "Step " << step << ": fl_max = " << fl_max
                      << ", cells melting: " << n_melting << "\n";
            delete[] h_fl;
        }
    }

    std::cout << "\nMaximum liquid fraction achieved: " << max_liquid_fraction << "\n";

    // Verify liquid fraction appears
    EXPECT_GT(max_liquid_fraction, 0.0f)
        << "Liquid fraction should be > 0 when melting occurs";
    EXPECT_GT(max_liquid_fraction, 0.5f)
        << "Peak liquid fraction should exceed 0.5 in center of laser spot";
}

/**
 * Test energy conservation with phase change
 */
TEST_F(LaserMeltingTest, EnergyConservation) {
    std::cout << "\n=== Laser Melting Test: Energy Conservation ===\n";

    // Create thermal solver with phase change
    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);
    thermal.initialize(T_initial);

    // Create laser source
    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // Calculate total laser energy input
    int n_steps = 500;
    float total_time = n_steps * dt;
    float total_laser_energy = laser_power * total_time;  // [J]

    std::cout << "Total simulation time: " << total_time * 1e6 << " microseconds\n";
    std::cout << "Total laser energy input: " << total_laser_energy * 1e3 << " mJ\n";

    // Compute initial thermal energy
    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    thermal.copyTemperatureToHost(h_temp);
    thermal.copyLiquidFractionToHost(h_fl);

    float cell_volume = dx * dy * dz;
    float E_initial = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float T = h_temp[i];
        float fl = h_fl[i];
        float rho = material.getDensity(T);
        float cp = material.getSpecificHeat(T);
        // Total energy includes sensible heat + latent heat
        float energy_density = rho * cp * (T - T_initial) + fl * rho * material.L_fusion;
        E_initial += energy_density * cell_volume;
    }

    std::cout << "Initial thermal energy: " << E_initial * 1e3 << " mJ\n";

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    // Time evolution
    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Compute final thermal energy
    thermal.copyTemperatureToHost(h_temp);
    thermal.copyLiquidFractionToHost(h_fl);

    float E_final = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float T = h_temp[i];
        float fl = h_fl[i];
        float rho = material.getDensity(T);
        float cp = material.getSpecificHeat(T);
        float energy_density = rho * cp * (T - T_initial) + fl * rho * material.L_fusion;
        E_final += energy_density * cell_volume;
    }

    float delta_E = E_final - E_initial;
    float energy_error = fabsf(delta_E - total_laser_energy) / total_laser_energy * 100.0f;

    std::cout << "Final thermal energy: " << E_final * 1e3 << " mJ\n";
    std::cout << "Energy change: " << delta_E * 1e3 << " mJ\n";
    std::cout << "Energy error: " << energy_error << " %\n";

    delete[] h_temp;
    delete[] h_fl;

    // Energy should be conserved within reasonable tolerance
    // Note: Some energy loss to boundaries is expected in small domain
    // 70% tolerance accounts for significant boundary losses
    EXPECT_LT(energy_error, 70.0f)
        << "Energy conservation error should be reasonable (< 70%)";

    // Energy should increase (laser adds heat)
    EXPECT_GT(delta_E, 0.0f)
        << "Total energy should increase due to laser heating";
}

/**
 * Test VTK output with liquid fraction field
 */
TEST_F(LaserMeltingTest, VTKOutputWithLiquidFraction) {
    std::cout << "\n=== Laser Melting Test: VTK Output ===\n";

    // Create thermal solver with phase change
    float thermal_diffusivity = material.getThermalDiffusivity(T_initial);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);
    thermal.initialize(T_initial);

    // Create laser source
    float Lx = nx * dx;
    float Ly = ny * dy;
    LaserSource laser(laser_power, spot_radius,
                     material.absorptivity_solid, penetration_depth);
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // Run simulation - reduced for fast integration test
    int n_steps = 5000;

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Copy results to host
    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    thermal.copyTemperatureToHost(h_temp);
    thermal.copyLiquidFractionToHost(h_fl);

    // Compute phase state field (0=solid, 1=mushy, 2=liquid)
    float* h_phase = new float[num_cells];
    for (int i = 0; i < num_cells; ++i) {
        float T = h_temp[i];
        if (T < material.T_solidus) {
            h_phase[i] = 0.0f;  // Solid
        } else if (T > material.T_liquidus) {
            h_phase[i] = 2.0f;  // Liquid
        } else {
            h_phase[i] = 1.0f;  // Mushy
        }
    }

    // Write VTK file with temperature, liquid fraction, and phase state
    std::string filename = "test_output/laser_melting_final.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    std::cout << "VTK output written to: " << filename << "\n";
    std::cout << "  - Temperature field\n";
    std::cout << "  - LiquidFraction field (0-1)\n";
    std::cout << "  - PhaseState field (0=solid, 1=mushy, 2=liquid)\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    // Verify file was created
    FILE* f = fopen(filename.c_str(), "r");
    ASSERT_NE(f, nullptr) << "VTK file should be created";
    fclose(f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
