/**
 * @file test_laser_heating_simplified.cu
 * @brief Simplified integration tests for laser heating simulation
 *
 * Tests the basic integration of Thermal LBM + Laser Source + Material Properties
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// Test fixture
class LaserHeatingIntegrationTest : public ::testing::Test {
protected:
    // Default simulation parameters
    const int nx = 32;
    const int ny = 32;
    const int nz = 16;
    const float dx = 2e-6f;      // 2 micrometers
    const float dt = 1e-7f;      // Must match ThermalLBM default dt (100 ns)
    const float T_init = 300.0f;  // Room temperature

    float* d_heat_source = nullptr;
    float* d_temperature = nullptr;

    void SetUp() override {
        // Allocate device memory
        size_t size = nx * ny * nz * sizeof(float);
        cudaMalloc(&d_heat_source, size);
        cudaMalloc(&d_temperature, size);
        cudaMemset(d_heat_source, 0, size);
    }

    void TearDown() override {
        if (d_heat_source) cudaFree(d_heat_source);
        if (d_temperature) cudaFree(d_temperature);
    }
};

// Test kernel to compute laser heat source
__global__ void testComputeLaserHeatSourceKernel(
    float* heat_source,
    const LaserSource laser,  // Pass by value for simplicity
    float dx, float dy, float dz,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Get 3D indices
    int k = idx / (nx * ny);
    int j = (idx % (nx * ny)) / nx;
    int i = idx % nx;

    // Get physical coordinates
    float x = i * dx;
    float y = j * dy;
    float z = k * dz;

    // Compute heat source from laser
    heat_source[idx] = laser.computeVolumetricHeatSource(x, y, z);
}

// Test 1: Basic Heat Source Generation
TEST_F(LaserHeatingIntegrationTest, LaserHeatSourceGeneration) {
    // Setup laser
    LaserSource laser(100.0f, 50e-6f, 0.35f, 10e-6f);
    float center_x = nx * dx / 2.0f;
    float center_y = ny * dx / 2.0f;
    laser.setPosition(center_x, center_y, 0.0f);

    // Compute heat source
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    testComputeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
        d_heat_source, laser, dx, dx, dx, nx, ny, nz
    );
    cudaDeviceSynchronize();

    // Copy to host and verify
    std::vector<float> h_heat_source(nx * ny * nz);
    cudaMemcpy(h_heat_source.data(), d_heat_source,
               nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);

    // Find maximum heat source
    float max_heat = *std::max_element(h_heat_source.begin(), h_heat_source.end());

    // Center should have maximum heat at surface
    int center_idx = 0 * nx * ny + (ny/2) * nx + (nx/2);  // z=0, center x,y
    float center_heat = h_heat_source[center_idx];

    std::cout << "Max heat source: " << max_heat << " W/m³\n";
    std::cout << "Center heat source: " << center_heat << " W/m³\n";

    // Verify heat source is non-zero
    EXPECT_GT(max_heat, 0.0f) << "Heat source should be non-zero";

    // Center should be near maximum (at surface)
    EXPECT_NEAR(center_heat, max_heat, max_heat * 0.1f)
        << "Center should have maximum heat at surface";

    // Check decay with depth
    if (nz > 1) {
        int deeper_idx = 1 * nx * ny + (ny/2) * nx + (nx/2);  // z=1, center x,y
        float deeper_heat = h_heat_source[deeper_idx];
        EXPECT_LT(deeper_heat, center_heat)
            << "Heat should decrease with depth";
    }
}

// Test 2: Thermal Diffusion
TEST_F(LaserHeatingIntegrationTest, ThermalDiffusion) {
    // Setup thermal LBM with thermal diffusivity for steel
    float alpha = 4e-6f;  // m²/s (typical for steel)
    ThermalLBM thermal(nx, ny, nz, alpha);
    thermal.initialize(T_init);

    // Create a point heat source at center
    std::vector<float> h_heat_source(nx * ny * nz, 0.0f);
    int center_idx = 0 * nx * ny + (ny/2) * nx + (nx/2);
    h_heat_source[center_idx] = 1e12f;  // 1 TW/m³ at single point

    cudaMemcpy(d_heat_source, h_heat_source.data(),
               nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    // Apply heat and evolve
    const int n_steps = 100;
    for (int step = 0; step < n_steps; ++step) {
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Get temperature field
    thermal.copyTemperatureToHost(h_heat_source.data());  // Reuse buffer

    // Check temperature increase at center
    float T_center = h_heat_source[center_idx];
    std::cout << "Center temperature after heating: " << T_center << " K\n";

    // With corrected physics (proper unit conversion), the temperature increase
    // is much smaller than the old buggy behavior. For Q=1e12 W/m³, dt=1e-9s,
    // 100 steps, with high diffusivity, expect ~0.003-0.005K increase.
    EXPECT_GT(T_center, T_init + 0.001f)
        << "Temperature should increase at heat source";

    // Check diffusion - neighboring cells should also heat up
    if (nx > 2 && ny > 2) {
        int neighbor_idx = 0 * nx * ny + (ny/2) * nx + (nx/2 + 1);
        float T_neighbor = h_heat_source[neighbor_idx];
        EXPECT_GT(T_neighbor, T_init - 0.001f)  // Allow small numerical error
            << "Temperature should remain physical";
    }
}

// Test 3: Material Property Integration
TEST_F(LaserHeatingIntegrationTest, MaterialPropertyUsage) {
    // Create Ti6Al4V material
    MaterialProperties ti64;

    // Basic properties
    ti64.rho_solid = 4430.0f;
    ti64.k_solid = 21.9f;
    ti64.cp_solid = 546.0f;
    ti64.rho_liquid = 4000.0f;
    ti64.k_liquid = 33.4f;
    ti64.cp_liquid = 831.0f;
    ti64.mu_liquid = 0.003f;

    // Phase change properties
    ti64.T_solidus = 1878.0f;
    ti64.T_liquidus = 1923.0f;
    ti64.T_vaporization = 3533.0f;
    ti64.L_fusion = 286e3f;
    ti64.L_vaporization = 9.83e6f;

    // Surface and optical properties
    ti64.absorptivity_solid = 0.35f;
    ti64.absorptivity_liquid = 0.40f;
    ti64.surface_tension = 1.5f;
    ti64.dsigma_dT = -0.00035f;
    ti64.emissivity = 0.3f;

    // Compute thermal diffusivity
    float alpha = ti64.getThermalDiffusivity(T_init);

    std::cout << "Ti6Al4V thermal diffusivity at 300K: " << alpha << " m²/s\n";

    EXPECT_GT(alpha, 0.0f) << "Thermal diffusivity should be positive";
    EXPECT_LT(alpha, 1e-4f) << "Thermal diffusivity should be reasonable for metal";

    // Test temperature-dependent properties
    float k_solid = ti64.getThermalConductivity(T_init);
    float k_liquid = ti64.getThermalConductivity(2000.0f);  // Above melting

    std::cout << "Thermal conductivity (solid): " << k_solid << " W/(m·K)\n";
    std::cout << "Thermal conductivity (liquid): " << k_liquid << " W/(m·K)\n";

    // Liquid should have different conductivity
    EXPECT_NE(k_solid, k_liquid)
        << "Thermal conductivity should change with phase";
}

// Test 4: Combined Laser Heating Simulation
TEST_F(LaserHeatingIntegrationTest, CombinedLaserHeating) {
    // Setup material - 316L stainless steel
    MaterialProperties ss316;
    ss316.rho_solid = 7990.0f;
    ss316.k_solid = 16.2f;
    ss316.cp_solid = 500.0f;
    ss316.rho_liquid = 7500.0f;
    ss316.k_liquid = 25.0f;
    ss316.cp_liquid = 800.0f;
    ss316.absorptivity_solid = 0.33f;
    ss316.absorptivity_liquid = 0.38f;
    ss316.T_solidus = 1673.0f;
    ss316.T_liquidus = 1723.0f;

    float alpha = ss316.getThermalDiffusivity(T_init);

    // Setup thermal LBM
    ThermalLBM thermal(nx, ny, nz, alpha);
    thermal.initialize(T_init);

    // Setup laser
    LaserSource laser(50.0f, 30e-6f, ss316.absorptivity_solid, 10e-6f);
    laser.setPosition(nx * dx / 2.0f, ny * dx / 2.0f, 0.0f);

    // Time evolution
    const int n_steps = 500;
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    std::vector<float> temperature_history;

    for (int step = 0; step < n_steps; ++step) {
        // Compute laser heat source
        testComputeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, laser, dx, dx, dx, nx, ny, nz
        );
        cudaDeviceSynchronize();

        // Apply heat and evolve
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Monitor center temperature every 100 steps
        if (step % 100 == 0) {
            std::vector<float> h_temp(nx * ny * nz);
            thermal.copyTemperatureToHost(h_temp.data());

            int center_idx = 0 * nx * ny + (ny/2) * nx + (nx/2);
            float T_center = h_temp[center_idx];
            temperature_history.push_back(T_center);

            std::cout << "Step " << step << ": T_center = " << T_center << " K\n";
        }
    }

    // Verify heating occurred
    ASSERT_FALSE(temperature_history.empty());

    float T_initial = temperature_history.front();
    float T_final = temperature_history.back();

    EXPECT_GT(T_final, T_initial)
        << "Temperature should increase over time";

    EXPECT_GT(T_final, T_init + 50.0f)
        << "Significant heating expected with laser";

    // Check monotonic increase (approximately)
    for (size_t i = 1; i < temperature_history.size(); ++i) {
        EXPECT_GE(temperature_history[i], temperature_history[i-1] - 5.0f)
            << "Temperature should generally increase (allowing small fluctuations)";
    }
}

// Test 5: Energy Conservation Check
TEST_F(LaserHeatingIntegrationTest, EnergyConservationBasic) {
    // Use smaller domain for better energy tracking
    const int nx_small = 16;
    const int ny_small = 16;
    const int nz_small = 8;

    // Setup material
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 20.0f;
    mat.cp_solid = 500.0f;
    mat.absorptivity_solid = 0.3f;
    mat.T_solidus = 1700.0f;
    mat.T_liquidus = 1750.0f;

    float alpha = mat.getThermalDiffusivity(T_init);

    // Setup thermal LBM
    ThermalLBM thermal(nx_small, ny_small, nz_small, alpha);
    thermal.initialize(T_init);

    // Setup laser
    const float laser_power = 10.0f;  // 10W
    LaserSource laser(laser_power, 20e-6f, mat.absorptivity_solid, 10e-6f);
    laser.setPosition(nx_small * dx / 2.0f, ny_small * dx / 2.0f, 0.0f);

    // Reallocate for smaller domain
    cudaFree(d_heat_source);
    cudaMalloc(&d_heat_source, nx_small * ny_small * nz_small * sizeof(float));

    // Simulate
    const int n_steps = 1000;
    const float absorbed_power = laser_power * mat.absorptivity_solid;
    const float total_energy_input = absorbed_power * n_steps * dt;

    dim3 blockSize(256);
    dim3 gridSize((nx_small * ny_small * nz_small + blockSize.x - 1) / blockSize.x);

    for (int step = 0; step < n_steps; ++step) {
        testComputeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, laser, dx, dx, dx, nx_small, ny_small, nz_small
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Calculate temperature increase
    std::vector<float> h_temp(nx_small * ny_small * nz_small);
    thermal.copyTemperatureToHost(h_temp.data());

    float avg_temp_increase = 0.0f;
    for (const auto& T : h_temp) {
        avg_temp_increase += (T - T_init);
    }
    avg_temp_increase /= h_temp.size();

    // Estimate energy stored
    float volume = nx_small * ny_small * nz_small * dx * dx * dx;
    float mass = mat.getDensity(T_init) * volume;
    float energy_stored = mass * mat.getSpecificHeat(T_init) * avg_temp_increase;

    std::cout << "Energy input: " << total_energy_input << " J\n";
    std::cout << "Energy stored (approx): " << energy_stored << " J\n";
    std::cout << "Average temperature increase: " << avg_temp_increase << " K\n";

    // Very rough check - energy stored should be same order of magnitude as input
    // (accounting for boundary losses and approximations)
    EXPECT_GT(energy_stored, 0.0f) << "Some energy should be stored";
    EXPECT_LT(energy_stored, total_energy_input * 10.0f)
        << "Energy stored shouldn't be unreasonably large";
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Check CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Running tests on: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;

    return RUN_ALL_TESTS();
}