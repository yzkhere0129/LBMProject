/**
 * @file test_laser_heating.cu
 * @brief Integration tests for laser heating simulation combining Thermal LBM + Laser Source + Material Properties
 *
 * These tests verify that all three modules work together correctly to simulate
 * realistic laser heating of metals in additive manufacturing processes.
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
#include <fstream>
#include <iomanip>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"

// Test fixture for laser heating integration tests
class LaserHeatingIntegrationTest : public ::testing::Test {
protected:
    // Default simulation parameters
    const int nx = 64;
    const int ny = 64;
    const int nz = 32;
    const float dx = 2e-6f;      // 2 micrometers
    const float dy = 2e-6f;
    const float dz = 2e-6f;
    const float dt = 1e-9f;       // 1 nanosecond
    const float T_init = 300.0f;  // Room temperature

    float* d_heat_source = nullptr;
    float* d_temperature = nullptr;

    void SetUp() override {
        // Allocate device memory
        size_t size = nx * ny * nz * sizeof(float);
        cudaMalloc(&d_heat_source, size);
        cudaMalloc(&d_temperature, size);

        // Clear heat source
        cudaMemset(d_heat_source, 0, size);
    }

    void TearDown() override {
        if (d_heat_source) cudaFree(d_heat_source);
        if (d_temperature) cudaFree(d_temperature);
    }

    // Helper function to compute total internal energy
    float computeTotalInternalEnergy(const ThermalLBM& thermal,
                                      const MaterialProperties& mat) {
        // Copy temperature field to host
        std::vector<float> h_temperature(nx * ny * nz);
        thermal.getTemperatureField(h_temperature.data());

        float total_energy = 0.0f;
        float dV = dx * dy * dz;

        for (int idx = 0; idx < nx * ny * nz; ++idx) {
            float T = h_temperature[idx];
            float rho = mat.getDensity(T);
            float cp = mat.getSpecificHeat(T);

            // Sensible heat
            total_energy += rho * cp * (T - T_init) * dV;

            // Latent heat if melting
            if (T > mat.T_solidus && T < mat.T_liquidus) {
                float fl = mat.liquidFraction(T);
                total_energy += rho * mat.L_fusion * fl * dV;
            }
        }

        return total_energy;
    }

    // Helper function to get temperature at a specific point
    float getTemperatureAt(const ThermalLBM& thermal, int i, int j, int k) {
        std::vector<float> h_temperature(nx * ny * nz);
        thermal.getTemperatureField(h_temperature.data());
        int idx = k * nx * ny + j * nx + i;
        return h_temperature[idx];
    }

    // Helper function to extract temperature profile along a line
    std::vector<float> getTemperatureProfile(const ThermalLBM& thermal,
                                              int j, int k) {
        std::vector<float> h_temperature(nx * ny * nz);
        thermal.getTemperatureField(h_temperature.data());

        std::vector<float> profile(nx);
        for (int i = 0; i < nx; ++i) {
            int idx = k * nx * ny + j * nx + i;
            profile[i] = h_temperature[idx];
        }
        return profile;
    }

    // Helper function to export temperature field to VTK
    void exportTemperatureFieldVTK(const ThermalLBM& thermal,
                                    const std::string& filename) {
        std::vector<float> h_temperature(nx * ny * nz);
        thermal.getTemperatureField(h_temperature.data());

        std::ofstream file(filename);
        file << "# vtk DataFile Version 3.0\n";
        file << "Temperature Field\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        file << "ORIGIN 0 0 0\n";
        file << "SPACING " << dx * 1e6 << " " << dy * 1e6 << " " << dz * 1e6 << "\n";
        file << "POINT_DATA " << nx * ny * nz << "\n";
        file << "SCALARS Temperature float 1\n";
        file << "LOOKUP_TABLE default\n";

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = k * nx * ny + j * nx + i;
                    file << h_temperature[idx] << "\n";
                }
            }
        }
        file.close();
    }
};

// Kernel to compute laser heat source
__global__ void computeLaserHeatSourceKernel(
    float* heat_source,
    const LaserSource* laser,
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
    heat_source[idx] = laser->computeVolumetricHeatSource(x, y, z);
}

// Test 1: Point Melting - Stationary laser heating
TEST_F(LaserHeatingIntegrationTest, PointMeltingHeating) {
    // 1. Setup material - Ti6Al4V
    MaterialProperties ti64;
    ti64.T_melt = 1923.0f;
    ti64.T_solidus = 1878.0f;
    ti64.T_liquidus = 1923.0f;
    ti64.T_vaporization = 3533.0f;
    ti64.L_fusion = 286e3f;        // J/kg
    ti64.L_vaporization = 9.83e6f; // J/kg
    ti64.absorptivity = 0.35f;

    // Temperature-dependent properties (simplified)
    ti64.rho_solid = 4430.0f;      // kg/m³
    ti64.rho_liquid = 4000.0f;
    ti64.k_solid = 21.9f;           // W/(m·K)
    ti64.k_liquid = 33.4f;
    ti64.cp_solid = 546.0f;         // J/(kg·K)
    ti64.cp_liquid = 831.0f;

    // 2. Initialize thermal LBM
    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    // Set material properties
    float alpha = ti64.k_solid / (ti64.rho_solid * ti64.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // 3. Setup laser - 100W at center
    LaserSource laser(100.0f, 50e-6f, ti64.absorptivity, 10e-6f);
    float center_x = nx * dx / 2.0f;
    float center_y = ny * dy / 2.0f;
    laser.setPosition(center_x, center_y, 0.0f);

    // Copy laser to device
    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // 4. Time evolution
    const int n_steps = 5000;
    const int check_interval = 500;

    std::vector<float> center_temp_history;
    center_temp_history.reserve(n_steps / check_interval);

    for (int step = 0; step < n_steps; ++step) {
        // Compute laser heat source
        dim3 blockSize(256);
        dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        // Add heat source to temperature field
        thermal.addHeatSource(d_heat_source, dt);

        // Evolve temperature field
        thermal.step();

        // Monitor progress
        if (step % check_interval == 0) {
            float T_center = getTemperatureAt(thermal, nx/2, ny/2, 0);
            center_temp_history.push_back(T_center);
            std::cout << "Step " << step << "/" << n_steps
                      << ": T_center = " << T_center << " K" << std::endl;
        }
    }

    // 5. Verify results
    float T_center = getTemperatureAt(thermal, nx/2, ny/2, 0);
    float T_edge = getTemperatureAt(thermal, 0, 0, 0);
    float T_corner = getTemperatureAt(thermal, nx-1, ny-1, nz-1);

    std::cout << "Final temperatures:\n";
    std::cout << "  Center (laser spot): " << T_center << " K\n";
    std::cout << "  Edge: " << T_edge << " K\n";
    std::cout << "  Corner: " << T_corner << " K\n";

    // Center temperature should increase significantly
    EXPECT_GT(T_center, T_init + 500.0f) << "Center temperature should rise by at least 500K";

    // Edge temperature should increase less
    EXPECT_LT(T_edge - T_init, 200.0f) << "Edge temperature rise should be limited";

    // Temperature gradient from center to edge
    EXPECT_GT(T_center, T_edge + 300.0f) << "Strong temperature gradient expected";

    // Temperature should not exceed vaporization (physically unreasonable)
    EXPECT_LT(T_center, ti64.T_vaporization) << "Temperature should not exceed vaporization";

    // Temperature should be monotonically increasing at center
    for (size_t i = 1; i < center_temp_history.size(); ++i) {
        EXPECT_GE(center_temp_history[i], center_temp_history[i-1])
            << "Temperature at center should monotonically increase";
    }

    // Export VTK for visualization (optional)
    exportTemperatureFieldVTK(thermal, "point_melting_final.vtk");

    cudaFree(d_laser);
}

// Test 2: Temperature Distribution Shape
TEST_F(LaserHeatingIntegrationTest, TemperatureDistributionShape) {
    // Setup material - 316L Stainless Steel
    MaterialProperties ss316L;
    ss316L.T_melt = 1673.0f;
    ss316L.rho_solid = 7990.0f;
    ss316L.k_solid = 16.2f;
    ss316L.cp_solid = 500.0f;
    ss316L.absorptivity = 0.33f;

    // Initialize thermal LBM
    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = ss316L.k_solid / (ss316L.rho_solid * ss316L.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Setup laser
    LaserSource laser(150.0f, 75e-6f, ss316L.absorptivity, 10e-6f);
    float center_x = nx * dx / 2.0f;
    float center_y = ny * dy / 2.0f;
    laser.setPosition(center_x, center_y, 0.0f);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Run simulation
    const int n_steps = 3000;
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();
    }

    // Extract temperature profiles
    std::vector<float> profile_x = getTemperatureProfile(thermal, ny/2, 0);
    std::vector<float> profile_y = getTemperatureProfile(thermal, nx/2, 0);

    // Find maximum temperature and its position
    auto max_it = std::max_element(profile_x.begin(), profile_x.end());
    int max_idx = std::distance(profile_x.begin(), max_it);
    float T_max = *max_it;

    std::cout << "Temperature distribution analysis:\n";
    std::cout << "  Maximum temperature: " << T_max << " K\n";
    std::cout << "  Maximum position index: " << max_idx << " (expected: " << nx/2 << ")\n";

    // Verify center is hottest
    EXPECT_EQ(max_idx, nx/2) << "Maximum temperature should be at center";
    EXPECT_FLOAT_EQ(T_max, profile_x[nx/2]) << "Center should be hottest point";

    // Verify monotonic decrease from center
    for (int i = nx/2; i < nx-1; ++i) {
        EXPECT_GE(profile_x[i], profile_x[i+1])
            << "Temperature should decrease from center outward";
    }
    for (int i = 1; i < nx/2; ++i) {
        EXPECT_LE(profile_x[i-1], profile_x[i])
            << "Temperature should increase toward center";
    }

    // Check for approximate Gaussian shape
    float fwhm_value = (T_max - T_init) / 2.0f + T_init;
    int left_idx = -1, right_idx = -1;

    for (int i = 0; i < nx/2; ++i) {
        if (profile_x[i] >= fwhm_value) {
            left_idx = i;
            break;
        }
    }

    for (int i = nx-1; i > nx/2; --i) {
        if (profile_x[i] >= fwhm_value) {
            right_idx = i;
            break;
        }
    }

    if (left_idx >= 0 && right_idx >= 0) {
        float fwhm = (right_idx - left_idx) * dx;
        float expected_fwhm = 2.355f * laser.spot_radius; // For Gaussian
        float relative_error = std::abs(fwhm - expected_fwhm) / expected_fwhm;

        std::cout << "  FWHM: " << fwhm * 1e6 << " µm\n";
        std::cout << "  Expected FWHM: " << expected_fwhm * 1e6 << " µm\n";
        std::cout << "  Relative error: " << relative_error * 100 << "%\n";

        EXPECT_LT(relative_error, 0.5f) << "FWHM should match Gaussian profile";
    }

    cudaFree(d_laser);
}

// Test 3: Energy Conservation
TEST_F(LaserHeatingIntegrationTest, EnergyConservation) {
    // Setup material
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 20.0f;
    mat.cp_solid = 500.0f;
    mat.absorptivity = 0.3f;
    mat.T_melt = 1800.0f;
    mat.L_fusion = 250e3f;

    // Use smaller domain for better energy tracking
    const int nx_small = 32;
    const int ny_small = 32;
    const int nz_small = 16;

    ThermalLBM thermal(nx_small, ny_small, nz_small, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Setup laser with known power
    const float laser_power = 50.0f; // Watts
    LaserSource laser(laser_power, 30e-6f, mat.absorptivity, 10e-6f);
    laser.setPosition(nx_small * dx / 2.0f, ny_small * dy / 2.0f, 0.0f);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Reallocate heat source for smaller domain
    cudaFree(d_heat_source);
    cudaMalloc(&d_heat_source, nx_small * ny_small * nz_small * sizeof(float));

    // Track energy
    float total_energy_input = 0.0f;
    const int n_steps = 2000;
    const float absorbed_power = laser_power * mat.absorptivity;

    // Initial internal energy (should be zero relative to T_init)
    float initial_energy = 0.0f;

    dim3 blockSize(256);
    dim3 gridSize((nx_small * ny_small * nz_small + blockSize.x - 1) / blockSize.x);

    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx_small, ny_small, nz_small
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();

        // Accumulate input energy
        total_energy_input += absorbed_power * dt;
    }

    // Compute final internal energy
    std::vector<float> h_temperature(nx_small * ny_small * nz_small);
    thermal.getTemperatureField(h_temperature.data());

    float final_energy = 0.0f;
    float dV = dx * dy * dz;

    for (int idx = 0; idx < nx_small * ny_small * nz_small; ++idx) {
        float T = h_temperature[idx];
        float dE = mat.rho_solid * mat.cp_solid * (T - T_init) * dV;
        final_energy += dE;
    }

    float energy_increase = final_energy - initial_energy;

    std::cout << "Energy conservation analysis:\n";
    std::cout << "  Total energy input: " << total_energy_input << " J\n";
    std::cout << "  Internal energy increase: " << energy_increase << " J\n";
    std::cout << "  Energy ratio: " << energy_increase / total_energy_input << "\n";

    // Energy increase should be less than input (boundary losses)
    EXPECT_LT(energy_increase, total_energy_input)
        << "Energy increase should not exceed input";

    // But should be a significant fraction (at least 30% retained)
    EXPECT_GT(energy_increase / total_energy_input, 0.3f)
        << "At least 30% of energy should be retained";

    // Check for reasonable efficiency
    float efficiency = energy_increase / total_energy_input;
    EXPECT_GT(efficiency, 0.0f) << "Some energy must be absorbed";
    EXPECT_LT(efficiency, 1.0f) << "Cannot exceed 100% efficiency";

    cudaFree(d_laser);
}

// Test 4: Material Property Response
TEST_F(LaserHeatingIntegrationTest, MaterialPropertyResponse) {
    // Setup Ti6Al4V with full property model
    MaterialProperties ti64;
    ti64.T_melt = 1923.0f;
    ti64.T_solidus = 1878.0f;
    ti64.T_liquidus = 1923.0f;
    ti64.rho_solid = 4430.0f;
    ti64.rho_liquid = 4000.0f;
    ti64.k_solid = 21.9f;
    ti64.k_liquid = 33.4f;
    ti64.cp_solid = 546.0f;
    ti64.cp_liquid = 831.0f;
    ti64.L_fusion = 286e3f;
    ti64.absorptivity = 0.35f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    // High power laser to reach melting
    LaserSource laser(200.0f, 40e-6f, ti64.absorptivity, 10e-6f);
    laser.setPosition(nx * dx / 2.0f, ny * dy / 2.0f, 0.0f);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Heat until melting
    const int n_steps = 8000;
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    for (int step = 0; step < n_steps; ++step) {
        // Update thermal diffusivity based on temperature
        if (step % 100 == 0) {
            float T_center = getTemperatureAt(thermal, nx/2, ny/2, 0);
            float k = ti64.getThermalConductivity(T_center);
            float rho = ti64.getDensity(T_center);
            float cp = ti64.getSpecificHeat(T_center);
            float alpha = k / (rho * cp);
            thermal.setThermalDiffusivity(alpha);
        }

        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();
    }

    // Check temperatures at different locations
    float T_hot = getTemperatureAt(thermal, nx/2, ny/2, 0);     // Center
    float T_warm = getTemperatureAt(thermal, nx/2 + 5, ny/2, 0); // Near center
    float T_cold = getTemperatureAt(thermal, 0, 0, 0);          // Corner

    std::cout << "Material property response:\n";
    std::cout << "  T_hot (center): " << T_hot << " K\n";
    std::cout << "  T_warm (near): " << T_warm << " K\n";
    std::cout << "  T_cold (edge): " << T_cold << " K\n";

    // Check thermal conductivity variation
    float k_hot = ti64.getThermalConductivity(T_hot);
    float k_cold = ti64.getThermalConductivity(T_cold);

    std::cout << "  k(T_hot): " << k_hot << " W/(m·K)\n";
    std::cout << "  k(T_cold): " << k_cold << " W/(m·K)\n";

    // If center is above melting, liquid properties should apply
    if (T_hot > ti64.T_melt) {
        EXPECT_NEAR(k_hot, ti64.k_liquid, ti64.k_liquid * 0.1f)
            << "Liquid thermal conductivity should apply above melting";

        float rho_hot = ti64.getDensity(T_hot);
        EXPECT_NEAR(rho_hot, ti64.rho_liquid, ti64.rho_liquid * 0.1f)
            << "Liquid density should apply above melting";
    }

    // Check liquid fraction in mushy zone
    if (T_warm > ti64.T_solidus && T_warm < ti64.T_liquidus) {
        float fl = ti64.liquidFraction(T_warm);
        EXPECT_GT(fl, 0.0f) << "Liquid fraction should be positive in mushy zone";
        EXPECT_LT(fl, 1.0f) << "Liquid fraction should be less than 1 in mushy zone";

        std::cout << "  Liquid fraction at T_warm: " << fl << "\n";
    }

    // Verify property continuity
    float T_test = ti64.T_melt;
    float k_below = ti64.getThermalConductivity(T_test - 1.0f);
    float k_above = ti64.getThermalConductivity(T_test + 1.0f);
    float k_jump = std::abs(k_above - k_below);

    EXPECT_LT(k_jump, 20.0f) << "Thermal conductivity should not have huge discontinuity";

    cudaFree(d_laser);
}

// Test 5: Moving Laser Scan
TEST_F(LaserHeatingIntegrationTest, MovingLaserScan) {
    // Setup 316L stainless steel
    MaterialProperties ss316L;
    ss316L.rho_solid = 7990.0f;
    ss316L.k_solid = 16.2f;
    ss316L.cp_solid = 500.0f;
    ss316L.absorptivity = 0.33f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = ss316L.k_solid / (ss316L.rho_solid * ss316L.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Setup moving laser
    LaserSource laser(150.0f, 60e-6f, ss316L.absorptivity, 10e-6f);
    float start_x = 10 * dx;
    float center_y = ny * dy / 2.0f;
    laser.setPosition(start_x, center_y, 0.0f);

    // Set scan velocity - 0.5 m/s in x direction
    const float scan_velocity = 0.5f;

    // Simulate scanning
    const int n_steps = 4000;
    const int snapshot_interval = 1000;

    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    std::vector<std::vector<float>> snapshots;

    for (int step = 0; step < n_steps; ++step) {
        // Update laser position
        float current_x = start_x + scan_velocity * step * dt;
        laser.setPosition(current_x, center_y, 0.0f);

        // Copy updated laser to device
        LaserSource* d_laser;
        cudaMalloc(&d_laser, sizeof(LaserSource));
        cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

        // Compute heat source at new position
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );

        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();

        cudaFree(d_laser);

        // Take snapshots
        if (step % snapshot_interval == 0) {
            snapshots.push_back(getTemperatureProfile(thermal, ny/2, 0));
            std::cout << "Step " << step << ": Laser at x = "
                      << current_x * 1e6 << " µm\n";
        }
    }

    // Analyze temperature trail
    std::vector<float> final_profile = getTemperatureProfile(thermal, ny/2, 0);

    // Find peak temperatures along scan path
    float max_temp_trail = *std::max_element(final_profile.begin(), final_profile.end());

    std::cout << "Moving laser scan results:\n";
    std::cout << "  Maximum trail temperature: " << max_temp_trail << " K\n";
    std::cout << "  Scan distance: " << scan_velocity * n_steps * dt * 1e3 << " mm\n";

    // Verify heating along path
    int n_heated = 0;
    for (size_t i = 0; i < final_profile.size(); ++i) {
        if (final_profile[i] > T_init + 50.0f) {
            n_heated++;
        }
    }

    float heated_fraction = static_cast<float>(n_heated) / nx;
    std::cout << "  Heated region: " << heated_fraction * 100 << "% of domain\n";

    EXPECT_GT(heated_fraction, 0.3f) << "At least 30% of path should be heated";
    EXPECT_GT(max_temp_trail, T_init + 200.0f) << "Significant heating expected along path";

    // Check cooling behind laser
    if (snapshots.size() >= 2) {
        // Compare early and late positions
        float early_temp = snapshots[0][20]; // Position laser passed early
        float late_temp = snapshots.back()[20]; // Same position, later time

        if (early_temp > T_init + 100.0f) {
            EXPECT_LT(late_temp, early_temp)
                << "Temperature should decrease after laser passes";
        }
    }

    // Export final state
    exportTemperatureFieldVTK(thermal, "moving_laser_scan.vtk");
}

// Test 6: Multiple Heat Sources
TEST_F(LaserHeatingIntegrationTest, MultipleHeatSources) {
    // Setup material
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 25.0f;
    mat.cp_solid = 500.0f;
    mat.absorptivity = 0.35f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Setup two lasers
    LaserSource laser1(75.0f, 40e-6f, mat.absorptivity, 10e-6f);
    LaserSource laser2(75.0f, 40e-6f, mat.absorptivity, 10e-6f);

    float spacing = 20 * dx; // 40 micrometers apart
    laser1.setPosition(nx * dx / 2.0f - spacing/2, ny * dy / 2.0f, 0.0f);
    laser2.setPosition(nx * dx / 2.0f + spacing/2, ny * dy / 2.0f, 0.0f);

    // Allocate device lasers
    LaserSource* d_lasers;
    cudaMalloc(&d_lasers, 2 * sizeof(LaserSource));
    cudaMemcpy(&d_lasers[0], &laser1, sizeof(LaserSource), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_lasers[1], &laser2, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Kernel for multiple lasers
    auto computeMultiLaserKernel = [](
        float* heat_source,
        const LaserSource* lasers,
        int n_lasers,
        float dx, float dy, float dz,
        int nx, int ny, int nz
    ) {
        // Implementation would sum contributions from all lasers
        // For this test, we'll simulate with sequential calls
    };

    // Simulate
    const int n_steps = 3000;
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    for (int step = 0; step < n_steps; ++step) {
        // Clear heat source
        cudaMemset(d_heat_source, 0, nx * ny * nz * sizeof(float));

        // Add both laser contributions
        for (int i = 0; i < 2; ++i) {
            float* d_temp_source;
            cudaMalloc(&d_temp_source, nx * ny * nz * sizeof(float));

            computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
                d_temp_source, &d_lasers[i], dx, dy, dz, nx, ny, nz
            );

            // Add to total heat source
            thrust::device_ptr<float> total_ptr(d_heat_source);
            thrust::device_ptr<float> temp_ptr(d_temp_source);
            thrust::transform(total_ptr, total_ptr + nx*ny*nz,
                              temp_ptr, total_ptr,
                              thrust::plus<float>());

            cudaFree(d_temp_source);
        }

        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();
    }

    // Verify two hot spots
    std::vector<float> profile = getTemperatureProfile(thermal, ny/2, 0);

    // Find local maxima
    std::vector<int> maxima_indices;
    for (int i = 1; i < nx-1; ++i) {
        if (profile[i] > profile[i-1] && profile[i] > profile[i+1] &&
            profile[i] > T_init + 100.0f) {
            maxima_indices.push_back(i);
        }
    }

    std::cout << "Multiple heat sources:\n";
    std::cout << "  Number of hot spots detected: " << maxima_indices.size() << "\n";

    EXPECT_EQ(maxima_indices.size(), 2) << "Should detect two hot spots from two lasers";

    if (maxima_indices.size() == 2) {
        float separation = std::abs(maxima_indices[1] - maxima_indices[0]) * dx;
        std::cout << "  Hot spot separation: " << separation * 1e6 << " µm\n";

        EXPECT_NEAR(separation, spacing, spacing * 0.3f)
            << "Hot spot separation should match laser spacing";
    }

    cudaFree(d_lasers);
}

// Test 7: Boundary Conditions
TEST_F(LaserHeatingIntegrationTest, BoundaryConditions) {
    // Test different boundary conditions
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 25.0f;
    mat.cp_solid = 500.0f;
    mat.absorptivity = 0.3f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Apply fixed temperature boundary at bottom
    thermal.setFixedTemperatureBoundary(T_init, BoundaryLocation::BOTTOM);

    // Laser near boundary
    LaserSource laser(100.0f, 50e-6f, mat.absorptivity, 10e-6f);
    laser.setPosition(nx * dx / 2.0f, ny * dy / 2.0f, (nz-1) * dz);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Simulate
    const int n_steps = 2000;
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + blockSize.x - 1) / blockSize.x);

    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();
    }

    // Check boundary temperature remains fixed
    float T_bottom = getTemperatureAt(thermal, nx/2, ny/2, 0);
    float T_top = getTemperatureAt(thermal, nx/2, ny/2, nz-1);

    std::cout << "Boundary conditions:\n";
    std::cout << "  Bottom temperature (fixed): " << T_bottom << " K\n";
    std::cout << "  Top temperature (heated): " << T_top << " K\n";

    EXPECT_NEAR(T_bottom, T_init, 10.0f) << "Bottom boundary should remain near initial temperature";
    EXPECT_GT(T_top, T_init + 100.0f) << "Top should heat up significantly";

    cudaFree(d_laser);
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
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

    return RUN_ALL_TESTS();
}