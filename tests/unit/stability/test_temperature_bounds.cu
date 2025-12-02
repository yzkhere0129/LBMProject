/**
 * @file test_temperature_bounds.cu
 * @brief Unit tests for temperature bounds enforcement
 *
 * CRITICAL REGRESSION TEST: Verifies that temperature is clamped to
 * physically realistic bounds [0, T_max].
 *
 * Context:
 * - Numerical errors can cause temperature to become negative or unrealistically high
 * - Negative temperatures create NaN in radiation BC (T^4)
 * - Extremely high temperatures cause evaporation model overflow
 *
 * Fix Location: thermal_lbm.cu::computeTemperatureKernel()
 * Fix Type: Clamp temperature to [0, 7000] K range
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/lattice_d3q7.h"
#include "physics/material_properties.h"
#include <cmath>
#include <vector>
#include <cstring>

using namespace lbm::physics;

// Test fixture
class TemperatureBoundsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }

        // Create material properties for Ti6Al4V
        strcpy(material.name, "Ti6Al4V");
        material.rho_solid = 4420.0f;
        material.rho_liquid = 4110.0f;
        material.cp_solid = 546.0f;
        material.cp_liquid = 831.0f;
        material.k_solid = 7.0f;
        material.k_liquid = 33.0f;
        material.T_solidus = 1878.0f;
        material.T_liquidus = 1928.0f;
        material.L_fusion = 286000.0f;
        material.T_vaporization = 3560.0f;
        material.L_vaporization = 9830000.0f;
    }

    MaterialProperties material;
};

/**
 * @brief Test upper temperature bound (T_max = 7000 K)
 */
TEST_F(TemperatureBoundsTest, UpperBoundEnforcement) {
    // Domain: 10x10x10
    int nx = 10, ny = 10, nz = 10;
    float thermal_diff = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, thermal_diff, false);

    // Create extreme heat source that would push T > 7000 K
    int num_cells = nx * ny * nz;
    std::vector<float> h_heat_source(num_cells, 1e12f);  // Extreme heating

    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));
    cudaMemcpy(d_heat_source, h_heat_source.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize to high temperature
    thermal.initialize(6500.0f);

    // Apply extreme heat source
    thermal.addHeatSource(d_heat_source, 1e-6f);
    thermal.computeTemperature();

    // Check that temperature is bounded
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_LE(h_temp[i], 7000.0f)
            << "REGRESSION: Temperature exceeded upper bound at cell " << i;
        EXPECT_GE(h_temp[i], 0.0f)
            << "Temperature became negative at cell " << i;
    }

    cudaFree(d_heat_source);
}

/**
 * @brief Test lower temperature bound (T_min = 0 K)
 */
TEST_F(TemperatureBoundsTest, LowerBoundEnforcement) {
    int nx = 10, ny = 10, nz = 10;
    float thermal_diff = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, thermal_diff, false);

    // Initialize to very low temperature
    thermal.initialize(10.0f);

    // Apply extreme cooling via radiation
    float dt = 1e-6f;
    float dx = 2e-6f;
    float epsilon = 0.9f;  // High emissivity
    float T_ambient = 0.0f;  // Absolute zero ambient

    // Apply radiation BC multiple times
    for (int step = 0; step < 100; ++step) {
        thermal.applyRadiationBC(dt, dx, epsilon, T_ambient);
    }

    thermal.computeTemperature();

    // Check that temperature is bounded above zero
    int num_cells = nx * ny * nz;
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_GE(h_temp[i], 0.0f)
            << "REGRESSION: Temperature became negative at cell " << i;
    }
}

/**
 * @brief Test that normal temperatures are unaffected
 */
TEST_F(TemperatureBoundsTest, NormalRangeUnaffected) {
    int nx = 10, ny = 10, nz = 10;
    float thermal_diff = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, thermal_diff, false);

    // Initialize to normal operating temperature
    float T_init = 3000.0f;  // Normal melt pool temperature
    thermal.initialize(T_init);

    thermal.computeTemperature();

    // Temperature should remain unchanged
    int num_cells = nx * ny * nz;
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_NEAR(h_temp[i], T_init, 1.0f)
            << "Normal temperature altered by bounds at cell " << i;
    }
}

/**
 * @brief GPU kernel test for temperature bounds
 */
__global__ void testTemperatureBoundsKernel(const float* g, float* T_out, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Sum distributions
    float T = 0.0f;
    for (int q = 0; q < 7; ++q) {
        T += g[idx * 7 + q];
    }

    // Apply bounds (as in computeTemperatureKernel)
    T = fmaxf(T, 0.0f);
    T = fminf(T, 7000.0f);

    T_out[idx] = T;
}

TEST_F(TemperatureBoundsTest, GPUBoundsKernelCorrectness) {
    int num_cells = 100;

    // Create distributions that sum to various temperatures
    std::vector<float> h_g(num_cells * 7);

    // Cell 0: Sum to -500 K (unphysical)
    for (int q = 0; q < 7; ++q) h_g[0 * 7 + q] = -71.43f;

    // Cell 1: Sum to 10000 K (too high)
    for (int q = 0; q < 7; ++q) h_g[1 * 7 + q] = 1428.57f;

    // Cell 2: Sum to 3000 K (normal)
    for (int q = 0; q < 7; ++q) h_g[2 * 7 + q] = 428.57f;

    // Fill rest with normal values
    for (int i = 3; i < num_cells; ++i) {
        for (int q = 0; q < 7; ++q) {
            h_g[i * 7 + q] = 300.0f / 7.0f;  // 300 K
        }
    }

    // Copy to device
    float *d_g, *d_T;
    cudaMalloc(&d_g, num_cells * 7 * sizeof(float));
    cudaMalloc(&d_T, num_cells * sizeof(float));
    cudaMemcpy(d_g, h_g.data(), num_cells * 7 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_cells + blockSize - 1) / blockSize;
    testTemperatureBoundsKernel<<<gridSize, blockSize>>>(d_g, d_T, num_cells);
    cudaDeviceSynchronize();

    // Copy results
    std::vector<float> h_T(num_cells);
    cudaMemcpy(h_T.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify bounds
    EXPECT_EQ(h_T[0], 0.0f) << "Lower bound not applied";
    EXPECT_EQ(h_T[1], 7000.0f) << "Upper bound not applied";
    EXPECT_NEAR(h_T[2], 3000.0f, 0.1f) << "Normal temperature altered";

    for (int i = 3; i < num_cells; ++i) {
        EXPECT_GE(h_T[i], 0.0f) << "Negative temperature at cell " << i;
        EXPECT_LE(h_T[i], 7000.0f) << "Temperature exceeded max at cell " << i;
    }

    cudaFree(d_g);
    cudaFree(d_T);
}

/**
 * @brief Test temperature bounds with phase change enabled
 */
TEST_F(TemperatureBoundsTest, BoundsWithPhaseChange) {
    int nx = 10, ny = 10, nz = 10;
    float thermal_diff = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, thermal_diff, true);

    // Initialize to extreme values
    int num_cells = nx * ny * nz;
    std::vector<float> h_temp_init(num_cells);

    // Half cells at 10000 K, half at -100 K
    for (int i = 0; i < num_cells; ++i) {
        h_temp_init[i] = (i < num_cells / 2) ? 10000.0f : -100.0f;
    }

    thermal.initialize(h_temp_init.data());
    thermal.computeTemperature();

    // Verify all cells are within bounds
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_GE(h_temp[i], 0.0f)
            << "Negative temperature with phase change at cell " << i;
        EXPECT_LE(h_temp[i], 7000.0f)
            << "Temperature exceeded max with phase change at cell " << i;
    }
}

/**
 * @brief Test that radiation BC handles bounded temperatures correctly
 */
TEST_F(TemperatureBoundsTest, RadiationBCWithBounds) {
    int nx = 10, ny = 10, nz = 10;
    float thermal_diff = 5.8e-6f;

    ThermalLBM thermal(nx, ny, nz, material, thermal_diff, false);

    // Initialize to near-upper-bound temperature
    thermal.initialize(6900.0f);

    // Apply radiation BC (should not cause overflow)
    float dt = 1e-7f;
    float dx = 2e-6f;
    thermal.applyRadiationBC(dt, dx, 0.35f, 300.0f);

    thermal.computeTemperature();

    // Verify no overflow or NaN
    int num_cells = nx * ny * nz;
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FALSE(std::isnan(h_temp[i]))
            << "NaN detected after radiation BC at cell " << i;
        EXPECT_LE(h_temp[i], 7000.0f)
            << "Temperature overflow after radiation BC at cell " << i;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
