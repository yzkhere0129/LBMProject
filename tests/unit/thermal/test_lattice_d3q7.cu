/**
 * @file test_lattice_d3q7.cu
 * @brief Unit tests for D3Q7 lattice structure
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/lattice_d3q7.h"
#include <cmath>

using namespace lbm::physics;

// Test fixture for D3Q7 lattice tests
class D3Q7Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the D3Q7 lattice on device if not already done
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }
    }
};

// Test 1: Verify lattice constants and weights
TEST_F(D3Q7Test, LatticeConstantsVerification) {
    // Test number of velocities
    EXPECT_EQ(D3Q7::Q, 7);

    // Test speed of sound squared for D3Q7 thermal lattice
    // Note: D3Q7 uses CS2 = 1/4, different from D3Q19 fluid (CS2 = 1/3)
    EXPECT_FLOAT_EQ(D3Q7::CS2, 1.0f/4.0f);

    // Verify weights sum to 1.0
    float weight_sum = 0.0f;
    for (int q = 0; q < D3Q7::Q; ++q) {
        weight_sum += D3Q7::getWeight(q);
    }
    EXPECT_NEAR(weight_sum, 1.0f, 1e-6f);

    // Verify individual weights
    EXPECT_FLOAT_EQ(D3Q7::getWeight(0), 1.0f/4.0f);  // Rest particle
    for (int q = 1; q < 7; ++q) {
        EXPECT_FLOAT_EQ(D3Q7::getWeight(q), 1.0f/8.0f);  // Face directions
    }
}

// Test 2: Temperature computation from distribution functions
TEST_F(D3Q7Test, TemperatureComputation) {
    // Create a simple distribution function array
    float g[7];

    // Test case 1: Uniform distribution with T = 1.0
    float T_expected = 1.0f;
    for (int q = 0; q < 7; ++q) {
        g[q] = D3Q7::getWeight(q) * T_expected;
    }

    float T_computed = D3Q7::computeTemperature(g);
    EXPECT_NEAR(T_computed, T_expected, 1e-6f);

    // Test case 2: Non-uniform distribution
    T_expected = 2.5f;
    g[0] = 0.5f;
    g[1] = 0.3f;
    g[2] = 0.3f;
    g[3] = 0.4f;
    g[4] = 0.4f;
    g[5] = 0.3f;
    g[6] = 0.3f;

    T_computed = D3Q7::computeTemperature(g);
    float sum = 0.0f;
    for (int q = 0; q < 7; ++q) {
        sum += g[q];
    }
    EXPECT_NEAR(T_computed, sum, 1e-6f);
}

// Test 3: Thermal equilibrium distribution at rest
TEST_F(D3Q7Test, ThermalEquilibriumAtRest) {
    float T = 1.5f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;

    // At rest, equilibrium should be g_eq = w_i * T
    for (int q = 0; q < 7; ++q) {
        float g_eq = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        float expected = D3Q7::getWeight(q) * T;
        EXPECT_NEAR(g_eq, expected, 1e-6f) << "Failed at q = " << q;
    }

    // Verify sum equals temperature
    float sum = 0.0f;
    for (int q = 0; q < 7; ++q) {
        sum += D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
    }
    EXPECT_NEAR(sum, T, 1e-6f);
}

// Test 4: Thermal equilibrium with flow
TEST_F(D3Q7Test, ThermalEquilibriumWithFlow) {
    float T = 2.0f;
    float ux = 0.1f, uy = 0.05f, uz = -0.02f;

    // Compute equilibrium for all directions
    float g_eq[7];
    float sum = 0.0f;

    for (int q = 0; q < 7; ++q) {
        g_eq[q] = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        sum += g_eq[q];
    }

    // Temperature should be conserved
    EXPECT_NEAR(sum, T, 1e-5f);

    // Test specific directions
    // For q=1 (+x direction), we should have enhancement due to positive ux
    float g_eq_1 = D3Q7::computeThermalEquilibrium(1, T, ux, uy, uz);
    float expected_1 = D3Q7::getWeight(1) * T * (1.0f + ux / D3Q7::CS2);
    EXPECT_NEAR(g_eq_1, expected_1, 1e-6f);

    // For q=2 (-x direction), we should have reduction
    float g_eq_2 = D3Q7::computeThermalEquilibrium(2, T, ux, uy, uz);
    float expected_2 = D3Q7::getWeight(2) * T * (1.0f - ux / D3Q7::CS2);
    EXPECT_NEAR(g_eq_2, expected_2, 1e-6f);
}

// Test 5: Neighbor index calculation
TEST_F(D3Q7Test, NeighborIndexCalculation) {
    int nx = 10, ny = 10, nz = 10;

    // Test case 1: Interior point
    int x = 5, y = 5, z = 5;

    // Rest particle should return same index
    int idx = D3Q7::getThermalNeighborIndex(x, y, z, 0, nx, ny, nz);
    int expected = x + y * nx + z * nx * ny;
    EXPECT_EQ(idx, expected);

    // +x direction (q=1)
    idx = D3Q7::getThermalNeighborIndex(x, y, z, 1, nx, ny, nz);
    expected = (x + 1) + y * nx + z * nx * ny;
    EXPECT_EQ(idx, expected);

    // -x direction (q=2)
    idx = D3Q7::getThermalNeighborIndex(x, y, z, 2, nx, ny, nz);
    expected = (x - 1) + y * nx + z * nx * ny;
    EXPECT_EQ(idx, expected);

    // Test case 2: Boundary with CLAMPED (adiabatic) conditions
    // Note: D3Q7 thermal lattice uses clamped boundaries, not periodic
    x = 0; y = 0; z = 0;

    // -x direction should clamp to x=0 (not wrap)
    idx = D3Q7::getThermalNeighborIndex(x, y, z, 2, nx, ny, nz);
    expected = 0 + 0 * nx + 0 * nx * ny;  // Clamped at boundary
    EXPECT_EQ(idx, expected);

    // -y direction should clamp to y=0 (not wrap)
    idx = D3Q7::getThermalNeighborIndex(x, y, z, 4, nx, ny, nz);
    expected = 0 + 0 * nx + 0 * nx * ny;  // Clamped at boundary
    EXPECT_EQ(idx, expected);

    // -z direction should clamp to z=0 (not wrap)
    idx = D3Q7::getThermalNeighborIndex(x, y, z, 6, nx, ny, nz);
    expected = 0 + 0 * nx + 0 * nx * ny;  // Clamped at boundary
    EXPECT_EQ(idx, expected);
}

// GPU kernel for testing thermal equilibrium on device
__global__ void testThermalEquilibriumKernel(float* d_results, float T,
                                             float ux, float uy, float uz) {
    int q = threadIdx.x;
    if (q < 7) {
        d_results[q] = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
    }
}

// Test 6: GPU kernel correctness
TEST_F(D3Q7Test, GPUKernelCorrectness) {
    const int n_tests = 7;
    float T = 1.8f;
    float ux = 0.15f, uy = -0.1f, uz = 0.05f;

    // Allocate device memory
    float* d_results;
    cudaMalloc(&d_results, n_tests * sizeof(float));

    // Launch kernel
    testThermalEquilibriumKernel<<<1, 32>>>(d_results, T, ux, uy, uz);
    cudaDeviceSynchronize();

    // Copy results back
    float h_results[n_tests];
    cudaMemcpy(h_results, d_results, n_tests * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for (int q = 0; q < 7; ++q) {
        float expected = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        EXPECT_NEAR(h_results[q], expected, 1e-6f) << "GPU computation failed at q = " << q;
    }

    // Check sum equals temperature
    float sum = 0.0f;
    for (int q = 0; q < 7; ++q) {
        sum += h_results[q];
    }
    EXPECT_NEAR(sum, T, 1e-5f);

    // Clean up
    cudaFree(d_results);
}

// Test 7: Initialization check
TEST_F(D3Q7Test, InitializationCheck) {
    EXPECT_TRUE(D3Q7::isInitialized());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}