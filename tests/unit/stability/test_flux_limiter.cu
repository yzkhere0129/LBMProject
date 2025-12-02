/**
 * @file test_flux_limiter.cu
 * @brief Unit tests for TVD flux limiter in thermal equilibrium computation
 *
 * CRITICAL REGRESSION TEST: Verifies that the flux limiter prevents
 * negative populations at high Peclet numbers (Pe >> 1).
 *
 * Context:
 * - At high fluid velocities (v >> cs), thermal advection can create
 *   negative equilibrium populations without a flux limiter
 * - This causes numerical instability and NaN propagation
 *
 * Fix Location: lattice_d3q7.cu::computeThermalEquilibrium()
 * Fix Type: TVD flux limiter applied when |c_i·u| is large
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/lattice_d3q7.h"
#include <cmath>

using namespace lbm::physics;

// Test fixture for flux limiter tests
class FluxLimiterTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }
    }
};

/**
 * @brief CRITICAL TEST: Verify flux limiter prevents negative populations
 *
 * Test Scenario: Extreme velocity (v_lattice = 5.0 >> cs = 0.577)
 * Expected: ALL equilibrium populations must be >= 0
 * Failure Mode: Without limiter, populations can go negative -> NaN -> runaway
 */
TEST_F(FluxLimiterTest, PreventNegativePopulations) {
    // Test extreme velocity: v_lattice = 5.0 (>> cs = sqrt(1/3) ≈ 0.577)
    // This represents Pe ≈ 10-20 in realistic LPBF scenarios
    float T = 1000.0f;
    float ux = 5.0f, uy = 0.0f, uz = 0.0f;

    for (int q = 0; q < 7; ++q) {
        float g_eq = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);

        // CRITICAL: g_eq must ALWAYS be >= 0
        EXPECT_GE(g_eq, 0.0f)
            << "REGRESSION: Negative population detected at q=" << q
            << " (ux=" << ux << ", T=" << T << ")";

        // With flux limiter: should be > 0.05*w*T minimum
        // (limiter prevents population from dropping below small fraction of isotropic value)
        float w = D3Q7::getWeight(q);
        EXPECT_GE(g_eq, 0.05f * w * T)
            << "Population too small at q=" << q
            << " (may indicate limiter failure)";
    }
}

/**
 * @brief Test that sum of populations equals temperature (mass conservation)
 */
TEST_F(FluxLimiterTest, ConservesTemperature) {
    float T = 1500.0f;
    float ux = 5.0f, uy = 2.0f, uz = 1.0f;

    float sum = 0.0f;
    for (int q = 0; q < 7; ++q) {
        float g_eq = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        sum += g_eq;
    }

    // Temperature must be conserved (within numerical precision)
    EXPECT_NEAR(sum, T, 1e-3f * T)
        << "REGRESSION: Temperature not conserved with flux limiter";
}

/**
 * @brief Test flux limiter at various Peclet numbers
 */
TEST_F(FluxLimiterTest, VaryingPecletNumbers) {
    float T = 2000.0f;

    // Test range: Pe ~ 0.1 to Pe ~ 50
    float velocities[] = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f};

    for (float v : velocities) {
        float ux = v, uy = 0.0f, uz = 0.0f;

        for (int q = 0; q < 7; ++q) {
            float g_eq = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);

            EXPECT_GE(g_eq, 0.0f)
                << "Negative population at v=" << v << ", q=" << q;
        }

        // Check mass conservation
        float sum = 0.0f;
        for (int q = 0; q < 7; ++q) {
            sum += D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        }
        EXPECT_NEAR(sum, T, 1e-3f * T)
            << "Mass not conserved at v=" << v;
    }
}

/**
 * @brief Test that low velocity case is unaffected by limiter
 *
 * At low velocity (Pe < 1), limiter should be inactive
 * Result should match analytical equilibrium
 */
TEST_F(FluxLimiterTest, LowVelocityUnaffected) {
    // At low velocity, limiter should be inactive
    float T = 1000.0f;
    float ux = 0.1f, uy = 0.0f, uz = 0.0f;

    // Test +x direction (q=1)
    float g_eq = D3Q7::computeThermalEquilibrium(1, T, ux, uy, uz);
    float w = D3Q7::getWeight(1);
    float cx = 1.0f;
    float expected = w * T * (1.0f + (cx * ux) / D3Q7::CS2);

    // Should match analytical equilibrium (limiter inactive)
    EXPECT_NEAR(g_eq, expected, 1e-5f)
        << "REGRESSION: Limiter active at low velocity (should be inactive)";
}

/**
 * @brief Test multidirectional high velocity
 */
TEST_F(FluxLimiterTest, MultidirectionalHighVelocity) {
    float T = 1200.0f;
    float ux = 3.0f, uy = 2.5f, uz = 2.0f;

    for (int q = 0; q < 7; ++q) {
        float g_eq = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);

        EXPECT_GE(g_eq, 0.0f)
            << "REGRESSION: Negative population in multidirectional flow at q=" << q;
    }
}

/**
 * @brief GPU kernel test for flux limiter on device
 */
__global__ void testFluxLimiterKernel(float* d_results, float T,
                                       float ux, float uy, float uz) {
    int q = threadIdx.x;
    if (q < 7) {
        d_results[q] = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
    }
}

TEST_F(FluxLimiterTest, GPUFluxLimiterCorrectness) {
    float T = 1500.0f;
    float ux = 5.0f, uy = 0.0f, uz = 0.0f;

    // Allocate device memory
    float* d_results;
    cudaMalloc(&d_results, 7 * sizeof(float));

    // Launch kernel
    testFluxLimiterKernel<<<1, 32>>>(d_results, T, ux, uy, uz);
    cudaDeviceSynchronize();

    // Copy results back
    float h_results[7];
    cudaMemcpy(h_results, d_results, 7 * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify all populations are non-negative
    for (int q = 0; q < 7; ++q) {
        EXPECT_GE(h_results[q], 0.0f)
            << "REGRESSION: GPU flux limiter failed at q=" << q;
    }

    // Verify host-device consistency
    for (int q = 0; q < 7; ++q) {
        float expected = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        EXPECT_NEAR(h_results[q], expected, 1e-5f)
            << "Host-device mismatch at q=" << q;
    }

    cudaFree(d_results);
}

/**
 * @brief Stress test: Randomized high velocity scenarios
 */
TEST_F(FluxLimiterTest, RandomizedStressTest) {
    float T = 1000.0f;

    // Test 100 random high-velocity scenarios
    for (int i = 0; i < 100; ++i) {
        float ux = (rand() / (float)RAND_MAX) * 10.0f - 5.0f;  // [-5, 5]
        float uy = (rand() / (float)RAND_MAX) * 10.0f - 5.0f;
        float uz = (rand() / (float)RAND_MAX) * 10.0f - 5.0f;

        for (int q = 0; q < 7; ++q) {
            float g_eq = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);

            ASSERT_GE(g_eq, 0.0f)
                << "REGRESSION: Negative population in random test #" << i
                << " at q=" << q << " (u=[" << ux << "," << uy << "," << uz << "])";
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
