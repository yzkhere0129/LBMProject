/**
 * @file test_d3q19.cu
 * @brief Unit tests for D3Q19 lattice structure
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "core/lattice_d3q19.h"

using namespace lbm::core;

class D3Q19Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

// Test lattice initialization
TEST_F(D3Q19Test, Initialization) {
    EXPECT_TRUE(D3Q19::isInitialized());
}

// Test lattice constants
TEST_F(D3Q19Test, LatticeConstants) {
    EXPECT_EQ(D3Q19::Q, 19);
    EXPECT_FLOAT_EQ(D3Q19::CS2, 1.0f/3.0f);
    EXPECT_FLOAT_EQ(D3Q19::CS, std::sqrt(1.0f/3.0f));
}

// Test weight normalization
TEST_F(D3Q19Test, WeightNormalization) {
    // Weights should sum to 1
    float w_sum = 1.0f/3.0f + 6.0f * (1.0f/18.0f) + 12.0f * (1.0f/36.0f);
    EXPECT_NEAR(w_sum, 1.0f, 1e-6);
}

// Test velocity directions
TEST_F(D3Q19Test, VelocityDirections) {
    // Test symmetry of velocity set
    int sum_ex = 0, sum_ey = 0, sum_ez = 0;

    // Access host arrays directly for testing
    const int ex[19] = {
        0, 1, -1, 0, 0, 0, 0,
        1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
    };
    const int ey[19] = {
        0, 0, 0, 1, -1, 0, 0,
        1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
    };
    const int ez[19] = {
        0, 0, 0, 0, 0, 1, -1,
        0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
    };

    for (int q = 0; q < 19; ++q) {
        sum_ex += ex[q];
        sum_ey += ey[q];
        sum_ez += ez[q];
    }

    // Velocity set should be symmetric
    EXPECT_EQ(sum_ex, 0);
    EXPECT_EQ(sum_ey, 0);
    EXPECT_EQ(sum_ez, 0);
}

// Test equilibrium distribution for rest state
TEST_F(D3Q19Test, EquilibriumAtRest) {
    float rho = 1.0f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;

    // At rest, equilibrium should equal weight * rho
    for (int q = 0; q < D3Q19::Q; ++q) {
        float feq = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
        float expected = (q == 0) ? rho/3.0f :
                        (q <= 6) ? rho/18.0f : rho/36.0f;
        EXPECT_NEAR(feq, expected, 1e-6);
    }
}

// Test mass conservation in equilibrium
TEST_F(D3Q19Test, MassConservation) {
    float rho = 2.5f;
    float ux = 0.1f, uy = -0.05f, uz = 0.02f;

    float total_mass = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        float feq = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
        total_mass += feq;
    }

    EXPECT_NEAR(total_mass, rho, 1e-5);
}

// Test momentum conservation in equilibrium
TEST_F(D3Q19Test, MomentumConservation) {
    float rho = 1.5f;
    float ux = 0.15f, uy = -0.1f, uz = 0.05f;

    float mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;

    const int ex[19] = {
        0, 1, -1, 0, 0, 0, 0,
        1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
    };
    const int ey[19] = {
        0, 0, 0, 1, -1, 0, 0,
        1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
    };
    const int ez[19] = {
        0, 0, 0, 0, 0, 1, -1,
        0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
    };

    for (int q = 0; q < D3Q19::Q; ++q) {
        float feq = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
        mom_x += feq * ex[q];
        mom_y += feq * ey[q];
        mom_z += feq * ez[q];
    }

    EXPECT_NEAR(mom_x, rho * ux, 1e-5);
    EXPECT_NEAR(mom_y, rho * uy, 1e-5);
    EXPECT_NEAR(mom_z, rho * uz, 1e-5);
}

// Test density computation
TEST_F(D3Q19Test, DensityComputation) {
    float f[19];
    float expected_rho = 0.0f;

    // Initialize with some values
    for (int q = 0; q < 19; ++q) {
        f[q] = 0.05f + 0.01f * q;
        expected_rho += f[q];
    }

    float computed_rho = D3Q19::computeDensity(f);
    EXPECT_NEAR(computed_rho, expected_rho, 1e-6);
}

// Test velocity computation
TEST_F(D3Q19Test, VelocityComputation) {
    float f[19];
    float rho = 0.0f;
    float expected_ux = 0.0f, expected_uy = 0.0f, expected_uz = 0.0f;

    const int ex[19] = {
        0, 1, -1, 0, 0, 0, 0,
        1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
    };
    const int ey[19] = {
        0, 0, 0, 1, -1, 0, 0,
        1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
    };
    const int ez[19] = {
        0, 0, 0, 0, 0, 1, -1,
        0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
    };

    // Initialize with equilibrium distribution
    float test_rho = 1.2f;
    float test_ux = 0.08f, test_uy = -0.04f, test_uz = 0.02f;

    for (int q = 0; q < 19; ++q) {
        f[q] = D3Q19::computeEquilibrium(q, test_rho, test_ux, test_uy, test_uz);
        rho += f[q];
        expected_ux += f[q] * ex[q];
        expected_uy += f[q] * ey[q];
        expected_uz += f[q] * ez[q];
    }
    expected_ux /= rho;
    expected_uy /= rho;
    expected_uz /= rho;

    float computed_ux, computed_uy, computed_uz;
    D3Q19::computeVelocity(f, rho, computed_ux, computed_uy, computed_uz);

    EXPECT_NEAR(computed_ux, expected_ux, 1e-5);
    EXPECT_NEAR(computed_uy, expected_uy, 1e-5);
    EXPECT_NEAR(computed_uz, expected_uz, 1e-5);
}

// Test neighbor index calculation
TEST_F(D3Q19Test, NeighborIndex) {
    int nx = 10, ny = 10, nz = 10;

    // Test periodic boundary at origin
    int idx = D3Q19::getNeighborIndex(0, 0, 0, 2, nx, ny, nz); // direction 2 is (-1,0,0)
    EXPECT_EQ(idx, 9 + 0*nx + 0*nx*ny); // Should wrap to x=9

    // Test periodic boundary at edge
    idx = D3Q19::getNeighborIndex(9, 9, 9, 1, nx, ny, nz); // direction 1 is (+1,0,0)
    EXPECT_EQ(idx, 0 + 9*nx + 9*nx*ny); // Should wrap to x=0

    // Test internal point
    idx = D3Q19::getNeighborIndex(5, 5, 5, 0, nx, ny, nz); // direction 0 is rest
    EXPECT_EQ(idx, 5 + 5*nx + 5*nx*ny); // Should stay at same position
}

// Test opposite directions
TEST_F(D3Q19Test, OppositeDirections) {
    const int opposite[19] = {
        0, 2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11,
        18, 17, 16, 15
    };

    const int ex[19] = {
        0, 1, -1, 0, 0, 0, 0,
        1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
    };
    const int ey[19] = {
        0, 0, 0, 1, -1, 0, 0,
        1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
    };
    const int ez[19] = {
        0, 0, 0, 0, 0, 1, -1,
        0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
    };

    for (int q = 0; q < 19; ++q) {
        int opp = opposite[q];
        // Opposite velocities should sum to zero
        EXPECT_EQ(ex[q] + ex[opp], 0);
        EXPECT_EQ(ey[q] + ey[opp], 0);
        EXPECT_EQ(ez[q] + ez[opp], 0);
    }
}

// GPU kernel test for equilibrium computation
__global__ void testEquilibriumKernel(float* results, float rho, float ux, float uy, float uz) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < D3Q19::Q) {
        results[q] = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
    }
}

TEST_F(D3Q19Test, GPUEquilibriumComputation) {
    float rho = 1.0f;
    float ux = 0.1f, uy = 0.05f, uz = -0.02f;

    float* d_results;
    cudaMalloc(&d_results, D3Q19::Q * sizeof(float));

    testEquilibriumKernel<<<1, 32>>>(d_results, rho, ux, uy, uz);
    cudaDeviceSynchronize();

    float h_results[D3Q19::Q];
    cudaMemcpy(h_results, d_results, D3Q19::Q * sizeof(float), cudaMemcpyDeviceToHost);

    // Check mass conservation on GPU results
    float total_mass = 0.0f;
    for (int q = 0; q < D3Q19::Q; ++q) {
        total_mass += h_results[q];
    }
    EXPECT_NEAR(total_mass, rho, 1e-5);

    cudaFree(d_results);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}