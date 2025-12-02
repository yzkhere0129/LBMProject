/**
 * @file test_streaming.cu
 * @brief Unit tests for streaming operations
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "core/streaming.h"
#include "core/lattice_d3q19.h"

using namespace lbm::core;

class StreamingTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

// Test boundary detection
TEST_F(StreamingTest, BoundaryDetection) {
    int nx = 10, ny = 10, nz = 10;

    // Test corners
    EXPECT_TRUE(Streaming::isAtBoundary(0, 0, 0, nx, ny, nz));
    EXPECT_TRUE(Streaming::isAtBoundary(nx-1, ny-1, nz-1, nx, ny, nz));

    // Test edges
    EXPECT_TRUE(Streaming::isAtBoundary(0, 5, 5, nx, ny, nz));
    EXPECT_TRUE(Streaming::isAtBoundary(nx-1, 5, 5, nx, ny, nz));

    // Test interior
    EXPECT_FALSE(Streaming::isAtBoundary(5, 5, 5, nx, ny, nz));
    EXPECT_FALSE(Streaming::isAtBoundary(3, 7, 2, nx, ny, nz));
}

// Test boundary type identification
TEST_F(StreamingTest, BoundaryTypeIdentification) {
    int nx = 10, ny = 10, nz = 10;

    // Test X boundaries
    unsigned int type = Streaming::getBoundaryType(0, 5, 5, nx, ny, nz);
    EXPECT_EQ(type & Streaming::BOUNDARY_X_MIN, Streaming::BOUNDARY_X_MIN);
    EXPECT_EQ(type & Streaming::BOUNDARY_X_MAX, 0);

    type = Streaming::getBoundaryType(nx-1, 5, 5, nx, ny, nz);
    EXPECT_EQ(type & Streaming::BOUNDARY_X_MAX, Streaming::BOUNDARY_X_MAX);
    EXPECT_EQ(type & Streaming::BOUNDARY_X_MIN, 0);

    // Test corner (multiple boundaries)
    type = Streaming::getBoundaryType(0, 0, 0, nx, ny, nz);
    EXPECT_EQ(type & Streaming::BOUNDARY_X_MIN, Streaming::BOUNDARY_X_MIN);
    EXPECT_EQ(type & Streaming::BOUNDARY_Y_MIN, Streaming::BOUNDARY_Y_MIN);
    EXPECT_EQ(type & Streaming::BOUNDARY_Z_MIN, Streaming::BOUNDARY_Z_MIN);

    // Test interior (no boundaries)
    type = Streaming::getBoundaryType(5, 5, 5, nx, ny, nz);
    EXPECT_EQ(type, Streaming::BOUNDARY_NONE);
}

// Test pull streaming kernel
TEST_F(StreamingTest, PullStreamingKernel) {
    const int nx = 8, ny = 8, nz = 8;
    const int n_cells = nx * ny * nz;
    const int n_total = n_cells * D3Q19::Q;

    // Allocate device memory
    float *d_f_src, *d_f_dst;
    cudaMalloc(&d_f_src, n_total * sizeof(float));
    cudaMalloc(&d_f_dst, n_total * sizeof(float));

    // Initialize source with unique values for tracking
    float* h_f_src = new float[n_total];
    for (int i = 0; i < n_total; ++i) {
        h_f_src[i] = static_cast<float>(i) / n_total;
    }

    cudaMemcpy(d_f_src, h_f_src, n_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_f_dst, 0, n_total * sizeof(float));

    // Launch pull streaming kernel
    dim3 block(4, 4, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    pullStreamingKernel<<<grid, block>>>(d_f_src, d_f_dst, nx, ny, nz);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

    // Copy results back
    float* h_f_dst = new float[n_total];
    cudaMemcpy(h_f_dst, d_f_dst, n_total * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify that streaming occurred (values should have moved)
    bool streaming_occurred = false;
    for (int i = 0; i < n_total; ++i) {
        if (h_f_dst[i] != 0.0f) {
            streaming_occurred = true;
            break;
        }
    }
    EXPECT_TRUE(streaming_occurred);

    // Clean up
    delete[] h_f_src;
    delete[] h_f_dst;
    cudaFree(d_f_src);
    cudaFree(d_f_dst);
}

// Test periodic streaming
TEST_F(StreamingTest, PeriodicStreamingKernel) {
    const int nx = 4, ny = 4, nz = 4;
    const int n_cells = nx * ny * nz;
    const int n_total = n_cells * D3Q19::Q;

    // Allocate device memory
    float *d_f_src, *d_f_dst;
    cudaMalloc(&d_f_src, n_total * sizeof(float));
    cudaMalloc(&d_f_dst, n_total * sizeof(float));

    // Initialize with a pattern to test periodicity
    float* h_f_src = new float[n_total];
    for (int id = 0; id < n_cells; ++id) {
        int x = id % nx;
        int y = (id / nx) % ny;
        int z = id / (nx * ny);

        for (int q = 0; q < D3Q19::Q; ++q) {
            // Create a pattern that will help verify periodic boundaries
            h_f_src[id + q * n_cells] = static_cast<float>(x + y*10 + z*100 + q*1000);
        }
    }

    cudaMemcpy(d_f_src, h_f_src, n_total * sizeof(float), cudaMemcpyHostToDevice);

    // Launch periodic streaming kernel
    dim3 block(4, 4, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    periodicStreamingKernel<<<grid, block>>>(d_f_src, d_f_dst, nx, ny, nz);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

    // Copy results back
    float* h_f_dst = new float[n_total];
    cudaMemcpy(h_f_dst, d_f_dst, n_total * sizeof(float), cudaMemcpyDeviceToHost);

    // Check mass conservation (total sum should be preserved)
    float sum_src = 0.0f, sum_dst = 0.0f;
    for (int i = 0; i < n_total; ++i) {
        sum_src += h_f_src[i];
        sum_dst += h_f_dst[i];
    }
    EXPECT_NEAR(sum_dst, sum_src, 1e-4);

    // Clean up
    delete[] h_f_src;
    delete[] h_f_dst;
    cudaFree(d_f_src);
    cudaFree(d_f_dst);
}

// Test streaming preserves uniform flow
TEST_F(StreamingTest, UniformFlowPreservation) {
    const int nx = 8, ny = 8, nz = 8;
    const int n_cells = nx * ny * nz;
    const int n_total = n_cells * D3Q19::Q;

    // Allocate device memory
    float *d_f_src, *d_f_dst;
    cudaMalloc(&d_f_src, n_total * sizeof(float));
    cudaMalloc(&d_f_dst, n_total * sizeof(float));

    // Initialize with uniform equilibrium distribution
    float* h_f_src = new float[n_total];
    float rho = 1.0f;
    float ux = 0.05f, uy = 0.0f, uz = 0.0f;

    for (int id = 0; id < n_cells; ++id) {
        for (int q = 0; q < D3Q19::Q; ++q) {
            h_f_src[id + q * n_cells] = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
        }
    }

    cudaMemcpy(d_f_src, h_f_src, n_total * sizeof(float), cudaMemcpyHostToDevice);

    // Perform multiple streaming steps
    for (int step = 0; step < 10; ++step) {
        dim3 block(4, 4, 4);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y,
                  (nz + block.z - 1) / block.z);

        periodicStreamingKernel<<<grid, block>>>(d_f_src, d_f_dst, nx, ny, nz);
        cudaDeviceSynchronize();

        // Swap buffers
        float* temp = d_f_src;
        d_f_src = d_f_dst;
        d_f_dst = temp;
    }

    // Copy final result back
    float* h_f_final = new float[n_total];
    cudaMemcpy(h_f_final, d_f_src, n_total * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that uniform flow is preserved
    for (int id = 0; id < n_cells; ++id) {
        float local_rho = 0.0f;
        for (int q = 0; q < D3Q19::Q; ++q) {
            local_rho += h_f_final[id + q * n_cells];
        }
        EXPECT_NEAR(local_rho, rho, 1e-5);
    }

    // Clean up
    delete[] h_f_src;
    delete[] h_f_final;
    cudaFree(d_f_src);
    cudaFree(d_f_dst);
}

// Test push streaming kernel
// DISABLED: pushStreamingKernel has been deprecated due to race conditions
// Use PullStreamingKernel or PeriodicStreamingKernel instead
TEST_F(StreamingTest, DISABLED_PushStreamingKernel_DEPRECATED) {
    // This test has been disabled because the push streaming kernel
    // has race conditions that can cause data corruption.
    // The function has been renamed to pushStreamingKernel_DEPRECATED_DO_NOT_USE
    // and should not be used in production code.
    //
    // Reason for deprecation:
    // - Multiple threads may write to the same destination cell
    // - atomicExch only makes individual writes atomic, but multiple writes
    //   to the same location will still overwrite each other, losing data
    //
    // Recommendation: Use pullStreamingKernel instead, which has no race conditions

    GTEST_SKIP() << "Push streaming kernel has been deprecated due to race conditions. "
                 << "Use pullStreamingKernel instead.";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}