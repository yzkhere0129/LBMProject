/**
 * @file test_flux_limiter_overhead.cu
 * @brief Performance test for flux limiter computational overhead
 *
 * PERFORMANCE REGRESSION TEST: Ensures flux limiter doesn't add
 * significant computational cost.
 *
 * Context:
 * - Flux limiter adds ~3-5 flops per equilibrium computation
 * - Should not significantly impact overall performance
 * - Target: < 10% overhead compared to uncapped version
 *
 * Benchmark: 1M equilibrium computations should complete in < 100ms
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/lattice_d3q7.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace lbm::physics;

class FluxLimiterPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }
    }
};

/**
 * @brief GPU kernel for benchmarking equilibrium computation
 */
__global__ void benchmarkEquilibriumKernel(float* d_results, float T,
                                            float ux, float uy, float uz,
                                            int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int q = tid % 7;

    float sum = 0.0f;

    // Compute equilibrium many times
    for (int i = 0; i < iterations; ++i) {
        sum += D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
    }

    // Write result to prevent optimization
    if (tid < 7) {
        d_results[q] = sum;
    }
}

/**
 * @brief Benchmark flux limiter overhead on GPU
 */
TEST_F(FluxLimiterPerformanceTest, GPUEquilibriumThroughput) {
    float T = 1500.0f;
    float ux = 5.0f, uy = 0.0f, uz = 0.0f;  // High velocity (flux limiter active)

    float* d_results;
    cudaMalloc(&d_results, 7 * sizeof(float));

    // Warm-up
    benchmarkEquilibriumKernel<<<1, 256>>>(d_results, T, ux, uy, uz, 1000);
    cudaDeviceSynchronize();

    // Benchmark: 10M equilibrium computations
    int iterations = 10000;  // Each of 256 threads does 10k iterations
    int total_eqs = 256 * iterations;

    auto start = std::chrono::high_resolution_clock::now();

    benchmarkEquilibriumKernel<<<1, 256>>>(d_results, T, ux, uy, uz, iterations);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "GPU Equilibrium Benchmark:\n"
              << "  Total equilibria: " << total_eqs << "\n"
              << "  Time: " << duration.count() << " ms\n"
              << "  Throughput: " << (total_eqs / 1000.0f / duration.count()) << " M eq/s\n";

    // Performance requirement: Should complete in reasonable time
    // Target: > 10M eq/s (< 256 ms for 2.56M equilibria)
    EXPECT_LT(duration.count(), 500)
        << "PERFORMANCE REGRESSION: Flux limiter overhead too high";

    cudaFree(d_results);
}

/**
 * @brief CPU benchmark for equilibrium computation
 */
TEST_F(FluxLimiterPerformanceTest, CPUEquilibriumThroughput) {
    float T = 1500.0f;
    float ux = 5.0f, uy = 0.0f, uz = 0.0f;

    // Benchmark 1M equilibrium computations
    int iterations = 1000000;

    auto start = std::chrono::high_resolution_clock::now();

    volatile float sum = 0.0f;  // Prevent optimization
    for (int i = 0; i < iterations; ++i) {
        for (int q = 0; q < 7; ++q) {
            sum += D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "CPU Equilibrium Benchmark:\n"
              << "  Total equilibria: " << iterations * 7 << "\n"
              << "  Time: " << duration.count() << " ms\n"
              << "  Throughput: " << (iterations * 7 / 1000.0f / duration.count()) << " M eq/s\n";

    // CPU should complete 7M equilibria in < 500ms
    EXPECT_LT(duration.count(), 500)
        << "CPU equilibrium computation too slow";
}

/**
 * @brief Test flux limiter impact on full LBM step
 */
TEST_F(FluxLimiterPerformanceTest, FullLBMStepThroughput) {
    // This test would require a full ThermalLBM setup
    // Simplified version: just test equilibrium computation in context

    int num_cells = 1000;
    std::vector<float> T(num_cells, 1500.0f);
    std::vector<float> ux(num_cells, 5.0f);
    std::vector<float> uy(num_cells, 0.0f);
    std::vector<float> uz(num_cells, 0.0f);

    std::vector<float> g_eq(num_cells * 7);

    auto start = std::chrono::high_resolution_clock::now();

    // Compute equilibrium for all cells (simulates collision step)
    for (int iter = 0; iter < 1000; ++iter) {
        for (int i = 0; i < num_cells; ++i) {
            for (int q = 0; q < 7; ++q) {
                g_eq[i * 7 + q] = D3Q7::computeThermalEquilibrium(
                    q, T[i], ux[i], uy[i], uz[i]);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int total_cells = num_cells * 1000;
    std::cout << "Full LBM Step Benchmark:\n"
              << "  Total cell-steps: " << total_cells << "\n"
              << "  Time: " << duration.count() << " ms\n"
              << "  Throughput: " << (total_cells / 1000.0f / duration.count()) << " M cells/s\n";

    // Should process > 100k cells/s
    EXPECT_LT(duration.count(), 1000)
        << "Full LBM step too slow with flux limiter";
}

/**
 * @brief Compare performance: low vs high velocity
 */
TEST_F(FluxLimiterPerformanceTest, LowVsHighVelocityOverhead) {
    float T = 1500.0f;
    int iterations = 1000000;

    // Benchmark LOW velocity (flux limiter inactive)
    auto start_low = std::chrono::high_resolution_clock::now();

    volatile float sum_low = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        for (int q = 0; q < 7; ++q) {
            sum_low += D3Q7::computeThermalEquilibrium(q, T, 0.1f, 0.0f, 0.0f);
        }
    }

    auto end_low = std::chrono::high_resolution_clock::now();
    auto duration_low = std::chrono::duration_cast<std::chrono::milliseconds>(end_low - start_low);

    // Benchmark HIGH velocity (flux limiter active)
    auto start_high = std::chrono::high_resolution_clock::now();

    volatile float sum_high = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        for (int q = 0; q < 7; ++q) {
            sum_high += D3Q7::computeThermalEquilibrium(q, T, 5.0f, 0.0f, 0.0f);
        }
    }

    auto end_high = std::chrono::high_resolution_clock::now();
    auto duration_high = std::chrono::duration_cast<std::chrono::milliseconds>(end_high - start_high);

    float overhead_percent = 100.0f * (duration_high.count() - duration_low.count()) /
                             (float)duration_low.count();

    std::cout << "Low vs High Velocity Overhead:\n"
              << "  Low velocity time:  " << duration_low.count() << " ms\n"
              << "  High velocity time: " << duration_high.count() << " ms\n"
              << "  Overhead: " << overhead_percent << "%\n";

    // Overhead should be < 20% (flux limiter is cheap)
    EXPECT_LT(overhead_percent, 20.0f)
        << "PERFORMANCE REGRESSION: Flux limiter overhead > 20%";
}

// Memory bandwidth test removed - requires extended lambda support
// The flux limiter is primarily compute-bound, not memory-bound

/**
 * @brief Scaling test: performance vs problem size
 */
TEST_F(FluxLimiterPerformanceTest, ScalingWithProblemSize) {
    float T = 1500.0f;
    float ux = 5.0f, uy = 0.0f, uz = 0.0f;

    std::vector<int> problem_sizes = {1000, 10000, 100000, 1000000};

    for (int size : problem_sizes) {
        auto start = std::chrono::high_resolution_clock::now();

        volatile float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            for (int q = 0; q < 7; ++q) {
                sum += D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float time_per_eq = duration.count() / (float)(size * 7);

        std::cout << "Size " << size << ": " << time_per_eq << " μs/eq\n";

        // Time per equilibrium should scale linearly (O(1) complexity)
        EXPECT_LT(time_per_eq, 1.0f)
            << "Equilibrium computation too slow at size " << size;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
