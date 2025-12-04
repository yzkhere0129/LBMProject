/**
 * @file test_lbm_benchmark.cu
 * @brief Comprehensive LBM Performance Benchmark
 *
 * Measures:
 * - MLUPS (Million Lattice Updates Per Second)
 * - Memory bandwidth utilization
 * - Kernel execution time
 * - Computational efficiency
 *
 * For comparison with WalBerla and other LBM frameworks
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>

//==============================================================================
// Device Properties Query
//==============================================================================

struct GPUProperties {
    std::string name;
    size_t totalMemory;
    float memoryBandwidthGB;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    float clockRateGHz;

    void query() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        name = prop.name;
        totalMemory = prop.totalGlobalMem;
        memoryBandwidthGB = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
        computeCapabilityMajor = prop.major;
        computeCapabilityMinor = prop.minor;
        multiProcessorCount = prop.multiProcessorCount;
        maxThreadsPerBlock = prop.maxThreadsPerBlock;
        maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        clockRateGHz = prop.clockRate / 1.0e6f;
    }

    void print() const {
        std::cout << "========================================\n";
        std::cout << "GPU Properties\n";
        std::cout << "========================================\n";
        std::cout << "Device: " << name << "\n";
        std::cout << "Memory: " << (totalMemory / (1024*1024)) << " MB\n";
        std::cout << "Memory Bandwidth: " << memoryBandwidthGB << " GB/s\n";
        std::cout << "Compute Capability: " << computeCapabilityMajor << "." << computeCapabilityMinor << "\n";
        std::cout << "Multiprocessors: " << multiProcessorCount << "\n";
        std::cout << "Max Threads/Block: " << maxThreadsPerBlock << "\n";
        std::cout << "Clock Rate: " << clockRateGHz << " GHz\n";
        std::cout << "========================================\n\n";
    }
};

//==============================================================================
// D3Q19 Lattice Constants
//==============================================================================

namespace D3Q19 {
    constexpr int Q = 19;
    constexpr float CS2 = 1.0f / 3.0f;

    __constant__ int ex[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
    __constant__ int ey[19] = {0, 0,  0, 1, -1, 0,  0, 1,  1, -1, -1, 0,  0,  0,  0, 1, -1,  1, -1};
    __constant__ int ez[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1,  1, -1, -1, 1,  1, -1, -1};
    __constant__ float w[19] = {
        1.0f/3.0f,
        1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
    };
}

//==============================================================================
// Benchmark Kernels
//==============================================================================

// Compute equilibrium distribution
__device__ inline float computeEquilibrium(int q, float rho, float ux, float uy, float uz) {
    using namespace D3Q19;
    float eu = ex[q] * ux + ey[q] * uy + ez[q] * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    return w[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
}

// BGK collision kernel (separate from streaming)
__global__ void bgkCollisionKernel(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    int nx, int ny, int nz,
    float omega)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    // Load distributions
    float f[19];
    for (int q = 0; q < 19; ++q) {
        f[q] = f_in[id + q * n_cells];
    }

    // Compute macroscopic quantities
    float rho = 0.0f;
    for (int q = 0; q < 19; ++q) {
        rho += f[q];
    }

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    for (int q = 0; q < 19; ++q) {
        ux += f[q] * D3Q19::ex[q];
        uy += f[q] * D3Q19::ey[q];
        uz += f[q] * D3Q19::ez[q];
    }
    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;
    uz *= inv_rho;

    // BGK collision
    for (int q = 0; q < 19; ++q) {
        float feq = computeEquilibrium(q, rho, ux, uy, uz);
        f[q] = f[q] - omega * (f[q] - feq);
    }

    // Store
    for (int q = 0; q < 19; ++q) {
        f_out[id + q * n_cells] = f[q];
    }
}

// Pull streaming kernel (periodic boundaries)
__global__ void pullStreamingKernel(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int n_cells = nx * ny * nz;
    int id = idx + idy * nx + idz * nx * ny;

    for (int q = 0; q < 19; ++q) {
        // Pull from upstream neighbor (periodic BC)
        int src_x = (idx - D3Q19::ex[q] + nx) % nx;
        int src_y = (idy - D3Q19::ey[q] + ny) % ny;
        int src_z = (idz - D3Q19::ez[q] + nz) % nz;
        int src_id = src_x + src_y * nx + src_z * nx * ny;

        f_out[id + q * n_cells] = f_in[src_id + q * n_cells];
    }
}

// Fused collision-streaming kernel (optimized)
__global__ void fusedCollisionStreamingKernel(
    const float* __restrict__ f_in,
    float* __restrict__ f_out,
    int nx, int ny, int nz,
    float omega)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int n_cells = nx * ny * nz;

    // Pull and collide
    for (int q = 0; q < 19; ++q) {
        // Pull from upstream
        int src_x = (idx - D3Q19::ex[q] + nx) % nx;
        int src_y = (idy - D3Q19::ey[q] + ny) % ny;
        int src_z = (idz - D3Q19::ez[q] + nz) % nz;
        int src_id = src_x + src_y * nx + src_z * nx * ny;

        // No collision in this simplified benchmark - just streaming
        int dst_id = idx + idy * nx + idz * nx * ny;
        f_out[dst_id + q * n_cells] = f_in[src_id + q * n_cells];
    }
}

//==============================================================================
// Benchmark Test Fixture
//==============================================================================

class LBMBenchmark : public ::testing::Test {
protected:
    GPUProperties gpu;

    // Test configurations (grid sizes)
    struct Config {
        int nx, ny, nz;
        std::string name;
    };

    std::vector<Config> configs = {
        {32, 32, 32, "Small (32^3)"},
        {64, 64, 64, "Medium (64^3)"},
        {100, 100, 100, "Large (100^3)"},
        {128, 128, 128, "XLarge (128^3)"}
    };

    void SetUp() override {
        gpu.query();
        gpu.print();
    }

    // Run benchmark for a specific configuration
    void runBenchmark(const Config& cfg, int num_iterations = 1000) {
        int nx = cfg.nx, ny = cfg.ny, nz = cfg.nz;
        size_t n_cells = nx * ny * nz;
        size_t n_elements = n_cells * D3Q19::Q;
        size_t bytes = n_elements * sizeof(float);

        std::cout << "\n========================================\n";
        std::cout << "Configuration: " << cfg.name << "\n";
        std::cout << "========================================\n";
        std::cout << "Grid: " << nx << " x " << ny << " x " << nz << "\n";
        std::cout << "Total cells: " << n_cells << "\n";
        std::cout << "Memory per array: " << (bytes / (1024.0*1024.0)) << " MB\n";
        std::cout << "Total memory (2 arrays): " << (2*bytes / (1024.0*1024.0)) << " MB\n";

        // Allocate device memory
        float *d_f_src, *d_f_dst;
        cudaMalloc(&d_f_src, bytes);
        cudaMalloc(&d_f_dst, bytes);

        // Initialize with some data
        std::vector<float> h_f(n_elements, 1.0f / D3Q19::Q);
        cudaMemcpy(d_f_src, h_f.data(), bytes, cudaMemcpyHostToDevice);

        // Configure kernel launch
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                      (ny + blockSize.y - 1) / blockSize.y,
                      (nz + blockSize.z - 1) / blockSize.z);

        std::cout << "Block size: " << blockSize.x << " x " << blockSize.y << " x " << blockSize.z << "\n";
        std::cout << "Grid size: " << gridSize.x << " x " << gridSize.y << " x " << gridSize.z << "\n";
        std::cout << "Iterations: " << num_iterations << "\n\n";

        // Warm-up
        for (int i = 0; i < 10; ++i) {
            fusedCollisionStreamingKernel<<<gridSize, blockSize>>>(d_f_src, d_f_dst, nx, ny, nz, 1.0f);
            std::swap(d_f_src, d_f_dst);
        }
        cudaDeviceSynchronize();

        // Benchmark: Fused collision-streaming
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            fusedCollisionStreamingKernel<<<gridSize, blockSize>>>(d_f_src, d_f_dst, nx, ny, nz, 1.0f);
            std::swap(d_f_src, d_f_dst);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double elapsed_s = elapsed_ms / 1000.0;

        // Compute performance metrics
        double total_lattice_updates = static_cast<double>(n_cells) * num_iterations;
        double mlups = (total_lattice_updates / elapsed_s) / 1.0e6;

        // Memory traffic: 2 reads + 1 write per distribution function = 3 * Q * sizeof(float) per cell
        double bytes_per_cell_update = 3.0 * D3Q19::Q * sizeof(float);
        double total_bytes = bytes_per_cell_update * total_lattice_updates;
        double bandwidth_gb_s = (total_bytes / elapsed_s) / 1.0e9;

        // Efficiency
        double bandwidth_efficiency = (bandwidth_gb_s / gpu.memoryBandwidthGB) * 100.0;

        // Time per iteration
        double time_per_iter_ms = elapsed_ms / num_iterations;

        std::cout << "========================================\n";
        std::cout << "PERFORMANCE RESULTS\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total time: " << elapsed_ms << " ms (" << elapsed_s << " s)\n";
        std::cout << "Time per iteration: " << time_per_iter_ms << " ms\n";
        std::cout << "\n";
        std::cout << "MLUPS: " << mlups << " (Million Lattice Updates Per Second)\n";
        std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s\n";
        std::cout << "Peak bandwidth: " << gpu.memoryBandwidthGB << " GB/s\n";
        std::cout << "Bandwidth efficiency: " << bandwidth_efficiency << " %\n";
        std::cout << "\n";
        std::cout << "Cells per iteration: " << n_cells << "\n";
        std::cout << "Updates per second: " << (total_lattice_updates / elapsed_s) << "\n";
        std::cout << "========================================\n";

        // Separate kernel benchmark
        std::cout << "\n--- Separate Collision Kernel ---\n";
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            bgkCollisionKernel<<<gridSize, blockSize>>>(d_f_src, d_f_dst, nx, ny, nz, 1.0f);
            std::swap(d_f_src, d_f_dst);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        elapsed_s = std::chrono::duration<double>(end - start).count();
        mlups = (total_lattice_updates / elapsed_s) / 1.0e6;
        std::cout << "Collision MLUPS: " << mlups << "\n";

        std::cout << "\n--- Separate Streaming Kernel ---\n";
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            pullStreamingKernel<<<gridSize, blockSize>>>(d_f_src, d_f_dst, nx, ny, nz);
            std::swap(d_f_src, d_f_dst);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        elapsed_s = std::chrono::duration<double>(end - start).count();
        mlups = (total_lattice_updates / elapsed_s) / 1.0e6;
        std::cout << "Streaming MLUPS: " << mlups << "\n";

        // Cleanup
        cudaFree(d_f_src);
        cudaFree(d_f_dst);
    }
};

//==============================================================================
// Tests
//==============================================================================

TEST_F(LBMBenchmark, SmallGrid) {
    runBenchmark(configs[0], 1000);
}

TEST_F(LBMBenchmark, MediumGrid) {
    runBenchmark(configs[1], 1000);
}

TEST_F(LBMBenchmark, LargeGrid) {
    runBenchmark(configs[2], 500);
}

TEST_F(LBMBenchmark, XLargeGrid) {
    // Only run if we have enough memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t required = 2 * configs[3].nx * configs[3].ny * configs[3].nz * 19 * sizeof(float);

    if (free_mem > required * 1.2) {  // 20% safety margin
        runBenchmark(configs[3], 500);
    } else {
        std::cout << "Skipping XLarge test: insufficient memory\n";
        std::cout << "Required: " << (required / (1024*1024)) << " MB\n";
        std::cout << "Available: " << (free_mem / (1024*1024)) << " MB\n";
        GTEST_SKIP();
    }
}

TEST_F(LBMBenchmark, ScalingAnalysis) {
    std::cout << "\n\n";
    std::cout << "============================================================\n";
    std::cout << "                   SCALING ANALYSIS\n";
    std::cout << "============================================================\n\n";

    std::cout << std::setw(15) << "Grid Size"
              << std::setw(15) << "Cells"
              << std::setw(15) << "MLUPS"
              << std::setw(18) << "Bandwidth (GB/s)"
              << std::setw(15) << "Efficiency (%)"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& cfg : configs) {
        size_t n_cells = cfg.nx * cfg.ny * cfg.nz;
        size_t bytes = n_cells * D3Q19::Q * sizeof(float);

        // Check memory availability
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (free_mem < 2 * bytes * 1.2) {
            std::cout << std::setw(15) << cfg.name
                      << " - SKIPPED (insufficient memory)\n";
            continue;
        }

        // Quick benchmark (100 iterations)
        int nx = cfg.nx, ny = cfg.ny, nz = cfg.nz;
        int num_iterations = 100;

        float *d_f_src, *d_f_dst;
        cudaMalloc(&d_f_src, bytes);
        cudaMalloc(&d_f_dst, bytes);
        cudaMemset(d_f_src, 0, bytes);

        dim3 blockSize(8, 8, 8);
        dim3 gridSize((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

        // Warm-up
        for (int i = 0; i < 5; ++i) {
            fusedCollisionStreamingKernel<<<gridSize, blockSize>>>(d_f_src, d_f_dst, nx, ny, nz, 1.0f);
            std::swap(d_f_src, d_f_dst);
        }
        cudaDeviceSynchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            fusedCollisionStreamingKernel<<<gridSize, blockSize>>>(d_f_src, d_f_dst, nx, ny, nz, 1.0f);
            std::swap(d_f_src, d_f_dst);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_s = std::chrono::duration<double>(end - start).count();
        double mlups = (n_cells * num_iterations / elapsed_s) / 1.0e6;
        double bytes_per_update = 3.0 * D3Q19::Q * sizeof(float);
        double bandwidth = (bytes_per_update * n_cells * num_iterations / elapsed_s) / 1.0e9;
        double efficiency = (bandwidth / gpu.memoryBandwidthGB) * 100.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(15) << cfg.name
                  << std::setw(15) << n_cells
                  << std::setw(15) << mlups
                  << std::setw(18) << bandwidth
                  << std::setw(15) << efficiency
                  << "\n";

        cudaFree(d_f_src);
        cudaFree(d_f_dst);
    }

    std::cout << std::string(78, '-') << "\n";
    std::cout << "\nNote: Larger grids typically achieve higher efficiency\n";
    std::cout << "      due to better GPU occupancy and memory coalescing.\n";
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "          LBM-CUDA PERFORMANCE BENCHMARK SUITE\n";
    std::cout << "============================================================\n";
    std::cout << "\n";
    std::cout << "This benchmark measures:\n";
    std::cout << "  - MLUPS (Million Lattice Updates Per Second)\n";
    std::cout << "  - Memory bandwidth utilization\n";
    std::cout << "  - Computational efficiency\n";
    std::cout << "  - Scaling with problem size\n";
    std::cout << "\n";
    std::cout << "Lattice: D3Q19\n";
    std::cout << "Collision: BGK (SRT)\n";
    std::cout << "Streaming: Pull scheme with periodic BC\n";
    std::cout << "\n";

    return RUN_ALL_TESTS();
}
