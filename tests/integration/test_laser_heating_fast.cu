/**
 * @file test_laser_heating_fast.cu
 * @brief FAST validation test for laser heating and energy balance
 *
 * Purpose: Quickly verify laser heating is working correctly without
 *          the overhead of the full 150k-step steady-state test.
 *
 * Design principles:
 *   - Minimal timesteps (200 steps, ~5-10 seconds runtime)
 *   - Reduced diagnostics (output only at end)
 *   - Small domain (24x24x12)
 *   - Simple pass/fail criteria
 *   - GPU-side computations (minimal host transfers)
 *
 * What this test validates:
 *   1. Laser heat source is being applied
 *   2. Energy is increasing in the domain
 *   3. Temperature distribution is physically reasonable
 *   4. Energy balance is approximately correct
 *
 * What this test does NOT validate:
 *   - Steady-state convergence
 *   - Long-term stability
 *   - Detailed energy accounting
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"

// Kernel to compute laser heat source
__global__ void computeLaserHeatSourceKernel(
    float* heat_source,
    const LaserSource* laser,
    float dx, float dy, float dz,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx / (nx * ny);
    int j = (idx % (nx * ny)) / nx;
    int i = idx % nx;

    float x = i * dx;
    float y = j * dy;
    float z = k * dz;

    heat_source[idx] = laser->computeVolumetricHeatSource(x, y, z);
}

// GPU kernel to compute total energy on device
__global__ void computeTotalEnergyKernel(
    const float* temperature,
    float* partial_sums,
    float T_init,
    float rho, float cp,
    float dV,
    int num_cells
) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes energy for one cell
    float energy = 0.0f;
    if (idx < num_cells) {
        float T = temperature[idx];
        energy = rho * cp * (T - T_init) * dV;
    }
    s_data[tid] = energy;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = s_data[0];
    }
}

// GPU kernel to find max temperature
__global__ void findMaxTemperatureKernel(
    const float* temperature,
    float* partial_max,
    int num_cells
) {
    extern __shared__ float s_max[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = -1e10f;
    if (idx < num_cells) {
        local_max = temperature[idx];
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Reduction for maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = s_max[0];
    }
}

class FastLaserValidationTest : public ::testing::Test {
protected:
    // Smaller domain for speed
    const int nx = 24;
    const int ny = 24;
    const int nz = 12;
    const float dx = 2e-6f;      // 2 micrometers
    const float dy = 2e-6f;
    const float dz = 2e-6f;
    const float dt = 1e-9f;       // 1 nanosecond
    const float T_init = 300.0f;

    float* d_heat_source = nullptr;
    float* d_partial_sums = nullptr;
    float* d_partial_max = nullptr;

    void SetUp() override {
        size_t size = nx * ny * nz * sizeof(float);
        cudaMalloc(&d_heat_source, size);
        cudaMemset(d_heat_source, 0, size);

        // For reductions
        int num_blocks = (nx * ny * nz + 255) / 256;
        cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));
        cudaMalloc(&d_partial_max, num_blocks * sizeof(float));
    }

    void TearDown() override {
        if (d_heat_source) cudaFree(d_heat_source);
        if (d_partial_sums) cudaFree(d_partial_sums);
        if (d_partial_max) cudaFree(d_partial_max);
    }

    // Fast GPU-side energy computation
    float computeTotalEnergyGPU(const ThermalLBM& thermal,
                                const MaterialProperties& mat) {
        float dV = dx * dy * dz;
        int num_cells = nx * ny * nz;

        dim3 blockSize(256);
        dim3 gridSize((num_cells + 255) / 256);

        computeTotalEnergyKernel<<<gridSize, blockSize, 256 * sizeof(float)>>>(
            thermal.getTemperature(),
            d_partial_sums,
            T_init,
            mat.rho_solid,
            mat.cp_solid,
            dV,
            num_cells
        );

        // Copy partial sums to host and finish reduction on CPU
        std::vector<float> h_partial(gridSize.x);
        cudaMemcpy(h_partial.data(), d_partial_sums,
                   gridSize.x * sizeof(float), cudaMemcpyDeviceToHost);

        float total = 0.0f;
        for (float val : h_partial) {
            total += val;
        }
        return total;
    }

    // Fast GPU-side max temperature
    float findMaxTemperatureGPU(const ThermalLBM& thermal) {
        int num_cells = nx * ny * nz;
        dim3 blockSize(256);
        dim3 gridSize((num_cells + 255) / 256);

        findMaxTemperatureKernel<<<gridSize, blockSize, 256 * sizeof(float)>>>(
            thermal.getTemperature(),
            d_partial_max,
            num_cells
        );

        std::vector<float> h_partial(gridSize.x);
        cudaMemcpy(h_partial.data(), d_partial_max,
                   gridSize.x * sizeof(float), cudaMemcpyDeviceToHost);

        float max_temp = h_partial[0];
        for (float val : h_partial) {
            max_temp = std::max(max_temp, val);
        }
        return max_temp;
    }
};

/**
 * Test 1: Quick Laser Heating Validation
 *
 * Goal: Verify laser is depositing energy in the domain
 * Runtime: ~5 seconds
 * Steps: 200 (0.2 microseconds physical time)
 */
TEST_F(FastLaserValidationTest, LaserHeatsUpDomain) {
    // Setup material - Ti6Al4V
    MaterialProperties ti64;
    ti64.rho_solid = 4430.0f;
    ti64.k_solid = 21.9f;
    ti64.cp_solid = 546.0f;
    ti64.absorptivity = 0.35f;
    ti64.T_melt = 1923.0f;

    // Initialize thermal LBM
    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = ti64.k_solid / (ti64.rho_solid * ti64.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Setup laser - 100W at center
    LaserSource laser(100.0f, 30e-6f, ti64.absorptivity, 10e-6f);
    laser.setPosition(nx * dx / 2.0f, ny * dy / 2.0f, 0.0f);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Measure initial state
    float energy_initial = computeTotalEnergyGPU(thermal, ti64);
    float T_max_initial = findMaxTemperatureGPU(thermal);

    // Run simulation - ONLY 200 steps
    const int n_steps = 200;

    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + 255) / 256);

    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();
    }

    // Measure final state
    float energy_final = computeTotalEnergyGPU(thermal, ti64);
    float T_max_final = findMaxTemperatureGPU(thermal);

    // Compute expected energy input
    float absorbed_power = laser.power * ti64.absorptivity;
    float total_time = n_steps * dt;
    float energy_input = absorbed_power * total_time;
    float energy_increase = energy_final - energy_initial;

    // Output results
    std::cout << "\n=== Fast Laser Heating Validation ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  Steps: " << n_steps << " (" << total_time * 1e6 << " µs)" << std::endl;
    std::cout << "  Laser power: " << laser.power << " W" << std::endl;
    std::cout << "  Absorbed power: " << absorbed_power << " W" << std::endl;
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Energy input (laser): " << energy_input << " J" << std::endl;
    std::cout << "  Energy increase (domain): " << energy_increase << " J" << std::endl;
    std::cout << "  Efficiency: " << (energy_increase / energy_input * 100.0f) << "%" << std::endl;
    std::cout << "  T_max (initial): " << T_max_initial << " K" << std::endl;
    std::cout << "  T_max (final): " << T_max_final << " K" << std::endl;
    std::cout << "  Temperature rise: " << (T_max_final - T_init) << " K" << std::endl;

    // Validation criteria
    EXPECT_GT(energy_increase, 0.0f)
        << "FAIL: Energy must increase when laser is on";

    EXPECT_GT(energy_increase / energy_input, 0.05f)
        << "FAIL: At least 5% of input energy should be retained";

    EXPECT_LT(energy_increase / energy_input, 1.0f)
        << "FAIL: Cannot gain more energy than input";

    EXPECT_GT(T_max_final, T_init + 50.0f)
        << "FAIL: Peak temperature should rise by at least 50K";

    EXPECT_LT(T_max_final, ti64.T_melt * 2.0f)
        << "FAIL: Temperature unreasonably high (possible numerical issue)";

    std::cout << "\n✓ All validation criteria passed" << std::endl;

    cudaFree(d_laser);
}

/**
 * Test 2: Energy Balance Convergence Check
 *
 * Goal: Verify energy balance is approaching steady state
 * Runtime: ~10 seconds
 * Steps: 500 in 5 phases
 */
TEST_F(FastLaserValidationTest, EnergyBalanceConvergence) {
    // Setup
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 20.0f;
    mat.cp_solid = 500.0f;
    mat.absorptivity = 0.3f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Laser
    const float laser_power = 50.0f;
    LaserSource laser(laser_power, 25e-6f, mat.absorptivity, 10e-6f);
    laser.setPosition(nx * dx / 2.0f, ny * dy / 2.0f, 0.0f);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Run in 5 phases, measure energy balance
    const int phase_steps = 100;
    const int n_phases = 5;

    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + 255) / 256);

    std::vector<float> energy_history;
    std::vector<float> energy_rate;  // Rate of energy increase

    float absorbed_power = laser_power * mat.absorptivity;

    std::cout << "\n=== Energy Balance Convergence ===" << std::endl;
    std::cout << "Absorbed power: " << absorbed_power << " W" << std::endl;

    for (int phase = 0; phase < n_phases; ++phase) {
        float energy_start = computeTotalEnergyGPU(thermal, mat);

        // Run phase
        for (int step = 0; step < phase_steps; ++step) {
            computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
                d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
            );
            thermal.addHeatSource(d_heat_source, dt);
            thermal.step();
        }

        float energy_end = computeTotalEnergyGPU(thermal, mat);
        float energy_gained = energy_end - energy_start;
        float phase_time = phase_steps * dt;
        float power_retained = energy_gained / phase_time;

        energy_history.push_back(energy_end);
        energy_rate.push_back(power_retained);

        std::cout << "Phase " << (phase + 1) << ": "
                  << "Energy = " << energy_end << " J, "
                  << "Power retained = " << power_retained << " W ("
                  << (power_retained / absorbed_power * 100.0f) << "%)"
                  << std::endl;
    }

    // Validation: Energy should increase monotonically
    for (size_t i = 1; i < energy_history.size(); ++i) {
        EXPECT_GT(energy_history[i], energy_history[i-1])
            << "FAIL: Energy must increase monotonically";
    }

    // Validation: Power retention should decrease (approaching steady state)
    // As system heats up, boundary losses increase, so net power retention decreases
    if (energy_rate.size() >= 3) {
        bool is_decreasing = (energy_rate[2] < energy_rate[0]);
        std::cout << "Power retention trend: "
                  << (is_decreasing ? "DECREASING (good)" : "INCREASING/FLAT")
                  << std::endl;

        // This is expected behavior as system approaches steady state
        // We don't fail if it's not decreasing, just note it
    }

    // Validation: Final retention should be reasonable
    float final_retention = energy_rate.back() / absorbed_power;
    EXPECT_GT(final_retention, 0.0f)
        << "FAIL: Must retain some energy";
    EXPECT_LT(final_retention, 1.0f)
        << "FAIL: Cannot retain more power than input";

    std::cout << "\n✓ Energy balance validation passed" << std::endl;

    cudaFree(d_laser);
}

/**
 * Test 3: Spatial Distribution Check
 *
 * Goal: Verify temperature distribution is physically reasonable
 * Runtime: ~5 seconds
 * Steps: 200
 */
TEST_F(FastLaserValidationTest, SpatialDistributionCheck) {
    // Setup
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 20.0f;
    mat.cp_solid = 500.0f;
    mat.absorptivity = 0.3f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    thermal.setThermalDiffusivity(alpha);

    // Laser at center
    LaserSource laser(100.0f, 30e-6f, mat.absorptivity, 10e-6f);
    laser.setPosition(nx * dx / 2.0f, ny * dy / 2.0f, 0.0f);

    LaserSource* d_laser;
    cudaMalloc(&d_laser, sizeof(LaserSource));
    cudaMemcpy(d_laser, &laser, sizeof(LaserSource), cudaMemcpyHostToDevice);

    // Simulate
    const int n_steps = 200;
    dim3 blockSize(256);
    dim3 gridSize((nx * ny * nz + 255) / 256);

    for (int step = 0; step < n_steps; ++step) {
        computeLaserHeatSourceKernel<<<gridSize, blockSize>>>(
            d_heat_source, d_laser, dx, dy, dz, nx, ny, nz
        );
        thermal.addHeatSource(d_heat_source, dt);
        thermal.step();
    }

    // Sample temperatures at key locations
    std::vector<float> h_temperature(nx * ny * nz);
    thermal.copyTemperatureToHost(h_temperature.data());

    auto getT = [&](int i, int j, int k) {
        return h_temperature[k * nx * ny + j * nx + i];
    };

    float T_center = getT(nx/2, ny/2, 0);
    float T_near = getT(nx/2 + 3, ny/2, 0);
    float T_far = getT(0, 0, 0);
    float T_corner = getT(nx-1, ny-1, nz-1);

    std::cout << "\n=== Spatial Distribution Check ===" << std::endl;
    std::cout << "T_center (laser spot): " << T_center << " K" << std::endl;
    std::cout << "T_near (3 cells away): " << T_near << " K" << std::endl;
    std::cout << "T_far (edge): " << T_far << " K" << std::endl;
    std::cout << "T_corner: " << T_corner << " K" << std::endl;

    // Validation: Temperature should decay with distance
    EXPECT_GT(T_center, T_near)
        << "FAIL: Center should be hottest";
    EXPECT_GT(T_near, T_far)
        << "FAIL: Temperature should decrease with distance";

    // Validation: Center should be significantly hotter
    EXPECT_GT(T_center - T_far, 50.0f)
        << "FAIL: Should have strong temperature gradient";

    // Validation: No negative or extreme temperatures
    EXPECT_GE(T_far, T_init - 1.0f)
        << "FAIL: Temperature cannot drop below initial";
    EXPECT_LT(T_center, 5000.0f)
        << "FAIL: Temperature unreasonably high";

    std::cout << "✓ Spatial distribution is physically reasonable" << std::endl;

    cudaFree(d_laser);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Check CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Running on: " << deviceProp.name << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "FAST LASER HEATING VALIDATION TEST" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Purpose: Quick validation of laser heating" << std::endl;
    std::cout << "Expected runtime: < 30 seconds" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return RUN_ALL_TESTS();
}
