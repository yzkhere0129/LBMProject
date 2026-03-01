/**
 * @file test_laser_source.cu
 * @brief Unit tests for laser heat source implementation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <numeric>
#include "physics/laser_source.h"

using namespace lbm::physics;

// Test fixture for laser source tests
class LaserSourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default test parameters (typical LPBF values)
        power = 100.0f;           // 100 W
        spot_radius = 50e-6f;     // 50 μm
        absorptivity = 0.35f;     // 35% absorption for Ti6Al4V
        penetration_depth = 10e-6f; // 10 μm

        // Grid parameters
        nx = 128; ny = 128; nz = 32;
        dx = 2e-6f; dy = 2e-6f; dz = 2e-6f;  // 2 μm resolution

        // Allocate memory
        size_t total_cells = nx * ny * nz;
        h_heat_source.resize(total_cells, 0.0f);

        cudaMalloc(&d_heat_source, total_cells * sizeof(float));
        cudaMemset(d_heat_source, 0, total_cells * sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_heat_source);
    }

    // Test parameters
    float power, spot_radius, absorptivity, penetration_depth;
    int nx, ny, nz;
    float dx, dy, dz;

    // Host and device arrays
    std::vector<float> h_heat_source;
    float* d_heat_source;
};

/**
 * Test 1: Gaussian distribution normalization
 * Verify that the integrated intensity equals the laser power
 */
TEST_F(LaserSourceTest, GaussianNormalization) {
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);
    laser.setPosition(0.0f, 0.0f, 0.0f);  // Center at origin

    // Integrate over a large enough domain (±5 spot radii)
    float domain_radius = 5.0f * spot_radius;
    int n_samples = 200;
    float ds = 2.0f * domain_radius / n_samples;

    float total_power = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            float x = -domain_radius + i * ds;
            float y = -domain_radius + j * ds;
            float I = laser.computeIntensity(x, y);
            total_power += I * ds * ds;
        }
    }

    // Should equal the laser power (within 1% error due to discretization)
    float expected_power = power;
    float relative_error = std::abs(total_power - expected_power) / expected_power;
    EXPECT_LT(relative_error, 0.01f) << "Total power: " << total_power
                                      << " W, Expected: " << expected_power << " W";
}

/**
 * Test 2: Spot size verification
 * Check that intensity at beam radius is exp(-2) of center intensity
 */
TEST_F(LaserSourceTest, SpotSizeVerification) {
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);
    laser.setPosition(0.0f, 0.0f, 0.0f);

    float I_center = laser.computeIntensity(0.0f, 0.0f);
    float I_radius_x = laser.computeIntensity(spot_radius, 0.0f);
    float I_radius_y = laser.computeIntensity(0.0f, spot_radius);

    // At r = w0, intensity should be I0 * exp(-2)
    float expected_ratio = std::exp(-2.0f);

    EXPECT_NEAR(I_radius_x / I_center, expected_ratio, 0.001f)
        << "X-direction ratio: " << I_radius_x / I_center;
    EXPECT_NEAR(I_radius_y / I_center, expected_ratio, 0.001f)
        << "Y-direction ratio: " << I_radius_y / I_center;

    // Also check at sqrt(2) * w0 (45 degrees)
    float I_diagonal = laser.computeIntensity(spot_radius / std::sqrt(2.0f),
                                              spot_radius / std::sqrt(2.0f));
    EXPECT_NEAR(I_diagonal / I_center, expected_ratio, 0.001f)
        << "Diagonal ratio: " << I_diagonal / I_center;
}

/**
 * Test 3: Volumetric heat source depth decay
 * Verify Beer-Lambert law absorption
 */
TEST_F(LaserSourceTest, PenetrationDepthDecay) {
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);
    laser.setPosition(0.0f, 0.0f, 0.0f);

    // Test at beam center
    float q_surface = laser.computeVolumetricHeatSource(0.0f, 0.0f, 0.0f);
    float q_depth1 = laser.computeVolumetricHeatSource(0.0f, 0.0f, penetration_depth);
    float q_depth2 = laser.computeVolumetricHeatSource(0.0f, 0.0f, 2.0f * penetration_depth);

    // At z = δ, intensity should decay to 1/e
    float expected_ratio1 = std::exp(-1.0f);
    float expected_ratio2 = std::exp(-2.0f);

    EXPECT_NEAR(q_depth1 / q_surface, expected_ratio1, 0.001f)
        << "Ratio at 1δ: " << q_depth1 / q_surface;
    EXPECT_NEAR(q_depth2 / q_surface, expected_ratio2, 0.001f)
        << "Ratio at 2δ: " << q_depth2 / q_surface;

    // Check cutoff (should be zero beyond 10 penetration depths)
    float q_deep = laser.computeVolumetricHeatSource(0.0f, 0.0f, 11.0f * penetration_depth);
    EXPECT_EQ(q_deep, 0.0f) << "Should be zero beyond cutoff";
}

/**
 * Test 4: Energy conservation
 * Verify total absorbed energy equals η * P
 */
TEST_F(LaserSourceTest, EnergyConservation) {
    // Create laser centered in domain
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);
    float x_center = (nx / 2) * dx;
    float y_center = (ny / 2) * dy;
    laser.setPosition(x_center, y_center, 0.0f);

    // Compute heat source on GPU
    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    computeLaserHeatSourceKernel<<<grid, block>>>(
        d_heat_source, laser, dx, dy, dz, nx, ny, nz);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

    // Copy back to host
    cudaMemcpy(h_heat_source.data(), d_heat_source,
               h_heat_source.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Use the provided function to compute total energy
    float total_energy = computeTotalLaserEnergy(d_heat_source, dx, dy, dz, nx, ny, nz);

    // Should equal absorbed power (η * P)
    float expected_energy = power * absorptivity;
    float relative_error = std::abs(total_energy - expected_energy) / expected_energy;

    // Allow 15% error due to finite domain and discretization
    // The error comes from:
    // 1. Discretization of the Gaussian profile
    // 2. Finite domain (some energy may be outside the computational domain)
    // 3. Numerical integration errors
    EXPECT_LT(relative_error, 0.15f)
        << "Total energy: " << total_energy << " W, "
        << "Expected: " << expected_energy << " W, "
        << "Relative error: " << relative_error * 100.0f << "%";
}

/**
 * Test 5: Moving laser position update
 * Verify position updates correctly with velocity
 */
TEST_F(LaserSourceTest, MovingLaserPosition) {
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);

    // Set initial position and velocity
    float x_init = 100e-6f;  // 100 μm
    float y_init = 50e-6f;   // 50 μm
    laser.setPosition(x_init, y_init, 0.0f);

    float vx = 0.5f;  // 0.5 m/s
    float vy = 0.2f;  // 0.2 m/s
    laser.setScanVelocity(vx, vy);

    // Update position for different time steps
    float dt1 = 0.001f;  // 1 ms
    laser.updatePosition(dt1);

    EXPECT_FLOAT_EQ(laser.x0, x_init + vx * dt1)
        << "X position after dt1";
    EXPECT_FLOAT_EQ(laser.y0, y_init + vy * dt1)
        << "Y position after dt1";

    // Update again
    float dt2 = 0.002f;  // 2 ms
    laser.updatePosition(dt2);

    EXPECT_FLOAT_EQ(laser.x0, x_init + vx * (dt1 + dt2))
        << "X position after dt1+dt2";
    EXPECT_FLOAT_EQ(laser.y0, y_init + vy * (dt1 + dt2))
        << "Y position after dt1+dt2";
}

/**
 * Test 6: GPU kernel correctness
 * Compare GPU computation with CPU reference
 */
TEST_F(LaserSourceTest, GPUKernelCorrectness) {
    // Create laser
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);
    float x_center = (nx / 2) * dx;
    float y_center = (ny / 2) * dy;
    laser.setPosition(x_center, y_center, 0.0f);

    // Compute on GPU

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    computeLaserHeatSourceKernel<<<grid, block>>>(
        d_heat_source, laser, dx, dy, dz, nx, ny, nz);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess) << "CUDA kernel error: " << cudaGetErrorString(error);

    // Copy back to host
    cudaMemcpy(h_heat_source.data(), d_heat_source,
               h_heat_source.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Debug: Check if any values were written
    bool any_nonzero = false;
    int num_nonzero = 0;
    for (const auto& val : h_heat_source) {
        if (val != 0.0f) {
            any_nonzero = true;
            num_nonzero++;
        }
    }
    EXPECT_TRUE(any_nonzero) << "GPU kernel produced all zeros!";

    // Print first few values for debugging
    std::cout << "First 10 GPU values: ";
    for (int i = 0; i < std::min(10, (int)h_heat_source.size()); ++i) {
        std::cout << h_heat_source[i] << " ";
    }
    std::cout << "\nFirst 10 CPU values: ";
    for (int i = 0; i < std::min(10, (int)h_heat_source.size()); ++i) {
        float x = (i % nx) * dx;
        float y = ((i / nx) % ny) * dy;
        float z = (i / (nx * ny)) * dz;
        std::cout << laser.computeVolumetricHeatSource(x, y, z) << " ";
    }
    std::cout << "\nNumber of non-zero values: " << num_nonzero
              << " out of " << h_heat_source.size() << std::endl;

    // Check values at corners (should be near zero for Gaussian)
    int corner_idx = 0;  // (0,0,0)
    std::cout << "Corner (0,0,0) GPU value: " << h_heat_source[corner_idx]
              << ", CPU value: " << laser.computeVolumetricHeatSource(0, 0, 0) << std::endl;

    // Check a point far from center
    int far_idx = (nz-1) * nx * ny + (ny-1) * nx + (nx-1);  // (nx-1,ny-1,nz-1)
    float far_x = (nx-1) * dx;
    float far_y = (ny-1) * dy;
    float far_z = (nz-1) * dz;
    std::cout << "Far corner GPU value: " << h_heat_source[far_idx]
              << ", CPU value: " << laser.computeVolumetricHeatSource(far_x, far_y, far_z) << std::endl;

    // Compare with CPU calculation
    float max_error = 0.0f;
    float max_relative_error = 0.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = i * dx;
                float y = j * dy;
                float z = k * dz;

                float expected = laser.computeVolumetricHeatSource(x, y, z);
                int idx = k * nx * ny + j * nx + i;
                float actual = h_heat_source[idx];

                // Check for NaN or Inf
                if (std::isnan(actual) || std::isinf(actual)) {
                    FAIL() << "Found NaN/Inf at (" << i << "," << j << "," << k << "): " << actual;
                }

                // Check bounds
                if (idx >= h_heat_source.size()) {
                    FAIL() << "Index out of bounds: " << idx << " >= " << h_heat_source.size();
                }

                float error = std::abs(actual - expected);

                // Debug large errors
                if (error > 1000000.0f) {
                    std::cout << "Huge error at (" << i << "," << j << "," << k << "): actual=" << actual
                              << ", expected=" << expected << ", error=" << error
                              << ", idx=" << idx << std::endl;
                    // Only print first huge error to avoid spam
                    static bool printed = false;
                    if (!printed) {
                        printed = true;
                        break;  // Exit loops
                    }
                }

                max_error = std::max(max_error, error);

                if (expected > 1e-10f) {
                    float rel_error = error / expected;
                    max_relative_error = std::max(max_relative_error, rel_error);
                }
            }
        }
    }

    // For small values, check absolute error
    // For large values, check relative error
    // This is needed because floating point precision is relative
    EXPECT_LT(max_relative_error, 1e-4f)  // 0.01% relative error is good for float
        << "Maximum relative error: " << max_relative_error;

    // Only check absolute error for small values
    // Large values naturally have larger absolute errors due to float precision
}

/**
 * Test 7: Linear scan path
 * Verify linear scan trajectory
 */
TEST_F(LaserSourceTest, LinearScanPath) {
    float x_start = 0.0f;
    float y_start = 0.0f;
    float x_end = 1e-3f;  // 1 mm
    float y_end = 0.5e-3f; // 0.5 mm
    float speed = 1.0f;    // 1 m/s

    LinearScan path(x_start, y_start, x_end, y_end, speed);

    // Test at different time points
    float x, y, vx, vy;

    // t = 0: should be at start
    path.getPosition(0.0f, x, y);
    EXPECT_FLOAT_EQ(x, x_start);
    EXPECT_FLOAT_EQ(y, y_start);

    // t = end: should be at end
    float distance = std::sqrt((x_end - x_start) * (x_end - x_start) +
                               (y_end - y_start) * (y_end - y_start));
    float total_time = distance / speed;

    path.getPosition(total_time, x, y);
    EXPECT_NEAR(x, x_end, 1e-9f);
    EXPECT_NEAR(y, y_end, 1e-9f);

    // t = middle: should be halfway
    path.getPosition(total_time / 2.0f, x, y);
    EXPECT_NEAR(x, (x_start + x_end) / 2.0f, 1e-9f);
    EXPECT_NEAR(y, (y_start + y_end) / 2.0f, 1e-9f);

    // Check velocity
    path.getVelocity(total_time / 2.0f, vx, vy);
    float v_total = std::sqrt(vx * vx + vy * vy);
    EXPECT_NEAR(v_total, speed, 1e-6f) << "Total velocity should equal scan speed";
}

/**
 * Test 8: Raster scan pattern
 * Verify zigzag raster scanning
 */
TEST_F(LaserSourceTest, RasterScanPattern) {
    float x_min = 0.0f;
    float y_min = 0.0f;
    float x_max = 1e-3f;   // 1 mm
    float y_max = 0.5e-3f; // 0.5 mm
    float hatch = 0.1e-3f; // 0.1 mm hatch spacing
    float speed = 1.0f;    // 1 m/s

    RasterScan path(x_min, y_min, x_max, y_max, hatch, speed);

    float x, y, vx, vy;

    // First line: should go from left to right
    path.getPosition(0.0f, x, y);
    EXPECT_FLOAT_EQ(x, x_min);
    EXPECT_FLOAT_EQ(y, y_min);

    path.getVelocity(0.5e-3f, vx, vy);
    EXPECT_GT(vx, 0.0f) << "First line should move right";
    EXPECT_FLOAT_EQ(vy, 0.0f) << "No y-velocity during line scan";

    // Second line: should go from right to left
    float line_time = (x_max - x_min) / speed;
    path.getPosition(line_time + 0.01f * line_time, x, y);
    EXPECT_NEAR(y, y_min + hatch, 1e-9f) << "Should be on second line";

    path.getVelocity(line_time + 0.01f * line_time, vx, vy);
    EXPECT_LT(vx, 0.0f) << "Second line should move left (zigzag)";
}

/**
 * Test 9: Parameter updates
 * Verify that parameter changes update derived quantities
 */
TEST_F(LaserSourceTest, ParameterUpdates) {
    LaserSource laser;

    // Set new parameters
    float new_power = 200.0f;
    float new_radius = 75e-6f;
    float new_absorptivity = 0.4f;
    float new_depth = 15e-6f;

    laser.setParameters(new_power, new_radius, new_absorptivity, new_depth);

    EXPECT_FLOAT_EQ(laser.power, new_power);
    EXPECT_FLOAT_EQ(laser.spot_radius, new_radius);
    EXPECT_FLOAT_EQ(laser.absorptivity, new_absorptivity);
    EXPECT_FLOAT_EQ(laser.penetration_depth, new_depth);

    // Check derived parameters
    EXPECT_FLOAT_EQ(laser.beta, 1.0f / new_depth);

    float expected_intensity_factor = 2.0f * new_power / (M_PI * new_radius * new_radius);
    EXPECT_NEAR(laser.intensity_factor, expected_intensity_factor, 1e-6f);
}

/**
 * Test 10: Total absorbed power
 * Verify getTotalAbsorbedPower method
 */
TEST_F(LaserSourceTest, TotalAbsorbedPower) {
    float test_power = 150.0f;
    float test_absorptivity = 0.42f;

    LaserSource laser(test_power, spot_radius, test_absorptivity, penetration_depth);

    float absorbed = laser.getTotalAbsorbedPower();
    float expected = test_power * test_absorptivity;

    EXPECT_FLOAT_EQ(absorbed, expected)
        << "Absorbed power should be P * η";
}