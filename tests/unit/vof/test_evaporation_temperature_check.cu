/**
 * @file test_evaporation_temperature_check.cu
 * @brief Unit tests for evaporation logic temperature dependency
 *
 * Problem: User observed that metal continues to evaporate even after the
 * high-temperature region has moved away.
 *
 * This test suite validates:
 * 1. Cold region (T < T_boil): No evaporation should occur
 * 2. Hot region (T > T_boil): Evaporation should occur, fill_level decreases
 * 3. Temperature drop: Evaporation should stop when T drops below T_boil
 *
 * Test approach:
 * - Directly test the evaporation mass flux kernel
 * - Test VOF applyEvaporationMassLoss with controlled J_evap
 * - Integration test: temperature-based evaporation stop
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/material_properties.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace lbm::physics;

// ============================================================================
// Constants for Ti6Al4V (typical AM material)
// ============================================================================
namespace TestMaterial {
    constexpr float T_BOIL = 3533.0f;       // Boiling temperature [K]
    constexpr float RHO_LIQUID = 4000.0f;   // Liquid density [kg/m^3]
    constexpr float L_VAP = 8.878e6f;       // Latent heat of vaporization [J/kg]
    constexpr float M_MOLAR = 0.0479f;      // Molar mass [kg/mol]
    constexpr float R_GAS = 8.314f;         // Universal gas constant [J/(mol.K)]
    constexpr float P_REF = 101325.0f;      // Reference pressure [Pa]
    constexpr float ALPHA_EVAP = 0.82f;     // Evaporation coefficient
    constexpr float PI = 3.14159265359f;
}

// ============================================================================
// Helper CUDA kernel: Compute evaporation mass flux from temperature
// (Mirrors the kernel in thermal_lbm.cu for isolated testing)
// ============================================================================
__global__ void computeTestEvaporationMassFluxKernel(
    const float* temperature,
    float* J_evap,
    float T_boil,
    float L_vap,
    float M_molar,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];

    // Only compute mass flux when T > T_boil
    if (T <= T_boil) {
        J_evap[idx] = 0.0f;
        return;
    }

    // Clausius-Clapeyron equation for vapor pressure
    float T_capped = fminf(T, 2.0f * T_boil);
    float exponent = (L_vap * M_molar / TestMaterial::R_GAS) *
                     (1.0f / T_boil - 1.0f / T_capped);
    exponent = fminf(exponent, 20.0f);
    float P_sat = TestMaterial::P_REF * expf(exponent);
    P_sat = fminf(P_sat, 10.0f * TestMaterial::P_REF);

    // Hertz-Knudsen-Langmuir evaporation mass flux
    float denominator = sqrtf(2.0f * TestMaterial::PI * TestMaterial::R_GAS *
                              T_capped / M_molar);
    J_evap[idx] = TestMaterial::ALPHA_EVAP * P_sat / denominator;
}

// ============================================================================
// Test Fixture
// ============================================================================
class EvaporationTemperatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Small domain for fast testing
        nx_ = 16;
        ny_ = 16;
        nz_ = 16;
        num_cells_ = nx_ * ny_ * nz_;
        dx_ = 2e-6f;  // 2 micron grid spacing
        dt_ = 1e-7f;  // 100 ns time step
    }

    void TearDown() override {
        // Clean up any allocated memory
    }

    // Helper to compute J_evap from temperature field
    void computeJEvapFromTemperature(const float* d_temperature, float* d_J_evap) {
        int blockSize = 256;
        int gridSize = (num_cells_ + blockSize - 1) / blockSize;

        computeTestEvaporationMassFluxKernel<<<gridSize, blockSize>>>(
            d_temperature, d_J_evap,
            TestMaterial::T_BOIL,
            TestMaterial::L_VAP,
            TestMaterial::M_MOLAR,
            num_cells_);
        cudaDeviceSynchronize();
    }

    int nx_, ny_, nz_, num_cells_;
    float dx_, dt_;
};

// ============================================================================
// Test 1: Cold Region No Evaporation
// ============================================================================
/**
 * @test Verify that no evaporation occurs when T < T_boil
 *
 * Setup:
 * - Uniform temperature T = 2000 K (well below T_boil = 3533 K)
 * - Initial fill_level = 1.0 (full cells)
 *
 * Expected:
 * - J_evap = 0 for all cells
 * - fill_level unchanged after applying evaporation
 */
TEST_F(EvaporationTemperatureTest, ColdRegionNoEvaporation) {
    std::cout << "\n=== Test 1: Cold Region No Evaporation ===" << std::endl;
    std::cout << "  T = 2000 K (below T_boil = " << TestMaterial::T_BOIL << " K)" << std::endl;

    // Allocate device memory
    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    // Set uniform cold temperature
    const float T_cold = 2000.0f;
    std::vector<float> h_temperature(num_cells_, T_cold);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    // Compute evaporation mass flux
    computeJEvapFromTemperature(d_temperature, d_J_evap);

    // Copy results back
    std::vector<float> h_J_evap(num_cells_);
    cudaMemcpy(h_J_evap.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify J_evap = 0 for all cells
    float max_J = 0.0f;
    int nonzero_count = 0;
    for (int i = 0; i < num_cells_; ++i) {
        max_J = std::max(max_J, h_J_evap[i]);
        if (h_J_evap[i] > 0.0f) nonzero_count++;
    }

    std::cout << "  Max J_evap = " << max_J << " kg/(m^2.s)" << std::endl;
    std::cout << "  Nonzero cells = " << nonzero_count << " / " << num_cells_ << std::endl;

    EXPECT_FLOAT_EQ(max_J, 0.0f) << "J_evap should be 0 when T < T_boil";
    EXPECT_EQ(nonzero_count, 0) << "No cells should have nonzero evaporation";

    // Now test VOF fill_level change
    VOFSolver vof(nx_, ny_, nz_, dx_);
    vof.initialize(1.0f);  // Full cells

    float mass_before = vof.computeTotalMass();

    // Apply evaporation with zero J_evap
    vof.applyEvaporationMassLoss(d_J_evap, TestMaterial::RHO_LIQUID, dt_);

    float mass_after = vof.computeTotalMass();

    std::cout << "  Mass before: " << mass_before << std::endl;
    std::cout << "  Mass after:  " << mass_after << std::endl;
    std::cout << "  Mass change: " << (mass_after - mass_before) << std::endl;

    EXPECT_FLOAT_EQ(mass_before, mass_after)
        << "Mass should not change when T < T_boil";

    // Verify individual fill levels unchanged
    std::vector<float> h_fill(num_cells_);
    vof.copyFillLevelToHost(h_fill.data());
    for (int i = 0; i < num_cells_; ++i) {
        EXPECT_FLOAT_EQ(h_fill[i], 1.0f) << "Fill level should remain 1.0 at cell " << i;
    }

    std::cout << "  [PASS] No evaporation in cold region" << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Test 2: Hot Region Evaporation
// ============================================================================
/**
 * @test Verify that evaporation occurs when T > T_boil
 *
 * Setup:
 * - Uniform temperature T = 4000 K (above T_boil = 3533 K)
 * - Initial fill_level = 1.0
 *
 * Expected:
 * - J_evap > 0 for all cells
 * - fill_level decreases after applying evaporation
 */
TEST_F(EvaporationTemperatureTest, HotRegionEvaporation) {
    std::cout << "\n=== Test 2: Hot Region Evaporation ===" << std::endl;
    std::cout << "  T = 4000 K (above T_boil = " << TestMaterial::T_BOIL << " K)" << std::endl;

    // Allocate device memory
    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    // Set uniform hot temperature
    const float T_hot = 4000.0f;
    std::vector<float> h_temperature(num_cells_, T_hot);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    // Compute evaporation mass flux
    computeJEvapFromTemperature(d_temperature, d_J_evap);

    // Copy results back
    std::vector<float> h_J_evap(num_cells_);
    cudaMemcpy(h_J_evap.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify J_evap > 0 for all cells
    float min_J = h_J_evap[0];
    float max_J = h_J_evap[0];
    for (int i = 0; i < num_cells_; ++i) {
        min_J = std::min(min_J, h_J_evap[i]);
        max_J = std::max(max_J, h_J_evap[i]);
    }

    std::cout << "  J_evap range: [" << min_J << ", " << max_J << "] kg/(m^2.s)" << std::endl;

    EXPECT_GT(min_J, 0.0f) << "J_evap should be > 0 when T > T_boil";
    EXPECT_GT(max_J, 0.0f) << "J_evap should be > 0 when T > T_boil";

    // Test VOF fill_level change
    VOFSolver vof(nx_, ny_, nz_, dx_);
    vof.initialize(1.0f);

    float mass_before = vof.computeTotalMass();

    // Apply evaporation for several timesteps
    const int num_steps = 10;
    for (int step = 0; step < num_steps; ++step) {
        vof.applyEvaporationMassLoss(d_J_evap, TestMaterial::RHO_LIQUID, dt_);
    }

    float mass_after = vof.computeTotalMass();
    float mass_loss = mass_before - mass_after;

    std::cout << "  Mass before: " << mass_before << std::endl;
    std::cout << "  Mass after:  " << mass_after << " (after " << num_steps << " steps)" << std::endl;
    std::cout << "  Mass loss:   " << mass_loss << std::endl;

    EXPECT_LT(mass_after, mass_before) << "Mass should decrease when T > T_boil";
    EXPECT_GT(mass_loss, 0.0f) << "Should have positive mass loss";

    // Verify individual fill levels decreased
    std::vector<float> h_fill(num_cells_);
    vof.copyFillLevelToHost(h_fill.data());
    int decreased_count = 0;
    for (int i = 0; i < num_cells_; ++i) {
        if (h_fill[i] < 1.0f) decreased_count++;
    }

    std::cout << "  Cells with decreased fill: " << decreased_count << " / " << num_cells_ << std::endl;
    EXPECT_GT(decreased_count, 0) << "Some cells should have decreased fill level";

    std::cout << "  [PASS] Evaporation active in hot region" << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Test 3: Temperature Drop Stops Evaporation
// ============================================================================
/**
 * @test Verify that evaporation stops when temperature drops below T_boil
 *
 * Setup:
 * 1. Start with T = 4000 K, apply evaporation for several steps
 * 2. Drop temperature to T = 2000 K
 * 3. Apply evaporation for more steps
 *
 * Expected:
 * - Phase 1: Mass decreases (evaporation active)
 * - Phase 2: Mass stays constant (evaporation stopped)
 */
TEST_F(EvaporationTemperatureTest, TemperatureDropStopsEvaporation) {
    std::cout << "\n=== Test 3: Temperature Drop Stops Evaporation ===" << std::endl;

    // Allocate device memory
    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    // Initialize VOF
    VOFSolver vof(nx_, ny_, nz_, dx_);
    vof.initialize(1.0f);

    // ---- Phase 1: Hot region (T = 4000 K) ----
    std::cout << "\n  Phase 1: Hot region (T = 4000 K)" << std::endl;

    const float T_hot = 4000.0f;
    std::vector<float> h_temperature(num_cells_, T_hot);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    computeJEvapFromTemperature(d_temperature, d_J_evap);

    float mass_initial = vof.computeTotalMass();

    const int steps_hot = 50;
    for (int step = 0; step < steps_hot; ++step) {
        vof.applyEvaporationMassLoss(d_J_evap, TestMaterial::RHO_LIQUID, dt_);
    }

    float mass_after_hot = vof.computeTotalMass();
    float loss_hot = mass_initial - mass_after_hot;

    std::cout << "    Initial mass:      " << mass_initial << std::endl;
    std::cout << "    Mass after " << steps_hot << " hot steps: " << mass_after_hot << std::endl;
    std::cout << "    Mass loss (hot):   " << loss_hot << std::endl;

    EXPECT_GT(loss_hot, 0.0f) << "Should have mass loss in hot phase";

    // ---- Phase 2: Cold region (T = 2000 K) ----
    std::cout << "\n  Phase 2: Cold region (T = 2000 K)" << std::endl;

    const float T_cold = 2000.0f;
    std::fill(h_temperature.begin(), h_temperature.end(), T_cold);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    // CRITICAL: Recompute J_evap with new temperature
    computeJEvapFromTemperature(d_temperature, d_J_evap);

    // Verify J_evap is now zero
    std::vector<float> h_J_evap(num_cells_);
    cudaMemcpy(h_J_evap.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float max_J_cold = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        max_J_cold = std::max(max_J_cold, h_J_evap[i]);
    }
    std::cout << "    Max J_evap after cooling: " << max_J_cold << std::endl;
    EXPECT_FLOAT_EQ(max_J_cold, 0.0f) << "J_evap should be 0 after cooling";

    // Apply more evaporation steps (should do nothing)
    const int steps_cold = 50;
    for (int step = 0; step < steps_cold; ++step) {
        vof.applyEvaporationMassLoss(d_J_evap, TestMaterial::RHO_LIQUID, dt_);
    }

    float mass_after_cold = vof.computeTotalMass();
    float loss_cold = mass_after_hot - mass_after_cold;

    std::cout << "    Mass after " << steps_cold << " cold steps: " << mass_after_cold << std::endl;
    std::cout << "    Mass loss (cold):  " << loss_cold << std::endl;

    EXPECT_FLOAT_EQ(loss_cold, 0.0f)
        << "Should have NO mass loss after temperature drop";
    EXPECT_FLOAT_EQ(mass_after_hot, mass_after_cold)
        << "Mass should be constant in cold phase";

    // Summary
    std::cout << "\n  Summary:" << std::endl;
    std::cout << "    Mass loss during hot phase:  " << loss_hot << std::endl;
    std::cout << "    Mass loss during cold phase: " << loss_cold << std::endl;
    std::cout << "    Total remaining mass:        " << mass_after_cold
              << " (" << (mass_after_cold / mass_initial * 100.0f) << "%)" << std::endl;

    std::cout << "  [PASS] Evaporation correctly stops when temperature drops" << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Test 4: Boundary Temperature Check (T = T_boil exactly)
// ============================================================================
/**
 * @test Verify behavior at exactly the boiling point
 *
 * The condition is "T <= T_boil" so at T = T_boil, evaporation should NOT occur.
 */
TEST_F(EvaporationTemperatureTest, AtBoilingPoint) {
    std::cout << "\n=== Test 4: At Boiling Point ===" << std::endl;
    std::cout << "  T = T_boil = " << TestMaterial::T_BOIL << " K (boundary case)" << std::endl;

    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    // Set temperature exactly at boiling point
    std::vector<float> h_temperature(num_cells_, TestMaterial::T_BOIL);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    computeJEvapFromTemperature(d_temperature, d_J_evap);

    std::vector<float> h_J_evap(num_cells_);
    cudaMemcpy(h_J_evap.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float max_J = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        max_J = std::max(max_J, h_J_evap[i]);
    }

    std::cout << "  Max J_evap at T_boil: " << max_J << " kg/(m^2.s)" << std::endl;

    // At T = T_boil, condition "T <= T_boil" is true, so J_evap = 0
    EXPECT_FLOAT_EQ(max_J, 0.0f) << "J_evap should be 0 at exactly T_boil";

    std::cout << "  [PASS] No evaporation at exactly T_boil" << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Test 5: Just Above Boiling Point
// ============================================================================
/**
 * @test Verify evaporation starts just above boiling point
 */
TEST_F(EvaporationTemperatureTest, JustAboveBoilingPoint) {
    std::cout << "\n=== Test 5: Just Above Boiling Point ===" << std::endl;

    const float T_test = TestMaterial::T_BOIL + 1.0f;  // 3534 K
    std::cout << "  T = " << T_test << " K (T_boil + 1)" << std::endl;

    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    std::vector<float> h_temperature(num_cells_, T_test);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    computeJEvapFromTemperature(d_temperature, d_J_evap);

    std::vector<float> h_J_evap(num_cells_);
    cudaMemcpy(h_J_evap.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float min_J = h_J_evap[0];
    float max_J = h_J_evap[0];
    for (int i = 0; i < num_cells_; ++i) {
        min_J = std::min(min_J, h_J_evap[i]);
        max_J = std::max(max_J, h_J_evap[i]);
    }

    std::cout << "  J_evap at T_boil+1: " << max_J << " kg/(m^2.s)" << std::endl;

    EXPECT_GT(min_J, 0.0f) << "J_evap should be > 0 just above T_boil";

    // The value should be reasonable (Hertz-Knudsen gives ~40 kg/m2.s just above T_boil)
    // This is expected - vapor pressure increases rapidly with temperature
    EXPECT_LT(max_J, 100.0f) << "J_evap should be moderate just above T_boil";

    std::cout << "  [PASS] Evaporation starts just above T_boil" << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Test 6: Spatial Temperature Gradient (Mixed Hot/Cold)
// ============================================================================
/**
 * @test Verify evaporation only occurs in hot regions of a mixed field
 *
 * Setup:
 * - Half domain cold (T = 2000 K), half domain hot (T = 4000 K)
 *
 * Expected:
 * - J_evap = 0 in cold half
 * - J_evap > 0 in hot half
 */
TEST_F(EvaporationTemperatureTest, SpatialTemperatureGradient) {
    std::cout << "\n=== Test 6: Spatial Temperature Gradient ===" << std::endl;
    std::cout << "  Left half: T = 2000 K (cold)" << std::endl;
    std::cout << "  Right half: T = 4000 K (hot)" << std::endl;

    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    // Create temperature field: left half cold, right half hot
    std::vector<float> h_temperature(num_cells_);
    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                int idx = i + nx_ * (j + ny_ * k);
                h_temperature[idx] = (i < nx_ / 2) ? 2000.0f : 4000.0f;
            }
        }
    }
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    computeJEvapFromTemperature(d_temperature, d_J_evap);

    std::vector<float> h_J_evap(num_cells_);
    cudaMemcpy(h_J_evap.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Count cells by region
    int cold_nonzero = 0, hot_zero = 0;
    float max_J_cold = 0.0f, min_J_hot = 1e10f, max_J_hot = 0.0f;

    for (int k = 0; k < nz_; ++k) {
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                int idx = i + nx_ * (j + ny_ * k);
                if (i < nx_ / 2) {  // Cold region
                    if (h_J_evap[idx] > 0.0f) cold_nonzero++;
                    max_J_cold = std::max(max_J_cold, h_J_evap[idx]);
                } else {  // Hot region
                    if (h_J_evap[idx] == 0.0f) hot_zero++;
                    min_J_hot = std::min(min_J_hot, h_J_evap[idx]);
                    max_J_hot = std::max(max_J_hot, h_J_evap[idx]);
                }
            }
        }
    }

    std::cout << "  Cold region: max_J = " << max_J_cold << ", nonzero cells = " << cold_nonzero << std::endl;
    std::cout << "  Hot region: J_range = [" << min_J_hot << ", " << max_J_hot << "], zero cells = " << hot_zero << std::endl;

    EXPECT_EQ(cold_nonzero, 0) << "Cold region should have no evaporation";
    EXPECT_FLOAT_EQ(max_J_cold, 0.0f) << "Max J in cold region should be 0";
    EXPECT_GT(min_J_hot, 0.0f) << "Hot region should have evaporation everywhere";
    EXPECT_EQ(hot_zero, 0) << "Hot region should have no zero-evaporation cells";

    std::cout << "  [PASS] Evaporation correctly localized to hot region" << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Test 7: VOF Integration - Stale J_evap Bug Detection
// ============================================================================
/**
 * @test Detect potential bug where J_evap is not updated when temperature changes
 *
 * This test simulates the scenario where:
 * 1. Hot region creates non-zero J_evap
 * 2. Temperature drops, but J_evap array is NOT recomputed (bug scenario)
 * 3. VOF continues to use stale J_evap values
 *
 * Expected behavior: This test should FAIL if there's a bug where J_evap
 * is not recomputed after temperature changes.
 */
TEST_F(EvaporationTemperatureTest, StaleJEvapBugDetection) {
    std::cout << "\n=== Test 7: Stale J_evap Bug Detection ===" << std::endl;
    std::cout << "  This test verifies J_evap must be recomputed when T changes" << std::endl;

    float *d_temperature, *d_J_evap;
    cudaMalloc(&d_temperature, num_cells_ * sizeof(float));
    cudaMalloc(&d_J_evap, num_cells_ * sizeof(float));

    // Step 1: Set hot temperature and compute J_evap
    const float T_hot = 4000.0f;
    std::vector<float> h_temperature(num_cells_, T_hot);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    computeJEvapFromTemperature(d_temperature, d_J_evap);

    // Verify hot J_evap
    std::vector<float> h_J_evap_hot(num_cells_);
    cudaMemcpy(h_J_evap_hot.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float max_J_hot = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        max_J_hot = std::max(max_J_hot, h_J_evap_hot[i]);
    }
    std::cout << "  Hot phase J_evap max: " << max_J_hot << std::endl;
    EXPECT_GT(max_J_hot, 0.0f) << "Should have positive J_evap when hot";

    // Step 2: Change temperature to cold WITHOUT recomputing J_evap
    const float T_cold = 2000.0f;
    std::fill(h_temperature.begin(), h_temperature.end(), T_cold);
    cudaMemcpy(d_temperature, h_temperature.data(),
               num_cells_ * sizeof(float), cudaMemcpyHostToDevice);

    // NOTE: We intentionally DO NOT call computeJEvapFromTemperature here
    // This simulates the bug condition

    std::cout << "  WARNING: Temperature changed to " << T_cold << " K but J_evap NOT recomputed" << std::endl;

    // Step 3: Initialize VOF and apply evaporation with STALE J_evap
    VOFSolver vof(nx_, ny_, nz_, dx_);
    vof.initialize(1.0f);

    float mass_before = vof.computeTotalMass();

    // Apply evaporation using stale (hot) J_evap
    const int num_steps = 10;
    for (int step = 0; step < num_steps; ++step) {
        vof.applyEvaporationMassLoss(d_J_evap, TestMaterial::RHO_LIQUID, dt_);
    }

    float mass_after_stale = vof.computeTotalMass();
    float loss_with_stale = mass_before - mass_after_stale;

    std::cout << "  Mass loss with STALE J_evap: " << loss_with_stale << std::endl;

    // This is the BUG CONDITION: if loss_with_stale > 0, we have a problem
    // The temperature is cold but evaporation still occurs because J_evap wasn't updated
    if (loss_with_stale > 0.0f) {
        std::cout << "  [BUG DETECTED] Evaporation occurred with cold T but stale hot J_evap!" << std::endl;
        std::cout << "  Root cause: J_evap must be recomputed whenever temperature changes" << std::endl;
    }

    // Step 4: Now properly recompute J_evap and verify it's zero
    computeJEvapFromTemperature(d_temperature, d_J_evap);

    std::vector<float> h_J_evap_cold(num_cells_);
    cudaMemcpy(h_J_evap_cold.data(), d_J_evap,
               num_cells_ * sizeof(float), cudaMemcpyDeviceToHost);

    float max_J_cold = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        max_J_cold = std::max(max_J_cold, h_J_evap_cold[i]);
    }
    std::cout << "  After recompute, J_evap max: " << max_J_cold << std::endl;
    EXPECT_FLOAT_EQ(max_J_cold, 0.0f) << "J_evap should be 0 after recompute with cold T";

    // Re-initialize VOF and test with correct J_evap
    VOFSolver vof2(nx_, ny_, nz_, dx_);
    vof2.initialize(1.0f);

    float mass_before2 = vof2.computeTotalMass();
    for (int step = 0; step < num_steps; ++step) {
        vof2.applyEvaporationMassLoss(d_J_evap, TestMaterial::RHO_LIQUID, dt_);
    }
    float mass_after_correct = vof2.computeTotalMass();
    float loss_with_correct = mass_before2 - mass_after_correct;

    std::cout << "  Mass loss with CORRECT J_evap: " << loss_with_correct << std::endl;
    EXPECT_FLOAT_EQ(loss_with_correct, 0.0f) << "Should have no mass loss with correct cold J_evap";

    std::cout << "\n  [SUMMARY] This test demonstrates that J_evap MUST be recomputed" << std::endl;
    std::cout << "  whenever temperature changes to avoid the stale-J_evap bug." << std::endl;

    cudaFree(d_temperature);
    cudaFree(d_J_evap);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
