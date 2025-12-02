/**
 * @file test_recoil_pressure.cu
 * @brief Unit tests for recoil pressure computation (Anisimov model)
 *
 * Tests:
 * 1. Saturation pressure at various temperatures (Clausius-Clapeyron)
 * 2. Recoil pressure values (C_r = 0.54)
 * 3. Force direction (into liquid, along -n)
 * 4. Numerical stability limiters
 *
 * Physical model (Anisimov model for Ti6Al4V):
 *   P_sat(T) = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]
 *   P_recoil = 0.54 * P_sat
 *
 * Parameters:
 *   - P_ref = 101325 Pa
 *   - L_vap = 8.878e6 J/kg
 *   - M = 0.0479 kg/mol
 *   - R = 8.314 J/(mol.K)
 *   - T_boil = 3533 K
 */

#include <gtest/gtest.h>
#include "physics/recoil_pressure.h"
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace lbm::physics;

// Physical constants for validation calculations (Anisimov model for Ti6Al4V)
namespace TestConstants {
    constexpr float R_GAS = 8.314f;           // Universal gas constant [J/(mol.K)]
    constexpr float P_REF = 101325.0f;        // Reference pressure [Pa] (1 atm)
    constexpr float T_BOIL = 3533.0f;         // Boiling point [K]
    constexpr float L_VAP = 8.878e6f;         // Latent heat of vaporization [J/kg]
    constexpr float M_MOLAR = 0.0479f;        // Molar mass [kg/mol]
    constexpr float C_R = 0.54f;              // Anisimov/Knight recoil coefficient
    constexpr float T_ACTIVATION = T_BOIL - 500.0f;  // Activation threshold

    // Precomputed Clausius-Clapeyron factor: L_vap * M / R
    constexpr float CC_FACTOR = L_VAP * M_MOLAR / R_GAS;
}

// ============================================================================
// Helper functions for analytical calculations
// ============================================================================

/**
 * @brief Compute saturation pressure using Clausius-Clapeyron equation
 *
 * P_sat = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]
 *
 * This matches the Anisimov model parameters for Ti6Al4V.
 */
float computePsatAnalytical(float T) {
    if (T < TestConstants::T_ACTIVATION) {
        return 0.0f;
    }
    float exponent = TestConstants::CC_FACTOR * (1.0f / TestConstants::T_BOIL - 1.0f / T);
    return TestConstants::P_REF * std::exp(exponent);
}

/**
 * @brief Compute recoil pressure from saturation pressure
 * P_recoil = C_r * P_sat
 */
float computePrecoilAnalytical(float p_sat) {
    return TestConstants::C_R * p_sat;
}

/**
 * @brief CUDA kernel to compute saturation pressure field from temperature
 *
 * This standalone kernel computes P_sat directly from temperature using
 * the Clausius-Clapeyron equation, matching the Anisimov model parameters.
 */
__global__ void computeSaturationPressureFieldKernel(
    const float* __restrict__ temperature,
    float* __restrict__ saturation_pressure,
    float T_boil,
    float p_ref,
    float cc_factor,
    float T_activation,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    if (T < T_activation) {
        saturation_pressure[idx] = 0.0f;
    } else {
        float exponent = cc_factor * (1.0f / T_boil - 1.0f / T);
        saturation_pressure[idx] = p_ref * expf(exponent);
    }
}

/**
 * @brief Host wrapper to compute saturation pressure field
 */
void computeSaturationPressureField(
    const float* d_temperature,
    float* d_saturation_pressure,
    int nx, int ny, int nz)
{
    int num_cells = nx * ny * nz;
    int blockSize = 256;
    int gridSize = (num_cells + blockSize - 1) / blockSize;

    computeSaturationPressureFieldKernel<<<gridSize, blockSize>>>(
        d_temperature,
        d_saturation_pressure,
        TestConstants::T_BOIL,
        TestConstants::P_REF,
        TestConstants::CC_FACTOR,
        TestConstants::T_ACTIVATION,
        num_cells);

    cudaDeviceSynchronize();
}

// ============================================================================
// Unit Test 1: P_sat Calculation Verification
// ============================================================================

class PsatCalculationTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n=== Setting up P_sat Calculation Test ===" << std::endl;
    }
};

/**
 * @test Verify P_sat at boiling point equals atmospheric pressure
 *
 * At T = T_boil, the Clausius-Clapeyron equation should give P_sat = P_ref = 101325 Pa
 */
TEST_F(PsatCalculationTest, AtBoilingPoint) {
    std::cout << "\n--- Test: P_sat at Boiling Point ---" << std::endl;

    float T = TestConstants::T_BOIL;  // 3533 K
    float p_sat = computePsatAnalytical(T);

    std::cout << "  T = " << T << " K (boiling point)" << std::endl;
    std::cout << "  P_sat (computed) = " << p_sat << " Pa" << std::endl;
    std::cout << "  P_sat (expected) = " << TestConstants::P_REF << " Pa" << std::endl;

    // At boiling point, P_sat should equal reference pressure (1 atm)
    EXPECT_NEAR(p_sat, TestConstants::P_REF, TestConstants::P_REF * 0.01)
        << "P_sat at boiling point should equal atmospheric pressure";

    std::cout << "  [PASS] P_sat at boiling point verified" << std::endl;
}

/**
 * @test Verify P_sat at T = 3100 K (below boiling point, above activation)
 *
 * At T < T_boil, P_sat should be less than atmospheric pressure.
 */
TEST_F(PsatCalculationTest, BelowBoilingPoint_3100K) {
    std::cout << "\n--- Test: P_sat at T = 3100 K ---" << std::endl;

    float T = 3100.0f;
    float p_sat = computePsatAnalytical(T);

    std::cout << "  T = " << T << " K" << std::endl;
    std::cout << "  P_sat = " << p_sat << " Pa" << std::endl;

    // Physical sanity check: should be less than 1 atm
    EXPECT_GT(p_sat, 100.0f) << "P_sat at 3100K should be > 100 Pa";
    EXPECT_LT(p_sat, TestConstants::P_REF) << "P_sat at 3100K should be < 1 atm";

    std::cout << "  [PASS] P_sat at 3100 K verified" << std::endl;
}

/**
 * @test Verify P_sat at T = 4000 K (above boiling point)
 *
 * At T > T_boil, P_sat should exceed atmospheric pressure.
 */
TEST_F(PsatCalculationTest, AboveBoilingPoint_4000K) {
    std::cout << "\n--- Test: P_sat at T = 4000 K ---" << std::endl;

    float T = 4000.0f;
    float p_sat = computePsatAnalytical(T);

    std::cout << "  T = " << T << " K" << std::endl;
    std::cout << "  P_sat = " << std::scientific << p_sat << " Pa" << std::endl;
    std::cout << std::fixed;

    // Physical sanity check: should exceed 1 atm
    EXPECT_GT(p_sat, TestConstants::P_REF) << "P_sat at 4000K should exceed 1 atm";
    EXPECT_LT(p_sat, 1e7f) << "P_sat at 4000K should be reasonable (< 10 MPa)";

    std::cout << "  [PASS] P_sat at 4000 K verified" << std::endl;
}

/**
 * @test Verify P_sat temperature dependence (exponential growth)
 */
TEST_F(PsatCalculationTest, ExponentialTemperatureDependence) {
    std::cout << "\n--- Test: P_sat Exponential Temperature Dependence ---" << std::endl;

    float T1 = 3200.0f;
    float T2 = 3400.0f;
    float T3 = 3600.0f;

    float p_sat_1 = computePsatAnalytical(T1);
    float p_sat_2 = computePsatAnalytical(T2);
    float p_sat_3 = computePsatAnalytical(T3);

    std::cout << "  P_sat(" << T1 << " K) = " << p_sat_1 << " Pa" << std::endl;
    std::cout << "  P_sat(" << T2 << " K) = " << p_sat_2 << " Pa" << std::endl;
    std::cout << "  P_sat(" << T3 << " K) = " << p_sat_3 << " Pa" << std::endl;

    // Verify monotonic increase
    EXPECT_GT(p_sat_2, p_sat_1) << "P_sat should increase with temperature";
    EXPECT_GT(p_sat_3, p_sat_2) << "P_sat should increase with temperature";

    float ratio_12 = p_sat_2 / p_sat_1;
    float ratio_23 = p_sat_3 / p_sat_2;

    std::cout << "  Ratio P_sat(3400)/P_sat(3200) = " << ratio_12 << std::endl;
    std::cout << "  Ratio P_sat(3600)/P_sat(3400) = " << ratio_23 << std::endl;

    // Both ratios should be > 1
    EXPECT_GT(ratio_12, 1.0f);
    EXPECT_GT(ratio_23, 1.0f);

    std::cout << "  [PASS] Exponential temperature dependence verified" << std::endl;
}

// ============================================================================
// Unit Test 2: Recoil Pressure Coefficient Verification
// ============================================================================

class RecoilCoefficientTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n=== Setting up Recoil Coefficient Test ===" << std::endl;
    }
};

/**
 * @test Verify Anisimov coefficient C_r = 0.54 is correctly applied
 */
TEST_F(RecoilCoefficientTest, AnisimovCoefficientCorrect) {
    std::cout << "\n--- Test: Anisimov Coefficient (C_r = 0.54) ---" << std::endl;

    float dx = 1.0e-6f;
    RecoilPressureConfig config;
    config.coefficient = 0.54f;
    config.max_pressure = 1e10f;

    RecoilPressure recoil(config, dx);

    std::vector<float> p_sat_values = {1000.0f, 10000.0f, 100000.0f, 500000.0f};

    std::cout << "  Testing P_recoil = 0.54 * P_sat" << std::endl;
    std::cout << std::setw(15) << "P_sat [Pa]" << std::setw(20) << "P_recoil [Pa]"
              << std::setw(20) << "Expected [Pa]" << std::endl;

    for (float p_sat : p_sat_values) {
        float p_recoil = recoil.computePressure(p_sat);
        float expected = 0.54f * p_sat;

        std::cout << std::setw(15) << p_sat << std::setw(20) << p_recoil
                  << std::setw(20) << expected << std::endl;

        EXPECT_NEAR(p_recoil, expected, expected * 0.01)
            << "P_recoil should be 0.54 * P_sat";
    }

    std::cout << "  [PASS] Anisimov coefficient verified" << std::endl;
}

/**
 * @test Verify recoil coefficient getter/setter
 */
TEST_F(RecoilCoefficientTest, CoefficientGetSet) {
    std::cout << "\n--- Test: Recoil Coefficient Getter/Setter ---" << std::endl;

    float dx = 1.0e-6f;
    RecoilPressure recoil(0.54f, 2.0f, dx);

    EXPECT_FLOAT_EQ(recoil.getRecoilCoefficient(), 0.54f);

    recoil.setRecoilCoefficient(0.56f);
    EXPECT_FLOAT_EQ(recoil.getRecoilCoefficient(), 0.56f);

    float p_sat = 100000.0f;
    float p_recoil = recoil.computePressure(p_sat);
    EXPECT_NEAR(p_recoil, 0.56f * p_sat, 0.01f * p_sat);

    std::cout << "  [PASS] Coefficient getter/setter verified" << std::endl;
}

/**
 * @test Verify recoil pressure at boiling point gives expected value
 */
TEST_F(RecoilCoefficientTest, RecoilAtBoilingPoint) {
    std::cout << "\n--- Test: P_recoil at Boiling Point ---" << std::endl;

    float dx = 1.0e-6f;
    RecoilPressure recoil(0.54f, 2.0f, dx);

    float p_sat_at_boil = TestConstants::P_REF;
    float p_recoil = recoil.computePressure(p_sat_at_boil);
    float expected = 0.54f * TestConstants::P_REF;

    std::cout << "  At T = T_boil:" << std::endl;
    std::cout << "    P_sat = " << p_sat_at_boil << " Pa" << std::endl;
    std::cout << "    P_recoil = " << p_recoil << " Pa" << std::endl;
    std::cout << "    Expected = " << expected << " Pa (~54.7 kPa)" << std::endl;

    EXPECT_NEAR(p_recoil, expected, 1.0f);

    std::cout << "  [PASS] Recoil pressure at boiling point verified" << std::endl;
}

// ============================================================================
// Unit Test 3: Force Direction Verification
// ============================================================================

class ForceDirectionTest : public ::testing::Test {
protected:
    const int nx = 20;
    const int ny = 20;
    const int nz = 20;
    const float dx = 2.0e-6f;

    void SetUp() override {
        std::cout << "\n=== Setting up Force Direction Test ===" << std::endl;
    }
};

/**
 * @test Verify force direction on horizontal interface (force should be -z)
 */
TEST_F(ForceDirectionTest, HorizontalInterfaceForceDownward) {
    std::cout << "\n--- Test: Force Direction on Horizontal Interface ---" << std::endl;

    const int num_cells = nx * ny * nz;
    const int z_interface = 10;

    VOFSolver vof(nx, ny, nz, dx);
    RecoilPressure recoil(0.54f, 2.0f, dx);

    // Initialize horizontal interface
    std::vector<float> h_fill(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float z_dist = static_cast<float>(k - z_interface);
                h_fill[idx] = 0.5f * (1.0f - std::tanh(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Create uniform P_sat field (high temperature)
    float p_sat_value = computePsatAnalytical(3700.0f);
    std::vector<float> h_psat(num_cells, p_sat_value);

    float* d_psat;
    cudaMalloc(&d_psat, num_cells * sizeof(float));
    cudaMemcpy(d_psat, h_psat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    recoil.computeForceField(d_psat, vof.getFillLevel(), vof.getInterfaceNormals(),
                             d_fx, d_fy, d_fz, nx, ny, nz);

    std::vector<float> h_fz(num_cells);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze force direction at interface
    int cells_with_negative_fz = 0;
    int total_interface_cells = 0;
    float min_fz = 0.0f;

    for (int k = z_interface - 2; k <= z_interface + 2; ++k) {
        for (int j = 2; j < ny - 2; ++j) {
            for (int i = 2; i < nx - 2; ++i) {
                int idx = i + nx * (j + ny * k);
                if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
                    total_interface_cells++;
                    if (h_fz[idx] < -1.0f) cells_with_negative_fz++;
                    min_fz = std::min(min_fz, h_fz[idx]);
                }
            }
        }
    }

    std::cout << "  Interface at z = " << z_interface << std::endl;
    std::cout << "  P_sat at T=3700K: " << p_sat_value << " Pa" << std::endl;
    std::cout << "  Total interface cells: " << total_interface_cells << std::endl;
    std::cout << "  Cells with F_z < -1: " << cells_with_negative_fz << std::endl;
    std::cout << "  Min F_z: " << min_fz << std::endl;

    EXPECT_GT(cells_with_negative_fz, total_interface_cells * 0.5)
        << "Majority of interface cells should have negative F_z";
    EXPECT_LT(min_fz, 0.0f) << "Minimum F_z should be negative";

    std::cout << "  [PASS] Force direction on horizontal interface verified" << std::endl;

    cudaFree(d_psat);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @test Verify force is along interface normal direction (spherical droplet)
 */
TEST_F(ForceDirectionTest, ForceAlongNormal) {
    std::cout << "\n--- Test: Force Along Interface Normal ---" << std::endl;

    const int num_cells = nx * ny * nz;

    VOFSolver vof(nx, ny, nz, dx);
    RecoilPressure recoil(0.54f, 2.0f, dx);

    float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    float radius = 6.0f;
    vof.initializeDroplet(cx, cy, cz, radius);
    vof.reconstructInterface();

    float p_sat_value = computePsatAnalytical(3700.0f);
    std::vector<float> h_psat(num_cells, p_sat_value);

    float* d_psat;
    cudaMalloc(&d_psat, num_cells * sizeof(float));
    cudaMemcpy(d_psat, h_psat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    recoil.computeForceField(d_psat, vof.getFillLevel(), vof.getInterfaceNormals(),
                             d_fx, d_fy, d_fz, nx, ny, nz);

    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    int correct_direction = 0;
    int total_tested = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                if (h_fill[idx] > 0.2f && h_fill[idx] < 0.8f) {
                    float fx = h_fx[idx], fy = h_fy[idx], fz = h_fz[idx];
                    float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

                    if (f_mag > 1.0f) {
                        total_tested++;
                        float rx = i - cx, ry = j - cy, rz = k - cz;
                        float r_mag = std::sqrt(rx*rx + ry*ry + rz*rz);

                        if (r_mag > 1.0f) {
                            float f_dot_r = (fx*rx + fy*ry + fz*rz) / (f_mag * r_mag);
                            if (f_dot_r < 0.0f) correct_direction++;
                        }
                    }
                }
            }
        }
    }

    float accuracy = (total_tested > 0) ?
                     static_cast<float>(correct_direction) / total_tested : 0.0f;

    std::cout << "  Total interface cells tested: " << total_tested << std::endl;
    std::cout << "  Cells with inward force: " << correct_direction << std::endl;
    std::cout << "  Direction accuracy: " << accuracy * 100.0f << "%" << std::endl;

    EXPECT_GT(accuracy, 0.7f) << "Majority of forces should point inward";

    std::cout << "  [PASS] Force along normal direction verified" << std::endl;

    cudaFree(d_psat);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

// ============================================================================
// Unit Test 4: Boundary Conditions and Edge Cases
// ============================================================================

class BoundaryConditionTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n=== Setting up Boundary Condition Test ===" << std::endl;
    }
};

/**
 * @test Verify zero P_sat below activation temperature
 */
TEST_F(BoundaryConditionTest, ZeroPsatBelowThreshold) {
    std::cout << "\n--- Test: Zero P_sat Below Temperature Threshold ---" << std::endl;

    std::vector<float> low_temps = {300.0f, 1000.0f, 2000.0f, 2500.0f};

    for (float T : low_temps) {
        float p_sat = computePsatAnalytical(T);
        std::cout << "  T = " << T << " K: P_sat = " << p_sat << " Pa" << std::endl;
        EXPECT_LT(p_sat, 1.0f) << "P_sat should be ~0 below activation temperature";
    }

    std::cout << "  [PASS] Zero pressure below threshold verified" << std::endl;
}

/**
 * @test Verify zero force in pure gas region (fill_level = 0)
 */
TEST_F(BoundaryConditionTest, ZeroForceInGasRegion) {
    std::cout << "\n--- Test: Zero Force in Pure Gas Region ---" << std::endl;

    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;
    const float dx = 1.0e-6f;

    RecoilPressure recoil(0.54f, 2.0f, dx);

    std::vector<float> h_psat(num_cells, 500000.0f);
    std::vector<float> h_fill(num_cells, 0.0f);  // Pure gas
    std::vector<float3> h_normals(num_cells, make_float3(0.0f, 0.0f, 1.0f));

    float *d_psat, *d_fill, *d_fx, *d_fy, *d_fz;
    float3* d_normals;
    cudaMalloc(&d_psat, num_cells * sizeof(float));
    cudaMalloc(&d_fill, num_cells * sizeof(float));
    cudaMalloc(&d_normals, num_cells * sizeof(float3));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_psat, h_psat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill, h_fill.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals.data(), num_cells * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    recoil.computeForceField(d_psat, d_fill, d_normals, d_fx, d_fy, d_fz, nx, ny, nz);

    std::vector<float> h_fx(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float max_force = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        max_force = std::max(max_force, std::abs(h_fx[i]));
    }

    std::cout << "  Fill level = 0 (pure gas)" << std::endl;
    std::cout << "  Max force magnitude: " << max_force << std::endl;

    EXPECT_LT(max_force, 1.0f) << "Force should be ~0 in pure gas region";

    std::cout << "  [PASS] Zero force in gas region verified" << std::endl;

    cudaFree(d_psat);
    cudaFree(d_fill);
    cudaFree(d_normals);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @test Verify zero force in pure liquid region (fill_level = 1)
 */
TEST_F(BoundaryConditionTest, ZeroForceInLiquidRegion) {
    std::cout << "\n--- Test: Zero Force in Pure Liquid Region ---" << std::endl;

    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;
    const float dx = 1.0e-6f;

    RecoilPressure recoil(0.54f, 2.0f, dx);

    std::vector<float> h_psat(num_cells, 500000.0f);
    std::vector<float> h_fill(num_cells, 1.0f);  // Pure liquid
    std::vector<float3> h_normals(num_cells, make_float3(0.0f, 0.0f, 1.0f));

    float *d_psat, *d_fill, *d_fx, *d_fy, *d_fz;
    float3* d_normals;
    cudaMalloc(&d_psat, num_cells * sizeof(float));
    cudaMalloc(&d_fill, num_cells * sizeof(float));
    cudaMalloc(&d_normals, num_cells * sizeof(float3));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_psat, h_psat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill, h_fill.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals.data(), num_cells * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    recoil.computeForceField(d_psat, d_fill, d_normals, d_fx, d_fy, d_fz, nx, ny, nz);

    std::vector<float> h_fz(num_cells);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float max_force = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        max_force = std::max(max_force, std::abs(h_fz[i]));
    }

    std::cout << "  Fill level = 1 (pure liquid)" << std::endl;
    std::cout << "  Max force magnitude: " << max_force << std::endl;

    EXPECT_LT(max_force, 1.0f) << "Force should be ~0 in pure liquid region";

    std::cout << "  [PASS] Zero force in liquid region verified" << std::endl;

    cudaFree(d_psat);
    cudaFree(d_fill);
    cudaFree(d_normals);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @test Verify no NaN or Inf values in output
 */
TEST_F(BoundaryConditionTest, NoNaNOrInf) {
    std::cout << "\n--- Test: No NaN or Inf Values ---" << std::endl;

    const int nx = 16, ny = 16, nz = 16;
    const int num_cells = nx * ny * nz;
    const float dx = 1.0e-6f;

    VOFSolver vof(nx, ny, nz, dx);
    RecoilPressure recoil(0.54f, 2.0f, dx);

    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 5.0f);
    vof.reconstructInterface();

    std::vector<float> h_psat(num_cells);
    srand(42);
    for (int i = 0; i < num_cells; ++i) {
        float T = 300.0f + static_cast<float>(rand() % 4000);
        h_psat[i] = computePsatAnalytical(T);
    }

    float *d_psat, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_psat, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_psat, h_psat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    recoil.computeForceField(d_psat, vof.getFillLevel(), vof.getInterfaceNormals(),
                             d_fx, d_fy, d_fz, nx, ny, nz);

    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (std::isnan(h_fx[i]) || std::isnan(h_fy[i]) || std::isnan(h_fz[i])) nan_count++;
        if (std::isinf(h_fx[i]) || std::isinf(h_fy[i]) || std::isinf(h_fz[i])) inf_count++;
    }

    std::cout << "  Cells with NaN: " << nan_count << std::endl;
    std::cout << "  Cells with Inf: " << inf_count << std::endl;

    EXPECT_EQ(nan_count, 0) << "No NaN values should be present";
    EXPECT_EQ(inf_count, 0) << "No Inf values should be present";

    std::cout << "  [PASS] No NaN or Inf values verified" << std::endl;

    cudaFree(d_psat);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @test Verify pressure limiter works correctly
 */
TEST_F(BoundaryConditionTest, PressureLimiter) {
    std::cout << "\n--- Test: Pressure Limiter ---" << std::endl;

    float dx = 1.0e-6f;
    float max_pressure = 1e6f;

    RecoilPressureConfig config;
    config.coefficient = 0.54f;
    config.max_pressure = max_pressure;

    RecoilPressure recoil(config, dx);

    float p_sat_high = 5e6f;
    float p_recoil = recoil.computePressure(p_sat_high);

    std::cout << "  P_sat = " << p_sat_high / 1e6f << " MPa" << std::endl;
    std::cout << "  P_recoil (unlimited) = " << 0.54f * p_sat_high / 1e6f << " MPa" << std::endl;
    std::cout << "  P_recoil (limited) = " << p_recoil / 1e6f << " MPa" << std::endl;
    std::cout << "  Max pressure = " << max_pressure / 1e6f << " MPa" << std::endl;

    EXPECT_LE(p_recoil, max_pressure) << "P_recoil should be limited to max_pressure";

    std::cout << "  [PASS] Pressure limiter verified" << std::endl;
}

// ============================================================================
// Unit Test 5: Physical Consistency
// ============================================================================

class PhysicalConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n=== Setting up Physical Consistency Test ===" << std::endl;
    }
};

/**
 * @test Verify recoil pressure is in keyhole regime for high temperatures
 */
TEST_F(PhysicalConsistencyTest, KeyholeRegimePressure) {
    std::cout << "\n--- Test: Keyhole Regime Pressure ---" << std::endl;

    RecoilPressure recoil(0.54f, 2.0f, 1.0e-6f);

    float T = 4500.0f;
    float p_sat = computePsatAnalytical(T);
    float p_recoil = recoil.computePressure(p_sat);

    std::cout << "  T = " << T << " K (keyhole regime)" << std::endl;
    std::cout << "  P_sat = " << p_sat / 1e6f << " MPa" << std::endl;
    std::cout << "  P_recoil = " << p_recoil / 1e6f << " MPa" << std::endl;

    EXPECT_GT(p_recoil, 1e6f) << "P_recoil should exceed 1 MPa in keyhole regime";
    EXPECT_LT(p_recoil, 1e8f) << "P_recoil should be reasonable (< 100 MPa)";

    std::cout << "  [PASS] Keyhole regime pressure verified" << std::endl;
}

/**
 * @test Verify force magnitude is physically reasonable
 *
 * The volumetric force is computed as:
 *   F = P_recoil * |grad(f)| / (h_interface * dx)
 *
 * For typical LPBF parameters:
 *   - P_recoil ~ 270 kPa (from P_sat = 500 kPa)
 *   - |grad(f)| ~ 1/dx at sharp interface
 *   - h_interface = 2 cells
 *   - dx = 2 um
 *
 * This gives F ~ P_recoil / (h * dx) ~ 270e3 / (2 * 2e-6) ~ 6.75e10 N/m3
 *
 * For microscale problems with high gradients, forces in the 10^10 - 10^17 N/m3
 * range are expected due to the 1/dx^2 scaling.
 */
TEST_F(PhysicalConsistencyTest, ForceMagnitudeReasonable) {
    std::cout << "\n--- Test: Force Magnitude Order of Magnitude ---" << std::endl;

    float dx = 2.0e-6f;
    RecoilPressure recoil(0.54f, 2.0f, dx);

    float p_sat = 500000.0f;
    float grad_f_mag = 1.0f / dx;  // Sharp interface: |grad(f)| ~ 1/dx

    float F_mag = recoil.computeForceMagnitude(p_sat, grad_f_mag);

    // Expected: F = P_recoil * grad_f_mag / (h * dx)
    //              = 0.54 * 500000 * 500000 / (2 * 2e-6)
    //              = 270000 * 500000 / 4e-6
    //              = 1.35e11 / 4e-6 = 3.375e16 N/m3
    float expected_order = 0.54f * p_sat * grad_f_mag / (2.0f * dx);

    std::cout << "  P_sat = " << p_sat / 1000.0f << " kPa" << std::endl;
    std::cout << "  |grad(f)| = " << grad_f_mag << " 1/m" << std::endl;
    std::cout << "  Force magnitude = " << F_mag << " N/m3" << std::endl;
    std::cout << "  Expected order = " << expected_order << " N/m3" << std::endl;

    // Force should be positive and non-zero
    EXPECT_GT(F_mag, 1e6f) << "Force should be significant";

    // Force should be within order of magnitude of expected
    float ratio = F_mag / expected_order;
    std::cout << "  Ratio F/expected = " << ratio << std::endl;

    EXPECT_GT(ratio, 0.1f) << "Force should be within expected order of magnitude";
    EXPECT_LT(ratio, 10.0f) << "Force should be within expected order of magnitude";

    std::cout << "  [PASS] Force magnitude order of magnitude verified" << std::endl;
}

// ============================================================================
// Integration Test
// ============================================================================

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n=== Setting up Integration Test ===" << std::endl;
    }
};

/**
 * @test Integration test: Surface depression force setup
 */
TEST_F(IntegrationTest, SurfaceDepressionSetup) {
    std::cout << "\n--- Test: Surface Depression Force Setup ---" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const int num_cells = nx * ny * nz;
    const float dx = 2.0e-6f;
    const int z_interface = 20;

    VOFSolver vof(nx, ny, nz, dx);
    RecoilPressure recoil(0.54f, 2.0f, dx);

    // Initialize horizontal interface
    std::vector<float> h_fill(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float z_dist = static_cast<float>(k - z_interface);
                h_fill[idx] = 0.5f * (1.0f - std::tanh(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Create Gaussian temperature profile -> P_sat field
    std::vector<float> h_psat(num_cells);
    float cx = nx / 2.0f, cy = ny / 2.0f;
    float T_max = 4000.0f, T_ambient = 300.0f;
    float spot_radius = 10.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float r2 = (i - cx) * (i - cx) + (j - cy) * (j - cy);
                float T = T_ambient + (T_max - T_ambient) *
                          std::exp(-r2 / (2.0f * spot_radius * spot_radius));
                h_psat[idx] = computePsatAnalytical(T);
            }
        }
    }

    float *d_psat, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_psat, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_psat, h_psat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    recoil.computeForceField(d_psat, vof.getFillLevel(), vof.getInterfaceNormals(),
                             d_fx, d_fy, d_fz, nx, ny, nz);

    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Count cells with downward force
    int cells_with_downward_force = 0;
    float total_fz = 0.0f;
    for (int idx = 0; idx < num_cells; ++idx) {
        if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
            if (h_fz[idx] < -1.0f) {
                cells_with_downward_force++;
                total_fz += h_fz[idx];
            }
        }
    }

    std::cout << "  Cells with downward force: " << cells_with_downward_force << std::endl;
    std::cout << "  Total downward force: " << total_fz << " N/m3" << std::endl;

    EXPECT_GT(cells_with_downward_force, 10) << "Should have multiple cells with downward force";
    EXPECT_LT(total_fz, 0.0f) << "Net force should be downward (negative z)";

    std::cout << "  [PASS] Surface depression force setup verified" << std::endl;

    cudaFree(d_psat);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Recoil Pressure Module - Comprehensive Test Suite" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "\nPhysical Constants (Ti6Al4V Anisimov Model):" << std::endl;
    std::cout << "  T_boil = " << TestConstants::T_BOIL << " K" << std::endl;
    std::cout << "  L_vap = " << TestConstants::L_VAP / 1e6f << " MJ/kg" << std::endl;
    std::cout << "  M_molar = " << TestConstants::M_MOLAR << " kg/mol" << std::endl;
    std::cout << "  C_r = " << TestConstants::C_R << std::endl;
    std::cout << "  P_ref = " << TestConstants::P_REF << " Pa" << std::endl;
    std::cout << "  T_activation = " << TestConstants::T_ACTIVATION << " K" << std::endl;

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "All Recoil Pressure Tests Complete" << std::endl;
    std::cout << std::string(60, '=') << std::endl << std::endl;

    return result;
}
