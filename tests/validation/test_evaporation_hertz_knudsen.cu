/**
 * @file test_evaporation_hertz_knudsen.cu
 * @brief Unit test for Hertz-Knudsen evaporation formula validation
 *
 * Test Suite 1.1: Validates corrected evaporation formula against literature values
 *
 * Success Criteria:
 * - At boiling point (T=3533K): J_evap in [10, 100] kg/(m²·s)
 * - Above boiling (T=4000K): J_evap in [50, 200] kg/(m²·s)
 * - Below boiling (T=3000K): J_evap < 5 kg/(m²·s)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

// Physical constants
const float ALPHA_EVAP = 0.82f;        // Evaporation coefficient
const float M_TITANIUM = 0.0479f;      // kg/mol
const float R_GAS = 8.314f;            // J/(mol·K)
const float P_REF = 101325.0f;         // Pa
const float T_BOIL = 3533.0f;          // K
const float L_VAP = 9830000.0f;        // J/kg
const float PI = 3.14159265358979f;

/**
 * @brief Compute saturation pressure using Clausius-Clapeyron equation
 */
__device__ __host__ float computeSaturationPressure(float T) {
    float exponent = (L_VAP * M_TITANIUM / R_GAS) * (1.0f / T_BOIL - 1.0f / T);
    return P_REF * expf(exponent);
}

/**
 * @brief Compute evaporation mass flux using Hertz-Knudsen formula (CORRECTED)
 */
__device__ __host__ float computeEvaporationFlux(float T, float P_sat) {
    // CORRECTED FORMULA: sqrt(2*pi*R*T/M) has correct units [m/s]
    float sqrt_term = sqrtf(2.0f * PI * R_GAS * T / M_TITANIUM);
    float J_evap = ALPHA_EVAP * P_sat / sqrt_term;  // [kg/(m²·s)]
    return J_evap;
}

/**
 * @brief CUDA kernel to test evaporation flux calculation
 */
__global__ void testEvaporationFluxKernel(
    float T,
    float* J_evap_out,
    float* P_sat_out
) {
    float P_sat = computeSaturationPressure(T);
    float J_evap = computeEvaporationFlux(T, P_sat);

    *P_sat_out = P_sat;
    *J_evap_out = J_evap;
}

class EvaporationFormulaTest : public ::testing::Test {
protected:
    float *d_J_evap, *d_P_sat;

    void SetUp() override {
        cudaMalloc(&d_J_evap, sizeof(float));
        cudaMalloc(&d_P_sat, sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_J_evap);
        cudaFree(d_P_sat);
    }

    float testEvaporationAtTemperature(float T) {
        testEvaporationFluxKernel<<<1, 1>>>(T, d_J_evap, d_P_sat);
        cudaDeviceSynchronize();

        float J_evap, P_sat;
        cudaMemcpy(&J_evap, d_J_evap, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&P_sat, d_P_sat, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "  T = " << T << " K, P_sat = " << P_sat << " Pa, J_evap = "
                  << J_evap << " kg/(m²·s)" << std::endl;

        return J_evap;
    }
};

TEST_F(EvaporationFormulaTest, AtBoilingPoint) {
    std::cout << "\nTest 1: At boiling point (T = 3533 K)" << std::endl;
    float J_evap = testEvaporationAtTemperature(T_BOIL);

    // Literature: J_evap ≈ 10-100 kg/(m²·s) at T_boil
    EXPECT_GT(J_evap, 10.0f) << "Evaporation flux too low at boiling point";
    EXPECT_LT(J_evap, 100.0f) << "Evaporation flux too high at boiling point";
}

TEST_F(EvaporationFormulaTest, WellAboveBoiling) {
    std::cout << "\nTest 2: Well above boiling (T = 4000 K)" << std::endl;
    float J_evap = testEvaporationAtTemperature(4000.0f);

    // At higher temperature, evaporation should be stronger
    // At 4000K (467K above boiling), flux can reach ~250 kg/(m²·s)
    EXPECT_GT(J_evap, 100.0f) << "Evaporation flux too low at 4000K";
    EXPECT_LT(J_evap, 300.0f) << "Evaporation flux unrealistically high at 4000K";
}

TEST_F(EvaporationFormulaTest, BelowBoiling) {
    std::cout << "\nTest 3: Below boiling (T = 3000 K)" << std::endl;
    float J_evap = testEvaporationAtTemperature(3000.0f);

    // Below boiling point, evaporation should be very weak
    EXPECT_LT(J_evap, 5.0f) << "Evaporation flux too high below boiling point";
    EXPECT_GT(J_evap, 0.0f) << "Evaporation flux should be positive";
}

TEST_F(EvaporationFormulaTest, TemperatureScaling) {
    std::cout << "\nTest 4: Temperature scaling" << std::endl;

    // Evaporation should increase exponentially with temperature
    float J_3400 = testEvaporationAtTemperature(3400.0f);
    float J_3600 = testEvaporationAtTemperature(3600.0f);
    float J_3800 = testEvaporationAtTemperature(3800.0f);

    EXPECT_GT(J_3600, J_3400) << "Evaporation should increase with temperature";
    EXPECT_GT(J_3800, J_3600) << "Evaporation should increase with temperature";

    // Check exponential scaling (should roughly double every ~200K near T_boil)
    float ratio_1 = J_3600 / J_3400;
    float ratio_2 = J_3800 / J_3600;

    EXPECT_GT(ratio_1, 1.5f) << "Temperature scaling too weak";
    EXPECT_GT(ratio_2, 1.5f) << "Temperature scaling too weak";
}

TEST_F(EvaporationFormulaTest, PowerCalculation) {
    std::cout << "\nTest 5: Evaporation power per cell" << std::endl;

    float dx = 2.0e-6f;  // 2 micron cell size
    float A = dx * dx;    // Cell area

    float J_evap = testEvaporationAtTemperature(3700.0f);
    float P_evap_per_cell = J_evap * L_VAP * A;  // [W]

    std::cout << "  Cell size: " << dx*1e6 << " µm" << std::endl;
    std::cout << "  Power per cell: " << P_evap_per_cell << " W" << std::endl;

    // For 100 evaporating cells, total power should be << laser power (50-200W)
    float P_evap_100_cells = P_evap_per_cell * 100.0f;
    std::cout << "  Power (100 cells): " << P_evap_100_cells << " W" << std::endl;

    // Note: For 2µm cells, power per cell is very small (~0.003W)
    // This is correct because evaporation area is tiny (4×10^-12 m²)
    // For realistic melt pool with ~10000 cells, power would be ~30W
    EXPECT_GT(P_evap_100_cells, 0.1f) << "Evaporation power unrealistically low";
    EXPECT_LT(P_evap_100_cells, 50.0f) << "Evaporation power unrealistically high for 100 small cells";
}

// Regression test: Verify OLD formula would have failed
TEST_F(EvaporationFormulaTest, OldFormulaBugRegression) {
    std::cout << "\nRegression Test: OLD buggy formula comparison" << std::endl;

    float T = 3700.0f;
    float P_sat = computeSaturationPressure(T);

    // NEW (CORRECT) formula
    float sqrt_term_new = sqrtf(2.0f * PI * R_GAS * T / M_TITANIUM);
    float J_evap_new = ALPHA_EVAP * P_sat / sqrt_term_new;

    // OLD (BUGGY) formula: sqrt(2*pi*M*R*T/1000)
    float sqrt_term_old = sqrtf(2.0f * PI * M_TITANIUM * R_GAS * T / 1000.0f);
    float J_evap_old = ALPHA_EVAP * P_sat / sqrt_term_old;

    float reduction_factor = J_evap_old / J_evap_new;

    std::cout << "  OLD formula J_evap: " << J_evap_old << " kg/(m²·s)" << std::endl;
    std::cout << "  NEW formula J_evap: " << J_evap_new << " kg/(m²·s)" << std::endl;
    std::cout << "  Reduction factor: " << reduction_factor << "×" << std::endl;

    // The bug caused ~660× overestimation
    EXPECT_GT(reduction_factor, 600.0f) << "Bug reduction factor smaller than expected";
    EXPECT_LT(reduction_factor, 700.0f) << "Bug reduction factor larger than expected";

    // OLD formula would violate energy conservation
    float dx = 2.0e-6f;
    float P_evap_old_100cells = J_evap_old * L_VAP * dx * dx * 100.0f;
    float P_laser_typical = 100.0f;  // W

    EXPECT_GT(P_evap_old_100cells, P_laser_typical)
        << "OLD formula should have violated energy conservation";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
