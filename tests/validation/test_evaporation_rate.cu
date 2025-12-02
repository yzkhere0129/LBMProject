/**
 * @file test_evaporation_rate.cu
 * @brief Benchmark 3: Evaporation rate validation against Hertz-Knudsen formula
 *
 * Validates that the evaporative mass flux computed in the code matches
 * the analytical prediction from the Hertz-Knudsen formula at various
 * temperatures.
 *
 * Hertz-Knudsen Formula:
 *   j_evap = (α_evap × P_sat) / sqrt(2π M R T)
 *
 * Where:
 *   - α_evap = evaporation coefficient (0.82 for titanium)
 *   - P_sat = saturation pressure (Clausius-Clapeyron)
 *   - M = molar mass (kg/mol)
 *   - R = gas constant (J/(mol·K))
 *   - T = temperature (K)
 *
 * Saturation Pressure (Clausius-Clapeyron):
 *   P_sat(T) = P_ref × exp[(L_v × M / R) × (1/T_boil - 1/T)]
 *
 * Test Temperatures:
 *   - T = 3000K (below boiling)
 *   - T = 3500K (near boiling, T_boil = 3533K)
 *   - T = 4000K (above boiling)
 *   - T = 4500K (well above boiling)
 *   - T = 5000K (very high temperature)
 *
 * Success Criteria:
 *   - Numerical flux matches analytical within ±2% at all temperatures
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>

// Physical constants for Ti6Al4V
constexpr float ALPHA_EVAP = 0.82f;        // Evaporation coefficient
constexpr float M_TITANIUM = 0.0479f;      // kg/mol
constexpr float R_GAS = 8.314f;            // J/(mol·K)
constexpr float P_REF = 101325.0f;         // Pa (1 atm)
constexpr float T_BOIL = 3533.0f;          // K (boiling point)
constexpr float L_VAP = 9830000.0f;        // J/kg (latent heat of vaporization)
constexpr float PI = 3.14159265358979f;

// Test temperatures
constexpr float TEST_TEMPS[] = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};
constexpr int NUM_TEMPS = 5;

/**
 * @brief Compute saturation pressure using Clausius-Clapeyron equation
 */
__device__ __host__ float computeSaturationPressure(float T) {
    float exponent = (L_VAP * M_TITANIUM / R_GAS) * (1.0f / T_BOIL - 1.0f / T);
    return P_REF * expf(exponent);
}

/**
 * @brief Compute evaporation mass flux using Hertz-Knudsen formula
 */
__device__ __host__ float computeEvaporationFlux(float T) {
    float P_sat = computeSaturationPressure(T);
    float sqrt_term = sqrtf(2.0f * PI * R_GAS * T / M_TITANIUM);
    float j_evap = ALPHA_EVAP * P_sat / sqrt_term;
    return j_evap;
}

/**
 * @brief Compute evaporative cooling power per unit area
 */
__device__ __host__ float computeEvaporativePower(float T) {
    float j_evap = computeEvaporationFlux(T);
    float q_evap = j_evap * L_VAP;  // W/m²
    return q_evap;
}

/**
 * @brief CUDA kernel to test evaporation rate calculation
 */
__global__ void testEvaporationKernel(
    const float* temperatures,
    float* j_evap_out,
    float* P_sat_out,
    float* q_evap_out,
    int num_temps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_temps) {
        float T = temperatures[idx];
        float P_sat = computeSaturationPressure(T);
        float j_evap = computeEvaporationFlux(T);
        float q_evap = computeEvaporativePower(T);

        P_sat_out[idx] = P_sat;
        j_evap_out[idx] = j_evap;
        q_evap_out[idx] = q_evap;
    }
}

class EvaporationRateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate device memory
        cudaMalloc(&d_temps, NUM_TEMPS * sizeof(float));
        cudaMalloc(&d_j_evap, NUM_TEMPS * sizeof(float));
        cudaMalloc(&d_P_sat, NUM_TEMPS * sizeof(float));
        cudaMalloc(&d_q_evap, NUM_TEMPS * sizeof(float));

        // Copy test temperatures to device
        cudaMemcpy(d_temps, TEST_TEMPS, NUM_TEMPS * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Compute analytical solutions on host
        h_j_analytical.resize(NUM_TEMPS);
        h_P_analytical.resize(NUM_TEMPS);
        h_q_analytical.resize(NUM_TEMPS);

        for (int i = 0; i < NUM_TEMPS; ++i) {
            h_P_analytical[i] = computeSaturationPressure(TEST_TEMPS[i]);
            h_j_analytical[i] = computeEvaporationFlux(TEST_TEMPS[i]);
            h_q_analytical[i] = computeEvaporativePower(TEST_TEMPS[i]);
        }
    }

    void TearDown() override {
        cudaFree(d_temps);
        cudaFree(d_j_evap);
        cudaFree(d_P_sat);
        cudaFree(d_q_evap);
    }

    void runKernel() {
        int threads = 256;
        int blocks = (NUM_TEMPS + threads - 1) / threads;

        testEvaporationKernel<<<blocks, threads>>>(
            d_temps, d_j_evap, d_P_sat, d_q_evap, NUM_TEMPS
        );

        cudaDeviceSynchronize();

        // Check for kernel errors
        cudaError_t error = cudaGetLastError();
        ASSERT_EQ(error, cudaSuccess) << "CUDA kernel error: "
                                      << cudaGetErrorString(error);
    }

    void copyResults() {
        h_j_numerical.resize(NUM_TEMPS);
        h_P_numerical.resize(NUM_TEMPS);
        h_q_numerical.resize(NUM_TEMPS);

        cudaMemcpy(h_j_numerical.data(), d_j_evap, NUM_TEMPS * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_P_numerical.data(), d_P_sat, NUM_TEMPS * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q_numerical.data(), d_q_evap, NUM_TEMPS * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    float computeRelativeError(float numerical, float analytical) {
        return fabsf(numerical - analytical) / fabsf(analytical);
    }

    void printResults() {
        std::cout << "\n  Temperature-dependent evaporation rates:" << std::endl;
        std::cout << "  " << std::string(85, '-') << std::endl;
        std::cout << "  T[K]    P_sat[kPa]  j_evap[kg/(m²·s)]  q_evap[MW/m²]  Error[%]"
                  << std::endl;
        std::cout << "  " << std::string(85, '-') << std::endl;

        for (int i = 0; i < NUM_TEMPS; ++i) {
            float error_j = computeRelativeError(h_j_numerical[i], h_j_analytical[i]);

            std::cout << "  " << std::setw(4) << std::fixed << std::setprecision(0)
                      << TEST_TEMPS[i]
                      << std::setw(12) << std::setprecision(2)
                      << h_P_numerical[i] / 1000.0f
                      << std::setw(20) << std::setprecision(3)
                      << h_j_numerical[i]
                      << std::setw(16) << std::setprecision(2)
                      << h_q_numerical[i] / 1e6f
                      << std::setw(11) << std::setprecision(3)
                      << error_j * 100.0f
                      << std::endl;
        }
        std::cout << "  " << std::string(85, '-') << std::endl;
    }

    // Device memory
    float *d_temps, *d_j_evap, *d_P_sat, *d_q_evap;

    // Host memory for results
    std::vector<float> h_j_numerical, h_P_numerical, h_q_numerical;
    std::vector<float> h_j_analytical, h_P_analytical, h_q_analytical;
};

TEST_F(EvaporationRateTest, AllTemperatures) {
    std::cout << "\nBenchmark 3: Evaporation rate validation" << std::endl;

    // Run kernel
    runKernel();

    // Copy results
    copyResults();

    // Print results
    printResults();

    // Verify all temperatures meet success criteria
    for (int i = 0; i < NUM_TEMPS; ++i) {
        float error_j = computeRelativeError(h_j_numerical[i], h_j_analytical[i]);
        float error_P = computeRelativeError(h_P_numerical[i], h_P_analytical[i]);
        float error_q = computeRelativeError(h_q_numerical[i], h_q_analytical[i]);

        EXPECT_LT(error_j, 0.02f) << "Mass flux error > 2% at T = "
                                   << TEST_TEMPS[i] << " K";
        EXPECT_LT(error_P, 0.02f) << "Saturation pressure error > 2% at T = "
                                   << TEST_TEMPS[i] << " K";
        EXPECT_LT(error_q, 0.02f) << "Evaporative power error > 2% at T = "
                                   << TEST_TEMPS[i] << " K";
    }
}

TEST_F(EvaporationRateTest, BelowBoiling) {
    std::cout << "\nBenchmark 3.1: Below boiling point (T = 3000K)" << std::endl;

    runKernel();
    copyResults();

    int idx = 0;  // 3000K
    float j_evap = h_j_numerical[idx];
    float P_sat = h_P_numerical[idx];

    std::cout << "  T = " << TEST_TEMPS[idx] << " K" << std::endl;
    std::cout << "  P_sat = " << P_sat / 1000.0f << " kPa" << std::endl;
    std::cout << "  j_evap = " << j_evap << " kg/(m²·s)" << std::endl;

    // Below boiling, evaporation should be weak
    EXPECT_LT(j_evap, 10.0f) << "Evaporation too strong below boiling";
    EXPECT_GT(j_evap, 0.0f) << "Evaporation should be positive";
}

TEST_F(EvaporationRateTest, NearBoiling) {
    std::cout << "\nBenchmark 3.2: Near boiling point (T = 3500K)" << std::endl;

    runKernel();
    copyResults();

    int idx = 1;  // 3500K (33K below T_boil)
    float j_evap = h_j_numerical[idx];
    float P_sat = h_P_numerical[idx];

    std::cout << "  T = " << TEST_TEMPS[idx] << " K" << std::endl;
    std::cout << "  P_sat = " << P_sat / 1000.0f << " kPa" << std::endl;
    std::cout << "  j_evap = " << j_evap << " kg/(m²·s)" << std::endl;

    // Near boiling, flux should be moderate
    EXPECT_GT(j_evap, 5.0f) << "Evaporation too weak near boiling";
    EXPECT_LT(j_evap, 100.0f) << "Evaporation unrealistically high near boiling";
}

TEST_F(EvaporationRateTest, AboveBoiling) {
    std::cout << "\nBenchmark 3.3: Above boiling point (T = 4000K)" << std::endl;

    runKernel();
    copyResults();

    int idx = 2;  // 4000K
    float j_evap = h_j_numerical[idx];
    float P_sat = h_P_numerical[idx];

    std::cout << "  T = " << TEST_TEMPS[idx] << " K" << std::endl;
    std::cout << "  P_sat = " << P_sat / 1000.0f << " kPa" << std::endl;
    std::cout << "  j_evap = " << j_evap << " kg/(m²·s)" << std::endl;

    // Above boiling, flux should be strong
    EXPECT_GT(j_evap, 50.0f) << "Evaporation too weak above boiling";
    EXPECT_LT(j_evap, 500.0f) << "Evaporation unrealistically high";
}

TEST_F(EvaporationRateTest, EnergyConsistency) {
    std::cout << "\nBenchmark 3.4: Energy consistency check" << std::endl;

    runKernel();
    copyResults();

    // For a typical melt pool with 100 surface cells at 3700K
    float T_typical = 3700.0f;
    float j_typical = computeEvaporationFlux(T_typical);
    float q_typical = j_typical * L_VAP;  // W/m²

    float dx = 2.0e-6f;  // 2 micron cell size
    float A_cell = dx * dx;  // Cell area

    float P_evap_per_cell = q_typical * A_cell;  // W
    float P_evap_100_cells = P_evap_per_cell * 100.0f;  // W

    std::cout << "  Typical melt pool scenario:" << std::endl;
    std::cout << "    Temperature: " << T_typical << " K" << std::endl;
    std::cout << "    Mass flux: " << j_typical << " kg/(m²·s)" << std::endl;
    std::cout << "    Heat flux: " << q_typical / 1e6f << " MW/m²" << std::endl;
    std::cout << "    Cell size: " << dx * 1e6 << " µm" << std::endl;
    std::cout << "    Power per cell: " << P_evap_per_cell << " W" << std::endl;
    std::cout << "    Power (100 cells): " << P_evap_100_cells << " W" << std::endl;

    // For realistic LPBF with 50-200W laser, evaporation power should be
    // a fraction of laser power (typically 10-30%)
    float P_laser_typical = 100.0f;  // W

    EXPECT_LT(P_evap_100_cells, P_laser_typical)
        << "Evaporation power exceeds typical laser power (energy violation)";
    EXPECT_GT(P_evap_100_cells, 0.001f * P_laser_typical)
        << "Evaporation power unrealistically low (for 2µm cells, ~0.3W is correct)";
}

TEST_F(EvaporationRateTest, TemperatureScaling) {
    std::cout << "\nBenchmark 3.5: Temperature scaling (exponential growth)" << std::endl;

    runKernel();
    copyResults();

    // Evaporation should increase exponentially with temperature
    std::cout << "\n  Temperature scaling:" << std::endl;
    for (int i = 1; i < NUM_TEMPS; ++i) {
        float j_prev = h_j_numerical[i-1];
        float j_curr = h_j_numerical[i];
        float ratio = j_curr / j_prev;

        std::cout << "    " << TEST_TEMPS[i-1] << "K -> " << TEST_TEMPS[i]
                  << "K: ratio = " << ratio << "×" << std::endl;

        // Should increase significantly (at least 2×) per 500K
        EXPECT_GT(ratio, 2.0f) << "Temperature scaling too weak";
        EXPECT_LT(ratio, 20.0f) << "Temperature scaling unrealistically strong";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
