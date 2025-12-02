/**
 * @file test_substrate_temperature_reduction.cu
 * @brief Integration test comparing substrate BC vs adiabatic bottom
 *
 * Test Suite 2.2: Validates that substrate cooling reduces temperature
 *
 * Success Criteria:
 * - Substrate BC reduces average temperature vs adiabatic case
 * - Temperature reduction is significant (>10%)
 * - System remains stable
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>

using namespace lbm::physics;

class SubstrateTemperatureReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        strncpy(material.name, "Ti6Al4V", sizeof(material.name)-1);
        material.name[sizeof(material.name)-1] = '\0';
        material.rho_solid = 4420.0f;
        material.cp_solid = 670.0f;
        material.k_solid = 6.8f;
        material.T_solidus = 1878.0f;
        material.T_liquidus = 1928.0f;
        material.L_fusion = 286000.0f;
    }

    MaterialProperties material;

    float computeAverageTemperature(ThermalLBM& thermal, int nx, int ny, int nz) {
        thermal.computeTemperature();
        std::vector<float> T_field(nx * ny * nz);
        thermal.copyTemperatureToHost(T_field.data());

        float sum = std::accumulate(T_field.begin(), T_field.end(), 0.0f);
        return sum / T_field.size();
    }
};

TEST_F(SubstrateTemperatureReductionTest, CompareAdiabaticVsConvective) {
    std::cout << "\nTest 1: Adiabatic vs Convective bottom BC" << std::endl;

    int nx = 30, ny = 30, nz = 40;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 1500.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  Initial T = " << T_init << " K" << std::endl;
    std::cout << "  Substrate T = " << T_substrate << " K" << std::endl;
    std::cout << "  h_conv = " << h_conv << " W/(m²·K)" << std::endl;

    // Test A: Adiabatic bottom (no substrate cooling)
    std::cout << "\n  Running adiabatic case..." << std::endl;
    ThermalLBM thermal_adiabatic(nx, ny, nz, material, alpha, false);
    thermal_adiabatic.initialize(T_init);

    for (int t = 0; t < 5000; ++t) {
        thermal_adiabatic.collisionBGK();
        thermal_adiabatic.streaming();
        // No substrate BC - adiabatic bottom

        if (t % 1000 == 0) {
            float T_avg = computeAverageTemperature(thermal_adiabatic, nx, ny, nz);
            std::cout << "    Step " << t << ": T_avg = " << T_avg << " K" << std::endl;
        }
    }

    float T_avg_adiabatic = computeAverageTemperature(thermal_adiabatic, nx, ny, nz);
    std::cout << "  Final T_avg (adiabatic): " << T_avg_adiabatic << " K" << std::endl;

    // Test B: Convective bottom (with substrate cooling)
    std::cout << "\n  Running convective case..." << std::endl;
    ThermalLBM thermal_convective(nx, ny, nz, material, alpha, false);
    thermal_convective.initialize(T_init);

    for (int t = 0; t < 5000; ++t) {
        thermal_convective.collisionBGK();
        thermal_convective.streaming();
        thermal_convective.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        if (t % 1000 == 0) {
            float T_avg = computeAverageTemperature(thermal_convective, nx, ny, nz);
            std::cout << "    Step " << t << ": T_avg = " << T_avg << " K" << std::endl;
        }
    }

    float T_avg_convective = computeAverageTemperature(thermal_convective, nx, ny, nz);
    std::cout << "  Final T_avg (convective): " << T_avg_convective << " K" << std::endl;

    // Substrate BC should reduce temperature
    EXPECT_LT(T_avg_convective, T_avg_adiabatic)
        << "Convective BC should reduce temperature vs adiabatic";

    // Reduction should be significant
    float reduction = (T_avg_adiabatic - T_avg_convective) / T_avg_adiabatic;
    std::cout << "\n  Temperature reduction: " << reduction * 100.0f << "%" << std::endl;

    EXPECT_GT(reduction, 0.10f) << "Temperature reduction < 10% (too small)";
    EXPECT_LT(reduction, 0.90f) << "Temperature reduction > 90% (unrealistic)";
}

TEST_F(SubstrateTemperatureReductionTest, ConvectionCoefficientScaling) {
    std::cout << "\nTest 2: Effect of h_conv on cooling" << std::endl;

    int nx = 20, ny = 20, nz = 30;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 2000.0f;
    float T_substrate = 300.0f;

    std::vector<float> h_values = {100.0f, 500.0f, 1000.0f, 5000.0f};
    std::vector<float> T_finals;

    for (float h_conv : h_values) {
        ThermalLBM thermal(nx, ny, nz, material, alpha, false);
        thermal.initialize(T_init);

        for (int t = 0; t < 3000; ++t) {
            thermal.collisionBGK();
            thermal.streaming();
            thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
        }

        float T_avg = computeAverageTemperature(thermal, nx, ny, nz);
        T_finals.push_back(T_avg);

        std::cout << "  h_conv = " << h_conv << " W/(m²·K): T_final = "
                  << T_avg << " K" << std::endl;
    }

    // Higher h_conv should lead to lower final temperature
    for (size_t i = 1; i < T_finals.size(); ++i) {
        EXPECT_LT(T_finals[i], T_finals[i-1])
            << "Temperature should decrease with higher h_conv";
    }

    // At very high h_conv, should approach substrate temperature
    float approach_ratio = (T_finals.back() - T_substrate) / (T_init - T_substrate);
    std::cout << "  Approach to substrate temp: " << (1.0f - approach_ratio) * 100.0f
              << "%" << std::endl;

    EXPECT_LT(approach_ratio, 0.80f) << "Should cool significantly with high h_conv";
}

TEST_F(SubstrateTemperatureReductionTest, SpatialTemperatureProfile) {
    std::cout << "\nTest 3: Spatial temperature profile with substrate cooling" << std::endl;

    int nx = 15, ny = 15, nz = 40;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 1800.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    // Run to quasi-steady state
    for (int t = 0; t < 8000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
    }

    thermal.computeTemperature();
    std::vector<float> T_field(nx * ny * nz);
    thermal.copyTemperatureToHost(T_field.data());

    // Compute average temperature at each z-level
    std::vector<float> T_z(nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                T_z[k] += T_field[idx];
            }
        }
        T_z[k] /= (nx * ny);
    }

    std::cout << "  Temperature profile (bottom to top):" << std::endl;
    for (int k = 0; k < nz; k += 8) {
        std::cout << "    z=" << k << " (" << k*dx*1e6 << " µm): T = "
                  << T_z[k] << " K" << std::endl;
    }

    // Bottom should be coolest
    EXPECT_LT(T_z[0], T_z[nz/2]) << "Bottom should be cooler than middle";
    EXPECT_LT(T_z[0], T_z[nz-1]) << "Bottom should be cooler than top";

    // Should have temperature gradient
    float dT = T_z[nz-1] - T_z[0];
    std::cout << "  Temperature difference (top-bottom): " << dT << " K" << std::endl;

    EXPECT_GT(dT, 100.0f) << "Temperature gradient too small";
}

TEST_F(SubstrateTemperatureReductionTest, TransientCoolingRate) {
    std::cout << "\nTest 4: Transient cooling rate with substrate BC" << std::endl;

    int nx = 20, ny = 20, nz = 30;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 2500.0f;
    float T_substrate = 300.0f;
    float h_conv = 2000.0f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    std::vector<float> T_avg_history;
    std::vector<int> time_steps;

    std::cout << "  Monitoring temperature evolution:" << std::endl;

    for (int t = 0; t <= 10000; t += 500) {
        if (t > 0) {
            for (int step = 0; step < 500; ++step) {
                thermal.collisionBGK();
                thermal.streaming();
                thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
            }
        }

        float T_avg = computeAverageTemperature(thermal, nx, ny, nz);
        T_avg_history.push_back(T_avg);
        time_steps.push_back(t);

        float physical_time = t * dt * 1e6;  // microseconds
        std::cout << "    t=" << physical_time << " µs: T_avg=" << T_avg << " K" << std::endl;
    }

    // Temperature should monotonically decrease
    for (size_t i = 1; i < T_avg_history.size(); ++i) {
        EXPECT_LT(T_avg_history[i], T_avg_history[i-1])
            << "Temperature should decrease over time";
    }

    // Should be approaching substrate temperature
    EXPECT_LT(T_avg_history.back(), T_init * 0.70f)
        << "Should cool significantly after 10000 steps";

    // Cooling rate should decrease over time (approaching equilibrium)
    float rate_early = (T_avg_history[0] - T_avg_history[1]) / 500.0f;
    float rate_late = (T_avg_history[T_avg_history.size()-2] - T_avg_history.back()) / 500.0f;

    std::cout << "  Early cooling rate: " << rate_early << " K/step" << std::endl;
    std::cout << "  Late cooling rate: " << rate_late << " K/step" << std::endl;

    EXPECT_GT(rate_early, rate_late) << "Cooling rate should decrease over time";
}

TEST_F(SubstrateTemperatureReductionTest, PowerConsistencyCheck) {
    std::cout << "\nTest 5: Substrate power consistency" << std::endl;

    int nx = 25, ny = 25, nz = 35;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 2000.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    // Run for some steps
    for (int t = 0; t < 1000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
    }

    // Compute substrate power
    float P_substrate = thermal.computeSubstratePower(dx, h_conv, T_substrate);
    std::cout << "  Substrate cooling power: " << P_substrate << " W" << std::endl;

    // Power should be positive (cooling)
    EXPECT_GT(P_substrate, 0.0f) << "Substrate power should be positive";

    // Power should be realistic for this domain size
    float domain_area = (nx * dx) * (ny * dx);  // m²
    float q_avg = P_substrate / domain_area;    // W/m²
    std::cout << "  Average heat flux: " << q_avg << " W/m²" << std::endl;

    // Heat flux should be reasonable
    thermal.computeTemperature();
    std::vector<float> T_field(nx * ny * nz);
    thermal.copyTemperatureToHost(T_field.data());

    // Average bottom surface temperature
    float T_bottom = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * 0);
            T_bottom += T_field[idx];
        }
    }
    T_bottom /= (nx * ny);

    float q_expected = h_conv * (T_bottom - T_substrate);
    std::cout << "  Bottom surface T: " << T_bottom << " K" << std::endl;
    std::cout << "  Expected heat flux: " << q_expected << " W/m²" << std::endl;

    // Should match within reasonable tolerance
    float error = fabs(q_avg - q_expected) / q_expected;
    std::cout << "  Relative error: " << error * 100.0f << "%" << std::endl;

    EXPECT_LT(error, 0.20f) << "Heat flux error > 20%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
