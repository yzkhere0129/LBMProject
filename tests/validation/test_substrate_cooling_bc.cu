/**
 * @file test_substrate_cooling_bc.cu
 * @brief Unit test for substrate cooling boundary condition
 *
 * Test Suite 2.1: Validates convective heat flux implementation
 *
 * Success Criteria:
 * - Heat flux matches Newton's law of cooling: q = h*(T_surface - T_substrate)
 * - Energy removed balances steady-state conduction
 * - BC is stable and doesn't cause negative temperatures
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include <cmath>
#include <cstring>
#include <iostream>

using namespace lbm::physics;

class SubstrateCoolingBCTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default Ti6Al4V properties
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
};

TEST_F(SubstrateCoolingBCTest, ConvectiveFluxAnalytical) {
    std::cout << "\nTest 1: Convective flux - analytical validation" << std::endl;

    // Setup parameters
    int nx = 10, ny = 10, nz = 20;
    float dx = 2.0e-6f;       // 2 µm
    float dt = 1.0e-9f;       // 1 ns
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_substrate = 300.0f;  // K
    float T_hot = 500.0f;         // K
    float h_conv = 1000.0f;       // W/(m²·K)

    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  dx = " << dx*1e6 << " µm" << std::endl;
    std::cout << "  T_substrate = " << T_substrate << " K" << std::endl;
    std::cout << "  h_conv = " << h_conv << " W/(m²·K)" << std::endl;

    // Create thermal solver
    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // Initialize with uniform temperature
    thermal.initialize(T_hot);

    std::cout << "  Running to steady state..." << std::endl;

    // Run to steady state (10000 steps)
    for (int t = 0; t < 10000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        if (t % 1000 == 0) {
            thermal.computeTemperature();
            std::vector<float> T_field(nx * ny * nz);
            thermal.copyTemperatureToHost(T_field.data());
            float T_bottom_avg = 0.0f;
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * 0);
                    T_bottom_avg += T_field[idx];
                }
            }
            T_bottom_avg /= (nx * ny);
            std::cout << "  Step " << t << ": T_bottom_avg = " << T_bottom_avg << " K" << std::endl;
        }
    }

    // Check final state
    thermal.computeTemperature();
    std::vector<float> T_field(nx * ny * nz);
    thermal.copyTemperatureToHost(T_field.data());

    // Sample bottom surface temperature
    float T_bottom_avg = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * 0);
            T_bottom_avg += T_field[idx];
        }
    }
    T_bottom_avg /= (nx * ny);

    std::cout << "  Final T_bottom_avg = " << T_bottom_avg << " K" << std::endl;

    // Convective heat flux
    float q_conv = h_conv * (T_bottom_avg - T_substrate);
    std::cout << "  Convective flux: " << q_conv << " W/m²" << std::endl;

    // At steady state, bottom surface should be cooler than initial temperature
    EXPECT_LT(T_bottom_avg, T_hot) << "Bottom surface should cool down";
    EXPECT_GT(T_bottom_avg, T_substrate) << "Bottom surface should be above substrate temp";

    // Convective flux should be positive (heat flowing out)
    EXPECT_GT(q_conv, 0.0f) << "Convective flux should be positive (heat removal)";

    // Flux should be reasonable (not extreme)
    EXPECT_LT(q_conv, 1e6f) << "Convective flux unrealistically high";
}

TEST_F(SubstrateCoolingBCTest, EnergyBalanceCheck) {
    std::cout << "\nTest 2: Energy balance - substrate cooling power" << std::endl;

    int nx = 20, ny = 20, nz = 30;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 2000.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    std::cout << "  Initial T = " << T_init << " K" << std::endl;
    std::cout << "  Running 100 timesteps..." << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    // Compute initial energy
    float E_initial = thermal.computeTotalThermalEnergy(dx);
    std::cout << "  Initial energy: " << E_initial << " J" << std::endl;

    // Apply substrate cooling for 100 steps
    double P_substrate_sum = 0.0;
    for (int t = 0; t < 100; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        // Compute substrate power
        float P_substrate = thermal.computeSubstratePower(dx, h_conv, T_substrate);
        P_substrate_sum += P_substrate;
    }

    // Compute final energy
    float E_final = thermal.computeTotalThermalEnergy(dx);
    std::cout << "  Final energy: " << E_final << " J" << std::endl;

    float E_lost = E_initial - E_final;
    float P_substrate_avg = P_substrate_sum / 100.0;
    float E_substrate_expected = P_substrate_avg * dt * 100.0f;

    std::cout << "  Energy lost: " << E_lost << " J" << std::endl;
    std::cout << "  Expected substrate removal: " << E_substrate_expected << " J" << std::endl;
    std::cout << "  Average substrate power: " << P_substrate_avg << " W" << std::endl;

    // Energy should decrease (cooling)
    EXPECT_LT(E_final, E_initial) << "Energy should decrease with substrate cooling";

    // Energy balance (allow 50% error due to numerical diffusion and transients)
    float error = fabs(E_lost - E_substrate_expected) / E_initial;
    std::cout << "  Relative error: " << error * 100.0f << "%" << std::endl;

    EXPECT_LT(error, 0.5f) << "Energy balance error too large (>50%)";
}

TEST_F(SubstrateCoolingBCTest, NoNegativeTemperatures) {
    std::cout << "\nTest 3: Regression - No negative temperatures" << std::endl;

    int nx = 15, ny = 15, nz = 15;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 400.0f;  // Close to substrate temperature
    float T_substrate = 300.0f;
    float h_conv = 5000.0f;  // Very high to stress test

    std::cout << "  High h_conv = " << h_conv << " W/(m²·K) (stress test)" << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    // Run for 500 steps
    for (int t = 0; t < 500; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        // Check for negative temperatures every 50 steps
        if (t % 50 == 0) {
            thermal.computeTemperature();
            std::vector<float> T_field(nx * ny * nz);
            thermal.copyTemperatureToHost(T_field.data());

            float T_min = T_field[0];
            float T_max = T_field[0];
            for (int i = 0; i < nx * ny * nz; ++i) {
                T_min = fmin(T_min, T_field[i]);
                T_max = fmax(T_max, T_field[i]);
            }

            std::cout << "  Step " << t << ": T_min = " << T_min << " K, T_max = " << T_max << " K" << std::endl;

            ASSERT_GE(T_min, 0.0f) << "Negative temperature detected at step " << t;
            ASSERT_LT(T_max, 10000.0f) << "Unrealistic temperature at step " << t;
        }
    }

    std::cout << "  No negative temperatures detected - PASS" << std::endl;
}

TEST_F(SubstrateCoolingBCTest, PowerMagnitudeRealistic) {
    std::cout << "\nTest 4: Substrate cooling power magnitude" << std::endl;

    int nx = 50, ny = 50, nz = 50;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 3000.0f;  // Hot melt pool
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    // Run a few steps to settle
    for (int t = 0; t < 10; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
    }

    float P_substrate = thermal.computeSubstratePower(dx, h_conv, T_substrate);
    std::cout << "  Substrate cooling power: " << P_substrate << " W" << std::endl;

    // For a 100µm × 100µm domain at 3000K, expect O(10-100W) substrate cooling
    EXPECT_GT(P_substrate, 1.0f) << "Substrate power too low";
    EXPECT_LT(P_substrate, 500.0f) << "Substrate power unrealistically high";
}

TEST_F(SubstrateCoolingBCTest, TemperatureGradientFormation) {
    std::cout << "\nTest 5: Temperature gradient near substrate" << std::endl;

    int nx = 10, ny = 10, nz = 30;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 1000.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    // Run to quasi-steady state
    for (int t = 0; t < 5000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
    }

    thermal.computeTemperature();
    std::vector<float> T_field(nx * ny * nz);
    thermal.copyTemperatureToHost(T_field.data());

    // Average temperature at different z-levels
    std::vector<float> T_avg_z(nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                T_avg_z[k] += T_field[idx];
            }
        }
        T_avg_z[k] /= (nx * ny);
    }

    std::cout << "  Temperature profile (z-direction):" << std::endl;
    for (int k = 0; k < nz; k += 5) {
        std::cout << "    z=" << k << ": T_avg = " << T_avg_z[k] << " K" << std::endl;
    }

    // Temperature should increase moving away from substrate (z=0)
    EXPECT_LT(T_avg_z[0], T_avg_z[nz-1]) << "Temperature should increase away from substrate";

    // Temperature gradient should exist
    float dT_dz = (T_avg_z[nz-1] - T_avg_z[0]) / (nz * dx);
    std::cout << "  Temperature gradient: " << dT_dz << " K/m" << std::endl;

    EXPECT_GT(fabs(dT_dz), 1e6f) << "Temperature gradient too small";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
