/**
 * @file test_substrate_bc_stability.cu
 * @brief Stress test for substrate cooling BC stability
 *
 * Test Suite 2.3: Long-run stability test
 *
 * Success Criteria:
 * - No NaN or Inf after 100,000 timesteps
 * - No negative temperatures
 * - Temperatures remain physical (<10,000 K)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace lbm::physics;

class SubstrateBCStabilityTest : public ::testing::Test {
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

    void checkTemperatureValidity(ThermalLBM& thermal, int nx, int ny, int nz, int step) {
        thermal.computeTemperature();
        std::vector<float> T_field(nx * ny * nz);
        thermal.copyTemperatureToHost(T_field.data());

        float T_min = T_field[0];
        float T_max = T_field[0];

        for (int i = 0; i < nx * ny * nz; ++i) {
            T_min = fmin(T_min, T_field[i]);
            T_max = fmax(T_max, T_field[i]);

            ASSERT_FALSE(std::isnan(T_field[i]))
                << "NaN detected at index " << i << " at step " << step;
            ASSERT_FALSE(std::isinf(T_field[i]))
                << "Inf detected at index " << i << " at step " << step;
        }

        ASSERT_GE(T_min, 0.0f) << "Negative temperature at step " << step;
        ASSERT_LT(T_max, 10000.0f) << "Unrealistic temperature at step " << step;
    }
};

TEST_F(SubstrateBCStabilityTest, LongRunStability) {
    std::cout << "\nTest 1: Long-run stability (100k steps)" << std::endl;

    int nx = 40, ny = 40, nz = 50;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 2000.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  Running 100,000 timesteps..." << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    for (int t = 0; t < 100000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        // Check validity every 1000 steps
        if (t % 1000 == 0) {
            checkTemperatureValidity(thermal, nx, ny, nz, t);

            if (t % 10000 == 0) {
                thermal.computeTemperature();
                std::vector<float> T_field(nx * ny * nz);
                thermal.copyTemperatureToHost(T_field.data());

                float T_min = T_field[0], T_max = T_field[0];
                for (float T : T_field) {
                    T_min = fmin(T_min, T);
                    T_max = fmax(T_max, T);
                }

                std::cout << "    Step " << t << ": T_min=" << T_min
                          << " K, T_max=" << T_max << " K" << std::endl;
            }
        }
    }

    std::cout << "  100,000 steps completed - STABLE" << std::endl;
}

TEST_F(SubstrateBCStabilityTest, ExtremeConvectionCoefficient) {
    std::cout << "\nTest 2: Extreme h_conv stress test" << std::endl;

    int nx = 20, ny = 20, nz = 25;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 1000.0f;
    float T_substrate = 300.0f;
    float h_conv = 50000.0f;  // Extremely high

    std::cout << "  Extreme h_conv = " << h_conv << " W/(m²·K)" << std::endl;
    std::cout << "  Running 10,000 timesteps..." << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    for (int t = 0; t < 10000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        if (t % 500 == 0) {
            checkTemperatureValidity(thermal, nx, ny, nz, t);

            if (t % 2000 == 0) {
                thermal.computeTemperature();
                std::vector<float> T_field(nx * ny * nz);
                thermal.copyTemperatureToHost(T_field.data());

                float T_min = T_field[0];
                for (float T : T_field) {
                    T_min = fmin(T_min, T);
                }

                std::cout << "    Step " << t << ": T_min=" << T_min << " K" << std::endl;
            }
        }
    }

    std::cout << "  Extreme h_conv test passed - STABLE" << std::endl;
}

TEST_F(SubstrateBCStabilityTest, SmallTemperatureDifference) {
    std::cout << "\nTest 3: Small temperature difference (near equilibrium)" << std::endl;

    int nx = 25, ny = 25, nz = 30;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 350.0f;      // Close to substrate
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    std::cout << "  Small ΔT = " << (T_init - T_substrate) << " K" << std::endl;
    std::cout << "  Running 20,000 timesteps..." << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    for (int t = 0; t < 20000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        if (t % 2000 == 0) {
            checkTemperatureValidity(thermal, nx, ny, nz, t);
        }
    }

    thermal.computeTemperature();
    std::vector<float> T_field(nx * ny * nz);
    thermal.copyTemperatureToHost(T_field.data());

    float T_avg = 0.0f;
    for (float T : T_field) {
        T_avg += T;
    }
    T_avg /= T_field.size();

    std::cout << "  Final T_avg = " << T_avg << " K" << std::endl;
    std::cout << "  Should be close to substrate T - STABLE" << std::endl;

    EXPECT_GT(T_avg, T_substrate) << "Temperature below substrate";
    EXPECT_LT(T_avg, T_init) << "Temperature didn't decrease";
}

TEST_F(SubstrateBCStabilityTest, LargeTemperatureDifference) {
    std::cout << "\nTest 4: Large temperature difference" << std::endl;

    int nx = 20, ny = 20, nz = 25;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 5000.0f;     // Very hot
    float T_substrate = 300.0f;
    float h_conv = 2000.0f;

    std::cout << "  Large ΔT = " << (T_init - T_substrate) << " K" << std::endl;
    std::cout << "  Running 15,000 timesteps..." << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    for (int t = 0; t < 15000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        if (t % 1000 == 0) {
            checkTemperatureValidity(thermal, nx, ny, nz, t);

            if (t % 3000 == 0) {
                thermal.computeTemperature();
                std::vector<float> T_field(nx * ny * nz);
                thermal.copyTemperatureToHost(T_field.data());

                float T_avg = 0.0f;
                for (float T : T_field) {
                    T_avg += T;
                }
                T_avg /= T_field.size();

                std::cout << "    Step " << t << ": T_avg=" << T_avg << " K" << std::endl;
            }
        }
    }

    std::cout << "  Large ΔT test passed - STABLE" << std::endl;
}

TEST_F(SubstrateBCStabilityTest, VaryingTimeStep) {
    std::cout << "\nTest 5: Stability with different timesteps" << std::endl;

    int nx = 20, ny = 20, nz = 25;
    float dx = 2.0e-6f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 2000.0f;
    float T_substrate = 300.0f;
    float h_conv = 1000.0f;

    std::vector<float> timesteps = {5.0e-10f, 1.0e-9f, 2.0e-9f, 5.0e-9f};

    for (float dt : timesteps) {
        std::cout << "  Testing dt = " << dt*1e9 << " ns" << std::endl;

        ThermalLBM thermal(nx, ny, nz, material, alpha, false);
        thermal.initialize(T_init);

        int n_steps = static_cast<int>(5000.0f * (1.0e-9f / dt));  // Equivalent physical time

        for (int t = 0; t < n_steps; ++t) {
            thermal.collisionBGK();
            thermal.streaming();
            thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

            if (t % (n_steps / 5) == 0) {
                checkTemperatureValidity(thermal, nx, ny, nz, t);
            }
        }

        std::cout << "    dt = " << dt*1e9 << " ns: STABLE (" << n_steps << " steps)" << std::endl;
    }
}

TEST_F(SubstrateBCStabilityTest, CombinedStressTest) {
    std::cout << "\nTest 6: Combined stress test (extreme conditions)" << std::endl;

    int nx = 30, ny = 30, nz = 35;
    float dx = 1.0e-6f;       // Smaller grid spacing
    float dt = 5.0e-10f;      // Smaller timestep
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    float T_init = 4000.0f;   // Very hot
    float T_substrate = 300.0f;
    float h_conv = 10000.0f;  // Very high

    std::cout << "  Stress conditions:" << std::endl;
    std::cout << "    dx = " << dx*1e6 << " µm" << std::endl;
    std::cout << "    dt = " << dt*1e9 << " ns" << std::endl;
    std::cout << "    T_init = " << T_init << " K" << std::endl;
    std::cout << "    h_conv = " << h_conv << " W/(m²·K)" << std::endl;
    std::cout << "  Running 50,000 timesteps..." << std::endl;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(T_init);

    for (int t = 0; t < 50000; ++t) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);

        if (t % 5000 == 0) {
            checkTemperatureValidity(thermal, nx, ny, nz, t);

            if (t % 10000 == 0) {
                thermal.computeTemperature();
                std::vector<float> T_field(nx * ny * nz);
                thermal.copyTemperatureToHost(T_field.data());

                float T_min = T_field[0], T_max = T_field[0];
                for (float T : T_field) {
                    T_min = fmin(T_min, T);
                    T_max = fmax(T_max, T);
                }

                std::cout << "    Step " << t << ": T_min=" << T_min
                          << " K, T_max=" << T_max << " K" << std::endl;
            }
        }
    }

    std::cout << "  Combined stress test PASSED - ROBUST" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
