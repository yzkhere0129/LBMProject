/**
 * @file test_liquid_fraction_copy.cu
 * @brief Unit test for liquid fraction device-to-host copy
 *
 * This test verifies that:
 * 1. copyLiquidFractionToHost correctly retrieves data from device
 * 2. Phase change solver properly computes and stores liquid fraction
 * 3. Data integrity is maintained through the copy operation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm;

class LiquidFractionCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        nx = 10;
        ny = 10;
        nz = 10;
        num_cells = nx * ny * nz;

        material = physics::MaterialDatabase::getTi6Al4V();
    }

    int nx, ny, nz, num_cells;
    physics::MaterialProperties material;
};

/**
 * Test 1: Verify copy returns zeros for solid phase
 */
TEST_F(LiquidFractionCopyTest, CopyReturnsZeroForSolidPhase) {
    std::cout << "\n=== TEST: Copy Liquid Fraction - Solid Phase ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(300.0f);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);

    // Initialize to room temperature (well below melting point)
    float T_solid = 300.0f;
    thermal.initialize(T_solid);

    std::cout << "Temperature: " << T_solid << " K (solid phase)\n";
    std::cout << "Melting point: " << material.T_liquidus << " K\n";

    // Copy liquid fraction to host
    float* h_fl = new float[num_cells];
    thermal.copyLiquidFractionToHost(h_fl);

    // Verify all values are zero
    bool all_zero = true;
    float max_fl = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        max_fl = fmaxf(max_fl, h_fl[i]);
        if (h_fl[i] != 0.0f) {
            all_zero = false;
        }
    }

    std::cout << "Maximum liquid fraction: " << max_fl << "\n";
    std::cout << "All values zero: " << (all_zero ? "YES" : "NO") << "\n";

    delete[] h_fl;

    EXPECT_TRUE(all_zero) << "Liquid fraction should be zero in solid phase";
    EXPECT_EQ(max_fl, 0.0f) << "Maximum liquid fraction should be exactly 0.0";
}

/**
 * Test 2: Verify copy returns ones for liquid phase
 */
TEST_F(LiquidFractionCopyTest, CopyReturnsOneForLiquidPhase) {
    std::cout << "\n=== TEST: Copy Liquid Fraction - Liquid Phase ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(2500.0f);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);

    // Initialize to well above melting point
    float T_liquid = 2500.0f;
    thermal.initialize(T_liquid);

    std::cout << "Temperature: " << T_liquid << " K (liquid phase)\n";
    std::cout << "Melting point: " << material.T_liquidus << " K\n";

    // Copy liquid fraction to host
    float* h_fl = new float[num_cells];
    thermal.copyLiquidFractionToHost(h_fl);

    // Verify all values are 1.0
    bool all_one = true;
    float min_fl = 1.0f;
    float max_fl = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        min_fl = fminf(min_fl, h_fl[i]);
        max_fl = fmaxf(max_fl, h_fl[i]);
        if (fabsf(h_fl[i] - 1.0f) > 1e-5f) {
            all_one = false;
        }
    }

    std::cout << "Liquid fraction range: [" << min_fl << ", " << max_fl << "]\n";
    std::cout << "All values ~1.0: " << (all_one ? "YES" : "NO") << "\n";

    delete[] h_fl;

    EXPECT_TRUE(all_one) << "Liquid fraction should be 1.0 in liquid phase";
    EXPECT_NEAR(min_fl, 1.0f, 1e-5f) << "Minimum should be close to 1.0";
    EXPECT_NEAR(max_fl, 1.0f, 1e-5f) << "Maximum should be close to 1.0";
}

/**
 * Test 3: Verify copy returns intermediate values in mushy zone
 */
TEST_F(LiquidFractionCopyTest, CopyReturnsIntermediateForMushyZone) {
    std::cout << "\n=== TEST: Copy Liquid Fraction - Mushy Zone ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(1900.0f);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);

    // Initialize to middle of melting range
    float T_mushy = (material.T_solidus + material.T_liquidus) / 2.0f;
    thermal.initialize(T_mushy);

    std::cout << "Temperature: " << T_mushy << " K (mushy zone)\n";
    std::cout << "Solidus: " << material.T_solidus << " K\n";
    std::cout << "Liquidus: " << material.T_liquidus << " K\n";

    // Expected liquid fraction at midpoint
    float expected_fl = (T_mushy - material.T_solidus) /
                        (material.T_liquidus - material.T_solidus);

    // Copy liquid fraction to host
    float* h_fl = new float[num_cells];
    thermal.copyLiquidFractionToHost(h_fl);

    // Verify values are in (0, 1)
    bool all_intermediate = true;
    float min_fl = 1.0f;
    float max_fl = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        min_fl = fminf(min_fl, h_fl[i]);
        max_fl = fmaxf(max_fl, h_fl[i]);
        if (h_fl[i] <= 0.0f || h_fl[i] >= 1.0f) {
            all_intermediate = false;
        }
    }

    std::cout << "Expected liquid fraction: " << expected_fl << "\n";
    std::cout << "Actual liquid fraction range: [" << min_fl << ", " << max_fl << "]\n";
    std::cout << "All values in (0, 1): " << (all_intermediate ? "YES" : "NO") << "\n";

    delete[] h_fl;

    EXPECT_TRUE(all_intermediate) << "Liquid fraction should be in (0, 1) in mushy zone";
    EXPECT_NEAR(min_fl, expected_fl, 0.01f) << "Should match expected value";
    EXPECT_NEAR(max_fl, expected_fl, 0.01f) << "Should match expected value";
}

/**
 * Test 4: Verify copy after temperature evolution
 *
 * This test sets up a non-uniform temperature field, evolves it,
 * then verifies that liquid fraction correctly reflects the temperature.
 */
TEST_F(LiquidFractionCopyTest, CopyAfterTemperatureEvolution) {
    std::cout << "\n=== TEST: Copy After Temperature Evolution ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(300.0f);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);

    // Create non-uniform temperature field
    // Center hot (above melting), edges cold (below melting)
    float* h_temp = new float[num_cells];
    int center_x = nx / 2;
    int center_y = ny / 2;
    int center_z = nz / 2;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;

                // Distance from center
                float dx = (i - center_x);
                float dy = (j - center_y);
                float dz = (k - center_z);
                float r2 = dx*dx + dy*dy + dz*dz;

                // Gaussian temperature profile
                // Center: 2500K (liquid), edges: 300K (solid)
                h_temp[idx] = 300.0f + 2200.0f * expf(-r2 / 10.0f);
            }
        }
    }

    thermal.initialize(h_temp);

    // Evolve a few steps
    for (int step = 0; step < 10; ++step) {
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Copy temperature and liquid fraction
    float* h_temp_final = new float[num_cells];
    float* h_fl = new float[num_cells];
    thermal.copyTemperatureToHost(h_temp_final);
    thermal.copyLiquidFractionToHost(h_fl);

    // Count cells in each phase
    int n_solid = 0, n_mushy = 0, n_liquid = 0;
    float max_fl_solid_region = 0.0f;
    float min_fl_liquid_region = 1.0f;

    for (int i = 0; i < num_cells; ++i) {
        float T = h_temp_final[i];
        float fl = h_fl[i];

        if (T < material.T_solidus) {
            n_solid++;
            max_fl_solid_region = fmaxf(max_fl_solid_region, fl);
        } else if (T > material.T_liquidus) {
            n_liquid++;
            min_fl_liquid_region = fminf(min_fl_liquid_region, fl);
        } else {
            n_mushy++;
        }
    }

    std::cout << "Phase distribution:\n";
    std::cout << "  Solid cells (T < " << material.T_solidus << "): " << n_solid << "\n";
    std::cout << "  Mushy cells: " << n_mushy << "\n";
    std::cout << "  Liquid cells (T > " << material.T_liquidus << "): " << n_liquid << "\n";
    std::cout << "Max fl in solid region: " << max_fl_solid_region << "\n";
    std::cout << "Min fl in liquid region: " << min_fl_liquid_region << "\n";

    delete[] h_temp;
    delete[] h_temp_final;
    delete[] h_fl;

    // Verify phase-liquid fraction consistency
    EXPECT_LT(max_fl_solid_region, 0.01f)
        << "Liquid fraction should be ~0 in solid region";
    EXPECT_GT(min_fl_liquid_region, 0.99f)
        << "Liquid fraction should be ~1 in liquid region";
}

/**
 * Test 5: Verify data integrity (no corruption)
 *
 * This test writes known pattern to temperature, verifies liquid fraction
 * copy doesn't corrupt the data.
 */
TEST_F(LiquidFractionCopyTest, DataIntegrityNoCorruption) {
    std::cout << "\n=== TEST: Data Integrity - No Corruption ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(300.0f);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);

    // Create temperature field with specific pattern
    float* h_temp_init = new float[num_cells];
    for (int i = 0; i < num_cells; ++i) {
        // Pattern: alternating hot and cold
        h_temp_init[i] = (i % 2 == 0) ? 2500.0f : 300.0f;
    }

    thermal.initialize(h_temp_init);

    // Copy liquid fraction
    float* h_fl = new float[num_cells];
    thermal.copyLiquidFractionToHost(h_fl);

    // Copy temperature back and verify it wasn't corrupted
    float* h_temp_verify = new float[num_cells];
    thermal.copyTemperatureToHost(h_temp_verify);

    // Check that temperature field is intact
    bool temp_intact = true;
    int n_mismatches = 0;
    for (int i = 0; i < num_cells; ++i) {
        float expected = h_temp_init[i];
        float actual = h_temp_verify[i];
        if (fabsf(actual - expected) > 1.0f) {
            temp_intact = false;
            n_mismatches++;
        }
    }

    // Check that liquid fraction matches pattern
    int n_fl_correct = 0;
    for (int i = 0; i < num_cells; ++i) {
        float expected_fl = (i % 2 == 0) ? 1.0f : 0.0f;
        float actual_fl = h_fl[i];
        if (fabsf(actual_fl - expected_fl) < 0.01f) {
            n_fl_correct++;
        }
    }

    std::cout << "Temperature field intact: " << (temp_intact ? "YES" : "NO") << "\n";
    std::cout << "Temperature mismatches: " << n_mismatches << " / " << num_cells << "\n";
    std::cout << "Liquid fraction correct: " << n_fl_correct << " / " << num_cells << "\n";

    delete[] h_temp_init;
    delete[] h_temp_verify;
    delete[] h_fl;

    EXPECT_TRUE(temp_intact) << "Temperature field should not be corrupted";
    EXPECT_EQ(n_fl_correct, num_cells)
        << "Liquid fraction should match temperature pattern";
}

/**
 * Test 6: Multiple consecutive copies
 *
 * Verify that calling copyLiquidFractionToHost multiple times
 * returns consistent results.
 */
TEST_F(LiquidFractionCopyTest, MultipleConsecutiveCopies) {
    std::cout << "\n=== TEST: Multiple Consecutive Copies ===\n";

    float thermal_diffusivity = material.getThermalDiffusivity(1900.0f);
    physics::ThermalLBM thermal(nx, ny, nz, material, thermal_diffusivity, true);

    float T_mushy = 1900.0f;
    thermal.initialize(T_mushy);

    // Perform multiple copies
    const int N_COPIES = 5;
    float** h_fl_copies = new float*[N_COPIES];
    for (int i = 0; i < N_COPIES; ++i) {
        h_fl_copies[i] = new float[num_cells];
        thermal.copyLiquidFractionToHost(h_fl_copies[i]);
    }

    // Verify all copies are identical
    bool all_identical = true;
    int n_differences = 0;
    for (int copy = 1; copy < N_COPIES; ++copy) {
        for (int i = 0; i < num_cells; ++i) {
            if (h_fl_copies[copy][i] != h_fl_copies[0][i]) {
                all_identical = false;
                n_differences++;
            }
        }
    }

    std::cout << "Performed " << N_COPIES << " consecutive copies\n";
    std::cout << "All copies identical: " << (all_identical ? "YES" : "NO") << "\n";
    std::cout << "Total differences: " << n_differences << "\n";

    for (int i = 0; i < N_COPIES; ++i) {
        delete[] h_fl_copies[i];
    }
    delete[] h_fl_copies;

    EXPECT_TRUE(all_identical) << "Multiple copies should return identical data";
    EXPECT_EQ(n_differences, 0) << "Should have zero differences between copies";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
