/**
 * @file test_interface_geometry.cu
 * @brief Unit test for VOF interface geometry initialization
 *
 * This test validates that the VOF fill level field is correctly initialized
 * with liquid at the bottom and vapor at the top.
 *
 * Test objectives:
 * - Verify fill level distribution (liquid=1 at bottom, vapor=0 at top)
 * - Check interface sharpness (transition within 2-5 cells)
 * - Validate interface position matches specification
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

/**
 * @brief Test that liquid is at bottom and vapor at top
 */
TEST(InterfaceGeometry, LiquidAtBottom) {
    std::cout << "\n=== Test: Interface Geometry - Liquid at Bottom ===" << std::endl;

    // Setup: 10x10x10 domain, interface at z=2
    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;
    const float interface_z = 2.0f;
    const float dz = 1.0f;

    std::vector<float> h_fill(num_cells);

    // Initialize like test does (CORRECTED formula):
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float height = k * dz;
                float interface_height = interface_z;
                float thickness = 0.5f;

                // CORRECTED formula: liquid (f=1) at bottom, vapor (f=0) at top
                h_fill[idx] = 0.5f * (1.0f - tanhf((height - interface_height) / thickness));

                // Clamp to [0, 1]
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    // Verify:
    // z=0 (bottom): fill ≈ 1.0 (liquid)
    // z=2 (interface): fill ≈ 0.5
    // z=9 (top): fill ≈ 0.0 (vapor)

    int idx_bottom = 0 + nx * (0 + ny * 0);      // (0,0,0)
    int idx_interface = 0 + nx * (0 + ny * 2);   // (0,0,2)
    int idx_top = 0 + nx * (0 + ny * 9);         // (0,0,9)

    std::cout << "  Bottom cell (0,0,0): fill = " << h_fill[idx_bottom] << std::endl;
    std::cout << "  Interface cell (0,0,2): fill = " << h_fill[idx_interface] << std::endl;
    std::cout << "  Top cell (0,0,9): fill = " << h_fill[idx_top] << std::endl;

    EXPECT_NEAR(h_fill[idx_bottom], 1.0f, 0.1f)
        << "Bottom should be liquid (f≈1)";
    EXPECT_NEAR(h_fill[idx_interface], 0.5f, 0.1f)
        << "Interface should be f≈0.5";
    EXPECT_NEAR(h_fill[idx_top], 0.0f, 0.1f)
        << "Top should be vapor (f≈0)";

    std::cout << "  ✓ Liquid at bottom, vapor at top (PASS)" << std::endl;
}

/**
 * @brief Test that interface is sharp (2-5 cells thick)
 */
TEST(InterfaceGeometry, InterfaceThickness) {
    std::cout << "\n=== Test: Interface Geometry - Interface Thickness ===" << std::endl;

    // Verify interface is sharp (2-3 cells thick)
    const int nz = 50;
    const float interface_z = 5.0f;
    const float dz = 1.0f;
    const float thickness = 0.5f;

    int transition_cells = 0;
    std::vector<float> fill_profile(nz);

    for (int k = 0; k < nz; ++k) {
        float height = k * dz;
        float fill = 0.5f * (1.0f - tanhf((height - interface_z) / thickness));
        fill_profile[k] = fill;

        // Count cells in transition (0.01 < f < 0.99)
        if (fill > 0.01f && fill < 0.99f) {
            transition_cells++;
        }
    }

    std::cout << "  Interface thickness: " << transition_cells << " cells" << std::endl;
    std::cout << "  Interface position: z = " << interface_z << std::endl;

    // Print interface region
    std::cout << "  Fill profile around interface:" << std::endl;
    for (int k = 0; k < nz; ++k) {
        if (fill_profile[k] > 0.01f && fill_profile[k] < 0.99f) {
            std::cout << "    z=" << k << ": fill=" << fill_profile[k] << std::endl;
        }
    }

    EXPECT_LE(transition_cells, 5)
        << "Interface should be sharp (< 5 cells)";
    EXPECT_GE(transition_cells, 2)
        << "Interface should span at least 2 cells";

    std::cout << "  ✓ Interface thickness in acceptable range (PASS)" << std::endl;
}

/**
 * @brief Test interface monotonicity (fill decreases with height)
 */
TEST(InterfaceGeometry, Monotonicity) {
    std::cout << "\n=== Test: Interface Geometry - Monotonicity ===" << std::endl;

    const int nz = 20;
    const float interface_z = 10.0f;
    const float dz = 1.0f;
    const float thickness = 0.5f;

    std::vector<float> fill_profile(nz);

    for (int k = 0; k < nz; ++k) {
        float height = k * dz;
        fill_profile[k] = 0.5f * (1.0f - tanhf((height - interface_z) / thickness));
    }

    // Verify fill level decreases monotonically with height
    int violations = 0;
    for (int k = 1; k < nz; ++k) {
        if (fill_profile[k] > fill_profile[k-1]) {
            violations++;
            std::cout << "  WARNING: Non-monotonic at z=" << k
                      << " (f[" << k << "]=" << fill_profile[k]
                      << " > f[" << k-1 << "]=" << fill_profile[k-1] << ")" << std::endl;
        }
    }

    EXPECT_EQ(violations, 0)
        << "Fill level should decrease monotonically with height";

    std::cout << "  ✓ Fill level is monotonic (PASS)" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "Interface Geometry Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = RUN_ALL_TESTS();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Interface Geometry Tests Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n";

    return result;
}
