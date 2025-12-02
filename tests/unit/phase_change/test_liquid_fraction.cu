/**
 * @file test_liquid_fraction.cu
 * @brief Unit tests for liquid fraction calculation
 *
 * Tests verify:
 * 1. fl = 0 for T < T_solidus
 * 2. fl = 1 for T > T_liquidus
 * 3. Linear variation in mushy zone
 * 4. Continuity at phase boundaries
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/material_properties.h"

using namespace lbm::physics;

class LiquidFractionTest : public ::testing::Test {
protected:
    MaterialProperties mat;

    void SetUp() override {
        // Use Ti6Al4V properties
        mat = MaterialDatabase::getTi6Al4V();
    }
};

/**
 * Test: Liquid fraction is zero below solidus
 */
TEST_F(LiquidFractionTest, ZeroBelowSolidus) {
    // Test at various temperatures below solidus
    float test_temps[] = {300.0f, 1000.0f, 1500.0f, 1877.0f};

    for (float T : test_temps) {
        ASSERT_LT(T, mat.T_solidus) << "Test temperature must be below solidus";

        float fl = mat.liquidFraction(T);

        EXPECT_FLOAT_EQ(fl, 0.0f)
            << "Liquid fraction should be zero at T = " << T << " K";
        EXPECT_TRUE(mat.isSolid(T))
            << "Material should be solid at T = " << T << " K";
        EXPECT_FALSE(mat.isLiquid(T))
            << "Material should not be liquid at T = " << T << " K";
        EXPECT_FALSE(mat.isMushy(T))
            << "Material should not be mushy at T = " << T << " K";
    }
}

/**
 * Test: Liquid fraction is one above liquidus
 */
TEST_F(LiquidFractionTest, OneAboveLiquidus) {
    // Test at various temperatures above liquidus
    float test_temps[] = {1929.0f, 2000.0f, 2500.0f, 3000.0f};

    for (float T : test_temps) {
        ASSERT_GT(T, mat.T_liquidus) << "Test temperature must be above liquidus";

        float fl = mat.liquidFraction(T);

        EXPECT_FLOAT_EQ(fl, 1.0f)
            << "Liquid fraction should be one at T = " << T << " K";
        EXPECT_TRUE(mat.isLiquid(T))
            << "Material should be liquid at T = " << T << " K";
        EXPECT_FALSE(mat.isSolid(T))
            << "Material should not be solid at T = " << T << " K";
        EXPECT_FALSE(mat.isMushy(T))
            << "Material should not be mushy at T = " << T << " K";
    }
}

/**
 * Test: Linear variation in mushy zone
 */
TEST_F(LiquidFractionTest, LinearInMushyZone) {
    // Ti6Al4V: T_solidus = 1878 K, T_liquidus = 1923 K
    // ΔT = 45 K

    float T_solidus = mat.T_solidus;
    float T_liquidus = mat.T_liquidus;
    float deltaT = T_liquidus - T_solidus;

    // Test at quarter points
    struct TestPoint {
        float T;
        float expected_fl;
    };

    TestPoint test_points[] = {
        {T_solidus, 0.0f},                      // At solidus
        {T_solidus + 0.25f * deltaT, 0.25f},    // Quarter melted
        {T_solidus + 0.5f * deltaT, 0.5f},      // Half melted
        {T_solidus + 0.75f * deltaT, 0.75f},    // Three-quarters melted
        {T_liquidus, 1.0f}                      // At liquidus
    };

    for (const auto& pt : test_points) {
        float fl = mat.liquidFraction(pt.T);

        EXPECT_NEAR(fl, pt.expected_fl, 1e-5f)
            << "Liquid fraction mismatch at T = " << pt.T << " K";
        EXPECT_TRUE(mat.isMushy(pt.T) || pt.T == T_solidus || pt.T == T_liquidus)
            << "Temperature " << pt.T << " K should be in mushy zone";
    }
}

/**
 * Test: Continuity at phase boundaries
 */
TEST_F(LiquidFractionTest, ContinuityAtBoundaries) {
    float epsilon = 0.01f;  // Small temperature increment

    // At solidus boundary
    {
        float T_below = mat.T_solidus - epsilon;
        float T_at = mat.T_solidus;
        float T_above = mat.T_solidus + epsilon;

        float fl_below = mat.liquidFraction(T_below);
        float fl_at = mat.liquidFraction(T_at);
        float fl_above = mat.liquidFraction(T_above);

        EXPECT_FLOAT_EQ(fl_below, 0.0f);
        EXPECT_FLOAT_EQ(fl_at, 0.0f);
        EXPECT_GT(fl_above, 0.0f);
        EXPECT_LT(fl_above, 0.01f);  // Should be very small
    }

    // At liquidus boundary
    {
        float T_below = mat.T_liquidus - epsilon;
        float T_at = mat.T_liquidus;
        float T_above = mat.T_liquidus + epsilon;

        float fl_below = mat.liquidFraction(T_below);
        float fl_at = mat.liquidFraction(T_at);
        float fl_above = mat.liquidFraction(T_above);

        EXPECT_LT(fl_below, 1.0f);
        EXPECT_GT(fl_below, 0.99f);  // Should be very close to 1
        EXPECT_FLOAT_EQ(fl_at, 1.0f);
        EXPECT_FLOAT_EQ(fl_above, 1.0f);
    }
}

/**
 * Test: Monotonicity (fl always increases with T)
 */
TEST_F(LiquidFractionTest, Monotonicity) {
    // Test from solid to liquid
    for (float T = 300.0f; T < 2500.0f; T += 10.0f) {
        float T_next = T + 10.0f;

        float fl_current = mat.liquidFraction(T);
        float fl_next = mat.liquidFraction(T_next);

        EXPECT_GE(fl_next, fl_current)
            << "Liquid fraction should be monotonically increasing: "
            << "fl(" << T << ") = " << fl_current << ", "
            << "fl(" << T_next << ") = " << fl_next;

        // Also check bounds
        EXPECT_GE(fl_current, 0.0f) << "Liquid fraction cannot be negative";
        EXPECT_LE(fl_current, 1.0f) << "Liquid fraction cannot exceed 1";
    }
}

/**
 * Test: Derivative in mushy zone
 *
 * dfl/dT = 1/(T_liquidus - T_solidus) in mushy zone
 */
TEST_F(LiquidFractionTest, DerivativeInMushyZone) {
    float deltaT = mat.T_liquidus - mat.T_solidus;
    float expected_dfl_dT = 1.0f / deltaT;

    // Test at midpoint of mushy zone
    float T_mid = (mat.T_solidus + mat.T_liquidus) / 2.0f;
    float dT = 0.1f;

    float fl1 = mat.liquidFraction(T_mid - dT / 2.0f);
    float fl2 = mat.liquidFraction(T_mid + dT / 2.0f);

    float numerical_dfl_dT = (fl2 - fl1) / dT;

    EXPECT_NEAR(numerical_dfl_dT, expected_dfl_dT, 1e-4f)
        << "Derivative dfl/dT in mushy zone should be constant";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
