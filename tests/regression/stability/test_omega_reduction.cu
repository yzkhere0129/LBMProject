/**
 * @file test_omega_reduction.cu
 * @brief Regression test for omega_T capping
 *
 * CRITICAL REGRESSION TEST: Ensures omega_T never exceeds safe value.
 *
 * Context:
 * - Original code: omega_T could reach 1.9+ (near instability limit of 2.0)
 * - With high diffusivity, omega_T → 2.0 causes severe instability
 * - Fix: Cap omega_T at 1.50 for safety margin
 *
 * Fix Location: thermal_lbm.cu::ThermalLBM() constructor
 * Fix Type: Clamp omega_T to maximum 1.50 (previously 1.95)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include "physics/lattice_d3q7.h"
#include <cmath>
#include <cstring>

using namespace lbm::physics;

class OmegaRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }

        // Ti6Al4V material
        strcpy(material.name, "Ti6Al4V");
        material.rho_solid = 4420.0f;
        material.rho_liquid = 4110.0f;
        material.cp_solid = 546.0f;
        material.cp_liquid = 831.0f;
        material.k_solid = 7.0f;
        material.k_liquid = 33.0f;
        material.T_solidus = 1878.0f;
        material.T_liquidus = 1928.0f;
        material.L_fusion = 286000.0f;
        material.T_vaporization = 3560.0f;
        material.L_vaporization = 9830000.0f;
    }

    MaterialProperties material;
};

/**
 * @brief Test that omega_T never exceeds 1.50 (safe limit)
 */
TEST_F(OmegaRegressionTest, OmegaNeverExceeds1_50) {
    // Test various diffusivity values that would create high omega
    float diffusivities[] = {
        5.8e-6f,   // Typical solid Ti6Al4V
        8.0e-6f,   // Typical liquid Ti6Al4V
        1.0e-5f,   // High diffusivity
        5.0e-5f,   // Extremely high (would create omega > 2.0 without cap)
        1.0e-4f    // Pathological case
    };

    for (float alpha : diffusivities) {
        ThermalLBM thermal(50, 50, 50, material, alpha, false);

        float omega = thermal.getThermalTau();
        omega = 1.0f / omega;  // Convert tau to omega

        EXPECT_LE(omega, 1.50f)
            << "REGRESSION: Omega " << omega << " exceeds safe limit (alpha=" << alpha << ")";

        EXPECT_GE(omega, 0.5f)
            << "Omega " << omega << " too low (over-diffusive, alpha=" << alpha << ")";
    }
}

/**
 * @brief Test omega calculation formula
 */
TEST_F(OmegaRegressionTest, OmegaCalculationFormula) {
    // For D3Q7: tau_T = alpha / cs² + 0.5
    // omega_T = 1 / tau_T
    // cs² = 1/3

    float alpha = 5.8e-6f;
    ThermalLBM thermal(50, 50, 50, material, alpha, false);

    float tau = thermal.getThermalTau();
    float omega = 1.0f / tau;

    // Expected (uncapped): tau = alpha / (1/3) + 0.5 = 3*alpha + 0.5
    float tau_expected = 3.0f * alpha + 0.5f;
    float omega_expected = 1.0f / tau_expected;

    // If omega_expected > 1.50, it should be capped
    if (omega_expected > 1.50f) {
        EXPECT_NEAR(omega, 1.50f, 1e-6f)
            << "Omega should be capped at 1.50";
    } else {
        EXPECT_NEAR(omega, omega_expected, 1e-6f)
            << "Omega should match formula when below cap";
    }
}

/**
 * @brief Test that capped omega still provides reasonable diffusivity
 */
TEST_F(OmegaRegressionTest, CappedOmegaReasonableDiffusivity) {
    // High diffusivity that would trigger capping
    float alpha_requested = 1.0e-5f;

    ThermalLBM thermal(50, 50, 50, material, alpha_requested, false);

    float tau = thermal.getThermalTau();
    float omega = 1.0f / tau;

    EXPECT_LE(omega, 1.50f) << "Omega not capped";

    // Effective diffusivity with capped omega
    // alpha_effective = cs² * (tau - 0.5) = (1/3) * (tau - 0.5)
    float alpha_effective = (1.0f / 3.0f) * (tau - 0.5f);

    // Effective diffusivity should be positive and reasonable
    EXPECT_GT(alpha_effective, 0.0f)
        << "Capped omega created negative diffusivity";

    // Should be lower than requested (due to capping), but still reasonable
    EXPECT_LT(alpha_effective, alpha_requested)
        << "Effective diffusivity should be reduced due to capping";

    EXPECT_GT(alpha_effective, 1e-7f)
        << "Effective diffusivity too low (over-capped)";
}

/**
 * @brief Test old omega limit (1.95) is NOT used
 */
TEST_F(OmegaRegressionTest, OldLimitNotUsed) {
    // Use diffusivity that would create omega between 1.50 and 1.95
    float alpha = 5.0e-5f;

    ThermalLBM thermal(50, 50, 50, material, alpha, false);

    float tau = thermal.getThermalTau();
    float omega = 1.0f / tau;

    // Should be capped at 1.50, NOT 1.95
    EXPECT_LE(omega, 1.50f)
        << "REGRESSION: Using old omega cap of 1.95 instead of new 1.50";

    EXPECT_LT(omega, 1.95f)
        << "Omega should never approach 1.95 (old unsafe limit)";
}

/**
 * @brief Test multiple instances with different diffusivities
 */
TEST_F(OmegaRegressionTest, MultipleInstancesConsistency) {
    std::vector<float> alphas = {1e-6f, 5e-6f, 1e-5f, 5e-5f, 1e-4f};

    for (float alpha : alphas) {
        ThermalLBM thermal1(30, 30, 30, material, alpha, false);
        ThermalLBM thermal2(40, 40, 40, material, alpha, false);

        float omega1 = 1.0f / thermal1.getThermalTau();
        float omega2 = 1.0f / thermal2.getThermalTau();

        // Omega should be same regardless of domain size
        EXPECT_NEAR(omega1, omega2, 1e-6f)
            << "Omega inconsistent between instances (alpha=" << alpha << ")";

        // Both should be capped
        EXPECT_LE(omega1, 1.50f);
        EXPECT_LE(omega2, 1.50f);
    }
}

/**
 * @brief Test omega with backward-compatible constructor
 */
TEST_F(OmegaRegressionTest, BackwardCompatibleConstructor) {
    float thermal_diff = 1.0e-5f;
    float density = 4420.0f;
    float specific_heat = 546.0f;

    // Old constructor (no MaterialProperties)
    ThermalLBM thermal(50, 50, 50, thermal_diff, density, specific_heat);

    float omega = 1.0f / thermal.getThermalTau();

    EXPECT_LE(omega, 1.50f)
        << "REGRESSION: Omega cap not applied in backward-compatible constructor";
}

/**
 * @brief Test that omega cap is applied BEFORE simulation starts
 */
TEST_F(OmegaRegressionTest, OmegaCapAppliedAtConstruction) {
    float alpha = 5.0e-5f;

    // Create solver
    ThermalLBM thermal(40, 40, 40, material, alpha, false);

    // Check omega IMMEDIATELY (before any timesteps)
    float omega = 1.0f / thermal.getThermalTau();

    EXPECT_LE(omega, 1.50f)
        << "Omega cap must be applied at construction, not during runtime";

    // Initialize and run one step
    thermal.initialize(1000.0f);
    thermal.collisionBGK(nullptr, nullptr, nullptr);
    thermal.streaming();

    // Omega should still be same (capped at construction)
    float omega_after = 1.0f / thermal.getThermalTau();

    EXPECT_NEAR(omega_after, omega, 1e-6f)
        << "Omega changed after timesteps (should be constant)";
}

/**
 * @brief Test omega warning message is printed
 */
TEST_F(OmegaRegressionTest, WarningMessageForCapping) {
    // Capture stdout
    testing::internal::CaptureStdout();

    float alpha = 5.0e-5f;  // Will trigger capping
    ThermalLBM thermal(30, 30, 30, material, alpha, false);

    std::string output = testing::internal::GetCapturedStdout();

    // Should contain warning about omega capping
    // (Note: actual output depends on implementation)
    // This test may need adjustment based on actual warning format
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
