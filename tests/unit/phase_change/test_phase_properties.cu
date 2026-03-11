/**
 * @file test_phase_properties.cu
 * @brief Unit tests for phase-dependent material properties
 *
 * Tests verify:
 * 1. Density ρ(T) transitions smoothly from solid to liquid
 * 2. Specific heat cp(T) interpolates correctly
 * 3. Thermal conductivity k(T) varies with phase
 * 4. Dynamic viscosity μ(T) transitions from rigid solid to liquid
 * 5. Property continuity at phase boundaries
 */

#include <gtest/gtest.h>
#include "physics/material_properties.h"
#include <cmath>

using namespace lbm::physics;

class PhasePropertiesTest : public ::testing::Test {
protected:
    MaterialProperties mat;

    void SetUp() override {
        mat = MaterialDatabase::getTi6Al4V();
    }
};

/**
 * Test: Density transitions from solid to liquid
 */
TEST_F(PhasePropertiesTest, DensityTransition) {
    // Solid phase
    float T_solid = 1500.0f;
    float rho_solid = mat.getDensity(T_solid);
    EXPECT_FLOAT_EQ(rho_solid, mat.rho_solid)
        << "In solid phase, density should equal rho_solid";

    // Liquid phase
    float T_liquid = 2200.0f;
    float rho_liquid = mat.getDensity(T_liquid);
    EXPECT_FLOAT_EQ(rho_liquid, mat.rho_liquid)
        << "In liquid phase, density should equal rho_liquid";

    // Mushy zone - midpoint should be average
    float T_mushy = (mat.T_solidus + mat.T_liquidus) / 2.0f;
    float rho_mushy = mat.getDensity(T_mushy);
    float rho_avg = (mat.rho_solid + mat.rho_liquid) / 2.0f;
    EXPECT_NEAR(rho_mushy, rho_avg, 10.0f)
        << "In mushy zone midpoint, density should be approximately average";

    // Density should be monotonic (decreasing with T for most metals)
    EXPECT_GT(rho_solid, rho_liquid)
        << "Solid density should be greater than liquid density for Ti6Al4V";
}

/**
 * Test: Specific heat transitions from solid to liquid
 */
TEST_F(PhasePropertiesTest, SpecificHeatTransition) {
    // Solid phase
    float T_solid = 1500.0f;
    float cp_solid = mat.getSpecificHeat(T_solid);
    EXPECT_FLOAT_EQ(cp_solid, mat.cp_solid)
        << "In solid phase, cp should equal cp_solid";

    // Liquid phase
    float T_liquid = 2200.0f;
    float cp_liquid = mat.getSpecificHeat(T_liquid);
    EXPECT_FLOAT_EQ(cp_liquid, mat.cp_liquid)
        << "In liquid phase, cp should equal cp_liquid";

    // Mushy zone - should interpolate
    float T_mushy = (mat.T_solidus + mat.T_liquidus) / 2.0f;
    float cp_mushy = mat.getSpecificHeat(T_mushy);
    float cp_avg = (mat.cp_solid + mat.cp_liquid) / 2.0f;
    EXPECT_NEAR(cp_mushy, cp_avg, 10.0f)
        << "In mushy zone midpoint, cp should be approximately average";
}

/**
 * Test: Thermal conductivity transitions from solid to liquid
 */
TEST_F(PhasePropertiesTest, ThermalConductivityTransition) {
    // Solid phase
    float T_solid = 1500.0f;
    float k_solid = mat.getThermalConductivity(T_solid);
    EXPECT_FLOAT_EQ(k_solid, mat.k_solid)
        << "In solid phase, k should equal k_solid";

    // Liquid phase
    float T_liquid = 2200.0f;
    float k_liquid = mat.getThermalConductivity(T_liquid);
    EXPECT_FLOAT_EQ(k_liquid, mat.k_liquid)
        << "In liquid phase, k should equal k_liquid";

    // Mushy zone - should interpolate
    float T_mushy = (mat.T_solidus + mat.T_liquidus) / 2.0f;
    float k_mushy = mat.getThermalConductivity(T_mushy);
    float k_avg = (mat.k_solid + mat.k_liquid) / 2.0f;
    EXPECT_NEAR(k_mushy, k_avg, 1.0f)
        << "In mushy zone midpoint, k should be approximately average";
}

/**
 * Test: Dynamic viscosity transitions from rigid to liquid
 */
TEST_F(PhasePropertiesTest, ViscosityTransition) {
    // Solid phase - should be very high (effectively rigid)
    float T_solid = 1500.0f;
    float mu_solid = mat.getDynamicViscosity(T_solid);
    EXPECT_GT(mu_solid, 1e9f)
        << "Solid phase viscosity should be very high (> 1e9 Pa·s)";

    // Liquid phase - should be reasonable for molten metal
    float T_liquid = 2200.0f;
    float mu_liquid = mat.getDynamicViscosity(T_liquid);
    EXPECT_FLOAT_EQ(mu_liquid, mat.mu_liquid)
        << "Liquid phase viscosity should equal mu_liquid";
    EXPECT_LT(mu_liquid, 0.1f)
        << "Liquid metal viscosity should be less than 0.1 Pa·s";

    // Mushy zone - should transition exponentially
    float T_mushy = (mat.T_solidus + mat.T_liquidus) / 2.0f;
    float mu_mushy = mat.getDynamicViscosity(T_mushy);
    EXPECT_GT(mu_mushy, mat.mu_liquid)
        << "Mushy zone viscosity should be higher than liquid";
    EXPECT_LT(mu_mushy, 1e9f)
        << "Mushy zone viscosity should be lower than solid";
}

/**
 * Test: Property continuity at solidus
 */
TEST_F(PhasePropertiesTest, ContinuityAtSolidus) {
    float epsilon = 0.01f;
    float T_below = mat.T_solidus - epsilon;
    float T_at = mat.T_solidus;
    float T_above = mat.T_solidus + epsilon;

    // Density should be continuous
    {
        float rho_below = mat.getDensity(T_below);
        float rho_at = mat.getDensity(T_at);
        float rho_above = mat.getDensity(T_above);

        EXPECT_NEAR(rho_below, rho_at, 1.0f);
        EXPECT_NEAR(rho_at, rho_above, 1.0f);
    }

    // Specific heat should be continuous
    {
        float cp_below = mat.getSpecificHeat(T_below);
        float cp_at = mat.getSpecificHeat(T_at);
        float cp_above = mat.getSpecificHeat(T_above);

        EXPECT_NEAR(cp_below, cp_at, 1.0f);
        EXPECT_NEAR(cp_at, cp_above, 1.0f);
    }

    // Thermal conductivity should be continuous
    {
        float k_below = mat.getThermalConductivity(T_below);
        float k_at = mat.getThermalConductivity(T_at);
        float k_above = mat.getThermalConductivity(T_above);

        EXPECT_NEAR(k_below, k_at, 0.1f);
        EXPECT_NEAR(k_at, k_above, 0.1f);
    }
}

/**
 * Test: Property continuity at liquidus
 */
TEST_F(PhasePropertiesTest, ContinuityAtLiquidus) {
    float epsilon = 0.01f;
    float T_below = mat.T_liquidus - epsilon;
    float T_at = mat.T_liquidus;
    float T_above = mat.T_liquidus + epsilon;

    // Density should be continuous
    {
        float rho_below = mat.getDensity(T_below);
        float rho_at = mat.getDensity(T_at);
        float rho_above = mat.getDensity(T_above);

        EXPECT_NEAR(rho_below, rho_at, 1.0f);
        EXPECT_NEAR(rho_at, rho_above, 1.0f);
    }

    // Specific heat should be continuous
    {
        float cp_below = mat.getSpecificHeat(T_below);
        float cp_at = mat.getSpecificHeat(T_at);
        float cp_above = mat.getSpecificHeat(T_above);

        EXPECT_NEAR(cp_below, cp_at, 1.0f);
        EXPECT_NEAR(cp_at, cp_above, 1.0f);
    }

    // Thermal conductivity should be continuous
    {
        float k_below = mat.getThermalConductivity(T_below);
        float k_at = mat.getThermalConductivity(T_at);
        float k_above = mat.getThermalConductivity(T_above);

        EXPECT_NEAR(k_below, k_at, 0.1f);
        EXPECT_NEAR(k_at, k_above, 0.1f);
    }
}

/**
 * Test: Thermal diffusivity calculation
 *
 * α = k/(ρ·cp)
 */
TEST_F(PhasePropertiesTest, ThermalDiffusivity) {
    float test_temps[] = {1500.0f, 1900.0f, 2200.0f};

    for (float T : test_temps) {
        float alpha = mat.getThermalDiffusivity(T);
        float k = mat.getThermalConductivity(T);
        float rho = mat.getDensity(T);
        float cp = mat.getSpecificHeat(T);

        float alpha_expected = k / (rho * cp);

        EXPECT_NEAR(alpha, alpha_expected, 1e-9f)
            << "Thermal diffusivity calculation error at T = " << T << " K"
            << "\nα = " << alpha << " m²/s"
            << "\nExpected: k/(ρ·cp) = " << alpha_expected << " m²/s";

        // Check reasonable values
        EXPECT_GT(alpha, 1e-7f) << "α too small at T = " << T;
        EXPECT_LT(alpha, 1e-3f) << "α too large at T = " << T;
    }
}

/**
 * Test: Effective heat capacity in mushy zone
 *
 * cp_eff = cp + L·(dfl/dT)
 */
TEST_F(PhasePropertiesTest, EffectiveHeatCapacity) {
    // In solid or liquid: cp_eff = cp
    {
        float T_solid = 1500.0f;
        float cp_eff = mat.getApparentHeatCapacity(T_solid);
        float cp = mat.getSpecificHeat(T_solid);
        EXPECT_FLOAT_EQ(cp_eff, cp)
            << "In solid, effective heat capacity should equal cp";
    }

    // In mushy zone: cp_eff > cp due to latent heat
    {
        float T_mushy = (mat.T_solidus + mat.T_liquidus) / 2.0f;
        float cp_eff = mat.getApparentHeatCapacity(T_mushy);
        float cp = mat.getSpecificHeat(T_mushy);

        EXPECT_GT(cp_eff, cp)
            << "In mushy zone, effective heat capacity should exceed cp";

        // Expected: cp_eff = cp + L/(T_liquidus - T_solidus)
        float dfl_dT = 1.0f / (mat.T_liquidus - mat.T_solidus);
        float cp_eff_expected = cp + mat.L_fusion * dfl_dT;

        EXPECT_NEAR(cp_eff, cp_eff_expected, 100.0f)
            << "Effective heat capacity calculation mismatch in mushy zone";
    }
}

/**
 * Test: All properties are positive
 */
TEST_F(PhasePropertiesTest, PropertiesArePositive) {
    // Test at various temperatures
    for (float T = 300.0f; T < 3000.0f; T += 100.0f) {
        EXPECT_GT(mat.getDensity(T), 0.0f)
            << "Density must be positive at T = " << T;
        EXPECT_GT(mat.getSpecificHeat(T), 0.0f)
            << "Specific heat must be positive at T = " << T;
        EXPECT_GT(mat.getThermalConductivity(T), 0.0f)
            << "Thermal conductivity must be positive at T = " << T;
        EXPECT_GT(mat.getDynamicViscosity(T), 0.0f)
            << "Viscosity must be positive at T = " << T;
        EXPECT_GT(mat.getThermalDiffusivity(T), 0.0f)
            << "Thermal diffusivity must be positive at T = " << T;
    }
}

/**
 * Test: Material validation
 */
TEST_F(PhasePropertiesTest, MaterialValidation) {
    EXPECT_TRUE(mat.validate())
        << "Ti6Al4V material properties should be valid";

    // Test other materials
    MaterialProperties ss316l = MaterialDatabase::get316L();
    EXPECT_TRUE(ss316l.validate())
        << "316L material properties should be valid";

    MaterialProperties in718 = MaterialDatabase::getInconel718();
    EXPECT_TRUE(in718.validate())
        << "IN718 material properties should be valid";

    MaterialProperties alsi10mg = MaterialDatabase::getAlSi10Mg();
    EXPECT_TRUE(alsi10mg.validate())
        << "AlSi10Mg material properties should be valid";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
