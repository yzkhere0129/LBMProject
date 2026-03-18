/**
 * @file test_material_database_values.cu
 * @brief Regression test for material database value correctness
 *
 * UPDATED (2026-03-01):
 * cp_liquid corrected from 526 to 831 J/(kg·K) per Mills 2002.
 * The old value (526) was identical to cp_solid for walberla validation
 * consistency, but this is physically incorrect for Ti6Al4V liquid phase.
 *
 * TEST STRATEGY:
 * 1. Verify Ti6Al4V material properties match database values exactly
 * 2. Test that no silent value changes occur (regression prevention)
 * 3. Validate all materials have internally consistent properties
 * 4. Cross-check with literature values where applicable
 *
 * REFERENCE:
 * - Material database: src/physics/materials/material_database.cu
 * - Mills 2002: cp_liquid(Ti6Al4V) = 831 J/(kg·K)
 */

#include <gtest/gtest.h>
#include "physics/material_properties.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace lbm::physics;

/**
 * @brief Helper to check if two floats are approximately equal
 */
bool approxEqual(float a, float b, float tolerance = 1e-5f) {
    return std::abs(a - b) < tolerance;
}

/**
 * @brief Test 1: Ti6Al4V critical properties regression
 *
 * Validates that Ti6Al4V properties match EXACTLY the values that have been
 * validated against walberla and literature. Any deviation indicates regression.
 */
TEST(MaterialDatabaseValuesRegression, Ti6Al4V_CriticalProperties) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    std::cout << "\n=== Ti6Al4V Critical Properties Validation ===\n";

    // CRITICAL: Solid properties (walberla match)
    const float expected_rho_solid = 4430.0f;
    const float expected_cp_solid = 526.0f;
    const float expected_k_solid = 6.7f;

    std::cout << "\nSolid properties (walberla validated):\n";
    std::cout << "  ρ_solid: " << ti64.rho_solid << " kg/m³ (expected: "
              << expected_rho_solid << ")\n";
    std::cout << "  cp_solid: " << ti64.cp_solid << " J/(kg·K) (expected: "
              << expected_cp_solid << ")\n";
    std::cout << "  k_solid: " << ti64.k_solid << " W/(m·K) (expected: "
              << expected_k_solid << ")\n";

    EXPECT_FLOAT_EQ(ti64.rho_solid, expected_rho_solid)
        << "REGRESSION: Solid density changed from validated value!";
    EXPECT_FLOAT_EQ(ti64.cp_solid, expected_cp_solid)
        << "REGRESSION: Solid heat capacity changed from validated value!";
    EXPECT_FLOAT_EQ(ti64.k_solid, expected_k_solid)
        << "REGRESSION: Solid conductivity changed from validated value!";

    // CRITICAL: Liquid properties
    const float expected_rho_liquid = 4110.0f;
    const float expected_cp_liquid = 831.0f;  // Mills 2002 corrected value
    const float expected_k_liquid = 33.0f;

    std::cout << "\nLiquid properties:\n";
    std::cout << "  ρ_liquid: " << ti64.rho_liquid << " kg/m³ (expected: "
              << expected_rho_liquid << ")\n";
    std::cout << "  cp_liquid: " << ti64.cp_liquid << " J/(kg·K) (expected: "
              << expected_cp_liquid << ")\n";
    std::cout << "  k_liquid: " << ti64.k_liquid << " W/(m·K) (expected: "
              << expected_k_liquid << ")\n";

    EXPECT_FLOAT_EQ(ti64.rho_liquid, expected_rho_liquid)
        << "REGRESSION: Liquid density changed!";

    EXPECT_FLOAT_EQ(ti64.cp_liquid, expected_cp_liquid)
        << "REGRESSION: Liquid heat capacity changed! "
        << "Must remain 831 J/(kg·K) per Mills 2002.";

    EXPECT_FLOAT_EQ(ti64.k_liquid, expected_k_liquid)
        << "REGRESSION: Liquid conductivity changed!";

    // CRITICAL: Phase change temperatures
    const float expected_T_solidus = 1878.0f;
    const float expected_T_liquidus = 1923.0f;
    const float expected_T_vaporization = 3560.0f;

    std::cout << "\nPhase change temperatures:\n";
    std::cout << "  T_solidus: " << ti64.T_solidus << " K (expected: "
              << expected_T_solidus << ")\n";
    std::cout << "  T_liquidus: " << ti64.T_liquidus << " K (expected: "
              << expected_T_liquidus << ")\n";
    std::cout << "  T_vaporization: " << ti64.T_vaporization << " K (expected: "
              << expected_T_vaporization << ")\n";

    EXPECT_FLOAT_EQ(ti64.T_solidus, expected_T_solidus)
        << "REGRESSION: Solidus temperature changed!";
    EXPECT_FLOAT_EQ(ti64.T_liquidus, expected_T_liquidus)
        << "REGRESSION: Liquidus temperature changed!";
    EXPECT_FLOAT_EQ(ti64.T_vaporization, expected_T_vaporization)
        << "REGRESSION: Vaporization temperature changed!";

    // CRITICAL: Latent heats
    const float expected_L_fusion = 286000.0f;
    const float expected_L_vaporization = 9830000.0f;

    std::cout << "\nLatent heats:\n";
    std::cout << "  L_fusion: " << ti64.L_fusion << " J/kg (expected: "
              << expected_L_fusion << ")\n";
    std::cout << "  L_vaporization: " << ti64.L_vaporization << " J/kg (expected: "
              << expected_L_vaporization << ")\n";

    EXPECT_FLOAT_EQ(ti64.L_fusion, expected_L_fusion)
        << "REGRESSION: Fusion latent heat changed!";
    EXPECT_FLOAT_EQ(ti64.L_vaporization, expected_L_vaporization)
        << "REGRESSION: Vaporization latent heat changed!";

    // CRITICAL: Optical properties
    const float expected_absorptivity_solid = 0.35f;
    const float expected_absorptivity_liquid = 0.40f;
    const float expected_emissivity = 0.25f;

    std::cout << "\nOptical properties:\n";
    std::cout << "  α_solid: " << ti64.absorptivity_solid << " (expected: "
              << expected_absorptivity_solid << ")\n";
    std::cout << "  α_liquid: " << ti64.absorptivity_liquid << " (expected: "
              << expected_absorptivity_liquid << ")\n";
    std::cout << "  ε: " << ti64.emissivity << " (expected: "
              << expected_emissivity << ")\n";

    EXPECT_FLOAT_EQ(ti64.absorptivity_solid, expected_absorptivity_solid)
        << "REGRESSION: Solid absorptivity changed!";
    EXPECT_FLOAT_EQ(ti64.absorptivity_liquid, expected_absorptivity_liquid)
        << "REGRESSION: Liquid absorptivity changed!";
    EXPECT_FLOAT_EQ(ti64.emissivity, expected_emissivity)
        << "REGRESSION: Emissivity changed!";

    // CRITICAL: Surface tension (Mills 2002 validated value)
    const float expected_surface_tension = 1.65f;
    const float expected_dsigma_dT = -2.6e-4f;

    std::cout << "\nSurface tension:\n";
    std::cout << "  σ: " << ti64.surface_tension << " N/m (expected: "
              << expected_surface_tension << ")\n";
    std::cout << "  dσ/dT: " << ti64.dsigma_dT << " N/(m·K) (expected: "
              << expected_dsigma_dT << ")\n";

    EXPECT_FLOAT_EQ(ti64.surface_tension, expected_surface_tension)
        << "REGRESSION: Surface tension changed from Mills 2002 value!";
    EXPECT_FLOAT_EQ(ti64.dsigma_dT, expected_dsigma_dT)
        << "REGRESSION: Surface tension gradient changed from Mills 2002 value!";

    std::cout << "\n[PASS] All Ti6Al4V properties match validated values!\n";
}

/**
 * @brief Test 2: Specific heat consistency check
 *
 * Validates Ti6Al4V cp_solid=526, cp_liquid=831 (Mills 2002)
 * and tests getSpecificHeat() method at various temperatures.
 */
TEST(MaterialDatabaseValuesRegression, SpecificHeatConsistency) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    std::cout << "\n=== Specific Heat Consistency Test ===\n";

    // cp_solid=526, cp_liquid=831 (Mills 2002 corrected)
    std::cout << "cp_solid: " << ti64.cp_solid << " J/(kg·K)\n";
    std::cout << "cp_liquid: " << ti64.cp_liquid << " J/(kg·K)\n";

    EXPECT_FLOAT_EQ(ti64.cp_solid, 526.0f)
        << "REGRESSION: Solid heat capacity changed!";
    EXPECT_FLOAT_EQ(ti64.cp_liquid, 831.0f)
        << "REGRESSION: Liquid heat capacity changed from Mills 2002 value!";

    // Test getSpecificHeat() at different temperatures
    std::cout << "\nSpecific heat vs temperature:\n";
    std::cout << std::setw(12) << "T [K]"
              << std::setw(15) << "cp [J/(kg·K)]"
              << std::setw(12) << "Phase\n";
    std::cout << std::string(39, '-') << "\n";

    std::vector<float> test_temps = {
        300.0f,             // Solid
        1800.0f,            // Solid (near melting)
        ti64.T_solidus,     // Solidus
        1900.0f,            // Mushy zone
        ti64.T_liquidus,    // Liquidus
        2200.0f,            // Liquid
        3000.0f             // Hot liquid
    };

    for (float T : test_temps) {
        float cp = ti64.getSpecificHeat(T);
        std::string phase;
        if (T < ti64.T_solidus) phase = "solid";
        else if (T > ti64.T_liquidus) phase = "liquid";
        else phase = "mushy";

        std::cout << std::setw(12) << T
                  << std::setw(15) << cp
                  << std::setw(12) << phase << "\n";

        if (phase == "solid") {
            EXPECT_FLOAT_EQ(cp, 526.0f)
                << "Solid specific heat should be 526 J/(kg·K) at T=" << T << " K";
        } else if (phase == "liquid") {
            EXPECT_FLOAT_EQ(cp, 831.0f)
                << "Liquid specific heat should be 831 J/(kg·K) at T=" << T << " K";
        }
    }

    std::cout << "\n[PASS] Specific heat consistency verified!\n";
}

/**
 * @brief Test 3: Thermal diffusivity calculation
 *
 * Validates α = k / (ρ·cp) calculation and consistency
 */
TEST(MaterialDatabaseValuesRegression, ThermalDiffusivityCalculation) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    std::cout << "\n=== Thermal Diffusivity Calculation Test ===\n";

    // Test at solid and liquid states
    struct TestCase {
        float T;
        std::string phase;
        float expected_rho;
        float expected_k;
        float expected_cp;
    };

    std::vector<TestCase> cases = {
        {300.0f, "solid", 4430.0f, 6.7f, 526.0f},
        {2200.0f, "liquid", 4110.0f, 33.0f, 831.0f}
    };

    std::cout << std::setw(12) << "T [K]"
              << std::setw(10) << "Phase"
              << std::setw(15) << "α [m²/s]"
              << std::setw(20) << "α_expected [m²/s]"
              << std::setw(12) << "Match\n";
    std::cout << std::string(69, '-') << "\n";

    for (const auto& tc : cases) {
        float alpha = ti64.getThermalDiffusivity(tc.T);
        float alpha_expected = tc.expected_k / (tc.expected_rho * tc.expected_cp);

        bool match = approxEqual(alpha, alpha_expected, 1e-9f);

        std::cout << std::setw(12) << tc.T
                  << std::setw(10) << tc.phase
                  << std::setw(15) << std::scientific << alpha
                  << std::setw(20) << alpha_expected
                  << std::setw(12) << (match ? "YES" : "NO") << "\n";

        EXPECT_NEAR(alpha, alpha_expected, 1e-9f)
            << "Thermal diffusivity calculation incorrect at T=" << tc.T << " K";
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[PASS] Thermal diffusivity calculations correct!\n";
}

/**
 * @brief Test 4: Cross-material validation
 *
 * Ensures all materials have valid and distinct properties
 */
TEST(MaterialDatabaseValuesRegression, CrossMaterialValidation) {
    std::cout << "\n=== Cross-Material Validation ===\n";

    std::vector<std::pair<std::string, MaterialProperties>> materials = {
        {"Ti6Al4V", MaterialDatabase::getTi6Al4V()},
        {"SS316L", MaterialDatabase::get316L()},
        {"IN718", MaterialDatabase::getInconel718()},
        {"AlSi10Mg", MaterialDatabase::getAlSi10Mg()}
    };

    // Create CSV for comparison
    std::ofstream csv("material_properties_regression.csv");
    csv << "Material,rho_solid,cp_solid,k_solid,rho_liquid,cp_liquid,k_liquid,"
        << "T_solidus,T_liquidus,T_vaporization,L_fusion,L_vaporization,"
        << "abs_solid,abs_liquid,emissivity,sigma,dsigma_dT\n";

    std::cout << "\n" << std::setw(12) << "Material"
              << std::setw(12) << "ρ_s [kg/m³]"
              << std::setw(12) << "cp_s [J/kg·K]"
              << std::setw(12) << "T_liq [K]"
              << std::setw(12) << "σ [N/m]\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& [name, mat] : materials) {
        std::cout << std::setw(12) << name
                  << std::setw(12) << mat.rho_solid
                  << std::setw(12) << mat.cp_solid
                  << std::setw(12) << mat.T_liquidus
                  << std::setw(12) << mat.surface_tension << "\n";

        // Write to CSV
        csv << name << ","
            << mat.rho_solid << "," << mat.cp_solid << "," << mat.k_solid << ","
            << mat.rho_liquid << "," << mat.cp_liquid << "," << mat.k_liquid << ","
            << mat.T_solidus << "," << mat.T_liquidus << "," << mat.T_vaporization << ","
            << mat.L_fusion << "," << mat.L_vaporization << ","
            << mat.absorptivity_solid << "," << mat.absorptivity_liquid << "," << mat.emissivity << ","
            << mat.surface_tension << "," << mat.dsigma_dT << "\n";

        // Validate each material
        EXPECT_TRUE(mat.validate())
            << "Material " << name << " failed validation!";

        // Check physical constraints
        EXPECT_GT(mat.rho_solid, 0.0f) << name << ": Invalid density";
        EXPECT_GT(mat.cp_solid, 0.0f) << name << ": Invalid heat capacity";
        EXPECT_GT(mat.k_solid, 0.0f) << name << ": Invalid conductivity";
        EXPECT_LT(mat.T_solidus, mat.T_liquidus) << name << ": Invalid melting range";
        EXPECT_LT(mat.T_liquidus, mat.T_vaporization) << name << ": Invalid boiling point";
        EXPECT_GT(mat.surface_tension, 0.0f) << name << ": Invalid surface tension";
        EXPECT_LT(mat.dsigma_dT, 0.0f) << name << ": Surface tension should decrease with T";
    }

    csv.close();

    std::cout << "\n[PASS] All materials validated!\n";
    std::cout << "Material comparison saved to: material_properties_regression.csv\n";
}

/**
 * @brief Test 5: Solid-phase walberla validation consistency
 *
 * Validates that SOLID-phase properties still match walberla reference values.
 * Note: cp_liquid was corrected to 831 (Mills 2002) and no longer equals cp_solid.
 * Walberla validation uses solid-phase diffusivity, so this remains consistent.
 */
TEST(MaterialDatabaseValuesRegression, WalberlaValidationConsistency) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    std::cout << "\n=== Walberla Validation Consistency ===\n";

    // Walberla solid-phase reference values
    const float walberla_rho = 4430.0f;
    const float walberla_cp_solid = 526.0f;
    const float walberla_k = 6.7f;

    std::cout << "Walberla validated solid properties:\n";
    std::cout << "  ρ = " << walberla_rho << " kg/m³\n";
    std::cout << "  cp_solid = " << walberla_cp_solid << " J/(kg·K)\n";
    std::cout << "  k = " << walberla_k << " W/(m·K)\n";

    std::cout << "\nLBMProject Ti6Al4V properties:\n";
    std::cout << "  ρ_solid = " << ti64.rho_solid << " kg/m³\n";
    std::cout << "  cp_solid = " << ti64.cp_solid << " J/(kg·K)\n";
    std::cout << "  cp_liquid = " << ti64.cp_liquid << " J/(kg·K) (Mills 2002)\n";
    std::cout << "  k_solid = " << ti64.k_solid << " W/(m·K)\n";

    // Solid-phase properties must match walberla
    EXPECT_FLOAT_EQ(ti64.rho_solid, walberla_rho)
        << "REGRESSION: Density no longer matches walberla!";
    EXPECT_FLOAT_EQ(ti64.cp_solid, walberla_cp_solid)
        << "REGRESSION: Solid cp no longer matches walberla!";
    EXPECT_FLOAT_EQ(ti64.k_solid, walberla_k)
        << "REGRESSION: Conductivity no longer matches walberla!";

    // cp_liquid is now 831 (Mills 2002), distinct from cp_solid
    EXPECT_FLOAT_EQ(ti64.cp_liquid, 831.0f)
        << "REGRESSION: cp_liquid must be 831 J/(kg·K) per Mills 2002.";

    // Compute solid-phase thermal diffusivity (walberla validation uses this)
    float alpha_solid = ti64.k_solid / (ti64.rho_solid * ti64.cp_solid);
    float alpha_walberla = walberla_k / (walberla_rho * walberla_cp_solid);

    std::cout << "\nSolid-phase thermal diffusivity:\n";
    std::cout << "  α_LBMProject = " << alpha_solid << " m²/s\n";
    std::cout << "  α_walberla = " << alpha_walberla << " m²/s\n";
    std::cout << "  Difference: " << std::abs(alpha_solid - alpha_walberla) * 100 / alpha_walberla << "%\n";

    EXPECT_NEAR(alpha_solid, alpha_walberla, 1e-9f)
        << "REGRESSION: Solid thermal diffusivity no longer matches walberla!";

    // Also verify liquid-phase thermal diffusivity uses corrected cp_liquid
    float alpha_liquid = ti64.k_liquid / (ti64.rho_liquid * ti64.cp_liquid);
    float alpha_liquid_expected = 33.0f / (4110.0f * 831.0f);
    EXPECT_NEAR(alpha_liquid, alpha_liquid_expected, 1e-9f)
        << "Liquid thermal diffusivity must use cp_liquid=831!";

    std::cout << "\n[PASS] Walberla validation consistency maintained!\n";
}

/**
 * @brief Test 6: Property value range checks
 *
 * Ensures all property values are in physically realistic ranges
 */
TEST(MaterialDatabaseValuesRegression, PropertyValueRanges) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    std::cout << "\n=== Property Value Range Checks ===\n";

    // Density ranges (kg/m³)
    EXPECT_GT(ti64.rho_solid, 1000.0f) << "Ti alloy density too low";
    EXPECT_LT(ti64.rho_solid, 10000.0f) << "Ti alloy density too high";
    EXPECT_GT(ti64.rho_liquid, 1000.0f) << "Liquid Ti density too low";
    EXPECT_LT(ti64.rho_liquid, ti64.rho_solid) << "Liquid should be less dense than solid (typical)";

    // Specific heat ranges (J/(kg·K))
    EXPECT_GT(ti64.cp_solid, 100.0f) << "Specific heat too low";
    EXPECT_LT(ti64.cp_solid, 2000.0f) << "Specific heat too high";

    // Thermal conductivity ranges (W/(m·K))
    EXPECT_GT(ti64.k_solid, 1.0f) << "Thermal conductivity too low";
    EXPECT_LT(ti64.k_solid, 500.0f) << "Thermal conductivity too high";
    EXPECT_GT(ti64.k_liquid, ti64.k_solid) << "Liquid should have higher conductivity (typical)";

    // Temperature ranges (K)
    EXPECT_GT(ti64.T_solidus, 1000.0f) << "Melting point too low for Ti";
    EXPECT_LT(ti64.T_solidus, 3000.0f) << "Melting point too high";
    EXPECT_GT(ti64.T_vaporization, ti64.T_liquidus + 1000.0f) << "Boiling point too close to melting";

    // Optical property ranges
    EXPECT_GT(ti64.absorptivity_solid, 0.0f) << "Absorptivity must be positive";
    EXPECT_LE(ti64.absorptivity_solid, 1.0f) << "Absorptivity cannot exceed 1.0";
    EXPECT_GT(ti64.absorptivity_liquid, ti64.absorptivity_solid) << "Liquid typically more absorptive";

    // Surface tension ranges (N/m)
    EXPECT_GT(ti64.surface_tension, 0.5f) << "Surface tension too low";
    EXPECT_LT(ti64.surface_tension, 3.0f) << "Surface tension too high";

    std::cout << "[PASS] All property values in physically realistic ranges!\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
