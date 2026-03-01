/**
 * @file test_materials.cu
 * @brief Unit tests for material properties database
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/material_properties.h"
#include <vector>
#include <cmath>

using namespace lbm::physics;

class MaterialPropertiesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaSetDevice(0);
    }

    void TearDown() override {
        // Clean up CUDA
        cudaDeviceReset();
    }

    /**
     * Helper function to check if two floats are approximately equal
     */
    bool approxEqual(float a, float b, float tolerance = 1e-5f) {
        return std::abs(a - b) < tolerance;
    }
};

/**
 * Test 1: Basic property access for Ti6Al4V
 */
TEST_F(MaterialPropertiesTest, Ti6Al4VBasicProperties) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Check basic solid properties (updated to match walberla)
    EXPECT_FLOAT_EQ(ti64.rho_solid, 4430.0f);  // kg/m³ (walberla)
    EXPECT_FLOAT_EQ(ti64.cp_solid, 526.0f);   // J/(kg·K) (walberla)
    EXPECT_FLOAT_EQ(ti64.k_solid, 6.7f);      // W/(m·K) (walberla)

    // Check liquid properties (cp_liquid uses Mills 2002 value)
    EXPECT_FLOAT_EQ(ti64.rho_liquid, 4110.0f);
    EXPECT_FLOAT_EQ(ti64.cp_liquid, 831.0f);  // J/(kg·K) - Mills 2002
    EXPECT_FLOAT_EQ(ti64.k_liquid, 33.0f);

    // Check phase change parameters
    EXPECT_FLOAT_EQ(ti64.T_solidus, 1878.0f);
    EXPECT_FLOAT_EQ(ti64.T_liquidus, 1923.0f);
    EXPECT_FLOAT_EQ(ti64.T_vaporization, 3560.0f);
    EXPECT_FLOAT_EQ(ti64.L_fusion, 286000.0f);
    EXPECT_FLOAT_EQ(ti64.L_vaporization, 9830000.0f);

    // Check optical properties
    EXPECT_FLOAT_EQ(ti64.absorptivity_solid, 0.35f);
    EXPECT_FLOAT_EQ(ti64.absorptivity_liquid, 0.40f);
    EXPECT_FLOAT_EQ(ti64.emissivity, 0.25f);

    // Check surface properties (CORRECTED VALUE from Mills 2002)
    EXPECT_FLOAT_EQ(ti64.surface_tension, 1.65f);
    EXPECT_FLOAT_EQ(ti64.dsigma_dT, -2.6e-4f);
}

/**
 * Test 2: Temperature-dependent thermal conductivity
 */
TEST_F(MaterialPropertiesTest, TemperatureDependentThermalConductivity) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Solid state (well below solidus)
    float k_300K = ti64.getThermalConductivity(300.0f);
    EXPECT_FLOAT_EQ(k_300K, ti64.k_solid);

    // Liquid state (well above liquidus)
    float k_2200K = ti64.getThermalConductivity(2200.0f);
    EXPECT_FLOAT_EQ(k_2200K, ti64.k_liquid);

    // Mushy zone - should be between solid and liquid values
    float k_1900K = ti64.getThermalConductivity(1900.0f);  // In mushy zone
    EXPECT_GT(k_1900K, ti64.k_solid);
    EXPECT_LT(k_1900K, ti64.k_liquid);

    // At solidus temperature
    float k_solidus = ti64.getThermalConductivity(ti64.T_solidus);
    EXPECT_FLOAT_EQ(k_solidus, ti64.k_solid);

    // At liquidus temperature
    float k_liquidus = ti64.getThermalConductivity(ti64.T_liquidus);
    EXPECT_FLOAT_EQ(k_liquidus, ti64.k_liquid);
}

/**
 * Test 3: Liquid fraction calculation
 */
TEST_F(MaterialPropertiesTest, LiquidFractionCalculation) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Below solidus - fully solid
    EXPECT_FLOAT_EQ(ti64.liquidFraction(ti64.T_solidus - 100.0f), 0.0f);
    EXPECT_FLOAT_EQ(ti64.liquidFraction(1800.0f), 0.0f);

    // Above liquidus - fully liquid
    EXPECT_FLOAT_EQ(ti64.liquidFraction(ti64.T_liquidus + 100.0f), 1.0f);
    EXPECT_FLOAT_EQ(ti64.liquidFraction(2000.0f), 1.0f);

    // At solidus - just starting to melt
    EXPECT_FLOAT_EQ(ti64.liquidFraction(ti64.T_solidus), 0.0f);

    // At liquidus - fully melted
    EXPECT_FLOAT_EQ(ti64.liquidFraction(ti64.T_liquidus), 1.0f);

    // Middle of mushy zone - should be between 0 and 1
    float T_middle = (ti64.T_solidus + ti64.T_liquidus) / 2.0f;
    float fl_middle = ti64.liquidFraction(T_middle);
    EXPECT_NEAR(fl_middle, 0.5f, 0.01f);

    // Check monotonic increase in mushy zone
    float fl_25 = ti64.liquidFraction(ti64.T_solidus + 0.25f * (ti64.T_liquidus - ti64.T_solidus));
    float fl_75 = ti64.liquidFraction(ti64.T_solidus + 0.75f * (ti64.T_liquidus - ti64.T_solidus));
    EXPECT_NEAR(fl_25, 0.25f, 0.01f);
    EXPECT_NEAR(fl_75, 0.75f, 0.01f);
}

/**
 * Test 4: Surface tension temperature dependence
 */
TEST_F(MaterialPropertiesTest, SurfaceTensionTemperatureDependence) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // At melting point
    float sigma_Tm = ti64.getSurfaceTension(ti64.T_liquidus);
    EXPECT_FLOAT_EQ(sigma_Tm, ti64.surface_tension);

    // Temperature increase should decrease surface tension (negative gradient)
    float sigma_Tm_plus_100 = ti64.getSurfaceTension(ti64.T_liquidus + 100.0f);
    float expected = ti64.surface_tension + ti64.dsigma_dT * 100.0f;
    EXPECT_NEAR(sigma_Tm_plus_100, expected, 1e-6f);
    EXPECT_LT(sigma_Tm_plus_100, sigma_Tm);  // Should decrease

    // Temperature decrease should increase surface tension
    float sigma_Tm_minus_100 = ti64.getSurfaceTension(ti64.T_liquidus - 100.0f);
    expected = ti64.surface_tension + ti64.dsigma_dT * (-100.0f);
    EXPECT_NEAR(sigma_Tm_minus_100, expected, 1e-6f);
    EXPECT_GT(sigma_Tm_minus_100, sigma_Tm);  // Should increase
}

/**
 * Test 5: Thermal diffusivity consistency
 */
TEST_F(MaterialPropertiesTest, ThermalDiffusivityConsistency) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Test at various temperatures
    std::vector<float> test_temps = {300.0f, 1500.0f, ti64.T_solidus, ti64.T_liquidus, 2200.0f};

    for (float T : test_temps) {
        float alpha = ti64.getThermalDiffusivity(T);
        float k = ti64.getThermalConductivity(T);
        float rho = ti64.getDensity(T);
        float cp = ti64.getSpecificHeat(T);

        // α = k / (ρ * cp)
        float expected_alpha = k / (rho * cp);
        EXPECT_NEAR(alpha, expected_alpha, 1e-9f) << "Failed at T = " << T << " K";
    }
}

/**
 * Test 6: Material validation
 */
TEST_F(MaterialPropertiesTest, MaterialValidation) {
    // Test valid materials
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();
    EXPECT_TRUE(ti64.validate());

    MaterialProperties ss316l = MaterialDatabase::get316L();
    EXPECT_TRUE(ss316l.validate());

    MaterialProperties in718 = MaterialDatabase::getInconel718();
    EXPECT_TRUE(in718.validate());

    MaterialProperties alsi10mg = MaterialDatabase::getAlSi10Mg();
    EXPECT_TRUE(alsi10mg.validate());

    // Test invalid material
    MaterialProperties invalid;
    invalid.rho_solid = -1.0f;  // Invalid negative density
    EXPECT_FALSE(invalid.validate());

    invalid = ti64;
    invalid.T_liquidus = invalid.T_solidus - 10.0f;  // Invalid temperature ordering
    EXPECT_FALSE(invalid.validate());

    invalid = ti64;
    invalid.absorptivity_solid = 1.5f;  // Invalid absorptivity > 1
    EXPECT_FALSE(invalid.validate());
}

/**
 * Test 7: Multiple materials distinction
 */
TEST_F(MaterialPropertiesTest, MultipleMaterialsDistinct) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();
    MaterialProperties ss316l = MaterialDatabase::get316L();
    MaterialProperties in718 = MaterialDatabase::getInconel718();
    MaterialProperties alsi10mg = MaterialDatabase::getAlSi10Mg();

    // Different materials should have different melting points
    EXPECT_NE(ti64.T_liquidus, ss316l.T_liquidus);
    EXPECT_NE(ti64.T_liquidus, in718.T_liquidus);
    EXPECT_NE(ti64.T_liquidus, alsi10mg.T_liquidus);

    // Ti6Al4V has highest melting point among these
    EXPECT_GT(ti64.T_liquidus, ss316l.T_liquidus);
    EXPECT_GT(ti64.T_liquidus, in718.T_liquidus);
    EXPECT_GT(ti64.T_liquidus, alsi10mg.T_liquidus);

    // AlSi10Mg has lowest melting point
    EXPECT_LT(alsi10mg.T_liquidus, ti64.T_liquidus);
    EXPECT_LT(alsi10mg.T_liquidus, ss316l.T_liquidus);
    EXPECT_LT(alsi10mg.T_liquidus, in718.T_liquidus);

    // Check densities are different
    EXPECT_NE(ti64.rho_solid, ss316l.rho_solid);
    EXPECT_NE(ti64.rho_solid, in718.rho_solid);
    EXPECT_NE(ti64.rho_solid, alsi10mg.rho_solid);

    // AlSi10Mg has much lower absorptivity (highly reflective)
    EXPECT_LT(alsi10mg.absorptivity_solid, ti64.absorptivity_solid);
    EXPECT_LT(alsi10mg.absorptivity_solid, ss316l.absorptivity_solid);
}

/**
 * Test 8: Temperature-dependent density
 */
TEST_F(MaterialPropertiesTest, TemperatureDependentDensity) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Solid state
    float rho_300K = ti64.getDensity(300.0f);
    EXPECT_FLOAT_EQ(rho_300K, ti64.rho_solid);

    // Liquid state
    float rho_2200K = ti64.getDensity(2200.0f);
    EXPECT_FLOAT_EQ(rho_2200K, ti64.rho_liquid);

    // Mushy zone - should interpolate
    float T_middle = (ti64.T_solidus + ti64.T_liquidus) / 2.0f;
    float rho_middle = ti64.getDensity(T_middle);
    float expected_rho = (ti64.rho_solid + ti64.rho_liquid) / 2.0f;
    EXPECT_NEAR(rho_middle, expected_rho, 1.0f);

    // Density should decrease from solid to liquid (typical for metals)
    EXPECT_LT(ti64.rho_liquid, ti64.rho_solid);
}

/**
 * Test 9: Dynamic viscosity behavior
 */
TEST_F(MaterialPropertiesTest, DynamicViscosityBehavior) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Solid state - should return very high viscosity
    float mu_solid = ti64.getDynamicViscosity(300.0f);
    EXPECT_GT(mu_solid, 1e9f);  // Effectively rigid

    // Liquid state - should return liquid viscosity
    float mu_liquid = ti64.getDynamicViscosity(2200.0f);
    EXPECT_FLOAT_EQ(mu_liquid, ti64.mu_liquid);

    // Mushy zone - should be between liquid and solid values
    float mu_mushy = ti64.getDynamicViscosity((ti64.T_solidus + ti64.T_liquidus) / 2.0f);
    EXPECT_GT(mu_mushy, ti64.mu_liquid);
    EXPECT_LT(mu_mushy, 1e9f);

    // Near solidus - viscosity should be very high
    float mu_near_solidus = ti64.getDynamicViscosity(ti64.T_solidus + 1.0f);
    EXPECT_GT(mu_near_solidus, 100.0f * ti64.mu_liquid);
}

/**
 * Test 10: Absorptivity temperature dependence
 */
TEST_F(MaterialPropertiesTest, AbsorptivityTemperatureDependence) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Solid state
    float abs_solid = ti64.getAbsorptivity(300.0f);
    EXPECT_FLOAT_EQ(abs_solid, ti64.absorptivity_solid);

    // Liquid state
    float abs_liquid = ti64.getAbsorptivity(2200.0f);
    EXPECT_FLOAT_EQ(abs_liquid, ti64.absorptivity_liquid);

    // Mushy zone - should interpolate
    float T_middle = (ti64.T_solidus + ti64.T_liquidus) / 2.0f;
    float abs_middle = ti64.getAbsorptivity(T_middle);
    float expected_abs = (ti64.absorptivity_solid + ti64.absorptivity_liquid) / 2.0f;
    EXPECT_NEAR(abs_middle, expected_abs, 0.01f);

    // Liquid typically has higher absorptivity
    EXPECT_GT(ti64.absorptivity_liquid, ti64.absorptivity_solid);
}

/**
 * Test 11: Phase state functions
 */
TEST_F(MaterialPropertiesTest, PhaseStateFunctions) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Test solid state
    EXPECT_TRUE(ti64.isSolid(300.0f));
    EXPECT_TRUE(ti64.isSolid(ti64.T_solidus - 1.0f));
    EXPECT_FALSE(ti64.isLiquid(300.0f));
    EXPECT_FALSE(ti64.isMushy(300.0f));

    // Test liquid state
    EXPECT_TRUE(ti64.isLiquid(2200.0f));
    EXPECT_TRUE(ti64.isLiquid(ti64.T_liquidus + 1.0f));
    EXPECT_FALSE(ti64.isSolid(2200.0f));
    EXPECT_FALSE(ti64.isMushy(2200.0f));

    // Test mushy zone
    float T_mushy = (ti64.T_solidus + ti64.T_liquidus) / 2.0f;
    EXPECT_TRUE(ti64.isMushy(T_mushy));
    EXPECT_FALSE(ti64.isSolid(T_mushy));
    EXPECT_FALSE(ti64.isLiquid(T_mushy));

    // Boundary conditions
    EXPECT_FALSE(ti64.isSolid(ti64.T_solidus));  // Solidus is start of mushy zone
    EXPECT_TRUE(ti64.isMushy(ti64.T_solidus));
    EXPECT_TRUE(ti64.isMushy(ti64.T_liquidus));
    EXPECT_FALSE(ti64.isLiquid(ti64.T_liquidus));  // Liquidus is end of mushy zone
}

/**
 * Test 12: Effective heat capacity with latent heat
 */
TEST_F(MaterialPropertiesTest, EffectiveHeatCapacity) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Outside mushy zone - should return normal cp
    float cp_solid = ti64.getEffectiveHeatCapacity(300.0f);
    EXPECT_FLOAT_EQ(cp_solid, ti64.cp_solid);

    float cp_liquid = ti64.getEffectiveHeatCapacity(2200.0f);
    EXPECT_FLOAT_EQ(cp_liquid, ti64.cp_liquid);

    // In mushy zone - should include latent heat contribution
    float T_mushy = (ti64.T_solidus + ti64.T_liquidus) / 2.0f;
    float cp_eff = ti64.getEffectiveHeatCapacity(T_mushy, 1.0f);
    float cp_base = ti64.getSpecificHeat(T_mushy);

    // Effective cp should be higher due to latent heat
    EXPECT_GT(cp_eff, cp_base);

    // Check the latent heat contribution
    float dfl_dT = 1.0f / (ti64.T_liquidus - ti64.T_solidus);
    float expected_cp_eff = cp_base + ti64.L_fusion * dfl_dT;
    EXPECT_NEAR(cp_eff, expected_cp_eff, 1.0f);
}

/**
 * Test 13: Unit conversion utilities
 */
TEST_F(MaterialPropertiesTest, UnitConversionUtilities) {
    // Test temperature conversion
    float T_ref = 300.0f;  // K
    float deltaT = 100.0f;  // K
    float T_phys = 500.0f;  // K

    float T_lattice = MaterialUnits::physicalToLatticeTemperature(T_phys, T_ref, deltaT);
    EXPECT_FLOAT_EQ(T_lattice, 2.0f);  // (500 - 300) / 100 = 2

    float T_phys_back = MaterialUnits::latticeToPhysicalTemperature(T_lattice, T_ref, deltaT);
    EXPECT_FLOAT_EQ(T_phys_back, T_phys);

    // Test diffusivity conversion
    float alpha_phys = 1e-5f;  // m²/s
    float dx = 1e-6f;  // m
    float dt = 1e-9f;  // s

    float alpha_lattice = MaterialUnits::physicalToLatticeDiffusivity(alpha_phys, dx, dt);
    float expected_alpha = alpha_phys * dt / (dx * dx);
    EXPECT_FLOAT_EQ(alpha_lattice, expected_alpha);

    float alpha_phys_back = MaterialUnits::latticeToPhysicalDiffusivity(alpha_lattice, dx, dt);
    EXPECT_FLOAT_EQ(alpha_phys_back, alpha_phys);
}

/**
 * Test 14: Material cache functionality
 */
TEST_F(MaterialPropertiesTest, MaterialCacheFunctionality) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();
    MaterialCache cache;

    // Initially invalid
    EXPECT_FALSE(cache.valid);

    // Update cache in mushy zone where properties change with temperature
    float T = 1900.0f;  // In mushy zone where properties vary with temperature
    cache.update(ti64, T);
    EXPECT_TRUE(cache.valid);
    EXPECT_FLOAT_EQ(cache.T, T);
    EXPECT_FLOAT_EQ(cache.k, ti64.getThermalConductivity(T));
    EXPECT_FLOAT_EQ(cache.cp, ti64.getSpecificHeat(T));
    EXPECT_FLOAT_EQ(cache.rho, ti64.getDensity(T));
    EXPECT_FLOAT_EQ(cache.alpha, ti64.getThermalDiffusivity(T));
    EXPECT_FLOAT_EQ(cache.liquid_fraction, ti64.liquidFraction(T));

    // Update with same temperature (within tolerance) - should not recalculate
    float old_k = cache.k;
    float old_T = cache.T;
    cache.update(ti64, T + 0.05f);  // Within 0.1K tolerance
    EXPECT_FLOAT_EQ(cache.T, old_T);  // Should not update
    EXPECT_FLOAT_EQ(cache.k, old_k);  // Should not change

    // Update with different temperature - should recalculate
    cache.update(ti64, T + 1.0f);  // Outside tolerance
    EXPECT_FLOAT_EQ(cache.T, T + 1.0f);
    // In mushy zone, thermal conductivity changes with temperature
    EXPECT_NE(cache.k, old_k);  // Should change

    // Verify cache values match direct calculations
    EXPECT_FLOAT_EQ(cache.k, ti64.getThermalConductivity(T + 1.0f));
    EXPECT_FLOAT_EQ(cache.cp, ti64.getSpecificHeat(T + 1.0f));
    EXPECT_FLOAT_EQ(cache.rho, ti64.getDensity(T + 1.0f));
}

/**
 * Test 15: Get material by name
 */
TEST_F(MaterialPropertiesTest, GetMaterialByName) {
    // Test various name formats
    MaterialProperties mat;

    // Ti6Al4V variations
    mat = MaterialDatabase::getMaterialByName("Ti6Al4V");
    EXPECT_STREQ(mat.name, "Ti6Al4V");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1923.0f);

    mat = MaterialDatabase::getMaterialByName("ti6al4v");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1923.0f);

    mat = MaterialDatabase::getMaterialByName("TI6AL4V");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1923.0f);

    // 316L variations
    mat = MaterialDatabase::getMaterialByName("316L");
    EXPECT_STREQ(mat.name, "SS316L");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1700.0f);

    mat = MaterialDatabase::getMaterialByName("SS316L");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1700.0f);

    mat = MaterialDatabase::getMaterialByName("ss316l");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1700.0f);

    // IN718 variations
    mat = MaterialDatabase::getMaterialByName("IN718");
    EXPECT_STREQ(mat.name, "IN718");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1609.0f);

    mat = MaterialDatabase::getMaterialByName("Inconel718");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 1609.0f);

    // AlSi10Mg variations
    mat = MaterialDatabase::getMaterialByName("AlSi10Mg");
    EXPECT_STREQ(mat.name, "AlSi10Mg");
    EXPECT_FLOAT_EQ(mat.T_liquidus, 873.0f);

    // Test unknown material
    EXPECT_THROW(MaterialDatabase::getMaterialByName("UnknownMaterial"), std::runtime_error);
}

// GPU kernel for testing device memory access
__global__ void testMaterialPropertiesKernel(float* results, float T) {
    // Test accessing material properties from constant memory
    results[0] = d_material.getDensity(T);
    results[1] = d_material.getThermalConductivity(T);
    results[2] = d_material.getSpecificHeat(T);
    results[3] = d_material.getThermalDiffusivity(T);
    results[4] = d_material.liquidFraction(T);
}

/**
 * Test 16: GPU device memory copy and access
 */
TEST_F(MaterialPropertiesTest, GPUDeviceMemoryCopy) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Copy to device constant memory
    MaterialDatabase::copyToDevice(ti64);

    // Allocate device memory for results
    float* d_results;
    cudaMalloc(&d_results, 5 * sizeof(float));

    // Test temperature
    float T = 1900.0f;  // In mushy zone

    // Launch kernel to test device access
    testMaterialPropertiesKernel<<<1, 1>>>(d_results, T);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(error, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(error);

    // Copy results back
    float h_results[5];
    cudaMemcpy(h_results, d_results, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results match CPU calculations
    EXPECT_NEAR(h_results[0], ti64.getDensity(T), 1e-3f);
    EXPECT_NEAR(h_results[1], ti64.getThermalConductivity(T), 1e-3f);
    EXPECT_NEAR(h_results[2], ti64.getSpecificHeat(T), 1e-1f);  // float precision between CPU/GPU
    EXPECT_NEAR(h_results[3], ti64.getThermalDiffusivity(T), 1e-9f);
    EXPECT_NEAR(h_results[4], ti64.liquidFraction(T), 1e-5f);

    // Clean up
    cudaFree(d_results);
}

/**
 * Test 17: 316L Stainless Steel specific properties
 */
TEST_F(MaterialPropertiesTest, SS316LSpecificProperties) {
    MaterialProperties ss316l = MaterialDatabase::get316L();

    // Verify properties from MATERIAL_DATABASE.yaml
    EXPECT_FLOAT_EQ(ss316l.rho_solid, 7990.0f);
    EXPECT_FLOAT_EQ(ss316l.cp_solid, 500.0f);
    EXPECT_FLOAT_EQ(ss316l.k_solid, 16.2f);
    EXPECT_FLOAT_EQ(ss316l.T_solidus, 1658.0f);
    EXPECT_FLOAT_EQ(ss316l.T_liquidus, 1700.0f);
    EXPECT_FLOAT_EQ(ss316l.surface_tension, 1.75f);
    EXPECT_FLOAT_EQ(ss316l.absorptivity_solid, 0.38f);

    // Test mushy zone width
    float mushy_width = ss316l.T_liquidus - ss316l.T_solidus;
    EXPECT_FLOAT_EQ(mushy_width, 42.0f);  // Narrow mushy zone for 316L
}

/**
 * Test 18: Aluminum alloy unique characteristics
 */
TEST_F(MaterialPropertiesTest, AluminumAlloyCharacteristics) {
    MaterialProperties alsi = MaterialDatabase::getAlSi10Mg();

    // Aluminum has much higher thermal conductivity
    EXPECT_GT(alsi.k_solid, 100.0f);

    // Much lower density than steel/titanium
    EXPECT_LT(alsi.rho_solid, 3000.0f);

    // Very low absorptivity (highly reflective)
    EXPECT_LT(alsi.absorptivity_solid, 0.1f);
    EXPECT_LT(alsi.absorptivity_liquid, 0.15f);

    // Lower melting point
    EXPECT_LT(alsi.T_liquidus, 1000.0f);

    // Lower surface tension
    EXPECT_LT(alsi.surface_tension, 1.0f);
}

/**
 * Test 19: Ti6Al4V Literature-Correct Properties (CRITICAL VALIDATION)
 */
TEST_F(MaterialPropertiesTest, Ti6Al4V_LiteratureCorrectProperties) {
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Test liquid density (Mills 2002)
    EXPECT_NEAR(ti64.rho_liquid, 4110.0f, 10.0f)
        << "Liquid density should be 4110 kg/m³ (Mills 2002)";

    // Test liquid viscosity (Valencia & Quested 2008)
    EXPECT_NEAR(ti64.mu_liquid, 5.0e-3f, 0.5e-3f)
        << "Liquid viscosity should be 0.005 Pa·s (Valencia & Quested 2008)";

    // Test surface tension coefficient (Mills 2002)
    EXPECT_NEAR(ti64.dsigma_dT, -2.6e-4f, 0.2e-4f)
        << "dσ/dT should be -2.6×10⁻⁴ N/(m·K) (Mills 2002)";

    // Test surface tension at melting point
    float sigma_at_Tm = ti64.getSurfaceTension(ti64.T_liquidus);
    EXPECT_NEAR(sigma_at_Tm, 1.65f, 0.05f)
        << "Surface tension at melting point should be ~1.65 N/m";

    // Verify surface tension decreases with temperature (dσ/dT < 0)
    float sigma_at_Tm_plus_100 = ti64.getSurfaceTension(ti64.T_liquidus + 100.0f);
    EXPECT_LT(sigma_at_Tm_plus_100, sigma_at_Tm)
        << "Surface tension should decrease with temperature";

    // Analytical Marangoni velocity estimate
    // Using stress balance: τ_Marangoni ~ μ v / δ
    // where τ_Marangoni = |dσ/dT| × |∇T|
    // and δ is viscous boundary layer thickness
    // This gives: v ~ (|dσ/dT| × |∇T| × δ) / μ

    float dsigma_dT_abs = std::abs(ti64.dsigma_dT);
    float mu = ti64.mu_liquid;

    // Typical LPBF melt pool conditions (Khairallah 2016)
    // Note: Different choices of length scales give different estimates
    // The actual value will be determined by simulation
    float deltaT = 500.0f;  // K (temperature drop across melt pool)
    float L_pool = 100.0e-6f;  // m (melt pool width)
    float gradT = deltaT / L_pool;  // 5×10⁶ K/m

    // Viscous boundary layer thickness ~ melt pool depth
    float delta = 50.0e-6f;  // m (typical depth scale)

    float v_predicted = (dsigma_dT_abs * gradT * delta) / mu;

    // Expected velocity should be O(1 m/s) for LPBF conditions
    // NOTE: Analytical estimates are rough - actual values depend on geometry
    // We're just checking order of magnitude here
    EXPECT_GT(v_predicted, 0.1f)
        << "Predicted Marangoni velocity too low (got " << v_predicted << " m/s)";
    EXPECT_LT(v_predicted, 20.0f)
        << "Predicted Marangoni velocity unrealistically high (got " << v_predicted << " m/s)";

    std::cout << "\n=== Ti6Al4V Marangoni Velocity Prediction ===" << std::endl;
    std::cout << "Material properties:" << std::endl;
    std::cout << "  ρ_liquid = " << ti64.rho_liquid << " kg/m³" << std::endl;
    std::cout << "  μ_liquid = " << ti64.mu_liquid << " Pa·s" << std::endl;
    std::cout << "  dσ/dT = " << ti64.dsigma_dT << " N/(m·K)" << std::endl;
    std::cout << "\nTypical LPBF conditions:" << std::endl;
    std::cout << "  ΔT = " << deltaT << " K across melt pool" << std::endl;
    std::cout << "  L_pool = " << L_pool * 1e6 << " μm" << std::endl;
    std::cout << "  ∇T = " << gradT << " K/m" << std::endl;
    std::cout << "  Boundary layer δ = " << delta * 1e6 << " μm" << std::endl;
    std::cout << "\nPredicted Marangoni velocity: " << v_predicted << " m/s" << std::endl;
    std::cout << "Literature range: 0.5-2 m/s (Khairallah 2016)" << std::endl;
    std::cout << "Expected range: 0.1-20 m/s (order of magnitude check)" << std::endl;

    if (v_predicted >= 0.5f && v_predicted <= 5.0f) {
        std::cout << "✓ PASS - Order of magnitude matches literature" << std::endl;
    } else if (v_predicted >= 0.1f && v_predicted <= 20.0f) {
        std::cout << "⚠ PARTIAL - Reasonable order of magnitude" << std::endl;
        std::cout << "  (exact value depends on geometry and will be determined by simulation)" << std::endl;
    } else {
        std::cout << "✗ FAIL - Velocity prediction unrealistic" << std::endl;
    }
}

/**
 * Test: Steel material properties validation
 * Validates that Steel (Fe) material matches validation case parameters
 */
TEST_F(MaterialPropertiesTest, SteelBasicProperties) {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // Check material name
    EXPECT_STREQ(steel.name, "Steel");

    // Validation case parameters (from test specification)
    EXPECT_FLOAT_EQ(steel.rho_solid, 7900.0f);    // kg/m³
    EXPECT_FLOAT_EQ(steel.rho_liquid, 7433.0f);   // kg/m³
    EXPECT_FLOAT_EQ(steel.T_liquidus, 1723.0f);   // K (melting point)
    EXPECT_FLOAT_EQ(steel.T_vaporization, 3090.0f); // K (boiling point)

    // Check phase change temperatures (200K mushy zone for numerical stability)
    // Widened from 50K to 200K (2026-01-27) to slow solidification and reduce positive feedback
    EXPECT_FLOAT_EQ(steel.T_solidus, 1523.0f);
    EXPECT_FLOAT_EQ(steel.T_liquidus - steel.T_solidus, 200.0f); // 200K mushy zone

    // Verify all properties are positive (basic validation)
    EXPECT_GT(steel.cp_solid, 0.0f);
    EXPECT_GT(steel.cp_liquid, 0.0f);
    EXPECT_GT(steel.k_solid, 0.0f);
    EXPECT_GT(steel.k_liquid, 0.0f);
    EXPECT_GT(steel.mu_liquid, 0.0f);
    EXPECT_GT(steel.L_fusion, 0.0f);
    EXPECT_GT(steel.L_vaporization, 0.0f);
    EXPECT_GT(steel.surface_tension, 0.0f);

    // Check surface tension temperature coefficient is negative (typical for metals)
    EXPECT_LT(steel.dsigma_dT, 0.0f);

    // Check optical properties are in valid range [0, 1]
    EXPECT_GE(steel.absorptivity_solid, 0.0f);
    EXPECT_LE(steel.absorptivity_solid, 1.0f);
    EXPECT_GE(steel.absorptivity_liquid, 0.0f);
    EXPECT_LE(steel.absorptivity_liquid, 1.0f);
    EXPECT_GE(steel.emissivity, 0.0f);
    EXPECT_LE(steel.emissivity, 1.0f);
}

/**
 * Test: Steel temperature-dependent properties
 */
TEST_F(MaterialPropertiesTest, SteelTemperatureDependence) {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // Test at room temperature (solid state)
    float T_room = 300.0f;
    EXPECT_FLOAT_EQ(steel.getDensity(T_room), steel.rho_solid);
    EXPECT_FLOAT_EQ(steel.getSpecificHeat(T_room), steel.cp_solid);
    EXPECT_FLOAT_EQ(steel.getThermalConductivity(T_room), steel.k_solid);
    EXPECT_GT(steel.getDynamicViscosity(T_room), 1e9f); // Very high for solid

    // Test at liquid temperature (above melting point)
    float T_liquid = 2000.0f;
    EXPECT_FLOAT_EQ(steel.getDensity(T_liquid), steel.rho_liquid);
    EXPECT_FLOAT_EQ(steel.getSpecificHeat(T_liquid), steel.cp_liquid);
    EXPECT_FLOAT_EQ(steel.getThermalConductivity(T_liquid), steel.k_liquid);
    EXPECT_FLOAT_EQ(steel.getDynamicViscosity(T_liquid), steel.mu_liquid);

    // Test at melting point (pure element - no mushy zone in this case)
    float T_melt = steel.T_liquidus;
    EXPECT_FLOAT_EQ(steel.getDensity(T_melt), steel.rho_liquid);
    EXPECT_FLOAT_EQ(steel.liquidFraction(T_melt), 1.0f);
}

/**
 * Test: Steel material validation
 */
TEST_F(MaterialPropertiesTest, SteelValidation) {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // Call validation function
    EXPECT_TRUE(steel.validate()) << "Steel material properties failed validation";

    // Verify temperature ordering
    EXPECT_LT(steel.T_solidus, steel.T_vaporization);
    EXPECT_LE(steel.T_solidus, steel.T_liquidus);
    EXPECT_LT(steel.T_liquidus, steel.T_vaporization);

    // Check density ratio (liquid should be less dense than solid)
    EXPECT_LT(steel.rho_liquid, steel.rho_solid);
    float shrinkage = steel.getShrinkageFactor();
    EXPECT_GT(shrinkage, 0.0f);
    EXPECT_LT(shrinkage, 0.2f); // Reasonable range for metals (<20% volume change)
}

/**
 * Test: Steel material lookup by name
 */
TEST_F(MaterialPropertiesTest, SteelNameLookup) {
    // Test various name variations
    std::vector<std::string> valid_names = {"Steel", "steel", "Fe", "fe", "Iron", "iron"};

    for (const auto& name : valid_names) {
        MaterialProperties mat = MaterialDatabase::getMaterialByName(name);
        EXPECT_STREQ(mat.name, "Steel") << "Name lookup failed for: " << name;
        EXPECT_FLOAT_EQ(mat.rho_solid, 7900.0f);
    }
}

/**
 * Test: Steel surface tension temperature dependence
 */
TEST_F(MaterialPropertiesTest, SteelSurfaceTension) {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // At melting point, should equal base surface tension
    float sigma_melt = steel.getSurfaceTension(steel.T_liquidus);
    EXPECT_FLOAT_EQ(sigma_melt, steel.surface_tension);

    // At higher temperature, should decrease (negative dsigma_dT)
    float T_high = steel.T_liquidus + 100.0f;
    float sigma_high = steel.getSurfaceTension(T_high);
    EXPECT_LT(sigma_high, steel.surface_tension);

    // Verify linear relationship: σ(T) = σ₀ + (dσ/dT) * (T - T_m)
    float expected_sigma = steel.surface_tension + steel.dsigma_dT * 100.0f;
    EXPECT_FLOAT_EQ(sigma_high, expected_sigma);
}

/**
 * Test: Steel thermal diffusivity calculation
 */
TEST_F(MaterialPropertiesTest, SteelThermalDiffusivity) {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // Calculate thermal diffusivity at room temperature
    // α = k / (ρ * cp)
    float alpha_expected = steel.k_solid / (steel.rho_solid * steel.cp_solid);
    float alpha_calculated = steel.getThermalDiffusivity(300.0f);

    EXPECT_NEAR(alpha_calculated, alpha_expected, 1e-6f);

    // Verify positive diffusivity
    EXPECT_GT(alpha_calculated, 0.0f);

    // Typical range for steel thermal diffusivity: 1e-5 to 3e-5 m²/s
    EXPECT_GT(alpha_calculated, 1e-6f);
    EXPECT_LT(alpha_calculated, 1e-4f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}