/**
 * @file test_enthalpy.cu
 * @brief Unit tests for enthalpy-temperature conversion
 *
 * Tests verify:
 * 1. H(T) calculation: H = ρ·cp·T + fl·ρ·L_fusion
 * 2. T(H) iterative solution (Newton-Raphson)
 * 3. Roundtrip consistency: T → H → T
 * 4. Dimensional analysis
 * 5. Energy conservation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/phase_change.h"
#include "physics/material_properties.h"
#include <vector>
#include <cmath>

using namespace lbm::physics;

class EnthalpyTest : public ::testing::Test {
protected:
    MaterialProperties mat;
    PhaseChangeSolver* solver;
    int nx, ny, nz;

    float* d_temperature;
    float* h_temperature;
    float* h_enthalpy;
    float* h_liquid_fraction;

    void SetUp() override {
        // Use Ti6Al4V properties
        mat = MaterialDatabase::getTi6Al4V();

        // Small test domain
        nx = 10;
        ny = 10;
        nz = 10;
        int num_cells = nx * ny * nz;

        // Allocate memory
        cudaMalloc(&d_temperature, num_cells * sizeof(float));
        h_temperature = new float[num_cells];
        h_enthalpy = new float[num_cells];
        h_liquid_fraction = new float[num_cells];

        // Create solver
        solver = new PhaseChangeSolver(nx, ny, nz, mat);
    }

    void TearDown() override {
        delete solver;
        cudaFree(d_temperature);
        delete[] h_temperature;
        delete[] h_enthalpy;
        delete[] h_liquid_fraction;
    }
};

/**
 * Test: Enthalpy calculation for solid phase (T < T_solidus)
 *
 * In solid: fl = 0, so H = ρ·cp·T
 */
TEST_F(EnthalpyTest, EnthalpySolidPhase) {
    int num_cells = nx * ny * nz;

    // Set uniform temperature in solid phase
    float T_test = 1500.0f;  // Well below T_solidus = 1878 K
    ASSERT_LT(T_test, mat.T_solidus);

    for (int i = 0; i < num_cells; ++i) {
        h_temperature[i] = T_test;
    }
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize solver
    solver->initializeFromTemperature(d_temperature);

    // Copy results back
    solver->copyEnthalpyToHost(h_enthalpy);
    solver->copyLiquidFractionToHost(h_liquid_fraction);

    // Expected: H = ρ_ref·cp_ref·T (using solid as reference)
    float rho_ref = mat.rho_solid;
    float cp_ref = mat.cp_solid;
    float H_expected = rho_ref * cp_ref * T_test;

    // Check all cells
    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FLOAT_EQ(h_liquid_fraction[i], 0.0f)
            << "Liquid fraction should be zero in solid phase";

        EXPECT_NEAR(h_enthalpy[i], H_expected, 1.0f)
            << "Enthalpy calculation error at cell " << i
            << "\nExpected: " << H_expected << " J/m³"
            << "\nGot: " << h_enthalpy[i] << " J/m³";
    }
}

/**
 * Test: Enthalpy calculation for liquid phase (T > T_liquidus)
 *
 * In liquid: fl = 1, so H = ρ·cp·T + ρ·L_fusion
 */
TEST_F(EnthalpyTest, EnthalpyLiquidPhase) {
    int num_cells = nx * ny * nz;

    // Set uniform temperature in liquid phase
    float T_test = 2200.0f;  // Well above T_liquidus = 1928 K
    ASSERT_GT(T_test, mat.T_liquidus);

    for (int i = 0; i < num_cells; ++i) {
        h_temperature[i] = T_test;
    }
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize solver
    solver->initializeFromTemperature(d_temperature);

    // Copy results back
    solver->copyEnthalpyToHost(h_enthalpy);
    solver->copyLiquidFractionToHost(h_liquid_fraction);

    // Expected: H = ρ_ref·cp_ref·T + ρ_ref·L_fusion (using solid as reference)
    float rho_ref = mat.rho_solid;
    float cp_ref = mat.cp_solid;
    float H_expected = rho_ref * cp_ref * T_test + rho_ref * mat.L_fusion;

    // Check all cells
    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FLOAT_EQ(h_liquid_fraction[i], 1.0f)
            << "Liquid fraction should be one in liquid phase";

        EXPECT_NEAR(h_enthalpy[i], H_expected, 1.0f)
            << "Enthalpy calculation error at cell " << i
            << "\nExpected: " << H_expected << " J/m³"
            << "\nGot: " << h_enthalpy[i] << " J/m³";
    }
}

/**
 * Test: Enthalpy calculation in mushy zone
 *
 * In mushy zone: H = ρ·cp·T + fl(T)·ρ·L_fusion
 * where fl = (T - T_solidus)/(T_liquidus - T_solidus)
 */
TEST_F(EnthalpyTest, EnthalpyMushyZone) {
    int num_cells = nx * ny * nz;

    // Set temperature at midpoint of mushy zone
    float T_test = (mat.T_solidus + mat.T_liquidus) / 2.0f;  // 1903 K
    ASSERT_TRUE(mat.isMushy(T_test));

    for (int i = 0; i < num_cells; ++i) {
        h_temperature[i] = T_test;
    }
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize solver
    solver->initializeFromTemperature(d_temperature);

    // Copy results back
    solver->copyEnthalpyToHost(h_enthalpy);
    solver->copyLiquidFractionToHost(h_liquid_fraction);

    // Expected: fl = 0.5 at midpoint
    float fl_expected = mat.liquidFraction(T_test);
    EXPECT_NEAR(fl_expected, 0.5f, 0.01f);

    // Expected enthalpy (using solid as reference)
    float rho_ref = mat.rho_solid;
    float cp_ref = mat.cp_solid;
    float H_expected = rho_ref * cp_ref * T_test + fl_expected * rho_ref * mat.L_fusion;

    // Check all cells
    for (int i = 0; i < num_cells; ++i) {
        EXPECT_NEAR(h_liquid_fraction[i], fl_expected, 1e-4f)
            << "Liquid fraction mismatch at cell " << i;

        EXPECT_NEAR(h_enthalpy[i], H_expected, 10.0f)
            << "Enthalpy calculation error at cell " << i
            << "\nExpected: " << H_expected << " J/m³"
            << "\nGot: " << h_enthalpy[i] << " J/m³";
    }
}

/**
 * Test: Roundtrip T → H → T should recover original temperature
 */
TEST_F(EnthalpyTest, RoundtripConsistency) {
    int num_cells = nx * ny * nz;

    // Test at various temperatures
    std::vector<float> test_temps = {
        300.0f,      // Solid
        1500.0f,     // Solid near melting
        1878.0f,     // At solidus
        1903.0f,     // Mushy zone midpoint
        1928.0f,     // At liquidus
        2200.0f,     // Liquid
        2500.0f      // Hot liquid
    };

    for (float T_original : test_temps) {
        // Set temperature
        for (int i = 0; i < num_cells; ++i) {
            h_temperature[i] = T_original;
        }
        cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float),
                   cudaMemcpyHostToDevice);

        // T → H
        solver->updateEnthalpyFromTemperature(d_temperature);

        // H → T (modify temperature first to test solver)
        for (int i = 0; i < num_cells; ++i) {
            h_temperature[i] = T_original * 0.9f;  // Wrong initial guess
        }
        cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float),
                   cudaMemcpyHostToDevice);

        solver->updateTemperatureFromEnthalpy(d_temperature, 0.01f, 20);

        // Copy back recovered temperature
        cudaMemcpy(h_temperature, d_temperature, num_cells * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Check all cells
        for (int i = 0; i < num_cells; ++i) {
            float error = std::abs(h_temperature[i] - T_original);
            EXPECT_LT(error, 0.1f)
                << "Roundtrip error too large at T = " << T_original << " K"
                << "\nOriginal: " << T_original << " K"
                << "\nRecovered: " << h_temperature[i] << " K"
                << "\nError: " << error << " K";
        }
    }
}

/**
 * Test: Latent heat jump across mushy zone
 *
 * When going from solid to liquid, enthalpy should increase by ρ·L_fusion
 * beyond the sensible heat increase
 */
TEST_F(EnthalpyTest, LatentHeatJump) {
    // Compute enthalpy just below and just above mushy zone
    float T_solid = mat.T_solidus - 10.0f;
    float T_liquid = mat.T_liquidus + 10.0f;

    // Enthalpy in solid
    float rho_solid = mat.getDensity(T_solid);
    float cp_solid = mat.getSpecificHeat(T_solid);
    float H_solid = rho_solid * cp_solid * T_solid;

    // Enthalpy in liquid
    float rho_liquid = mat.getDensity(T_liquid);
    float cp_liquid = mat.getSpecificHeat(T_liquid);
    float H_liquid = rho_liquid * cp_liquid * T_liquid + rho_liquid * mat.L_fusion;

    // Sensible heat increase if no phase change
    float deltaT = T_liquid - T_solid;
    float avg_cp = (cp_solid + cp_liquid) / 2.0f;
    float avg_rho = (rho_solid + rho_liquid) / 2.0f;
    float H_sensible_only = rho_solid * cp_solid * T_solid + avg_rho * avg_cp * deltaT;

    // Actual enthalpy increase
    float deltaH_actual = H_liquid - H_solid;

    // Latent heat contribution
    float latent_heat = rho_liquid * mat.L_fusion;

    // The actual increase should be significantly larger due to latent heat
    EXPECT_GT(deltaH_actual, H_sensible_only - H_solid)
        << "Enthalpy jump should include latent heat";

    // Check that latent heat is significant
    float sensible_heat_change = avg_rho * avg_cp * deltaT;
    EXPECT_GT(latent_heat, sensible_heat_change)
        << "Latent heat should dominate over sensible heat in this temperature range";

    std::cout << "Latent heat: " << latent_heat / 1e9 << " GJ/m³" << std::endl;
    std::cout << "Sensible heat change: " << sensible_heat_change / 1e9 << " GJ/m³" << std::endl;
    std::cout << "Ratio: " << latent_heat / sensible_heat_change << std::endl;
}

/**
 * Test: Dimensional analysis
 *
 * Verify that enthalpy has correct units [J/m³]
 */
TEST_F(EnthalpyTest, DimensionalAnalysis) {
    // Physical values for Ti6Al4V at 2000 K (liquid)
    // Using solid reference: ρ_ref ~ 4420 kg/m³, cp_ref ~ 610 J/(kg·K)
    // T ~ 2000 K
    // L_fusion ~ 2.86e5 J/kg

    float T = 2000.0f;
    float rho_ref = mat.rho_solid;    // Reference density
    float cp_ref = mat.cp_solid;      // Reference cp
    float L = mat.L_fusion;

    // Expected order of magnitude:
    // H = ρ_ref·cp_ref·T + ρ_ref·L
    // H ~ 4420 * 610 * 2000 + 4420 * 2.86e5
    // H ~ 5.39e9 + 1.26e9 = 6.65e9 J/m³

    float H_expected_order = 6.5e9;  // ~6.5 GJ/m³

    // Set temperature
    int num_cells = nx * ny * nz;
    for (int i = 0; i < num_cells; ++i) {
        h_temperature[i] = T;
    }
    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    solver->initializeFromTemperature(d_temperature);
    solver->copyEnthalpyToHost(h_enthalpy);

    // Check order of magnitude
    float H_computed = h_enthalpy[0];
    EXPECT_GT(H_computed, 1e9)
        << "Enthalpy should be on order of GJ/m³ for metals at high T";
    EXPECT_LT(H_computed, 1e10)
        << "Enthalpy seems too large";

    EXPECT_NEAR(H_computed, H_expected_order, H_expected_order * 0.5f)
        << "Enthalpy order of magnitude mismatch"
        << "\nComputed: " << H_computed / 1e9 << " GJ/m³"
        << "\nExpected: ~" << H_expected_order / 1e9 << " GJ/m³";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
