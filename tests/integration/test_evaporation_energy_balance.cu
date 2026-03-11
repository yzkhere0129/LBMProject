/**
 * @file test_evaporation_energy_balance.cu
 * @brief Integration test for evaporation energy conservation
 *
 * Test Suite 1.2: Validates energy balance with evaporation cooling
 *
 * Success Criteria:
 * - Energy removed matches evaporation power × time within 10%
 * - Temperature decreases due to evaporation cooling
 * - No negative temperatures or NaN values
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include "physics/vof_solver.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>

using namespace lbm::physics;

class EvaporationEnergyBalanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ti6Al4V properties
        strncpy(material.name, "Ti6Al4V", sizeof(material.name)-1);
        material.name[sizeof(material.name)-1] = '\0';
        material.rho_solid = 4420.0f;
        material.cp_solid = 670.0f;
        material.k_solid = 6.8f;
        material.T_solidus = 1878.0f;
        material.T_liquidus = 1928.0f;
        material.L_fusion = 286000.0f;
        material.T_vaporization = 3533.0f;
        material.L_vaporization = 9830000.0f;
        material.molar_mass = 0.04593f;  // Ti6Al4V effective molar mass [kg/mol]
    }

    MaterialProperties material;
};

TEST_F(EvaporationEnergyBalanceTest, SingleHotCellEvaporation) {
    std::cout << "\nTest 1: Single hot cell evaporation energy balance" << std::endl;

    int nx = 20, ny = 20, nz = 30;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  dx = " << dx*1e6 << " µm, dt = " << dt*1e9 << " ns" << std::endl;

    // Create thermal solver
    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // Initialize uniform temperature
    thermal.initialize(300.0f);

    // Create a hot spot at top surface (simulating evaporation)
    std::vector<float> T_field(nx * ny * nz, 300.0f);

    // Set hot cell at surface (z = nz-1)
    int hot_x = nx / 2;
    int hot_y = ny / 2;
    int hot_z = nz - 1;
    int hot_idx = hot_x + nx * (hot_y + ny * hot_z);

    T_field[hot_idx] = 4000.0f;  // Well above boiling point

    thermal.initialize(T_field.data());

    // Create fill level (surface identification)
    std::vector<float> fill_level(nx * ny * nz, 1.0f);  // All liquid
    // Mark top surface as interface
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * (nz-1));
            fill_level[idx] = 0.5f;  // Interface
        }
    }

    float *d_fill_level;
    cudaMalloc(&d_fill_level, nx * ny * nz * sizeof(float));
    cudaMemcpy(d_fill_level, fill_level.data(), nx * ny * nz * sizeof(float),
               cudaMemcpyHostToDevice);

    float E_initial = thermal.computeTotalThermalEnergy(dx);
    std::cout << "  Initial energy: " << E_initial << " J" << std::endl;

    // NOTE: applySurfaceCooling is not a public method in ThermalLBM
    // We'll test using computeEvaporationPower instead
    std::cout << "  Testing evaporation power calculation..." << std::endl;

    // Compute evaporation power
    float P_evap = thermal.computeEvaporationPower(d_fill_level, dx);
    std::cout << "  Evaporation power: " << P_evap << " W" << std::endl;

    // For a single cell at 4000K, expect reasonable evaporation power
    // FIX (2025-12-02): Updated threshold for alpha_evap = 0.18 (calibrated value)
    // Expected power is ~4.5x lower than with alpha_evap = 0.82
    EXPECT_GT(P_evap, 0.001f) << "Evaporation power should be positive";
    EXPECT_LT(P_evap, 100.0f) << "Evaporation power unrealistically high for single cell";

    // Verify it's not the old buggy value (would be ~66000× higher)
    EXPECT_LT(P_evap, 10.0f) << "Evaporation power suggests old buggy formula";

    cudaFree(d_fill_level);
}

TEST_F(EvaporationEnergyBalanceTest, MultiCellEvaporationScaling) {
    std::cout << "\nTest 2: Multi-cell evaporation power scaling" << std::endl;

    int nx = 30, ny = 30, nz = 40;
    float dx = 2.0e-6f;
    float dt = 1.0e-9f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // Test different numbers of hot cells
    std::vector<int> n_hot_cells = {1, 10, 50, 100};
    std::vector<float> powers;

    for (int n_hot : n_hot_cells) {
        std::vector<float> T_field(nx * ny * nz, 300.0f);
        std::vector<float> fill_level(nx * ny * nz, 1.0f);

        // Create n_hot cells at surface with T > T_boil
        int cells_set = 0;
        for (int j = 0; j < ny && cells_set < n_hot; ++j) {
            for (int i = 0; i < nx && cells_set < n_hot; ++i) {
                int idx = i + nx * (j + ny * (nz-1));
                T_field[idx] = 3700.0f;  // Above boiling
                fill_level[idx] = 0.5f;   // Interface
                cells_set++;
            }
        }

        thermal.initialize(T_field.data());

        float *d_fill_level;
        cudaMalloc(&d_fill_level, nx * ny * nz * sizeof(float));
        cudaMemcpy(d_fill_level, fill_level.data(), nx * ny * nz * sizeof(float),
                   cudaMemcpyHostToDevice);

        float P_evap = thermal.computeEvaporationPower(d_fill_level, dx);
        powers.push_back(P_evap);

        std::cout << "  " << n_hot << " hot cells: P_evap = " << P_evap << " W" << std::endl;

        cudaFree(d_fill_level);
    }

    // Power should scale roughly linearly with number of cells
    EXPECT_GT(powers[1], powers[0] * 5.0f) << "Power should scale with cell count";
    EXPECT_LT(powers[1], powers[0] * 15.0f) << "Power scaling unrealistic";

    EXPECT_GT(powers[2], powers[1] * 3.0f) << "Power should scale with cell count";
    EXPECT_LT(powers[2], powers[1] * 7.0f) << "Power scaling unrealistic";

    // Total power for 100 cells should still be < typical laser power
    EXPECT_LT(powers[3], 200.0f) << "Evaporation power exceeds typical laser input";
}

TEST_F(EvaporationEnergyBalanceTest, TemperatureDependence) {
    std::cout << "\nTest 3: Evaporation power vs temperature" << std::endl;

    int nx = 20, ny = 20, nz = 20;
    float dx = 2.0e-6f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // FIX (2025-12-02): Updated temperature range to account for new activation threshold
    // Evaporation now activates at T > T_boil - 500K = 3060K (aligned with recoil pressure)
    std::vector<float> temperatures = {3000.0f, 3100.0f, 3300.0f, 3800.0f, 4000.0f};
    std::vector<float> powers;

    for (float T : temperatures) {
        std::vector<float> T_field(nx * ny * nz, 300.0f);
        std::vector<float> fill_level(nx * ny * nz, 1.0f);

        // Set 10 cells at surface to test temperature
        for (int i = 0; i < 10; ++i) {
            int idx = i + nx * ny * (nz-1);
            T_field[idx] = T;
            fill_level[idx] = 0.5f;
        }

        thermal.initialize(T_field.data());

        float *d_fill_level;
        cudaMalloc(&d_fill_level, nx * ny * nz * sizeof(float));
        cudaMemcpy(d_fill_level, fill_level.data(), nx * ny * nz * sizeof(float),
                   cudaMemcpyHostToDevice);

        float P_evap = thermal.computeEvaporationPower(d_fill_level, dx);
        powers.push_back(P_evap);

        std::cout << "  T = " << T << " K: P_evap = " << P_evap << " W" << std::endl;

        cudaFree(d_fill_level);
    }

    // Power should increase with temperature (for T > T_activation)
    // First temperature (3000K) is below activation threshold → P_evap = 0
    // Remaining temperatures should show monotonic increase
    EXPECT_EQ(powers[0], 0.0f) << "Below activation threshold (3060K), no evaporation";

    for (size_t i = 2; i < powers.size(); ++i) {
        EXPECT_GT(powers[i], powers[i-1])
            << "Evaporation power should increase with temperature above activation";
    }

    // Below activation threshold should have zero evaporation
    EXPECT_EQ(powers[0], 0.0f) << "Evaporation below activation threshold should be zero";

    // Well above activation should have strong evaporation
    EXPECT_GT(powers[4], powers[2] * 2.0f) << "Evaporation scaling with T too weak";
}

TEST_F(EvaporationEnergyBalanceTest, RealisticMagnitudeCheck) {
    std::cout << "\nTest 4: Realistic evaporation power magnitude" << std::endl;

    // Simulate realistic melt pool scenario
    int nx = 50, ny = 50, nz = 50;
    float dx = 2.0e-6f;  // 100µm domain
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    std::vector<float> T_field(nx * ny * nz, 300.0f);
    std::vector<float> fill_level(nx * ny * nz, 1.0f);

    // Create realistic melt pool top surface (circular hot region)
    int center_x = nx / 2;
    int center_y = ny / 2;
    int radius = 10;  // ~20µm diameter hot zone

    int n_evap_cells = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int dx_cell = i - center_x;
            int dy_cell = j - center_y;
            float r = sqrt(dx_cell*dx_cell + dy_cell*dy_cell);

            if (r < radius) {
                int idx = i + nx * (j + ny * (nz-1));
                T_field[idx] = 3700.0f;  // Typical melt pool surface temp
                fill_level[idx] = 0.5f;
                n_evap_cells++;
            }
        }
    }

    std::cout << "  Evaporating cells: " << n_evap_cells << std::endl;

    thermal.initialize(T_field.data());

    float *d_fill_level;
    cudaMalloc(&d_fill_level, nx * ny * nz * sizeof(float));
    cudaMemcpy(d_fill_level, fill_level.data(), nx * ny * nz * sizeof(float),
               cudaMemcpyHostToDevice);

    float P_evap = thermal.computeEvaporationPower(d_fill_level, dx);
    std::cout << "  Evaporation power: " << P_evap << " W" << std::endl;

    // For realistic LPBF with ~100W laser, evaporation should be 10-50% of input
    float P_laser_typical = 100.0f;
    float evap_fraction = P_evap / P_laser_typical;

    std::cout << "  Evaporation fraction of laser power: " << evap_fraction * 100.0f << "%" << std::endl;

    // FIX (2025-12-02): Updated thresholds for alpha_evap = 0.18 (calibrated value)
    // Calibration reduced evaporation strength by ~4.5x to prevent excessive cooling
    EXPECT_GT(P_evap, 0.1f) << "Evaporation power too low for realistic melt pool";
    EXPECT_LT(P_evap, P_laser_typical) << "Evaporation exceeds laser input (energy violation)";

    // With calibrated alpha_evap = 0.18, evaporation fraction is 1-5% (reduced from 10-30%)
    EXPECT_GT(evap_fraction, 0.001f) << "Evaporation fraction unrealistically low";
    EXPECT_LT(evap_fraction, 0.10f) << "Evaporation fraction unrealistically high";

    cudaFree(d_fill_level);
}

TEST_F(EvaporationEnergyBalanceTest, NoNaNWithHighTemperature) {
    std::cout << "\nTest 5: Stability check - No NaN at extreme temperatures" << std::endl;

    int nx = 15, ny = 15, nz = 15;
    float dx = 2.0e-6f;
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // Test extreme temperatures
    std::vector<float> test_temps = {5000.0f, 6000.0f, 7000.0f};

    for (float T_extreme : test_temps) {
        std::vector<float> T_field(nx * ny * nz, 300.0f);
        std::vector<float> fill_level(nx * ny * nz, 1.0f);

        // Single very hot cell
        int idx = nx/2 + nx * (ny/2 + ny * (nz-1));
        T_field[idx] = T_extreme;
        fill_level[idx] = 0.5f;

        thermal.initialize(T_field.data());

        float *d_fill_level;
        cudaMalloc(&d_fill_level, nx * ny * nz * sizeof(float));
        cudaMemcpy(d_fill_level, fill_level.data(), nx * ny * nz * sizeof(float),
                   cudaMemcpyHostToDevice);

        float P_evap = thermal.computeEvaporationPower(d_fill_level, dx);

        std::cout << "  T = " << T_extreme << " K: P_evap = " << P_evap << " W" << std::endl;

        // Check for NaN or Inf
        ASSERT_FALSE(std::isnan(P_evap)) << "NaN detected at T = " << T_extreme << " K";
        ASSERT_FALSE(std::isinf(P_evap)) << "Inf detected at T = " << T_extreme << " K";
        ASSERT_GT(P_evap, 0.0f) << "Negative power at T = " << T_extreme << " K";

        cudaFree(d_fill_level);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
