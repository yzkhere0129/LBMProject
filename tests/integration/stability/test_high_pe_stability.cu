/**
 * @file test_high_pe_stability.cu
 * @brief Integration test for high Peclet number stability
 *
 * CRITICAL REGRESSION TEST: End-to-end test of full solver at high Pe.
 *
 * Context:
 * - In LPBF, Peclet number Pe = v*L/α can reach 10-20 in melt pool
 * - Without stability fixes, simulation diverges within 100-500 steps
 * - This test ensures the entire thermal-fluid coupling remains stable
 *
 * Fixes Tested:
 * 1. Flux limiter in thermal equilibrium
 * 2. Temperature bounds enforcement
 * 3. Omega reduction (tau_T capped)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/material_properties.h"
#include "physics/lattice_d3q7.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstring>

using namespace lbm::physics;

class HighPeStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }

        // Ti6Al4V material properties
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
 * @brief Test thermal solver stability at high Pe for 500 steps
 */
TEST_F(HighPeStabilityTest, ThermalSolver500Steps) {
    // Small domain to run quickly
    int nx = 20, ny = 20, nz = 20;
    int num_cells = nx * ny * nz;

    // High thermal diffusivity (liquid metal)
    float alpha = 8.0e-6f;  // m²/s

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // Initialize to high temperature (melt pool)
    thermal.initialize(2500.0f);

    // Create high velocity field (Pe ~ 5)
    // u=0.5 in lattice units ≈ Ma=1.0 (cs=0.5), realistic high-Pe for LPBF
    std::vector<float> h_ux(num_cells, 0.5f);
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));
    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Run 500 timesteps
    bool diverged = false;
    float max_T = 0.0f;
    int divergence_step = -1;

    std::vector<float> h_temp(num_cells);

    for (int step = 0; step < 500; ++step) {
        // LBM cycle
        thermal.collisionBGK(d_ux, d_uy, d_uz);
        thermal.streaming();
        thermal.computeTemperature();

        // Check for divergence every 50 steps
        if (step % 50 == 0) {
            thermal.copyTemperatureToHost(h_temp.data());

            // Check max temperature
            max_T = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                if (std::isnan(h_temp[i]) || std::isinf(h_temp[i])) {
                    diverged = true;
                    divergence_step = step;
                    break;
                }
                max_T = std::max(max_T, h_temp[i]);
            }

            if (diverged) break;

            // Temperature should not explode to infinity
            // Note: With advection and boundary clamping, T can grow beyond
            // initial value but must remain finite and bounded
            EXPECT_LT(max_T, 1e6f)
                << "REGRESSION: Temperature runaway at step " << step;
            EXPECT_GT(max_T, 0.0f)
                << "Temperature collapsed to zero at step " << step;
        }
    }

    EXPECT_FALSE(diverged)
        << "REGRESSION: Simulation diverged at step " << divergence_step;

    // Final check
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FALSE(std::isnan(h_temp[i]))
            << "NaN detected in final state at cell " << i;
        EXPECT_GE(h_temp[i], 0.0f)
            << "Negative temperature in final state";
        EXPECT_LE(h_temp[i], 1e6f)
            << "Temperature explosion in final state";
    }

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test stability with varying velocities (adaptive Pe test)
 */
TEST_F(HighPeStabilityTest, VaryingVelocityStability) {
    int nx = 15, ny = 15, nz = 15;
    int num_cells = nx * ny * nz;
    float alpha = 8.0e-6f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(2000.0f);

    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    // Test with ramping velocity (0 -> 1.0 lattice units)
    for (int step = 0; step < 200; ++step) {
        // Ramp velocity up to Ma≈2 (beyond normal validity, tests limiter)
        float v = (step / 200.0f) * 1.0f;

        for (int i = 0; i < num_cells; ++i) {
            h_ux[i] = v;
            h_uy[i] = 0.0f;
            h_uz[i] = 0.0f;
        }

        cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

        // LBM step
        thermal.collisionBGK(d_ux, d_uy, d_uz);
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Check final state
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FALSE(std::isnan(h_temp[i]))
            << "REGRESSION: NaN after velocity ramp at cell " << i;
        EXPECT_GE(h_temp[i], 0.0f);
        EXPECT_LE(h_temp[i], 1e6f);
    }

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test stability with localized high-velocity region (realistic melt pool)
 */
TEST_F(HighPeStabilityTest, LocalizedHighVelocityRegion) {
    int nx = 30, ny = 30, nz = 20;
    int num_cells = nx * ny * nz;
    float alpha = 8.0e-6f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);

    // Temperature field: hot center, cool exterior
    std::vector<float> h_temp_init(num_cells);
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx = x + y * nx + z * nx * ny;

                // Gaussian temperature profile
                float dx = x - nx / 2;
                float dy = y - ny / 2;
                float dz = z - nz / 2;
                float r2 = dx * dx + dy * dy + dz * dz;
                h_temp_init[idx] = 300.0f + 3000.0f * expf(-r2 / 50.0f);
            }
        }
    }

    thermal.initialize(h_temp_init.data());

    // Velocity field: high in center (Marangoni-like)
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx = x + y * nx + z * nx * ny;

                float dx = x - nx / 2;
                float dy = y - ny / 2;
                float r = sqrtf(dx * dx + dy * dy);

                // Radial velocity profile
                if (r > 0.1f) {
                    h_ux[idx] = 0.5f * dx / r;
                    h_uy[idx] = 0.5f * dy / r;
                } else {
                    h_ux[idx] = 0.0f;
                    h_uy[idx] = 0.0f;
                }
                h_uz[idx] = 0.0f;
            }
        }
    }

    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));
    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Run 300 steps
    for (int step = 0; step < 300; ++step) {
        thermal.collisionBGK(d_ux, d_uy, d_uz);
        thermal.streaming();
        thermal.computeTemperature();

        // Periodic check
        if (step % 100 == 0) {
            std::vector<float> h_temp(num_cells);
            thermal.copyTemperatureToHost(h_temp.data());

            for (int i = 0; i < num_cells; ++i) {
                ASSERT_FALSE(std::isnan(h_temp[i]))
                    << "REGRESSION: NaN at step " << step << ", cell " << i;
            }
        }
    }

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test stability with heat source and high velocity
 */
TEST_F(HighPeStabilityTest, HeatSourceWithHighVelocity) {
    int nx = 20, ny = 20, nz = 20;
    int num_cells = nx * ny * nz;
    float alpha = 8.0e-6f;

    ThermalLBM thermal(nx, ny, nz, material, alpha, false);
    thermal.initialize(1500.0f);

    // Heat source at center
    std::vector<float> h_heat_source(num_cells, 0.0f);
    int center = (nx / 2) + (ny / 2) * nx + (nz / 2) * nx * ny;
    h_heat_source[center] = 1e11f;  // W/m³

    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));
    cudaMemcpy(d_heat_source, h_heat_source.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // High velocity (Ma≈0.6, tests limiter + heat source interaction)
    std::vector<float> h_ux(num_cells, 0.3f);
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));
    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    float dt = 1e-7f;

    // Run 200 steps with heating
    for (int step = 0; step < 200; ++step) {
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK(d_ux, d_uy, d_uz);
        thermal.streaming();
        thermal.computeTemperature();
    }

    // Verify stability
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FALSE(std::isnan(h_temp[i]))
            << "NaN with heat source + high velocity at cell " << i;
        EXPECT_LE(h_temp[i], 1e6f)
            << "Temperature runaway with heat source";
    }

    cudaFree(d_heat_source);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
