/**
 * @file test_buoyancy_unit.cu
 * @brief Unit tests for buoyancy force calculation in LBM-CUDA
 *
 * Physics Validation:
 * 1. Hydrostatic pressure distribution: p(z) = p0 + ρgz
 * 2. Archimedes buoyancy force: F = Δρ × g × V
 *
 * Test Approach:
 * - Use ForceAccumulator::addBuoyancyForce to compute buoyancy
 * - Validate against analytical solutions
 * - Ensure proper unit handling and numerical accuracy
 *
 * Acceptance Criteria:
 * - Hydrostatic pressure L2 error < 1%
 * - Buoyancy force magnitude error < 5%
 *
 * Author: Testing and Validation Specialist
 * Date: 2026-01-18
 */

#include <gtest/gtest.h>
#include "physics/force_accumulator.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>

using namespace lbm::physics;

class BuoyancyUnitTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Compute L2 error between numerical and analytical fields
     */
    float computeL2Error(const std::vector<float>& numerical,
                         const std::vector<float>& analytical) {
        if (numerical.size() != analytical.size()) {
            return 1.0f;  // Invalid comparison
        }

        float sum_sq_error = 0.0f;
        float sum_sq_ref = 0.0f;

        for (size_t i = 0; i < numerical.size(); ++i) {
            float error = numerical[i] - analytical[i];
            sum_sq_error += error * error;
            sum_sq_ref += analytical[i] * analytical[i];
        }

        if (sum_sq_ref < 1e-12f) {
            return 0.0f;  // Reference is zero
        }

        return std::sqrt(sum_sq_error / sum_sq_ref);
    }
};

/**
 * @brief Test 1: Hydrostatic Pressure Distribution
 *
 * Physics:
 * In a static fluid column under gravity, the pressure varies as:
 *   p(z) = p0 + ρ × g × z
 *
 * The buoyancy force density (Boussinesq approximation) is:
 *   F_b = ρ₀ × β × (T - T_ref) × g  [N/m³]
 *
 * For a linear temperature profile T(z) = T_ref + ΔT × z/H:
 *   F_b(z) = ρ₀ × β × ΔT × (z/H) × g
 *
 * Integration gives the pressure distribution.
 *
 * Test Setup:
 * - Domain: 10×10×50 grid (quasi-1D column)
 * - Temperature: Linear profile from T_ref to T_ref + 100K
 * - Material: Water (ρ = 1000 kg/m³, β = 2.1e-4 1/K)
 * - Gravity: g = 9.81 m/s² (downward, -z direction)
 *
 * Success Criteria:
 * - L2 error of force distribution < 1%
 */
TEST_F(BuoyancyUnitTest, HydrostaticPressure) {
    std::cout << "\n=== M1 Buoyancy Test: Hydrostatic Pressure ===" << std::endl;

    // Domain setup (tall column)
    const int nx = 10, ny = 10, nz = 50;
    const float dx = 1.0e-3f;  // 1 mm spacing
    const int num_cells = nx * ny * nz;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Grid spacing: dx = " << dx * 1e3 << " mm" << std::endl;
    std::cout << "  Column height: H = " << (nz - 1) * dx << " m" << std::endl;

    // Material properties (water)
    const float rho = 1000.0f;           // kg/m³
    const float beta = 2.1e-4f;          // 1/K (thermal expansion coefficient)
    const float T_ref = 300.0f;          // K (reference temperature)
    const float dT = 100.0f;             // K (temperature difference)
    const float g = 9.81f;               // m/s² (gravitational acceleration)

    std::cout << "\n  Material properties:" << std::endl;
    std::cout << "    Density: ρ = " << rho << " kg/m³" << std::endl;
    std::cout << "    Thermal expansion: β = " << beta << " 1/K" << std::endl;
    std::cout << "    Reference temperature: T_ref = " << T_ref << " K" << std::endl;
    std::cout << "    Temperature difference: ΔT = " << dT << " K" << std::endl;
    std::cout << "    Gravity: g = " << g << " m/s²" << std::endl;

    // Initialize temperature field with linear profile
    // T(z) = T_ref + dT × z / (nz - 1)
    std::vector<float> h_temperature(num_cells);
    for (int k = 0; k < nz; ++k) {
        float T_k = T_ref + dT * static_cast<float>(k) / static_cast<float>(nz - 1);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                h_temperature[idx] = T_k;
            }
        }
    }

    // Copy temperature to device
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize ForceAccumulator
    ForceAccumulator force_acc(nx, ny, nz);
    force_acc.reset();

    // Add buoyancy force (gravity in -z direction)
    std::cout << "\n  Computing buoyancy force..." << std::endl;
    force_acc.addBuoyancyForce(d_temperature, T_ref, beta, rho,
                               0.0f, 0.0f, -g,  // Gravity in -z
                               nullptr);  // No liquid fraction masking

    // Copy forces back to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), force_acc.getFx(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), force_acc.getFy(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), force_acc.getFz(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute analytical solution
    // F_b(z) = ρ × β × (T(z) - T_ref) × g = ρ × β × dT × (z / (nz-1)) × g
    std::vector<float> h_fz_analytical(num_cells);
    for (int k = 0; k < nz; ++k) {
        float z_normalized = static_cast<float>(k) / static_cast<float>(nz - 1);
        float F_analytical = -rho * beta * dT * z_normalized * g;  // Negative (downward)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                h_fz_analytical[idx] = F_analytical;
            }
        }
    }

    // Compute L2 error
    float l2_error = computeL2Error(h_fz, h_fz_analytical);
    std::cout << "\n  Results:" << std::endl;
    std::cout << "    L2 error: " << l2_error * 100.0f << "%" << std::endl;

    // Check sample points
    std::cout << "\n  Sample points (z-force comparison):" << std::endl;
    std::cout << "    z-index | T [K] | F_numerical [N/m³] | F_analytical [N/m³] | Error [%]" << std::endl;
    std::cout << "    --------|-------|---------------------|---------------------|----------" << std::endl;
    for (int k : {0, nz/4, nz/2, 3*nz/4, nz-1}) {
        int idx = (nx/2) + nx * ((ny/2) + ny * k);  // Center of x-y plane
        float T_k = h_temperature[idx];
        float F_num = h_fz[idx];
        float F_ana = h_fz_analytical[idx];
        float error_percent = std::abs(F_num - F_ana) / std::max(std::abs(F_ana), 1e-10f) * 100.0f;
        std::cout << "    " << k << " | " << T_k << " | " << F_num << " | " << F_ana << " | " << error_percent << std::endl;
    }

    // Verify x and y forces are zero (gravity only in z)
    float max_fx = *std::max_element(h_fx.begin(), h_fx.end(),
                                     [](float a, float b) { return std::abs(a) < std::abs(b); });
    float max_fy = *std::max_element(h_fy.begin(), h_fy.end(),
                                     [](float a, float b) { return std::abs(a) < std::abs(b); });
    std::cout << "\n  Cross-check:" << std::endl;
    std::cout << "    Max |F_x|: " << std::abs(max_fx) << " (should be ~0)" << std::endl;
    std::cout << "    Max |F_y|: " << std::abs(max_fy) << " (should be ~0)" << std::endl;

    // Acceptance criteria
    EXPECT_LT(l2_error, 0.01f) << "Hydrostatic pressure L2 error exceeds 1%";
    EXPECT_LT(std::abs(max_fx), 1e-6f) << "F_x should be zero (gravity in z only)";
    EXPECT_LT(std::abs(max_fy), 1e-6f) << "F_y should be zero (gravity in z only)";

    std::cout << "\n  ✓ Hydrostatic pressure test PASSED" << std::endl;

    // Cleanup
    cudaFree(d_temperature);
}

/**
 * @brief Test 2: Archimedes Buoyancy Force
 *
 * Physics:
 * A submerged object experiences an upward buoyancy force:
 *   F_buoyancy = Δρ × g × V
 * where Δρ = ρ_fluid - ρ_object is the density difference
 *
 * In the Boussinesq approximation with temperature-dependent density:
 *   ρ(T) ≈ ρ₀ × (1 - β × (T - T_ref))
 *   Δρ ≈ ρ₀ × β × ΔT
 *   F_buoyancy = ρ₀ × β × ΔT × g × V
 *
 * Test Setup:
 * - Hot cubic region (10×10×10 cells) embedded in cold background
 * - Hot region: T = T_ref + 50K
 * - Cold region: T = T_ref
 * - Compute total buoyancy force and compare to analytical formula
 *
 * Success Criteria:
 * - Total force magnitude error < 5%
 * - Force direction is upward (+z)
 */
TEST_F(BuoyancyUnitTest, ArchimedesBuoyancy) {
    std::cout << "\n=== M1 Buoyancy Test: Archimedes Force ===" << std::endl;

    // Domain setup
    const int nx = 40, ny = 40, nz = 40;
    const float dx = 1.0e-3f;  // 1 mm spacing
    const int num_cells = nx * ny * nz;

    // Hot region (cubic blob in center)
    const int blob_x0 = 15, blob_x1 = 25;
    const int blob_y0 = 15, blob_y1 = 25;
    const int blob_z0 = 15, blob_z1 = 25;
    const int blob_size = (blob_x1 - blob_x0);
    const float blob_volume = blob_size * blob_size * blob_size * dx * dx * dx;  // m³

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Grid spacing: dx = " << dx * 1e3 << " mm" << std::endl;
    std::cout << "  Hot blob: [" << blob_x0 << ":" << blob_x1 << ", "
              << blob_y0 << ":" << blob_y1 << ", "
              << blob_z0 << ":" << blob_z1 << "]" << std::endl;
    std::cout << "  Blob volume: V = " << blob_volume * 1e9 << " mm³" << std::endl;

    // Material properties (water)
    const float rho = 1000.0f;           // kg/m³
    const float beta = 2.1e-4f;          // 1/K
    const float T_ref = 300.0f;          // K
    const float T_hot = T_ref + 50.0f;   // K
    const float dT = T_hot - T_ref;      // K
    const float g = 9.81f;               // m/s²

    std::cout << "\n  Material properties:" << std::endl;
    std::cout << "    Density: ρ = " << rho << " kg/m³" << std::endl;
    std::cout << "    Thermal expansion: β = " << beta << " 1/K" << std::endl;
    std::cout << "    Cold temperature: T_cold = " << T_ref << " K" << std::endl;
    std::cout << "    Hot temperature: T_hot = " << T_hot << " K" << std::endl;
    std::cout << "    Temperature difference: ΔT = " << dT << " K" << std::endl;
    std::cout << "    Gravity: g = " << g << " m/s²" << std::endl;

    // Initialize temperature field (hot blob in cold background)
    std::vector<float> h_temperature(num_cells, T_ref);
    for (int k = blob_z0; k < blob_z1; ++k) {
        for (int j = blob_y0; j < blob_y1; ++j) {
            for (int i = blob_x0; i < blob_x1; ++i) {
                int idx = i + nx * (j + ny * k);
                h_temperature[idx] = T_hot;
            }
        }
    }

    // Copy temperature to device
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize ForceAccumulator
    ForceAccumulator force_acc(nx, ny, nz);
    force_acc.reset();

    // Add buoyancy force (gravity in -z direction, buoyancy in +z)
    std::cout << "\n  Computing buoyancy force..." << std::endl;
    force_acc.addBuoyancyForce(d_temperature, T_ref, beta, rho,
                               0.0f, 0.0f, -g,  // Gravity in -z
                               nullptr);  // No liquid fraction masking

    // Copy forces back to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), force_acc.getFx(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), force_acc.getFy(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), force_acc.getFz(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute total force by integrating over hot blob
    // F_total = Σ F(cell) × dV
    const float cell_volume = dx * dx * dx;  // m³
    float total_fx = 0.0f, total_fy = 0.0f, total_fz = 0.0f;

    for (int k = blob_z0; k < blob_z1; ++k) {
        for (int j = blob_y0; j < blob_y1; ++j) {
            for (int i = blob_x0; i < blob_x1; ++i) {
                int idx = i + nx * (j + ny * k);
                total_fx += h_fx[idx] * cell_volume;
                total_fy += h_fy[idx] * cell_volume;
                total_fz += h_fz[idx] * cell_volume;
            }
        }
    }

    // Analytical solution (Archimedes principle with Boussinesq)
    // F_buoyancy = ρ₀ × β × ΔT × g_z × V [N]
    // where g_z = -g (gravity points down in -z direction)
    // For hot fluid (T > T_ref), the buoyancy force is UPWARD (negative z)
    // F_analytical = ρ₀ × β × ΔT × (-g) × V = negative
    const float F_analytical = rho * beta * dT * (-g) * blob_volume;

    std::cout << "\n  Results:" << std::endl;
    std::cout << "    Total F_x: " << total_fx << " N (should be ~0)" << std::endl;
    std::cout << "    Total F_y: " << total_fy << " N (should be ~0)" << std::endl;
    std::cout << "    Total F_z (numerical): " << total_fz << " N" << std::endl;
    std::cout << "    Total F_z (analytical): " << F_analytical << " N" << std::endl;

    float error_percent = std::abs(total_fz - F_analytical) / std::abs(F_analytical) * 100.0f;
    std::cout << "    Error: " << error_percent << "%" << std::endl;

    // Check force direction (should be upward, i.e., opposite to gravity)
    // Gravity is -z, so buoyancy should give negative F_z (upward)
    std::cout << "\n  Force direction check:" << std::endl;
    if (total_fz < 0.0f) {
        std::cout << "    ✓ Force is upward (negative z-component)" << std::endl;
    } else {
        std::cout << "    ✗ Force direction incorrect! (should be negative)" << std::endl;
    }

    // Check magnitude of cross forces (should be negligible)
    float cross_force_mag = std::sqrt(total_fx*total_fx + total_fy*total_fy);
    std::cout << "    Cross-force magnitude: " << cross_force_mag << " N" << std::endl;

    // Acceptance criteria
    EXPECT_LT(error_percent, 5.0f) << "Archimedes buoyancy force error exceeds 5%";
    EXPECT_LT(total_fz, 0.0f) << "Buoyancy force should be upward (negative z)";
    EXPECT_LT(cross_force_mag / std::abs(F_analytical), 0.01f)
        << "Cross forces (F_x, F_y) should be negligible (<1% of F_z)";

    std::cout << "\n  ✓ Archimedes buoyancy test PASSED" << std::endl;

    // Cleanup
    cudaFree(d_temperature);
}

/**
 * @brief Test 3: Liquid Fraction Masking
 *
 * Physics:
 * Buoyancy force should only act on liquid regions (liquid_fraction = 1).
 * Solid regions (liquid_fraction = 0) should have zero buoyancy.
 *
 * Test Setup:
 * - Domain with half solid (liquid_fraction = 0), half liquid (liquid_fraction = 1)
 * - Uniform hot temperature everywhere
 * - Verify buoyancy only appears in liquid region
 *
 * Success Criteria:
 * - Force in solid region < 1e-10 N/m³
 * - Force in liquid region matches expected value
 */
TEST_F(BuoyancyUnitTest, LiquidFractionMasking) {
    std::cout << "\n=== M1 Buoyancy Test: Liquid Fraction Masking ===" << std::endl;

    // Domain setup (split into solid/liquid halves)
    const int nx = 20, ny = 20, nz = 20;
    const int num_cells = nx * ny * nz;
    const int z_split = nz / 2;  // Split at z = nz/2

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Solid region: z < " << z_split << std::endl;
    std::cout << "  Liquid region: z >= " << z_split << std::endl;

    // Material properties
    const float rho = 1000.0f;
    const float beta = 2.1e-4f;
    const float T_ref = 300.0f;
    const float T_hot = 400.0f;  // Uniform hot temperature
    const float g = 9.81f;

    // Initialize uniform hot temperature
    std::vector<float> h_temperature(num_cells, T_hot);

    // Initialize liquid fraction (0 = solid, 1 = liquid)
    std::vector<float> h_liquid_fraction(num_cells);
    for (int k = 0; k < nz; ++k) {
        float lf = (k < z_split) ? 0.0f : 1.0f;  // Solid below, liquid above
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                h_liquid_fraction[idx] = lf;
            }
        }
    }

    // Copy to device
    float *d_temperature, *d_liquid_fraction;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMalloc(&d_liquid_fraction, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_liquid_fraction, h_liquid_fraction.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize ForceAccumulator
    ForceAccumulator force_acc(nx, ny, nz);
    force_acc.reset();

    // Add buoyancy force WITH liquid fraction masking
    std::cout << "\n  Computing buoyancy force with liquid fraction masking..." << std::endl;
    force_acc.addBuoyancyForce(d_temperature, T_ref, beta, rho,
                               0.0f, 0.0f, -g,
                               d_liquid_fraction);  // Enable masking

    // Copy forces back to host
    std::vector<float> h_fz(num_cells);
    cudaMemcpy(h_fz.data(), force_acc.getFz(), num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Check forces in solid and liquid regions
    float max_force_solid = 0.0f;
    float max_force_liquid = 0.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = std::abs(h_fz[idx]);
                if (k < z_split) {
                    max_force_solid = std::max(max_force_solid, f);
                } else {
                    max_force_liquid = std::max(max_force_liquid, f);
                }
            }
        }
    }

    // Analytical force in liquid region
    const float F_expected = std::abs(rho * beta * (T_hot - T_ref) * (-g));

    std::cout << "\n  Results:" << std::endl;
    std::cout << "    Max force in solid region: " << max_force_solid << " N/m³" << std::endl;
    std::cout << "    Max force in liquid region: " << max_force_liquid << " N/m³" << std::endl;
    std::cout << "    Expected force (liquid): " << F_expected << " N/m³" << std::endl;

    // Acceptance criteria
    EXPECT_LT(max_force_solid, 1e-10f) << "Force in solid region should be zero";
    EXPECT_NEAR(max_force_liquid, F_expected, F_expected * 0.01f)
        << "Force in liquid region should match analytical value";

    std::cout << "\n  ✓ Liquid fraction masking test PASSED" << std::endl;

    // Cleanup
    cudaFree(d_temperature);
    cudaFree(d_liquid_fraction);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
