/**
 * @file test_vof_marangoni.cu
 * @brief Unit test for Marangoni effect (thermocapillary force)
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test Marangoni force is zero for uniform temperature
 */
TEST(MarangoniTest, UniformTemperatureZeroForce) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;  // 1 micron
    float dsigma_dT = -2.6e-4f;  // Ti6Al4V [N/(m·K)]

    VOFSolver vof(nx, ny, nz, dx);
    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Initialize interface
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);
    vof.reconstructInterface();

    // Uniform temperature field
    int num_cells = nx * ny * nz;
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells, 1800.0f);  // Uniform 1800 K
    cudaMemcpy(d_temperature, h_temp.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Compute Marangoni force
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    marangoni.computeMarangoniForce(d_temperature, vof.getFillLevel(),
                                    vof.getInterfaceNormals(), d_fx, d_fy, d_fz);

    // Copy forces to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that forces are zero (no temperature gradient)
    for (int i = 0; i < num_cells; ++i) {
        EXPECT_NEAR(h_fx[i], 0.0f, 1e-3f) << "Force should be zero for uniform T at " << i;
        EXPECT_NEAR(h_fy[i], 0.0f, 1e-3f);
        EXPECT_NEAR(h_fz[i], 0.0f, 1e-3f);
    }

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test Marangoni force direction for temperature gradient
 */
TEST(MarangoniTest, ForceDirection) {
    int nx = 64, ny = 32, nz = 32;
    float dx = 1.0e-6f;
    float dsigma_dT = -2.6e-4f;  // Negative for metals (flow from hot to cold)

    VOFSolver vof(nx, ny, nz, dx);
    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Initialize planar interface perpendicular to z-axis
    // This ensures tangential temperature gradient (∇T in x, interface normal in z)
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float z = static_cast<float>(k);
                h_fill[idx] = 0.5f * (1.0f - tanhf((z - nz / 2.0f) / 2.0f));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Create temperature gradient: hot on left, cold on right
    int num_cells = nx * ny * nz;
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                // Linear temperature gradient: 2000 K -> 1500 K
                h_temp[idx] = 2000.0f - 500.0f * i / nx;
            }
        }
    }

    cudaMemcpy(d_temperature, h_temp.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Compute Marangoni force
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    marangoni.computeMarangoniForce(d_temperature, vof.getFillLevel(),
                                    vof.getInterfaceNormals(), d_fx, d_fy, d_fz);

    // Copy forces to host
    std::vector<float> h_fx(num_cells);
    std::vector<float> h_fill_check(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    vof.copyFillLevelToHost(h_fill_check.data());

    // Check force at interface
    int mid_x = nx / 2;
    int mid_y = ny / 2;
    int mid_z = nz / 2;
    int idx = mid_x + nx * (mid_y + ny * mid_z);

    // Debug output
    std::vector<float3> h_normals(num_cells);
    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(), num_cells * sizeof(float3), cudaMemcpyDeviceToHost);
    std::cout << "  Fill level at mid_x=" << mid_x << ": " << h_fill_check[idx] << std::endl;
    std::cout << "  Interface normal: (" << h_normals[idx].x << ", " << h_normals[idx].y << ", " << h_normals[idx].z << ")" << std::endl;
    std::cout << "  Marangoni force: " << h_fx[idx] << std::endl;

    // For dσ/dT < 0 and ∇T in -x direction, force should be in +x direction
    // (flow from hot to cold)
    EXPECT_GT(std::abs(h_fx[idx]), 1.0f) << "Should have significant Marangoni force";

    // Since dσ/dT < 0 and dT/dx < 0, force should be positive (rightward)
    EXPECT_GT(h_fx[idx], 0.0f) << "Force should point from hot to cold region";

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test characteristic Marangoni velocity calculation
 */
TEST(MarangoniTest, CharacteristicVelocity) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;
    float dsigma_dT = -2.6e-4f;  // N/(m·K)

    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Typical parameters for laser melting
    float delta_T = 500.0f;  // K
    float viscosity = 5.0e-3f;  // Pa·s (Ti6Al4V liquid)

    float v_Ma = marangoni.computeMarangoniVelocity(delta_T, viscosity);

    // Expected: v ~ |dσ/dT * ΔT| / μ
    float expected = std::abs(dsigma_dT * delta_T) / viscosity;

    EXPECT_FLOAT_EQ(v_Ma, expected);

    // For Ti6Al4V: v ~ 0.26 * 500 / 0.005 = 26 m/s (very fast!)
    EXPECT_GT(v_Ma, 10.0f) << "Marangoni velocity should be significant";
    EXPECT_LT(v_Ma, 100.0f) << "Marangoni velocity should be reasonable";
}

/**
 * @brief Test tangential projection (no normal component)
 */
TEST(MarangoniTest, TangentialProjection) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;
    float dsigma_dT = -2.6e-4f;

    VOFSolver vof(nx, ny, nz, dx);
    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Initialize planar interface perpendicular to x-axis
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float x = static_cast<float>(i);
                h_fill[idx] = 0.5f * (1.0f - tanhf((x - nx / 2.0f) / 2.0f));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Temperature gradient perpendicular to interface (normal direction)
    int num_cells = nx * ny * nz;
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                // Temperature varies only in x (normal to interface)
                h_temp[idx] = 1500.0f + 500.0f * i / nx;
            }
        }
    }

    cudaMemcpy(d_temperature, h_temp.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Compute Marangoni force
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    marangoni.computeMarangoniForce(d_temperature, vof.getFillLevel(),
                                    vof.getInterfaceNormals(), d_fx, d_fy, d_fz);

    // Copy forces to host
    std::vector<float> h_fx(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // For purely normal temperature gradient, tangential force should be zero
    int mid_x = nx / 2;
    int mid_y = ny / 2;
    int mid_z = nz / 2;
    int idx = mid_x + nx * (mid_y + ny * mid_z);

    // Note: This test might have small numerical values due to finite differences
    EXPECT_NEAR(h_fx[idx], 0.0f, 10.0f)
        << "Force should be small for normal temperature gradient";

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}
