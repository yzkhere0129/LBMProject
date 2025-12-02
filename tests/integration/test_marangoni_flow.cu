/**
 * @file test_marangoni_flow.cu
 * @brief Integration test: Marangoni-driven surface flow
 *
 * Test verifies:
 * 1. Temperature gradient drives surface flow
 * 2. Flow direction is from hot to cold (for dσ/dT < 0)
 * 3. Flow magnitude scales with temperature gradient
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test Marangoni flow direction
 */
TEST(MarangoniFlowTest, FlowDirection) {
    // Domain setup
    int nx = 64, ny = 32, nz = 32;
    float dx = 1.0e-6f;  // 1 micron

    // Physical properties
    float dsigma_dT = -2.6e-4f;  // N/(m·K) for Ti6Al4V

    // Initialize solvers
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
    float T_hot = 2000.0f;  // K
    float T_cold = 1500.0f;  // K

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                // Linear temperature gradient
                h_temp[idx] = T_hot - (T_hot - T_cold) * i / (nx - 1);
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
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check force at interface
    std::vector<float> h_fill_check(num_cells);
    vof.copyFillLevelToHost(h_fill_check.data());

    // Sample forces at interface cells
    std::vector<float> interface_fx;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Interface cells
                if (h_fill_check[idx] > 0.1f && h_fill_check[idx] < 0.9f) {
                    if (std::abs(h_fx[idx]) > 1.0f) {
                        interface_fx.push_back(h_fx[idx]);
                    }
                }
            }
        }
    }

    ASSERT_GT(interface_fx.size(), 5) << "Should have Marangoni forces at interface";

    // Compute mean force
    float mean_fx = 0.0f;
    for (float fx : interface_fx) {
        mean_fx += fx;
    }
    mean_fx /= interface_fx.size();

    // Force should be positive (flow from hot to cold for dσ/dT < 0)
    // Temperature decreases in +x, so force should be in +x direction
    EXPECT_GT(mean_fx, 0.0f)
        << "Marangoni force should point from hot to cold (mean fx: " << mean_fx << ")";

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test velocity scaling with temperature gradient
 */
TEST(MarangoniFlowTest, VelocityScaling) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;
    float dsigma_dT = -2.6e-4f;

    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Test two different temperature gradients
    float viscosity = 5.0e-3f;  // Pa·s
    float delta_T1 = 200.0f;  // K
    float delta_T2 = 400.0f;  // K (double)

    float v1 = marangoni.computeMarangoniVelocity(delta_T1, viscosity);
    float v2 = marangoni.computeMarangoniVelocity(delta_T2, viscosity);

    // Velocity should scale linearly with ΔT
    float ratio = v2 / v1;
    EXPECT_NEAR(ratio, 2.0f, 0.01f)
        << "Velocity should scale linearly with ΔT (ratio: " << ratio << ")";
}

/**
 * @brief Test Marangoni force magnitude
 */
TEST(MarangoniFlowTest, ForceMagnitude) {
    int nx = 64, ny = 32, nz = 32;
    float dx = 1.0e-6f;
    float dsigma_dT = -2.6e-4f;

    VOFSolver vof(nx, ny, nz, dx);
    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Initialize interface
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 12.0f);
    vof.reconstructInterface();

    // Steep temperature gradient
    int num_cells = nx * ny * nz;
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells);
    float T_center = 2000.0f;
    float T_edge = 1500.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float dx_c = i - nx / 2.0f;
                float dy_c = j - ny / 2.0f;
                float dz_c = k - nz / 2.0f;
                float r = std::sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c);
                // Radial temperature profile
                h_temp[idx] = T_edge + (T_center - T_edge) * std::exp(-r / 10.0f);
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
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that forces are significant at interface
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    int n_significant_forces = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_fill[i] > 0.1f && h_fill[i] < 0.9f) {
            float force_mag = std::sqrt(h_fx[i] * h_fx[i] +
                                       h_fy[i] * h_fy[i] +
                                       h_fz[i] * h_fz[i]);
            if (force_mag > 10.0f) {  // Significant force
                n_significant_forces++;
            }
        }
    }

    EXPECT_GT(n_significant_forces, 10)
        << "Should have significant Marangoni forces at interface";

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}
