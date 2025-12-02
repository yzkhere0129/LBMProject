/**
 * @file test_vof_surface_tension.cu
 * @brief Unit test for CSF surface tension force
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/surface_tension.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test that surface tension force is zero for planar interface
 */
TEST(SurfaceTensionTest, PlanarInterfaceZeroForce) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;  // 1 micron
    float sigma = 1.65f;  // Ti6Al4V surface tension [N/m]

    VOFSolver vof(nx, ny, nz, dx);
    SurfaceTension st(nx, ny, nz, sigma, dx);

    // Initialize planar interface
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
    vof.computeCurvature();

    // Compute surface tension force
    int num_cells = nx * ny * nz;
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(), d_fx, d_fy, d_fz);

    // Copy forces to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that forces are nearly zero (planar interface has zero curvature)
    int mid_x = nx / 2;
    int mid_y = ny / 2;
    int mid_z = nz / 2;
    int idx = mid_x + nx * (mid_y + ny * mid_z);

    EXPECT_NEAR(h_fx[idx], 0.0f, 1.0f) << "Force should be small for planar interface";
    EXPECT_NEAR(h_fy[idx], 0.0f, 1.0f);
    EXPECT_NEAR(h_fz[idx], 0.0f, 1.0f);

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test CSF force for spherical droplet
 */
TEST(SurfaceTensionTest, SphericalDropletForce) {
    int nx = 64, ny = 64, nz = 64;
    float dx = 1.0e-6f;  // 1 micron
    float sigma = 1.65f;  // N/m

    VOFSolver vof(nx, ny, nz, dx);
    SurfaceTension st(nx, ny, nz, sigma, dx);

    // Initialize droplet
    float radius = 16.0f;  // lattice units
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, radius);
    vof.reconstructInterface();
    vof.computeCurvature();

    // Compute surface tension force
    int num_cells = nx * ny * nz;
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(), d_fx, d_fy, d_fz);

    // Copy forces to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that forces exist at interface
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    int n_nonzero_forces = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_fill[i] > 0.1f && h_fill[i] < 0.9f) {
            float force_mag = std::sqrt(h_fx[i] * h_fx[i] +
                                        h_fy[i] * h_fy[i] +
                                        h_fz[i] * h_fz[i]);
            if (force_mag > 1.0f) {
                n_nonzero_forces++;
            }
        }
    }

    EXPECT_GT(n_nonzero_forces, 10) << "Should have non-zero forces at interface";

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test Laplace pressure calculation
 */
TEST(SurfaceTensionTest, LaplacePressure) {
    float sigma = 1.65f;  // N/m
    float radius = 10.0e-6f;  // 10 microns

    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;

    SurfaceTension st(nx, ny, nz, sigma, dx);

    // Curvature for sphere: κ = 2/R
    float curvature = 2.0f / radius;

    // Laplace pressure: ΔP = σ * κ
    float pressure_jump = st.computeLaplacePressure(curvature);
    float expected = sigma * curvature;

    EXPECT_FLOAT_EQ(pressure_jump, expected);

    // For R = 10 μm: ΔP = 1.65 * 2 / 10e-6 = 330,000 Pa = 330 kPa
    EXPECT_NEAR(pressure_jump, 330000.0f, 1.0f);
}

/**
 * @brief Test force magnitude scaling with surface tension coefficient
 */
TEST(SurfaceTensionTest, ForceSigmaScaling) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 8.0f);
    vof.reconstructInterface();
    vof.computeCurvature();

    int num_cells = nx * ny * nz;
    float* d_fx1, *d_fy1, *d_fz1;
    float* d_fx2, *d_fy2, *d_fz2;

    cudaMalloc(&d_fx1, num_cells * sizeof(float));
    cudaMalloc(&d_fy1, num_cells * sizeof(float));
    cudaMalloc(&d_fz1, num_cells * sizeof(float));
    cudaMalloc(&d_fx2, num_cells * sizeof(float));
    cudaMalloc(&d_fy2, num_cells * sizeof(float));
    cudaMalloc(&d_fz2, num_cells * sizeof(float));

    // Test with sigma1 = 1.0
    float sigma1 = 1.0f;
    SurfaceTension st1(nx, ny, nz, sigma1, dx);
    st1.computeCSFForce(vof.getFillLevel(), vof.getCurvature(), d_fx1, d_fy1, d_fz1);

    // Test with sigma2 = 2.0
    float sigma2 = 2.0f;
    SurfaceTension st2(nx, ny, nz, sigma2, dx);
    st2.computeCSFForce(vof.getFillLevel(), vof.getCurvature(), d_fx2, d_fy2, d_fz2);

    // Copy forces to host
    std::vector<float> h_fx1(num_cells), h_fx2(num_cells);
    cudaMemcpy(h_fx1.data(), d_fx1, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx2.data(), d_fx2, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that force doubles when sigma doubles
    int test_idx = nx / 2 + nx * (ny / 2 + ny * nz / 2);
    if (std::abs(h_fx1[test_idx]) > 0.1f) {
        float ratio = h_fx2[test_idx] / h_fx1[test_idx];
        EXPECT_NEAR(ratio, 2.0f, 0.1f) << "Force should scale linearly with sigma";
    }

    cudaFree(d_fx1);
    cudaFree(d_fy1);
    cudaFree(d_fz1);
    cudaFree(d_fx2);
    cudaFree(d_fy2);
    cudaFree(d_fz2);
}
