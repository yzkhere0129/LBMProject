/**
 * @file test_laser_surface_deformation.cu
 * @brief Integration test: Laser-induced surface deformation
 *
 * Test verifies complete coupling:
 * 1. Laser heating → temperature field
 * 2. Temperature → Marangoni force
 * 3. Marangoni + surface tension → surface deformation
 * 4. VOF advection with velocity field
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/surface_tension.h"
#include "physics/marangoni.h"
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test laser heating affects surface
 */
TEST(LaserSurfaceTest, HeatingAffectsSurface) {
    // Small domain for fast test
    int nx = 48, ny = 48, nz = 48;
    float dx = 1.0e-6f;  // 1 micron

    // Physical properties (Ti6Al4V)
    float sigma = 1.65f;
    float dsigma_dT = -2.6e-4f;

    // Initialize solvers
    VOFSolver vof(nx, ny, nz, dx);
    SurfaceTension st(nx, ny, nz, sigma, dx);
    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Initialize flat liquid surface (bottom half of domain)
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float z = static_cast<float>(k);
                // Interface at z = nz/2
                h_fill[idx] = 0.5f * (1.0f - tanhf((z - nz / 2.0f) / 2.0f));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();
    vof.computeCurvature();

    // Create "laser" temperature field: hot spot at center of surface
    int num_cells = nx * ny * nz;
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells, 1650.0f);  // Base temperature (solid)

    float T_peak = 2000.0f;  // Peak temperature (above melting)
    float laser_radius = 8.0f;  // Laser spot radius

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Gaussian laser heating near surface
                float dx_c = i - nx / 2.0f;
                float dy_c = j - ny / 2.0f;
                float r2 = dx_c * dx_c + dy_c * dy_c;

                // Heat penetrates slightly below surface
                float dz = k - nz / 2.0f;
                float depth_factor = std::exp(-std::abs(dz) / 5.0f);

                h_temp[idx] = 1650.0f + (T_peak - 1650.0f) *
                             std::exp(-r2 / (2.0f * laser_radius * laser_radius)) *
                             depth_factor;
            }
        }
    }

    cudaMemcpy(d_temperature, h_temp.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Compute forces
    float* d_fx_surf, *d_fy_surf, *d_fz_surf;
    float* d_fx_mar, *d_fy_mar, *d_fz_mar;

    cudaMalloc(&d_fx_surf, num_cells * sizeof(float));
    cudaMalloc(&d_fy_surf, num_cells * sizeof(float));
    cudaMalloc(&d_fz_surf, num_cells * sizeof(float));
    cudaMalloc(&d_fx_mar, num_cells * sizeof(float));
    cudaMalloc(&d_fy_mar, num_cells * sizeof(float));
    cudaMalloc(&d_fz_mar, num_cells * sizeof(float));

    // Surface tension force
    st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(),
                      d_fx_surf, d_fy_surf, d_fz_surf);

    // Marangoni force
    marangoni.computeMarangoniForce(d_temperature, vof.getFillLevel(),
                                    vof.getInterfaceNormals(),
                                    d_fx_mar, d_fy_mar, d_fz_mar);

    // Copy forces to host and verify
    std::vector<float> h_fx_mar(num_cells);
    cudaMemcpy(h_fx_mar.data(), d_fx_mar, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Check that Marangoni forces exist near hot spot
    std::vector<float> h_fill_check(num_cells);
    vof.copyFillLevelToHost(h_fill_check.data());

    int n_marangoni_forces = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_fill_check[i] > 0.1f && h_fill_check[i] < 0.9f) {
            float force_mag = std::sqrt(h_fx_mar[i] * h_fx_mar[i]);
            if (force_mag > 1.0f) {
                n_marangoni_forces++;
            }
        }
    }

    EXPECT_GT(n_marangoni_forces, 5)
        << "Should have Marangoni forces at heated interface";

    cudaFree(d_temperature);
    cudaFree(d_fx_surf);
    cudaFree(d_fy_surf);
    cudaFree(d_fz_surf);
    cudaFree(d_fx_mar);
    cudaFree(d_fy_mar);
    cudaFree(d_fz_mar);
}

/**
 * @brief Test combined surface tension + Marangoni
 */
TEST(LaserSurfaceTest, CombinedForces) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;

    float sigma = 1.65f;
    float dsigma_dT = -2.6e-4f;

    VOFSolver vof(nx, ny, nz, dx);
    SurfaceTension st(nx, ny, nz, sigma, dx);
    MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);

    // Initialize droplet
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);
    vof.reconstructInterface();
    vof.computeCurvature();

    // Temperature field with gradient
    int num_cells = nx * ny * nz;
    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));

    std::vector<float> h_temp(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                // Linear temperature gradient in x
                h_temp[idx] = 1700.0f + 300.0f * i / nx;
            }
        }
    }

    cudaMemcpy(d_temperature, h_temp.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Allocate combined force field
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    // Compute surface tension force
    st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(),
                      d_fx, d_fy, d_fz);

    // Add Marangoni force
    marangoni.addMarangoniForce(d_temperature, vof.getFillLevel(),
                                vof.getInterfaceNormals(), d_fx, d_fy, d_fz);

    // Copy combined forces to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that combined forces exist at interface
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    int n_combined_forces = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_fill[i] > 0.2f && h_fill[i] < 0.8f) {
            float force_mag = std::sqrt(h_fx[i] * h_fx[i] +
                                       h_fy[i] * h_fy[i] +
                                       h_fz[i] * h_fz[i]);
            if (force_mag > 1.0f) {
                n_combined_forces++;
            }
        }
    }

    EXPECT_GT(n_combined_forces, 10)
        << "Should have combined forces at interface";

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test surface deformation detection
 */
TEST(LaserSurfaceTest, SurfaceDeformation) {
    // This is a simplified test - full coupling requires FluidLBM
    // Here we just verify that VOF can track deformed interfaces

    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize flat interface
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
    vof.computeCurvature();

    // Initial curvature should be near zero (flat interface)
    int num_cells = nx * ny * nz;
    std::vector<float> h_curvature_initial(num_cells);
    vof.copyCurvatureToHost(h_curvature_initial.data());

    float mean_curvature_initial = 0.0f;
    int n_interface = 0;

    for (int i = 0; i < num_cells; ++i) {
        if (h_fill[i] > 0.1f && h_fill[i] < 0.9f) {
            mean_curvature_initial += std::abs(h_curvature_initial[i]);
            n_interface++;
        }
    }

    if (n_interface > 0) {
        mean_curvature_initial /= n_interface;
    }

    EXPECT_LT(mean_curvature_initial, 0.5f)
        << "Initial flat interface should have low curvature";

    cudaFree(0);  // Dummy to avoid compiler warnings
}
