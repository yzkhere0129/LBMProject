/**
 * @file test_vof_curvature.cu
 * @brief Unit test for interface curvature computation
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test curvature of planar interface (should be zero)
 */
TEST(VOFCurvatureTest, PlanarInterfaceZeroCurvature) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize planar interface
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float x = static_cast<float>(i);
                float x_interface = nx / 2.0f;
                h_fill[idx] = 0.5f * (1.0f - tanhf((x - x_interface) / 2.0f));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();
    vof.computeCurvature();

    // Check curvature at interface
    int num_cells = nx * ny * nz;
    std::vector<float> h_curvature(num_cells);
    vof.copyCurvatureToHost(h_curvature.data());

    // Planar interface should have zero curvature
    int mid_x = nx / 2;
    int mid_y = ny / 2;
    int mid_z = nz / 2;
    int idx = mid_x + nx * (mid_y + ny * mid_z);

    EXPECT_NEAR(h_curvature[idx], 0.0f, 0.1f)
        << "Planar interface should have zero curvature";
}

/**
 * @brief Test curvature of spherical interface
 */
TEST(VOFCurvatureTest, SphericalCurvature) {
    int nx = 64, ny = 64, nz = 64;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize spherical droplet
    float cx = nx / 2.0f;
    float cy = ny / 2.0f;
    float cz = nz / 2.0f;
    float radius = 16.0f;

    vof.initializeDroplet(cx, cy, cz, radius);
    vof.reconstructInterface();
    vof.computeCurvature();

    // Theoretical curvature for sphere: κ = 2/R
    float theoretical_curvature = 2.0f / radius;

    // Check curvature at interface cells
    int num_cells = nx * ny * nz;
    std::vector<float> h_curvature(num_cells);
    std::vector<float> h_fill(num_cells);

    vof.copyCurvatureToHost(h_curvature.data());
    vof.copyFillLevelToHost(h_fill.data());

    // Sample curvature values at interface
    std::vector<float> interface_curvatures;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Only check interface cells
                if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
                    if (std::abs(h_curvature[idx]) > 0.01f) {
                        interface_curvatures.push_back(h_curvature[idx]);
                    }
                }
            }
        }
    }

    ASSERT_GT(interface_curvatures.size(), 10) << "Should have interface curvature values";

    // Compute mean curvature
    float mean_curvature = 0.0f;
    for (float kappa : interface_curvatures) {
        mean_curvature += kappa;
    }
    mean_curvature /= interface_curvatures.size();

    // Allow 30% error (curvature is challenging to compute accurately on Cartesian grid)
    float relative_error = std::abs(mean_curvature - theoretical_curvature) / theoretical_curvature;
    EXPECT_LT(relative_error, 0.3f)
        << "Mean curvature: " << mean_curvature
        << ", theoretical: " << theoretical_curvature
        << ", error: " << relative_error * 100 << "%";
}

/**
 * @brief Test that curvature is zero in bulk regions
 */
TEST(VOFCurvatureTest, BulkRegionsZeroCurvature) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize with interface in middle
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx / 2 - 3; ++i) {
                h_fill[i + nx * (j + ny * k)] = 1.0f;
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();
    vof.computeCurvature();

    // Check curvature in bulk regions
    int num_cells = nx * ny * nz;
    std::vector<float> h_curvature(num_cells);
    vof.copyCurvatureToHost(h_curvature.data());

    // Bulk liquid region
    int test_x = nx / 4;
    int test_y = ny / 2;
    int test_z = nz / 2;
    int idx = test_x + nx * (test_y + ny * test_z);

    EXPECT_NEAR(h_curvature[idx], 0.0f, 1e-6f)
        << "Curvature in bulk liquid should be zero";

    // Bulk gas region
    test_x = 3 * nx / 4;
    idx = test_x + nx * (test_y + ny * test_z);

    EXPECT_NEAR(h_curvature[idx], 0.0f, 1e-6f)
        << "Curvature in bulk gas should be zero";
}

/**
 * @brief Test curvature sign convention
 * @note Positive curvature means concave (from liquid side)
 */
TEST(VOFCurvatureTest, SignConvention) {
    int nx = 64, ny = 64, nz = 64;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Liquid droplet (curvature should be positive)
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 12.0f);
    vof.reconstructInterface();
    vof.computeCurvature();

    int num_cells = nx * ny * nz;
    std::vector<float> h_curvature(num_cells);
    std::vector<float> h_fill(num_cells);

    vof.copyCurvatureToHost(h_curvature.data());
    vof.copyFillLevelToHost(h_fill.data());

    // Sample interface curvatures
    int n_positive = 0;
    int n_negative = 0;
    int n_interface = 0;

    for (int i = 0; i < num_cells; ++i) {
        if (h_fill[i] > 0.1f && h_fill[i] < 0.9f) {
            n_interface++;
            if (h_curvature[i] > 0.01f) n_positive++;
            if (h_curvature[i] < -0.01f) n_negative++;
        }
    }

    ASSERT_GT(n_interface, 10) << "Should have interface cells";

    // For a liquid droplet in gas, curvature should be predominantly positive
    EXPECT_GT(n_positive, n_negative)
        << "Droplet curvature should be positive: +"
        << n_positive << ", -" << n_negative;
}
