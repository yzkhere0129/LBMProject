/**
 * @file test_vof_reconstruction.cu
 * @brief Unit test for interface reconstruction
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test interface normal computation for planar interface
 */
TEST(VOFReconstructionTest, PlanarInterface) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize planar interface perpendicular to x-axis
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                // Smooth interface using tanh
                float x = static_cast<float>(i);
                float x_interface = nx / 2.0f;
                h_fill[idx] = 0.5f * (1.0f - tanhf((x - x_interface) / 2.0f));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Check interface normals
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals(num_cells);
    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    // At interface (x = nx/2), normal should point in +x direction
    // (from liquid to gas)
    int mid_x = nx / 2;
    int mid_y = ny / 2;
    int mid_z = nz / 2;
    int idx = mid_x + nx * (mid_y + ny * mid_z);

    float3 n = h_normals[idx];

    // Normal should be approximately (1, 0, 0)
    EXPECT_NEAR(n.x, 1.0f, 0.1f) << "Normal x-component should be ~1";
    EXPECT_NEAR(n.y, 0.0f, 0.1f) << "Normal y-component should be ~0";
    EXPECT_NEAR(n.z, 0.0f, 0.1f) << "Normal z-component should be ~0";

    // Check normalization
    float norm = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    EXPECT_NEAR(norm, 1.0f, 0.01f) << "Normal should be unit vector";
}

/**
 * @brief Test interface normal for spherical interface
 */
TEST(VOFReconstructionTest, SphericalInterface) {
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

    // Check normals at interface
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals(num_cells);
    std::vector<float> h_fill(num_cells);

    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);
    vof.copyFillLevelToHost(h_fill.data());

    // Check several points on the sphere
    int n_test_points = 0;
    int n_correct_directions = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Only check interface cells
                if (h_fill[idx] < 0.1f || h_fill[idx] > 0.9f) continue;

                float3 n = h_normals[idx];
                float norm = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

                // Skip if normal is too small
                if (norm < 0.1f) continue;

                n_test_points++;

                // Radial direction from center
                float dx_r = i - cx;
                float dy_r = j - cy;
                float dz_r = k - cz;
                float r = std::sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r);

                if (r < 0.1f) continue;

                // Normalize radial vector
                dx_r /= r;
                dy_r /= r;
                dz_r /= r;

                // Normal should align with radial direction
                float dot_product = n.x * dx_r + n.y * dy_r + n.z * dz_r;

                // Check alignment (allow some tolerance)
                if (std::abs(dot_product) > 0.7f) {
                    n_correct_directions++;
                }
            }
        }
    }

    // At least 80% of interface normals should point radially
    ASSERT_GT(n_test_points, 10) << "Should have interface points";
    float correct_fraction = static_cast<float>(n_correct_directions) / n_test_points;
    EXPECT_GT(correct_fraction, 0.8f)
        << "Normal directions: " << n_correct_directions << "/" << n_test_points;
}

/**
 * @brief Test that normals are zero in bulk regions
 */
TEST(VOFReconstructionTest, BulkRegionsZeroNormal) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize with pure liquid in left half, pure gas in right half
    std::vector<float> h_fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx / 2 - 2; ++i) {
                h_fill[i + nx * (j + ny * k)] = 1.0f;
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Check normals in bulk liquid region (far from interface)
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals(num_cells);
    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    // Check bulk liquid (x < nx/4)
    int test_x = nx / 4;
    int test_y = ny / 2;
    int test_z = nz / 2;
    int idx = test_x + nx * (test_y + ny * test_z);

    float3 n = h_normals[idx];
    float norm = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

    EXPECT_LT(norm, 0.1f) << "Normal in bulk liquid should be nearly zero";

    // Check bulk gas (x > 3*nx/4)
    test_x = 3 * nx / 4;
    idx = test_x + nx * (test_y + ny * test_z);

    n = h_normals[idx];
    norm = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

    EXPECT_LT(norm, 0.1f) << "Normal in bulk gas should be nearly zero";
}
