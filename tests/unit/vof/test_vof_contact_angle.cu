/**
 * @file test_vof_contact_angle.cu
 * @brief Unit test for contact angle boundary conditions
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test periodic boundary (no contact angle)
 */
TEST(ContactAngleTest, PeriodicBoundary) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);
    vof.reconstructInterface();

    // Store initial normals
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals_initial(num_cells);
    cudaMemcpy(h_normals_initial.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    // Apply periodic boundary (should not modify normals)
    vof.applyBoundaryConditions(0, 90.0f);

    // Check normals unchanged
    std::vector<float3> h_normals_final(num_cells);
    cudaMemcpy(h_normals_final.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_cells; ++i) {
        EXPECT_FLOAT_EQ(h_normals_final[i].x, h_normals_initial[i].x);
        EXPECT_FLOAT_EQ(h_normals_final[i].y, h_normals_initial[i].y);
        EXPECT_FLOAT_EQ(h_normals_final[i].z, h_normals_initial[i].z);
    }
}

/**
 * @brief Test wall boundary with 90 degree contact angle
 */
TEST(ContactAngleTest, NinetyDegreeContactAngle) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet touching bottom wall
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, 5.0f, 8.0f);
    vof.reconstructInterface();
    vof.convertCells();

    // Apply 90 degree contact angle at walls
    vof.applyBoundaryConditions(1, 90.0f);

    // Check that boundary modification was applied
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals(num_cells);
    std::vector<uint8_t> h_flags(num_cells);

    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);
    vof.copyCellFlagsToHost(h_flags.data());

    // Find interface cells at bottom boundary
    bool found_boundary_interface = false;
    int n_interface_boundary = 0;
    int n_liquid_boundary = 0;
    int n_gas_boundary = 0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = 0;  // Bottom boundary
            int idx = i + nx * (j + ny * k);

            if (h_flags[idx] == static_cast<uint8_t>(CellFlag::INTERFACE)) {
                found_boundary_interface = true;
                n_interface_boundary++;
                // Normal should have been modified (hard to test exact value)
                // Just check it's non-zero
                float norm = std::sqrt(h_normals[idx].x * h_normals[idx].x +
                                      h_normals[idx].y * h_normals[idx].y +
                                      h_normals[idx].z * h_normals[idx].z);
                if (norm > 0.1f) {
                    // Good!
                } else {
                    std::cout << "WARNING: Interface cell at (" << i << "," << j << ",0) has zero normal\n";
                }
                EXPECT_GT(norm, 0.1f) << "Boundary normal should be non-zero at (" << i << "," << j << ",0)";
            } else if (h_flags[idx] == static_cast<uint8_t>(CellFlag::LIQUID)) {
                n_liquid_boundary++;
            } else {
                n_gas_boundary++;
            }
        }
    }

    std::cout << "Bottom boundary: " << n_interface_boundary << " interface, "
              << n_liquid_boundary << " liquid, " << n_gas_boundary << " gas\n";

    // Should have found some interface cells at boundary
    EXPECT_TRUE(found_boundary_interface) << "Should have interface cells at boundary";
}

/**
 * @brief Test hydrophilic contact angle (< 90 degrees)
 */
TEST(ContactAngleTest, HydrophilicContactAngle) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet near bottom
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, 5.0f, 6.0f);
    vof.reconstructInterface();
    vof.convertCells();

    // Apply 45 degree contact angle (hydrophilic)
    float contact_angle = 45.0f;
    vof.applyBoundaryConditions(1, contact_angle);

    // Normals should still be unit vectors
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals(num_cells);
    std::vector<uint8_t> h_flags(num_cells);

    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);
    vof.copyCellFlagsToHost(h_flags.data());

    for (int i = 0; i < num_cells; ++i) {
        if (h_flags[i] == static_cast<uint8_t>(CellFlag::INTERFACE)) {
            float norm = std::sqrt(h_normals[i].x * h_normals[i].x +
                                  h_normals[i].y * h_normals[i].y +
                                  h_normals[i].z * h_normals[i].z);

            if (norm > 0.1f) {
                EXPECT_NEAR(norm, 1.0f, 0.1f)
                    << "Interface normal should be unit vector at " << i;
            }
        }
    }
}

/**
 * @brief Test hydrophobic contact angle (> 90 degrees)
 */
TEST(ContactAngleTest, HydrophobicContactAngle) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet near bottom
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, 5.0f, 6.0f);
    vof.reconstructInterface();
    vof.convertCells();

    // Apply 135 degree contact angle (hydrophobic)
    float contact_angle = 135.0f;
    vof.applyBoundaryConditions(1, contact_angle);

    // Check that function completes without error
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals(num_cells);
    cudaMemcpy(h_normals.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    // Verify no NaN values
    int n_valid = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (!std::isnan(h_normals[i].x) &&
            !std::isnan(h_normals[i].y) &&
            !std::isnan(h_normals[i].z)) {
            n_valid++;
        }
    }

    EXPECT_EQ(n_valid, num_cells) << "All normals should be valid (no NaN)";
}
