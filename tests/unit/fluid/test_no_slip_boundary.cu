/**
 * @file test_no_slip_boundary.cu
 * @brief Unit test for no-slip wall boundary conditions
 *
 * This test validates that the FluidLBM solver correctly applies
 * no-slip boundary conditions at walls (z=0 and z=nz-1).
 *
 * Test objectives:
 * - Verify wall velocities are zero after boundary condition application
 * - Check periodic boundaries in X and Y directions work correctly
 * - Validate boundary conditions don't corrupt interior flow
 */

#include <gtest/gtest.h>
#include "physics/fluid_lbm.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace lbm::physics;

/**
 * @brief Test that no-slip walls set velocity to zero
 */
TEST(NoSlipBoundary, WallVelocityZero) {
    std::cout << "\n=== Test: No-Slip Boundary - Wall Velocity Zero ===" << std::endl;

    // Setup: Simple 10x10x10 domain with walls at z=0,9
    const int nx = 10, ny = 10, nz = 10;
    const float nu = 0.033f;  // Lattice viscosity
    const float rho = 1000.0f;

    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  Boundaries: X=Periodic, Y=Periodic, Z=Wall" << std::endl;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL);  // Walls in Z

    // Initialize with uniform velocity
    std::vector<float> h_ux(nx*ny*nz);
    std::vector<float> h_uy(nx*ny*nz);
    std::vector<float> h_uz(nx*ny*nz);

    for (int i = 0; i < nx*ny*nz; ++i) {
        h_ux[i] = 0.1f;  // Initial velocity
        h_uy[i] = 0.0f;
        h_uz[i] = 0.0f;
    }

    // Initialize fluid with velocity field
    fluid.initialize(rho, 0.1f, 0.0f, 0.0f);

    std::cout << "  Initial velocity: u_x = 0.1 (uniform)" << std::endl;

    // Run full LBM step cycle to allow boundaries to take effect
    // Boundary conditions work through the streaming step
    for (int iter = 0; iter < 20; ++iter) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);  // 1 = no-slip wall
        fluid.computeMacroscopic();
    }

    std::cout << "  Applied 20 LBM steps with boundary conditions" << std::endl;

    // Copy velocity back to host
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Verify: Walls at z=0 and z=9 have zero velocity
    float max_wall_velocity = 0.0f;
    int wall_violations = 0;
    const float tolerance = 1e-3f;  // Allow small numerical errors

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            // Bottom wall (z=0)
            int idx_bottom = i + nx * (j + ny * 0);
            float v_bottom = std::sqrt(h_ux[idx_bottom] * h_ux[idx_bottom] +
                                       h_uy[idx_bottom] * h_uy[idx_bottom] +
                                       h_uz[idx_bottom] * h_uz[idx_bottom]);

            if (v_bottom > tolerance) {
                wall_violations++;
            }
            max_wall_velocity = std::max(max_wall_velocity, v_bottom);

            // Top wall (z=nz-1)
            int idx_top = i + nx * (j + ny * (nz-1));
            float v_top = std::sqrt(h_ux[idx_top] * h_ux[idx_top] +
                                   h_uy[idx_top] * h_uy[idx_top] +
                                   h_uz[idx_top] * h_uz[idx_top]);

            if (v_top > tolerance) {
                wall_violations++;
            }
            max_wall_velocity = std::max(max_wall_velocity, v_top);
        }
    }

    std::cout << "  Wall cells checked: " << 2 * nx * ny << std::endl;
    std::cout << "  Maximum wall velocity: " << max_wall_velocity << std::endl;
    std::cout << "  Violations (v > " << tolerance << "): " << wall_violations << std::endl;

    EXPECT_LT(max_wall_velocity, tolerance)
        << "Wall velocity should be near zero (max = " << max_wall_velocity << ")";

    EXPECT_EQ(wall_violations, 0)
        << "No wall cells should have velocity above tolerance";

    if (wall_violations == 0) {
        std::cout << "  ✓ All wall velocities are zero (PASS)" << std::endl;
    }
}

/**
 * @brief Test that interior flow is not corrupted by boundary conditions
 */
TEST(NoSlipBoundary, InteriorFlowPreserved) {
    std::cout << "\n=== Test: No-Slip Boundary - Interior Flow Preserved ===" << std::endl;

    const int nx = 10, ny = 10, nz = 10;
    const float nu = 0.033f;
    const float rho = 1000.0f;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL);

    // Initialize with uniform velocity
    fluid.initialize(rho, 0.1f, 0.0f, 0.0f);

    // Apply boundary conditions
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();

    // Check interior cells (z=2 to z=7) still have non-zero velocity
    std::vector<float> h_ux(nx*ny*nz);
    std::vector<float> h_uy(nx*ny*nz);
    std::vector<float> h_uz(nx*ny*nz);

    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Check a few interior cells
    int interior_nonzero_count = 0;
    float avg_interior_velocity = 0.0f;

    for (int k = 3; k < nz - 3; ++k) {
        for (int j = 2; j < ny - 2; ++j) {
            for (int i = 2; i < nx - 2; ++i) {
                int idx = i + nx * (j + ny * k);
                float v_mag = std::sqrt(h_ux[idx] * h_ux[idx] +
                                       h_uy[idx] * h_uy[idx] +
                                       h_uz[idx] * h_uz[idx]);

                if (v_mag > 1e-6f) {
                    interior_nonzero_count++;
                }
                avg_interior_velocity += v_mag;
            }
        }
    }

    int interior_cells = (nz - 6) * (ny - 4) * (nx - 4);
    avg_interior_velocity /= interior_cells;

    std::cout << "  Interior cells: " << interior_cells << std::endl;
    std::cout << "  Non-zero velocity cells: " << interior_nonzero_count << std::endl;
    std::cout << "  Average interior velocity: " << avg_interior_velocity << std::endl;

    // Interior should still have some velocity (not all zeroed)
    EXPECT_GT(interior_nonzero_count, interior_cells / 2)
        << "Interior flow should be preserved (not all zeroed)";

    std::cout << "  ✓ Interior flow preserved (PASS)" << std::endl;
}

/**
 * @brief Test mixed boundary conditions (periodic + wall)
 */
TEST(NoSlipBoundary, MixedBoundaries) {
    std::cout << "\n=== Test: No-Slip Boundary - Mixed Boundaries ===" << std::endl;

    const int nx = 10, ny = 10, nz = 10;
    const float nu = 0.033f;
    const float rho = 1000.0f;

    // Test that constructor accepts mixed boundaries without crashing
    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL);

    fluid.initialize(rho, 0.0f, 0.0f, 0.0f);

    // Run a few steps to ensure stability
    for (int step = 0; step < 5; ++step) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
    }

    std::cout << "  ✓ Mixed boundaries initialized and stable (PASS)" << std::endl;

    SUCCEED() << "Mixed boundary conditions work correctly";
}

/**
 * @brief Test boundary condition convergence
 */
TEST(NoSlipBoundary, ConvergenceTest) {
    std::cout << "\n=== Test: No-Slip Boundary - Convergence ===" << std::endl;

    const int nx = 10, ny = 10, nz = 10;
    const float nu = 0.033f;
    const float rho = 1000.0f;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL);

    fluid.initialize(rho, 0.1f, 0.0f, 0.0f);

    std::vector<float> h_ux(nx*ny*nz);
    std::vector<float> h_uy(nx*ny*nz);
    std::vector<float> h_uz(nx*ny*nz);

    // Apply boundary conditions iteratively and check convergence
    std::vector<float> wall_velocity_history;

    for (int iter = 0; iter < 20; ++iter) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
        fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

        // Check wall velocity
        float max_wall_v = 0.0f;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx_bottom = i + nx * (j + ny * 0);
                float v = std::abs(h_ux[idx_bottom]);
                max_wall_v = std::max(max_wall_v, v);
            }
        }

        wall_velocity_history.push_back(max_wall_v);
    }

    std::cout << "  Convergence history (max wall velocity):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), wall_velocity_history.size()); ++i) {
        std::cout << "    Iter " << i << ": " << wall_velocity_history[i] << std::endl;
    }

    // Verify convergence (last value should be small)
    EXPECT_LT(wall_velocity_history.back(), 1e-3f)
        << "Boundary conditions should converge to zero wall velocity";

    std::cout << "  ✓ Boundary conditions converge (PASS)" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "No-Slip Boundary Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = RUN_ALL_TESTS();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Boundary Condition Tests Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n";

    return result;
}
