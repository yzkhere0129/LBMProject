/**
 * @file test_moving_wall_bc.cu
 * @brief Unit test for moving wall boundary conditions
 *
 * This test validates the moving wall boundary condition implementation
 * needed for lid-driven cavity and other canonical flow benchmarks.
 *
 * Physical Background:
 * - Moving wall BC enforces a prescribed velocity at a wall boundary
 * - Essential for lid-driven cavity (moving top wall)
 * - Implemented using Zou-He velocity boundary condition
 *
 * Test objectives:
 * 1. Verify wall velocity is correctly imposed
 * 2. Test momentum balance at moving wall
 * 3. Validate for different wall velocities and directions
 * 4. Check compatibility with existing no-slip walls
 *
 * Reference:
 * Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions
 * for the lattice Boltzmann BGK model. Physics of Fluids, 9(6), 1591-1598.
 */

#include <gtest/gtest.h>
#include "physics/fluid_lbm.h"
#include "core/streaming.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace lbm::physics;
using namespace lbm::core;

class MovingWallBCTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

/**
 * @brief Test that moving wall velocity is correctly imposed
 *
 * Setup: Channel with moving top wall (y=ny-1)
 * Expected: Top wall should have prescribed velocity after BC application
 */
TEST_F(MovingWallBCTest, TopWallVelocityImposed) {
    std::cout << "\n=== Test: Moving Wall - Top Wall Velocity Imposed ===" << std::endl;

    const int nx = 16, ny = 16, nz = 16;
    const float nu = 0.033f;  // Lattice viscosity
    const float rho = 1000.0f;
    const float wall_velocity = 0.1f;  // Moving wall velocity

    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  Wall velocity: " << wall_velocity << " (x-direction)" << std::endl;

    // Create fluid solver with walls in Y direction
    FluidLBM fluid(nx, ny, nz, nu, rho,
                   lbm::physics::BoundaryType::WALL,      // X walls
                   lbm::physics::BoundaryType::WALL,      // Y walls
                   lbm::physics::BoundaryType::WALL,      // Z walls
                   1.0f, 1.0f);  // dt, dx (lattice units)

    // Initialize with zero velocity
    fluid.initialize(rho, 0.0f, 0.0f, 0.0f);

    // Set top wall (y=ny-1) as moving wall
    fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, wall_velocity, 0.0f, 0.0f);

    std::cout << "  Set moving wall at Y_MAX with u_x = " << wall_velocity << std::endl;

    // Run several LBM steps to allow BC to take effect
    for (int step = 0; step < 50; ++step) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);  // Apply boundaries
        fluid.computeMacroscopic();
    }

    std::cout << "  Applied 50 LBM steps" << std::endl;

    // Copy velocity to host
    std::vector<float> h_ux(nx * ny * nz);
    std::vector<float> h_uy(nx * ny * nz);
    std::vector<float> h_uz(nx * ny * nz);

    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Verify top wall velocity
    float max_error = 0.0f;
    float avg_wall_ux = 0.0f;
    int wall_count = 0;

    for (int k = 1; k < nz - 1; ++k) {
        for (int i = 1; i < nx - 1; ++i) {
            int idx = i + nx * ((ny - 1) + ny * k);  // Top wall (y=ny-1)
            float ux = h_ux[idx];
            float error = std::abs(ux - wall_velocity);
            max_error = std::max(max_error, error);
            avg_wall_ux += ux;
            wall_count++;
        }
    }

    avg_wall_ux /= wall_count;

    std::cout << "  Wall nodes checked: " << wall_count << std::endl;
    std::cout << "  Average wall u_x: " << avg_wall_ux << std::endl;
    std::cout << "  Expected u_x: " << wall_velocity << std::endl;
    std::cout << "  Maximum error: " << max_error << std::endl;

    // Allow some tolerance due to compressibility effects and corners
    const float tolerance = 0.05f;
    EXPECT_NEAR(avg_wall_ux, wall_velocity, tolerance)
        << "Average wall velocity should match prescribed value";

    if (std::abs(avg_wall_ux - wall_velocity) < tolerance) {
        std::cout << "  PASS: Wall velocity correctly imposed" << std::endl;
    }
}

/**
 * @brief Test momentum balance at moving wall
 *
 * The moving wall should impart momentum to the fluid
 */
TEST_F(MovingWallBCTest, MomentumImparted) {
    std::cout << "\n=== Test: Moving Wall - Momentum Imparted ===" << std::endl;

    const int nx = 16, ny = 16, nz = 16;
    const float nu = 0.033f;
    const float rho = 1000.0f;
    const float wall_velocity = 0.1f;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL, 1.0f, 1.0f);

    fluid.initialize(rho, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, wall_velocity, 0.0f, 0.0f);

    // Measure momentum before and after
    std::vector<float> h_ux(nx * ny * nz);
    std::vector<float> h_uy(nx * ny * nz);
    std::vector<float> h_uz(nx * ny * nz);

    // Initial momentum (should be near zero)
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    float initial_momentum = 0.0f;
    for (int i = 0; i < nx * ny * nz; ++i) {
        initial_momentum += h_ux[i];
    }

    std::cout << "  Initial total x-momentum: " << initial_momentum << std::endl;

    // Run simulation
    for (int step = 0; step < 100; ++step) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
    }

    // Final momentum (should be positive - fluid dragged by wall)
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    float final_momentum = 0.0f;
    for (int i = 0; i < nx * ny * nz; ++i) {
        final_momentum += h_ux[i];
    }

    std::cout << "  Final total x-momentum: " << final_momentum << std::endl;
    std::cout << "  Momentum increase: " << (final_momentum - initial_momentum) << std::endl;

    // Moving wall should have increased fluid momentum
    EXPECT_GT(final_momentum, initial_momentum)
        << "Moving wall should impart momentum to fluid";

    std::cout << "  PASS: Momentum correctly imparted by moving wall" << std::endl;
}

/**
 * @brief Test different wall velocities
 *
 * Verify BC works for different velocities
 */
TEST_F(MovingWallBCTest, VariableWallVelocity) {
    std::cout << "\n=== Test: Moving Wall - Variable Wall Velocity ===" << std::endl;

    const int nx = 12, ny = 12, nz = 12;
    const float nu = 0.033f;
    const float rho = 1000.0f;

    std::vector<float> test_velocities = {0.05f, 0.1f, 0.15f};

    for (float wall_vel : test_velocities) {
        std::cout << "  Testing wall velocity: " << wall_vel << std::endl;

        FluidLBM fluid(nx, ny, nz, nu, rho,
                       lbm::physics::BoundaryType::WALL,
                       lbm::physics::BoundaryType::WALL,
                       lbm::physics::BoundaryType::WALL, 1.0f, 1.0f);

        fluid.initialize(rho, 0.0f, 0.0f, 0.0f);
        fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, wall_vel, 0.0f, 0.0f);

        // Run simulation
        for (int step = 0; step < 50; ++step) {
            fluid.collisionBGK();
            fluid.streaming();
            fluid.applyBoundaryConditions(1);
            fluid.computeMacroscopic();
        }

        // Check wall velocity
        std::vector<float> h_ux(nx * ny * nz);
        std::vector<float> h_uy(nx * ny * nz);
        std::vector<float> h_uz(nx * ny * nz);
        fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

        float avg_ux = 0.0f;
        int count = 0;
        for (int k = 1; k < nz - 1; ++k) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = i + nx * ((ny - 1) + ny * k);
                avg_ux += h_ux[idx];
                count++;
            }
        }
        avg_ux /= count;

        std::cout << "    Average wall u_x: " << avg_ux << " (expected: " << wall_vel << ")" << std::endl;

        EXPECT_NEAR(avg_ux, wall_vel, 0.05f)
            << "Wall velocity should match for u_wall = " << wall_vel;
    }

    std::cout << "  PASS: All wall velocities correctly imposed" << std::endl;
}

/**
 * @brief Test moving wall in different directions
 *
 * Verify BC works for different wall orientations
 */
TEST_F(MovingWallBCTest, DifferentWallDirections) {
    std::cout << "\n=== Test: Moving Wall - Different Directions ===" << std::endl;

    const int nx = 12, ny = 12, nz = 12;
    const float nu = 0.033f;
    const float rho = 1000.0f;
    const float wall_velocity = 0.1f;

    struct WallTest {
        unsigned int direction;
        std::string name;
        int check_coord;
        int check_dim;  // 0=x, 1=y, 2=z
    };

    std::vector<WallTest> tests = {
        {Streaming::BOUNDARY_X_MAX, "X_MAX (right wall)", nx - 1, 0},
        {Streaming::BOUNDARY_Y_MAX, "Y_MAX (top wall)", ny - 1, 1},
        {Streaming::BOUNDARY_Z_MAX, "Z_MAX (front wall)", nz - 1, 2}
    };

    for (const auto& test : tests) {
        std::cout << "  Testing wall: " << test.name << std::endl;

        FluidLBM fluid(nx, ny, nz, nu, rho,
                       lbm::physics::BoundaryType::WALL,
                       lbm::physics::BoundaryType::WALL,
                       lbm::physics::BoundaryType::WALL, 1.0f, 1.0f);

        fluid.initialize(rho, 0.0f, 0.0f, 0.0f);

        // Set moving wall with velocity in x-direction
        fluid.setMovingWall(test.direction, wall_velocity, 0.0f, 0.0f);

        // Run simulation
        for (int step = 0; step < 30; ++step) {
            fluid.collisionBGK();
            fluid.streaming();
            fluid.applyBoundaryConditions(1);
            fluid.computeMacroscopic();
        }

        // Verify velocity field has been affected
        std::vector<float> h_ux(nx * ny * nz);
        std::vector<float> h_uy(nx * ny * nz);
        std::vector<float> h_uz(nx * ny * nz);
        fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

        float max_velocity = 0.0f;
        for (int i = 0; i < nx * ny * nz; ++i) {
            max_velocity = std::max(max_velocity, std::abs(h_ux[i]));
        }

        std::cout << "    Maximum u_x in domain: " << max_velocity << std::endl;

        // Moving wall should induce some flow
        EXPECT_GT(max_velocity, 0.01f)
            << "Moving wall should induce flow in domain";
    }

    std::cout << "  PASS: All wall directions work correctly" << std::endl;
}

/**
 * @brief Test compatibility with mixed boundary conditions
 *
 * Moving wall + no-slip walls should coexist
 */
TEST_F(MovingWallBCTest, MixedBoundaryConditions) {
    std::cout << "\n=== Test: Moving Wall - Mixed BC Compatibility ===" << std::endl;

    const int nx = 16, ny = 16, nz = 16;
    const float nu = 0.033f;
    const float rho = 1000.0f;
    const float wall_velocity = 0.1f;

    std::cout << "  Domain: Lid-driven cavity configuration" << std::endl;
    std::cout << "    Top wall (y=ny-1): Moving with u_x = " << wall_velocity << std::endl;
    std::cout << "    Other walls: No-slip (stationary)" << std::endl;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL, 1.0f, 1.0f);

    fluid.initialize(rho, 0.0f, 0.0f, 0.0f);

    // Set top wall as moving, others remain no-slip
    fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, wall_velocity, 0.0f, 0.0f);

    // Run simulation
    for (int step = 0; step < 50; ++step) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
    }

    std::vector<float> h_ux(nx * ny * nz);
    std::vector<float> h_uy(nx * ny * nz);
    std::vector<float> h_uz(nx * ny * nz);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Check top wall is moving
    float top_wall_avg = 0.0f;
    int top_count = 0;
    for (int k = 1; k < nz - 1; ++k) {
        for (int i = 1; i < nx - 1; ++i) {
            int idx = i + nx * ((ny - 1) + ny * k);
            top_wall_avg += h_ux[idx];
            top_count++;
        }
    }
    top_wall_avg /= top_count;

    std::cout << "  Top wall average u_x: " << top_wall_avg << std::endl;

    // Check bottom wall is stationary
    float bottom_wall_avg = 0.0f;
    int bottom_count = 0;
    for (int k = 1; k < nz - 1; ++k) {
        for (int i = 1; i < nx - 1; ++i) {
            int idx = i + nx * (0 + ny * k);
            bottom_wall_avg += std::abs(h_ux[idx]);
            bottom_count++;
        }
    }
    bottom_wall_avg /= bottom_count;

    std::cout << "  Bottom wall average |u_x|: " << bottom_wall_avg << std::endl;

    EXPECT_NEAR(top_wall_avg, wall_velocity, 0.05f)
        << "Top wall should be moving";

    EXPECT_LT(bottom_wall_avg, 0.01f)
        << "Bottom wall should remain stationary (no-slip)";

    std::cout << "  PASS: Mixed boundary conditions work correctly" << std::endl;
}

/**
 * @brief Test mass conservation with moving wall
 *
 * Moving wall should conserve mass
 */
TEST_F(MovingWallBCTest, MassConservation) {
    std::cout << "\n=== Test: Moving Wall - Mass Conservation ===" << std::endl;

    const int nx = 16, ny = 16, nz = 16;
    const float nu = 0.033f;
    const float rho = 1000.0f;
    const float wall_velocity = 0.1f;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::WALL, 1.0f, 1.0f);

    fluid.initialize(rho, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, wall_velocity, 0.0f, 0.0f);

    // Compute initial total mass
    std::vector<float> h_rho(nx * ny * nz);
    fluid.copyDensityToHost(h_rho.data());
    float initial_mass = 0.0f;
    for (float r : h_rho) {
        initial_mass += r;
    }

    std::cout << "  Initial total mass: " << initial_mass << std::endl;

    // Run simulation
    for (int step = 0; step < 100; ++step) {
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
    }

    // Compute final total mass
    fluid.copyDensityToHost(h_rho.data());
    float final_mass = 0.0f;
    for (float r : h_rho) {
        final_mass += r;
    }

    std::cout << "  Final total mass: " << final_mass << std::endl;
    std::cout << "  Relative change: " << std::abs(final_mass - initial_mass) / initial_mass << std::endl;

    // Mass should be conserved (allow small numerical error)
    EXPECT_NEAR(final_mass, initial_mass, initial_mass * 0.01f)
        << "Total mass should be conserved";

    std::cout << "  PASS: Mass conserved with moving wall BC" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "Moving Wall Boundary Condition Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPhysics:" << std::endl;
    std::cout << "  - Moving wall BC via Zou-He velocity BC" << std::endl;
    std::cout << "  - Essential for lid-driven cavity test" << std::endl;
    std::cout << "  - Enforces prescribed wall velocity" << std::endl;
    std::cout << "\n";

    int result = RUN_ALL_TESTS();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Moving Wall BC Tests Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n";

    return result;
}
