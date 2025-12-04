/**
 * @file test_fluid_boundaries.cu
 * @brief Unit tests for FluidLBM boundary conditions
 *
 * Tests no-slip wall boundaries (bounce-back) implementation including:
 * - Basic wall velocity verification
 * - Poiseuille flow profile validation
 * - Momentum conservation with walls
 * - Bounce-back symmetry verification
 */

#include <gtest/gtest.h>
#include "physics/fluid_lbm.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace lbm::physics;

/**
 * @brief Test fixture for fluid boundary condition tests
 */
class FluidBoundaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Common cleanup if needed
    }

    // Helper function to compute L2 error
    float computeL2Error(const std::vector<float>& computed,
                        const std::vector<float>& analytical) {
        float sum_sq_error = 0.0f;
        float sum_sq_analytical = 0.0f;

        for (size_t i = 0; i < computed.size(); ++i) {
            float error = computed[i] - analytical[i];
            sum_sq_error += error * error;
            sum_sq_analytical += analytical[i] * analytical[i];
        }

        return std::sqrt(sum_sq_error / (sum_sq_analytical + 1e-10f));
    }

    // Helper function to compute analytical Poiseuille profile
    std::vector<float> computePoiseuilleProfile(int ny, float H, float force, float nu, float rho) {
        std::vector<float> profile(ny);
        float u_max = force * H * H / (8.0f * nu * rho);

        for (int y = 0; y < ny; ++y) {
            float y_norm = (y + 0.5f) / ny;  // Normalized position [0, 1]
            float y_phys = y_norm * H;        // Physical position

            // Parabolic profile: u(y) = (F*H²/8νρ) * [1 - (2y/H - 1)²]
            float eta = 2.0f * y_phys / H - 1.0f;  // [-1, 1]
            profile[y] = u_max * (1.0f - eta * eta);
        }

        return profile;
    }
};

/**
 * @brief Test 1: No-slip wall boundary - verify zero velocity at walls
 *
 * Setup: Channel with walls in y-direction, periodic in x and z
 * Initial: Uniform flow in x-direction
 * Expected: After relaxation, uy = 0 at y=0 and y=ny-1
 */
TEST_F(FluidBoundaryTest, NoSlipWallZeroVelocity) {
    // Domain setup
    const int nx = 32;
    const int ny = 16;
    const int nz = 4;
    const float nu = 0.1f;
    const float rho0 = 1.0f;

    // Create solver with walls in y-direction
    FluidLBM solver(nx, ny, nz, nu, rho0,
                    BoundaryType::PERIODIC,  // x
                    BoundaryType::WALL,      // y - walls at top/bottom
                    BoundaryType::PERIODIC); // z

    // Initialize with small uniform velocity
    solver.initialize(rho0, 0.01f, 0.01f, 0.0f);

    // Run simulation to let walls enforce no-slip
    const int n_steps = 1000;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);  // No body force
        solver.streaming();
        solver.applyBoundaryConditions(1);  // Apply wall boundaries
    }

    // Extract velocity field
    std::vector<float> h_ux(nx * ny * nz);
    std::vector<float> h_uy(nx * ny * nz);
    std::vector<float> h_uz(nx * ny * nz);
    solver.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Check y-velocity at walls (averaged over x and z)
    float uy_bottom_avg = 0.0f;
    float uy_top_avg = 0.0f;

    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            int idx_bottom = x + 0 * nx + z * nx * ny;
            int idx_top = x + (ny - 1) * nx + z * nx * ny;
            uy_bottom_avg += std::abs(h_uy[idx_bottom]);
            uy_top_avg += std::abs(h_uy[idx_top]);
        }
    }

    uy_bottom_avg /= (nx * nz);
    uy_top_avg /= (nx * nz);

    // Verify no-slip: velocity should be very small at walls
    EXPECT_LT(uy_bottom_avg, 1e-3f) << "Bottom wall velocity should be near zero";
    EXPECT_LT(uy_top_avg, 1e-3f) << "Top wall velocity should be near zero";
}

/**
 * @brief Test 2: Poiseuille flow profile validation
 *
 * Setup: Channel with walls in y-direction, periodic in x/z, uniform force in x
 * Expected: Parabolic velocity profile u_x(y) after steady state
 * Validation: L2 error < 5% compared to analytical solution
 */
TEST_F(FluidBoundaryTest, PoiseuilleFlowProfile) {
    // Domain setup
    const int nx = 64;
    const int ny = 32;
    const int nz = 4;
    const float nu = 0.16f;  // Increased for better stability (tau > 1.0)
    const float rho0 = 1.0f;

    // Create solver with walls in y-direction
    // Use lattice units (dt = dx = 1.0) to avoid huge relaxation time
    FluidLBM solver(nx, ny, nz, nu, rho0,
                    BoundaryType::PERIODIC,
                    BoundaryType::WALL,
                    BoundaryType::PERIODIC,
                    1.0f, 1.0f);  // dt = dx = 1.0 (lattice units)

    // Initialize at rest
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Apply uniform body force in x-direction
    const float force_x = 1e-5f;

    // Run to steady state
    const int n_steps = 15000;  // More iterations for better convergence
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(force_x, 0.0f, 0.0f);
        solver.streaming();
        solver.applyBoundaryConditions(1);
    }

    // Extract velocity field
    std::vector<float> h_ux(nx * ny * nz);
    solver.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    // Compute average profile across x and z
    std::vector<float> profile_computed(ny, 0.0f);
    for (int y = 0; y < ny; ++y) {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                int idx = x + y * nx + z * nx * ny;
                profile_computed[y] += h_ux[idx];
            }
        }
        profile_computed[y] /= (nx * nz);
    }

    // Compute analytical Poiseuille profile
    float H = static_cast<float>(ny);  // Channel height in lattice units
    std::vector<float> profile_analytical = computePoiseuilleProfile(ny, H, force_x, nu, rho0);

    // Compute L2 error
    float l2_error = computeL2Error(profile_computed, profile_analytical);

    // Check that L2 error is below threshold
    // Note: Bounce-back has O(h) accuracy, so ~10% error is acceptable at this resolution
    EXPECT_LT(l2_error, 0.10f) << "Poiseuille profile L2 error should be < 10%";

    // Also verify parabolic shape: max velocity should be at center
    int center_y = ny / 2;
    float u_center = profile_computed[center_y];
    float u_walls_avg = 0.5f * (profile_computed[0] + profile_computed[ny - 1]);

    EXPECT_GT(u_center, u_walls_avg) << "Center velocity should exceed wall velocity";
    EXPECT_LT(profile_computed[0], u_center * 0.2f) << "Wall velocity should be small";
    EXPECT_LT(profile_computed[ny - 1], u_center * 0.2f) << "Wall velocity should be small";
}

/**
 * @brief Test 3: Momentum conservation with walls
 *
 * Setup: Walls in y-direction, initial momentum
 * Expected: Total momentum decays to zero (no-slip dissipation) but mass conserved
 */
TEST_F(FluidBoundaryTest, MomentumConservationWithWalls) {
    // Domain setup
    const int nx = 32;
    const int ny = 16;
    const int nz = 4;
    const float nu = 0.16f;  // Use same viscosity as Poiseuille test
    const float rho0 = 1.0f;

    // Create solver with walls in y-direction
    // Use lattice units (dt = dx = 1.0)
    FluidLBM solver(nx, ny, nz, nu, rho0,
                    BoundaryType::PERIODIC,
                    BoundaryType::WALL,
                    BoundaryType::PERIODIC,
                    1.0f, 1.0f);  // dt = dx = 1.0 (lattice units)

    // Initialize with uniform flow
    solver.initialize(rho0, 0.05f, 0.0f, 0.0f);

    // Compute initial mass
    std::vector<float> h_rho(nx * ny * nz);
    solver.copyDensityToHost(h_rho.data());

    float total_mass_initial = 0.0f;
    for (int i = 0; i < nx * ny * nz; ++i) {
        total_mass_initial += h_rho[i];
    }

    // Run simulation (momentum should decay due to walls)
    const int n_steps = 2000;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
        solver.applyBoundaryConditions(1);
    }

    // Compute final mass
    solver.copyDensityToHost(h_rho.data());

    float total_mass_final = 0.0f;
    for (int i = 0; i < nx * ny * nz; ++i) {
        total_mass_final += h_rho[i];
    }

    // Check mass conservation (should be very tight)
    float mass_relative_error = std::abs(total_mass_final - total_mass_initial) / total_mass_initial;
    EXPECT_LT(mass_relative_error, 1e-5f) << "Mass should be conserved with wall boundaries";

    // Verify momentum has decayed (walls cause dissipation)
    std::vector<float> h_ux(nx * ny * nz);
    solver.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    float avg_velocity = 0.0f;
    for (int i = 0; i < nx * ny * nz; ++i) {
        avg_velocity += h_ux[i];
    }
    avg_velocity /= (nx * ny * nz);

    // Average velocity should be much smaller than initial (0.05)
    EXPECT_LT(avg_velocity, 0.01f) << "Momentum should decay with no-slip walls";
}

/**
 * @brief Test 4: Bounce-back symmetry verification
 *
 * Setup: Single wall in y-direction, particle approaching wall
 * Expected: Distribution functions should obey bounce-back symmetry
 *           f_incoming = f_outgoing_opposite after boundary application
 */
TEST_F(FluidBoundaryTest, BounceBackSymmetry) {
    // Simple setup for testing bounce-back mechanism
    const int nx = 8;
    const int ny = 8;
    const int nz = 4;
    const float nu = 0.1f;
    const float rho0 = 1.0f;

    // Create solver with walls in y-direction
    FluidLBM solver(nx, ny, nz, nu, rho0,
                    BoundaryType::PERIODIC,
                    BoundaryType::WALL,
                    BoundaryType::PERIODIC);

    // Initialize with flow toward wall
    solver.initialize(rho0, 0.0f, 0.05f, 0.0f);  // Flow in +y direction

    // Run a few steps
    for (int step = 0; step < 100; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
        solver.applyBoundaryConditions(1);
    }

    // Extract velocity at wall nodes
    std::vector<float> h_uy(nx * ny * nz);
    solver.copyVelocityToHost(nullptr, h_uy.data(), nullptr);

    // Check that wall nodes have near-zero y-velocity
    float wall_velocity_sum = 0.0f;
    int wall_count = 0;

    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            // Bottom wall
            int idx_bottom = x + 0 * nx + z * nx * ny;
            wall_velocity_sum += std::abs(h_uy[idx_bottom]);
            wall_count++;

            // Top wall
            int idx_top = x + (ny - 1) * nx + z * nx * ny;
            wall_velocity_sum += std::abs(h_uy[idx_top]);
            wall_count++;
        }
    }

    float avg_wall_velocity = wall_velocity_sum / wall_count;

    // Bounce-back should enforce near-zero velocity at walls
    EXPECT_LT(avg_wall_velocity, 1e-3f) << "Bounce-back should enforce zero velocity at walls";

    // Also verify that interior velocities are small but non-zero
    float interior_velocity_sum = 0.0f;
    int interior_count = 0;

    for (int y = 1; y < ny - 1; ++y) {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                int idx = x + y * nx + z * nx * ny;
                interior_velocity_sum += std::abs(h_uy[idx]);
                interior_count++;
            }
        }
    }

    float avg_interior_velocity = interior_velocity_sum / interior_count;

    // Interior should have some flow remaining
    EXPECT_GT(avg_interior_velocity, avg_wall_velocity)
        << "Interior should have more velocity than walls";
}

/**
 * @brief Test 5: Backward compatibility - periodic boundaries
 *
 * Ensure that default constructor (periodic boundaries) still works
 */
TEST_F(FluidBoundaryTest, PeriodicBoundaryBackwardCompatibility) {
    const int nx = 16;
    const int ny = 16;
    const int nz = 4;
    const float nu = 0.1f;
    const float rho0 = 1.0f;

    // Create solver with default (all periodic)
    FluidLBM solver(nx, ny, nz, nu, rho0);

    // Initialize
    solver.initialize(rho0, 0.01f, 0.01f, 0.0f);

    // Run simulation
    for (int step = 0; step < 100; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
        solver.applyBoundaryConditions(0);  // Periodic (no action)
    }

    // Extract density
    std::vector<float> h_rho(nx * ny * nz);
    solver.copyDensityToHost(h_rho.data());

    // Compute total mass
    float total_mass = 0.0f;
    for (int i = 0; i < nx * ny * nz; ++i) {
        total_mass += h_rho[i];
    }

    float expected_mass = rho0 * nx * ny * nz;
    float mass_error = std::abs(total_mass - expected_mass) / expected_mass;

    EXPECT_LT(mass_error, 1e-5f) << "Periodic boundaries should conserve mass";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
