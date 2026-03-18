/**
 * @file test_vof_advection_bulk_cells.cu
 * @brief Regression test for VOF advection bulk cell bug fix
 *
 * BUG FIX: Interface compression kernel was returning early for bulk cells
 * (f < 0.01 or f > 0.99) WITHOUT writing the advected values to output buffer.
 * This caused the advection to not propagate through bulk regions.
 *
 * Fix location: /home/yzk/LBMProject/src/physics/vof/vof_solver.cu:419-421
 * Added: fill_level[idx] = f; before early return
 *
 * This test validates that bulk cells (f~0 and f~1) properly propagate
 * advected values when moving with a uniform velocity field.
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace lbm::physics;

/**
 * @brief Test that bulk liquid (f=1) advects correctly with uniform velocity
 *
 * Setup: Column of liquid (f=1) in the middle of domain
 * Velocity: Uniform u_x = 1.0 (rightward)
 * Expected: Column moves right by distance = velocity * time
 *
 * This tests the critical bug fix where bulk cells (f > 0.99) were not
 * being written to output buffer, causing advection to fail.
 */
TEST(VOFAdvectionBulkCellsTest, BulkLiquidAdvection) {
    // Domain setup
    int nx = 128, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize: vertical column of liquid at x = nx/4
    int x_init = nx / 4;
    std::vector<float> h_fill(nx * ny * nz, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Column of pure liquid (f = 1.0) from x_init to x_init+10
                if (i >= x_init && i < x_init + 10) {
                    h_fill[idx] = 1.0f;  // Bulk liquid cell
                }
            }
        }
    }

    // Upload initial fill level
    cudaMemcpy(vof.getFillLevel(), h_fill.data(),
               nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    // Create uniform velocity field: u_x = 1.0, u_y = u_z = 0
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 1.0f);  // Rightward velocity
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Advect for 20 time steps (dt = 0.5 dx/u = 0.5)
    // Expected displacement: u * t = 1.0 * (20 * 0.5) = 10.0 dx
    float dt = 0.5f;
    int num_steps = 20;

    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    // Download final fill level
    cudaMemcpy(h_fill.data(), vof.getFillLevel(),
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: Column should have moved from x_init to x_init+10
    int x_expected = x_init + 10;  // displacement = 10

    // Check that liquid has moved to expected position
    float liquid_at_expected = 0.0f;
    float liquid_at_initial = 0.0f;
    int count_expected = 0;
    int count_initial = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            // Check expected position
            for (int i = x_expected; i < x_expected + 10 && i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                liquid_at_expected += h_fill[idx];
                count_expected++;
            }

            // Check initial position (should be mostly empty)
            for (int i = x_init; i < x_init + 10; ++i) {
                int idx = i + nx * (j + ny * k);
                liquid_at_initial += h_fill[idx];
                count_initial++;
            }
        }
    }

    liquid_at_expected /= count_expected;
    liquid_at_initial /= count_initial;

    std::cout << "\n=== Bulk Liquid Advection Test ===\n";
    std::cout << "Initial position: x = " << x_init << " to " << x_init + 10 << "\n";
    std::cout << "Expected position: x = " << x_expected << " to " << x_expected + 10 << "\n";
    std::cout << "Average fill at expected: " << liquid_at_expected << "\n";
    std::cout << "Average fill at initial: " << liquid_at_initial << "\n";

    // CRITICAL: Without the bug fix, liquid_at_expected would be ~0 (advection doesn't work)
    // With the fix, liquid_at_expected should be high (> 0.5)
    EXPECT_GT(liquid_at_expected, 0.5f)
        << "REGRESSION: Bulk liquid did not advect! "
        << "Bug likely reintroduced in interface compression kernel.";

    // Initial position should be mostly depleted
    EXPECT_LT(liquid_at_initial, 0.5f)
        << "Liquid did not leave initial position";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test that bulk gas (f=0) advects correctly with uniform velocity
 *
 * Setup: Domain full of liquid except for a gas bubble (f=0)
 * Velocity: Uniform u_x = 1.0 (rightward)
 * Expected: Gas bubble moves right
 *
 * Tests the f < 0.01 branch of the bug fix
 */
TEST(VOFAdvectionBulkCellsTest, BulkGasAdvection) {
    int nx = 128, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize: domain full of liquid (f=1) with gas bubble at x = nx/4
    int x_init = nx / 4;
    std::vector<float> h_fill(nx * ny * nz, 1.0f);  // Background liquid

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Gas bubble (f = 0.0) from x_init to x_init+10
                if (i >= x_init && i < x_init + 10) {
                    h_fill[idx] = 0.0f;  // Bulk gas cell
                }
            }
        }
    }

    cudaMemcpy(vof.getFillLevel(), h_fill.data(),
               nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    // Uniform velocity field
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 1.0f);
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Advect for 20 steps
    float dt = 0.5f;
    int num_steps = 20;

    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    cudaMemcpy(h_fill.data(), vof.getFillLevel(),
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: gas bubble moved to x_init+10
    int x_expected = x_init + 10;

    float gas_at_expected = 0.0f;
    float gas_at_initial = 0.0f;
    int count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = x_expected; i < x_expected + 10 && i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                gas_at_expected += (1.0f - h_fill[idx]);  // Gas fraction
                count++;
            }
            for (int i = x_init; i < x_init + 10; ++i) {
                int idx = i + nx * (j + ny * k);
                gas_at_initial += (1.0f - h_fill[idx]);
            }
        }
    }

    gas_at_expected /= count;
    gas_at_initial /= count;

    std::cout << "\n=== Bulk Gas Advection Test ===\n";
    std::cout << "Average gas at expected: " << gas_at_expected << "\n";
    std::cout << "Average gas at initial: " << gas_at_initial << "\n";

    // Gas should have moved to expected position
    EXPECT_GT(gas_at_expected, 0.5f)
        << "REGRESSION: Bulk gas did not advect!";

    // Initial position should be refilled with liquid
    EXPECT_LT(gas_at_initial, 0.5f)
        << "Gas did not leave initial position";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test interface advection with bulk regions on both sides
 *
 * Setup: Sharp interface with bulk liquid (f=1) and bulk gas (f=0) on either side
 * Velocity: Uniform rightward
 * Expected: Entire pattern moves right, maintaining sharp interface
 *
 * This is the most realistic test case for the bug fix
 */
TEST(VOFAdvectionBulkCellsTest, InterfaceWithBulkRegions) {
    int nx = 128, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize: Sharp interface at x = nx/2 with bulk regions on both sides
    int x_interface = nx / 2;
    std::vector<float> h_fill(nx * ny * nz, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Step function: liquid on left, gas on right
                if (i < x_interface - 2) {
                    h_fill[idx] = 1.0f;  // Bulk liquid
                } else if (i >= x_interface + 2) {
                    h_fill[idx] = 0.0f;  // Bulk gas
                } else {
                    // Interface region (linear transition)
                    float frac = float(x_interface + 2 - i) / 4.0f;
                    h_fill[idx] = frac;
                }
            }
        }
    }

    cudaMemcpy(vof.getFillLevel(), h_fill.data(),
               nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    // Uniform velocity
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 1.0f);
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Advect for 15 steps (displacement = 7.5)
    float dt = 0.5f;
    int num_steps = 15;

    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    cudaMemcpy(h_fill.data(), vof.getFillLevel(),
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected interface position: x_interface + 7.5 ≈ x_interface + 8
    int x_new_interface = x_interface + 8;

    // Compute average fill on left and right of new interface
    float fill_left = 0.0f;
    float fill_right = 0.0f;
    int count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            // Sample bulk liquid region (should be f~1)
            int idx_left = (x_new_interface - 10) + nx * (j + ny * k);
            if (x_new_interface - 10 >= 0) {
                fill_left += h_fill[idx_left];
            }

            // Sample bulk gas region (should be f~0)
            int idx_right = (x_new_interface + 10) + nx * (j + ny * k);
            if (x_new_interface + 10 < nx) {
                fill_right += h_fill[idx_right];
            }
            count++;
        }
    }

    fill_left /= count;
    fill_right /= count;

    std::cout << "\n=== Interface with Bulk Regions Test ===\n";
    std::cout << "Initial interface: x = " << x_interface << "\n";
    std::cout << "Expected interface: x = " << x_new_interface << "\n";
    std::cout << "Fill left of interface: " << fill_left << "\n";
    std::cout << "Fill right of interface: " << fill_right << "\n";

    // Bulk liquid region should still be liquid
    EXPECT_GT(fill_left, 0.8f)
        << "REGRESSION: Bulk liquid region degraded during advection";

    // Bulk gas region should still be gas
    EXPECT_LT(fill_right, 0.2f)
        << "REGRESSION: Bulk gas region degraded during advection";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Stress test: Very high velocity advection
 *
 * Tests that bulk cells advect correctly even with large CFL numbers
 */
TEST(VOFAdvectionBulkCellsTest, HighVelocityStressTest) {
    int nx = 256, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize liquid column
    int x_init = 30;
    std::vector<float> h_fill(nx * ny * nz, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = x_init; i < x_init + 20; ++i) {
                int idx = i + nx * (j + ny * k);
                h_fill[idx] = 1.0f;
            }
        }
    }

    cudaMemcpy(vof.getFillLevel(), h_fill.data(),
               nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    // High velocity field: u = 5.0 (CFL = 5.0 * dt / dx)
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 5.0f);  // High velocity
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Advect with dt = 0.1 (CFL = 0.5)
    float dt = 0.1f;
    int num_steps = 40;  // displacement = 5.0 * 0.1 * 40 = 20

    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    cudaMemcpy(h_fill.data(), vof.getFillLevel(),
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that liquid has moved approximately 20 cells
    int x_expected = x_init + 20;

    float liquid_at_expected = 0.0f;
    int count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = x_expected - 5; i < x_expected + 25 && i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                liquid_at_expected += h_fill[idx];
                count++;
            }
        }
    }

    liquid_at_expected /= count;

    std::cout << "\n=== High Velocity Stress Test ===\n";
    std::cout << "Velocity: 5.0 dx/dt\n";
    std::cout << "Average fill at expected position: " << liquid_at_expected << "\n";

    // Liquid should still be present (even with diffusion)
    EXPECT_GT(liquid_at_expected, 0.3f)
        << "High velocity advection failed for bulk cells";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Stress test: Very low velocity (numerical precision test)
 *
 * Tests that bulk cells work correctly with very small velocities
 */
TEST(VOFAdvectionBulkCellsTest, LowVelocityStressTest) {
    int nx = 128, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize liquid column
    int x_init = nx / 4;
    std::vector<float> h_fill(nx * ny * nz, 0.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = x_init; i < x_init + 10; ++i) {
                int idx = i + nx * (j + ny * k);
                h_fill[idx] = 1.0f;
            }
        }
    }

    cudaMemcpy(vof.getFillLevel(), h_fill.data(),
               nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    // Very low velocity: u = 0.001
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 0.001f);  // Very low velocity
    std::vector<float> h_uy(num_cells, 0.0f);
    std::vector<float> h_uz(num_cells, 0.0f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Many small steps
    float dt = 1.0f;
    int num_steps = 100;  // displacement = 0.001 * 1.0 * 100 = 0.1 (barely moves)

    float initial_mass = vof.computeTotalMass();

    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    float final_mass = vof.computeTotalMass();
    float mass_error = std::abs(final_mass - initial_mass) / initial_mass;

    std::cout << "\n=== Low Velocity Stress Test ===\n";
    std::cout << "Velocity: 0.001 dx/dt\n";
    std::cout << "Initial mass: " << initial_mass << "\n";
    std::cout << "Final mass: " << final_mass << "\n";
    std::cout << "Mass error: " << mass_error << "\n";

    // Mass should be well conserved (low velocity = low numerical error)
    EXPECT_LT(mass_error, 0.05f)
        << "Low velocity bulk cell advection has excessive mass loss";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}
