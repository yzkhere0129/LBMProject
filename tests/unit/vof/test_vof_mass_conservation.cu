/**
 * @file test_vof_mass_conservation.cu
 * @brief Unit test for VOF mass conservation
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test mass conservation during advection
 */
TEST(VOFMassConservationTest, AdvectionConservation) {
    int nx = 64, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);

    // Record initial mass
    float initial_mass = vof.computeTotalMass();
    EXPECT_GT(initial_mass, 0.0f) << "Initial mass should be positive";

    // Create velocity field
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 1.5f);
    std::vector<float> h_uy(num_cells, 0.5f);
    std::vector<float> h_uz(num_cells, 0.0f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Advect for 20 time steps
    float dt = 0.05f;
    for (int step = 0; step < 20; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    // Check mass conservation
    float final_mass = vof.computeTotalMass();
    float mass_error = std::abs(final_mass - initial_mass) / initial_mass;

    // With Olsson-Kreiss compression: expect < 5% error (major improvement over 32.6%)
    EXPECT_LT(mass_error, 0.05f)  // 5% tolerance with compression
        << "Mass conservation error: " << mass_error
        << " (initial: " << initial_mass << ", final: " << final_mass << ")";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test mass conservation with complex velocity field
 */
TEST(VOFMassConservationTest, ComplexVelocityField) {
    int nx = 48, ny = 48, nz = 48;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet
    float cx = nx / 2.0f;
    float cy = ny / 2.0f;
    float cz = nz / 2.0f;
    vof.initializeDroplet(cx, cy, cz, 12.0f);

    float initial_mass = vof.computeTotalMass();

    // Create rotating velocity field (should conserve mass exactly)
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Rotating flow: u = -ω*(y-cy), v = ω*(x-cx), w = 0
                float omega = 0.5f;
                h_ux[idx] = -omega * (j - cy);
                h_uy[idx] = omega * (i - cx);
                h_uz[idx] = 0.0f;
            }
        }
    }

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Advect
    float dt = 0.05f;
    for (int step = 0; step < 10; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    // Check mass conservation
    float final_mass = vof.computeTotalMass();
    float mass_error = std::abs(final_mass - initial_mass) / initial_mass;

    // With compression: expect < 6% for rotating flow (complex deformation)
    EXPECT_LT(mass_error, 0.06f)  // 6% tolerance for complex rotating flow
        << "Mass error with rotating flow: " << mass_error;

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test mass conservation over long time
 */
TEST(VOFMassConservationTest, LongTimeAdvection) {
    int nx = 64, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 8.0f);

    float initial_mass = vof.computeTotalMass();

    // Velocity field
    int num_cells = nx * ny * nz;
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 1.0f);
    std::vector<float> h_uy(num_cells, 0.5f);
    std::vector<float> h_uz(num_cells, 0.25f);

    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Long-time advection (100 steps)
    float dt = 0.05f;
    for (int step = 0; step < 100; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    // Check mass conservation
    float final_mass = vof.computeTotalMass();
    float mass_error = std::abs(final_mass - initial_mass) / initial_mass;

    // With compression: expect < 15% for long-time advection (100 steps)
    // This is a significant improvement over 32.6% without compression
    EXPECT_LT(mass_error, 0.15f)  // 15% tolerance for long-time advection (100 steps)
        << "Mass error after 100 steps: " << mass_error;

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test that mass stays within physical bounds
 */
TEST(VOFMassConservationTest, MassBounds) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);

    float initial_mass = vof.computeTotalMass();

    // Mass should be positive and less than total cells
    int num_cells = nx * ny * nz;
    EXPECT_GT(initial_mass, 0.0f);
    EXPECT_LT(initial_mass, static_cast<float>(num_cells));

    // Advect
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    cudaMemset(d_ux, 0, num_cells * sizeof(float));
    cudaMemset(d_uy, 0, num_cells * sizeof(float));
    cudaMemset(d_uz, 0, num_cells * sizeof(float));

    std::vector<float> h_ux(num_cells, 2.0f);
    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    float dt = 0.1f;
    for (int step = 0; step < 50; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);

        float mass = vof.computeTotalMass();
        EXPECT_GT(mass, 0.0f) << "Mass negative at step " << step;
        EXPECT_LT(mass, static_cast<float>(num_cells))
            << "Mass exceeds total cells at step " << step;
    }

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}
