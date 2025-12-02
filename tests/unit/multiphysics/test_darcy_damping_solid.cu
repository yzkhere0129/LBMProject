/**
 * @file test_darcy_damping_solid.cu
 * @brief Unit test for Darcy damping in solid regions
 *
 * This test validates that Darcy damping correctly suppresses velocity
 * in solid regions (liquid_fraction = 0) while allowing flow in liquid
 * regions (liquid_fraction = 1).
 *
 * Test scenario:
 * - Domain: 50x50x50 cells (2 μm spacing)
 * - Phase field: Bottom half solid (fl=0.0), top half liquid (fl=1.0)
 * - Initial velocity: Uniform 0.1 m/s in x-direction
 * - Apply Darcy damping for 100 iterations
 *
 * Expected results:
 * - Solid region (k < 25): velocity decays to < 1e-6 m/s
 * - Liquid region (k >= 25): velocity remains > 0.05 m/s
 *
 * Physics:
 * Darcy damping force: F = -C·(1-fl)²/(fl³+ε)·u
 * - In solid (fl=0): F ≈ -C·u (strong damping)
 * - In liquid (fl=1): F ≈ 0 (no damping)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/fluid_lbm.h"

using namespace lbm::physics;

// Helper kernel to apply force to velocity (simplified explicit update)
__global__ void applyForceToVelocityKernel(
    float* ux, float* uy, float* uz,
    const float* fx, const float* fy, const float* fz,
    float dt, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Simple explicit update: v += F*dt
    // In real LBM, force is incorporated into collision step
    ux[idx] += fx[idx] * dt;
    uy[idx] += fy[idx] * dt;
    uz[idx] += fz[idx] * dt;
}

class DarcyDampingTest : public ::testing::Test {
protected:
    void SetUp() override {
        nx = 50;
        ny = 50;
        nz = 50;
        num_cells = nx * ny * nz;
        dx = 2e-6f;  // 2 μm
        dt = 1e-10f;  // 0.1 ns (small timestep for numerical stability)

        // Allocate host memory
        h_liquid_fraction = new float[num_cells];
        h_ux = new float[num_cells];
        h_uy = new float[num_cells];
        h_uz = new float[num_cells];

        // Initialize: bottom half solid, top half liquid
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);

                    // Liquid fraction: 0.0 (solid) at bottom, 1.0 (liquid) at top
                    h_liquid_fraction[idx] = (k < nz/2) ? 0.0f : 1.0f;

                    // Initial uniform velocity
                    h_ux[idx] = 0.1f;  // 0.1 m/s in x-direction
                    h_uy[idx] = 0.0f;
                    h_uz[idx] = 0.0f;
                }
            }
        }

        // Allocate device memory
        cudaMalloc(&d_liquid_fraction, num_cells * sizeof(float));
        cudaMalloc(&d_ux, num_cells * sizeof(float));
        cudaMalloc(&d_uy, num_cells * sizeof(float));
        cudaMalloc(&d_uz, num_cells * sizeof(float));
        cudaMalloc(&d_fx, num_cells * sizeof(float));
        cudaMalloc(&d_fy, num_cells * sizeof(float));
        cudaMalloc(&d_fz, num_cells * sizeof(float));

        // Copy to device
        cudaMemcpy(d_liquid_fraction, h_liquid_fraction, num_cells * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_ux, h_ux, num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uy, h_uy, num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uz, h_uz, num_cells * sizeof(float), cudaMemcpyHostToDevice);

        // Zero force arrays
        cudaMemset(d_fx, 0, num_cells * sizeof(float));
        cudaMemset(d_fy, 0, num_cells * sizeof(float));
        cudaMemset(d_fz, 0, num_cells * sizeof(float));
    }

    void TearDown() override {
        delete[] h_liquid_fraction;
        delete[] h_ux;
        delete[] h_uy;
        delete[] h_uz;

        cudaFree(d_liquid_fraction);
        cudaFree(d_ux);
        cudaFree(d_uy);
        cudaFree(d_uz);
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_fz);
    }

    int nx, ny, nz, num_cells;
    float dx, dt;
    float *h_liquid_fraction, *h_ux, *h_uy, *h_uz;
    float *d_liquid_fraction, *d_ux, *d_uy, *d_uz;
    float *d_fx, *d_fy, *d_fz;
};

TEST_F(DarcyDampingTest, SolidVelocitySuppression) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST: Darcy Damping - Solid Velocity Suppression" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Resolution: dx = " << dx * 1e6 << " μm" << std::endl;
    std::cout << "Phase field:" << std::endl;
    std::cout << "  Bottom half (k < " << nz/2 << "): solid (fl = 0.0)" << std::endl;
    std::cout << "  Top half (k >= " << nz/2 << "): liquid (fl = 1.0)" << std::endl;
    std::cout << "Initial velocity: u = (0.1, 0, 0) m/s everywhere" << std::endl;
    std::cout << std::endl;

    // Create FluidLBM instance
    const float nu_lattice = 0.0333f;  // Lattice viscosity (tau=0.6)
    const float rho = 4110.0f;  // Ti6Al4V density
    FluidLBM fluid_lbm(nx, ny, nz, nu_lattice, rho);

    // Initialize FluidLBM with uniform velocity
    // Note: velocity in physical units
    fluid_lbm.initialize(1.0f, 0.1f, 0.0f, 0.0f);

    const float darcy_coefficient = 1e7f;  // Darcy damping constant
    const int num_iterations = 1000;  // More iterations for gradual damping

    std::cout << "Darcy coefficient: C = " << darcy_coefficient << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    std::cout << "Time step: dt = " << dt * 1e9 << " ns" << std::endl;
    std::cout << std::endl;

    // Get pointers to FluidLBM velocity arrays (to update them)
    float* d_fluid_ux = fluid_lbm.getVelocityX();
    float* d_fluid_uy = fluid_lbm.getVelocityY();
    float* d_fluid_uz = fluid_lbm.getVelocityZ();

    // Apply Darcy damping iteratively
    std::cout << "Running simulation..." << std::endl;
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Zero forces before each iteration
        cudaMemset(d_fx, 0, num_cells * sizeof(float));
        cudaMemset(d_fy, 0, num_cells * sizeof(float));
        cudaMemset(d_fz, 0, num_cells * sizeof(float));

        // Apply Darcy damping (modifies forces in-place)
        fluid_lbm.applyDarcyDamping(d_liquid_fraction, darcy_coefficient,
                                    d_fx, d_fy, d_fz);

        // Update velocity (simplified: v += F*dt)
        // In real LBM, this happens in collision step
        int block_size = 256;
        int grid_size = (num_cells + block_size - 1) / block_size;
        applyForceToVelocityKernel<<<grid_size, block_size>>>(
            d_fluid_ux, d_fluid_uy, d_fluid_uz, d_fx, d_fy, d_fz, dt, num_cells);
        cudaDeviceSynchronize();

        // Print progress every 100 iterations
        if ((iter + 1) % 100 == 0 || iter == 0) {
            fluid_lbm.copyVelocityToHost(h_ux, h_uy, h_uz);

            // Sample solid and liquid velocities
            int solid_idx = nx/2 + nx * (ny/2 + ny * 5);  // k=5 (solid)
            int liquid_idx = nx/2 + nx * (ny/2 + ny * 35);  // k=35 (liquid)

            std::cout << "  Iteration " << std::setw(4) << (iter + 1)
                      << ": solid ux = " << std::scientific << std::setprecision(3)
                      << h_ux[solid_idx]
                      << " m/s, liquid ux = " << h_ux[liquid_idx] << " m/s" << std::endl;
        }
    }
    std::cout << std::endl;

    // Copy results back
    fluid_lbm.copyVelocityToHost(h_ux, h_uy, h_uz);

    // Analyze results
    float max_solid_velocity = 0.0f;
    float min_liquid_velocity = 1e10f;
    float mean_solid_velocity = 0.0f;
    float mean_liquid_velocity = 0.0f;
    int solid_violations = 0;
    int solid_count = 0;
    int liquid_count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float v_mag = std::sqrt(h_ux[idx]*h_ux[idx] +
                                       h_uy[idx]*h_uy[idx] +
                                       h_uz[idx]*h_uz[idx]);

                if (k < nz/2) {  // Solid region
                    max_solid_velocity = std::max(max_solid_velocity, v_mag);
                    mean_solid_velocity += v_mag;
                    solid_count++;

                    if (v_mag > 1e-6f) {
                        solid_violations++;
                        if (solid_violations <= 5) {  // Print first 5 violations
                            std::cout << "  WARNING: Solid violation at (" << i << ","
                                      << j << "," << k << "): |v| = "
                                      << std::scientific << v_mag << " m/s" << std::endl;
                        }
                    }
                } else {  // Liquid region
                    min_liquid_velocity = std::min(min_liquid_velocity, v_mag);
                    mean_liquid_velocity += v_mag;
                    liquid_count++;
                }
            }
        }
    }

    mean_solid_velocity /= solid_count;
    mean_liquid_velocity /= liquid_count;

    // Print results
    std::cout << "\n=== Darcy Damping Unit Test Results ===\n" << std::endl;

    std::cout << "Solid region (bottom half, k < " << nz/2 << "):" << std::endl;
    std::cout << "  Cell count: " << solid_count << std::endl;
    std::cout << "  Max velocity: " << std::scientific << std::setprecision(3)
              << max_solid_velocity << " m/s" << std::endl;
    std::cout << "  Mean velocity: " << mean_solid_velocity << " m/s" << std::endl;
    std::cout << "  Violations (v > 1e-6): " << solid_violations << " / "
              << solid_count << std::endl;
    std::cout << std::endl;

    std::cout << "Liquid region (top half, k >= " << nz/2 << "):" << std::endl;
    std::cout << "  Cell count: " << liquid_count << std::endl;
    std::cout << "  Min velocity: " << std::scientific << std::setprecision(3)
              << min_liquid_velocity << " m/s" << std::endl;
    std::cout << "  Mean velocity: " << mean_liquid_velocity << " m/s" << std::endl;
    std::cout << std::endl;

    // Assertions
    EXPECT_LT(max_solid_velocity, 1e-6)
        << "Solid region velocity too high: " << max_solid_velocity
        << " m/s (expected < 1e-6 m/s)";

    EXPECT_EQ(solid_violations, 0)
        << solid_violations << " cells in solid exceed velocity threshold";

    EXPECT_GT(min_liquid_velocity, 0.05)
        << "Liquid region velocity too low (over-damped): " << min_liquid_velocity
        << " m/s (expected > 0.05 m/s)";

    if (max_solid_velocity < 1e-6 && solid_violations == 0 && min_liquid_velocity > 0.05) {
        std::cout << "✅ PASS: Darcy damping correctly suppresses solid flow" << std::endl;
    } else {
        std::cout << "❌ FAIL: Darcy damping test failed" << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
