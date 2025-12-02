/**
 * @file test_vof_advection_uniform.cu
 * @brief Test VOF advection with uniform velocity - validates mass conservation
 *
 * Physics: Interface advection under uniform velocity field
 * - Advection equation: ∂f/∂t + ∇·(f·u) = 0
 * - Expected: Interface translates without shape change
 * - Mass conservation: Σf_i = constant
 *
 * Test scenario:
 * - Plane interface perpendicular to flow
 * - Uniform velocity u = constant
 * - Periodic boundaries in x,y
 * - Validate: displacement matches u*t, mass conserved < 1%
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace lbm::physics;

class VOFAdvectionUniformTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Initialize vertical plane interface at x = x0
    void initializePlaneInterface(VOFSolver& vof, int nx, int ny, int nz, float x0) {
        std::vector<float> h_fill(nx * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    float x = static_cast<float>(i);

                    // Smooth interface using tanh profile (width ~ 2 cells)
                    h_fill[idx] = 0.5f * (1.0f - tanhf((x - x0) / 2.0f));
                }
            }
        }

        vof.initialize(h_fill.data());
    }

    // Find interface position (x where f ≈ 0.5)
    float findInterfaceX(const std::vector<float>& fill, int nx, int ny, int nz) {
        int mid_y = ny / 2;
        int mid_z = nz / 2;

        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (mid_y + ny * mid_z);
            if (std::abs(fill[idx] - 0.5f) < 0.1f) {
                return static_cast<float>(i);
            }
        }

        // Fallback: find transition
        for (int i = 0; i < nx - 1; ++i) {
            int idx = i + nx * (mid_y + ny * mid_z);
            int idx_next = (i + 1) + nx * (mid_y + ny * mid_z);

            if (fill[idx] > 0.7f && fill[idx_next] < 0.3f) {
                return static_cast<float>(i) + 0.5f;
            }
        }

        return -1.0f;
    }

    // Set uniform velocity field
    void setUniformVelocity(float* d_ux, float* d_uy, float* d_uz,
                           float ux, float uy, float uz, int num_cells) {
        std::vector<float> h_ux(num_cells, ux);
        std::vector<float> h_uy(num_cells, uy);
        std::vector<float> h_uz(num_cells, uz);

        cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }
};

/**
 * @brief Test 1: Interface Displacement
 *
 * Validates that interface moves with velocity u*t
 */
TEST_F(VOFAdvectionUniformTest, InterfaceDisplacement) {
    std::cout << "\n=== VOF Advection Uniform: Interface Displacement ===" << std::endl;

    // Domain setup
    const int nx = 64, ny = 32, nz = 32;
    const float dx = 2e-6f;  // 2 μm
    const float dt = 1e-7f;  // 0.1 μs
    const int num_cells = nx * ny * nz;

    // Initial interface at center
    const float x0 = nx / 2.0f;  // 32 cells

    // Velocity and time
    const float u_phys = 0.1f;  // 0.1 m/s
    const int num_steps = 500;
    const float total_time = dt * num_steps;  // 50 μs

    // Expected displacement
    const float expected_disp_m = u_phys * total_time;  // 5 μm
    const float expected_disp_cells = expected_disp_m / dx;  // 2.5 cells

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Initial interface: x = " << x0 << " cells" << std::endl;
    std::cout << "  Velocity: u = " << u_phys << " m/s" << std::endl;
    std::cout << "  Time: " << total_time * 1e6 << " μs (" << num_steps << " steps)" << std::endl;
    std::cout << "  Expected displacement: " << expected_disp_cells << " cells" << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializePlaneInterface(vof, nx, ny, nz, x0);

    // Get initial interface position
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());
    float x_initial = findInterfaceX(h_fill, nx, ny, nz);

    std::cout << "  Measured initial interface: x = " << x_initial << " cells" << std::endl;

    // Setup velocity
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    setUniformVelocity(d_ux, d_uy, d_uz, u_phys, 0.0f, 0.0f, num_cells);

    // Advect
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();
    }

    // Measure final interface position
    vof.copyFillLevelToHost(h_fill.data());
    float x_final = findInterfaceX(h_fill, nx, ny, nz);
    float displacement = x_final - x_initial;

    std::cout << "  Final interface: x = " << x_final << " cells" << std::endl;
    std::cout << "  Measured displacement: " << displacement << " cells" << std::endl;
    std::cout << "  Error: " << std::abs(displacement - expected_disp_cells) << " cells" << std::endl;

    // Validation: displacement within 0.5 cells (accounting for diffusion)
    float error_cells = std::abs(displacement - expected_disp_cells);
    EXPECT_LT(error_cells, 0.5f)
        << "Interface displacement error too large: " << error_cells << " cells";

    std::cout << "  ✓ Test passed" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test 2: Mass Conservation
 *
 * Validates that total liquid mass is conserved during advection
 * Strict tolerance: < 1% error
 */
TEST_F(VOFAdvectionUniformTest, MassConservation) {
    std::cout << "\n=== VOF Advection Uniform: Mass Conservation ===" << std::endl;

    // Domain setup (large domain to keep interface away from boundaries)
    const int nx = 100, ny = 32, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    // Initial interface at center
    const float x0 = nx / 2.0f;

    // Moderate velocity and short time
    const float u_phys = 0.05f;  // 0.05 m/s
    const int num_steps = 200;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Advection: " << num_steps << " steps" << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializePlaneInterface(vof, nx, ny, nz, x0);

    // Record initial mass
    float mass_initial = vof.computeTotalMass();
    std::cout << "  Initial mass: M0 = " << mass_initial << std::endl;

    // Theoretical mass: half domain filled
    float theoretical_mass = num_cells / 2.0f;
    std::cout << "  Theoretical mass: " << theoretical_mass << std::endl;

    // Setup velocity
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    setUniformVelocity(d_ux, d_uy, d_uz, u_phys, 0.0f, 0.0f, num_cells);

    // Advect with periodic mass checks
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();

        if ((step + 1) % 50 == 0) {
            float mass = vof.computeTotalMass();
            float error = std::abs(mass - mass_initial) / mass_initial;
            std::cout << "    Step " << (step + 1) << ": M = " << mass
                      << ", error = " << error * 100.0f << "%" << std::endl;
        }
    }

    // Final mass check
    float mass_final = vof.computeTotalMass();
    float mass_error = std::abs(mass_final - mass_initial) / mass_initial;

    std::cout << "  Final mass: M = " << mass_final << std::endl;
    std::cout << "  Mass change: ΔM = " << (mass_final - mass_initial) << std::endl;
    std::cout << "  Relative error: " << mass_error * 100.0f << "%" << std::endl;

    // Validation: mass conserved within 1% (upwind scheme has numerical diffusion)
    EXPECT_LT(mass_error, 0.01f)
        << "Mass conservation violated: error = " << mass_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (mass conserved within 1%)" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test 3: Interface Shape Preservation
 *
 * Validates that plane interface remains planar (no distortion)
 */
TEST_F(VOFAdvectionUniformTest, InterfaceShapePreservation) {
    std::cout << "\n=== VOF Advection Uniform: Interface Shape Preservation ===" << std::endl;

    const int nx = 64, ny = 64, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const float x0 = nx / 2.0f;
    const float u_phys = 0.1f;
    const int num_steps = 300;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializePlaneInterface(vof, nx, ny, nz, x0);

    // Setup velocity
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    setUniformVelocity(d_ux, d_uy, d_uz, u_phys, 0.0f, 0.0f, num_cells);

    // Advect
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();
    }

    // Check interface planarity
    // Sample interface positions at different y,z locations
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    std::vector<float> interface_positions;

    // Sample at multiple y,z positions
    for (int k = nz / 4; k < 3 * nz / 4; k += 4) {
        for (int j = ny / 4; j < 3 * ny / 4; j += 4) {
            // Find interface x at this (y,z)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                if (std::abs(h_fill[idx] - 0.5f) < 0.15f) {
                    interface_positions.push_back(static_cast<float>(i));
                    break;
                }
            }
        }
    }

    ASSERT_GT(interface_positions.size(), 5)
        << "Could not find enough interface samples";

    // Compute mean and standard deviation
    float mean_x = 0.0f;
    for (float x : interface_positions) {
        mean_x += x;
    }
    mean_x /= interface_positions.size();

    float std_dev = 0.0f;
    for (float x : interface_positions) {
        float diff = x - mean_x;
        std_dev += diff * diff;
    }
    std_dev = std::sqrt(std_dev / interface_positions.size());

    std::cout << "  Interface samples: " << interface_positions.size() << std::endl;
    std::cout << "  Mean interface position: " << mean_x << " cells" << std::endl;
    std::cout << "  Standard deviation: " << std_dev << " cells" << std::endl;

    // Validation: interface remains planar (std dev < 1 cell)
    EXPECT_LT(std_dev, 1.0f)
        << "Interface distorted: std dev = " << std_dev << " cells";

    std::cout << "  ✓ Test passed (interface remains planar)" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
