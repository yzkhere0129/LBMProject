/**
 * @file test_vof_advection_shear.cu
 * @brief Test VOF advection under shear flow - measures interface diffusion
 *
 * Physics: Interface deformation in shear flow
 * - Shear velocity: u(y) = γ * y (linear shear profile)
 * - Expected: Interface tilts, some numerical diffusion
 * - Measure: interface thickness growth rate
 *
 * Test scenario:
 * - Vertical plane interface
 * - Linear shear velocity profile u = γ*y
 * - Validate: interface tilts at correct angle, limited diffusion
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace lbm::physics;

class VOFAdvectionShearTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Initialize vertical plane interface
    void initializePlaneInterface(VOFSolver& vof, int nx, int ny, int nz, float x0) {
        std::vector<float> h_fill(nx * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    float x = static_cast<float>(i);
                    h_fill[idx] = 0.5f * (1.0f - tanhf((x - x0) / 2.0f));
                }
            }
        }

        vof.initialize(h_fill.data());
    }

    // Set linear shear velocity field: u(y) = gamma * y
    void setShearVelocity(float* d_ux, float* d_uy, float* d_uz,
                         float gamma, int nx, int ny, int nz) {
        int num_cells = nx * ny * nz;
        std::vector<float> h_ux(num_cells);
        std::vector<float> h_uy(num_cells, 0.0f);
        std::vector<float> h_uz(num_cells, 0.0f);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    float y = static_cast<float>(j);
                    h_ux[idx] = gamma * y;  // Linear shear profile
                }
            }
        }

        cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Measure interface thickness (width of 0.1 < f < 0.9 region)
    float measureInterfaceThickness(const std::vector<float>& fill,
                                    int nx, int ny, int nz) {
        int mid_y = ny / 2;
        int mid_z = nz / 2;

        int i_left = -1, i_right = -1;

        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (mid_y + ny * mid_z);
            if (fill[idx] > 0.9f && i_left < 0) {
                i_left = i;
            }
            if (fill[idx] < 0.1f && i_right < 0) {
                i_right = i;
                break;
            }
        }

        if (i_left >= 0 && i_right >= 0) {
            return static_cast<float>(i_right - i_left);
        }

        return -1.0f;
    }
};

/**
 * @brief Test 1: Shear Flow Interface Tilting
 *
 * Validates that interface tilts under shear flow
 * Shear angle: tan(θ) ≈ γ*t (for small deformations)
 */
TEST_F(VOFAdvectionShearTest, InterfaceTilting) {
    std::cout << "\n=== VOF Advection Shear: Interface Tilting ===" << std::endl;

    const int nx = 64, ny = 64, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const float x0 = nx / 2.0f;

    // Shear rate: γ = 1000 s^-1 (moderate)
    const float gamma_phys = 1000.0f;  // 1/s
    const float gamma_lattice = gamma_phys * dt;  // dimensionless per timestep
    const int num_steps = 200;
    const float total_time = dt * num_steps;  // 20 μs

    // Expected shear angle: tan(θ) ≈ γ*t (small angle approximation)
    const float expected_tan_theta = gamma_phys * total_time;
    const float expected_theta_deg = std::atan(expected_tan_theta) * 180.0f / M_PI;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Shear rate: γ = " << gamma_phys << " s^-1" << std::endl;
    std::cout << "  Time: " << total_time * 1e6 << " μs" << std::endl;
    std::cout << "  Expected tilt angle: θ ≈ " << expected_theta_deg << "°" << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializePlaneInterface(vof, nx, ny, nz, x0);

    // Setup velocity
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    setShearVelocity(d_ux, d_uy, d_uz, gamma_lattice, nx, ny, nz);

    // Advect
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();
    }

    // Measure interface tilt
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    // Find interface positions at y_bottom and y_top
    int y_bottom = ny / 4;
    int y_top = 3 * ny / 4;
    int mid_z = nz / 2;

    float x_bottom = -1.0f, x_top = -1.0f;

    // Find interface at y_bottom
    for (int i = 0; i < nx; ++i) {
        int idx = i + nx * (y_bottom + ny * mid_z);
        if (std::abs(h_fill[idx] - 0.5f) < 0.15f) {
            x_bottom = static_cast<float>(i);
            break;
        }
    }

    // Find interface at y_top
    for (int i = 0; i < nx; ++i) {
        int idx = i + nx * (y_top + ny * mid_z);
        if (std::abs(h_fill[idx] - 0.5f) < 0.15f) {
            x_top = static_cast<float>(i);
            break;
        }
    }

    ASSERT_GT(x_bottom, 0.0f) << "Could not find interface at y_bottom";
    ASSERT_GT(x_top, 0.0f) << "Could not find interface at y_top";

    float dx_interface = x_top - x_bottom;
    float dy_interface = y_top - y_bottom;
    float tan_theta = dx_interface / dy_interface;
    float theta_deg = std::atan(tan_theta) * 180.0f / M_PI;

    std::cout << "  Interface positions:" << std::endl;
    std::cout << "    At y = " << y_bottom << ": x = " << x_bottom << std::endl;
    std::cout << "    At y = " << y_top << ": x = " << x_top << std::endl;
    std::cout << "  Measured tilt: Δx = " << dx_interface << " over Δy = " << dy_interface << std::endl;
    std::cout << "  Measured angle: θ = " << theta_deg << "°" << std::endl;

    // Validation: tilt angle within 30% (numerical diffusion reduces tilt)
    float angle_error = std::abs(theta_deg - expected_theta_deg) / expected_theta_deg;
    EXPECT_LT(angle_error, 0.3f)
        << "Interface tilt angle error: " << angle_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (tilt angle within 30%)" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test 2: Interface Diffusion Measurement
 *
 * Measures numerical diffusion in shear flow
 * Interface should thicken due to numerical diffusion
 */
TEST_F(VOFAdvectionShearTest, InterfaceDiffusion) {
    std::cout << "\n=== VOF Advection Shear: Interface Diffusion ===" << std::endl;

    const int nx = 64, ny = 64, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const float x0 = nx / 2.0f;
    const float gamma_phys = 500.0f;  // 500 s^-1
    const float gamma_lattice = gamma_phys * dt;
    const int num_steps = 500;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Shear rate: γ = " << gamma_phys << " s^-1" << std::endl;
    std::cout << "  Steps: " << num_steps << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializePlaneInterface(vof, nx, ny, nz, x0);

    // Measure initial interface thickness
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());
    float thickness_initial = measureInterfaceThickness(h_fill, nx, ny, nz);

    std::cout << "  Initial interface thickness: " << thickness_initial << " cells" << std::endl;

    // Setup velocity
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    setShearVelocity(d_ux, d_uy, d_uz, gamma_lattice, nx, ny, nz);

    // Advect
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();
    }

    // Measure final interface thickness
    vof.copyFillLevelToHost(h_fill.data());
    float thickness_final = measureInterfaceThickness(h_fill, nx, ny, nz);

    std::cout << "  Final interface thickness: " << thickness_final << " cells" << std::endl;

    float thickness_growth = thickness_final - thickness_initial;
    float growth_rate = thickness_growth / num_steps;

    std::cout << "  Thickness growth: " << thickness_growth << " cells" << std::endl;
    std::cout << "  Growth rate: " << growth_rate << " cells/step" << std::endl;

    // Validation: interface should thicken but not excessively
    // Upwind scheme has inherent diffusion: expect 1-5 cells growth over 500 steps
    EXPECT_GT(thickness_growth, 0.5f)
        << "Interface should show some diffusion (upwind scheme)";

    EXPECT_LT(thickness_growth, 10.0f)
        << "Excessive interface diffusion: " << thickness_growth << " cells";

    std::cout << "  ✓ Test passed (diffusion within acceptable range)" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test 3: Mass Conservation in Shear Flow
 *
 * Validates mass conservation under shear deformation
 */
TEST_F(VOFAdvectionShearTest, MassConservationShear) {
    std::cout << "\n=== VOF Advection Shear: Mass Conservation ===" << std::endl;

    const int nx = 64, ny = 64, nz = 32;
    const float dx = 2e-6f;
    const float dt = 1e-7f;
    const int num_cells = nx * ny * nz;

    const float x0 = nx / 2.0f;
    const float gamma_phys = 1000.0f;
    const float gamma_lattice = gamma_phys * dt;
    const int num_steps = 300;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Shear rate: γ = " << gamma_phys << " s^-1" << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializePlaneInterface(vof, nx, ny, nz, x0);

    float mass_initial = vof.computeTotalMass();
    std::cout << "  Initial mass: M0 = " << mass_initial << std::endl;

    // Setup velocity
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    setShearVelocity(d_ux, d_uy, d_uz, gamma_lattice, nx, ny, nz);

    // Advect
    for (int step = 0; step < num_steps; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();
    }

    float mass_final = vof.computeTotalMass();
    float mass_error = std::abs(mass_final - mass_initial) / mass_initial;

    std::cout << "  Final mass: M = " << mass_final << std::endl;
    std::cout << "  Mass error: " << mass_error * 100.0f << "%" << std::endl;

    // Validation: mass conserved within 2% (shear adds more numerical error)
    EXPECT_LT(mass_error, 0.02f)
        << "Mass conservation violated in shear flow: error = " << mass_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (mass conserved within 2%)" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
