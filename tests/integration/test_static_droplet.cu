/**
 * @file test_static_droplet.cu
 * @brief Integration test: Static droplet - Laplace pressure validation
 *
 * Test verifies:
 * 1. Spherical shape is maintained under surface tension
 * 2. Pressure jump across interface: ΔP = σ * κ = σ * 2/R
 * 3. No spurious currents (velocity should remain zero)
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/surface_tension.h"
#include "physics/fluid_lbm.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

/**
 * @brief Test static droplet maintains spherical shape
 */
TEST(StaticDropletTest, SphericalShape) {
    // Domain setup
    int nx = 64, ny = 64, nz = 64;
    float dx = 1.0e-6f;  // 1 micron
    float dt = 1.0e-9f;  // 1 nanosecond

    // Physical properties
    float sigma = 1.65f;  // N/m (Ti6Al4V)
    float nu = 1.0e-6f;   // m²/s (kinematic viscosity)
    float rho0 = 4420.0f; // kg/m³

    // Initialize solvers
    VOFSolver vof(nx, ny, nz, dx);
    SurfaceTension st(nx, ny, nz, sigma, dx);

    // Initialize spherical droplet
    float cx = nx / 2.0f;
    float cy = ny / 2.0f;
    float cz = nz / 2.0f;
    float radius = 16.0f;  // lattice units

    vof.initializeDroplet(cx, cy, cz, radius);

    // Run for a few time steps (droplet should remain spherical)
    int num_cells = nx * ny * nz;
    float* d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    for (int step = 0; step < 5; ++step) {
        vof.reconstructInterface();
        vof.computeCurvature();
        vof.convertCells();

        // Surface tension should balance pressure
        st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(),
                          d_fx, d_fy, d_fz);
    }

    // Check that droplet is still approximately spherical
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    // Sample interface cells and check radial distance
    std::vector<float> radii;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Interface cells
                if (h_fill[idx] > 0.3f && h_fill[idx] < 0.7f) {
                    float dx_r = i - cx;
                    float dy_r = j - cy;
                    float dz_r = k - cz;
                    float r = std::sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r);
                    radii.push_back(r);
                }
            }
        }
    }

    ASSERT_GT(radii.size(), 50) << "Should have interface cells";

    // Compute mean and std deviation of radii
    float mean_radius = 0.0f;
    for (float r : radii) mean_radius += r;
    mean_radius /= radii.size();

    float std_dev = 0.0f;
    for (float r : radii) {
        float diff = r - mean_radius;
        std_dev += diff * diff;
    }
    std_dev = std::sqrt(std_dev / radii.size());

    // Droplet should be approximately spherical (low std deviation)
    EXPECT_LT(std_dev / mean_radius, 0.1f)
        << "Droplet should maintain spherical shape (std_dev/radius = "
        << std_dev / mean_radius << ")";

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test Laplace pressure jump
 */
TEST(StaticDropletTest, LaplacePressure) {
    // Domain setup
    int nx = 64, ny = 64, nz = 64;
    float dx = 1.0e-6f;  // 1 micron

    // Physical properties
    float sigma = 1.65f;  // N/m
    float radius_lattice = 20.0f;
    float radius_physical = radius_lattice * dx;  // meters

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    SurfaceTension st(nx, ny, nz, sigma, dx);

    float cx = nx / 2.0f;
    float cy = ny / 2.0f;
    float cz = nz / 2.0f;

    vof.initializeDroplet(cx, cy, cz, radius_lattice);
    vof.reconstructInterface();
    vof.computeCurvature();

    // Theoretical Laplace pressure: ΔP = σ * 2/R
    float theoretical_curvature = 2.0f / radius_physical;
    float theoretical_pressure_jump = sigma * theoretical_curvature;

    // Sample curvature at interface
    int num_cells = nx * ny * nz;
    std::vector<float> h_curvature(num_cells);
    std::vector<float> h_fill(num_cells);

    vof.copyCurvatureToHost(h_curvature.data());
    vof.copyFillLevelToHost(h_fill.data());

    std::vector<float> interface_curvatures;

    for (int i = 0; i < num_cells; ++i) {
        if (h_fill[i] > 0.2f && h_fill[i] < 0.8f) {
            if (std::abs(h_curvature[i]) > 0.01f) {
                interface_curvatures.push_back(h_curvature[i]);
            }
        }
    }

    ASSERT_GT(interface_curvatures.size(), 20) << "Should have curvature values";

    // Mean curvature
    float mean_curvature = 0.0f;
    for (float kappa : interface_curvatures) {
        mean_curvature += kappa;
    }
    mean_curvature /= interface_curvatures.size();

    // Compute pressure jump
    float computed_pressure_jump = sigma * mean_curvature;

    // Relative error
    float relative_error = std::abs(computed_pressure_jump - theoretical_pressure_jump)
                          / theoretical_pressure_jump;

    EXPECT_LT(relative_error, 0.3f)
        << "Laplace pressure error: " << relative_error * 100 << "%\n"
        << "Theoretical: " << theoretical_pressure_jump << " Pa\n"
        << "Computed: " << computed_pressure_jump << " Pa";
}

/**
 * @brief Test absence of spurious currents (velocities should be zero)
 */
TEST(StaticDropletTest, NoSpuriousCurrent) {
    // This is a simplified test - full test would couple with FluidLBM
    // Here we just verify that VOF advection with zero velocity doesn't create artifacts

    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0e-6f;

    VOFSolver vof(nx, ny, nz, dx);
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);

    // Store initial state
    int num_cells = nx * ny * nz;
    std::vector<float> h_fill_initial(num_cells);
    vof.copyFillLevelToHost(h_fill_initial.data());

    // Zero velocity advection
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    cudaMemset(d_ux, 0, num_cells * sizeof(float));
    cudaMemset(d_uy, 0, num_cells * sizeof(float));
    cudaMemset(d_uz, 0, num_cells * sizeof(float));

    float dt = 1.0e-9f;
    for (int step = 0; step < 10; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    }

    // Check that droplet didn't move
    std::vector<float> h_fill_final(num_cells);
    vof.copyFillLevelToHost(h_fill_final.data());

    float max_change = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float change = std::abs(h_fill_final[i] - h_fill_initial[i]);
        max_change = std::max(max_change, change);
    }

    EXPECT_LT(max_change, 1e-5f)
        << "Droplet should not move with zero velocity (max change: " << max_change << ")";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}
