/**
 * @file test_divergence_free.cu
 * @brief Test that LBM velocity field is divergence-free
 *
 * This test creates a simple fluid simulation with uniform flow
 * and verifies that div(u) ≈ 0, which is required for incompressible flow.
 *
 * Test cases:
 * 1. Zero force: uniform flow should remain divergence-free
 * 2. Uniform force: should maintain incompressibility
 * 3. Spatially-varying force: should still be incompressible
 */

#include "physics/fluid_lbm.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace lbm::physics;

// Compute divergence using central differences
void computeDivergence(const std::vector<float>& ux,
                      const std::vector<float>& uy,
                      const std::vector<float>& uz,
                      int nx, int ny, int nz,
                      float dx,
                      std::vector<float>& div)
{
    div.resize(nx * ny * nz, 0.0f);

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = i + nx * (j + ny * k);

                // Central differences
                float dvx_dx = (ux[idx + 1] - ux[idx - 1]) / (2.0f * dx);
                float dvy_dy = (uy[idx + nx] - uy[idx - nx]) / (2.0f * dx);
                float dvz_dz = (uz[idx + nx*ny] - uz[idx - nx*ny]) / (2.0f * dx);

                div[idx] = dvx_dx + dvy_dy + dvz_dz;
            }
        }
    }
}

// Print divergence statistics
void printDivergenceStats(const std::vector<float>& div, const std::string& test_name) {
    float min_div = *std::min_element(div.begin(), div.end());
    float max_div = *std::max_element(div.begin(), div.end());

    float mean = 0.0f;
    float rms = 0.0f;
    for (float d : div) {
        mean += d;
        rms += d * d;
    }
    mean /= div.size();
    rms = std::sqrt(rms / div.size());

    int count_high = 0;
    const float threshold = 1e-3f;
    for (float d : div) {
        if (std::abs(d) > threshold) count_high++;
    }

    std::cout << "\n" << test_name << " - Divergence Statistics:\n";
    std::cout << "  Min:  " << min_div << "\n";
    std::cout << "  Max:  " << max_div << "\n";
    std::cout << "  Mean: " << mean << "\n";
    std::cout << "  RMS:  " << rms << "\n";
    std::cout << "  Cells with |div| > " << threshold << ": "
              << count_high << " (" << 100.0f * count_high / div.size() << "%)\n";
}

bool testCase1_ZeroForce() {
    std::cout << "\n========================================\n";
    std::cout << "TEST 1: Zero Force - Uniform Flow\n";
    std::cout << "========================================\n";

    // Small domain for quick test
    const int nx = 32, ny = 32, nz = 32;
    const float dx = 1e-5f; // 10 μm (not used here, just for divergence calculation)
    const float nu_lattice = 0.1f; // Lattice viscosity (needs omega in stable range 0.5-1.9)
    const float rho0 = 1.0f; // Lattice density (normalized)

    // Create solver
    FluidLBM solver(nx, ny, nz, nu_lattice, rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC);

    // Initialize with uniform velocity (0.01, 0, 0)
    const float u0 = 0.01f;
    solver.initialize(rho0, u0, 0.0f, 0.0f);

    // Run 100 steps with NO force
    for (int step = 0; step < 100; ++step) {
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();
        solver.computeMacroscopic();
    }

    // Get velocity field
    std::vector<float> ux(nx * ny * nz);
    std::vector<float> uy(nx * ny * nz);
    std::vector<float> uz(nx * ny * nz);
    solver.copyVelocityToHost(ux.data(), uy.data(), uz.data());

    // Compute divergence
    std::vector<float> div;
    computeDivergence(ux, uy, uz, nx, ny, nz, dx, div);

    // Print stats
    printDivergenceStats(div, "TEST 1");

    // Check: max divergence should be < 1e-6
    float max_div = *std::max_element(div.begin(), div.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });

    bool passed = std::abs(max_div) < 1e-3f;
    std::cout << "\nRESULT: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "  Expected: |div| < 1e-3\n";
    std::cout << "  Actual:   |div| = " << std::abs(max_div) << "\n";

    return passed;
}

bool testCase2_UniformForce() {
    std::cout << "\n========================================\n";
    std::cout << "TEST 2: Uniform Force (Gravity)\n";
    std::cout << "========================================\n";

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 1e-5f;
    const float nu_lattice = 0.1f;
    const float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu_lattice, rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC);

    // Initialize with zero velocity
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Apply uniform gravity force (in lattice units)
    const float fx = 0.0f;
    const float fy = 0.0f;
    const float fz = -1e-5f; // Small downward force in lattice units

    // Run 100 steps with uniform force
    for (int step = 0; step < 100; ++step) {
        solver.collisionBGK(fx, fy, fz);
        solver.streaming();
        solver.computeMacroscopic();
    }

    // Get velocity field
    std::vector<float> ux(nx * ny * nz);
    std::vector<float> uy(nx * ny * nz);
    std::vector<float> uz(nx * ny * nz);
    solver.copyVelocityToHost(ux.data(), uy.data(), uz.data());

    // Compute divergence
    std::vector<float> div;
    computeDivergence(ux, uy, uz, nx, ny, nz, dx, div);

    printDivergenceStats(div, "TEST 2");

    float max_div = *std::max_element(div.begin(), div.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });

    bool passed = std::abs(max_div) < 1e-3f;
    std::cout << "\nRESULT: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "  Expected: |div| < 1e-3\n";
    std::cout << "  Actual:   |div| = " << std::abs(max_div) << "\n";

    return passed;
}

bool testCase3_SpatiallyVaryingForce() {
    std::cout << "\n========================================\n";
    std::cout << "TEST 3: Spatially-Varying Force (Buoyancy)\n";
    std::cout << "========================================\n";

    const int nx = 32, ny = 32, nz = 32;
    const int num_cells = nx * ny * nz;
    const float dx = 1e-5f;
    const float nu_lattice = 0.1f;
    const float rho0 = 1.0f;

    FluidLBM solver(nx, ny, nz, nu_lattice, rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC);

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Create DIVERGENCE-FREE spatially-varying force (curl-driven)
    // This is essential: div(F) = 0 is required for LBM to maintain div(u) = 0
    // Use a rotational force field: F = curl(A) where A is a vector potential
    std::vector<float> h_fx(num_cells);
    std::vector<float> h_fy(num_cells);
    std::vector<float> h_fz(num_cells, 0.0f);

    float center_x = nx / 2.0f;
    float center_y = ny / 2.0f;
    float force_strength = 1e-6f;  // Small force magnitude

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Rotational force field (vortex-like): F = (-y', x', 0) * strength
                // This has div(F) = 0 automatically
                float dx_rel = (i - center_x) / nx;
                float dy_rel = (j - center_y) / ny;

                h_fx[idx] = -dy_rel * force_strength;
                h_fy[idx] = dx_rel * force_strength;
                // h_fz[idx] already initialized to 0
            }
        }
    }

    // Copy to device
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_fx, h_fx.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, h_fy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, h_fz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Run 100 steps
    for (int step = 0; step < 100; ++step) {
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();
        solver.computeMacroscopic();
    }

    // Get velocity field
    std::vector<float> ux(num_cells);
    std::vector<float> uy(num_cells);
    std::vector<float> uz(num_cells);
    solver.copyVelocityToHost(ux.data(), uy.data(), uz.data());

    // Compute divergence
    std::vector<float> div;
    computeDivergence(ux, uy, uz, nx, ny, nz, dx, div);

    printDivergenceStats(div, "TEST 3 - VELOCITY");

    // Also check force divergence
    std::vector<float> div_force;
    computeDivergence(h_fx, h_fy, h_fz, nx, ny, nz, dx, div_force);
    printDivergenceStats(div_force, "TEST 3 - FORCE");

    float max_div = *std::max_element(div.begin(), div.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    bool passed = std::abs(max_div) < 1e-3f;
    std::cout << "\nRESULT: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "  Expected: |div| < 1e-3\n";
    std::cout << "  Actual:   |div| = " << std::abs(max_div) << "\n";

    return passed;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "DIVERGENCE-FREE VELOCITY FIELD TEST\n";
    std::cout << "========================================\n";
    std::cout << "\nPurpose: Verify that LBM enforces incompressibility\n";
    std::cout << "Requirement: div(u) < 1e-3 for all test cases\n";

    bool test1 = testCase1_ZeroForce();
    bool test2 = testCase2_UniformForce();
    bool test3 = testCase3_SpatiallyVaryingForce();

    std::cout << "\n========================================\n";
    std::cout << "FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Test 1 (Zero Force):        " << (test1 ? "PASS" : "FAIL") << "\n";
    std::cout << "Test 2 (Uniform Force):     " << (test2 ? "PASS" : "FAIL") << "\n";
    std::cout << "Test 3 (Varying Force):     " << (test3 ? "PASS" : "FAIL") << "\n";
    std::cout << "\n";

    if (test1 && test2 && test3) {
        std::cout << "ALL TESTS PASSED ✓\n";
        std::cout << "LBM velocity field is divergence-free!\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED ✗\n";
        std::cout << "LBM is NOT enforcing incompressibility!\n";
        std::cout << "This will cause mass conservation violations in VOF.\n";
        return 1;
    }
}
