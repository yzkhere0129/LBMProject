/**
 * @file test_marangoni_force_validation.cu
 * @brief Integration test for Marangoni force driven by temperature gradient
 *
 * This test validates the coupling between thermal gradients, surface tension,
 * and fluid motion (Marangoni effect) using the Guo forcing scheme.
 *
 * Physics:
 * - Marangoni stress: τ = dσ/dT * ∇_s T (tangential surface stress)
 * - Induced velocity: u ∝ (dσ/dT * ΔT * L) / (μ * h)
 * - Expected velocity scale: O(0.1 - 1.0 m/s) for metal additive manufacturing
 *
 * Test configuration:
 * - Liquid pool with free surface (VOF interface)
 * - Temperature gradient applied across surface
 * - Measure induced velocity and compare with analytical scaling
 *
 * References:
 * - Levich & Krylov (1969). Surface-Tension-Driven Phenomena
 * - Khairallah et al. (2016). Laser powder-bed fusion additive manufacturing
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "physics/fluid_lbm.h"
#include "physics/force_accumulator.h"
#include "core/lattice_d3q19.h"

using namespace lbm::core;

class MarangoniForceValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    /**
     * @brief Analytical Marangoni velocity estimate
     *
     * From Levich & Krylov (1969):
     * u ~ (dσ/dT * ΔT * L) / (μ * h)
     *
     * where:
     * - dσ/dT: temperature coefficient of surface tension [N/(m·K)]
     * - ΔT: temperature difference across interface [K]
     * - L: characteristic length of gradient [m]
     * - μ: dynamic viscosity [Pa·s]
     * - h: liquid depth [m]
     */
    float estimateMarangoniVelocity(float dsigma_dT, float deltaT,
                                      float L, float mu, float h) {
        return std::abs(dsigma_dT) * deltaT * L / (mu * h);
    }

    /**
     * @brief Check if steady state is reached
     */
    bool isConverged(const std::vector<float>& u_old,
                     const std::vector<float>& u_new,
                     float tolerance) {
        float max_change = 0.0f;
        for (size_t i = 0; i < u_old.size(); ++i) {
            max_change = std::max(max_change, std::abs(u_new[i] - u_old[i]));
        }
        return max_change < tolerance;
    }
};

/**
 * Test 1: Marangoni force from horizontal temperature gradient
 *
 * Configuration:
 * - Thin liquid layer with free surface at top
 * - Temperature gradient in x-direction (cold on left, hot on right)
 * - Marangoni stress drives flow from hot to cold (for metals, dσ/dT < 0)
 * - Physics: Low σ at hot side pulls fluid toward high σ at cold side
 */
TEST_F(MarangoniForceValidationTest, HorizontalTemperatureGradient) {
    // Domain configuration (quasi-2D liquid layer)
    const int nx = 32;  // Gradient direction
    const int ny = 8;   // Depth (wall at bottom, free surface at top)
    const int nz = 4;   // Thin in spanwise
    const int num_cells = nx * ny * nz;

    // Physical parameters (Ti-6Al-4V liquid at ~1943 K)
    const float dx = 2.0e-6f;           // 2 μm lattice spacing
    const float dt = 1.0e-7f;           // 0.1 μs time step
    const float rho = 4110.0f;          // kg/m³
    const float nu = 4.5e-7f;           // m²/s (kinematic viscosity)
    const float mu = rho * nu;          // Dynamic viscosity [Pa·s]
    const float dsigma_dT = -2.6e-4f;   // N/(m·K) (negative for metals)

    // Temperature field (linear gradient in x-direction)
    const float T_cold = 1943.0f;       // K (melting point)
    const float T_hot = 2443.0f;        // K (T_m + 500K)
    const float deltaT = T_hot - T_cold;

    // Characteristic length and depth
    const float L = nx * dx;            // Gradient length scale
    const float h = ny * dx;            // Liquid depth

    // Analytical velocity estimate
    float u_analytical = estimateMarangoniVelocity(dsigma_dT, deltaT, L, mu, h);

    std::cout << "\n=== Marangoni Force Validation Test ===" << std::endl;
    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Lattice spacing: " << (dx * 1e6) << " μm" << std::endl;
    std::cout << "Time step: " << (dt * 1e6) << " μs" << std::endl;
    std::cout << "\nPhysical parameters:" << std::endl;
    std::cout << "  Density: " << rho << " kg/m³" << std::endl;
    std::cout << "  Kinematic viscosity: " << (nu * 1e6) << " mm²/s" << std::endl;
    std::cout << "  dσ/dT: " << dsigma_dT << " N/(m·K)" << std::endl;
    std::cout << "  Temperature range: " << T_cold << " - " << T_hot << " K" << std::endl;
    std::cout << "  ΔT: " << deltaT << " K" << std::endl;
    std::cout << "\nAnalytical velocity estimate: " << u_analytical << " m/s" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Initialize FluidLBM solver
    // WALL at bottom (y=0), PERIODIC elsewhere (simplified free surface)
    lbm::physics::FluidLBM solver(nx, ny, nz, nu, rho,
                    lbm::physics::BoundaryType::PERIODIC,
                    lbm::physics::BoundaryType::WALL,      // Bottom wall
                    lbm::physics::BoundaryType::PERIODIC,
                    dt, dx);

    solver.initialize(rho, 0.0f, 0.0f, 0.0f);

    // Create temperature field (linear gradient in x)
    std::vector<float> h_temperature(num_cells);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = ix + iy * nx + iz * nx * ny;
                // Linear interpolation: T(x) = T_cold + (T_hot - T_cold) * x/L
                float x_frac = static_cast<float>(ix) / (nx - 1);
                h_temperature[idx] = T_cold + deltaT * x_frac;
            }
        }
    }

    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // Create smoothed VOF field (interface at top)
    // FIX (2026-01-12): Use smoothed interface to avoid excessive |∇f|
    // Sharp interfaces (f jumping from 1→0.5 in one cell) produce |∇f| ~ 2.5e5 1/m
    // which leads to unrealistically large Marangoni forces
    std::vector<float> h_fill_level(num_cells);
    std::vector<float3> h_normals(num_cells);
    const int interface_width = 3;  // Smooth interface over 3 cells
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = ix + iy * nx + iz * nx * ny;

                // Smooth interface from y=ny-interface_width to y=ny-1
                if (iy < ny - interface_width) {
                    h_fill_level[idx] = 1.0f;  // Full liquid
                    h_normals[idx] = make_float3(0.0f, 0.0f, 0.0f);
                } else {
                    // Linear transition: f goes from 1.0 (at y=ny-interface_width) to 0.3 (at y=ny-1)
                    int dist_from_liquid = iy - (ny - interface_width);
                    float frac = static_cast<float>(dist_from_liquid) / (interface_width - 1);
                    h_fill_level[idx] = 1.0f - 0.7f * frac;  // Range: [1.0, 0.3]

                    // Set normal for ALL interface cells (kernel requires |n| > 0.01)
                    h_normals[idx] = make_float3(0.0f, 1.0f, 0.0f);
                }
            }
        }
    }

    float* d_fill_level;
    float3* d_normals;
    cudaMalloc(&d_fill_level, num_cells * sizeof(float));
    cudaMalloc(&d_normals, num_cells * sizeof(float3));
    cudaMemcpy(d_fill_level, h_fill_level.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals.data(), num_cells * sizeof(float3),
               cudaMemcpyHostToDevice);

    // Initialize ForceAccumulator
    lbm::physics::ForceAccumulator forces(nx, ny, nz);

    // Time evolution
    const int max_steps = 10000;
    const int check_interval = 500;
    const float convergence_tol = 1e-7f;

    std::vector<float> u_old(num_cells, 0.0f);
    std::vector<float> u_new(num_cells);

    bool converged = false;
    int convergence_step = -1;

    std::cout << "Running simulation with Marangoni forcing..." << std::endl;

    for (int step = 1; step <= max_steps; ++step) {
        // Reset and accumulate forces
        forces.reset();

        // Add Marangoni force
        // FIX (2026-01-12): Use h_interface=3 to match the smoothed interface width
        forces.addMarangoniForce(d_temperature, d_fill_level, d_normals,
                                 dsigma_dT, nx, ny, nz, dx, 3.0f);

        // Convert to lattice units
        forces.convertToLatticeUnits(dx, dt, rho);

        // Apply forces to fluid
        solver.computeMacroscopic();
        solver.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        solver.streaming();
        solver.applyBoundaryConditions(1);

        // Check convergence
        if (step % check_interval == 0) {
            solver.copyVelocityToHost(u_new.data(), nullptr, nullptr);

            if (isConverged(u_old, u_new, convergence_tol)) {
                converged = true;
                convergence_step = step;
                std::cout << "Converged at step " << step << std::endl;
                break;
            }

            u_old = u_new;

            // Print max velocity
            float max_u = *std::max_element(u_new.begin(), u_new.end(),
                [](float a, float b) { return std::abs(a) < std::abs(b); });
            float max_u_physical = max_u * dx / dt;  // Convert to physical units
            std::cout << "Step " << step << ": max velocity = "
                      << max_u_physical << " m/s" << std::endl;
        }
    }

    // Final state
    solver.computeMacroscopic();

    std::vector<float> ux(num_cells);
    std::vector<float> uy(num_cells);
    std::vector<float> uz(num_cells);
    solver.copyVelocityToHost(ux.data(), uy.data(), uz.data());

    // Extract velocity at interface (top layer, y = ny-1)
    std::vector<float> interface_velocity(nx, 0.0f);
    for (int ix = 0; ix < nx; ++ix) {
        float sum = 0.0f;
        for (int iz = 0; iz < nz; ++iz) {
            int idx = ix + (ny - 1) * nx + iz * nx * ny;
            sum += ux[idx];
        }
        interface_velocity[ix] = sum / nz;
    }

    // Find maximum interface velocity (in lattice units)
    auto it_max = std::max_element(interface_velocity.begin(), interface_velocity.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    float u_max_lattice = *it_max;
    float u_max_physical = u_max_lattice * dx / dt;  // Convert to m/s

    // Compute overall statistics
    float avg_interface_u = std::accumulate(interface_velocity.begin(),
                                            interface_velocity.end(), 0.0f) / nx;
    avg_interface_u *= dx / dt;  // Convert to m/s

    // Print results
    std::cout << "\n=== Validation Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Convergence: " << (converged ? "YES" : "NO") << std::endl;
    if (converged) {
        std::cout << "Converged at step: " << convergence_step << std::endl;
    }
    std::cout << "\nInterface velocity:" << std::endl;
    std::cout << "  Maximum:    " << u_max_physical << " m/s" << std::endl;
    std::cout << "  Average:    " << avg_interface_u << " m/s" << std::endl;
    std::cout << "  Analytical: " << u_analytical << " m/s" << std::endl;
    std::cout << "  Ratio (numerical/analytical): "
              << (u_max_physical / u_analytical) << std::endl;
    std::cout << "==========================\n" << std::endl;

    // Print velocity profile along interface
    std::cout << "Interface velocity profile (x-direction):" << std::endl;
    std::cout << "  x\tVelocity (m/s)\tTemperature (K)" << std::endl;
    for (int ix = 0; ix < nx; ix += 4) {  // Sample every 4th point
        float u_phys = interface_velocity[ix] * dx / dt;
        int idx = ix + (ny - 1) * nx;  // Top layer, z=0
        std::cout << "  " << ix << "\t" << u_phys << "\t"
                  << h_temperature[idx] << std::endl;
    }

    // Save profile to file
    std::ofstream file("marangoni_velocity_profile.txt");
    if (file.is_open()) {
        file << "# Marangoni Force Validation\n";
        file << "# x\tVelocity_m/s\tTemperature_K\n";
        for (int ix = 0; ix < nx; ++ix) {
            float u_phys = interface_velocity[ix] * dx / dt;
            int idx = ix + (ny - 1) * nx;
            file << ix << "\t" << u_phys << "\t" << h_temperature[idx] << "\n";
        }
        file.close();
        std::cout << "\nProfile saved to: marangoni_velocity_profile.txt\n" << std::endl;
    }

    // Validation assertions
    EXPECT_TRUE(converged) << "Simulation did not converge";
    EXPECT_GT(u_max_physical, 0.0f) << "No Marangoni flow generated";

    // For metals with dσ/dT < 0, flow is from hot to cold
    // Since temperature increases with x, and dσ/dT < 0, flow should be in -x direction
    // FIX (2026-01-12): Corrected physics - Marangoni flow goes from LOW σ to HIGH σ
    // For dσ/dT < 0: hot (low σ) → cold (high σ), which is -x direction
    EXPECT_LT(avg_interface_u, 0.0f)
        << "Flow direction incorrect (should be from hot to cold for dσ/dT < 0)";

    // Velocity should be within order of magnitude of analytical estimate
    // Analytical estimate is rough, so we allow 0.1x to 10x range
    float ratio = u_max_physical / u_analytical;
    EXPECT_GT(ratio, 0.1f)
        << "Velocity too low compared to analytical estimate (factor of "
        << (1.0f / ratio) << ")";
    EXPECT_LT(ratio, 10.0f)
        << "Velocity too high compared to analytical estimate (factor of "
        << ratio << ")";

    // For better accuracy, expect within factor of 2-3
    EXPECT_GT(ratio, 0.3f)
        << "Target: velocity within factor of 3 of analytical estimate";
    EXPECT_LT(ratio, 3.0f)
        << "Target: velocity within factor of 3 of analytical estimate";

    // Check velocity magnitude is greater at cold side than hot side (for dσ/dT < 0)
    // Flow is in -x direction, so at cold side (x=0), velocity should be more negative
    // FIX (2026-01-12): For dσ/dT < 0, flow is from hot to cold (-x direction)
    float u_cold_phys = interface_velocity[0] * dx / dt;
    float u_hot_phys = interface_velocity[nx-1] * dx / dt;
    EXPECT_LT(u_cold_phys, u_hot_phys)
        << "Velocity should be more negative at cold end (stronger outflow) for dσ/dT < 0";

    // Free memory
    cudaFree(d_temperature);
    cudaFree(d_fill_level);
    cudaFree(d_normals);
}

/**
 * Test 2: Marangoni force direction with positive dσ/dT
 *
 * For fluids with positive dσ/dT (e.g., water), flow is from cold to hot.
 * FIX (2026-01-12): Corrected physics
 * - dσ/dT > 0 means σ increases with T
 * - Hot region has HIGH σ, cold region has LOW σ
 * - Flow goes from LOW σ to HIGH σ, i.e., cold to hot (+x direction)
 * This test verifies the force direction is correct for both signs of dσ/dT.
 */
TEST_F(MarangoniForceValidationTest, ForcDirectionWithPositiveDsigmaDT) {
    const int nx = 16, ny = 8, nz = 4;
    const int num_cells = nx * ny * nz;

    const float dx = 1.0e-6f;
    const float dt = 1.0e-7f;
    const float rho = 1000.0f;      // Water-like
    const float nu = 1.0e-6f;       // Water viscosity
    const float dsigma_dT = 1.5e-4f;  // POSITIVE (water-like)

    const float T_cold = 300.0f;
    const float T_hot = 350.0f;

    lbm::physics::FluidLBM solver(nx, ny, nz, nu, rho,
                    lbm::physics::BoundaryType::PERIODIC,
                    lbm::physics::BoundaryType::WALL,
                    lbm::physics::BoundaryType::PERIODIC,
                    dt, dx);
    solver.initialize(rho, 0.0f, 0.0f, 0.0f);

    // Temperature gradient
    std::vector<float> h_temperature(num_cells);
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = ix + iy * nx + iz * nx * ny;
                float x_frac = static_cast<float>(ix) / (nx - 1);
                h_temperature[idx] = T_cold + (T_hot - T_cold) * x_frac;
            }
        }
    }

    float* d_temperature;
    cudaMalloc(&d_temperature, num_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);

    // VOF field (interface at top, smoothed)
    // FIX (2026-01-12): Use smoothed interface to avoid excessive |∇f|
    std::vector<float> h_fill_level(num_cells);
    std::vector<float3> h_normals(num_cells);
    const int interface_width = 3;
    for (int idx = 0; idx < num_cells; ++idx) {
        int iy = (idx / nx) % ny;

        if (iy < ny - interface_width) {
            h_fill_level[idx] = 1.0f;
            h_normals[idx] = make_float3(0.0f, 0.0f, 0.0f);
        } else {
            int dist_from_liquid = iy - (ny - interface_width);
            float frac = static_cast<float>(dist_from_liquid) / (interface_width - 1);
            h_fill_level[idx] = 1.0f - 0.7f * frac;

            // Set normal for ALL interface cells
            h_normals[idx] = make_float3(0.0f, 1.0f, 0.0f);
        }
    }

    float* d_fill_level;
    float3* d_normals;
    cudaMalloc(&d_fill_level, num_cells * sizeof(float));
    cudaMalloc(&d_normals, num_cells * sizeof(float3));
    cudaMemcpy(d_fill_level, h_fill_level.data(), num_cells * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals.data(), num_cells * sizeof(float3),
               cudaMemcpyHostToDevice);

    lbm::physics::ForceAccumulator forces(nx, ny, nz);

    std::cout << "\n=== Positive dσ/dT Test ===" << std::endl;
    std::cout << "dσ/dT = " << dsigma_dT << " (positive, water-like)" << std::endl;

    // Run simulation
    for (int step = 0; step < 5000; ++step) {
        forces.reset();
        // FIX (2026-01-12): Use h_interface=3 to match the smoothed interface width
        forces.addMarangoniForce(d_temperature, d_fill_level, d_normals,
                                 dsigma_dT, nx, ny, nz, dx, 3.0f);
        forces.convertToLatticeUnits(dx, dt, rho);

        solver.computeMacroscopic();
        solver.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        solver.streaming();
        solver.applyBoundaryConditions(1);
    }

    solver.computeMacroscopic();

    std::vector<float> ux(num_cells);
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    // Extract interface velocity
    float u_cold_end = 0.0f;
    float u_hot_end = 0.0f;
    for (int iz = 0; iz < nz; ++iz) {
        int idx_cold = 0 + (ny - 1) * nx + iz * nx * ny;
        int idx_hot = (nx - 1) + (ny - 1) * nx + iz * nx * ny;
        u_cold_end += ux[idx_cold];
        u_hot_end += ux[idx_hot];
    }
    u_cold_end /= nz;
    u_hot_end /= nz;

    std::cout << "Velocity at cold end: " << (u_cold_end * dx / dt) << " m/s" << std::endl;
    std::cout << "Velocity at hot end:  " << (u_hot_end * dx / dt) << " m/s" << std::endl;

    // For positive dσ/dT, flow is from cold to hot (positive x-direction)
    // FIX (2026-01-12): Corrected physics - high σ at hot side pulls fluid from cold side
    // So average velocity should be positive
    float avg_u = std::accumulate(ux.begin(), ux.end(), 0.0f) / ux.size();
    avg_u *= dx / dt;

    std::cout << "Average velocity: " << avg_u << " m/s" << std::endl;
    std::cout << "Expected: positive (flow from cold to hot)" << std::endl;

    EXPECT_GT(avg_u, 0.0f)
        << "Flow direction incorrect for positive dσ/dT (should be cold to hot)";

    cudaFree(d_temperature);
    cudaFree(d_fill_level);
    cudaFree(d_normals);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
