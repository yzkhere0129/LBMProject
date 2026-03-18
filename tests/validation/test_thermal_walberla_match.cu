/**
 * @file test_thermal_walberla_match.cu
 * @brief Pure thermal test to exactly match walberla FD solver
 *
 * This test isolates the thermal solver from all multiphysics effects to
 * achieve numerical consistency with walberla's heat equation solver.
 *
 * WALBERLA CONFIGURATION (LaserHeating.cpp):
 * - Explicit FD heat equation: dT/dt = α∇²T + Q/(ρ·cp)
 * - 3D Laplacian (6-point stencil)
 * - Gaussian laser with Beer-Lambert absorption
 * - Domain: 400×400×200 μm (200×200×100 cells)
 * - dx = 2 μm, dt = 100 ns
 * - Ti6Al4V: ρ=4430, cp=526, k=6.7, α=2.874e-6 m²/s
 * - Laser: P=200W, r0=50μm, η=0.35, δ=50μm (δ = r0 in walberla)
 * - Dirichlet BC: T=300K on all boundaries
 *
 * Goal: Match walberla peak temperature within 5%
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

// Direct ThermalLBM test without MultiphysicsSolver
#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            FAIL() << "CUDA error: " << cudaGetErrorString(err); \
        } \
    } while(0)

// Helper to compute max temperature from ThermalLBM
float getMaxTemp(ThermalLBM& thermal) {
    int num_cells = thermal.getNx() * thermal.getNy() * thermal.getNz();
    std::vector<float> h_temp(num_cells);
    thermal.copyTemperatureToHost(h_temp.data());
    return *std::max_element(h_temp.begin(), h_temp.end());
}

// Kernel to compute laser heat source (exact walberla formula)
__global__ void computeWalberlaHeatSource(
    float* heat_source,
    float power, float absorptivity, float spot_radius, float absorption_depth,
    float laser_x, float laser_y,
    float dx, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Physical position [m]
    float x = i * dx;
    float y = j * dx;

    // walberla uses z from top surface (z=0 at top)
    // Our domain has z=0 at bottom, so surface is at z = (nz-1)*dx
    // Depth from surface (positive downward into material)
    float surface_z = (nz - 1) * dx;
    float depth = surface_z - k * dx;  // positive when below surface

    // Only heat below surface (depth >= 0)
    if (depth < 0.0f) {
        heat_source[idx] = 0.0f;
        return;
    }

    // Distance from laser center
    float dx_laser = x - laser_x;
    float dy_laser = y - laser_y;
    float r2 = dx_laser * dx_laser + dy_laser * dy_laser;
    float r0_2 = spot_radius * spot_radius;

    // EXACT walberla formula (LaserHeating.cpp lines 87-92):
    // Q = (2*P*η) / (π*r0²) * exp(-2*r²/r0²) * exp(-|z|/δ) / δ
    float I = (2.0f * power * absorptivity) / (M_PI * r0_2) * expf(-2.0f * r2 / r0_2);
    float Q = I * expf(-depth / absorption_depth) / absorption_depth;

    heat_source[idx] = Q;
}

/**
 * @brief Test: Pure thermal solver matching walberla exactly
 *
 * Disables all multiphysics to isolate thermal behavior
 */
TEST(ThermalWalberlaMatch, PureThermalTest) {
    std::cout << "\n=== Pure Thermal Test: walberla FD Match ===\n";

    // ========================================================================
    // Domain setup (EXACT walberla match)
    // ========================================================================
    const int nx = 200;
    const int ny = 200;
    const int nz = 100;
    const float dx = 2.0e-6f;  // 2 μm
    const int num_cells = nx * ny * nz;

    std::cout << "Grid: " << nx << " × " << ny << " × " << nz << "\n";
    std::cout << "dx = " << dx * 1e6 << " μm\n";

    // ========================================================================
    // Material properties (Ti6Al4V - EXACT walberla match)
    // ========================================================================
    const float rho = 4430.0f;    // kg/m³
    const float cp = 526.0f;      // J/(kg·K)
    const float k = 6.7f;         // W/(m·K)
    const float alpha = k / (rho * cp);  // 2.874e-6 m²/s

    std::cout << "Material: Ti6Al4V\n";
    std::cout << "  α = " << alpha << " m²/s\n";

    // ========================================================================
    // Time parameters (EXACT walberla match)
    // ========================================================================
    const float dt = 100e-9f;  // 100 ns
    const float laser_shutoff_time = 50e-6f;  // 50 μs
    const float total_time = 100e-6f;  // 100 μs
    const int total_steps = static_cast<int>(total_time / dt);
    const int output_interval = 100;  // Every 10 μs

    // LBM omega
    float omega = 1.0f / (0.5f + 3.0f * alpha * dt / (dx * dx));
    std::cout << "dt = " << dt * 1e9 << " ns\n";
    std::cout << "LBM omega = " << omega << " (stable if < 2.0)\n";
    ASSERT_LT(omega, 2.0f) << "omega unstable!";

    // ========================================================================
    // Laser parameters (EXACT walberla match)
    // ========================================================================
    const float laser_power = 200.0f;       // W
    const float laser_spot_radius = 50e-6f; // 50 μm
    const float laser_absorptivity = 0.35f;
    const float laser_absorption_depth = 50e-6f;  // walberla uses δ = r0
    const float laser_x = nx * dx / 2.0f;   // Center
    const float laser_y = ny * dx / 2.0f;   // Center

    std::cout << "Laser: P=" << laser_power << "W, r0=" << laser_spot_radius*1e6
              << "μm, η=" << laser_absorptivity << ", δ=" << laser_absorption_depth*1e6 << "μm\n";

    // ========================================================================
    // Initialize ThermalLBM solver
    // ========================================================================
    // Constructor: ThermalLBM(nx, ny, nz, thermal_diffusivity, density, specific_heat, dt, dx)
    ThermalLBM thermal(nx, ny, nz, alpha, rho, cp, dt, dx);
    thermal.initialize(300.0f);  // Initial temperature

    // Allocate heat source array
    float* d_heat_source;
    CUDA_CHECK(cudaMalloc(&d_heat_source, num_cells * sizeof(float)));

    std::cout << "\nInitial temperature: 300 K\n";

    // ========================================================================
    // Boundary condition
    // ========================================================================
    const float T_boundary = 300.0f;  // Dirichlet BC

    // ========================================================================
    // Time integration
    // ========================================================================
    std::cout << "\n=== Time Integration ===\n";
    std::cout << "Step      Time[μs]    Max_T[K]    Status\n";
    std::cout << std::string(60, '-') << "\n";

    dim3 threads(8, 8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    float peak_temperature = 300.0f;
    int peak_step = 0;

    for (int step = 0; step <= total_steps; ++step) {
        float current_time = step * dt;
        bool laser_on = (current_time <= laser_shutoff_time);

        // 1. Apply Dirichlet BC (constant temperature on all faces)
        thermal.applyBoundaryConditions(1, T_boundary);

        // 2. Compute temperature from distributions
        thermal.computeTemperature();

        // 3. LBM collision (pure diffusion, no advection)
        thermal.collisionBGK(nullptr, nullptr, nullptr);

        // 4. LBM streaming
        thermal.streaming();

        // 5. Add heat source (only when laser is on)
        if (laser_on) {
            // Compute heat source
            computeWalberlaHeatSource<<<blocks, threads>>>(
                d_heat_source,
                laser_power, laser_absorptivity, laser_spot_radius, laser_absorption_depth,
                laser_x, laser_y,
                dx, nx, ny, nz
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            // Add to thermal solver
            thermal.addHeatSource(d_heat_source, dt);
        }

        // Diagnostics
        if (step % output_interval == 0 || step == total_steps) {
            float max_T = getMaxTemp(thermal);

            if (max_T > peak_temperature) {
                peak_temperature = max_T;
                peak_step = step;
            }

            std::string status = laser_on ? "LASER ON" : "cooling";
            std::cout << std::setw(6) << step << "    "
                      << std::setw(8) << std::fixed << std::setprecision(1) << current_time * 1e6 << "    "
                      << std::setw(10) << std::fixed << std::setprecision(1) << max_T << "    "
                      << status << "\n";
        }
    }

    // ========================================================================
    // Results
    // ========================================================================
    float final_temp = getMaxTemp(thermal);

    std::cout << "\n=== Results ===\n";
    std::cout << "Peak temperature: " << peak_temperature << " K (step " << peak_step << ")\n";
    std::cout << "Final temperature: " << final_temp << " K\n";
    std::cout << "Peak time: " << peak_step * dt * 1e6 << " μs\n";

    // ========================================================================
    // walberla Reference (verified Dec 2025)
    // ========================================================================
    // walberla peak: 4,099.37 K at t=50 μs with EXACT same parameters:
    //   Grid: 200×200×100, dx=2μm, dt=100ns
    //   Ti6Al4V: ρ=4430, cp=526, k=6.7
    //   Laser: P=200W, r₀=50μm, η=0.35, δ=50μm (δ=r₀ in walberla)
    //
    // Note: Earlier reports of 17,500 K used DIFFERENT parameters:
    //   r₀=30μm, δ=30μm, smaller 40×80×40 grid
    //   Smaller spot/absorption depth → higher energy density → higher T
    const float walberla_peak = 4099.37f;  // K

    // Validation criteria
    const float T_melt = 1923.0f;  // Ti6Al4V liquidus

    std::cout << "\n=== Validation vs walberla ===\n";

    // 1. Compare with walberla reference (target: <5% error)
    float error_vs_walberla = std::abs(peak_temperature - walberla_peak) / walberla_peak * 100.0f;
    std::cout << "walberla reference: " << walberla_peak << " K\n";
    std::cout << "LBMProject peak:    " << peak_temperature << " K\n";
    std::cout << "Error vs walberla:  " << std::fixed << std::setprecision(2)
              << error_vs_walberla << "% "
              << (error_vs_walberla < 5.0f ? "PASS (<5%)" : "FAIL (>=5%)") << "\n";
    EXPECT_NEAR(peak_temperature, walberla_peak, walberla_peak * 0.05f)
        << "Peak temperature should match walberla within 5%";

    // 2. Must achieve melting
    bool melting_achieved = peak_temperature > T_melt;
    std::cout << "Melting achieved (T > " << T_melt << " K): "
              << (melting_achieved ? "PASS" : "FAIL") << "\n";
    EXPECT_GT(peak_temperature, T_melt) << "Peak temperature should exceed melting point";

    // 3. Peak should occur at or near laser shutoff
    float peak_time_us = peak_step * dt * 1e6;
    bool peak_timing_ok = std::abs(peak_time_us - 50.0f) < 5.0f;  // Within 5 μs of shutoff
    std::cout << "Peak timing (near 50 μs): "
              << (peak_timing_ok ? "PASS" : "FAIL")
              << " (actual: " << peak_time_us << " μs)\n";
    EXPECT_NEAR(peak_time_us, 50.0f, 5.0f) << "Peak should be near laser shutoff";

    // 4. Cooling after shutoff
    bool cooling_observed = final_temp < peak_temperature;
    std::cout << "Cooling observed: " << (cooling_observed ? "PASS" : "FAIL") << "\n";
    EXPECT_LT(final_temp, peak_temperature) << "Temperature should decrease after laser off";

    // Cleanup
    CUDA_CHECK(cudaFree(d_heat_source));

    std::cout << "\n=== Test Complete ===\n";
}

/**
 * @brief Test: Compare LBM with explicit FD reference solver
 *
 * Implements explicit FD heat equation exactly like walberla to verify LBM accuracy
 */
TEST(ThermalWalberlaMatch, FDReferenceComparison) {
    std::cout << "\n=== FD Reference Comparison ===\n";

    // Smaller grid for faster testing
    const int nx = 100;
    const int ny = 100;
    const int nz = 50;
    const float dx = 4.0e-6f;  // 4 μm
    const int num_cells = nx * ny * nz;

    // Material
    const float rho = 4430.0f;
    const float cp = 526.0f;
    const float k = 6.7f;
    const float alpha = k / (rho * cp);

    // Time
    const float dt = 200e-9f;  // 200 ns for stability
    const float laser_shutoff_time = 50e-6f;
    const float total_time = 60e-6f;  // Shorter run
    const int total_steps = static_cast<int>(total_time / dt);

    // Check FD stability (von Neumann): α·dt/dx² < 1/6 (3D)
    float fd_stability = alpha * dt / (dx * dx);
    std::cout << "FD stability number: " << fd_stability << " (must be < 0.167)\n";
    ASSERT_LT(fd_stability, 0.167f) << "FD scheme unstable!";

    // Laser
    const float laser_power = 200.0f;
    const float laser_spot_radius = 50e-6f;
    const float laser_absorptivity = 0.35f;
    const float laser_absorption_depth = 50e-6f;
    const float laser_x = nx * dx / 2.0f;
    const float laser_y = ny * dx / 2.0f;

    // Allocate CPU arrays for FD solver
    std::vector<float> T_fd(num_cells, 300.0f);
    std::vector<float> T_fd_new(num_cells, 300.0f);
    std::vector<float> Q(num_cells, 0.0f);

    // Initialize LBM solver
    ThermalLBM thermal_lbm(nx, ny, nz, alpha, rho, cp, dt, dx);
    thermal_lbm.initialize(300.0f);

    float* d_heat_source;
    CUDA_CHECK(cudaMalloc(&d_heat_source, num_cells * sizeof(float)));

    dim3 threads(8, 8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    std::cout << "\nStep   Time[μs]   FD_max[K]   LBM_max[K]   Error[%]\n";
    std::cout << std::string(60, '-') << "\n";

    float fd_factor = alpha * dt / (dx * dx);
    float source_factor = dt / (rho * cp);

    for (int step = 0; step <= total_steps; ++step) {
        float current_time = step * dt;
        bool laser_on = (current_time <= laser_shutoff_time);

        // Compute heat source (CPU)
        if (laser_on) {
            for (int kk = 0; kk < nz; ++kk) {
                for (int jj = 0; jj < ny; ++jj) {
                    for (int ii = 0; ii < nx; ++ii) {
                        int idx = ii + nx * (jj + ny * kk);

                        float x = ii * dx;
                        float y = jj * dx;
                        float surface_z = (nz - 1) * dx;
                        float depth = surface_z - kk * dx;

                        if (depth < 0.0f) {
                            Q[idx] = 0.0f;
                            continue;
                        }

                        float dx_laser = x - laser_x;
                        float dy_laser = y - laser_y;
                        float r2 = dx_laser * dx_laser + dy_laser * dy_laser;
                        float r0_2 = laser_spot_radius * laser_spot_radius;

                        float I = (2.0f * laser_power * laser_absorptivity) /
                                  (M_PI * r0_2) * expf(-2.0f * r2 / r0_2);
                        Q[idx] = I * expf(-depth / laser_absorption_depth) / laser_absorption_depth;
                    }
                }
            }
        } else {
            std::fill(Q.begin(), Q.end(), 0.0f);
        }

        // FD update (explicit 3D Laplacian)
        for (int kk = 1; kk < nz - 1; ++kk) {
            for (int jj = 1; jj < ny - 1; ++jj) {
                for (int ii = 1; ii < nx - 1; ++ii) {
                    int idx = ii + nx * (jj + ny * kk);

                    // 3D Laplacian
                    float laplacian = T_fd[idx - 1] + T_fd[idx + 1]           // x neighbors
                                    + T_fd[idx - nx] + T_fd[idx + nx]         // y neighbors
                                    + T_fd[idx - nx*ny] + T_fd[idx + nx*ny]   // z neighbors
                                    - 6.0f * T_fd[idx];

                    T_fd_new[idx] = T_fd[idx] + fd_factor * laplacian + source_factor * Q[idx];
                }
            }
        }

        // Dirichlet BC (boundaries stay at 300K)
        for (int kk = 0; kk < nz; ++kk) {
            for (int jj = 0; jj < ny; ++jj) {
                for (int ii = 0; ii < nx; ++ii) {
                    if (ii == 0 || ii == nx-1 || jj == 0 || jj == ny-1 || kk == 0 || kk == nz-1) {
                        int idx = ii + nx * (jj + ny * kk);
                        T_fd_new[idx] = 300.0f;
                    }
                }
            }
        }

        std::swap(T_fd, T_fd_new);

        // LBM update
        thermal_lbm.applyBoundaryConditions(1, 300.0f);
        thermal_lbm.computeTemperature();
        thermal_lbm.collisionBGK(nullptr, nullptr, nullptr);
        thermal_lbm.streaming();

        if (laser_on) {
            computeWalberlaHeatSource<<<blocks, threads>>>(
                d_heat_source,
                laser_power, laser_absorptivity, laser_spot_radius, laser_absorption_depth,
                laser_x, laser_y,
                dx, nx, ny, nz
            );
            cudaDeviceSynchronize();
            thermal_lbm.addHeatSource(d_heat_source, dt);
        }

        // Output
        if (step % 50 == 0 || step == total_steps) {
            float fd_max = *std::max_element(T_fd.begin(), T_fd.end());
            float lbm_max = getMaxTemp(thermal_lbm);
            float error = 100.0f * std::abs(fd_max - lbm_max) / fd_max;

            std::cout << std::setw(5) << step << "   "
                      << std::setw(8) << std::fixed << std::setprecision(1) << current_time * 1e6 << "   "
                      << std::setw(10) << fd_max << "   "
                      << std::setw(10) << lbm_max << "   "
                      << std::setw(8) << std::setprecision(2) << error << "\n";
        }
    }

    // Final comparison
    float fd_max = *std::max_element(T_fd.begin(), T_fd.end());
    float lbm_max = getMaxTemp(thermal_lbm);
    float final_error = 100.0f * std::abs(fd_max - lbm_max) / fd_max;

    std::cout << "\nFinal error: " << final_error << "%\n";

    // Target: <10% error between LBM and FD
    EXPECT_LT(final_error, 10.0f) << "LBM should match FD within 10%";

    CUDA_CHECK(cudaFree(d_heat_source));

    std::cout << "\n=== FD Reference Test Complete ===\n";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
