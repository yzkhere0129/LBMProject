/**
 * @file test_phase_inversion.cu
 * @brief Phase Inversion validation test - Oil block rising in water
 *
 * Physics: A block of lighter oil rises through heavier water due to buoyancy
 * - Initial condition: Rectangular oil block at bottom-left corner
 * - Oil (lighter, ρ=900 kg/m³) rises through water (heavier, ρ=1000 kg/m³)
 * - Tests VOF interface tracking under buoyancy and surface tension
 *
 * This test validates:
 * - VOF interface tracking with density-ratio ~1.1
 * - Buoyancy-driven interface motion
 * - Surface tension effects at liquid-liquid interface
 * - Mass conservation during phase rearrangement
 *
 * Solver: FluidLBM with TRT collision + EDM forcing (zero Darcy)
 * - LBM collision-streaming enforces near-divergence-free velocity
 * - EDM forcing avoids distribution anisotropy accumulation
 * - TRT suppresses checkerboard instability at low tau
 *
 * Reference:
 * - Classic VOF benchmark (Hirt & Nichols, JCP 1981)
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/fluid_lbm.h"
#include "physics/force_accumulator.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace lbm::physics;

class PhaseInversionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Initialize rectangular oil block in bottom-left corner
     *
     * Oil block: f = 0 (gas-like in VOF convention)
     * Water background: f = 1 (liquid-like)
     */
    void initializeOilBlock(VOFSolver& vof, int nx, int ny, int nz,
                           int block_x0, int block_y0,
                           int block_width, int block_height) {
        std::vector<float> h_fill(nx * ny * nz);

        const float interface_width = 2.0f;  // Smooth interface

        for (int kz = 0; kz < nz; ++kz) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * kz);

                    float dx_left = static_cast<float>(i - block_x0);
                    float dx_right = static_cast<float>(block_x0 + block_width - i);
                    float dy_bottom = static_cast<float>(j - block_y0);
                    float dy_top = static_cast<float>(block_y0 + block_height - j);

                    float f_left = 0.5f * (1.0f + std::tanh(dx_left / interface_width));
                    float f_right = 0.5f * (1.0f + std::tanh(dx_right / interface_width));
                    float f_bottom = 0.5f * (1.0f + std::tanh(dy_bottom / interface_width));
                    float f_top = 0.5f * (1.0f + std::tanh(dy_top / interface_width));

                    float inside_block = f_left * f_right * f_bottom * f_top;

                    // Oil (lighter) is f=0, Water (heavier) is f=1
                    h_fill[idx] = 1.0f - inside_block;
                }
            }
        }

        vof.initialize(h_fill.data());
    }

    void computeOilCentroid(const std::vector<float>& fill,
                           int nx, int ny, int nz,
                           float& cx, float& cy) {
        float sum_x = 0.0f, sum_y = 0.0f;
        float total_oil = 0.0f;

        for (int kz = 0; kz < nz; ++kz) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * kz);
                    float f = fill[idx];
                    float oil_frac = 1.0f - f;
                    if (oil_frac > 0.01f) {
                        sum_x += oil_frac * static_cast<float>(i);
                        sum_y += oil_frac * static_cast<float>(j);
                        total_oil += oil_frac;
                    }
                }
            }
        }

        if (total_oil > 0.0f) {
            cx = sum_x / total_oil;
            cy = sum_y / total_oil;
        } else {
            cx = 0.0f;
            cy = 0.0f;
        }
    }

    float computeOilMass(const std::vector<float>& fill) {
        float mass = 0.0f;
        for (float f : fill) {
            mass += (1.0f - f);
        }
        return mass;
    }

    bool hasNaNOrInf(const std::vector<float>& data) {
        for (float val : data) {
            if (std::isnan(val) || std::isinf(val)) return true;
        }
        return false;
    }
};

/**
 * @brief Phase Inversion - Oil Block Rising in Water (FluidLBM + TRT + EDM)
 *
 * Configuration:
 * - Domain: 128×128×4 cells (8mm × 8mm, quasi-2D)
 * - Grid spacing: dx = 62.5 μm
 * - Water: ρ_w = 1000 kg/m³
 * - Oil: ρ_o = 900 kg/m³
 * - Dynamic viscosity: μ = 0.01 Pa·s (equal for both phases)
 *   NOTE: Original 100:1 viscosity ratio (μ_oil/μ_water) gives tau_oil ≈ 23,
 *   far outside LBM stable range. Equal μ is standard for density-driven VOF
 *   benchmarks in LBM.
 * - Surface tension: σ = 4.5×10⁻² N/m
 * - Gravity: g = 9.81 m/s² (downward, -y)
 *
 * LBM parameters (designed in lattice units first):
 * - tau = 0.7, ν_LB = 0.0667
 * - dt derived from tau and ν = μ/ρ_water
 * - g_LB ≈ 1e-4 (safe for LBM stability)
 */
TEST_F(PhaseInversionTest, OilBlockRising) {
    std::cout << "\n=== Phase Inversion: Oil Rising in Water (FluidLBM + TRT + EDM) ===" << std::endl;

    // ========================================================================
    // Domain Configuration
    // ========================================================================
    const int nx = 128;
    const int ny = 128;
    const int nz = 4;  // Quasi-2D
    const float dx = 62.5e-6f;  // 62.5 μm
    const int num_cells = nx * ny * nz;

    // ========================================================================
    // Physical Parameters
    // ========================================================================
    const float rho_water = 1000.0f;     // kg/m³ (heavy phase, f=1)
    const float rho_oil = 900.0f;        // kg/m³ (light phase, f=0)
    const float mu = 0.05f;              // Pa·s (equal for both phases)
    // NOTE: Original had μ_water=0.001, μ_oil=0.1 (100:1 ratio).
    // A 100:1 viscosity ratio gives tau_oil ≈ 23 — far outside LBM stable
    // range. Equal μ = 0.05 Pa·s is the standard approach for density-driven
    // VOF benchmarks in LBM. This keeps tau uniform and Ma < 0.01.
    const float nu = mu / rho_water;     // 5e-5 m²/s kinematic viscosity
    const float sigma = 4.5e-2f;         // N/m (water-oil interface)
    const float g_phys = 9.81f;          // m/s²

    // ========================================================================
    // LBM Parameter Design (lattice units first)
    // ========================================================================
    const float tau = 0.7f;
    const float nu_LB = (tau - 0.5f) / 3.0f;  // 0.0667
    const float dt = nu_LB * dx * dx / nu;     // ~26 μs
    const float g_LB = g_phys * dt * dt / dx;  // ~1e-4
    const float vel_conv = dx / dt;             // lattice→physical velocity conversion

    // Total simulation time: 50 ms
    const float t_total = 0.05f;
    const int num_steps = static_cast<int>(t_total / dt);
    const int output_every = std::max(1, num_steps / 6);

    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << " cells" << std::endl;
    std::cout << "  dx = " << dx * 1e6f << " um, dt = " << dt * 1e6f << " us" << std::endl;
    std::cout << "  tau = " << tau << ", nu_LB = " << nu_LB << std::endl;
    std::cout << "  g_LB = " << g_LB << " (safe < 0.001)" << std::endl;
    std::cout << "  Steps: " << num_steps << " (" << t_total * 1e3f << " ms)" << std::endl;
    std::cout << "  rho_water = " << rho_water << ", rho_oil = " << rho_oil
              << ", At = " << (rho_water - rho_oil) / (rho_water + rho_oil) << std::endl;

    // ========================================================================
    // Initialize FluidLBM (TRT + EDM, no-slip walls on x/y, periodic z)
    // ========================================================================
    FluidLBM fluid(nx, ny, nz, nu, rho_water,
                   BoundaryType::WALL,     // x: no-slip bounce-back
                   BoundaryType::WALL,     // y: no-slip bounce-back
                   BoundaryType::PERIODIC, // z: quasi-2D
                   dt, dx);
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);  // Lattice density = 1, zero velocity
    fluid.setTRT(3.0f / 16.0f);  // Λ = 3/16 for optimal wall accuracy

    std::cout << "  FluidLBM: TRT (Lambda=3/16) + EDM forcing" << std::endl;
    std::cout << "  Boundaries: WALL (x,y), PERIODIC (z)" << std::endl;

    // ========================================================================
    // Initialize VOF Solver (WALL on x/y, PERIODIC z)
    // ========================================================================
    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::WALL,     // x
                  VOFSolver::BoundaryType::WALL,     // y
                  VOFSolver::BoundaryType::PERIODIC);// z

    const int block_x0 = 10;
    const int block_y0 = 10;
    const int block_width = nx / 3;   // ~42 cells
    const int block_height = ny / 3;  // ~42 cells

    initializeOilBlock(vof, nx, ny, nz, block_x0, block_y0, block_width, block_height);
    // PLIC geometric advection: Strang-split directional sweeps with
    // Youngs normal + Scardovelli-Zaleski alpha inversion.
    // Exactly mass-conservative by construction (no numerical diffusion).
    vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
    vof.reconstructInterface();
    vof.computeCurvature();

    std::cout << "  VOF: PLIC geometric advection, WALL boundaries" << std::endl;
    std::cout << "  Oil block: (" << block_x0 << "," << block_y0 << ") "
              << block_width << "x" << block_height << " cells" << std::endl;

    // Copy initial state
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());
    float oil_mass_initial = computeOilMass(h_fill);
    float cx_initial, cy_initial;
    computeOilCentroid(h_fill, nx, ny, nz, cx_initial, cy_initial);

    std::cout << "  Initial oil mass = " << oil_mass_initial
              << ", centroid = (" << cx_initial << ", " << cy_initial << ")" << std::endl;

    // ========================================================================
    // Force Accumulator + zero Darcy buffer (for EDM API)
    // ========================================================================
    ForceAccumulator forces(nx, ny, nz);

    float* d_zero_darcy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_zero_darcy, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_zero_darcy, 0, num_cells * sizeof(float)));

    // Physical velocity buffers for VOF advection
    float *d_vx_phys = nullptr, *d_vy_phys = nullptr, *d_vz_phys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vx_phys, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy_phys, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz_phys, num_cells * sizeof(float)));

    // ========================================================================
    // Time Integration Loop
    // ========================================================================
    std::cout << "\n  Running simulation..." << std::endl;

    for (int step = 0; step <= num_steps; ++step) {
        float t = step * dt;

        // Diagnostics
        if (step % output_every == 0) {
            vof.copyFillLevelToHost(h_fill.data());

            if (hasNaNOrInf(h_fill)) {
                FAIL() << "NaN/Inf in fill level at step " << step;
            }

            float oil_mass = computeOilMass(h_fill);
            float mass_error = std::abs(oil_mass - oil_mass_initial) / oil_mass_initial;
            float cx, cy;
            computeOilCentroid(h_fill, nx, ny, nz, cx, cy);

            // Get max lattice velocity for stability check
            std::vector<float> h_vx(num_cells), h_vy(num_cells);
            cudaMemcpy(h_vx.data(), fluid.getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy.data(), fluid.getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            float v_max_LU = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float v = std::sqrt(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i]);
                v_max_LU = std::max(v_max_LU, v);
            }

            std::cout << "  Step " << step << " (t=" << t*1e3f << " ms):"
                      << " cy=" << cy
                      << ", rise=" << (cy - cy_initial) << " cells"
                      << ", mass_err=" << mass_error * 100.0f << "%"
                      << ", v_max=" << v_max_LU * vel_conv * 1e3f << " mm/s"
                      << std::endl;
        }

        if (step >= num_steps) break;

        // ==================================================================
        // 1. Accumulate forces in physical units [N/m³]
        // ==================================================================
        forces.reset();

        // Buoyancy: F = (f - 0.5) × Δρ × g [N/m³]
        forces.addVOFBuoyancyForce(
            vof.getFillLevel(),
            rho_water,   // heavy phase (f=1)
            rho_oil,     // light phase (f=0)
            0.0f, -g_phys, 0.0f  // gravity in -y
        );

        // Surface tension: F = σ × κ × ∇f [N/m³]
        forces.addSurfaceTensionForce(
            vof.getCurvature(),
            vof.getFillLevel(),
            sigma,
            nx, ny, nz, dx
        );

        // Convert [N/m³] → lattice units
        forces.convertToLatticeUnits(dx, dt, rho_water);

        // ==================================================================
        // 2. LBM collision (TRT + EDM) + streaming + macroscopic
        // ==================================================================
        fluid.collisionBGKwithEDM(forces.getFx(), forces.getFy(), forces.getFz(),
                                   d_zero_darcy);
        fluid.streaming();
        fluid.computeMacroscopicEDM(forces.getFx(), forces.getFy(), forces.getFz(),
                                     d_zero_darcy);

        // ==================================================================
        // 3. Convert lattice velocity → physical [m/s] for VOF advection
        // ==================================================================
        std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
        cudaMemcpy(h_vx.data(), fluid.getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vy.data(), fluid.getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vz.data(), fluid.getVelocityZ(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_cells; ++i) {
            h_vx[i] *= vel_conv;
            h_vy[i] *= vel_conv;
            h_vz[i] *= vel_conv;
        }

        cudaMemcpy(d_vx_phys, h_vx.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy_phys, h_vy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz_phys, h_vz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

        // ==================================================================
        // 4. VOF advection + interface reconstruction
        // ==================================================================
        vof.advectFillLevel(d_vx_phys, d_vy_phys, d_vz_phys, dt);
        vof.reconstructInterface();
        vof.computeCurvature();
    }

    // ========================================================================
    // Final Validation
    // ========================================================================
    vof.copyFillLevelToHost(h_fill.data());
    float oil_mass_final = computeOilMass(h_fill);
    float mass_error = std::abs(oil_mass_final - oil_mass_initial) / oil_mass_initial;

    float cx_final, cy_final;
    computeOilCentroid(h_fill, nx, ny, nz, cx_final, cy_final);
    float total_rise = cy_final - cy_initial;

    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "  Initial centroid: (" << cx_initial << ", " << cy_initial << ")" << std::endl;
    std::cout << "  Final centroid:   (" << cx_final << ", " << cy_final << ")" << std::endl;
    std::cout << "  Total rise: " << total_rise << " cells = "
              << total_rise * dx * 1e3f << " mm" << std::endl;
    std::cout << "  Oil mass initial: " << oil_mass_initial << std::endl;
    std::cout << "  Oil mass final:   " << oil_mass_final << std::endl;
    std::cout << "  Mass error: " << mass_error * 100.0f << "%" << std::endl;

    // ========================================================================
    // Validation Criteria
    // ========================================================================
    std::cout << "\n=== Validation Criteria ===" << std::endl;

    // 1. Oil rises (positive y-displacement)
    bool rises_ok = (total_rise > 0.5f);
    std::cout << "  [" << (rises_ok ? "PASS" : "FAIL")
              << "] Oil rises: dy = " << total_rise << " cells (>0.5)" << std::endl;

    // 2. Mass conservation (LBM enforces div-free → should be < 1%)
    bool mass_ok = (mass_error < 0.01f);
    std::cout << "  [" << (mass_ok ? "PASS" : "FAIL")
              << "] Mass conservation: " << mass_error * 100.0f << "% < 1%" << std::endl;

    // 3. No NaN/Inf
    bool no_nan = !hasNaNOrInf(h_fill);
    std::cout << "  [" << (no_nan ? "PASS" : "FAIL") << "] No NaN/Inf" << std::endl;

    // 4. Centroid moves (not frozen)
    float cx_movement = std::abs(cx_final - cx_initial);
    float total_movement = std::sqrt(cx_movement * cx_movement + total_rise * total_rise);
    bool active_ok = (total_movement > 0.5f);
    std::cout << "  [" << (active_ok ? "PASS" : "FAIL")
              << "] Oil block active: movement = " << total_movement << " cells (>0.5)" << std::endl;

    std::cout << "\n=== Test Complete ===" << std::endl;

    // Cleanup
    cudaFree(d_zero_darcy);
    cudaFree(d_vx_phys);
    cudaFree(d_vy_phys);
    cudaFree(d_vz_phys);

    // Assertions
    EXPECT_TRUE(rises_ok) << "Oil block did not rise";
    EXPECT_TRUE(mass_ok) << "Mass conservation violated: " << mass_error * 100.0f << "%";
    EXPECT_TRUE(no_nan) << "NaN/Inf detected";
    EXPECT_TRUE(active_ok) << "Oil block did not move";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
