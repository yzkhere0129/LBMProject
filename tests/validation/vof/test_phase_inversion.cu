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
 * @brief Phase Inversion - Oil Block Rising in Water
 *
 * FluidLBM + TRT variable-omega (Guo forcing) + PLIC advection
 *
 * Configuration:
 * - Domain: 128×128×4 (quasi-2D), dx = 62.5 μm
 * - Water: ρ = 1000 kg/m³, μ = 0.005 Pa·s
 * - Oil:   ρ =  900 kg/m³, μ = 0.05  Pa·s  (10:1 viscosity ratio)
 * - σ = 4.5e-2 N/m, g = 9.81 m/s²
 *
 * LBM parameter design (water phase controls dt):
 * - tau_water ≈ 0.6 (TRT stabilizes near tau=0.5)
 * - tau_oil   ≈ 1.17 (well within stable range)
 * - g_LB ≈ 1e-4
 *
 * NOTE: 100:1 viscosity ratio gives tau_oil ≈ 23 — out of LBM range.
 * 10:1 is the maximum tractable ratio that keeps both tau in [0.55, 1.6].
 * Variable-omega TRT handles the per-cell tau variation.
 *
 * NOTE: EDM collision does not support variable omega (per-cell viscosity).
 * Variable-omega TRT uses Guo forcing, which is correct for two-phase flow
 * without Darcy damping. EDM is needed only when Darcy K is active.
 */
TEST_F(PhaseInversionTest, OilBlockRising) {
    std::cout << "\n=== Phase Inversion: Oil Rising in Water ===" << std::endl;
    std::cout << "  Collision: TRT variable-omega + Guo forcing" << std::endl;
    std::cout << "  Advection: PLIC geometric (Weymouth-Yue)" << std::endl;

    // ========================================================================
    // Domain
    // ========================================================================
    const int nx = 128;
    const int ny = 128;
    const int nz = 4;
    const float dx = 62.5e-6f;
    const int num_cells = nx * ny * nz;

    // ========================================================================
    // Physical Parameters — 10:1 viscosity ratio
    // ========================================================================
    const float rho_water = 1000.0f;     // kg/m³ (heavy, f=1)
    const float rho_oil = 900.0f;        // kg/m³ (light, f=0)
    const float mu_water = 0.03f;        // Pa·s
    const float mu_oil = 0.3f;           // Pa·s (10× more viscous)
    // Ma constraint: at μ=0.005 the peak Ma was 0.52 (compressible).
    // Increasing μ 6× maps the same physical flow to 6× lower lattice
    // velocity, bringing the terminal Ma to ~0.03 and transient peaks
    // to ~0.08. tau_water and tau_oil are UNCHANGED at 0.6 and 1.61.
    const float nu_water = mu_water / rho_water;  // 3e-5 m²/s
    const float nu_oil = mu_oil / rho_oil;        // 3.33e-4 m²/s
    const float sigma = 4.5e-2f;         // N/m
    const float g_phys = 9.81f;          // m/s²

    // ========================================================================
    // LBM parameters — designed from water phase (controls stability)
    // ========================================================================
    const float tau_water = 0.6f;
    const float nu_water_LB = (tau_water - 0.5f) / 3.0f;  // 0.0333
    const float dt = nu_water_LB * dx * dx / nu_water;
    const float g_LB = g_phys * dt * dt / dx;
    const float vel_conv = dx / dt;

    // Verify oil tau is in range
    const float nu_oil_LB = nu_oil * dt / (dx * dx);
    const float tau_oil = 3.0f * nu_oil_LB + 0.5f;

    // Longer simulation to compensate for higher viscosity (rise ∝ t/μ)
    const float t_total = 0.40f;  // 400 ms
    const int num_steps = static_cast<int>(t_total / dt);
    const int output_every = std::max(1, num_steps / 8);

    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  dx=" << dx*1e6f << " um, dt=" << dt*1e6f << " us" << std::endl;
    std::cout << "  mu_water=" << mu_water << ", mu_oil=" << mu_oil
              << " (ratio " << mu_oil/mu_water << ":1)" << std::endl;
    std::cout << "  tau_water=" << tau_water << ", tau_oil=" << tau_oil << std::endl;
    std::cout << "  g_LB=" << g_LB << ", steps=" << num_steps << std::endl;

    // ========================================================================
    // FluidLBM — TRT with variable omega
    // ========================================================================
    FluidLBM fluid(nx, ny, nz, nu_water, rho_water,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   dt, dx);
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // ========================================================================
    // VOF — PLIC, WALL boundaries
    // ========================================================================
    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::PERIODIC);

    const int block_x0 = 10;
    const int block_y0 = 10;
    const int block_width = nx / 3;
    const int block_height = ny / 3;

    initializeOilBlock(vof, nx, ny, nz, block_x0, block_y0, block_width, block_height);
    vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
    vof.reconstructInterface();
    vof.computeCurvature();

    std::cout << "  Oil block: (" << block_x0 << "," << block_y0 << ") "
              << block_width << "x" << block_height << " cells" << std::endl;

    // Initial state
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());
    float oil_mass_initial = computeOilMass(h_fill);
    float cx_initial, cy_initial;
    computeOilCentroid(h_fill, nx, ny, nz, cx_initial, cy_initial);

    std::cout << "  Initial oil mass=" << oil_mass_initial
              << ", centroid=(" << cx_initial << "," << cy_initial << ")" << std::endl;

    // ========================================================================
    // Force accumulator + physical velocity buffers
    // ========================================================================
    ForceAccumulator forces(nx, ny, nz);

    float *d_vx_phys = nullptr, *d_vy_phys = nullptr, *d_vz_phys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vx_phys, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy_phys, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz_phys, num_cells * sizeof(float)));

    // ========================================================================
    // Time integration
    // ========================================================================
    std::cout << "\n  Running..." << std::endl;

    for (int step = 0; step <= num_steps; ++step) {
        float t = step * dt;

        if (step % output_every == 0) {
            vof.copyFillLevelToHost(h_fill.data());
            if (hasNaNOrInf(h_fill)) {
                FAIL() << "NaN/Inf at step " << step;
            }

            float oil_mass = computeOilMass(h_fill);
            float mass_error = std::abs(oil_mass - oil_mass_initial) / oil_mass_initial;
            float cx, cy;
            computeOilCentroid(h_fill, nx, ny, nz, cx, cy);

            std::vector<float> h_vx(num_cells), h_vy(num_cells);
            cudaMemcpy(h_vx.data(), fluid.getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy.data(), fluid.getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            float v_max_LU = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float v = std::sqrt(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i]);
                v_max_LU = std::max(v_max_LU, v);
            }

            printf("  Step %5d (t=%6.2f ms): cy=%6.2f rise=%6.2f cells "
                   "mass_err=%.3f%% v_max=%.1f mm/s (Ma=%.4f)\n",
                   step, t*1e3f, cy, cy - cy_initial,
                   mass_error * 100.0f, v_max_LU * vel_conv * 1e3f,
                   v_max_LU / 0.577f);
        }

        if (step >= num_steps) break;

        // 1. Forces in physical units [N/m³]
        forces.reset();
        forces.addVOFBuoyancyForce(vof.getFillLevel(),
                                    rho_water, rho_oil,
                                    0.0f, -g_phys, 0.0f);
        forces.addSurfaceTensionForce(vof.getCurvature(), vof.getFillLevel(),
                                       sigma, nx, ny, nz, dx);
        forces.convertToLatticeUnits(dx, dt, rho_water);

        // 2. Update variable omega field from VOF (10:1 viscosity ratio)
        fluid.computeVariableViscosity(vof.getFillLevel(),
                                        rho_water, rho_oil,
                                        mu_water, mu_oil);

        // 3. TRT collision with per-cell omega + Guo forcing
        fluid.collisionTRTVariable(forces.getFx(), forces.getFy(), forces.getFz(),
                                    vof.getFillLevel(),
                                    rho_water, rho_oil);
        fluid.streaming();
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(), forces.getFz());

        // 4. Lattice velocity → physical [m/s] for VOF
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

        // 5. PLIC advection + interface reconstruction
        vof.advectFillLevel(d_vx_phys, d_vy_phys, d_vz_phys, dt);
        vof.reconstructInterface();
        vof.computeCurvature();
    }

    // ========================================================================
    // Final validation
    // ========================================================================
    vof.copyFillLevelToHost(h_fill.data());
    float oil_mass_final = computeOilMass(h_fill);
    float mass_error = std::abs(oil_mass_final - oil_mass_initial) / oil_mass_initial;

    float cx_final, cy_final;
    computeOilCentroid(h_fill, nx, ny, nz, cx_final, cy_final);
    float total_rise = cy_final - cy_initial;

    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "  Viscosity ratio: " << mu_oil/mu_water << ":1" << std::endl;
    std::cout << "  tau range: [" << tau_water << ", " << tau_oil << "]" << std::endl;
    std::cout << "  Initial centroid: (" << cx_initial << ", " << cy_initial << ")" << std::endl;
    std::cout << "  Final centroid:   (" << cx_final << ", " << cy_final << ")" << std::endl;
    std::cout << "  Total rise: " << total_rise << " cells = "
              << total_rise * dx * 1e3f << " mm" << std::endl;
    std::cout << "  Mass error: " << mass_error * 100.0f << "%" << std::endl;

    std::cout << "\n=== Validation ===" << std::endl;

    bool rises_ok = (total_rise > 15.0f);
    std::cout << "  [" << (rises_ok ? "PASS" : "FAIL")
              << "] Oil rises: dy=" << total_rise << " cells (>15)" << std::endl;

    bool mass_ok = (mass_error < 0.01f);
    std::cout << "  [" << (mass_ok ? "PASS" : "FAIL")
              << "] Mass conservation: " << mass_error * 100.0f << "% (<1%)" << std::endl;

    bool no_nan = !hasNaNOrInf(h_fill);
    std::cout << "  [" << (no_nan ? "PASS" : "FAIL") << "] No NaN/Inf" << std::endl;

    // Cleanup
    cudaFree(d_vx_phys);
    cudaFree(d_vy_phys);
    cudaFree(d_vz_phys);

    EXPECT_TRUE(rises_ok) << "Oil block did not rise >15 cells";
    EXPECT_TRUE(mass_ok) << "Mass error: " << mass_error * 100.0f << "%";
    EXPECT_TRUE(no_nan) << "NaN/Inf detected";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
