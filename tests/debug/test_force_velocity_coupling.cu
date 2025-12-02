/**
 * @file test_force_velocity_coupling.cu
 * @brief Diagnostic test to debug force-velocity coupling issue
 *
 * This test examines the complete force → velocity pipeline to identify
 * where forces are being lost or incorrectly applied.
 *
 * Test Strategy:
 * 1. Apply known Marangoni force
 * 2. Trace force through unit conversions
 * 3. Verify force reaches FluidLBM collision kernel
 * 4. Check velocity response
 * 5. Identify where coupling breaks
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "physics/multiphysics_solver.h"
#include "physics/vof_solver.h"
#include "physics/marangoni.h"
#include "physics/fluid_lbm.h"

using namespace lbm::physics;

/**
 * @brief CUDA kernel to directly print force values at specific cells
 */
__global__ void printForceKernel(
    const float* fx, const float* fy, const float* fz,
    const float* fill, const float* ux,
    int nx, int ny, int nz, int print_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= print_count) return;

    // Sample interface cells
    int k = nz / 2;  // Mid-height (interface)
    int j = ny / 2;  // Center y
    int i = nx / 2 + tid - print_count / 2;  // Cells around center x

    if (i < 0 || i >= nx) return;

    int idx = i + nx * (j + ny * k);

    printf("[GPU] Cell (%d,%d,%d): fill=%.3f, fx=%.3e, fy=%.3e, fz=%.3e, ux=%.3e\n",
           i, j, k, fill[idx], fx[idx], fy[idx], fz[idx], ux[idx]);
}

/**
 * @brief Test 1: Simple Marangoni Force Application
 */
TEST(ForceVelocityCouplingDebug, SimpleForceApplication) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG TEST: Force-Velocity Coupling" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Small domain for easier debugging
    MultiphysicsConfig config;
    config.nx = 40;
    config.ny = 40;
    config.nz = 20;
    config.dx = 2e-6f;  // 2 μm
    config.dt = 1e-7f;  // 0.1 μs

    // Enable only necessary physics
    config.enable_thermal = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;  // No advection, isolate force-velocity
    config.enable_surface_tension = false;
    config.enable_marangoni = true;
    config.enable_darcy = true;  // Keep enabled to test if it's interfering
    config.enable_laser = false;

    // Material properties
    config.material = MaterialDatabase::getTi6Al4V();
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;  // Lattice units
    config.density = 4110.0f;
    config.dsigma_dT = -0.26e-3f;
    config.darcy_coefficient = 1e7f;  // Default value

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Grid: " << config.nx << "×" << config.ny << "×" << config.nz << std::endl;
    std::cout << "  dx = " << config.dx * 1e6 << " μm" << std::endl;
    std::cout << "  dt = " << config.dt * 1e9 << " ns" << std::endl;
    std::cout << "  Darcy C = " << config.darcy_coefficient << std::endl;
    std::cout << "  dσ/dT = " << config.dsigma_dT * 1e3 << " mN/(m·K)" << std::endl;
    std::cout << std::endl;

    MultiphysicsSolver solver(config);

    // Initialize with planar interface at mid-height
    solver.initialize(2300.0f, 0.5f);

    // ===== STEP 1: Create strong temperature gradient =====
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);

    // Radial temperature gradient (hot center, cold edge)
    float T_hot = 2500.0f;
    float T_cold = 2000.0f;
    float R_hot = 10 * config.dx;  // 20 μm hot zone

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                float x = (i - config.nx / 2.0f) * config.dx;
                float y = (j - config.ny / 2.0f) * config.dx;
                float r = sqrtf(x * x + y * y);

                h_temp[idx] = (r < R_hot) ? T_hot : T_cold;
            }
        }
    }

    float* d_temp;
    cudaMalloc(&d_temp, num_cells * sizeof(float));
    cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver.setStaticTemperature(d_temp);

    std::cout << "Temperature field:" << std::endl;
    std::cout << "  T_hot = " << T_hot << " K (center)" << std::endl;
    std::cout << "  T_cold = " << T_cold << " K (edge)" << std::endl;
    std::cout << "  ΔT = " << (T_hot - T_cold) << " K" << std::endl;
    std::cout << "  Gradient ~ " << (T_hot - T_cold) / R_hot * 1e-6 << " K/μm" << std::endl;
    std::cout << std::endl;

    // ===== STEP 2: Verify liquid fraction is 1.0 everywhere =====
    std::cout << "Checking liquid fraction field..." << std::endl;
    // Note: MultiphysicsSolver initializes d_liquid_fraction_static_ to 1.0 by default
    // We don't need to set it explicitly
    std::cout << "  Liquid fraction = 1.0 (fully liquid) everywhere [default]" << std::endl;
    std::cout << "  Expected Darcy damping = 0 (since fl=1.0)" << std::endl;
    std::cout << std::endl;

    // ===== STEP 3: Run 1 timestep and examine forces =====
    std::cout << "Running 1 timestep with diagnostics..." << std::endl;
    std::cout << std::endl;

    // Initial velocity should be zero
    std::vector<float> h_ux_init(num_cells);
    solver.copyVelocityToHost(h_ux_init.data(), nullptr, nullptr);
    float max_ux_init = *std::max_element(h_ux_init.begin(), h_ux_init.end());

    std::cout << "Initial velocity: max_ux = " << max_ux_init << " (should be ~0)" << std::endl;

    // Take 1 step
    solver.step(config.dt);

    // Check velocity after 1 step
    std::vector<float> h_ux_step1(num_cells);
    solver.copyVelocityToHost(h_ux_step1.data(), nullptr, nullptr);

    float max_ux_step1 = *std::max_element(h_ux_step1.begin(), h_ux_step1.end());

    std::cout << "After 1 step: max_ux = " << max_ux_step1 << " (lattice units)" << std::endl;
    std::cout << "              = " << (max_ux_step1 * config.dx / config.dt) << " m/s (physical)" << std::endl;
    std::cout << std::endl;

    // ===== STEP 4: Use GPU printf to trace forces in kernel =====
    std::cout << "GPU force diagnostics (from kernel printf):" << std::endl;
    std::cout << "  [Look for [Darcy DEBUG] and [GPU] output above]" << std::endl;
    std::cout << std::endl;

    // ===== STEP 5: Run more steps and track velocity evolution =====
    std::cout << "Running 100 steps to observe velocity evolution..." << std::endl;
    std::cout << std::setw(8) << "Step" << std::setw(18) << "max_ux (lattice)"
              << std::setw(18) << "max_v (m/s)" << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    for (int step = 0; step < 100; ++step) {
        solver.step(config.dt);

        if ((step + 1) % 20 == 0) {
            solver.copyVelocityToHost(h_ux_step1.data(), nullptr, nullptr);
            float max_ux = *std::max_element(h_ux_step1.begin(), h_ux_step1.end());
            float max_v_phys = max_ux * config.dx / config.dt;

            std::cout << std::setw(8) << (step + 1)
                      << std::setw(18) << std::scientific << std::setprecision(3) << max_ux
                      << std::setw(18) << std::fixed << std::setprecision(6) << max_v_phys
                      << std::endl;
        }
    }

    float final_v_max = solver.getMaxVelocity();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "DIAGNOSTIC RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Final velocity: " << final_v_max << " m/s" << std::endl;
    std::cout << std::endl;

    if (final_v_max < 0.01f) {
        std::cout << "❌ PROBLEM CONFIRMED: Velocity is essentially zero" << std::endl;
        std::cout << "   Forces are not generating motion!" << std::endl;
    } else if (final_v_max < 0.1f) {
        std::cout << "⚠️  WEAK RESPONSE: Velocity is too small" << std::endl;
        std::cout << "   Expected: 0.7-1.5 m/s for Marangoni" << std::endl;
    } else if (final_v_max < 0.5f) {
        std::cout << "⚠️  PARTIAL SUCCESS: Velocity present but weak" << std::endl;
        std::cout << "   May indicate force attenuation or unit conversion issue" << std::endl;
    } else {
        std::cout << "✓ SUCCESS: Significant velocity generated" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Expected velocity (from analytical Marangoni):" << std::endl;
    std::cout << "  v ~ |dσ/dT| * ΔT / μ" << std::endl;
    std::cout << "    = " << std::abs(config.dsigma_dT) << " * " << (T_hot - T_cold)
              << " / " << (config.density * config.kinematic_viscosity * config.dx * config.dx / config.dt) << std::endl;
    std::cout << "    ~ " << (std::abs(config.dsigma_dT) * (T_hot - T_cold) / 0.005f) << " m/s" << std::endl;
    std::cout << std::endl;

    cudaFree(d_temp);

    // Test expectation: velocity should be non-zero
    EXPECT_GT(final_v_max, 0.01f) << "Velocity should be generated by Marangoni forces";
}

/**
 * @brief Test 2: Isolate Darcy Damping Effect
 */
TEST(ForceVelocityCouplingDebug, DarcyDampingIsolation) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG TEST: Darcy Damping Isolation" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Configuration WITHOUT Darcy
    MultiphysicsConfig config_no_darcy;
    config_no_darcy.nx = 40;
    config_no_darcy.ny = 40;
    config_no_darcy.nz = 20;
    config_no_darcy.dx = 2e-6f;
    config_no_darcy.dt = 1e-7f;

    config_no_darcy.enable_thermal = true;
    config_no_darcy.enable_fluid = true;
    config_no_darcy.enable_vof = true;
    config_no_darcy.enable_vof_advection = false;
    config_no_darcy.enable_surface_tension = false;
    config_no_darcy.enable_marangoni = true;
    config_no_darcy.enable_darcy = false;  // ← DISABLED
    config_no_darcy.enable_laser = false;

    config_no_darcy.material = MaterialDatabase::getTi6Al4V();
    config_no_darcy.thermal_diffusivity = 5.8e-6f;
    config_no_darcy.kinematic_viscosity = 0.0333f;
    config_no_darcy.density = 4110.0f;
    config_no_darcy.dsigma_dT = -0.26e-3f;

    std::cout << "Test A: WITHOUT Darcy damping" << std::endl;
    MultiphysicsSolver solver_no_darcy(config_no_darcy);
    solver_no_darcy.initialize(2300.0f, 0.5f);

    // Set temperature gradient
    int num_cells = config_no_darcy.nx * config_no_darcy.ny * config_no_darcy.nz;
    std::vector<float> h_temp(num_cells);
    for (int i = 0; i < num_cells; ++i) {
        h_temp[i] = 2000.0f + 500.0f * sinf(i * 0.01f);  // Varying temperature
    }
    float* d_temp;
    cudaMalloc(&d_temp, num_cells * sizeof(float));
    cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    solver_no_darcy.setStaticTemperature(d_temp);

    // Run 100 steps
    for (int step = 0; step < 100; ++step) {
        solver_no_darcy.step(config_no_darcy.dt);
    }

    float v_max_no_darcy = solver_no_darcy.getMaxVelocity();
    std::cout << "  Final velocity: " << v_max_no_darcy << " m/s" << std::endl;
    std::cout << std::endl;

    // Configuration WITH Darcy
    MultiphysicsConfig config_with_darcy = config_no_darcy;
    config_with_darcy.enable_darcy = true;  // ← ENABLED
    config_with_darcy.darcy_coefficient = 1e7f;

    std::cout << "Test B: WITH Darcy damping (C=" << config_with_darcy.darcy_coefficient << ")" << std::endl;
    MultiphysicsSolver solver_with_darcy(config_with_darcy);
    solver_with_darcy.initialize(2300.0f, 0.5f);
    solver_with_darcy.setStaticTemperature(d_temp);

    // Run 100 steps
    for (int step = 0; step < 100; ++step) {
        solver_with_darcy.step(config_with_darcy.dt);
    }

    float v_max_with_darcy = solver_with_darcy.getMaxVelocity();
    std::cout << "  Final velocity: " << v_max_with_darcy << " m/s" << std::endl;
    std::cout << std::endl;

    cudaFree(d_temp);

    // Analysis
    std::cout << "========================================" << std::endl;
    std::cout << "COMPARISON" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Velocity without Darcy: " << v_max_no_darcy << " m/s" << std::endl;
    std::cout << "Velocity with Darcy:    " << v_max_with_darcy << " m/s" << std::endl;
    std::cout << "Ratio (with/without):   " << (v_max_no_darcy > 0 ? v_max_with_darcy / v_max_no_darcy : 0.0f) << std::endl;
    std::cout << std::endl;

    if (v_max_no_darcy > 0.1f && v_max_with_darcy < 0.01f) {
        std::cout << "❌ DARCY IS THE PROBLEM!" << std::endl;
        std::cout << "   Darcy damping is suppressing all motion even in fully liquid regions" << std::endl;
    } else if (v_max_no_darcy < 0.01f && v_max_with_darcy < 0.01f) {
        std::cout << "❌ FORCE GENERATION IS THE PROBLEM!" << std::endl;
        std::cout << "   No velocity even without Darcy → forces not being generated/applied" << std::endl;
    } else {
        std::cout << "✓ Both configurations show motion" << std::endl;
    }
    std::cout << std::endl;

    // Expectations
    EXPECT_GT(v_max_no_darcy, 0.1f) << "Should have significant velocity without Darcy";
    EXPECT_GT(v_max_with_darcy / v_max_no_darcy, 0.8f)
        << "Darcy should not significantly affect fully liquid regions (fl=1.0)";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
