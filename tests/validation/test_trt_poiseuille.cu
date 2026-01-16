/**
 * @file test_trt_poiseuille.cu
 * @brief Pure Poiseuille Flow Validation with TRT Collision Model
 *
 * This test validates pure pressure-driven flow with NO moving walls using
 * the Two-Relaxation-Time (TRT) collision operator for improved accuracy.
 *
 * The analytical solution is a PARABOLIC velocity profile:
 *   u(y) = u_max * 4 * eta * (1 - eta)
 *
 * TRT improves accuracy for force-driven flows and should achieve better
 * convergence (~0.3-0.5% L2 error) compared to BGK's 1.13% error.
 *
 * CONFIGURATION (matching test_pure_poiseuille.cu):
 * - Grid: 10 × 30 × 10
 * - Reynolds number: 10
 * - Body force: Computed for u_max ~ 0.15 lattice units
 * - Moving walls: NONE (both walls no-slip)
 * - Magic number: Λ = 3/16 (optimal for wall accuracy)
 *
 * EXPECTED IMPROVEMENT:
 * - BGK: 1.13% L2 error
 * - TRT: ~0.3-0.5% L2 error (2-3× better accuracy)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"

using namespace lbm::physics;

// ============================================================================
// CONFIGURATION
// ============================================================================

// Domain parameters - matching test_pure_poiseuille.cu
constexpr int NX = 10;                    // Grid resolution x
constexpr int NY = 30;                    // Grid resolution y (channel height)
constexpr int NZ = 10;                    // Grid resolution z
constexpr int NUM_CELLS = NX * NY * NZ;

// Channel geometry in LATTICE UNITS
constexpr float H = static_cast<float>(NY - 1);  // Channel height = 29 [lattice units]

// Flow parameters - MATCHING WALBERLA
constexpr float RE = 10.0f;
constexpr float NU = 0.309524f;           // Kinematic viscosity
constexpr float U_MAX_TARGET = 0.154762f; // Target max velocity at channel center
constexpr float RHO0 = 1.0f;

// Body force computed from: u_max = f_x * H² / (8 * nu)
// => f_x = 8 * nu * u_max / H²
constexpr float FX = 8.0f * NU * U_MAX_TARGET / (H * H);  // ~0.000426
constexpr float FY = 0.0f;
constexpr float FZ = 0.0f;

// TRT parameters (hardcoded in implementation)
// Magic number Λ = 3/16 (optimal for walls) is built into collisionTRT()

// Time parameters
constexpr int MAX_STEPS = 7000;           // Same as BGK test
constexpr float DT = 1.0f;

// Validation thresholds - STRICTER for TRT
constexpr float L2_ERROR_THRESHOLD = 0.006f;   // 0.6% (2× better than BGK's 1.13%)
constexpr float FLOW_RATE_ERROR_THRESHOLD = 0.01f;  // 1% flow rate error

// Output
constexpr int OUTPUT_INTERVAL = 500;
const std::string OUTPUT_DIR = "output_trt_poiseuille";

// ============================================================================
// ANALYTICAL SOLUTION - PURE POISEUILLE (PARABOLIC)
// ============================================================================

float analytical_pure_poiseuille(float y) {
    // Pure Poiseuille: u(y) = u_max * 4 * eta * (1 - eta)
    // where eta = y/H and u_max = f_x * H² / (8 * nu)
    float eta = y / H;
    float u_max = FX * H * H / (8.0f * NU);
    return u_max * 4.0f * eta * (1.0f - eta);
}

float analytical_flow_rate() {
    // Flow rate Q = (2/3) * u_max * H * W (for parallel plates)
    float u_max = FX * H * H / (8.0f * NU);
    return (2.0f / 3.0f) * u_max * H * static_cast<float>(NX);
}

// ============================================================================
// ERROR COMPUTATION
// ============================================================================

float compute_L2_error(const std::vector<float>& u_numerical, int nx, int ny, int nz) {
    float sum_sq_error = 0.0f;
    float sum_sq_ana = 0.0f;
    int count = 0;

    for (int j = 1; j < ny - 1; j++) {  // Skip boundary cells
        float y = static_cast<float>(j);
        float u_analytical = analytical_pure_poiseuille(y);

        // Average numerical velocity at this y-level
        float u_avg = 0.0f;
        for (int k = 0; k < nz; k++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + nx * (j + ny * k);
                u_avg += u_numerical[idx];
            }
        }
        u_avg /= (nx * nz);

        float error = u_avg - u_analytical;
        sum_sq_error += error * error;
        sum_sq_ana += u_analytical * u_analytical;
        count++;
    }

    // Relative L2 error (normalized by analytical norm)
    float l2_error = std::sqrt(sum_sq_error / count);
    float l2_norm = std::sqrt(sum_sq_ana / count);

    return (l2_norm > 1e-10f) ? (l2_error / l2_norm) : l2_error;
}

float compute_flow_rate(const std::vector<float>& u_numerical, int nx, int ny, int nz) {
    // Compute numerical flow rate across cross-section
    int k_center = nz / 2;
    float flow_rate = 0.0f;

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = i + nx * (j + ny * k_center);
            flow_rate += u_numerical[idx];
        }
    }

    return flow_rate;
}

void save_velocity_profile(const std::string& filename,
                          const std::vector<float>& u_numerical,
                          int nx, int ny, int nz) {
    std::ofstream file(filename);
    file << std::setprecision(16);
    file << "# TRT Pure Poiseuille Flow Velocity Profile\n";
    file << "# y,eta,u_numerical,u_analytical,error\n";
    file << "y,eta,u_numerical,u_analytical,error\n";

    for (int j = 0; j < ny; j++) {
        float y = static_cast<float>(j);
        float eta = y / H;
        float u_analytical = analytical_pure_poiseuille(y);

        // Average numerical velocity at this y-level
        float u_avg = 0.0f;
        for (int k = 0; k < nz; k++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + nx * (j + ny * k);
                u_avg += u_numerical[idx];
            }
        }
        u_avg /= (nx * nz);

        float error = u_avg - u_analytical;
        file << y << "," << eta << "," << u_avg << ","
             << u_analytical << "," << error << "\n";
    }
    file.close();
}

// ============================================================================
// TEST CASE
// ============================================================================

TEST(FluidValidation, TRTPoiseuilleFlow) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  TRT PURE POISEUILLE FLOW VALIDATION\n";
    std::cout << "  (Two-Relaxation-Time collision model)\n";
    std::cout << "========================================\n\n";

    // Create output directory
    mkdir(OUTPUT_DIR.c_str(), 0777);

    float u_max_ana = FX * H * H / (8.0f * NU);

    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << NX << " x " << NY << " x " << NZ << std::endl;
    std::cout << "  Channel height H: " << H << std::endl;
    std::cout << "  Reynolds number: " << RE << std::endl;
    std::cout << "  Kinematic viscosity nu: " << NU << std::endl;
    std::cout << "  Body force f_x: " << std::scientific << FX << std::endl;
    std::cout << "  Expected u_max: " << u_max_ana << std::endl;
    std::cout << "  Moving walls: NONE (both no-slip)" << std::endl;
    std::cout << "  Magic number Λ: 3/16 (hardcoded, optimal for walls)" << std::endl;
    std::cout << "  Analytical solution: u(y) = u_max * 4*eta*(1-eta) [PARABOLIC]" << std::endl;
    std::cout << std::endl;

    // Initialize solver with body force
    FluidLBM fluid(NX, NY, NZ,
                   NU,                              // kinematic viscosity
                   RHO0,                            // reference density
                   BoundaryType::PERIODIC,          // periodic in x
                   BoundaryType::WALL,              // walls in y (both no-slip)
                   BoundaryType::PERIODIC,          // periodic in z
                   DT,                              // dt = 1.0
                   1.0f);                           // dx = 1.0

    std::cout << "TRT solver parameters:" << std::endl;
    std::cout << "  tau = " << fluid.getTau() << std::endl;
    std::cout << "  omega = " << fluid.getOmega() << std::endl;
    std::cout << "  Magic number Λ = 3/16 (hardcoded)" << std::endl;
    std::cout << std::endl;

    // Initialize with zero velocity
    fluid.initialize(RHO0, 0.0f, 0.0f, 0.0f);

    std::cout << "  Body force (applied in collision): (" << FX << ", 0, 0)\n\n";

    // Allocate host arrays
    std::vector<float> h_ux(NUM_CELLS);
    std::vector<float> h_uy(NUM_CELLS);
    std::vector<float> h_uz(NUM_CELLS);

    std::cout << "Running TRT simulation for " << MAX_STEPS << " steps..." << std::endl;

    // Time loop - PURE POISEUILLE: body force but no moving wall, TRT collision!
    for (int step = 0; step <= MAX_STEPS; step++) {
        if (step > 0) {
            // LBM execution order:
            // 1. Apply boundary conditions
            // 2. Compute macroscopic quantities
            // 3. TRT collision with body force (Guo forcing)
            // 4. Streaming
            fluid.applyBoundaryConditions(1);
            fluid.computeMacroscopic();
            fluid.collisionTRT(FX, FY, FZ);  // TRT with body force and Λ=3/16 (hardcoded)
            fluid.streaming();
        }

        if (step % OUTPUT_INTERVAL == 0 && step > 0) {
            // Copy velocity to host
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            float l2_error = compute_L2_error(h_ux, NX, NY, NZ);
            std::cout << "Step " << step << ": L2 error = "
                      << std::fixed << std::setprecision(4) << l2_error * 100.0f << "%" << std::endl;
        }
    }

    // Final validation
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    float final_l2_error = compute_L2_error(h_ux, NX, NY, NZ);
    float numerical_flow_rate = compute_flow_rate(h_ux, NX, NY, NZ);
    float analytical_flow = analytical_flow_rate();
    float flow_rate_error = std::abs(numerical_flow_rate - analytical_flow) / analytical_flow;

    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "L2 relative error: " << final_l2_error * 100.0f << "%" << std::endl;
    std::cout << "Flow rate - numerical: " << numerical_flow_rate << ", analytical: " << analytical_flow << std::endl;
    std::cout << "Flow rate error: " << flow_rate_error * 100.0f << "%" << std::endl;
    std::cout << "Thresholds: L2 < " << L2_ERROR_THRESHOLD * 100.0f << "%, Flow rate < "
              << FLOW_RATE_ERROR_THRESHOLD * 100.0f << "%" << std::endl;

    // Save profile for analysis
    save_velocity_profile(OUTPUT_DIR + "/velocity_profile.csv", h_ux, NX, NY, NZ);
    std::cout << "Velocity profile saved to: " << OUTPUT_DIR << "/velocity_profile.csv" << std::endl;

    // Check center velocity (should be u_max)
    int j_center = NY / 2;
    float u_center = 0.0f;
    for (int k = 0; k < NZ; k++) {
        for (int i = 0; i < NX; i++) {
            int idx = i + NX * (j_center + NY * k);
            u_center += h_ux[idx];
        }
    }
    u_center /= (NX * NZ);

    std::cout << "\nCenter velocity (y=" << j_center << "):" << std::endl;
    std::cout << "  Numerical: " << u_center << std::endl;
    std::cout << "  Analytical (u_max): " << u_max_ana << std::endl;

    // Assertions
    EXPECT_LT(final_l2_error, L2_ERROR_THRESHOLD)
        << "TRT L2 error " << final_l2_error * 100.0f << "% exceeds threshold "
        << L2_ERROR_THRESHOLD * 100.0f << "% (expected ~0.3-0.5% vs BGK's 1.13%)";

    EXPECT_LT(flow_rate_error, FLOW_RATE_ERROR_THRESHOLD)
        << "Flow rate error " << flow_rate_error * 100.0f << "% exceeds threshold "
        << FLOW_RATE_ERROR_THRESHOLD * 100.0f << "%";

    std::cout << "\n=== TRT vs BGK comparison ===" << std::endl;
    std::cout << "BGK (test_pure_poiseuille): ~1.13% L2 error" << std::endl;
    std::cout << "TRT (this test): " << final_l2_error * 100.0f << "% L2 error" << std::endl;
    std::cout << "Expected improvement: ~2-3× better accuracy" << std::endl;
    std::cout << "walberla PoiseuilleChannel achieves 0.65% flow rate error" << std::endl;
    std::cout << "Our TRT implementation: " << flow_rate_error * 100.0f << "% flow rate error" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
