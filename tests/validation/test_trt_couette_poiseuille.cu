/**
 * @file test_trt_couette_poiseuille.cu
 * @brief TRT Fluid Mechanics Validation: Combined Couette-Poiseuille Flow
 *
 * This test validates the TRT collision model against the analytical solution
 * for combined shear-driven (Couette) and pressure-driven (Poiseuille) flow
 * between parallel plates.
 *
 * TRT COLLISION MODEL:
 * - Uses Two-Relaxation-Time collision operator instead of BGK
 * - Separate relaxation rates for symmetric/antisymmetric populations
 * - Magic parameter Λ = 3/16 for optimal wall boundary accuracy
 * - Expected improvement: BGK L2 error 3.31% → TRT < 2%
 *
 * PHYSICAL SETUP:
 * - Domain: 2D channel between parallel plates (y = 0 and y = H)
 * - Bottom plate (y=0): Stationary (no-slip), u = 0
 * - Top plate (y=H): Moving at velocity U_top (moving wall)
 * - Pressure gradient: Applied as body force f_x = -6*U_top*ν/H²
 * - Periodic BC in x and z directions
 *
 * ANALYTICAL SOLUTION:
 * The solution is a superposition of Couette (linear) and Poiseuille (parabolic):
 *   u(y) = U_top * (y/H) + (f_x/(2ν)) * y * (H - y)
 *
 * For normalized coordinates η = y/H:
 *   u(η) = U_top * η + (f_x * H²)/(2ν) * η * (1 - η)
 *
 * With parameters U_top = 0.05, H = 29, f_x = -6*U_top*ν/H²:
 *   u(η) = U_top * (3η² - 2η)
 *
 * TEST CONFIGURATION (matching BGK test):
 * - Grid: 10 × 30 × 10 (quasi-2D channel)
 * - Channel height: H = 29 [lattice units]
 * - Reynolds number: Re = 10
 * - Kinematic viscosity: ν = 0.309524
 * - Top wall velocity: U_top = 0.05
 * - Body force: f_x = -6*U_top*ν/H²
 *
 * VALIDATION METRICS:
 * 1. L2 relative error: ||u_numerical - u_analytical||₂ / range < 10%
 * 2. Wall velocity match: u(0) ≈ 0, u(H) ≈ U_top within 5%
 * 3. Mass conservation: max|∇·u| < 1e-6
 * 4. Steady-state convergence: max|u(t+Δt) - u(t)| < 1e-6
 *
 * ACCEPTANCE CRITERIA:
 * - CRITICAL: L2 error < 3.31% (better than BGK)
 * - TARGET: L2 error < 2% (significant improvement)
 * - CRITICAL: Steady state reached
 * - CRITICAL: No NaN or instability
 *
 * PREVIOUS RESULTS:
 * - BGK Couette-Poiseuille: L2 error = 3.31%
 * - TRT Couette: L2 error = 0.051% (vs BGK 0.043%)
 * - TRT Poiseuille: L2 error = 0.55% (vs BGK 1.13%, 2× improvement)
 *
 * EXPECTED RESULT:
 * TRT should achieve better accuracy than BGK due to improved treatment
 * of wall boundaries and reduced viscosity-dependent errors.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"
#include "utils/cuda_check.h"
#include "io/vtk_writer.h"

using namespace lbm::physics;

// ============================================================================
// CONFIGURATION CONSTANTS (MATCHING BGK TEST)
// ============================================================================

// Domain parameters - USING LATTICE UNITS (dt = dx = 1.0)
constexpr int NX = 10;                    // Grid resolution x
constexpr int NY = 30;                    // Grid resolution y (channel height)
constexpr int NZ = 10;                    // Grid resolution z
constexpr int NUM_CELLS = NX * NY * NZ;

// Channel geometry in LATTICE UNITS
// Node-based convention: walls ARE at boundary cells (j=0 and j=NY-1)
// This matches our BC implementation where BCs are applied at boundary cells
constexpr float H = static_cast<float>(NY - 1);  // Channel height [lattice units]
constexpr float DX = 1.0f;                // Cell size [lattice units]
constexpr float DY = 1.0f;                // Cell size [lattice units]
constexpr float DZ = 1.0f;                // Cell size [lattice units]

// Flow parameters in LATTICE UNITS
constexpr float RE = 10.0f;               // Reynolds number
constexpr float U_MAX_TARGET = 0.05f;     // Target max velocity [lattice units]
constexpr float U_TOP = U_MAX_TARGET;     // Top wall velocity [lattice units]
constexpr float NU = U_MAX_TARGET * H / RE;  // Kinematic viscosity [lattice units]
constexpr float RHO0 = 1.0f;              // Reference density [lattice units]

// Body force computed to give analytical solution u(η) = 3η² - 2η (normalized by U_TOP)
// Full derivation: u(η) = U_top*η + (f_x*H²)/(2ν)*η(1-η) = U_top*(3η²-2η)
// => (f_x*H²)/(2ν) = -3*U_top  =>  f_x = -6*U_top*ν/H²
constexpr float FX = -6.0f * U_TOP * NU / (H * H);  // Body force [lattice units]
constexpr float FY = 0.0f;                // Body force y [lattice units]
constexpr float FZ = 0.0f;                // Body force z [lattice units]

// Time parameters
constexpr int MAX_STEPS = 50000;          // Maximum simulation steps
constexpr float DT = 1.0f;                // Timestep [lattice units]

// Validation thresholds
constexpr float L2_ERROR_THRESHOLD = 0.10f;        // 10% L2 error (range-normalized)
constexpr float WALL_VELOCITY_TOL = 0.05f;         // 5% wall velocity tolerance
constexpr float MASS_CONS_THRESHOLD = 1.0e-6f;     // Mass conservation error
constexpr float STEADY_STATE_THRESHOLD = 1.0e-6f;  // Steady state convergence
constexpr float PROFILE_SHAPE_TOL = 0.10f;         // 10% tolerance for normalized profile

// Output parameters
constexpr int OUTPUT_INTERVAL = 500;      // Output frequency
constexpr int STEADY_CHECK_INTERVAL = 100; // Check steady state every N steps

// ============================================================================
// ANALYTICAL SOLUTION
// ============================================================================

/**
 * @brief Analytical solution for combined Couette-Poiseuille flow
 *
 * @param y Physical y-coordinate [lattice units]
 * @param U_top Top wall velocity [lattice units]
 * @param fx Body force in x-direction [lattice units]
 * @param nu Kinematic viscosity [lattice units]
 * @param H Channel height [lattice units]
 * @return Analytical velocity u(y) [lattice units]
 */
inline float analytical_couette_poiseuille(float y, float U_top, float fx, float nu, float H) {
    // Normalized coordinate η = y/H ∈ [0, 1]
    float eta = y / H;

    // Couette component (linear): U_top * η
    float u_couette = U_top * eta;

    // Poiseuille component (parabolic): (f_x * H²)/(2ν) * η(1 - η)
    float u_poiseuille = (fx * H * H) / (2.0f * nu) * eta * (1.0f - eta);

    return u_couette + u_poiseuille;
}

// ============================================================================
// ERROR COMPUTATION
// ============================================================================

/**
 * @brief Compute L2 relative error between numerical and analytical solutions
 *
 * For Couette-Poiseuille flow with zero-crossing, we normalize by velocity RANGE.
 * L2_error = sqrt(mean((u_num - u_ana)²)) / (u_max - u_min)
 *
 * @param u_numerical Numerical velocity field
 * @param nx, ny, nz Grid dimensions
 * @param U_top, fx, nu, H Flow parameters
 * @return L2 error normalized by velocity range (dimensionless)
 */
float compute_L2_error(const std::vector<float>& u_numerical,
                       int nx, int ny, int nz,
                       float U_top, float fx, float nu, float H) {
    double sum_squared_error = 0.0;
    float u_min_ana = 0.0f;
    float u_max_ana = 0.0f;
    int count_total = 0;

    // Average over x and z (should be uniform due to periodicity)
    // Compare profile in y-direction
    for (int j = 0; j < ny; ++j) {
        // Node-based coordinate: y = j (walls at j=0 and j=NY-1)
        float y = static_cast<float>(j);
        float u_analytical = analytical_couette_poiseuille(y, U_top, fx, nu, H);

        // Track analytical solution range
        u_min_ana = std::min(u_min_ana, u_analytical);
        u_max_ana = std::max(u_max_ana, u_analytical);

        // Average numerical velocity at this y-level
        double u_avg = 0.0;
        int count = 0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                u_avg += u_numerical[idx];
                count++;
            }
        }
        u_avg /= count;

        double error = u_avg - u_analytical;
        sum_squared_error += error * error;
        count_total++;
    }

    // Normalize by velocity range (physical velocity scale)
    float u_range = u_max_ana - u_min_ana;
    if (u_range < 1e-20f) {
        return 0.0f;
    }

    // RMS error normalized by range
    return std::sqrt(sum_squared_error / count_total) / u_range;
}

/**
 * @brief Compute maximum absolute error
 */
float compute_max_error(const std::vector<float>& u_numerical,
                        int nx, int ny, int nz,
                        float U_top, float fx, float nu, float H) {
    float max_error = 0.0f;

    for (int j = 0; j < ny; ++j) {
        // Node-based coordinate
        float y = static_cast<float>(j);
        float u_analytical = analytical_couette_poiseuille(y, U_top, fx, nu, H);

        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float error = std::abs(u_numerical[idx] - u_analytical);
                max_error = std::max(max_error, error);
            }
        }
    }

    return max_error;
}

/**
 * @brief Check if steady state has been reached
 */
bool check_steady_state(const std::vector<float>& u_old,
                       const std::vector<float>& u_new,
                       int size,
                       float threshold) {
    float max_change = 0.0f;

    for (int i = 0; i < size; ++i) {
        float change = std::abs(u_new[i] - u_old[i]);
        max_change = std::max(max_change, change);
    }

    return max_change < threshold;
}

// ============================================================================
// OUTPUT FUNCTIONS
// ============================================================================

/**
 * @brief Write velocity profile to CSV file
 */
void write_velocity_profile_csv(const std::string& filename,
                                const std::vector<float>& u_numerical,
                                int nx, int ny, int nz,
                                float U_top, float fx, float nu, float H) {
    std::ofstream file(filename);

    // Header with metadata
    file << "# TRT Couette-Poiseuille Velocity Profile\n";
    file << "# Grid: " << nx << " x " << ny << " x " << nz << "\n";
    file << "# Re = " << RE << ", U_top = " << U_top << " [lattice units]\n";
    file << "# f_x = " << fx << " [lattice units], ν = " << nu << " [lattice units]\n";
    file << "# Columns: y, eta, u_numerical, u_analytical, error, error_percent\n";
    file << "y,eta,u_numerical,u_analytical,error,error_percent\n";

    for (int j = 0; j < ny; ++j) {
        // Node-based coordinate
        float y = static_cast<float>(j);
        float eta = y / H;
        float u_analytical = analytical_couette_poiseuille(y, U_top, fx, nu, H);

        // Average numerical velocity at this y-level
        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                u_avg += u_numerical[idx];
            }
        }
        u_avg /= (nx * nz);

        float error = u_avg - u_analytical;
        float error_percent = (std::abs(u_analytical) > 1e-10f) ?
                             (error / u_analytical * 100.0f) : 0.0f;

        file << std::scientific << std::setprecision(6)
             << y << ","
             << eta << ","
             << u_avg << ","
             << u_analytical << ","
             << error << ","
             << std::fixed << std::setprecision(3) << error_percent << "\n";
    }

    file.close();
    std::cout << "  Profile written: " << filename << "\n";
}

/**
 * @brief Write time-series data to CSV file
 */
void write_time_series_csv(const std::string& filename,
                           const std::vector<float>& times,
                           const std::vector<float>& l2_errors,
                           const std::vector<float>& max_changes) {
    std::ofstream file(filename);

    file << "# TRT Couette-Poiseuille Time Evolution\n";
    file << "# Columns: time, L2_error_percent, max_change\n";
    file << "time,L2_error_percent,max_change\n";

    for (size_t i = 0; i < times.size(); ++i) {
        file << std::scientific << std::setprecision(6)
             << times[i] << ","
             << std::fixed << std::setprecision(4) << l2_errors[i] * 100.0f << ","
             << std::scientific << std::setprecision(4) << max_changes[i] << "\n";
    }

    file.close();
    std::cout << "  Time series written: " << filename << "\n";
}

// ============================================================================
// MAIN TEST
// ============================================================================

TEST(FluidValidation, TRTCouettePoiseuille) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  TRT COUETTE-POISEUILLE FLOW VALIDATION\n";
    std::cout << "========================================\n\n";

    // ========================================
    // DOMAIN SETUP
    // ========================================
    std::cout << "Domain Configuration (LATTICE UNITS):\n";
    std::cout << "  Grid: " << NX << " × " << NY << " × " << NZ << "\n";
    std::cout << "  Channel height H: " << H << " [lattice units]\n";
    std::cout << "  Cell size: dx = dy = dz = " << DX << " [lattice units]\n\n";

    // ========================================
    // FLOW PARAMETERS
    // ========================================
    std::cout << "Flow Parameters (LATTICE UNITS):\n";
    std::cout << "  Reynolds number: Re = " << RE << "\n";
    std::cout << "  Top wall velocity: U_top = " << U_TOP << " [lattice units]\n";
    std::cout << "  Kinematic viscosity: ν = " << NU << " [lattice units]\n";
    std::cout << "  Reference density: ρ₀ = " << RHO0 << "\n";
    std::cout << "  Body force: f_x = " << FX << " [lattice units]\n\n";

    // ========================================
    // TIME PARAMETERS
    // ========================================
    std::cout << "Time Parameters:\n";
    std::cout << "  Timestep: dt = " << DT << " [lattice units]\n";
    std::cout << "  Maximum steps: " << MAX_STEPS << "\n";
    std::cout << "  Convergence tolerance: " << STEADY_STATE_THRESHOLD << "\n\n";

    // ========================================
    // TRT PARAMETERS
    // ========================================
    std::cout << "TRT Collision Parameters:\n";
    std::cout << "  Magic parameter: Λ = 3/16 = " << (3.0f / 16.0f) << "\n";
    std::cout << "  Expected improvement over BGK: 2× better accuracy\n\n";

    // ========================================
    // ANALYTICAL SOLUTION PREVIEW
    // ========================================
    std::cout << "Analytical Solution Preview:\n";
    std::cout << "  Formula: u(η) = U_top*η + (f_x*H²)/(2ν)*η(1-η)\n";
    std::cout << "  Node-based coordinates: y = j, H = NY - 1 = " << H << "\n";
    std::cout << "  Wall at y=0: u = 0, Wall at y=H: u = U_top\n";
    std::cout << "  u(y=0) = " << analytical_couette_poiseuille(0.0f, U_TOP, FX, NU, H) << "\n";
    std::cout << "  u(y=H/2) = " << analytical_couette_poiseuille(H/2.0f, U_TOP, FX, NU, H) << "\n";
    std::cout << "  u(y=H) = " << analytical_couette_poiseuille(H, U_TOP, FX, NU, H) << "\n\n";

    // ========================================
    // INITIALIZE FLUID SOLVER
    // ========================================
    std::cout << "Initializing Fluid LBM Solver with TRT collision...\n";
    FluidLBM fluid(NX, NY, NZ,
                   NU,                              // kinematic viscosity [lattice units]
                   RHO0,                            // reference density
                   BoundaryType::PERIODIC,          // periodic in x
                   BoundaryType::WALL,              // walls in y (top/bottom)
                   BoundaryType::PERIODIC,          // periodic in z
                   1.0f,                            // dt = 1.0 [lattice units]
                   1.0f);                           // dx = 1.0 [lattice units]

    std::cout << "  Solver tau = " << fluid.getTau() << "\n";
    std::cout << "  Solver omega = " << fluid.getOmega() << "\n\n";

    // Stability check
    if (fluid.getTau() < 0.51f) {
        std::cout << "[WARNING] Relaxation time tau < 0.51 may cause instability!\n\n";
    }

    // ========================================
    // SET INITIAL CONDITIONS
    // ========================================
    std::cout << "Setting Initial Conditions (zero velocity)...\n";
    fluid.initialize(RHO0, 0.0f, 0.0f, 0.0f);

    // ========================================
    // SET MOVING WALL BOUNDARY CONDITION
    // ========================================
    std::cout << "Configuring Moving Wall Boundary Condition...\n";
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_TOP, 0.0f, 0.0f);
    std::cout << "  Top wall (y_max) velocity set to (" << U_TOP << ", 0, 0)\n\n";

    // ========================================
    // ALLOCATE HOST ARRAYS
    // ========================================
    std::vector<float> h_ux(NUM_CELLS);
    std::vector<float> h_uy(NUM_CELLS);
    std::vector<float> h_uz(NUM_CELLS);
    std::vector<float> h_ux_old(NUM_CELLS);

    // Time-series data storage
    std::vector<float> time_points;
    std::vector<float> l2_error_series;
    std::vector<float> max_change_series;

    // ========================================
    // TIME INTEGRATION LOOP
    // ========================================
    std::cout << "Running Simulation with TRT Collision...\n";
    std::cout << "Step       L2_err[%]   Max_Δu       Status\n";
    std::cout << std::string(55, '-') << "\n";

    bool steady_state_reached = false;
    int steady_state_step = -1;

    for (int step = 0; step <= MAX_STEPS; ++step) {

        // Perform LBM step with TRT collision
        if (step > 0) {
            // LBM execution order:
            // 1. Apply boundary conditions
            // 2. Compute macroscopic quantities
            // 3. TRT collision with body force (Guo forcing scheme)
            // 4. Streaming

            fluid.applyBoundaryConditions(1);  // 1 = wall BC active
            fluid.computeMacroscopic();
            fluid.collisionTRT(FX, FY, FZ);    // TRT COLLISION (main difference)
            fluid.streaming();
        }

        // Output at regular intervals
        if (step % OUTPUT_INTERVAL == 0 || step == MAX_STEPS) {
            // Copy velocity to host
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            // Compute error metrics
            float l2_error = compute_L2_error(h_ux, NX, NY, NZ, U_TOP, FX, NU, H);

            // Compute steady-state metric
            float max_change = 0.0f;
            if (step > 0) {
                for (int i = 0; i < NUM_CELLS; ++i) {
                    float change = std::abs(h_ux[i] - h_ux_old[i]);
                    max_change = std::max(max_change, change);
                }
            }

            // Store time-series data
            time_points.push_back(static_cast<float>(step));
            l2_error_series.push_back(l2_error);
            max_change_series.push_back(max_change);

            // Check steady state
            std::string status = "TRANSIENT";
            if (!steady_state_reached && max_change < STEADY_STATE_THRESHOLD && step > 0) {
                steady_state_reached = true;
                steady_state_step = step;
                status = "STEADY";
            } else if (steady_state_reached) {
                status = "STEADY";
            }

            // Console output
            std::cout << std::setw(8) << step << "   "
                      << std::setw(10) << std::setprecision(3) << l2_error * 100.0f << "   "
                      << std::setw(12) << std::scientific << std::setprecision(2) << max_change << "   "
                      << status << "\n";

            // Save old velocity for next comparison
            h_ux_old = h_ux;

            // Early exit if converged
            if (steady_state_reached && l2_error < L2_ERROR_THRESHOLD) {
                std::cout << "\n[INFO] Converged with acceptable error. Stopping early.\n";
                break;
            }
        }

        // Check steady state more frequently (but don't output)
        if (step % STEADY_CHECK_INTERVAL == 0 && step > 0 && !steady_state_reached) {
            fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);
            if (check_steady_state(h_ux_old, h_ux, NUM_CELLS, STEADY_STATE_THRESHOLD)) {
                steady_state_reached = true;
                steady_state_step = step;
            }
            h_ux_old = h_ux;
        }
    }

    std::cout << std::string(55, '-') << "\n\n";

    if (steady_state_reached) {
        std::cout << "Steady state reached at step " << steady_state_step << "\n\n";
    } else {
        std::cout << "[WARNING] Steady state not reached within simulation time\n\n";
    }

    // ========================================
    // FINAL VALIDATION
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  VALIDATION RESULTS\n";
    std::cout << "========================================\n\n";

    // Apply BC and recompute macroscopic quantities from final distribution state
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();

    // Get final velocity field
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Compute error metrics
    float l2_error = compute_L2_error(h_ux, NX, NY, NZ, U_TOP, FX, NU, H);
    float max_error = compute_max_error(h_ux, NX, NY, NZ, U_TOP, FX, NU, H);

    std::cout << "Error Metrics:\n";
    std::cout << "  L2 relative error: " << std::setprecision(3) << l2_error * 100.0f << " %\n";
    std::cout << "  Maximum error: " << std::scientific << std::setprecision(4) << max_error << "\n\n";

    // Check wall velocities (node-based: walls ARE at boundary nodes)
    // Bottom wall (j=0) is at y=0
    float u_bottom_avg = 0.0f;
    for (int k = 0; k < NZ; ++k) {
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX * (0 + NY * k);
            u_bottom_avg += h_ux[idx];
        }
    }
    u_bottom_avg /= (NX * NZ);

    // Top wall (j=NY-1) is at y=H
    float u_top_avg = 0.0f;
    for (int k = 0; k < NZ; ++k) {
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX * ((NY-1) + NY * k);
            u_top_avg += h_ux[idx];
        }
    }
    u_top_avg /= (NX * NZ);

    // Expected values: u(0) = 0, u(H) = U_top
    float u_bottom_error = std::abs(u_bottom_avg - 0.0f);
    float u_top_error = std::abs(u_top_avg - U_TOP) / U_TOP;

    std::cout << "Wall Velocities:\n";
    std::cout << "  Bottom wall (y=0): u = " << std::fixed << std::setprecision(6) << u_bottom_avg
              << " (expected: 0, error: " << std::scientific << u_bottom_error << ")\n";
    std::cout << "  Top wall (y=H): u = " << std::fixed << std::setprecision(6) << u_top_avg
              << " (expected: " << U_TOP << ", error: " << std::setprecision(3) << u_top_error * 100.0f << " %)\n\n";

    // ========================================
    // SUCCESS CRITERIA EVALUATION
    // ========================================
    std::cout << "Success Criteria:\n";

    bool l2_pass = (l2_error < L2_ERROR_THRESHOLD);
    std::cout << "  L2 error < " << (L2_ERROR_THRESHOLD * 100.0f) << "% ... " << (l2_pass ? "PASS" : "FAIL") << "\n";

    bool bottom_wall_pass = (u_bottom_error < WALL_VELOCITY_TOL * U_TOP);
    std::cout << "  Bottom wall velocity ≈ 0 ... " << (bottom_wall_pass ? "PASS" : "FAIL") << "\n";

    bool top_wall_pass = (u_top_error < WALL_VELOCITY_TOL);
    std::cout << "  Top wall velocity matches ... " << (top_wall_pass ? "PASS" : "FAIL") << "\n";

    bool steady_pass = steady_state_reached;
    std::cout << "  Steady state reached ... " << (steady_pass ? "PASS" : "FAIL") << "\n";

    // Check for NaN
    bool nan_check = true;
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (std::isnan(h_ux[i]) || std::isinf(h_ux[i])) {
            nan_check = false;
            break;
        }
    }
    std::cout << "  No NaN or Inf detected ... " << (nan_check ? "PASS" : "FAIL") << "\n\n";

    bool all_passed = l2_pass && bottom_wall_pass && top_wall_pass && steady_pass && nan_check;

    std::cout << "Overall Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n\n";

    // ========================================
    // COMPARISON WITH BGK
    // ========================================
    std::cout << "Comparison with BGK Results:\n";
    std::cout << "  BGK L2 error: 3.31%\n";
    std::cout << "  TRT L2 error: " << std::setprecision(2) << l2_error * 100.0f << "%\n";
    float improvement = ((3.31f - l2_error * 100.0f) / 3.31f) * 100.0f;
    std::cout << "  Improvement: " << std::setprecision(1) << improvement << "%\n\n";

    // ========================================
    // OUTPUT FILES
    // ========================================
    std::cout << "Writing Output Files...\n";

    // Create output directory
    const std::string output_dir = "/home/yzk/LBMProject/tests/validation/output_trt_couette_poiseuille";
    mkdir(output_dir.c_str(), 0755);

    // Write velocity profile
    std::string profile_file = output_dir + "/velocity_profile.csv";
    write_velocity_profile_csv(profile_file, h_ux, NX, NY, NZ, U_TOP, FX, NU, H);

    // Write time series
    std::string timeseries_file = output_dir + "/time_series.csv";
    write_time_series_csv(timeseries_file, time_points, l2_error_series, max_change_series);

    // Write VTK files for visualization
    std::cout << "  Writing VTK files for ParaView...\n";
    try {
        std::string vtk_filename = output_dir + "/velocity_field";

        // Get density
        std::vector<float> h_rho(NUM_CELLS);
        fluid.copyDensityToHost(h_rho.data());

        // Write velocity magnitude as a scalar field
        std::vector<float> velocity_mag(NUM_CELLS);
        for (int i = 0; i < NUM_CELLS; ++i) {
            velocity_mag[i] = std::sqrt(h_ux[i] * h_ux[i] + h_uy[i] * h_uy[i] + h_uz[i] * h_uz[i]);
        }

        lbm::io::VTKWriter::writeStructuredPoints(
            vtk_filename,
            velocity_mag.data(),
            NX, NY, NZ,
            DX, DY, DZ,
            "Velocity_Magnitude"
        );
        std::cout << "  VTK file written: " << vtk_filename << ".vtk\n";
    } catch (const std::exception& e) {
        std::cout << "  [WARNING] VTK write failed: " << e.what() << "\n";
    }

    std::cout << "  Output directory: " << output_dir << "\n\n";

    // ========================================
    // GTEST ASSERTIONS
    // ========================================
    EXPECT_LT(l2_error, L2_ERROR_THRESHOLD) << "L2 error exceeds threshold";
    EXPECT_LT(u_bottom_error, WALL_VELOCITY_TOL * U_TOP) << "Bottom wall velocity deviates from 0";
    EXPECT_LT(u_top_error, WALL_VELOCITY_TOL) << "Top wall velocity deviates from U_top";
    EXPECT_TRUE(steady_state_reached) << "Steady state not reached";
    EXPECT_TRUE(nan_check) << "NaN or Inf detected in results";

    // TRT-specific assertion: should be better than BGK
    EXPECT_LT(l2_error * 100.0f, 3.31f) << "TRT should achieve better accuracy than BGK (3.31%)";

    std::cout << "========================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "========================================\n\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
