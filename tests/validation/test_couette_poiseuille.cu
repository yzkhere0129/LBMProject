/**
 * @file test_couette_poiseuille.cu
 * @brief Fluid Mechanics Validation: Combined Couette-Poiseuille Flow (BGK)
 *
 * Validates the BGK fluid LBM solver against the analytical solution for
 * combined shear-driven (Couette) and pressure-driven (Poiseuille) flow
 * between parallel plates.
 *
 * PHYSICAL SETUP:
 * - Domain: 2D channel between parallel plates
 * - On-node bounce-back: walls AT y=0 and y=NY-1, channel height H=NY-1
 * - Bottom plate (y=0): Stationary (no-slip bounce-back)
 * - Top plate (y=H): Moving wall (Ladd bounce-back with velocity)
 * - Body force f_x drives Poiseuille component
 * - Periodic BC in x and z directions
 *
 * ANALYTICAL SOLUTION:
 *   u(y) = U_top * (y/H) + (f_x/(2*nu)) * y * (H - y)
 *
 * With force chosen so that (f_x*H^2)/(2*nu) = -3*U_top:
 *   u(eta) = U_top * (3*eta^2 - 2*eta),  eta = y/H
 *
 * ERROR ANALYSIS:
 * On-node bounce-back has O(dx) wall placement error for BGK (tau-dependent).
 * At NY=128 (H=127), the inherent error is ~0.5-1.5%.
 * The L2 error excludes wall nodes (j=0, j=NY-1) since their velocity is
 * imposed, not computed by the solver.
 *
 * ACCEPTANCE CRITERIA:
 * - L2 error < 2% (interior nodes, range-normalized)
 * - Wall velocity match within 1%
 * - Steady state reached
 *
 * REFERENCES:
 * - Guo, Z., Zheng, C., & Shi, B. (2002). PRE 65, 046308.
 * - Krueger, T., et al. (2017). "The Lattice Boltzmann Method." Springer.
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

using namespace lbm::physics;

// ============================================================================
// CONFIGURATION
// ============================================================================

// Domain: increased NY from 64 to 128 for better accuracy
constexpr int NX = 4;
constexpr int NY = 128;
constexpr int NZ = 4;
constexpr int NUM_CELLS = NX * NY * NZ;

// On-node bounce-back: walls at j=0 and j=NY-1
constexpr float H = static_cast<float>(NY - 1);

// Flow parameters in lattice units
constexpr float RE = 10.0f;
constexpr float U_TOP = 0.05f;
constexpr float NU = U_TOP * H / RE;
constexpr float RHO0 = 1.0f;

// Body force: f_x = -6*U_top*nu/H^2 gives u(eta) = U_top*(3*eta^2 - 2*eta)
constexpr float FX = -6.0f * U_TOP * NU / (H * H);
constexpr float FY = 0.0f;
constexpr float FZ = 0.0f;

// Time parameters
// Diffusion time scale: t_diff = H^2/nu = H*Re/U_top
constexpr int MAX_STEPS = 120000;
constexpr int OUTPUT_INTERVAL = 2000;
constexpr int STEADY_CHECK_INTERVAL = 200;
constexpr float STEADY_STATE_THRESHOLD = 1.0e-7f;

// Validation thresholds (tightened from 10% to 2%)
constexpr float L2_ERROR_THRESHOLD = 0.02f;
constexpr float WALL_VELOCITY_TOL = 0.01f;

// ============================================================================
// ANALYTICAL SOLUTION
// ============================================================================

inline float analytical_couette_poiseuille(float y, float U_top, float fx,
                                           float nu, float H_val) {
    float eta = y / H_val;
    float u_couette = U_top * eta;
    float u_poiseuille = (fx * H_val * H_val) / (2.0f * nu) * eta * (1.0f - eta);
    return u_couette + u_poiseuille;
}

// ============================================================================
// ERROR COMPUTATION (excluding wall nodes)
// ============================================================================

/**
 * @brief L2 error over interior nodes only (j=1..ny-2).
 *
 * Wall nodes (j=0, j=ny-1) have imposed velocity via setBoundaryVelocityKernel,
 * so including them in the error metric would mask the actual solver accuracy.
 * Normalizes by velocity range (appropriate for zero-crossing profiles).
 */
float compute_L2_error(const std::vector<float>& u_numerical,
                       int nx, int ny, int nz,
                       float U_top, float fx, float nu, float H_val) {
    double sum_sq = 0.0;
    int count = 0;

    // Compute analytical range over full domain for normalization
    float u_min_ana = 0.0f, u_max_ana = 0.0f;
    for (int j = 0; j < ny; ++j) {
        float y = static_cast<float>(j);
        float ua = analytical_couette_poiseuille(y, U_top, fx, nu, H_val);
        u_min_ana = std::min(u_min_ana, ua);
        u_max_ana = std::max(u_max_ana, ua);
    }

    // L2 error over interior nodes only
    for (int j = 1; j < ny - 1; ++j) {
        float y = static_cast<float>(j);
        float ua = analytical_couette_poiseuille(y, U_top, fx, nu, H_val);

        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                u_avg += u_numerical[idx];
            }
        }
        u_avg /= (nx * nz);

        double err = u_avg - ua;
        sum_sq += err * err;
        count++;
    }

    float u_range = u_max_ana - u_min_ana;
    if (u_range < 1e-20f) return 0.0f;
    return std::sqrt(sum_sq / count) / u_range;
}

float compute_max_error(const std::vector<float>& u_numerical,
                        int nx, int ny, int nz,
                        float U_top, float fx, float nu, float H_val) {
    float max_err = 0.0f;
    for (int j = 1; j < ny - 1; ++j) {
        float y = static_cast<float>(j);
        float ua = analytical_couette_poiseuille(y, U_top, fx, nu, H_val);
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float err = std::abs(u_numerical[idx] - ua);
                max_err = std::max(max_err, err);
            }
        }
    }
    return max_err;
}

bool check_steady_state(const std::vector<float>& u_old,
                        const std::vector<float>& u_new,
                        int size, float threshold) {
    float max_change = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_change = std::max(max_change, std::abs(u_new[i] - u_old[i]));
    }
    return max_change < threshold;
}

// ============================================================================
// OUTPUT
// ============================================================================

void write_velocity_profile_csv(const std::string& filename,
                                const std::vector<float>& u_numerical,
                                int nx, int ny, int nz,
                                float U_top, float fx, float nu, float H_val) {
    std::ofstream file(filename);
    file << "# Couette-Poiseuille Velocity Profile (BGK, NY=" << ny << ")\n";
    file << "# On-node bounce-back: walls at j=0 and j=" << (ny-1) << ", H=" << H_val << "\n";
    file << "# Re=" << RE << ", U_top=" << U_top << ", nu=" << nu << ", fx=" << fx << "\n";
    file << "y,eta,u_numerical,u_analytical,error,error_percent\n";

    for (int j = 0; j < ny; ++j) {
        float y = static_cast<float>(j);
        float eta = y / H_val;
        float ua = analytical_couette_poiseuille(y, U_top, fx, nu, H_val);

        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i < nx; ++i)
                u_avg += u_numerical[i + nx * (j + ny * k)];
        u_avg /= (nx * nz);

        float err = u_avg - ua;
        float err_pct = (std::abs(ua) > 1e-10f) ? (err / ua * 100.0f) : 0.0f;

        file << std::scientific << std::setprecision(6)
             << y << "," << eta << "," << u_avg << "," << ua << ","
             << err << "," << std::fixed << std::setprecision(3) << err_pct << "\n";
    }
    file.close();
}

void write_time_series_csv(const std::string& filename,
                           const std::vector<float>& times,
                           const std::vector<float>& l2_errors,
                           const std::vector<float>& max_changes) {
    std::ofstream file(filename);
    file << "# Couette-Poiseuille Time Evolution (BGK)\n";
    file << "time,L2_error_percent,max_change\n";
    for (size_t i = 0; i < times.size(); ++i) {
        file << std::scientific << std::setprecision(6) << times[i] << ","
             << std::fixed << std::setprecision(4) << l2_errors[i] * 100.0f << ","
             << std::scientific << std::setprecision(4) << max_changes[i] << "\n";
    }
    file.close();
}

// ============================================================================
// HELPER: Run a single Couette-Poiseuille simulation and return L2 error
// ============================================================================

/**
 * @brief Run Couette-Poiseuille flow at given resolution and return L2 error.
 *
 * Used by both the main test and the convergence study.
 * Collision model: BGK.
 *
 * When fix_nu > 0, uses that viscosity (keeping tau fixed across resolutions).
 * This is essential for BGK convergence studies since BGK wall error depends on tau.
 * When fix_nu <= 0, computes nu from Re=10 (default for the main test helper).
 */
float run_couette_poiseuille_bgk(int ny, bool verbose, float fix_nu = -1.0f) {
    const int nx = 4, nz = 4;
    const int num_cells = nx * ny * nz;
    const float h = static_cast<float>(ny - 1);
    const float u_top = 0.05f;
    const float nu = (fix_nu > 0.0f) ? fix_nu : (u_top * h / 10.0f);
    const float fx = -6.0f * u_top * nu / (h * h);

    // Diffusion time scale: t_diff = h^2/nu
    const float t_diff = h * h / nu;
    const int max_steps = static_cast<int>(8.0f * t_diff);
    const float ss_threshold = 1.0e-7f;

    FluidLBM fluid(nx, ny, nz, nu, RHO0,
                   BoundaryType::PERIODIC, BoundaryType::WALL, BoundaryType::PERIODIC,
                   1.0f, 1.0f);
    fluid.initialize(RHO0, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, u_top, 0.0f, 0.0f);

    std::vector<float> h_ux(num_cells), h_ux_old(num_cells, 0.0f);
    bool steady = false;

    for (int step = 1; step <= max_steps; ++step) {
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
        fluid.collisionBGK(fx, 0.0f, 0.0f);
        fluid.streaming();

        if (step % 500 == 0) {
            fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);
            if (check_steady_state(h_ux_old, h_ux, num_cells, ss_threshold)) {
                steady = true;
                if (verbose) {
                    std::cout << "    NY=" << ny << ": steady state at step " << step << "\n";
                }
                break;
            }
            h_ux_old = h_ux;
        }
    }

    if (!steady && verbose) {
        std::cout << "    NY=" << ny << ": [WARNING] steady state not reached in "
                  << max_steps << " steps\n";
    }

    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();
    fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);

    return compute_L2_error(h_ux, nx, ny, nz, u_top, fx, nu, h);
}

// ============================================================================
// MAIN TEST
// ============================================================================

TEST(FluidValidation, CouettePoiseuille) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  COUETTE-POISEUILLE FLOW (BGK)\n";
    std::cout << "========================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << NX << " x " << NY << " x " << NZ << "\n";
    std::cout << "  Channel height H = " << H << " (on-node bounce-back)\n";
    std::cout << "  Re = " << RE << ", U_top = " << U_TOP << "\n";
    std::cout << "  nu = " << NU << ", tau = " << (NU / (1.0f/3.0f) + 0.5f) << "\n";
    std::cout << "  f_x = " << FX << "\n";
    float t_diff = H * RE / U_TOP;
    std::cout << "  Diffusion time scale: " << t_diff << " steps\n";
    std::cout << "  Max steps: " << MAX_STEPS << " (~" << (MAX_STEPS / t_diff)
              << " diffusion times)\n\n";

    // Analytical solution preview
    std::cout << "Analytical solution: u(eta) = U_top*(3*eta^2 - 2*eta)\n";
    std::cout << "  u(0) = " << analytical_couette_poiseuille(0, U_TOP, FX, NU, H) << "\n";
    std::cout << "  u(H/2) = " << analytical_couette_poiseuille(H/2, U_TOP, FX, NU, H) << "\n";
    std::cout << "  u(H) = " << analytical_couette_poiseuille(H, U_TOP, FX, NU, H) << "\n\n";

    // Initialize solver
    FluidLBM fluid(NX, NY, NZ, NU, RHO0,
                   BoundaryType::PERIODIC, BoundaryType::WALL, BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    std::cout << "  Solver tau = " << fluid.getTau() << "\n";
    std::cout << "  Solver omega = " << fluid.getOmega() << "\n\n";

    fluid.initialize(RHO0, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_TOP, 0.0f, 0.0f);

    std::vector<float> h_ux(NUM_CELLS), h_uy(NUM_CELLS), h_uz(NUM_CELLS);
    std::vector<float> h_ux_old(NUM_CELLS, 0.0f);
    std::vector<float> time_points, l2_series, change_series;

    // Time integration
    std::cout << "Running simulation...\n";
    std::cout << std::setw(8) << "Step" << "   "
              << std::setw(10) << "L2[%]" << "   "
              << std::setw(12) << "Max_du" << "   "
              << "Status\n";
    std::cout << std::string(50, '-') << "\n";

    bool steady_state_reached = false;
    int steady_step = -1;

    for (int step = 1; step <= MAX_STEPS; ++step) {
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
        fluid.collisionBGK(FX, FY, FZ);
        fluid.streaming();

        if (step % OUTPUT_INTERVAL == 0 || step == MAX_STEPS) {
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            float l2 = compute_L2_error(h_ux, NX, NY, NZ, U_TOP, FX, NU, H);

            float max_change = 0.0f;
            for (int i = 0; i < NUM_CELLS; ++i)
                max_change = std::max(max_change, std::abs(h_ux[i] - h_ux_old[i]));

            time_points.push_back(static_cast<float>(step));
            l2_series.push_back(l2);
            change_series.push_back(max_change);

            std::string status = steady_state_reached ? "STEADY" : "TRANSIENT";
            if (!steady_state_reached && max_change < STEADY_STATE_THRESHOLD) {
                steady_state_reached = true;
                steady_step = step;
                status = "STEADY";
            }

            std::cout << std::setw(8) << step << "   "
                      << std::setw(10) << std::fixed << std::setprecision(3) << l2 * 100.0f << "   "
                      << std::setw(12) << std::scientific << std::setprecision(2) << max_change << "   "
                      << status << "\n";

            h_ux_old = h_ux;

            if (steady_state_reached && l2 < L2_ERROR_THRESHOLD) {
                std::cout << "\n[INFO] Converged. Stopping.\n";
                break;
            }
        }

        // More frequent steady-state check
        if (step % STEADY_CHECK_INTERVAL == 0 && !steady_state_reached) {
            fluid.copyVelocityToHost(h_ux.data(), nullptr, nullptr);
            if (check_steady_state(h_ux_old, h_ux, NUM_CELLS, STEADY_STATE_THRESHOLD)) {
                steady_state_reached = true;
                steady_step = step;
            }
            h_ux_old = h_ux;
        }
    }

    std::cout << std::string(50, '-') << "\n";
    if (steady_state_reached)
        std::cout << "Steady state at step " << steady_step << "\n\n";
    else
        std::cout << "[WARNING] Steady state not reached\n\n";

    // Final validation
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    float l2_error = compute_L2_error(h_ux, NX, NY, NZ, U_TOP, FX, NU, H);
    float max_error = compute_max_error(h_ux, NX, NY, NZ, U_TOP, FX, NU, H);

    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n\n";
    std::cout << "  L2 error (interior, range-norm): " << std::fixed << std::setprecision(3)
              << l2_error * 100.0f << " %\n";
    std::cout << "  Max error (interior): " << std::scientific << std::setprecision(4)
              << max_error << "\n\n";

    // Wall velocities
    float u_bottom_avg = 0.0f, u_top_avg = 0.0f;
    for (int k = 0; k < NZ; ++k) {
        for (int i = 0; i < NX; ++i) {
            u_bottom_avg += h_ux[i + NX * (0 + NY * k)];
            u_top_avg += h_ux[i + NX * ((NY-1) + NY * k)];
        }
    }
    u_bottom_avg /= (NX * NZ);
    u_top_avg /= (NX * NZ);

    float u_bottom_err = std::abs(u_bottom_avg);
    float u_top_err = std::abs(u_top_avg - U_TOP) / U_TOP;

    std::cout << "  Bottom wall: u = " << std::fixed << std::setprecision(6) << u_bottom_avg
              << " (expected 0, |err| = " << std::scientific << u_bottom_err << ")\n";
    std::cout << "  Top wall: u = " << std::fixed << std::setprecision(6) << u_top_avg
              << " (expected " << U_TOP << ", rel err = "
              << std::setprecision(3) << u_top_err * 100.0f << " %)\n\n";

    // Print a few interior profile points for diagnostics
    std::cout << "Profile sample (interior):\n";
    std::cout << std::setw(6) << "j" << std::setw(10) << "eta"
              << std::setw(14) << "u_num" << std::setw(14) << "u_ana"
              << std::setw(12) << "err[%]\n";
    for (int j : {1, NY/4, NY/2, 3*NY/4, NY-2}) {
        float y = static_cast<float>(j);
        float eta = y / H;
        float ua = analytical_couette_poiseuille(y, U_TOP, FX, NU, H);
        double u_avg = 0.0;
        for (int k = 0; k < NZ; ++k)
            for (int i = 0; i < NX; ++i)
                u_avg += h_ux[i + NX * (j + NY * k)];
        u_avg /= (NX * NZ);
        float pct = (std::abs(ua) > 1e-10f) ? ((u_avg - ua) / ua * 100.0f) : 0.0f;
        std::cout << std::setw(6) << j
                  << std::setw(10) << std::fixed << std::setprecision(4) << eta
                  << std::setw(14) << std::scientific << std::setprecision(5) << u_avg
                  << std::setw(14) << ua
                  << std::setw(12) << std::fixed << std::setprecision(3) << pct << "\n";
    }
    std::cout << "\n";

    // Success criteria
    std::cout << "Criteria:\n";
    bool l2_pass = (l2_error < L2_ERROR_THRESHOLD);
    std::cout << "  L2 < " << L2_ERROR_THRESHOLD * 100.0f << "% ... "
              << (l2_pass ? "PASS" : "FAIL") << " (" << l2_error * 100.0f << "%)\n";

    bool bottom_pass = (u_bottom_err < WALL_VELOCITY_TOL * U_TOP);
    std::cout << "  Bottom wall ... " << (bottom_pass ? "PASS" : "FAIL") << "\n";

    bool top_pass = (u_top_err < WALL_VELOCITY_TOL);
    std::cout << "  Top wall ... " << (top_pass ? "PASS" : "FAIL") << "\n";

    bool steady_pass = steady_state_reached;
    std::cout << "  Steady state ... " << (steady_pass ? "PASS" : "FAIL") << "\n";

    bool nan_ok = true;
    for (int i = 0; i < NUM_CELLS; ++i) {
        if (std::isnan(h_ux[i]) || std::isinf(h_ux[i])) { nan_ok = false; break; }
    }
    std::cout << "  No NaN/Inf ... " << (nan_ok ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Overall: " << ((l2_pass && bottom_pass && top_pass && steady_pass && nan_ok) ? "PASS" : "FAIL")
              << "\n========================================\n\n";

    // Write output files
    const std::string output_dir = "/home/yzk/LBMProject/tests/validation/output_couette_poiseuille";
    mkdir(output_dir.c_str(), 0755);
    write_velocity_profile_csv(output_dir + "/velocity_profile.csv",
                               h_ux, NX, NY, NZ, U_TOP, FX, NU, H);
    write_time_series_csv(output_dir + "/time_series.csv",
                          time_points, l2_series, change_series);

    // Assertions
    EXPECT_LT(l2_error, L2_ERROR_THRESHOLD)
        << "L2 error " << l2_error * 100.0f << "% exceeds " << L2_ERROR_THRESHOLD * 100.0f << "% threshold";
    EXPECT_LT(u_bottom_err, WALL_VELOCITY_TOL * U_TOP) << "Bottom wall velocity error";
    EXPECT_LT(u_top_err, WALL_VELOCITY_TOL) << "Top wall velocity error";
    EXPECT_TRUE(steady_state_reached) << "Steady state not reached";
    EXPECT_TRUE(nan_ok) << "NaN or Inf detected";
}

// ============================================================================
// GRID CONVERGENCE TEST (BGK)
// ============================================================================

/**
 * @brief Verify spatial convergence rate of BGK Couette-Poiseuille.
 *
 * Runs at NY=32, 64, 128 with FIXED tau (nu=0.315, tau=1.445) across all
 * resolutions. This is essential because BGK wall error depends on tau.
 * If tau changes with resolution, the convergence rate is contaminated.
 *
 * With fixed tau and on-node bounce-back, BGK gives O(dx) to O(dx^2)
 * convergence depending on how close tau is to 1.
 */
TEST(FluidValidation, CouettePoiseuilleBGKConvergence) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  BGK GRID CONVERGENCE STUDY\n";
    std::cout << "  (fixed tau across resolutions)\n";
    std::cout << "========================================\n\n";

    // Fix nu=0.315 so tau=1.445 at all resolutions
    // This decouples spatial discretization error from BGK wall error
    const float fixed_nu = 0.315f;
    const float fixed_tau = fixed_nu / (1.0f/3.0f) + 0.5f;
    std::cout << "  Fixed nu = " << fixed_nu << ", tau = " << fixed_tau << "\n\n";

    const int resolutions[] = {32, 64, 128};
    const int n_res = 3;
    float errors[3];

    for (int r = 0; r < n_res; ++r) {
        std::cout << "  Running NY=" << resolutions[r] << "...\n";
        errors[r] = run_couette_poiseuille_bgk(resolutions[r], true, fixed_nu);
        std::cout << "    L2 error: " << std::fixed << std::setprecision(4)
                  << errors[r] * 100.0f << " %\n\n";
    }

    // Compute convergence orders
    std::cout << "Convergence rates:\n";
    std::cout << std::setw(8) << "NY" << std::setw(14) << "L2[%]"
              << std::setw(10) << "Order\n";
    std::cout << std::string(32, '-') << "\n";

    std::cout << std::setw(8) << resolutions[0]
              << std::setw(14) << std::fixed << std::setprecision(4) << errors[0] * 100.0f
              << std::setw(10) << "---" << "\n";

    for (int r = 1; r < n_res; ++r) {
        float order = std::log(errors[r-1] / errors[r]) / std::log(2.0f);
        std::cout << std::setw(8) << resolutions[r]
                  << std::setw(14) << std::fixed << std::setprecision(4) << errors[r] * 100.0f
                  << std::setw(10) << std::setprecision(2) << order << "\n";

        // Error must decrease with resolution
        EXPECT_LT(errors[r], errors[r-1])
            << "Error should decrease: NY=" << resolutions[r]
            << " (" << errors[r]*100 << "%) vs NY=" << resolutions[r-1]
            << " (" << errors[r-1]*100 << "%)";
    }

    // With fixed tau=1.445, BGK on-node bounce-back should give order ~1-2.
    // We require at least 0.8 (allowing some margin for the tau-dependent
    // wall placement error that persists even at fixed tau).
    float order_fine = std::log(errors[1] / errors[2]) / std::log(2.0f);
    EXPECT_GT(order_fine, 0.8f)
        << "Convergence order " << order_fine << " is below 0.8 (expected ~1-2 for BGK with fixed tau)";

    std::cout << "\n========================================\n\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
