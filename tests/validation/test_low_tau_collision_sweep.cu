/**
 * @file test_low_tau_collision_sweep.cu
 * @brief Parameterized Couette-Poiseuille accuracy sweep across TRT and
 *        Regularized collision operators at low τ (0.51 – 0.70).
 *
 * MOTIVATION
 * ----------
 * Track-A of the current sprint reduces dt by 4× (Phase-5 Ma 0.43 → 0.11),
 * driving the operating τ from ~0.695 down to ~0.54.  Before this test existed,
 * TRT was only validated at τ=0.94 (test_trt_couette_poiseuille.cu) and
 * Regularized had ZERO test coverage.  This file provides permanent regression
 * guards for both operators across the full low-τ window.
 *
 * This file supersedes the informal counterparts:
 *   apps/sanity_trt_low_tau.cu     (TRT sweep, CLI-only)
 *   apps/sanity_reg_low_tau.cu     (Regularized sweep, CLI-only)
 * Those apps are kept for ad-hoc manual runs but are not authoritative tests.
 *
 * PHYSICS
 * -------
 * Combined Couette-Poiseuille channel:
 *   Domain:   NX × NY × NZ = 8 × 20 × 8  (quasi-2D, periodic x/z, walls y)
 *   Top wall: moves at U_TOP = 0.005 LU  (Ma ≈ 0.0087, negligible compressibility)
 *   Body force: f_x = -6 U_TOP ν / H²   (chosen so the analytical profile is
 *               exactly u(η) = U_TOP (3η² - 2η), which has a zero crossing)
 *
 * Analytical solution: u(y) = U_TOP (y/H) + (f_x H²)/(2ν) η(1-η)
 * L2 error: range-normalised RMS over the y-averaged profile.
 * Threshold: 0.5% (measured values are 0.15–0.20%; 2–3× safety margin).
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"

using namespace lbm::physics;

// ============================================================================
// DOMAIN CONSTANTS  (match sanity apps exactly so numbers are comparable)
// ============================================================================

namespace {

constexpr int   NX    = 8;
constexpr int   NY    = 20;
constexpr int   NZ    = 8;
constexpr float H_LU  = static_cast<float>(NY - 1);   // = 19 LU
constexpr float U_TOP = 0.005f;

// ============================================================================
// ANALYTICAL SOLUTION
// ============================================================================

inline float analytical_cp(float y, float U_top, float fx, float nu, float H) {
    const float eta = y / H;
    return U_top * eta + (fx * H * H) / (2.0f * nu) * eta * (1.0f - eta);
}

// ============================================================================
// L2 COMPUTATION  (range-normalised, x/z-averaged, same as sanity apps)
// ============================================================================

float compute_l2(const std::vector<float>& ux,
                 float U_top, float fx, float nu) {
    float  u_min = 0.0f, u_max = 0.0f;
    double sum_sq = 0.0;
    int    count  = 0;

    for (int j = 0; j < NY; ++j) {
        const float y   = static_cast<float>(j);
        const float u_a = analytical_cp(y, U_top, fx, nu, H_LU);
        u_min = std::min(u_min, u_a);
        u_max = std::max(u_max, u_a);

        double avg = 0.0;
        for (int k = 0; k < NZ; ++k) {
            for (int i = 0; i < NX; ++i) {
                avg += ux[i + NX * (j + NY * k)];
            }
        }
        avg /= (NX * NZ);

        const double err = avg - u_a;
        sum_sq += err * err;
        ++count;
    }

    const float u_range = u_max - u_min;
    if (u_range < 1e-20f) return 0.0f;
    return static_cast<float>(std::sqrt(sum_sq / count) / u_range);
}

// ============================================================================
// CORE RUN HELPER  (returns L2; -1.0 if NaN/Inf encountered)
// ============================================================================

enum class Operator { TRT, REGULARIZED };

float run_couette_poiseuille(float tau, int max_steps,
                              Operator op, float fx_scale = 1.0f) {
    const float nu = (tau - 0.5f) / 3.0f;
    const float fx = fx_scale * (-6.0f * U_TOP * nu / (H_LU * H_LU));

    FluidLBM fluid(NX, NY, NZ,
                   nu, 1.0f,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    if (op == Operator::TRT) {
        fluid.setTRT(3.0f / 16.0f);
    } else {
        fluid.setRegularized(true);
    }

    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(lbm::core::Streaming::BOUNDARY_Y_MAX, U_TOP, 0.0f, 0.0f);

    for (int step = 0; step < max_steps; ++step) {
        fluid.collisionTRT(fx, 0.0f, 0.0f);
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
    }

    const int N = NX * NY * NZ;
    std::vector<float> ux(N);
    cudaMemcpy(ux.data(), fluid.getVelocityX(),
               N * sizeof(float), cudaMemcpyDeviceToHost);

    for (float v : ux) {
        if (!std::isfinite(v)) return -1.0f;
    }

    return compute_l2(ux, U_TOP, fx, nu);
}

}  // namespace

// ============================================================================
// PARAMETERIZED FIXTURE
// ============================================================================

struct LowTauCase {
    float tau;
    int   max_steps;
    enum class Operator { TRT, REGULARIZED } op;
    const char* name;
    float l2_threshold;
};

class LowTauCollisionSweep : public ::testing::TestWithParam<LowTauCase> {};

TEST_P(LowTauCollisionSweep, CouettePoiseuille) {
    const LowTauCase& c = GetParam();

    const Operator op = (c.op == LowTauCase::Operator::TRT)
                            ? Operator::TRT
                            : Operator::REGULARIZED;

    std::cout << "\n[" << c.name << "]"
              << "  tau=" << c.tau
              << "  nu=" << std::scientific << std::setprecision(3)
              << (c.tau - 0.5f) / 3.0f
              << "  steps=" << c.max_steps << "\n";

    const float l2 = run_couette_poiseuille(c.tau, c.max_steps, op);

    // NaN / Inf check
    ASSERT_GE(l2, 0.0f) << "NaN or Inf in final velocity field — solver diverged";

    std::cout << "  L2 = " << std::fixed << std::setprecision(4)
              << l2 * 100.0f << " %  (threshold "
              << c.l2_threshold * 100.0f << " %)\n";

    ASSERT_LT(l2, c.l2_threshold)
        << "L2 = " << l2 * 100.0f << " % exceeds "
        << c.l2_threshold * 100.0f << " % for operator "
        << c.name << " at tau=" << c.tau;
}

// ============================================================================
// INSTANTIATION: {TRT, REGULARIZED} × {0.51, 0.53, 0.54, 0.55, 0.60, 0.70}
// ============================================================================

// Step counts are conservative enough for steady state but still finish
// within ~10 min total on an RTX 3050 (serial execution).
// τ=0.51 is the hardest case; convergence is slowest due to lowest ν.

static const LowTauCase kCases[] = {
    // --- TRT ---
    {0.51f, 80000, LowTauCase::Operator::TRT,         "TRT_051", 0.005f},
    {0.53f, 60000, LowTauCase::Operator::TRT,         "TRT_053", 0.005f},
    {0.54f, 60000, LowTauCase::Operator::TRT,         "TRT_054", 0.005f},
    {0.55f, 60000, LowTauCase::Operator::TRT,         "TRT_055", 0.005f},
    {0.60f, 40000, LowTauCase::Operator::TRT,         "TRT_060", 0.005f},
    {0.70f, 30000, LowTauCase::Operator::TRT,         "TRT_070", 0.005f},
    // --- Regularized ---
    {0.51f, 80000, LowTauCase::Operator::REGULARIZED, "REG_051", 0.005f},
    {0.53f, 60000, LowTauCase::Operator::REGULARIZED, "REG_053", 0.005f},
    {0.54f, 60000, LowTauCase::Operator::REGULARIZED, "REG_054", 0.005f},
    {0.55f, 60000, LowTauCase::Operator::REGULARIZED, "REG_055", 0.005f},
    {0.60f, 40000, LowTauCase::Operator::REGULARIZED, "REG_060", 0.005f},
    {0.70f, 30000, LowTauCase::Operator::REGULARIZED, "REG_070", 0.005f},
};

INSTANTIATE_TEST_SUITE_P(
    AllOperators,
    LowTauCollisionSweep,
    ::testing::ValuesIn(kCases),
    [](const ::testing::TestParamInfo<LowTauCase>& info) {
        return std::string(info.param.name);
    });

// ============================================================================
// MUTATION DISCRIMINATOR TESTS
//
// Goal: verify that a body-force miscalculation is detectable via L2.
// These protect against the class of bugs that test_trt_omega_minus_arithmetic
// was created to catch: a silent change to the force channel that shifts the
// velocity profile without causing NaN.
//
// WHY PURE POISEUILLE (no moving wall)
// -------------------------------------
// In the Couette-Poiseuille sweep the wall shear (U_TOP) and the body force
// are comparable in magnitude.  A 5% drift on f_x changes only the parabolic
// Poiseuille component, which is O(U_TOP × 5%) — giving L2 change of ~0.01%,
// too small to reliably exceed 2× the ~0.16% baseline across all hardware.
//
// For a PURE Poiseuille channel (no moving wall, wall velocity = 0), the
// analytical profile is entirely determined by f_x:
//   u(η) = (f_x H²)/(2ν) η(1-η)
// A 20% drift on f_x shifts every node by 20%, so L2_drifted ≈ 20%,
// easily > 2 × a sub-0.5% baseline on any hardware.
//
// Configuration:  same domain (8×20×8), τ=0.54, 60k steps.
//                 f_x chosen for Ma ≈ 0.02 at the peak.
// ============================================================================

namespace {

// Run a pure Poiseuille channel (no moving wall) and return the
// range-normalised L2 error vs the analytical parabola.
// fx_scale multiplies the body force passed to the collision kernel while
// the analytical reference always uses fx_scale=1.  This simulates a
// multiplicative drift in the force path.
float run_poiseuille_mutation(float tau, int max_steps,
                               Operator op, float fx_scale) {
    const float nu    = (tau - 0.5f) / 3.0f;
    // Peak velocity target ≈ 0.012 LU → Ma ≈ 0.021  (sub-sonic, clean signal)
    const float u_pk  = 0.012f;
    const float fx    = 8.0f * nu * u_pk / (H_LU * H_LU);   // exact Poiseuille

    FluidLBM fluid(NX, NY, NZ,
                   nu, 1.0f,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    if (op == Operator::TRT) {
        fluid.setTRT(3.0f / 16.0f);
    } else {
        fluid.setRegularized(true);
    }

    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);
    // No moving wall — pure Poiseuille.

    const float fx_run = fx_scale * fx;
    for (int step = 0; step < max_steps; ++step) {
        fluid.collisionTRT(fx_run, 0.0f, 0.0f);
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();
    }

    const int N = NX * NY * NZ;
    std::vector<float> ux(N);
    cudaMemcpy(ux.data(), fluid.getVelocityX(),
               N * sizeof(float), cudaMemcpyDeviceToHost);

    for (float v : ux) {
        if (!std::isfinite(v)) return -1.0f;
    }

    // Compute L2 vs the CORRECT parabola (fx_scale=1 always as reference).
    // u_analytical uses fx, not fx_run.
    float  u_min = 0.0f, u_max = 0.0f;
    double sum_sq = 0.0;
    int    count  = 0;
    for (int j = 0; j < NY; ++j) {
        const float y   = static_cast<float>(j);
        const float u_a = analytical_cp(y, 0.0f, fx, nu, H_LU);
        u_min = std::min(u_min, u_a);
        u_max = std::max(u_max, u_a);

        double avg = 0.0;
        for (int k = 0; k < NZ; ++k) {
            for (int i = 0; i < NX; ++i) {
                avg += ux[i + NX * (j + NY * k)];
            }
        }
        avg /= (NX * NZ);
        const double err = avg - u_a;
        sum_sq += err * err;
        ++count;
    }

    const float u_range = u_max - u_min;
    if (u_range < 1e-20f) return 0.0f;
    return static_cast<float>(std::sqrt(sum_sq / count) / u_range);
}

}  // namespace

TEST(LowTauCollision, MutationCatchesEDMForceShift_TRT) {
    // 20% body-force drift: simulates a corrupted force path (e.g., missing
    // factor, unit error, wrong Guo correction coefficient).
    constexpr float tau       = 0.54f;
    constexpr int   steps     = 60000;
    constexpr float drift     = 1.20f;    // 20% multiplicative error

    const float l2_baseline = run_poiseuille_mutation(tau, steps, Operator::TRT, 1.00f);
    const float l2_drifted  = run_poiseuille_mutation(tau, steps, Operator::TRT, drift);

    ASSERT_GE(l2_baseline, 0.0f) << "Baseline run produced NaN";
    ASSERT_GE(l2_drifted,  0.0f) << "Drifted run produced NaN";

    std::cout << "\n[MutationTest TRT tau=0.54 drift=" << drift << "x]"
              << "  baseline L2=" << std::fixed << std::setprecision(4)
              << l2_baseline * 100.0f << " %"
              << "  drifted L2=" << l2_drifted * 100.0f << " %\n";

    EXPECT_GT(l2_drifted, 2.0f * l2_baseline + 0.05f)
        << "20% body-force drift not detected: L2_drifted=" << l2_drifted * 100.0f
        << " % should be clearly above baseline=" << l2_baseline * 100.0f << " %";
}

TEST(LowTauCollision, MutationCatchesEDMForceShift_Regularized) {
    constexpr float tau       = 0.54f;
    constexpr int   steps     = 60000;
    constexpr float drift     = 1.20f;

    const float l2_baseline = run_poiseuille_mutation(tau, steps, Operator::REGULARIZED, 1.00f);
    const float l2_drifted  = run_poiseuille_mutation(tau, steps, Operator::REGULARIZED, drift);

    ASSERT_GE(l2_baseline, 0.0f) << "Baseline run produced NaN";
    ASSERT_GE(l2_drifted,  0.0f) << "Drifted run produced NaN";

    std::cout << "\n[MutationTest Regularized tau=0.54 drift=" << drift << "x]"
              << "  baseline L2=" << std::fixed << std::setprecision(4)
              << l2_baseline * 100.0f << " %"
              << "  drifted L2=" << l2_drifted * 100.0f << " %\n";

    EXPECT_GT(l2_drifted, 2.0f * l2_baseline + 0.05f)
        << "20% body-force drift not detected: L2_drifted=" << l2_drifted * 100.0f
        << " % should be clearly above baseline=" << l2_baseline * 100.0f << " %";
}
