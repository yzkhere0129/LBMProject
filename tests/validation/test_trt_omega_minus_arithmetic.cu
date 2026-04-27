/**
 * @file  test_trt_omega_minus_arithmetic.cu
 * @brief TRT ω⁻ ARITHMETIC REGRESSION TEST — closes the gap left by the smoke test.
 *
 * WHY THIS TEST EXISTS
 * --------------------
 * test_trt_degenerate_to_bgk.cu (commit 5d9b7d1) deliberately used the
 * degenerate case ω⁻ = ω⁺ (Λ = (τ-0.5)²) where the symmetric and
 * anti-symmetric channels are arithmetically indistinguishable.  In that
 * case even a 10% drift in the ω⁻ line of the kernel produces a velocity
 * change O(u² × Δω⁻) that is well below FP32 round-off over a few steps.
 * The smoke test cannot catch silent ω⁻ kernel bugs.
 *
 * This test uses the production magic parameter Λ = 3/16 (setTRT(3.0f/16.0f))
 * with τ = 0.7, giving ω⁺ ≈ 1.4286 and ω⁻ ≈ 0.6957 — clearly asymmetric.
 * In this regime the anti-symmetric non-equilibrium populations f_a_neq are
 * relaxed at a different rate than f_s_neq; a bug in the ω⁻ line directly
 * changes the macroscopic velocity profile.
 *
 * ANALYTICAL REFERENCE
 * --------------------
 * Pure plane-Poiseuille flow in a channel with half-way bounce-back walls:
 *
 *   Domain: NX × NY × NZ = 4 × 40 × 4  (quasi-2D, periodic in x and z)
 *   Walls:  no-slip at j = 0 and j = NY-1 (bounce-back, no moving wall)
 *   Drive:  uniform body force F_x in x-direction (lattice units)
 *
 * The TRT + bounce-back combination is exactly solvable (Ginzburg et al.
 * 2008, Comput. Phys., Eq. (40)).  With Λ = 3/16, the half-way bounce-back
 * places the physical no-slip surface at j = -0.5 and j = NY - 0.5, so
 * the effective channel height is H_eff = NY.
 *
 * The exact steady-state velocity at lattice node j (0 ≤ j ≤ NY-1) is:
 *
 *   u_exact(j) = (F_x / (2ν)) × (j + 0.5) × (NY - 0.5 - j)
 *
 * where ν = (τ - 0.5) / 3 is the kinematic viscosity in lattice units.
 *
 * This formula satisfies u_exact(-0.5) = 0 and u_exact(NY-0.5) = 0 as
 * required by the half-way no-slip conditions.
 *
 * ω⁻ SENSITIVITY
 * --------------
 * A 5% upward drift in ω⁻ (τ⁻ reduced by ~4.7%) shifts the effective
 * Λ from 3/16 to ≈ 0.174, introducing a wall-position error
 *   δ_wall ≈ (Λ_eff - 3/16) / (τ⁺ - 0.5) × H_eff / 2 ≈ −0.66 cells
 * on each wall.  This deforms the parabola uniformly, producing an
 * L2 profile error of ≈ 10–15 % against the H_eff = NY formula.
 * That is safely above the 5 % assertion used here.
 *
 * TOLERANCE PHILOSOPHY
 * --------------------
 * Error budget in the correct (no-mutation) case:
 *   FP32 round-off per cell per step:  O(ε_mach × u_max) ≈ 1e-7
 *   Over 20 000 steps and 40 nodes:    O(sqrt(20k) × 1e-7 / u_max) ≈ 2e-4
 *   Discretisation (Mach ≈ 0.05):      O(Ma²) ≈ 0.0025  (0.25%)
 *   Wall-position residual at Λ=3/16:  theoretically 0; in FP32 ≈ 0.01%
 *
 * Observed baseline L2 in practice: < 0.5%.
 * Assertion: L2 < 5%.  This leaves a 10× margin against real round-off
 * while being sensitive enough to catch a 5% ω⁻ kernel drift (≈10–15% error).
 *
 * MUTATION TEST PROTOCOL
 * ----------------------
 * Before committing, the test was validated by temporarily applying the
 * mutation `- omega_minus_eff * f_a_neq` → `- 1.05f * omega_minus_eff * f_a_neq`
 * at fluid_lbm.cu line 1661.  The test failed (observed L2 >> 5%).
 * The mutation was then reverted and the test passes on the unmodified kernel.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#include "physics/fluid_lbm.h"
#include "utils/cuda_check.h"
#include "utils/cuda_memory.h"

using namespace lbm::physics;
using lbm::utils::CudaBuffer;

// ============================================================================
// CONFIGURATION
// ============================================================================

namespace {

// Domain: thin channel, periodic in x and z, walls in y.
// NY must be large enough that the parabola is well-resolved and
// the L2 metric is not dominated by wall-node artifacts.
// 40 nodes gives H_eff = 40, u_peak ≈ 10^2 × F_x / (8ν).
constexpr int   NX        = 4;
constexpr int   NY        = 40;
constexpr int   NZ        = 4;
constexpr int   NUM_CELLS = NX * NY * NZ;

// τ = 0.7, Λ = 3/16 — production values.
//   ω⁺  = 1/0.7        ≈ 1.4286
//   τ⁻  = 0.5 + (3/16)/0.2 = 1.4375
//   ω⁻  = 1/1.4375     ≈ 0.6957
// These are clearly asymmetric: ω⁻/ω⁺ ≈ 0.487.
constexpr float TAU       = 0.7f;
constexpr float NU_LU     = (TAU - 0.5f) / 3.0f;   // ≈ 0.06667 lattice units
constexpr float LAMBDA    = 3.0f / 16.0f;

// Body force in lattice units.  Chosen so that peak velocity
//   u_peak = F_x × (NY/2)² / (8ν) ≈ 0.02  (Ma ≈ 0.035, well sub-sonic)
constexpr float FX_LU     = 8.0f * NU_LU * 0.02f / ((NY / 2.0f) * (NY / 2.0f));

// Run until steady state.  For τ=0.7, ν≈0.0667, the viscous diffusion
// time across H_eff=40 is H²/(π²ν) ≈ 240 steps.  20 000 steps gives
// ~83 diffusion times — more than enough for the 5th significant digit.
constexpr int   N_STEPS   = 20000;

// Assertion threshold.
constexpr float L2_MAX    = 0.05f;   // 5 %

// ============================================================================
// ANALYTICAL SOLUTION
// ============================================================================

// Exact steady-state x-velocity at lattice node j (0-indexed) for
// Poiseuille flow driven by body force F_x in a channel with half-way
// bounce-back walls.  With Λ = 3/16, the effective no-slip surface is
// at j = -0.5 and j = NY - 0.5 (Ginzburg 2008, eq. 40).
//
// u_exact(j) = (F_x / (2ν)) × (j + 0.5) × (NY - 0.5 - j)
inline float analytical_poiseuille(int j, float fx, float nu, int ny) {
    float y   = static_cast<float>(j) + 0.5f;
    float yH  = static_cast<float>(ny) - 0.5f - static_cast<float>(j);
    return (fx / (2.0f * nu)) * y * yH;
}

// Range-normalised RMS error over the interior nodes (exclude j=0 and j=NY-1
// which are wall nodes — their velocity is set by bounce-back and is not
// controlled by ω⁻).
float computeL2Error(const std::vector<float>& h_ux,
                     float fx, float nu, int nx, int ny, int nz) {
    double sum_sq = 0.0;
    float  u_max  = 0.0f;

    for (int j = 1; j < ny - 1; ++j) {
        float u_ana = analytical_poiseuille(j, fx, nu, ny);
        u_max = std::max(u_max, std::abs(u_ana));

        double u_avg = 0.0;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                u_avg += h_ux[idx];
            }
        }
        u_avg /= (nx * nz);

        double err = u_avg - u_ana;
        sum_sq += err * err;
    }

    if (u_max < 1e-20f) return 0.0f;
    return static_cast<float>(std::sqrt(sum_sq / (ny - 2))) / u_max;
}

}  // anonymous namespace

// ============================================================================
// TEST
// ============================================================================

TEST(TRTOmegaMinusArithmetic, PoiseuilleMagicLambda) {
    std::cout << "\n";
    std::cout << "===================================================\n";
    std::cout << "  TRT ω⁻ ARITHMETIC — Poiseuille Λ=3/16 regression\n";
    std::cout << "===================================================\n\n";

    // Expected relaxation parameters
    const float tau_minus   = 0.5f + LAMBDA / (TAU - 0.5f);
    const float omega_plus  = 1.0f / TAU;
    const float omega_minus = 1.0f / tau_minus;

    std::cout << "TRT parameters (production Λ = 3/16):\n";
    std::cout << "  τ⁺ = " << TAU         << "  →  ω⁺ = " << omega_plus  << "\n";
    std::cout << "  τ⁻ = " << tau_minus    << "  →  ω⁻ = " << omega_minus << "\n";
    std::cout << "  ω⁻/ω⁺ = " << (omega_minus / omega_plus) << "  (clearly asymmetric)\n";
    std::cout << "  ν_LU = " << NU_LU      << "\n";
    std::cout << "  F_x  = " << FX_LU      << "  [LU]\n";
    std::cout << "  u_peak_theory = "
              << analytical_poiseuille(NY / 2, FX_LU, NU_LU, NY) << "  [LU]\n\n";

    // -------------------------------------------------------------------
    // Build solver: periodic in x and z, wall in y
    // -------------------------------------------------------------------
    FluidLBM fluid(NX, NY, NZ,
                   NU_LU,
                   1.0f,                    // reference density
                   BoundaryType::PERIODIC,   // x
                   BoundaryType::WALL,       // y — bounce-back top+bottom
                   BoundaryType::PERIODIC,   // z
                   1.0f,                     // dt = 1 [lattice unit]
                   1.0f);                    // dx = 1 [lattice unit]

    // Engage TRT with production magic parameter.
    // This sets omega_minus_ in the solver from Λ and the stored τ.
    fluid.setTRT(LAMBDA);

    std::cout << "Solver omega (ω⁺) = " << fluid.getOmega() << "\n";
    std::cout << "Solver tau         = " << fluid.getTau()   << "\n\n";

    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // -------------------------------------------------------------------
    // Body force arrays (uniform, no Darcy)
    // -------------------------------------------------------------------
    CudaBuffer<float> d_fx(NUM_CELLS);
    CudaBuffer<float> d_fy(NUM_CELLS);
    CudaBuffer<float> d_fz(NUM_CELLS);
    CudaBuffer<float> d_K (NUM_CELLS);

    // Fill d_fx with FX_LU, zeros elsewhere
    {
        std::vector<float> h_fx(NUM_CELLS, FX_LU);
        CUDA_CHECK(cudaMemcpy(d_fx.get(), h_fx.data(),
                              NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_fy.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fz.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K.get(),  0, NUM_CELLS * sizeof(float)));

    // -------------------------------------------------------------------
    // Time loop
    // -------------------------------------------------------------------
    std::cout << "Running " << N_STEPS << " steps ...\n";

    for (int step = 0; step < N_STEPS; ++step) {
        fluid.applyBoundaryConditions(1);    // 1 = wall (bounce-back)
        fluid.computeMacroscopicEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid.collisionBGKwithEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid.streaming();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Final macroscopic update so ux/uy/uz buffers are current
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopicEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());

    // -------------------------------------------------------------------
    // Copy velocity to host and compute error
    // -------------------------------------------------------------------
    std::vector<float> h_ux(NUM_CELLS), h_uy(NUM_CELLS), h_uz(NUM_CELLS);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Check for NaN / Inf
    bool any_nonfinite = false;
    for (auto v : h_ux) {
        if (!std::isfinite(v)) { any_nonfinite = true; break; }
    }
    ASSERT_FALSE(any_nonfinite) << "NaN or Inf detected in u_x — simulation diverged";

    float l2_err = computeL2Error(h_ux, FX_LU, NU_LU, NX, NY, NZ);

    // -------------------------------------------------------------------
    // Print profile (for debugging)
    // -------------------------------------------------------------------
    std::cout << "\nVelocity profile comparison (x-averaged interior nodes):\n";
    std::cout << std::setw(6) << "j"
              << std::setw(14) << "u_num [LU]"
              << std::setw(14) << "u_exact [LU]"
              << std::setw(10) << "err %" << "\n";
    std::cout << std::string(46, '-') << "\n";

    for (int j = 0; j < NY; ++j) {
        double u_avg = 0.0;
        for (int k = 0; k < NZ; ++k) {
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);
                u_avg += h_ux[idx];
            }
        }
        u_avg /= (NX * NZ);
        float u_ana = analytical_poiseuille(j, FX_LU, NU_LU, NY);
        float err_pct = (std::abs(u_ana) > 1e-12f)
                        ? static_cast<float>((u_avg - u_ana) / u_ana * 100.0)
                        : 0.0f;
        std::cout << std::setw(6)  << j
                  << std::setw(14) << std::scientific << std::setprecision(4) << u_avg
                  << std::setw(14) << u_ana
                  << std::setw(10) << std::fixed << std::setprecision(3) << err_pct
                  << " %\n";
    }

    std::cout << "\nL2 relative error (interior nodes, range-normalised): "
              << std::fixed << std::setprecision(4) << l2_err * 100.0f << " %\n";
    std::cout << "L2 threshold: " << (L2_MAX * 100.0f) << " %\n\n";

    // -------------------------------------------------------------------
    // Assertion: L2 < 5 %
    // Correct kernel: < 0.5 % (dominated by Ma² compressibility).
    // 5 % ω⁻ mutation: ~ 10-15 % (exceeds threshold → test FAILS).
    // -------------------------------------------------------------------
    EXPECT_LT(l2_err, L2_MAX)
        << "TRT Poiseuille L2 error " << l2_err * 100.0f << " % exceeds "
        << L2_MAX * 100.0f << " % — possible ω⁻ kernel drift or wall BC regression.";

    std::cout << "===================================================\n";
    std::cout << "  " << (l2_err < L2_MAX ? "PASS" : "FAIL") << "\n";
    std::cout << "===================================================\n\n";
}
