/**
 * @file test_cumulant_stages.cu
 * @brief Per-stage unit tests for D3Q27 Cumulant LBM
 *
 * Tests:
 *   Stage 2a: raw moments of f_eq match analytical Maxwell-Boltzmann moments
 *   Stage 2b: central moments at u_macro recover ρ at order 0, 0 at order 1
 *   Stage 2c: cumulants of f_eq are zero at orders ≥ 3 (Maxwell is Gaussian)
 *             and ρ·cs² at order-2 diagonal, 0 at order-2 off-diagonal
 *   Stage 2e: f_eq → m → κ → K → κ → m → f recovers f_eq (round-trip)
 *   Stage 2f: collision with ω=0 (no relaxation) preserves f_eq exactly
 */

#include "physics/cumulant/cumulant_d3q27.h"
#include "core/lattice_d3q27.h"
#include <cstdio>
#include <cmath>

using namespace lbm::core;
using namespace lbm::physics::cumulant;

#define TIDX(p, q, r) ((p) + 3 * (q) + 9 * (r))

static int passed = 0, failed = 0;

#define EXPECT(cond, msg, ...) do { \
    if (cond) { printf("PASS: " msg "\n", ##__VA_ARGS__); ++passed; } \
    else      { printf("FAIL: " msg "\n", ##__VA_ARGS__); ++failed; } \
} while(0)

#define EXPECT_NEAR(a, b, tol, msg, ...) do { \
    if (std::abs((a) - (b)) < (tol)) { \
        printf("PASS: " msg "  (a=%.6e, b=%.6e, |a-b|=%.2e)\n", \
               ##__VA_ARGS__, (double)(a), (double)(b), (double)std::abs((a)-(b))); \
        ++passed; \
    } else { \
        printf("FAIL: " msg "  (a=%.6e, b=%.6e, |a-b|=%.2e, tol=%.2e)\n", \
               ##__VA_ARGS__, (double)(a), (double)(b), (double)std::abs((a)-(b)), (double)(tol)); \
        ++failed; \
    } \
} while(0)

// Helper: run all stage tests at a given (rho, ux, uy, uz). At u=0 the
// truncated 2nd-order Hermite equilibrium IS exact MB, so cumulants ≥ 3 are
// exactly 0. At u≠0, cumulants ≥ 3 are O(u³) — small but nonzero.
static void runStageTestsAt(const char* label, float rho, float ux, float uy, float uz);

int main() {
    printf("=== D3Q27 Cumulant per-stage tests ===\n\n");

    runStageTestsAt("(rho=1, u=0)        — exact MB", 1.0f, 0.0f, 0.0f, 0.0f);
    runStageTestsAt("(rho=1, u=(.05,.03,-.02)) — moderate u, K_3+ should be O(u³)",
                    1.0f, 0.05f, 0.03f, -0.02f);

    printf("\n=== Final summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

static void runStageTestsAt(const char* label, float rho, float ux, float uy, float uz) {
    printf("\n>>> Run: %s\n\n", label);
    const float cs2 = 1.0f / 3.0f;
    // Tolerance for "K_3+ should be zero": at u=0 → exact zero (1e-5);
    // at moderate u → O(u³) (5e-4 = tol * (0.05)³)
    const float u_mag = std::sqrt(ux*ux + uy*uy + uz*uz);
    const float K3_tol = (u_mag < 1e-6f) ? 1e-5f : 5.0f * u_mag * u_mag * u_mag;
    const float K6_tol = (u_mag < 1e-6f) ? 1e-5f : 1e-4f;

    // Build f_eq (compressible 2nd-order Hermite)
    float f_eq[27];
    for (int q = 0; q < 27; ++q) {
        f_eq[q] = D3Q27::computeEquilibrium(q, rho, ux, uy, uz);
    }

    // ------------------------------------------------------------------
    // Stage 2a: raw moments of f_eq
    // Analytical: m_000 = ρ, m_100 = ρ·ux, m_010 = ρ·uy, m_001 = ρ·uz
    // m_200 = ρ·(ux² + cs²), m_020 = ρ·(uy² + cs²), m_002 = ρ·(uz² + cs²)
    // m_110 = ρ·ux·uy, m_101 = ρ·ux·uz, m_011 = ρ·uy·uz
    // ------------------------------------------------------------------
    printf("--- Stage 2a: raw moments of f_eq ---\n");
    float m[27];
    computeRawMoments27(f_eq, m);
    EXPECT_NEAR(m[TIDX(0,0,0)], rho, 1e-5f, "m_000 = ρ");
    EXPECT_NEAR(m[TIDX(1,0,0)], rho * ux, 1e-5f, "m_100 = ρ·ux");
    EXPECT_NEAR(m[TIDX(0,1,0)], rho * uy, 1e-5f, "m_010 = ρ·uy");
    EXPECT_NEAR(m[TIDX(0,0,1)], rho * uz, 1e-5f, "m_001 = ρ·uz");
    EXPECT_NEAR(m[TIDX(2,0,0)], rho * (ux*ux + cs2), 1e-5f, "m_200 = ρ·(ux²+cs²)");
    EXPECT_NEAR(m[TIDX(0,2,0)], rho * (uy*uy + cs2), 1e-5f, "m_020 = ρ·(uy²+cs²)");
    EXPECT_NEAR(m[TIDX(0,0,2)], rho * (uz*uz + cs2), 1e-5f, "m_002 = ρ·(uz²+cs²)");
    EXPECT_NEAR(m[TIDX(1,1,0)], rho * ux * uy, 1e-5f, "m_110 = ρ·ux·uy");
    EXPECT_NEAR(m[TIDX(1,0,1)], rho * ux * uz, 1e-5f, "m_101 = ρ·ux·uz");
    EXPECT_NEAR(m[TIDX(0,1,1)], rho * uy * uz, 1e-5f, "m_011 = ρ·uy·uz");
    printf("\n");

    // ------------------------------------------------------------------
    // Stage 2b: central moments at u (should give κ_000 = ρ, κ_1xx = 0,
    //           κ_2nd_diag = ρ·cs², κ_2nd_offdiag = 0 for f_eq)
    // ------------------------------------------------------------------
    printf("--- Stage 2b: central moments of f_eq ---\n");
    float kappa[27];
    shiftToCentralMoments27(m, ux, uy, uz, kappa);
    EXPECT_NEAR(kappa[TIDX(0,0,0)], rho, 1e-5f, "κ_000 = ρ");
    EXPECT_NEAR(kappa[TIDX(1,0,0)], 0.0f, 1e-5f, "κ_100 = 0 (in central frame)");
    EXPECT_NEAR(kappa[TIDX(0,1,0)], 0.0f, 1e-5f, "κ_010 = 0");
    EXPECT_NEAR(kappa[TIDX(0,0,1)], 0.0f, 1e-5f, "κ_001 = 0");
    EXPECT_NEAR(kappa[TIDX(2,0,0)], rho * cs2, 1e-5f, "κ_200 = ρ·cs²");
    EXPECT_NEAR(kappa[TIDX(0,2,0)], rho * cs2, 1e-5f, "κ_020 = ρ·cs²");
    EXPECT_NEAR(kappa[TIDX(0,0,2)], rho * cs2, 1e-5f, "κ_002 = ρ·cs²");
    EXPECT_NEAR(kappa[TIDX(1,1,0)], 0.0f, 1e-5f, "κ_110 = 0");
    EXPECT_NEAR(kappa[TIDX(1,0,1)], 0.0f, 1e-5f, "κ_101 = 0");
    EXPECT_NEAR(kappa[TIDX(0,1,1)], 0.0f, 1e-5f, "κ_011 = 0");
    printf("\n");

    // ------------------------------------------------------------------
    // Stage 2c: cumulants of f_eq
    // Maxwell-Boltzmann is Gaussian → cumulants beyond order 2 are EXACTLY 0.
    // Order 0: K_000 = ρ
    // Order 2 diagonal: K = ρ·cs²
    // Order 2 off-diagonal: K = 0
    // Order 3+: K = 0
    // ------------------------------------------------------------------
    printf("--- Stage 2c: cumulants of f_eq (MB → 0 at order ≥ 3) ---\n");
    float K[27];
    centralToCumulants27(kappa, rho, K);
    EXPECT_NEAR(K[TIDX(0,0,0)], rho, 1e-5f, "K_000 = ρ");
    EXPECT_NEAR(K[TIDX(2,0,0)], rho * cs2, 1e-5f, "K_200 = ρ·cs²");
    EXPECT_NEAR(K[TIDX(0,2,0)], rho * cs2, 1e-5f, "K_020 = ρ·cs²");
    EXPECT_NEAR(K[TIDX(0,0,2)], rho * cs2, 1e-5f, "K_002 = ρ·cs²");
    EXPECT_NEAR(K[TIDX(1,1,0)], 0.0f, 1e-5f, "K_110 = 0");
    EXPECT_NEAR(K[TIDX(1,0,1)], 0.0f, 1e-5f, "K_101 = 0");
    EXPECT_NEAR(K[TIDX(0,1,1)], 0.0f, 1e-5f, "K_011 = 0");
    EXPECT_NEAR(K[TIDX(1,1,1)], 0.0f, K3_tol, "K_111 = 0 (3rd, O(u³) tol)");
    EXPECT_NEAR(K[TIDX(2,1,0)], 0.0f, K3_tol, "K_210 = 0 (3rd)");
    EXPECT_NEAR(K[TIDX(2,2,0)], 0.0f, K3_tol, "K_220 = 0 (4th)");
    EXPECT_NEAR(K[TIDX(2,0,2)], 0.0f, K3_tol, "K_202 = 0 (4th)");
    EXPECT_NEAR(K[TIDX(0,2,2)], 0.0f, K3_tol, "K_022 = 0 (4th)");
    EXPECT_NEAR(K[TIDX(2,1,1)], 0.0f, K3_tol, "K_211 = 0 (4th)");
    EXPECT_NEAR(K[TIDX(1,2,1)], 0.0f, K3_tol, "K_121 = 0 (4th)");
    EXPECT_NEAR(K[TIDX(1,1,2)], 0.0f, K3_tol, "K_112 = 0 (4th)");
    EXPECT_NEAR(K[TIDX(2,2,1)], 0.0f, K3_tol, "K_221 = 0 (5th)");
    EXPECT_NEAR(K[TIDX(2,1,2)], 0.0f, K3_tol, "K_212 = 0 (5th)");
    EXPECT_NEAR(K[TIDX(1,2,2)], 0.0f, K3_tol, "K_122 = 0 (5th)");
    EXPECT_NEAR(K[TIDX(2,2,2)], 0.0f, K6_tol, "K_222 = 0 (6th)");
    printf("\n");

    // ------------------------------------------------------------------
    // Stage 2e: round-trip identity (no relaxation: K stays as input)
    // f_eq → m → κ → K → κ → m → f, expect f ≈ f_eq
    // ------------------------------------------------------------------
    printf("--- Stage 2e: round-trip identity (no relaxation) ---\n");
    float kappa2[27], m2[27], f_round[27];
    cumulantsToCentralMoments27(K, rho, kappa2);
    centralToRawMoments27(kappa2, ux, uy, uz, m2);
    rawMomentsToPDFs27(m2, f_round);

    float max_diff = 0.0f;
    for (int q = 0; q < 27; ++q) {
        float d = std::abs(f_round[q] - f_eq[q]);
        if (d > max_diff) max_diff = d;
    }
    EXPECT_NEAR(max_diff, 0.0f, 1e-5f,
                "max |f_round - f_eq| over 27 directions");

    printf("\n");

    // ------------------------------------------------------------------
    // Stage 2f: collision with ω=0 (no relaxation) is identity on f_eq
    // (Cumulant collision must preserve equilibrium when no relaxation.)
    // ------------------------------------------------------------------
    printf("--- Stage 2f: collision identity for ω=0 ---\n");
    float K_after[27];
    for (int i = 0; i < 27; ++i) K_after[i] = K[i];
    relaxCumulants27(K_after, rho, /*ω*/ 0, 0, 0, 0, 0, 0);
    bool all_unchanged = true;
    for (int i = 0; i < 27; ++i) {
        if (std::abs(K_after[i] - K[i]) > 1e-6f) { all_unchanged = false; break; }
    }
    EXPECT(all_unchanged, "K_after = K when all ω = 0");

    // Better test: full collision (ω=1, full relaxation) on f_eq.
    // Mass and momentum MUST be conserved exactly. K_3+ go to 0 (which
    // ACTUALLY removes the spurious O(u³) error in truncated equilibrium —
    // this is the value-add of Cumulant LBM).
    for (int i = 0; i < 27; ++i) K_after[i] = K[i];
    relaxCumulants27(K_after, rho, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    EXPECT_NEAR(K_after[TIDX(0,0,0)], rho, 1e-5f, "ω=1 conserves K_000 = ρ");
    EXPECT_NEAR(K_after[TIDX(2,0,0)], rho * cs2, 1e-5f, "ω=1 sets K_200 → K_200_eq = ρ·cs²");

    // Reconstruct f_post and verify mass/momentum conservation
    float kappa3[27], m3[27], f_post[27];
    cumulantsToCentralMoments27(K_after, rho, kappa3);
    centralToRawMoments27(kappa3, ux, uy, uz, m3);
    rawMomentsToPDFs27(m3, f_post);
    float rho_post = 0.0f, mx_post = 0.0f, my_post = 0.0f, mz_post = 0.0f;
    for (int q = 0; q < 27; ++q) {
        rho_post += f_post[q];
        mx_post += D3Q27::h_ex[q] * f_post[q];
        my_post += D3Q27::h_ey[q] * f_post[q];
        mz_post += D3Q27::h_ez[q] * f_post[q];
    }
    EXPECT_NEAR(rho_post, rho, 1e-4f, "post-collision: ρ conserved");
    EXPECT_NEAR(mx_post, rho * ux, 1e-4f, "post-collision: ρ·ux conserved");
    EXPECT_NEAR(my_post, rho * uy, 1e-4f, "post-collision: ρ·uy conserved");
    EXPECT_NEAR(mz_post, rho * uz, 1e-4f, "post-collision: ρ·uz conserved");
}
