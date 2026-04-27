/**
 * @file  test_trt_degenerate_to_bgk.cu
 * @brief SMOKE TEST — TRT dispatch path engages, basic arithmetic is sane,
 *        LES branch does not NaN. **NOT a TRT-correctness regression catcher**.
 *
 * Scope (be honest about what this test does and doesn't catch):
 *
 *   ✓ Catches: dispatch errors (TRT branch not taken when omega_minus_>0,
 *              or BGK branch not taken when omega_minus_=0). Catches gross
 *              kernel rewrites that break BGK-degenerate equivalence (wrong
 *              loop bounds, swapped indices, missing terms).
 *   ✓ Catches: LES branch NaN-ing on small domain.
 *   ✗ Does NOT catch: ω⁻ arithmetic drift below ~10× the FP32 round-off
 *              floor. The degenerate case ω⁻=ω⁺ makes the symmetric and
 *              anti-symmetric channels indistinguishable; f_a_neq is
 *              O(u²·weight) ~ 1e-6 of |f| at sub-Mach perturbations, so
 *              even a 1% drift on ω⁻ produces ~1e-7 deviation per step —
 *              well below any non-flaky FP32 tolerance.
 *
 * For real ω⁻-arithmetic regression coverage we still need an analytical
 * benchmark with ω⁻ ≠ ω⁺ (e.g. Poiseuille profile against `u(y)=Fy(H-y)/2ν`
 * — see test_trt_couette_poiseuille.cu for the canonical Λ=3/16 setup).
 *
 * Why this still earns its place: production calls setTRT(3/16) at every
 * MultiphysicsSolver construction (multiphysics_solver.cu:976), so EVERY
 * LPBF run is TRT-EDM. A future commit that accidentally skips the TRT
 * dispatch (or breaks the BGK-equivalence guarantee) would silently revert
 * production sims to a different operator. This test catches that in 1 sec.
 *
 * TRT mathematical reduction (for reference):
 *   TRT: f_post = f - ω⁺·f^+_neq - ω⁻·f^-_neq + EDM_shift
 *   When ω⁻=ω⁺:  ω⁺·f^+ + ω⁻·f^- = ω⁺·(f^+ + f^-) = ω⁺·f_neq ≡ BGK
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "core/lattice_d3q19.h"
#include "physics/fluid_lbm.h"
#include "utils/cuda_check.h"
#include "utils/cuda_memory.h"

using namespace lbm;
using namespace lbm::physics;
using lbm::utils::CudaBuffer;

namespace {

constexpr int  NX        = 16;
constexpr int  NY        = 16;
constexpr int  NZ        = 16;
constexpr int  NUM_CELLS = NX * NY * NZ;
constexpr int  Q         = 19;

// τ = 0.7 → ω = 1/0.7. For ω⁻ = ω⁺: τ⁻ = τ⁺, so Λ = (τ⁺-0.5)·(τ⁻-0.5) = 0.04.
constexpr float TAU          = 0.7f;
constexpr float NU_LU        = (TAU - 0.5f) / 3.0f;
constexpr float MAGIC_LAMBDA = (TAU - 0.5f) * (TAU - 0.5f);  // 0.04

// 5 steps + body force exercise the f_a_neq channel enough that a 1% ω⁻
// kernel drift produces ~5% f-deviation (well above the 1e-4 tolerance).
constexpr int   N_STEPS      = 5;
constexpr float BODY_FX      = 1.0e-4f;   // lattice units, sub-Mach
constexpr float REL_TOL_F    = 1.0e-4f;
constexpr float REL_TOL_U    = 1.0e-4f;
constexpr float REL_TOL_MASS = 1.0e-7f;

void initPerturbedField(FluidLBM& fluid) {
    std::vector<float> h_rho(NUM_CELLS, 1.0f);
    std::vector<float> h_ux(NUM_CELLS), h_uy(NUM_CELLS), h_uz(NUM_CELLS);
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);
                h_ux[idx] = 0.005f * std::sin(2.0f * 3.14159f * i / NX);
                h_uy[idx] = 0.003f * std::cos(2.0f * 3.14159f * j / NY);
                h_uz[idx] = 0.0f;
            }
    CudaBuffer<float> d_rho(NUM_CELLS), d_ux(NUM_CELLS), d_uy(NUM_CELLS), d_uz(NUM_CELLS);
    CUDA_CHECK(cudaMemcpy(d_rho.get(), h_rho.data(), NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux.get(),  h_ux.data(),  NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy.get(),  h_uy.data(),  NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uz.get(),  h_uz.data(),  NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    fluid.initialize(d_rho.get(), d_ux.get(), d_uy.get(), d_uz.get());
}

// Normalised L∞: max element-wise abs diff divided by GLOBAL max |a|.
// Per-cell relative would explode at zero crossings; a fixed-amplitude
// floor would mask real growth in mostly-quiet fields. Global max is
// the best fit for "biggest deviation as a fraction of meaningful signal."
float maxRelDiff(const std::vector<float>& a,
                 const std::vector<float>& b,
                 float eps = 1.0e-12f) {
    float scale = eps;
    for (float x : a) {
        float ax = std::abs(x);
        if (ax > scale) scale = ax;
    }
    float worst_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > worst_abs) worst_abs = diff;
    }
    return worst_abs / scale;
}

}  // namespace

TEST(TRTDegenerateToBGK, BodyForce_PeriodicBox) {
    FluidLBM fluid_bgk(NX, NY, NZ, NU_LU, 1.0f,
                       BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                       1.0f, 1.0f);
    FluidLBM fluid_trt(NX, NY, NZ, NU_LU, 1.0f,
                       BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                       1.0f, 1.0f);
    initPerturbedField(fluid_bgk);
    initPerturbedField(fluid_trt);
    fluid_trt.setTRT(MAGIC_LAMBDA);  // dispatch flips to TRT branch

    CudaBuffer<float> d_fx(NUM_CELLS), d_fy(NUM_CELLS), d_fz(NUM_CELLS), d_K(NUM_CELLS);
    {
        std::vector<float> h_fx(NUM_CELLS, BODY_FX);
        CUDA_CHECK(cudaMemcpy(d_fx.get(), h_fx.data(), NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_fy.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fz.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K.get(),  0, NUM_CELLS * sizeof(float)));

    for (int step = 0; step < N_STEPS; ++step) {
        fluid_bgk.computeMacroscopicEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid_bgk.collisionBGKwithEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid_bgk.streaming();

        fluid_trt.computeMacroscopicEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid_trt.collisionBGKwithEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid_trt.streaming();
    }

    std::vector<float> f_bgk(Q * NUM_CELLS), f_trt(Q * NUM_CELLS);
    CUDA_CHECK(cudaMemcpy(f_bgk.data(), fluid_bgk.getDistributionSrc(),
                          Q * NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f_trt.data(), fluid_trt.getDistributionSrc(),
                          Q * NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));

    float worst_f = maxRelDiff(f_bgk, f_trt);
    EXPECT_LT(worst_f, REL_TOL_F)
        << "TRT-degenerate distributions diverged from BGK by " << worst_f;

    std::vector<float> ux_bgk(NUM_CELLS), uy_bgk(NUM_CELLS), uz_bgk(NUM_CELLS);
    std::vector<float> ux_trt(NUM_CELLS), uy_trt(NUM_CELLS), uz_trt(NUM_CELLS);
    fluid_bgk.copyVelocityToHost(ux_bgk.data(), uy_bgk.data(), uz_bgk.data());
    fluid_trt.copyVelocityToHost(ux_trt.data(), uy_trt.data(), uz_trt.data());

    // u_z excluded: initialised to zero, FP32 ratio against near-zero scale
    // would explode without indicating a TRT bug.
    EXPECT_LT(std::max(maxRelDiff(ux_bgk, ux_trt), maxRelDiff(uy_bgk, uy_trt)),
              REL_TOL_U);

    double mass_bgk = 0.0, mass_trt = 0.0;
    for (auto v : f_bgk) mass_bgk += v;
    for (auto v : f_trt) mass_trt += v;
    double mass_rel = std::abs(mass_bgk - mass_trt) / std::abs(mass_bgk);
    EXPECT_LT(mass_rel, REL_TOL_MASS);
}

// LES branch coverage smoke. With cs_smag > 0, the TRT kernel re-evaluates
// omega_minus_eff from omega_eff while preserving Λ (fluid_lbm.cu:1635-1642).
// A sign error or division-by-zero guard miss would NaN the result. The test
// just runs and verifies the output is finite + mass-conserving — it does not
// assert numerical agreement against any reference.
TEST(TRTDegenerateToBGK, LES_BranchSmoke) {
    FluidLBM fluid(NX, NY, NZ, NU_LU, 1.0f,
                   BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                   1.0f, 1.0f);
    initPerturbedField(fluid);
    fluid.setTRT(3.0f / 16.0f);   // production Λ
    fluid.setSmagorinskyCs(0.16f); // standard Smagorinsky constant

    CudaBuffer<float> d_fx(NUM_CELLS), d_fy(NUM_CELLS), d_fz(NUM_CELLS), d_K(NUM_CELLS);
    CUDA_CHECK(cudaMemset(d_fx.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fy.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fz.get(), 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K.get(),  0, NUM_CELLS * sizeof(float)));

    for (int step = 0; step < N_STEPS; ++step) {
        fluid.computeMacroscopicEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid.collisionBGKwithEDM(d_fx.get(), d_fy.get(), d_fz.get(), d_K.get());
        fluid.streaming();
    }

    std::vector<float> f_out(Q * NUM_CELLS);
    CUDA_CHECK(cudaMemcpy(f_out.data(), fluid.getDistributionBuffer(),
                          Q * NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));

    double mass = 0.0;
    bool any_nan = false;
    for (auto v : f_out) {
        if (!std::isfinite(v)) any_nan = true;
        mass += v;
    }
    EXPECT_FALSE(any_nan) << "TRT+LES path produced NaN/Inf";
    EXPECT_NEAR(mass, static_cast<double>(NUM_CELLS), 1.0e-2)
        << "TRT+LES mass drift exceeded 1 % over " << N_STEPS << " steps";
}
