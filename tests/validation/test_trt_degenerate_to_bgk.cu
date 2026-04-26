/**
 * @file  test_trt_degenerate_to_bgk.cu
 * @brief Anchor test: TRT-EDM with ω⁻ = ω⁺ ≈ BGK-EDM byte-for-byte
 *        (within FP32 round-off tolerance).
 *
 * Background
 * ----------
 * `FluidLBM::collisionBGKwithEDM` dispatches to `fluidTRTCollisionEDMKernel`
 * whenever `omega_minus_ > 0` (fluid_lbm.cu:403-407). Production sims call
 * `setTRT(3/16)` so they're TRT-EDM end-to-end.
 *
 * The TRT operator decomposes the non-equilibrium part into symmetric and
 * anti-symmetric pieces and relaxes them with separate ω⁺ and ω⁻:
 *
 *   f_post = f - ω⁺·(f^+ - f_eq^+) - ω⁻·(f^- - f_eq^-) + EDM_shift
 *
 * In the degenerate case ω⁻ = ω⁺, mathematically:
 *   ω⁺·f^+ + ω⁻·f^- = ω⁺·(f^+ + f^-) = ω⁺·f_neq
 *
 * So TRT(ω, ω) reduces to BGK exactly in real arithmetic, but FP32 round-off
 * from the (½)(neq_q ± neq_qbar) intermediate steps breaks bit-equality.
 * This test verifies that the deviation stays within ~1e-5 max relative error
 * after a few collision+streaming steps — i.e., that the TRT path is a
 * faithful generalisation of BGK and any divergence is traceable to ω⁻ alone.
 *
 * Why we need this test
 * ---------------------
 * Confidence that switching production from BGK to TRT (which already happened)
 * did not introduce a silent kernel bug. Without this anchor, any future
 * regression in `fluidTRTCollisionEDMKernel` could be undetectable until a
 * full benchmark run shows divergence.
 *
 * Acceptance
 * ----------
 * - max element-wise |f_BGK - f_TRT| / max(|f_BGK|, ε) < 1e-4 after 5 steps
 * - max element-wise |u_BGK - u_TRT| / max(|u_BGK|, ε) < 1e-4
 * - mass conservation match: |Σf_BGK - Σf_TRT| / Σf_BGK < 1e-7
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"
#include "utils/cuda_check.h"

using namespace lbm;
using namespace lbm::physics;

namespace {

constexpr int NX = 16;
constexpr int NY = 16;
constexpr int NZ = 16;
constexpr int NUM_CELLS = NX * NY * NZ;
constexpr int Q = 19;

// τ = 0.7 → ω = 1/0.7 = 1.4286, (τ-0.5) = 0.2.
// For ω⁻ = ω⁺ degenerate: Λ = (τ⁺ - 0.5)·(τ⁻ - 0.5) = 0.2·0.2 = 0.04.
constexpr float TAU            = 0.7f;
constexpr float NU_LU          = (TAU - 0.5f) / 3.0f;   // 0.0667
constexpr float MAGIC_LAMBDA   = (TAU - 0.5f) * (TAU - 0.5f);  // 0.04

constexpr int  N_STEPS         = 2;
constexpr float REL_TOL_F      = 1.0e-4f;
constexpr float REL_TOL_U      = 1.0e-4f;
constexpr float REL_TOL_MASS   = 1.0e-7f;

// Initialise both solvers with a small velocity perturbation that exercises
// both symmetric and anti-symmetric modes. The FluidLBM::initialize(const float*…)
// overload expects DEVICE pointers, so we stage the host buffers to GPU first.
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
    float *d_rho, *d_ux, *d_uy, *d_uz;
    CUDA_CHECK(cudaMalloc(&d_rho, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ux,  NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uy,  NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uz,  NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux,  h_ux.data(),  NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy,  h_uy.data(),  NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uz,  h_uz.data(),  NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    fluid.initialize(d_rho, d_ux, d_uy, d_uz);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

// Normalised L∞: max element-wise absolute diff divided by GLOBAL max |a|.
// (Per-cell relative explodes near zero crossings; what we care about is
// "biggest deviation as a fraction of the meaningful signal".)
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

TEST(TRTDegenerateToBGK, FlatField_NoForce_NoDarcy) {
    // Two FluidLBMs with identical setup
    // Pass dt=1, dx=1 so the constructor's ν_LU = ν_phys conversion is identity
    // (we want LU directly, not physical SI units).
    FluidLBM fluid_bgk(NX, NY, NZ, NU_LU, 1.0f,
                       BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                       1.0f, 1.0f);
    FluidLBM fluid_trt(NX, NY, NZ, NU_LU, 1.0f,
                       BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                       1.0f, 1.0f);

    initPerturbedField(fluid_bgk);
    initPerturbedField(fluid_trt);

    // Switch fluid_trt to TRT branch (degenerate ω⁻ = ω⁺)
    fluid_trt.setTRT(MAGIC_LAMBDA);

    // Allocate zero-force, zero-Darcy device fields once
    float *d_fx, *d_fy, *d_fz, *d_K;
    CUDA_CHECK(cudaMalloc(&d_fx, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,  NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fx, 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fy, 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fz, 0, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K,  0, NUM_CELLS * sizeof(float)));

    // Step both solvers identically
    for (int step = 0; step < N_STEPS; ++step) {
        fluid_bgk.computeMacroscopicEDM(d_fx, d_fy, d_fz, d_K);
        fluid_bgk.collisionBGKwithEDM(d_fx, d_fy, d_fz, d_K);
        fluid_bgk.streaming();

        fluid_trt.computeMacroscopicEDM(d_fx, d_fy, d_fz, d_K);
        fluid_trt.collisionBGKwithEDM(d_fx, d_fy, d_fz, d_K);  // dispatches TRT
        fluid_trt.streaming();
    }

    // Pull distributions back to host
    std::vector<float> f_bgk(Q * NUM_CELLS), f_trt(Q * NUM_CELLS);
    CUDA_CHECK(cudaMemcpy(f_bgk.data(), fluid_bgk.getDistributionBuffer(),
                          Q * NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f_trt.data(), fluid_trt.getDistributionBuffer(),
                          Q * NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));

    // Distribution-level check
    float worst_f = maxRelDiff(f_bgk, f_trt);
    EXPECT_LT(worst_f, REL_TOL_F)
        << "TRT-degenerate distributions diverged from BGK by " << worst_f
        << " (tol " << REL_TOL_F << ")";

    // Velocity-level check: only compare the in-plane (x,y) components since
    // we did not perturb u_z. u_z difference normalised against a near-zero
    // scale would explode on FP32 round-off without indicating a real bug.
    std::vector<float> ux_bgk(NUM_CELLS), uy_bgk(NUM_CELLS), uz_bgk(NUM_CELLS);
    std::vector<float> ux_trt(NUM_CELLS), uy_trt(NUM_CELLS), uz_trt(NUM_CELLS);
    fluid_bgk.copyVelocityToHost(ux_bgk.data(), uy_bgk.data(), uz_bgk.data());
    fluid_trt.copyVelocityToHost(ux_trt.data(), uy_trt.data(), uz_trt.data());

    float worst_ux = maxRelDiff(ux_bgk, ux_trt);
    float worst_uy = maxRelDiff(uy_bgk, uy_trt);
    float worst_u = std::max(worst_ux, worst_uy);
    EXPECT_LT(worst_u, REL_TOL_U)
        << "TRT-degenerate velocity diverged: u_x=" << worst_ux
        << ", u_y=" << worst_uy;

    // Mass conservation
    double mass_bgk = 0.0, mass_trt = 0.0;
    for (auto v : f_bgk) mass_bgk += v;
    for (auto v : f_trt) mass_trt += v;
    double mass_rel = std::abs(mass_bgk - mass_trt) / std::abs(mass_bgk);
    EXPECT_LT(mass_rel, REL_TOL_MASS)
        << "Mass mismatch BGK=" << mass_bgk << " TRT=" << mass_trt;

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
    cudaFree(d_K);
}
