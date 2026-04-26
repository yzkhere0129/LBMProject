/**
 * @file  collision_trt_draft.cuh
 * @brief Two-Relaxation-Time (TRT) collision operator — DRAFT, DO NOT BUILD.
 *
 * Status: design draft per Dawn-2 protocol (2026-04-26 night run).
 * NOT registered in CMakeLists.txt. Not included by any .cu file.
 *
 * --------------------------------------------------------------------------
 * Why TRT
 * --------------------------------------------------------------------------
 * The current `fluidBGKCollisionEDMKernel` (`fluid_lbm.cu:~1300`) uses BGK
 * with a single τ. To stay numerically stable on LPBF flows it forces
 * τ ≈ 0.7, which inflates physical viscosity 4.3× over Mills 316L
 * (ν_LU = 0.065 vs target 0.0167 for ν_phys ≈ 8e-7 m²/s at dx=2μm/dt=80ns).
 *
 * TRT splits the distribution function into symmetric and anti-symmetric
 * parts and relaxes them with separate rates ω+ and ω-:
 *
 *   f_i^+ = ½(f_i + f_{ī}),         f_i^- = ½(f_i - f_{ī})
 *   f_eq^+ = ½(f_i^eq + f_{ī}^eq),  f_eq^- = ½(f_i^eq - f_{ī}^eq)
 *
 *   f_i^new = f_i  - ω+ (f_i^+ - f_eq^+)  - ω- (f_i^- - f_eq^-)  + S_i
 *
 * The "magic parameter" Λ = (1/ω+ - ½)(1/ω- - ½) controls boundary
 * accuracy and stability. Choices:
 *   - Λ = 1/4   : optimal stability for Stokes-like flow
 *   - Λ = 3/16  : exact-location half-way bounce-back for Poiseuille
 *   - Λ = 1/12  : 4th-order convergence for laplacian
 * Default in this draft: **Λ = 3/16** (matches existing Marangoni return-flow
 * benchmark setup, see scripts/viz/viz_marangoni_returnflow.cu).
 *
 * Physical viscosity: ν_LU = (1/ω+ - 1/2) / 3
 * ω- is solved from: ω- = 1 / (1/2 + Λ/(1/ω+ - 1/2))
 *
 * --------------------------------------------------------------------------
 * Integration into existing platform (TODO list for future merge)
 * --------------------------------------------------------------------------
 * 1. `include/physics/fluid_lbm.h`:
 *    - Add `enum class CollisionScheme { BGK_EDM, TRT_EDM };`
 *    - Add `void setTRT(float omega_plus, float lambda);` — host setter
 *    - Add member `float omega_plus_, omega_minus_, magic_lambda_;`
 *    - Default scheme stays BGK_EDM for backward compatibility.
 *
 * 2. `src/physics/fluid/fluid_lbm.cu`:
 *    - Add a wrapper `void FluidLBM::collisionTRTwithEDM(...)` that
 *      dispatches to `fluidTRTCollisionEDMKernel<<<>>>` (the kernel
 *      below).
 *    - Modify `MultiphysicsSolver::fluidStep()` to switch on scheme.
 *
 * 3. `apps/sim_linescan_phase{1,4}.cu`:
 *    - Replace `config.kinematic_viscosity = 0.065f` with explicit
 *      `setTRT(omega_plus, 3.0f/16.0f)` after construction.
 *
 * 4. Halfway bounce-back at walls: pair the wall-direction with its
 *    opposite. Λ=3/16 places the no-slip plane *exactly* on the wall;
 *    no need to shift indexing.
 *
 * 5. Validation tests to add (validation/test_trt_*.cu):
 *    - Poiseuille: confirm parabolic profile, viscosity-independent
 *      asymptote (the BGK τ=0.5+ε collapses; TRT survives down to
 *      τ+ ≈ 0.501 with Λ=3/16).
 *    - Couette-Poiseuille at varying τ+: should retain accuracy where
 *      BGK loses the inflection point.
 *    - 1D Marangoni return flow (already exists for BGK at TRT
 *      `viz_marangoni_returnflow.cu` — re-run with new operator).
 *    - Lid-driven cavity Re=400 / Re=1000.
 *
 * 6. Risks (call out before merging):
 *    - EDM forcing was derived assuming a single relaxation. The natural
 *      generalization is to apply the EDM increment to BOTH symmetric
 *      and anti-symmetric parts:
 *         Δf_i = f_eq(ρ, u + Δu) - f_eq(ρ, u)
 *         (f_i+, f_eq+) and (f_i-, f_eq-) shift consistently because the
 *         shift is at the macroscopic-velocity level. A short symbolic
 *         check (sympy script) is cheap and worth doing before deployment.
 *    - τ+ < 0.55 with strong Marangoni shear: same instability as BGK
 *      may re-appear; expected workaround is to clamp ω+ inside the
 *      kernel or fall back to MRT for those regimes.
 *    - At extremely strong recoil (catastrophic-cap zone) TRT alone may
 *      not save it — recoil is a bulk-momentum injection, not a viscous
 *      problem. A separate force-cap is still needed.
 *
 * --------------------------------------------------------------------------
 * Reference D3Q19 opposite table (already exists in lattice_d3q19.cu)
 * --------------------------------------------------------------------------
 *   i :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
 *   ī :  0  2  1  4  3  6  5  8  7 10  9 12 11 14 13 16 15 18 17
 *   (rest = 0; opposite of any axis-aligned vector flips that axis)
 */
#pragma once

#ifndef LBM_TRT_DRAFT_HEADER
#define LBM_TRT_DRAFT_HEADER

#include <cuda_runtime.h>
#include "core/lattice_d3q19.h"

namespace lbm {
namespace physics {
namespace trt_draft {

/**
 * TRT-EDM collision kernel for D3Q19.
 *
 * Inputs (device pointers):
 *   f       : SoA layout, 19 slabs of nx×ny×nz floats (existing convention)
 *   d_rho   : macroscopic density [nx*ny*nz]
 *   d_u     : macroscopic velocity components, 3 slabs of nx*ny*nz
 *   d_fx,d_fy,d_fz : non-Darcy force [nx*ny*nz]
 *   d_darcy : Darcy K coefficient (semi-implicit) [nx*ny*nz]
 *   omega_plus, omega_minus : TRT relaxation rates
 *
 * Same launch geometry as `fluidBGKCollisionEDMKernel` — drop-in compatible.
 *
 * Per-cell algorithm (NOT yet hand-unrolled):
 *   1. Compute u_bare = m / (ρ + 0.5K) with m = Σ ci·fi  (Darcy-aware).
 *   2. Δu = F_other / (ρ + 0.5K)         (Darcy-shared denominator).
 *   3. u_phys = u_bare + 0.5·Δu          (Guo-compatible 2nd-order velocity).
 *   4. Compute equilibria f_i^eq(ρ, u_bare) and f_i^eq(ρ, u_bare + Δu).
 *   5. EDM source S_i = f_i^eq(u_bare + Δu) - f_i^eq(u_bare).
 *   6. Build symmetric/anti-symmetric parts:
 *        f^+ = ½(f_i + f_{ī}),  f^- = ½(f_i - f_{ī})
 *        e^+ = ½(f_i^eq + f_{ī}^eq) at u_bare,  e^- analogous.
 *   7. Relax:
 *        f_i_post = f_i - ω+ (f^+ - e^+) - ω- (f^- - e^-) + S_i
 *
 * Resting direction (i=0): only ω+ acts (it's its own opposite, so f^- = 0).
 */
__global__ void fluidTRTCollisionEDMKernel_DRAFT(
    float*        f,
    const float*  d_rho,
    const float*  d_u,
    const float*  d_fx,
    const float*  d_fy,
    const float*  d_fz,
    const float*  d_darcy,
    float         omega_plus,
    float         omega_minus,
    int           nx,
    int           ny,
    int           nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    // ---- 1. Read distributions and macroscopic moments ----
    // (skeleton — not optimised; real kernel should use shared memory
    //  or thread-coarsening for D3Q19's 19 reads/writes per cell)
    float fi[19];
    int slab = total;
    #pragma unroll
    for (int q = 0; q < 19; ++q) fi[q] = f[q * slab + idx];

    float rho = d_rho[idx];
    float ux  = d_u[idx];
    float uy  = d_u[idx + slab];
    float uz  = d_u[idx + 2*slab];
    float fx  = d_fx ? d_fx[idx] : 0.0f;
    float fy  = d_fy ? d_fy[idx] : 0.0f;
    float fz  = d_fz ? d_fz[idx] : 0.0f;
    float K   = d_darcy ? d_darcy[idx] : 0.0f;

    // ---- 2. Darcy-aware u_bare and Δu ----
    float inv_denom = 1.0f / fmaxf(rho + 0.5f * K, 1e-6f);
    float ub_x = ux,  ub_y = uy,  ub_z = uz;       // u was already u_bare
    float dux = fx * inv_denom;
    float duy = fy * inv_denom;
    float duz = fz * inv_denom;
    float us_x = ub_x + dux,  us_y = ub_y + duy,  us_z = ub_z + duz;

    // ---- 3. Equilibrium evaluation ----
    // f_eq_i(ρ,u) = w_i ρ [1 + 3(ci·u) + 4.5(ci·u)² - 1.5 u²]
    auto feq = [&](int q, float ux_, float uy_, float uz_) -> float {
        float cu = lattice::D3Q19::ex_h[q]*ux_ + lattice::D3Q19::ey_h[q]*uy_
                 + lattice::D3Q19::ez_h[q]*uz_;
        float u2 = ux_*ux_ + uy_*uy_ + uz_*uz_;
        return lattice::D3Q19::w_h[q] * rho
               * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*u2);
    };

    // ---- 4. TRT relaxation per direction ----
    float fi_new[19];
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        int qbar = lattice::D3Q19::opposite_h[q];

        float feq_q   = feq(q,    ub_x, ub_y, ub_z);
        float feq_qbar= feq(qbar, ub_x, ub_y, ub_z);
        float fi_q    = fi[q];
        float fi_qbar = fi[qbar];

        float fp = 0.5f * (fi_q + fi_qbar);
        float fm = 0.5f * (fi_q - fi_qbar);
        float ep = 0.5f * (feq_q + feq_qbar);
        float em = 0.5f * (feq_q - feq_qbar);

        // EDM source per direction (shifted equilibrium difference)
        float S_q = feq(q, us_x, us_y, us_z) - feq_q;

        fi_new[q] = fi_q - omega_plus  * (fp - ep)
                          - omega_minus * (fm - em)
                          + S_q;
    }

    // ---- 5. Write back ----
    #pragma unroll
    for (int q = 0; q < 19; ++q) f[q * slab + idx] = fi_new[q];
}

/// Compute ω- from ω+ and Λ (host helper).
inline float omega_minus_from_lambda(float omega_plus, float lambda) {
    float tp_minus_half = 1.0f / omega_plus - 0.5f;
    float tm_minus_half = lambda / tp_minus_half;
    return 1.0f / (tm_minus_half + 0.5f);
}

} // namespace trt_draft
} // namespace physics
} // namespace lbm

#endif // LBM_TRT_DRAFT_HEADER
