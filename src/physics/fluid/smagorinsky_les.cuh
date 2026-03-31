/**
 * @file smagorinsky_les.cuh
 * @brief Smagorinsky LES subgrid-scale viscosity for D3Q19 LBM
 *
 * Computes local eddy viscosity from the non-equilibrium stress tensor,
 * then adjusts the relaxation parameter omega per cell.
 *
 * The strain rate is extracted DIRECTLY from the non-equilibrium part of
 * the distribution functions — no finite differences needed (one of LBM's
 * unique advantages for LES).
 *
 * Formula:
 *   S_ij = -(omega / 2·rho·cs²) · Σ_q (c_qi·c_qj · f_q^neq)
 *   |S| = sqrt(2·S_ij·S_ij)
 *   ν_sgs = (C_s · Δ)² · |S|
 *   ν_eff = ν_0 + ν_sgs
 *   τ_eff = 3·ν_eff + 0.5  (in lattice units where ν_eff is already in LU)
 *   ω_eff = 1 / τ_eff
 *
 * References:
 *   - Hou et al. (1996), J. Comput. Phys. 118:329-347
 *   - Teixeira (1998), Int. J. Mod. Phys. C 9:1159-1175
 *   - Yu et al. (2005), Comput. Fluids 35:957-965 (LBM-specific)
 */

#pragma once
#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Compute Smagorinsky effective omega from non-equilibrium distributions
 *
 * Called INSIDE the collision kernel, BEFORE the BGK/TRT relaxation step.
 * Uses the pre-collision distributions and the current macroscopic velocity
 * to extract the strain rate tensor.
 *
 * @param f_local   Local distribution values f[q] (pre-collision, 19 floats)
 * @param rho       Local density
 * @param ux,uy,uz  Local velocity (u_bare for EDM)
 * @param omega_0   Base relaxation parameter (from physical viscosity)
 * @param Cs        Smagorinsky constant (typically 0.1-0.2)
 * @return          Effective omega incorporating subgrid viscosity
 *
 * Implementation note: This is a DEVICE FUNCTION, not a kernel.
 * It is called per-cell within the collision kernel to avoid a separate pass.
 */
__device__ __forceinline__ float computeSmagorinskyOmega(
    const float* f_local,   // f[0..18] at this cell
    float rho,
    float ux, float uy, float uz,
    float omega_0,
    float Cs)
{
    // D3Q19 lattice vectors (compile-time constants for inlining)
    constexpr int cx[19] = {0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0};
    constexpr int cy[19] = {0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1};
    constexpr int cz[19] = {0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1};
    constexpr float wt[19] = {
        1.f/3, 1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
        1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,
        1.f/36,1.f/36,1.f/36,1.f/36
    };
    constexpr float cs2 = 1.0f / 3.0f;

    // ================================================================
    // Step 1: Compute non-equilibrium stress tensor Q_ij = Σ_q c_qi·c_qj·f_q^neq
    //
    // f_q^neq = f_q - f_q^eq
    // f_q^eq  = w_q · rho · (1 + c·u/cs² + (c·u)²/(2cs⁴) - u²/(2cs²))
    //
    // We only need the 6 independent components of the symmetric tensor:
    // Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz
    // ================================================================
    float Qxx = 0, Qyy = 0, Qzz = 0;
    float Qxy = 0, Qxz = 0, Qyz = 0;

    float usq = ux*ux + uy*uy + uz*uz;

    for (int q = 0; q < 19; q++) {
        float cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
        float feq = wt[q] * rho * (1.0f + cu/cs2 + 0.5f*cu*cu/(cs2*cs2) - 0.5f*usq/cs2);
        float fneq = f_local[q] - feq;

        Qxx += cx[q] * cx[q] * fneq;
        Qyy += cy[q] * cy[q] * fneq;
        Qzz += cz[q] * cz[q] * fneq;
        Qxy += cx[q] * cy[q] * fneq;
        Qxz += cx[q] * cz[q] * fneq;
        Qyz += cy[q] * cz[q] * fneq;
    }

    // ================================================================
    // Step 2: Non-equilibrium stress magnitude Q_mag = √(2·Q_ij·Q_ij)
    //
    // Used directly in the Hou (1996) algebraic formula.
    // No intermediate S_ij computation needed — the quadratic
    // formula absorbs the ω-dependent scaling.
    // ================================================================
    float Q2 = Qxx*Qxx + Qyy*Qyy + Qzz*Qzz
             + 2.0f*(Qxy*Qxy + Qxz*Qxz + Qyz*Qyz);

    // ================================================================
    // Step 3: Exact algebraic solution for τ_eff (Hou et al. 1996)
    //
    // The nonlinear coupling S(ω) ↔ ω(S) is resolved EXACTLY:
    //
    //   Q_mag = √(2·Q_ij·Q_ij)
    //   τ_eff = 0.5·(τ_0 + √(τ_0² + 18·Cs²·Q_mag/ρ))
    //   ω_eff = 1/τ_eff
    //
    // Derivation: S_ij = -ω/(2ρcs²)·Q_ij → |S| = ω/(2ρcs²)·Q_mag
    // Substituting into ν_sgs = (Cs·Δ)²·|S| and τ = 3(ν₀+ν_sgs)+0.5
    // gives a quadratic in τ with the above closed-form solution.
    //
    // This eliminates ALL lagging error — critical at τ₀ ≈ 0.507.
    // ================================================================
    float Q_mag = sqrtf(2.0f * Q2);
    float tau_0 = 1.0f / omega_0;
    float tau_eff = 0.5f * (tau_0 + sqrtf(tau_0 * tau_0
                    + 18.0f * Cs * Cs * Q_mag / fmaxf(rho, 1e-6f)));

    // Safety clamp: τ_eff ∈ [0.505, 5.0]
    tau_eff = fmaxf(0.505f, fminf(tau_eff, 5.0f));

    return 1.0f / tau_eff;
}

} // namespace physics
} // namespace lbm
