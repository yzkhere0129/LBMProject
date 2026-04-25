/**
 * @file thermal_fdm.cu
 * @brief Explicit FDM thermal solver — CUDA kernels and host implementation
 *
 * Core kernel: fdmAdvDiffKernel (fused advection + diffusion)
 *   T_new = T + dt_sub * [-u·grad(T) + alpha·laplacian(T)]
 *
 * Advection: first-order upwind (branch-free, compiler emits FSEL)
 * Diffusion: central difference, 7-point stencil
 * Phase change: ESM post-correction (same formula as LBM ESM)
 */

#include "physics/thermal_fdm.h"
#include "physics/phase_change.h"
#include "utils/cuda_check.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>

namespace lbm {
namespace physics {

// ============================================================================
// CUDA Kernel: Fused Advection-Diffusion Step
// ============================================================================
//
// Stability criterion: 6·Fo + |Cx| + |Cy| + |Cz| <= 1
//   where Fo = alpha·dt_sub/dx², Cd = ud_phys·dt_sub/dx
//
// Boundary: index clamping → zero-gradient (adiabatic) by default.
//           Periodic: wrap indices via modular arithmetic.
//           Dirichlet: applied separately after this kernel.
// ============================================================================
// ---- TVD Flux Limiters ----
// MINMOD: most diffusive TVD, φ(r) = max(0, min(r, 1))
// SUPERBEE: least diffusive TVD, φ(r) = max(0, max(min(2r,1), min(r,2)))
// VAN_LEER: smooth compromise, φ(r) = (r + |r|) / (1 + |r|)

__device__ __forceinline__ float limiter_superbee(float r) {
    return fmaxf(0.0f, fmaxf(fminf(2.0f*r, 1.0f), fminf(r, 2.0f)));
}

__device__ __forceinline__ float limiter_vanleer(float r) {
    return (r + fabsf(r)) / (1.0f + fabsf(r));
}

__device__ __forceinline__ float limiter_minmod(float r) {
    return fmaxf(0.0f, fminf(r, 1.0f));
}

// Active limiter selection (compile-time) — used as fallback at boundaries
#define TVD_LIMITER limiter_superbee

// WENO5 reconstruction
#include "weno5.cuh"

// ---- Clamped index helper (adiabatic = zero gradient at walls) ----
__device__ __forceinline__ int clamp_idx(int val, int lo, int hi) {
    return max(lo, min(hi, val));
}

__global__ void fdmAdvDiffKernel(
    const float* __restrict__ T_old,
    float* __restrict__       T_new,
    const float* __restrict__ ux_lu,
    const float* __restrict__ uy_lu,
    const float* __restrict__ uz_lu,
    const float* __restrict__ fill_level,  // VOF field (nullable: no masking)
    float alpha,
    float dt_sub,
    float dx,
    float v_conv,
    int nx, int ny, int nz,
    bool per_x, bool per_y, bool per_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int idx = i + nx * (j + ny * k);

    // ---- Neighbor indices: ±1 (diffusion + upwind) ----
    int im = per_x ? ((i == 0)    ? nx-1 : i-1) : clamp_idx(i-1, 0, nx-1);
    int ip = per_x ? ((i == nx-1) ? 0    : i+1) : clamp_idx(i+1, 0, nx-1);
    int jm = per_y ? ((j == 0)    ? ny-1 : j-1) : clamp_idx(j-1, 0, ny-1);
    int jp = per_y ? ((j == ny-1) ? 0    : j+1) : clamp_idx(j+1, 0, ny-1);
    int km = per_z ? ((k == 0)    ? nz-1 : k-1) : clamp_idx(k-1, 0, nz-1);
    int kp = per_z ? ((k == nz-1) ? 0    : k+1) : clamp_idx(k+1, 0, nz-1);

    // ---- Far-upwind indices: ±2 (needed for TVD gradient ratio) ----
    int imm = per_x ? ((i <= 1)    ? nx-2+i : i-2) : clamp_idx(i-2, 0, nx-1);
    int ipp = per_x ? ((i >= nx-2) ? i-nx+2 : i+2) : clamp_idx(i+2, 0, nx-1);
    int jmm = per_y ? ((j <= 1)    ? ny-2+j : j-2) : clamp_idx(j-2, 0, ny-1);
    int jpp = per_y ? ((j >= ny-2) ? j-ny+2 : j+2) : clamp_idx(j+2, 0, ny-1);
    int kmm = per_z ? ((k <= 1)    ? nz-2+k : k-2) : clamp_idx(k-2, 0, nz-1);
    int kpp = per_z ? ((k >= nz-2) ? k-nz+2 : k+2) : clamp_idx(k+2, 0, nz-1);

    // ---- Load 7-point diffusion stencil ----
    float Tc  = T_old[idx];
    float Txm = T_old[im + nx*(j  + ny*k )];
    float Txp = T_old[ip + nx*(j  + ny*k )];
    float Tym = T_old[i  + nx*(jm + ny*k )];
    float Typ = T_old[i  + nx*(jp + ny*k )];
    float Tzm = T_old[i  + nx*(j  + ny*km)];
    float Tzp = T_old[i  + nx*(j  + ny*kp)];

    // ---- VOF interface masking: gas neighbors → use Tc (adiabatic) ----
    // Prevents cold gas (300K) from being included in diffusion/advection stencil.
    // Without this, the gas-metal interface acts as an artificial cold wall.
    if (fill_level != nullptr) {
        float fc = fill_level[idx];
        if (fc < 0.05f) {
            // Pure gas cell: no thermal evolution, will be wiped later
            T_new[idx] = Tc;
            return;
        }
        // Metal cell: mask gas neighbors
        if (fill_level[im + nx*(j +ny*k )] < 0.05f) Txm = Tc;
        if (fill_level[ip + nx*(j +ny*k )] < 0.05f) Txp = Tc;
        if (fill_level[i  + nx*(jm+ny*k )] < 0.05f) Tym = Tc;
        if (fill_level[i  + nx*(jp+ny*k )] < 0.05f) Typ = Tc;
        if (fill_level[i  + nx*(j +ny*km)] < 0.05f) Tzm = Tc;
        if (fill_level[i  + nx*(j +ny*kp)] < 0.05f) Tzp = Tc;
    }

    // ---- Diffusion: central difference, Fo·(Σneighbors - 6·Tc) ----
    float Fo = alpha * dt_sub / (dx * dx);
    float diff = Fo * (Txm + Txp + Tym + Typ + Tzm + Tzp - 6.0f * Tc);

    // ---- Advection: WENO5 (5th-order) with Lax-Friedrichs flux splitting ----
    //
    // Flux at face i+1/2:
    //   F_{i+1/2} = u^+ · T^L_{i+1/2} + u^- · T^R_{i+1/2}
    // where u^+ = max(u,0), u^- = min(u,0)
    //   T^L = weno5_left(T[i-2]..T[i+2])    (left-biased, for u>0)
    //   T^R = weno5_right(T[i-1]..T[i+3])   (right-biased, for u<0)
    //
    // Stencil: 5 points per direction (i-2..i+2). At boundaries (i<2 or i>n-3),
    // indices are clamped (adiabatic ghost cells: T_ghost = T_boundary).
    //
    // Reference: Jiang & Shu (1996), J. Comput. Phys. 126:202-228
    // ============================================================
    float adv = 0.0f;
    if (ux_lu != nullptr || uy_lu != nullptr || uz_lu != nullptr) {
        float ux = ux_lu ? ux_lu[idx] * v_conv : 0.0f;
        float uy = uy_lu ? uy_lu[idx] * v_conv : 0.0f;
        float uz = uz_lu ? uz_lu[idx] * v_conv : 0.0f;
        float Co = dt_sub / dx;

        // Load extended stencil values (±2 already computed above)
        float Txmm = T_old[imm + nx*(j + ny*k)];
        float Txpp = T_old[ipp + nx*(j + ny*k)];
        float Tymm = T_old[i + nx*(jmm + ny*k)];
        float Typp = T_old[i + nx*(jpp + ny*k)];
        float Tzmm = T_old[i + nx*(j + ny*kmm)];
        float Tzpp = T_old[i + nx*(j + ny*kpp)];

        // --- X direction: face i-1/2 and i+1/2 ---
        {
            float up = fmaxf(ux, 0.0f);  // positive part
            float um = fminf(ux, 0.0f);  // negative part

            // Face i-1/2: left-biased from {i-3,i-2,i-1,i,i+1}
            // But we only have ±2, so stencil is {imm,im,i-1→im,i,ip} shifted
            // For face at (i-1/2): stencil centered at im
            // Left: weno5_left(T[im-2], T[im-1], T[im], T[i], T[ip])
            int imm_l = per_x ? ((im <= 0) ? nx-1+im : im-1) : clamp_idx(i-2, 0, nx-1);
            int immm  = per_x ? ((i <= 2) ? nx-3+i : i-3) : clamp_idx(i-3, 0, nx-1);
            float T_imm_l = T_old[immm + nx*(j+ny*k)];

            float TL_left = weno5_left(T_imm_l, Txmm, Txm, Tc, Txp);
            float TR_left = weno5_right(Txmm, Txm, Tc, Txp, Txpp);
            float F_left = up * TL_left + um * TR_left;

            // Face i+1/2: stencil centered at i
            int ippp = per_x ? ((i >= nx-3) ? i-nx+3 : i+3) : clamp_idx(i+3, 0, nx-1);
            float T_ippp = T_old[ippp + nx*(j+ny*k)];

            float TL_right = weno5_left(Txmm, Txm, Tc, Txp, Txpp);
            float TR_right = weno5_right(Txm, Tc, Txp, Txpp, T_ippp);
            float F_right = up * TL_right + um * TR_right;

            adv -= Co * (F_right - F_left);
        }

        // --- Y direction ---
        {
            float up = fmaxf(uy, 0.0f);
            float um = fminf(uy, 0.0f);

            int jmmm = per_y ? ((j <= 2) ? ny-3+j : j-3) : clamp_idx(j-3, 0, ny-1);
            int jppp = per_y ? ((j >= ny-3) ? j-ny+3 : j+3) : clamp_idx(j+3, 0, ny-1);
            float T_jmmm = T_old[i + nx*(jmmm + ny*k)];
            float T_jppp = T_old[i + nx*(jppp + ny*k)];

            float TL_left = weno5_left(T_jmmm, Tymm, Tym, Tc, Typ);
            float TR_left = weno5_right(Tymm, Tym, Tc, Typ, Typp);
            float F_left = up * TL_left + um * TR_left;

            float TL_right = weno5_left(Tymm, Tym, Tc, Typ, Typp);
            float TR_right = weno5_right(Tym, Tc, Typ, Typp, T_jppp);
            float F_right = up * TL_right + um * TR_right;

            adv -= Co * (F_right - F_left);
        }

        // --- Z direction ---
        {
            float up = fmaxf(uz, 0.0f);
            float um = fminf(uz, 0.0f);

            int kmmm = per_z ? ((k <= 2) ? nz-3+k : k-3) : clamp_idx(k-3, 0, nz-1);
            int kppp = per_z ? ((k >= nz-3) ? k-nz+3 : k+3) : clamp_idx(k+3, 0, nz-1);
            float T_kmmm = T_old[i + nx*(j + ny*kmmm)];
            float T_kppp = T_old[i + nx*(j + ny*kppp)];

            float TL_left = weno5_left(T_kmmm, Tzmm, Tzm, Tc, Tzp);
            float TR_left = weno5_right(Tzmm, Tzm, Tc, Tzp, Tzpp);
            float F_left = up * TL_left + um * TR_left;

            float TL_right = weno5_left(Tzmm, Tzm, Tc, Tzp, Tzpp);
            float TR_right = weno5_right(Tzm, Tc, Tzp, Tzpp, T_kppp);
            float F_right = up * TL_right + um * TR_right;

            adv -= Co * (F_right - F_left);
        }
    }

    T_new[idx] = Tc + diff + adv;
}

// ============================================================================
// CUDA Kernel: ESM Phase Change Correction — Bisection Enthalpy Inversion
// ============================================================================
// Strict enthalpy-method corrector. Given T_old (pre-step) and T_star (post-
// heat-source + advDiff), reconstructs the exact delivered energy
//   ΔE = ρ(T_old)·cp(T_old)·(T_star − T_old)            [J/m³]
// and inverts H(T_new) = H(T_old) + ΔE using the ground-truth piecewise H(T)
// defined in MaterialProperties::enthalpyPerVolume (includes latent heat).
//
// Three-branch dispatch:
//   Solid  (H_target ≤ H_solidus):   T_new = H_target / k_s, fl = 0
//   Liquid (H_target ≥ H_liquidus):  T_new = T_l + (H_target − H_liq)/k_l, fl = 1
//   Mushy  (H_s < H_target < H_l):   bisect on [T_s, T_l] (N_max=25, tol=1e-6)
//
// This is the true enthalpy-method corrector — latent heat is honored exactly;
// the "phantom energy" bug (~6% residual) is eliminated.
// Reference: Voller & Prakash (1987), Bisection as in any standard root-finder.
// ============================================================================
__global__ void fdmESMKernel(
    const float* __restrict__ T_old,    // snapshot: T at start of step
    float* __restrict__ T,              // IN: T_star (post-source, post-advDiff); OUT: T_new
    float* __restrict__ fl,             // OUT: f_l_new (equilibrium value from T_new)
    const float* __restrict__ fill_level,
    MaterialProperties mat,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // VOF mask: skip gas cells (no phase change in inert atmosphere)
    if (fill_level != nullptr && fill_level[idx] < 0.01f) {
        fl[idx] = 0.0f;
        return;
    }

    const float T_sol = mat.T_solidus;
    const float T_liq = mat.T_liquidus;
    const float dT_m  = T_liq - T_sol;
    if (dT_m < 1e-8f || mat.L_fusion < 1e-6f) return;

    const float To = T_old[idx];
    const float Ts = T[idx];                 // T_star

    // Energy delivered this step per unit volume.
    // For the heat-source kernel this equals Q·dt exactly; for advDiff it
    // equals the net flux divergence·dt (first-order operator-split accuracy).
    const float rho_old = mat.getDensity(To);
    const float cp_old  = mat.getSpecificHeat(To);
    const float dE      = rho_old * cp_old * (Ts - To);

    // Target enthalpy (piecewise analytic H(T) from material_properties.h)
    const float H_old    = mat.enthalpyPerVolume(To);
    const float H_target = H_old + dE;

    // Enthalpy boundaries — evaluate ground-truth H(T) at T_sol and T_liq
    const float H_solidus  = mat.enthalpyPerVolume(T_sol);
    const float H_liquidus = mat.enthalpyPerVolume(T_liq);

    const float k_s = mat.rho_solid  * mat.cp_solid;   // J/(m³·K)
    const float k_l = mat.rho_liquid * mat.cp_liquid;

    float T_new, fl_new;

    if (H_target <= H_solidus) {
        // --- Regime A: pure solid ---
        // H_sensible_solid(T) = k_s·T  →  T = H_target / k_s
        // Guard against extreme cooling pulling T below 0K (applyTemperatureSafetyCap
        // catches this downstream, but defensive clamp keeps the kernel self-consistent).
        T_new  = fmaxf(H_target / k_s, 1.0f);
        fl_new = 0.0f;
    } else if (H_target >= H_liquidus) {
        // --- Regime B: pure liquid ---
        // Above T_liq, H(T) = H_liquidus + k_l·(T − T_liq) → invert linearly
        T_new  = T_liq + (H_target - H_liquidus) / k_l;
        fl_new = 1.0f;
    } else {
        // --- Regime C: mushy zone — bisect on [T_sol, T_liq] ---
        // H(T) is strictly monotonically increasing on (T_sol, T_liq) with
        // dH/dT = k(T) + ρ_ref·L/ΔT_m > 0. Unique root guaranteed.
        float T_lo = T_sol;
        float T_hi = T_liq;
        float T_mid = 0.5f * (T_lo + T_hi);

        // Robust absolute tolerance: rel 1e-6 vs |H_target|, floored at ρ_ref·L.
        const float H_floor  = 0.5f * (mat.rho_solid + mat.rho_liquid) * mat.L_fusion;
        const float tol_abs  = 1e-6f * fmaxf(fabsf(H_target), H_floor);

        #pragma unroll 1   // early-exit varies per cell; do not unroll
        for (int iter = 0; iter < 25; ++iter) {
            T_mid = 0.5f * (T_lo + T_hi);
            float H_mid = mat.enthalpyPerVolume(T_mid);
            float diff  = H_mid - H_target;
            if (fabsf(diff) < tol_abs) break;
            if (diff < 0.0f) T_lo = T_mid;
            else             T_hi = T_mid;
        }
        T_new  = T_mid;
        fl_new = (T_new - T_sol) / dT_m;
        fl_new = fmaxf(0.0f, fminf(1.0f, fl_new));
    }

    T[idx]  = T_new;
    fl[idx] = fl_new;
}

// ============================================================================
// CUDA Kernel: Heat Source Injection
// ============================================================================
__global__ void fdmHeatSourceKernel(
    float* __restrict__ T,
    const float* __restrict__ Q_vol,
    const float* __restrict__ fill_level,
    float dt,
    MaterialProperties mat,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // Allow heat deposition at interface cells (f>=0.05) for powder bed compatibility.
    // Gas-wipe (f<0.01) handles thermal isolation of pure gas cells.
    if (fill_level != nullptr && fill_level[idx] < 0.05f) return;

    float Q = Q_vol[idx];
    if (Q <= 0.0f) return;

    float Tl = T[idx];
    float rho = mat.getDensity(Tl);
    float cp  = mat.getSpecificHeat(Tl);
    float rho_cp = rho * cp;
    if (rho_cp < 1.0f) return;

    T[idx] += Q * dt / rho_cp;
}

// ============================================================================
// CUDA Kernel: Dirichlet BC on a face
// ============================================================================
__global__ void fdmDirichletKernel(
    float* __restrict__ T,
    float T_wall,
    int face, int nx, int ny, int nz)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = -1;

    switch (face) {
        case 0: if (a<ny && b<nz) idx = 0     + nx*(a + ny*b); break; // x_min
        case 1: if (a<ny && b<nz) idx = nx-1   + nx*(a + ny*b); break; // x_max
        case 2: if (a<nx && b<nz) idx = a      + nx*(0 + ny*b); break; // y_min
        case 3: if (a<nx && b<nz) idx = a      + nx*(ny-1+ny*b); break; // y_max
        case 4: if (a<nx && b<ny) idx = a      + nx*(b + ny*0); break;  // z_min
        case 5: if (a<nx && b<ny) idx = a      + nx*(b + ny*(nz-1)); break; // z_max
    }

    if (idx >= 0) T[idx] = T_wall;
}

// ============================================================================
// CUDA Kernel: Evaporation Cooling (Hertz-Knudsen-Langmuir + anti-oscillation)
// ============================================================================
// At surface cells where T > T_boil:
//   J_evap = α_evap · P_sat(T) / √(2π·R·T/M)
//   P_sat  = P_ref · exp[(L_vap·M/R)·(1/T_boil - 1/T)]
//   q_evap = J_evap · L_vap   [W/m²]
//   dT     = q_evap · dt / (ρ·cp·dx)
//
// Anti-oscillation: T_new = max(T_boil, T - dT)
// This guarantees T never undershoots T_boil in a single step,
// even when q_evap grows exponentially near T >> T_boil.
// ============================================================================
__global__ void fdmEvaporationCoolingKernel(
    float* __restrict__ T,
    const float* __restrict__ fill_level,  // VOF fill level (nullable)
    int nx, int ny, int nz,
    int z_surface,       // -1 = volumetric mode (all cells)
    float dt, float dx,
    MaterialProperties mat,
    float cooling_factor)
{
    // Volumetric mode: 1D thread indexing over all cells
    // Surface mode: 2D thread indexing over one z-layer
    int idx;
    if (z_surface < 0) {
        idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nx * ny * nz) return;
    } else {
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx || iy >= ny) return;
        idx = ix + iy * nx + z_surface * nx * ny;
    }

    // BUG FIX: Only apply evaporation at the free surface (interface cells).
    // Previous code had NO VOF check, cooling ALL cells with T > T_boil including
    // bulk liquid and even gas cells, over-cooling by 4× in typical domains.
    // Physical reality: HKL evaporation requires a vapor-liquid interface.
    // Interior liquid at T > T_boil is superheated but cannot evaporate without
    // a free surface. Only cells with f ∈ (0.01, 0.99) are actual interface cells.
    if (fill_level != nullptr) {
        float f = fill_level[idx];
        if (f <= 0.01f || f >= 0.99f) return;  // not at free surface
    }

    float Tc = T[idx];

    float T_boil = mat.T_vaporization;
    if (Tc <= T_boil) return;  // no evaporation below boiling point

    // Clausius-Clapeyron saturation pressure
    float L_vap = mat.L_vaporization;  // J/kg
    float M     = mat.molar_mass;       // kg/mol
    constexpr float R_gas = 8.314f;     // J/(mol·K)
    constexpr float P_ref = 101325.0f;  // Pa (1 atm)
    // R6 OpenFOAM-alignment: Hertz-Knudsen sticking coefficient from laserMeltFoam.
    // Previous 0.18 was the Semak-1995 "partial-pressure" value; laserMeltFoam's
    // UEqn.H/TEqn.H uses σ=0.82 (~full accommodation), which dominates the T
    // regulation and drops steady-state T from ~3760 K to near T_boil.
    constexpr float alpha_evap = 0.82f;

    float exponent = (L_vap * M / R_gas) * (1.0f / T_boil - 1.0f / Tc);
    // Clamp exponent to prevent overflow (exp(88) ≈ FLT_MAX)
    exponent = fminf(exponent, 80.0f);
    float P_sat = P_ref * expf(exponent);

    // Hertz-Knudsen-Langmuir evaporation mass flux [kg/(m²·s)]
    float J_evap = alpha_evap * P_sat / sqrtf(2.0f * 3.14159265f * R_gas * Tc / M);

    float q_evap = J_evap * L_vap * cooling_factor;

    // Temperature decrement [K]
    float rho = mat.getDensity(Tc);
    float cp  = mat.getSpecificHeat(Tc);
    float dT  = q_evap * dt / (rho * cp * dx);

    // Anti-oscillation red line: T_new >= T_boil (NEVER undershoot)
    T[idx] = fmaxf(T_boil, Tc - dT);
}

// ============================================================================
// CUDA Kernel: Gas phase temperature wipe
// ============================================================================
// ============================================================================
// Gas wipe protection: only reset far-field gas cells to T_ambient.
// Near-interface gas (within N layers of any f>0 cell) retains temperature
// to prevent the gas wipe from stealing energy from the melt pool.
// ============================================================================

// Step 1: Initialize protection mask from fill level
__global__ void initGasProtectionMaskKernel(
    uint8_t* __restrict__ mask,
    const float* __restrict__ fill_level,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    mask[idx] = (fill_level[idx] >= 0.01f) ? 1 : 0;
}

// Step 2: Binary dilation — expand protected region by 1 cell (6-neighbor)
// Run N times for N-layer protection. In-place update is safe because
// we only set 0→1 (monotone), so race conditions don't cause errors.
__global__ void dilateProtectionMaskKernel(
    uint8_t* __restrict__ mask,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + j*nx + k*nx*ny;
    if (mask[idx] == 1) return;  // already protected

    // Check 6 face neighbors
    if (i > 0    && mask[(i-1) + j*nx + k*nx*ny]) { mask[idx] = 1; return; }
    if (i < nx-1 && mask[(i+1) + j*nx + k*nx*ny]) { mask[idx] = 1; return; }
    if (j > 0    && mask[i + (j-1)*nx + k*nx*ny]) { mask[idx] = 1; return; }
    if (j < ny-1 && mask[i + (j+1)*nx + k*nx*ny]) { mask[idx] = 1; return; }
    if (k > 0    && mask[i + j*nx + (k-1)*nx*ny]) { mask[idx] = 1; return; }
    if (k < nz-1 && mask[i + j*nx + (k+1)*nx*ny]) { mask[idx] = 1; return; }
}

// Step 3: Gas wipe with protection mask
// Far-field gas (mask=0): hard reset to T_ambient.
// Near-interface gas (mask=1, f<0.01): exponential relaxation toward T_ambient.
//   This prevents "ghost heat layers" where evacuated cavities near the
//   interface retain T_boil forever, blocking thermal relaxation and
//   distorting Marangoni gradients.
//   Relaxation rate: T_new = T - α*(T - T_amb), α = 0.01 per step (~1/100 decay).
__global__ void fdmGasWipeKernel(
    float* __restrict__ T,
    const float* __restrict__ fill_level,
    const uint8_t* __restrict__ protection_mask,
    float T_ambient,
    int num_cells)
{
    // NOTE: energy removed by gas wipe = material.enthalpyIncrement(T_old, T_new)*V.
    // Use MaterialProperties::enthalpyIncrement as the canonical primitive if tracking
    // is ever added here (avoids the phantom mushy-zone jump in naive ρ·cp·ΔT·V).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    if (fill_level == nullptr) return;

    float f = fill_level[idx];
    if (f >= 0.01f) return;  // Metal cell — don't touch

    if (protection_mask[idx] == 0) {
        // Far-field gas: hard reset
        T[idx] = T_ambient;
    } else {
        // Near-interface gas cavity: gentle exponential relaxation
        // Prevents ghost heat layers while allowing transient thermal coupling
        T[idx] -= 0.01f * (T[idx] - T_ambient);
    }
}

// ============================================================================
// CUDA Kernel: Sub-surface volumetric boiling cap with energy tracking
// ============================================================================
// For bulk liquid cells (f >= 0.99) with T > T_cap:
//   - Cap temperature to T_cap
//   - Accumulate removed energy (physically: latent heat of volumetric boiling)
// This prevents sub-surface cells from overheating to 3798K while tracking
// the removed energy for accurate energy balance diagnostics.
// ============================================================================
__global__ void fdmSubsurfaceBoilingCapKernel(
    float* __restrict__ T,
    const float* __restrict__ fill_level,
    float T_cap,
    MaterialProperties mat, float dV,
    unsigned long long* __restrict__ d_energy_removed_raw,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    if (fill_level == nullptr) return;

    float f = fill_level[idx];
    if (f < 0.99f) return;  // Only bulk liquid (not interface)

    float Tc = T[idx];
    if (Tc <= T_cap) return;

    float dT = Tc - T_cap;
    T[idx] = T_cap;

    // R6 FIX: Use temperature-dependent rho(T) and cp(T) matching computeTotalThermalEnergy().
    // Bulk liquid cells at T > T_boil have rho = rho_liquid, not rho_solid.
    float rho = mat.getDensity(Tc);
    float cp = mat.getSpecificHeat(Tc);

    float E_removed = rho * cp * dT * dV;
    unsigned long long E_fixed = (unsigned long long)(E_removed * 1.0e12);
    atomicAdd(d_energy_removed_raw, E_fixed);
}

// ============================================================================
// CUDA Kernel: Gas wipe with energy tracking
// ============================================================================
// Same as fdmGasWipeKernel but also tracks removed energy via atomicAdd.
// ============================================================================
__global__ void fdmGasWipeWithTrackingKernel(
    float* __restrict__ T,
    const float* __restrict__ fill_level,
    const uint8_t* __restrict__ protection_mask,
    float T_ambient,
    MaterialProperties mat, float dV,
    unsigned long long* __restrict__ d_energy_removed_raw,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    if (fill_level == nullptr) return;

    float f = fill_level[idx];
    if (f >= 0.01f) return;

    float T_old = T[idx];
    float T_new;

    if (protection_mask[idx] == 0) {
        T_new = T_ambient;
    } else {
        // Sprint-1 (2026-04-25): was 0.01·(T-T_amb) per step which over a few
        // hundred steps still drains protected hot-vapor cells. Now zero — let
        // the protected interior evolve under diffusion + advection only.
        // Streaming artifacts inside D3Q7 LBM-thermal would re-emerge here, but
        // FDM is the production thermal path and it has no streaming, so the
        // gentle relaxation was over-kill.
        T_new = T_old;
    }

    T[idx] = T_new;

    // R6 FIX: Use temperature-dependent rho(T) and cp(T) matching computeTotalThermalEnergy().
    // For hot gas cells near the metal (T > T_solidus), rho drops from rho_solid to rho_liquid.
    // Using constant rho_solid overestimates energy change by up to 2x.
    float rho = mat.getDensity(T_old);
    float cp = mat.getSpecificHeat(T_old);

    // Track NET energy change (positive = removed/cooling, negative = added/heating)
    float dT = T_old - T_new;
    float E_change = rho * cp * dT * dV;
    long long E_fixed = (long long)(E_change * 1.0e12);
    if (E_fixed >= 0) {
        atomicAdd(d_energy_removed_raw, (unsigned long long)E_fixed);
    } else {
        unsigned long long neg = (unsigned long long)(-E_fixed);
        atomicAdd(d_energy_removed_raw, ~neg + 1ULL);
    }
}

// ============================================================================
// CUDA Kernel: Hard T_boil cap on ALL cells (not just surface)
// ============================================================================
// Without VOF free-surface tracking, material above T_boil would vaporize
// and leave the domain. The cap enforces this physical constraint:
// any cell with T > T_boil has its excess energy removed (as if vaporized).
// ============================================================================
__global__ void fdmBoilCapKernel(float* T, float T_boil, int num_cells) {
    // NOTE: energy removed per cell by boil cap = material.enthalpyIncrement(T_boil, T[idx])*V
    // (positive when T[idx] > T_boil). Use enthalpyIncrement if tracking is added here.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    if (T[idx] > T_boil) T[idx] = T_boil;
}

// ============================================================================
// CUDA Kernel: Temperature Safety Cap
// ============================================================================
__global__ void fdmTCapKernel(float* T, float T_cap, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    T[idx] = fminf(T[idx], T_cap);
}

// ============================================================================
// CUDA Kernel: Fill with constant value
// ============================================================================
__global__ void fdmFillKernel(float* arr, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

// ============================================================================
// CUDA Kernel: Copy array (fl → fl_prev)
// ============================================================================
__global__ void fdmCopyKernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ============================================================================
// Host Implementation
// ============================================================================

ThermalFDM::ThermalFDM(int nx, int ny, int nz,
                       const MaterialProperties& material,
                       float thermal_diffusivity,
                       bool enable_phase_change,
                       float dt, float dx)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      dt_(dt), dx_(dx), alpha_phys_(thermal_diffusivity),
      material_(material)
{
    // Compute subcycle count from combined stability: 6·Fo + CFL ≤ 1
    // Assume max advection velocity ~ 20 m/s (typical Marangoni)
    float Fo = alpha_phys_ * dt_ / (dx_ * dx_);
    float CFL_est = 20.0f * dt_ / dx_;  // conservative estimate
    float combined = 6.0f * Fo + CFL_est;
    if (combined > 0.9f) {
        n_subcycle_ = static_cast<int>(std::ceil(combined / 0.9f));
    }

    float dt_sub = dt_ / static_cast<float>(n_subcycle_);
    float Fo_sub = alpha_phys_ * dt_sub / (dx_ * dx_);

    std::cout << "ThermalFDM initialized:\n"
              << "  Domain: " << nx_ << " x " << ny_ << " x " << nz_ << "\n"
              << "  dt = " << dt_ << " s, dx = " << dx_ << " m\n"
              << "  alpha = " << alpha_phys_ << " m^2/s\n"
              << "  Fo = " << Fo << " (per full dt)\n"
              << "  Subcycles: " << n_subcycle_
              << " (dt_sub = " << dt_sub << " s, Fo_sub = " << Fo_sub << ")\n"
              << "  Phase change: " << (enable_phase_change ? "ENABLED" : "DISABLED")
              << std::endl;

    allocateMemory();

    if (enable_phase_change) {
        phase_solver_ = new PhaseChangeSolver(nx_, ny_, nz_, material_);
    }
}

ThermalFDM::~ThermalFDM() {
    freeMemory();
    if (phase_solver_) delete phase_solver_;
}

ThermalFDM::ThermalFDM(ThermalFDM&& o) noexcept
    : nx_(o.nx_), ny_(o.ny_), nz_(o.nz_), num_cells_(o.num_cells_),
      dt_(o.dt_), dx_(o.dx_), alpha_phys_(o.alpha_phys_),
      material_(o.material_), phase_solver_(o.phase_solver_),
      d_T_(o.d_T_), d_T_new_(o.d_T_new_), d_T_old_(o.d_T_old_),
      d_vof_fill_(o.d_vof_fill_), d_gas_wipe_mask_(o.d_gas_wipe_mask_),
      skip_T_cap_(o.skip_T_cap_),
      z_periodic_(o.z_periodic_), n_subcycle_(o.n_subcycle_)
{
    o.d_T_ = o.d_T_new_ = o.d_T_old_ = nullptr;
    o.d_gas_wipe_mask_ = nullptr;
    o.phase_solver_ = nullptr;
}

void ThermalFDM::allocateMemory() {
    CUDA_CHECK(cudaMalloc(&d_T_,     num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T_new_, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T_old_, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_T_,     0, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_T_new_, 0, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_T_old_, 0, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gas_wipe_mask_, num_cells_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_gas_wipe_mask_, 0, num_cells_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_gas_wipe_energy_raw_, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_boiling_cap_energy_raw_, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_gas_wipe_energy_raw_, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_boiling_cap_energy_raw_, 0, sizeof(unsigned long long)));
}

void ThermalFDM::freeMemory() {
    if (d_T_)     { cudaFree(d_T_);     d_T_ = nullptr; }
    if (d_T_new_) { cudaFree(d_T_new_); d_T_new_ = nullptr; }
    if (d_T_old_) { cudaFree(d_T_old_); d_T_old_ = nullptr; }
    if (d_gas_wipe_mask_) { cudaFree(d_gas_wipe_mask_); d_gas_wipe_mask_ = nullptr; }
    if (d_gas_wipe_energy_raw_) { cudaFree(d_gas_wipe_energy_raw_); d_gas_wipe_energy_raw_ = nullptr; }
    if (d_boiling_cap_energy_raw_) { cudaFree(d_boiling_cap_energy_raw_); d_boiling_cap_energy_raw_ = nullptr; }
}

void ThermalFDM::initialize(float initial_temp) {
    T_initial_ = initial_temp;
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    fdmFillKernel<<<gs, bs>>>(d_T_, initial_temp, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Safe fallback: d_T_old_ = d_T_ so bisection ESM has valid input
    // even if caller forgets to call storePreviousTemperature() on first step.
    CUDA_CHECK(cudaMemcpy(d_T_old_, d_T_,
                          num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));

    if (phase_solver_) {
        phase_solver_->initializeFromTemperature(d_T_);
    }
}

void ThermalFDM::initialize(const float* temp_field) {
    // Compute mean temperature from host array for enthalpy reference.
    // Diagnostic approximation: mean T used as T_initial_ when field is non-uniform.
    double sum = 0.0;
    for (int i = 0; i < num_cells_; ++i) sum += temp_field[i];
    T_initial_ = static_cast<float>(sum / num_cells_);

    CUDA_CHECK(cudaMemcpy(d_T_, temp_field,
                          num_cells_ * sizeof(float), cudaMemcpyHostToDevice));

    // Safe fallback: d_T_old_ = d_T_ (see initialize(float) for rationale)
    CUDA_CHECK(cudaMemcpy(d_T_old_, d_T_,
                          num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));

    if (phase_solver_) {
        phase_solver_->initializeFromTemperature(d_T_);
    }
}

// collisionBGK → FDM advection + diffusion (writes to d_T_new_)
void ThermalFDM::collisionBGK(const float* ux, const float* uy, const float* uz) {
    dim3 block(8, 8, 4);
    dim3 grid((nx_ + block.x - 1) / block.x,
              (ny_ + block.y - 1) / block.y,
              (nz_ + block.z - 1) / block.z);

    // Velocity conversion: v_phys = v_lu * dx / dt_fluid
    // dt_fluid = dt_ (the full LBM time step, NOT dt_sub)
    float v_conv = dx_ / dt_;

    float dt_sub = dt_ / static_cast<float>(n_subcycle_);

    for (int s = 0; s < n_subcycle_; s++) {
        fdmAdvDiffKernel<<<grid, block>>>(
            d_T_, d_T_new_, ux, uy, uz, d_vof_fill_,
            alpha_phys_, dt_sub, dx_, v_conv,
            nx_, ny_, nz_,
            false, false, z_periodic_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        // For multi-subcycle: swap and re-do (except last — leave in d_T_new_)
        if (s < n_subcycle_ - 1) {
            swapBuffers();
        }
    }
    // After all subcycles: d_T_new_ has the final result.
    // streaming() will swap it into d_T_.
}

// streaming → buffer swap
void ThermalFDM::streaming() {
    swapBuffers();
    // d_T_ now holds the post-advDiff temperature
}

// Snapshot T into d_T_old_ — MUST be called at the top of each step loop,
// before addHeatSource / collisionBGK / any T-modifying operation.
// Consumed by fdmESMKernel for strict enthalpy-method inversion.
void ThermalFDM::storePreviousTemperature() {
    CUDA_CHECK(cudaMemcpy(d_T_old_, d_T_,
                          num_cells_ * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Compute gas wipe protection mask via iterative binary dilation
void ThermalFDM::computeGasWipeProtectionMask(int protection_layers) {
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;

    // Init: mask=1 where fill>=0.01, mask=0 elsewhere
    initGasProtectionMaskKernel<<<gs, bs>>>(d_gas_wipe_mask_, d_vof_fill_, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dilate N times for N-layer protection
    dim3 blk(8, 8, 4);
    dim3 grd((nx_ + blk.x - 1) / blk.x,
             (ny_ + blk.y - 1) / blk.y,
             (nz_ + blk.z - 1) / blk.z);
    // Sprint-1 perf (2026-04-25): each dilation iteration USED to sync after,
    // turning N layers into N×~50μs sync overhead. With protection_layers=30
    // and 15.6 M cells (M16 domain), 30 syncs/step blew wall time 30× over
    // the small-domain baseline. Default stream serializes kernel launches
    // automatically, so inter-iteration sync is unnecessary; only sync once
    // at the end of the chain.
    for (int iter = 0; iter < protection_layers; iter++) {
        dilateProtectionMaskKernel<<<grd, blk>>>(d_gas_wipe_mask_, nx_, ny_, nz_);
    }
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// computeTemperature → ESM phase change correction (bisection enthalpy inversion)
void ThermalFDM::computeTemperature() {
    if (phase_solver_) {
        // Save current fl as fl_prev (legacy: kernel no longer reads it, kept for diagnostics)
        phase_solver_->storePreviousLiquidFraction();

        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        fdmESMKernel<<<gs, bs>>>(
            d_T_old_,                          // T at start of step (snapshot)
            d_T_,                              // T* post-source+advDiff (IN), T_new (OUT)
            phase_solver_->getLiquidFraction(),// OUT: f_l_new
            d_vof_fill_,
            material_,
            num_cells_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Gas phase wipe with energy tracking
    if (d_vof_fill_ != nullptr) {
        // Sprint-1 M16 (2026-04-25): tuned protection_layers based on dx.
        // 50 layers × 15.6 M cells (M16 domain) = 780 M dilation ops per step
        // → wall time blew up 30× vs M13b small domain. Adjusted to be ~ 50 μm
        // physical reach regardless of dx, by computing layers from dx.
        // 50 μm covers most of keyhole depth (typical 60-80 μm) plus margin.
        // Combined with M13b's "no gentle relaxation in protected zone", the
        // remaining gentle-edge cells around the protected zone don't drain
        // significant energy.
        int protection_phys_um = 50;
        int n_layers = (int)((float)protection_phys_um / (dx_ * 1e6f) + 0.5f);
        if (n_layers < 5) n_layers = 5;
        if (n_layers > 30) n_layers = 30;  // cap to keep wall time reasonable
        computeGasWipeProtectionMask(n_layers);
        float dV = dx_ * dx_ * dx_;
        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        // Accumulate energy (do NOT reset — getGasWipeEnergyRemoved() handles reset)
        fdmGasWipeWithTrackingKernel<<<gs, bs>>>(
            d_T_, d_vof_fill_, d_gas_wipe_mask_, 600.0f,
            material_, dV,
            d_gas_wipe_energy_raw_, num_cells_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void ThermalFDM::addHeatSource(const float* heat_source, float dt) {
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    fdmHeatSourceKernel<<<gs, bs>>>(
        d_T_, heat_source, d_vof_fill_, dt, material_, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ThermalFDM::applyBoundaryConditions(int boundary_type, float boundary_value) {
    if (boundary_type == 2) {
        // Dirichlet on all faces
        for (int face = 0; face < 6; face++) {
            applyFaceThermalBC(face, 2, dt_, dx_, boundary_value);
        }
    }
    // Adiabatic (type 1) is handled by index clamping in the stencil — no action needed
}

void ThermalFDM::applyFaceThermalBC(int face, int bc_type,
                                     float dt, float dx,
                                     float dirichlet_T,
                                     float h_conv, float T_inf,
                                     float emissivity, float T_ambient) {
    if (bc_type == 0 || bc_type == 1) return;  // PERIODIC / ADIABATIC: no action

    // Compute 2D launch dimensions for this face
    int w, h;
    if (face < 2) { w = ny_; h = nz_; }
    else if (face < 4) { w = nx_; h = nz_; }
    else { w = nx_; h = ny_; }

    dim3 block2d(16, 16);
    dim3 grid2d((w + 15) / 16, (h + 15) / 16);

    if (bc_type == 2) {  // DIRICHLET
        fdmDirichletKernel<<<grid2d, block2d>>>(
            d_T_, dirichlet_T, face, nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();
    }
    // TODO: CONVECTIVE (bc_type=3) and RADIATION (bc_type=4)
    // can be added following the same pattern as ThermalLBM
}

void ThermalFDM::applyRadiationBC(float dt, float dx, float eps, float T_amb) {
    // Simplified: radiation on z_max face only (top surface)
    // q_rad = eps * sigma * (T^4 - T_amb^4), applied as dT = -q/(rho*cp*dx) * dt
    // TODO: implement as a face-specific radiation kernel
}

void ThermalFDM::applySubstrateCoolingBC(float dt, float dx, float h, float T_sub) {
    // TODO: implement convective BC at z_min
}

// R7 IMPLICIT OPENFOAM-ALIGNED: Backward-Euler Newton iteration for
// evaporation cooling. Solves locally at each cell:
//
//   T_new - T_old = -β · J_surf(T_new)
//   β = |∇f| · L_vap · factor · dt / (ρ·cp)
//
// J_surf(T) = α · P_sat(T) / √(2π·R·T/M)     (Hertz-Knudsen)
// dJ/dT    = J · [L_vap·M/(R·T²) - 1/(2T)]    (analytic derivative)
//
// This prevents the explicit-scheme overshoot where T lags one step behind
// the cooling rate, which becomes catastrophic when P_sat(T) grows by
// 10× in ~100 K. The implicit form tracks the cooling self-consistently.
//
// Reads fill_level for |∇f| reconstruction (central diff).
__global__ void fdmEvapCoolingImplicitKernel(
    float* __restrict__ T,
    const float* __restrict__ fill_level,
    MaterialProperties mat,
    float dt, float inv_dx,
    float cooling_factor,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = nx * ny * nz;
    if (idx >= num_cells) return;
    if (fill_level == nullptr) return;

    int k = idx / (nx * ny);
    int r = idx - k * nx * ny;
    int j = r / nx;
    int i = r - j * nx;

    int ip = (i < nx - 1) ? i + 1 : i;
    int im = (i > 0)      ? i - 1 : i;
    int jp = (j < ny - 1) ? j + 1 : j;
    int jm = (j > 0)      ? j - 1 : j;
    int kp = (k < nz - 1) ? k + 1 : k;
    int km = (k > 0)      ? k - 1 : k;

    float denom_x = (float)(ip - im); if (denom_x < 1e-6f) denom_x = 1.0f;
    float denom_y = (float)(jp - jm); if (denom_y < 1e-6f) denom_y = 1.0f;
    float denom_z = (float)(kp - km); if (denom_z < 1e-6f) denom_z = 1.0f;

    float dfx = (fill_level[ip + nx*(j  + ny*k )] - fill_level[im + nx*(j  + ny*k )]) * inv_dx / denom_x;
    float dfy = (fill_level[i  + nx*(jp + ny*k )] - fill_level[i  + nx*(jm + ny*k )]) * inv_dx / denom_y;
    float dfz = (fill_level[i  + nx*(j  + ny*kp)] - fill_level[i  + nx*(j  + ny*km)]) * inv_dx / denom_z;
    float grad_f = sqrtf(dfx*dfx + dfy*dfy + dfz*dfz);

    const float grad_cap = 0.866f * inv_dx;
    if (grad_f > grad_cap) grad_f = grad_cap;
    if (grad_f < 0.02f * inv_dx) return;

    float T_old = T[idx];
    float T_boil = mat.T_vaporization;
    if (T_old <= T_boil) return;

    constexpr float R_gas = 8.314f;
    constexpr float P_ref = 101325.0f;
    constexpr float alpha_evap = 0.82f;

    float rho = mat.getDensity(T_old);
    float cp  = mat.getSpecificHeat(T_old);
    float L_vap = mat.L_vaporization;
    float M     = mat.molar_mass;

    // β groups all cell-local constants
    float beta = grad_f * L_vap * cooling_factor * dt / (rho * cp);

    // Newton iteration to solve  T_new - T_old + β · J_surf(T_new) = 0
    float T_new = T_old;
    #pragma unroll 8
    for (int it = 0; it < 8; ++it) {
        float expo = (L_vap * M / R_gas) * (1.0f / T_boil - 1.0f / T_new);
        expo = fminf(expo, 50.0f);
        float P_sat = P_ref * expf(expo);
        float sqrt_fac = sqrtf(2.0f * 3.14159265f * R_gas * T_new / M);
        float J_surf = alpha_evap * P_sat / sqrt_fac;
        // dJ/dT via log-derivative:
        //   d(ln P_sat)/dT = L·M/(R·T²)
        //   d(ln sqrt_fac)/dT = 1/(2T)
        //   dJ/dT = J · [L·M/(R·T²) − 1/(2T)]
        float dJ_dT = J_surf * (L_vap * M / (R_gas * T_new * T_new) - 0.5f / T_new);

        float F  = T_new - T_old + beta * J_surf;
        float Fp = 1.0f + beta * dJ_dT;
        float dT = F / Fp;
        T_new -= dT;
        if (fabsf(dT) < 1e-2f) break;   // 0.01 K tolerance
        if (T_new < T_boil) { T_new = T_boil; break; }
    }

    if (T_new < T_boil) T_new = T_boil;
    if (T_new > T_old)  T_new = T_old;   // cooling only (safety)
    T[idx] = T_new;
}

// Explicit variant kept for fallback / regression testing.
__global__ void fdmEvapCoolingFromFluxKernel(
    float* __restrict__ T,
    const float* __restrict__ J_evap,
    const float* __restrict__ fill_level,
    float L_vap, float dt, float dx,
    MaterialProperties mat,
    float factor, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    float J_vol = J_evap[idx];
    if (J_vol <= 0.0f) return;

    float Tc = T[idx];
    float rho = mat.getDensity(Tc);
    float cp = mat.getSpecificHeat(Tc);
    float q_vol = J_vol * L_vap * factor;
    float dT = q_vol * dt / (rho * cp);
    float T_boil = mat.T_vaporization;
    float T_new = Tc - dT;
    if (T_new < T_boil) T_new = T_boil;
    T[idx] = T_new;
}

void ThermalFDM::applyEvaporationCooling(const float* J_evap, const float* fill,
                                          float dt, float dx, float factor) {
    if (J_evap != nullptr) {
        // R7 IMPLICIT PATH: backward-Euler Newton iteration on T_new.
        // J_evap is still computed upstream (consumed by VOF mass loss).
        // For thermal cooling we re-derive J(T_new) to avoid the explicit
        // lag that lets T overshoot T_boil by hundreds of Kelvin.
        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        float inv_dx = 1.0f / dx;
        fdmEvapCoolingImplicitKernel<<<gs, bs>>>(
            d_T_, d_vof_fill_, material_, dt, inv_dx, factor,
            nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();
    } else {
        // Fallback path: HKL cooling on overheated interface cells only.
        // Uses d_vof_fill_ (set via setVOFFillLevel) to restrict to interface.
        // If no VOF fill is set, no evaporation is applied (no free surface,
        // so evaporation has no physical meaning).
        if (d_vof_fill_ == nullptr) {
            // No VOF context — skip evaporation entirely.
            // (Cannot evaporate without a free surface.)
            return;
        }
        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        fdmEvaporationCoolingKernel<<<gs, bs>>>(
            d_T_, d_vof_fill_, nx_, ny_, nz_, -1,
            dt, dx, material_, factor);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ThermalFDM::applyTemperatureSafetyCap() {
    if (skip_T_cap_ || material_.T_vaporization <= 0.0f) return;
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    fdmTCapKernel<<<gs, bs>>>(d_T_, material_.T_vaporization, num_cells_);
    CUDA_CHECK_KERNEL();
}

void ThermalFDM::applyTemperatureFailsafeCap(float T_max) {
    if (T_max <= 0.0f) return;
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    fdmTCapKernel<<<gs, bs>>>(d_T_, T_max, num_cells_);
    CUDA_CHECK_KERNEL();
}

float* ThermalFDM::getLiquidFraction() {
    return phase_solver_ ? phase_solver_->getLiquidFraction() : nullptr;
}

const float* ThermalFDM::getLiquidFraction() const {
    return phase_solver_ ? phase_solver_->getLiquidFraction() : nullptr;
}

void ThermalFDM::copyTemperatureToHost(float* h_T) const {
    CUDA_CHECK(cudaMemcpy(h_T, d_T_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void ThermalFDM::copyLiquidFractionToHost(float* h_fl) const {
    if (phase_solver_) {
        phase_solver_->copyLiquidFractionToHost(h_fl);
    }
}

float ThermalFDM::computeTotalThermalEnergy(float dx) const {
    // Enthalpy-based reduction (single pass, sensible + latent unified):
    //   E = Σ [H(T_i) − H(T_initial_)] · dV
    // H(T) = sensibleEnthalpyPerVolume + latentEnthalpyPerVolume
    // Subtracting h0 is equivalent to choosing T_initial_ as the reference state;
    // the offset cancels in dE/dt but ensures E=0 at t=0 for diagnostics.
    // f_l(T) is recomputed analytically from T via liquidFraction() — consistent
    // with the equilibrium mushy-zone convention used throughout the solver.
    std::vector<float> h_T(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_T.data(), d_T_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    const float dV = dx * dx * dx;
    const float h0  = material_.enthalpyPerVolume(T_initial_);
    double E = 0.0;
    for (int i = 0; i < num_cells_; i++) {
        E += (material_.enthalpyPerVolume(h_T[i]) - h0) * dV;
    }

    return static_cast<float>(E);
}

// Stub implementations for diagnostics not yet needed
float ThermalFDM::computeEvaporationPower(const float*, float) const { return 0.0f; }
float ThermalFDM::computeRadiationPower(const float*, float, float, float) const { return 0.0f; }
float ThermalFDM::computeSubstratePower(float, float, float) const { return 0.0f; }
float ThermalFDM::computeCapPower(float, float) const { return 0.0f; }
// R7 OPENFOAM-ALIGNED: Evaporation mass flux distributed via |∇f|.
// OpenFOAM laserMeltFoam TEqn.H applies Qv through the VOF transition band
// using `delGradAlpha = |∇α|` as a surface delta function. We mirror that:
//
//   J_surf [kg/(m²·s)] = α · P_sat(T) / √(2π·R·T/M)     (Hertz-Knudsen-Langmuir)
//   J_vol  [kg/(m³·s)] = J_surf × |∇f|                   (CSF distribution)
//
// Units change: the buffer J_evap now carries [kg/(m³·s)] — a volumetric
// mass-source rate. All downstream consumers (fdmEvapCoolingFromFluxKernel
// in this file, applyEvaporationMassLossKernel in vof_solver.cu) must drop
// their historical `/ dx` factor, since |∇f| already has units of 1/length.
//
// Conservation: ∫J_vol·dV = J_surf·∫|∇f|·dV ≈ J_surf·A_surface, i.e. the
// correct surface integral of the Hertz-Knudsen flux.
__global__ void fdmEvapMassFluxKernel(
    const float* __restrict__ T,
    const float* __restrict__ fill_level,
    float* __restrict__ J_evap,
    MaterialProperties mat,
    float inv_dx,               // 1/dx [1/m] for central differences
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    J_evap[idx] = 0.0f;
    if (fill_level == nullptr) return;

    // Decompose idx → (i, j, k)
    int k = idx / (nx * ny);
    int r = idx - k * nx * ny;
    int j = r / nx;
    int i = r - j * nx;

    // |∇f| central diff (one-sided clamp at domain edges).
    // Use 1/dx (inv_dx) so result is in [1/m].
    int ip = (i < nx - 1) ? i + 1 : i;
    int im = (i > 0)      ? i - 1 : i;
    int jp = (j < ny - 1) ? j + 1 : j;
    int jm = (j > 0)      ? j - 1 : j;
    int kp = (k < nz - 1) ? k + 1 : k;
    int km = (k > 0)      ? k - 1 : k;

    float denom_x = (float)(ip - im);
    float denom_y = (float)(jp - jm);
    float denom_z = (float)(kp - km);
    if (denom_x < 1e-6f) denom_x = 1.0f;
    if (denom_y < 1e-6f) denom_y = 1.0f;
    if (denom_z < 1e-6f) denom_z = 1.0f;

    float dfx = (fill_level[ip + nx*(j  + ny*k )] - fill_level[im + nx*(j  + ny*k )]) * inv_dx / denom_x;
    float dfy = (fill_level[i  + nx*(jp + ny*k )] - fill_level[i  + nx*(jm + ny*k )]) * inv_dx / denom_y;
    float dfz = (fill_level[i  + nx*(j  + ny*kp)] - fill_level[i  + nx*(j  + ny*km)]) * inv_dx / denom_z;

    float grad_f = sqrtf(dfx*dfx + dfy*dfy + dfz*dfz);

    // Fix 1 style clamp: |∇f| ≤ √3/(2·dx) (theoretical max for tanh interface)
    const float grad_cap = 0.866f * inv_dx;  // √3/2 / dx
    if (grad_f > grad_cap) grad_f = grad_cap;

    // No evaporation far from an interface
    const float grad_threshold = 0.02f * inv_dx;   // ~2% of the cap
    if (grad_f < grad_threshold) return;

    float Tc = T[idx];
    float T_boil = mat.T_vaporization;
    if (Tc <= T_boil) return;

    // Hertz-Knudsen-Langmuir surface mass flux [kg/(m²·s)]
    constexpr float R_gas = 8.314f;
    constexpr float P_ref = 101325.0f;
    constexpr float alpha_evap = 0.82f;  // R6 OpenFOAM-alignment

    float exponent = (mat.L_vaporization * mat.molar_mass / R_gas)
                   * (1.0f / T_boil - 1.0f / Tc);
    exponent = fminf(exponent, 50.0f);
    float P_sat = P_ref * expf(exponent);

    float J_surf = alpha_evap * P_sat
                 / sqrtf(2.0f * 3.14159265f * R_gas * Tc / mat.molar_mass);

    // Output is volumetric: [kg/(m²·s)] × [1/m] = [kg/(m³·s)]
    J_evap[idx] = J_surf * grad_f;
}

void ThermalFDM::computeEvaporationMassFlux(float* d_J_evap,
                                             const float* fill_level) const {
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    float inv_dx = 1.0f / dx_;
    fdmEvapMassFluxKernel<<<gs, bs>>>(
        d_T_, fill_level, d_J_evap, material_, inv_dx, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Host: Apply sub-surface boiling cap
// ============================================================================
void ThermalFDM::applySubsurfaceBoilingCap(float T_boil, float overshoot_K) {
    if (d_vof_fill_ == nullptr) return;
    float T_cap = T_boil + overshoot_K;
    float dV = dx_ * dx_ * dx_;
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    // Accumulate energy (do NOT reset — getBoilingCapEnergyRemoved() handles reset)
    fdmSubsurfaceBoilingCapKernel<<<gs, bs>>>(
        d_T_, d_vof_fill_, T_cap,
        material_, dV,
        d_boiling_cap_energy_raw_, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Host: Retrieve gas wipe energy removed [J], resets counter
// ============================================================================
double ThermalFDM::getGasWipeEnergyRemoved() {
    unsigned long long h_raw = 0;
    CUDA_CHECK(cudaMemcpy(&h_raw, d_gas_wipe_energy_raw_, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_gas_wipe_energy_raw_, 0, sizeof(unsigned long long)));
    // Interpret as signed (two's complement) for net energy tracking
    long long h_signed = static_cast<long long>(h_raw);
    return static_cast<double>(h_signed) * 1.0e-12;
}

// ============================================================================
// Host: Retrieve boiling cap energy removed [J], resets counter
// ============================================================================
double ThermalFDM::getBoilingCapEnergyRemoved() {
    unsigned long long h_raw = 0;
    CUDA_CHECK(cudaMemcpy(&h_raw, d_boiling_cap_energy_raw_, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_boiling_cap_energy_raw_, 0, sizeof(unsigned long long)));
    return static_cast<double>(h_raw) * 1.0e-12;
}

// ============================================================================
// Host: Compute thermal energy of metal cells only (fill >= 0.01)
// ============================================================================
double ThermalFDM::computeMetalThermalEnergy(float dx) const {
    std::vector<float> h_T(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_T.data(), d_T_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_fill(num_cells_, 1.0f);  // default: all metal if no VOF
    if (d_vof_fill_) {
        CUDA_CHECK(cudaMemcpy(h_fill.data(), d_vof_fill_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float dV = dx * dx * dx;
    double E = 0.0;
    for (int i = 0; i < num_cells_; i++) {
        if (h_fill[i] < 0.01f) continue;  // skip gas cells
        float rho = material_.getDensity(h_T[i]);
        float cp  = material_.getSpecificHeat(h_T[i]);
        E += rho * cp * h_T[i] * dV;
    }

    if (phase_solver_) {
        std::vector<float> h_fl(num_cells_);
        phase_solver_->copyLiquidFractionToHost(h_fl.data());
        for (int i = 0; i < num_cells_; i++) {
            if (h_fill[i] < 0.01f) continue;
            float rho = material_.getDensity(h_T[i]);
            E += h_fl[i] * rho * material_.L_fusion * dV;
        }
    }

    return E;
}

} // namespace physics
} // namespace lbm
