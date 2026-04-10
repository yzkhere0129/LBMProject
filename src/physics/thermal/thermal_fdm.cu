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
// CUDA Kernel: ESM Phase Change Correction
// ============================================================================
// After advection-diffusion gives T*, enforce enthalpy conservation:
//   H = cp·T* + fl_old·L  →  decode(T_new, fl_new)
//
// Identical formula to enthalpySourceTermKernel in thermal_lbm.cu,
// but operates directly on T (no distribution function correction needed).
// ============================================================================
__global__ void fdmESMKernel(
    float* __restrict__ T,
    float* __restrict__ fl,
    const float* __restrict__ fl_prev,
    const float* __restrict__ fill_level,
    MaterialProperties mat,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    // VOF mask: skip gas cells
    if (fill_level != nullptr && fill_level[idx] < 0.01f) {
        fl[idx] = 0.0f;
        return;
    }

    float T_star = T[idx];
    float fl_old = fl_prev[idx];

    float cp    = mat.cp_solid;
    float L     = mat.L_fusion;
    float T_sol = mat.T_solidus;
    float T_liq = mat.T_liquidus;
    float dT_m  = T_liq - T_sol;

    if (dT_m < 1e-8f || L < 1e-6f) return;

    // Total specific enthalpy
    float H = cp * T_star + fl_old * L;

    float H_sol = cp * T_sol;
    float H_liq = cp * T_liq + L;

    float T_new, fl_new;
    if (H <= H_sol) {
        T_new  = H / cp;
        fl_new = 0.0f;
    } else if (H >= H_liq) {
        T_new  = (H - L) / cp;
        fl_new = 1.0f;
    } else {
        fl_new = (H - H_sol) / (cp * dT_m + L);
        fl_new = fmaxf(0.0f, fminf(1.0f, fl_new));
        T_new  = T_sol + fl_new * dT_m;
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
    float Tc = T[idx];

    float T_boil = mat.T_vaporization;
    if (Tc <= T_boil) return;  // no evaporation below boiling point

    // Clausius-Clapeyron saturation pressure
    float L_vap = mat.L_vaporization;  // J/kg
    float M     = mat.molar_mass;       // kg/mol
    constexpr float R_gas = 8.314f;     // J/(mol·K)
    constexpr float P_ref = 101325.0f;  // Pa (1 atm)
    // Evaporation coefficient: 0.18 (physical value for metals).
    // With volumetric evaporation cooling on ALL overheated cells,
    // T naturally stabilizes at ~3400K. P_recoil at 3400K = 137 kPa
    // which exceeds P_Laplace = 70 kPa → keyhole forms.
    constexpr float alpha_evap = 0.18f;

    float exponent = (L_vap * M / R_gas) * (1.0f / T_boil - 1.0f / Tc);
    // Clamp exponent to prevent overflow (exp(88) ≈ FLT_MAX)
    exponent = fminf(exponent, 80.0f);
    float P_sat = P_ref * expf(exponent);

    // Hertz-Knudsen-Langmuir evaporation mass flux [kg/(m²·s)]
    float J_evap = alpha_evap * P_sat / sqrtf(2.0f * 3.14159265f * R_gas * Tc / M);

    // Energy flux [W/m²] with interface area compensation.
    // VOF interface is smeared over 2-3 cells → effective evaporation area
    // is underestimated. Multiplier compensates for this discrete deficit.
    constexpr float evap_area_compensation = 1.0f;
    float q_evap = J_evap * L_vap * cooling_factor * evap_area_compensation;

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
// CUDA Kernel: Hard T_boil cap on ALL cells (not just surface)
// ============================================================================
// Without VOF free-surface tracking, material above T_boil would vaporize
// and leave the domain. The cap enforces this physical constraint:
// any cell with T > T_boil has its excess energy removed (as if vaporized).
// ============================================================================
__global__ void fdmBoilCapKernel(float* T, float T_boil, int num_cells) {
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
      d_T_(o.d_T_), d_T_new_(o.d_T_new_),
      d_vof_fill_(o.d_vof_fill_), d_gas_wipe_mask_(o.d_gas_wipe_mask_),
      skip_T_cap_(o.skip_T_cap_),
      z_periodic_(o.z_periodic_), n_subcycle_(o.n_subcycle_)
{
    o.d_T_ = o.d_T_new_ = nullptr;
    o.d_gas_wipe_mask_ = nullptr;
    o.phase_solver_ = nullptr;
}

void ThermalFDM::allocateMemory() {
    CUDA_CHECK(cudaMalloc(&d_T_,     num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T_new_, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_T_,     0, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_T_new_, 0, num_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gas_wipe_mask_, num_cells_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_gas_wipe_mask_, 0, num_cells_ * sizeof(uint8_t)));
}

void ThermalFDM::freeMemory() {
    if (d_T_)     { cudaFree(d_T_);     d_T_ = nullptr; }
    if (d_T_new_) { cudaFree(d_T_new_); d_T_new_ = nullptr; }
    if (d_gas_wipe_mask_) { cudaFree(d_gas_wipe_mask_); d_gas_wipe_mask_ = nullptr; }
}

void ThermalFDM::initialize(float initial_temp) {
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    fdmFillKernel<<<gs, bs>>>(d_T_, initial_temp, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (phase_solver_) {
        phase_solver_->initializeFromTemperature(d_T_);
    }
}

void ThermalFDM::initialize(const float* temp_field) {
    CUDA_CHECK(cudaMemcpy(d_T_, temp_field,
                          num_cells_ * sizeof(float), cudaMemcpyHostToDevice));
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
    for (int iter = 0; iter < protection_layers; iter++) {
        dilateProtectionMaskKernel<<<grd, blk>>>(d_gas_wipe_mask_, nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// computeTemperature → ESM phase change correction
void ThermalFDM::computeTemperature() {
    if (phase_solver_) {
        // Save current fl as fl_prev
        phase_solver_->storePreviousLiquidFraction();

        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        fdmESMKernel<<<gs, bs>>>(
            d_T_,
            phase_solver_->getLiquidFraction(),
            phase_solver_->getPreviousLiquidFraction(),
            d_vof_fill_,
            material_,
            num_cells_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Gas phase wipe: reset far-field gas cells to ambient temperature
    // Near-interface gas (within 5 layers) is protected to prevent energy theft
    if (d_vof_fill_ != nullptr) {
        computeGasWipeProtectionMask(5);
        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        fdmGasWipeKernel<<<gs, bs>>>(d_T_, d_vof_fill_, d_gas_wipe_mask_,
                                      600.0f, num_cells_);
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

// Kernel: apply evaporation cooling from pre-computed mass flux at ALL interface cells
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
    float J = J_evap[idx];
    if (J <= 0.0f) return;

    float Tc = T[idx];
    float rho = mat.getDensity(Tc);
    float cp = mat.getSpecificHeat(Tc);
    float q = J * L_vap * factor;
    float dT = q * dt / (rho * cp * dx);

    T[idx] = fmaxf(mat.T_vaporization, Tc - dT);
}

void ThermalFDM::applyEvaporationCooling(const float* J_evap, const float* fill,
                                          float dt, float dx, float factor) {
    if (J_evap != nullptr) {
        // Recoil-enabled path: use pre-computed mass flux from computeEvaporationMassFlux
        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        fdmEvapCoolingFromFluxKernel<<<gs, bs>>>(
            d_T_, J_evap, d_vof_fill_,
            material_.L_vaporization, dt, dx, material_, factor, num_cells_);
        CUDA_CHECK_KERNEL();
    } else {
        // Volumetric evaporation: cool ALL cells with T > T_boil.
        // Physics: any overheated metal loses energy through latent heat.
        // dT = q_evap * dt / (rho*cp*dx), clamped at T_boil (no undershoot).
        // This is the primary temperature regulation mechanism.
        int bs = 256, gs = (num_cells_ + bs - 1) / bs;
        fdmEvaporationCoolingKernel<<<gs, bs>>>(
            d_T_, nx_, ny_, nz_, -1,  // z_surface=-1 signals "all cells" mode
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
    // Reduction: E = sum(rho * cp * T * dx^3) + sum(fl * rho * L * dx^3)
    // Simple host-side reduction for now (same as ThermalLBM)
    std::vector<float> h_T(num_cells_);
    CUDA_CHECK(cudaMemcpy(h_T.data(), d_T_, num_cells_ * sizeof(float), cudaMemcpyDeviceToHost));

    float dV = dx * dx * dx;
    double E = 0.0;
    for (int i = 0; i < num_cells_; i++) {
        float rho = material_.getDensity(h_T[i]);
        float cp  = material_.getSpecificHeat(h_T[i]);
        E += rho * cp * h_T[i] * dV;
    }

    if (phase_solver_) {
        std::vector<float> h_fl(num_cells_);
        phase_solver_->copyLiquidFractionToHost(h_fl.data());
        for (int i = 0; i < num_cells_; i++) {
            float rho = material_.getDensity(h_T[i]);
            E += h_fl[i] * rho * material_.L_fusion * dV;
        }
    }

    return static_cast<float>(E);
}

// Stub implementations for diagnostics not yet needed
float ThermalFDM::computeEvaporationPower(const float*, float) const { return 0.0f; }
float ThermalFDM::computeRadiationPower(const float*, float, float, float) const { return 0.0f; }
float ThermalFDM::computeSubstratePower(float, float, float) const { return 0.0f; }
float ThermalFDM::computeCapPower(float, float) const { return 0.0f; }
// Evaporation mass flux kernel for VOF mass loss
__global__ void fdmEvapMassFluxKernel(
    const float* __restrict__ T,
    const float* __restrict__ fill_level,
    float* __restrict__ J_evap,
    MaterialProperties mat,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    J_evap[idx] = 0.0f;

    // Only at interface cells on the metal side
    if (fill_level == nullptr) return;
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) return;

    float Tc = T[idx];
    float T_boil = mat.T_vaporization;
    if (Tc <= T_boil) return;

    // Hertz-Knudsen-Langmuir mass flux
    constexpr float R_gas = 8.314f;
    constexpr float P_ref = 101325.0f;
    constexpr float alpha_evap = 0.18f;

    float exponent = (mat.L_vaporization * mat.molar_mass / R_gas)
                   * (1.0f / T_boil - 1.0f / Tc);
    exponent = fminf(exponent, 50.0f);
    float P_sat = P_ref * expf(exponent);

    J_evap[idx] = alpha_evap * P_sat
                / sqrtf(2.0f * 3.14159265f * R_gas * Tc / mat.molar_mass);
}

void ThermalFDM::computeEvaporationMassFlux(float* d_J_evap,
                                             const float* fill_level) const {
    int bs = 256, gs = (num_cells_ + bs - 1) / bs;
    fdmEvapMassFluxKernel<<<gs, bs>>>(
        d_T_, fill_level, d_J_evap, material_, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace physics
} // namespace lbm
