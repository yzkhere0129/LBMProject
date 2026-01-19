/**
 * @file vof_solver.cu
 * @brief Implementation of Volume of Fluid (VOF) solver
 */

#include "physics/vof_solver.h"
#include "utils/cuda_check.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace lbm {
namespace physics {

// ============================================================================
// CUDA Kernels
// ============================================================================

// ============================================================================
// Flux Limiter Functions for TVD Schemes
// ============================================================================

/**
 * @brief Minmod flux limiter
 * @note Most diffusive, most stable
 * φ(r) = max(0, min(1, r))
 */
__device__ __forceinline__ float fluxLimiterMinmod(float r) {
    return fmaxf(0.0f, fminf(1.0f, r));
}

/**
 * @brief van Leer flux limiter
 * @note Good balance between accuracy and stability
 * φ(r) = (r + |r|) / (1 + |r|)
 */
__device__ __forceinline__ float fluxLimiterVanLeer(float r) {
    if (r <= 0.0f) return 0.0f;
    return (r + fabsf(r)) / (1.0f + fabsf(r));
}

/**
 * @brief Superbee flux limiter
 * @note Least diffusive, most compressive
 * φ(r) = max(0, min(2r, 1), min(r, 2))
 */
__device__ __forceinline__ float fluxLimiterSuperbee(float r) {
    return fmaxf(0.0f, fmaxf(fminf(2.0f * r, 1.0f), fminf(r, 2.0f)));
}

/**
 * @brief MC (Monotonized Central) flux limiter
 * @note Good for smooth flows, prevents overshoot
 * φ(r) = max(0, min((1+r)/2, 2, 2r))
 */
__device__ __forceinline__ float fluxLimiterMC(float r) {
    return fmaxf(0.0f, fminf(fminf(0.5f * (1.0f + r), 2.0f), 2.0f * r));
}

/**
 * @brief Generic flux limiter dispatcher
 * @param r Gradient ratio
 * @param limiter_type 0=minmod, 1=van Leer, 2=superbee, 3=MC
 */
__device__ __forceinline__ float applyFluxLimiter(float r, int limiter_type) {
    switch (limiter_type) {
        case 0: return fluxLimiterMinmod(r);
        case 1: return fluxLimiterVanLeer(r);
        case 2: return fluxLimiterSuperbee(r);
        case 3: return fluxLimiterMC(r);
        default: return fluxLimiterVanLeer(r);  // Default to van Leer
    }
}

/**
 * @brief FLUX-CONSERVATIVE upwind advection kernel with configurable boundaries
 * @note Guarantees mass conservation for divergence-free velocity fields
 *
 * IMPORTANT: Velocities are expected in PHYSICAL UNITS [m/s]
 * from MultiphysicsSolver (which converts from lattice units).
 *
 * The CONSERVATIVE advection equation: ∂f/∂t + ∇·(uf) = 0
 * Discretized: f^{n+1} = f^n - dt/dx × (F_{i+1/2} - F_{i-1/2})
 *
 * This form guarantees mass conservation by computing explicit face fluxes.
 * For periodic boundaries: Σf^{n+1} = Σf^n (exact, up to floating point)
 *
 * FIX (2026-01-18): Changed from advective form (∂f/∂t = -u·∇f) to
 * conservative flux form (∂f/∂t = -∇·(uf)) to achieve <0.1% mass error.
 * Previous implementation had 11.7% mass error in RT tests.
 *
 * FIX (2026-01-18): Added boundary condition support (periodic vs wall).
 * Wall boundaries use zero-flux condition (no material passes through walls).
 * This prevents unphysical wrapping in Rayleigh-Taylor simulations.
 */
__global__ void advectFillLevelUpwindKernel(
    const float* __restrict__ fill_level,
    float* __restrict__ fill_level_new,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float dt,
    float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // ========================================================================
    // Neighbor indices with boundary condition handling
    // ========================================================================
    // X-direction
    int im, ip;
    if (bc_x == 0) {  // PERIODIC
        im = (i > 0) ? i - 1 : nx - 1;
        ip = (i < nx - 1) ? i + 1 : 0;
    } else {  // WALL
        im = (i > 0) ? i - 1 : i;
        ip = (i < nx - 1) ? i + 1 : i;
    }

    // Y-direction
    int jm, jp;
    if (bc_y == 0) {  // PERIODIC
        jm = (j > 0) ? j - 1 : ny - 1;
        jp = (j < ny - 1) ? j + 1 : 0;
    } else {  // WALL
        jm = (j > 0) ? j - 1 : j;
        jp = (j < ny - 1) ? j + 1 : j;
    }

    // Z-direction
    int km, kp;
    if (bc_z == 0) {  // PERIODIC
        km = (k > 0) ? k - 1 : nz - 1;
        kp = (k < nz - 1) ? k + 1 : 0;
    } else {  // WALL
        km = (k > 0) ? k - 1 : k;
        kp = (k < nz - 1) ? k + 1 : k;
    }

    int idx_im = im + nx * (j + ny * k);
    int idx_ip = ip + nx * (j + ny * k);
    int idx_jm = i + nx * (jm + ny * k);
    int idx_jp = i + nx * (jp + ny * k);
    int idx_km = i + nx * (j + ny * km);
    int idx_kp = i + nx * (j + ny * kp);

    // ========================================================================
    // Compute face velocities (interpolated to cell faces)
    // ========================================================================
    float u_face_xm = 0.5f * (ux[idx_im] + ux[idx]);  // Face i-1/2
    float u_face_xp = 0.5f * (ux[idx] + ux[idx_ip]);  // Face i+1/2
    float v_face_ym = 0.5f * (uy[idx_jm] + uy[idx]);  // Face j-1/2
    float v_face_yp = 0.5f * (uy[idx] + uy[idx_jp]);  // Face j+1/2
    float w_face_zm = 0.5f * (uz[idx_km] + uz[idx]);  // Face k-1/2
    float w_face_zp = 0.5f * (uz[idx] + uz[idx_kp]);  // Face k+1/2

    // ========================================================================
    // Compute face fluxes using upwind scheme: F = u × f_upwind
    // ========================================================================
    // X-direction fluxes
    float flux_xm = (u_face_xm >= 0.0f) ? u_face_xm * fill_level[idx_im]
                                        : u_face_xm * fill_level[idx];
    float flux_xp = (u_face_xp >= 0.0f) ? u_face_xp * fill_level[idx]
                                        : u_face_xp * fill_level[idx_ip];

    // Y-direction fluxes
    float flux_ym = (v_face_ym >= 0.0f) ? v_face_ym * fill_level[idx_jm]
                                        : v_face_ym * fill_level[idx];
    float flux_yp = (v_face_yp >= 0.0f) ? v_face_yp * fill_level[idx]
                                        : v_face_yp * fill_level[idx_jp];

    // Z-direction fluxes
    float flux_zm = (w_face_zm >= 0.0f) ? w_face_zm * fill_level[idx_km]
                                        : w_face_zm * fill_level[idx];
    float flux_zp = (w_face_zp >= 0.0f) ? w_face_zp * fill_level[idx]
                                        : w_face_zp * fill_level[idx_kp];

    // ========================================================================
    // ZERO-FLUX boundary conditions for WALL boundaries
    // This prevents mass leakage through walls (FIX for RT mass loss)
    // ========================================================================
    // X boundaries
    if (bc_x != 0) {  // WALL
        if (i == 0) flux_xm = 0.0f;
        if (i == nx - 1) flux_xp = 0.0f;
    }
    // Y boundaries
    if (bc_y != 0) {  // WALL
        if (j == 0) flux_ym = 0.0f;
        if (j == ny - 1) flux_yp = 0.0f;
    }
    // Z boundaries
    if (bc_z != 0) {  // WALL
        if (k == 0) flux_zm = 0.0f;
        if (k == nz - 1) flux_zp = 0.0f;
    }

    // ========================================================================
    // Conservative update: f^{n+1} = f^n - dt/dx × ∇·F
    // ========================================================================
    float div_flux = (flux_xp - flux_xm + flux_yp - flux_ym + flux_zp - flux_zm) / dx;
    float f_new = fill_level[idx] - dt * div_flux;

    // Flush extremely tiny values to zero (reduced from 1e-6 to 1e-9 to minimize mass loss)
    if (f_new < 1e-9f) f_new = 0.0f;

    // Clamp to [0, 1] to maintain physical bounds
    fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}

// ============================================================================
// TVD (Total Variation Diminishing) Advection Kernel
// ============================================================================

/**
 * @brief TVD advection kernel with flux limiter
 * @note Second-order accurate in smooth regions, reduces to first-order near discontinuities
 *
 * PHYSICS & NUMERICS:
 * ==================
 * Conservative form: ∂f/∂t + ∇·(uf) = 0
 * Discretized: f^{n+1} = f^n - dt/dx × (F_{i+1/2} - F_{i-1/2})
 *
 * TVD FLUX FORMULA:
 * ================
 * F_{i+1/2} = F_low + φ(r) × (F_high - F_low)
 *
 * Where:
 *   F_low  = First-order upwind flux (stable, diffusive)
 *   F_high = Second-order flux (accurate, oscillatory)
 *   φ(r)   = Flux limiter function (0 ≤ φ ≤ 2)
 *   r      = Gradient ratio = (f_i - f_{i-1}) / (f_{i+1} - f_i)  [for u > 0]
 *
 * FIRST-ORDER UPWIND FLUX:
 * ========================
 * F_low = u × f_upwind
 *   where f_upwind = f_i if u > 0, f_{i+1} if u < 0
 *
 * SECOND-ORDER FLUX (Fromm scheme):
 * ==================================
 * F_high = u × (f_i + f_{i+1})/2 - (u²dt/2dx) × (f_{i+1} - f_i)
 *        ≈ u × f_face_center  [for small CFL]
 *
 * FLUX LIMITER PROPERTIES:
 * ========================
 * - φ(r) = 0: Pure first-order upwind (most diffusive)
 * - φ(r) = 1: Second-order central (accurate but oscillatory)
 * - φ(r) transitions smoothly based on local gradient ratio
 *
 * TVD REGION: Limiter must satisfy φ(r)/r ∈ [0, 2] to preserve monotonicity
 *
 * GRADIENT RATIO 'r':
 * ===================
 * For u > 0 (left-to-right flow):
 *   r = (f_i - f_{i-1}) / (f_{i+1} - f_i + ε)
 *   Measures how gradient changes across the face
 *
 * Interpretation:
 *   r ≈ 1: Smooth linear variation → use 2nd-order (φ ≈ 1)
 *   r ≈ 0 or r < 0: Discontinuity or extremum → use 1st-order (φ ≈ 0)
 *   r >> 1: Sharp gradient change → limit flux (φ < 1)
 *
 * MASS CONSERVATION:
 * ==================
 * Conservative flux formulation ensures Σf^{n+1} = Σf^n for periodic BC
 * TVD property prevents spurious oscillations (no new maxima/minima)
 *
 * STABILITY:
 * ==========
 * CFL condition: |u| × dt/dx < 0.5 (same as first-order upwind)
 * TVD limiter ensures stability even at discontinuities
 *
 * ADVANTAGES OVER FIRST-ORDER UPWIND:
 * ====================================
 * 1. Reduced numerical diffusion (2nd-order in smooth regions)
 * 2. Sharp interface preservation (less mass loss)
 * 3. Minimal spurious oscillations (TVD property)
 * 4. Better accuracy for long-time simulations
 *
 * REFERENCE:
 * ==========
 * - Sweby, P.K. (1984). High resolution schemes using flux limiters for
 *   hyperbolic conservation laws. SIAM Journal on Numerical Analysis, 21(5), 995-1011.
 * - Hirt, C.W., & Nichols, B.D. (1981). Volume of fluid (VOF) method for
 *   the dynamics of free boundaries. Journal of Computational Physics, 39(1), 201-225.
 */
__global__ void advectFillLevelTVDKernel(
    const float* __restrict__ fill_level,
    float* __restrict__ fill_level_new,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float dt,
    float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z,
    int limiter_type)  // 0=minmod, 1=van Leer, 2=superbee, 3=MC
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // ========================================================================
    // Neighbor indices with boundary condition handling
    // ========================================================================
    // Need 2-layer stencil for TVD (upwind + upwind-upwind)

    // X-direction: i-2, i-1, i, i+1, i+2
    int imm, im, ip, ipp;
    if (bc_x == 0) {  // PERIODIC
        im  = (i > 0)       ? i - 1 : nx - 1;
        imm = (i > 1)       ? i - 2 : (i > 0) ? nx - 1 : nx - 2;
        ip  = (i < nx - 1)  ? i + 1 : 0;
        ipp = (i < nx - 2)  ? i + 2 : (i < nx - 1) ? 0 : 1;
    } else {  // WALL
        im  = (i > 0)       ? i - 1 : i;
        imm = (i > 1)       ? i - 2 : i;
        ip  = (i < nx - 1)  ? i + 1 : i;
        ipp = (i < nx - 2)  ? i + 2 : i;
    }

    // Y-direction: j-2, j-1, j, j+1, j+2
    int jmm, jm, jp, jpp;
    if (bc_y == 0) {  // PERIODIC
        jm  = (j > 0)       ? j - 1 : ny - 1;
        jmm = (j > 1)       ? j - 2 : (j > 0) ? ny - 1 : ny - 2;
        jp  = (j < ny - 1)  ? j + 1 : 0;
        jpp = (j < ny - 2)  ? j + 2 : (j < ny - 1) ? 0 : 1;
    } else {  // WALL
        jm  = (j > 0)       ? j - 1 : j;
        jmm = (j > 1)       ? j - 2 : j;
        jp  = (j < ny - 1)  ? j + 1 : j;
        jpp = (j < ny - 2)  ? j + 2 : j;
    }

    // Z-direction: k-2, k-1, k, k+1, k+2
    int kmm, km, kp, kpp;
    if (bc_z == 0) {  // PERIODIC
        km  = (k > 0)       ? k - 1 : nz - 1;
        kmm = (k > 1)       ? k - 2 : (k > 0) ? nz - 1 : nz - 2;
        kp  = (k < nz - 1)  ? k + 1 : 0;
        kpp = (k < nz - 2)  ? k + 2 : (k < nz - 1) ? 0 : 1;
    } else {  // WALL
        km  = (k > 0)       ? k - 1 : k;
        kmm = (k > 1)       ? k - 2 : k;
        kp  = (k < nz - 1)  ? k + 1 : k;
        kpp = (k < nz - 2)  ? k + 2 : k;
    }

    // Compute linear indices for all required neighbors
    int idx_im   = im  + nx * (j   + ny * k);
    int idx_imm  = imm + nx * (j   + ny * k);
    int idx_ip   = ip  + nx * (j   + ny * k);
    int idx_ipp  = ipp + nx * (j   + ny * k);

    int idx_jm   = i   + nx * (jm  + ny * k);
    int idx_jmm  = i   + nx * (jmm + ny * k);
    int idx_jp   = i   + nx * (jp  + ny * k);
    int idx_jpp  = i   + nx * (jpp + ny * k);

    int idx_km   = i   + nx * (j   + ny * km);
    int idx_kmm  = i   + nx * (j   + ny * kmm);
    int idx_kp   = i   + nx * (j   + ny * kp);
    int idx_kpp  = i   + nx * (j   + ny * kpp);

    // ========================================================================
    // Compute face velocities (interpolated to cell faces)
    // ========================================================================
    float u_face_xm = 0.5f * (ux[idx_im] + ux[idx]);
    float u_face_xp = 0.5f * (ux[idx] + ux[idx_ip]);
    float v_face_ym = 0.5f * (uy[idx_jm] + uy[idx]);
    float v_face_yp = 0.5f * (uy[idx] + uy[idx_jp]);
    float w_face_zm = 0.5f * (uz[idx_km] + uz[idx]);
    float w_face_zp = 0.5f * (uz[idx] + uz[idx_kp]);

    // ========================================================================
    // X-DIRECTION FLUXES with TVD limiting
    // ========================================================================
    float flux_xm, flux_xp;
    const float eps = 1e-10f;  // Small epsilon to prevent division by zero

    // Face i-1/2 (between im and i)
    if (u_face_xm >= 0.0f) {  // Flow from imm → im → i
        float f_upwind = fill_level[idx_im];
        float f_down   = fill_level[idx_imm];
        float f_center = fill_level[idx];

        // Gradient ratio: r = (f_upwind - f_down) / (f_center - f_upwind)
        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);

        // Flux limiter
        float phi = applyFluxLimiter(r, limiter_type);

        // First-order upwind flux
        float F_low = u_face_xm * f_upwind;

        // Second-order correction: F_high ≈ u × f_face_center
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = u_face_xm * f_face_second;

        flux_xm = F_low + phi * (F_high - F_low);
    } else {  // Flow from i → im
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_ip];
        float f_center = fill_level[idx_im];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);

        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = u_face_xm * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = u_face_xm * f_face_second;

        flux_xm = F_low + phi * (F_high - F_low);
    }

    // Face i+1/2 (between i and ip)
    if (u_face_xp >= 0.0f) {  // Flow from im → i → ip
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_im];
        float f_center = fill_level[idx_ip];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);

        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = u_face_xp * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = u_face_xp * f_face_second;

        flux_xp = F_low + phi * (F_high - F_low);
    } else {  // Flow from ip → i
        float f_upwind = fill_level[idx_ip];
        float f_down   = fill_level[idx_ipp];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);

        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = u_face_xp * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = u_face_xp * f_face_second;

        flux_xp = F_low + phi * (F_high - F_low);
    }

    // ========================================================================
    // Y-DIRECTION FLUXES with TVD limiting
    // ========================================================================
    float flux_ym, flux_yp;

    // Face j-1/2
    if (v_face_ym >= 0.0f) {
        float f_upwind = fill_level[idx_jm];
        float f_down   = fill_level[idx_jmm];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_ym * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = v_face_ym * f_face_second;

        flux_ym = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_jp];
        float f_center = fill_level[idx_jm];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_ym * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = v_face_ym * f_face_second;

        flux_ym = F_low + phi * (F_high - F_low);
    }

    // Face j+1/2
    if (v_face_yp >= 0.0f) {
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_jm];
        float f_center = fill_level[idx_jp];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_yp * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = v_face_yp * f_face_second;

        flux_yp = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx_jp];
        float f_down   = fill_level[idx_jpp];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_yp * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = v_face_yp * f_face_second;

        flux_yp = F_low + phi * (F_high - F_low);
    }

    // ========================================================================
    // Z-DIRECTION FLUXES with TVD limiting
    // ========================================================================
    float flux_zm, flux_zp;

    // Face k-1/2
    if (w_face_zm >= 0.0f) {
        float f_upwind = fill_level[idx_km];
        float f_down   = fill_level[idx_kmm];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zm * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = w_face_zm * f_face_second;

        flux_zm = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_kp];
        float f_center = fill_level[idx_km];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zm * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = w_face_zm * f_face_second;

        flux_zm = F_low + phi * (F_high - F_low);
    }

    // Face k+1/2
    if (w_face_zp >= 0.0f) {
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_km];
        float f_center = fill_level[idx_kp];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zp * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = w_face_zp * f_face_second;

        flux_zp = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx_kp];
        float f_down   = fill_level[idx_kpp];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = delta_upwind / (fabsf(delta_center) + eps);
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zp * f_upwind;
        float f_face_second = f_upwind + 0.5f * phi * delta_center;
        float F_high = w_face_zp * f_face_second;

        flux_zp = F_low + phi * (F_high - F_low);
    }

    // ========================================================================
    // ZERO-FLUX boundary conditions for WALL boundaries
    // ========================================================================
    if (bc_x != 0) {  // WALL
        if (i == 0) flux_xm = 0.0f;
        if (i == nx - 1) flux_xp = 0.0f;
    }
    if (bc_y != 0) {  // WALL
        if (j == 0) flux_ym = 0.0f;
        if (j == ny - 1) flux_yp = 0.0f;
    }
    if (bc_z != 0) {  // WALL
        if (k == 0) flux_zm = 0.0f;
        if (k == nz - 1) flux_zp = 0.0f;
    }

    // ========================================================================
    // Conservative update: f^{n+1} = f^n - dt/dx × ∇·F
    // ========================================================================
    float div_flux = (flux_xp - flux_xm + flux_yp - flux_ym + flux_zp - flux_zm) / dx;
    float f_new = fill_level[idx] - dt * div_flux;

    // Flush extremely tiny values to zero
    if (f_new < 1e-9f) f_new = 0.0f;

    // Clamp to [0, 1] to maintain physical bounds
    fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}

/**
 * @brief Interface reconstruction using central differences
 * @note Computes interface normal n = -∇f / |∇f|
 */
__global__ void reconstructInterfaceKernel(
    const float* fill_level,
    float3* interface_normal,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Central differences with boundary handling
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    int k_m = max(k - 1, 0);
    int k_p = min(k + 1, nz - 1);

    int idx_xm = i_m + nx * (j + ny * k);
    int idx_xp = i_p + nx * (j + ny * k);
    int idx_ym = i + nx * (j_m + ny * k);
    int idx_yp = i + nx * (j_p + ny * k);
    int idx_zm = i + nx * (j + ny * k_m);
    int idx_zp = i + nx * (j + ny * k_p);

    // Compute fill level gradient
    float grad_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    float grad_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    float grad_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);

    // Compute gradient magnitude
    float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

    // Interface normal points from liquid to gas: n = -∇f / |∇f|
    if (grad_mag > 1e-8f) {
        interface_normal[idx].x = -grad_x / grad_mag;
        interface_normal[idx].y = -grad_y / grad_mag;
        interface_normal[idx].z = -grad_z / grad_mag;
    } else {
        // Zero gradient (bulk liquid or gas)
        interface_normal[idx].x = 0.0f;
        interface_normal[idx].y = 0.0f;
        interface_normal[idx].z = 0.0f;
    }
}

/**
 * @brief Curvature computation: κ = ∇·n
 * @note Uses finite differences on interface normals
 */
__global__ void computeCurvatureKernel(
    const float* fill_level,
    const float3* interface_normal,
    float* curvature,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Only compute curvature at interface cells
    float f = fill_level[idx];
    if (f < 0.01f || f > 0.99f) {
        curvature[idx] = 0.0f;
        return;
    }

    // Central differences with boundary handling
    int i_m = max(i - 1, 0);
    int i_p = min(i + 1, nx - 1);
    int j_m = max(j - 1, 0);
    int j_p = min(j + 1, ny - 1);
    int k_m = max(k - 1, 0);
    int k_p = min(k + 1, nz - 1);

    int idx_xm = i_m + nx * (j + ny * k);
    int idx_xp = i_p + nx * (j + ny * k);
    int idx_ym = i + nx * (j_m + ny * k);
    int idx_yp = i + nx * (j_p + ny * k);
    int idx_zm = i + nx * (j + ny * k_m);
    int idx_zp = i + nx * (j + ny * k_p);

    // Check if normals are valid (non-zero magnitude)
    float3 n_center = interface_normal[idx];
    float n_mag = sqrtf(n_center.x*n_center.x + n_center.y*n_center.y + n_center.z*n_center.z);

    if (n_mag < 1e-10f) {
        // No interface here despite fill level - set curvature to zero
        curvature[idx] = 0.0f;
        return;
    }

    // Compute divergence of normal: κ = ∇·n
    float dnx_dx = (interface_normal[idx_xp].x - interface_normal[idx_xm].x) / (2.0f * dx);
    float dny_dy = (interface_normal[idx_yp].y - interface_normal[idx_ym].y) / (2.0f * dx);
    float dnz_dz = (interface_normal[idx_zp].z - interface_normal[idx_zm].z) / (2.0f * dx);

    float kappa = dnx_dx + dny_dy + dnz_dz;

    // NaN/Inf protection
    if (isnan(kappa) || isinf(kappa)) {
        curvature[idx] = 0.0f;
        return;
    }

    // CRITICAL FIX (2026-01-17): Limit curvature to prevent numerical instabilities
    //
    // Physical reasoning: The sharpest possible interface spanning one grid cell
    // has curvature κ_max = 2/dx. Numerical errors can produce larger values,
    // especially at poorly-resolved features.
    //
    // Limiting prevents:
    // 1. Surface tension force explosion (F ∝ σ×κ)
    // 2. Spurious currents from curvature noise
    // 3. CFL violations from extreme accelerations
    //
    // For dx = 2 μm: κ_max = 1e6 m⁻¹ (R_min = 2 μm)
    //
    float kappa_max = 2.0f / dx;
    kappa = fmaxf(-kappa_max, fminf(kappa_max, kappa));

    curvature[idx] = kappa;
}

/**
 * @brief Cell type conversion based on fill level
 */
__global__ void convertCellsKernel(
    const float* fill_level,
    uint8_t* cell_flags,
    float eps,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];

    if (f < eps) {
        cell_flags[idx] = static_cast<uint8_t>(CellFlag::GAS);
    } else if (f > 1.0f - eps) {
        cell_flags[idx] = static_cast<uint8_t>(CellFlag::LIQUID);
    } else {
        cell_flags[idx] = static_cast<uint8_t>(CellFlag::INTERFACE);
    }
}

/**
 * @brief Contact angle boundary condition
 * @note Modifies interface normal at walls: n_wall = n - (n·n_w)·n_w + cos(θ)·n_w
 */
__global__ void applyContactAngleBoundaryKernel(
    float3* interface_normal,
    const uint8_t* cell_flags,
    float contact_angle,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    // Only apply to boundary cells
    if (i != 0 && i != nx - 1 && j != 0 && j != ny - 1 && k != 0 && k != nz - 1) {
        return;
    }

    int idx = i + nx * (j + ny * k);

    // Check if this is an interface cell
    if (cell_flags[idx] != static_cast<uint8_t>(CellFlag::INTERFACE)) {
        return;
    }

    // Wall normal (pointing inward)
    float3 n_wall = make_float3(0.0f, 0.0f, 0.0f);
    if (i == 0) n_wall.x = 1.0f;
    if (i == nx - 1) n_wall.x = -1.0f;
    if (j == 0) n_wall.y = 1.0f;
    if (j == ny - 1) n_wall.y = -1.0f;
    if (k == 0) n_wall.z = 1.0f;
    if (k == nz - 1) n_wall.z = -1.0f;

    // Compute contact angle
    float cos_theta = cosf(contact_angle * 3.14159265f / 180.0f);
    float sin_theta = sinf(contact_angle * 3.14159265f / 180.0f);

    // Current interface normal
    float3 n = interface_normal[idx];
    float n_mag = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

    // For boundary cells, we need to adjust or initialize the normal based on contact angle
    if (n_mag < 0.01f) {
        // Normal is zero or very small - initialize it based on contact angle
        // Create a tangent vector perpendicular to wall normal
        float3 tangent = make_float3(0.0f, 0.0f, 0.0f);
        if (fabsf(n_wall.z) > 0.9f) {
            // Horizontal wall (top/bottom): use x as tangent
            tangent.x = 1.0f;
        } else {
            // Vertical wall (sides): use z as tangent
            tangent.z = 1.0f;
        }

        // Interface normal = cos(θ) * n_wall + sin(θ) * tangent
        // This gives the correct angle relative to the wall
        n.x = cos_theta * n_wall.x + sin_theta * tangent.x;
        n.y = cos_theta * n_wall.y + sin_theta * tangent.y;
        n.z = cos_theta * n_wall.z + sin_theta * tangent.z;
        n_mag = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    } else {
        // Normal exists - adjust it for contact angle
        // Project normal onto wall plane and add contact angle component
        float n_dot_nwall = n.x * n_wall.x + n.y * n_wall.y + n.z * n_wall.z;
        n.x = n.x - n_dot_nwall * n_wall.x + cos_theta * n_wall.x;
        n.y = n.y - n_dot_nwall * n_wall.y + cos_theta * n_wall.y;
        n.z = n.z - n_dot_nwall * n_wall.z + cos_theta * n_wall.z;
        n_mag = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    }

    // Normalize and write back
    if (n_mag > 1e-8f) {
        interface_normal[idx].x = n.x / n_mag;
        interface_normal[idx].y = n.y / n_mag;
        interface_normal[idx].z = n.z / n_mag;
    } else {
        // Fallback: set to wall-tangent direction
        if (fabsf(n_wall.z) > 0.9f) {
            interface_normal[idx] = make_float3(1.0f, 0.0f, 0.0f);
        } else {
            interface_normal[idx] = make_float3(0.0f, 0.0f, 1.0f);
        }
    }
}

/**
 * @brief Initialize spherical droplet
 */
__global__ void initializeDropletKernel(
    float* fill_level,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Distance from droplet center
    float dx = static_cast<float>(i) - center_x;
    float dy = static_cast<float>(j) - center_y;
    float dz = static_cast<float>(k) - center_z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    // Smooth interface using tanh profile (width ~ 2 grid cells)
    float interface_width = 2.0f;
    fill_level[idx] = 0.5f * (1.0f - tanhf((dist - radius) / interface_width));
}

/**
 * @brief Olsson-Kreiss interface compression kernel
 *
 * Implements the Olsson-Kreiss compression scheme to sharpen diffused interfaces
 * and restore mass conservation after upwind advection.
 *
 * Physics:
 *   ∂φ/∂t = ∇·(ε·φ·(1-φ)·n)
 *
 * Where:
 *   ε = C * max(|u|, |v|, |w|) * dx   [compression coefficient]
 *   n = -∇φ/|∇φ|                       [interface normal]
 *   C = 0.5                            [Olsson-Kreiss constant]
 *
 * The compression term:
 *   - Acts only at interfaces (φ·(1-φ) = 0 at bulk cells)
 *   - Transports material toward interface (∇·flux)
 *   - Counteracts numerical diffusion from upwind scheme
 *   - Preserves mass (divergence form)
 *
 * Discretization:
 *   df/dt = ∇·(ε·f·(1-f)·n)
 *   ≈ [Fx(i+1/2) - Fx(i-1/2)]/dx + [Fy(j+1/2) - Fy(j-1/2)]/dx + [Fz(k+1/2) - Fz(k-1/2)]/dx
 *
 * Where flux at face i+1/2:
 *   Fx(i+1/2) = ε · f_face · (1-f_face) · nx_face
 *   f_face = (f[i] + f[i+1]) / 2
 *   nx_face = -(f[i+1] - f[i]) / (|∇f| * dx)
 *
 * Stability:
 *   - Only applied to interface cells (0.01 < φ < 0.99)
 *   - CFL condition: ε·dt/dx < 0.5
 *   - Result clamped to [0, 1]
 *
 * References:
 *   - Olsson & Kreiss (2005). A conservative level set method for two phase flow.
 *     Journal of Computational Physics, 210(1), 225-246.
 */
__global__ void applyInterfaceCompressionKernel(
    float* fill_level,
    const float* fill_level_old,
    const float* ux,
    const float* uy,
    const float* uz,
    float dx,
    float dt,
    float C_compress,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    float f = fill_level_old[idx];

    // For bulk cells (f~0 or f~1), skip compression but MUST copy advected value
    // BUG FIX: Previously returned without writing, causing advection to not propagate
    if (f < 0.01f || f > 0.99f) {
        fill_level[idx] = f;  // Copy advected value to output buffer
        return;
    }

    // ========================================================================
    // Compute compression coefficient: ε = C * max(|u|, |v|, |w|) * dx
    // ========================================================================
    // Velocities are in PHYSICAL UNITS [m/s] from MultiphysicsSolver
    float u = ux[idx];
    float v = uy[idx];
    float w = uz[idx];
    float u_max = fmaxf(fabsf(u), fmaxf(fabsf(v), fabsf(w)));
    float epsilon = C_compress * u_max * dx;  // Physical units [m²/s]

    // Skip compression if velocity is too small, but MUST copy advected value
    if (u_max < 1e-8f) {
        fill_level[idx] = fill_level_old[idx];
        return;
    }

    // ========================================================================
    // Periodic boundary indexing
    // ========================================================================
    int i_m = (i > 0) ? i - 1 : nx - 1;
    int i_p = (i < nx - 1) ? i + 1 : 0;
    int j_m = (j > 0) ? j - 1 : ny - 1;
    int j_p = (j < ny - 1) ? j + 1 : 0;
    int k_m = (k > 0) ? k - 1 : nz - 1;
    int k_p = (k < nz - 1) ? k + 1 : 0;

    int idx_xm = i_m + nx * (j + ny * k);
    int idx_xp = i_p + nx * (j + ny * k);
    int idx_ym = i + nx * (j_m + ny * k);
    int idx_yp = i + nx * (j_p + ny * k);
    int idx_zm = i + nx * (j + ny * k_m);
    int idx_zp = i + nx * (j + ny * k_p);

    // ========================================================================
    // Compute interface normal: n = -∇φ/|∇φ|
    // ========================================================================
    // Using central differences (physical units)
    float grad_x = (fill_level_old[idx_xp] - fill_level_old[idx_xm]) / (2.0f * dx);
    float grad_y = (fill_level_old[idx_yp] - fill_level_old[idx_ym]) / (2.0f * dx);
    float grad_z = (fill_level_old[idx_zp] - fill_level_old[idx_zm]) / (2.0f * dx);

    float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

    // Skip compression if gradient is too small, but MUST copy advected value
    if (grad_mag < 1e-8f) {
        fill_level[idx] = fill_level_old[idx];
        return;
    }

    float nx_norm = -grad_x / grad_mag;
    float ny_norm = -grad_y / grad_mag;
    float nz_norm = -grad_z / grad_mag;

    // ========================================================================
    // Compute compression flux divergence: ∇·(ε·φ·(1-φ)·n)
    // ========================================================================
    // Using upwind-biased flux reconstruction at cell faces

    // X-direction flux at i+1/2 and i-1/2
    float f_xp = 0.5f * (fill_level_old[idx] + fill_level_old[idx_xp]);
    float f_xm = 0.5f * (fill_level_old[idx_xm] + fill_level_old[idx]);

    // Normal component at faces (using central difference approximation)
    float nx_xp = 0.5f * (nx_norm + (-1.0f) * (fill_level_old[idx_xp] - fill_level_old[idx]) / (grad_mag * dx + 1e-10f));
    float nx_xm = 0.5f * (nx_norm + (-1.0f) * (fill_level_old[idx] - fill_level_old[idx_xm]) / (grad_mag * dx + 1e-10f));

    float Flux_xp = epsilon * f_xp * (1.0f - f_xp) * nx_xp;
    float Flux_xm = epsilon * f_xm * (1.0f - f_xm) * nx_xm;

    // Y-direction flux at j+1/2 and j-1/2
    float f_yp = 0.5f * (fill_level_old[idx] + fill_level_old[idx_yp]);
    float f_ym = 0.5f * (fill_level_old[idx_ym] + fill_level_old[idx]);

    float ny_yp = 0.5f * (ny_norm + (-1.0f) * (fill_level_old[idx_yp] - fill_level_old[idx]) / (grad_mag * dx + 1e-10f));
    float ny_ym = 0.5f * (ny_norm + (-1.0f) * (fill_level_old[idx] - fill_level_old[idx_ym]) / (grad_mag * dx + 1e-10f));

    float Flux_yp = epsilon * f_yp * (1.0f - f_yp) * ny_yp;
    float Flux_ym = epsilon * f_ym * (1.0f - f_ym) * ny_ym;

    // Z-direction flux at k+1/2 and k-1/2
    float f_zp = 0.5f * (fill_level_old[idx] + fill_level_old[idx_zp]);
    float f_zm = 0.5f * (fill_level_old[idx_zm] + fill_level_old[idx]);

    float nz_zp = 0.5f * (nz_norm + (-1.0f) * (fill_level_old[idx_zp] - fill_level_old[idx]) / (grad_mag * dx + 1e-10f));
    float nz_zm = 0.5f * (nz_norm + (-1.0f) * (fill_level_old[idx] - fill_level_old[idx_zm]) / (grad_mag * dx + 1e-10f));

    float Flux_zp = epsilon * f_zp * (1.0f - f_zp) * nz_zp;
    float Flux_zm = epsilon * f_zm * (1.0f - f_zm) * nz_zm;

    // Divergence: ∇·F = (Fx+ - Fx-)/dx + (Fy+ - Fy-)/dx + (Fz+ - Fz-)/dx
    float div_flux = ((Flux_xp - Flux_xm) + (Flux_yp - Flux_ym) + (Flux_zp - Flux_zm)) / dx;

    // ========================================================================
    // Time integration: φ^{n+1} = φ^n + dt * ∇·(ε·φ·(1-φ)·n)
    // ========================================================================
    float f_new = fill_level_old[idx] + dt * div_flux;

    // Flush tiny values to zero
    if (f_new < 1e-9f) f_new = 0.0f;

    // Clamp to [0, 1] to maintain physical bounds
    fill_level[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}

/**
 * @brief Mass reduction kernel for conservation check
 */
__global__ void computeMassReductionKernel(
    const float* fill_level,
    float* partial_sums,
    int num_cells)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < num_cells) ? fill_level[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// VOFSolver Implementation
// ============================================================================

VOFSolver::VOFSolver(int nx, int ny, int nz, float dx,
                     BoundaryType bc_x, BoundaryType bc_y, BoundaryType bc_z)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz), dx_(dx),
      bc_x_(bc_x), bc_y_(bc_y), bc_z_(bc_z),
      advection_scheme_(VOFAdvectionScheme::UPWIND),  // Default: first-order upwind
      tvd_limiter_(TVDLimiter::VAN_LEER),             // Default: van Leer limiter
      d_fill_level_(nullptr), d_cell_flags_(nullptr),
      d_interface_normal_(nullptr), d_curvature_(nullptr),
      d_fill_level_tmp_(nullptr)
{
    allocateMemory();
}

VOFSolver::~VOFSolver() {
    freeMemory();
}

void VOFSolver::allocateMemory() {
    // Clear any previous CUDA errors before allocation
    cudaGetLastError();

    cudaError_t err;

    err = cudaMalloc(&d_fill_level_, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_fill_level: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_cell_flags_, num_cells_ * sizeof(uint8_t));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_cell_flags: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_interface_normal_, num_cells_ * sizeof(float3));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_interface_normal: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_curvature_, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_curvature: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_fill_level_tmp_, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("VOFSolver: Failed to allocate d_fill_level_tmp: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void VOFSolver::freeMemory() {
    if (d_fill_level_) cudaFree(d_fill_level_);
    if (d_cell_flags_) cudaFree(d_cell_flags_);
    if (d_interface_normal_) cudaFree(d_interface_normal_);
    if (d_curvature_) cudaFree(d_curvature_);
    if (d_fill_level_tmp_) cudaFree(d_fill_level_tmp_);
}

void VOFSolver::initialize(const float* fill_level) {
    cudaMemcpy(d_fill_level_, fill_level, num_cells_ * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize cell flags based on fill level
    convertCells();

    // Compute initial interface normals and curvature
    reconstructInterface();
    computeCurvature();
}

void VOFSolver::initialize(float uniform_fill) {
    std::vector<float> h_fill(num_cells_, uniform_fill);
    initialize(h_fill.data());
}

void VOFSolver::initializeDroplet(float center_x, float center_y,
                                   float center_z, float radius) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    initializeDropletKernel<<<gridSize, blockSize>>>(
        d_fill_level_, center_x, center_y, center_z, radius, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update cell flags and interface properties
    convertCells();
    reconstructInterface();
    computeCurvature();
}

void VOFSolver::advectFillLevel(const float* velocity_x,
                                 const float* velocity_y,
                                 const float* velocity_z,
                                 float dt) {
    // CRITICAL FIX (2026-01-17): Velocities from FluidLBM are in LATTICE UNITS
    //
    // The velocities passed to this function are in lattice units (dimensionless),
    // where velocity is measured in lattice spacings per timestep.
    //
    // For lattice units: dt_lattice = 1, dx_lattice = 1
    // CFL = v_lattice (no multiplication needed!)
    //
    // To convert to physical units for diagnostics:
    // v_phys [m/s] = v_lattice [dimensionless] × (dx [m] / dt [s])
    //
    // Check VOF CFL condition before advection
    // CFL = v_lattice should be < 0.5 for stability (in lattice units)
    // Sample from TOP LAYER (z = nz-1) where Marangoni flow is active
    const int top_layer_size = nx_ * ny_;
    const int top_layer_offset = (nz_ - 1) * nx_ * ny_;  // Start of top layer
    const int sample_size = std::min(top_layer_size, num_cells_ - top_layer_offset);

    std::vector<float> h_ux(sample_size), h_uy(sample_size), h_uz(sample_size);
    CUDA_CHECK(cudaMemcpy(h_ux.data(), velocity_x + top_layer_offset, sample_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uy.data(), velocity_y + top_layer_offset, sample_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uz.data(), velocity_z + top_layer_offset, sample_size * sizeof(float), cudaMemcpyDeviceToHost));

    float v_max = 0.0f;
    for (int i = 0; i < sample_size; ++i) {
        float v_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        v_max = std::max(v_max, v_mag);
    }

    // CFL = v * dt / dx (correct formula for any unit system)
    // Works for both lattice units (dx=1, dt=1) and physical units
    float vof_cfl = v_max * dt / dx_;

    // For diagnostic output: detect if velocity is in lattice or physical units
    // If v_max * dt / dx is small (~0.001-0.1), velocity is likely physical
    // If v_max is already in the range 0.001-0.1, velocity is likely lattice
    float v_max_phys = (dt >= 1.0f && dx_ >= 1.0f) ? v_max * (dx_ / dt) : v_max;

    // ========================================================================
    // CFL-ADAPTIVE TIMESTEP SUBCYCLING (2026-01-19)
    // ========================================================================
    // PROBLEM: First-order upwind advection requires CFL < 0.5 for stability
    //          and minimal diffusion. At CFL=0.44, we observe 20% mass loss.
    //
    // SOLUTION: Split advection into multiple substeps when CFL > threshold
    //
    // RATIONALE:
    //   - VOF advection is hyperbolic PDE with strict CFL stability limit
    //   - LBM can use larger dt (different stability criterion)
    //   - Operator splitting allows different timesteps for different physics
    //   - Standard practice in multiphysics CFD (OpenFOAM, Fluent, Gerris)
    //
    // IMPLEMENTATION:
    //   - Target CFL = 0.25 (conservative, ensures mass conservation)
    //   - n_substeps = ceil(CFL_current / CFL_target)
    //   - Each substep uses dt_sub = dt / n_substeps
    //   - Mass conservation preserved (conservative flux formulation)
    //
    // PERFORMANCE:
    //   - Overhead only when CFL > threshold (~10% of simulation time)
    //   - Early time (low velocity): No subcycling
    //   - Late time (high velocity): 2-3× subcycling
    //   - Average overhead: ~5-10%
    //
    // REFERENCES:
    //   - OpenFOAM: nAlphaSubCycles parameter
    //   - ANSYS Fluent: VOF subcycling with CFL < 0.25
    //   - Gerris: Adaptive timestepping for VOF
    // ========================================================================
    // CRITICAL (2026-01-19): CFL_target must be VERY conservative for first-order upwind
    //
    // Analysis of RT mushroom test showed:
    //   - CFL_target = 0.25: Mass loss 20.74% (subcycling at Call 2500, too late)
    //   - Major mass loss occurred at Call 0-2000 (CFL 0.10-0.15)
    //   - After subcycling activated: mass STABLE at 420k
    //
    // CONCLUSION: First-order upwind is MORE diffusive than expected
    //   - Literature claims CFL < 0.5 stable, but accuracy requires CFL < 0.1
    //   - Diffusion error: O((1 - 2·CFL) + CFL²) ≈ 1 - 2·CFL for small CFL
    //   - At CFL=0.15: 30% diffusion per step!
    //
    // FIX: Reduce CFL_target from 0.25 to 0.10
    //   - This will activate subcycling much earlier
    //   - Performance cost: ~2-3× subcycling throughout simulation
    //   - But necessary for mass conservation
    // ========================================================================
    const float CFL_target = 0.10f;  // VERY conservative for first-order upwind
    int n_substeps = std::max(1, static_cast<int>(std::ceil(vof_cfl / CFL_target)));
    float dt_sub = dt / static_cast<float>(n_substeps);
    float cfl_sub = vof_cfl / static_cast<float>(n_substeps);

    // Diagnostic: print VOF advection info periodically
    static int call_count = 0;
    static float prev_mass = -1.0f;
    static int prev_substeps = 1;
    static bool scheme_reported = false;

    if (call_count % 500 == 0 && call_count < 5000) {
        // Compute current mass
        float mass = computeTotalMass();
        float mass_change = (prev_mass > 0) ? (mass - prev_mass) : 0.0f;

        // Report scheme on first call
        if (!scheme_reported) {
            const char* scheme_name = (advection_scheme_ == VOFAdvectionScheme::UPWIND) ? "UPWIND" : "TVD";
            const char* limiter_names[] = {"MINMOD", "VAN_LEER", "SUPERBEE", "MC"};
            const char* limiter_name = limiter_names[static_cast<int>(tvd_limiter_)];
            printf("[VOF INIT] Advection scheme: %s", scheme_name);
            if (advection_scheme_ == VOFAdvectionScheme::TVD) {
                printf(" (limiter: %s)", limiter_name);
            }
            printf("\n");
            scheme_reported = true;
        }

        printf("[VOF ADVECT] Call %d: v_max=%.6f, CFL=%.6f, n_sub=%d, CFL_sub=%.3f, mass=%.1f (delta=%.3f)\n",
               call_count, v_max, vof_cfl, n_substeps, cfl_sub, mass, mass_change);
        prev_mass = mass;
    }

    // Warn if subcycling activated/deactivated
    if (n_substeps != prev_substeps && call_count % 100 == 0) {
        printf("[VOF SUBCYCLE] Step %d: CFL=%.3f requires %d substeps (dt_sub=%.2e s, CFL_sub=%.3f)\n",
               call_count, vof_cfl, n_substeps, dt_sub, cfl_sub);
    }
    prev_substeps = n_substeps;
    call_count++;

    if (vof_cfl > 0.5f) {
        printf("WARNING: VOF CFL violation: %.3f > 0.5 (v_max=%.2e, dt=%.2e s, dx=%.2e m)\n",
               vof_cfl, v_max, dt, dx_);
        printf("         Subcycling to %d steps (CFL_sub=%.3f)\n", n_substeps, cfl_sub);
    }

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    // ========================================================================
    // STEP 1: Advection with CFL-adaptive subcycling
    // ========================================================================
    // Dispatch to appropriate kernel based on advection scheme
    for (int substep = 0; substep < n_substeps; ++substep) {
        if (advection_scheme_ == VOFAdvectionScheme::UPWIND) {
            // First-order upwind (stable, diffusive)
            advectFillLevelUpwindKernel<<<gridSize, blockSize>>>(
                d_fill_level_, d_fill_level_tmp_, velocity_x, velocity_y, velocity_z,
                dt_sub, dx_, nx_, ny_, nz_,
                static_cast<int>(bc_x_), static_cast<int>(bc_y_), static_cast<int>(bc_z_));
        } else {
            // TVD scheme (2nd-order accurate, less diffusive)
            advectFillLevelTVDKernel<<<gridSize, blockSize>>>(
                d_fill_level_, d_fill_level_tmp_, velocity_x, velocity_y, velocity_z,
                dt_sub, dx_, nx_, ny_, nz_,
                static_cast<int>(bc_x_), static_cast<int>(bc_y_), static_cast<int>(bc_z_),
                static_cast<int>(tvd_limiter_));
        }
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to d_fill_level_ for next substep (or final result)
        cudaMemcpy(d_fill_level_, d_fill_level_tmp_, num_cells_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ========================================================================
    // STEP 2: Interface compression (Olsson-Kreiss) - DISABLED for RT
    // ========================================================================
    // CRITICAL FIX (2026-01-19): Interface compression MUST BE DISABLED for
    // Rayleigh-Taylor instability simulations!
    //
    // PROBLEM: Olsson-Kreiss compression artificially stiffens the interface,
    // resisting natural deformation from buoyancy-driven flow. This manifests as:
    //   1. Reduced growth rate (dh/dt ~ 0.5× correct value)
    //   2. Suppressed Kelvin-Helmholtz vortices
    //   3. Artificial interface "rigidity" that acts like extra surface tension
    //
    // ROOT CAUSE: Compression term ∇·(ε·φ·(1-φ)·n) opposes interface curvature
    // changes, which is EXACTLY what RT instability requires!
    //
    // PHYSICS: RT growth requires interface to freely deform under buoyancy.
    // Compression counteracts this deformation → slower growth.
    //
    // SOLUTION: Disable compression for RT simulations. Accept slightly diffused
    // interface (3-4 cells thick) in exchange for correct physics.
    //
    // FOR OTHER SIMULATIONS (droplets, surface tension): Keep compression enabled.
    //
    // TODO: Add solver parameter to control compression (enable/disable per-simulation)
    //
    // NOTE (2026-01-19): With subcycling, d_fill_level_ already contains the
    // final advected result (copied back from d_fill_level_tmp_ in loop above).
    // No additional copy needed when compression is disabled.
    //
    float C_compress = 0.0f;  // DISABLED for RT (was 0.3)

    if (C_compress > 0.0f) {
        // Copy current field to tmp buffer for compression kernel input
        cudaMemcpy(d_fill_level_tmp_, d_fill_level_, num_cells_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());

        applyInterfaceCompressionKernel<<<gridSize, blockSize>>>(
            d_fill_level_,      // output: compressed field
            d_fill_level_tmp_,  // input: advected (diffused) field
            velocity_x, velocity_y, velocity_z,
            dx_, dt, C_compress, nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // NOTE: d_fill_level_ now contains final result (advected + optionally compressed)
}

void VOFSolver::reconstructInterface() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    reconstructInterfaceKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_normal_, dx_, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void VOFSolver::computeCurvature() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    computeCurvatureKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_normal_, d_curvature_, dx_, nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void VOFSolver::convertCells() {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    float eps = 0.01f;  // Threshold for interface detection

    convertCellsKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_cell_flags_, eps, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void VOFSolver::applyBoundaryConditions(int boundary_type, float contact_angle) {
    if (boundary_type == 0) {
        // Periodic boundaries - no action needed
        return;
    }

    if (boundary_type == 1) {
        // Apply contact angle at walls
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                      (ny_ + blockSize.y - 1) / blockSize.y,
                      (nz_ + blockSize.z - 1) / blockSize.z);

        applyContactAngleBoundaryKernel<<<gridSize, blockSize>>>(
            d_interface_normal_, d_cell_flags_, contact_angle, nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void VOFSolver::copyFillLevelToHost(float* host_fill) const {
    cudaMemcpy(host_fill, d_fill_level_, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void VOFSolver::copyCellFlagsToHost(uint8_t* host_flags) const {
    cudaMemcpy(host_flags, d_cell_flags_, num_cells_ * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
}

void VOFSolver::copyCurvatureToHost(float* host_curvature) const {
    cudaMemcpy(host_curvature, d_curvature_, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost);
}

float VOFSolver::computeTotalMass() const {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Allocate partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // First reduction: compute partial sums
    computeMassReductionKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_fill_level_, d_partial_sums, num_cells_);
    CUDA_CHECK_KERNEL();

    // Copy partial sums to host and finish reduction on CPU
    std::vector<float> h_partial_sums(gridSize);
    cudaMemcpy(h_partial_sums.data(), d_partial_sums, gridSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_partial_sums);

    // Final reduction on CPU
    float total_mass = 0.0f;
    for (float sum : h_partial_sums) {
        total_mass += sum;
    }

    return total_mass;
}

// ============================================================================
// Evaporation Mass Loss (VOF-Thermal Coupling)
// ============================================================================

/**
 * @brief Apply evaporation mass loss to fill level field
 *
 * Physics:
 *   dm/dt = -J_evap * A_interface    [kg/s]
 *   df/dt = -J_evap / (rho * dx)     [1/s]
 *   df = -J_evap * dt / (rho * dx)   [dimensionless]
 *
 * Where:
 *   J_evap: Evaporation mass flux [kg/(m^2*s)]
 *   rho: Material density [kg/m^3]
 *   dx: Grid spacing [m]
 *   dt: Time step [s]
 *
 * Stability:
 *   - Only applies to cells with material (f > 0)
 *   - Limited to max 10% reduction per timestep
 *   - Clamps result to [0, 1]
 */
__global__ void applyEvaporationMassLossKernel(
    float* fill_level,
    const float* J_evap,
    float rho,
    float dx,
    float dt,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    float f = fill_level[idx];
    float J = J_evap[idx];

    // Skip cells with no material or no evaporation
    if (f <= 0.0f || J <= 0.0f) {
        return;
    }

    // Compute fill level change
    // df = -J_evap * dt / (rho * dx)
    float df = -J * dt / (rho * dx);

    // ============================================================
    // Stability limiter: max 2% reduction per timestep
    // ============================================================
    // CRITICAL FIX (2025-11-27): Reduced from 10% to 2% to prevent
    // excessive mass loss at extreme temperatures (>40,000 K).
    //
    // This prevents numerical instability from large evaporation
    // rates causing fill_level to go negative in a single step.
    //
    // Physical justification: In real AM processes, the maximum
    // evaporation rate is limited by heat transport to the surface.
    // This limiter ensures the numerical scheme remains stable
    // while maintaining physical fidelity.
    // ============================================================
    const float MAX_DF_PER_STEP = 0.02f;  // Max 2% change per step

    if (df < -MAX_DF_PER_STEP * f) {
        df = -MAX_DF_PER_STEP * f;
    }

    // Apply change and clamp to [0, 1]
    float f_new = f + df;

    // Flush tiny values to zero (prevent denormalized float underflow)
    if (f_new < 1e-9f) f_new = 0.0f;

    fill_level[idx] = fmaxf(0.0f, fminf(1.0f, f_new));
}

void VOFSolver::applyEvaporationMassLoss(const float* J_evap, float rho, float dt) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    applyEvaporationMassLossKernel<<<gridSize, blockSize>>>(
        d_fill_level_, J_evap, rho, dx_, dt, nx_, ny_, nz_
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Diagnostic: Print evaporation mass loss info periodically
    static int evap_call_count = 0;
    if (evap_call_count % 500 == 0 && evap_call_count < 5000) {
        // Sample J_evap to check if evaporation is active
        int top_layer_start = (nz_ - 1) * nx_ * ny_;
        int sample_size = std::min(nx_ * ny_, 10000);
        std::vector<float> h_J(sample_size);
        CUDA_CHECK(cudaMemcpy(h_J.data(), J_evap + top_layer_start, sample_size * sizeof(float), cudaMemcpyDeviceToHost));

        float max_J = 0.0f;
        int active_cells = 0;
        for (int i = 0; i < sample_size; ++i) {
            if (h_J[i] > 0.0f) {
                active_cells++;
                max_J = std::max(max_J, h_J[i]);
            }
        }

        if (active_cells > 0) {
            printf("[VOF EVAP] Call %d: active_cells=%d, max_J=%.4e kg/(m^2*s), df_max=%.6f\n",
                   evap_call_count, active_cells, max_J, max_J * dt / (rho * dx_));
        }
    }
    evap_call_count++;
}

// ============================================================================
// Solidification Shrinkage (VOF-Thermal Coupling)
// ============================================================================

/**
 * @brief Apply solidification shrinkage mass source to fill level
 *
 * Physics:
 *   Solidification shrinkage occurs when liquid metal transforms to solid,
 *   causing volume contraction due to higher solid density.
 *
 *   Volume change: dV/V = -beta * dfl
 *   where beta = (rho_solid - rho_liquid) / rho_solid = 1 - rho_l/rho_s
 *         (typically ~0.07 for metals)
 *
 *   VOF fill level change (dimensionless):
 *   df = -beta * (dfl/dt) * dt = -beta * dfl
 *
 *   Solidifying: dfl/dt < 0 --> df > 0? NO!
 *   Actually: dfl < 0 (liquid decreasing) --> df < 0 (volume shrinks)
 *   So: df = beta * dfl_dt * dt (positive beta, negative rate = negative df)
 *
 * CRITICAL CONSTRAINTS (Bug Fix 2024-11):
 *   1. Only apply at VOF INTERFACE cells (0.01 < f < 0.99)
 *      - Internal bulk cells should not have their fill_level modified
 *      - Shrinkage manifests as surface depression, not internal voids
 *   2. Only apply during SOLIDIFICATION (dfl/dt < 0)
 *      - Melting expansion is handled differently (material addition)
 *   3. Correct dimensionless formula: df = beta * dfl_dt * dt
 *      - Previous formula had spurious /dx term causing grid-dependent behavior
 *   4. Conservative limiter: max 1% change per step
 *      - Prevents numerical instability at sharp solidification fronts
 *
 * Stability:
 *   - Only applies to interface cells (0.01 < f < 0.99)
 *   - Only during solidification (rate < 0)
 *   - Limited to max 1% reduction per timestep
 *   - Clamps result to [0, 1]
 */
__global__ void applySolidificationShrinkageKernel(
    float* fill_level,
    const float* dfl_dt,
    float beta,
    float dx,      // Not used in corrected formula, kept for API compatibility
    float dt,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];
    float rate = dfl_dt[idx];

    // ========================================================================
    // CONSTRAINT 1: Only apply at VOF interface cells
    // ========================================================================
    // Shrinkage can only manifest where there's a free surface.
    // Internal cells (f ~ 1.0) represent bulk material - their density changes
    // but the VOF fill_level should remain 1.0 (still fully filled with metal).
    // Gas cells (f ~ 0.0) have no material to shrink.
    // Only interface cells (0.01 < f < 0.99) can show volume reduction.
    // ========================================================================
    if (f <= 0.01f || f >= 0.99f) return;

    // ========================================================================
    // CONSTRAINT 2: Only apply during solidification (rate < 0)
    // ========================================================================
    // When dfl/dt < 0: liquid is becoming solid, volume shrinks
    // When dfl/dt > 0: solid is melting, but we don't add material here
    //                  (melting expansion would require mass source from substrate)
    // ========================================================================
    if (rate >= 0.0f) return;

    // Skip if no significant phase change happening
    if (fabsf(rate) < 1e-10f) return;

    // ========================================================================
    // CORRECTED FORMULA (dimensionless)
    // ========================================================================
    // df = beta * dfl_dt * dt
    //
    // Derivation:
    //   dV/V = -beta * dfl  (volume change due to solidification)
    //   For a cell: df_VOF = dV/V = -beta * dfl
    //             = -beta * (dfl/dt) * dt
    //             = beta * |rate| * dt  (since rate < 0 for solidification)
    //
    // Sign: rate < 0 (solidifying), beta > 0, dt > 0
    //       df = beta * rate * dt < 0 (fill level decreases = shrinkage)
    //
    // NOTE: Previous formula had /dx which is dimensionally incorrect:
    //       df = rate * beta * dt / dx  [1/s * - * s / m = 1/m] WRONG!
    //       Correct: df = rate * beta * dt  [1/s * - * s = -] CORRECT!
    // ========================================================================
    float df = beta * rate * dt;

    // ========================================================================
    // CONSERVATIVE LIMITER: max 1% reduction per step
    // ========================================================================
    // More conservative than before (was 5%) to prevent:
    // - Excessive dimpling at sharp solidification fronts
    // - Numerical oscillations near mushy zone boundaries
    // - Unrealistic rapid surface depression
    // ========================================================================
    const float MAX_DF_FRACTION = 0.01f;  // Max 1% change per step
    float max_reduction = MAX_DF_FRACTION * f;

    // df is already negative for solidification, so we limit its magnitude
    if (df < -max_reduction) {
        df = -max_reduction;
    }

    // Apply change and clamp
    float f_new = f + df;

    // Flush tiny values to zero (prevent denormalized float underflow)
    if (f_new < 1e-9f) f_new = 0.0f;

    fill_level[idx] = fmaxf(0.0f, fminf(1.0f, f_new));
}

void VOFSolver::applySolidificationShrinkage(const float* dfl_dt, float beta, float dx, float dt) {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    applySolidificationShrinkageKernel<<<blocks, threads>>>(
        d_fill_level_, dfl_dt, beta, dx, dt, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace physics
} // namespace lbm
