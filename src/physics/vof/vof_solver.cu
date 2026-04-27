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

// ============================================================================
// Forward Declarations (kernels defined later in file)
// ============================================================================
__global__ void applyMassCorrectionKernel(
    float* fill_level,
    float mass_correction,
    int nx, int ny, int nz,
    int interface_count,
    float damping);

__global__ void countInterfaceCellsKernel(
    const float* fill_level,
    int* partial_counts,
    int num_cells);

// ============================================================================
// Advection Kernels
// ============================================================================

/**
 * @brief GPU reduction kernel to find max velocity magnitude.
 * Avoids expensive full-domain D2H copy for CFL check.
 */
__global__ void maxVelocityMagnitudeKernel(
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float* __restrict__ block_max,
    int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        float vx = ux[idx], vy = uy[idx], vz = uz[idx];
        val = sqrtf(vx*vx + vy*vy + vz*vz);
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
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
    // Zero wall face velocities (FIX #4 - prevents mass advection through walls)
    // ========================================================================
    if (bc_x != 0) {  // X is WALL
        if (i == 0) u_face_xm = 0.0f;
        if (i == nx - 1) u_face_xp = 0.0f;
    }
    if (bc_y != 0) {  // Y is WALL for RT
        if (j == 0) v_face_ym = 0.0f;
        if (j == ny - 1) v_face_yp = 0.0f;
    }
    if (bc_z != 0) {  // Z is WALL
        if (k == 0) w_face_zm = 0.0f;
        if (k == nz - 1) w_face_zp = 0.0f;
    }

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

    // Symmetric flushing for minimal mass loss (BUG FIX #3)
    if (f_new < 1e-9f) f_new = 0.0f;
    if (f_new > 1.0f - 1e-9f) f_new = 1.0f;

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
    // Zero wall face velocities (FIX #4 - prevents mass advection through walls)
    // ========================================================================
    if (bc_x != 0) {  // X is WALL
        if (i == 0) u_face_xm = 0.0f;
        if (i == nx - 1) u_face_xp = 0.0f;
    }
    if (bc_y != 0) {  // Y is WALL for RT
        if (j == 0) v_face_ym = 0.0f;
        if (j == ny - 1) v_face_yp = 0.0f;
    }
    if (bc_z != 0) {  // Z is WALL
        if (k == 0) w_face_zm = 0.0f;
        if (k == nz - 1) w_face_zp = 0.0f;
    }

    // ========================================================================
    // X-DIRECTION FLUXES with TVD limiting
    // ========================================================================
    float flux_xm, flux_xp;
    // Face i-1/2 (between im and i)
    if (u_face_xm >= 0.0f) {  // Flow from imm → im → i
        float f_upwind = fill_level[idx_im];
        float f_down   = fill_level[idx_imm];
        float f_center = fill_level[idx];

        // Gradient ratio: r = (f_upwind - f_down) / (f_center - f_upwind)
        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;

        // Flux limiter
        float phi = applyFluxLimiter(r, limiter_type);

        // First-order upwind flux
        float F_low = u_face_xm * f_upwind;

        // Second-order correction: F_high ≈ u × f_face_center
        // BUGFIX: phi should only be applied ONCE in the flux blending, not in f_face_second
        float f_face_second = f_upwind + 0.5f * delta_center;  // NO phi here!
        float F_high = u_face_xm * f_face_second;

        flux_xm = F_low + phi * (F_high - F_low);  // phi applied here ONLY
    } else {  // Flow from i → im
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_ip];
        float f_center = fill_level[idx_im];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;

        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = u_face_xm * f_upwind;
        // BUGFIX: phi should only be applied ONCE in the flux blending
        float f_face_second = f_upwind + 0.5f * delta_center;  // NO phi here!
        float F_high = u_face_xm * f_face_second;

        flux_xm = F_low + phi * (F_high - F_low);  // phi applied here ONLY
    }

    // Face i+1/2 (between i and ip)
    if (u_face_xp >= 0.0f) {  // Flow from im → i → ip
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_im];
        float f_center = fill_level[idx_ip];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;

        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = u_face_xp * f_upwind;
        // BUGFIX: phi applied only in flux blending
        float f_face_second = f_upwind + 0.5f * delta_center;
        float F_high = u_face_xp * f_face_second;

        flux_xp = F_low + phi * (F_high - F_low);
    } else {  // Flow from ip → i
        float f_upwind = fill_level[idx_ip];
        float f_down   = fill_level[idx_ipp];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;

        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = u_face_xp * f_upwind;
        // BUGFIX: phi applied only in flux blending
        float f_face_second = f_upwind + 0.5f * delta_center;
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
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_ym * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
        float F_high = v_face_ym * f_face_second;

        flux_ym = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_jp];
        float f_center = fill_level[idx_jm];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_ym * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
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
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_yp * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
        float F_high = v_face_yp * f_face_second;

        flux_yp = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx_jp];
        float f_down   = fill_level[idx_jpp];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = v_face_yp * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
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
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zm * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
        float F_high = w_face_zm * f_face_second;

        flux_zm = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx];
        float f_down   = fill_level[idx_kp];
        float f_center = fill_level[idx_km];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zm * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
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
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zp * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
        float F_high = w_face_zp * f_face_second;

        flux_zp = F_low + phi * (F_high - F_low);
    } else {
        float f_upwind = fill_level[idx_kp];
        float f_down   = fill_level[idx_kpp];
        float f_center = fill_level[idx];

        float delta_upwind = f_upwind - f_down;
        float delta_center = f_center - f_upwind;
        float r = (fabsf(delta_center) > 1e-6f) ? (delta_upwind / delta_center) : 0.0f;
        float phi = applyFluxLimiter(r, limiter_type);

        float F_low = w_face_zp * f_upwind;
        float f_face_second = f_upwind + 0.5f * delta_center;  // BUGFIX: no phi here
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

    // Symmetric flushing for minimal mass loss (BUG FIX #3)
    if (f_new < 1e-9f) f_new = 0.0f;
    if (f_new > 1.0f - 1e-9f) f_new = 1.0f;

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

    // Compute fill level gradient with adaptive denominator at boundaries
    // At boundaries, i_m==i or i_p==i, so the stencil span is 1*dx, not 2*dx.
    // Using 2*dx at boundaries halves the gradient magnitude (incorrect).
    float denom_x = (i > 0 && i < nx - 1) ? 2.0f * dx : dx;
    float denom_y = (j > 0 && j < ny - 1) ? 2.0f * dx : dx;
    float denom_z = (k > 0 && k < nz - 1) ? 2.0f * dx : dx;

    float grad_x = (fill_level[idx_xp] - fill_level[idx_xm]) / denom_x;
    float grad_y = (fill_level[idx_yp] - fill_level[idx_ym]) / denom_y;
    float grad_z = (fill_level[idx_zp] - fill_level[idx_zm]) / denom_z;

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
    // Use adaptive denominator at boundaries (same fix as reconstructInterfaceKernel)
    float denom_x = (i > 0 && i < nx - 1) ? 2.0f * dx : dx;
    float denom_y = (j > 0 && j < ny - 1) ? 2.0f * dx : dx;
    float denom_z = (k > 0 && k < nz - 1) ? 2.0f * dx : dx;

    float dnx_dx = (interface_normal[idx_xp].x - interface_normal[idx_xm].x) / denom_x;
    float dny_dy = (interface_normal[idx_yp].y - interface_normal[idx_ym].y) / denom_y;
    float dnz_dz = (interface_normal[idx_zp].z - interface_normal[idx_zm].z) / denom_z;

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

/**
 * @brief Global mass conservation correction kernel (legacy uniform scaling)
 * @note Scales all fill levels uniformly to enforce exact mass conservation.
 * @warning Concentrates redistributed mass in interface cells regardless of
 *          physical context — known to worsen LPBF centerline depression.
 *          Prefer the v_z-weighted overload of enforceGlobalMassConservation.
 */
__global__ void enforceGlobalMassConservationKernel(
    float* fill_level,
    float scale_factor,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_cells) return;

    float f_old = fill_level[idx];
    float f_new = f_old * scale_factor;

    // Clamp to [0, 1] to maintain physical bounds
    fill_level[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}

// ============================================================================
// A1: v_z-Weighted Additive Mass Correction (2026-04-26)
// ============================================================================
// Replaces uniform multiplicative scaling with physics-aware redistribution:
// mass deficit is deposited preferentially at interface cells whose top
// surface flows upward (the capillary back-flow zone trying to refill the
// trailing groove), and removed preferentially from cells flowing downward.
// Diagnosed Phase-2 failure: uniform scale dumps reclaimed mass on already-
// over-deposited splash deposits + side ridges, deepening the centerline.
//
// Pass 1: reduce W = Σ max(sign(Δm) * v_z, 0) over interface cells.
// Pass 2: apply f += (Δm/W) * max(sign(Δm) * v_z, 0) with clamp.
// ============================================================================

/**
 * @brief Pass 1 — compute Σw over interface cells.
 * @param fill_level   VOF fill level
 * @param velocity_z   Vertical velocity [m/s]
 * @param sign_dm      +1.0f for mass deficit, -1.0f for mass excess
 * @param partial_sums One float per block (host completes the reduction)
 * @param num_cells    Total cell count
 */
__global__ void computeVzWeightSumKernel(
    const float* __restrict__ fill_level,
    const float* __restrict__ velocity_z,
    float sign_dm,
    float* __restrict__ partial_sums,
    int num_cells)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float w = 0.0f;
    if (idx < num_cells) {
        float f = fill_level[idx];
        // Interface cells only — same threshold as countInterfaceCellsKernel
        // and applyMassCorrectionKernel (B2 fix 2026-04-27, was strict 0/1).
        // Saturated bulk (f≤0.01 or f≥0.99) has unreliable gradients/normals.
        if (f > 0.01f && f < 0.99f) {
            w = fmaxf(sign_dm * velocity_z[idx], 0.0f);
        }
    }
    sdata[tid] = w;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

/**
 * @brief Pass 2 — apply v_z-weighted additive correction.
 * @param fill_level   VOF fill level (modified in-place)
 * @param velocity_z   Vertical velocity [m/s]
 * @param sign_dm      +1.0f for deficit, -1.0f for excess (same as pass 1)
 * @param delta_per_W  (target_mass - current_mass) / W  (cells per unit weight)
 * @param num_cells    Total cell count
 */
__global__ void applyVzWeightedMassCorrectionKernel(
    float* fill_level,
    const float* __restrict__ velocity_z,
    float sign_dm,
    float delta_per_W,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];
    // B2 fix (2026-04-27): match Pass-1 threshold and applyMassCorrectionKernel.
    if (f <= 0.01f || f >= 0.99f) return;      // skip near-pure cells

    float w = fmaxf(sign_dm * velocity_z[idx], 0.0f);
    if (w <= 0.0f) return;                     // wrong flow direction

    float f_new = f + delta_per_W * w;
    fill_level[idx] = fmaxf(0.0f, fminf(1.0f, f_new));
}

// ============================================================================
// B1: Inward-Flux-Weighted Additive Mass Correction (2026-04-27)
// ============================================================================
// Track-B replaces Track-A's max(v_z, 0) weight with the un-normalised
// interface inward-flux:
//
//   w_i = max( sign(Δm) * (∇f · v),  0 )    [SIGN CORRECTION 2026-04-27]
//
// where f=1 inside liquid → ∇f points TOWARD liquid → outward unit normal is
// n = -∇f/|∇f|. Math expert's `max(-n·v, 0)` translates to `max(+∇f·v, 0)` in
// this convention (NOT `max(-∇f·v, 0)`, which would invert the physics).
//
// computed inline from 6 face-neighbour fill-levels via central differences.
// The unnormalised gradient form (recommended by cfd-math-expert and
// validated against actual Phase-2 VTK by vtk-data-analyzer) keeps the
// natural |∇f| amplitude factor — sharp groove edges have larger |∇f|
// than gentle side ridges, automatically biasing the correction toward
// real refill sites and away from over-deposited ridge cells.
//
// Why inline ∇f rather than reading d_interface_normal_:
//   1. Normals are stale at advect time (reconstructInterface runs later)
//   2. Stored normals are normalized — discards |∇f| factor that is the
//      key discriminator (verified empirically: normalized form gives
//      side/center ratio 0.50, unnormalized gives 0.23)
//   3. Removes a pointer dependency from the API (no float3* needed)
//
// Pass 1: reduce W = Σ max(sign(Δm) * (∇f·v), 0) over interface cells.
// Pass 2: apply f += (Δm/W) * w_i with clamp.
// ============================================================================

/**
 * @brief Compute -∇f·v at cell (i,j,k) via central differences from 6 neighbors.
 * @param fill_level Device array of fill_level values
 * @param i, j, k    Cell indices
 * @param nx, ny, nz Domain dims
 * @param dx         Grid spacing [m]
 * @param vx, vy, vz Velocity at this cell
 * @param bc_x, bc_y, bc_z Boundary types (0=PERIODIC, 1=WALL)
 * @return -∇f·v [1/s · m/s = 1/s with f dimensionless]
 *
 * Boundary handling: for WALL, use one-sided difference at the edge cell;
 * for PERIODIC, wrap.
 */
__device__ inline float fluxWeightAtCell(
    const float* __restrict__ fill_level,
    int i, int j, int k, int nx, int ny, int nz, float dx,
    float vx, float vy, float vz,
    int bc_x, int bc_y, int bc_z)
{
    auto wrap = [](int q, int qmax, int bc) -> int {
        if (q < 0)      return (bc == 0) ? (qmax - 1) : 0;       // PERIODIC : WALL
        if (q >= qmax)  return (bc == 0) ? 0          : (qmax - 1);
        return q;
    };
    int ip = wrap(i + 1, nx, bc_x), im = wrap(i - 1, nx, bc_x);
    int jp = wrap(j + 1, ny, bc_y), jm = wrap(j - 1, ny, bc_y);
    int kp = wrap(k + 1, nz, bc_z), km = wrap(k - 1, nz, bc_z);
    float fxp = fill_level[ip + nx * (j  + ny * k )];
    float fxm = fill_level[im + nx * (j  + ny * k )];
    float fyp = fill_level[i  + nx * (jp + ny * k )];
    float fym = fill_level[i  + nx * (jm + ny * k )];
    float fzp = fill_level[i  + nx * (j  + ny * kp)];
    float fzm = fill_level[i  + nx * (j  + ny * km)];
    float gx = (fxp - fxm) * (0.5f / dx);
    float gy = (fyp - fym) * (0.5f / dx);
    float gz = (fzp - fzm) * (0.5f / dx);
    // ∇f · v  (positive = inflow into liquid, since f=1 in liquid means
    // ∇f points TOWARD liquid; outward normal n = -∇f/|∇f|; math expert's
    // max(-n·v, 0) becomes max(+∇f·v, 0) in this convention).
    // Negative = outflow from liquid (recoil splash); zeroed by the caller's
    // fmaxf guard to give w=0 on side ridges.
    return (gx * vx + gy * vy + gz * vz);
}

/**
 * @brief Pass 1 — Track-C reduce W = Σ max(sign_dm * (∇f·v), 0) over interface cells.
 *
 * Track-C augments Track-B with two geometric gates that zero w before accumulation:
 *   Gate 1 (trailing-band x-mask): skip cells ahead of the laser spot, where recoil
 *     dominates and ∇f·v falsely appears inward.
 *     Active when laser_x_lu >= 0; skips cell i > laser_x_lu - trailing_margin_lu.
 *   Gate 2 (z-floor gate): skip elevated cells (side ridges, splash deposits) that
 *     should never be refill targets.
 *     Active when z_substrate_lu >= 0; skips cell k > z_substrate_lu + z_offset_lu.
 *
 * Both gates cost 2 integer compares per thread and have zero impact when disabled
 * (laser_x_lu < 0 or z_substrate_lu < 0).
 */
__global__ void computeFluxWeightSumKernel(
    const float* __restrict__ fill_level,
    const float* __restrict__ velocity_x,
    const float* __restrict__ velocity_y,
    const float* __restrict__ velocity_z,
    float sign_dm,
    float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z,
    float* __restrict__ partial_sums,
    int num_cells,
    float laser_x_lu,        ///< Current laser x in lattice units; <0 = gate disabled
    float trailing_margin_lu, ///< x-exclusion half-width [lu]; cells with i > laser_x_lu-margin excluded
    float z_substrate_lu,     ///< Substrate top index [lu]; <0 = gate disabled
    float z_offset_lu)        ///< Extra allowance above substrate [lu] before exclusion kicks in
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float w = 0.0f;
    if (idx < num_cells) {
        float f = fill_level[idx];
        if (f > 0.01f && f < 0.99f) {
            int i = idx % nx;
            int j = (idx / nx) % ny;
            int k = idx / (nx * ny);

            // Gate 1: trailing-band x-mask — exclude active laser zone
            if (laser_x_lu >= 0.0f && (float)i > laser_x_lu - trailing_margin_lu) {
                // inside or ahead of laser spot; skip
            }
            // Gate 2: z-floor gate — exclude elevated cells above substrate
            else if (z_substrate_lu >= 0.0f && (float)k > z_substrate_lu + z_offset_lu) {
                // elevated cell (ridge/splash); skip
            }
            else {
                float vx = velocity_x[idx];
                float vy = velocity_y[idx];
                float vz = velocity_z[idx];
                float ndotv_inward = fluxWeightAtCell(
                    fill_level, i, j, k, nx, ny, nz, dx, vx, vy, vz,
                    bc_x, bc_y, bc_z);
                w = fmaxf(sign_dm * ndotv_inward, 0.0f);
            }
        }
    }
    sdata[tid] = w;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

/**
 * @brief Pass 2 — Track-C apply (Δm/W) * max(sign_dm * (∇f·v), 0) per cell.
 * Same geometric gates as computeFluxWeightSumKernel (must be identical to keep
 * the Pass-1 / Pass-2 cell sets consistent).
 */
__global__ void applyFluxWeightedMassCorrectionKernel(
    float* fill_level,
    const float* __restrict__ velocity_x,
    const float* __restrict__ velocity_y,
    const float* __restrict__ velocity_z,
    float sign_dm,
    float delta_per_W,
    float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z,
    int num_cells,
    float laser_x_lu,
    float trailing_margin_lu,
    float z_substrate_lu,
    float z_offset_lu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];
    if (f <= 0.01f || f >= 0.99f) return;

    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);

    // Gate 1: trailing-band x-mask
    if (laser_x_lu >= 0.0f && (float)i > laser_x_lu - trailing_margin_lu) return;
    // Gate 2: z-floor gate
    if (z_substrate_lu >= 0.0f && (float)k > z_substrate_lu + z_offset_lu) return;

    float vx = velocity_x[idx];
    float vy = velocity_y[idx];
    float vz = velocity_z[idx];
    float ndotv_inward = fluxWeightAtCell(
        fill_level, i, j, k, nx, ny, nz, dx, vx, vy, vz,
        bc_x, bc_y, bc_z);
    float w = fmaxf(sign_dm * ndotv_inward, 0.0f);
    if (w <= 0.0f) return;

    float f_new = f + delta_per_W * w;
    fill_level[idx] = fmaxf(0.0f, fminf(1.0f, f_new));
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
      mass_correction_enabled_(false),                // Default: disabled (backward compatible)
      mass_correction_damping_(0.7f),                 // Default: moderate damping
      mass_reference_(-1.0f),                         // Default: uninitialized
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
    // Bug-3 fix (2026-04-26): release cached scratch buffers.
    if (d_mass_partial_sums_) {
        cudaFree(d_mass_partial_sums_);
        d_mass_partial_sums_ = nullptr;
        d_mass_partial_sums_size_ = 0;
    }
    if (d_interface_partial_counts_) {
        cudaFree(d_interface_partial_counts_);
        d_interface_partial_counts_ = nullptr;
        d_interface_partial_counts_size_ = 0;
    }
}

void VOFSolver::initialize(const float* fill_level) {
    CUDA_CHECK(cudaMemcpy(d_fill_level_, fill_level, num_cells_ * sizeof(float),
               cudaMemcpyHostToDevice));

    // B4 fix (2026-04-27): reset mass-correction state on (re)initialize so
    // multi-instance / re-initialise tests don't inherit a stale baseline.
    mass_reference_ = -1.0f;
    mass_correction_call_count_ = 0;

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

void VOFSolver::setInterfaceCompression(bool enabled, float coefficient) {
    interface_compression_enabled_ = enabled;
    C_compress_coeff_ = coefficient;
}

void VOFSolver::advectFillLevel(const float* velocity_x,
                                 const float* velocity_y,
                                 const float* velocity_z,
                                 float dt) {
    // ========================================================================
    // PLIC geometric advection dispatch
    // ========================================================================
    // PLIC with CFL-based sub-stepping.
    // PLIC geometric fluxes require CFL < 1 per sweep direction. When the
    // fluid velocity exceeds dx/dt, the departure slab exceeds one cell and
    // the geometric intersection is invalid — producing overshoots that the
    // final clamp deletes, causing permanent mass loss.
    //
    // Fix: dynamically compute n_subs = ceil(CFL / CFL_target) and split
    // the advection into n_subs sub-steps of dt_sub = dt / n_subs.
    if (advection_scheme_ == VOFAdvectionScheme::PLIC) {
        // Compute max CFL from velocity field (GPU reduction already exists)
        const int rt = 256;
        const int rb = (num_cells_ + rt - 1) / rt;

        static float* d_block_max = nullptr;
        static int d_block_max_size = 0;
        if (d_block_max_size < rb) {
            if (d_block_max) cudaFree(d_block_max);
            CUDA_CHECK(cudaMalloc(&d_block_max, rb * sizeof(float)));
            d_block_max_size = rb;
        }

        maxVelocityMagnitudeKernel<<<rb, rt, rt * sizeof(float)>>>(
            velocity_x, velocity_y, velocity_z, d_block_max, num_cells_);
        CUDA_CHECK_KERNEL();

        std::vector<float> h_bmax(rb);
        CUDA_CHECK(cudaMemcpy(h_bmax.data(), d_block_max,
                              rb * sizeof(float), cudaMemcpyDeviceToHost));
        float v_max = 0.0f;
        for (int i = 0; i < rb; ++i)
            v_max = std::max(v_max, h_bmax[i]);

        float cfl = v_max * dt / dx_;
        const float cfl_target = 0.3f;
        int n_subs = std::max(1, static_cast<int>(std::ceil(cfl / cfl_target)));
        float dt_sub = dt / n_subs;

        static int call_count = 0;
        if (call_count % 500 == 0 || n_subs > 1) {
            printf("[VOF PLIC] Call %d: v_max=%.4f, CFL=%.3f, n_subs=%d, dt_sub=%.2e\n",
                   call_count, v_max, cfl, n_subs, dt_sub);
        }
        call_count++;

        for (int sub = 0; sub < n_subs; ++sub) {
            advectFillLevelPLIC(velocity_x, velocity_y, velocity_z, dt_sub);
        }
        return;
    }

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
    // Check VOF CFL condition before advection using GPU max-reduction.
    // BUG FIX: Was sampling only top layer (z=nz-1) = gas phase = zero velocity.
    // Now uses full-domain GPU reduction — O(N) on device, O(num_blocks) D2H copy.
    const int reduction_threads = 256;
    const int reduction_blocks = (num_cells_ + reduction_threads - 1) / reduction_threads;

    // Lazy-allocate reduction buffer (persists across calls)
    static float* d_block_max = nullptr;
    static int d_block_max_size = 0;
    if (d_block_max_size < reduction_blocks) {
        if (d_block_max) cudaFree(d_block_max);
        CUDA_CHECK(cudaMalloc(&d_block_max, reduction_blocks * sizeof(float)));
        d_block_max_size = reduction_blocks;
    }

    maxVelocityMagnitudeKernel<<<reduction_blocks, reduction_threads,
                                  reduction_threads * sizeof(float)>>>(
        velocity_x, velocity_y, velocity_z, d_block_max, num_cells_);
    CUDA_CHECK_KERNEL();

    std::vector<float> h_block_max(reduction_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max,
                          reduction_blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float v_max = 0.0f;
    for (int i = 0; i < reduction_blocks; ++i) {
        v_max = std::max(v_max, h_block_max[i]);
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
    // OPTIMIZED (2026-01-20): CFL_target = 0.20 for TVD scheme
    //   - TVD scheme is more stable than first-order upwind
    //   - Can use higher CFL without mass loss
    //   - Conservative: 0.20 to avoid subcycling edge cases
    //   - Better performance than 0.10 (less subcycling)
    // ========================================================================
    const float CFL_target = 0.20f;  // Optimized for TVD scheme
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
        CUDA_CHECK(cudaMemcpy(d_fill_level_, d_fill_level_tmp_, num_cells_ * sizeof(float),
                   cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ========================================================================
    // STEP 2: Mass conservation correction (2026-01-20)
    // ========================================================================
    // FIX: Correct cumulative mass loss from numerical diffusion and clamping
    //
    // PROBLEM: Even conservative flux formulation loses mass over time:
    //   - Clamping f < 1e-9 to zero (cumulative loss in bulk regions)
    //   - Numerical diffusion spreading interface over many cells
    //   - Floating-point roundoff errors in flux computation
    //   - Boundary flux truncation errors
    //
    // SOLUTION: After advection, measure mass loss and redistribute to interface
    //
    // MECHANISM:
    //   1. Compute current mass: M_new = Σf_i
    //   2. Compare to reference: ΔM = M_ref - M_new
    //   3. Count interface cells: N_int (0.01 < f < 0.99)
    //   4. Redistribute: Add ΔM/N_int to each interface cell (with damping)
    //
    // CONSERVATION:
    //   Σf_corrected = Σf + ΔM = M_ref (exact)
    //
    // COST: ~5% overhead (2 reductions + 1 correction kernel)
    // BENEFIT: <1% mass error (vs 5-20% without correction)
    //
    // NOTE: Only applied if mass_correction_enabled_ = true (default: false)
    // ========================================================================
    // Mass-correction (Track-A v_z OR Track-B inline-∇f flux weight).
    // velocity_x/y/z are passed to advection so they're in scope here.
    if (mass_correction_use_flux_weight_) {
        applyMassCorrectionInline(velocity_x, velocity_y, velocity_z);  // Track-B
    } else {
        applyMassCorrectionInline(velocity_z);                          // Track-A
    }

    // ========================================================================
    // STEP 3: Interface compression (Olsson-Kreiss) - DISABLED for RT
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
    // OPTIMIZED (2026-01-20): Mild compression for sharper interface
    if (!interface_compression_enabled_) return;
    float C_compress = C_compress_coeff_;

    {
        // Copy current field to tmp buffer for compression kernel input
        CUDA_CHECK(cudaMemcpy(d_fill_level_tmp_, d_fill_level_, num_cells_ * sizeof(float),
                   cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        applyInterfaceCompressionKernel<<<gridSize, blockSize>>>(
            d_fill_level_,      // output: compressed field
            d_fill_level_tmp_,  // input: advected (diffused) field
            velocity_x, velocity_y, velocity_z,
            dx_, dt, C_compress, nx_, ny_, nz_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // NOTE: d_fill_level_ now contains final result (advected + optionally compressed + mass corrected)
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
    CUDA_CHECK(cudaMemcpy(host_fill, d_fill_level_, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost));
}

void VOFSolver::copyCellFlagsToHost(uint8_t* host_flags) const {
    CUDA_CHECK(cudaMemcpy(host_flags, d_cell_flags_, num_cells_ * sizeof(uint8_t),
               cudaMemcpyDeviceToHost));
}

void VOFSolver::copyCurvatureToHost(float* host_curvature) const {
    CUDA_CHECK(cudaMemcpy(host_curvature, d_curvature_, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost));
}

float VOFSolver::computeTotalMass() const {
    int blockSize = 256;
    int gridSize = (num_cells_ + blockSize - 1) / blockSize;

    // Bug-3 fix (2026-04-26): lazy-cache scratch buffer instead of allocating
    // every call. computeTotalMass is invoked up to 3× per advection step.
    if (d_mass_partial_sums_size_ < gridSize) {
        if (d_mass_partial_sums_) cudaFree(d_mass_partial_sums_);
        CUDA_CHECK(cudaMalloc(&d_mass_partial_sums_, gridSize * sizeof(float)));
        d_mass_partial_sums_size_ = gridSize;
    }

    // First reduction: compute partial sums
    computeMassReductionKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_fill_level_, d_mass_partial_sums_, num_cells_);
    CUDA_CHECK_KERNEL();

    // Copy partial sums to host and finish reduction on CPU
    std::vector<float> h_partial_sums(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_mass_partial_sums_, gridSize * sizeof(float),
               cudaMemcpyDeviceToHost));

    // Final reduction on CPU using double precision + Kahan compensation.
    // Sprint-1 fix (2026-04-25): old code used FP32 sequential accumulation;
    // for ~3.8M cells the relative error was O(N)·ε_mach ≈ 4×10⁻¹, completely
    // washing out the GPU tree-reduction precision and making the 0.001%
    // mass-drift target unmeasurable in diagnostics.
    double total_mass_dbl = 0.0;
    double kahan_c = 0.0;
    for (float sum : h_partial_sums) {
        double y = static_cast<double>(sum) - kahan_c;
        double t = total_mass_dbl + y;
        kahan_c = (t - total_mass_dbl) - y;
        total_mass_dbl = t;
    }
    return static_cast<float>(total_mass_dbl);
}

// ============================================================================
// A1 inline helper — shared by TVD and PLIC advection paths.
// Replaces the duplicated correction blocks (Bug-4 dedupe, 2026-04-26).
// ============================================================================
void VOFSolver::applyMassCorrectionInline(const float* d_vz) {
    if (!mass_correction_enabled_) return;

    // B1 fix (2026-04-27): explicit sync before reading d_fill_level_.
    // Callers (advectFillLevel TVD/PLIC) end with cudaDeviceSynchronize but
    // the ordering was implicit. Make it explicit so future call sites are safe.
    CUDA_CHECK(cudaDeviceSynchronize());

    float mass_current = computeTotalMass();
    if (mass_reference_ < 0.0f) {
        mass_reference_ = mass_current;   // first-call initialization
        return;                           // nothing to correct yet
    }

    float delta_m = mass_reference_ - mass_current;     // positive when mass lost
    float mass_error_fraction = (mass_reference_ > 0.0f)
        ? fabsf(delta_m) / mass_reference_ : 0.0f;

    constexpr float CORRECTION_THRESHOLD = 1e-6f;       // absolute (Σf units)
    if (fabsf(delta_m) <= CORRECTION_THRESHOLD) return;

    int blockSize = 256;
    int gridSize  = (num_cells_ + blockSize - 1) / blockSize;

    // ---- A1 attempt: v_z-weighted additive correction ----
    double W_dbl = 0.0;
    if (d_vz != nullptr) {
        if (d_mass_partial_sums_size_ < gridSize) {
            if (d_mass_partial_sums_) cudaFree(d_mass_partial_sums_);
            CUDA_CHECK(cudaMalloc(&d_mass_partial_sums_, gridSize * sizeof(float)));
            d_mass_partial_sums_size_ = gridSize;
        }
        float sign_dm = (delta_m >= 0.0f) ? 1.0f : -1.0f;

        computeVzWeightSumKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
            d_fill_level_, d_vz, sign_dm, d_mass_partial_sums_, num_cells_);
        CUDA_CHECK_KERNEL();

        std::vector<float> h_partial(gridSize);
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_mass_partial_sums_,
                              gridSize * sizeof(float), cudaMemcpyDeviceToHost));
        double kc = 0.0;
        for (float p : h_partial) {
            double y = static_cast<double>(p) - kc;
            double t = W_dbl + y;
            kc = (t - W_dbl) - y;
            W_dbl = t;
        }

        if (W_dbl > 1e-12) {
            float delta_per_W = static_cast<float>(static_cast<double>(delta_m) / W_dbl);
            applyVzWeightedMassCorrectionKernel<<<gridSize, blockSize>>>(
                d_fill_level_, d_vz, sign_dm, delta_per_W, num_cells_);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            if (mass_correction_call_count_ % 500 == 0) {
                float mass_after = computeTotalMass();
                printf("[VOF MASS CORRECTION A1-vz] ΔM=%.3e (%.4f%%), W=%.3e, "
                       "applied=%.3e\n",
                       delta_m, mass_error_fraction * 100.0f,
                       W_dbl, mass_after - mass_current);
            }
            mass_correction_call_count_++;
            return;
        }
        // W ≈ 0 → fall through to uniform additive fallback
    }

    // ---- Fallback: uniform additive over interface cells ----
    if (d_interface_partial_counts_size_ < gridSize) {
        if (d_interface_partial_counts_) cudaFree(d_interface_partial_counts_);
        CUDA_CHECK(cudaMalloc(&d_interface_partial_counts_, gridSize * sizeof(int)));
        d_interface_partial_counts_size_ = gridSize;
    }
    countInterfaceCellsKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_partial_counts_, num_cells_);
    CUDA_CHECK_KERNEL();

    std::vector<int> h_counts(gridSize);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_interface_partial_counts_,
                          gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    long long interface_count = 0;
    for (int c : h_counts) interface_count += c;

    if (interface_count > 0) {
        dim3 mc_block(8, 8, 8);
        dim3 mc_grid((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 7) / 8);
        applyMassCorrectionKernel<<<mc_grid, mc_block>>>(
            d_fill_level_, delta_m, nx_, ny_, nz_,
            static_cast<int>(interface_count), mass_correction_damping_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        if (mass_correction_call_count_ % 500 == 0) {
            float mass_after = computeTotalMass();
            const char* tag = (d_vz != nullptr) ? "A1-fallback-uniform" : "uniform";
            printf("[VOF MASS CORRECTION %s] ΔM=%.3e (%.4f%%), N_int=%lld, "
                   "applied=%.3e\n",
                   tag, delta_m, mass_error_fraction * 100.0f,
                   interface_count, mass_after - mass_current);
        }
    } else if (mass_error_fraction > 0.01f &&
               mass_correction_call_count_ % 1000 == 0) {
        printf("[VOF MASS WARNING] Cannot correct %.1f%% mass loss — "
               "no interface cells.\n", mass_error_fraction * 100.0f);
    }
    mass_correction_call_count_++;
}

// ============================================================================
// Track-B helper — w = max(sign(Δm)·(-∇f·v), 0) inline-gradient flux weight.
// ============================================================================
void VOFSolver::applyMassCorrectionInline(const float* d_vx,
                                           const float* d_vy,
                                           const float* d_vz) {
    if (!mass_correction_enabled_) return;
    if (d_vx == nullptr || d_vy == nullptr || d_vz == nullptr) return;

    // B1 fix: explicit sync before reading d_fill_level_.
    CUDA_CHECK(cudaDeviceSynchronize());

    float mass_current = computeTotalMass();
    if (mass_reference_ < 0.0f) {
        mass_reference_ = mass_current;
        return;
    }

    float delta_m = mass_reference_ - mass_current;
    float mass_error_fraction = (mass_reference_ > 0.0f)
        ? fabsf(delta_m) / mass_reference_ : 0.0f;
    constexpr float CORRECTION_THRESHOLD = 1e-6f;
    if (fabsf(delta_m) <= CORRECTION_THRESHOLD) return;

    int blockSize = 256;
    int gridSize  = (num_cells_ + blockSize - 1) / blockSize;

    // Damping is applied to delta_m (softer correction) rather than to the
    // weight, so cell discrimination is preserved while the magnitude shrinks.
    float damped_dm = mass_correction_damping_ * delta_m;
    float sign_dm   = (damped_dm >= 0.0f) ? 1.0f : -1.0f;

    // ---- Pass 1: reduce W = Σ max(sign_dm·(-∇f·v), 0) over interface cells ----
    if (d_mass_partial_sums_size_ < gridSize) {
        if (d_mass_partial_sums_) cudaFree(d_mass_partial_sums_);
        CUDA_CHECK(cudaMalloc(&d_mass_partial_sums_, gridSize * sizeof(float)));
        d_mass_partial_sums_size_ = gridSize;
    }
    int bc_x_i = static_cast<int>(bc_x_);
    int bc_y_i = static_cast<int>(bc_y_);
    int bc_z_i = static_cast<int>(bc_z_);
    computeFluxWeightSumKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_fill_level_, d_vx, d_vy, d_vz, sign_dm, dx_,
        nx_, ny_, nz_, bc_x_i, bc_y_i, bc_z_i,
        d_mass_partial_sums_, num_cells_,
        mass_correction_laser_x_lu_, mass_correction_trailing_margin_lu_,
        mass_correction_z_substrate_lu_, mass_correction_z_offset_lu_);
    CUDA_CHECK_KERNEL();

    std::vector<float> h_partial(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_mass_partial_sums_,
                          gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    double W_dbl = 0.0, kc = 0.0;
    for (float p : h_partial) {
        double y = static_cast<double>(p) - kc;
        double t = W_dbl + y;
        kc = (t - W_dbl) - y;
        W_dbl = t;
    }

    if (W_dbl > 1e-12) {
        float delta_per_W = static_cast<float>(static_cast<double>(damped_dm) / W_dbl);
        applyFluxWeightedMassCorrectionKernel<<<gridSize, blockSize>>>(
            d_fill_level_, d_vx, d_vy, d_vz, sign_dm, delta_per_W,
            dx_, nx_, ny_, nz_, bc_x_i, bc_y_i, bc_z_i, num_cells_,
            mass_correction_laser_x_lu_, mass_correction_trailing_margin_lu_,
            mass_correction_z_substrate_lu_, mass_correction_z_offset_lu_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        if (mass_correction_call_count_ % 500 == 0) {
            float mass_after = computeTotalMass();
            printf("[VOF MASS CORRECTION C-flux] ΔM=%.3e (%.4f%%), W=%.3e, "
                   "applied=%.3e (damp=%.2f, laser_x=%.1f, z_sub=%.1f)\n",
                   delta_m, mass_error_fraction * 100.0f,
                   W_dbl, mass_after - mass_current, mass_correction_damping_,
                   mass_correction_laser_x_lu_, mass_correction_z_substrate_lu_);
        }
        mass_correction_call_count_++;
        return;
    }
    // W ≈ 0 → fall through to uniform additive over interface cells

    if (d_interface_partial_counts_size_ < gridSize) {
        if (d_interface_partial_counts_) cudaFree(d_interface_partial_counts_);
        CUDA_CHECK(cudaMalloc(&d_interface_partial_counts_, gridSize * sizeof(int)));
        d_interface_partial_counts_size_ = gridSize;
    }
    countInterfaceCellsKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_partial_counts_, num_cells_);
    CUDA_CHECK_KERNEL();

    std::vector<int> h_counts(gridSize);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_interface_partial_counts_,
                          gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    long long interface_count = 0;
    for (int c : h_counts) interface_count += c;

    if (interface_count > 0) {
        dim3 mc_block(8, 8, 8);
        dim3 mc_grid((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 7) / 8);
        applyMassCorrectionKernel<<<mc_grid, mc_block>>>(
            d_fill_level_, damped_dm, nx_, ny_, nz_,
            static_cast<int>(interface_count), 1.0f /* damping=1: full additive */);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        if (mass_correction_call_count_ % 500 == 0) {
            float mass_after = computeTotalMass();
            printf("[VOF MASS CORRECTION B1-fallback-uniform] ΔM=%.3e (%.4f%%), "
                   "N_int=%lld, applied=%.3e\n",
                   delta_m, mass_error_fraction * 100.0f,
                   interface_count, mass_after - mass_current);
        }
    }
    mass_correction_call_count_++;
}

// ============================================================================
// Track-C public entry point — exercises gates with caller-supplied params.
// Designed so unit tests can drive the Track-C kernels with synthetic fields
// and arbitrary gate values, without going through advectFillLevel().
// ============================================================================
void VOFSolver::enforceMassConservationFlux(
    float target_mass,
    const float* d_vx, const float* d_vy, const float* d_vz,
    float laser_x_lu, float trailing_margin_lu,
    float z_substrate_lu, float z_offset_lu)
{
    if (d_vx == nullptr || d_vy == nullptr || d_vz == nullptr) return;

    CUDA_CHECK(cudaDeviceSynchronize());

    float mass_current = computeTotalMass();
    float delta_m = target_mass - mass_current;
    if (fabsf(delta_m) <= 1e-6f) return;

    int blockSize = 256;
    int gridSize  = (num_cells_ + blockSize - 1) / blockSize;

    if (d_mass_partial_sums_size_ < gridSize) {
        if (d_mass_partial_sums_) cudaFree(d_mass_partial_sums_);
        CUDA_CHECK(cudaMalloc(&d_mass_partial_sums_, gridSize * sizeof(float)));
        d_mass_partial_sums_size_ = gridSize;
    }

    float sign_dm  = (delta_m >= 0.0f) ? 1.0f : -1.0f;
    int bc_x_i = static_cast<int>(bc_x_);
    int bc_y_i = static_cast<int>(bc_y_);
    int bc_z_i = static_cast<int>(bc_z_);

    computeFluxWeightSumKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_fill_level_, d_vx, d_vy, d_vz, sign_dm, dx_,
        nx_, ny_, nz_, bc_x_i, bc_y_i, bc_z_i,
        d_mass_partial_sums_, num_cells_,
        laser_x_lu, trailing_margin_lu, z_substrate_lu, z_offset_lu);
    CUDA_CHECK_KERNEL();

    std::vector<float> h_partial(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_mass_partial_sums_,
                          gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    double W_dbl = 0.0, kc = 0.0;
    for (float p : h_partial) {
        double y = static_cast<double>(p) - kc;
        double t = W_dbl + y;
        kc = (t - W_dbl) - y;
        W_dbl = t;
    }

    if (W_dbl > 1e-12) {
        float delta_per_W = static_cast<float>(static_cast<double>(delta_m) / W_dbl);
        applyFluxWeightedMassCorrectionKernel<<<gridSize, blockSize>>>(
            d_fill_level_, d_vx, d_vy, d_vz, sign_dm, delta_per_W,
            dx_, nx_, ny_, nz_, bc_x_i, bc_y_i, bc_z_i, num_cells_,
            laser_x_lu, trailing_margin_lu, z_substrate_lu, z_offset_lu);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // Fallback: uniform additive over all interface cells (gates don't apply here —
    // this is the safety net for the degenerate case W=0, not a physics path).
    if (d_interface_partial_counts_size_ < gridSize) {
        if (d_interface_partial_counts_) cudaFree(d_interface_partial_counts_);
        CUDA_CHECK(cudaMalloc(&d_interface_partial_counts_, gridSize * sizeof(int)));
        d_interface_partial_counts_size_ = gridSize;
    }
    countInterfaceCellsKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_interface_partial_counts_, num_cells_);
    CUDA_CHECK_KERNEL();

    std::vector<int> h_counts(gridSize);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_interface_partial_counts_,
                          gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    long long n_int = 0;
    for (int c : h_counts) n_int += c;
    if (n_int > 0) {
        dim3 mc_block(8, 8, 8);
        dim3 mc_grid((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 7) / 8);
        applyMassCorrectionKernel<<<mc_grid, mc_block>>>(
            d_fill_level_, delta_m, nx_, ny_, nz_,
            static_cast<int>(n_int), 1.0f);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void VOFSolver::enforceGlobalMassConservation(float target_mass,
                                              const float* d_vz) {
    // Bug-2 fix (2026-04-26): guard against zero-division.
    // target_mass == 0 happens if caller passes uninitialized state;
    // current_mass == 0 happens if the entire domain is gas (e.g. before fill).
    if (target_mass <= 0.0f) return;

    float current_mass = computeTotalMass();
    if (current_mass <= 0.0f) return;

    // Check if correction is needed (only if relative error > 0.1%)
    float mass_error_abs = current_mass - target_mass;       // signed
    float mass_error_rel = fabsf(mass_error_abs) / target_mass;
    if (mass_error_rel < 0.001f) return;

    int blockSize = 256;
    int gridSize  = (num_cells_ + blockSize - 1) / blockSize;

    // ------------------------------------------------------------------------
    // Backward-compat path: no velocity field → uniform multiplicative scaling.
    // ------------------------------------------------------------------------
    if (d_vz == nullptr) {
        float scale_factor = target_mass / current_mass;
        enforceGlobalMassConservationKernel<<<gridSize, blockSize>>>(
            d_fill_level_, scale_factor, num_cells_);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // ------------------------------------------------------------------------
    // A1 path: v_z-weighted additive correction.
    //   delta_m = target - current  (positive when mass was lost)
    //   sign_dm = sign(delta_m); use +v_z to refill, -v_z to drain
    // ------------------------------------------------------------------------
    float delta_m = target_mass - current_mass;             // signed
    float sign_dm = (delta_m >= 0.0f) ? 1.0f : -1.0f;

    // Pass 1 — reduce W = Σ max(sign_dm * v_z, 0) over interface cells.
    if (d_mass_partial_sums_size_ < gridSize) {
        if (d_mass_partial_sums_) cudaFree(d_mass_partial_sums_);
        CUDA_CHECK(cudaMalloc(&d_mass_partial_sums_, gridSize * sizeof(float)));
        d_mass_partial_sums_size_ = gridSize;
    }

    computeVzWeightSumKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_fill_level_, d_vz, sign_dm, d_mass_partial_sums_, num_cells_);
    CUDA_CHECK_KERNEL();

    std::vector<float> h_partial(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_mass_partial_sums_,
                          gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Kahan-compensated CPU finish (matches computeTotalMass style).
    double W_dbl = 0.0, kc = 0.0;
    for (float p : h_partial) {
        double y = static_cast<double>(p) - kc;
        double t = W_dbl + y;
        kc = (t - W_dbl) - y;
        W_dbl = t;
    }

    // ------------------------------------------------------------------------
    // Fallback: if no cells with the right flow direction, redistribute mass
    // uniformly over interface cells via the existing applyMassCorrectionKernel.
    // This avoids the multiplicative scaling failure mode entirely.
    // ------------------------------------------------------------------------
    if (W_dbl < 1e-12) {
        // Count interface cells via existing helper for uniform additive fallback.
        int* d_counts = nullptr;
        if (d_interface_partial_counts_size_ < gridSize) {
            if (d_interface_partial_counts_) cudaFree(d_interface_partial_counts_);
            CUDA_CHECK(cudaMalloc(&d_interface_partial_counts_, gridSize * sizeof(int)));
            d_interface_partial_counts_size_ = gridSize;
        }
        d_counts = d_interface_partial_counts_;

        countInterfaceCellsKernel<<<gridSize, blockSize>>>(
            d_fill_level_, d_counts, num_cells_);
        CUDA_CHECK_KERNEL();

        std::vector<int> h_counts(gridSize);
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                              gridSize * sizeof(int), cudaMemcpyDeviceToHost));
        long long n_int = 0;
        for (int c : h_counts) n_int += c;

        if (n_int > 0) {
            // Use 3D grid as the legacy applyMassCorrectionKernel expects.
            dim3 mc_block(8, 8, 8);
            dim3 mc_grid((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 7) / 8);
            applyMassCorrectionKernel<<<mc_grid, mc_block>>>(
                d_fill_level_, delta_m, nx_, ny_, nz_,
                static_cast<int>(n_int), 1.0f /* damping=1: full additive */);
            CUDA_CHECK_KERNEL();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // Pass 2 — apply weighted additive correction.
    // delta_m carries its sign; w_i is non-negative, so the product follows
    // sign(delta_m). delta_per_W = |delta_m| / W when sign_dm=+1, else
    // -|delta_m|/W; equivalently delta_m / W (because |delta_m| = sign_dm * delta_m).
    float delta_per_W = static_cast<float>(static_cast<double>(delta_m) / W_dbl);

    applyVzWeightedMassCorrectionKernel<<<gridSize, blockSize>>>(
        d_fill_level_, d_vz, sign_dm, delta_per_W, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Track-B public entry — wraps the private 3-velocity helper with a
// target-mass mode (sets mass_reference_ from target so a one-shot test can
// drive the kernel without relying on first-call initialisation chain).
// ============================================================================
void VOFSolver::enforceGlobalMassConservation(float target_mass,
                                              const float* d_vx,
                                              const float* d_vy,
                                              const float* d_vz)
{
    if (target_mass <= 0.0f) return;
    if (d_vx == nullptr || d_vy == nullptr || d_vz == nullptr) {
        // No velocities → fall back to legacy uniform-scale path.
        enforceGlobalMassConservation(target_mass, /*d_vz=*/(const float*)nullptr);
        return;
    }

    // Set mass_reference_ to target so the private helper sees correct delta_m.
    // The helper uses delta_m = mass_reference_ - mass_current, so mass_reference_
    // = target gives the standard "current → target" semantics.
    // Damping is forced to 1.0 here (full correction in one shot) — the public
    // API is for one-shot correction (called by tests / by external code that
    // wants exact correction); the per-step inline correction in the advection
    // loop uses the configurable damping (typically 0.5-0.7) for stability.
    bool  was_enabled  = mass_correction_enabled_;
    float saved_ref    = mass_reference_;
    float saved_damp   = mass_correction_damping_;
    mass_correction_enabled_ = true;
    mass_reference_          = target_mass;
    mass_correction_damping_ = 1.0f;

    applyMassCorrectionInline(d_vx, d_vy, d_vz);

    mass_correction_enabled_ = was_enabled;
    mass_reference_          = saved_ref;
    mass_correction_damping_ = saved_damp;
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
    float J_vol = J_evap[idx];   // R7: volumetric [kg/(m³·s)] (|∇f|-weighted)

    // Skip cells with no material or no evaporation
    if (f <= 0.0f || J_vol <= 0.0f) {
        return;
    }

    // R7 OPENFOAM-ALIGNED: J_evap carries volumetric mass-loss rate.
    // df = -J_vol · dt / ρ   (no /dx; |∇f| already provided that factor).
    float df = -J_vol * dt / rho;

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

// ============================================================================
// Mass Conservation Correction (2026-01-20)
// ============================================================================

/**
 * @brief CUDA kernel for mass redistribution to correct numerical mass loss
 *
 * PROBLEM:
 * ========
 * Even with conservative flux formulation and TVD schemes, cumulative mass loss
 * occurs due to:
 *   1. Clamping small values (f < 1e-9) to zero
 *   2. Numerical diffusion spreading interface over many cells
 *   3. Floating-point roundoff in flux computations
 *   4. Boundary flux truncation errors
 *
 * Typical mass loss: 5-20% over long simulations (unacceptable!)
 *
 * SOLUTION:
 * =========
 * After each advection step:
 *   1. Compute total mass: M_new = Σf_i
 *   2. Compare to reference: ΔM = M_ref - M_new
 *   3. If mass lost (ΔM > 0), redistribute proportionally to interface cells
 *
 * REDISTRIBUTION STRATEGY:
 * ========================
 * Only add mass to INTERFACE cells (0.01 < f < 0.99) because:
 *   - Pure liquid cells (f ≈ 1) are already full
 *   - Pure gas cells (f ≈ 0) should remain empty
 *   - Interface cells can absorb small corrections without artifacts
 *
 * Correction formula:
 *   f_corrected = f + α · ΔM / N_interface
 *   where α ∈ [0.1, 1.0] is damping factor (prevent overcorrection)
 *
 * MASS CONSERVATION:
 * ==================
 * This correction is CONSERVATIVE:
 *   Σf_corrected = Σf + ΔM = M_ref  (exact, up to FP precision)
 *
 * PHYSICS:
 * ========
 * Numerical mass loss is non-physical. This correction restores the correct
 * mass without violating any physical laws. The redistributed mass represents
 * material that was "artificially diffused" by numerical errors.
 *
 * LIMITATIONS:
 * ============
 * 1. Cannot correct mass if NO interface cells exist (pure liquid/gas domain)
 * 2. Large corrections (>5% per step) indicate severe numerical issues
 * 3. Does not fix root cause (only symptom) - better schemes still preferred
 *
 * PERFORMANCE:
 * ============
 * Cost: 2 reduction kernels + 1 correction kernel ≈ 5% overhead
 * Benefit: <1% mass error (down from 20%)
 *
 * REFERENCES:
 * ===========
 * - Rudman (1997). Volume-tracking methods for interfacial flow calculations.
 *   International Journal for Numerical Methods in Fluids, 24(7), 671-691.
 * - Ubbink & Issa (1999). A method for capturing sharp fluid interfaces on
 *   arbitrary meshes. Journal of Computational Physics, 153(1), 26-50.
 * - Deshpande et al. (2012). Evaluating the performance of the two-phase flow
 *   solver interFoam. Computational Science & Discovery, 5(1), 014016.
 *
 * @param fill_level Fill level field [0-1] (modified in-place)
 * @param mass_correction Global mass correction to add
 * @param nx, ny, nz Grid dimensions
 * @param interface_count Total number of interface cells (computed by caller)
 * @param damping Damping factor [0.1-1.0] to prevent overcorrection
 */
__global__ void applyMassCorrectionKernel(
    float* fill_level,
    float mass_correction,
    int nx, int ny, int nz,
    int interface_count,
    float damping)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);
    float f = fill_level[idx];

    // ========================================================================
    // Only redistribute mass to interface cells
    // ========================================================================
    // Interface threshold: 0.01 < f < 0.99
    // - Pure liquid (f ≈ 1.0): Already full, cannot add more
    // - Pure gas (f ≈ 0.0): Should remain empty
    // - Interface cells: Can absorb correction without visual artifacts
    const float F_MIN = 0.01f;
    const float F_MAX = 0.99f;

    if (f <= F_MIN || f >= F_MAX) {
        return;  // Skip pure liquid/gas cells
    }

    // Avoid division by zero if no interface cells
    if (interface_count == 0) return;

    // ========================================================================
    // Compute per-cell correction
    // ========================================================================
    // Distribute mass evenly among all interface cells
    // Apply damping to prevent overshoot (typical: 0.5-0.8)
    float delta_f = damping * mass_correction / static_cast<float>(interface_count);

    // ========================================================================
    // Apply correction with bounds checking
    // ========================================================================
    float f_new = f + delta_f;

    // Clamp to [0, 1] to maintain physical bounds
    f_new = fmaxf(0.0f, fminf(1.0f, f_new));

    // Flush extremely tiny values to zero (consistent with advection kernel)
    if (f_new < 1e-9f) f_new = 0.0f;

    fill_level[idx] = f_new;
}

/**
 * @brief CUDA kernel to count interface cells
 * @note Required for mass redistribution normalization
 */
__global__ void countInterfaceCellsKernel(
    const float* fill_level,
    int* partial_counts,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for block-level reduction
    __shared__ int shared_count[256];

    int local_count = 0;
    if (idx < num_cells) {
        float f = fill_level[idx];
        // Count interface cells: 0.01 < f < 0.99
        if (f > 0.01f && f < 0.99f) {
            local_count = 1;
        }
    }

    // Store in shared memory
    int tid = threadIdx.x;
    shared_count[tid] = local_count;
    __syncthreads();

    // Block-level reduction (parallel sum)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) {
        partial_counts[blockIdx.x] = shared_count[0];
    }
}

// ============================================================================
// PLIC (Piecewise Linear Interface Calculation) Geometric Advection
// ============================================================================
//
// Reference: Scardovelli & Zaleski (2000), JCP 164:228-237
// Parker & Youngs (1992) for normal estimation
//
// Summary of the PLIC approach:
//   1. Reconstruct a plane n·x = α per interface cell (Youngs normal + α inversion)
//   2. For each directional sweep, compute the geometric flux through each face
//      by intersecting the PLIC plane with the departure slab
//   3. Update the fill level conservatively, one direction at a time (Strang splitting)
//
// The flux through the right face of cell (i,j,k) for x-sweep with u > 0 is:
//   flux = volume of fluid in the slab [1-d, 1] × [0,1] × [0,1]
// where d = u_face * dt / dx (CFL number for this face).
//
// Coordinate transformation maps the slab to [0,1]³ with a modified alpha.
// The volume function plicVolume3D uses the inclusion-exclusion formula of
// Scardovelli & Zaleski, valid for 0 ≤ α ≤ (m1+m2+m3)/2 (symmetry handles rest).
// ============================================================================

// ----------------------------------------------------------------------------
// Device helper: volume below plane m·x = α in the unit cube [0,1]³
// Requires m1 ≥ m2 ≥ m3 ≥ 0 and uses symmetry for α > S/2.
//
// Inclusion-exclusion formula (Scardovelli & Zaleski 2000, eq. 28):
//   V = [α³ - Σ(α - mi)³₊ + Σ(α - mi - mj)³₊] / (6·m1·m2·m3)
// valid for α ≤ S/2; use V(α) = 1 - V(S-α) for α > S/2.
// ----------------------------------------------------------------------------
// Evaluate the inclusion-exclusion formula for alpha in [0, S/2] (first half only).
// Caller must apply symmetry if alpha > S/2.
__device__ __forceinline__ float plicVolumeFirstHalf(float alpha,
                                                      float m1, float m2, float m3) {
    // 2D degenerate case (m3 ≈ 0, e.g. quasi-2D or thin-slab domain)
    if (m3 < 1e-8f) {
        float S2 = m1 + m2;
        if (alpha >= S2) return 1.0f;
        if (alpha <= 0.0f) return 0.0f;
        float vol2d;
        if (alpha <= m2) {
            vol2d = (alpha * alpha) / (2.0f * m1 * m2);
        } else if (alpha <= m1) {
            vol2d = (alpha - 0.5f * m2) / m1;
        } else {
            float t = S2 - alpha;
            vol2d = 1.0f - (t * t) / (2.0f * m1 * m2);
        }
        return fmaxf(0.0f, fminf(1.0f, vol2d));
    }

    // 3D inclusion-exclusion (Scardovelli & Zaleski 2000, eq. 28)
    float denom = 6.0f * m1 * m2 * m3;
    float a = alpha;
    float vol = a * a * a;
    float t1 = a - m1; if (t1 > 0.0f) vol -= t1 * t1 * t1;
    float t2 = a - m2; if (t2 > 0.0f) vol -= t2 * t2 * t2;
    float t3 = a - m3; if (t3 > 0.0f) vol -= t3 * t3 * t3;
    float t12 = a - m1 - m2; if (t12 > 0.0f) vol += t12 * t12 * t12;
    float t13 = a - m1 - m3; if (t13 > 0.0f) vol += t13 * t13 * t13;
    float t23 = a - m2 - m3; if (t23 > 0.0f) vol += t23 * t23 * t23;
    return fmaxf(0.0f, fminf(1.0f, vol / denom));
}

// Volume below plane m1*x + m2*y + m3*z = alpha in unit cube [0,1]^3.
// Requires m1 >= m2 >= m3 >= 0. Non-recursive.
__device__ float plicVolume3D(float alpha, float m1, float m2, float m3) {
    if (alpha <= 0.0f) return 0.0f;
    float S = m1 + m2 + m3;
    if (alpha >= S) return 1.0f;
    // Use symmetry V(alpha) = 1 - V(S-alpha) for the upper half
    if (alpha > 0.5f * S) {
        return 1.0f - plicVolumeFirstHalf(S - alpha, m1, m2, m3);
    }
    return plicVolumeFirstHalf(alpha, m1, m2, m3);
}

// ----------------------------------------------------------------------------
// Device helper: find α such that plicVolume3D(α, m1, m2, m3) = C
// Uses bisection (40 iterations → machine precision for float).
// ----------------------------------------------------------------------------
// Find alpha s.t. plicVolume3D(alpha, m1, m2, m3) = C. Non-recursive.
__device__ float plicAlphaFromVolume3D(float C, float m1, float m2, float m3) {
    C = fmaxf(1e-8f, fminf(1.0f - 1e-8f, C));
    float S = m1 + m2 + m3;

    // Exploit symmetry: V(S-α) = 1 - V(α)
    // For C > 0.5 → solve for (1-C) in [0, S/2], then reflect
    bool flipped = (C > 0.5f);
    float Csearch = flipped ? (1.0f - C) : C;

    // Bisect in [0, S/2] where V is monotone increasing 0→0.5
    float a_lo = 0.0f, a_hi = 0.5f * S;
    for (int iter = 0; iter < 56; ++iter) {  // 56 iterations for float precision
        float a_mid = 0.5f * (a_lo + a_hi);
        float v = plicVolumeFirstHalf(a_mid, m1, m2, m3);
        if (v < Csearch) a_lo = a_mid; else a_hi = a_mid;
        if (a_hi - a_lo < 1e-7f * (S + 1e-30f)) break;
    }
    float alpha = 0.5f * (a_lo + a_hi);
    return flipped ? (S - alpha) : alpha;
}

// ----------------------------------------------------------------------------
// Device helper: volume of fluid in a rectangular box [0,Lx]×[0,Ly]×[0,Lz]
// below the plane nx*x + ny*y + nz*z = alpha_orig (original coordinates).
// Handles sign flips (negative normals) and scaling to unit cube internally.
// ----------------------------------------------------------------------------
__device__ float plicVolumeInBox(float alpha_orig,
                                 float nx_raw, float ny_raw, float nz_raw,
                                 float Lx, float Ly, float Lz) {
    // Handle sign flips: for each negative component, reflect that axis.
    // If ni < 0: xi → Ldi - xi, which maps alpha → alpha + |ni| * Ldi.
    float alpha = alpha_orig;
    float anx = fabsf(nx_raw), any = fabsf(ny_raw), anz = fabsf(nz_raw);
    if (nx_raw < 0.0f) alpha += anx * Lx;
    if (ny_raw < 0.0f) alpha += any * Ly;
    if (nz_raw < 0.0f) alpha += anz * Lz;

    // Scale to unit cube: x' = x/Lx, y' = y/Ly, z' = z/Lz
    // Plane: (anx*Lx)*x' + (any*Ly)*y' + (anz*Lz)*z' = alpha
    float mx = anx * Lx;
    float my = any * Ly;
    float mz = anz * Lz;

    // Sort so m1 ≥ m2 ≥ m3 (required by plicVolume3D)
    if (mx < my) { float t = mx; mx = my; my = t; }
    if (mx < mz) { float t = mx; mx = mz; mz = t; }
    if (my < mz) { float t = my; my = mz; mz = t; }

    // Volume in unit cube times box volume
    float unit_vol = plicVolume3D(alpha, mx, my, mz);
    return unit_vol * Lx * Ly * Lz;
}

// ============================================================================
// Kernel 1: Youngs Normal Computation (Parker-Youngs 3×3×3 stencil)
// ============================================================================
__global__ void computePlicNormalsKernel(
    const float* __restrict__ fill,
    float* __restrict__ nx_out,
    float* __restrict__ ny_out,
    float* __restrict__ nz_out,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);
    float f = fill[idx];

    // Bulk cells: no plane needed, zero normal
    if (f <= 0.0f || f >= 1.0f) {
        nx_out[idx] = 0.0f;
        ny_out[idx] = 0.0f;
        nz_out[idx] = 0.0f;
        return;
    }

    // Neighbor index helpers respecting boundary conditions
    int im = (bc_x == 0) ? ((i > 0)      ? i-1 : nx-1) : max(0,    i-1);
    int ip = (bc_x == 0) ? ((i < nx-1)   ? i+1 : 0)    : min(nx-1, i+1);
    int jm = (bc_y == 0) ? ((j > 0)      ? j-1 : ny-1) : max(0,    j-1);
    int jp = (bc_y == 0) ? ((j < ny-1)   ? j+1 : 0)    : min(ny-1, j+1);
    int km = (bc_z == 0) ? ((k > 0)      ? k-1 : nz-1) : max(0,    k-1);
    int kp = (bc_z == 0) ? ((k < nz-1)   ? k+1 : 0)    : min(nz-1, k+1);

    // Parker-Youngs 3×3×3 weighted gradient:
    //   ∂C/∂x ≈ Σ_{dj,dk} w(dj,dk) * (C[ip,j+dj,k+dk] - C[im,j+dj,k+dk]) / (2*h)
    //   weights: face=2, edge=1, corner=0.5  (total x-weight = 4·(2+1+1+1+1+0.5*4) = ... )
    // Implemented as weighted sum over the 3×3 face slices:
    //   center slice (dj=0,dk=0): weight 2
    //   edge faces (|dj|+|dk|=1): weight 1
    //   corners (|dj|=|dk|=1): weight 0.5

    // Macro: read fill level with BC-aware index clamping/wrapping
    // Evaluates to a float from global memory. ii/jj/kk may be negative or OOB.
#define FILL(ii, jj, kk) \
    fill[ \
        ((bc_x == 0) ? (((ii) % nx + nx) % nx) : max(0, min(nx-1, (ii)))) \
        + nx * ( \
            ((bc_y == 0) ? (((jj) % ny + ny) % ny) : max(0, min(ny-1, (jj)))) \
            + ny * ((bc_z == 0) ? (((kk) % nz + nz) % nz) : max(0, min(nz-1, (kk)))) \
        ) \
    ]

    // X-gradient (Youngs 1982, Parker-Youngs 3×3×3 extension)
    // weights: center face = 2, edge = 1, corner = 0.5
    float gx = 0.0f;
    gx += 2.0f * (FILL(ip,j ,k ) - FILL(im,j ,k ));   // center face: w=2
    gx += 1.0f * (FILL(ip,jp,k ) - FILL(im,jp,k ));   // y-edge: w=1
    gx += 1.0f * (FILL(ip,jm,k ) - FILL(im,jm,k ));
    gx += 1.0f * (FILL(ip,j ,kp) - FILL(im,j ,kp));   // z-edge: w=1
    gx += 1.0f * (FILL(ip,j ,km) - FILL(im,j ,km));
    gx += 0.5f * (FILL(ip,jp,kp) - FILL(im,jp,kp));   // corners: w=0.5
    gx += 0.5f * (FILL(ip,jp,km) - FILL(im,jp,km));
    gx += 0.5f * (FILL(ip,jm,kp) - FILL(im,jm,kp));
    gx += 0.5f * (FILL(ip,jm,km) - FILL(im,jm,km));

    // Y-gradient
    float gy = 0.0f;
    gy += 2.0f * (FILL(i ,jp,k ) - FILL(i ,jm,k ));
    gy += 1.0f * (FILL(ip,jp,k ) - FILL(ip,jm,k ));
    gy += 1.0f * (FILL(im,jp,k ) - FILL(im,jm,k ));
    gy += 1.0f * (FILL(i ,jp,kp) - FILL(i ,jm,kp));
    gy += 1.0f * (FILL(i ,jp,km) - FILL(i ,jm,km));
    gy += 0.5f * (FILL(ip,jp,kp) - FILL(ip,jm,kp));
    gy += 0.5f * (FILL(ip,jp,km) - FILL(ip,jm,km));
    gy += 0.5f * (FILL(im,jp,kp) - FILL(im,jm,kp));
    gy += 0.5f * (FILL(im,jp,km) - FILL(im,jm,km));

    // Z-gradient
    float gz = 0.0f;
    gz += 2.0f * (FILL(i ,j ,kp) - FILL(i ,j ,km));
    gz += 1.0f * (FILL(ip,j ,kp) - FILL(ip,j ,km));
    gz += 1.0f * (FILL(im,j ,kp) - FILL(im,j ,km));
    gz += 1.0f * (FILL(i ,jp,kp) - FILL(i ,jp,km));
    gz += 1.0f * (FILL(i ,jm,kp) - FILL(i ,jm,km));
    gz += 0.5f * (FILL(ip,jp,kp) - FILL(ip,jp,km));
    gz += 0.5f * (FILL(ip,jm,kp) - FILL(ip,jm,km));
    gz += 0.5f * (FILL(im,jp,kp) - FILL(im,jp,km));
    gz += 0.5f * (FILL(im,jm,kp) - FILL(im,jm,km));

#undef FILL

    // Interface normal points from liquid to gas: n̂ = -∇C / |∇C|
    // (∇C points toward increasing C = toward liquid)
    float mag = sqrtf(gx*gx + gy*gy + gz*gz);
    if (mag < 1e-8f) {
        // Flat or unresolved region: use (1,0,0) fallback
        nx_out[idx] = 1.0f;
        ny_out[idx] = 0.0f;
        nz_out[idx] = 0.0f;
    } else {
        nx_out[idx] = -gx / mag;
        ny_out[idx] = -gy / mag;
        nz_out[idx] = -gz / mag;
    }
}

// ============================================================================
// Kernel 2: Alpha (plane constant) inversion — one thread per cell
// ============================================================================
__global__ void computePlicAlphaKernel(
    const float* __restrict__ fill,
    const float* __restrict__ nx_in,
    const float* __restrict__ ny_in,
    const float* __restrict__ nz_in,
    float* __restrict__ alpha_out,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill[idx];
    if (f <= 0.0f || f >= 1.0f) {
        alpha_out[idx] = f;  // 0 or 1 (unused in flux kernel for bulk cells)
        return;
    }

    float pnx = nx_in[idx];
    float pny = ny_in[idx];
    float pnz = nz_in[idx];

    // Convention: store alpha for the equation with ABSOLUTE VALUE normals in unit cube.
    // That is, we find alpha s.t.:
    //   Volume({x in [0,1]^3 : |nx|*x + |ny|*y + |nz|*z < alpha_abs}) = f
    // where alpha_abs is in the sorted-|n| frame.
    //
    // When the flux kernel calls plicVolumeInBox(alpha_signed, nx_signed, ny_signed, ...),
    // it first does sign-flip reflections that map alpha_signed → alpha_abs by adding
    // |ni| * Li for each negative ni. So we must store:
    //   alpha_out = alpha_abs - Σ_{ni < 0} |ni|   (in the original signed frame)
    // so that plicVolumeInBox recovers alpha_abs correctly.
    //
    // In the full-cell case (Lx=Ly=Lz=1), this adjustment is:
    //   alpha_signed = alpha_abs - (|pnx| if pnx<0 else 0) - (|pny| if pny<0 else 0) - ...

    float mx = fabsf(pnx);
    float my = fabsf(pny);
    float mz = fabsf(pnz);

    // Sort for plicAlphaFromVolume3D: m1 ≥ m2 ≥ m3
    float sm1 = mx, sm2 = my, sm3 = mz;
    if (sm1 < sm2) { float t = sm1; sm1 = sm2; sm2 = t; }
    if (sm1 < sm3) { float t = sm1; sm1 = sm3; sm3 = t; }
    if (sm2 < sm3) { float t = sm2; sm2 = sm3; sm3 = t; }

    // Find alpha in the sorted-|n| frame (unit cube, all-positive normals)
    float alpha_abs = plicAlphaFromVolume3D(f, sm1, sm2, sm3);

    // Convert to signed frame: subtract contributions for negative components
    // plicVolumeInBox will add them back via its sign-flip logic
    float alpha_signed = alpha_abs;
    if (pnx < 0.0f) alpha_signed -= mx;   // subtract |pnx| * Lx (Lx=1)
    if (pny < 0.0f) alpha_signed -= my;   // subtract |pny| * Ly (Ly=1)
    if (pnz < 0.0f) alpha_signed -= mz;   // subtract |pnz| * Lz (Lz=1)

    alpha_out[idx] = alpha_signed;
}

// ============================================================================
// Kernel 3: Face velocity interpolation (cell-centered → face-centered)
// ============================================================================
// For dir=0 (x-faces): output array has (nx+1)*ny*nz entries
// face index: fi + (nx+1)*(j + ny*k),  fi = 0..nx  (faces between cells)
// ============================================================================
__global__ void interpolateFaceVelocityKernel(
    const float* __restrict__ u_cell,
    float* __restrict__ u_face,
    int nx, int ny, int nz,
    int dir,     // 0=x-faces, 1=y-faces, 2=z-faces
    int bc_dir)  // 0=periodic, 1=wall
{
    // Grid layout: each thread handles one face
    // Launch with face-count threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (dir == 0) {
        // x-faces: (nx+1)*ny*nz faces, index fi + (nx+1)*(j + ny*k)
        int total = (nx+1) * ny * nz;
        if (idx >= total) return;
        int fi = idx % (nx+1);
        int tmp = idx / (nx+1);
        int j  = tmp % ny;
        int k  = tmp / ny;

        // Wall BC: zero velocity at domain boundaries
        if (bc_dir != 0 && (fi == 0 || fi == nx)) {
            u_face[idx] = 0.0f;
            return;
        }
        // Periodic or interior: average adjacent cell values
        int il = (fi == 0)  ? (bc_dir == 0 ? nx-1 : 0)    : fi-1;
        int ir = (fi == nx) ? (bc_dir == 0 ? 0    : nx-1)  : fi;
        int cidx_l = il + nx * (j + ny * k);
        int cidx_r = ir + nx * (j + ny * k);
        u_face[idx] = 0.5f * (u_cell[cidx_l] + u_cell[cidx_r]);

    } else if (dir == 1) {
        // y-faces: nx*(ny+1)*nz
        int total = nx * (ny+1) * nz;
        if (idx >= total) return;
        int i  = idx % nx;
        int tmp = idx / nx;
        int fj  = tmp % (ny+1);
        int k   = tmp / (ny+1);

        if (bc_dir != 0 && (fj == 0 || fj == ny)) {
            u_face[idx] = 0.0f;
            return;
        }
        int jl = (fj == 0)  ? (bc_dir == 0 ? ny-1 : 0)    : fj-1;
        int jr = (fj == ny) ? (bc_dir == 0 ? 0    : ny-1)  : fj;
        int cidx_l = i + nx * (jl + ny * k);
        int cidx_r = i + nx * (jr + ny * k);
        u_face[idx] = 0.5f * (u_cell[cidx_l] + u_cell[cidx_r]);

    } else {
        // z-faces: nx*ny*(nz+1)
        int total = nx * ny * (nz+1);
        if (idx >= total) return;
        int i  = idx % nx;
        int tmp = idx / nx;
        int j   = tmp % ny;
        int fk  = tmp / ny;

        if (bc_dir != 0 && (fk == 0 || fk == nz)) {
            u_face[idx] = 0.0f;
            return;
        }
        int kl = (fk == 0)  ? (bc_dir == 0 ? nz-1 : 0)    : fk-1;
        int kr = (fk == nz) ? (bc_dir == 0 ? 0    : nz-1)  : fk;
        int cidx_l = i + nx * (j + ny * kl);
        int cidx_r = i + nx * (j + ny * kr);
        u_face[idx] = 0.5f * (u_cell[cidx_l] + u_cell[cidx_r]);
    }
}

// ============================================================================
// Kernel 4: Geometric PLIC flux computation (one direction per call)
// ============================================================================
// Each thread computes the volumetric flux through one face.
// The flux is in units of [cell volume] (dimensionless fill fraction × volume).
//
// Convention: flux[face] = volume of fluid that crosses this face in time dt.
//   Positive = flow in +dir direction.
// ============================================================================
__global__ void computePlicFluxKernel(
    const float* __restrict__ fill,
    const float* __restrict__ plic_nx,
    const float* __restrict__ plic_ny,
    const float* __restrict__ plic_nz,
    const float* __restrict__ plic_alpha,
    const float* __restrict__ u_face,
    float* __restrict__ flux_out,
    float dt, float dx,
    int nx, int ny, int nz,
    int dir, int bc_dir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // -------------------------------------------------------------------------
    // Decode face index → (fi, j, k) depending on direction
    // -------------------------------------------------------------------------
    int fi = 0, fj = 0, fk = 0;
    int donor_i = 0, donor_j = 0, donor_k = 0;

    if (dir == 0) {
        int total = (nx+1) * ny * nz;
        if (idx >= total) return;
        fi = idx % (nx+1);
        int tmp = idx / (nx+1);
        fj = tmp % ny;
        fk = tmp / ny;
    } else if (dir == 1) {
        int total = nx * (ny+1) * nz;
        if (idx >= total) return;
        fi = idx % nx;
        int tmp = idx / nx;
        fj = tmp % (ny+1);
        fk = tmp / (ny+1);
    } else {
        int total = nx * ny * (nz+1);
        if (idx >= total) return;
        fi = idx % nx;
        int tmp = idx / nx;
        fj = tmp % ny;
        fk = tmp / ny;
    }

    float u = u_face[idx];

    // Wall faces: always zero flux
    if (bc_dir != 0) {
        bool is_boundary = (dir == 0 && (fi == 0 || fi == nx)) ||
                           (dir == 1 && (fj == 0 || fj == ny)) ||
                           (dir == 2 && (fk == 0 || fk == nz));
        if (is_boundary) { flux_out[idx] = 0.0f; return; }
    }

    // Zero velocity: no flux
    if (fabsf(u) < 1e-15f) { flux_out[idx] = 0.0f; return; }

    // -------------------------------------------------------------------------
    // Determine donor cell (upwind cell)
    // For x-faces: face fi sits between cell fi-1 (left) and cell fi (right)
    // u > 0 → material moves right → donor = left cell (fi-1, fj, fk)
    // u < 0 → material moves left  → donor = right cell (fi, fj, fk)
    // -------------------------------------------------------------------------
    if (dir == 0) {
        if (u > 0.0f) {
            donor_i = fi - 1;
            donor_j = fj;
            donor_k = fk;
            // Periodic wrap for left-most face
            if (donor_i < 0) donor_i = (bc_dir == 0) ? nx-1 : 0;
        } else {
            donor_i = fi;
            donor_j = fj;
            donor_k = fk;
            if (donor_i >= nx) donor_i = (bc_dir == 0) ? 0 : nx-1;
        }
    } else if (dir == 1) {
        if (u > 0.0f) {
            donor_i = fi;
            donor_j = fj - 1;
            donor_k = fk;
            if (donor_j < 0) donor_j = (bc_dir == 0) ? ny-1 : 0;
        } else {
            donor_i = fi;
            donor_j = fj;
            donor_k = fk;
            if (donor_j >= ny) donor_j = (bc_dir == 0) ? 0 : ny-1;
        }
    } else {
        if (u > 0.0f) {
            donor_i = fi;
            donor_j = fj;
            donor_k = fk - 1;
            if (donor_k < 0) donor_k = (bc_dir == 0) ? nz-1 : 0;
        } else {
            donor_i = fi;
            donor_j = fj;
            donor_k = fk;
            if (donor_k >= nz) donor_k = (bc_dir == 0) ? 0 : nz-1;
        }
    }

    // Clamp donor indices to valid range
    donor_i = max(0, min(nx-1, donor_i));
    donor_j = max(0, min(ny-1, donor_j));
    donor_k = max(0, min(nz-1, donor_k));

    int d_idx = donor_i + nx * (donor_j + ny * donor_k);

    float f  = fill[d_idx];
    float d  = fabsf(u) * dt / dx;  // CFL depth in [0,1] units

    // Clamp CFL to 1 for safety (should be ≤ 1 if CFL condition holds)
    d = fminf(d, 1.0f);

    // -------------------------------------------------------------------------
    // Compute geometric flux volume using PLIC plane in the departure slab
    // -------------------------------------------------------------------------
    float vol_flux;

    // Near-uniform and out-of-range cells: use simple upwind flux = f*d.
    // PLIC reconstruction is ill-conditioned for f ≈ 0 or f ≈ 1 because
    // the interface normal is dominated by floating-point noise, producing
    // spurious confetti fragments. The 1e-6 threshold catches these cases.
    // For f outside [0,1] (Strang overshoot), upwind preserves the excess.
    if (f < 1e-6f || f > (1.0f - 1e-6f)) {
        vol_flux = f * d;
    } else {
        // Interface cell: compute volume in departure slab
        float pnx_d = plic_nx[d_idx];
        float pny_d = plic_ny[d_idx];
        float pnz_d = plic_nz[d_idx];
        float alpha = plic_alpha[d_idx];  // In |n|-sorted frame

        // The slab geometry depends on direction and flow sign:
        //   u > 0: departure slab is the RIGHT portion of donor cell: [1-d,1]×[0,1]×[0,1] (for dir=0)
        //   u < 0: departure slab is the LEFT  portion of donor cell: [0,d]×[0,1]×[0,1]
        //
        // For u > 0 (right face): translate x' = x - (1-d) so slab becomes [0,d]×[0,1]×[0,1]
        //   New alpha: alpha' = alpha - n_sweep * (1-d)
        //   where n_sweep is the component of n in the sweep direction.
        // For u < 0 (left face): slab is already [0,d]×[0,1]×[0,1], no translation.
        //
        // Then scale the slab to unit cube [0,1]³ for the sweep dimension:
        //   vol in [0,d]×[0,1]×[0,1] = d * plicVolume3D(alpha' , |n_sweep|*d, |ny|, |nz|)
        // The sign-flip convention for negative normals is handled inside plicVolumeInBox.

        float n_sweep;
        float Lx_slab, Ly_slab, Lz_slab;
        float alpha_slab;

        if (dir == 0) {
            n_sweep = pnx_d;
            if (u > 0.0f) {
                // Right slab [1-d, 1]×[0,1]×[0,1] → translate x' = x-(1-d)
                // alpha' = alpha - n_sweep*(1-d)
                alpha_slab = alpha - n_sweep * (1.0f - d);
            } else {
                // Left slab [0, d]×[0,1]×[0,1], no translation
                alpha_slab = alpha;
            }
            Lx_slab = d;
            Ly_slab = 1.0f;
            Lz_slab = 1.0f;
            vol_flux = plicVolumeInBox(alpha_slab, n_sweep, pny_d, pnz_d,
                                       Lx_slab, Ly_slab, Lz_slab);
        } else if (dir == 1) {
            n_sweep = pny_d;
            if (u > 0.0f) {
                alpha_slab = alpha - n_sweep * (1.0f - d);
            } else {
                alpha_slab = alpha;
            }
            Lx_slab = 1.0f;
            Ly_slab = d;
            Lz_slab = 1.0f;
            vol_flux = plicVolumeInBox(alpha_slab, pnx_d, n_sweep, pnz_d,
                                       Lx_slab, Ly_slab, Lz_slab);
        } else {
            n_sweep = pnz_d;
            if (u > 0.0f) {
                alpha_slab = alpha - n_sweep * (1.0f - d);
            } else {
                alpha_slab = alpha;
            }
            Lx_slab = 1.0f;
            Ly_slab = 1.0f;
            Lz_slab = d;
            vol_flux = plicVolumeInBox(alpha_slab, pnx_d, pny_d, n_sweep,
                                       Lx_slab, Ly_slab, Lz_slab);
        }

        // vol_flux is in physical units [dx³ fractions], normalize to per-cell-volume
        // Actually plicVolumeInBox returns vol in units of Lx*Ly*Lz (already in dx units)
        // We want flux in [cell volume] = 1 (normalized), so vol_flux is already correct.
        vol_flux = fmaxf(0.0f, fminf(d, vol_flux));
    }

    // Sign: flux is positive in the +dir direction
    // (we store signed flux; updateVofFromFlux uses it as: C -= flux_right - flux_left)
    flux_out[idx] = (u >= 0.0f) ? vol_flux : -vol_flux;
}

// ============================================================================
// Kernel 5: VOF update from PLIC fluxes (one direction)
// ============================================================================
// C_new[i,j,k] = C_old[i,j,k] - (flux[i+1/2] - flux[i-1/2])
// No clamping: geometric fluxes are naturally bounded.
// ============================================================================
__global__ void updateVofFromFluxKernel(
    const float* __restrict__ fill_old,
    float* __restrict__ fill_new,
    const float* __restrict__ flux,
    const float* __restrict__ u_face,   // face velocities for divergence correction
    float dt, float dx,
    int nx, int ny, int nz,
    int dir, int bc_dir)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);
    float f = fill_old[idx];

    float flux_plus, flux_minus;
    float u_plus, u_minus;

    if (dir == 0) {
        int fp = (i+1) + (nx+1) * (j + ny * k);
        int fm = (i)   + (nx+1) * (j + ny * k);
        flux_plus  = flux[fp];
        flux_minus = flux[fm];
        u_plus  = u_face[fp];
        u_minus = u_face[fm];
    } else if (dir == 1) {
        int fp = i + nx * ((j+1) + (ny+1) * k);
        int fm = i + nx * ( j    + (ny+1) * k);
        flux_plus  = flux[fp];
        flux_minus = flux[fm];
        u_plus  = u_face[fp];
        u_minus = u_face[fm];
    } else {
        int fp = i + nx * (j + ny * (k+1));
        int fm = i + nx * (j + ny *  k   );
        flux_plus  = flux[fp];
        flux_minus = flux[fm];
        u_plus  = u_face[fp];
        u_minus = u_face[fm];
    }

    // Weymouth-Yue (2010) divergence-corrected VOF update:
    //   f_new = f * (1 + δ) - (Φ+ - Φ-)
    //   where δ = dt * (u_face+ - u_face-) / dx  (local velocity divergence)
    //
    // Without this correction, weakly compressible LBM velocity fields
    // (∇·u ~ O(Ma²)) cause systematic mass loss of ~3% over 4000 steps.
    // The δ term compensates for the cell "stretching" due to divergence,
    // keeping the volume fraction consistent with the actual fluid volume.
    //
    // WALL FIX: At wall-adjacent cells, the wall face velocity is forced to 0
    // by interpolateFaceVelocityKernel, but the interior face retains the
    // physical velocity. This creates artificial divergence (δ ≠ 0) that acts
    // as a mass sink/source. Fix: mirror the interior face velocity to the
    // wall face for the divergence computation only. The flux at the wall
    // face is already correctly zero, so this only affects the δ term.
    float u_div_plus = u_plus, u_div_minus = u_minus;
    if (bc_dir != 0) {
        if (dir == 0) {
            if (i == 0)      u_div_minus = u_div_plus;   // x_min wall
            if (i == nx - 1) u_div_plus  = u_div_minus;  // x_max wall
        } else if (dir == 1) {
            if (j == 0)      u_div_minus = u_div_plus;   // y_min wall
            if (j == ny - 1) u_div_plus  = u_div_minus;  // y_max wall
        } else {
            if (k == 0)      u_div_minus = u_div_plus;   // z_min wall
            if (k == nz - 1) u_div_plus  = u_div_minus;  // z_max wall
        }
    }
    float delta = dt * (u_div_plus - u_div_minus) / dx;
    float f_new = f * (1.0f + delta) - (flux_plus - flux_minus);

    fill_new[idx] = f_new;
}

// ============================================================================
// Post-sweep wall sealing: zero-gradient BC on fill_level at WALL faces
// ============================================================================
__global__ void plicWallSealKernel(
    float* __restrict__ fill,
    int nx, int ny, int nz,
    int seal_y, int seal_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    if (seal_y) {
        if (j == 0)      fill[idx] = fill[i + nx * (1       + ny * k)];
        if (j == ny - 1) fill[idx] = fill[i + nx * ((ny-2)  + ny * k)];
    }
    if (seal_z) {
        if (k == 0)      fill[idx] = fill[i + nx * (j + ny * 1)];
    }
}

// ============================================================================
// Post-sweep floating-point tolerance clamp for PLIC
// ============================================================================
__global__ void plicFinalClampKernel(float* __restrict__ fill, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    fill[idx] = fmaxf(0.0f, fminf(1.0f, fill[idx]));
}

// ============================================================================
// VOFSolver::plicAllocateIfNeeded()
// ============================================================================
void VOFSolver::plicAllocateIfNeeded() {
    if (plic_nx_.size() > 0) return;

    int N = num_cells_;
    int max_face = std::max({(nx_+1)*ny_*nz_,
                            nx_*(ny_+1)*nz_,
                            nx_*ny_*(nz_+1)});

    plic_nx_       = lbm::utils::CudaBuffer<float>(N);
    plic_ny_       = lbm::utils::CudaBuffer<float>(N);
    plic_nz_       = lbm::utils::CudaBuffer<float>(N);
    plic_alpha_    = lbm::utils::CudaBuffer<float>(N);
    plic_flux_     = lbm::utils::CudaBuffer<float>(max_face);
    plic_face_vel_ = lbm::utils::CudaBuffer<float>(max_face);

    plic_nx_.zero();
    plic_ny_.zero();
    plic_nz_.zero();
    plic_alpha_.zero();
    plic_flux_.zero();
    plic_face_vel_.zero();
}

// ============================================================================
// VOFSolver::advectFillLevelPLIC()
// Strang-split directional sweeps with full PLIC reconstruction each sweep.
// ============================================================================
void VOFSolver::advectFillLevelPLIC(const float* d_ux, const float* d_uy,
                                     const float* d_uz, float dt) {
    plicAllocateIfNeeded();

    int N = num_cells_;
    int bcs[3] = { static_cast<int>(bc_x_),
                   static_cast<int>(bc_y_),
                   static_cast<int>(bc_z_) };

    // Face array sizes for each direction
    int face_sizes[3] = {
        (nx_+1) * ny_  * nz_,
        nx_  * (ny_+1) * nz_,
        nx_  * ny_  * (nz_+1)
    };

    const float* vel_ptrs[3] = { d_ux, d_uy, d_uz };

    // Strang splitting: alternate XYZ / YXZ order each step
    int sweeps[3];
    if (plic_strang_x_first_) {
        sweeps[0] = 0; sweeps[1] = 1; sweeps[2] = 2;
    } else {
        sweeps[0] = 1; sweeps[1] = 0; sweeps[2] = 2;
    }

    // Work on d_fill_level_ / d_fill_level_tmp_ ping-pong
    float* src = d_fill_level_;
    float* dst = d_fill_level_tmp_;

    // 1-D block for flat kernels
    dim3 cell_block(256);
    dim3 cell_grid((N + 255) / 256);

    // 3-D block for normal kernel (same as existing kernels)
    dim3 blk3(8, 8, 8);
    dim3 grd3((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 7) / 8);

    for (int s = 0; s < 3; ++s) {
        int dir = sweeps[s];
        int face_N = face_sizes[dir];
        dim3 face_block(256);
        dim3 face_grid((face_N + 255) / 256);

        // Step 1: Compute Youngs normals from current fill
        computePlicNormalsKernel<<<grd3, blk3>>>(
            src,
            plic_nx_.get(), plic_ny_.get(), plic_nz_.get(),
            nx_, ny_, nz_, bcs[0], bcs[1], bcs[2]);
        CUDA_CHECK_KERNEL();

        // Step 2: Invert alpha from volume fraction + normal
        computePlicAlphaKernel<<<cell_grid, cell_block>>>(
            src,
            plic_nx_.get(), plic_ny_.get(), plic_nz_.get(),
            plic_alpha_.get(),
            N);
        CUDA_CHECK_KERNEL();

        // Step 3: Interpolate face velocities for this direction
        interpolateFaceVelocityKernel<<<face_grid, face_block>>>(
            vel_ptrs[dir], plic_face_vel_.get(),
            nx_, ny_, nz_, dir, bcs[dir]);
        CUDA_CHECK_KERNEL();

        // Step 4: Compute geometric flux at each face
        computePlicFluxKernel<<<face_grid, face_block>>>(
            src,
            plic_nx_.get(), plic_ny_.get(), plic_nz_.get(),
            plic_alpha_.get(),
            plic_face_vel_.get(),
            plic_flux_.get(),
            dt, dx_,
            nx_, ny_, nz_, dir, bcs[dir]);
        CUDA_CHECK_KERNEL();

        // Step 5: Update VOF fill level from fluxes (with divergence correction)
        updateVofFromFluxKernel<<<grd3, blk3>>>(
            src, dst, plic_flux_.get(),
            plic_face_vel_.get(),
            dt, dx_,
            nx_, ny_, nz_, dir, bcs[dir]);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap ping-pong buffers
        float* tmp = src; src = dst; dst = tmp;
    }

    // Ensure d_fill_level_ holds the final result
    if (src != d_fill_level_) {
        CUDA_CHECK(cudaMemcpy(d_fill_level_, src, N * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Diagnostic: measure mass before and after clamp to quantify loss
    static int plic_call = 0;
    if (plic_call % 500 == 0) {
        float mass_before = computeTotalMass();
        plicFinalClampKernel<<<(N + 255) / 256, 256>>>(d_fill_level_, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        float mass_after = computeTotalMass();
        float clamp_loss = mass_before - mass_after;
        printf("[PLIC CLAMP] Call %d: mass_before=%.1f, mass_after=%.1f, "
               "clamp_deleted=%.4f (%.4f%%)\n",
               plic_call, mass_before, mass_after,
               clamp_loss, clamp_loss / mass_before * 100.0f);
    } else {
        plicFinalClampKernel<<<(N + 255) / 256, 256>>>(d_fill_level_, N);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    plic_call++;

    // WALL BOUNDARY SEALING: zero-gradient on fill_level at WALL faces.
    // Seals Y-walls and Z-min to prevent side fragmentation.
    {
        dim3 blk3(8, 8, 8);
        dim3 grd3((nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 7) / 8);
        if (bc_y_ != VOFSolver::BoundaryType::PERIODIC ||
            bc_z_ != VOFSolver::BoundaryType::PERIODIC) {
            plicWallSealKernel<<<grd3, blk3>>>(
                d_fill_level_, nx_, ny_, nz_,
                (bc_y_ != VOFSolver::BoundaryType::PERIODIC) ? 1 : 0,
                (bc_z_ != VOFSolver::BoundaryType::PERIODIC) ? 1 : 0);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Mass-correction (Track-A v_z OR Track-B inline-∇f) shared with TVD path.
    if (mass_correction_use_flux_weight_) {
        applyMassCorrectionInline(d_ux, d_uy, d_uz);   // Track-B
    } else {
        applyMassCorrectionInline(d_uz);               // Track-A
    }

    // Alternate Strang sweep order for next call
    plic_strang_x_first_ = !plic_strang_x_first_;
}

} // namespace physics
} // namespace lbm
