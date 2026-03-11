/**
 * @file viz_marangoni_cavity.cu
 * @brief 2D Thermocapillary Convection in Square Cavity
 *
 * Benchmark: Zebib, Homsy & Meiburg (1985), Ma=100/1000, Pr=1
 * Setup: Hot left wall, cold right wall, adiabatic bottom, free surface top
 * Physics: Surface tension gradient drives flow from hot→cold at top surface
 *
 * Method:
 *   Fluid:   D3Q19 TRT (Λ=3/16), WALL on x,y, PERIODIC on z
 *   Thermal: D3Q7 BGK, Dirichlet left/right, adiabatic top/bottom
 *   Marangoni: Inamuro specular + CE-corrected stress BC at top surface
 *
 * Compile-time options:
 *   -DMA_NUMBER=100    (default 100)
 *   -DGRID_N=100       (default 100, use 200 for Ma=1000)
 *
 * CRITICAL: Thermal BCs must be applied BEFORE computeTemperature()
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <algorithm>

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/lattice_d3q7.h"
#include "core/lattice_d3q19.h"
#include "utils/cuda_check.h"

using lbm::physics::ThermalLBM;
using lbm::physics::FluidLBM;
using lbm::physics::BoundaryType;
using lbm::physics::D3Q7;
using lbm::core::D3Q19;

// ======================== Configurable Parameters ========================
#ifndef MA_NUMBER
#define MA_NUMBER 100
#endif
#ifndef GRID_N
#define GRID_N 100
#endif

static constexpr int NX = GRID_N;
static constexpr int NY = GRID_N;
static constexpr int NZ = 1;
static constexpr int NC = NX * NY * NZ;

static constexpr float MA      = (float)MA_NUMBER;
static constexpr float PR      = 1.0f;
static constexpr float RHO     = 1.0f;

// Auto-compute tau_v: use tau=0.8 when Ma_LBM is safe, reduce for high Ma
// Ma_LBM = U_ref * sqrt(3) = Ma * nu * sqrt(3) / NX
// Target: Ma_LBM < 0.2
static constexpr float TAU_V_PREFERRED = 0.8f;
static constexpr float NU_LB_PREFERRED = (TAU_V_PREFERRED - 0.5f) / 3.0f;
static constexpr float MA_LBM_PREFERRED = MA * NU_LB_PREFERRED * 1.7320508f / (float)NX;
static constexpr float MA_LBM_LIMIT = 0.2f;
static constexpr float NU_LB_SAFE = MA_LBM_LIMIT * (float)NX / (MA * 1.7320508f);
static constexpr float TAU_V_SAFE = 3.0f * NU_LB_SAFE + 0.5f;
// Use preferred tau if Ma_LBM is OK, otherwise use safe tau (clamped >= 0.52)
static constexpr float TAU_V   = (MA_LBM_PREFERRED <= MA_LBM_LIMIT) ? TAU_V_PREFERRED
                                : ((TAU_V_SAFE > 0.52f) ? TAU_V_SAFE : 0.52f);
static constexpr float NU_LB   = (TAU_V - 0.5f) / 3.0f;
static constexpr float U_REF   = MA * NU_LB / (float)NX;
static constexpr float ALPHA_LB= NU_LB / PR;                  // = NU_LB for Pr=1
static constexpr float TAU_T   = 4.0f * ALPHA_LB + 0.5f;
static constexpr float LAMBDA  = 3.0f / 16.0f;

static constexpr float T_HOT   = 1.0f;
static constexpr float T_COLD  = 0.0f;

// γ_T = ∂σ/∂T in lattice units: derived from Ma = γ_T·ΔT·H/(μ·α)
// → γ_T = Ma·μ·α/(ΔT·H) = Ma·ρ·ν²/(ΔT·H) (since α=ν for Pr=1)
// Equivalently: γ_T = U_REF·ρ·ν/ΔT
static constexpr float GAMMA_T = U_REF * RHO * NU_LB;

// Inamuro CE factor: stress = (1-1/(2τ)) × Σ c_α c_β f_neq
static constexpr float CE_FACTOR = 1.0f - 0.5f / TAU_V;

// Simulation control: run for >= 2 thermal diffusion times
// t_diff = H²/α = NX²/ALPHA_LB
static constexpr int   T_DIFF     = (int)((float)(NX * NX) / ALPHA_LB);
static constexpr int   MAX_STEPS  = 4 * T_DIFF;  // 4× diffusion time
static constexpr int   CHECK_INTERVAL = 5000;
static constexpr float CONV_TOL   = 1e-4f;       // relative Nu change

// ======================== CUDA Kernels ========================

// Fused D3Q19 TRT collision + streaming + bounce-back walls + Inamuro BC
__global__ void fusedFluidStep_D3Q19(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    float* __restrict__ rho_out,
    float* __restrict__ ux_out,
    float* __restrict__ uy_out,
    float* __restrict__ uz_out,
    const float* __restrict__ temperature,
    float tau_v, float lambda,
    float gamma_T, float ce_factor,
    int nx, int ny, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    const int nc = nx * ny * nz;
    const int id = ix + iy * nx;

    // D3Q19 lattice vectors and weights
    constexpr int ex[19] = {0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0};
    constexpr int ey[19] = {0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1};
    constexpr int ez[19] = {0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1};
    constexpr int opp[19]= {0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15};
    constexpr float w[19] = {
        1.0f/3.0f,                               // rest
        1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,  // faces
        1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,  // edges xy
        1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,  // edges xz
        1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f   // edges yz
    };

    // TRT relaxation
    float omega_plus = 1.0f / tau_v;
    float tau_minus = 0.5f + lambda / (tau_v - 0.5f);
    float omega_minus = 1.0f / tau_minus;

    // 1. Read distributions and compute macroscopic
    float f[19];
    float m_rho = 0, m_ux = 0, m_uy = 0, m_uz = 0;
    for (int q = 0; q < 19; q++) {
        f[q] = f_src[id + q * nc];
        m_rho += f[q];
        m_ux += ex[q] * f[q];
        m_uy += ey[q] * f[q];
        m_uz += ez[q] * f[q];
    }
    float inv_rho = 1.0f / fmaxf(m_rho, 1e-12f);
    m_ux *= inv_rho;
    m_uy *= inv_rho;
    m_uz *= inv_rho;

    // Store macroscopic
    rho_out[id] = m_rho;
    ux_out[id] = m_ux;
    uy_out[id] = m_uy;
    uz_out[id] = m_uz;

    // 2. TRT collision
    float usq = m_ux*m_ux + m_uy*m_uy + m_uz*m_uz;
    for (int q = 0; q < 19; q++) {
        float cu = ex[q]*m_ux + ey[q]*m_uy + ez[q]*m_uz;
        float feq = w[q] * m_rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usq);

        int qo = opp[q];
        float cu_opp = ex[qo]*m_ux + ey[qo]*m_uy + ez[qo]*m_uz;
        float feq_opp = w[qo] * m_rho * (1.0f + 3.0f*cu_opp + 4.5f*cu_opp*cu_opp - 1.5f*usq);

        float f_sym  = 0.5f * (f[q] + f[qo]);
        float feq_sym = 0.5f * (feq + feq_opp);
        float f_asym  = 0.5f * (f[q] - f[qo]);
        float feq_asym = 0.5f * (feq - feq_opp);

        f[q] -= omega_plus * (f_sym - feq_sym) + omega_minus * (f_asym - feq_asym);
    }

    // 3. Streaming with boundary handling
    bool is_wall = (ix == 0 || ix == nx-1 || iy == 0);
    bool is_top  = (iy == ny - 1);

    if (is_wall) {
        // Bounce-back: post-collision populations reflect back
        for (int q = 1; q < 19; q++) {
            int qo = opp[q];
            f_dst[id + qo * nc] = f[q];
        }
        f_dst[id] = f[0];
    } else if (is_top && ix > 0 && ix < nx - 1) {
        // Inamuro specular + Marangoni stress at free surface
        // Stream interior directions normally
        for (int q = 0; q < 19; q++) {
            int nx_dst = ix + ex[q];
            int ny_dst = iy + ey[q];

            if (ny_dst >= ny) {
                // Would go above: specular reflect handled below
                continue;
            }
            if (nx_dst >= 0 && nx_dst < nx && ny_dst >= 0) {
                int dst_id = nx_dst + ny_dst * nx;
                f_dst[dst_id + q * nc] = f[q];
            }
        }

        // Specular reflections for upward-going populations
        // q=3 (+y) → q=4 (-y)
        f_dst[id + 4 * nc] = f[3];
        // q=7 (+x,+y) → q=9 (+x,-y) but with Marangoni stress
        // q=8 (-x,+y) → q=10 (-x,-y) but with Marangoni stress

        // Compute Marangoni stress
        float T_plus  = temperature[id + 1];
        float T_minus = temperature[id - 1];
        float dTdx = (T_plus - T_minus) * 0.5f;
        float tau_s = -gamma_T * dTdx;
        float delta_f = tau_s / (2.0f * ce_factor);

        f_dst[id +  9 * nc] = f[7] + delta_f;
        f_dst[id + 10 * nc] = f[8] - delta_f;

        // q=15 (0,+y,+z) → q=16 (0,-y,+z)
        // q=17 (0,+y,-z) → q=18 (0,-y,-z)
        f_dst[id + 16 * nc] = f[15];
        f_dst[id + 18 * nc] = f[17];
    } else {
        // Interior cell: push to neighbors
        for (int q = 0; q < 19; q++) {
            int nx_dst = ix + ex[q];
            int ny_dst = iy + ey[q];
            // z is periodic for NZ=1 (stays at iz=0)

            if (nx_dst >= 0 && nx_dst < nx && ny_dst >= 0 && ny_dst < ny) {
                int dst_id = nx_dst + ny_dst * nx;
                f_dst[dst_id + q * nc] = f[q];
            }
            // Distributions going out of bounds bounce back (handled by wall cells)
        }
    }
}

__global__ void applyNoSlipWalls(float* f, int nx, int ny, int nz)
{
    static constexpr int ex[19] = {0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0};
    static constexpr int ey[19] = {0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1};
    static constexpr int opp[19] = {0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15};

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = nx * ny * nz;

    int n_x_wall = 2 * ny * nz;
    int n_y_bot  = nx * nz;
    int n_total  = n_x_wall + n_y_bot;
    if (tid >= n_total) return;

    int ix, iy, iz;
    unsigned int wall_dirs = 0;

    if (tid < n_x_wall) {
        int side = tid / (ny * nz);
        int rem  = tid % (ny * nz);
        iz = rem / ny;
        iy = rem % ny;
        ix = (side == 0) ? 0 : nx - 1;
        wall_dirs = (side == 0) ? 1u : 2u;
        if (iy == 0) wall_dirs |= 4u;
        if (iy == ny - 1) wall_dirs |= 8u;
    } else {
        int bid = tid - n_x_wall;
        iz = bid / nx;
        ix = bid % nx;
        iy = 0;
        wall_dirs = 4u;
        if (ix == 0) wall_dirs |= 1u;
        if (ix == nx - 1) wall_dirs |= 2u;
    }

    int id = ix + iy * nx + iz * nx * ny;

    for (int q = 1; q < 19; q++) {
        bool needs_bc = false;
        if ((wall_dirs & 1u) && ex[q] > 0) needs_bc = true;
        if ((wall_dirs & 2u) && ex[q] < 0) needs_bc = true;
        if ((wall_dirs & 4u) && ey[q] > 0) needs_bc = true;
        if ((wall_dirs & 8u) && ey[q] < 0) needs_bc = true;
        if (needs_bc) {
            f[id + q * n_cells] = f[id + opp[q] * n_cells];
        }
    }
}

__global__ void applyTopInamuroBC(
    float* __restrict__ f,
    const float* __restrict__ temperature,
    int nx, int ny, int nz,
    float gamma_T, float ce_factor)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iz >= nz) return;
    if (ix == 0 || ix == nx - 1) return;  // wall corners: keep bounce-back

    const int n_cells = nx * ny * nz;
    const int id = ix + (ny - 1) * nx + iz * nx * ny;

    // Central difference ∂T/∂x
    float T_plus  = temperature[id + 1];
    float T_minus = temperature[id - 1];
    float dTdx = (T_plus - T_minus) * 0.5f;

    // Marangoni stress τ_s = ∂σ/∂x = −γ_T · ∂T/∂x
    float tau_s   = -gamma_T * dTdx;
    float delta_f = tau_s / (2.0f * ce_factor);

    // q=4 (0,-1,0) ← specular of q=3 (0,+1,0)
    f[id + 4 * n_cells] = f[id + 3 * n_cells];

    // q=9 (1,-1,0) ← specular of q=7 (1,+1,0) + stress
    // q=10 (-1,-1,0) ← specular of q=8 (-1,+1,0) - stress
    float f7 = f[id + 7 * n_cells];
    float f8 = f[id + 8 * n_cells];
    f[id +  9 * n_cells] = f7 + delta_f;
    f[id + 10 * n_cells] = f8 - delta_f;

    // q=16 (0,-1,+1) ← specular of q=15 (0,+1,+1)
    // q=18 (0,-1,-1) ← specular of q=17 (0,+1,-1)
    f[id + 16 * n_cells] = f[id + 15 * n_cells];
    f[id + 18 * n_cells] = f[id + 17 * n_cells];
}

__global__ void computeMacroKernel(
    const float* __restrict__ f,
    float* __restrict__ rho,
    float* __restrict__ ux,
    float* __restrict__ uy,
    float* __restrict__ uz,
    int n_cells)
{
    static constexpr int ex[19] = {0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0};
    static constexpr int ey[19] = {0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1};
    static constexpr int ez[19] = {0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1};

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;

    float m_rho = 0.0f, m_ux = 0.0f, m_uy = 0.0f, m_uz = 0.0f;
    for (int q = 0; q < 19; q++) {
        float fq = f[id + q * n_cells];
        m_rho += fq;
        m_ux += ex[q] * fq;
        m_uy += ey[q] * fq;
        m_uz += ez[q] * fq;
    }
    float inv_rho = 1.0f / fmaxf(m_rho, 1e-12f);
    rho[id] = m_rho;
    ux[id]  = m_ux * inv_rho;
    uy[id]  = m_uy * inv_rho;
    uz[id]  = m_uz * inv_rho;
}

// Moment-based Dirichlet BC for D3Q7:
// Only modifies the MISSING distribution direction, preserving incoming flux.
// x_min (face=0): q=1 (+x) is missing → g[1] = T_wall - Σ_{q≠1} g[q]
// x_max (face=1): q=2 (-x) is missing → g[2] = T_wall - Σ_{q≠2} g[q]
__global__ void applyDirichletBC_D3Q7(
    float* __restrict__ g,    // D3Q7 distributions [NC×7]
    float T_wall,
    int face,                 // 0=x_min, 1=x_max
    int nx, int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_face = ny * nz;
    if (tid >= n_face) return;

    int iy = tid % ny;
    int iz = tid / ny;

    int ix = (face == 0) ? 0 : nx - 1;
    int id = ix + iy * nx + iz * nx * ny;
    int nc = nx * ny * nz;

    // Missing direction: q=1 (+x) at x_min, q=2 (-x) at x_max
    int q_miss = (face == 0) ? 1 : 2;

    // Sum all distributions EXCEPT the missing one
    float sum_known = 0.0f;
    for (int q = 0; q < 7; q++) {
        if (q != q_miss)
            sum_known += g[id + q * nc];
    }

    // Set missing distribution so that T = Σ g_q = T_wall
    g[id + q_miss * nc] = T_wall - sum_known;
}

// Fused D3Q7 thermal step: collision → streaming → Dirichlet BCs → compute T
// Eliminates all cudaDeviceSynchronize() overhead from library calls
__global__ void fusedThermalStep_D3Q7(
    float* __restrict__ g_src,       // input distributions  [NC × 7]
    float* __restrict__ g_dst,       // output distributions [NC × 7]
    float* __restrict__ temperature, // output temperature field [NC]
    const float* __restrict__ ux,    // velocity x
    const float* __restrict__ uy,    // velocity y
    const float* __restrict__ uz,    // velocity z
    float omega_T,
    float T_hot, float T_cold,
    int nx, int ny, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    const int nc = nx * ny * nz;
    const int iz = 0;  // 2D (NZ=1)
    const int id = ix + iy * nx;

    // D3Q7 lattice: rest, +x, -x, +y, -y, +z, -z
    constexpr int cx[7] = {0, 1, -1, 0, 0, 0, 0};
    constexpr int cy[7] = {0, 0, 0, 1, -1, 0, 0};
    constexpr float w[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    constexpr float CS2 = 0.25f;

    // 1. COLLISION: read distributions, compute equilibrium, relax
    float g[7];
    for (int q = 0; q < 7; q++)
        g[q] = g_src[id + q * nc];

    float T = 0.0f;
    for (int q = 0; q < 7; q++) T += g[q];

    float vx = ux[id], vy = uy[id];

    for (int q = 0; q < 7; q++) {
        float cu = (cx[q] * vx + cy[q] * vy) / CS2;
        // TVD limiter for high-Pe
        cu = fminf(fmaxf(cu, -0.9f), 0.9f);
        float g_eq = w[q] * T * (1.0f + cu);
        g[q] += omega_T * (g_eq - g[q]);
    }

    // 2. STREAMING: push to neighbors with boundary handling
    for (int q = 0; q < 7; q++) {
        int nx_dst = ix + cx[q];
        int ny_dst = iy + cy[q];

        // Bounce-back at y boundaries (streaming kernel behavior)
        // z-periodic for NZ=1 (q=5,6 map to same cell)
        if (nx_dst < 0 || nx_dst >= nx || ny_dst < 0 || ny_dst >= ny) {
            // Bounce-back: write to current cell, opposite direction
            constexpr int opp[7] = {0, 2, 1, 4, 3, 6, 5};
            g_dst[id + opp[q] * nc] = g[q];
        } else {
            int dst_id = nx_dst + ny_dst * nx;
            g_dst[dst_id + q * nc] = g[q];
        }
    }
}

// Apply Dirichlet BCs and compute temperature (post-streaming)
// Adiabatic (zero-flux) at y=0 and y=NY-1 is handled by bounce-back in the
// streaming kernel — no explicit override needed.  Previous copy-from-interior
// was first-order and overrode Dirichlet at corners.
__global__ void applyBCsAndComputeT_D3Q7(
    float* __restrict__ g,           // distributions [NC × 7]
    float* __restrict__ temperature, // output temperature [NC]
    float T_hot, float T_cold,
    int nx, int ny, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    const int nc = nx * ny * nz;
    const int id = ix + iy * nx;

    // Dirichlet BC: moment-based (only modify missing direction)
    // Applied at ALL y values (including corners) — Dirichlet takes priority
    if (ix == 0) {
        // x_min: q=1 (+x) is the missing direction
        float sum = 0.0f;
        for (int q = 0; q < 7; q++)
            if (q != 1) sum += g[id + q * nc];
        g[id + 1 * nc] = T_hot - sum;
    }
    if (ix == nx - 1) {
        // x_max: q=2 (-x) is the missing direction
        float sum = 0.0f;
        for (int q = 0; q < 7; q++)
            if (q != 2) sum += g[id + q * nc];
        g[id + 2 * nc] = T_cold - sum;
    }

    // Adiabatic at y boundaries: streaming bounce-back already handles this
    // (second-order accurate, preserves non-equilibrium flux)

    // Compute temperature
    float T = 0.0f;
    for (int q = 0; q < 7; q++)
        T += g[id + q * nc];
    temperature[id] = T;
}

// ======================== Host utilities ========================

static float computeNusselt(const float* T, int nx, int ny) {
    float Nu = 0.0f;
    for (int iy = 0; iy < ny; iy++)
        Nu += -(T[1 + iy * nx] - T_HOT) * (nx - 1);
    return Nu / ny;
}

static float computeStreamFunctionMax(const float* ux, int nx, int ny) {
    // ψ(x,0) = 0, ∂ψ/∂y = ux → ψ(x,j) = Σ ux·dy
    float dy = 1.0f / (ny - 1);
    float psi_max = 0.0f;
    for (int ix = 0; ix < nx; ix++) {
        float psi = 0.0f;
        for (int iy = 1; iy < ny; iy++) {
            float ux_prev = ux[ix + (iy - 1) * nx];
            float ux_curr = ux[ix + iy * nx];
            psi += 0.5f * (ux_prev + ux_curr) * dy;
            if (fabsf(psi) > fabsf(psi_max)) psi_max = psi;
        }
    }
    // Non-dimensionalise: ψ* = ψ / α (α in lattice units is per-cell)
    // ψ is in lattice-velocity × non-dim-length units
    // To get Zebib's ψ*: ψ_zebib = ψ_lattice * NX / ALPHA_LB
    return psi_max * (float)nx / ALPHA_LB;
}

// ======================== Main ========================
int main() {
    CUDA_CHECK(cudaSetDevice(0));
    auto t0 = std::chrono::high_resolution_clock::now();

    float ma_lbm = U_REF / sqrtf(1.0f / 3.0f);
    printf("=== Thermocapillary Cavity (Zebib 1985) ===\n");
    printf("Grid: %dx%d  Ma=%g  Pr=%g\n", NX, NY, MA, PR);
    printf("nu=%.4f  alpha=%.4f  tau_v=%.3f  tau_T=%.3f\n",
           NU_LB, ALPHA_LB, TAU_V, TAU_T);
    printf("U_ref=%.4f  gamma_T=%.4e  CE_factor=%.4f\n", U_REF, GAMMA_T, CE_FACTOR);
    printf("Ma_LBM=%.4f (should be <0.1)  t_diff=%d steps\n", ma_lbm, T_DIFF);
    printf("MAX_STEPS=%d  CHECK=%d  CONV_TOL=%.1e\n\n", MAX_STEPS, CHECK_INTERVAL, CONV_TOL);
    if (ma_lbm > 0.15f)
        printf("*** WARNING: Ma_LBM=%.3f > 0.15 — compressibility errors expected ***\n\n", ma_lbm);
    fflush(stdout);

    // --- Lattice constants ---
    if (!D3Q7::isInitialized())  D3Q7::initializeDevice();
    if (!D3Q19::isInitialized()) D3Q19::initializeDevice();

    // --- Solvers ---
    FluidLBM fluid(NX, NY, NZ, NU_LB, RHO,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   1.0f, 1.0f);

    ThermalLBM thermal(NX, NY, NZ, ALPHA_LB, RHO, 1.0f, 1.0f, 1.0f);
    thermal.setZPeriodic(true);

    // --- Initialize: linear temperature profile ---
    fluid.initialize(RHO, 0.0f, 0.0f, 0.0f);

    std::vector<float> T_init(NC);
    for (int iz = 0; iz < NZ; iz++)
        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++)
                T_init[ix + iy * NX + iz * NX * NY] =
                    T_HOT + (T_COLD - T_HOT) * ix / (float)(NX - 1);
    thermal.initialize(T_init.data());

    // Kernel configs
    dim3 blk_top(128, 1);
    dim3 grd_top((NX + blk_top.x - 1) / blk_top.x, NZ);

    int n_wall_nodes = 2 * NY * NZ + NX * NZ;
    int blk_wall = 256;
    int grd_wall = (n_wall_nodes + blk_wall - 1) / blk_wall;

    int blk_macro = 256;
    int grd_macro = (NC + blk_macro - 1) / blk_macro;

    // Moment-based Dirichlet BC launch config (NY*NZ threads per face)
    int n_face = NY * NZ;
    int blk_dirichlet = 256;
    int grd_dirichlet = (n_face + blk_dirichlet - 1) / blk_dirichlet;

    // Custom rho/velocity arrays (avoid FluidLBM's internal boundary zeroing)
    float *d_rho_c, *d_ux_c, *d_uy_c, *d_uz_c;
    CUDA_CHECK(cudaMalloc(&d_rho_c, NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ux_c,  NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uy_c,  NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uz_c,  NC * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ux_c, 0, NC * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_uy_c, 0, NC * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_uz_c, 0, NC * sizeof(float)));

    // D3Q7 thermal: own two distribution buffers for fused kernel (avoid library sync)
    float *d_g_A, *d_g_B;
    CUDA_CHECK(cudaMalloc(&d_g_A, NC * 7 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_B, NC * 7 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_g_A, thermal.getDistributionSrc(),
                          NC * 7 * sizeof(float), cudaMemcpyDeviceToDevice));
    float* d_temperature = thermal.getTemperature();
    float* d_g_src = d_g_A;
    float* d_g_dst = d_g_B;

    // Fused kernel launch config (2D: NX × NY threads)
    dim3 blk_fused(16, 16);
    dim3 grd_fused((NX + 15) / 16, (NY + 15) / 16);
    float omega_T = 1.0f / TAU_T;

    // Convergence buffers
    std::vector<float> h_ux(NC, 0.0f), h_uy(NC, 0.0f), h_T(NC);

    FILE* clog = fopen("marangoni_cavity_convergence.csv", "w");
    if (!clog) { fprintf(stderr, "Cannot open convergence log!\n"); return 1; }
    fprintf(clog, "step,Nu,psi_max,u_max_star,dNu_rel\n");

    bool converged = false;
    int  conv_step = 0;
    float Nu_prev  = 1.0f;  // start at conduction Nu
    int   stable_count = 0; // consecutive intervals with |dNu/Nu| < tol

    // Pure conduction diagnostic verified: Nu=0.999977 with moment-based Dirichlet BC
    // (Equilibrium Dirichlet gave Nu=0.557 — the bug that caused all prior low-Nu results)

    // ======================== Time loop ========================
    for (int step = 1; step <= MAX_STEPS; step++) {

        // 1. Thermal: fused collision+streaming (zero sync overhead)
        cudaMemset(d_g_dst, 0, NC * 7 * sizeof(float));
        fusedThermalStep_D3Q7<<<grd_fused, blk_fused>>>(
            d_g_src, d_g_dst, d_temperature,
            d_ux_c, d_uy_c, d_uz_c, omega_T, T_HOT, T_COLD, NX, NY, NZ);
        { float* tmp = d_g_src; d_g_src = d_g_dst; d_g_dst = tmp; }

        // 1b. Apply BCs and compute temperature
        applyBCsAndComputeT_D3Q7<<<grd_fused, blk_fused>>>(
            d_g_src, d_temperature, T_HOT, T_COLD, NX, NY, NZ);

        // 2. Fluid: TRT collision → streaming → walls → Inamuro → macro
        fluid.collisionTRT(0.0f, 0.0f, 0.0f, LAMBDA);
        fluid.streaming();
        applyNoSlipWalls<<<grd_wall, blk_wall>>>(
            fluid.getDistributionSrc(), NX, NY, NZ);
        applyTopInamuroBC<<<grd_top, blk_top>>>(
            fluid.getDistributionSrc(), d_temperature,
            NX, NY, NZ, GAMMA_T, CE_FACTOR);
        computeMacroKernel<<<grd_macro, blk_macro>>>(
            fluid.getDistributionSrc(), d_rho_c, d_ux_c, d_uy_c, d_uz_c, NC);

        // 3. NaN check (early steps only)
        if (step <= 5 || (step <= 100 && step % 20 == 0)) {
            cudaDeviceSynchronize();
            float dbg;
            cudaMemcpy(&dbg, d_ux_c + NX/2 + (NY-1)*NX,
                       sizeof(float), cudaMemcpyDeviceToHost);
            if (std::isnan(dbg)) {
                printf("  [%d] NaN detected! Aborting.\n", step);
                break;
            }
        }

        // 7. Convergence: track Nusselt number stability
        if (step % CHECK_INTERVAL == 0) {
            CUDA_CHECK(cudaMemcpy(h_T.data(), d_temperature,
                                  NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ux.data(), d_ux_c,
                                  NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy.data(), d_uy_c,
                                  NC * sizeof(float), cudaMemcpyDeviceToHost));

            float Nu = computeNusselt(h_T.data(), NX, NY);
            float psi_max = computeStreamFunctionMax(h_ux.data(), NX, NY);

            // Max velocity on surface (non-dimensional: u* = u·H/α)
            float u_max_lu = 0.0f;
            for (int i = 0; i < NC; i++)
                u_max_lu = fmaxf(u_max_lu, sqrtf(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i]));
            float u_max_star = u_max_lu * (float)NX / ALPHA_LB;

            float dNu_rel = fabsf(Nu - Nu_prev) / fmaxf(fabsf(Nu), 1e-6f);

            fprintf(clog, "%d,%.6f,%.4f,%.2f,%.6e\n",
                    step, Nu, psi_max, u_max_star, dNu_rel);
            fflush(clog);

            float t_frac = (float)step / (float)T_DIFF;
            printf("  step %7d (%.2f t_diff): Nu=%.4f  ψ*=%.3f  u*=%.1f  dNu=%.2e\n",
                   step, t_frac, Nu, psi_max, u_max_star, dNu_rel);

            // Diagnostic: surface temperature gradient and stress at x=NX/2
            if (step == CHECK_INTERVAL || step % (CHECK_INTERVAL * 10) == 0) {
                int xm = NX / 2;
                int surf = NY - 1;
                float T_surf_hot  = h_T[1 + surf * NX];
                float T_surf_mid  = h_T[xm + surf * NX];
                float T_surf_cold = h_T[(NX-2) + surf * NX];
                float dTdx_surf   = (h_T[(xm+1) + surf*NX] - h_T[(xm-1) + surf*NX]) * 0.5f;
                float dTdx_cond   = -1.0f / (float)(NX - 1);
                float tau_M       = -GAMMA_T * dTdx_surf;
                float du_dy       = (h_ux[xm + surf*NX] - h_ux[xm + (surf-1)*NX]);
                float tau_visc    = NU_LB * du_dy;
                printf("    Surface T: hot=%.4f mid=%.4f cold=%.4f\n",
                       T_surf_hot, T_surf_mid, T_surf_cold);
                printf("    dTdx_surf=%.5e (cond=%.5e, ratio=%.2f)\n",
                       dTdx_surf, dTdx_cond, dTdx_surf/dTdx_cond);
                printf("    tau_M=%.5e  tau_visc=%.5e  ratio=%.3f\n",
                       tau_M, tau_visc, tau_visc / fmaxf(fabsf(tau_M), 1e-12f));
                printf("    u_surf(x=0.5)=%.6f  U_REF=%.6f  u/U=%.3f\n",
                       h_ux[xm + surf*NX], U_REF, h_ux[xm + surf*NX] / U_REF);
            }
            fflush(stdout);

            // Check Ma_LBM safety
            float ma_lbm_curr = u_max_lu / sqrtf(1.0f/3.0f);
            if (ma_lbm_curr > 0.3f) {
                printf("  *** ABORT: Ma_LBM=%.3f > 0.3 — simulation unstable ***\n",
                       ma_lbm_curr);
                break;
            }

            // Converge when Nu stable for 3 consecutive checks, after 0.5 t_diff
            if (step > T_DIFF / 2) {
                if (dNu_rel < CONV_TOL)
                    stable_count++;
                else
                    stable_count = 0;

                if (stable_count >= 3) {
                    printf("  → CONVERGED (Nu stable for %d checks)\n", stable_count);
                    converged = true;
                    conv_step = step;
                    break;
                }
            }
            Nu_prev = Nu;
        }
    }
    fclose(clog);

    float elapsed = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("\nWall time: %.1f s  Converged: %s",
           elapsed, converged ? "YES" : "NO");
    if (converged) printf(" (step %d)", conv_step);
    printf("\n");

    // ======================== Final output ========================
    if (h_T.empty() || !converged) {
        // Re-copy if we didn't break from convergence check
        CUDA_CHECK(cudaMemcpy(h_T.data(), d_temperature, NC * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ux.data(), d_ux_c, NC * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_uy.data(), d_uy_c, NC * sizeof(float), cudaMemcpyDeviceToHost));
    }

    std::vector<float> h_rho(NC);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho_c, NC * sizeof(float), cudaMemcpyDeviceToHost));

    // Field CSV
    {
        std::ofstream f("marangoni_cavity_field.csv");
        f << "ix,iy,x,y,T,ux,uy,rho\n";
        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++) {
                int id = ix + iy * NX;
                f << ix << ',' << iy << ','
                  << ix / (float)(NX - 1) << ',' << iy / (float)(NY - 1) << ','
                  << h_T[id] << ',' << h_ux[id] << ',' << h_uy[id] << ','
                  << h_rho[id] << '\n';
            }
    }

    // Midline profiles CSV
    {
        std::ofstream f("marangoni_cavity_profiles.csv");
        f << "type,pos,vel\n";
        int xm = NX / 2, ym = NY / 2;
        for (int iy = 0; iy < NY; iy++)
            f << "vmid," << iy / (float)(NY - 1) << ',' << h_ux[xm + iy * NX] << '\n';
        for (int ix = 0; ix < NX; ix++)
            f << "hmid," << ix / (float)(NX - 1) << ',' << h_uy[ix + ym * NX] << '\n';
    }

    // ======================== Validation metrics ========================
    float Nu = computeNusselt(h_T.data(), NX, NY);
    float psi_max = computeStreamFunctionMax(h_ux.data(), NX, NY);

    int xm = NX / 2, ym = NY / 2;
    float ux_max = 0, ux_y = 0, uy_max = 0, uy_x = 0;
    for (int iy = 0; iy < NY; iy++) {
        float v = h_ux[xm + iy * NX];
        if (fabsf(v) > fabsf(ux_max)) { ux_max = v; ux_y = iy / (float)(NY - 1); }
    }
    for (int ix = 0; ix < NX; ix++) {
        float v = h_uy[ix + ym * NX];
        if (fabsf(v) > fabsf(uy_max)) { uy_max = v; uy_x = ix / (float)(NX - 1); }
    }

    float ux_max_star = ux_max * (float)NX / ALPHA_LB;
    float uy_max_star = uy_max * (float)NX / ALPHA_LB;

    printf("\n=== Validation Metrics (Ma=%g, Pr=%g, %dx%d) ===\n", MA, PR, NX, NY);
    printf("Nusselt (hot wall):  %.3f\n", Nu);
    printf("ψ*_max:              %.3f\n", psi_max);
    printf("u*_max midline:      %.2f at y=%.3f\n", ux_max_star, ux_y);
    printf("v*_max midline:      %.2f at x=%.3f\n", uy_max_star, uy_x);

    // Reference: Zebib, Homsy & Meiburg (1985); Carpenter & Homsy (1990)
    if (MA_NUMBER == 100)
        printf("\nReference (Ma=100): Nu≈2.15, ψ*≈-3.22\n");
    else if (MA_NUMBER == 1000)
        printf("\nReference (Ma=1000): Nu≈4.67, ψ*≈-9.05\n");

    printf("Output: marangoni_cavity_field.csv, _profiles.csv, _convergence.csv\n");

    cudaFree(d_rho_c);
    cudaFree(d_ux_c);
    cudaFree(d_uy_c);
    cudaFree(d_uz_c);
    cudaFree(d_g_A);
    cudaFree(d_g_B);
    return 0;
}
