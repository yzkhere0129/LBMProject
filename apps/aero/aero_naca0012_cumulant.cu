/**
 * @file aero_naca0012_cumulant.cu
 * @brief NACA0012 airfoil drag/lift benchmark using D3Q27 Cumulant LBM
 *
 * Adapted from aero_st_2d2_cumulant.cu. Differences:
 *   - Geometry: NACA0012 airfoil (4-digit symmetric, t/c=0.12) instead of cylinder
 *   - Inlet: uniform freestream (NOT parabolic) via Ladd moving-wall
 *   - Domain: ~30c × 20c (low blockage); top/bot walls halfway BB far from airfoil
 *   - Cd/Cl normalised by chord c (not diameter D)
 *
 * Reference benchmark: Kurtulus D.F. 2015, Int. J. Micro Air Veh. 7:301
 *   "On the Wake Pattern of Symmetric Airfoils for Different Incidence Angles
 *    at Re=1000". NACA0012 at Re=1000 (laminar regime), α sweep:
 *   - α=8°:  Cd_mean ≈ 0.34, Cl_mean ≈ 0.49, St ≈ 0.81 (LEV shedding)
 *   - α=12°: Cd_mean ≈ 0.49, Cl_mean ≈ 0.62, St ≈ 0.65
 *   - α=16°: Cd_mean ≈ 0.66, Cl_mean ≈ 0.70, St ≈ 0.50
 *
 * Default: α=8°, Re=1000.
 */

#include "core/lattice_d3q27.h"
#include "physics/cumulant/cumulant_d3q27.h"
#include "physics/cumulant/streaming_d3q27_qbb.h"
#include "physics/aero/obstacle_geometry.h"
#include "io/vtk_writer.h"
#include "utils/cuda_check.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>

using namespace lbm;
namespace fs = std::filesystem;

struct Args {
    int   resolution = 40;       // cells per chord
    float re         = 1000.0f;  // Reynolds = U_inf · c / nu
    float alpha_deg  = 8.0f;     // angle of attack [deg]
    int   steps      = -1;
    int   probe_every = 100;
    int   vtk_every   = 0;
    std::string output_dir = "output_aero_naca0012";
    int   nz_thin    = 4;
    float u_max_lu   = 0.05f;
    float omega_3    = 1.0f;
    float omega_4    = 1.0f;
    float omega_5    = 1.0f;
    float omega_6    = 1.0f;
    float lx_over_c  = 30.0f;    // domain length in chord units
    float ly_over_c  = 20.0f;    // domain height in chord units
    float xLE_over_c = 10.0f;    // LE position from inlet, chord units
    int   use_bgk    = 0;        // 1 = use BGK D3Q27 instead of Cumulant
    int   use_trt    = 0;        // 1 = use TRT D3Q27 (magic Λ=3/16 by default)
    float trt_lambda = 0.1875f;  // TRT magic parameter Λ = (τ+-1/2)(τ--1/2)
    int   sparse_qfrac = 0;      // 1 = use sparse-CSR qfrac (saves ~99% mem)
    // Shape: 0=NACA0012 (default), 1=cylinder, 2=flat plate
    int   shape      = 0;
    float cyl_off_y  = 0.0f;     // cylinder y offset from y_LE (for asymmetry)
    float plate_thick_cells = 1.0f;  // flat plate thickness in cell units (= 1 cell)
    // BC at obstacle: "stair" = halfway BB on cell-centre stamp; "qbb-snode"
    // = D3Q27 single-node Bouzidi (curved BC, q-fraction per link). NACA only.
    std::string bc_mode = "stair";
    // Wall ω used inside QBB streaming kernel + MEM-QBB force probe. Defaults
    // to ω_nu (BGK-equivalent). Setting to a different value emulates TRT-style
    // Bouzidi (e.g. Λ_eo = 3/16 magic gives ω_wall = 1/(0.5 + 3/(16·(τ-0.5)))).
    // -1 sentinel = "use ω_nu" (i.e. unchanged behavior).
    float wall_omega_override = -1.0f;
};

// Simple BGK D3Q27 collision (control test for Cumulant)
// TRT D3Q27 collision: split f into symmetric (even) and antisymmetric (odd)
// parts and apply distinct relaxation rates. For magic parameter Λ = 3/16,
// the no-slip wall condition becomes exact (independent of ν). For our setup
// at ω+ = 1.976 (τ+ = 0.506), magic ω- = 1 / (0.5 + 3/(16·(τ+-0.5))) ≈ 0.031.
__global__ void fluidTRTD3Q27Kernel(
    const float* f_src, float* f_dst,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int nx, int ny, int nz, float omega_plus, float omega_minus)
{
    using lbm::core::D3Q27;
    using lbm::core::opposite27;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    float f[27];
    #pragma unroll
    for (int q = 0; q < 27; ++q) f[q] = f_src[id + q * n_cells];

    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[q];
        rho += fq;
        mx += ::lbm::core::ex27[q] * fq;
        my += ::lbm::core::ey27[q] * fq;
        mz += ::lbm::core::ez27[q] * fq;
    }
    const float inv_rho = (rho > 1e-12f) ? (1.0f / rho) : 0.0f;
    const float ux = mx * inv_rho;
    const float uy = my * inv_rho;
    const float uz = mz * inv_rho;

    float feq[27];
    #pragma unroll
    for (int q = 0; q < 27; ++q)
        feq[q] = D3Q27::computeEquilibrium(q, rho, ux, uy, uz);

    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const int qo = opposite27[q];
        const float f_plus  = 0.5f * (f[q] + f[qo]);
        const float f_minus = 0.5f * (f[q] - f[qo]);
        const float feq_plus  = 0.5f * (feq[q] + feq[qo]);
        const float feq_minus = 0.5f * (feq[q] - feq[qo]);
        const float coll = omega_plus  * (f_plus  - feq_plus)
                         + omega_minus * (f_minus - feq_minus);
        f_dst[id + q * n_cells] = f[q] - coll;
    }
    if (rho_out) rho_out[id] = rho;
    if (ux_out)  ux_out[id]  = ux;
    if (uy_out)  uy_out[id]  = uy;
    if (uz_out)  uz_out[id]  = uz;
}

__global__ void fluidBGKD3Q27Kernel(
    const float* f_src, float* f_dst,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int nx, int ny, int nz, float omega)
{
    using lbm::core::D3Q27;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    float f[27];
    #pragma unroll
    for (int q = 0; q < 27; ++q) f[q] = f_src[id + q * n_cells];

    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[q];
        rho += fq;
        mx += ::lbm::core::ex27[q] * fq;
        my += ::lbm::core::ey27[q] * fq;
        mz += ::lbm::core::ez27[q] * fq;
    }
    const float inv_rho = (rho > 1e-12f) ? (1.0f / rho) : 0.0f;
    const float ux = mx * inv_rho;
    const float uy = my * inv_rho;
    const float uz = mz * inv_rho;

    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float feq = D3Q27::computeEquilibrium(q, rho, ux, uy, uz);
        f_dst[id + q * n_cells] = f[q] - omega * (f[q] - feq);
    }
    if (rho_out) rho_out[id] = rho;
    if (ux_out)  ux_out[id]  = ux;
    if (uy_out)  uy_out[id]  = uy;
    if (uz_out)  uz_out[id]  = uz;
}

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value after " << s << std::endl; std::exit(1); }
            return argv[++i];
        };
        if      (s == "--resolution")  a.resolution = std::stoi(next());
        else if (s == "--re")          a.re = std::stof(next());
        else if (s == "--alpha")       a.alpha_deg = std::stof(next());
        else if (s == "--steps")       a.steps = std::stoi(next());
        else if (s == "--probe-every") a.probe_every = std::stoi(next());
        else if (s == "--vtk-every")   a.vtk_every = std::stoi(next());
        else if (s == "--output-dir")  a.output_dir = next();
        else if (s == "--nz-thin")     a.nz_thin = std::stoi(next());
        else if (s == "--u-max-lu")    a.u_max_lu = std::stof(next());
        else if (s == "--lx-over-c")   a.lx_over_c = std::stof(next());
        else if (s == "--ly-over-c")   a.ly_over_c = std::stof(next());
        else if (s == "--xle-over-c")  a.xLE_over_c = std::stof(next());
        else if (s == "--bgk")         a.use_bgk = 1;
        else if (s == "--trt")         a.use_trt = 1;
        else if (s == "--trt-lambda")  a.trt_lambda = std::stof(next());
        else if (s == "--sparse-qfrac") a.sparse_qfrac = 1;
        else if (s == "--omega-3")     a.omega_3 = std::stof(next());
        else if (s == "--omega-4")     a.omega_4 = std::stof(next());
        else if (s == "--omega-5")     a.omega_5 = std::stof(next());
        else if (s == "--omega-6")     a.omega_6 = std::stof(next());
        else if (s == "--wall-omega")  a.wall_omega_override = std::stof(next());
        // Shape selector. --cylinder kept as backward-compat alias.
        else if (s == "--cylinder")    a.shape = 1;
        else if (s == "--shape") {
            std::string v = next();
            if      (v == "naca")      a.shape = 0;
            else if (v == "cylinder")  a.shape = 1;
            else if (v == "flatplate") a.shape = 2;
            else { std::cerr << "Unknown shape: " << v << " (naca|cylinder|flatplate)\n"; std::exit(1); }
        }
        else if (s == "--cyl-off-y")   a.cyl_off_y = std::stof(next());
        else if (s == "--plate-thick-cells") a.plate_thick_cells = std::stof(next());
        else if (s == "--bc") {
            std::string v = next();
            if (v != "stair" && v != "qbb-snode") {
                std::cerr << "Unknown --bc: " << v << " (stair|qbb-snode)\n";
                std::exit(1);
            }
            a.bc_mode = v;
        }
        else if (s == "-h" || s == "--help") {
            std::cout <<
              "Usage: aero_naca0012_cumulant [opts]\n"
              "  --resolution N    cells per chord (default 40)\n"
              "  --re R            Reynolds = U·c/ν (default 1000)\n"
              "  --alpha A         angle of attack [deg] (default 8)\n"
              "  --u-max-lu X      target u_∞ in LU (default 0.05; Mach control)\n"
              "  --steps N         total LBM steps (auto if -1)\n"
              "  --probe-every N   Cd/Cl sample interval (default 100)\n"
              "  --vtk-every N     VTK snapshot interval (0=off)\n"
              "  --lx-over-c X     domain length in chord units (default 30)\n"
              "  --ly-over-c Y     domain height in chord units (default 20)\n"
              "  --xle-over-c X    LE position from inlet (default 10)\n"
              "  --bc M            obstacle BC: stair (default) | qbb-snode (NACA only)\n";
            std::exit(0);
        } else { std::cerr << "Unknown: " << s << std::endl; std::exit(1); }
    }
    return a;
}

using core::ex27;
using core::ey27;
using core::ez27;
using core::w27;
using core::opposite27;
using core::D3Q27;

// Initialize all PDFs with f_eq(ρ=1, u_∞=(u_x_inf, 0, 0))
__global__ void initializeFreestream(float* f, int n_cells, float u_x_inf) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f[id + q * n_cells] =
            D3Q27::computeEquilibrium(q, 1.0f, u_x_inf, 0.0f, 0.0f);
    }
}

// Uniform freestream inlet (Ladd moving-wall, post-stream).
// Same Cd-rescued formula as ST 2D-2 — only difference is u_inlet is constant
// across y instead of parabolic.
__global__ void applyInletFreestream(
    float* f, int nx, int ny, int nz, float u_x_inf)
{
    const int idy = blockIdx.x * blockDim.x + threadIdx.x;
    const int idz = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= ny || idz >= nz) return;

    const int idx = 0;
    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        if (ex27[q] > 0) {
            const int q_opp = opposite27[q];
            const float cu = (float)ex27[q] * u_x_inf;
            f[id + q * n_cells] = f[id + q_opp * n_cells]
                                + 6.0f * w27[q] * 1.0f * cu;
        }
    }
}

// Outlet: zero-gradient extrapolation from i=nx-2.
__global__ void applyOutletExtrap(
    float* f, int nx, int ny, int nz)
{
    const int idy = blockIdx.x * blockDim.x + threadIdx.x;
    const int idz = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= ny || idz >= nz) return;

    const int idx = nx - 1;
    const int id = idx + idy * nx + idz * nx * ny;
    const int src_id = (nx - 2) + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f[id + q * n_cells] = f[src_id + q * n_cells];
    }
}

// y-mirror table for D3Q27 free-slip walls: only flips c_y component
// (preserves c_x and c_z). Computed by hand from D3Q27 table.
__constant__ int y_mirror27[27] = {
    0, 1, 2, 4, 3, 5, 6,           // rest + faces
    9, 10, 7, 8,                    // xy edges (q=7..10)
    11, 12, 13, 14,                 // xz edges (q=11..14)
    16, 15, 18, 17,                 // yz edges (q=15..18)
    23, 24, 26, 25, 19, 20, 22, 21  // corners (q=19..26)
};

// Streaming with FREE-SLIP on Y faces (mirror, preserves tangential u_x),
// z-periodic, halfway BB on solid (NACA airfoil) cells.
// Skip i=0/nx-1 (inlet/outlet handled separately by post-stream kernels).
__global__ void streamD3Q27_naca(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    const unsigned char* __restrict__ solid_mask,
    int nx, int ny, int nz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    if (solid_mask[id] != 0) return;

    for (int q = 0; q < 27; ++q) {
        int src_x = idx - ex27[q];
        int src_y = idy - ey27[q];
        int src_z = idz - ez27[q];
        if (src_z < 0)  src_z += nz;
        if (src_z >= nz) src_z -= nz;
        bool out_of_domain = false;
        if (src_x < 0 || src_x >= nx) out_of_domain = true;
        if (src_y < 0 || src_y >= ny) out_of_domain = true;
        if (out_of_domain) {
            // X face: temporary halfway BB; will be overwritten by
            // applyInlet/Outlet for i=0/nx-1 cells anyway.
            // Y face: FREE-SLIP (mirror reflection) — preserves tangential
            // u_x velocity, only flips u_y. Critical for free-stream airfoil
            // simulation; halfway BB would create spurious wall boundary
            // layers that propagate into the airfoil region.
            const int reflected_q = (src_y < 0 || src_y >= ny)
                                  ? y_mirror27[q]      // free-slip Y wall
                                  : opposite27[q];     // X face: halfway BB
            f_dst[id + q * n_cells] = f_src[id + reflected_q * n_cells];
            continue;
        }
        const int src_id = src_x + src_y * nx + src_z * nx * ny;
        if (solid_mask[src_id] != 0) {
            f_dst[id + q * n_cells] = f_src[id + opposite27[q] * n_cells];
        } else {
            f_dst[id + q * n_cells] = f_src[src_id + q * n_cells];
        }
    }
}

// MEM force on NACA airfoil — pure halfway BB version.
// F_link = c_q · 2·f_in (used with stair-step streaming).
__global__ void memForceNaca(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    int nx, int ny, int nz,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    if (solid_mask[id] != 0) return;

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;
    for (int q = 1; q < 27; ++q) {
        int dst_x = idx + ex27[q];
        int dst_y = idy + ey27[q];
        int dst_z = idz + ez27[q];
        if (dst_z < 0)  dst_z += nz;
        if (dst_z >= nz) dst_z -= nz;
        if (dst_x < 0 || dst_x >= nx || dst_y < 0 || dst_y >= ny) continue;
        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == 0) continue;
        const float f_in = f[id + q * n_cells];
        const double scale = 2.0 * (double)f_in;
        fx_local += (double)ex27[q] * scale;
        fy_local += (double)ey27[q] * scale;
        fz_local += (double)ez27[q] * scale;
    }
    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}

// MEM force consistent with single-node QBB streaming. For each fluid cell X
// with a solid neighbour in direction q (q = q_push, link from X to solid):
//   F_link = c_q · (f_in + f_out_new)
// where f_in = f^pre(X, q) and f_out_new is the lbmpy single-node QBB result
// that the streaming kernel would write into f_dst[X, opp(q)]. Computing it
// inline (using only f_src + local moments + qfrac) keeps the force probe a
// single read of f_src — no dependence on f_dst order.
//
// Reference: same formula as `streamD3Q27_naca_qbb`. Must use the same omega.
__global__ void memForceNaca_QBB(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const float* __restrict__ qfrac,
    float omega,
    int nx, int ny, int nz,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    using lbm::core::D3Q27;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    if (solid_mask[id] != 0) return;

    // Local moments for f_eq.
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[id + q * n_cells];
        rho += fq;
        mx += ex27[q] * fq;
        my += ey27[q] * fq;
        mz += ez27[q] * fq;
    }
    const float rho_safe = fmaxf(rho, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    const float inv_one_minus_omega = 1.0f / (1.0f - omega);
    constexpr float QMIN_F = 0.05f;
    constexpr float QMAX_F = 0.95f;

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;
    for (int q = 1; q < 27; ++q) {
        int dst_x = idx + ex27[q];
        int dst_y = idy + ey27[q];
        int dst_z = idz + ez27[q];
        if (dst_z < 0)  dst_z += nz;
        if (dst_z >= nz) dst_z -= nz;
        if (dst_x < 0 || dst_x >= nx || dst_y < 0 || dst_y >= ny) continue;
        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == 0) continue;

        const int q_opp = opposite27[q];
        float qf = qfrac[id + q * n_cells];   // qf along push direction q
        if (qf > QMAX_F) qf = QMAX_F;
        if (qf < QMIN_F) qf = QMIN_F;

        const float f_in  = f[id + q     * n_cells];  // pop into solid
        const float f_out = f[id + q_opp * n_cells];  // pop away from solid

        const float feq_a   = D3Q27::computeEquilibrium(q,     rho, ux, uy, uz);
        const float feq_b   = D3Q27::computeEquilibrium(q_opp, rho, ux, uy, uz);
        const float feq_sym = feq_a + feq_b;

        const float t1 = (f_in - f_out)
                       + (f_in + f_out - omega * feq_sym) * inv_one_minus_omega;
        const float t2 = qf * (f_in + f_out) / (1.0f + qf);
        const float f_out_new = ((1.0f - qf) / (1.0f + qf)) * 0.5f * t1 + t2;

        const double sum = (double)f_in + (double)f_out_new;
        fx_local += (double)ex27[q] * sum;
        fy_local += (double)ey27[q] * sum;
        fz_local += (double)ez27[q] * sum;
    }
    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}

// Sparse-qfrac MEM force probe. Identical formula as memForceNaca_QBB but
// reads qfrac via CSR-like lookup. Used in --sparse-qfrac mode for D/dx≥120.
__device__ inline float lookup_sparse_qf_local(
    int id, unsigned char Q,
    const int* __restrict__ offset,
    const unsigned char* __restrict__ link_q,
    const float* __restrict__ link_val)
{
    int start = offset[id];
    int end   = offset[id + 1];
    for (int e = start; e < end; ++e)
        if (link_q[e] == Q) return link_val[e];
    return 0.5f;
}

__global__ void memForceNaca_QBB_sparse(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const int*           __restrict__ qf_offset,
    const unsigned char* __restrict__ qf_link_q,
    const float*         __restrict__ qf_link_val,
    float omega,
    int nx, int ny, int nz,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    using lbm::core::D3Q27;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    if (solid_mask[id] != 0) return;

    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[id + q * n_cells];
        rho += fq;
        mx += ex27[q] * fq;
        my += ey27[q] * fq;
        mz += ez27[q] * fq;
    }
    const float rho_safe = fmaxf(rho, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    const float inv_one_minus_omega = 1.0f / (1.0f - omega);
    constexpr float QMIN_F = 0.05f;
    constexpr float QMAX_F = 0.95f;

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;
    for (int q = 1; q < 27; ++q) {
        int dst_x = idx + ex27[q];
        int dst_y = idy + ey27[q];
        int dst_z = idz + ez27[q];
        if (dst_z < 0)  dst_z += nz;
        if (dst_z >= nz) dst_z -= nz;
        if (dst_x < 0 || dst_x >= nx || dst_y < 0 || dst_y >= ny) continue;
        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == 0) continue;

        const int q_opp = opposite27[q];
        float qf = lookup_sparse_qf_local(id, (unsigned char)q,
            qf_offset, qf_link_q, qf_link_val);
        if (qf > QMAX_F) qf = QMAX_F;
        if (qf < QMIN_F) qf = QMIN_F;

        const float f_in  = f[id + q     * n_cells];
        const float f_out = f[id + q_opp * n_cells];

        const float feq_a   = D3Q27::computeEquilibrium(q,     rho, ux, uy, uz);
        const float feq_b   = D3Q27::computeEquilibrium(q_opp, rho, ux, uy, uz);
        const float feq_sym = feq_a + feq_b;

        const float t1 = (f_in - f_out)
                       + (f_in + f_out - omega * feq_sym) * inv_one_minus_omega;
        const float t2 = qf * (f_in + f_out) / (1.0f + qf);
        const float f_out_new = ((1.0f - qf) / (1.0f + qf)) * 0.5f * t1 + t2;

        const double sum = (double)f_in + (double)f_out_new;
        fx_local += (double)ex27[q] * sum;
        fy_local += (double)ey27[q] * sum;
        fz_local += (double)ez27[q] * sum;
    }
    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}

__global__ void copyMacroD3Q27_naca(
    const float* __restrict__ f,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int n_cells)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[id + q * n_cells];
        rho += fq;
        mx += ex27[q] * fq;
        my += ey27[q] * fq;
        mz += ez27[q] * fq;
    }
    rho_out[id] = rho;
    const float inv_rho = (rho > 1e-12f) ? (1.0f / rho) : 0.0f;
    ux_out[id] = mx * inv_rho;
    uy_out[id] = my * inv_rho;
    uz_out[id] = mz * inv_rho;
}

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    // ---- Geometry: chord = 1, dimension is normalised so 1 chord = 1.0 m
    const float chord = 1.0f;
    const float Lx = args.lx_over_c * chord;
    const float Ly = args.ly_over_c * chord;
    const float xLE = args.xLE_over_c * chord;
    const float yLE = 0.5f * Ly;       // centred vertically
    const float alpha_rad = args.alpha_deg * (3.14159265358979f / 180.0f);

    // ---- Re from U_∞ (= u_x_inf in physical), c, ν
    const float U_inf_phys = 1.0f;     // 1 m/s reference
    const float nu_phys = U_inf_phys * chord / args.re;

    const float dx = chord / args.resolution;
    const int nx = (int)std::round(Lx / dx);
    const int ny = (int)std::round(Ly / dx) + 1;
    const int nz = args.nz_thin;
    const int n_cells = nx * ny * nz;

    const float dt = args.u_max_lu * dx / U_inf_phys;
    const float nu_lat = nu_phys * dt / (dx * dx);
    const float tau = nu_lat / D3Q27::CS2 + 0.5f;
    const float omega_nu = 1.0f / tau;
    const float wall_omega = (args.wall_omega_override > 0.0f)
                           ? args.wall_omega_override : omega_nu;

    int total_steps = args.steps;
    if (total_steps < 0) {
        // ~20 convective times (T_conv = c/U_∞)
        const float T_conv = chord / U_inf_phys;
        total_steps = (int)std::ceil(20.0f * T_conv / dt);
    }

    fs::create_directories(args.output_dir);

    std::cout << "================================================================\n"
              << " NACA0012 D3Q27 Cumulant\n"
              << "================================================================\n"
              << " Domain: " << Lx << "×" << Ly << "×" << (nz*dx) << " m  ("
              << args.lx_over_c << "c × " << args.ly_over_c << "c × thin)\n"
              << " Mesh: " << nx << "×" << ny << "×" << nz
              << " (= " << (long long)n_cells << " cells)\n"
              << " dx=" << dx << " m, dt=" << dt << " s\n"
              << " chord=" << chord << " m  (cells/c = " << args.resolution << ")\n"
              << " U_∞=" << U_inf_phys << " m/s, nu=" << nu_phys << " m²/s\n"
              << " Re=" << args.re << ", alpha=" << args.alpha_deg << "°\n"
              << " tau=" << tau << ", omega_nu=" << omega_nu
              << ", wall_omega=" << wall_omega << "\n"
              << " u_max_LU=" << args.u_max_lu << " (Ma="
              << (args.u_max_lu * std::sqrt(3.0f)) << ")\n"
              << " LE position: (" << xLE << ", " << yLE << ") m\n"
              << " Obstacle BC: " << args.bc_mode << "\n"
              << " Total steps: " << total_steps << "\n"
              << "================================================================\n";

    D3Q27::initializeDevice();

    // Allocate device fields
    float *d_f_src = nullptr, *d_f_dst = nullptr;
    float *d_rho = nullptr, *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr;
    unsigned char *d_solid = nullptr;
    const size_t f_size = (size_t)n_cells * 27 * sizeof(float);
    const size_t macro_size = (size_t)n_cells * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_f_src, f_size));
    CUDA_CHECK(cudaMalloc(&d_f_dst, f_size));
    CUDA_CHECK(cudaMalloc(&d_rho, macro_size));
    CUDA_CHECK(cudaMalloc(&d_ux, macro_size));
    CUDA_CHECK(cudaMalloc(&d_uy, macro_size));
    CUDA_CHECK(cudaMalloc(&d_uz, macro_size));
    CUDA_CHECK(cudaMalloc(&d_solid, n_cells * sizeof(unsigned char)));

    // Build mask: --shape selects naca|cylinder|flatplate
    auto h_mask = physics::aero::makeFluidMask(nx, ny, nz);
    const char* shape_name = "?";
    if (args.shape == 1) {
        // Cylinder of diameter = chord, centered at (xLE+0.5c, yLE+cyl_off_y)
        const float cz = 0.5f * nz * dx;
        physics::aero::stampSphere(
            h_mask, nx, ny, nz, dx,
            xLE + 0.5f * chord, yLE + args.cyl_off_y, cz, 0.5f * chord);
        shape_name = "cylinder";
    } else if (args.shape == 2) {
        // Thin flat plate at AOA, plate_thick_cells cells thick
        physics::aero::stampFlatPlate(
            h_mask, nx, ny, nz, dx,
            xLE, yLE, chord, args.plate_thick_cells * dx, alpha_rad);
        shape_name = "flatplate";
    } else {
        physics::aero::stampNacaAirfoil4Digit(
            h_mask, nx, ny, nz, dx,
            xLE, yLE, chord, /*thickness%*/ 12.0f, alpha_rad);
        shape_name = "naca0012";
    }
    std::cout << " Shape: " << shape_name << "\n";
    long long n_solid = 0; for (auto v : h_mask) if (v) ++n_solid;
    std::cout << " NACA0012 stamped: " << n_solid << " solid cells ("
              << (100.0 * n_solid / n_cells) << "% of domain)\n";

    // DEBUG: dump z-mid slice of the actual stamped mask to ASCII for verification
    {
        std::ofstream mf(args.output_dir + "/mask_zmid.txt");
        const int kz = nz / 2;
        // header so Python can parse it
        mf << "# nx=" << nx << " ny=" << ny << " dx=" << dx << " kz=" << kz << "\n";
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t id = (size_t)i + (size_t)j * nx + (size_t)kz * nx * ny;
                mf << (int)h_mask[id];
                if (i + 1 < nx) mf << " ";
            }
            mf << "\n";
        }
        std::cout << " Mask z-mid slice dumped to " << args.output_dir << "/mask_zmid.txt\n";
    }

    CUDA_CHECK(cudaMemcpy(d_solid, h_mask.data(), n_cells * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Build per-link q-fractions (only for --bc qbb-snode + NACA shape).
    float*         d_qfrac = nullptr;          // dense path
    int*           d_qf_offset = nullptr;      // sparse path: offset[N+1]
    unsigned char* d_qf_link_q = nullptr;      // sparse path: link_q[K]
    float*         d_qf_link_val = nullptr;    // sparse path: link_val[K]
    const bool use_qbb = (args.bc_mode == "qbb-snode");
    if (use_qbb) {
        if (args.shape != 0) {
            std::cerr << "ERROR: --bc qbb-snode currently supports only NACA "
                         "(shape=naca). Got shape=" << args.shape << ". Aborting.\n";
            std::exit(1);
        }
        std::cout << " Building D3Q27 q-fractions (sub-sample + bisect)...\n";
        if (args.sparse_qfrac) {
            auto sq = physics::aero::makeNacaQFractionSparse(
                h_mask, nx, ny, nz, dx,
                xLE, yLE, chord, 12.0f, alpha_rad);
            const int K = (int)sq.link_q.size();
            std::cout << " QBB SPARSE enabled: " << K << " fractional links over "
                      << n_cells << " cells (offset+q+val: "
                      << (sq.offset.size()*4 + K*1 + K*4) / 1048576.0
                      << " MB vs dense "
                      << (n_cells * 27 * 4) / 1048576.0 << " MB)\n";
            CUDA_CHECK(cudaMalloc(&d_qf_offset, sq.offset.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_qf_link_q, K * sizeof(unsigned char)));
            CUDA_CHECK(cudaMalloc(&d_qf_link_val, K * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_qf_offset, sq.offset.data(),
                sq.offset.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_qf_link_q, sq.link_q.data(),
                K * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_qf_link_val, sq.link_qfrac.data(),
                K * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            auto h_qfrac = physics::aero::makeNacaQFraction(
                h_mask, nx, ny, nz, dx,
                xLE, yLE, chord, /*thickness%*/ 12.0f, alpha_rad);
            long long n_fractional = 0;
            for (auto v : h_qfrac) if (v < 0.999f) ++n_fractional;
            std::cout << " QBB enabled: " << n_fractional << " fractional links of "
                      << h_qfrac.size() << " total ("
                      << (100.0 * n_fractional / h_qfrac.size()) << "%)\n";
            const size_t qf_bytes = h_qfrac.size() * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_qfrac, qf_bytes));
            CUDA_CHECK(cudaMemcpy(d_qfrac, h_qfrac.data(), qf_bytes,
                                  cudaMemcpyHostToDevice));
        }
    }

    // Initialise fluid at uniform freestream
    {
        const int block = 256;
        const int grid = (n_cells + block - 1) / block;
        initializeFreestream<<<grid, block>>>(d_f_src, n_cells, args.u_max_lu);
        CUDA_CHECK_KERNEL();
    }

    std::ofstream csv(args.output_dir + "/forces.csv");
    csv << "step,t,Fx_LU,Fy_LU,Fz_LU,Fx_phys_per_m,Fy_phys_per_m,Cd,Cl\n";
    csv.precision(8);

    const float rho_phys = 1.0f;
    const float Lz_phys = nz * dx;
    const float force_lu_to_N = rho_phys * dx * dx * dx * dx / (dt * dt);
    // Cd = 2·F_x' / (rho·U_∞²·c)  with F_x' = force per unit z [N/m]
    const float cd_denom_per_m = 0.5f * rho_phys * U_inf_phys * U_inf_phys * chord;

    auto t_start = std::chrono::steady_clock::now();
    int next_log = 1000;

    dim3 block3(4, 4, 4);
    dim3 grid3((nx + 3) / 4, (ny + 3) / 4, (nz + 3) / 4);
    const int macro_grid = (n_cells + 255) / 256;
    dim3 block_face(16, 4, 1);
    dim3 grid_face((ny + 15) / 16, (nz + 3) / 4, 1);

    // TRT magic ω_minus from Λ = (τ+-0.5)(τ--0.5). For ω+ ≡ omega_nu,
    // τ+ = 1/omega_nu, Λ_+ = τ+-0.5, Λ_- = trt_lambda/Λ_+, τ_- = 0.5 + Λ_-,
    // ω_- = 1/τ_-.
    const float tau_plus  = 1.0f / omega_nu;
    const float lambda_plus_trt = tau_plus - 0.5f;
    const float lambda_minus_trt = args.trt_lambda / fmaxf(lambda_plus_trt, 1e-6f);
    const float omega_minus_trt = 1.0f / (0.5f + lambda_minus_trt);
    if (args.use_trt) {
        std::cout << " TRT magic Λ=" << args.trt_lambda
                  << ", ω_+=" << omega_nu << " (shear),"
                  << " ω_-=" << omega_minus_trt << "\n";
    }

    for (int step = 0; step < total_steps; ++step) {
        if (args.use_bgk) {
            fluidBGKD3Q27Kernel<<<grid3, block3>>>(
                d_f_src, d_f_src, d_rho, d_ux, d_uy, d_uz,
                nx, ny, nz, omega_nu);
        } else if (args.use_trt) {
            fluidTRTD3Q27Kernel<<<grid3, block3>>>(
                d_f_src, d_f_src, d_rho, d_ux, d_uy, d_uz,
                nx, ny, nz, omega_nu, omega_minus_trt);
        } else {
            physics::cumulant::fluidCumulantCollisionKernel<<<grid3, block3>>>(
                d_f_src, d_f_src, d_rho, d_ux, d_uy, d_uz,
                nx, ny, nz,
                omega_nu, omega_nu,
                args.omega_3, args.omega_4, args.omega_5, args.omega_6);
        }
        CUDA_CHECK_KERNEL();

        if (args.probe_every > 0 && (step % args.probe_every == 0)) {
            double *d_Fx, *d_Fy, *d_Fz;
            CUDA_CHECK(cudaMalloc(&d_Fx, sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Fy, sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Fz, sizeof(double)));
            const double zero = 0.0;
            CUDA_CHECK(cudaMemcpy(d_Fx, &zero, sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Fy, &zero, sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Fz, &zero, sizeof(double), cudaMemcpyHostToDevice));
            if (use_qbb && args.sparse_qfrac) {
                memForceNaca_QBB_sparse<<<grid3, block3>>>(
                    d_f_src, d_solid, d_qf_offset, d_qf_link_q, d_qf_link_val,
                    wall_omega, nx, ny, nz, d_Fx, d_Fy, d_Fz);
            } else if (use_qbb) {
                memForceNaca_QBB<<<grid3, block3>>>(
                    d_f_src, d_solid, d_qfrac, wall_omega,
                    nx, ny, nz, d_Fx, d_Fy, d_Fz);
            } else {
                memForceNaca<<<grid3, block3>>>(d_f_src, d_solid, nx, ny, nz,
                                                d_Fx, d_Fy, d_Fz);
            }
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            double Fx, Fy, Fz;
            CUDA_CHECK(cudaMemcpy(&Fx, d_Fx, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&Fy, d_Fy, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&Fz, d_Fz, sizeof(double), cudaMemcpyDeviceToHost));
            cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);
            const double Fx_N = Fx * force_lu_to_N;
            const double Fy_N = Fy * force_lu_to_N;
            const double Fx_per_m = Fx_N / Lz_phys;
            const double Fy_per_m = Fy_N / Lz_phys;
            const double Cd = Fx_per_m / cd_denom_per_m;
            const double Cl = Fy_per_m / cd_denom_per_m;
            csv << step << "," << (step * dt) << ","
                << Fx << "," << Fy << "," << Fz << ","
                << Fx_per_m << "," << Fy_per_m << ","
                << Cd << "," << Cl << "\n";
        }

        if (use_qbb && args.sparse_qfrac) {
            physics::cumulant::streamD3Q27_naca_qbb_sparse<<<grid3, block3>>>(
                d_f_src, d_f_dst, d_solid,
                d_qf_offset, d_qf_link_q, d_qf_link_val,
                nx, ny, nz, wall_omega);
        } else if (use_qbb) {
            physics::cumulant::streamD3Q27_naca_qbb<<<grid3, block3>>>(
                d_f_src, d_f_dst, d_solid, d_qfrac,
                nx, ny, nz, wall_omega);
        } else {
            streamD3Q27_naca<<<grid3, block3>>>(d_f_src, d_f_dst, d_solid,
                                                nx, ny, nz);
        }
        CUDA_CHECK_KERNEL();

        applyInletFreestream<<<grid_face, block_face>>>(
            d_f_dst, nx, ny, nz, args.u_max_lu);
        CUDA_CHECK_KERNEL();
        applyOutletExtrap<<<grid_face, block_face>>>(d_f_dst, nx, ny, nz);
        CUDA_CHECK_KERNEL();

        std::swap(d_f_src, d_f_dst);

        if (step + 1 == next_log) {
            csv.flush();
            auto t_now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t_now - t_start).count();
            double mlups = (double)n_cells * (step + 1) / 1e6 / secs;
            std::cout << "[" << (step + 1) << "/" << total_steps
                      << "] t=" << ((step + 1) * dt) << " s, wall=" << secs
                      << " s, " << mlups << " MLUPS" << std::endl;
            next_log = std::min(total_steps, next_log * 2);
        }

        if (args.vtk_every > 0 && ((step + 1) % args.vtk_every == 0)) {
            copyMacroD3Q27_naca<<<macro_grid, 256>>>(d_f_src, d_rho, d_ux, d_uy, d_uz, n_cells);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> h_ux(n_cells), h_uy(n_cells), h_uz(n_cells), h_rho(n_cells);
            CUDA_CHECK(cudaMemcpy(h_ux.data(), d_ux, macro_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy.data(), d_uy, macro_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uz.data(), d_uz, macro_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho, macro_size, cudaMemcpyDeviceToHost));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "/snap_%07d", step + 1);
            io::VTKWriter::writeVectorField(
                args.output_dir + buf,
                h_ux.data(), h_uy.data(), h_uz.data(),
                nx, ny, nz, dx, dx, dx, "velocity");
            // Dump rho z-mid slice as ASCII for pressure analysis
            std::snprintf(buf, sizeof(buf), "/rho_zmid_%07d.txt", step + 1);
            std::ofstream rf(args.output_dir + buf);
            const int kz = nz / 2;
            rf << "# nx=" << nx << " ny=" << ny << " dx=" << dx << "\n";
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const size_t id = (size_t)i + (size_t)j * nx + (size_t)kz * nx * ny;
                    rf << h_rho[id];
                    if (i + 1 < nx) rf << " ";
                }
                rf << "\n";
            }
        }
    }

    csv.close();
    auto t_end = std::chrono::steady_clock::now();
    double tot_secs = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Done. Wall time " << tot_secs << " s, "
              << ((double)n_cells * total_steps / 1e6 / tot_secs) << " MLUPS\n"
              << "Forces written to " << args.output_dir << "/forces.csv\n";

    cudaFree(d_f_src); cudaFree(d_f_dst);
    cudaFree(d_rho); cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
    cudaFree(d_solid);
    if (d_qfrac) cudaFree(d_qfrac);
    if (d_qf_offset)  cudaFree(d_qf_offset);
    if (d_qf_link_q)  cudaFree(d_qf_link_q);
    if (d_qf_link_val) cudaFree(d_qf_link_val);
    return 0;
}
