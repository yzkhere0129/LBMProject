/**
 * @file aero_naca0012_ib.cu
 * @brief NACA0012 + Immersed Boundary LBM (Wu-Shu 2009 framework).
 *
 * Sister driver to aero_naca0012_cumulant.cu — same Cumulant D3Q27 collision,
 * same Ladd inlet, same free-slip Y, same outlet extrapolation. Differs in:
 *   - NO solid-mask stamping. Whole domain is fluid.
 *   - NO obstacle BC in streaming. Standard PULL stream everywhere.
 *   - Surface enforced via Lagrangian markers + IB direct-forcing (Wu-Shu).
 *   - Force probed by reducing Σ_k -ρ · F_L · ds_k over markers.
 *
 * Time step (V0 = 1 explicit IB iteration):
 *   1. Cumulant collide (no force, in-place)
 *   2. Stream (PULL, no obstacle, free-slip Y, periodic z)
 *   3. Inlet (Ladd freestream) + outlet (zero-grad)
 *   4. swap pointers
 *   5. Compute (ρ, u_pre) from f
 *   6. Interpolate u_pre at markers → u_L
 *      F_L = (U_target − u_L) / dt   with U_target = 0   (no-slip wall)
 *      cudaMemset F_field to 0; spread F_L · ds → F_field (atomic)
 *   7. Apply Guo (2002) source term to f using (ρ, u_pre, F_field)
 *   8. Probe F_body = -ρ_phys · Σ_k F_L · ds_k  (CPU reduce, n_markers small)
 *
 * V0 simplifications (intentional, will revisit in Phase D):
 *   - 1 explicit IB iteration (no Wu-Shu implicit loop)
 *   - Guo source uses u_pre (not u_pre + 0.5·F/ρ) — small bias for converged IB
 *   - All IB math in lattice units (markers uploaded as X_LU = X_phys / dx)
 */

#include "core/lattice_d3q27.h"
#include "physics/cumulant/cumulant_d3q27.h"
#include "physics/aero/ib_markers.h"
#include "physics/aero/ib_spread_interp.h"
#include "io/vtk_writer.h"
#include "utils/cuda_check.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace lbm;
namespace fs = std::filesystem;
namespace aero = lbm::physics::aero;
namespace cumulant = lbm::physics::cumulant;

using lbm::core::D3Q27;
using lbm::core::ex27;
using lbm::core::ey27;
using lbm::core::ez27;
using lbm::core::w27;
using lbm::core::opposite27;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    int   resolution  = 40;
    float re          = 1000.0f;
    float alpha_deg   = 8.0f;
    int   steps       = -1;
    int   probe_every = 100;
    int   vtk_every   = 0;
    std::string output_dir = "output_naca_ib";
    int   nz_thin     = 8;
    float u_max_lu    = 0.05f;
    float lx_over_c   = 30.0f;
    float ly_over_c   = 20.0f;
    float xLE_over_c  = 10.0f;
    int   use_bgk     = 0;
    int   ib_iters    = 1;          // V0: 1 explicit iter
    float omega_3 = 1.0f, omega_4 = 1.0f, omega_5 = 1.0f, omega_6 = 1.0f;
    int   shape       = 0;          // 0=NACA, 1=cylinder
    float cyl_radius_phys = 0.5f;   // for --shape cylinder (D=2R; D1 default)
    float ds_per_dx   = 1.0f;       // marker spacing in dx units (V0: 1.0)
};

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
        else if (s == "--ib-iters")    a.ib_iters = std::stoi(next());
        else if (s == "--ds-per-dx")   a.ds_per_dx = std::stof(next());
        else if (s == "--shape") {
            std::string v = next();
            if      (v == "naca")     a.shape = 0;
            else if (v == "cylinder") a.shape = 1;
            else { std::cerr << "Unknown shape: " << v << " (naca|cylinder)\n"; std::exit(1); }
        }
        else if (s == "--cyl-radius") a.cyl_radius_phys = std::stof(next());
        else if (s == "-h" || s == "--help") {
            std::cout <<
              "Usage: aero_naca0012_ib [opts]\n"
              "  --resolution N    cells per chord (NACA) or per diameter (cyl) (default 40)\n"
              "  --re R            Reynolds (default 1000)\n"
              "  --alpha A         alpha [deg] (NACA only, default 8)\n"
              "  --shape S         naca | cylinder (default naca)\n"
              "  --cyl-radius R    cylinder physical radius [m] (default 0.5)\n"
              "  --ib-iters N      Wu-Shu explicit iterations (V0 default 1)\n"
              "  --ds-per-dx X     marker spacing in dx units (default 1.0)\n"
              "  --u-max-lu X      target u_inf in LU (default 0.05)\n"
              "  --steps N         total LBM steps (auto if -1)\n"
              "  --probe-every N   force-probe interval (default 100)\n"
              "  --vtk-every N     VTK snapshot interval (0=off)\n"
              "  --bgk             use BGK (control test for Cumulant)\n";
            std::exit(0);
        } else { std::cerr << "Unknown: " << s << std::endl; std::exit(1); }
    }
    return a;
}

// ---------------------------------------------------------------------------
// y-mirror table for D3Q27 free-slip wall (flips c_y, preserves c_x, c_z)
// ---------------------------------------------------------------------------
__constant__ int y_mirror27_ib[27] = {
    0, 1, 2, 4, 3, 5, 6,
    9, 10, 7, 8,
    11, 12, 13, 14,
    16, 15, 18, 17,
    23, 24, 26, 25, 19, 20, 22, 21
};

// ---------------------------------------------------------------------------
// Initialise f to f_eq(rho=1, u=(u_inf, 0, 0))
// ---------------------------------------------------------------------------
__global__ void initializeFreestream(float* f, int n_cells, float u_x_inf) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f[id + q * n_cells] =
            D3Q27::computeEquilibrium(q, 1.0f, u_x_inf, 0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Ladd freestream inlet (post-stream, x=0 face, q with c_x>0)
// ---------------------------------------------------------------------------
__global__ void applyInletFreestream(float* f, int nx, int ny, int nz, float u_x_inf) {
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

// ---------------------------------------------------------------------------
// Outlet zero-gradient (copy from i=nx-2 to i=nx-1)
// ---------------------------------------------------------------------------
__global__ void applyOutletExtrap(float* f, int nx, int ny, int nz) {
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

// ---------------------------------------------------------------------------
// PULL streaming, NO obstacle (whole domain fluid). Free-slip on Y faces,
// periodic Z, halfway BB on X faces (overwritten by inlet/outlet kernels).
// ---------------------------------------------------------------------------
__global__ void streamD3Q27_freestream(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    int nx, int ny, int nz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;
    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    for (int q = 0; q < 27; ++q) {
        int src_x = idx - ex27[q];
        int src_y = idy - ey27[q];
        int src_z = idz - ez27[q];
        if (src_z < 0)  src_z += nz;
        if (src_z >= nz) src_z -= nz;
        bool oob_y = (src_y < 0 || src_y >= ny);
        bool oob_x = (src_x < 0 || src_x >= nx);
        if (oob_y || oob_x) {
            const int reflected_q = oob_y ? y_mirror27_ib[q]
                                          : opposite27[q];
            f_dst[id + q * n_cells] = f_src[id + reflected_q * n_cells];
            continue;
        }
        const int src_id = src_x + src_y * nx + src_z * nx * ny;
        f_dst[id + q * n_cells] = f_src[src_id + q * n_cells];
    }
}

// ---------------------------------------------------------------------------
// Compute macro (rho, u) from f. No force correction — gives u_pre.
// ---------------------------------------------------------------------------
__global__ void copyMacroD3Q27(
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

// ---------------------------------------------------------------------------
// Compute IB force at each marker: F_L = (U_target - u_L) / dt_LU, U_target=0.
// dt_LU = 1, so F_L = -u_L.
// ---------------------------------------------------------------------------
__global__ void computeIBForceKernel(
    const float* __restrict__ d_uL_x,
    const float* __restrict__ d_uL_y,
    const float* __restrict__ d_uL_z,
    int n_markers,
    float* __restrict__ d_FL_x,
    float* __restrict__ d_FL_y,
    float* __restrict__ d_FL_z)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_markers) return;
    d_FL_x[k] = -d_uL_x[k];
    d_FL_y[k] = -d_uL_y[k];
    d_FL_z[k] = -d_uL_z[k];
}

// ---------------------------------------------------------------------------
// Apply Guo (2002) forcing source term to f.
//
// S_q = w_q · (1 − ω/2) · [ 3·(c_q · F − u · F) + 9·(c_q · u)·(c_q · F) ]
// f_q[id] += S_q · dt_LU  (dt_LU = 1)
//
// Uses u_pre (read from d_ux/d_uy/d_uz) — V0 simplification.
// Skips cells with no force (early return) for cheap no-op outside obstacle.
// ---------------------------------------------------------------------------
__global__ void applyGuoForcing(
    float* f,
    const float* __restrict__ rho,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    const float* __restrict__ Fx_field,
    const float* __restrict__ Fy_field,
    const float* __restrict__ Fz_field,
    int nx, int ny, int nz, float omega)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;
    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    const float Fx = Fx_field[id];
    const float Fy = Fy_field[id];
    const float Fz = Fz_field[id];
    if (Fx == 0.0f && Fy == 0.0f && Fz == 0.0f) return;

    const float ux_ = ux[id];
    const float uy_ = uy[id];
    const float uz_ = uz[id];
    const float coef = (1.0f - 0.5f * omega);
    const float u_dot_F = ux_ * Fx + uy_ * Fy + uz_ * Fz;

    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float cx = (float)ex27[q];
        const float cy = (float)ey27[q];
        const float cz = (float)ez27[q];
        const float c_dot_u = cx * ux_ + cy * uy_ + cz * uz_;
        const float c_dot_F = cx * Fx + cy * Fy + cz * Fz;
        const float S = w27[q] * coef * (3.0f * (c_dot_F - u_dot_F)
                                       + 9.0f * c_dot_u * c_dot_F);
        f[id + q * n_cells] += S;
    }
}

// ---------------------------------------------------------------------------
// Simple BGK D3Q27 collision (control)
// ---------------------------------------------------------------------------
__global__ void fluidBGKD3Q27Kernel(
    const float* f_src, float* f_dst,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int nx, int ny, int nz, float omega)
{
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
        mx += ex27[q] * fq;
        my += ey27[q] * fq;
        mz += ez27[q] * fq;
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);
    fs::create_directories(args.output_dir);

    D3Q27::initializeDevice();

    // Domain
    const float chord = 1.0f;                   // [m] reference length
    const float dx = chord / float(args.resolution);
    const int   nx = int(std::round(args.lx_over_c * chord / dx));
    const int   ny = int(std::round(args.ly_over_c * chord / dx));
    const int   nz = args.nz_thin;
    const size_t n_cells = size_t(nx) * ny * nz;

    // Physics in physical units
    const float U_inf_phys = 1.0f;              // [m/s]
    const float nu_phys = U_inf_phys * chord / args.re;
    const float dt = args.u_max_lu * dx / U_inf_phys;
    const float nu_LU = args.u_max_lu * float(args.resolution) / args.re;
    const float tau   = 3.0f * nu_LU + 0.5f;
    const float omega_nu = 1.0f / tau;

    const float xLE = args.xLE_over_c * chord;
    const float yLE = 0.5f * args.ly_over_c * chord;
    const float alpha_rad = args.alpha_deg * float(M_PI) / 180.0f;

    std::cout << "================================================================\n"
              << " NACA0012 IB-LBM (Wu-Shu 2009)\n"
              << " Domain " << nx << "×" << ny << "×" << nz
              << "  dx=" << dx << " m  dt=" << dt << " s\n"
              << " U_inf=" << U_inf_phys << " m/s, nu=" << nu_phys << " m²/s\n"
              << " Re=" << args.re << ", alpha=" << args.alpha_deg << "°\n"
              << " tau=" << tau << ", omega_nu=" << omega_nu << "\n"
              << " u_max_LU=" << args.u_max_lu
              << " (Ma=" << (args.u_max_lu * std::sqrt(3.0f)) << ")\n"
              << " LE=(" << xLE << ", " << yLE << ") m\n"
              << " Shape: " << (args.shape == 0 ? "NACA0012" : "cylinder") << "\n";

    int total_steps = args.steps;
    if (total_steps < 0) total_steps = int(std::round(15.0f * float(nx) / args.u_max_lu));
    std::cout << " Total steps: " << total_steps << "\n"
              << " IB iters/step: " << args.ib_iters << "\n";

    // ---- Build markers (host AoS in physical units) ----
    std::vector<aero::IBMarker> m2d;
    const float ds_target_phys = args.ds_per_dx * dx;
    if (args.shape == 0) {
        m2d = aero::buildNacaMarkers2D(
            chord, /*thick%*/ 12.0f, alpha_rad, xLE, yLE, ds_target_phys);
    } else {
        m2d = aero::buildCircleMarkers2D(
            xLE, yLE, args.cyl_radius_phys, ds_target_phys);
    }
    std::vector<aero::IBMarker> markers = aero::extrudeZ(m2d, nz, dx);
    const int n_markers = int(markers.size());
    std::cout << " Markers per layer: " << m2d.size()
              << ", total (× nz=" << nz << "): " << n_markers << "\n";

    // Convert to LU & SoA. Marker coords in LU = phys/dx; ds_LU = ds_phys/dx.
    std::vector<float> h_xL_lu(n_markers), h_yL_lu(n_markers), h_zL_lu(n_markers),
                       h_ds_lu(n_markers);
    for (int k = 0; k < n_markers; ++k) {
        h_xL_lu[k] = markers[k].x / dx;
        h_yL_lu[k] = markers[k].y / dx;
        h_zL_lu[k] = markers[k].z / dx;
        h_ds_lu[k] = markers[k].ds / dx;
    }

    // ---- Allocate GPU buffers ----
    float *d_f_src, *d_f_dst;
    float *d_rho, *d_ux, *d_uy, *d_uz;
    float *d_Fx_field, *d_Fy_field, *d_Fz_field;
    float *d_xL, *d_yL, *d_zL, *d_ds;
    float *d_uL_x, *d_uL_y, *d_uL_z;
    float *d_FL_x, *d_FL_y, *d_FL_z;
    const size_t f_size = n_cells * 27 * sizeof(float);
    const size_t macro_size = n_cells * sizeof(float);
    const size_t marker_size = n_markers * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_f_src, f_size));
    CUDA_CHECK(cudaMalloc(&d_f_dst, f_size));
    CUDA_CHECK(cudaMalloc(&d_rho, macro_size));
    CUDA_CHECK(cudaMalloc(&d_ux,  macro_size));
    CUDA_CHECK(cudaMalloc(&d_uy,  macro_size));
    CUDA_CHECK(cudaMalloc(&d_uz,  macro_size));
    CUDA_CHECK(cudaMalloc(&d_Fx_field, macro_size));
    CUDA_CHECK(cudaMalloc(&d_Fy_field, macro_size));
    CUDA_CHECK(cudaMalloc(&d_Fz_field, macro_size));
    CUDA_CHECK(cudaMalloc(&d_xL, marker_size));
    CUDA_CHECK(cudaMalloc(&d_yL, marker_size));
    CUDA_CHECK(cudaMalloc(&d_zL, marker_size));
    CUDA_CHECK(cudaMalloc(&d_ds, marker_size));
    CUDA_CHECK(cudaMalloc(&d_uL_x, marker_size));
    CUDA_CHECK(cudaMalloc(&d_uL_y, marker_size));
    CUDA_CHECK(cudaMalloc(&d_uL_z, marker_size));
    CUDA_CHECK(cudaMalloc(&d_FL_x, marker_size));
    CUDA_CHECK(cudaMalloc(&d_FL_y, marker_size));
    CUDA_CHECK(cudaMalloc(&d_FL_z, marker_size));

    CUDA_CHECK(cudaMemcpy(d_xL, h_xL_lu.data(), marker_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_yL, h_yL_lu.data(), marker_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_zL, h_zL_lu.data(), marker_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ds, h_ds_lu.data(), marker_size, cudaMemcpyHostToDevice));

    // Init freestream
    {
        const int block = 256;
        const int grid = (n_cells + block - 1) / block;
        initializeFreestream<<<grid, block>>>(d_f_src, n_cells, args.u_max_lu);
        CUDA_CHECK_KERNEL();
    }

    // Open forces.csv
    std::ofstream csv(args.output_dir + "/forces.csv");
    csv << "step,t,Fx_LU,Fy_LU,Fz_LU,Fx_phys_per_m,Fy_phys_per_m,Cd,Cl\n";
    csv.precision(8);

    const float rho_phys = 1.0f;
    const float Lz_phys = nz * dx;
    const float force_lu_to_N = rho_phys * dx * dx * dx * dx / (dt * dt);
    const float ref_len = (args.shape == 0) ? chord : (2.0f * args.cyl_radius_phys);
    const float cd_denom_per_m = 0.5f * rho_phys * U_inf_phys * U_inf_phys * ref_len;

    auto t_start = std::chrono::steady_clock::now();
    int next_log = 1000;

    // Launch configs
    dim3 block3(4, 4, 4);
    dim3 grid3((nx + 3) / 4, (ny + 3) / 4, (nz + 3) / 4);
    const int macro_block = 256;
    const int macro_grid = (n_cells + macro_block - 1) / macro_block;
    dim3 block_face(16, 4, 1);
    dim3 grid_face((ny + 15) / 16, (nz + 3) / 4, 1);
    const int marker_block = 128;
    const int marker_grid = (n_markers + marker_block - 1) / marker_block;

    std::vector<float> h_FL_x(n_markers), h_FL_y(n_markers), h_FL_z(n_markers);

    for (int step = 0; step < total_steps; ++step) {
        // 1. Cumulant collide (in-place)
        if (args.use_bgk) {
            fluidBGKD3Q27Kernel<<<grid3, block3>>>(
                d_f_src, d_f_src, d_rho, d_ux, d_uy, d_uz,
                nx, ny, nz, omega_nu);
        } else {
            cumulant::fluidCumulantCollisionKernel<<<grid3, block3>>>(
                d_f_src, d_f_src, d_rho, d_ux, d_uy, d_uz,
                nx, ny, nz,
                omega_nu, omega_nu,
                args.omega_3, args.omega_4, args.omega_5, args.omega_6);
        }
        CUDA_CHECK_KERNEL();

        // 2. Stream
        streamD3Q27_freestream<<<grid3, block3>>>(d_f_src, d_f_dst, nx, ny, nz);
        CUDA_CHECK_KERNEL();

        // 3. BC
        applyInletFreestream<<<grid_face, block_face>>>(d_f_dst, nx, ny, nz, args.u_max_lu);
        CUDA_CHECK_KERNEL();
        applyOutletExtrap<<<grid_face, block_face>>>(d_f_dst, nx, ny, nz);
        CUDA_CHECK_KERNEL();

        // 4. Swap
        std::swap(d_f_src, d_f_dst);

        // 5. Compute u_pre
        copyMacroD3Q27<<<macro_grid, macro_block>>>(
            d_f_src, d_rho, d_ux, d_uy, d_uz, n_cells);
        CUDA_CHECK_KERNEL();

        // 6. IB iterations (V0: 1 explicit)
        for (int it = 0; it < args.ib_iters; ++it) {
            aero::interpolateVelocityKernel<<<marker_grid, marker_block>>>(
                d_ux, d_uy, d_uz, nx, ny, nz, /*dx_LU=*/1.0f, /*periodic_z=*/1,
                d_xL, d_yL, d_zL, n_markers,
                d_uL_x, d_uL_y, d_uL_z);
            CUDA_CHECK_KERNEL();

            computeIBForceKernel<<<marker_grid, marker_block>>>(
                d_uL_x, d_uL_y, d_uL_z, n_markers,
                d_FL_x, d_FL_y, d_FL_z);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemset(d_Fx_field, 0, macro_size));
            CUDA_CHECK(cudaMemset(d_Fy_field, 0, macro_size));
            CUDA_CHECK(cudaMemset(d_Fz_field, 0, macro_size));

            aero::spreadForceKernel<<<marker_grid, marker_block>>>(
                d_xL, d_yL, d_zL, d_ds,
                d_FL_x, d_FL_y, d_FL_z,
                n_markers, nx, ny, nz, /*dx_LU=*/1.0f, /*periodic_z=*/1,
                d_Fx_field, d_Fy_field, d_Fz_field);
            CUDA_CHECK_KERNEL();

            // 7. Guo source on f
            applyGuoForcing<<<grid3, block3>>>(
                d_f_src, d_rho, d_ux, d_uy, d_uz,
                d_Fx_field, d_Fy_field, d_Fz_field,
                nx, ny, nz, omega_nu);
            CUDA_CHECK_KERNEL();
        }

        // 8. Force probe
        if (args.probe_every > 0 && (step % args.probe_every == 0)) {
            CUDA_CHECK(cudaMemcpy(h_FL_x.data(), d_FL_x, marker_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_FL_y.data(), d_FL_y, marker_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_FL_z.data(), d_FL_z, marker_size, cudaMemcpyDeviceToHost));
            // F_body_LU = -ρ_LU · Σ_k F_L · ds_LU. ρ_LU ≈ 1.
            double Fx_LU = 0, Fy_LU = 0, Fz_LU = 0;
            for (int k = 0; k < n_markers; ++k) {
                Fx_LU -= double(h_FL_x[k]) * double(h_ds_lu[k]);
                Fy_LU -= double(h_FL_y[k]) * double(h_ds_lu[k]);
                Fz_LU -= double(h_FL_z[k]) * double(h_ds_lu[k]);
            }
            const double Fx_N = Fx_LU * force_lu_to_N;
            const double Fy_N = Fy_LU * force_lu_to_N;
            const double Fx_per_m = Fx_N / Lz_phys;
            const double Fy_per_m = Fy_N / Lz_phys;
            const double Cd = Fx_per_m / cd_denom_per_m;
            const double Cl = Fy_per_m / cd_denom_per_m;
            csv << step << "," << (step * dt) << ","
                << Fx_LU << "," << Fy_LU << "," << Fz_LU << ","
                << Fx_per_m << "," << Fy_per_m << ","
                << Cd << "," << Cl << "\n";
        }

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

        // VTK
        if (args.vtk_every > 0 && ((step + 1) % args.vtk_every == 0)) {
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
    cudaFree(d_Fx_field); cudaFree(d_Fy_field); cudaFree(d_Fz_field);
    cudaFree(d_xL); cudaFree(d_yL); cudaFree(d_zL); cudaFree(d_ds);
    cudaFree(d_uL_x); cudaFree(d_uL_y); cudaFree(d_uL_z);
    cudaFree(d_FL_x); cudaFree(d_FL_y); cudaFree(d_FL_z);
    return 0;
}
