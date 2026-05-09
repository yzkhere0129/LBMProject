/**
 * @file aero_st_2d2_cumulant.cu
 * @brief Schäfer-Turek 2D-2 cylinder benchmark with D3Q27 Cumulant LBM
 *
 * Self-contained driver — does NOT use FluidLBM (which is D3Q19 + TRT-EDM).
 * Manages its own D3Q27 PDF arrays, streaming, BCs, and Cumulant collision.
 *
 * Goal: break the +12% Mach plateau hit by D3Q19 TRT, by using Cumulant LBM
 * which has improved Galilean invariance.
 *
 * Setup:
 *   Domain: 2.2 m × 0.41 m × Lz (z-periodic, Nz=4)
 *   Cylinder: D=0.1 m at (0.2, 0.2)
 *   Inlet (x_min):  parabolic u_x(y) = 4·U·y(H-y)/H², set via f = f_eq(ρ=1, u)
 *   Outlet (x_max): zero-gradient extrapolation from i=nx-2 to i=nx-1
 *   Top/bot walls (y_min, y_max): halfway BB at link midpoint
 *   Cylinder surface: stair-step halfway BB at link midpoint (Phase 4 first cut;
 *                     BFL / quad-Bouzidi on D3Q27 deferred)
 *   z faces: periodic
 *
 * Cumulant relaxation rates (default, Geier 2017 §3.3 for moderate Re):
 *   ω_ν = 1/τ from physical viscosity
 *   ω_b = ω_ν (bulk = shear unless tuned)
 *   ω_3 = ω_4 = ω_5 = ω_6 = 1.0 (full relaxation to MB equilibrium)
 */

#include "core/lattice_d3q27.h"
#include "physics/cumulant/cumulant_d3q27.h"
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

// ============================================================================
// CLI parsing
// ============================================================================
struct Args {
    int   resolution = 20;       // cells per cylinder diameter
    float re         = 100.0f;
    int   steps      = -1;
    int   probe_every = 100;
    int   vtk_every   = 0;
    std::string output_dir = "output_aero_st2d2_cumulant";
    int   nz_thin    = 4;
    float u_max_lu   = 0.05f;    // target u_max in lattice units
    float omega_3    = 1.0f;
    float omega_4    = 1.0f;
    float omega_5    = 1.0f;
    float omega_6    = 1.0f;
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
        else if (s == "--steps")       a.steps = std::stoi(next());
        else if (s == "--probe-every") a.probe_every = std::stoi(next());
        else if (s == "--vtk-every")   a.vtk_every = std::stoi(next());
        else if (s == "--output-dir")  a.output_dir = next();
        else if (s == "--nz-thin")     a.nz_thin = std::stoi(next());
        else if (s == "--u-max-lu")    a.u_max_lu = std::stof(next());
        else if (s == "--omega-3")     a.omega_3 = std::stof(next());
        else if (s == "--omega-4")     a.omega_4 = std::stof(next());
        else if (s == "--omega-5")     a.omega_5 = std::stof(next());
        else if (s == "--omega-6")     a.omega_6 = std::stof(next());
        else if (s == "-h" || s == "--help") {
            std::cout <<
              "Usage: aero_st_2d2_cumulant [opts]\n"
              "  --resolution N    cells per D (default 20)\n"
              "  --re R            Reynolds (default 100)\n"
              "  --steps N         total LBM steps (auto if -1)\n"
              "  --u-max-lu X      target u_max in LU (default 0.05; Mach control)\n"
              "  --probe-every N   Cd/Cl sample interval (default 100)\n"
              "  --vtk-every N     VTK snapshot interval (0 = off)\n"
              "  --omega-3/4/5/6 X higher-order cumulant relaxation rates\n";
            std::exit(0);
        } else { std::cerr << "Unknown: " << s << std::endl; std::exit(1); }
    }
    return a;
}

// ============================================================================
// Device-side kernels
// ============================================================================
using core::ex27;
using core::ey27;
using core::ez27;
using core::w27;
using core::opposite27;
using core::D3Q27;

// Stamp cylinder mask (host)
static std::vector<unsigned char> makeCylinderMask(
    int nx, int ny, int nz, float dx,
    float cx, float cy, float R)
{
    std::vector<unsigned char> mask((size_t)nx * ny * nz, 0);
    const float R2 = R * R;
    for (int j = 0; j < ny; ++j) {
        const float y = (j + 0.5f) * dx;
        const float dy = y - cy;
        for (int i = 0; i < nx; ++i) {
            const float x = (i + 0.5f) * dx;
            const float dxc = x - cx;
            if (dxc * dxc + dy * dy <= R2) {
                for (int k = 0; k < nz; ++k) {
                    mask[(size_t)i + nx * (j + (size_t)ny * k)] = 1;
                }
            }
        }
    }
    return mask;
}

// Initialize all PDFs with f_eq(ρ=1, u=0)
__global__ void initializeUniformF27(float* f, int n_cells) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f[id + q * n_cells] = D3Q27::computeEquilibrium(q, 1.0f, 0.0f, 0.0f, 0.0f);
    }
}

// Apply parabolic-velocity inlet to f at i=0 (overwrite all 27 PDFs with f_eq).
// Run BEFORE collision; Cumulant collision preserves equilibrium so the
// inlet condition stays valid through collision + streaming.
__global__ void applyInletD3Q27(
    float* f, int nx, int ny, int nz,
    float u_max_lu, float dx_phys)
{
    const int idy = blockIdx.x * blockDim.x + threadIdx.x;
    const int idz = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= ny || idz >= nz) return;

    const int idx = 0;
    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    const float H_phys = (ny - 1) * dx_phys;
    const float y_phys = idy * dx_phys;
    const float u_x = 4.0f * u_max_lu * y_phys * (H_phys - y_phys)
                     / (H_phys * H_phys);
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f[id + q * n_cells] = D3Q27::computeEquilibrium(q, 1.0f, u_x, 0.0f, 0.0f);
    }
}

// Apply outlet (zero-gradient extrapolation): f_src[nx-1] = f_src[nx-2].
// Run BEFORE collision so collision propagates correctly.
__global__ void applyOutletD3Q27(
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

// Combined streaming + BC kernel for D3Q27.
// Halfway BB on Y faces, periodic on Z, parabolic inlet on X_min via f_eq,
// extrapolation outlet on X_max, halfway BB at solid cells.
__global__ void streamD3Q27(
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

    // Inlet/outlet handled by separate pre-collision kernels. Here just
    // stream — at i=0 cells we still stream from (i=-1) which is OOB; treat
    // as halfway BB so f_dst[i=0, q] for q with ex<0 stays defined. The
    // applyInlet kernel runs before the next collision, overwriting.

    // Generic interior fluid cell: stream from each direction's pre-image.
    // For each q, source = X - e_q. If source is solid OR out-of-domain on
    // a non-periodic face, apply halfway BB: take f_src at THIS cell in
    // direction opp[q] (= the population that was about to leave in -q
    // direction last step).
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        int src_x = idx - ex27[q];
        int src_y = idy - ey27[q];
        int src_z = idz - ez27[q];

        // Z-periodic
        if (src_z < 0)  src_z += nz;
        if (src_z >= nz) src_z -= nz;

        bool out_of_domain = false;
        if (src_x < 0 || src_x >= nx) out_of_domain = true;
        if (src_y < 0 || src_y >= ny) out_of_domain = true;

        if (out_of_domain) {
            // Y face wall (halfway BB). X face will not happen in interior
            // (we already handled idx=0 and nx-1 above, so src_x=-1 means
            // we're streaming from inlet which is at x=0 inside domain — OK).
            // For y boundaries: bounce
            if (src_y < 0 || src_y >= ny) {
                // Halfway BB: read from THIS cell in opposite direction
                f_dst[id + q * n_cells] = f_src[id + opposite27[q] * n_cells];
                continue;
            }
            // X out of domain shouldn't occur for idx ∈ [1, nx-2]; defensive.
            f_dst[id + q * n_cells] = f_src[id + opposite27[q] * n_cells];
            continue;
        }

        const int src_id = src_x + src_y * nx + src_z * nx * ny;
        if (solid_mask[src_id] != 0) {
            // Solid neighbour: halfway BB
            f_dst[id + q * n_cells] = f_src[id + opposite27[q] * n_cells];
        } else {
            f_dst[id + q * n_cells] = f_src[src_id + q * n_cells];
        }
    }
}

// MEM force for D3Q27
__global__ void memForceD3Q27(
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

        // Halfway BB MEM: F = 2·c·f_in
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

// Macro field copy for VTK output
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

// ============================================================================
int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    // ---- Schäfer-Turek 2D-2 spec ----
    const float Lx = 2.2f, Ly = 0.41f, D = 0.1f;
    const float cx = 0.2f, cy = 0.2f;
    const float U_max_phys = 1.5f;
    const float U_avg_phys = (2.0f / 3.0f) * U_max_phys;
    const float nu_phys = U_avg_phys * D / args.re;

    const float dx = D / args.resolution;
    const int nx = (int)std::round(Lx / dx);
    const int ny = (int)std::round(Ly / dx) + 1;
    const int nz = args.nz_thin;
    const int n_cells = nx * ny * nz;

    const float dt = args.u_max_lu * dx / U_max_phys;
    const float nu_lat = nu_phys * dt / (dx * dx);
    const float tau = nu_lat / D3Q27::CS2 + 0.5f;
    const float omega_nu = 1.0f / tau;

    int total_steps = args.steps;
    if (total_steps < 0) {
        const float St_guess = 0.30f;
        const float T_shed = D / (St_guess * U_avg_phys);
        const float T_total = 12.0f * T_shed;
        total_steps = (int)std::ceil(T_total / dt);
    }

    fs::create_directories(args.output_dir);

    std::cout << "================================================================\n"
              << " ST 2D-2 Cumulant D3Q27\n"
              << "================================================================\n"
              << " Mesh: " << nx << "×" << ny << "×" << nz
              << " (= " << (long long)n_cells << " cells)\n"
              << " dx=" << dx << " m, dt=" << dt << " s\n"
              << " nu=" << nu_phys << " m²/s, tau=" << tau
              << ", omega_nu=" << omega_nu << "\n"
              << " u_max_LU=" << args.u_max_lu << " (Ma="
              << (args.u_max_lu * std::sqrt(3.0f)) << ")\n"
              << " Re=" << args.re << ", total steps=" << total_steps << "\n"
              << " omega 3/4/5/6 = " << args.omega_3 << " / "
              << args.omega_4 << " / " << args.omega_5 << " / "
              << args.omega_6 << "\n"
              << "================================================================\n";

    // Initialise D3Q27 lattice tables on device
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

    // Build cylinder mask
    auto h_mask = makeCylinderMask(nx, ny, nz, dx, cx, cy, 0.5f * D);
    long long n_solid = 0; for (auto v : h_mask) if (v) ++n_solid;
    std::cout << " Solid cells: " << n_solid << " ("
              << (100.0 * n_solid / n_cells) << "% of domain)\n";
    CUDA_CHECK(cudaMemcpy(d_solid, h_mask.data(), n_cells * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Initialize at rest
    {
        const int block = 256;
        const int grid = (n_cells + block - 1) / block;
        initializeUniformF27<<<grid, block>>>(d_f_src, n_cells);
        CUDA_CHECK_KERNEL();
    }

    // Open Cd/Cl CSV
    std::ofstream csv(args.output_dir + "/forces.csv");
    csv << "step,t,Fx_LU,Fy_LU,Fz_LU,Fx_phys_per_m,Fy_phys_per_m,Cd,Cl\n";
    csv.precision(8);

    const float rho_phys = 1.0f;
    const float Lz_phys = nz * dx;
    const float force_lu_to_N = rho_phys * dx * dx * dx * dx / (dt * dt);
    const float cd_denom_per_m = 0.5f * rho_phys * U_avg_phys * U_avg_phys * D;

    auto t_start = std::chrono::steady_clock::now();
    int next_log = 1000;

    // Cumulant kernel uses ~108 floats (~432 B) per thread of register-spillable
    // local arrays; small block size needed to fit register budget on RTX 30xx.
    dim3 block3(4, 4, 4);
    dim3 grid3((nx + 3) / 4, (ny + 3) / 4, (nz + 3) / 4);
    const int macro_grid = (n_cells + 255) / 256;

    // 2D grid for inlet/outlet (per (j,k) face cell)
    dim3 block_face(16, 4, 1);
    dim3 grid_face((ny + 15) / 16, (nz + 3) / 4, 1);

    for (int step = 0; step < total_steps; ++step) {
        // Apply inlet (parabolic velocity via f = f_eq) on f_src[i=0]
        applyInletD3Q27<<<grid_face, block_face>>>(
            d_f_src, nx, ny, nz, args.u_max_lu, dx);
        CUDA_CHECK_KERNEL();

        // Apply outlet (extrapolation) on f_src[i=nx-1]
        applyOutletD3Q27<<<grid_face, block_face>>>(d_f_src, nx, ny, nz);
        CUDA_CHECK_KERNEL();

        // Cumulant collision (in-place: f_src → f_src)
        physics::cumulant::fluidCumulantCollisionKernel<<<grid3, block3>>>(
            d_f_src, d_f_src, d_rho, d_ux, d_uy, d_uz,
            nx, ny, nz,
            omega_nu, omega_nu,
            args.omega_3, args.omega_4, args.omega_5, args.omega_6);
        CUDA_CHECK_KERNEL();

        // MEM force from post-collision pre-streaming
        if (args.probe_every > 0 && (step % args.probe_every == 0)) {
            double *d_Fx, *d_Fy, *d_Fz;
            CUDA_CHECK(cudaMalloc(&d_Fx, sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Fy, sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Fz, sizeof(double)));
            const double zero = 0.0;
            CUDA_CHECK(cudaMemcpy(d_Fx, &zero, sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Fy, &zero, sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Fz, &zero, sizeof(double), cudaMemcpyHostToDevice));
            memForceD3Q27<<<grid3, block3>>>(d_f_src, d_solid, nx, ny, nz,
                                             d_Fx, d_Fy, d_Fz);
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

        // Streaming (interior only; inlet/outlet enforced pre-collision)
        streamD3Q27<<<grid3, block3>>>(
            d_f_src, d_f_dst, d_solid, nx, ny, nz);
        CUDA_CHECK_KERNEL();

        // Swap
        std::swap(d_f_src, d_f_dst);

        // Logging
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
            // Recompute macroscopic from current f_src
            copyMacroD3Q27<<<macro_grid, 256>>>(d_f_src, d_rho, d_ux, d_uy, d_uz,
                                                n_cells);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> h_ux(n_cells), h_uy(n_cells), h_uz(n_cells);
            CUDA_CHECK(cudaMemcpy(h_ux.data(), d_ux, macro_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy.data(), d_uy, macro_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uz.data(), d_uz, macro_size, cudaMemcpyDeviceToHost));
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
    cudaFree(d_solid);
    return 0;
}
