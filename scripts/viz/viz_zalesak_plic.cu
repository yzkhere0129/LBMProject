/**
 * @file viz_zalesak_plic.cu
 * @brief Zalesak slotted-disk benchmark with PLIC-VOF advection.
 *
 * Same setup as viz_zalesak.cu but uses geometric PLIC instead of algebraic TVD.
 * Sharp initialization (no tanh smoothing) to match published benchmarks.
 *
 * Compile (from build/):
 *   nvcc -o ../scripts/viz/viz_zalesak_plic ../scripts/viz/viz_zalesak_plic.cu \
 *        -I../include -L. -llbm_physics -llbm_io -llbm_diagnostics -llbm_core \
 *        --std=c++17 -rdc=true -Xcompiler -fPIC -O2
 */

#include "physics/vof_solver.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm::physics;

// ---- helpers ----------------------------------------------------------------

static void dumpSliceCSV(const std::string& path,
                          const std::vector<float>& fill,
                          int nx, int ny, int nz)
{
    int k0 = nz / 2;
    std::ofstream f(path);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * k0);
            f << fill[idx];
            if (i < nx - 1) f << ',';
        }
        f << '\n';
    }
}

static void initSlottedDiskSharp(VOFSolver& vof,
                                  int nx, int ny, int nz,
                                  float cx, float cy,
                                  float R, float slot_w, float slot_d)
{
    std::vector<float> h(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float dx = i - cx, dy = j - cy;
                float r  = std::sqrt(dx*dx + dy*dy);
                bool in_disk = (r <= R);
                bool in_slot = (std::fabs(dx) <= slot_w * 0.5f)
                             && (dy >= 0.0f) && (dy <= slot_d);
                int idx = i + nx*(j + ny*k);
                if (in_disk && !in_slot) {
                    h[idx] = 1.0f;  // sharp initialization
                }
            }
    vof.initialize(h.data());
}

static void uploadRotationVelocity(float* d_ux, float* d_uy, float* d_uz,
                                    float omega, float cx, float cy,
                                    int nx, int ny, int nz)
{
    int N = nx * ny * nz;
    std::vector<float> hx(N), hy(N), hz(N, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx*(j + ny*k);
                hx[idx] = -omega * (j - cy);
                hy[idx] =  omega * (i - cx);
            }
    cudaMemcpy(d_ux, hx.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, hy.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, hz.data(), N*sizeof(float), cudaMemcpyHostToDevice);
}

// ---- main -------------------------------------------------------------------

int main()
{
    const int nx = 128, ny = 128, nz = 4;
    const float dx = 1.0f;
    const int   N  = nx * ny * nz;

    const float cx = nx * 0.5f;
    const float cy = ny * 0.5f;
    const float R       = 30.0f;
    const float slot_w  = 10.0f;
    const float slot_d  = 50.0f;

    const int   steps_per_rev = 800;
    const float omega = 2.0f * M_PI / static_cast<float>(steps_per_rev);
    const float dt    = 1.0f;

    std::printf("Zalesak Disk — PLIC-VOF\n");
    std::printf("  Grid: %d x %d x %d,  R=%.0f,  omega=%.5f rad/step\n",
                nx, ny, nz, R, omega);
    std::printf("  CFL_max = %.3f\n", omega * R * dt / dx);
    std::printf("  Steps per revolution: %d\n", steps_per_rev);

    // PLIC geometric advection
    VOFSolver vof(nx, ny, nz, dx);
    vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
    initSlottedDiskSharp(vof, nx, ny, nz, cx, cy, R, slot_w, slot_d);

    const float mass0 = vof.computeTotalMass();

    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, N * sizeof(float));
    cudaMalloc(&d_uy, N * sizeof(float));
    cudaMalloc(&d_uz, N * sizeof(float));
    uploadRotationVelocity(d_ux, d_uy, d_uz, omega, cx, cy, nx, ny, nz);

    std::vector<float> h_fill(N);
    const char* out_dir = "/home/yzk/LBMProject/scripts/viz";

    const int snap_steps[] = {0, steps_per_rev/4, steps_per_rev/2,
                               3*steps_per_rev/4, steps_per_rev};

    auto saveSnap = [&](int s) {
        vof.copyFillLevelToHost(h_fill.data());
        char fname[256];
        std::snprintf(fname, sizeof(fname), "%s/zalesak_plic_deg%03d.csv",
                      out_dir, static_cast<int>(s * 360.0f / steps_per_rev + 0.5f));
        dumpSliceCSV(fname, h_fill, nx, ny, nz);
        float mass  = vof.computeTotalMass();
        float merr  = (mass - mass0) / mass0 * 100.0f;
        std::printf("  step %4d  (%3d deg)  mass_err=%.4f%%  -> %s\n",
                    s, static_cast<int>(s * 360.0f / steps_per_rev + 0.5f),
                    merr, fname);
    };

    int snap_idx = 0;
    for (int step = 0; step <= steps_per_rev; ++step) {
        if (snap_idx < 5 && step == snap_steps[snap_idx]) {
            saveSnap(step);
            ++snap_idx;
        }
        if (step < steps_per_rev) {
            vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
            cudaDeviceSynchronize();
        }
    }

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    std::printf("Done.  CSV files in %s/zalesak_plic_deg*.csv\n", out_dir);
    return 0;
}
