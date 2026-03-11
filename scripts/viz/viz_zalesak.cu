/**
 * @file viz_zalesak.cu
 * @brief Zalesak slotted-disk benchmark visualization.
 *
 * Initialises a slotted disk in solid-body rotation, runs one full 360°
 * revolution, and dumps the VOF fill level at 0°, 90°, 180°, 270°, and 360°
 * as CSV files for plotting.
 *
 * Compile (from build/):
 *   nvcc -o ../scripts/viz/viz_zalesak ../scripts/viz/viz_zalesak.cu \
 *        -I../include -L. -llbm_physics -llbm_core \
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

static void initSlottedDisk(VOFSolver& vof,
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
                // Slot: vertical cut above center
                bool in_slot = (std::fabs(dx) <= slot_w * 0.5f)
                             && (dy >= 0.0f) && (dy <= slot_d);
                int idx = i + nx*(j + ny*k);
                if (in_disk && !in_slot) {
                    float d = R - r;
                    h[idx] = 0.5f * (1.0f + std::tanh(d / 1.5f));
                }
            }
    vof.initialize(h.data());
}

// Upload solid-body rotation velocity field to device
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
    // -------------------------------------------------------------------------
    // Domain (2-D problem, thin slab in z)
    // -------------------------------------------------------------------------
    const int nx = 128, ny = 128, nz = 4;
    const float dx = 1.0f;   // dimensionless lattice units
    const int   N  = nx * ny * nz;

    // -------------------------------------------------------------------------
    // Disk geometry (scaled to 128x128; original Zalesak test: 100x100)
    // -------------------------------------------------------------------------
    const float cx = nx * 0.5f;       // 64
    const float cy = ny * 0.5f;       // 64
    const float R       = 30.0f;      // radius [cells]
    const float slot_w  = 10.0f;      // slot width [cells]
    const float slot_d  = 50.0f;      // slot depth [cells]

    // -------------------------------------------------------------------------
    // Rotation: one full revolution in 'steps_per_rev' steps
    // omega satisfies CFL: max_v = omega * R * dt/dx < 0.5
    //   omega * R < 0.5  →  omega < 0.5/30 ≈ 0.0167 rad/step
    // Use 800 steps/rev for comfortable CFL (~0.017 * 30 = 0.5)
    // -------------------------------------------------------------------------
    const int   steps_per_rev = 800;
    const float omega = 2.0f * M_PI / static_cast<float>(steps_per_rev);
    const float dt    = 1.0f;         // dimensionless

    std::printf("Zalesak Disk Visualization\n");
    std::printf("  Grid: %d x %d x %d,  R=%.0f,  omega=%.5f rad/step\n",
                nx, ny, nz, R, omega);
    std::printf("  CFL_max = %.3f\n", omega * R * dt / dx);
    std::printf("  Steps per revolution: %d\n", steps_per_rev);

    // -------------------------------------------------------------------------
    // VOF solver with TVD-MC (best for rotation benchmarks per project memory)
    // -------------------------------------------------------------------------
    VOFSolver vof(nx, ny, nz, dx);
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);
    initSlottedDisk(vof, nx, ny, nz, cx, cy, R, slot_w, slot_d);

    const float mass0 = vof.computeTotalMass();

    // Velocity device arrays (constant solid-body rotation)
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, N * sizeof(float));
    cudaMalloc(&d_uy, N * sizeof(float));
    cudaMalloc(&d_uz, N * sizeof(float));
    uploadRotationVelocity(d_ux, d_uy, d_uz, omega, cx, cy, nx, ny, nz);

    std::vector<float> h_fill(N);
    const char* out_dir = "/home/yzk/LBMProject/scripts/viz";

    // Snapshots at 0°, 90°, 180°, 270°, 360°
    const int snap_steps[] = {0, steps_per_rev/4, steps_per_rev/2,
                               3*steps_per_rev/4, steps_per_rev};

    auto saveSnap = [&](int s) {
        vof.copyFillLevelToHost(h_fill.data());
        char fname[256];
        std::snprintf(fname, sizeof(fname), "%s/zalesak_deg%03d.csv",
                      out_dir, static_cast<int>(s * 360.0f / steps_per_rev + 0.5f));
        dumpSliceCSV(fname, h_fill, nx, ny, nz);
        float mass  = vof.computeTotalMass();
        float merr  = (mass - mass0) / mass0 * 100.0f;
        std::printf("  step %4d  (%3d deg)  mass_err=%.2f%%  -> %s\n",
                    s, static_cast<int>(s * 360.0f / steps_per_rev + 0.5f),
                    merr, fname);
    };

    // ---- Time loop ---------------------------------------------------------
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

    std::printf("Done.  CSV files in %s/zalesak_deg*.csv\n", out_dir);
    return 0;
}
