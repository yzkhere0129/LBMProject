/**
 * @file viz_zalesak_compare.cu
 * @brief Fair comparison: TVD-MC vs PLIC on same sharp-init Zalesak disk.
 */
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm::physics;

static void dumpSliceCSV(const std::string& path, const std::vector<float>& fill,
                          int nx, int ny, int nz) {
    int k0 = nz / 2;
    std::ofstream f(path);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            f << fill[i + nx * (j + ny * k0)];
            if (i < nx - 1) f << ',';
        }
        f << '\n';
    }
}

static void initSlottedDiskSharp(VOFSolver& vof, int nx, int ny, int nz,
                                  float cx, float cy, float R, float sw, float sd) {
    std::vector<float> h(nx*ny*nz, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float dx = i - cx, dy = j - cy;
                float r = std::sqrt(dx*dx + dy*dy);
                bool in = (r <= R) && !(std::fabs(dx) <= sw*0.5f && dy >= 0.0f && dy <= sd);
                if (in) h[i + nx*(j + ny*k)] = 1.0f;
            }
    vof.initialize(h.data());
}

static void uploadRotVel(float* ux, float* uy, float* uz,
                          float omega, float cx, float cy, int nx, int ny, int nz) {
    int N = nx*ny*nz;
    std::vector<float> hx(N), hy(N), hz(N, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx*(j + ny*k);
                hx[idx] = -omega * (j - cy);
                hy[idx] =  omega * (i - cx);
            }
    cudaMemcpy(ux, hx.data(), N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(uy, hy.data(), N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(uz, hz.data(), N*4, cudaMemcpyHostToDevice);
}

static void runTest(const char* name, VOFAdvectionScheme scheme, const char* prefix) {
    const int nx=128, ny=128, nz=4, N=nx*ny*nz;
    const float dx=1.0f, cx=64, cy=64, R=30, sw=10, sd=50;
    const int steps=800;
    const float omega = 2*M_PI/steps, dt=1.0f;

    VOFSolver vof(nx, ny, nz, dx);
    vof.setAdvectionScheme(scheme);
    if (scheme == VOFAdvectionScheme::TVD) vof.setTVDLimiter(TVDLimiter::MC);
    initSlottedDiskSharp(vof, nx, ny, nz, cx, cy, R, sw, sd);
    float m0 = vof.computeTotalMass();

    float *ux, *uy, *uz;
    cudaMalloc(&ux, N*4); cudaMalloc(&uy, N*4); cudaMalloc(&uz, N*4);
    uploadRotVel(ux, uy, uz, omega, cx, cy, nx, ny, nz);

    std::vector<float> h(N);
    char fname[256];
    const char* dir = "/home/yzk/LBMProject/scripts/viz";

    // Save initial
    vof.copyFillLevelToHost(h.data());
    std::snprintf(fname, 256, "%s/%s_deg000.csv", dir, prefix);
    dumpSliceCSV(fname, h, nx, ny, nz);

    for (int s = 0; s < steps; ++s) {
        vof.advectFillLevel(ux, uy, uz, dt);
        cudaDeviceSynchronize();
    }

    vof.copyFillLevelToHost(h.data());
    std::snprintf(fname, 256, "%s/%s_deg360.csv", dir, prefix);
    dumpSliceCSV(fname, h, nx, ny, nz);

    float m1 = vof.computeTotalMass();
    std::printf("  %s: mass_err = %.4f%%\n", name, (m1-m0)/m0*100);

    cudaFree(ux); cudaFree(uy); cudaFree(uz);
}

int main() {
    std::printf("Zalesak Comparison (sharp init, 128x128, 800 steps)\n");
    runTest("TVD-MC", VOFAdvectionScheme::TVD,  "cmp_tvd");
    runTest("PLIC",   VOFAdvectionScheme::PLIC, "cmp_plic");
    return 0;
}
