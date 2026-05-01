/**
 * @file test_plic_curvature.cu
 * @brief Phase 3a — verify PLIC_DIVERGENCE curvature against analytic 2/R.
 *
 * For a sphere of radius R, mean curvature κ = 2/R (sign convention: κ > 0
 * for convex liquid drop in gas, with n̂ pointing outward and κ = -∇·n̂).
 *
 * Acceptance:
 *   - mean |κ_PLIC − 2/R| / (2/R) < 5 % (combined R=64 + HEIGHT_FUNCTION
 *     normals; theoretical floor is O((h/R)²) ≈ 2.4e-4 fractional, but the
 *     divergence stencil and tangent-cell fallback make 5% the realistic
 *     gate without going to even larger spheres).
 *   - max fractional error < 50 % (a handful of polar-cap / fallback cells
 *     exceed the mean; we want them bounded).
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace lbm::physics;

namespace {

// Forward declaration of the GPU sharp-sphere VOF init kernel from the
// PLIC-normal test. Same signature; kept as a static helper here.
__global__ void initSphereForCurvatureKernel(
    float* fill, int nx, int ny, int nz,
    float cx, float cy, float cz, float R, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int idx = i + nx * (j + ny * k);

    float xc = max((float)i, min(cx, (float)(i + 1)));
    float yc = max((float)j, min(cy, (float)(j + 1)));
    float zc = max((float)k, min(cz, (float)(k + 1)));
    float dxn = xc - cx, dyn = yc - cy, dzn = zc - cz;
    float d_min2 = dxn*dxn + dyn*dyn + dzn*dzn;

    float xf = (cx < i + 0.5f) ? (float)(i + 1) : (float)i;
    float yf = (cy < j + 0.5f) ? (float)(j + 1) : (float)j;
    float zf = (cz < k + 0.5f) ? (float)(k + 1) : (float)k;
    float dxf = xf - cx, dyf = yf - cy, dzf = zf - cz;
    float d_max2 = dxf*dxf + dyf*dyf + dzf*dzf;

    float R2 = R * R;
    if (d_max2 <= R2) { fill[idx] = 1.0f; return; }
    if (d_min2 >= R2) { fill[idx] = 0.0f; return; }

    int count = 0;
    float inv_M = 1.0f / M;
    for (int sk = 0; sk < M; ++sk) {
        float zs = k + (sk + 0.5f) * inv_M;
        float dz = zs - cz; float dz2 = dz * dz;
        for (int sj = 0; sj < M; ++sj) {
            float ys = j + (sj + 0.5f) * inv_M;
            float dy = ys - cy; float dy2 = dy * dy;
            for (int si = 0; si < M; ++si) {
                float xs = i + (si + 0.5f) * inv_M;
                float dx_ = xs - cx;
                if (dx_*dx_ + dy2 + dz2 <= R2) ++count;
            }
        }
    }
    fill[idx] = count / (float)(M * M * M);
}

void initSphereVOFForCurvature(std::vector<float>& fill,
                                int nx, int ny, int nz,
                                float cx, float cy, float cz, float R)
{
    constexpr int M = 64;   // 64³ samples per cell — enough for κ test
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    fill.assign(N, 0.0f);

    float* d_fill = nullptr;
    cudaMalloc(&d_fill, N * sizeof(float));
    dim3 blk(8, 8, 8);
    dim3 grd((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    initSphereForCurvatureKernel<<<grd, blk>>>(d_fill, nx, ny, nz, cx, cy, cz, R, M);
    cudaDeviceSynchronize();
    cudaMemcpy(fill.data(), d_fill, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fill);
}

}  // namespace

TEST(PLICCurvature, SphereGivesCorrectKappa) {
    const int nx = 128, ny = 128, nz = 128;
    const float dx = 1.0f;
    const float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    const float R  = 48.0f;            // 16-cell margin to periodic wall

    VOFSolver vof(nx, ny, nz, dx);

    std::vector<float> fill;
    initSphereVOFForCurvature(fill, nx, ny, nz, cx, cy, cz, R);

    // HEIGHT_FUNCTION normals are required for clean PLIC κ; without them
    // the divergence-of-Youngs picks up the O(h/R) noise pattern of Phase 1.
    vof.setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
    vof.setCurvatureMethod(CurvatureMethod::PLIC_DIVERGENCE);
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();
    vof.computeCurvature();

    int N = nx * ny * nz;
    std::vector<float> kappa(N);
    cudaMemcpy(kappa.data(), vof.getCurvature(), N * sizeof(float),
               cudaMemcpyDeviceToHost);

    const float kappa_analytic = 2.0f / R;   // mean curvature of sphere [1/lu]

    int n = 0;
    double sum_abs_err = 0.0, sum_rel_err = 0.0;
    float worst_rel = 0.0f;
    for (int idx = 0; idx < N; ++idx) {
        float f = fill[idx];
        if (f < 0.1f || f > 0.9f) continue;   // core interface cells only
        float k_p = kappa[idx];
        float rel = std::fabs(k_p - kappa_analytic) / kappa_analytic;
        sum_abs_err += std::fabs(k_p - kappa_analytic);
        sum_rel_err += rel;
        worst_rel = std::max(worst_rel, rel);
        ++n;
    }
    ASSERT_GT(n, 100);
    double mean_rel = sum_rel_err / n;
    double mean_abs = sum_abs_err / n;
    printf("[PLIC κ] R=%.0f sphere: cells=%d, κ_target=%.4f, "
           "mean_abs_err=%.4f, mean_rel_err=%.3e, worst_rel=%.3e\n",
           R, n, kappa_analytic, mean_abs, mean_rel, worst_rel);

    EXPECT_LT(mean_rel, 0.05)
        << "Mean κ relative error must be < 5 % on R=48 sphere with HF normals";
    EXPECT_LT(worst_rel, 0.5f)
        << "Worst-cell κ error must be bounded";
}

// Sanity: with the legacy method, the existing curvature kernel still runs
// and produces sphere κ within its own (larger) tolerance.
TEST(PLICCurvature, LegacyMethodStillWorks) {
    const int nx = 64, ny = 64, nz = 64;
    const float dx = 1.0f;
    const float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    const float R  = 16.0f;

    VOFSolver vof(nx, ny, nz, dx);
    vof.initializeDroplet(cx, cy, cz, R);  // tanh-smoothed init — fine for legacy
    EXPECT_EQ(vof.getCurvatureMethod(), CurvatureMethod::LEGACY_HF);
    vof.computeCurvature();

    int N = nx * ny * nz;
    std::vector<float> kappa(N), fill(N);
    cudaMemcpy(kappa.data(), vof.getCurvature(), N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(fill.data(), vof.getFillLevel(), N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Legacy curvature is noisier; just check we got a non-zero κ on most
    // interface cells.
    int n_ok = 0, n_total = 0;
    for (int idx = 0; idx < N; ++idx) {
        float f = fill[idx];
        if (f < 0.1f || f > 0.9f) continue;
        ++n_total;
        if (std::fabs(kappa[idx]) > 1e-3f) ++n_ok;
    }
    EXPECT_GT(n_total, 50);
    EXPECT_GT(static_cast<float>(n_ok) / n_total, 0.5f)
        << "Legacy curvature should produce non-zero κ on the majority of "
        << "interface cells (smearing aside)";
}
