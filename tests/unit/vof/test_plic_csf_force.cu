/**
 * @file test_plic_csf_force.cu
 * @brief Phase 3b — verify PLIC sharp-delta CSF kernel.
 *
 * Two property tests on a stationary R=48 sphere of liquid:
 *
 *   A. Direction:
 *      The accumulated force at every interface cell is parallel to the
 *      analytic radial unit vector (within 5° angular tolerance — the
 *      bound here is set by the HF normal accuracy, NOT by the kernel
 *      under test).
 *
 *   B. Spatial concentration ("sharpness"):
 *      The force is essentially confined to the PLIC interface band.
 *      Specifically, the L¹ norm of the force outside cells with
 *      |signed distance to plane| < h_smooth_lu must be ≤ 1 % of the
 *      L¹ norm inside that band. The legacy |∇f|-smeared kernel cannot
 *      meet this gate: ∇f is non-zero across 4-5 cells of a tanh
 *      interface, so a substantial fraction of the legacy force lives
 *      outside the cosine-kernel support.
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace lbm::physics;

namespace {

// GPU sharp-sphere init (re-used pattern from test_plic_curvature.cu).
__global__ void initSphereCSFKernel(
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

void initSphereVOF(std::vector<float>& fill,
                   int nx, int ny, int nz,
                   float cx, float cy, float cz, float R)
{
    constexpr int M = 64;
    const size_t N = static_cast<size_t>(nx) * ny * nz;
    fill.assign(N, 0.0f);
    float* d_fill = nullptr;
    cudaMalloc(&d_fill, N * sizeof(float));
    dim3 blk(8, 8, 8);
    dim3 grd((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    initSphereCSFKernel<<<grd, blk>>>(d_fill, nx, ny, nz, cx, cy, cz, R, M);
    cudaDeviceSynchronize();
    cudaMemcpy(fill.data(), d_fill, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fill);
}

}  // namespace

TEST(PLICCsfForce, DirectionAndSharpness) {
    const int nx = 128, ny = 128, nz = 128;
    const float dx = 1.0f;
    const float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    const float R = 48.0f;
    const float sigma = 1.0f;          // dimensionless for the test

    VOFSolver vof(nx, ny, nz, dx);
    vof.setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
    vof.setCurvatureMethod(CurvatureMethod::PLIC_DIVERGENCE);

    std::vector<float> fill;
    initSphereVOF(fill, nx, ny, nz, cx, cy, cz, R);
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();
    vof.computeCurvature();

    ForceAccumulator forces(nx, ny, nz);
    forces.reset();
    forces.addSurfaceTensionForcePLIC(
        vof.getInterfaceGeometry(),
        vof.getCurvature(),
        sigma, dx, /*h_smooth_lu=*/1.5f);

    int N = nx * ny * nz;
    std::vector<float> fx(N), fy(N), fz(N);
    cudaMemcpy(fx.data(), forces.getFx(), N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), forces.getFy(), N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), forces.getFz(), N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ---- Direction check ---------------------------------------------------
    // Surface tension squeezes a convex liquid drop inward, so F should be
    // anti-parallel to the OUTWARD radial unit vector (= aligned with the
    // INWARD direction). Test against -r̂.
    int n_dir = 0;
    int well_aligned = 0;
    double sum_angle = 0.0;
    for (int idx = 0; idx < N; ++idx) {
        float f = fill[idx];
        if (f < 0.2f || f > 0.8f) continue;
        float fmag = std::sqrt(fx[idx]*fx[idx] + fy[idx]*fy[idx] + fz[idx]*fz[idx]);
        if (fmag < 1e-6f) continue;
        int kk = idx / (nx*ny), jj = (idx/nx)%ny, ii = idx%nx;
        float xc_ = ii + 0.5f, yc_ = jj + 0.5f, zc_ = kk + 0.5f;
        float dxr = xc_ - cx, dyr = yc_ - cy, dzr = zc_ - cz;
        float r = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);
        if (r < 1e-3f) continue;
        // Inward unit vector (-r̂)
        float inwardx = -dxr / r, inwardy = -dyr / r, inwardz = -dzr / r;
        float fhx = fx[idx] / fmag, fhy = fy[idx] / fmag, fhz = fz[idx] / fmag;
        float dot = inwardx*fhx + inwardy*fhy + inwardz*fhz;
        dot = std::min(1.0f, std::max(-1.0f, dot));
        float angle = std::acos(dot);
        sum_angle += angle;
        if (angle < 5.0f * M_PI / 180.0f) ++well_aligned;
        ++n_dir;
    }
    ASSERT_GT(n_dir, 1000);
    double mean_angle = sum_angle / n_dir;
    float frac_aligned = static_cast<float>(well_aligned) / n_dir;
    printf("[PLIC CSF] direction (vs INWARD): cells=%d, mean=%.3e rad, "
           "well_aligned(<5deg)=%.3f\n", n_dir, mean_angle, frac_aligned);
    EXPECT_LT(mean_angle, 0.05);
    EXPECT_GT(frac_aligned, 0.9f);

    // ---- Sharpness check ---------------------------------------------------
    // Force outside the |d|<h_smooth band must be small. We measure the
    // signed distance d using the same plicSignedDistanceFromCenter logic
    // (host re-implementation) on the cached PLIC plane.
    auto v = vof.getInterfaceGeometry();
    std::vector<float> nxh(N), nyh(N), nzh(N), alphah(N);
    cudaMemcpy(nxh.data(),    v.d_normal_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nyh.data(),    v.d_normal_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nzh.data(),    v.d_normal_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(alphah.data(), v.d_alpha,    N*sizeof(float), cudaMemcpyDeviceToHost);

    const float h_smooth = 1.5f;
    double L1_inside = 0.0, L1_outside = 0.0;
    for (int idx = 0; idx < N; ++idx) {
        float fmag = std::sqrt(fx[idx]*fx[idx] + fy[idx]*fy[idx] + fz[idx]*fz[idx]);
        if (fmag < 1e-12f) continue;
        // For cells where the PLIC plane is meaningful (interface cells), we
        // can compute d. For non-interface cells with non-zero force, count
        // them as "outside" (they should not exist in a sharp kernel).
        float f = fill[idx];
        if (f < 0.01f || f > 0.99f) {
            L1_outside += fmag;
            continue;
        }
        float n_x = nxh[idx], n_y = nyh[idx], n_z = nzh[idx];
        float a = alphah[idx];
        float d = a - 0.5f * (n_x + n_y + n_z);
        if (std::fabs(d) < h_smooth) L1_inside += fmag;
        else                          L1_outside += fmag;
    }
    double frac_outside = L1_outside / (L1_inside + L1_outside);
    printf("[PLIC CSF] sharpness: L1_in=%.3e, L1_out=%.3e, frac_out=%.3e\n",
           L1_inside, L1_outside, frac_outside);
    EXPECT_LT(frac_outside, 0.01)
        << "PLIC sharp-delta CSF should keep > 99% of its force inside |d|<h";
}
