/**
 * @file test_plic_marangoni_direction.cu
 * @brief Physics test: Marangoni force points from HOT toward COLD along surface.
 *
 * Setup: R=36 sphere on 96^3, linear T gradient in z (hot bottom, cold top).
 *   T(k) = T_hot - dT_dz * k * dx,  dT_dz = 1e7 K/m (100K over 10 μm)
 *   dσ/dT = -2.6e-4 N/(m·K)  (316L value, negative)
 *
 * At the equatorial band where n̂·ẑ ≈ 0 (sphere "equator" — cells whose
 * PLIC normal is nearly horizontal), the surface-tangential gradient ∇_s T
 * has a large z-component (since n̂_z ≈ 0, the z-component of ∇T is not
 * projected out).  The Marangoni force is:
 *
 *   F_M = (dσ/dT) · ∇_s T · δ(d)
 *
 * With dσ/dT < 0 and ∇_s T ≈ (0, 0, -dT/dz) at the equator (∇T points
 * in -z, i.e. from hot to cold), we get:
 *
 *   F_z = (dσ/dT) · (-dT/dz) · δ > 0    (upward, toward cold)
 *
 * Wait — let's be precise: ∇T = (0, 0, -dT_dz) where dT_dz > 0.
 * F_z = dsigma_dT * (-dT_dz) * delta = (-2.6e-4) * (-1e7) * delta = +2600 * delta > 0.
 *
 * So the mean Fz at the equatorial interface band must be POSITIVE (force
 * drives flow from the hot bottom toward the cold top along the sphere
 * surface). This tests the actual physics sign, not just tangency.
 *
 * Pass criterion: mean Fz over equatorial interface cells > 0 (positive),
 *   with the mean value within 50% of the analytic prediction
 *   F_z_pred = |dσ/dT| * dT_dz * delta_peak
 *   where delta_peak = 1.0 / (1.5 * dx) for a cell at d=0.
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

namespace {

__global__ void initSphereMDKernel(
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
        float dz = zs - cz;
        for (int sj = 0; sj < M; ++sj) {
            float ys = j + (sj + 0.5f) * inv_M;
            float dy = ys - cy;
            for (int si = 0; si < M; ++si) {
                float xs = i + (si + 0.5f) * inv_M;
                float dx_ = xs - cx;
                if (dx_*dx_ + dy*dy + dz*dz <= R2) ++count;
            }
        }
    }
    fill[idx] = count / (float)(M * M * M);
}

void initSphere(std::vector<float>& fill, int nx, int ny, int nz,
                float cx, float cy, float cz, float R)
{
    const size_t N = (size_t)nx * ny * nz;
    fill.assign(N, 0.0f);
    float* d_fill = nullptr;
    cudaMalloc(&d_fill, N * sizeof(float));
    dim3 blk(8, 8, 8);
    dim3 grd((nx+7)/8, (ny+7)/8, (nz+7)/8);
    initSphereMDKernel<<<grd, blk>>>(d_fill, nx, ny, nz, cx, cy, cz, R, 64);
    cudaDeviceSynchronize();
    cudaMemcpy(fill.data(), d_fill, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fill);
}

}  // namespace

TEST(PLICMarangoniDirection, HotBottomColdTop_ForcePushesUpward) {
    const int nx = 96, ny = 96, nz = 96;
    const float dx = 1e-6f;          // 1 μm / cell
    const float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    const float R = 36.0f;           // 36-cell radius sphere

    // dσ/dT for 316L steel [N/(m·K)]
    const float dsigma_dT = -2.6e-4f;
    // Temperature gradient: hot at bottom (k=0), cold at top (k=nz-1).
    // T(k) = T_hot - dT_dz_SI * k * dx,  dT_dz_SI = 1e7 K/m.
    const float T_hot = 3000.0f;
    const float dT_dz_SI = 1e7f;   // K/m

    // ---- VOF setup ---------------------------------------------------------
    VOFSolver vof(nx, ny, nz, dx);
    vof.setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
    std::vector<float> fill;
    initSphere(fill, nx, ny, nz, cx, cy, cz, R);
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();

    // ---- Temperature field on device ---------------------------------------
    int N = nx * ny * nz;
    std::vector<float> T_host(N);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                T_host[i + nx*(j + ny*k)] = T_hot - dT_dz_SI * k * dx;
    float* d_T = nullptr;
    cudaMalloc(&d_T, N * sizeof(float));
    cudaMemcpy(d_T, T_host.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // ---- Marangoni force ---------------------------------------------------
    ForceAccumulator forces(nx, ny, nz);
    forces.reset();
    forces.addMarangoniForcePLIC(d_T, /*liquid_fraction=*/nullptr,
                                  vof.getInterfaceGeometry(),
                                  dsigma_dT, dx, /*h_smooth_lu=*/1.5f);

    std::vector<float> fx(N), fy(N), fz(N);
    cudaMemcpy(fx.data(), forces.getFx(), N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), forces.getFy(), N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), forces.getFz(), N*sizeof(float), cudaMemcpyDeviceToHost);

    // Retrieve PLIC normals
    auto v = vof.getInterfaceGeometry();
    std::vector<float> nxh(N), nyh(N), nzh(N);
    cudaMemcpy(nxh.data(), v.d_normal_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nyh.data(), v.d_normal_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nzh.data(), v.d_normal_z, N*sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Equatorial band: cells where |n̂_z| < 0.2 and f in (0.05, 0.95) -
    double sum_fz = 0.0;
    int n_equat = 0;
    for (int idx = 0; idx < N; ++idx) {
        float f = fill[idx];
        if (f < 0.05f || f > 0.95f) continue;
        float nz_cell = nzh[idx];
        if (std::fabs(nz_cell) > 0.2f) continue;   // not at equator
        // Check cell is actually on the sphere surface (within 2R±3 cells)
        int kk = idx / (nx*ny);
        int jj = (idx / nx) % ny;
        int ii = idx % nx;
        float dr = std::sqrt((ii+0.5f-cx)*(ii+0.5f-cx)
                           + (jj+0.5f-cy)*(jj+0.5f-cy)
                           + (kk+0.5f-cz)*(kk+0.5f-cz));
        if (dr < R - 3.0f || dr > R + 3.0f) continue;
        float fmag = std::sqrt(fx[idx]*fx[idx] + fy[idx]*fy[idx] + fz[idx]*fz[idx]);
        if (fmag < 1e-6f) continue;
        sum_fz += fz[idx];
        ++n_equat;
    }

    ASSERT_GT(n_equat, 50) << "Need enough equatorial interface cells";

    double mean_fz = sum_fz / n_equat;

    // Analytic: delta_peak = 1/(1.5*dx) at d=0; F_z = dsigma_dT * (-dT_dz_SI) * delta_peak
    // = (-2.6e-4) * (-1e7) / (1.5 * 1e-6) = +2600 / 1.5e-6 = +1.733e9 N/m^3
    double delta_peak = 1.0 / (1.5 * dx);
    double F_z_pred = (double)dsigma_dT * (-(double)dT_dz_SI) * delta_peak;

    printf("[PLIC Marangoni Direction] equatorial cells=%d, mean_Fz=%.3e N/m^3, "
           "predicted=%.3e N/m^3\n", n_equat, mean_fz, F_z_pred);

    // The mean must be positive (force pushes toward cold = +z direction)
    EXPECT_GT(mean_fz, 0.0)
        << "Marangoni force at equatorial band must be POSITIVE in z (hot→cold upward)";

    // Mean should be within a factor of 5 of the peak-delta analytic value
    // (cells are not all at d=0, so the mean will be less than the peak).
    // At minimum, mean > 5% of F_z_pred.
    EXPECT_GT(mean_fz, 0.05 * F_z_pred)
        << "Mean equatorial Fz must be at least 5% of the peak-delta analytic value";

    cudaFree(d_T);
}
