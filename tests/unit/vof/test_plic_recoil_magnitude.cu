/**
 * @file test_plic_recoil_magnitude.cu
 * @brief Physics test: recoil force magnitude matches Clausius-Clapeyron prediction.
 *
 * Setup: R=36 sphere on 96^3 with dx=1e-6 m, uniform T=3500 K (above 316L
 * T_boil=3134 K so recoil is active everywhere on the interface).
 *
 * The Clausius-Clapeyron saturation pressure at 3500 K is:
 *
 *   P_sat = P_atm · exp[ (L_v · M / R_gas) · (1/T_boil - 1/T) ]
 *         = 101325 · exp[ (7.45e6 · 0.0556 / 8.314) · (1/3134 - 1/3500) ]
 *
 * The PLIC kernel applies:
 *   F = C_r · P_sat · n̂ · δ_h(d) / dx
 *
 * For interface cells where the signed distance d ≈ 0 (cell centre is
 * very close to the PLIC plane), δ_h(0) = 1 / h_smooth_lu = 1/1.5 in
 * lattice units, so δ_phys(0) = 1 / (1.5 · dx).
 * Therefore: |F| at d≈0 = C_r · P_sat / (1.5 · dx).
 *
 * We sample cells where |d| < 0.05 (approximately at the PLIC plane),
 * compute the mean |F|, and compare to the analytic prediction.
 *
 * Pass criterion: |mean_F_sampled / F_analytic - 1| < 10%.
 *
 * NOTE: d is signed distance in LU from cell centre to PLIC plane.
 * Interface cells often have d in [-0.5, 0.5]. Cells very close to d=0
 * are cells where the PLIC plane passes nearly through the cell centre,
 * meaning f ≈ 0.5 geometrically. We use |d| < 0.15 as the selection band
 * (generous enough to collect >30 cells, tight enough to get near-peak δ).
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

__global__ void initSphereRMKernel(
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
    initSphereRMKernel<<<grd, blk>>>(d_fill, nx, ny, nz, cx, cy, cz, R, 64);
    cudaDeviceSynchronize();
    cudaMemcpy(fill.data(), d_fill, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fill);
}

// Compute PLIC signed distance on host using the same formula as the kernel:
//   d = alpha - 0.5*(n_x + n_y + n_z)
// This is the distance from the cell centre (0.5,0.5,0.5) to the plane
// n·X = alpha in the unit-cube cell frame.
float plicSignedDist(float alpha, float nx, float ny, float nz) {
    return alpha - 0.5f * (nx + ny + nz);
}

}  // namespace

TEST(PLICRecoilMagnitude, RecoilMagnitudeMatchesClausiusClapeyron) {
    const int nx = 96, ny = 96, nz = 96;
    const float dx = 1e-6f;
    const float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    const float R = 36.0f;

    // 316L parameters
    const float T_boil  = 3134.0f;     // K
    const float L_v     = 7.45e6f;     // J/kg
    const float Mmol    = 0.0556f;     // kg/mol
    const float P_atm   = 101325.0f;   // Pa
    const float C_r     = 0.54f;
    const float P_max   = 1e9f;
    const float h_smooth = 1.5f;       // lattice units
    const float T_test  = 3500.0f;     // K, above T_boil

    // ---- Analytic recoil pressure at T_test ---
    const float R_gas = 8.314f;
    float exponent = (L_v * Mmol / R_gas) * (1.0f / T_boil - 1.0f / T_test);
    exponent = std::fmin(50.0f, std::fmax(-50.0f, exponent));
    float P_sat_analytic = P_atm * std::exp(exponent);
    float P_recoil_analytic = std::fmin(C_r * P_sat_analytic, P_max);
    // Force magnitude at d=0: delta_phys = 1/(h_smooth*dx)
    float delta_at_d0 = 1.0f / (h_smooth * dx);
    float F_analytic = P_recoil_analytic * delta_at_d0;

    printf("[PLIC Recoil Magnitude] analytic: P_sat=%.3e Pa, P_recoil=%.3e Pa, "
           "F_at_d0=%.3e N/m^3\n", P_sat_analytic, P_recoil_analytic, F_analytic);

    // ---- VOF setup --------------------------------------------------------
    VOFSolver vof(nx, ny, nz, dx);
    vof.setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
    std::vector<float> fill;
    initSphere(fill, nx, ny, nz, cx, cy, cz, R);
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();

    // ---- Uniform T = T_test on device -------------------------------------
    int N = nx * ny * nz;
    std::vector<float> T_host(N, T_test);
    float* d_T = nullptr;
    cudaMalloc(&d_T, N * sizeof(float));
    cudaMemcpy(d_T, T_host.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // ---- Run PLIC recoil kernel -------------------------------------------
    ForceAccumulator forces(nx, ny, nz);
    forces.reset();
    forces.addRecoilPressureForcePLIC(d_T, vof.getInterfaceGeometry(),
                                       T_boil, L_v, Mmol, P_atm, C_r, P_max,
                                       dx, h_smooth, /*mult=*/1.0f);

    std::vector<float> fx(N), fy(N), fz(N);
    cudaMemcpy(fx.data(), forces.getFx(), N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), forces.getFy(), N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), forces.getFz(), N*sizeof(float), cudaMemcpyDeviceToHost);

    // Retrieve alpha to compute signed distance on host
    auto v = vof.getInterfaceGeometry();
    std::vector<float> nxh(N), nyh(N), nzh(N), alphah(N);
    cudaMemcpy(nxh.data(),   v.d_normal_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nyh.data(),   v.d_normal_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nzh.data(),   v.d_normal_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(alphah.data(), v.d_alpha,   N*sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Sample cells with |d| < 0.15 LU (close to the interface plane) --
    const float d_threshold = 0.15f;
    double sum_fmag = 0.0;
    int n_sampled = 0;
    for (int idx = 0; idx < N; ++idx) {
        float f = fill[idx];
        if (f < 0.05f || f > 0.95f) continue;
        float d = plicSignedDist(alphah[idx], nxh[idx], nyh[idx], nzh[idx]);
        if (std::fabs(d) > d_threshold) continue;
        float fmag = std::sqrt(fx[idx]*fx[idx] + fy[idx]*fy[idx] + fz[idx]*fz[idx]);
        if (fmag < 1.0f) continue;  // skip numerically dead cells
        sum_fmag += fmag;
        ++n_sampled;
    }

    ASSERT_GT(n_sampled, 30)
        << "Need at least 30 near-plane interface cells; reduce d_threshold or "
        << "increase sphere radius if this fails";

    double mean_F = sum_fmag / n_sampled;
    double rel_err = std::fabs(mean_F / F_analytic - 1.0);
    printf("[PLIC Recoil Magnitude] n_sampled=%d, mean_F=%.3e, F_analytic=%.3e, "
           "rel_err=%.3f\n", n_sampled, mean_F, (double)F_analytic, rel_err);

    EXPECT_LT(rel_err, 0.10)
        << "Recoil force magnitude at |d|<0.15 cells must match Clausius-Clapeyron "
        << "prediction within 10%";

    cudaFree(d_T);
}
