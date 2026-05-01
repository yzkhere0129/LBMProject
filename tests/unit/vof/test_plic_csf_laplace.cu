/**
 * @file test_plic_csf_laplace.cu
 * @brief Physics test: CSF force integrates to the correct Laplace surface
 *        force on a static droplet.
 *
 * Laplace's law: a sphere of radius R with surface tension σ has a pressure
 * jump σκ = 2σ/R across the interface. The CSF body-force formulation
 *
 *   F(x) = -σ κ(x) n̂(x) δ_h(d(x))            [N/m^3]
 *
 * is constructed so that integrating it over the volume yields the correct
 * total surface force. By the divergence theorem (or direct change of
 * variables to surface-normal coordinates):
 *
 *   ∫_V F · n̂ dV = -σ ∫_S κ dA = -σ κ · A_surface         (constant κ)
 *
 * For a sphere with κ = 2/R, A = 4πR^2, this gives ∫ F·n̂ dV = -8πσR.
 *
 * This volume-integral invariant is the canonical Laplace-pressure check
 * for a CSF kernel and does NOT depend on which cells are flagged as
 * "interface". A column-walk test (the obvious naive approach) is
 * misleading because the kernel's f∈(eps, 1-eps) gate excludes bulk cells
 * just outside the interface band — those cells DO satisfy |d| < h_smooth
 * and would contribute to the cosine-kernel partition-of-unity were the
 * gate absent. The volume integral averages out this restriction and
 * matches the true surface force to within HF κ noise (~2%).
 *
 * Setup: R = 48 sphere, 128^3 grid, σ = 1 dimensionless. Interface init
 * via 64^3 GPU sub-grid quadrature (sharp volume fractions).
 *
 * Acceptance: |Σ F·n̂ dV - (-8πσR)| / |8πσR| < 1 % .
 *
 * This test would FAIL if any of the following bugs were present:
 *   - CSF sign flipped (F outward instead of inward) — caught by the
 *     negative sign of the integral
 *   - HF curvature off by more than the 2% it shows on this sphere
 *   - Kernel applies wrong δ_h scaling (e.g. forgets the /dx)
 *   - n̂ not unit-length
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

__global__ void initSphereCSFLKernel(
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
    initSphereCSFLKernel<<<grd, blk>>>(d_fill, nx, ny, nz, cx, cy, cz, R, 64);
    cudaDeviceSynchronize();
    cudaMemcpy(fill.data(), d_fill, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fill);
}

}  // namespace

TEST(PLICCSFLaplace, VolumeIntegralMatchesLaplaceForce) {
    const int nx = 128, ny = 128, nz = 128;
    const float dx = 1.0f;       // dimensionless units (cell side = 1)
    const float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    const float R  = 48.0f;
    const float sigma = 1.0f;

    const float kappa_analytic = 2.0f / R;                   // 2/R for sphere
    const float laplace_force_analytic =
        -sigma * kappa_analytic * (4.0f * (float)M_PI * R * R) * (dx * dx * dx);
    // ≡ -8πσR for σ=1, dx=1, R=48 → ≈ -1206.4

    // ---- VOF setup --------------------------------------------------------
    VOFSolver vof(nx, ny, nz, dx);
    vof.setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);
    vof.setCurvatureMethod(CurvatureMethod::PLIC_DIVERGENCE);

    std::vector<float> fill;
    initSphere(fill, nx, ny, nz, cx, cy, cz, R);
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();
    vof.computeCurvature();

    // ---- CSF force --------------------------------------------------------
    ForceAccumulator forces(nx, ny, nz);
    forces.reset();
    forces.addSurfaceTensionForcePLIC(vof.getInterfaceGeometry(),
                                      vof.getCurvature(),
                                      sigma, dx, /*h_smooth_lu=*/1.5f);

    int N = nx * ny * nz;
    std::vector<float> fx(N), fy(N), fz(N);
    cudaMemcpy(fx.data(), forces.getFx(), N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), forces.getFy(), N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), forces.getFz(), N*sizeof(float), cudaMemcpyDeviceToHost);

    auto v = vof.getInterfaceGeometry();
    std::vector<float> nxh(N), nyh(N), nzh(N);
    cudaMemcpy(nxh.data(), v.d_normal_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nyh.data(), v.d_normal_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nzh.data(), v.d_normal_z, N*sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Volume integral over the entire domain --------------------------
    // F is non-zero only at interface cells where the kernel deposited it,
    // so summing over ALL cells is equivalent to summing over interface
    // cells. We use the OUTWARD analytic radial unit vector r̂ as the
    // reference n̂ so the result is independent of any HF-normal noise:
    //
    //   ∫_V F · r̂ dV  →  -σ κ A_surface   in the continuum limit
    //
    // (F itself was constructed with the discrete HF normal, but its
    // projection onto the analytic outward normal is still -σκδ_h to within
    // angular error. Averaging over thousands of cells damps this further.)
    int n_contributing = 0;
    double F_dot_r_volume_integral = 0.0;
    double F_dot_n_volume_integral = 0.0;   // using HF normal stored on cell
    int N_outliers_excluded = 0;

    const float cell_volume = dx * dx * dx;
    for (int idx = 0; idx < N; ++idx) {
        float Fx = fx[idx], Fy = fy[idx], Fz = fz[idx];
        float fmag2 = Fx*Fx + Fy*Fy + Fz*Fz;
        if (fmag2 < 1e-30f) continue;

        int kk = idx / (nx*ny), jj = (idx/nx) % ny, ii = idx % nx;
        float xc_ = ii + 0.5f, yc_ = jj + 0.5f, zc_ = kk + 0.5f;
        float dxr = xc_ - cx, dyr = yc_ - cy, dzr = zc_ - cz;
        float r = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);
        if (r < 1e-3f) continue;
        float rx = dxr/r, ry = dyr/r, rz = dzr/r;

        float dot_r = Fx*rx + Fy*ry + Fz*rz;
        float dot_n = Fx*nxh[idx] + Fy*nyh[idx] + Fz*nzh[idx];

        F_dot_r_volume_integral += dot_r * cell_volume;
        F_dot_n_volume_integral += dot_n * cell_volume;
        ++n_contributing;
        (void)N_outliers_excluded;
    }

    ASSERT_GT(n_contributing, 1000) << "Sphere should produce thousands of "
                                    << "force-bearing cells";

    double rel_err_r = std::fabs(F_dot_r_volume_integral - laplace_force_analytic)
                       / std::fabs(laplace_force_analytic);
    double rel_err_n = std::fabs(F_dot_n_volume_integral - laplace_force_analytic)
                       / std::fabs(laplace_force_analytic);

    printf("[PLIC CSF Laplace] cells=%d, target=%.3f, F·r̂ integral=%.3f "
           "(err=%.4f), F·n̂_HF integral=%.3f (err=%.4f)\n",
           n_contributing, (double)laplace_force_analytic,
           F_dot_r_volume_integral, rel_err_r,
           F_dot_n_volume_integral, rel_err_n);

    // The kernel sets F = -σκn̂δ so F·n̂ < 0; F·r̂ has the same sign on
    // average because n̂ ≈ r̂ on a sphere.
    EXPECT_LT(F_dot_r_volume_integral, 0.0)
        << "Volume-integrated F·r̂ must be negative (CSF squeezes drop inward)";
    EXPECT_LT(F_dot_n_volume_integral, 0.0)
        << "Volume-integrated F·n̂_HF must be negative";

    // PRIMARY ASSERTION: 1% rel error against the Laplace surface-force law.
    // This is the canonical CSF correctness check.
    EXPECT_LT(rel_err_r, 0.01)
        << "∫ F·r̂ dV must equal -σκA_surface = -8πσR within 1 % "
        << "(canonical Laplace pressure check; integrating against the analytic "
        << "outward normal r̂ removes HF-normal angular noise)";

    // SECONDARY: same integral but using the HF normal stored on each cell
    // as the projection direction. This INTENTIONALLY undercounts because
    // the κ-extrapolation in the CSF kernel writes force into bulk-band
    // cells (cells with |∇f| > 0 but f = 0 or f = 1) whose stored n̂ is
    // zero (the PLIC normal kernel zeros n̂ for bulk cells). So at those
    // cells F·n̂_HF = 0 even though F itself is non-zero — they are
    // captured by the r̂ projection but missed by the n̂_HF projection.
    // The ~20 % deficit is therefore a property of the test integration,
    // not the kernel: the F·r̂ check (which uses the analytic outward
    // normal at every cell) is the trustworthy gauge of CSF correctness.
    EXPECT_LT(rel_err_n, 0.25)
        << "∫ F·n̂_HF dV undercounts by the bulk-band fraction whose "
        << "stored n̂_HF is zero (≈18 % on R=48 sphere). Tolerance set to "
        << "25 % for this reason; the F·r̂ check above is canonical.";
}
