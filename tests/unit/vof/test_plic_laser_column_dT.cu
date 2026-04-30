/**
 * @file test_plic_laser_column_dT.cu
 * @brief Physics test: laser column-march formula gives the correct ΔT per step.
 *
 * Rationale: the diag_plic_smoke diagnostic showed a T_peak reduction compared
 * to a reference.  This test pins the isolated formula chain:
 *
 *   Q_vol = I_xy * absorptivity / (dx * max(f, f_min))    [W/m³]
 *   ΔT    = Q_vol * dt / (ρ * cp)                         [K]
 *
 * by directly evaluating Q_vol from the PLIC column-march kernel and then
 * computing the implied ΔT. If this test passes, the formula and the kernel
 * are consistent; any T_peak discrepancy must come from conduction / diffusion
 * in subsequent steps, not from a bug in the deposition kernel itself.
 *
 * Setup: 1×1×N column, single interface cell at k=k_top with f=0.4.
 *   - nx=4, ny=4, nz=24, dx=2μm, dt=10ns
 *   - Laser centred on (2*dx, 2*dx) with P=200W, w0=40μm, absorptivity=0.35
 *   - 316L parameters: ρ=7900 kg/m³, cp=500 J/(kg·K)
 *
 * Analytic ΔT:
 *   I_xy  = (2P / (π w0²)) * exp(-2 r² / w0²)   at the column centre (r small)
 *   q_abs = I_xy * absorptivity                  [W/m²] — already absorbed power density
 *   Q_vol = q_abs / (dx * f)                     [W/m³]
 *   ΔT    = Q_vol * dt / (ρ * cp)               [K]
 *
 * Assert: |ΔT_kernel / ΔT_analytic - 1| < 1%.
 *
 * If this test FAILS, there is a bug in the column-march formula itself
 * (wrong f normalization, wrong I_xy evaluation, or wrong dx factor).
 * If it PASSES, the deposition physics are correct and T_peak reduction
 * comes from other sources (conduction, BC, etc.).
 */

#include <gtest/gtest.h>
#include "physics/laser_source.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace lbm {
namespace physics {
// Forward declaration of the PLIC laser kernel (defined in laser_source.cu).
__global__ void computeLaserHeatSourcePLICColumnKernel(
    float* d_heat_source,
    const float* fill_level,
    LaserSource laser,
    int nx, int ny, int nz,
    float dx,
    float f_min_deposit);
}  // namespace physics
}  // namespace lbm

using namespace lbm::physics;

TEST(PLICLaserColumnDT, DeltaTMatchesAnalytic) {
    // ---- Problem parameters -----------------------------------------------
    const int nx = 4, ny = 4, nz = 24;
    const float dx  = 2e-6f;       // 2 μm / cell
    const float dt  = 10e-9f;      // 10 ns time step
    const float rho = 7900.0f;     // 316L density [kg/m³]
    const float cp  = 500.0f;      // 316L solid Cp [J/(kg·K)]
    const float P   = 200.0f;      // laser power [W]
    const float w0  = 40e-6f;      // beam radius [m]
    const float absorptivity = 0.35f;
    const float penetration_depth = 10e-6f;
    const float f_interface = 0.4f;
    const int   k_interface = 16;
    const float f_min = 0.05f;

    // ---- Build LaserSource -------------------------------------------------
    LaserSource laser(P, w0, absorptivity, penetration_depth);
    // Centre the beam at the middle of the (i=2, j=2) cell column centre.
    float x_c = (nx * 0.5f) * dx;   // column centre in physical coords
    float y_c = (ny * 0.5f) * dx;
    laser.setPosition(x_c, y_c, 0.0f);
    laser.vx = 0.0f;
    laser.vy = 0.0f;

    // ---- Fill level: interface cell at k=k_interface, bulk below, gas above -
    int N = nx * ny * nz;
    std::vector<float> fill_h(N, 0.0f);
    for (int k = 0; k < nz; ++k) {
        float fv;
        if      (k < k_interface)  fv = 1.0f;
        else if (k == k_interface) fv = f_interface;
        else                       fv = 0.0f;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                fill_h[i + nx*(j + ny*k)] = fv;
    }

    // ---- Run the PLIC column-march kernel ----------------------------------
    float *d_fill = nullptr, *d_heat = nullptr;
    cudaMalloc(&d_fill, N * sizeof(float));
    cudaMalloc(&d_heat, N * sizeof(float));
    cudaMemcpy(d_fill, fill_h.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_heat, 0, N * sizeof(float));

    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);
    computeLaserHeatSourcePLICColumnKernel<<<blocks, threads>>>(
        d_heat, d_fill, laser, nx, ny, nz, dx, f_min);
    cudaDeviceSynchronize();

    std::vector<float> heat_h(N, 0.0f);
    cudaMemcpy(heat_h.data(), d_heat, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Probe the central column (i=nx/2, j=ny/2) -------------------------
    int i_c = nx / 2, j_c = ny / 2;
    int idx_iface = i_c + nx * (j_c + ny * k_interface);
    float Q_vol_kernel = heat_h[idx_iface];    // [W/m³] from kernel

    // ---- Analytic prediction -----------------------------------------------
    // The kernel evaluates intensity at the cell centre (i+0.5)*dx.
    float xc_phys = (i_c + 0.5f) * dx;
    float yc_phys = (j_c + 0.5f) * dx;
    float I_xy    = laser.computeIntensity(xc_phys, yc_phys) * absorptivity;
    float Q_vol_analytic = I_xy / (dx * f_interface);

    // The kernel clamps f at f_min, but f_interface=0.4 > f_min=0.05, so no clamping.
    float ΔT_kernel   = Q_vol_kernel   * dt / (rho * cp);
    float ΔT_analytic = Q_vol_analytic * dt / (rho * cp);

    float rel_err = std::fabs(ΔT_kernel / ΔT_analytic - 1.0f);

    printf("[PLIC Laser dT] Q_vol_kernel=%.4e W/m³, Q_vol_analytic=%.4e W/m³\n",
           Q_vol_kernel, Q_vol_analytic);
    printf("[PLIC Laser dT] ΔT_kernel=%.4e K, ΔT_analytic=%.4e K, rel_err=%.4f\n",
           ΔT_kernel, ΔT_analytic, rel_err);

    ASSERT_GT(Q_vol_kernel, 0.0f) << "Interface cell must receive non-zero heat";
    EXPECT_LT(rel_err, 0.01f)
        << "ΔT from PLIC laser kernel must match analytic formula within 1%";

    // Verify: cells strictly below the interface get zero deposition.
    for (int k = 0; k < k_interface; ++k) {
        int idx = i_c + nx * (j_c + ny * k);
        EXPECT_FLOAT_EQ(heat_h[idx], 0.0f)
            << "Sub-interface cell k=" << k << " must get zero direct heating";
    }
    // Verify: gas cells strictly above the interface get zero.
    for (int k = k_interface + 1; k < nz; ++k) {
        int idx = i_c + nx * (j_c + ny * k);
        EXPECT_FLOAT_EQ(heat_h[idx], 0.0f)
            << "Gas cell k=" << k << " must get zero heating";
    }

    cudaFree(d_fill);
    cudaFree(d_heat);
}
