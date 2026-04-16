/**
 * @file test_marangoni_scale.cu
 * @brief Isolated Marangoni force scale test — no fluid/thermal/laser solver.
 *
 * Instantiates ForceAccumulator directly and verifies that the kernel
 * produces the correct physical force at the interface cell.
 *
 * Analytical setup:
 *   T(x) = 1000 + (dT/dx) * i * dx,   dT/dx = 1.0e7 K/m
 *   f = 1.0 for k < nz/2, 0.5 at k = nz/2, 0.0 for k > nz/2
 *   n = (0, 0, -1) at interface layer (liquid below, gas above)
 *
 * Expected at the interface cell (nx/2, ny/2, nz/2):
 *   |∇f|  = |(f[k+1]-f[k-1])/(2dx)| = |(0-1)/(2*2.5e-6)| = 2.0e5 m^-1
 *   ∇T    = (1e7, 0, 0) K/m
 *   n     = (0, 0, -1)  [unit, from liquid to gas upward]
 *   ∇_sT  = (I - nnᵀ)∇T = (1e7, 0, 0) K/m  [z-component zeroed]
 *   F_x   = dσ/dT × ∇_sT_x × |∇f| = -4.3e-4 × 1e7 × 2e5 = -8.6e8 N/m³
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "physics/force_accumulator.h"

using namespace lbm::physics;

int main() {
    // ---- Grid ---------------------------------------------------------------
    const int nx = 32, ny = 8, nz = 16;
    const int num_cells = nx * ny * nz;
    const int k_iface = nz / 2;   // = 8

    // ---- Physical parameters ------------------------------------------------
    const float dx        = 2.5e-6f;   // [m]
    const float dt        = 1.0e-9f;   // [s]  — only used for LU conversion check
    const float rho       = 7900.0f;   // [kg/m³] 316L
    const float dsigma_dT = -4.3e-4f;  // [N/(m·K)]
    const float dT_dx     = 1.0e7f;    // [K/m]  temperature gradient

    printf("==========================================================\n");
    printf("  Marangoni Force Scale Test — Isolated Kernel Verification\n");
    printf("==========================================================\n");
    printf("Grid: %dx%dx%d, dx=%.2f µm, dt=%.1e s\n", nx, ny, nz, dx*1e6f, dt);
    printf("Interface at k=%d (sharp: f=1 below, f=0 above, f=0.5 at k=%d)\n",
           k_iface, k_iface);
    printf("dT/dx   = %.3e K/m  (linear in x)\n", dT_dx);
    printf("dσ/dT   = %.3e N/(m·K)\n", dsigma_dT);
    printf("rho     = %.1f kg/m³\n", rho);

    // ---- Build host fields --------------------------------------------------

    // Temperature: T(i,j,k) = 1000 + dT_dx * i * dx
    std::vector<float> h_T(num_cells);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int idx = i + nx * (j + ny * k);
                h_T[idx] = 1000.0f + dT_dx * i * dx;
            }

    // Fill level: sharp interface at k = k_iface
    std::vector<float> h_f(num_cells, 0.0f);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int idx = i + nx * (j + ny * k);
                if (k < k_iface)       h_f[idx] = 1.0f;
                else if (k == k_iface) h_f[idx] = 0.5f;
                else                   h_f[idx] = 0.0f;
            }

    // Normals: (0, 0, -1) at interface layer; zero elsewhere.
    // Convention in kernel: n points from liquid to gas.
    // Liquid is below (k < k_iface), gas above → normal is +z direction.
    // However the kernel only uses the normal to project ∇T tangentially:
    //   ∇_sT = ∇T - (∇T·n)·n
    // For n=(0,0,1): ∇_sT_x = ∇T_x, ∇_sT_z = 0.  Sign of n does not
    // affect the tangential projection magnitude, only z zeroing matters.
    // We use (0,0,-1) as specified; both give |∇_sT_x| = |∇T_x|.
    //
    // IMPORTANT: The kernel gates on n_mag > 0.01, so cells not at the
    // interface layer must have |n| = 0 (or < 0.01) to be skipped.
    std::vector<float3> h_normals(num_cells, make_float3(0.0f, 0.0f, 0.0f));
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            int idx = i + nx * (j + ny * k_iface);
            h_normals[idx] = make_float3(0.0f, 0.0f, -1.0f);
        }

    // ---- Device allocations -------------------------------------------------
    float*  d_T       = nullptr;
    float*  d_f       = nullptr;
    float3* d_normals = nullptr;

    cudaMalloc(&d_T,       num_cells * sizeof(float));
    cudaMalloc(&d_f,       num_cells * sizeof(float));
    cudaMalloc(&d_normals, num_cells * sizeof(float3));

    cudaMemcpy(d_T,       h_T.data(),       num_cells * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,       h_f.data(),       num_cells * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals.data(), num_cells * sizeof(float3), cudaMemcpyHostToDevice);

    // ---- Run ForceAccumulator -----------------------------------------------
    ForceAccumulator fa(nx, ny, nz);
    fa.reset();
    fa.addMarangoniForce(d_T, d_f, nullptr /* no liquid_fraction */,
                         d_normals, dsigma_dT,
                         nx, ny, nz, dx, 1.0f);

    // ---- Extract F_phys (before LU conversion) ------------------------------
    std::vector<float> h_fx_phys(num_cells), h_fy_phys(num_cells), h_fz_phys(num_cells);
    cudaMemcpy(h_fx_phys.data(), fa.getFx(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy_phys.data(), fa.getFy(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz_phys.data(), fa.getFz(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Probe center cell at interface
    const int i_probe = nx / 2;   // = 16
    const int j_probe = ny / 2;   // = 4
    const int k_probe = k_iface;  // = 8
    const int idx_probe = i_probe + nx * (j_probe + ny * k_probe);

    float Fx_phys = h_fx_phys[idx_probe];
    float Fy_phys = h_fy_phys[idx_probe];
    float Fz_phys = h_fz_phys[idx_probe];

    // Analytical reference
    // |∇f| = 1/(2*dx) = 2e5 m^-1
    // ∇_sT_x = dT_dx (n has no x-component, so no projection onto x)
    // F_x = dσ/dT * dT_dx * |∇f|
    const float grad_f_mag_expected  = 1.0f / (2.0f * dx);   // 2.0e5 m^-1
    const float grad_T_s_x_expected  = dT_dx;                 // 1.0e7 K/m
    const float Fx_phys_expected = dsigma_dT * grad_T_s_x_expected * grad_f_mag_expected;
    // = -4.3e-4 * 1e7 * 2e5 = -8.6e8 N/m³

    printf("\n=== Physical Force at probe cell (%d, %d, %d) ===\n",
           i_probe, j_probe, k_probe);
    printf("  Fx_phys          = %+.6e N/m³\n", Fx_phys);
    printf("  Fy_phys          = %+.6e N/m³\n", Fy_phys);
    printf("  Fz_phys          = %+.6e N/m³\n", Fz_phys);
    printf("  Fx_phys_expected = %+.6e N/m³\n", Fx_phys_expected);

    float ratio_phys = (fabsf(Fx_phys_expected) > 1e-30f)
                       ? Fx_phys / Fx_phys_expected
                       : 0.0f;
    printf("  Ratio Fx_phys / expected = %.6f\n", ratio_phys);

    bool pass_phys = fabsf(ratio_phys - 1.0f) < 0.01f;
    printf("  Physical force test: %s (|ratio-1| = %.4f, tolerance 1%%)\n",
           pass_phys ? "PASS" : "FAIL", fabsf(ratio_phys - 1.0f));

    // ---- Scan all interface cells to confirm uniformity ---------------------
    printf("\n=== Interface layer (k=%d) scan ===\n", k_iface);
    float Fx_min =  1e30f, Fx_max = -1e30f;
    int active_cells = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = i + nx * (j + ny * k_iface);
            float fx = h_fx_phys[idx];
            if (fabsf(fx) > 1.0f) {
                active_cells++;
                Fx_min = fminf(Fx_min, fx);
                Fx_max = fmaxf(Fx_max, fx);
            }
        }
    }
    printf("  Active cells (|Fx|>1): %d / %d\n", active_cells, nx * ny);
    if (active_cells > 0) {
        printf("  Fx range: [%.4e, %.4e] N/m³\n", Fx_min, Fx_max);
    }

    // ---- Verify no force in bulk regions ------------------------------------
    int bulk_nonzero = 0;
    for (int k = 0; k < nz; k++) {
        if (k == k_iface) continue;
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int idx = i + nx * (j + ny * k);
                if (fabsf(h_fx_phys[idx]) > 1.0f || fabsf(h_fz_phys[idx]) > 1.0f)
                    bulk_nonzero++;
            }
    }
    printf("  Bulk cells with |F|>1 N/m³: %d  (should be 0)\n", bulk_nonzero);

    // ---- Convert to lattice units -------------------------------------------
    fa.convertToLatticeUnits(dx, dt, rho);

    std::vector<float> h_fx_lu(num_cells), h_fy_lu(num_cells), h_fz_lu(num_cells);
    cudaMemcpy(h_fx_lu.data(), fa.getFx(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy_lu.data(), fa.getFy(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz_lu.data(), fa.getFz(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float Fx_LU = h_fx_lu[idx_probe];

    // LU conversion: F_LU = F_phys * dt^2 / (dx * rho)
    float conv  = dt * dt / (dx * rho);
    float Fx_LU_expected = Fx_phys_expected * conv;

    printf("\n=== Lattice Unit Force at probe cell (%d, %d, %d) ===\n",
           i_probe, j_probe, k_probe);
    printf("  conversion factor = dt²/(dx·ρ) = %.6e\n", conv);
    printf("  Fx_LU             = %+.6e  (dimensionless)\n", Fx_LU);
    printf("  Fx_LU_expected    = %+.6e  (dimensionless)\n", Fx_LU_expected);

    float ratio_lu = (fabsf(Fx_LU_expected) > 1e-40f)
                     ? Fx_LU / Fx_LU_expected
                     : 0.0f;
    printf("  Ratio Fx_LU / expected = %.6f\n", ratio_lu);

    bool pass_lu = fabsf(ratio_lu - 1.0f) < 0.01f;
    printf("  Lattice unit test: %s (|ratio-1| = %.4f, tolerance 1%%)\n",
           pass_lu ? "PASS" : "FAIL", fabsf(ratio_lu - 1.0f));

    // ---- Velocity scale context ---------------------------------------------
    // One step: Δu_LU = F_LU / rho_LU ≈ F_LU (rho_LU = 1 in LBM)
    // Over N steps with tau damping: u_ss ~ F_LU * tau (steady-state)
    // Physical velocity: v_phys = u_LU * dx / dt
    printf("\n=== Velocity Scale Context ===\n");
    printf("  F_LU at interface  = %.4e LU\n", fabsf(Fx_LU));
    printf("  Single-step Δu_LU ≈ %.4e  (with rho_LU=1)\n", fabsf(Fx_LU));
    float v_phys_per_step = fabsf(Fx_LU) * dx / dt;
    printf("  Physical equiv     = %.4e m/s per step\n", v_phys_per_step);
    printf("  Note: steady-state u_LU << F_LU due to viscous damping\n");
    printf("        typical tau=0.6 gives damping ~(tau-0.5)·cs2 per step\n");

    // ---- Summary ------------------------------------------------------------
    printf("\n==========================================================\n");
    printf("  RESULT: Physical force %s, LU force %s\n",
           pass_phys ? "PASS" : "FAIL",
           pass_lu   ? "PASS" : "FAIL");
    if (!pass_phys || !pass_lu) {
        printf("  >> Kernel scale bug detected — see ratios above.\n");
    } else {
        printf("  >> Kernel is correct. Scale bug (if any) is downstream.\n");
    }
    printf("==========================================================\n");

    cudaFree(d_T);
    cudaFree(d_f);
    cudaFree(d_normals);

    return (pass_phys && pass_lu) ? 0 : 1;
}
