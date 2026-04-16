/**
 * @file test_recoil_scale.cu
 * @brief Isolated recoil-pressure scale validation: flat 316L surface at T=3500 K
 *
 * Directly exercises ForceAccumulator::addRecoilPressureForce() without
 * MultiphysicsSolver — bypasses fl masking, force smoothing, and CFL limiting.
 * Pure unit test of the CC formula + CSF force kernel.
 *
 * Grid: 8x8x20, dx=2.5e-6 m, dt=1e-8 s
 * Interface: f=1 for k<10, f=0.5 at k=10, f=0 for k>10
 * T = 3500 K everywhere, u = 0.
 *
 * Theory (from round5/recoil_theory.md):
 *   P_sat   = 3.867e5 Pa
 *   P_recoil = 2.088e5 Pa
 *   |grad_f| at k=10 = 2.0e5 /m   (central diff: (0-1)/(2*2.5e-6))
 *   F_z at k=10  = -4.18e10 N/m^3
 *   a_LU = 2.42e-4  (safe << 0.01)
 *   Delta_u/step = 0.0605 m/s
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

int main() {
    // ------------------------------------------------------------------ //
    // 1. Grid and physical parameters
    // ------------------------------------------------------------------ //
    const int   nx = 8, ny = 8, nz = 20;
    const float dx = 2.5e-6f;   // m
    const float dt = 1.0e-8f;   // s
    const int   k_iface = 10;   // interface z-layer
    const int   n = nx * ny * nz;

    // Theory constants (from recoil_theory.md)
    const float R_gas    = 8.314f;
    const float T_surf   = 3500.0f;
    const float T_boil   = 3200.0f;
    const float L_v      = 7.45e6f;
    const float M        = 0.0558f;     // 316L molar mass
    const float P_atm    = 101325.0f;
    const float C_r      = 0.54f;
    const float rho_liq  = 6900.0f;    // 316L liquid density
    const float max_pres = 1e8f;

    printf("=================================================================\n");
    printf("  Recoil-Pressure Scale Validation  (316L, T=3500 K, flat surface)\n");
    printf("=================================================================\n");
    printf("Grid: %dx%dx%d  dx=%.2e m  dt=%.2e s\n", nx, ny, nz, (double)dx, (double)dt);
    printf("Interface at k=%d (f=0.5), liquid below, gas above\n\n", k_iface);

    // ------------------------------------------------------------------ //
    // 2. CPU-side theory computation (same formula as GPU kernel)
    // ------------------------------------------------------------------ //
    float exponent_cpu = (L_v * M / R_gas) * (1.0f / T_boil - 1.0f / T_surf);
    exponent_cpu = std::min(50.0f, std::max(-50.0f, exponent_cpu));
    float P_sat_cpu    = P_atm * expf(exponent_cpu);
    float P_recoil_cpu = std::min(C_r * P_sat_cpu, max_pres);

    // Theoretical |grad_f| at k=10 (central diff over 2 cells each side)
    // f[k=9]=1.0, f[k=10]=0.5, f[k=11]=0.0
    // grad_f_z = (0.0 - 1.0) / (2 * dx) = -2.0e5 /m
    float grad_f_theory = 1.0f / (2.0f * dx);  // |grad_f|

    float F_z_theory    = -P_recoil_cpu * grad_f_theory; // N/m^3 (into liquid = negative z)
    float a_LU_theory   = fabsf(F_z_theory) * dt * dt / (rho_liq * dx);
    float du_theory     = fabsf(F_z_theory) * dt / rho_liq;

    printf("[THEORY] CPU recompute:\n");
    printf("  exponent = %.6f\n", (double)exponent_cpu);
    printf("  P_sat    = %.4e Pa  (expect 3.867e5)\n", (double)P_sat_cpu);
    printf("  P_recoil = %.4e Pa  (expect 2.088e5)\n", (double)P_recoil_cpu);
    printf("  |grad_f| = %.4e /m (expect 2.0e5)\n",   (double)grad_f_theory);
    printf("  F_z      = %.4e N/m^3 (expect -4.18e10)\n", (double)F_z_theory);
    printf("  a_LU     = %.4e  (expect 2.42e-4)\n",    (double)a_LU_theory);
    printf("  Du/step  = %.4e m/s (expect 0.0605)\n\n",(double)du_theory);

    // ------------------------------------------------------------------ //
    // 3. Initialize VOF fill level
    // ------------------------------------------------------------------ //
    std::vector<float> h_fill(n, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                if      (k < k_iface)  h_fill[idx] = 1.0f;
                else if (k == k_iface) h_fill[idx] = 0.5f;
                else                   h_fill[idx] = 0.0f;
            }

    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL);
    vof.initialize(h_fill.data());
    vof.reconstructInterface();  // computes Youngs normals (unused by kernel, but API requires non-null)

    // ------------------------------------------------------------------ //
    // 4. Allocate and initialize temperature on device (T = 3500 K everywhere)
    // ------------------------------------------------------------------ //
    float* d_T = nullptr;
    cudaMalloc(&d_T, n * sizeof(float));
    std::vector<float> h_T(n, T_surf);
    cudaMemcpy(d_T, h_T.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------ //
    // 5. Construct ForceAccumulator and call recoil kernel exactly once
    // ------------------------------------------------------------------ //
    ForceAccumulator fa(nx, ny, nz);
    fa.reset();

    fa.addRecoilPressureForce(
        d_T,
        vof.getFillLevel(),
        vof.getInterfaceNormals(),
        /*T_boil*/         T_boil,
        /*L_v*/            L_v,
        /*M*/              M,
        /*P_atm*/          P_atm,
        /*C_r*/            C_r,
        /*smoothing_width*/2.0f,
        /*max_pressure*/   max_pres,
        nx, ny, nz, dx,
        /*force_multiplier*/1.0f);

    cudaDeviceSynchronize();

    // ------------------------------------------------------------------ //
    // 6. Copy results to host
    // ------------------------------------------------------------------ //
    std::vector<float> h_fx(n), h_fy(n), h_fz(n), h_f_out(n);
    cudaMemcpy(h_fx.data(),     fa.getFx(),           n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(),     fa.getFy(),            n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(),     fa.getFz(),            n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_out.data(), vof.getFillLevel(),     n*sizeof(float), cudaMemcpyDeviceToHost);

    // Probe at the centre of the interface layer
    int i_probe = nx / 2;
    int j_probe = ny / 2;
    int k_probe = k_iface;
    int idx_probe = i_probe + nx * (j_probe + ny * k_probe);

    float Fz    = h_fz[idx_probe];
    float Fx    = h_fx[idx_probe];
    float Fy    = h_fy[idx_probe];
    float F_mag = sqrtf(Fx*Fx + Fy*Fy + Fz*Fz);

    // Reconstruct |grad_f| exactly as the kernel does (central difference)
    int idx_zp = i_probe + nx * (j_probe + ny * (k_probe + 1));
    int idx_zm = i_probe + nx * (j_probe + ny * (k_probe - 1));
    float grad_f_z_code = (h_f_out[idx_zp] - h_f_out[idx_zm]) / (2.0f * dx);
    float grad_f_mag_code = fabsf(grad_f_z_code);

    // Back-calculate P_recoil and P_sat from the kernel's output
    float P_recoil_backcalc = (grad_f_mag_code > 1e-12f) ? F_mag / grad_f_mag_code : 0.0f;
    float P_sat_backcalc    = P_recoil_backcalc / C_r;

    // LBM stability metrics
    float a_LU_code = F_mag * dt * dt / (rho_liq * dx);
    float du_code   = F_mag * dt / rho_liq;

    // ------------------------------------------------------------------ //
    // 7. Print diagnostics in spec format
    // ------------------------------------------------------------------ //
    printf("[RECOIL_SCALE] T_surface = %.1f K\n", (double)T_surf);
    printf("[RECOIL_SCALE] P_sat (host recompute)    = %.3e Pa  (theory: 3.87e+05 Pa)\n", (double)P_sat_cpu);
    printf("[RECOIL_SCALE] P_recoil (host recompute) = %.3e Pa  (theory: 2.09e+05 Pa)\n", (double)P_recoil_cpu);
    printf("[RECOIL_SCALE] |grad f| at k=%d          = %.3e 1/m  (theory: 2.00e+05 for 2-cell step)\n", k_probe, (double)grad_f_mag_code);
    printf("[RECOIL_SCALE] F_z at k=%d               = %+.3e N/m^3  (theory: -4.18e+10, negative=into liquid)\n", k_probe, (double)Fz);
    printf("[RECOIL_SCALE] F_x,F_y at k=%d           = %+.3e, %+.3e N/m^3  (theory: ~0)\n", k_probe, (double)Fx, (double)Fy);
    printf("[RECOIL_SCALE] |F| at k=%d               = %.3e N/m^3  (theory: 4.18e+10)\n", k_probe, (double)F_mag);
    printf("[RECOIL_SCALE] a_LU = |F|*dt^2/(rho*dx)  = %.3e  (safe: <0.01; theory: ~2.4e-4)\n", (double)a_LU_code);
    printf("[RECOIL_SCALE] Delta_u over 1 step        = %.3e m/s (theory: ~0.06 m/s)\n", (double)du_code);
    printf("[RECOIL_SCALE] P_recoil_backcalc = |F|/|grad f| = %.3e Pa  (should equal CPU P_recoil)\n", (double)P_recoil_backcalc);
    printf("[RECOIL_SCALE] P_sat_backcalc = P_recoil_bc/C_r  = %.3e Pa  (should equal CPU P_sat)\n\n", (double)P_sat_backcalc);

    // ------------------------------------------------------------------ //
    // 8. Assertions (PASS/FAIL)
    // ------------------------------------------------------------------ //
    const float TOL5   = 0.05f;   // ±5%
    const float TOL2   = 0.02f;   // ±2%
    const float TOL20  = 0.20f;   // ±20% (interface-thickness dependent)

    int failures = 0;

    auto check = [&](const char* name, float value, float ref, float tol) -> bool {
        float ratio = (ref != 0.0f) ? fabsf(value - ref) / fabsf(ref) : fabsf(value);
        bool pass = (ratio <= tol);
        printf("  %-45s  %s  (got %.4e, ref %.4e, err %.1f%%)\n",
               name, pass ? "PASS" : "FAIL", (double)value, (double)ref, (double)(ratio*100.0f));
        if (!pass) ++failures;
        return pass;
    };
    auto check_lt = [&](const char* name, float value, float limit) -> bool {
        bool pass = (value < limit);
        printf("  %-45s  %s  (got %.4e, limit %.4e)\n",
               name, pass ? "PASS" : "FAIL", (double)value, (double)limit);
        if (!pass) ++failures;
        return pass;
    };

    printf("--- Pass/Fail Assertions ---\n");

    // CPU formula vs theory
    check("P_sat_cpu vs 3.867e5 Pa (±5%)",       P_sat_cpu,    3.867e5f, TOL5);
    check("P_recoil_cpu vs 2.088e5 Pa (±5%)",    P_recoil_cpu, 2.088e5f, TOL5);

    // Kernel |grad_f|
    check("|grad_f| at k=10 vs 2.0e5 /m (±20%)", grad_f_mag_code, 2.0e5f, TOL20);

    // Kernel F_z magnitude
    check("|F_z| at k=10 vs 4.18e10 N/m^3 (±5%)", fabsf(Fz), 4.18e10f, TOL5);

    // F_z sign: must be negative (into liquid)
    {
        bool pass = (Fz < 0.0f);
        printf("  %-45s  %s  (Fz=%.4e, must be <0)\n",
               "F_z sign (into liquid = negative z)", pass ? "PASS" : "FAIL", (double)Fz);
        if (!pass) ++failures;
    }

    // F_x, F_y near zero (symmetry)
    {
        float xy_frac = (F_mag > 0.0f) ? sqrtf(Fx*Fx + Fy*Fy) / F_mag : 0.0f;
        bool pass = (xy_frac < 0.01f);
        printf("  %-45s  %s  (|Fxy|/|F|=%.4e, must be <0.01)\n",
               "Horizontal symmetry |Fxy|/|F| < 1%", pass ? "PASS" : "FAIL", (double)xy_frac);
        if (!pass) ++failures;
    }

    // LBM stability
    check_lt("a_LU < 0.01 (LBM stability threshold)", a_LU_code, 0.01f);
    check_lt("|Du|/step < 1 m/s (sanity cap)",         du_code,   1.0f);

    // Back-calculation consistency: P_recoil from |F|/|grad_f|
    check("P_recoil backcalc / P_recoil_cpu (±5%)",
          P_recoil_backcalc, P_recoil_cpu, TOL5);

    // Kernel C_r consistency: P_recoil_bc / P_sat_cpu should = C_r (±2%)
    {
        float ratio_cr = (P_sat_cpu > 0.0f) ? P_recoil_backcalc / P_sat_cpu : 0.0f;
        check("Kernel C_r = P_recoil_bc/P_sat_cpu vs 0.54 (±2%)", ratio_cr, C_r, TOL2);
    }

    // Hard-fail: pressure cap must NOT be firing
    {
        bool pass = (P_recoil_cpu < max_pres * 0.5f);
        printf("  %-45s  %s  (P_recoil=%.2e, cap=%.2e)\n",
               "P_recoil cap NOT firing (< 50% of cap)", pass ? "PASS" : "FAIL",
               (double)P_recoil_cpu, (double)max_pres);
        if (!pass) ++failures;
    }

    // ------------------------------------------------------------------ //
    // 9. Summary
    // ------------------------------------------------------------------ //
    printf("\n");
    printf("=================================================================\n");
    if (failures == 0) {
        printf("  OVERALL RESULT: PASS  (%d/12 checks)\n", 12);
    } else {
        printf("  OVERALL RESULT: FAIL  (%d assertion(s) failed)\n", failures);
    }
    printf("=================================================================\n");

    // Cleanup
    cudaFree(d_T);

    return (failures > 0) ? 1 : 0;
}
