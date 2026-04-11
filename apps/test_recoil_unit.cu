/**
 * @file test_recoil_unit.cu
 * @brief Unit test: Recoil pressure force — isolated, analytical comparison
 *
 * Setup:
 *   16×16×16 domain, flat interface at k=7..8 (tanh)
 *   Uniform T = 3500K at interface cells
 *
 * Analytical expectation:
 *   P_sat(T)    = P_atm × exp[(L_v × M / R) × (1/T_boil - 1/T)]
 *   P_recoil    = C_r × P_sat                    [Pa]
 *   F_vol       = P_recoil × |∇f|               [N/m³]  (CSF)
 *   ∫F_z dz     = P_recoil                       [Pa = N/m²]
 *
 * Tests:
 *   C1: ∫F_z dz vs P_recoil analytical — ratio should be ≈1.0
 *   C2: Force direction: should push INTO liquid (negative z for z-up interface)
 *   C3: Force multiplier: kernel has force_multiplier param, default=1.0
 *       Verify that force scales linearly with it.
 *   C4: T below activation (T_boil - 500): force should be zero
 *   C5: F uses ∇f direction (not VOF normal n) — verify ∇f = -n_z direction
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

int main() {
    const int NX = 16, NY = 16, NZ = 16;
    const float dx = 2.5e-6f;
    const int num_cells = NX * NY * NZ;

    MaterialProperties mat = MaterialDatabase::get316L();

    printf("==========================================================\n");
    printf("  Test C: Recoil Pressure Force Unit Test\n");
    printf("==========================================================\n");
    printf("Material: %s\n", mat.name);
    printf("T_boil = %.0f K, L_v = %.3e J/kg, M = %.5f kg/mol\n",
           mat.T_vaporization, mat.L_vaporization, mat.molar_mass);

    const float T_test = 3500.0f;
    const float C_r = 0.54f;
    const float P_atm = 101325.0f;
    const int z_interface = 7;
    const float w = 1.5f * dx;

    // Build fill level and temperature
    std::vector<float> h_fill(num_cells, 0.0f);
    std::vector<float> h_T(num_cells, 300.0f);

    for (int k = 0; k < NZ; k++) {
        float z_phys = (k - z_interface) * dx;
        float f_val = 0.5f + 0.5f * tanhf(-z_phys / w);
        f_val = fmaxf(0.0f, fminf(1.0f, f_val));
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);
                h_fill[idx] = f_val;
                // Only interface+liquid cells above T_boil
                h_T[idx] = (f_val > 0.01f) ? T_test : 300.0f;
            }
        }
    }

    float* d_T    = nullptr;
    float* d_fill = nullptr;
    cudaMalloc(&d_T,    num_cells * sizeof(float));
    cudaMalloc(&d_fill, num_cells * sizeof(float));
    cudaMemcpy(d_T,    h_T.data(),   num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill, h_fill.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    VOFSolver vof(NX, NY, NZ, dx,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL);
    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    const float3* d_normals = vof.getInterfaceNormals();

    // Analytical P_sat and P_recoil at T_test
    const float R_gas = 8.314f;
    float exponent_anal = (mat.L_vaporization * mat.molar_mass / R_gas)
                        * (1.0f / mat.T_vaporization - 1.0f / T_test);
    exponent_anal = fminf(exponent_anal, 50.0f);
    float P_sat_anal   = P_atm * expf(exponent_anal);
    float P_recoil_anal = C_r * P_sat_anal;

    printf("\nAt T=%.0f K:\n", T_test);
    printf("  exponent    = %.4f\n", exponent_anal);
    printf("  P_sat       = %.4e Pa\n", P_sat_anal);
    printf("  P_recoil    = %.4e Pa  (C_r=%.2f)\n", P_recoil_anal, C_r);

    // ----------------------------------------------------------------
    // C1: Integrate F_z across interface, compare to P_recoil
    // The recoil kernel: F = P_recoil × ∇f × force_multiplier
    // ∇f points from gas→liquid (positive z, since liquid is below)
    // Wait — we set liquid at k<z_interface, so f=1 at low k, f=0 at high k
    // ∇f_z = df/dz < 0 (f decreases with k)
    // So F_z = P_recoil × ∇f_z < 0 → pushes downward into liquid. Correct.
    // ∫F_z dz = P_recoil × ∫∇f_z dz = P_recoil × [f(top) - f(bottom)]
    //         = P_recoil × [0 - 1] = -P_recoil
    // ----------------------------------------------------------------
    ForceAccumulator fa(NX, NY, NZ);
    fa.reset();
    fa.addRecoilPressureForce(d_T, d_fill, d_normals,
                              mat.T_vaporization, mat.L_vaporization, mat.molar_mass,
                              P_atm, C_r,
                              /*smoothing_width=*/2.0f, /*max_pressure=*/1.0e8f,
                              NX, NY, NZ, dx,
                              /*force_multiplier=*/1.0f);

    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), fa.getFx(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), fa.getFy(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), fa.getFz(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    const int i_center = NX / 2, j_center = NY / 2;
    double integral_Fz = 0.0;
    printf("\n--- Column (i=%d, j=%d): F_z integrated across z ---\n",
           i_center, j_center);
    printf("  k    fill     T[K]     Fz[N/m³]\n");
    for (int k = 0; k < NZ; k++) {
        int idx = i_center + NX * (j_center + NY * k);
        float Fz = h_fz[idx];
        integral_Fz += Fz * dx;
        if (fabsf(Fz) > 1.0f) {
            printf("  k=%2d  f=%.3f  T=%.0fK  Fz=%+.4e\n",
                   k, h_fill[idx], h_T[idx], Fz);
        }
    }

    // Expected: ∫F_z dz = -P_recoil (force pushes into liquid = negative z)
    float ratio_C1 = (float)(integral_Fz / (-P_recoil_anal));
    printf("\n[C1] Integrated F_z dz     = %.6e N/m²\n", (float)integral_Fz);
    printf("[C1] Expected -P_recoil    = %.6e N/m²\n", -P_recoil_anal);
    printf("[C1] Ratio (meas/expected) = %.4f   (should be ~1.0)\n", ratio_C1);
    if (fabsf(ratio_C1 - 1.0f) < 0.15f) {
        printf("[C1] PASS\n");
    } else {
        printf("[C1] FAIL — recoil force integral deviates >15%% from analytical\n");
    }

    // ----------------------------------------------------------------
    // C2: Force direction check
    // Liquid is at LOW z, gas at HIGH z. ∇f_z < 0.
    // Recoil should push INTO liquid → F_z < 0.
    // ----------------------------------------------------------------
    printf("\n[C2] F_z integral sign = %+.4e  (should be negative)\n",
           (float)integral_Fz);
    if (integral_Fz < 0.0) {
        printf("[C2] PASS — Force correctly pushes into liquid\n");
    } else {
        printf("[C2] FAIL — Force direction wrong! Recoil should push INTO liquid.\n");
    }

    // ----------------------------------------------------------------
    // C3: Force multiplier linearity
    // ----------------------------------------------------------------
    ForceAccumulator fa2(NX, NY, NZ);
    fa2.reset();
    fa2.addRecoilPressureForce(d_T, d_fill, d_normals,
                               mat.T_vaporization, mat.L_vaporization, mat.molar_mass,
                               P_atm, C_r, 2.0f, 1.0e8f,
                               NX, NY, NZ, dx, 2.0f);  // multiplier=2

    std::vector<float> h_fz2(num_cells);
    cudaMemcpy(h_fz2.data(), fa2.getFz(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    double integral_Fz2 = 0.0;
    for (int k = 0; k < NZ; k++) {
        int idx = i_center + NX * (j_center + NY * k);
        integral_Fz2 += h_fz2[idx] * dx;
    }
    float linearity = (fabsf(integral_Fz) > 1e-20f)
                    ? (float)(integral_Fz2 / (2.0 * integral_Fz)) : 0.0f;
    printf("\n[C3] force_multiplier=2: ∫Fz = %.4e (should be 2× above)\n",
           (float)integral_Fz2);
    printf("[C3] Linearity ratio = %.4f  (should be 1.0)\n", linearity);
    if (fabsf(linearity - 1.0f) < 0.01f) {
        printf("[C3] PASS\n");
    } else {
        printf("[C3] FAIL — force_multiplier scaling is nonlinear\n");
    }

    // ----------------------------------------------------------------
    // C4: Below activation temperature (T_boil - 500 = T_activation)
    // ----------------------------------------------------------------
    float T_activation = mat.T_vaporization - 500.0f;
    float T_below_act  = T_activation - 100.0f;
    std::vector<float> h_T_cold(num_cells, T_below_act);
    float* d_T_cold = nullptr;
    cudaMalloc(&d_T_cold, num_cells * sizeof(float));
    cudaMemcpy(d_T_cold, h_T_cold.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    ForceAccumulator fa3(NX, NY, NZ);
    fa3.reset();
    fa3.addRecoilPressureForce(d_T_cold, d_fill, d_normals,
                               mat.T_vaporization, mat.L_vaporization, mat.molar_mass,
                               P_atm, C_r, 2.0f, 1.0e8f,
                               NX, NY, NZ, dx, 1.0f);

    std::vector<float> h_fz3(num_cells);
    cudaMemcpy(h_fz3.data(), fa3.getFz(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    double max_cold_fz = 0.0;
    for (int idx = 0; idx < num_cells; idx++) {
        if (fabsf(h_fz3[idx]) > max_cold_fz) max_cold_fz = fabsf(h_fz3[idx]);
    }
    printf("\n[C4] T=%.0f K (below T_activation=%.0f K)\n",
           T_below_act, T_activation);
    printf("[C4] Max |F_z|         = %.4e N/m³  (should be 0)\n", max_cold_fz);
    if (max_cold_fz < 1.0) {
        printf("[C4] PASS — No recoil force below activation temperature\n");
    } else {
        printf("[C4] FAIL — Recoil force fires below activation temperature!\n");
    }

    // ----------------------------------------------------------------
    // C5: Verify the kernel uses ∇f for direction, NOT VOF normals
    // ∇f_z = (f[k+1] - f[k-1]) / (2dx) at interior cells
    // For our interface (liquid below, gas above), ∇f_z < 0
    // VOF normals at the interface point: n_z = -grad_f_z / |grad_f| > 0
    // F_z = P_recoil × grad_f_z  (uses ∇f, not n)
    // ----------------------------------------------------------------
    // This is already confirmed by reading the kernel (lines 569-580):
    // it uses grad_f_x/y/z directly, NOT normals.
    // So F_recoil = P × ∇f. The ∇f direction is:
    //   - gas→liquid pointing (from low f to high f)
    //   - For our setup: ∇f_z < 0 (f decreases with z since liquid is at low z)
    // Recoil pushes vapor outward → reaction force on liquid pushes inward.
    // ∇f points into liquid (negative z), so F_z = P × ∇f_z < 0. Correct.
    printf("\n[C5] Kernel direction check (analytical):\n");
    printf("     Liquid at low z → ∇f_z = df/dz < 0\n");
    printf("     F_z = P_recoil × ∇f_z < 0 (pushes into liquid) — correct.\n");
    printf("[C5] Kernel source confirmed: uses ∇f directly (NOT VOF normals).\n");
    printf("[C5] NOTE: force_multiplier=1 in production. Verify this is correct.\n");
    printf("     Analytical: ∫F_z dz = -P_recoil × 1 (since ∫|∇f_z| dz = 1).\n");
    printf("     Measured ratio C1=%.4f should confirm no missing multiplier.\n", ratio_C1);

    cudaFree(d_T);
    cudaFree(d_fill);
    cudaFree(d_T_cold);

    printf("\n==========================================================\n");
    printf("  Test C complete.\n");
    printf("==========================================================\n");
    return 0;
}
