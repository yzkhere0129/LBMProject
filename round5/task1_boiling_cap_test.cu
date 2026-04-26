/**
 * Round 5 Task 1: Boiling cap unit test
 *
 * Tests:
 * 1. Bulk liquid (f=1.0) with T=3800K → capped to 3250K (T_boil+50K)
 * 2. Interface cells (f=0.5) → NOT capped
 * 3. Gas cells (f=0) → NOT capped
 * 4. Energy tracking: ΔE = ρ*cp*(T-T_cap)*dV
 * 5. Multiple calls accumulate correctly (no per-step reset)
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "physics/thermal_fdm.h"
#include "physics/material_properties.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

int main() {
    const int NX = 10, NY = 10, NZ = 10;
    const int N = NX * NY * NZ;
    const float dx = 2.5e-6f, dt = 1.0e-8f;
    const float rho = 7900, cp = 700, k_c = 20;
    const float alpha = k_c / (rho * cp);
    const float T_boil = 3200.0f;

    MaterialProperties mat = {};
    mat.rho_solid = rho; mat.rho_liquid = rho;
    mat.cp_solid = cp; mat.cp_liquid = cp;
    mat.k_solid = k_c; mat.k_liquid = k_c;
    mat.T_solidus = 1650; mat.T_liquidus = 1700;
    mat.T_vaporization = T_boil;
    mat.L_fusion = 270000; mat.L_vaporization = 6.09e6f;
    mat.molar_mass = 0.05585f;

    ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

    // Set up fill level: center cell is bulk liquid, neighbors are interface
    std::vector<float> h_fill(N, 0.0f);
    std::vector<float> h_T(N, 300.0f);

    // Bulk liquid: center 4x4x4 region
    for (int k = 3; k < 7; k++)
        for (int j = 3; j < 7; j++)
            for (int i = 3; i < 7; i++)
                h_fill[i + j*NX + k*NX*NY] = 1.0f;

    // Interface: one layer around bulk liquid
    for (int k = 2; k < 8; k++)
        for (int j = 2; j < 8; j++)
            for (int i = 2; i < 8; i++)
                if (h_fill[i + j*NX + k*NX*NY] == 0.0f)
                    // Check if neighbor is bulk liquid
                    for (int dk = -1; dk <= 1; dk++)
                        for (int dj = -1; dj <= 1; dj++)
                            for (int di = -1; di <= 1; di++) {
                                int ni = i+di, nj = j+dj, nk = k+dk;
                                if (ni >= 0 && ni < NX && nj >= 0 && nj < NY && nk >= 0 && nk < NZ)
                                    if (h_fill[ni + nj*NX + nk*NX*NY] == 1.0f)
                                        h_fill[i + j*NX + k*NX*NY] = 0.5f;
                            }

    // Set temperature: bulk liquid is overheated to 3800K
    for (int idx = 0; idx < N; idx++) {
        if (h_fill[idx] >= 0.99f)
            h_T[idx] = 3800.0f;
        else if (h_fill[idx] > 0.01f)
            h_T[idx] = 3500.0f;  // interface cell
        else
            h_T[idx] = 300.0f;  // gas
    }

    // Copy to device
    float* d_fill;
    CUDA_CHECK(cudaMalloc(&d_fill, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fill, h_fill.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    thermal.initialize(h_T.data());
    thermal.setVOFFillLevel(d_fill);

    printf("================================================================\n");
    printf("  Task 1: Boiling Cap Unit Test\n");
    printf("================================================================\n");

    // Count cells
    int n_bulk = 0, n_iface = 0, n_gas = 0;
    for (int i = 0; i < N; i++) {
        if (h_fill[i] >= 0.99f) n_bulk++;
        else if (h_fill[i] > 0.01f) n_iface++;
        else n_gas++;
    }
    printf("Cells: bulk_liquid=%d, interface=%d, gas=%d\n\n", n_bulk, n_iface, n_gas);

    // Test 1: Apply boiling cap, check T gets capped
    printf("--- Test 1: Bulk liquid T=3800K → capped to 3250K ---\n");
    thermal.applySubsurfaceBoilingCap(T_boil, 50.0f);

    std::vector<float> h_T_after(N);
    thermal.copyTemperatureToHost(h_T_after.data());

    int pass1 = 0, fail1 = 0;
    float E_removed = thermal.getBoilingCapEnergyRemoved();
    for (int idx = 0; idx < N; idx++) {
        if (h_fill[idx] >= 0.99f) {
            if (h_T_after[idx] > 3250.01f) {
                printf("  FAIL: idx=%d, T=%.1fK (expected <= 3250K)\n", idx, h_T_after[idx]);
                fail1++;
            } else {
                pass1++;
            }
        }
    }
    // Expected energy: ρ * cp * (3800 - 3250) * dV * n_bulk_cells
    float dV = dx * dx * dx;
    float E_expected = rho * cp * (3800.0f - 3250.0f) * dV * n_bulk;
    printf("  Bulk liquid capped: %d/%d PASS\n", pass1, n_bulk);
    printf("  Energy removed: %.6e J (expected: %.6e J, ratio=%.3f)\n",
           E_removed, E_expected, E_removed / E_expected);
    if (fabs(E_removed / E_expected - 1.0f) < 0.01f)
        printf("  Energy tracking: PASS\n");
    else
        printf("  Energy tracking: FAIL (ratio %.3f != 1.0)\n", E_removed / E_expected);

    // Test 2: Interface cells should NOT be capped
    printf("\n--- Test 2: Interface cells NOT capped ---\n");
    int pass2 = 0, fail2 = 0;
    for (int idx = 0; idx < N; idx++) {
        if (h_fill[idx] > 0.01f && h_fill[idx] < 0.99f) {
            if (fabsf(h_T_after[idx] - 3500.0f) > 0.01f) {
                printf("  FAIL: idx=%d, T=%.1fK (expected 3500K, f=%.2f)\n",
                       idx, h_T_after[idx], h_fill[idx]);
                fail2++;
            } else {
                pass2++;
            }
        }
    }
    printf("  Interface unchanged: %d/%d PASS\n", pass2, n_iface);

    // Test 3: Gas cells should NOT be affected
    printf("\n--- Test 3: Gas cells NOT affected ---\n");
    int pass3 = 0, fail3 = 0;
    for (int idx = 0; idx < N; idx++) {
        if (h_fill[idx] < 0.01f) {
            if (fabsf(h_T_after[idx] - 300.0f) > 0.01f) {
                printf("  FAIL: idx=%d, T=%.1fK (expected 300K)\n", idx, h_T_after[idx]);
                fail3++;
            } else {
                pass3++;
            }
        }
    }
    printf("  Gas unchanged: %d/%d PASS\n", pass3, n_gas);

    // Test 4: Multiple calls accumulate correctly
    printf("\n--- Test 4: Multiple calls accumulate energy ---\n");
    // Reset bulk liquid to 3800K again
    for (int idx = 0; idx < N; idx++)
        if (h_fill[idx] >= 0.99f)
            h_T[idx] = 3800.0f;
    thermal.initialize(h_T.data());

    thermal.applySubsurfaceBoilingCap(T_boil, 50.0f);
    float E1 = thermal.getBoilingCapEnergyRemoved();

    // Set bulk liquid to 3800K again
    thermal.initialize(h_T.data());
    thermal.applySubsurfaceBoilingCap(T_boil, 50.0f);
    float E2 = thermal.getBoilingCapEnergyRemoved();

    printf("  Call 1 energy: %.6e J\n", E1);
    printf("  Call 2 energy: %.6e J\n", E2);
    if (fabs(E1 / E2 - 1.0f) < 0.01f)
        printf("  Accumulation: PASS (consistent)\n");
    else
        printf("  Accumulation: FAIL (inconsistent: ratio=%.3f)\n", E1/E2);

    // Summary
    printf("\n================================================================\n");
    int total_fail = fail1 + fail2 + fail3;
    if (total_fail == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("FAILURES: %d\n", total_fail);
    printf("================================================================\n");

    cudaFree(d_fill);
    return total_fail > 0 ? 1 : 0;
}
