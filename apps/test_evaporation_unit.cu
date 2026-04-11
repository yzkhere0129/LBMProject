/**
 * @file test_evaporation_unit.cu
 * @brief Unit test: HKL evaporation cooling — isolated, analytical comparison
 *
 * Setup:
 *   16×16×16 domain
 *   f=1 (liquid) for k<8, f=0 (gas) for k>=8  [sharp interface at k=7.5]
 *   Interface cells are NOT created — this tests the VOLUMETRIC path
 *
 * Tests:
 *   B1: Volumetric mode (J=nullptr) — how many cells get cooled?
 *       Expected: only interface cells (f∈[0.01,0.99]).
 *       Bug: ALL cells with T>T_boil get cooled (even deep bulk).
 *   B2: Analytical HKL dT/dt at T=3500K — compare to measured.
 *   B3: Interface-only path (with J_evap) — verify correct cell selection.
 *   B4: Energy conservation: total ΔE should match ∫q_evap dA × dt
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "physics/thermal_fdm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// Replicate HKL formula for analytical comparison
float hkl_dT(float T, const MaterialProperties& mat, float dt, float dx) {
    const float R_gas = 8.314f;
    const float P_ref = 101325.0f;
    const float alpha_evap = 0.18f;
    float T_boil = mat.T_vaporization;
    float exponent = (mat.L_vaporization * mat.molar_mass / R_gas)
                   * (1.0f / T_boil - 1.0f / T);
    exponent = fminf(exponent, 80.0f);
    float P_sat = P_ref * expf(exponent);
    float J_evap = alpha_evap * P_sat
                 / sqrtf(2.0f * 3.14159265f * R_gas * T / mat.molar_mass);
    float q_evap = J_evap * mat.L_vaporization;  // cooling_factor=1
    float rho = mat.getDensity(T);
    float cp  = mat.getSpecificHeat(T);
    return q_evap * dt / (rho * cp * dx);
}

int main() {
    const int NX = 16, NY = 16, NZ = 16;
    const float dx = 2.5e-6f;
    const float dt = 1.0e-9f;  // tiny dt so we measure per-step dT accurately
    const int num_cells = NX * NY * NZ;

    // 316L stainless steel
    MaterialProperties mat = MaterialDatabase::get316L();

    printf("==========================================================\n");
    printf("  Test B: Evaporation Cooling Unit Test\n");
    printf("==========================================================\n");
    printf("Material: %s, T_boil=%.0f K\n", mat.name, mat.T_vaporization);
    printf("Domain: %d×%d×%d, dx=%.2f µm, dt=%.2e s\n", NX, NY, NZ, dx*1e6f, dt);

    // alpha: 316L thermal diffusivity ≈ 4e-6 m²/s
    const float alpha = mat.k_liquid / (mat.rho_liquid * mat.cp_liquid);
    ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

    // Build fill level: flat slab, f=1 for k<8, f=0 for k>=8
    // We use a sharp step — the interface cells are at k=7 (last liquid cell)
    // With sharp step, |∇f|=0 in bulk and |∇f|=1/dx at k=7 boundary.
    // For the volumetric mode test we want bulk metal cells above T_boil.
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int k = 0; k < NZ / 2; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                h_fill[i + NX * (j + NY * k)] = 1.0f;  // fully liquid
            }
        }
    }
    // Add a thin interface layer at k=NZ/2-1 and k=NZ/2
    const int k_iface = NZ / 2 - 1;
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            h_fill[i + NX * (j + NY * k_iface)]       = 0.7f;
            h_fill[i + NX * (j + NY * (k_iface + 1))] = 0.3f;
        }
    }

    float* d_fill = nullptr;
    cudaMalloc(&d_fill, num_cells * sizeof(float));
    cudaMemcpy(d_fill, h_fill.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    thermal.setVOFFillLevel(d_fill);

    const float T_test = 3500.0f;

    // ----------------------------------------------------------------
    // B1: Volumetric mode (J=nullptr) — which cells actually get cooled?
    // Initialize ALL cells to T_test. Count how many change.
    // Expected: only cells with T>T_boil get dT.
    // Bug check: does it cool ONLY interface cells, or ALL metal cells?
    // ----------------------------------------------------------------
    printf("\n--- B1: Volumetric evaporation cooling (J=nullptr) ---\n");
    thermal.initialize(T_test);

    // Copy T before
    std::vector<float> T_before(num_cells), T_after(num_cells);
    thermal.copyTemperatureToHost(T_before.data());

    // Apply volumetric evaporation (nullptr path → all T>T_boil cells)
    thermal.applyEvaporationCooling(nullptr, nullptr, dt, dx, 1.0f);

    thermal.copyTemperatureToHost(T_after.data());

    int cells_cooled_liquid = 0;   // f=1 (bulk liquid) cells that got cooled
    int cells_cooled_iface  = 0;   // f∈(0.01,0.99) interface cells cooled
    int cells_cooled_gas    = 0;   // f=0 (gas) cells cooled (should be zero)
    double total_dT_liquid  = 0.0;
    double total_dT_iface   = 0.0;

    for (int idx = 0; idx < num_cells; idx++) {
        float f  = h_fill[idx];
        float dT = T_before[idx] - T_after[idx];
        if (dT > 0.0f) {
            if (f >= 0.99f)                    { cells_cooled_liquid++; total_dT_liquid += dT; }
            else if (f > 0.01f && f < 0.99f)  { cells_cooled_iface++;  total_dT_iface  += dT; }
            else if (f <= 0.01f)               { cells_cooled_gas++; }
        }
    }

    printf("  Cells cooled in bulk liquid (f=1.0):      %d  (WRONG if >0)\n",
           cells_cooled_liquid);
    printf("  Cells cooled at interface (f∈(0.01,0.99)): %d  (expected: %d)\n",
           cells_cooled_iface, 2 * NX * NY);
    printf("  Cells cooled in gas (f=0.0):               %d  (should be 0)\n",
           cells_cooled_gas);
    if (cells_cooled_liquid > 0) {
        printf("  BUG: Volumetric evaporation cooling fires in %d BULK LIQUID cells!\n",
               cells_cooled_liquid);
        printf("  This over-cools the melt pool interior (no free surface there).\n");
    } else {
        printf("  OK: No bulk liquid cells cooled.\n");
    }

    // ----------------------------------------------------------------
    // B2: Analytical HKL dT — compare kernel result to formula
    // At T=3500K, T>T_boil, compute expected dT per step.
    // ----------------------------------------------------------------
    printf("\n--- B2: Analytical HKL dT comparison ---\n");
    float T_test_vals[] = {3200.0f, 3400.0f, 3500.0f, 3600.0f, 3800.0f};
    printf("  T[K]    |  dT_analytical[K] | dT_kernel (iface) | ratio\n");
    printf("  --------|-------------------|-------------------|------\n");

    for (float T_val : T_test_vals) {
        if (T_val <= mat.T_vaporization) {
            printf("  %.0fK    |  (below T_boil)     |  -                | -\n", T_val);
            continue;
        }
        float dT_anal = hkl_dT(T_val, mat, dt, dx);

        // Set interface cells to T_val, measure cooling
        std::vector<float> h_T_init(num_cells, 300.0f);  // gas/substrate cold
        for (int k = 0; k < NZ / 2 - 1; k++) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    h_T_init[i + NX*(j + NY*k)] = T_val;  // bulk liquid at T_val
                }
            }
        }
        // Interface cells
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                h_T_init[i + NX*(j + NY * k_iface)]       = T_val;
                h_T_init[i + NX*(j + NY * (k_iface + 1))] = T_val;
            }
        }
        thermal.initialize(h_T_init.data());

        // Apply interface-only path: pre-compute J_evap
        float* d_J_evap = nullptr;
        cudaMalloc(&d_J_evap, num_cells * sizeof(float));
        thermal.computeEvaporationMassFlux(d_J_evap, d_fill);

        std::vector<float> T_b(num_cells), T_a(num_cells);
        thermal.copyTemperatureToHost(T_b.data());
        thermal.applyEvaporationCooling(d_J_evap, d_fill, dt, dx, 1.0f);
        thermal.copyTemperatureToHost(T_a.data());

        // Average dT at interface cells
        double sum_dT = 0.0; int count = 0;
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX*(j + NY * k_iface);
                float dT_obs = T_b[idx] - T_a[idx];
                if (dT_obs > 0.0f) { sum_dT += dT_obs; count++; }
            }
        }
        float dT_kernel = (count > 0) ? (float)(sum_dT / count) : 0.0f;
        float ratio = (dT_anal > 1e-10f) ? dT_kernel / dT_anal : 0.0f;
        printf("  %.0fK    |  %16.4e   |  %16.4e  | %.4f%s\n",
               T_val, dT_anal, dT_kernel, ratio,
               (fabsf(ratio - 1.0f) < 0.05f ? "  PASS" : "  CHECK"));
        cudaFree(d_J_evap);
    }

    // ----------------------------------------------------------------
    // B3: Verify that volumetric mode cools ALL metal (not just interface)
    // This is the documented bug: line 1810 in multiphysics_solver.cu
    // calls applyEvaporationCooling(nullptr, nullptr, ...) which hits
    // fdmEvaporationCoolingKernel with z_surface=-1 → ALL T>T_boil cells.
    // ----------------------------------------------------------------
    printf("\n--- B3: Volumetric mode scope check ---\n");
    // Uniform T_test in all metal cells (k < NZ/2)
    std::vector<float> h_T_flat(num_cells, 300.0f);
    for (int k = 0; k < NZ / 2; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                h_T_flat[i + NX*(j + NY*k)] = T_test;
            }
        }
    }
    thermal.initialize(h_T_flat.data());

    std::vector<float> T_flat_before(num_cells), T_flat_after(num_cells);
    thermal.copyTemperatureToHost(T_flat_before.data());
    // Volumetric: no J_evap, no fill passed to kernel
    thermal.applyEvaporationCooling(nullptr, nullptr, dt, dx, 1.0f);
    thermal.copyTemperatureToHost(T_flat_after.data());

    int total_cooled = 0;
    int expected_iface_cells = 2 * NX * NY;  // 2 layers of interface
    int total_metal_cells = (NZ / 2) * NX * NY;
    for (int idx = 0; idx < num_cells; idx++) {
        if (T_flat_before[idx] - T_flat_after[idx] > 0.0f) total_cooled++;
    }
    printf("  Total metal cells:        %d\n", total_metal_cells);
    printf("  Expected interface cells: %d\n", expected_iface_cells);
    printf("  Actual cells cooled:      %d\n", total_cooled);
    printf("  Bulk liquid cells cooled: %d  (WRONG if nonzero)\n",
           total_cooled - expected_iface_cells);

    if (total_cooled > expected_iface_cells) {
        printf("  BUG CONFIRMED: fdmEvaporationCoolingKernel with J=nullptr\n");
        printf("  fires on ALL %d metal cells, not just %d interface cells!\n",
               total_cooled, expected_iface_cells);
        printf("  Over-cools melt pool by factor ~%.1f×\n",
               (float)total_cooled / expected_iface_cells);
    } else {
        printf("  OK: Volumetric mode only hits interface cells.\n");
    }

    cudaFree(d_fill);

    printf("\n==========================================================\n");
    printf("  Test B complete.\n");
    printf("==========================================================\n");
    return 0;
}
