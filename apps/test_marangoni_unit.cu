/**
 * @file test_marangoni_unit.cu
 * @brief Unit test: Marangoni force kernel — isolated, analytical comparison
 *
 * Setup:
 *   32×32×32 domain, dx=2.5μm
 *   Flat interface at z=15 (tanh VOF profile smeared over ~3 cells)
 *   Linear T field: T(y) = 500 + (y/NY)*2500 → grad_T_y = const
 *   liquid_fraction = nullptr (no mushy gate, pure surface)
 *
 * Analytical expectation:
 *   tau_s = |dσ/dT| × |∇T_y|           [N/m²]  surface shear stress
 *   F_y_vol = tau_s × |∇f|              [N/m³]  at each interface cell
 *   ∫F_y dz = tau_s                     [N/m²]  across interface thickness
 *
 * Tests:
 *   A1: ∫F_y dz at center column vs analytical tau_s  — ratio should be ≈1.0
 *   A2: ∫F_z dz at center column — should be near zero (tangential proj OK)
 *   A3: Per-cell F_y direction consistent with dsigma_dT<0 → force toward hot
 *   A4: Bulk liquid/gas cells (|∇f|≈0) get zero Marangoni force
 *   A5: Check if n_mag re-normalization in tangential projection is consistent
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"

using namespace lbm::physics;

int main() {
    const int NX = 32, NY = 32, NZ = 32;
    const float dx = 2.5e-6f;   // 2.5 µm
    const int num_cells = NX * NY * NZ;

    // Linear temperature field: T(j) = T_min + (j/(NY-1)) * DeltaT
    const float T_min = 500.0f, DeltaT = 2500.0f;
    const float grad_T_y = DeltaT / ((NY - 1) * dx);   // [K/m]

    // Surface tension coefficient for 316L stainless steel
    // dσ/dT ≈ -4e-4 N/(m·K)  (negative: σ decreases with T)
    const float dsigma_dT = -4.0e-4f;  // N/(m·K)

    // Interface at z=15, tanh half-width ~ 1.5 cells
    const int z_interface = 15;
    const float w = 1.5f * dx;  // tanh half-width

    printf("==========================================================\n");
    printf("  Test A: Marangoni Force Unit Test\n");
    printf("==========================================================\n");
    printf("Domain: %d×%d×%d, dx=%.2f µm\n", NX, NY, NZ, dx*1e6f);
    printf("Interface at z=%d (tanh, w=%.2f dx)\n", z_interface, w/dx);
    printf("grad_T_y = %.4e K/m  (linear field)\n", grad_T_y);
    printf("dσ/dT    = %.4e N/(m·K)\n", dsigma_dT);

    // Build host arrays
    std::vector<float> h_fill(num_cells, 0.0f);
    std::vector<float> h_T(num_cells, 0.0f);

    for (int k = 0; k < NZ; k++) {
        // tanh profile: f = 0.5 + 0.5 * tanh((z_interface - k) * dx / w)
        // f=1 for k < z_interface (liquid), f=0 for k > z_interface (gas)
        float z_phys = (k - z_interface) * dx;
        float f_val = 0.5f + 0.5f * tanhf(-z_phys / w);
        f_val = fmaxf(0.0f, fminf(1.0f, f_val));
        for (int j = 0; j < NY; j++) {
            float T_val = T_min + (j / (float)(NY - 1)) * DeltaT;
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);
                h_fill[idx] = f_val;
                h_T[idx] = T_val;
            }
        }
    }

    // Device allocations
    float* d_T     = nullptr;
    float* d_fill  = nullptr;
    cudaMalloc(&d_T,    num_cells * sizeof(float));
    cudaMalloc(&d_fill, num_cells * sizeof(float));
    cudaMemcpy(d_T,    h_T.data(),   num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill, h_fill.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Compute VOF normals via VOFSolver
    VOFSolver vof(NX, NY, NZ, dx,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL);
    vof.initialize(h_fill.data());
    vof.reconstructInterface();  // populates d_interface_normal_

    const float3* d_normals = vof.getInterfaceNormals();

    // ForceAccumulator
    ForceAccumulator fa(NX, NY, NZ);
    fa.reset();
    fa.addMarangoniForce(d_T, d_fill, nullptr /* no liquid_fraction */,
                         d_normals, dsigma_dT,
                         NX, NY, NZ, dx, 1.0f);

    // Copy forces to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), fa.getFx(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), fa.getFy(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), fa.getFz(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy fill and normals for analysis
    std::vector<float>  h_fill_out(num_cells);
    cudaMemcpy(h_fill_out.data(), d_fill, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float3> h_normals(num_cells);
    cudaMemcpy(h_normals.data(), d_normals, num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    // ----------------------------------------------------------------
    // A1: Integrate F_y across interface at center column (i=NX/2, j=NY/2)
    // ∫F_y dz ≈ sum(F_y[k]) * dx
    // Compare to analytical: tau_s = |dσ/dT| * grad_T_y
    // Note: dsigma_dT < 0, so tau_s should be negative (force toward low y)
    // But ∇T_y > 0, so τ_s = dsigma_dT * grad_T_y < 0
    // The force direction: lower T → higher σ → pulls surface toward high-T side
    // With dsigma_dT<0 and ∇T_y>0: σ decreases with y → surface tension gradient
    // pulls toward high-T (high y) → F_y < 0 (wait: dσ/dy = dσ/dT * dT/dy < 0)
    // so ∇σ points toward low y, Marangoni pulls toward low y... direction matters
    // The kernel: F_y += coeff * grad_T_s_y where coeff = dsigma_dT * |∇f|
    // grad_T_s_y = grad_T_y (tangential, z is normal), so
    // F_y = dsigma_dT * grad_T_y * |∇f|  [N/m³]  (negative for our params)
    // ∫F_y dz = dsigma_dT * grad_T_y * ∫|∇f|dz ≈ dsigma_dT * grad_T_y [N/m²]
    // ----------------------------------------------------------------
    const int i_center = NX / 2;
    const int j_center = NY / 2;

    double integral_Fy = 0.0, integral_Fz = 0.0, integral_Fx = 0.0;
    printf("\n--- Column (i=%d, j=%d): F integrated across z ---\n", i_center, j_center);
    printf("  k    fill       Fy [N/m³]      Fz [N/m³]   nz\n");
    for (int k = 0; k < NZ; k++) {
        int idx = i_center + NX * (j_center + NY * k);
        float Fy = h_fy[idx], Fz = h_fz[idx], Fx = h_fx[idx];
        float f = h_fill_out[idx];
        float nz = h_normals[idx].z;
        if (fabsf(Fy) > 1.0f || fabsf(Fz) > 1.0f || f > 0.01f) {
            printf("  k=%2d  f=%.3f  Fy=%+.4e  Fz=%+.4e  nz=%.3f\n",
                   k, f, Fy, Fz, nz);
        }
        integral_Fy += Fy * dx;
        integral_Fz += Fz * dx;
        integral_Fx += Fx * dx;
    }

    float tau_s_analytical = dsigma_dT * grad_T_y;  // [N/m²]
    float ratio_Fy = (float)(integral_Fy / tau_s_analytical);

    printf("\n[A1] Integrated F_y dz  = %.6e N/m²\n", (float)integral_Fy);
    printf("[A1] Analytical tau_s   = %.6e N/m²\n", tau_s_analytical);
    printf("[A1] Ratio (meas/anal)  = %.4f   (should be ~1.0)\n", ratio_Fy);
    if (fabsf(ratio_Fy - 1.0f) < 0.15f) {
        printf("[A1] PASS — Marangoni integral within 15%% of analytical\n");
    } else {
        printf("[A1] FAIL — ratio %.4f deviates >15%% from 1.0\n", ratio_Fy);
    }

    printf("\n[A2] Integrated F_z dz  = %.6e N/m²\n", (float)integral_Fz);
    printf("[A2] Integrated F_x dz  = %.6e N/m²\n", (float)integral_Fx);
    float z_leakage = fabsf((float)integral_Fz / (fabsf(tau_s_analytical) + 1e-20f));
    float x_leakage = fabsf((float)integral_Fx / (fabsf(tau_s_analytical) + 1e-20f));
    printf("[A2] |F_z|/tau_s        = %.4f   (should be ~0, tangential proj)\n", z_leakage);
    printf("[A2] |F_x|/tau_s        = %.4f   (should be ~0, no x-gradient)\n", x_leakage);
    if (z_leakage < 0.05f) {
        printf("[A2] PASS — F_z leakage < 5%%\n");
    } else {
        printf("[A2] FAIL — F_z leakage %.4f > 5%%  (tangential projection bug!)\n", z_leakage);
    }

    // ----------------------------------------------------------------
    // A3: Check that bulk liquid (f>0.99) and bulk gas (f<0.01) are zero
    // ----------------------------------------------------------------
    double max_bulk_force = 0.0;
    int bulk_nonzero = 0;
    for (int idx = 0; idx < num_cells; idx++) {
        float f = h_fill_out[idx];
        if (f < 0.01f || f > 0.99f) {
            double fm = sqrt((double)h_fx[idx]*h_fx[idx] +
                             (double)h_fy[idx]*h_fy[idx] +
                             (double)h_fz[idx]*h_fz[idx]);
            if (fm > max_bulk_force) max_bulk_force = fm;
            if (fm > 1.0) bulk_nonzero++;
        }
    }
    printf("\n[A3] Max force in bulk (f<0.01 or f>0.99): %.4e N/m³\n", max_bulk_force);
    printf("[A3] Cells with bulk |F|>1 N/m³: %d\n", bulk_nonzero);
    if (bulk_nonzero == 0) {
        printf("[A3] PASS — No Marangoni force leaks into bulk\n");
    } else {
        printf("[A3] FAIL — Marangoni force leaking into %d bulk cells!\n", bulk_nonzero);
    }

    // ----------------------------------------------------------------
    // A4: Check normal vector normalization at interface cells
    // The kernel uses n raw from VOF. If |n| != 1, tangential projection wrong.
    // ----------------------------------------------------------------
    double max_n_mag_deviation = 0.0;
    int interface_cell_count = 0;
    for (int idx = 0; idx < num_cells; idx++) {
        float f = h_fill_out[idx];
        if (f > 0.01f && f < 0.99f) {
            float3 n = h_normals[idx];
            float n_mag = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
            if (n_mag > 0.01f) {
                double dev = fabs(n_mag - 1.0);
                if (dev > max_n_mag_deviation) max_n_mag_deviation = dev;
                interface_cell_count++;
            }
        }
    }
    printf("\n[A4] Interface cells: %d\n", interface_cell_count);
    printf("[A4] Max |n| deviation from 1.0: %.6f\n", max_n_mag_deviation);
    if (max_n_mag_deviation < 0.01) {
        printf("[A4] PASS — VOF normals are unit vectors (within 1%%)\n");
    } else {
        printf("[A4] FAIL — VOF normals are NOT unit vectors! "
               "Tangential projection is wrong.\n");
    }

    // ----------------------------------------------------------------
    // A5: fl_gate = (fl-0.1)/0.1 clamp — test if it wrongly kills full liquid
    // With liquid_fraction=nullptr, fl_gate=1 always. No bug here.
    // But in integration runs, fl_gate can prematurely kill Marangoni.
    // Test: with fl=0.15 (just inside mushy), what fraction of force survives?
    // Expected fl_gate = (0.15-0.1)/0.1 = 0.5 — half the force killed
    // ----------------------------------------------------------------
    {
        float fl_test = 0.15f;
        float fl_gate_test = fminf(fmaxf((fl_test - 0.1f) / 0.1f, 0.0f), 1.0f);
        printf("\n[A5] fl=0.15 → fl_gate = %.2f  (kills %.0f%% of Marangoni)\n",
               fl_gate_test, (1.0f - fl_gate_test) * 100.0f);
        float fl_test2 = 0.20f;
        float fl_gate_test2 = fminf(fmaxf((fl_test2 - 0.1f) / 0.1f, 0.0f), 1.0f);
        printf("[A5] fl=0.20 → fl_gate = %.2f  (full force at fl>=0.2)\n", fl_gate_test2);
        printf("[A5] NOTE: Interface cells at f≈0.5 but fl in mushy zone [0.1,0.2]\n");
        printf("[A5]       will have Marangoni severely reduced. Verify this is intended.\n");
    }

    cudaFree(d_T);
    cudaFree(d_fill);

    printf("\n==========================================================\n");
    printf("  Test A complete.\n");
    printf("==========================================================\n");
    return 0;
}
