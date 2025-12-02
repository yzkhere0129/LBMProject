/**
 * @file test_marangoni_velocity_diagnosis.cu
 * @brief Diagnostic test to identify 10-100x velocity discrepancy
 *
 * This test creates a simple LPBF-like scenario and prints detailed diagnostics:
 * - Temperature gradient magnitude
 * - VOF gradient magnitude
 * - Marangoni force magnitude (before and after limiting)
 * - Resulting velocity
 * - Comparison with expected values at each step
 *
 * Goal: Identify which step introduces the 10-100x discrepancy
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace lbm::physics;

// LPBF-like parameters
constexpr int NX = 20;
constexpr int NY = 20;
constexpr int NZ = 20;
constexpr float DX = 2.0e-6f;         // 2 micron grid spacing
constexpr float DSIGMA_DT = -0.00026f; // Ti6Al4V: N/(m·K)
constexpr float MU_LIQUID = 0.005f;    // Liquid viscosity: Pa·s
constexpr float RHO_LIQUID = 4110.0f;  // Liquid density: kg/m³

// Helper function
template<typename T>
T* allocateDeviceArray(int size, T init_value = T{}) {
    T* d_ptr;
    cudaMalloc(&d_ptr, size * sizeof(T));
    T* h_ptr = new T[size];
    for (int i = 0; i < size; i++) h_ptr[i] = init_value;
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    delete[] h_ptr;
    return d_ptr;
}

void diagnoseMarangoniVelocity() {
    printf("\n" "=" "===============================================\n");
    printf("MARANGONI VELOCITY DIAGNOSTIC TEST\n");
    printf("=" "===============================================\n\n");

    printf("SETUP:\n");
    printf("  Grid: %dx%dx%d cells\n", NX, NY, NZ);
    printf("  Grid spacing: %.2e m (%.1f microns)\n", DX, DX * 1e6f);
    printf("  Material: Ti6Al4V\n");
    printf("  dσ/dT: %.2e N/(m·K)\n", DSIGMA_DT);
    printf("  Viscosity: %.3f Pa·s\n", MU_LIQUID);
    printf("  Density: %.0f kg/m³\n\n", RHO_LIQUID);

    const int num_cells = NX * NY * NZ;

    // Allocate arrays
    float* d_temperature = allocateDeviceArray<float>(num_cells);
    float* d_fill_level = allocateDeviceArray<float>(num_cells);
    float3* d_interface_normal = allocateDeviceArray<float3>(num_cells);
    float* d_force_x = allocateDeviceArray<float>(num_cells);
    float* d_force_y = allocateDeviceArray<float>(num_cells);
    float* d_force_z = allocateDeviceArray<float>(num_cells);

    float* h_temperature = new float[num_cells];
    float* h_fill_level = new float[num_cells];
    float3* h_interface_normal = new float3[num_cells];

    // Scenario: Hot center (laser spot), cool edges
    // Flat interface perpendicular to Z
    const float T_center = 3000.0f;  // K (laser-heated)
    const float T_edge = 1900.0f;    // K (near melting point)
    const int interface_z = NZ / 2;

    float cx_phys = (NX / 2.0f) * DX;
    float cy_phys = (NY / 2.0f) * DX;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = i + NX * (j + NY * k);

                // Radial temperature profile
                float x_phys = i * DX;
                float y_phys = j * DX;
                float r = std::sqrt((x_phys - cx_phys) * (x_phys - cx_phys) +
                                   (y_phys - cy_phys) * (y_phys - cy_phys));
                float r_max = std::sqrt(cx_phys * cx_phys + cy_phys * cy_phys);

                h_temperature[idx] = T_center - (T_center - T_edge) * (r / r_max);

                // Sharp interface perpendicular to Z
                if (k < interface_z) {
                    h_fill_level[idx] = 1.0f;
                } else if (k == interface_z) {
                    h_fill_level[idx] = 0.5f;
                } else {
                    h_fill_level[idx] = 0.0f;
                }

                h_interface_normal[idx] = make_float3(0.0f, 0.0f, 1.0f);
            }
        }
    }

    cudaMemcpy(d_temperature, h_temperature, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill_level, h_fill_level, num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_interface_normal, h_interface_normal, num_cells * sizeof(float3), cudaMemcpyHostToDevice);

    // Compute Marangoni force
    MarangoniEffect marangoni(NX, NY, NZ, DSIGMA_DT, DX);
    marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                    d_force_x, d_force_y, d_force_z);

    // Copy results
    float* h_force_x = new float[num_cells];
    float* h_force_y = new float[num_cells];
    float* h_force_z = new float[num_cells];

    cudaMemcpy(h_force_x, d_force_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_y, d_force_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_z, d_force_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze center interface cell
    int ic = NX / 2;
    int jc = NY / 2;
    int kc = interface_z;
    int idx_center = ic + NX * (jc + NY * kc);

    printf("DIAGNOSTIC ANALYSIS AT CENTER INTERFACE CELL:\n");
    printf("  Cell coordinates: (%d, %d, %d)\n", ic, jc, kc);
    printf("  Fill level: %.3f\n", h_fill_level[idx_center]);
    printf("  Temperature: %.1f K\n\n", h_temperature[idx_center]);

    // Step 1: Temperature gradient
    int idx_xm = (ic - 1) + NX * (jc + NY * kc);
    int idx_xp = (ic + 1) + NX * (jc + NY * kc);
    int idx_ym = ic + NX * ((jc - 1) + NY * kc);
    int idx_yp = ic + NX * ((jc + 1) + NY * kc);

    float grad_T_x = (h_temperature[idx_xp] - h_temperature[idx_xm]) / (2.0f * DX);
    float grad_T_y = (h_temperature[idx_yp] - h_temperature[idx_ym]) / (2.0f * DX);
    float grad_T_mag = std::sqrt(grad_T_x * grad_T_x + grad_T_y * grad_T_y);

    printf("STEP 1: TEMPERATURE GRADIENT\n");
    printf("  ∇T = (%.3e, %.3e, 0) K/m\n", grad_T_x, grad_T_y);
    printf("  |∇T| = %.3e K/m\n", grad_T_mag);
    printf("  Expected for LPBF: 10^7 - 10^8 K/m\n");
    printf("  Status: %s\n\n", (grad_T_mag >= 1e7f && grad_T_mag <= 1e8f) ? "OK" : "CHECK");

    // Step 2: VOF gradient
    int idx_zm = ic + NX * (jc + NY * (kc - 1));
    int idx_zp = ic + NX * (jc + NY * (kc + 1));

    float grad_f_z = (h_fill_level[idx_zp] - h_fill_level[idx_zm]) / (2.0f * DX);
    float grad_f_mag = std::abs(grad_f_z);

    printf("STEP 2: VOF GRADIENT\n");
    printf("  ∇f·z = %.3e 1/m\n", grad_f_z);
    printf("  |∇f| = %.3e 1/m\n", grad_f_mag);
    printf("  Expected: ~0.5 / DX = %.3e 1/m\n", 0.5f / DX);
    printf("  Status: %s\n\n", (std::abs(grad_f_mag - 0.5f/DX) / (0.5f/DX) < 0.2f) ? "OK" : "CHECK");

    // Step 3: Tangential gradient (should equal total gradient since normal is perpendicular)
    float grad_Ts_mag = grad_T_mag;  // Since normal is in Z, tangential is in XY plane

    printf("STEP 3: TANGENTIAL TEMPERATURE GRADIENT\n");
    printf("  |∇_s T| = %.3e K/m\n", grad_Ts_mag);
    printf("  (Same as |∇T| since normal ⊥ gradient)\n\n");

    // Step 4: Expected Marangoni force magnitude
    float F_expected = std::abs(DSIGMA_DT) * grad_Ts_mag * grad_f_mag;

    printf("STEP 4: EXPECTED MARANGONI FORCE\n");
    printf("  F = |dσ/dT| × |∇_s T| × |∇f|\n");
    printf("    = %.3e × %.3e × %.3e\n", std::abs(DSIGMA_DT), grad_Ts_mag, grad_f_mag);
    printf("    = %.3e N/m³\n", F_expected);
    printf("  Expected for LPBF: 10^8 - 10^9 N/m³\n");
    printf("  Status: %s\n\n", (F_expected >= 1e8f && F_expected <= 1e10f) ? "OK" : "CHECK");

    // Step 5: Computed force
    float fx = h_force_x[idx_center];
    float fy = h_force_y[idx_center];
    float fz = h_force_z[idx_center];
    float F_computed = std::sqrt(fx*fx + fy*fy + fz*fz);

    printf("STEP 5: COMPUTED MARANGONI FORCE\n");
    printf("  F = (%.3e, %.3e, %.3e) N/m³\n", fx, fy, fz);
    printf("  |F| = %.3e N/m³\n", F_computed);
    printf("  Expected: %.3e N/m³\n", F_expected);
    printf("  Relative error: %.1f%%\n", 100.0f * std::abs(F_computed - F_expected) / F_expected);

    if (F_computed < F_expected * 0.01f) {
        printf("  STATUS: CRITICAL - Force is 100x too small!\n\n");
    } else if (F_computed < F_expected * 0.1f) {
        printf("  STATUS: WARNING - Force is 10x too small!\n\n");
    } else if (F_computed < F_expected * 0.5f) {
        printf("  STATUS: CHECK - Force is lower than expected\n\n");
    } else {
        printf("  STATUS: OK\n\n");
    }

    // Step 6: Expected velocity
    // v = F / (ρ * Ma) where Ma is Marangoni number
    // Simple estimate: v ~ (dσ/dT * ΔT) / μ
    float delta_T = T_center - T_edge;
    float v_expected = std::abs(DSIGMA_DT * delta_T) / MU_LIQUID;

    printf("STEP 6: EXPECTED MARANGONI VELOCITY\n");
    printf("  v ~ |dσ/dT| × ΔT / μ\n");
    printf("    = %.3e × %.1f / %.3f\n", std::abs(DSIGMA_DT), delta_T, MU_LIQUID);
    printf("    = %.3f m/s\n", v_expected);
    printf("  Expected for LPBF: 0.5 - 2.0 m/s\n");
    printf("  Status: %s\n\n", (v_expected >= 0.5f && v_expected <= 2.0f) ? "OK" : "CHECK");

    // Step 7: Velocity from computed force
    // For steady-state balance: F_marangoni = μ × ∇²v
    // Rough estimate: v ~ F × L² / μ where L is length scale
    float L_scale = (NX / 2.0f) * DX;  // Radius of hot spot
    float v_computed_1 = (F_computed * L_scale) / (MU_LIQUID);

    // Alternative: Direct from force acceleration
    // a = F / ρ, assuming flow develops over time scale τ ~ L/v
    // v ~ sqrt(F × L / ρ)
    float v_computed_2 = std::sqrt((F_computed * L_scale) / RHO_LIQUID);

    printf("STEP 7: COMPUTED VELOCITY ESTIMATES\n");
    printf("  Method 1 (viscous balance): v ~ F×L/μ = %.3f m/s\n", v_computed_1);
    printf("  Method 2 (acceleration):     v ~ sqrt(F×L/ρ) = %.3f m/s\n", v_computed_2);
    printf("  Expected: %.3f m/s\n", v_expected);
    printf("\n");

    // Summary
    printf("=" "===============================================\n");
    printf("DIAGNOSIS SUMMARY:\n");
    printf("=" "===============================================\n");

    bool grad_T_ok = (grad_T_mag >= 1e6f && grad_T_mag <= 1e9f);
    bool grad_f_ok = (grad_f_mag >= 1e5f && grad_f_mag <= 1e7f);
    bool force_ok = (F_computed >= F_expected * 0.5f);
    bool velocity_ok = (v_expected >= 0.5f && v_expected <= 2.0f);

    printf("  Temperature gradient:  %s (%.2e K/m)\n",
           grad_T_ok ? "✓ OK" : "✗ ISSUE", grad_T_mag);
    printf("  VOF gradient:          %s (%.2e 1/m)\n",
           grad_f_ok ? "✓ OK" : "✗ ISSUE", grad_f_mag);
    printf("  Force magnitude:       %s (%.2e N/m³)\n",
           force_ok ? "✓ OK" : "✗ ISSUE", F_computed);
    printf("  Velocity scale:        %s (%.2f m/s)\n",
           velocity_ok ? "✓ OK" : "✗ ISSUE", v_expected);

    if (!force_ok) {
        printf("\n  PRIMARY ISSUE: Marangoni force is %.1fx too small\n",
               F_expected / F_computed);
        printf("  This will cause velocities to be %.1fx too small\n",
               F_expected / F_computed);
    }

    if (!grad_T_ok) {
        printf("\n  ISSUE: Temperature gradient may be too small for LPBF\n");
        printf("  Recommendation: Check grid resolution and temperature field setup\n");
    }

    if (!grad_f_ok) {
        printf("\n  ISSUE: VOF gradient magnitude unexpected\n");
        printf("  Recommendation: Check interface thickness and fill level setup\n");
    }

    printf("=" "===============================================\n\n");

    // Cleanup
    delete[] h_temperature;
    delete[] h_fill_level;
    delete[] h_interface_normal;
    delete[] h_force_x;
    delete[] h_force_y;
    delete[] h_force_z;

    cudaFree(d_temperature);
    cudaFree(d_fill_level);
    cudaFree(d_interface_normal);
    cudaFree(d_force_x);
    cudaFree(d_force_y);
    cudaFree(d_force_z);
}

int main() {
    diagnoseMarangoniVelocity();
    return 0;
}
