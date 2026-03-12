/**
 * @file stefan_benchmark.cu
 * @brief Pure-metal 1D Stefan problem benchmark
 *
 * Validates the enthalpy source term (ESM) phase change implementation
 * against the Neumann analytical solution for 1D semi-infinite melting.
 *
 * Setup:
 *   - Ideal pure metal: T_solidus ≈ T_liquidus (ΔT = 0.1 K)
 *   - Constant properties: cp, k, ρ identical in solid and liquid
 *   - Stefan number Ste = 0.5 → λ ≈ 0.4654
 *   - Semi-infinite solid at T_melt, left wall at T_hot
 *
 * Analytical solution (Neumann):
 *   s(t) = 2λ√(αt)        (front position)
 *   T(x) = T_hot - ΔT·erf(x/(2√αt)) / erf(λ)   (liquid region)
 *
 * Output: CSV file for Python plotting script
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// ============================================================================
// Pure metal material (eliminates mushy-zone ambiguity)
// ============================================================================
static MaterialProperties createPureMetal() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "PureMetal_Ste0.5", sizeof(mat.name) - 1);

    // Constant thermophysical properties (solid = liquid)
    mat.rho_solid  = 7000.0f;   mat.rho_liquid  = 7000.0f;
    mat.cp_solid   = 500.0f;    mat.cp_liquid   = 500.0f;
    mat.k_solid    = 50.0f;     mat.k_liquid    = 50.0f;
    mat.mu_liquid  = 0.005f;

    // Near-isothermal melting (ΔT = 0.1 K avoids /0, approximates pure metal)
    mat.T_solidus      = 1000.0f;
    mat.T_liquidus     = 1000.1f;
    mat.T_vaporization = 3000.0f;

    // Latent heat chosen for Ste = cp·ΔT_wall / L = 0.5
    // With ΔT_wall = T_hot - T_melt = 100 K:
    //   L = cp·100/0.5 = 100,000 J/kg
    mat.L_fusion       = 100000.0f;
    mat.L_vaporization = 6.0e6f;

    // Surface properties (unused for thermal-only benchmark)
    mat.surface_tension    = 1.0f;
    mat.dsigma_dT          = -1.0e-4f;
    mat.absorptivity_solid = 0.3f;
    mat.absorptivity_liquid= 0.3f;
    mat.emissivity         = 0.3f;

    return mat;
}

// ============================================================================
// Solve Neumann transcendental equation: λ·exp(λ²)·erf(λ) = Ste/√π
// ============================================================================
static float solveLambda(float Ste) {
    const float target = Ste / sqrtf(static_cast<float>(M_PI));
    float lam = 0.3f;  // initial guess

    for (int iter = 0; iter < 200; ++iter) {
        float exp_l2 = expf(lam * lam);
        float erf_l  = erff(lam);
        float f  = lam * exp_l2 * erf_l - target;
        float df = exp_l2 * erf_l
                 + lam * exp_l2 * (2.0f * lam * erf_l + 2.0f / sqrtf(static_cast<float>(M_PI)));
        float dlam = f / df;
        lam -= dlam;
        if (fabsf(dlam) < 1.0e-10f) break;
    }
    return lam;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // Material
    MaterialProperties mat = createPureMetal();
    const float T_melt = mat.T_solidus;          // 1000 K
    const float T_hot  = T_melt + 100.0f;        // 1100 K  (ΔT = 100 K)
    const float T_cold = T_melt;                  // 1000 K

    // Derived quantities
    const float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);   // 1.4286e-5 m²/s
    const float Ste   = mat.cp_solid * (T_hot - T_melt) / mat.L_fusion; // 0.5
    const float lam   = solveLambda(Ste);

    printf("=== Pure-Metal 1D Stefan Problem (ESM Validation) ===\n");
    printf("Material: %s\n", mat.name);
    printf("  T_solidus  = %.1f K,  T_liquidus = %.1f K  (ΔT_melt = %.1f K)\n",
           mat.T_solidus, mat.T_liquidus, mat.T_liquidus - mat.T_solidus);
    printf("  T_hot = %.1f K,  T_cold = %.1f K  (ΔT_wall = %.1f K)\n",
           T_hot, T_cold, T_hot - T_cold);
    printf("  cp = %.0f J/(kg·K),  k = %.0f W/(m·K),  ρ = %.0f kg/m³\n",
           mat.cp_solid, mat.k_solid, mat.rho_solid);
    printf("  L_fusion = %.0f J/kg\n", mat.L_fusion);
    printf("  α = %.6e m²/s\n", alpha);
    printf("  Stefan number Ste = %.4f\n", Ste);
    printf("  Neumann λ = %.6f\n", lam);

    // LBM discretization
    const int NX = 400;
    const int NY = 1;
    const int NZ = 1;
    const float domain_length = 2.0e-3f;                // 2 mm
    const float dx = domain_length / static_cast<float>(NX);
    const float tau = 0.8f;
    const float alpha_LU = (tau - 0.5f) * D3Q7::CS2;    // 0.075
    const float dt = alpha_LU * dx * dx / alpha;

    printf("\nLBM parameters:\n");
    printf("  NX = %d,  dx = %.3e m,  dt = %.6e s\n", NX, dx, dt);
    printf("  τ = %.2f,  ω = %.4f,  α_LU = %.4f\n", tau, 1.0f / tau, alpha_LU);

    // Snapshot times
    const float t_snap[] = {0.2e-3f, 0.5e-3f, 1.0e-3f, 1.5e-3f, 2.0e-3f};
    const int n_snap = 5;

    // Initialize D3Q7 lattice
    if (!D3Q7::isInitialized()) {
        D3Q7::initializeDevice();
    }

    // Create thermal solver with phase change
    ThermalLBM thermal(NX, NY, NZ, mat, alpha, true, dt, dx);
    thermal.initialize(T_cold);

    // Open CSV output
    FILE* fout = fopen("stefan_pure_metal.csv", "w");
    if (!fout) { fprintf(stderr, "Cannot open output file\n"); return 1; }

    // Header
    fprintf(fout, "# Pure-Metal Stefan Problem: ESM Validation\n");
    fprintf(fout, "# Ste=%.4f lambda=%.6f alpha=%.6e dx=%.6e NX=%d\n",
            Ste, lam, alpha, dx, NX);
    fprintf(fout, "# Columns: snapshot_index, x_m, T_K, fl, T_analytical_K, fl_analytical\n");

    std::vector<float> h_temp(NX);
    std::vector<float> h_fl(NX);

    int snap_idx = 0;
    int total_step = 0;
    float current_time = 0.0f;

    printf("\n%-6s  %10s  %10s  %10s  %8s\n",
           "snap", "t [ms]", "s_anal [µm]", "s_num [µm]", "err [%]");
    printf("------  ----------  -----------  ----------  --------\n");

    while (snap_idx < n_snap) {
        float t_target = t_snap[snap_idx];
        int steps_needed = static_cast<int>((t_target - current_time) / dt + 0.5f);

        for (int s = 0; s < steps_needed; ++s) {
            thermal.collisionBGK();
            thermal.streaming();
            thermal.computeTemperature();  // T* + ESM correction

            // Dirichlet BC at x=0 (face 0): T = T_hot
            // bc_type: 0=PERIODIC, 1=ADIABATIC, 2=DIRICHLET, 3=CONVECTIVE, 4=RADIATION
            // MUST be applied AFTER computeTemperature so the ESM correction
            // at the boundary is overwritten by the true wall temperature.
            thermal.applyFaceThermalBC(0, 2, dt, dx, T_hot);
            // x=NX-1: adiabatic (bounce-back, handled by streaming kernel)

            total_step++;
        }
        current_time = t_target;

        // Extract fields
        thermal.copyTemperatureToHost(h_temp.data());
        thermal.copyLiquidFractionToHost(h_fl.data());

        // Analytical front position
        float s_anal = 2.0f * lam * sqrtf(alpha * current_time);

        // Numerical front position (fl = 0.5 crossing, linear interpolation)
        float s_num = 0.0f;
        for (int i = 1; i < NX; ++i) {
            if (h_fl[i - 1] >= 0.5f && h_fl[i] < 0.5f) {
                float x0 = (i - 1) * dx;
                s_num = x0 + (0.5f - h_fl[i - 1]) / (h_fl[i] - h_fl[i - 1]) * dx;
                break;
            }
        }

        float err_pct = (s_anal > 0.0f) ? fabsf(s_num - s_anal) / s_anal * 100.0f : -1.0f;

        printf("  %d     %8.3f    %9.1f    %9.1f    %6.2f\n",
               snap_idx, current_time * 1e3f, s_anal * 1e6f, s_num * 1e6f, err_pct);

        // Write profile to CSV
        float sqrt_alpha_t = sqrtf(alpha * current_time);
        float erf_lam = erff(lam);

        for (int i = 0; i < NX; ++i) {
            float x = i * dx;

            // Analytical temperature and liquid fraction
            float T_anal, fl_anal;
            if (x < s_anal) {
                // Liquid region
                float eta = x / (2.0f * sqrt_alpha_t);
                T_anal = T_hot - (T_hot - T_melt) * erff(eta) / erf_lam;
                fl_anal = 1.0f;
            } else {
                // Solid region
                T_anal = T_melt;
                fl_anal = 0.0f;
            }

            fprintf(fout, "%d,%.8e,%.4f,%.6f,%.4f,%.6f\n",
                    snap_idx, x, h_temp[i], h_fl[i], T_anal, fl_anal);
        }

        snap_idx++;
    }

    fclose(fout);

    printf("\nCSV written to stefan_pure_metal.csv\n");
    printf("Run: python3 plot_stefan_pure_metal.py\n");

    return 0;
}
