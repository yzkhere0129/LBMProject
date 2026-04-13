/**
 * Phase 2.1: Pure Conduction Verification
 *
 * Test A: 3D Gaussian decay with Dirichlet BCs
 *   T(x,y,z,0) = 300 + 1000*exp(-r²/σ²) with σ=10 cells
 *   BCs: 6-face Dirichlet at 300K
 *   Track: T_max, total energy E = Σ(ρ·cp·T·dx³)
 *   E must decrease monotonically
 *
 * Test B: 1D sinusoidal decay (analytical solution)
 *   T(x,0) = 300 + 500·sin(π·x/L)
 *   BCs: T(0)=T(L)=300K (Dirichlet)
 *   Analytical: T(x,t) = 300 + 500·sin(πx/L)·exp(-α(π/L)²t)
 *   Compare L2 error
 *
 * Uses ThermalFDM directly — no FluidLBM, no VOF, no laser.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "physics/thermal_fdm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

int main() {
    // Use production parameters
    const float dx = 2.5e-6f;
    const float dt = 1.0e-8f;
    const float rho = 7900.0f, cp = 700.0f, k = 20.0f;
    const float alpha = k / (rho * cp);  // 3.617e-6 m²/s

    printf("================================================================\n");
    printf("  Phase 2.1: Pure Conduction Verification\n");
    printf("  dx=%.1fμm, dt=%.0fns, α=%.4e m²/s\n", dx*1e6, dt*1e9, alpha);
    printf("================================================================\n");

    // ====================================================================
    // TEST A: 3D Gaussian Decay
    // ====================================================================
    {
        printf("\n--- TEST A: 3D Gaussian Decay with Dirichlet BCs ---\n");
        const int NX = 40, NY = 40, NZ = 40;
        const int N = NX * NY * NZ;
        const float T_bc = 300.0f;
        const float T_peak = 1300.0f;  // peak at 300+1000=1300K (below solidus, no phase change)
        const float sigma_cells = 10.0f;
        const float sigma_m = sigma_cells * dx;

        MaterialProperties mat = {};
        mat.rho_solid = rho; mat.rho_liquid = rho;
        mat.cp_solid = cp; mat.cp_liquid = cp;
        mat.k_solid = k; mat.k_liquid = k;
        mat.T_solidus = 1650; mat.T_liquidus = 1700;
        mat.T_vaporization = 3200;

        ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

        // Initialize: Gaussian centered at domain center
        std::vector<float> h_T(N);
        float cx = NX * 0.5f, cy = NY * 0.5f, cz = NZ * 0.5f;
        float E_init = 0.0f;
        for (int k = 0; k < NZ; k++)
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++) {
                    float r2 = ((i+0.5f-cx)*(i+0.5f-cx) +
                                (j+0.5f-cy)*(j+0.5f-cy) +
                                (k+0.5f-cz)*(k+0.5f-cz)) * dx * dx;
                    float T = T_bc + (T_peak - T_bc) * expf(-r2 / (sigma_m * sigma_m));
                    h_T[i + j*NX + k*NX*NY] = T;
                    E_init += rho * cp * T * dx * dx * dx;
                }

        thermal.initialize(h_T.data());
        float T_max_prev = T_peak;
        float E_prev = E_init;
        bool monotonic_T = true, monotonic_E = true;

        printf("  Step    T_max [K]    E [J]      dE/E_init   T_mono  E_mono\n");
        printf("  ----    ---------    -----      ---------   ------  ------\n");

        int total_steps = 10000;
        for (int step = 1; step <= total_steps; step++) {
            // Apply Dirichlet BCs on all 6 faces
            for (int face = 0; face < 6; face++) {
                thermal.applyFaceThermalBC(face, 2, dt, dx, T_bc);
            }
            thermal.collisionBGK(nullptr, nullptr, nullptr);  // no velocity
            thermal.streaming();
            thermal.computeTemperature();

            if (step % 1000 == 0 || step == total_steps) {
                std::vector<float> h_T_out(N);
                thermal.copyTemperatureToHost(h_T_out.data());

                float T_max = 0, E_total = 0;
                for (int idx = 0; idx < N; idx++) {
                    if (h_T_out[idx] > T_max) T_max = h_T_out[idx];
                    E_total += rho * cp * h_T_out[idx] * dx * dx * dx;
                }

                if (T_max > T_max_prev + 0.01f) monotonic_T = false;
                if (E_total > E_prev + 1e-10f) monotonic_E = false;

                printf("  %5d   %9.2f    %.4e  %+.6f    %s    %s\n",
                       step, T_max, E_total,
                       (E_total - E_init) / E_init,
                       monotonic_T ? "OK" : "FAIL",
                       monotonic_E ? "OK" : "FAIL");

                T_max_prev = T_max;
                E_prev = E_total;
            }
        }

        printf("\n  RESULT: T_max monotonic = %s\n", monotonic_T ? "PASS" : "FAIL");
        printf("  RESULT: E monotonic     = %s\n", monotonic_E ? "PASS" : "FAIL");
        printf("  Final T_max = %.2f K (should approach %.0f K)\n", T_max_prev, T_bc);
        printf("  Energy change: %.4f%%\n", (E_prev - E_init) / E_init * 100);
    }

    // ====================================================================
    // TEST B: 1D Sinusoidal Decay (Analytical Comparison)
    // ====================================================================
    {
        printf("\n--- TEST B: 1D Sinusoidal Decay (Analytical Solution) ---\n");
        // Quasi-1D: NX=100, NY=1, NZ=1 with periodic in Y,Z
        // But ThermalFDM needs 3D. Use NX=100, NY=4, NZ=4 with adiabatic Y,Z
        const int NX = 100, NY = 4, NZ = 4;
        const int N = NX * NY * NZ;
        const float L = NX * dx;  // domain length in x
        const float T_bc = 300.0f;
        const float dT = 500.0f;

        MaterialProperties mat = {};
        mat.rho_solid = rho; mat.rho_liquid = rho;
        mat.cp_solid = cp; mat.cp_liquid = cp;
        mat.k_solid = k; mat.k_liquid = k;
        mat.T_solidus = 1650; mat.T_liquidus = 1700;
        mat.T_vaporization = 3200;

        ThermalFDM thermal(NX, NY, NZ, mat, alpha, false, dt, dx);

        // Initialize: T(x) = 300 + 500*sin(πx/L)
        std::vector<float> h_T(N);
        for (int kk = 0; kk < NZ; kk++)
            for (int jj = 0; jj < NY; jj++)
                for (int ii = 0; ii < NX; ii++) {
                    float x = (ii + 0.5f) * dx;
                    float T = T_bc + dT * sinf(M_PI * x / L);
                    h_T[ii + jj*NX + kk*NX*NY] = T;
                }

        thermal.initialize(h_T.data());

        // Analytical decay rate
        float decay_rate = alpha * (M_PI / L) * (M_PI / L);
        printf("  L = %.1f μm, decay rate = %.4e /s\n", L*1e6, decay_rate);
        printf("  τ_decay = %.2f μs\n", 1.0 / decay_rate * 1e6);

        printf("\n  Step     t[μs]   T_max_num   T_max_ana   Ratio    L2_err\n");
        printf("  ----     -----   ---------   ---------   -----    ------\n");

        int total_steps = 20000;
        for (int step = 1; step <= total_steps; step++) {
            // X BCs: Dirichlet 300K at x_min (face 0) and x_max (face 1)
            thermal.applyFaceThermalBC(0, 2, dt, dx, T_bc);  // x_min
            thermal.applyFaceThermalBC(1, 2, dt, dx, T_bc);  // x_max
            // Y,Z BCs: Adiabatic (face 2-5 → type 1)
            for (int face = 2; face < 6; face++)
                thermal.applyFaceThermalBC(face, 1, dt, dx, T_bc);

            thermal.collisionBGK(nullptr, nullptr, nullptr);
            thermal.streaming();
            thermal.computeTemperature();

            if (step % 2000 == 0 || step == total_steps) {
                float t = step * dt;
                float decay = expf(-decay_rate * t);
                float T_max_analytical = T_bc + dT * decay;

                std::vector<float> h_T_out(N);
                thermal.copyTemperatureToHost(h_T_out.data());

                // Find numerical T_max (should be at x=L/2)
                float T_max_num = 0;
                float L2_sum = 0, L2_norm = 0;
                int jj_mid = NY/2, kk_mid = NZ/2;
                for (int ii = 0; ii < NX; ii++) {
                    float x = (ii + 0.5f) * dx;
                    float T_ana = T_bc + dT * sinf(M_PI * x / L) * decay;
                    float T_num = h_T_out[ii + jj_mid*NX + kk_mid*NX*NY];
                    if (T_num > T_max_num) T_max_num = T_num;
                    L2_sum += (T_num - T_ana) * (T_num - T_ana);
                    L2_norm += (T_ana - T_bc) * (T_ana - T_bc);
                }
                float L2_err = (L2_norm > 0) ? sqrtf(L2_sum / L2_norm) : 0;
                float ratio = (T_max_analytical > T_bc + 1) ?
                    (T_max_num - T_bc) / (T_max_analytical - T_bc) : 1.0f;

                printf("  %5d   %6.1f   %9.2f   %9.2f   %5.3f    %.4f\n",
                       step, t*1e6, T_max_num, T_max_analytical, ratio, L2_err);
            }
        }

        // Final comparison
        float t_final = total_steps * dt;
        float decay_final = expf(-decay_rate * t_final);
        float T_max_ana_final = T_bc + dT * decay_final;

        std::vector<float> h_T_final(N);
        thermal.copyTemperatureToHost(h_T_final.data());
        float T_max_final = 0;
        float L2_sum = 0, L2_norm = 0;
        int jj_mid = NY/2, kk_mid = NZ/2;
        for (int ii = 0; ii < NX; ii++) {
            float x = (ii + 0.5f) * dx;
            float T_ana = T_bc + dT * sinf(M_PI * x / L) * decay_final;
            float T_num = h_T_final[ii + jj_mid*NX + kk_mid*NX*NY];
            if (T_num > T_max_final) T_max_final = T_num;
            L2_sum += (T_num - T_ana) * (T_num - T_ana);
            L2_norm += (T_ana - T_bc) * (T_ana - T_bc);
        }
        float L2_final = (L2_norm > 0) ? sqrtf(L2_sum / L2_norm) : 0;

        printf("\n  FINAL: T_max numerical=%.2f, analytical=%.2f, L2=%.4f\n",
               T_max_final, T_max_ana_final, L2_final);
        printf("  RESULT: %s (L2 < 0.02 = 2%%)\n", L2_final < 0.02 ? "PASS" : "FAIL");
    }

    printf("\n================================================================\n");
    printf("  Phase 2.1 Complete\n");
    printf("================================================================\n");
    return 0;
}
