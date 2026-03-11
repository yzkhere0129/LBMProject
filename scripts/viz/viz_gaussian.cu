/**
 * @file viz_gaussian.cu
 * @brief Visualization data generator for 3D Gaussian heat diffusion
 *
 * Initializes a Gaussian temperature distribution and runs pure diffusion
 * using ThermalLBM (D3Q7). Dumps the z-midplane temperature field at:
 *   - t = 0 (initial Gaussian)
 *   - t = final (diffused Gaussian)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "physics/thermal_lbm.h"

using namespace lbm::physics;

int main() {
    // ========================================================================
    // Domain and physics setup
    // ========================================================================
    const int n = 65;            // Grid size per dimension
    const float dx = 2.0e-5f;   // 20 um
    const int num_cells = n * n * n;

    // Simple material (for clean diffusion physics)
    const float rho = 4430.0f;   // Ti6Al4V-like density [kg/m3]
    const float cp = 526.0f;     // [J/(kg*K)]
    const float k = 6.7f;        // [W/(m*K)]
    const float alpha = k / (rho * cp);  // thermal diffusivity [m2/s]

    // Time step chosen for stable omega (D3Q7: tau = 4*alpha_lattice + 0.5)
    // alpha_lattice = alpha * dt / dx^2
    // For omega = 1/tau ~ 1.5 (safe): tau ~ 0.667, alpha_lattice = (tau-0.5)/4 = 0.0417
    // dt = alpha_lattice * dx^2 / alpha = 0.0417 * (2e-5)^2 / (2.874e-6) = 5.8e-6 s
    const float dt = 5.0e-6f;
    const int num_steps = 400;
    const float final_time = num_steps * dt;

    // Verify lattice parameters
    float alpha_lattice = alpha * dt / (dx * dx);
    float tau = 4.0f * alpha_lattice + 0.5f;
    float omega = 1.0f / tau;

    printf("=== Gaussian Heat Diffusion Visualization ===\n");
    printf("Grid: %d^3 = %d cells\n", n, num_cells);
    printf("dx = %.1f um, dt = %.1f us\n", dx * 1e6f, dt * 1e6f);
    printf("alpha = %.4e m^2/s\n", alpha);
    printf("alpha_lattice = %.4f, tau = %.4f, omega = %.4f\n",
           alpha_lattice, tau, omega);
    printf("Simulation: %d steps, %.1f us total\n", num_steps, final_time * 1e6f);

    // ========================================================================
    // Gaussian parameters
    // ========================================================================
    const float T0 = 300.0f;          // Background temperature [K]
    const float A = 1500.0f;          // Peak amplitude [K] (vivid visualization)
    const float sigma0 = 3.5f * dx;   // Initial width ~ 3.5 cells
    const int i_center = n / 2;
    const float center = i_center * dx;

    // Analytical solution for sigma(t)
    float sigma_final = sqrtf(sigma0 * sigma0 + 2.0f * alpha * final_time);
    float ratio = sigma0 / sigma_final;
    float peak_final = T0 + A * ratio * ratio * ratio;  // 3D: (sigma0/sigma)^3

    printf("\nGaussian: T0=%.0f K, A=%.0f K, sigma0=%.1f um\n",
           T0, A, sigma0 * 1e6f);
    printf("Center: cell %d (%.1f um)\n", i_center, center * 1e6f);
    printf("Initial peak: %.0f K\n", T0 + A);
    printf("Expected final peak: %.1f K\n", peak_final);
    printf("Width broadening: %.1f -> %.1f um\n",
           sigma0 * 1e6f, sigma_final * 1e6f);

    // ========================================================================
    // Initialize thermal solver (8-arg constructor with dt, dx)
    // ========================================================================
    ThermalLBM thermal(n, n, n, alpha, rho, cp, dt, dx);

    // Set up initial Gaussian temperature field
    std::vector<float> h_temp(num_cells);
    for (int kk = 0; kk < n; ++kk) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                float x = i * dx;
                float y = j * dx;
                float z = kk * dx;
                float ddx = x - center;
                float ddy = y - center;
                float ddz = z - center;
                float r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                int idx = i + n * (j + n * kk);
                h_temp[idx] = T0 + A * expf(-r2 / (2.0f * sigma0 * sigma0));
            }
        }
    }

    thermal.initialize(h_temp.data());

    // ========================================================================
    // Dump initial z-midplane temperature
    // ========================================================================
    int z_mid = n / 2;
    printf("\nWriting initial temperature z-slice (z=%d) ...\n", z_mid);

    FILE* f0 = fopen("temp_initial.csv", "w");
    if (!f0) { fprintf(stderr, "Cannot open temp_initial.csv\n"); return 1; }
    // Header: x positions
    for (int i = 0; i < n; ++i) {
        fprintf(f0, "%.2f", i * dx * 1e6f);
        if (i < n - 1) fprintf(f0, ",");
    }
    fprintf(f0, "\n");
    // Data: y rows
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = i + n * (j + n * z_mid);
            fprintf(f0, "%.4f", h_temp[idx]);
            if (i < n - 1) fprintf(f0, ",");
        }
        fprintf(f0, "\n");
    }
    fclose(f0);
    printf("  -> temp_initial.csv written\n");

    // ========================================================================
    // Run diffusion simulation
    // ========================================================================
    printf("\nRunning %d diffusion steps ...\n", num_steps);

    for (int step = 0; step < num_steps; ++step) {
        // Pure diffusion: adiabatic BC (type 0), no velocity coupling
        thermal.applyBoundaryConditions(0, T0);
        thermal.computeTemperature();
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();

        if ((step + 1) % 100 == 0) {
            thermal.copyTemperatureToHost(h_temp.data());
            float peak = *std::max_element(h_temp.begin(), h_temp.end());
            printf("  Step %4d/%d: T_peak = %.1f K\n", step + 1, num_steps, peak);
        }
    }

    // ========================================================================
    // Dump final z-midplane temperature
    // ========================================================================
    thermal.copyTemperatureToHost(h_temp.data());
    float actual_peak = *std::max_element(h_temp.begin(), h_temp.end());
    printf("\nFinal peak: %.1f K (expected: %.1f K)\n", actual_peak, peak_final);

    printf("Writing final temperature z-slice (z=%d) ...\n", z_mid);

    FILE* f1 = fopen("temp_final.csv", "w");
    if (!f1) { fprintf(stderr, "Cannot open temp_final.csv\n"); return 1; }
    for (int i = 0; i < n; ++i) {
        fprintf(f1, "%.2f", i * dx * 1e6f);
        if (i < n - 1) fprintf(f1, ",");
    }
    fprintf(f1, "\n");
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = i + n * (j + n * z_mid);
            fprintf(f1, "%.4f", h_temp[idx]);
            if (i < n - 1) fprintf(f1, ",");
        }
        fprintf(f1, "\n");
    }
    fclose(f1);
    printf("  -> temp_final.csv written\n");

    // Also dump analytical solution for reference
    FILE* fa = fopen("temp_analytical.csv", "w");
    if (!fa) { fprintf(stderr, "Cannot open temp_analytical.csv\n"); return 1; }
    for (int i = 0; i < n; ++i) {
        fprintf(fa, "%.2f", i * dx * 1e6f);
        if (i < n - 1) fprintf(fa, ",");
    }
    fprintf(fa, "\n");
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            float x = i * dx;
            float y = j * dx;
            float z = z_mid * dx;
            float ddx = x - center;
            float ddy = y - center;
            float ddz = z - center;
            float r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            float T_anal = T0 + A * ratio * ratio * ratio
                           * expf(-r2 / (2.0f * sigma_final * sigma_final));
            fprintf(fa, "%.4f", T_anal);
            if (i < n - 1) fprintf(fa, ",");
        }
        fprintf(fa, "\n");
    }
    fclose(fa);
    printf("  -> temp_analytical.csv written\n");

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n=== Done ===\n");
    printf("Files generated:\n");
    printf("  temp_initial.csv    - Initial Gaussian (z-midplane)\n");
    printf("  temp_final.csv      - Diffused field (z-midplane)\n");
    printf("  temp_analytical.csv - Analytical solution at t=final\n");

    return 0;
}
