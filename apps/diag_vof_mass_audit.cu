/**
 * @file diag_vof_mass_audit.cu
 * @brief VOF Mass Conservation "Closed Box" Audit
 *
 * Translating metal sphere (r=20μm) in a 100μm³ periodic box.
 * No source terms: no laser, no evaporation, no gravity.
 * Measures total ΣF drift over 5000 steps with PLIC and TVD advection.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "physics/vof_solver.h"

using namespace lbm::physics;

int main() {
    // Domain: 50×50×50 at dx=2μm (100μm cube)
    const int N = 50;
    const float dx = 2.0e-6f;
    const float dt = 1.0e-8f;
    const int num_cells = N * N * N;
    const int total_steps = 5000;

    // Sphere: r=10 cells (20μm) at center
    const float cx = N * 0.5f, cy = N * 0.5f, cz = N * 0.5f;
    const float r = 10.0f;

    // Velocity: u_x = 40 m/s → CFL = u*dt/dx = 0.2
    const float u_phys = 40.0f;
    const float cfl = u_phys * dt / dx;

    printf("============================================================\n");
    printf("  VOF Mass Conservation Closed-Box Audit\n");
    printf("============================================================\n");
    printf("Domain: %d³ = %d cells (dx=%.0fμm, PERIODIC)\n", N, num_cells, dx*1e6f);
    printf("Sphere: r=%.0f cells (%.0fμm), center=(%.0f,%.0f,%.0f)\n",
           r, r*dx*1e6f, cx, cy, cz);
    printf("Velocity: u_x=%.0f m/s, CFL=%.3f\n", u_phys, cfl);
    printf("Steps: %d (sphere traverses %.0f domain widths)\n\n",
           total_steps, cfl * total_steps / N);

    // Initialize sphere fill level on host
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int iz = 0; iz < N; iz++)
        for (int iy = 0; iy < N; iy++)
            for (int ix = 0; ix < N; ix++) {
                float ddx = ix + 0.5f - cx;
                float ddy = iy + 0.5f - cy;
                float ddz = iz + 0.5f - cz;
                float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
                float f = r + 0.5f - dist;
                f = fmaxf(0.0f, fminf(1.0f, f));
                h_fill[ix + iy*N + iz*N*N] = f;
            }

    // Count initial cell types
    int n_full = 0, n_iface = 0;
    float vol_analytical = (4.0f/3.0f) * 3.14159265f * r*r*r;
    for (auto f : h_fill) {
        if (f > 0.99f) n_full++;
        else if (f > 0.01f) n_iface++;
    }
    printf("Initial: %d full + %d interface cells\n", n_full, n_iface);
    printf("Analytical volume: %.1f cells³, ΣF analytical: ~%.1f\n\n", vol_analytical, vol_analytical);

    // Allocate device velocity (uniform u_x)
    float *d_vx, *d_vy, *d_vz;
    cudaMalloc(&d_vx, num_cells * sizeof(float));
    cudaMalloc(&d_vy, num_cells * sizeof(float));
    cudaMalloc(&d_vz, num_cells * sizeof(float));
    {
        std::vector<float> h_vx(num_cells, u_phys);
        std::vector<float> h_vy(num_cells, 0.0f);
        std::vector<float> h_vz(num_cells, 0.0f);
        cudaMemcpy(d_vx, h_vx.data(), num_cells*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy, h_vy.data(), num_cells*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz, h_vz.data(), num_cells*sizeof(float), cudaMemcpyHostToDevice);
    }

    // ============================================================
    // Test A: PLIC advection
    // ============================================================
    printf("=== Test A: PLIC Advection ===\n");
    {
        VOFSolver vof(N, N, N, dx,
                      VOFSolver::BoundaryType::PERIODIC,
                      VOFSolver::BoundaryType::PERIODIC,
                      VOFSolver::BoundaryType::PERIODIC);
        vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
        vof.initialize(h_fill.data());

        float mass0 = vof.computeTotalMass();
        printf("Initial ΣF = %.6f\n", mass0);
        printf("%-6s  %14s  %+12s  %12s\n", "Step", "ΣF", "ΔΣF", "ΔΣF/ΣF₀[%]");
        printf("-------------------------------------------------------\n");

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int step = 1; step <= total_steps; step++) {
            vof.advectFillLevel(d_vx, d_vy, d_vz, dt);

            if (step % 500 == 0 || step == total_steps) {
                float mass = vof.computeTotalMass();
                float delta = mass - mass0;
                float err_pct = delta / mass0 * 100.0f;
                printf("%6d  %14.6f  %+12.6f  %+12.6f\n", step, mass, delta, err_pct);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(t1 - t0).count();

        float mass_final = vof.computeTotalMass();
        printf("PLIC final: ΣF=%.6f, error=%.6f%%, time=%.1fs\n\n",
               mass_final, (mass_final-mass0)/mass0*100.0f, elapsed);

        // Analyze final fill level distribution
        std::vector<float> h_f_final(num_cells);
        vof.copyFillLevelToHost(h_f_final.data());
        int n_neg = 0, n_over = 0, n_tiny = 0;
        float f_min = 1e9f, f_max = -1e9f;
        for (int i = 0; i < num_cells; i++) {
            float f = h_f_final[i];
            if (f < 0.0f) n_neg++;
            if (f > 1.0f) n_over++;
            if (f > 0.0f && f < 0.01f) n_tiny++;
            f_min = fminf(f_min, f);
            f_max = fmaxf(f_max, f);
        }
        printf("Fill level bounds: [%.6f, %.6f]\n", f_min, f_max);
        printf("Cells f<0: %d, f>1: %d, 0<f<0.01: %d\n\n", n_neg, n_over, n_tiny);
    }

    // ============================================================
    // Test B: TVD + MC advection
    // ============================================================
    printf("=== Test B: TVD + MC Limiter ===\n");
    {
        VOFSolver vof(N, N, N, dx,
                      VOFSolver::BoundaryType::PERIODIC,
                      VOFSolver::BoundaryType::PERIODIC,
                      VOFSolver::BoundaryType::PERIODIC);
        vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
        vof.setTVDLimiter(TVDLimiter::MC);
        vof.initialize(h_fill.data());

        float mass0 = vof.computeTotalMass();
        printf("Initial ΣF = %.6f\n", mass0);
        printf("%-6s  %14s  %+12s  %12s\n", "Step", "ΣF", "ΔΣF", "ΔΣF/ΣF₀[%]");
        printf("-------------------------------------------------------\n");

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int step = 1; step <= total_steps; step++) {
            vof.advectFillLevel(d_vx, d_vy, d_vz, dt);

            if (step % 500 == 0 || step == total_steps) {
                float mass = vof.computeTotalMass();
                float delta = mass - mass0;
                float err_pct = delta / mass0 * 100.0f;
                printf("%6d  %14.6f  %+12.6f  %+12.6f\n", step, mass, delta, err_pct);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(t1 - t0).count();

        float mass_final = vof.computeTotalMass();
        printf("TVD+MC final: ΣF=%.6f, error=%.6f%%, time=%.1fs\n\n",
               mass_final, (mass_final-mass0)/mass0*100.0f, elapsed);
    }

    // ============================================================
    // Test C: PLIC with WALL boundary (matches production)
    // ============================================================
    printf("=== Test C: PLIC + WALL Boundary ===\n");
    {
        VOFSolver vof(N, N, N, dx,
                      VOFSolver::BoundaryType::WALL,
                      VOFSolver::BoundaryType::WALL,
                      VOFSolver::BoundaryType::WALL);
        vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
        vof.initialize(h_fill.data());

        float mass0 = vof.computeTotalMass();
        printf("Initial ΣF = %.6f\n", mass0);
        printf("%-6s  %14s  %+12s  %12s\n", "Step", "ΣF", "ΔΣF", "ΔΣF/ΣF₀[%]");
        printf("-------------------------------------------------------\n");

        for (int step = 1; step <= total_steps; step++) {
            vof.advectFillLevel(d_vx, d_vy, d_vz, dt);

            if (step % 500 == 0 || step == total_steps) {
                float mass = vof.computeTotalMass();
                float delta = mass - mass0;
                float err_pct = delta / mass0 * 100.0f;
                printf("%6d  %14.6f  %+12.6f  %+12.6f\n", step, mass, delta, err_pct);
            }
        }

        float mass_final = vof.computeTotalMass();
        printf("PLIC+WALL final: ΣF=%.6f, error=%.6f%%\n\n",
               mass_final, (mass_final-mass0)/mass0*100.0f);
    }

    printf("============================================================\n");
    printf("  Audit Complete\n");
    printf("============================================================\n");

    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    return 0;
}
