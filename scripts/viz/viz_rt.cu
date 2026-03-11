/**
 * @file viz_rt.cu
 * @brief Rayleigh-Taylor instability visualization: generates CSV snapshots of
 *        the VOF fill level field for mushroom-cloud image generation.
 *
 * Physics: heavy fluid (f=1) on top, light fluid (f=0) below.
 * Gravity drives interface downward.  A cosine perturbation seeds the mode.
 *
 * Compile (from build/):
 *   nvcc -o ../scripts/viz/viz_rt ../scripts/viz/viz_rt.cu \
 *        -I../include -L. -llbm_physics -llbm_core \
 *        --std=c++17 -rdc=true -Xcompiler -fPIC -O2
 */

#include "physics/vof_solver.h"
#include "physics/fluid_lbm.h"
#include "physics/force_accumulator.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm::physics;

// ---- helpers ----------------------------------------------------------------

static void dumpSliceCSV(const std::string& path,
                          const std::vector<float>& fill,
                          int nx, int ny, int nz)
{
    // Take the middle z-slice (k = nz/2)
    int k0 = nz / 2;
    std::ofstream f(path);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * k0);
            f << fill[idx];
            if (i < nx - 1) f << ',';
        }
        f << '\n';
    }
}

static void initPerturbedInterface(VOFSolver& vof,
                                    int nx, int ny, int nz,
                                    float amp_cells, float y_iface)
{
    std::vector<float> h(nx * ny * nz);
    const float kx = 2.0f * M_PI / static_cast<float>(nx);
    const float w  = 2.0f;   // tanh half-width [cells]

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float y_if = y_iface + amp_cells * std::cos(kx * i);
                float dist = static_cast<float>(j) - y_if;
                // f=1 (heavy) above, f=0 (light) below
                h[i + nx*(j + ny*k)] = 0.5f * (1.0f + std::tanh(dist / w));
            }
    vof.initialize(h.data());
}

// ---- main -------------------------------------------------------------------

int main()
{
    // -------------------------------------------------------------------------
    // Domain: thin 3-D slab, quasi-2-D in x-y
    // -------------------------------------------------------------------------
    const int nx = 128, ny = 512, nz = 4;
    const float dx = 1e-4f;          // 100 µm / cell
    const int   N  = nx * ny * nz;

    // -------------------------------------------------------------------------
    // Physical parameters
    // -------------------------------------------------------------------------
    const float rho_h = 1000.0f;    // heavy fluid density  [kg/m³]
    const float rho_l =  500.0f;    // light fluid density  [kg/m³]
    // Atwood = 0.333
    const float nu    = 5e-6f;      // kinematic viscosity [m²/s]
    const float g     = 9.81f;      // gravity [m/s²]

    // -------------------------------------------------------------------------
    // Time step via LBM stability: tau=0.6 → nu_lbm = (tau-0.5)/3
    // nu_phys = nu_lbm * dx² / dt  →  dt = nu_lbm * dx² / nu_phys
    // -------------------------------------------------------------------------
    const float tau_f = 0.6f;
    const float nu_lbm = (tau_f - 0.5f) / 3.0f;    // 1/30 lattice units
    float dt = nu_lbm * dx * dx / nu;
    // CFL safety: max lattice speed ≈ 0.05 → physical = 0.05*dx/dt
    // Just cap dt to guarantee stability
    dt = std::min(dt, 5e-5f);

    const float y_iface = ny * 0.5f;
    const float amp0    = 3.0f;     // perturbation amplitude [cells]

    // Snapshot schedule: capture enough frames to see mushroom formation
    // We target ~8000 steps total; output every 1000 steps
    const int n_steps  = 8000;
    const int out_freq = 1000;

    std::printf("RT Visualization\n");
    std::printf("  Grid: %d x %d x %d\n", nx, ny, nz);
    std::printf("  dx=%.1e m  dt=%.2e s  steps=%d\n", dx, dt, n_steps);
    std::printf("  At=%.3f  nu=%.2e  g=%.2f\n",
                (rho_h-rho_l)/(rho_h+rho_l), nu, g);

    // -------------------------------------------------------------------------
    // Solvers
    // -------------------------------------------------------------------------
    FluidLBM fluid(nx, ny, nz, nu, rho_h,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   dt, dx);
    fluid.initialize(rho_h, 0.0f, 0.0f, 0.0f);

    VOFSolver vof(nx, ny, nz, dx);
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);
    initPerturbedInterface(vof, nx, ny, nz, amp0, y_iface);
    vof.reconstructInterface();
    vof.computeCurvature();

    ForceAccumulator forces(nx, ny, nz);

    // Device velocity buffers for physical-unit advection
    float *d_ux_p, *d_uy_p, *d_uz_p;
    cudaMalloc(&d_ux_p, N * sizeof(float));
    cudaMalloc(&d_uy_p, N * sizeof(float));
    cudaMalloc(&d_uz_p, N * sizeof(float));

    std::vector<float> h_ux(N), h_uy(N), h_uz(N), h_fill(N);
    const float v_conv = dx / dt;   // lattice→physical velocity

    // Output directory
    const char* out_dir = "/home/yzk/LBMProject/scripts/viz";

    // ---- Dump initial state ------------------------------------------------
    vof.copyFillLevelToHost(h_fill.data());
    dumpSliceCSV(std::string(out_dir) + "/rt_step0000.csv", h_fill, nx, ny, nz);
    std::printf("  Saved rt_step0000.csv\n");

    // ---- Time loop ---------------------------------------------------------
    for (int step = 1; step <= n_steps; ++step)
    {
        // VOF buoyancy force
        forces.reset();
        forces.addVOFBuoyancyForce(vof.getFillLevel(), rho_h, rho_l, 0.0f, -g, 0.0f);
        forces.convertToLatticeUnits(dx, dt, rho_h);

        // LBM collision + streaming
        fluid.collisionTRT(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();
        fluid.computeMacroscopic();

        // Convert lattice velocity to physical units on host
        cudaMemcpy(h_ux.data(), fluid.getVelocityX(), N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), fluid.getVelocityY(), N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uz.data(), fluid.getVelocityZ(), N*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            h_ux[i] *= v_conv;
            h_uy[i] *= v_conv;
            h_uz[i] *= v_conv;
        }
        cudaMemcpy(d_ux_p, h_ux.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uy_p, h_uy.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uz_p, h_uz.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        // VOF advection
        vof.advectFillLevel(d_ux_p, d_uy_p, d_uz_p, dt);
        vof.reconstructInterface();
        vof.computeCurvature();

        if (step % out_freq == 0) {
            vof.copyFillLevelToHost(h_fill.data());
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "%s/rt_step%04d.csv", out_dir, step);
            dumpSliceCSV(fname, h_fill, nx, ny, nz);
            std::printf("  step %5d / %d  ->  %s\n", step, n_steps, fname);
        }
    }

    cudaFree(d_ux_p);
    cudaFree(d_uy_p);
    cudaFree(d_uz_p);

    std::printf("Done.  CSV files written to %s/rt_step*.csv\n", out_dir);
    return 0;
}
