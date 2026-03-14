/**
 * @file viz_phase_inversion.cu
 * @brief Phase Inversion: At=0.5 Rayleigh-Taylor with 10:1 viscosity ratio
 *
 * Classic heavy-over-light RT instability run to FULL INVERSION.
 *
 * Parameters (all lattice units, dx=1, dt=1):
 *   At  = 0.5  (rho_H=3, rho_L=1)
 *   g_LB = 1e-5  (Ma ≈ 0.06, safely incompressible)
 *   mu_H/mu_L = 10:1  (variable-omega TRT)
 *   tau range [0.554, 0.68]
 *   sigma = 0  (pure RT, no surface tension)
 *   PLIC geometric advection
 *   WALL y, PERIODIC x/z
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

#include "physics/fluid_lbm.h"
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "core/lattice_d3q19.h"

using lbm::physics::FluidLBM;
using lbm::physics::VOFSolver;
using lbm::physics::VOFAdvectionScheme;
using lbm::physics::ForceAccumulator;
using lbm::physics::BoundaryType;
using lbm::core::D3Q19;

// Grid
static constexpr int NX = 128;
static constexpr int NY = 512;
static constexpr int NZ = 1;  // Strict 2D
static constexpr int NC = NX * NY * NZ;

// Physics — At=0.5 with low g for Ma < 0.1
static constexpr float RHO_H = 3.0f;
static constexpr float RHO_L = 1.0f;
static constexpr float AT    = (RHO_H - RHO_L) / (RHO_H + RHO_L);  // 0.5
static constexpr float G_LB  = 1e-5f;

// Viscosity — 10:1 dynamic viscosity ratio
// nu_L = 0.018 (tau_L = 0.554), nu_H = 0.06 (tau_H = 0.68)
// mu_L = nu_L * rho_L = 0.018, mu_H = nu_H * rho_H = 0.18 → ratio = 10
static constexpr float NU_L  = 0.018f;
static constexpr float MU_L  = NU_L * RHO_L;   // 0.018
static constexpr float MU_H  = 10.0f * MU_L;   // 0.18
static constexpr float NU_H  = MU_H / RHO_H;   // 0.06

static constexpr float TAU_L = 3.0f * NU_L + 0.5f;  // 0.554
static constexpr float TAU_H = 3.0f * NU_H + 0.5f;  // 0.68

// Lattice units
static constexpr float DX = 1.0f;
static constexpr float DT = 1.0f;

// Interface: heavy on top at 75% height, cosine perturbation
static constexpr float Y_INTF = 0.75f * NY;   // 384
static constexpr float AMP    = 0.05f * NX;   // 6.4 cells

// Timing
static constexpr int TOTAL_STEPS = 25000;
static constexpr int KEYFRAME_STEPS[] = {3000, 8000, 14000, 25000};
static constexpr int N_KEYFRAMES = 4;

static void dumpField(const char* path, int step,
                      const float* h_fill, const float* h_vx, const float* h_vy) {
    FILE* fp = fopen(path, "w");
    fprintf(fp, "ix,iy,x_mm,y_mm,fill_level,vx_ms,vy_ms,vmag_ms\n");
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX * j;
            float f = h_fill[idx];
            float vx = h_vx[idx];
            float vy = h_vy[idx];
            float vm = sqrtf(vx*vx + vy*vy);
            fprintf(fp, "%d,%d,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f\n",
                    i, j, (float)i, (float)j, f, vx, vy, vm);
        }
    fclose(fp);
    printf("  Wrote %s (step=%d)\n", path, step);
}

int main() {
    cudaSetDevice(0);
    auto wall0 = std::chrono::high_resolution_clock::now();

    float U_c = sqrtf(G_LB * NX);
    float Ma  = U_c * sqrtf(3.0f);
    float Re  = NX * U_c / NU_L;

    printf("=== Phase Inversion: At=0.5 RT with 10:1 Viscosity ===\n");
    printf("Grid: %dx%dx%d (strict 2D)\n", NX, NY, NZ);
    printf("At=%.2f  rho_H=%.1f  rho_L=%.1f\n", AT, RHO_H, RHO_L);
    printf("g_LB=%.1e  U_c=%.4f  Ma=%.3f  Re=%.0f\n", G_LB, U_c, Ma, Re);
    printf("mu_H/mu_L = %.1f  (mu_H=%.3f, mu_L=%.3f)\n", MU_H/MU_L, MU_H, MU_L);
    printf("tau_L=%.3f  tau_H=%.3f\n", TAU_L, TAU_H);
    printf("Interface: y=%.0f  amplitude=%.1f cells\n", Y_INTF, AMP);
    printf("Steps: %d  sigma=0 (pure RT)\n\n", TOTAL_STEPS);
    fflush(stdout);

    if (!D3Q19::isInitialized()) D3Q19::initializeDevice();

    // Fluid: WALL in y, PERIODIC in x/z
    FluidLBM fluid(NX, NY, NZ, NU_L, RHO_H,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   DT, DX);
    fluid.initialize(RHO_H, 0.0f, 0.0f, 0.0f);

    // VOF: PLIC, WALL in y, PERIODIC in x/z
    VOFSolver vof(NX, NY, NZ, DX,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::PERIODIC);

    // IC: heavy (f=1) on top, light (f=0) on bottom, cosine perturbation
    {
        std::vector<float> h_fill(NC);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * j;
                float x_norm = static_cast<float>(i) / NX;
                float y_intf = Y_INTF + AMP * cosf(2.0f * M_PI * x_norm);
                float dist = static_cast<float>(j) - y_intf;
                h_fill[idx] = 0.5f * (1.0f + tanhf(dist / 2.0f));
            }
        vof.initialize(h_fill.data());
    }
    vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
    vof.reconstructInterface();
    vof.computeCurvature();

    ForceAccumulator forces(NX, NY, NZ);

    std::vector<float> h_fill(NC), h_vx(NC), h_vy(NC);
    float initial_mass = vof.computeTotalMass();

    int next_kf = 0;
    int print_every = 500;

    for (int step = 0; step <= TOTAL_STEPS; ++step) {
        // Keyframe dump
        if (next_kf < N_KEYFRAMES && step >= KEYFRAME_STEPS[next_kf]) {
            cudaMemcpy(h_fill.data(), vof.getFillLevel(), NC*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vx.data(), fluid.getVelocityX(), NC*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy.data(), fluid.getVelocityY(), NC*sizeof(float), cudaMemcpyDeviceToHost);

            char path[256];
            snprintf(path, sizeof(path), "scripts/viz/phase_inversion_step%05d.csv",
                     KEYFRAME_STEPS[next_kf]);
            dumpField(path, step, h_fill.data(), h_vx.data(), h_vy.data());
            next_kf++;
        }

        // Progress
        if (step % print_every == 0) {
            cudaMemcpy(h_vx.data(), fluid.getVelocityX(), NC*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy.data(), fluid.getVelocityY(), NC*sizeof(float), cudaMemcpyDeviceToHost);
            float vmax = 0.0f;
            for (int i = 0; i < NC; ++i) {
                float v = sqrtf(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i]);
                vmax = fmaxf(vmax, v);
            }
            float mass = vof.computeTotalMass();
            float merr = fabsf(mass - initial_mass) / initial_mass * 100.0f;
            printf("  step %5d/%d  Ma=%.4f  mass_err=%.4f%%\n",
                   step, TOTAL_STEPS, vmax/0.577f, merr);
            fflush(stdout);
        }

        if (step >= TOTAL_STEPS) break;

        // Forces (lattice units, no conversion needed)
        forces.reset();
        forces.addVOFBuoyancyForce(vof.getFillLevel(), RHO_H, RHO_L,
                                    0.0f, -G_LB, 0.0f);
        // No surface tension (pure RT)

        // Variable viscosity: 10:1 dynamic viscosity ratio
        fluid.computeVariableViscosity(vof.getFillLevel(),
                                        RHO_H, RHO_L, MU_H, MU_L);

        // TRT collision with per-cell omega
        fluid.collisionTRTVariable(forces.getFx(), forces.getFy(), forces.getFz(),
                                    vof.getFillLevel(), RHO_H, RHO_L);
        fluid.streaming();
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(), forces.getFz());

        // VOF advection (lattice units: dx=1, dt=1, velocities already in LU)
        vof.advectFillLevel(fluid.getVelocityX(), fluid.getVelocityY(),
                            fluid.getVelocityZ(), DT);
        vof.reconstructInterface();
        vof.computeCurvature();
    }

    auto wall1 = std::chrono::high_resolution_clock::now();
    printf("\nDone in %.1f s\n", std::chrono::duration<float>(wall1 - wall0).count());
    return 0;
}
