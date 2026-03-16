/**
 * @file viz_rt_plic.cu
 * @brief Classic Re=256 Rayleigh-Taylor instability benchmark (strict 2D)
 *
 * Pure PLIC-VOF + TRT + Guo forcing. NO surface tension. NO mass correction.
 * Mass conservation relies solely on geometric face fluxes.
 *
 * Parameters (all lattice units, dx=1, dt=1):
 *   Re = NX * sqrt(g_LB * NX) / nu_LB = 256
 *   At = 0.5  (rho_H=3, rho_L=1)
 *   sigma = 0
 *   Interface at y = 0.75*NY = 384
 *   Perturbation: 0.05*NX * cos(2*pi*x/NX)
 *   BCs: x=periodic, y=wall, z=periodic
 *   STRICT 2D: NZ = 1 (no cross-plane pollution)
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

// ======================== Grid ========================
static constexpr int NX = 128;
static constexpr int NY = 512;
static constexpr int NZ = 1;   // STRICT 2D — no z-direction artifacts
static constexpr int NC = NX * NY * NZ;

// ======================== Physics ========================
static constexpr float RHO_HEAVY = 3.0f;
static constexpr float RHO_LIGHT = 1.0f;
static constexpr float AT = (RHO_HEAVY - RHO_LIGHT) / (RHO_HEAVY + RHO_LIGHT); // 0.5

// Re=256 derivation:
//   g_LB  = 1e-4
//   U_c   = sqrt(g_LB * NX) = sqrt(0.0128) = 0.11314
//   nu_LB = NX * U_c / Re = 128 * 0.11314 / 256 = 0.05657
//   tau   = 3*nu_LB + 0.5 = 0.6697
//   Ma    = U_c * sqrt(3) = 0.196
static constexpr float G_LB  = 1e-4f;
static constexpr float NU_LB = 0.05657f;
static constexpr float TAU   = 3.0f * NU_LB + 0.5f;
static constexpr float DX    = 1.0f;
static constexpr float DT    = 1.0f;

// Interface
static constexpr float Y_INTERFACE = 0.50f * NY;   // 256 (1:1 heavy:light)
static constexpr float AMPLITUDE   = 0.05f * NX;   // 6.4 cells

// Timing — finer snapshots for better frame selection
static constexpr int TOTAL_STEPS   = 20000;
static constexpr int SNAP_INTERVAL = 1000;

// ======================== Main ========================
int main() {
    cudaSetDevice(0);
    auto t0 = std::chrono::high_resolution_clock::now();

    float U_char = sqrtf(G_LB * NX);
    float Re = NX * U_char / NU_LB;
    float Ma = U_char * sqrtf(3.0f);

    printf("=== RT Instability: Classic Re=256, At=0.5, sigma=0, STRICT 2D ===\n");
    printf("Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("At=%.2f  rho_H=%.1f  rho_L=%.1f\n", AT, RHO_HEAVY, RHO_LIGHT);
    printf("tau=%.4f  nu_LB=%.5f  g_LB=%.1e  sigma=0\n", TAU, NU_LB, G_LB);
    printf("Re=%.0f  Ma=%.3f\n", Re, Ma);
    printf("Interface: y=%.0f  amplitude=%.1f cells\n", Y_INTERFACE, AMPLITUDE);
    printf("Steps: %d  snap every %d\n", TOTAL_STEPS, SNAP_INTERVAL);
    printf("Mass correction: DISABLED (geometric fluxes only)\n\n");
    fflush(stdout);

    if (!D3Q19::isInitialized()) D3Q19::initializeDevice();

    // Fluid: TRT, wall in y, periodic in x/z
    FluidLBM fluid(NX, NY, NZ, NU_LB, RHO_HEAVY,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   DT, DX);
    fluid.initialize(RHO_HEAVY, 0.0f, 0.0f, 0.0f);

    // VOF: PLIC geometric advection — NO mass correction
    VOFSolver vof(NX, NY, NZ, DX,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::PERIODIC);
    vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);
    // Mass correction explicitly DISABLED — rely on geometric fluxes only

    // Fill level: heavy (f=1) on top, light (f=0) on bottom
    {
        std::vector<float> h_fill(NC);
        const float k = 2.0f * M_PI / (float)NX;
        const float w = 2.0f;  // tanh smoothing width

        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++) {
                int idx = ix + NX * iy;
                float y_if = Y_INTERFACE + AMPLITUDE * cosf(k * ix);
                float dist = (float)iy - y_if;
                h_fill[idx] = 0.5f * (1.0f + tanhf(dist / w));
            }
        vof.initialize(h_fill.data());
    }

    ForceAccumulator forces(NX, NY, NZ);

    std::vector<float> h_fill(NC);
    float mass0 = vof.computeTotalMass();

    (void)system("rm -f /home/yzk/LBMProject/scripts/viz/rt_plic_data/rt_step*.csv");
    (void)system("mkdir -p /home/yzk/LBMProject/scripts/viz/rt_plic_data");

    auto saveSnapshot = [&](int step) {
        vof.copyFillLevelToHost(h_fill.data());
        float mass = vof.computeTotalMass();
        float mass_err = fabsf(mass - mass0) / mass0 * 100.0f;
        printf("  step %5d  mass_err=%.4f%%\n", step, mass_err);

        char fname[256];
        snprintf(fname, sizeof(fname),
                 "/home/yzk/LBMProject/scripts/viz/rt_plic_data/rt_step%05d.csv", step);
        FILE* fp = fopen(fname, "w");
        // NZ=1: direct output, no z-averaging needed
        for (int iy = 0; iy < NY; iy++) {
            for (int ix = 0; ix < NX; ix++) {
                if (ix > 0) fprintf(fp, ",");
                fprintf(fp, "%.4f", h_fill[ix + NX * iy]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    };

    printf("Saving snapshots...\n");
    saveSnapshot(0);

    // ======================== Time loop ========================
    for (int step = 1; step <= TOTAL_STEPS; step++) {

        // 1. VOF advection (PLIC)
        vof.advectFillLevel(
            fluid.getVelocityX(),
            fluid.getVelocityY(),
            fluid.getVelocityZ(),
            DT);
        vof.reconstructInterface();
        vof.convertCells();

        // 2. Buoyancy ONLY — no surface tension
        forces.reset();
        forces.addVOFBuoyancyForce(
            vof.getFillLevel(),
            RHO_HEAVY, RHO_LIGHT,
            0.0f, -G_LB, 0.0f);
        forces.convertToLatticeUnits(DX, DT, RHO_HEAVY);

        // 3. TRT collision + streaming + Guo-corrected u
        fluid.collisionTRT(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(), forces.getFz());

        // 4. Snapshot + NaN check
        if (step % SNAP_INTERVAL == 0) {
            saveSnapshot(step);
            float mass = vof.computeTotalMass();
            if (std::isnan(mass) || std::isinf(mass)) {
                printf("*** NaN/Inf at step %d — stopping ***\n", step);
                break;
            }
        }
    }

    float elapsed = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("\nDone. Wall time: %.1f s\n", elapsed);

    return 0;
}
