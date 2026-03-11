/**
 * @file viz_bubble.cu
 * @brief Rising bubble: real LBM-VOF simulation with stable dimensionless parameters.
 *
 * Physics: circular gas bubble (f=0) in liquid (f=1) driven by buoyancy.
 * All parameters chosen in LATTICE UNITS first, then mapped to physical.
 *
 * Design constraints:
 *   - g_LU ~ 1e-5  (standard for LBM buoyancy, keeps F_LU << 0.01)
 *   - tau = 0.7     (stable, Guo forcing prefactor 0.286)
 *   - D_bubble = 24 cells
 *   - rho_L/rho_G = 10
 *
 * Output:
 *   bubble_trajectory.csv  — per-step centroid trajectory
 *   bubble_snapshots.csv   — VOF snapshots at selected times
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "physics/fluid_lbm.h"
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "core/streaming.h"

using namespace lbm::physics;

// ---------------------------------------------------------------------------
// Domain (lattice units)
// ---------------------------------------------------------------------------
static constexpr int NX = 80;
static constexpr int NY = 240;
static constexpr int NZ = 1;
static constexpr int NC = NX * NY * NZ;

// ---------------------------------------------------------------------------
// Lattice parameters (chosen for stability)
// ---------------------------------------------------------------------------
static constexpr float TAU   = 0.7f;
static constexpr float NU_LB = (TAU - 0.5f) / 3.0f;    // 0.0667
static constexpr float G_LB  = 3.0e-5f;                 // lattice gravity (safe: <<0.01)
static constexpr float RHO_RATIO = 10.0f;               // rho_L / rho_G

// Bubble geometry (lattice units, dx=1)
static constexpr float R_BUB = 12.0f;                   // radius in cells
static constexpr float X0_LB = 0.5f * NX;               // center x
static constexpr float Y0_LB = 0.25f * NY;              // center y (lower quarter)

// Physical mapping (for display only)
static constexpr float DX = 5.0e-4f;            // 0.5 mm per cell
// DT_PHYS computed at runtime: sqrtf(G_LB * DX / 9.81)

static constexpr int TOTAL_STEPS = 3000;

// ---------------------------------------------------------------------------
// Analytical references (lattice units)
// ---------------------------------------------------------------------------
// Stokes: V_t = 2R²·Δρ·g / (9·ν·ρ_L)  — INVALID for Re>1
static float stokesVelocityLU() {
    float delta_rho = 1.0f - 1.0f/RHO_RATIO;  // (rho_L - rho_G)/rho_L
    return 2.0f * R_BUB * R_BUB * delta_rho * G_LB / (9.0f * NU_LB);
}

// Schiller-Naumann iterative
static float schillerNaumannLU() {
    float delta_rho = 1.0f - 1.0f/RHO_RATIO;
    float D = 2.0f * R_BUB;
    float V = stokesVelocityLU();
    for (int i = 0; i < 50; ++i) {
        float Re = fabs(V) * D / NU_LB;
        if (Re < 0.01f) Re = 0.01f;
        float Cd = 24.0f / Re * (1.0f + 0.15f * powf(Re, 0.687f));
        float V_new = sqrtf(8.0f * R_BUB * delta_rho * G_LB / (3.0f * Cd));
        if (fabsf(V_new - V) < 1e-8f) break;
        V = V_new;
    }
    return V;
}

// ---------------------------------------------------------------------------
// Build initial fill-level (sharp initialization)
// ---------------------------------------------------------------------------
static std::vector<float> buildBubble()
{
    std::vector<float> f(NC, 1.0f);  // all liquid
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            float dx = i + 0.5f - X0_LB;
            float dy = j + 0.5f - Y0_LB;
            float r = sqrtf(dx*dx + dy*dy);
            if (r < R_BUB + 2.0f) {
                float d = R_BUB - r;
                f[i + NX * j] = 0.5f * (1.0f - tanhf(d * 2.0f));  // smooth over ~1 cell
            }
        }
    return f;
}

// ---------------------------------------------------------------------------
// Compute bubble centroid (lattice units)
// ---------------------------------------------------------------------------
static float centroidY(const std::vector<float>& fill)
{
    double num = 0.0, den = 0.0;
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            float gas = 1.0f - fill[i + NX * j];
            if (gas > 0.01f) {
                num += gas * (j + 0.5f);
                den += gas;
            }
        }
    return (den > 1e-6) ? (float)(num / den) : Y0_LB;
}

// ---------------------------------------------------------------------------
int main()
{
    // Physical mapping
    float dt_phys = sqrtf(G_LB * DX / 9.81f);
    float nu_phys = NU_LB * DX * DX / dt_phys;
    float vel_conv = DX / dt_phys;  // LU velocity → m/s

    float V_stokes_LU = stokesVelocityLU();
    float V_sn_LU = schillerNaumannLU();
    float Re_stokes = V_stokes_LU * 2.0f * R_BUB / NU_LB;
    float Re_sn = V_sn_LU * 2.0f * R_BUB / NU_LB;

    printf("=== Rising Bubble (2D VOF-LBM) ===\n");
    printf("  Grid: %d x %d,  R=%.0f cells,  rho_L/rho_G=%.0f\n", NX, NY, R_BUB, RHO_RATIO);
    printf("  tau=%.3f,  nu_LB=%.4f,  g_LB=%.1e\n", TAU, NU_LB, G_LB);
    printf("  Physical mapping: dx=%.1e m, dt=%.4e s, nu=%.2e m²/s\n", DX, dt_phys, nu_phys);
    printf("  vel_conv=%.3f m/s per LU\n", vel_conv);
    printf("  V_stokes = %.5f LU  (%.2f mm/s,  Re=%.1f) — INVALID at this Re\n",
           V_stokes_LU, V_stokes_LU*vel_conv*1e3f, Re_stokes);
    printf("  V_SN     = %.5f LU  (%.2f mm/s,  Re=%.1f)\n",
           V_sn_LU, V_sn_LU*vel_conv*1e3f, Re_sn);
    printf("  F_buoy_max = %.2e LU  (safe: <<0.01)\n",
           (1.0f - 1.0f/RHO_RATIO) * G_LB);
    printf("  Total steps: %d  (physical time: %.3f s)\n\n",
           TOTAL_STEPS, TOTAL_STEPS * dt_phys);

    // ---- Construct solvers ----
    // FluidLBM uses physical units internally, so pass physical NU, RHO, DT, DX
    float rho_L_phys = 1000.0f;  // arbitrary choice for physical density
    float rho_G_phys = rho_L_phys / RHO_RATIO;

    FluidLBM fluid(NX, NY, NZ, nu_phys, rho_L_phys,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   dt_phys, DX);
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // VOF: uses dx=1 in lattice units, velocity in lattice units
    // We'll pass lattice velocities directly (no conversion)
    VOFSolver vof(NX, NY, NZ, 1.0f,  // dx=1 in lattice units
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::PERIODIC);
    std::vector<float> h_fill = buildBubble();
    vof.initialize(h_fill.data());
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);

    ForceAccumulator forces(NX, NY, NZ);

    // ---- Output files ----
    const std::string dir = "/home/yzk/LBMProject/scripts/viz";
    std::ofstream traj(dir + "/bubble_trajectory.csv");
    traj << "# Rising bubble trajectory (lattice units)\n";
    traj << "# g_LB=" << G_LB << " tau=" << TAU << " nu_LB=" << NU_LB
         << " rho_ratio=" << RHO_RATIO << " R=" << R_BUB
         << " dx=" << DX << " dt=" << dt_phys << " vel_conv=" << vel_conv << "\n";
    traj << "step,time_s,centroid_y_mm,velocity_mm_s,mass\n";

    std::ofstream snap(dir + "/bubble_snapshots.csv");
    snap << "# Rising bubble snapshots  NX=" << NX << " NY=" << NY << "\n";
    snap << "# dx=" << DX << " dt=" << dt_phys << " g_LB=" << G_LB << "\n";
    snap << "step,x,y,fill_level\n";

    // Snapshot schedule
    const int snap_steps[] = {0, 500, 1000, 1500, 2000, 2500, 3000};
    const int n_snaps = 7;
    int snap_idx = 0;

    auto dumpSnap = [&](int step) {
        vof.copyFillLevelToHost(h_fill.data());
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                snap << step << "," << std::setprecision(6)
                     << (i+0.5f)*DX << "," << (j+0.5f)*DX << ","
                     << h_fill[i + NX*j] << "\n";
        snap << "\n";
    };

    // ---- Step 0 ----
    vof.copyFillLevelToHost(h_fill.data());
    float cy_prev = centroidY(h_fill);
    float mass0 = vof.computeTotalMass();
    traj << 0 << "," << 0.0f << "," << cy_prev*DX*1e3f << "," << 0.0f << "," << mass0 << "\n";
    dumpSnap(0);
    snap_idx = 1;

    printf("  step    time[ms]   y_c[mm]    Vy[mm/s]   Re_inst    mass\n");
    printf("  %4d   %8.3f   %7.2f     %7.1f   %7.1f   %.4f\n",
           0, 0.0f, cy_prev*DX*1e3f, 0.0f, 0.0f, mass0);

    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Main loop ----
    for (int step = 1; step <= TOTAL_STEPS; ++step) {

        // 1. Reset forces, compute buoyancy IN LATTICE UNITS directly
        //    F_LU = (f - 0.5) * delta_rho_LU * g_LU  (but sign: gas rises)
        //    Use ForceAccumulator with lattice-unit parameters
        forces.reset();
        // Pass lattice-unit density and gravity directly, skip convertToLatticeUnits
        forces.addVOFBuoyancyForce(vof.getFillLevel(),
                                   1.0f,                   // rho_L in LU
                                   1.0f / RHO_RATIO,       // rho_G in LU
                                   0.0f, -G_LB, 0.0f);    // gravity in LU
        // No convertToLatticeUnits! Forces are already in LU

        // 2. LBM: BC → collision(+Guo forcing) → streaming
        fluid.applyBoundaryConditions(1);
        fluid.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();

        // 3. VOF advection: pass lattice velocities directly (dt_LB=1, dx_LB=1)
        vof.advectFillLevel(fluid.getVelocityX(), fluid.getVelocityY(),
                            fluid.getVelocityZ(), 1.0f);  // dt_LB = 1

        // 4. Track centroid every 10 steps
        if (step % 10 == 0 || (snap_idx < n_snaps && step == snap_steps[snap_idx])) {
            vof.copyFillLevelToHost(h_fill.data());
            float cy = centroidY(h_fill);
            float vy_LU = (cy - cy_prev) / 10.0f;  // lattice velocity
            float vy_phys = vy_LU * vel_conv;        // m/s
            float Re_inst = fabsf(vy_LU) * 2.0f * R_BUB / NU_LB;
            float mass = vof.computeTotalMass();

            traj << step << "," << std::setprecision(6) << step * dt_phys << ","
                 << cy*DX*1e3f << "," << vy_phys*1e3f << "," << mass << "\n";

            if (step % 100 == 0) {
                printf("  %4d   %8.3f   %7.2f     %7.1f   %7.1f   %.4f\n",
                       step, step*dt_phys*1e3f, cy*DX*1e3f, vy_phys*1e3f, Re_inst, mass);
                fflush(stdout);
            }

            if (snap_idx < n_snaps && step == snap_steps[snap_idx]) {
                dumpSnap(step);
                snap_idx++;
            }

            cy_prev = cy;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    traj.close();
    snap.close();
    printf("\nDone. %d steps in %.1f s (%.2f ms/step)\n", TOTAL_STEPS, elapsed, elapsed/TOTAL_STEPS*1000);
    printf("Output in %s/bubble_*.csv\n", dir.c_str());

    return 0;
}
