/**
 * @file viz_square_droplet.cu
 * @brief Square-droplet relaxation visualization.
 *
 * A square region of liquid (f=1) surrounded by gas (f=0) contracts to a circle
 * under surface tension — minimizing interfacial energy.  This is a classic
 * qualitative test for CSF-based surface tension solvers.
 *
 * Physics:
 *   - Domain: 128 x 128 x 3 (quasi-2D slab, z-periodic)
 *   - Square droplet: half-side HS = 32 cells, centred at (64,64)
 *     Interface smoothed with tanh over 3 cells to avoid curvature spikes
 *   - σ = 0.02 N/m,  ν = 5e-5 m²/s,  ρ = 1000 kg/m³
 *   - dx = 100 µm/cell,  dt = 50 µs  (τ = 1.25)
 *   - Oh = ν·ρ / sqrt(σ·ρ·R) = 0.25 (underdamped, oscillating then decaying)
 *   - Capillary time t_cap = sqrt(ρ·R³/σ) = 20 ms,  R = 20·dx = 2mm
 *   - Run 2000 steps = 5 t_cap: full square→circle relaxation with damping
 *   - At steady state: Laplace pressure ΔP = σ/R = 10 Pa
 *
 * Time-stepping order (correct LBM sequence, matching viz_bubble.cu):
 *   applyBC → computeMacroscopic(F) → collision(F) → streaming → VOF advect
 *
 * Outputs (all in scripts/viz/):
 *   square_drop_step0000.csv … square_drop_step3000.csv  (z-midplane fill level)
 *   square_drop_summary.csv   (step, time_ms, mass_mid, mass_error_pct,
 *                               max_vel_ms, R_eff_cells, circularity)
 *
 * Compile (from /home/yzk/LBMProject/build/):
 *   nvcc -o ../scripts/viz/viz_square_droplet ../scripts/viz/viz_square_droplet.cu \
 *        -I../include -L. -llbm_physics -llbm_core -llbm_io -llbm_diagnostics \
 *        --std=c++17 -rdc=true -Xcompiler -fPIC -O2 -arch=sm_86
 */

#include "physics/fluid_lbm.h"
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace lbm::physics;

// ---- domain -----------------------------------------------------------------

static constexpr int NX = 128;
static constexpr int NY = 128;
static constexpr int NZ = 3;
static constexpr int NC = NX * NY * NZ;

// ---- physical parameters ----------------------------------------------------
// σ=0.005 N/m keeps corner velocity < 0.05 lattice (safe CFL margin)
// τ = 3*ν*dt/dx² + 0.5 = 3*(1e-5)*(5e-5)/(1e-4)² + 0.5 = 0.65

static constexpr float DX    = 1.0e-4f;   // 100 µm / cell
static constexpr float DT    = 5.0e-5f;   // 50 µs time step  → τ = 1.25
static constexpr float NU    = 5.0e-5f;   // kinematic viscosity [m²/s]  (Oh=0.25)
static constexpr float RHO0  = 1000.0f;   // reference density [kg/m³]
static constexpr float SIGMA = 0.02f;     // surface tension [N/m]

// Velocity conversion: lattice → physical [m/s]
static constexpr float V_CONV = DX / DT;  // = 2.0 m/s per unit lattice velocity

// ---- square droplet geometry ------------------------------------------------

static constexpr float CX      = 0.5f * NX;   // 64.0
static constexpr float CY      = 0.5f * NY;   // 64.0
static constexpr float HS      = 32.0f;        // half-side [cells]  → side = 64 cells
static constexpr float SMOOTH_W = 3.0f;        // tanh half-width [cells]

// ---- simulation schedule ----------------------------------------------------

static constexpr int TOTAL_STEPS = 2000;
static constexpr int OUT_FREQ    = 200;    // dump every N steps  → 11 files

// ---- output directory -------------------------------------------------------

static constexpr const char* OUT_DIR = "/home/yzk/LBMProject/scripts/viz";

// ---- helpers ----------------------------------------------------------------

/**
 * Build initial fill-level: smoothed square droplet.
 *
 * The product of two tanh in x and y gives fill level 1 inside the square,
 * 0 outside, and a smooth transition over SMOOTH_W cells at every edge and corner.
 * This avoids discontinuous normals on the first curvature evaluation.
 */
static std::vector<float> buildSquareFill()
{
    std::vector<float> h(NC, 0.0f);
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                float dx_i = std::fabs((i + 0.5f) - CX);
                float dy_j = std::fabs((j + 0.5f) - CY);
                float tx = std::tanh((HS - dx_i) / SMOOTH_W);
                float ty = std::tanh((HS - dy_j) / SMOOTH_W);
                float f  = 0.25f * (1.0f + tx) * (1.0f + ty);
                h[i + NX * (j + NY * k)] = std::max(0.0f, std::min(1.0f, f));
            }
        }
    }
    return h;
}

/**
 * Write z-midplane fill level as a 2D CSV (NY rows × NX columns).
 */
static void dumpSliceCSV(const std::string& path,
                          const std::vector<float>& fill)
{
    const int k0 = NZ / 2;
    std::ofstream out(path);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            out << fill[i + NX * (j + NY * k0)];
            if (i < NX - 1) out << ',';
        }
        out << '\n';
    }
}

/**
 * Scalar diagnostics on the z-midplane.
 *
 * circularity = 4π·A / P²  (1 = circle, π/4 ≈ 0.785 for square)
 */
struct Diagnostics {
    float mass_mid;
    float area_cells;
    float R_eff_cells;
    float circularity;
    float max_vel_phys;
};

static Diagnostics computeDiagnostics(const std::vector<float>& fill,
                                       const std::vector<float>& ux_lat,
                                       const std::vector<float>& uy_lat)
{
    const int k0 = NZ / 2;
    double mass = 0.0, area = 0.0;
    int perim = 0;

    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            float fv = fill[i + NX * (j + NY * k0)];
            mass += fv;
            if (fv > 0.5f) {
                area += 1.0;
                int left  = std::max(i - 1, 0);
                int right = std::min(i + 1, NX - 1);
                int down  = std::max(j - 1, 0);
                int up    = std::min(j + 1, NY - 1);
                if (fill[left  + NX * (j   + NY * k0)] <= 0.5f ||
                    fill[right + NX * (j   + NY * k0)] <= 0.5f ||
                    fill[i     + NX * (down + NY * k0)] <= 0.5f ||
                    fill[i     + NX * (up   + NY * k0)] <= 0.5f)
                    ++perim;
            }
        }
    }

    float R_eff = static_cast<float>(std::sqrt(area / M_PI));
    float circ  = (perim > 0)
        ? static_cast<float>(4.0 * M_PI * area / (static_cast<double>(perim) * perim))
        : 0.0f;

    float max_v2 = 0.0f;
    for (int idx = 0; idx < NC; ++idx) {
        float v2 = ux_lat[idx] * ux_lat[idx] + uy_lat[idx] * uy_lat[idx];
        if (v2 > max_v2) max_v2 = v2;
    }

    return {static_cast<float>(mass), static_cast<float>(area),
            R_eff, circ, std::sqrt(max_v2) * V_CONV};
}

// ---- main -------------------------------------------------------------------

int main()
{
    const float nu_lbm = NU * DT / (DX * DX);
    const float tau    = 3.0f * nu_lbm + 0.5f;
    const float t_cap  = std::sqrt(RHO0 * std::pow(20.0f * DX, 3.0f) / SIGMA);

    printf("Square Droplet Relaxation\n");
    printf("  Grid: %d x %d x %d,  dx=%.1e m,  dt=%.1e s\n", NX, NY, NZ, DX, DT);
    printf("  nu=%.2e m2/s  ->  nu_lbm=%.4f  tau=%.4f\n", NU, nu_lbm, tau);
    printf("  sigma=%.4f N/m,  rho=%.1f kg/m3\n", SIGMA, RHO0);
    printf("  t_cap~%.4e s,  run %.1f cap times  (%d steps)\n",
           t_cap, TOTAL_STEPS * DT / t_cap, TOTAL_STEPS);
    printf("  Laplace pressure: dP=sigma/R~%.3f Pa  (R=20 cells)\n",
           SIGMA / (20.0f * DX));
    printf("  Initial circularity: pi/4 = %.4f  (circle = 1.0)\n", M_PI / 4.0);
    if (tau < 0.55f || tau > 2.0f)
        printf("WARNING: tau=%.4f outside safe [0.55, 2.0]\n", tau);

    // ---- Device workspace for physical-unit velocities ----------------------
    float *d_vx_phys = nullptr, *d_vy_phys = nullptr, *d_vz_phys = nullptr;
    cudaMalloc(&d_vx_phys, NC * sizeof(float));
    cudaMalloc(&d_vy_phys, NC * sizeof(float));
    cudaMalloc(&d_vz_phys, NC * sizeof(float));

    // ---- Build initial fill level -------------------------------------------
    std::vector<float> h_fill = buildSquareFill();

    // ---- Construct solvers --------------------------------------------------
    FluidLBM fluid(NX, NY, NZ, NU, RHO0,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   DT, DX);
    // Initialize with physical density so rho_lbm = 1000 in the kernels.
    // This ensures Guo force correction du = 0.5 * F_lat / rho_lbm gives the
    // correct physical acceleration (F_lat from convertToLatticeUnits does not
    // divide by rho, so rho must be in the LBM density field itself).
    fluid.initialize(RHO0, 0.0f, 0.0f, 0.0f);

    VOFSolver vof(NX, NY, NZ, DX,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::PERIODIC);
    vof.initialize(h_fill.data());
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);
    vof.setMassConservationCorrection(true, 0.7f);

    // Prime interface geometry
    vof.reconstructInterface();
    vof.computeCurvature();
    vof.convertCells();

    ForceAccumulator forces(NX, NY, NZ);

    std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC);

    // Initial mass reference
    vof.copyFillLevelToHost(h_fill.data());
    float mass0_mid = 0.0f;
    {
        const int k0 = NZ / 2;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                mass0_mid += h_fill[i + NX * (j + NY * k0)];
    }

    // ---- Summary CSV --------------------------------------------------------
    std::ofstream summary(std::string(OUT_DIR) + "/square_drop_summary.csv");
    summary << "step,time_ms,mass_mid,mass_error_pct,max_vel_ms,R_eff_cells,circularity\n";

    // ---- Step 0 snapshot ----------------------------------------------------
    {
        std::fill(h_ux.begin(), h_ux.end(), 0.0f);
        std::fill(h_uy.begin(), h_uy.end(), 0.0f);
        dumpSliceCSV(std::string(OUT_DIR) + "/square_drop_step0000.csv", h_fill);
        Diagnostics d = computeDiagnostics(h_fill, h_ux, h_uy);
        summary << "0,0.0," << d.mass_mid << ",0.0,0.0,"
                << d.R_eff_cells << "," << d.circularity << "\n";
        printf("  step %4d  mass_mid=%.0f  R_eff=%.1f  circ=%.4f\n",
               0, d.mass_mid, d.R_eff_cells, d.circularity);
    }

    // ---- Main time loop -----------------------------------------------------
    // Order: applyBC → computeMacroscopic(F) → collision(F) → streaming
    //        → VOF advect → reconstructInterface → computeCurvature
    for (int step = 1; step <= TOTAL_STEPS; ++step) {

        // 1. Surface tension force (CSF, physical [N/m³])
        forces.reset();
        forces.addSurfaceTensionForce(vof.getCurvature(), vof.getFillLevel(),
                                      SIGMA, NX, NY, NZ, DX);

        // 2. Convert to lattice acceleration: a_lat = F_phys * dt² / dx
        forces.convertToLatticeUnits(DX, DT, RHO0);

        // 3. Apply wall BCs, then get current macroscopic state with Guo correction
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(), forces.getFz());

        // 4. Collision with Guo forcing + streaming
        fluid.collisionTRT(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();

        // 5. Convert lattice velocity → physical [m/s] for VOF advection
        cudaMemcpy(h_ux.data(), fluid.getVelocityX(), NC * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), fluid.getVelocityY(), NC * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uz.data(), fluid.getVelocityZ(), NC * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < NC; ++i) {
            h_ux[i] *= V_CONV;
            h_uy[i] *= V_CONV;
            h_uz[i] *= V_CONV;
        }
        cudaMemcpy(d_vx_phys, h_ux.data(), NC * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy_phys, h_uy.data(), NC * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz_phys, h_uz.data(), NC * sizeof(float), cudaMemcpyHostToDevice);

        // 6. VOF advection (TVD+MC, physical [m/s])
        vof.advectFillLevel(d_vx_phys, d_vy_phys, d_vz_phys, DT);

        // 7. Rebuild interface geometry for next curvature
        vof.reconstructInterface();
        vof.computeCurvature();
        vof.convertCells();

        // 8. Diagnostics and snapshot
        if (step % OUT_FREQ == 0) {
            vof.copyFillLevelToHost(h_fill.data());
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "%s/square_drop_step%04d.csv", OUT_DIR, step);
            dumpSliceCSV(fname, h_fill);

            Diagnostics d = computeDiagnostics(h_fill, h_ux, h_uy);
            float err_pct = 100.0f * (d.mass_mid - mass0_mid) / (mass0_mid + 1e-12f);
            float t_ms    = step * DT * 1e3f;
            summary << step << "," << t_ms << ","
                    << d.mass_mid << "," << err_pct << ","
                    << d.max_vel_phys << ","
                    << d.R_eff_cells << "," << d.circularity << "\n";
            printf("  step %4d  t=%.1f ms  mass=%.0f (%.2f%%)  "
                   "R_eff=%.1f  circ=%.4f  v_max=%.4f m/s\n",
                   step, t_ms, d.mass_mid, err_pct,
                   d.R_eff_cells, d.circularity, d.max_vel_phys);
        }
    }

    // ---- Laplace pressure check ---------------------------------------------
    {
        vof.copyFillLevelToHost(h_fill.data());
        const int k0 = NZ / 2;
        double area = 0.0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                if (h_fill[i + NX * (j + NY * k0)] > 0.5f) area += 1.0;
        float R_m = static_cast<float>(std::sqrt(area / M_PI)) * DX;
        printf("\nLaplace pressure: sigma/R_eff = %.4f / %.4e = %.3f Pa\n",
               SIGMA, R_m, SIGMA / R_m);
    }

    summary.close();
    cudaFree(d_vx_phys);
    cudaFree(d_vy_phys);
    cudaFree(d_vz_phys);

    printf("Done. Visualize: python3 %s/plot_square_droplet.py\n", OUT_DIR);
    return 0;
}
