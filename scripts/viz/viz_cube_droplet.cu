/**
 * @file viz_cube_droplet.cu
 * @brief Cubic droplet relaxation visualization (3D).
 *
 * A cube of liquid (f=1) surrounded by gas (f=0) contracts to a sphere
 * under surface tension — minimizing interfacial energy.  This is the 3D
 * generalization of viz_square_droplet.cu.
 *
 * Physics:
 *   - Domain: 80 x 80 x 80 (fits in ~300 MB on RTX 3050 4GB)
 *   - Cube droplet: side 32 cells, centred at (40,40,40)
 *     Interface smoothed with tanh over 3 cells to avoid curvature spikes
 *   - σ = 0.02 N/m,  ν = 5e-5 m²/s,  ρ = 1000 kg/m³
 *   - dx = 100 µm/cell,  dt = 50 µs  (τ = 1.25)
 *   - Equivalent sphere radius: R = (side³ · 3/(4π))^(1/3) ≈ 19.7 cells
 *   - Young-Laplace (3D): ΔP = 2σ/R ≈ 20 Pa
 *   - Capillary time t_cap = sqrt(ρ·R³/σ)
 *   - All boundaries: WALL (no periodicity trick needed in 3D)
 *
 * Time-stepping order (identical to viz_square_droplet.cu):
 *   applyBC → computeMacroscopic(F) → collision(F) → streaming → VOF advect
 *
 * Outputs (all in scripts/viz/):
 *   cube_drop_xy_stepNNNN.csv   (z=40 midplane fill level, NY rows × NX cols)
 *   cube_drop_xz_stepNNNN.csv   (y=40 midplane fill level, NZ rows × NX cols)
 *   cube_drop_summary.csv       (step, time_ms, mass_total, mass_error_pct,
 *                                max_vel_ms, R_eff_cells, sphericity)
 *
 * Compile (from /home/yzk/LBMProject/build/):
 *   nvcc -o ../scripts/viz/viz_cube_droplet ../scripts/viz/viz_cube_droplet.cu \
 *        -I../include -L. -llbm_physics -llbm_core -llbm_io -llbm_diagnostics \
 *        --std=c++17 -rdc=true -Xcompiler -fPIC -O2 -arch=sm_86
 */

#include "physics/fluid_lbm.h"
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "io/vtk_writer.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace lbm::physics;

// ---- domain -----------------------------------------------------------------

static constexpr int NX = 80;
static constexpr int NY = 80;
static constexpr int NZ = 80;
static constexpr int NC = NX * NY * NZ;

// ---- physical parameters ----------------------------------------------------
// Same σ, ν, dt as the 2D version — verified stable there.
// τ = 3 · ν_lbm + 0.5 = 3 · (5e-5 · 5e-5 / (1e-4)²) + 0.5 = 1.25
// Capillary velocity scale: v_cap = sqrt(σ/(ρ·R)) ≈ 0.032 m/s → 0.016 lattice units (safe)

static constexpr float DX    = 1.0e-4f;   // 100 µm / cell
static constexpr float DT    = 5.0e-5f;   // 50 µs time step
static constexpr float NU    = 5.0e-5f;   // kinematic viscosity [m²/s]
static constexpr float RHO0  = 1000.0f;   // reference density [kg/m³]
static constexpr float SIGMA = 0.02f;     // surface tension [N/m]

// Velocity conversion: lattice → physical [m/s]
static constexpr float V_CONV = DX / DT;  // 2.0 m/s per lattice velocity unit

// ---- cube droplet geometry --------------------------------------------------

static constexpr float CX      = 0.5f * NX;   // 40.0
static constexpr float CY      = 0.5f * NY;   // 40.0
static constexpr float CZ      = 0.5f * NZ;   // 40.0
static constexpr float HS      = 16.0f;        // half-side [cells]  → side = 32 cells
static constexpr float SMOOTH_W = 3.0f;        // tanh half-width [cells]

// Equivalent sphere radius from volume conservation: V_cube = side³
// R_sphere = (V_cube · 3/(4π))^(1/3) = (32³ · 3/(4π))^(1/3) ≈ 19.7 cells
static constexpr float SIDE       = 2.0f * HS;
// Non-constexpr: std::pow is not constexpr, compute as a global initialised once.
static const float R_EQ_CELLS = static_cast<float>(
    std::pow(static_cast<double>(SIDE) * SIDE * SIDE * 3.0 / (4.0 * M_PI), 1.0 / 3.0));

// ---- simulation schedule ----------------------------------------------------

static constexpr int TOTAL_STEPS = 2000;
static constexpr int OUT_FREQ    = 200;    // dump every N steps → 11 outputs

// ---- output directory -------------------------------------------------------

static constexpr const char* OUT_DIR = "/home/yzk/LBMProject/scripts/viz";

// ---- helpers ----------------------------------------------------------------

/**
 * Build initial fill-level: smoothed cubic droplet.
 *
 * Product of three tanh in x, y, z gives 1 inside the cube and 0 outside,
 * with a smooth SMOOTH_W-cell transition at every face, edge, and corner.
 */
static std::vector<float> buildCubeFill()
{
    std::vector<float> h(NC, 0.0f);
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                float dxi = std::fabs((i + 0.5f) - CX);
                float dyj = std::fabs((j + 0.5f) - CY);
                float dzk = std::fabs((k + 0.5f) - CZ);
                float tx  = std::tanh((HS - dxi) / SMOOTH_W);
                float ty  = std::tanh((HS - dyj) / SMOOTH_W);
                float tz  = std::tanh((HS - dzk) / SMOOTH_W);
                // 0.5*(1+tx) ranges in [0,1] for each direction
                float f   = 0.125f * (1.0f + tx) * (1.0f + ty) * (1.0f + tz);
                h[i + NX * (j + NY * k)] = std::max(0.0f, std::min(1.0f, f));
            }
        }
    }
    return h;
}

/**
 * Write the z=z0 (XY) midplane fill level as a 2-D CSV (NY rows × NX cols).
 */
static void dumpSliceXY(const std::string& path, const std::vector<float>& fill, int z0)
{
    std::ofstream out(path);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            out << fill[i + NX * (j + NY * z0)];
            if (i < NX - 1) out << ',';
        }
        out << '\n';
    }
}

/**
 * Write the y=y0 (XZ) midplane fill level as a 2-D CSV (NZ rows × NX cols).
 */
static void dumpSliceXZ(const std::string& path, const std::vector<float>& fill, int y0)
{
    std::ofstream out(path);
    for (int k = 0; k < NZ; ++k) {
        for (int i = 0; i < NX; ++i) {
            out << fill[i + NX * (y0 + NY * k)];
            if (i < NX - 1) out << ',';
        }
        out << '\n';
    }
}

/**
 * 3D diagnostics on the full volume.
 *
 * sphericity = π^(1/3) · (6·V)^(2/3) / A  (1 = sphere, <1 for all other shapes)
 * We approximate the surface area A by counting interface cells.
 */
struct Diagnostics3D {
    float mass_total;
    float volume_cells;
    float R_eff_cells;
    float sphericity;
    float max_vel_phys;
};

static Diagnostics3D computeDiagnostics3D(const std::vector<float>& fill,
                                           const std::vector<float>& ux_lat,
                                           const std::vector<float>& uy_lat,
                                           const std::vector<float>& uz_lat)
{
    double mass = 0.0, volume = 0.0;
    int surface_cells = 0;

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);
                float fv = fill[idx];
                mass += fv;
                if (fv > 0.5f) {
                    volume += 1.0;
                    // Count as surface if any neighbour is outside
                    int im = std::max(i-1, 0),   ip = std::min(i+1, NX-1);
                    int jm = std::max(j-1, 0),   jp = std::min(j+1, NY-1);
                    int km = std::max(k-1, 0),   kp = std::min(k+1, NZ-1);
                    if (fill[im + NX*(j  + NY*k )] <= 0.5f ||
                        fill[ip + NX*(j  + NY*k )] <= 0.5f ||
                        fill[i  + NX*(jm + NY*k )] <= 0.5f ||
                        fill[i  + NX*(jp + NY*k )] <= 0.5f ||
                        fill[i  + NX*(j  + NY*km)] <= 0.5f ||
                        fill[i  + NX*(j  + NY*kp)] <= 0.5f)
                        ++surface_cells;
                }
            }
        }
    }

    // R_eff from volume: V = 4/3 π R³  →  R = (3V/(4π))^(1/3)
    float R_eff = static_cast<float>(
        std::pow(volume * 3.0 / (4.0 * M_PI), 1.0 / 3.0));

    // Sphericity = π^(1/3) · (6V)^(2/3) / A_approx
    float sphericity = 0.0f;
    if (surface_cells > 0) {
        double A = static_cast<double>(surface_cells);
        double V = volume;
        sphericity = static_cast<float>(
            std::pow(M_PI, 1.0/3.0) * std::pow(6.0 * V, 2.0/3.0) / A);
    }

    float max_v2 = 0.0f;
    for (int idx = 0; idx < NC; ++idx) {
        float v2 = ux_lat[idx]*ux_lat[idx]
                 + uy_lat[idx]*uy_lat[idx]
                 + uz_lat[idx]*uz_lat[idx];
        if (v2 > max_v2) max_v2 = v2;
    }

    return {static_cast<float>(mass), static_cast<float>(volume),
            R_eff, sphericity, std::sqrt(max_v2) * V_CONV};
}

// ---- main -------------------------------------------------------------------

int main()
{
    const float nu_lbm = NU * DT / (DX * DX);
    const float tau    = 3.0f * nu_lbm + 0.5f;
    const float R_m    = R_EQ_CELLS * DX;
    const float t_cap  = std::sqrt(RHO0 * R_m * R_m * R_m / SIGMA);

    printf("Cube Droplet Relaxation (3D)\n");
    printf("  Grid: %d x %d x %d,  dx=%.1e m,  dt=%.1e s\n", NX, NY, NZ, DX, DT);
    printf("  nu=%.2e m2/s  ->  nu_lbm=%.4f  tau=%.4f\n", NU, nu_lbm, tau);
    printf("  sigma=%.4f N/m,  rho=%.1f kg/m3\n", SIGMA, RHO0);
    printf("  Cube side=%.0f cells,  R_eq=%.2f cells (%.2e m)\n",
           SIDE, R_EQ_CELLS, R_m);
    printf("  t_cap~%.4e s,  run %.1f cap times  (%d steps)\n",
           t_cap, TOTAL_STEPS * DT / t_cap, TOTAL_STEPS);
    printf("  Laplace pressure (3D): 2*sigma/R_eq = %.3f Pa\n", 2.0f * SIGMA / R_m);
    if (tau < 0.55f || tau > 2.0f)
        printf("WARNING: tau=%.4f outside safe [0.55, 2.0]\n", tau);

    // ---- Device workspace for physical-unit velocities ----------------------
    float *d_vx_phys = nullptr, *d_vy_phys = nullptr, *d_vz_phys = nullptr;
    cudaMalloc(&d_vx_phys, NC * sizeof(float));
    cudaMalloc(&d_vy_phys, NC * sizeof(float));
    cudaMalloc(&d_vz_phys, NC * sizeof(float));

    // ---- Build initial fill level -------------------------------------------
    std::vector<float> h_fill = buildCubeFill();

    // ---- Construct solvers --------------------------------------------------
    // All boundaries WALL — no z-periodic trick needed in true 3D.
    FluidLBM fluid(NX, NY, NZ, NU, RHO0,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   DT, DX);
    // Physical density so Guo correction du = 0.5*F_lat/rho_lbm is correct.
    fluid.initialize(RHO0, 0.0f, 0.0f, 0.0f);

    VOFSolver vof(NX, NY, NZ, DX,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::WALL);
    vof.initialize(h_fill.data());
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);
    vof.setMassConservationCorrection(true, 0.7f);

    // Prime interface geometry before first force evaluation
    vof.reconstructInterface();
    vof.computeCurvature();
    vof.convertCells();

    ForceAccumulator forces(NX, NY, NZ);

    std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC);

    // Initial mass reference (full 3D volume sum)
    vof.copyFillLevelToHost(h_fill.data());
    double mass0 = 0.0;
    for (int idx = 0; idx < NC; ++idx) mass0 += h_fill[idx];

    // ---- Summary CSV --------------------------------------------------------
    std::ofstream summary(std::string(OUT_DIR) + "/cube_drop_summary.csv");
    summary << "step,time_ms,mass_total,mass_error_pct,max_vel_ms,R_eff_cells,sphericity\n";

    const int kMid = NZ / 2;   // z midplane index
    const int jMid = NY / 2;   // y midplane index

    // ---- Step 0 snapshot ----------------------------------------------------
    {
        std::fill(h_ux.begin(), h_ux.end(), 0.0f);
        std::fill(h_uy.begin(), h_uy.end(), 0.0f);
        std::fill(h_uz.begin(), h_uz.end(), 0.0f);

        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/cube_drop_xy_step%04d.csv", OUT_DIR, 0);
        dumpSliceXY(fname, h_fill, kMid);
        std::snprintf(fname, sizeof(fname), "%s/cube_drop_xz_step%04d.csv", OUT_DIR, 0);
        dumpSliceXZ(fname, h_fill, jMid);

        // VTK output for ParaView
        std::snprintf(fname, sizeof(fname), "%s/cube_drop_%04d", OUT_DIR, 0);
        lbm::io::VTKWriter::writeStructuredPoints(
            fname, h_fill.data(), NX, NY, NZ, DX, DX, DX, "FillLevel");

        Diagnostics3D d = computeDiagnostics3D(h_fill, h_ux, h_uy, h_uz);
        summary << "0,0.0," << d.mass_total << ",0.0,0.0,"
                << d.R_eff_cells << "," << d.sphericity << "\n";
        printf("  step %4d  mass=%.0f  R_eff=%.2f  sphericity=%.4f\n",
               0, d.mass_total, d.R_eff_cells, d.sphericity);
    }

    // ---- Main time loop -----------------------------------------------------
    // Order: applyBC → computeMacroscopic(F) → collision(F) → streaming
    //        → VOF advect → reconstructInterface → computeCurvature
    for (int step = 1; step <= TOTAL_STEPS; ++step) {

        // 1. Surface tension force (CSF, physical [N/m³])
        forces.reset();
        forces.addSurfaceTensionForce(vof.getCurvature(), vof.getFillLevel(),
                                      SIGMA, NX, NY, NZ, DX);

        // 2. Convert to lattice acceleration: a_lat = F_phys * dt² / (dx * rho)
        forces.convertToLatticeUnits(DX, DT, RHO0);

        // 3. Apply wall BCs, then compute macroscopic with Guo correction
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(), forces.getFz());

        // 4. TRT collision with Guo forcing + streaming
        fluid.collisionTRT(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();

        // 5. Convert lattice velocities → physical [m/s] for VOF advection
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

            char fname[512];
            std::snprintf(fname, sizeof(fname),
                          "%s/cube_drop_xy_step%04d.csv", OUT_DIR, step);
            dumpSliceXY(fname, h_fill, kMid);
            std::snprintf(fname, sizeof(fname),
                          "%s/cube_drop_xz_step%04d.csv", OUT_DIR, step);
            dumpSliceXZ(fname, h_fill, jMid);

            // VTK output for ParaView (fill level + velocity)
            std::snprintf(fname, sizeof(fname), "%s/cube_drop_%04d", OUT_DIR, step);
            lbm::io::VTKWriter::writeStructuredPoints(
                fname, h_fill.data(), NX, NY, NZ, DX, DX, DX, "FillLevel");
            std::snprintf(fname, sizeof(fname), "%s/cube_drop_vel_%04d", OUT_DIR, step);
            lbm::io::VTKWriter::writeVectorField(
                fname, h_ux.data(), h_uy.data(), h_uz.data(),
                NX, NY, NZ, DX, DX, DX, "Velocity");

            Diagnostics3D d = computeDiagnostics3D(h_fill, h_ux, h_uy, h_uz);
            float err_pct = 100.0f * (d.mass_total - static_cast<float>(mass0))
                            / (static_cast<float>(mass0) + 1e-12f);
            float t_ms    = step * DT * 1e3f;
            summary << step << "," << t_ms << ","
                    << d.mass_total << "," << err_pct << ","
                    << d.max_vel_phys << ","
                    << d.R_eff_cells << "," << d.sphericity << "\n";
            printf("  step %4d  t=%.1f ms  mass=%.0f (%.2f%%)  "
                   "R_eff=%.2f  sphericity=%.4f  v_max=%.4f m/s\n",
                   step, t_ms, d.mass_total, err_pct,
                   d.R_eff_cells, d.sphericity, d.max_vel_phys);
        }
    }

    // ---- Laplace pressure check (3D: ΔP = 2σ/R) ----------------------------
    {
        vof.copyFillLevelToHost(h_fill.data());
        double vol = 0.0;
        for (int idx = 0; idx < NC; ++idx)
            if (h_fill[idx] > 0.5f) vol += 1.0;
        float R_final = static_cast<float>(std::pow(vol * 3.0 / (4.0 * M_PI), 1.0/3.0)) * DX;
        printf("\nFinal Laplace pressure (3D): 2*sigma/R_eff = 2*%.4f/%.4e = %.3f Pa\n",
               SIGMA, R_final, 2.0f * SIGMA / R_final);
        printf("Expected: 2*sigma/R_eq = %.3f Pa\n", 2.0f * SIGMA / R_m);
    }

    summary.close();
    cudaFree(d_vx_phys);
    cudaFree(d_vy_phys);
    cudaFree(d_vz_phys);

    printf("Done. Visualize: python3 %s/plot_cube_droplet.py\n", OUT_DIR);
    return 0;
}
