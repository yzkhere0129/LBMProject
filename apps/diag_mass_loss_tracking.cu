/**
 * @file diag_mass_loss_tracking.cu
 * @brief Mass Loss Destination Analysis for Powder Bed Simulation
 *
 * Identical physics to benchmark_powder_bed_316L, but adds detailed
 * mass accounting to answer: where did the 3807 cells go?
 *
 * Tracks per-step:
 *   - Total ΣF and ΔΣF
 *   - Per-z-layer ΣF (identifies which layers lose mass)
 *   - Boundary face mass (6 faces: leak detection)
 *   - VOF cutoff cells (0 < f < 0.01: numerical erosion)
 *   - Cell transition census (liquid→gas, interface→gas, etc.)
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"
#include "io/field_registry.h"

using namespace lbm;
using namespace lbm::physics;

static MaterialProperties createMat() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_MassTrack", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700;
    mat.T_vaporization=3200.0f;
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=-4.3e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;
    mat.emissivity=0.3f;
    return mat;
}

static void initializePowderBed(
    std::vector<float>& fill,
    int NX, int NY, int NZ, float dx,
    int z_sub, int z_pow,
    float r_min, float r_max,
    float target_density)
{
    const int num_cells = NX * NY * NZ;
    fill.assign(num_cells, 0.0f);

    for (int k = 0; k < z_sub; k++)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
                fill[i + j*NX + k*NX*NY] = 1.0f;

    struct Sphere { float cx, cy, cz, r; };
    std::vector<Sphere> spheres;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_x(0, NX * dx);
    std::uniform_real_distribution<float> dist_y(0, NY * dx);
    std::uniform_real_distribution<float> dist_z(z_sub * dx, z_pow * dx);

    float powder_volume = NX * dx * NY * dx * (z_pow - z_sub) * dx;
    float target_solid_volume = target_density * powder_volume;
    float current_solid_volume = 0.0f;
    int placed = 0;

    float cell_size = 2.0f * r_max;
    int hNX = (int)(NX * dx / cell_size) + 1;
    int hNY = (int)(NY * dx / cell_size) + 1;
    int hNZ = (int)((z_pow - z_sub) * dx / cell_size) + 1;
    std::vector<std::vector<int>> hash_grid(hNX * hNY * hNZ);

    auto hash_idx = [&](float x, float y, float z) -> int {
        int hi = std::min((int)(x / cell_size), hNX-1);
        int hj = std::min((int)(y / cell_size), hNY-1);
        int hk = std::min((int)((z - z_sub*dx) / cell_size), hNZ-1);
        hi = std::max(0, hi); hj = std::max(0, hj); hk = std::max(0, hk);
        return hi + hj*hNX + hk*hNX*hNY;
    };

    auto check_overlap = [&](float cx, float cy, float cz, float r) -> bool {
        int hi = (int)(cx / cell_size);
        int hj = (int)(cy / cell_size);
        int hk = (int)((cz - z_sub*dx) / cell_size);
        for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
                for (int dk = -1; dk <= 1; dk++) {
                    int ni=hi+di, nj=hj+dj, nk=hk+dk;
                    if (ni<0||ni>=hNX||nj<0||nj>=hNY||nk<0||nk>=hNZ) continue;
                    int cell = ni + nj*hNX + nk*hNX*hNY;
                    for (int si : hash_grid[cell]) {
                        const auto& s = spheres[si];
                        float d2 = (cx-s.cx)*(cx-s.cx)+(cy-s.cy)*(cy-s.cy)+(cz-s.cz)*(cz-s.cz);
                        if (d2 < (r+s.r)*(r+s.r)*0.95f) return true;
                    }
                }
        return false;
    };

    float radii[] = {r_max, (r_min+r_max)*0.5f, r_min, r_min*0.7f};
    for (int pass = 0; pass < 4 && current_solid_volume < target_solid_volume; pass++) {
        float r_base = radii[pass];
        std::uniform_real_distribution<float> dist_r_pass(r_base*0.9f, r_base*1.1f);
        for (int att = 0; att < 100000 && current_solid_volume < target_solid_volume; att++) {
            float r = dist_r_pass(rng);
            float cx = dist_x(rng), cy = dist_y(rng), cz = dist_z(rng);
            cz = std::max(z_sub*dx + r, std::min(z_pow*dx - r, cz));
            if (check_overlap(cx, cy, cz, r)) continue;
            int sid = spheres.size();
            spheres.push_back({cx, cy, cz, r});
            hash_grid[hash_idx(cx, cy, cz)].push_back(sid);
            current_solid_volume += (4.0f/3.0f)*3.14159265f*r*r*r;
            placed++;
        }
        printf("  Pass %d (r~%.0fμm): %d particles, packing=%.1f%%\n",
               pass, r_base*1e6f, placed, current_solid_volume/powder_volume*100);
    }
    printf("Powder bed: %d particles, packing = %.1f%%\n", placed, current_solid_volume/powder_volume*100);

    for (const auto& s : spheres) {
        int i_min = std::max(0, (int)((s.cx - s.r) / dx) - 1);
        int i_max = std::min(NX-1, (int)((s.cx + s.r) / dx) + 1);
        int j_min = std::max(0, (int)((s.cy - s.r) / dx) - 1);
        int j_max = std::min(NY-1, (int)((s.cy + s.r) / dx) + 1);
        int k_min = std::max(z_sub, (int)((s.cz - s.r) / dx) - 1);
        int k_max = std::min(z_pow-1, (int)((s.cz + s.r) / dx) + 1);
        for (int k = k_min; k <= k_max; k++)
            for (int j = j_min; j <= j_max; j++)
                for (int i = i_min; i <= i_max; i++) {
                    float x = (i+0.5f)*dx, y = (j+0.5f)*dx, z = (k+0.5f)*dx;
                    float d2 = (x-s.cx)*(x-s.cx)+(y-s.cy)*(y-s.cy)+(z-s.cz)*(z-s.cz);
                    if (d2 <= s.r*s.r) fill[i + j*NX + k*NX*NY] = 1.0f;
                }
    }
    for (const auto& s : spheres) {
        int i_min = std::max(0, (int)((s.cx - s.r - dx) / dx));
        int i_max = std::min(NX-1, (int)((s.cx + s.r + dx) / dx));
        int j_min = std::max(0, (int)((s.cy - s.r - dx) / dx));
        int j_max = std::min(NY-1, (int)((s.cy + s.r + dx) / dx));
        int k_min = std::max(z_sub, (int)((s.cz - s.r - dx) / dx));
        int k_max = std::min(z_pow-1, (int)((s.cz + s.r + dx) / dx));
        for (int k = k_min; k <= k_max; k++)
            for (int j = j_min; j <= j_max; j++)
                for (int i = i_min; i <= i_max; i++) {
                    int idx = i + j*NX + k*NX*NY;
                    if (fill[idx] >= 1.0f) continue;
                    float x = (i+0.5f)*dx, y = (j+0.5f)*dx, z = (k+0.5f)*dx;
                    float dist = sqrtf((x-s.cx)*(x-s.cx)+(y-s.cy)*(y-s.cy)+(z-s.cz)*(z-s.cz));
                    float frac = (s.r + 0.5f*dx - dist) / dx;
                    frac = std::max(0.0f, std::min(1.0f, frac));
                    fill[idx] = std::max(fill[idx], frac);
                }
    }
}

// ============================================================
// Mass accounting helper
// ============================================================
struct MassAudit {
    float total_fill;            // ΣF
    float substrate_fill;        // ΣF for z < z_sub
    float powder_fill;           // ΣF for z_sub <= z < z_pow
    float gas_buffer_fill;       // ΣF for z >= z_pow

    // Boundary face mass (1-cell layer at each face)
    float face_xmin, face_xmax;
    float face_ymin, face_ymax;
    float face_zmin, face_zmax;

    // Cell census
    int n_full;                  // f > 0.99
    int n_interface;             // 0.01 < f <= 0.99
    int n_tiny;                  // 0 < f <= 0.01 (cutoff candidates)
    int n_gas;                   // f == 0

    // Cells that lost fill since last audit
    int cells_lost_to_zero;      // were f>0, now f=0
    float fill_lost_to_zero;     // total fill that went to zero
    int cells_gained_from_zero;  // were f=0, now f>0
};

static MassAudit computeMassAudit(
    const std::vector<float>& f_now,
    const std::vector<float>& f_prev,  // empty if first audit
    int NX, int NY, int NZ,
    int z_sub, int z_pow)
{
    MassAudit a = {};
    bool has_prev = !f_prev.empty();

    for (int iz = 0; iz < NZ; iz++)
        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++) {
                int idx = ix + iy*NX + iz*NX*NY;
                float f = f_now[idx];

                a.total_fill += f;

                if (iz < z_sub)       a.substrate_fill += f;
                else if (iz < z_pow)  a.powder_fill += f;
                else                  a.gas_buffer_fill += f;

                // Boundary faces
                if (ix == 0)    a.face_xmin += f;
                if (ix == NX-1) a.face_xmax += f;
                if (iy == 0)    a.face_ymin += f;
                if (iy == NY-1) a.face_ymax += f;
                if (iz == 0)    a.face_zmin += f;
                if (iz == NZ-1) a.face_zmax += f;

                // Cell census
                if (f > 0.99f)       a.n_full++;
                else if (f > 0.01f)  a.n_interface++;
                else if (f > 0.0f)   a.n_tiny++;
                else                 a.n_gas++;

                // Transition tracking
                if (has_prev) {
                    float fp = f_prev[idx];
                    if (fp > 0.01f && f <= 0.0f) {
                        a.cells_lost_to_zero++;
                        a.fill_lost_to_zero += fp;
                    }
                    if (fp <= 0.0f && f > 0.01f) {
                        a.cells_gained_from_zero++;
                    }
                }
            }
    return a;
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_cond=20, mu=0.005f;
    const float alpha=k_cond/(rho*cp), nu=mu/rho;

    const int NX = 80, NY = 160, NZ = 80;
    const float dx = 2.5e-6f, dt = 1.0e-8f;
    const float t_total = 200.0e-6f;
    const int z_substrate = 40, z_powder_top = 56;
    const int num_cells = NX*NY*NZ;

    printf("============================================================\n");
    printf("  Mass Loss Destination Analysis: Powder Bed 316L\n");
    printf("============================================================\n");
    printf("Domain: %d×%d×%d (dx=%.1fμm), t_total=%.0fμs\n\n",
           NX, NY, NZ, dx*1e6f, t_total*1e6f);

    // Generate powder bed (identical to production)
    std::vector<float> h_fill;
    initializePowderBed(h_fill, NX, NY, NZ, dx,
                        z_substrate, z_powder_top,
                        10.0e-6f, 15.0e-6f, 0.55f);

    // Solver config (identical to production)
    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;

    config.enable_thermal=true; config.enable_thermal_advection=true;
    config.use_fdm_thermal=true; config.enable_phase_change=true;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=true;
    config.enable_laser=true; config.enable_darcy=true;
    config.enable_marangoni=true; config.enable_surface_tension=true;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=true;
    config.enable_recoil_pressure=true;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity = nu * dt / (dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=1.0f;
    config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=1.0f;
    config.evap_cooling_factor=1.0f;

    config.laser_power=150.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=100.0e-6f;
    config.laser_start_y=100.0e-6f;
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=1.0f;

    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=0.15f;
    config.cfl_use_gradual_scaling=true;
    config.vof_subcycles=1;
    config.enable_vof_mass_correction=false;

    config.boundaries.x_min=config.boundaries.x_max=BoundaryType::WALL;
    config.boundaries.y_min=config.boundaries.y_max=BoundaryType::WALL;
    config.boundaries.z_min=config.boundaries.z_max=BoundaryType::WALL;
    config.boundaries.thermal_x_min=config.boundaries.thermal_x_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min=config.boundaries.thermal_y_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min=ThermalBCType::DIRICHLET;
    config.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature=300.0f;

    MultiphysicsSolver solver(config);
    solver.setSmagorinskyCs(0.20f);

    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    mkdir("output_mass_track", 0755);

    // Initial mass audit
    MassAudit a0 = computeMassAudit(h_fill, {}, NX, NY, NZ, z_substrate, z_powder_top);

    printf("\n=== Initial Mass Audit ===\n");
    printf("Total ΣF:      %.2f\n", a0.total_fill);
    printf("  Substrate:   %.2f (z < %d)\n", a0.substrate_fill, z_substrate);
    printf("  Powder zone: %.2f (z %d-%d)\n", a0.powder_fill, z_substrate, z_powder_top-1);
    printf("  Gas buffer:  %.2f (z >= %d)\n", a0.gas_buffer_fill, z_powder_top);
    printf("Cells: full=%d, interface=%d, tiny=%d, gas=%d\n\n",
           a0.n_full, a0.n_interface, a0.n_tiny, a0.n_gas);

    // Per-z-layer initial mass
    std::vector<float> layer_mass_init(NZ, 0.0f);
    for (int iz = 0; iz < NZ; iz++)
        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++)
                layer_mass_init[iz] += h_fill[ix + iy*NX + iz*NX*NY];

    // Time loop with mass tracking
    int total_steps = static_cast<int>(t_total / dt + 0.5f);
    int audit_interval = 1000;  // Detailed audit every 1000 steps (10μs)

    printf("%-6s %7s %10s %+10s %8s %8s %8s %8s %8s\n",
           "Step", "t[μs]", "ΣF", "ΔΣF", "Full", "Iface", "Tiny", "→Zero", "Zmax_f");
    printf("------------------------------------------------------------------------------------\n");

    FILE* f_audit = fopen("output_mass_track/mass_audit.csv", "w");
    fprintf(f_audit, "step,t_us,total_fill,delta_fill,substrate,powder,gas_buffer,"
                     "face_xmin,face_xmax,face_ymin,face_ymax,face_zmin,face_zmax,"
                     "n_full,n_interface,n_tiny,n_gas,"
                     "cells_lost_to_zero,fill_lost_to_zero,cells_gained_from_zero\n");

    FILE* f_layer = fopen("output_mass_track/layer_mass.csv", "w");
    fprintf(f_layer, "step,t_us");
    for (int z = 0; z < NZ; z++) fprintf(f_layer, ",z%d", z);
    fprintf(f_layer, "\n");

    std::vector<float> h_f_prev = h_fill;  // Previous fill level
    float cumulative_lost_to_zero = 0.0f;
    int cumulative_cells_lost = 0;

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        if (step > 0) solver.step();

        if (step % audit_interval == 0 || step == total_steps) {
            std::vector<float> h_f(num_cells);
            solver.copyFillLevelToHost(h_f.data());

            MassAudit a = computeMassAudit(h_f, h_f_prev, NX, NY, NZ, z_substrate, z_powder_top);
            float delta = a.total_fill - a0.total_fill;
            cumulative_lost_to_zero += a.fill_lost_to_zero;
            cumulative_cells_lost += a.cells_lost_to_zero;

            // Find max fill in gas buffer (z >= z_pow)
            float zmax_fill = 0;
            for (int iz = z_powder_top; iz < NZ; iz++)
                for (int iy = 0; iy < NY; iy++)
                    for (int ix = 0; ix < NX; ix++) {
                        float f = h_f[ix + iy*NX + iz*NX*NY];
                        if (f > zmax_fill) zmax_fill = f;
                    }

            printf("%6d %7.1f %10.2f %+10.2f %8d %8d %8d %8d %8.3f\n",
                   step, t*1e6f, a.total_fill, delta,
                   a.n_full, a.n_interface, a.n_tiny,
                   a.cells_lost_to_zero, zmax_fill);

            fprintf(f_audit, "%d,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                    "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                    "%d,%d,%d,%d,%d,%.4f,%d\n",
                    step, t*1e6f, a.total_fill, delta,
                    a.substrate_fill, a.powder_fill, a.gas_buffer_fill,
                    a.face_xmin, a.face_xmax, a.face_ymin, a.face_ymax,
                    a.face_zmin, a.face_zmax,
                    a.n_full, a.n_interface, a.n_tiny, a.n_gas,
                    a.cells_lost_to_zero, a.fill_lost_to_zero,
                    a.cells_gained_from_zero);

            // Per-z-layer mass
            fprintf(f_layer, "%d,%.3f", step, t*1e6f);
            for (int iz = 0; iz < NZ; iz++) {
                float lm = 0;
                for (int iy = 0; iy < NY; iy++)
                    for (int ix = 0; ix < NX; ix++)
                        lm += h_f[ix + iy*NX + iz*NX*NY];
                fprintf(f_layer, ",%.2f", lm);
            }
            fprintf(f_layer, "\n");

            h_f_prev = h_f;
        }
    }
    fclose(f_audit);
    fclose(f_layer);

    // Final summary
    std::vector<float> h_f_final(num_cells);
    solver.copyFillLevelToHost(h_f_final.data());

    printf("\n============================================================\n");
    printf("  MASS LOSS SUMMARY\n");
    printf("============================================================\n");

    float final_total = 0;
    for (auto f : h_f_final) final_total += f;
    float total_loss = a0.total_fill - final_total;
    float loss_pct = total_loss / a0.total_fill * 100.0f;

    printf("Initial ΣF:  %.2f\n", a0.total_fill);
    printf("Final ΣF:    %.2f\n", final_total);
    printf("Total loss:  %.2f (%.2f%%)\n\n", total_loss, loss_pct);

    // Per-zone breakdown
    float sub_final = 0, pow_final = 0, gas_final = 0;
    for (int iz = 0; iz < NZ; iz++)
        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++) {
                float f = h_f_final[ix + iy*NX + iz*NX*NY];
                if (iz < z_substrate)       sub_final += f;
                else if (iz < z_powder_top) pow_final += f;
                else                        gas_final += f;
            }

    printf("Zone breakdown:\n");
    printf("  Substrate (z<40):  %.2f → %.2f (Δ=%+.2f)\n",
           a0.substrate_fill, sub_final, sub_final - a0.substrate_fill);
    printf("  Powder (z=40-55):  %.2f → %.2f (Δ=%+.2f)\n",
           a0.powder_fill, pow_final, pow_final - a0.powder_fill);
    printf("  Gas (z>=56):       %.2f → %.2f (Δ=%+.2f)\n\n",
           a0.gas_buffer_fill, gas_final, gas_final - a0.gas_buffer_fill);

    // Boundary face analysis
    MassAudit af = computeMassAudit(h_f_final, h_f_prev, NX, NY, NZ, z_substrate, z_powder_top);
    printf("Boundary face ΣF (final):\n");
    printf("  x_min=%.2f  x_max=%.2f\n", af.face_xmin, af.face_xmax);
    printf("  y_min=%.2f  y_max=%.2f\n", af.face_ymin, af.face_ymax);
    printf("  z_min=%.2f  z_max=%.2f\n\n", af.face_zmin, af.face_zmax);

    // Cutoff analysis
    printf("Cumulative cells that went to f=0: %d\n", cumulative_cells_lost);
    printf("Cumulative fill lost in those cells: %.2f\n", cumulative_lost_to_zero);
    printf("Remaining tiny cells (0<f<0.01): %d\n\n", af.n_tiny);

    // Per-z-layer loss chart
    printf("Per-z-layer mass change (largest losses):\n");
    printf("%-5s %10s %10s %+10s %8s\n", "z", "Initial", "Final", "Delta", "Loss%");
    printf("---------------------------------------------------\n");

    std::vector<float> layer_mass_final(NZ, 0.0f);
    for (int iz = 0; iz < NZ; iz++)
        for (int iy = 0; iy < NY; iy++)
            for (int ix = 0; ix < NX; ix++)
                layer_mass_final[iz] += h_f_final[ix + iy*NX + iz*NX*NY];

    // Sort by delta for top 15 layers
    std::vector<int> layer_idx(NZ);
    for (int i = 0; i < NZ; i++) layer_idx[i] = i;
    std::sort(layer_idx.begin(), layer_idx.end(), [&](int a, int b) {
        float da = layer_mass_final[a] - layer_mass_init[a];
        float db = layer_mass_final[b] - layer_mass_init[b];
        return da < db;  // Most negative first
    });

    for (int i = 0; i < std::min(15, NZ); i++) {
        int z = layer_idx[i];
        float init = layer_mass_init[z];
        float fin = layer_mass_final[z];
        float delta = fin - init;
        float pct = (init > 0.1f) ? delta/init*100.0f : 0.0f;
        if (fabsf(delta) < 0.1f) continue;
        printf("z=%-3d %10.1f %10.1f %+10.1f %+7.1f%%\n", z, init, fin, delta, pct);
    }

    printf("\nOutput: output_mass_track/mass_audit.csv\n");
    printf("        output_mass_track/layer_mass.csv\n");

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Wall time: %.1f s\n", std::chrono::duration<float>(t1 - t0).count());

    return 0;
}
