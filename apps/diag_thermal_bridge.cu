/**
 * @file diag_thermal_bridge.cu
 * @brief Diagnose powder thermal isolation by adding thermal bridges
 *
 * Same setup as benchmark_powder_bed_316L, but:
 *   1. Fill gas gaps between substrate and powder with f=0.3 bridges
 *   2. Set T=1000K at contact cells to bootstrap heat transfer
 *
 * Compare liquid cell count at t=100μs vs baseline (no bridge).
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>
#include <random>
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
    std::strncpy(mat.name, "316L_ThermalBridge", sizeof(mat.name) - 1);
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

/**
 * @brief Generate random powder bed (identical to benchmark_powder_bed_316L)
 */
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

    std::mt19937 rng(42);  // Same seed as production
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
    int attempts_per_pass = 100000;

    for (int pass = 0; pass < 4 && current_solid_volume < target_solid_volume; pass++) {
        float r_base = radii[pass];
        std::uniform_real_distribution<float> dist_r_pass(r_base*0.9f, r_base*1.1f);
        for (int att = 0; att < attempts_per_pass && current_solid_volume < target_solid_volume; att++) {
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

    // Rasterize spheres
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
                    float x = (i + 0.5f) * dx, y = (j + 0.5f) * dx, z = (k + 0.5f) * dx;
                    float d2 = (x-s.cx)*(x-s.cx) + (y-s.cy)*(y-s.cy) + (z-s.cz)*(z-s.cz);
                    if (d2 <= s.r * s.r) fill[i + j*NX + k*NX*NY] = 1.0f;
                }
    }

    // Interface smoothing
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
                    float x = (i + 0.5f) * dx, y = (j + 0.5f) * dx, z = (k + 0.5f) * dx;
                    float dist = sqrtf((x-s.cx)*(x-s.cx) + (y-s.cy)*(y-s.cy) + (z-s.cz)*(z-s.cz));
                    float frac = (s.r + 0.5f*dx - dist) / dx;
                    frac = std::max(0.0f, std::min(1.0f, frac));
                    fill[idx] = std::max(fill[idx], frac);
                }
    }
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_cond=20, mu=0.005f;
    const float alpha=k_cond/(rho*cp), nu=mu/rho;

    const int NX = 80, NY = 160, NZ = 80;
    const float dx = 2.5e-6f;
    const float dt = 1.0e-8f;
    const float t_total = 100.0e-6f;  // 100μs (enough to see effect)
    const int z_substrate = 40;
    const int z_powder_top = 56;

    printf("============================================================\n");
    printf("  Thermal Bridge Diagnostic: Powder-Substrate Contact Fix\n");
    printf("============================================================\n");
    printf("Domain: %d×%d×%d (dx=%.1fμm, dt=%.0fns)\n", NX, NY, NZ, dx*1e6f, dt*1e9f);
    printf("Run to t=%.0fμs, compare liquid cells vs baseline\n\n", t_total*1e6f);

    // Generate powder bed
    const int num_cells = NX*NY*NZ;
    std::vector<float> h_fill;
    initializePowderBed(h_fill, NX, NY, NZ, dx,
                        z_substrate, z_powder_top,
                        10.0e-6f, 15.0e-6f, 0.55f);

    // === THERMAL BRIDGE FIX ===
    // For each (ix,iy), find lowest powder cell above substrate.
    // If there's a gas gap, fill it with f=0.3 bridge material.
    std::vector<float> h_temp(num_cells, 300.0f);
    int bridge_cells = 0;
    int contact_columns = 0;
    int gap_columns = 0;

    for (int jj = 0; jj < NY; jj++) {
        for (int ii = 0; ii < NX; ii++) {
            // Find lowest powder cell in this column
            int lowest_powder_z = -1;
            for (int kk = z_substrate; kk < z_powder_top; kk++) {
                if (h_fill[ii + jj*NX + kk*NX*NY] > 0.1f) {
                    lowest_powder_z = kk;
                    break;
                }
            }

            if (lowest_powder_z < 0) continue;  // No powder in this column

            if (lowest_powder_z == z_substrate) {
                // Direct contact: powder sits on substrate
                contact_columns++;
                // Boost temperature at contact for faster heat transfer
                h_temp[ii + jj*NX + z_substrate*NX*NY] = 1000.0f;
                h_temp[ii + jj*NX + (z_substrate-1)*NX*NY] = 500.0f;
            } else {
                // Gap exists: fill with thermal bridge
                gap_columns++;
                for (int kk = z_substrate; kk < lowest_powder_z; kk++) {
                    int idx = ii + jj*NX + kk*NX*NY;
                    h_fill[idx] = std::max(h_fill[idx], 0.3f);
                    h_temp[idx] = 1000.0f;
                    bridge_cells++;
                }
                // Also warm the powder contact cell
                h_temp[ii + jj*NX + lowest_powder_z*NX*NY] = 1000.0f;
            }
        }
    }

    printf("\n=== Thermal Bridge Statistics ===\n");
    printf("Columns with direct contact: %d\n", contact_columns);
    printf("Columns with gas gap (bridged): %d\n", gap_columns);
    printf("Bridge cells added (f=0.3, T=1000K): %d\n", bridge_cells);
    printf("Total columns with powder: %d / %d (%.1f%%)\n",
           contact_columns + gap_columns, NX*NY,
           100.0f*(contact_columns + gap_columns)/(NX*NY));

    // Recount cell types after bridge
    int n_solid=0, n_gas=0, n_interface=0;
    for (int i = 0; i < num_cells; i++) {
        if (h_fill[i] > 0.99f) n_solid++;
        else if (h_fill[i] < 0.01f) n_gas++;
        else n_interface++;
    }
    printf("Post-bridge: Solid=%d, Gas=%d, Interface=%d\n\n", n_solid, n_gas, n_interface);

    // Setup solver (same as production benchmark)
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
    solver.initialize(h_temp.data(), h_fill.data());

    mkdir("output_thermal_bridge", 0755);

    int total_steps = static_cast<int>(t_total / dt + 0.5f);
    int print_interval = std::max(1, total_steps / 20);
    int vtk_interval = static_cast<int>(25.0e-6f / dt);

    printf("Total steps: %d\n\n", total_steps);
    printf("%-6s %7s %9s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "T_max[K]", "v_max", "Depth", "Liq.Cells", "ΣF");
    printf("----------------------------------------------------------------------\n");

    FILE* f_ts = fopen("output_thermal_bridge/timeseries.csv", "w");
    fprintf(f_ts, "step,t_us,T_max,v_max_phys,depth_um,liquid_cells,total_fill\n");

    float initial_fill_sum = 0;
    for (auto f : h_fill) initial_fill_sum += f;

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        if (step > 0) solver.step();

        bool do_print = (step % print_interval == 0) || (step == total_steps);
        bool do_vtk = (step % vtk_interval == 0);

        if (do_print || do_vtk) {
            float T_max = solver.getMaxTemperature();
            float v_max_phys = solver.getMaxVelocity();

            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            // Melt pool depth
            float max_depth = 0;
            for (int kk = z_substrate-1; kk >= 0; kk--)
                for (int jj = 0; jj < NY; jj++)
                    for (int ii = 0; ii < NX; ii++)
                        if (h_T[ii+jj*NX+kk*NX*NY] >= 1650.0f) {
                            float d = (z_substrate-1-kk)*dx*1e6f;
                            if (d > max_depth) max_depth = d;
                        }

            // Count liquid cells (T >= T_liquidus AND f > 0.01)
            int liquid_cells = 0;
            float total_fill = 0;
            for (int i = 0; i < num_cells; i++) {
                total_fill += h_f[i];
                if (h_T[i] >= 1650.0f && h_f[i] > 0.01f)
                    liquid_cells++;
            }

            if (do_print) {
                printf("%6d %7.1f %9.0f %9.1f %7.1fμm %9d %9.1f\n",
                       step, t*1e6f, T_max, v_max_phys, max_depth,
                       liquid_cells, total_fill);
            }

            fprintf(f_ts, "%d,%.3f,%.1f,%.2f,%.1f,%d,%.1f\n",
                    step, t*1e6f, T_max, v_max_phys, max_depth,
                    liquid_cells, total_fill);

            if (do_vtk) {
                char fname[128];
                snprintf(fname, sizeof(fname), "output_thermal_bridge/bridge_%06d", step);
                const auto& registry = solver.getFieldRegistry();
                io::VTKWriter::writeFields(fname, registry, {}, NX, NY, NZ, dx);
            }
        }
    }
    fclose(f_ts);

    auto t1 = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(t1 - t0).count();
    printf("\nWall time: %.1f s\n", elapsed);
    printf("Initial ΣF: %.1f, compare liquid cells to baseline benchmark\n", initial_fill_sum);

    return 0;
}
