/**
 * @file benchmark_powder_bed_316L.cu
 * @brief LPBF Powder Bed Single-Track: laserMeltFoam tutorial 1:1 reproduction
 *
 * Domain: 200×800×200 μm (80×320×80 cells at dx=2.5μm)
 * Layout:
 *   z=0-100μm (0-39):    solid substrate (f=1, T=300K)
 *   z=100-140μm (40-55): powder layer with random spherical particles
 *   z=140-200μm (56-79): gas buffer (f=0)
 *
 * Laser: P=150W, r0=25μm, v=1000mm/s, scan along +Y
 *        start (100,100)μm → end (100,700)μm
 *
 * All physics: VOF+PLIC, recoil, evaporation, Marangoni CSF×4,
 * surface tension, Darcy, Smagorinsky LES, FDM WENO5 thermal
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
    std::strncpy(mat.name, "316L_PowderBed", sizeof(mat.name) - 1);
    mat.rho_solid=7900; mat.rho_liquid=7900;
    mat.cp_solid=700; mat.cp_liquid=700;
    mat.k_solid=20; mat.k_liquid=20;
    mat.mu_liquid=0.005f;
    mat.T_solidus=1650; mat.T_liquidus=1700;
    mat.T_vaporization=3200.0f;
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=+1e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;
    mat.emissivity=0.3f;
    return mat;
}

/**
 * @brief Generate random powder bed with spherical particles
 *
 * Places non-overlapping spheres in the powder layer zone.
 * Uses rejection sampling: try random positions, skip if overlapping.
 *
 * @param fill     Output fill_level array (NX*NY*NZ)
 * @param NX,NY,NZ Grid dimensions
 * @param dx       Grid spacing [m]
 * @param z_sub    Substrate top z-index (exclusive: z < z_sub is solid)
 * @param z_pow    Powder top z-index (exclusive: z_sub <= z < z_pow is powder zone)
 * @param r_min    Min particle radius [m]
 * @param r_max    Max particle radius [m]
 * @param target_density Target packing fraction in powder zone (0-1)
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

    // Substrate: solid metal
    for (int k = 0; k < z_sub; k++)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
                fill[i + j*NX + k*NX*NY] = 1.0f;

    // Powder zone: random sphere packing
    struct Sphere { float cx, cy, cz, r; };
    std::vector<Sphere> spheres;

    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist_x(0, NX * dx);
    std::uniform_real_distribution<float> dist_y(0, NY * dx);
    std::uniform_real_distribution<float> dist_z(z_sub * dx, z_pow * dx);
    std::uniform_real_distribution<float> dist_r(r_min, r_max);

    float powder_volume = NX * dx * NY * dx * (z_pow - z_sub) * dx;
    float target_solid_volume = target_density * powder_volume;
    float current_solid_volume = 0.0f;

    int max_attempts = 500000;
    int placed = 0;

    for (int attempt = 0; attempt < max_attempts && current_solid_volume < target_solid_volume; attempt++) {
        float r = dist_r(rng);
        float cx = dist_x(rng);
        float cy = dist_y(rng);
        float cz = dist_z(rng);

        // Clamp center so sphere stays within powder zone
        cz = std::max(z_sub * dx + r, std::min((z_pow * dx) - r, cz));

        // Check overlap with existing spheres (min gap = 0.5*dx)
        bool overlap = false;
        for (const auto& s : spheres) {
            float d2 = (cx-s.cx)*(cx-s.cx) + (cy-s.cy)*(cy-s.cy) + (cz-s.cz)*(cz-s.cz);
            float min_d = r + s.r + 0.5f * dx;
            if (d2 < min_d * min_d) { overlap = true; break; }
        }
        if (overlap) continue;

        spheres.push_back({cx, cy, cz, r});
        current_solid_volume += (4.0f/3.0f) * 3.14159265f * r*r*r;
        placed++;
    }

    float actual_density = current_solid_volume / powder_volume;
    printf("Powder bed: %d particles placed, packing = %.1f%%\n", placed, actual_density * 100);

    // Rasterize spheres to fill_level
    for (const auto& s : spheres) {
        // Bounding box in grid indices
        int i_min = std::max(0, (int)((s.cx - s.r) / dx) - 1);
        int i_max = std::min(NX-1, (int)((s.cx + s.r) / dx) + 1);
        int j_min = std::max(0, (int)((s.cy - s.r) / dx) - 1);
        int j_max = std::min(NY-1, (int)((s.cy + s.r) / dx) + 1);
        int k_min = std::max(z_sub, (int)((s.cz - s.r) / dx) - 1);
        int k_max = std::min(z_pow-1, (int)((s.cz + s.r) / dx) + 1);

        for (int k = k_min; k <= k_max; k++)
            for (int j = j_min; j <= j_max; j++)
                for (int i = i_min; i <= i_max; i++) {
                    float x = (i + 0.5f) * dx;
                    float y = (j + 0.5f) * dx;
                    float z = (k + 0.5f) * dx;
                    float d2 = (x-s.cx)*(x-s.cx) + (y-s.cy)*(y-s.cy) + (z-s.cz)*(z-s.cz);
                    if (d2 <= s.r * s.r) {
                        fill[i + j*NX + k*NX*NY] = 1.0f;
                    }
                }
    }

    // Interface smoothing: cells partially covered get fractional fill
    // (Simple: cells on sphere boundary get f based on distance)
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
                    if (fill[idx] >= 1.0f) continue;  // already full
                    float x = (i + 0.5f) * dx;
                    float y = (j + 0.5f) * dx;
                    float z = (k + 0.5f) * dx;
                    float dist = sqrtf((x-s.cx)*(x-s.cx) + (y-s.cy)*(y-s.cy) + (z-s.cz)*(z-s.cz));
                    float frac = (s.r + 0.5f*dx - dist) / dx;
                    frac = std::max(0.0f, std::min(1.0f, frac));
                    fill[idx] = std::max(fill[idx], frac);
                }
    }

    // Stats
    int n_solid=0, n_gas=0, n_interface=0;
    for (int i = 0; i < num_cells; i++) {
        if (fill[i] > 0.99f) n_solid++;
        else if (fill[i] < 0.01f) n_gas++;
        else n_interface++;
    }
    printf("  Solid: %d (%.1f%%), Gas: %d (%.1f%%), Interface: %d (%.1f%%)\n",
           n_solid, 100.0f*n_solid/num_cells,
           n_gas, 100.0f*n_gas/num_cells,
           n_interface, 100.0f*n_interface/num_cells);
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k=20, mu=0.005f;
    const float alpha=k/(rho*cp), nu=mu/rho;

    // Domain: 200×800×200 μm at dx=2.5μm
    const int NX = 80, NY = 320, NZ = 80;
    const float dx = 2.5e-6f;
    const float dt = 1.0e-8f;  // 10 ns
    const float t_total = 200.0e-6f;  // 200 μs initially

    printf("============================================================\n");
    printf("  LPBF Powder Bed Single-Track: 316L\n");
    printf("  laserMeltFoam Tutorial Reproduction\n");
    printf("============================================================\n");
    printf("Domain: %d×%d×%d = %.1fM cells (dx=%.1fμm)\n",
           NX, NY, NZ, NX*NY*NZ/1e6f, dx*1e6f);
    printf("dt=%.0f ns, t_total=%.0f μs\n\n", dt*1e9f, t_total*1e6f);

    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;

    // All physics ON
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

    // Fluid
    config.kinematic_viscosity = nu * dt / (dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;

    // Thermal
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    // Surface
    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=1.0f;
    config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=4.0f;
    config.evap_cooling_factor=1.0f;

    // Laser: scan along +Y
    config.laser_power=150.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=-1.0f;
    config.laser_start_x=100.0e-6f;   // center X
    config.laser_start_y=100.0e-6f;   // start Y
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=1.0f;         // 1000 mm/s along +Y

    config.ray_tracing.enabled=false;

    // CFL
    config.cfl_velocity_target=0.15f;
    config.cfl_use_gradual_scaling=true;

    // VOF
    config.vof_subcycles=1;
    config.enable_vof_mass_correction=false;

    // Boundaries
    config.boundaries.x_min=config.boundaries.x_max=BoundaryType::WALL;
    config.boundaries.y_min=config.boundaries.y_max=BoundaryType::WALL;
    config.boundaries.z_min=config.boundaries.z_max=BoundaryType::WALL;
    config.boundaries.thermal_x_min=config.boundaries.thermal_x_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min=config.boundaries.thermal_y_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_min=ThermalBCType::DIRICHLET;
    config.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature=300.0f;

    // Initialize solver
    MultiphysicsSolver solver(config);
    solver.setSmagorinskyCs(0.20f);

    // Generate powder bed
    const int num_cells = NX*NY*NZ;
    std::vector<float> h_fill;
    const int z_substrate = 40;  // z < 40 = solid (100μm)
    const int z_powder_top = 56; // z 40-55 = powder zone (40μm = 16 cells)

    initializePowderBed(h_fill, NX, NY, NZ, dx,
                        z_substrate, z_powder_top,
                        10.0e-6f, 15.0e-6f,  // radius 10-15μm
                        0.55f);               // target 55% packing

    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    // Output
    mkdir("output_powder", 0755);

    int total_steps = static_cast<int>(t_total / dt + 0.5f);
    int vtk_interval = static_cast<int>(25.0e-6f / dt);
    int print_interval = std::max(1, total_steps / 40);

    printf("\nTotal steps: %d, VTK every %d steps\n\n", total_steps, vtk_interval);
    printf("%-6s %7s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "T_max[K]", "v_max", "Depth", "laser_y");
    printf("--------------------------------------------------------------\n");

    FILE* f_ts = fopen("output_powder/timeseries.csv", "w");
    fprintf(f_ts, "step,t_us,T_max,v_max_phys,depth_um,laser_y_um\n");

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        if (step > 0) solver.step();

        bool do_print = (step % print_interval == 0) || (step == total_steps);
        bool do_vtk = (step % vtk_interval == 0);

        if (do_print || do_vtk) {
            float T_max = solver.getMaxTemperature();
            float v_max_LU = solver.getMaxVelocity();
            float v_max_phys = v_max_LU * dx / dt;
            float laser_y = (config.laser_start_y + 1.0f * t) * 1e6f;

            // Depth
            std::vector<float> h_T(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            float max_depth = 0;
            for (int kk = z_substrate-1; kk >= 0; kk--)
                for (int jj = 0; jj < NY; jj++)
                    for (int ii = 0; ii < NX; ii++)
                        if (h_T[ii+jj*NX+kk*NX*NY] >= 1650.0f) {
                            float d = (z_substrate-1-kk)*dx*1e6f;
                            if (d > max_depth) max_depth = d;
                        }

            if (do_print)
                printf("%6d %7.1f %9.0f %9.1f %7.1fμm %7.1fμm\n",
                       step, t*1e6f, T_max, v_max_phys, max_depth, laser_y);

            fprintf(f_ts, "%d,%.3f,%.1f,%.2f,%.1f,%.1f\n",
                    step, t*1e6f, T_max, v_max_phys, max_depth, laser_y);
        }

        if (do_vtk) {
            char fname[128];
            snprintf(fname, sizeof(fname), "output_powder/powder_%06d", step);
            const auto& registry = solver.getFieldRegistry();
            io::VTKWriter::writeFields(fname, registry, {},
                                       NX, NY, NZ, dx);
        }
    }

    fclose(f_ts);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\nWall time: %.1f s\n", std::chrono::duration<float>(t1-t0).count());
    return 0;
}
