/**
 * @file diag_spot_melt_solidify.cu
 * @brief Static spot melting + solidification recoil test
 *
 * Laser at (100,150)μm for 100μs, then OFF.
 * Continue to 300μs and observe surface tension pulling liquid into a dome.
 * Reports z_max of solidified bump to validate surface tension + wetting.
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
    std::strncpy(mat.name, "316L_SpotMelt", sizeof(mat.name) - 1);
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
    std::vector<float>& fill, int NX, int NY, int NZ, float dx,
    int z_sub, int z_pow, float r_min, float r_max, float target_density)
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
    std::uniform_real_distribution<float> dist_x(0, NX*dx), dist_y(0, NY*dx), dist_z(z_sub*dx, z_pow*dx);
    float powder_vol = NX*dx * NY*dx * (z_pow-z_sub)*dx;
    float target_vol = target_density * powder_vol;
    float cur_vol = 0; int placed = 0;
    float cell_size = 2.0f*r_max;
    int hNX=(int)(NX*dx/cell_size)+1, hNY=(int)(NY*dx/cell_size)+1, hNZ=(int)((z_pow-z_sub)*dx/cell_size)+1;
    std::vector<std::vector<int>> hash(hNX*hNY*hNZ);
    auto hidx = [&](float x, float y, float z) {
        return std::max(0,std::min((int)(x/cell_size),hNX-1))
             + std::max(0,std::min((int)(y/cell_size),hNY-1))*hNX
             + std::max(0,std::min((int)((z-z_sub*dx)/cell_size),hNZ-1))*hNX*hNY;
    };
    auto overlap = [&](float cx, float cy, float cz, float r) {
        int hi=(int)(cx/cell_size), hj=(int)(cy/cell_size), hk=(int)((cz-z_sub*dx)/cell_size);
        for (int di=-1;di<=1;di++) for (int dj=-1;dj<=1;dj++) for (int dk=-1;dk<=1;dk++) {
            int ni=hi+di,nj=hj+dj,nk=hk+dk;
            if (ni<0||ni>=hNX||nj<0||nj>=hNY||nk<0||nk>=hNZ) continue;
            for (int si : hash[ni+nj*hNX+nk*hNX*hNY]) {
                auto& s=spheres[si]; float d2=(cx-s.cx)*(cx-s.cx)+(cy-s.cy)*(cy-s.cy)+(cz-s.cz)*(cz-s.cz);
                if (d2<(r+s.r)*(r+s.r)*0.95f) return true;
            }
        } return false;
    };
    float radii[]={r_max,(r_min+r_max)*0.5f,r_min,r_min*0.7f};
    for (int pass=0; pass<4 && cur_vol<target_vol; pass++) {
        float rb=radii[pass]; std::uniform_real_distribution<float> dr(rb*0.9f,rb*1.1f);
        for (int a=0; a<100000 && cur_vol<target_vol; a++) {
            float r=dr(rng), cx=dist_x(rng), cy=dist_y(rng), cz=dist_z(rng);
            cz=std::max(z_sub*dx+r,std::min(z_pow*dx-r,cz));
            if (overlap(cx,cy,cz,r)) continue;
            spheres.push_back({cx,cy,cz,r}); hash[hidx(cx,cy,cz)].push_back(placed);
            cur_vol+=(4.f/3.f)*3.14159265f*r*r*r; placed++;
        }
    }
    printf("Powder: %d particles, packing=%.1f%%\n", placed, cur_vol/powder_vol*100);
    for (auto& s:spheres) {
        int imin=std::max(0,(int)((s.cx-s.r)/dx)-1), imax=std::min(NX-1,(int)((s.cx+s.r)/dx)+1);
        int jmin=std::max(0,(int)((s.cy-s.r)/dx)-1), jmax=std::min(NY-1,(int)((s.cy+s.r)/dx)+1);
        int kmin=std::max(z_sub,(int)((s.cz-s.r)/dx)-1), kmax=std::min(z_pow-1,(int)((s.cz+s.r)/dx)+1);
        for (int k=kmin;k<=kmax;k++) for (int j=jmin;j<=jmax;j++) for (int i=imin;i<=imax;i++) {
            float x=(i+.5f)*dx,y=(j+.5f)*dx,z=(k+.5f)*dx;
            if ((x-s.cx)*(x-s.cx)+(y-s.cy)*(y-s.cy)+(z-s.cz)*(z-s.cz)<=s.r*s.r) fill[i+j*NX+k*NX*NY]=1.f;
        }
    }
    for (auto& s:spheres) {
        int imin=std::max(0,(int)((s.cx-s.r-dx)/dx)), imax=std::min(NX-1,(int)((s.cx+s.r+dx)/dx));
        int jmin=std::max(0,(int)((s.cy-s.r-dx)/dx)), jmax=std::min(NY-1,(int)((s.cy+s.r+dx)/dx));
        int kmin=std::max(z_sub,(int)((s.cz-s.r-dx)/dx)), kmax=std::min(z_pow-1,(int)((s.cz+s.r+dx)/dx));
        for (int k=kmin;k<=kmax;k++) for (int j=jmin;j<=jmax;j++) for (int i=imin;i<=imax;i++) {
            int idx=i+j*NX+k*NX*NY; if (fill[idx]>=1.f) continue;
            float x=(i+.5f)*dx,y=(j+.5f)*dx,z=(k+.5f)*dx;
            float d=sqrtf((x-s.cx)*(x-s.cx)+(y-s.cy)*(y-s.cy)+(z-s.cz)*(z-s.cz));
            float f=std::max(0.f,std::min(1.f,(s.r+.5f*dx-d)/dx));
            fill[idx]=std::max(fill[idx],f);
        }
    }
    // Contact fix
    for (int j=0;j<NY;j++) for (int i=0;i<NX;i++) {
        int lp=-1;
        for (int k=z_sub;k<z_pow;k++) if (fill[i+j*NX+k*NX*NY]>0.1f) { lp=k; break; }
        if (lp<=z_sub) continue;
        for (int k=z_sub;k<lp;k++) { int idx=i+j*NX+k*NX*NY; if (fill[idx]<1.f) fill[idx]=1.f; }
    }
}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();
    MaterialProperties mat = createMat();
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;

    const int NX=80, NY=160, NZ=80;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const float t_laser_off = 100.0e-6f;  // Laser OFF at 100μs
    const float t_total     = 300.0e-6f;  // Run until 300μs
    const int z_substrate=40, z_powder_top=56;
    const int num_cells = NX*NY*NZ;

    printf("============================================================\n");
    printf("  Static Spot Melting + Solidification Recoil Test\n");
    printf("============================================================\n");
    printf("Laser at (100,150)μm, ON for %.0fμs, then OFF\n", t_laser_off*1e6f);
    printf("Total simulation: %.0fμs (%.0fμs cooling after laser off)\n\n",
           t_total*1e6f, (t_total-t_laser_off)*1e6f);

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

    config.kinematic_viscosity = nu*dt/(dx*dx);
    config.density=rho; config.darcy_coefficient=1e6f;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;

    config.surface_tension_coeff=mat.surface_tension;
    config.dsigma_dT=mat.dsigma_dT;
    config.recoil_force_multiplier=1.0f;
    config.recoil_max_pressure=1e8f;
    config.marangoni_csf_multiplier=4.0f;
    config.evap_cooling_factor=1.0f;

    // STATIONARY laser at (100μm, 150μm)
    config.laser_power=150.0f;
    config.laser_spot_radius=25.0e-6f;
    config.laser_absorptivity=0.35f;
    config.laser_penetration_depth=dx;
    config.laser_shutoff_time=t_laser_off;  // OFF at 100μs
    config.laser_start_x=100.0e-6f;
    config.laser_start_y=150.0e-6f;
    config.laser_scan_vx=0.0f;
    config.laser_scan_vy=0.0f;   // STATIONARY

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

    std::vector<float> h_fill;
    initializePowderBed(h_fill, NX, NY, NZ, dx,
                        z_substrate, z_powder_top, 10e-6f, 15e-6f, 0.55f);
    std::vector<float> h_temp(num_cells, 300.0f);
    solver.initialize(h_temp.data(), h_fill.data());

    mkdir("output_spot_melt", 0755);

    int total_steps = (int)(t_total/dt + 0.5f);
    int vtk_interval = (int)(25.0e-6f/dt);
    int print_interval = std::max(1, total_steps/60);

    printf("%-6s %7s %5s %9s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "Laser", "T_max[K]", "v_max", "LiqCells", "z_max_f", "SumF");
    printf("------------------------------------------------------------------------\n");

    FILE* f_ts = fopen("output_spot_melt/timeseries.csv", "w");
    fprintf(f_ts, "step,t_us,laser_on,T_max,v_max,liquid_cells,z_max_fill_um,sum_fill\n");

    for (int step = 0; step <= total_steps; step++) {
        float t = step * dt;
        if (step > 0) solver.step();

        bool do_print = (step % print_interval == 0) || (step == total_steps);
        bool do_vtk = (step % vtk_interval == 0);

        if (do_print || do_vtk) {
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity() * dx / dt;
            bool laser_on = (t <= t_laser_off);

            std::vector<float> h_T(num_cells), h_f(num_cells);
            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_f.data());

            int liquid_cells = 0;
            float sum_fill = 0;
            float z_max_fill = 0;  // highest z with f>0.5 (in μm)

            for (int kk = 0; kk < NZ; kk++)
                for (int jj = 0; jj < NY; jj++)
                    for (int ii = 0; ii < NX; ii++) {
                        int idx = ii + jj*NX + kk*NX*NY;
                        sum_fill += h_f[idx];
                        if (h_T[idx] >= 1650.0f && h_f[idx] > 0.01f)
                            liquid_cells++;
                        if (h_f[idx] > 0.5f) {
                            float z_um = (kk + 0.5f) * dx * 1e6f;
                            if (z_um > z_max_fill) z_max_fill = z_um;
                        }
                    }

            if (do_print)
                printf("%6d %7.1f %5s %9.0f %9.1f %9d %9.1fμm %9.0f\n",
                       step, t*1e6f, laser_on?"ON":"OFF",
                       T_max, v_max, liquid_cells, z_max_fill, sum_fill);

            fprintf(f_ts, "%d,%.3f,%d,%.1f,%.2f,%d,%.2f,%.1f\n",
                    step, t*1e6f, laser_on?1:0, T_max, v_max,
                    liquid_cells, z_max_fill, sum_fill);

            if (do_vtk) {
                char fname[128];
                snprintf(fname, sizeof(fname), "output_spot_melt/spot_%06d", step);
                const auto& reg = solver.getFieldRegistry();
                io::VTKWriter::writeFields(fname, reg, {}, NX, NY, NZ, dx);
            }
        }
    }
    fclose(f_ts);

    // Final z_max report
    {
        std::vector<float> h_f(num_cells);
        solver.copyFillLevelToHost(h_f.data());
        // Find max z near laser spot (within 30μm radius of laser center)
        float lx = 100e-6f, ly = 150e-6f;
        float z_max_spot = 0, z_max_global = 0;
        for (int kk = NZ-1; kk >= 0; kk--)
            for (int jj = 0; jj < NY; jj++)
                for (int ii = 0; ii < NX; ii++) {
                    if (h_f[ii+jj*NX+kk*NX*NY] < 0.5f) continue;
                    float z_um = (kk+0.5f)*dx*1e6f;
                    if (z_um > z_max_global) z_max_global = z_um;
                    float x=(ii+0.5f)*dx, y=(jj+0.5f)*dx;
                    float r2=(x-lx)*(x-lx)+(y-ly)*(y-ly);
                    if (r2 < 30e-6f*30e-6f && z_um > z_max_spot)
                        z_max_spot = z_um;
                }
        float substrate_top = z_substrate * dx * 1e6f;
        printf("\n============================================================\n");
        printf("  SOLIDIFICATION RESULT (t=300μs)\n");
        printf("============================================================\n");
        printf("Substrate top:        %.1f μm\n", substrate_top);
        printf("z_max (global):       %.1f μm\n", z_max_global);
        printf("z_max (near spot):    %.1f μm\n", z_max_spot);
        printf("Bump height (spot):   %.1f μm above substrate\n", z_max_spot - substrate_top);
        printf("============================================================\n");
    }

    printf("Wall time: %.1f s\n", std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count());
    return 0;
}
