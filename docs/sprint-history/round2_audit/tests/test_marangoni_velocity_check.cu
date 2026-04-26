/**
 * Task 3+7: Controlled Marangoni velocity test
 *
 * Pre-heated liquid slab with imposed linear temperature gradient.
 * NO Darcy, NO phase change, NO evaporation, NO laser, NO recoil.
 * Just Marangoni force on a liquid surface.
 *
 * Analytical: u_surface ≈ |dσ/dT| × |dT/dy| × h / (2μ_eff)
 * where μ_eff = ρ × ν_eff = ρ × (τ-0.5)/3 × dx²/dt
 *
 * Purpose: is the measured velocity consistent with the effective viscosity?
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm;
using namespace lbm::physics;

int main() {
    const float rho=7900, cp=700, k_c=20, mu=0.005f;
    const float alpha=k_c/(rho*cp), nu=mu/rho;
    const int NX=20, NY=80, NZ=20;
    const float dx=2.5e-6f, dt=1.0e-8f;
    const int z_surface=10;
    const int num_cells=NX*NY*NZ;
    const float cs=dx/(dt*sqrtf(3.0f));

    // Effective viscosity from clamped τ
    float tau_eff = 0.51f;  // clamped
    float nu_eff = (tau_eff - 0.5f) / 3.0f * dx * dx / dt;
    float mu_eff = rho * nu_eff;

    printf("================================================================\n");
    printf("  Controlled Marangoni Velocity Test\n");
    printf("  τ_eff=%.3f, ν_eff=%.4e m²/s (%.2f× physical)\n",
           tau_eff, nu_eff, nu_eff/nu);
    printf("  μ_eff=%.4f Pa·s (physical: %.4f)\n", mu_eff, mu);
    printf("================================================================\n");

    MaterialProperties mat = {};
    std::strncpy(mat.name, "316L_ctrl", sizeof(mat.name)-1);
    mat.rho_solid=rho; mat.rho_liquid=rho;
    mat.cp_solid=cp; mat.cp_liquid=cp;
    mat.k_solid=k_c; mat.k_liquid=k_c;
    mat.mu_liquid=mu;
    mat.T_solidus=100; mat.T_liquidus=110;  // Very low → everything is liquid
    mat.T_vaporization=50000.0f;  // Very high → no evaporation
    mat.L_fusion=260000; mat.L_vaporization=7.45e6f;
    mat.molar_mass=0.0558f;
    mat.surface_tension=1.75f;
    mat.dsigma_dT=-4.3e-4f;
    mat.absorptivity_solid=0.35f; mat.absorptivity_liquid=0.35f;
    mat.emissivity=0.3f;

    MultiphysicsConfig config;
    config.nx=NX; config.ny=NY; config.nz=NZ; config.dx=dx; config.dt=dt;
    config.material=mat;
    config.enable_thermal=true; config.enable_thermal_advection=false; // no advection
    config.use_fdm_thermal=true; config.enable_phase_change=false;
    config.enable_fluid=true;
    config.enable_vof=true; config.enable_vof_advection=false;  // static interface
    config.enable_laser=false;
    config.enable_darcy=false;  // NO Darcy
    config.enable_marangoni=true;
    config.enable_surface_tension=false;
    config.enable_buoyancy=false;
    config.enable_evaporation_mass_loss=false;
    config.enable_recoil_pressure=false;
    config.enable_radiation_bc=false;

    config.kinematic_viscosity=nu*dt/(dx*dx);
    config.density=rho;
    config.thermal_diffusivity=alpha; config.ambient_temperature=300.0f;
    config.dsigma_dT=mat.dsigma_dT;
    config.marangoni_csf_multiplier=1.0f;

    config.ray_tracing.enabled=false;
    config.cfl_velocity_target=0.30f;
    config.cfl_use_gradual_scaling=true;
    config.vof_subcycles=1;
    config.enable_vof_mass_correction=false;

    // Periodic in Y (flow direction), wall in X/Z
    config.boundaries.x_min=config.boundaries.x_max=BoundaryType::WALL;
    config.boundaries.y_min=config.boundaries.y_max=BoundaryType::PERIODIC;
    config.boundaries.z_min=config.boundaries.z_max=BoundaryType::WALL;
    config.boundaries.thermal_x_min=config.boundaries.thermal_x_max=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_y_min=config.boundaries.thermal_y_max=ThermalBCType::PERIODIC;
    config.boundaries.thermal_z_min=ThermalBCType::ADIABATIC;
    config.boundaries.thermal_z_max=ThermalBCType::ADIABATIC;
    config.boundaries.dirichlet_temperature=300.0f;

    MultiphysicsSolver solver(config);
    solver.setSmagorinskyCs(0.0f);  // NO Smagorinsky — pure LBM viscosity

    // Initialize: liquid slab z<z_surface, tanh interface
    std::vector<float> h_fill(num_cells, 0.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                float dist = (kk+0.5f) - (float)z_surface;
                h_fill[ii+jj*NX+kk*NX*NY] = 0.5f*(1.0f-tanhf(2.0f*dist));
            }

    // Linear temperature: T(y) = 1000 + 1000 × (y/Ly) → ∇T_y = 1000 / (NY×dx)
    float T_cold = 1000.0f, T_hot = 2000.0f;
    float dTdy = (T_hot - T_cold) / (NY * dx);
    std::vector<float> h_temp(num_cells, 300.0f);
    for (int kk=0; kk<NZ; kk++)
        for (int jj=0; jj<NY; jj++)
            for (int ii=0; ii<NX; ii++) {
                if (h_fill[ii+jj*NX+kk*NX*NY] > 0.01f) {
                    float y = (jj + 0.5f) * dx;
                    h_temp[ii+jj*NX+kk*NX*NY] = T_cold + (T_hot - T_cold) * y / (NY * dx);
                } else {
                    h_temp[ii+jj*NX+kk*NX*NY] = 300.0f;
                }
            }

    solver.initialize(h_temp.data(), h_fill.data());

    // Analytical steady-state Marangoni velocity
    float h = z_surface * dx;  // liquid depth
    float tau_s = fabsf(mat.dsigma_dT) * dTdy;  // surface stress [N/m²]
    float u_analytical_real = tau_s * h / (2.0f * mu);  // with real viscosity
    float u_analytical_eff = tau_s * h / (2.0f * mu_eff);  // with effective viscosity

    printf("\n  Liquid depth: h = %.1f μm (%d cells)\n", h*1e6, z_surface);
    printf("  ∇T_y = %.4e K/m\n", dTdy);
    printf("  τ_s = |dσ/dT| × |∇T| = %.1f N/m²\n", tau_s);
    printf("  u_analytical (real μ):   %.2f m/s (Ma=%.4f)\n", u_analytical_real, u_analytical_real/cs);
    printf("  u_analytical (eff μ):    %.2f m/s (Ma=%.4f)\n", u_analytical_eff, u_analytical_eff/cs);
    printf("\n");

    printf("  %6s %7s %9s %9s %9s %9s\n",
           "Step", "t[μs]", "v_max_y", "v_surf_y", "ratio_eff", "ratio_real");

    // Run and measure
    int total_steps = 10000;
    for (int step=0; step<=total_steps; step++) {
        if (step>0) solver.step();

        if (step % 1000 == 0) {
            std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
            std::vector<float> h_f(num_cells);
            solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());
            solver.copyFillLevelToHost(h_f.data());

            // Measure v_y at surface (z=z_surface-1, near interface)
            float v_max_y = 0, v_surf_y = 0;
            int x_mid = NX/2;
            for (int idx=0; idx<num_cells; idx++) {
                if (h_f[idx] > 0.01f) {
                    float vy = fabsf(h_vy[idx]) * dx / dt;
                    if (vy > v_max_y) v_max_y = vy;
                }
            }
            // Surface velocity at z=z_surface-1 (top liquid cell)
            for (int jj=0; jj<NY; jj++) {
                int idx = x_mid + jj*NX + (z_surface-1)*NX*NY;
                float vy = fabsf(h_vy[idx]) * dx / dt;
                if (vy > v_surf_y) v_surf_y = vy;
            }

            float ratio_eff = (u_analytical_eff > 0) ? v_max_y / u_analytical_eff : 0;
            float ratio_real = (u_analytical_real > 0) ? v_max_y / u_analytical_real : 0;

            printf("  %6d %7.1f %9.3f %9.3f %9.4f %9.4f\n",
                   step, step*dt*1e6, v_max_y, v_surf_y, ratio_eff, ratio_real);
        }
    }

    printf("\n  If ratio_eff → 1.0: velocity matches τ_eff prediction (LBM correct)\n");
    printf("  If ratio_real → 1.0: velocity matches real μ (LBM accurate)\n");
    printf("  If ratio_eff < 0.5: something else is suppressing flow\n");
    return 0;
}
