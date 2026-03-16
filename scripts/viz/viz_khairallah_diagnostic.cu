/**
 * @file viz_khairallah_diagnostic.cu
 * @brief Crime-scene diagnostic for KhairallahMeltPool NaN at step 8
 *
 * Runs the exact same config as test_khairallah_melt_pool.cu for 8 steps,
 * dumping T_max, v_max, force diagnostics, and XZ midplane at each step.
 */

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <float.h>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Check for NaN/Inf in a host array
struct FieldStats {
    float min_val, max_val, mean_val;
    int nan_count, inf_count;
    int nan_first_i, nan_first_j, nan_first_k;  // Location of first NaN
};

FieldStats analyzeField(const float* data, int nx, int ny, int nz, const char* name) {
    FieldStats s = {};
    s.min_val = FLT_MAX;
    s.max_val = -FLT_MAX;
    s.nan_first_i = s.nan_first_j = s.nan_first_k = -1;
    double sum = 0.0;
    int count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float v = data[idx];
                if (std::isnan(v)) {
                    if (s.nan_count == 0) {
                        s.nan_first_i = i;
                        s.nan_first_j = j;
                        s.nan_first_k = k;
                    }
                    s.nan_count++;
                } else if (std::isinf(v)) {
                    s.inf_count++;
                } else {
                    s.min_val = std::min(s.min_val, v);
                    s.max_val = std::max(s.max_val, v);
                    sum += v;
                    count++;
                }
            }
        }
    }
    s.mean_val = count > 0 ? (float)(sum / count) : 0.0f;

    printf("  %-20s: min=%.4e  max=%.4e  mean=%.4e  NaN=%d  Inf=%d",
           name, s.min_val, s.max_val, s.mean_val, s.nan_count, s.inf_count);
    if (s.nan_count > 0) {
        printf("  [first NaN at (%d,%d,%d)]", s.nan_first_i, s.nan_first_j, s.nan_first_k);
    }
    printf("\n");
    return s;
}

void dumpXZMidplane(const char* filename,
                    const float* T, const float* fl, const float* fill,
                    const float* vx, const float* vy, const float* vz,
                    const float* fx, const float* fy, const float* fz,
                    int nx, int ny, int nz, float dx) {
    int j_mid = ny / 2;

    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return; }

    fprintf(fp, "i,k,x_um,z_um,T,fl,fill,vx,vy,vz,vmag,fx,fy,fz,fmag\n");
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j_mid + ny * k);
            float v_mag = sqrtf(vx[idx]*vx[idx] + vy[idx]*vy[idx] + vz[idx]*vz[idx]);
            float f_mag = 0.0f;
            if (fx && fy && fz) {
                f_mag = sqrtf(fx[idx]*fx[idx] + fy[idx]*fy[idx] + fz[idx]*fz[idx]);
            }
            fprintf(fp, "%d,%d,%.2f,%.2f,%.4e,%.6f,%.6f,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                    i, k, i * dx * 1e6f, k * dx * 1e6f,
                    T[idx], fl[idx], fill[idx],
                    vx[idx], vy[idx], vz[idx], v_mag,
                    fx ? fx[idx] : 0.0f,
                    fy ? fy[idx] : 0.0f,
                    fz ? fz[idx] : 0.0f, f_mag);
        }
    }
    fclose(fp);
    printf("  XZ midplane saved to %s\n", filename);
}

int main() {
    printf("=======================================================\n");
    printf("KHAIRALLAH MELT POOL - CRIME SCENE DIAGNOSTIC\n");
    printf("=======================================================\n\n");

    // ========================================================================
    // EXACT SAME CONFIG AS test_khairallah_melt_pool.cu
    // ========================================================================
    const int nx = 300, ny = 200, nz = 100;
    const float dx = 2.0e-6f;
    const float dt = 100e-9f;

    MaterialProperties material = MaterialDatabase::get316L();

    // Print key material properties
    printf("=== 316L Material ===\n");
    printf("  rho_s=%.0f  rho_l=%.0f  cp_s=%.0f  cp_l=%.0f\n",
           material.rho_solid, material.rho_liquid, material.cp_solid, material.cp_liquid);
    printf("  k_s=%.1f  k_l=%.1f  mu_l=%.2e\n",
           material.k_solid, material.k_liquid, material.mu_liquid);
    printf("  T_sol=%.0f  T_liq=%.0f  T_vap=%.0f\n",
           material.T_solidus, material.T_liquidus, material.T_vaporization);
    printf("  L_fus=%.0f  L_vap=%.0f\n", material.L_fusion, material.L_vaporization);
    printf("  sigma=%.2f  dsigma_dT=%.2e\n", material.surface_tension, material.dsigma_dT);

    // Thermal diffusivity
    float alpha = material.k_solid / (material.rho_solid * material.cp_solid);
    float alpha_LU = alpha * dt / (dx * dx);
    float tau_D3Q7 = 4.0f * alpha_LU + 0.5f;
    float omega_D3Q7 = 1.0f / tau_D3Q7;
    printf("\n=== Thermal LBM ===\n");
    printf("  alpha_phys = %.4e m^2/s\n", alpha);
    printf("  alpha_LU   = %.6f\n", alpha_LU);
    printf("  tau_D3Q7   = %.4f (omega=%.4f)\n", tau_D3Q7, omega_D3Q7);

    // Fluid tau
    float nu_LU = 0.0333f;
    float tau_fluid = 3.0f * nu_LU + 0.5f;
    printf("  tau_fluid  = %.4f (nu_LU=%.4f)\n", tau_fluid, nu_LU);

    // Laser
    float P = 195.0f, w0 = 40e-6f, eta = 0.38f, delta = 10e-6f;
    float I0 = 2.0f * P / (M_PI * w0 * w0);
    float beta = 1.0f / delta;
    float q_peak = eta * I0 * beta;
    float dT_per_step = q_peak * dt / (material.rho_solid * material.cp_solid);
    printf("\n=== Laser ===\n");
    printf("  P=%.0fW  w0=%.0fum  eta=%.2f  delta=%.0fum\n", P, w0*1e6, eta, delta*1e6);
    printf("  I0 = %.4e W/m^2\n", I0);
    printf("  q_peak = %.4e W/m^3\n", q_peak);
    printf("  dT/step at peak = %.1f K\n", dT_per_step);
    printf("  P_absorbed = %.1f W\n", P * eta);

    // ========================================================================
    // CONFIGURE SOLVER
    // ========================================================================
    MultiphysicsConfig config;
    config.nx = nx; config.ny = ny; config.nz = nz;
    config.dx = dx; config.dt = dt;
    config.material = material;

    config.enable_thermal = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_surface_tension = true;
    config.enable_marangoni = true;
    config.enable_laser = true;
    config.enable_darcy = true;
    config.enable_buoyancy = true;
    config.enable_evaporation_mass_loss = true;
    config.enable_recoil_pressure = false;
    config.enable_solidification_shrinkage = true;

    config.thermal_diffusivity = alpha;
    config.kinematic_viscosity = 0.0333f;
    config.density = material.rho_liquid;
    config.darcy_coefficient = 1e7f;

    config.surface_tension_coeff = material.surface_tension;
    config.dsigma_dT = material.dsigma_dT;

    config.thermal_expansion_coeff = 1.5e-5f;
    config.gravity_x = 0.0f;
    config.gravity_y = 0.0f;
    config.gravity_z = -9.81f;
    config.reference_temperature = material.T_liquidus;

    config.laser_power = P;
    config.laser_spot_radius = w0;
    config.laser_absorptivity = eta;
    config.laser_penetration_depth = delta;
    config.laser_shutoff_time = 400e-6f;
    config.laser_start_x = 100e-6f;
    config.laser_start_y = ny * dx / 2.0f;
    config.laser_scan_vx = 1.0f;
    config.laser_scan_vy = 0.0f;

    config.boundary_type = 1;
    config.enable_radiation_bc = false;
    config.enable_substrate_cooling = true;
    config.substrate_h_conv = 2000.0f;
    config.substrate_temperature = 300.0f;

    config.vof_subcycles = 10;
    config.cfl_limit = 0.6f;
    config.cfl_velocity_target = 0.15f;

    printf("\n=== Initializing Solver ===\n");
    MultiphysicsSolver solver(config);

    const float T_initial = 300.0f;
    const float interface_height = 0.5f;
    solver.initialize(T_initial, interface_height);

    printf("Solver initialized.\n\n");

    // ========================================================================
    // ALLOCATE HOST BUFFERS
    // ========================================================================
    int N = nx * ny * nz;
    std::vector<float> h_T(N), h_fl(N), h_fill(N);
    std::vector<float> h_vx(N), h_vy(N), h_vz(N);
    std::vector<float> h_fx(N, 0.0f), h_fy(N, 0.0f), h_fz(N, 0.0f);

    // ========================================================================
    // TIME SERIES: T_max, v_max per step
    // ========================================================================
    FILE* ts_fp = fopen("scripts/viz/khairallah_crime_scene_timeseries.csv", "w");
    fprintf(ts_fp, "step,time_ns,T_max,T_min,v_max,fl_max,fill_min,fill_max\n");

    const int MAX_STEPS = 6000;

    printf("=======================================================\n");
    printf("STEP-BY-STEP DIAGNOSTIC (steps 0-%d)\n", MAX_STEPS);
    printf("=======================================================\n\n");

    for (int step = 0; step <= MAX_STEPS; ++step) {
        float t_ns = step * dt * 1e9f;
        float t_us = step * dt * 1e6f;

        // Detailed diagnostic every step for first 10, then every 50 steps
        bool detailed = (step <= 10) || (step % 50 == 0);
        // Lightweight NaN check + T_max every step
        bool check_nan = true;

        if (detailed) {
            printf("--- Step %d (t = %.1f us) ---\n", step, t_us);

            solver.copyTemperatureToHost(h_T.data());
            solver.copyFillLevelToHost(h_fill.data());
            solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());
            if (solver.getLiquidFraction()) {
                CUDA_CHECK(cudaMemcpy(h_fl.data(), solver.getLiquidFraction(),
                                      N * sizeof(float), cudaMemcpyDeviceToHost));
            }

            FieldStats T_stats = analyzeField(h_T.data(), nx, ny, nz, "Temperature [K]");
            FieldStats fl_stats = analyzeField(h_fl.data(), nx, ny, nz, "LiquidFraction");
            FieldStats fill_stats = analyzeField(h_fill.data(), nx, ny, nz, "FillLevel");
            FieldStats vx_stats = analyzeField(h_vx.data(), nx, ny, nz, "Vx [LU]");
            FieldStats vy_stats = analyzeField(h_vy.data(), nx, ny, nz, "Vy [LU]");
            FieldStats vz_stats = analyzeField(h_vz.data(), nx, ny, nz, "Vz [LU]");

            float v_max_LU = 0.0f;
            for (int idx = 0; idx < N; ++idx) {
                float v2 = h_vx[idx]*h_vx[idx] + h_vy[idx]*h_vy[idx] + h_vz[idx]*h_vz[idx];
                v_max_LU = std::max(v_max_LU, sqrtf(v2));
            }
            float v_max_phys = v_max_LU * dx / dt;
            printf("  v_max = %.4e LU = %.4f m/s\n", v_max_LU, v_max_phys);

            fprintf(ts_fp, "%d,%.0f,%.4e,%.4e,%.4e,%.6f,%.6f,%.6f\n",
                    step, t_ns, T_stats.max_val, T_stats.min_val,
                    v_max_LU, fl_stats.max_val, fill_stats.min_val, fill_stats.max_val);

            bool has_nan = (T_stats.nan_count > 0 || vx_stats.nan_count > 0 ||
                            vy_stats.nan_count > 0 || vz_stats.nan_count > 0 ||
                            fl_stats.nan_count > 0 || fill_stats.nan_count > 0);

            if (step <= 10 || has_nan) {
                char fname[256];
                snprintf(fname, sizeof(fname),
                         "scripts/viz/khairallah_xz_step%04d.csv", step);
                dumpXZMidplane(fname, h_T.data(), h_fl.data(), h_fill.data(),
                               h_vx.data(), h_vy.data(), h_vz.data(),
                               nullptr, nullptr, nullptr,
                               nx, ny, nz, dx);
            }

            if (has_nan) {
                printf("\n  *** NaN DETECTED at step %d (t=%.1f us)! ***\n", step, t_us);
                int nan_printed = 0;
                for (int k = 0; k < nz && nan_printed < 20; ++k) {
                    for (int j = 0; j < ny && nan_printed < 20; ++j) {
                        for (int i = 0; i < nx && nan_printed < 20; ++i) {
                            int idx = i + nx * (j + ny * k);
                            bool t_nan = std::isnan(h_T[idx]);
                            bool vx_nan = std::isnan(h_vx[idx]);
                            bool fill_nan = std::isnan(h_fill[idx]);
                            if (t_nan || vx_nan || fill_nan) {
                                printf("    (%3d,%3d,%3d): T=%+.4e  vx=%+.4e  fill=%.4f  fl=%.4f\n",
                                       i, j, k, h_T[idx], h_vx[idx], h_fill[idx], h_fl[idx]);
                                nan_printed++;
                            }
                        }
                    }
                }
                // Dump the crime scene and the step before
                char fname[256];
                snprintf(fname, sizeof(fname),
                         "scripts/viz/khairallah_xz_NaN_step%04d.csv", step);
                dumpXZMidplane(fname, h_T.data(), h_fl.data(), h_fill.data(),
                               h_vx.data(), h_vy.data(), h_vz.data(),
                               nullptr, nullptr, nullptr,
                               nx, ny, nz, dx);
                break;
            }
            printf("\n");
        } else if (check_nan) {
            // Lightweight: just get T_max and v_max from solver
            float T_max = solver.getMaxTemperature();
            float v_max = solver.getMaxVelocity();
            if (std::isnan(T_max) || std::isnan(v_max)) {
                printf("*** NaN DETECTED at step %d (t=%.1f us)! T_max=%e v_max=%e ***\n",
                       step, t_us, T_max, v_max);
                // Dump full state
                solver.copyTemperatureToHost(h_T.data());
                solver.copyFillLevelToHost(h_fill.data());
                solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());
                if (solver.getLiquidFraction()) {
                    CUDA_CHECK(cudaMemcpy(h_fl.data(), solver.getLiquidFraction(),
                                          N * sizeof(float), cudaMemcpyDeviceToHost));
                }
                analyzeField(h_T.data(), nx, ny, nz, "Temperature [K]");
                analyzeField(h_fl.data(), nx, ny, nz, "LiquidFraction");
                analyzeField(h_fill.data(), nx, ny, nz, "FillLevel");
                analyzeField(h_vx.data(), nx, ny, nz, "Vx [LU]");
                analyzeField(h_vy.data(), nx, ny, nz, "Vy [LU]");
                analyzeField(h_vz.data(), nx, ny, nz, "Vz [LU]");

                char fname[256];
                snprintf(fname, sizeof(fname),
                         "scripts/viz/khairallah_xz_NaN_step%04d.csv", step);
                dumpXZMidplane(fname, h_T.data(), h_fl.data(), h_fill.data(),
                               h_vx.data(), h_vy.data(), h_vz.data(),
                               nullptr, nullptr, nullptr,
                               nx, ny, nz, dx);
                break;
            }

            // Print one-liner every 10 steps
            if (step % 10 == 0) {
                fprintf(ts_fp, "%d,%.0f,%.4e,0,%.4e,0,0,0\n",
                        step, t_ns, T_max, v_max);
            }
            if (step % 100 == 0) {
                printf("Step %d (t=%.1f us): T_max=%.0f K  v_max=%.4f m/s\n",
                       step, t_us, T_max, v_max * dx / dt);
            }
        }

        // Advance
        if (step < MAX_STEPS) {
            solver.step(dt);
        }
    }

    fclose(ts_fp);
    printf("\nTime series saved to scripts/viz/khairallah_crime_scene_timeseries.csv\n");
    printf("Done.\n");
    return 0;
}
