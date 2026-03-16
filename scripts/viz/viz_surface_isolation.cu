/**
 * @file viz_surface_isolation.cu
 * @brief Isolation tests: identify which surface force causes NaN in LaserMeltingIron
 *
 * Runs three configurations of the exact LaserMeltingIron setup:
 *   A) Full physics (baseline) — expected NaN at step ~23
 *   B) Marangoni OFF (dsigma_dT=0), surface tension ON
 *   C) Both Marangoni and surface tension OFF
 *
 * Also dumps curvature + force diagnostics at step 20 for config A.
 *
 * Usage: ./viz_surface_isolation
 */

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

static const std::string OUT_DIR = "/home/yzk/LBMProject/scripts/viz/";

/// Build the exact LaserMeltingIron config (shared by all 3 runs)
MultiphysicsConfig buildBaseConfig() {
    MultiphysicsConfig config;

    config.nx = 40;
    config.ny = 80;
    config.nz = 80;
    config.dx = 3.75e-6f;
    config.dt = 75e-9f;

    config.material = MaterialDatabase::getSteel();

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
    config.enable_buoyancy = false;
    config.enable_evaporation_mass_loss = true;
    config.enable_recoil_pressure = false;

    float alpha_phys = config.material.k_liquid /
                       (config.material.rho_liquid * config.material.cp_liquid);
    float nu_physical = config.material.mu_liquid / config.material.rho_liquid;
    float nu_lattice_phys = nu_physical * config.dt / (config.dx * config.dx);
    float nu_min = 0.0333f * 1.5f;
    config.kinematic_viscosity = std::max(nu_lattice_phys, nu_min);
    config.density = config.material.rho_liquid;
    config.darcy_coefficient = 1e7f;

    config.thermal_diffusivity = alpha_phys;
    config.enable_radiation_bc = true;
    config.enable_substrate_cooling = true;
    config.substrate_h_conv = 5.0f;
    config.substrate_temperature = 300.0f;

    config.surface_tension_coeff = config.material.surface_tension;
    config.dsigma_dT = config.material.dsigma_dT;

    float domain_x = config.nx * config.dx;
    float domain_y = config.ny * config.dx;
    config.laser_power = 250.0f;
    config.laser_spot_radius = 50e-6f;
    config.laser_absorptivity = 0.43f;
    config.laser_penetration_depth = 20e-6f;
    config.laser_shutoff_time = 60e-6f;
    config.laser_start_x = domain_x / 2.0f;
    config.laser_start_y = domain_y / 2.0f;
    config.laser_scan_vx = 0.0f;
    config.laser_scan_vy = 0.0f;

    config.cfl_velocity_target = 0.1f;
    config.cfl_use_gradual_scaling = true;
    config.cfl_force_ramp_factor = 0.8f;

    config.vof_subcycles = 100;

    config.boundaries.x_min = BoundaryType::PERIODIC;
    config.boundaries.x_max = BoundaryType::PERIODIC;
    config.boundaries.y_min = BoundaryType::PERIODIC;
    config.boundaries.y_max = BoundaryType::PERIODIC;
    config.boundaries.z_min = BoundaryType::WALL;
    config.boundaries.z_max = BoundaryType::PERIODIC;
    config.boundaries.thermal_z_min = ThermalBCType::CONVECTIVE;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.convective_h = 5.0f;
    config.boundaries.convective_T_inf = 300.0f;

    return config;
}

/// Run a single configuration and report per-step diagnostics
struct StepResult {
    int step;
    float max_v;
    float max_T;
    bool has_nan;
};

std::vector<StepResult> runConfig(const std::string& name,
                                  MultiphysicsConfig& config,
                                  int max_steps,
                                  bool dump_curvature_at_step20 = false) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Config: " << name << std::endl;
    std::cout << "  ST=" << config.enable_surface_tension
              << " Marangoni=" << config.enable_marangoni
              << " sigma=" << config.surface_tension_coeff
              << " dsigma_dT=" << config.dsigma_dT << std::endl;
    std::cout << "========================================" << std::endl;

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 1.0f);

    const int NC = config.nx * config.ny * config.nz;
    std::vector<StepResult> results;

    for (int step = 1; step <= max_steps; ++step) {
        solver.step();

        StepResult r;
        r.step = step;
        r.has_nan = solver.checkNaN();
        r.max_v = solver.getMaxVelocity();
        r.max_T = solver.getMaxTemperature();
        results.push_back(r);

        // Print every step from 15 onward, or every 5 steps before that
        if (step >= 15 || step % 5 == 0) {
            std::cout << "  Step " << std::setw(3) << step
                      << ": max|u|=" << std::scientific << std::setprecision(3) << r.max_v
                      << " max_T=" << std::fixed << std::setprecision(1) << r.max_T << " K"
                      << (r.has_nan ? " *** NaN ***" : "") << std::endl;
        }

        // Curvature + fill + force diagnostics at step 20 (config A only)
        if (dump_curvature_at_step20 && step == 20) {
            std::cout << "\n  --- Curvature/Normal diagnostics at step 20 ---" << std::endl;

            std::vector<float> h_curv(NC), h_fill(NC), h_T(NC), h_fl(NC);
            std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC);

            CUDA_CHECK(cudaMemcpy(h_curv.data(), solver.getCurvature(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_fill.data(), solver.getFillLevel(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_T.data(), solver.getTemperature(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_fl.data(), solver.getLiquidFraction(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ux.data(), solver.getVelocityX(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy.data(), solver.getVelocityY(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uz.data(), solver.getVelocityZ(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));

            // Find top 20 cells by |curvature|
            struct CurvInfo {
                int idx;
                float kappa;
                float fill;
                float T;
                float fl;
                float vmag;
            };
            std::vector<CurvInfo> interface_cells;
            interface_cells.reserve(10000);

            for (int idx = 0; idx < NC; ++idx) {
                if (h_fill[idx] > 0.01f && h_fill[idx] < 0.99f) {
                    float vmag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx] +
                                           h_uz[idx]*h_uz[idx]);
                    interface_cells.push_back({idx, h_curv[idx], h_fill[idx],
                                               h_T[idx], h_fl[idx], vmag});
                }
            }

            // Sort by |curvature| descending
            std::sort(interface_cells.begin(), interface_cells.end(),
                      [](const CurvInfo& a, const CurvInfo& b) {
                          return std::abs(a.kappa) > std::abs(b.kappa);
                      });

            std::cout << "  Interface cells: " << interface_cells.size() << std::endl;
            std::cout << "  Top 20 by |kappa|:" << std::endl;
            std::cout << "  " << std::setw(5) << "i" << std::setw(5) << "j" << std::setw(5) << "k"
                      << std::setw(14) << "kappa" << std::setw(10) << "fill"
                      << std::setw(10) << "T(K)" << std::setw(10) << "fl"
                      << std::setw(14) << "|u|" << std::endl;

            int show = std::min((int)interface_cells.size(), 20);
            for (int n = 0; n < show; ++n) {
                auto& c = interface_cells[n];
                int ii = c.idx % config.nx;
                int jj = (c.idx / config.nx) % config.ny;
                int kk = c.idx / (config.nx * config.ny);
                std::cout << "  " << std::setw(5) << ii << std::setw(5) << jj << std::setw(5) << kk
                          << std::scientific << std::setprecision(3)
                          << std::setw(14) << c.kappa
                          << std::fixed << std::setprecision(4)
                          << std::setw(10) << c.fill
                          << std::setprecision(1)
                          << std::setw(10) << c.T
                          << std::setprecision(4)
                          << std::setw(10) << c.fl
                          << std::scientific << std::setprecision(3)
                          << std::setw(14) << c.vmag
                          << std::endl;
            }

            // Curvature statistics
            float kappa_max = 0, kappa_mean = 0;
            for (auto& c : interface_cells) {
                float ak = std::abs(c.kappa);
                if (ak > kappa_max) kappa_max = ak;
                kappa_mean += ak;
            }
            if (!interface_cells.empty()) kappa_mean /= interface_cells.size();
            float kappa_limit = 2.0f / config.dx;
            std::cout << "  kappa: max=" << std::scientific << kappa_max
                      << " mean=" << kappa_mean
                      << " limit(2/dx)=" << kappa_limit << std::endl;

            // Top 20 by velocity
            std::sort(interface_cells.begin(), interface_cells.end(),
                      [](const CurvInfo& a, const CurvInfo& b) {
                          return a.vmag > b.vmag;
                      });
            std::cout << "\n  Top 20 interface cells by |u|:" << std::endl;
            std::cout << "  " << std::setw(5) << "i" << std::setw(5) << "j" << std::setw(5) << "k"
                      << std::setw(14) << "|u|" << std::setw(14) << "kappa"
                      << std::setw(10) << "fill" << std::setw(10) << "T(K)" << std::endl;
            show = std::min((int)interface_cells.size(), 20);
            for (int n = 0; n < show; ++n) {
                auto& c = interface_cells[n];
                int ii = c.idx % config.nx;
                int jj = (c.idx / config.nx) % config.ny;
                int kk = c.idx / (config.nx * config.ny);
                std::cout << "  " << std::setw(5) << ii << std::setw(5) << jj << std::setw(5) << kk
                          << std::scientific << std::setprecision(3)
                          << std::setw(14) << c.vmag
                          << std::setw(14) << c.kappa
                          << std::fixed << std::setprecision(4)
                          << std::setw(10) << c.fill
                          << std::setprecision(1)
                          << std::setw(10) << c.T
                          << std::endl;
            }

            // Dump full curvature diagnostics CSV for z=74-77 (NaN ring zone)
            std::string csv_path = OUT_DIR + "surface_diagnostics_step20.csv";
            std::ofstream csv(csv_path);
            csv << "i,j,k,fill,curvature,temperature,liquid_frac,ux,uy,uz,vmag" << std::endl;
            for (int k = 74; k <= 77 && k < config.nz; ++k) {
                for (int j = 0; j < config.ny; ++j) {
                    for (int i = 0; i < config.nx; ++i) {
                        int idx = i + j * config.nx + k * config.nx * config.ny;
                        float vmag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx] +
                                               h_uz[idx]*h_uz[idx]);
                        csv << i << "," << j << "," << k
                            << std::scientific << std::setprecision(6)
                            << "," << h_fill[idx]
                            << "," << h_curv[idx]
                            << "," << h_T[idx]
                            << "," << h_fl[idx]
                            << "," << h_ux[idx]
                            << "," << h_uy[idx]
                            << "," << h_uz[idx]
                            << "," << vmag << std::endl;
                    }
                }
            }
            csv.close();
            std::cout << "  Saved: " << csv_path << std::endl;
            std::cout << "  --- End diagnostics ---\n" << std::endl;
        }

        if (r.has_nan) {
            std::cout << "  NaN at step " << step << " — stopping." << std::endl;
            break;
        }
    }

    return results;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "=== Surface Force Isolation Tests ===" << std::endl;
    std::cout << "Goal: identify which force causes exponential velocity growth at step 20-22\n"
              << std::endl;

    const int MAX_STEPS = 30;

    // ---- Config A: Full physics (baseline) ----
    {
        auto config = buildBaseConfig();
        auto results = runConfig("A: Full Physics (baseline)",
                                  config, MAX_STEPS, true);

        std::cout << "\n  Summary A:" << std::endl;
        for (auto& r : results) {
            if (r.step >= 18) {
                std::cout << "    Step " << r.step << ": |u|="
                          << std::scientific << std::setprecision(3) << r.max_v
                          << (r.has_nan ? " NaN" : "") << std::endl;
            }
        }
    }

    // ---- Config B: Marangoni OFF, surface tension ON ----
    {
        auto config = buildBaseConfig();
        config.enable_marangoni = false;
        config.dsigma_dT = 0.0f;
        auto results = runConfig("B: Marangoni OFF, Surface Tension ON",
                                  config, MAX_STEPS, false);

        std::cout << "\n  Summary B:" << std::endl;
        for (auto& r : results) {
            if (r.step >= 18) {
                std::cout << "    Step " << r.step << ": |u|="
                          << std::scientific << std::setprecision(3) << r.max_v
                          << (r.has_nan ? " NaN" : "") << std::endl;
            }
        }
    }

    // ---- Config C: Both Marangoni AND surface tension OFF ----
    {
        auto config = buildBaseConfig();
        config.enable_surface_tension = false;
        config.enable_marangoni = false;
        config.surface_tension_coeff = 0.0f;
        config.dsigma_dT = 0.0f;
        auto results = runConfig("C: Surface Tension OFF + Marangoni OFF",
                                  config, MAX_STEPS, false);

        std::cout << "\n  Summary C:" << std::endl;
        for (auto& r : results) {
            if (r.step >= 18) {
                std::cout << "    Step " << r.step << ": |u|="
                          << std::scientific << std::setprecision(3) << r.max_v
                          << (r.has_nan ? " NaN" : "") << std::endl;
            }
        }
    }

    std::cout << "\n=== Isolation tests complete ===" << std::endl;
    std::cout << "Compare: if B survives but A dies → Marangoni is the culprit" << std::endl;
    std::cout << "         if B dies but C survives → Surface tension curvature is the culprit" << std::endl;
    std::cout << "         if C also dies → problem is elsewhere (evaporation, CFL, VOF)" << std::endl;

    return 0;
}
