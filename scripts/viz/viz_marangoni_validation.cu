/**
 * @file viz_marangoni_validation.cu
 * @brief Marangoni fix validation: 1D interface profile + 2D vortex structure
 *
 * Runs LaserMeltingIron with fixed Marangoni (CSF smearing + Δu cap) and dumps:
 *   Plot A data: 1D z-line at laser spot edge showing fill, T, |F_M|
 *   Plot B data: 2D x-z midplane velocity field for vortex visualization
 *
 * Also dumps VTK at step 40 for ParaView inspection.
 *
 * Usage: ./viz_marangoni_validation
 */

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

static const std::string OUT_DIR = "/home/yzk/LBMProject/scripts/viz/";

MultiphysicsConfig buildConfig() {
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

    float nu_physical = config.material.mu_liquid / config.material.rho_liquid;
    float nu_lattice_phys = nu_physical * config.dt / (config.dx * config.dx);
    float nu_min = 0.01f;  // tau ≈ 0.53, stable with TRT
    config.kinematic_viscosity = std::max(nu_lattice_phys, nu_min);
    config.density = config.material.rho_liquid;
    config.darcy_coefficient = 1e7f;

    float alpha_phys = config.material.k_liquid /
                       (config.material.rho_liquid * config.material.cp_liquid);
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
    config.boundaries.z_max = BoundaryType::WALL;
    config.boundaries.thermal_z_min = ThermalBCType::CONVECTIVE;
    config.boundaries.thermal_z_max = ThermalBCType::ADIABATIC;
    config.boundaries.convective_h = 5.0f;
    config.boundaries.convective_T_inf = 300.0f;

    return config;
}

void writeVTK3D(const std::string& path, const std::string& title,
                int nx, int ny, int nz, float dx,
                const std::vector<float>& pressure,
                const std::vector<float>& temperature,
                const std::vector<float>& liquid_fraction,
                const std::vector<float>& fill_level,
                const std::vector<float>& ux,
                const std::vector<float>& uy,
                const std::vector<float>& uz) {
    std::ofstream vtk(path);
    vtk << "# vtk DataFile Version 3.0\n" << title << "\nASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    vtk << "ORIGIN 0 0 0\n";
    vtk << "SPACING " << dx*1e6 << " " << dx*1e6 << " " << dx*1e6 << "\n";
    int nc = nx * ny * nz;
    vtk << "POINT_DATA " << nc << "\n";

    auto writeScalar = [&](const std::string& name, const std::vector<float>& data) {
        vtk << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    vtk << std::scientific << std::setprecision(6)
                        << data[i + j * nx + k * nx * ny] << "\n";
    };

    writeScalar("pressure", pressure);
    writeScalar("temperature", temperature);
    writeScalar("liquid_fraction", liquid_fraction);
    writeScalar("fill_level", fill_level);

    vtk << "VECTORS velocity float\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                vtk << ux[idx] << " " << uy[idx] << " " << uz[idx] << "\n";
            }
    vtk.close();
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "=== Marangoni Fix Validation ===" << std::endl;

    auto config = buildConfig();
    MultiphysicsSolver solver(config);
    // Interface at 85% height: substrate (fill=1) below, gas (fill=0) above
    // This creates a clear liquid-gas interface for CSF Marangoni to act on
    solver.initialize(300.0f, 0.85f);

    const int NC = config.nx * config.ny * config.nz;
    const int MAX_STEPS = 1000;

    // Laser center: i=20, j=40 (cell indices)
    // Laser spot edge at r=50μm: ~13 cells from center at dx=3.75μm
    // Line profile at i=20+13=33 (just outside spot edge, max ∇T)
    const int PROFILE_I = 33;   // x-position for 1D z-profile
    const int PROFILE_J = 40;   // y-position (laser center y)

    for (int step = 1; step <= MAX_STEPS; ++step) {
        solver.step();

        bool has_nan = solver.checkNaN();
        float max_v = solver.getMaxVelocity();
        float max_T = solver.getMaxTemperature();

        if (step % 50 == 0 || has_nan) {
            std::cout << "  Step " << std::setw(4) << step
                      << ": max|u|=" << std::scientific << std::setprecision(3) << max_v
                      << " max_T=" << std::fixed << std::setprecision(1) << max_T << " K"
                      << " t=" << std::setprecision(2) << step * config.dt * 1e6 << " μs"
                      << (has_nan ? " *** NaN ***" : "") << std::endl;
        }

        // Dump data at 333 (25μs), 667 (50μs), 800 (60μs=shutoff), 1000 (75μs=mature)
        if (step == 333 || step == 667 || step == 800 || step == 1000 || has_nan) {
            std::vector<float> h_p(NC), h_T(NC), h_fl(NC), h_fill(NC), h_curv(NC);
            std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC);

            CUDA_CHECK(cudaMemcpy(h_p.data(), solver.getPressure(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_T.data(), solver.getTemperature(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_fl.data(), solver.getLiquidFraction(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_fill.data(), solver.getFillLevel(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_curv.data(), solver.getCurvature(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ux.data(), solver.getVelocityX(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy.data(), solver.getVelocityY(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uz.data(), solver.getVelocityZ(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));

            // ============================================================
            // Plot A: 1D z-profile at laser spot edge
            // ============================================================
            {
                std::ostringstream csv_name;
                csv_name << OUT_DIR << "marangoni_zprofile_step"
                         << std::setfill('0') << std::setw(4) << step << ".csv";
                std::ofstream csv(csv_name.str());
                csv << "k,z_um,fill,temperature,liquid_frac,curvature,ux,uy,uz,vmag" << std::endl;

                int i = PROFILE_I;
                int j = PROFILE_J;
                for (int k = 0; k < config.nz; ++k) {
                    int idx = i + j * config.nx + k * config.nx * config.ny;
                    float vmag = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx] +
                                           h_uz[idx]*h_uz[idx]);
                    csv << k << ","
                        << std::fixed << std::setprecision(2) << k * config.dx * 1e6 << ","
                        << std::scientific << std::setprecision(8)
                        << h_fill[idx] << ","
                        << h_T[idx] << ","
                        << h_fl[idx] << ","
                        << h_curv[idx] << ","
                        << h_ux[idx] << ","
                        << h_uy[idx] << ","
                        << h_uz[idx] << ","
                        << vmag << std::endl;
                }
                csv.close();
                std::cout << "  Plot A data: " << csv_name.str() << std::endl;
            }

            // ============================================================
            // Plot B: 2D x-z midplane at j=40 (laser center y)
            // ============================================================
            {
                std::ostringstream csv_name;
                csv_name << OUT_DIR << "marangoni_xz_midplane_step"
                         << std::setfill('0') << std::setw(4) << step << ".csv";
                std::ofstream csv(csv_name.str());
                csv << "i,k,x_um,z_um,fill,temperature,liquid_frac,ux,uz,vmag" << std::endl;

                int j = PROFILE_J;
                for (int k = 0; k < config.nz; ++k) {
                    for (int i = 0; i < config.nx; ++i) {
                        int idx = i + j * config.nx + k * config.nx * config.ny;
                        float vmag = std::sqrt(h_ux[idx]*h_ux[idx] +
                                               h_uy[idx]*h_uy[idx] +
                                               h_uz[idx]*h_uz[idx]);
                        csv << i << "," << k << ","
                            << std::fixed << std::setprecision(2)
                            << i * config.dx * 1e6 << ","
                            << k * config.dx * 1e6 << ","
                            << std::scientific << std::setprecision(8)
                            << h_fill[idx] << ","
                            << h_T[idx] << ","
                            << h_fl[idx] << ","
                            << h_ux[idx] << ","
                            << h_uz[idx] << ","
                            << vmag << std::endl;
                    }
                }
                csv.close();
                std::cout << "  Plot B data: " << csv_name.str() << std::endl;
            }

            // VTK for ParaView
            {
                std::ostringstream vtk_name;
                vtk_name << OUT_DIR << "marangoni_validation_step"
                         << std::setfill('0') << std::setw(4) << step << ".vtk";
                writeVTK3D(vtk_name.str(),
                           "Marangoni validation step " + std::to_string(step),
                           config.nx, config.ny, config.nz, config.dx,
                           h_p, h_T, h_fl, h_fill, h_ux, h_uy, h_uz);
                std::cout << "  VTK: " << vtk_name.str() << std::endl;
            }
        }

        if (has_nan) {
            std::cout << "  NaN at step " << step << " — stopping." << std::endl;
            break;
        }
    }

    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  Simulation reached step " << MAX_STEPS << std::endl;
    std::cout << "  Profile line: i=" << PROFILE_I << ", j=" << PROFILE_J
              << " (laser spot edge, max grad_T)" << std::endl;
    std::cout << "  If no NaN: Marangoni fix is working!" << std::endl;
    std::cout << "  Plot A: marangoni_zprofile_stepXX.csv (1D z-line)" << std::endl;
    std::cout << "  Plot B: marangoni_xz_midplane_stepXX.csv (2D velocity)" << std::endl;

    return 0;
}
