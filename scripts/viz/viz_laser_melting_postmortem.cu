/**
 * @file viz_laser_melting_postmortem.cu
 * @brief LaserMeltingIron post-mortem: dump VTK at step 21 (right before NaN at step 22)
 *
 * Runs the exact LaserMeltingIron configuration (nx=40, ny=80, nz=80, Steel)
 * and dumps full 3D VTK fields every step from step 15 to step 22:
 *   - Velocity (ux, uy, uz), Pressure, Temperature, Liquid Fraction, Fill Level
 *
 * This allows ParaView inspection of what is blowing up.
 *
 * Usage:  ./viz_laser_melting_postmortem
 * Output: scripts/viz/laser_postmortem_stepXX.vtk
 */

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "physics/multiphysics_solver.h"
#include "physics/material_properties.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

static const std::string OUT_DIR = "/home/yzk/LBMProject/scripts/viz/";

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
    vtk << "SPACING " << dx*1e6 << " " << dx*1e6 << " " << dx*1e6 << "\n";  // μm
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
    std::cout << "=== LaserMeltingIron Post-Mortem ===" << std::endl;

    // Exact config from test_laser_melting_iron.cu FullPhysics test
    MultiphysicsConfig config;

    // Domain
    config.nx = 40;
    config.ny = 80;
    config.nz = 80;
    config.dx = 3.75e-6f;
    const int NC = config.nx * config.ny * config.nz;

    // Timestep
    config.dt = 75e-9f;

    // Material: Steel
    config.material = MaterialDatabase::getSteel();

    // Physics flags
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

    // Fluid
    float alpha_phys = config.material.k_liquid /
                       (config.material.rho_liquid * config.material.cp_liquid);
    float alpha_lattice = alpha_phys * config.dt / (config.dx * config.dx);
    float nu_physical = config.material.mu_liquid / config.material.rho_liquid;
    float nu_lattice_phys = nu_physical * config.dt / (config.dx * config.dx);
    float nu_min = 0.0333f * 1.5f;
    config.kinematic_viscosity = std::max(nu_lattice_phys, nu_min);
    config.density = config.material.rho_liquid;
    config.darcy_coefficient = 1e7f;

    // Thermal
    config.thermal_diffusivity = alpha_phys;
    config.enable_radiation_bc = true;
    config.enable_substrate_cooling = true;
    config.substrate_h_conv = 5.0f;
    config.substrate_temperature = 300.0f;

    // Surface
    config.surface_tension_coeff = config.material.surface_tension;
    config.dsigma_dT = config.material.dsigma_dT;

    // Laser (stationary, center)
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

    // CFL
    config.cfl_velocity_target = 0.1f;
    config.cfl_use_gradual_scaling = true;
    config.cfl_force_ramp_factor = 0.8f;

    // VOF
    config.vof_subcycles = 100;

    // Boundaries (LPBF: periodic x/y, wall z-min substrate, periodic z-max open)
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

    std::cout << "  Config: " << config.nx << "x" << config.ny << "x" << config.nz
              << "  dx=" << config.dx << "  dt=" << config.dt << std::endl;
    std::cout << "  nu_lattice=" << config.kinematic_viscosity
              << "  darcy_C=" << config.darcy_coefficient << std::endl;

    // Create and initialize solver
    MultiphysicsSolver solver(config);
    float T_initial = 300.0f;
    float interface_height = 1.0f;  // Top surface
    solver.initialize(T_initial, interface_height);

    std::cout << "  Solver initialized. Running steps..." << std::endl;

    // Run and dump VTK at each step from 15 to 25 (or until NaN)
    const int DUMP_START = 15;
    const int MAX_STEPS = 25;

    for (int step = 1; step <= MAX_STEPS; ++step) {
        solver.step();

        bool has_nan = solver.checkNaN();
        float max_v = solver.getMaxVelocity();
        float max_T = solver.getMaxTemperature();

        std::cout << "  Step " << std::setw(3) << step
                  << ": max|u|=" << std::scientific << std::setprecision(3) << max_v
                  << " max_T=" << std::fixed << std::setprecision(1) << max_T << " K"
                  << (has_nan ? " *** NaN ***" : "") << std::endl;

        // Dump VTK from step DUMP_START onward
        if (step >= DUMP_START) {
            std::vector<float> h_p(NC), h_T(NC), h_fl(NC), h_fill(NC);
            std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC);

            // Copy from device
            CUDA_CHECK(cudaMemcpy(h_p.data(), solver.getPressure(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_T.data(), solver.getTemperature(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_fl.data(), solver.getLiquidFraction(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_fill.data(), solver.getFillLevel(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ux.data(), solver.getVelocityX(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy.data(), solver.getVelocityY(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uz.data(), solver.getVelocityZ(),
                                   NC * sizeof(float), cudaMemcpyDeviceToHost));

            std::ostringstream vtk_name;
            vtk_name << OUT_DIR << "laser_postmortem_step"
                     << std::setfill('0') << std::setw(2) << step << ".vtk";

            writeVTK3D(vtk_name.str(),
                       "LaserMeltingIron step " + std::to_string(step),
                       config.nx, config.ny, config.nz, config.dx,
                       h_p, h_T, h_fl, h_fill, h_ux, h_uy, h_uz);

            std::cout << "    VTK: " << vtk_name.str() << std::endl;

            // Also dump mid-plane CSV for quick inspection
            if (step == 21 || has_nan) {
                std::ostringstream csv_name;
                csv_name << OUT_DIR << "laser_postmortem_midplane_step"
                         << std::setfill('0') << std::setw(2) << step << ".csv";
                std::ofstream csv(csv_name.str());
                csv << "i,j,k,x_um,y_um,z_um,pressure,temperature,liquid_fraction,"
                    << "fill_level,ux,uy,uz,vmag" << std::endl;

                int k_mid = config.nz / 2;
                for (int j = 0; j < config.ny; ++j) {
                    for (int i = 0; i < config.nx; ++i) {
                        int idx = i + j * config.nx + k_mid * config.nx * config.ny;
                        float vmag = std::sqrt(h_ux[idx]*h_ux[idx] +
                                               h_uy[idx]*h_uy[idx] +
                                               h_uz[idx]*h_uz[idx]);
                        csv << i << "," << j << "," << k_mid << ","
                            << std::fixed << std::setprecision(2)
                            << i * config.dx * 1e6 << ","
                            << j * config.dx * 1e6 << ","
                            << k_mid * config.dx * 1e6 << ","
                            << std::scientific << std::setprecision(8)
                            << h_p[idx] << "," << h_T[idx] << ","
                            << h_fl[idx] << "," << h_fill[idx] << ","
                            << h_ux[idx] << "," << h_uy[idx] << ","
                            << h_uz[idx] << "," << vmag << std::endl;
                    }
                }
                csv.close();
                std::cout << "    CSV midplane: " << csv_name.str() << std::endl;

                // Find and report NaN locations
                int nan_count = 0;
                for (int idx = 0; idx < NC && nan_count < 20; ++idx) {
                    if (std::isnan(h_ux[idx]) || std::isnan(h_uy[idx]) ||
                        std::isnan(h_uz[idx]) || std::isnan(h_p[idx]) ||
                        std::isnan(h_T[idx])) {
                        int ii = idx % config.nx;
                        int jj = (idx / config.nx) % config.ny;
                        int kk = idx / (config.nx * config.ny);
                        std::cout << "    NaN at (" << ii << "," << jj << "," << kk
                                  << "): T=" << h_T[idx] << " p=" << h_p[idx]
                                  << " ux=" << h_ux[idx] << " fl=" << h_fl[idx]
                                  << " fill=" << h_fill[idx] << std::endl;
                        nan_count++;
                    }
                }

                // Find max force locations
                float max_vmag = 0;
                int max_idx = 0;
                for (int idx = 0; idx < NC; ++idx) {
                    float v = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx] +
                                        h_uz[idx]*h_uz[idx]);
                    if (!std::isnan(v) && v > max_vmag) {
                        max_vmag = v;
                        max_idx = idx;
                    }
                }
                int mi = max_idx % config.nx;
                int mj = (max_idx / config.nx) % config.ny;
                int mk = max_idx / (config.nx * config.ny);
                std::cout << "    Max |u| = " << max_vmag << " at ("
                          << mi << "," << mj << "," << mk
                          << ") T=" << h_T[max_idx]
                          << " fl=" << h_fl[max_idx]
                          << " fill=" << h_fill[max_idx] << std::endl;
            }
        }

        if (has_nan) {
            std::cout << "  NaN detected at step " << step << " — stopping." << std::endl;
            break;
        }
    }

    std::cout << "\n=== Post-mortem complete ===" << std::endl;
    return 0;
}
