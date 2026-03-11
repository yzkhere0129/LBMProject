/**
 * @file viz_cavity.cu
 * @brief Lid-driven cavity Re=100 simulation for visualization
 *
 * Runs a 2D lid-driven cavity at Re=100 to steady state, then dumps
 * the velocity field (ux, uy) on the z-midplane to CSV for matplotlib plotting.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "physics/fluid_lbm.h"
#include "core/streaming.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;
using lbm::core::Streaming;

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    // --- Domain ---
    const int nx = 129;
    const int ny = 129;
    const int nz = 3;
    const int num_cells = nx * ny * nz;

    // --- Physics ---
    const float Re = 100.0f;
    const float U_lid = 0.1f;
    const float L = static_cast<float>(nx - 1);
    const float nu = U_lid * L / Re;
    const float dx = 1.0f;
    const float dt = 1.0f;

    std::cout << "=== Lid-Driven Cavity Re=100 Visualization ===" << std::endl;
    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Re = " << Re << ", U_lid = " << U_lid << ", nu = " << nu << std::endl;

    // --- Create solver ---
    FluidLBM fluid(nx, ny, nz, nu, 1.0f,
                   BoundaryType::WALL,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   dt, dx);

    std::cout << "tau = " << fluid.getTau() << ", omega = " << fluid.getOmega() << std::endl;

    // --- Initialize ---
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);
    fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, U_lid, 0.0f, 0.0f);

    // --- Run to steady state ---
    const int max_steps = 80000;
    const int check_interval = 5000;
    std::vector<float> h_ux_old(num_cells, 0.0f);
    std::vector<float> h_ux_new(num_cells, 0.0f);

    std::cout << "Running..." << std::endl;

    for (int step = 0; step < max_steps; ++step) {
        fluid.collisionBGK(0.0f, 0.0f, 0.0f);
        fluid.streaming();
        fluid.applyBoundaryConditions(1);
        fluid.computeMacroscopic();

        if ((step + 1) % check_interval == 0) {
            fluid.copyVelocityToHost(h_ux_new.data(), nullptr, nullptr);

            float max_change = 0.0f;
            float max_u = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                max_change = std::max(max_change, std::abs(h_ux_new[i] - h_ux_old[i]));
                max_u = std::max(max_u, std::abs(h_ux_new[i]));
            }
            float rel_change = max_change / (max_u + 1e-10f);

            std::cout << "  Step " << (step + 1) << ": rel_change = " << rel_change << std::endl;

            if (rel_change < 1e-5f && step > 10000) {
                std::cout << "  Converged at step " << (step + 1) << std::endl;
                break;
            }
            h_ux_old = h_ux_new;
        }
    }

    // --- Extract velocity field ---
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // --- Dump z-midplane to CSV ---
    const int k_mid = nz / 2;
    const std::string csv_path = "/home/yzk/LBMProject/scripts/viz/cavity_velocity.csv";
    std::ofstream ofs(csv_path);
    ofs << "i,j,x,y,ux,uy" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + j * nx + k_mid * nx * ny;
            float x = static_cast<float>(i) / static_cast<float>(nx - 1);
            float y = static_cast<float>(j) / static_cast<float>(ny - 1);
            ofs << i << "," << j << ","
                << std::fixed << std::setprecision(6)
                << x << "," << y << ","
                << std::scientific << std::setprecision(8)
                << h_ux[idx] << "," << h_uy[idx] << std::endl;
        }
    }
    ofs.close();
    std::cout << "Velocity field written to: " << csv_path << std::endl;

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
