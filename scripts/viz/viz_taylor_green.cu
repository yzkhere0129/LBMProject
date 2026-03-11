/**
 * @file viz_taylor_green.cu
 * @brief Taylor-Green 2D vortex simulation for visualization
 *
 * Runs 2D Taylor-Green vortex, dumps velocity field at t=0 and t=T_decay
 * to CSV for matplotlib vorticity contour plots.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "physics/fluid_lbm.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

/**
 * Dump the z-midplane velocity field to CSV, computing vorticity numerically
 * via central differences: omega_z = dv/dx - du/dy
 */
static void dumpField(const std::vector<float>& ux,
                      const std::vector<float>& uy,
                      int nx, int ny, int nz,
                      float dx_phys, float U_lid,
                      const std::string& path,
                      const std::string& label) {
    const int k_mid = nz / 2;
    std::ofstream ofs(path);
    ofs << "i,j,x,y,ux,uy,vorticity" << std::endl;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + j * nx + k_mid * nx * ny;
            float x = (i + 0.5f) * dx_phys;
            float y = (j + 0.5f) * dx_phys;

            // Central-difference vorticity (periodic wrapping)
            int ip = (i + 1) % nx;
            int im = (i - 1 + nx) % nx;
            int jp = (j + 1) % ny;
            int jm = (j - 1 + ny) % ny;

            float dv_dx = (uy[ip + j * nx + k_mid * nx * ny]
                         - uy[im + j * nx + k_mid * nx * ny]) / (2.0f * dx_phys);
            float du_dy = (ux[i + jp * nx + k_mid * nx * ny]
                         - ux[i + jm * nx + k_mid * nx * ny]) / (2.0f * dx_phys);

            float vorticity = dv_dx - du_dy;

            ofs << i << "," << j << ","
                << std::scientific << std::setprecision(8)
                << x << "," << y << ","
                << ux[idx] << "," << uy[idx] << ","
                << vorticity << std::endl;
        }
    }
    ofs.close();
    std::cout << label << " written to: " << path << std::endl;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    // --- Domain ---
    const int nx = 128;
    const int ny = 128;
    const int nz = 3;
    const int num_cells = nx * ny * nz;

    // --- Physics ---
    const float Lx = 1.0e-3f;  // 1 mm
    const float dx_phys = Lx / nx;
    const float U0 = 0.1f;       // m/s
    const float Re = 100.0f;
    const float nu = U0 * Lx / Re;
    const float rho0 = 1.0f;
    const float k = 2.0f * M_PI / Lx;

    // Time parameters
    const float tau_visc = Lx * Lx / (2.0f * M_PI * M_PI * nu);
    const float nu_lattice_target = 0.0667f;
    const float dt = nu_lattice_target * dx_phys * dx_phys / nu;
    const float final_time = 1.0f * tau_visc;
    const int num_steps = static_cast<int>(final_time / dt);

    std::cout << "=== Taylor-Green 2D Vortex Visualization ===" << std::endl;
    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Re = " << Re << ", U0 = " << U0 << ", nu = " << nu << std::endl;
    std::cout << "tau_visc = " << tau_visc * 1e6 << " us" << std::endl;
    std::cout << "dt = " << dt * 1e9 << " ns, num_steps = " << num_steps << std::endl;

    // --- Create solver ---
    FluidLBM fluid(nx, ny, nz, nu, rho0,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   dt, dx_phys);

    std::cout << "tau = " << fluid.getTau() << ", omega = " << fluid.getOmega() << std::endl;

    // --- Set initial conditions: Taylor-Green vortex ---
    std::vector<float> h_rho(num_cells, rho0);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells, 0.0f);

    for (int kk = 0; kk < nz; ++kk) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * kk);
                float x = (i + 0.5f) * dx_phys;
                float y = (j + 0.5f) * dx_phys;
                h_ux[idx] =  U0 * sinf(k * x) * cosf(k * y);
                h_uy[idx] = -U0 * cosf(k * x) * sinf(k * y);
            }
        }
    }

    // Copy to device
    float *d_rho, *d_ux, *d_uy, *d_uz;
    CUDA_CHECK(cudaMalloc(&d_rho, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ux,  num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uy,  num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uz,  num_cells * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux,  h_ux.data(),  num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy,  h_uy.data(),  num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uz,  h_uz.data(),  num_cells * sizeof(float), cudaMemcpyHostToDevice));

    fluid.initialize(d_rho, d_ux, d_uy, d_uz);

    // --- Dump t=0 ---
    dumpField(h_ux, h_uy, nx, ny, nz, dx_phys, U0,
              "/home/yzk/LBMProject/scripts/viz/tg_t0.csv", "t=0 field");

    // --- Run to t = tau_visc ---
    std::cout << "Running " << num_steps << " steps..." << std::endl;
    const int print_interval = num_steps / 10;
    for (int step = 1; step <= num_steps; ++step) {
        fluid.collisionBGK(0.0f, 0.0f, 0.0f);
        fluid.streaming();
        fluid.computeMacroscopic();

        if (step % print_interval == 0) {
            std::cout << "  Step " << step << " / " << num_steps << std::endl;
        }
    }

    // --- Dump t=tau_visc ---
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
    dumpField(h_ux, h_uy, nx, ny, nz, dx_phys, U0,
              "/home/yzk/LBMProject/scripts/viz/tg_tfinal.csv", "t=tau_visc field");

    // Compute energy ratio
    float E0 = 0.0f, Ef = 0.0f;
    for (int kk = 0; kk < nz; ++kk) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * kk);
                float x = (i + 0.5f) * dx_phys;
                float y = (j + 0.5f) * dx_phys;
                float u0 = U0 * sinf(k * x) * cosf(k * y);
                float v0 = -U0 * cosf(k * x) * sinf(k * y);
                E0 += 0.5f * rho0 * (u0 * u0 + v0 * v0);
                Ef += 0.5f * rho0 * (h_ux[idx] * h_ux[idx] + h_uy[idx] * h_uy[idx]);
            }
        }
    }
    float decay_sim = Ef / E0;
    float decay_theory = expf(-4.0f * nu * k * k * final_time);
    std::cout << "Energy decay: simulated = " << decay_sim
              << ", analytical = " << decay_theory
              << ", error = " << fabsf(decay_sim - decay_theory) / decay_theory * 100.0f << "%" << std::endl;

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
