/**
 * @file test_rayleigh_taylor_gerris_exact.cu
 * @brief EXACT Gerris/Thibault RT benchmark with EXACT parameters (μ=3.13e-3, σ=0.01)
 *
 * This test uses the EXACT physical parameters from Gerris/Thibault Table 3.3:
 * - Viscosity: μ = 3.13×10⁻³ Pa·s (Re ≈ 622)
 * - Surface tension: σ = 0.01 N/m (provides stabilization, NOT zero)
 * - Resolution: 512×2048×4 (4× Gerris for K-H spiral resolution)
 *
 * CRITICAL PARAMETER COMPARISON:
 * ================================
 *
 * CURRENT CODE (test_rayleigh_taylor_mushroom.cu):
 * - μ = 7.5×10⁻⁴ Pa·s (REDUCED by 4.2×, gives Re≈2600)
 * - σ = 0.0 N/m (ZERO surface tension)
 * - Purpose: Attempt to increase Reynolds number for faster bubble rise
 * - Result: h₁=1.28m @ t=1.0s (still 36% below Gerris 2.0m target)
 *
 * GERRIS EXACT (this file):
 * - μ = 3.13×10⁻³ Pa·s (EXACT from Thibault Table 3.3)
 * - σ = 0.01 N/m (EXACT from Thibault Table 3.3, NOT zero!)
 * - Purpose: Direct 1:1 comparison with Gerris reference
 * - Expected: h₁≈2.0m @ t=1.0s (if our physics is correct)
 *
 * WHY THIS TEST MATTERS:
 * ======================
 * 1. Isolates numerical method differences from parameter differences
 * 2. Tests if higher viscosity (μ=3.13e-3) + surface tension (σ=0.01) can still achieve h₁≈2.0m
 * 3. Validates that Gerris uses σ=0.01 N/m (NOT zero) for stabilization
 * 4. If this fails to match Gerris, confirms the discrepancy is algorithmic, not parametric
 *
 * Reference:
 * - Thibault, J. P., & Senocak, I. (2009). CUDA implementation of a Navier-Stokes solver
 *   on multi-GPU desktop platforms for incompressible flows.
 * - Gerris Flow Solver: http://gfs.sourceforge.net/
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/force_accumulator.h"
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>

using namespace lbm::physics;
using namespace lbm::core;

// ============================================================================
// VOF Boundary Enforcement Kernel
// ============================================================================
__global__ void enforceVOFBoundaryKernel(
    float* fill_level,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || k >= nz) return;

    // Bottom boundary (y=0): Light fluid reservoir (f=0)
    int idx_bottom = i + nx * (0 + ny * k);
    fill_level[idx_bottom] = 0.0f;

    // Top boundary (y=ny-1): Heavy fluid reservoir (f=1)
    int idx_top = i + nx * ((ny-1) + ny * k);
    fill_level[idx_top] = 1.0f;
}

// ============================================================================
// VTK Output Helper
// ============================================================================
void writeVTK(const std::string& filename, const std::vector<float>& fill,
              const std::vector<float>& ux, const std::vector<float>& uy,
              int nx, int ny, int nz, float dx) {
    std::ofstream file(filename);
    file << "# vtk DataFile Version 3.0\n";
    file << "Rayleigh-Taylor Gerris EXACT\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << dx << " " << dx << " " << dx << "\n";
    file << "POINT_DATA " << (nx * ny * nz) << "\n";

    // Fill level field
    file << "SCALARS fill_level float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                file << fill[idx] << "\n";
            }
        }
    }

    // Velocity field
    file << "VECTORS velocity float\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                file << ux[idx] << " " << uy[idx] << " 0.0\n";
            }
        }
    }

    file.close();
}

// ============================================================================
// Interface Tracking Functions
// ============================================================================
void measureBubbleAndSpike(const std::vector<float>& fill, int nx, int ny, int nz,
                          float interface_y0, float& h1, float& h2) {
    float y_max = interface_y0;
    float y_min = interface_y0;

    int k_center = nz / 2;

    for (int i = 0; i < nx; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            int idx0 = i + nx * ((j - 1) + ny * k_center);
            int idx1 = i + nx * (j + ny * k_center);
            float f0 = fill[idx0];
            float f1 = fill[idx1];

            if ((f0 - 0.5f) * (f1 - 0.5f) < 0) {
                float y_interface = (j - 1) + (0.5f - f0) / (f1 - f0);
                y_max = std::max(y_max, y_interface);
                y_min = std::min(y_min, y_interface);
            }
        }
    }

    h1 = y_max - interface_y0;
    h2 = interface_y0 - y_min;
}

// ============================================================================
// Test Class
// ============================================================================
class RayleighTaylorGerrisExactTest : public ::testing::Test {
protected:
    void createOutputDirectory(const std::string& path) {
        mkdir(path.c_str(), 0755);
    }

    void initializePerturbedInterface(VOFSolver& vof, int nx, int ny, int nz,
                                     float amplitude, float wavelength, float interface_y) {
        std::vector<float> h_fill(nx * ny * nz);
        const float interface_width = 8.0f;
        const float k = 2.0f * M_PI / wavelength;

        for (int kz = 0; kz < nz; ++kz) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * kz);
                    float x = i;
                    float y = j;
                    float y_interface = interface_y + amplitude * std::cos(k * x);
                    float dist = y - y_interface;
                    h_fill[idx] = 0.5f * (1.0f + std::tanh(dist / interface_width));
                }
            }
        }
        vof.initialize(h_fill.data());
    }
};

// ============================================================================
// Main Test: EXACT Gerris Parameters
// ============================================================================
TEST_F(RayleighTaylorGerrisExactTest, ExactGerrisParameters) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  RAYLEIGH-TAYLOR: EXACT GERRIS PARAMETERS" << std::endl;
    std::cout << "  μ = 3.13e-3 Pa·s (Re≈622), σ = 0.01 N/m" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ========================================================================
    // Domain Configuration
    // ========================================================================
    const float Lx = 1.0f;
    const float Ly = 4.0f;
    const float Lz = 0.1f;
    const int nx = 512;
    const int ny = 2048;
    const int nz = 4;
    const float dx = Lx / nx;
    const int num_cells = nx * ny * nz;

    std::cout << "\n=== Domain Configuration ===" << std::endl;
    std::cout << "  Physical size: " << Lx << "m × " << Ly << "m × " << Lz << "m" << std::endl;
    std::cout << "  Grid: " << nx << " × " << ny << " × " << nz << std::endl;
    std::cout << "  dx = " << dx * 1e3f << " mm (4× finer than Gerris 7.8mm)" << std::endl;

    // ========================================================================
    // Physical Parameters - EXACT Gerris/Thibault Table 3.3
    // ========================================================================
    const float rho_heavy = 1.255f;
    const float rho_light = 0.1694f;
    const float mu = 3.13e-3f;         // EXACT Gerris value
    const float nu = mu / rho_heavy;
    const float g = 9.81f;
    const float sigma = 0.01f;         // EXACT Gerris value (NOT zero!)

    const float interface_y0 = Ly / 2.0f;
    const float eta0 = -0.05f;
    const float lambda = Lx;

    const float interface_y0_cells = interface_y0 / dx;
    const float eta0_cells = eta0 / dx;
    const float lambda_cells = lambda / dx;

    const float At = (rho_heavy - rho_light) / (rho_heavy + rho_light);
    const float rho_avg = (rho_heavy + rho_light) / 2.0f;
    const float U_characteristic = std::sqrt(At * g * lambda);
    const float Re = (rho_avg * U_characteristic * lambda) / mu;
    const float We = (rho_avg * U_characteristic * U_characteristic * lambda) / sigma;
    const float Bo = (rho_heavy - rho_light) * g * lambda * lambda / sigma;

    std::cout << "\n=== Physical Parameters (EXACT Gerris) ===" << std::endl;
    std::cout << "  Density heavy: " << rho_heavy << " kg/m³" << std::endl;
    std::cout << "  Density light: " << rho_light << " kg/m³" << std::endl;
    std::cout << "  Atwood number: At = " << At << std::endl;
    std::cout << "  Reynolds number: Re = " << Re << " (EXACT: ≈622)" << std::endl;
    std::cout << "  Weber number: We = " << We << std::endl;
    std::cout << "  Bond number: Bo = " << Bo << std::endl;
    std::cout << "  Dynamic viscosity: μ = " << mu << " Pa·s (EXACT Gerris)" << std::endl;
    std::cout << "  Kinematic viscosity: ν = " << nu << " m²/s" << std::endl;
    std::cout << "  Surface tension: σ = " << sigma << " N/m (EXACT Gerris, NOT zero!)" << std::endl;

    std::cout << "\n=== Parameter Comparison ===" << std::endl;
    std::cout << "  CURRENT CODE (mushroom test): μ=7.5e-4, σ=0.0 → Re≈2600" << std::endl;
    std::cout << "  GERRIS EXACT (this test):    μ=3.13e-3, σ=0.01 → Re≈622" << std::endl;
    std::cout << "  Viscosity ratio: μ_gerris/μ_current = " << (3.13e-3f / 7.5e-4f) << "×" << std::endl;

    // ========================================================================
    // Time Integration
    // ========================================================================
    const float U_max_estimate = U_characteristic * 1.5f;
    const float dt_CFL = 0.3f * dx / U_max_estimate;
    const float tau_target = 0.7f;
    const float nu_lattice_target = (tau_target - 0.5f) * D3Q19::CS2;
    const float dt_LBM = nu_lattice_target * dx * dx / nu;
    const float dt = std::min(dt_CFL, dt_LBM);
    const float t_final = 1.0f;
    const int num_steps = static_cast<int>(t_final / dt);

    std::vector<float> keyframe_times = {0.0f, 0.2f, 0.7f, 0.8f, 0.9f, 1.0f};
    std::vector<int> keyframe_steps;
    for (float t_key : keyframe_times) {
        keyframe_steps.push_back(static_cast<int>(t_key / dt));
    }

    int output_every = static_cast<int>(0.01f / dt);
    output_every = std::max(1, output_every);

    std::cout << "\n=== Time Integration ===" << std::endl;
    std::cout << "  dt = " << dt * 1e6f << " μs" << std::endl;
    std::cout << "  Total steps: " << num_steps << std::endl;

    float CFL = U_max_estimate * dt / dx;
    float nu_lattice = nu * dt / (dx * dx);
    float tau = nu_lattice / D3Q19::CS2 + 0.5f;

    std::cout << "\n=== Stability Analysis ===" << std::endl;
    std::cout << "  CFL = " << CFL << std::endl;
    std::cout << "  tau = " << tau << std::endl;

    // ========================================================================
    // Initialize Solvers
    // ========================================================================
    std::cout << "\n=== Initializing Solvers ===" << std::endl;

    FluidLBM fluid(nx, ny, nz, nu, rho_heavy,
                   lbm::physics::BoundaryType::PERIODIC,
                   lbm::physics::BoundaryType::WALL,
                   lbm::physics::BoundaryType::PERIODIC,
                   dt, dx);
    fluid.initialize(rho_heavy, 0.0f, 0.0f, 0.0f);

    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL,
                  VOFSolver::BoundaryType::PERIODIC);

    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::SUPERBEE);
    vof.setMassConservationCorrection(true, 0.7f);

    initializePerturbedInterface(vof, nx, ny, nz, eta0_cells, lambda_cells, interface_y0_cells);
    vof.reconstructInterface();
    vof.computeCurvature();

    ForceAccumulator forces(nx, ny, nz);

    // ========================================================================
    // Data Storage
    // ========================================================================
    std::vector<float> times, h1_values, h2_values, masses;
    std::vector<float> u_max_values;
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);

    float mass_initial = vof.computeTotalMass();

    const std::string output_dir = "/home/yzk/LBMProject/build/output_rt_gerris_exact";
    createOutputDirectory(output_dir);

    float* d_ux_phys;
    float* d_uy_phys;
    float* d_uz_phys;
    cudaMalloc(&d_ux_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uy_phys, num_cells * sizeof(float));
    cudaMalloc(&d_uz_phys, num_cells * sizeof(float));

    std::cout << "\n=== Running Simulation ===" << std::endl;
    std::cout << std::setw(8) << "Step"
              << std::setw(12) << "Time(s)"
              << std::setw(12) << "h₁(m)"
              << std::setw(12) << "h₂(m)"
              << std::setw(12) << "Mass(%)"
              << std::setw(12) << "u_max(m/s)" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    // ========================================================================
    // Main Simulation Loop
    // ========================================================================
    for (int step = 0; step <= num_steps; ++step) {
        float t = step * dt;

        bool is_keyframe = false;
        for (size_t k = 0; k < keyframe_steps.size(); ++k) {
            if (step == keyframe_steps[k]) {
                is_keyframe = true;
                break;
            }
        }

        bool is_output = (step % output_every == 0) || is_keyframe || (step == num_steps);

        if (is_output) {
            vof.copyFillLevelToHost(h_fill.data());

            const float* d_ux = fluid.getVelocityX();
            const float* d_uy = fluid.getVelocityY();
            const float* d_uz = fluid.getVelocityZ();

            cudaMemcpy(h_ux.data(), d_ux, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_uy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_uz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float v_conv = dx / dt;
            for (int i = 0; i < num_cells; ++i) {
                h_ux[i] *= v_conv;
                h_uy[i] *= v_conv;
                h_uz[i] *= v_conv;
            }

            float h1, h2;
            measureBubbleAndSpike(h_fill, nx, ny, nz, interface_y0_cells, h1, h2);

            float h1_phys = h1 * dx;
            float h2_phys = h2 * dx;

            float mass = vof.computeTotalMass();
            float mass_error = std::abs(mass - mass_initial) / mass_initial * 100.0f;

            float u_max = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float u_mag = std::sqrt(h_ux[i] * h_ux[i] + h_uy[i] * h_uy[i]);
                u_max = std::max(u_max, u_mag);
            }

            times.push_back(t);
            h1_values.push_back(h1_phys);
            h2_values.push_back(h2_phys);
            masses.push_back(mass);
            u_max_values.push_back(u_max);

            std::cout << std::setw(8) << step
                      << std::setw(12) << std::fixed << std::setprecision(3) << t
                      << std::setw(12) << std::setprecision(4) << h1_phys
                      << std::setw(12) << std::setprecision(4) << h2_phys
                      << std::setw(12) << std::setprecision(4) << mass_error
                      << std::setw(12) << std::setprecision(4) << u_max << std::endl;

            if (is_keyframe) {
                char vtk_filename[512];
                snprintf(vtk_filename, sizeof(vtk_filename),
                        "%s/rt_gerris_exact_t%.2f.vtk", output_dir.c_str(), t);
                writeVTK(vtk_filename, h_fill, h_ux, h_uy, nx, ny, nz, dx);
            }
        }

        if (step < num_steps) {
            forces.reset();
            forces.addVOFBuoyancyForce(vof.getFillLevel(), rho_heavy, rho_light, 0, -g, 0);

            if (sigma > 0) {
                forces.addSurfaceTensionForce(vof.getCurvature(), vof.getFillLevel(),
                                             sigma, nx, ny, nz, dx);
            }

            forces.convertToLatticeUnits(dx, dt, rho_heavy);

            fluid.computeVariableViscosity(vof.getFillLevel(), rho_heavy, rho_light, mu);

            fluid.collisionTRTVariable(forces.getFx(), forces.getFy(), forces.getFz(),
                                      vof.getFillLevel(), rho_heavy, rho_light);

            fluid.applyBoundaryConditions(1);
            fluid.streaming();
            fluid.computeMacroscopic();

            const float* d_ux = fluid.getVelocityX();
            const float* d_uy = fluid.getVelocityY();
            const float* d_uz = fluid.getVelocityZ();

            cudaMemcpy(h_ux.data(), d_ux, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_uy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_uz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float v_conv = dx / dt;
            for (int i = 0; i < num_cells; ++i) {
                h_ux[i] *= v_conv;
                h_uy[i] *= v_conv;
                h_uz[i] *= v_conv;
            }

            cudaMemcpy(d_ux_phys, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_uy_phys, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_uz_phys, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

            vof.advectFillLevel(d_ux_phys, d_uy_phys, d_uz_phys, dt);

            {
                dim3 blockSize(16, 16);
                dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                              (nz + blockSize.y - 1) / blockSize.y);
                enforceVOFBoundaryKernel<<<gridSize, blockSize>>>(
                    const_cast<float*>(vof.getFillLevel()), nx, ny, nz);
                cudaDeviceSynchronize();
            }

            vof.reconstructInterface();
            vof.computeCurvature();
        }
    }

    cudaFree(d_ux_phys);
    cudaFree(d_uy_phys);
    cudaFree(d_uz_phys);

    // ========================================================================
    // Write Results
    // ========================================================================
    std::string csv_filename = output_dir + "/rt_gerris_exact_evolution.csv";
    std::ofstream csv(csv_filename);
    csv << "time_s,h1_m,h2_m,mass,mass_error_pct,u_max_m_per_s\n";
    for (size_t i = 0; i < times.size(); ++i) {
        float mass_err = std::abs(masses[i] - mass_initial) / mass_initial * 100.0f;
        csv << std::fixed << std::setprecision(6) << times[i] << ","
            << h1_values[i] << ","
            << h2_values[i] << ","
            << masses[i] << ","
            << mass_err << ","
            << u_max_values[i] << "\n";
    }
    csv.close();

    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cout << "\n=== Final Results ===" << std::endl;

    float mass_final = masses.back();
    float mass_error = std::abs(mass_final - mass_initial) / mass_initial * 100.0f;
    float h1_final = h1_values.back();
    float h2_final = h2_values.back();

    std::cout << "  Bubble height h₁: " << h1_final << " m (Gerris target: ≈2.0m)" << std::endl;
    std::cout << "  Spike depth h₂: " << h2_final << " m (Gerris target: ≈1.0m)" << std::endl;
    std::cout << "  Mass conservation: " << mass_error << "%" << std::endl;

    std::cout << "\n=== Comparison with Current Code ===" << std::endl;
    std::cout << "  Current code (μ=7.5e-4, σ=0.0): h₁≈1.28m, h₂≈1.07m" << std::endl;
    std::cout << "  Gerris exact (μ=3.13e-3, σ=0.01): h₁=" << h1_final << "m, h₂=" << h2_final << "m" << std::endl;

    std::cout << "\n=== Output Files ===" << std::endl;
    std::cout << "  VTK keyframes: " << output_dir << "/rt_gerris_exact_t*.vtk" << std::endl;
    std::cout << "  CSV data: " << csv_filename << std::endl;

    EXPECT_LT(mass_error, 7.0f) << "Mass error exceeds 7%";
    EXPECT_GT(h1_final, 0.1f) << "Bubble should rise significantly";
    EXPECT_GT(h2_final, 0.1f) << "Spike should fall significantly";

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Test completed successfully!" << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
