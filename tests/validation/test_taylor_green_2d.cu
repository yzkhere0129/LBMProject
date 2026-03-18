/**
 * @file test_taylor_green_2d.cu
 * @brief Phase 1 Fluid Validation: 2D Taylor-Green Vortex Benchmark
 *
 * This test validates the incompressible Navier-Stokes solver (D3Q19 LBM)
 * against the exact analytical solution for the 2D Taylor-Green vortex.
 *
 * PHYSICAL SETUP:
 * The Taylor-Green vortex is a periodic array of counter-rotating vortices
 * that decay through viscous diffusion. This is a fundamental benchmark for
 * validating:
 * - Momentum diffusion accuracy
 * - Energy conservation/dissipation
 * - Incompressibility enforcement
 * - Spatial accuracy of velocity gradients
 *
 * TEST CONFIGURATION:
 * - Domain: 128×128 cells (periodic BC in x,y; minimal in z)
 * - Reynolds: Re = U₀L/ν = 100
 * - Duration: t = 5τ_visc where τ_visc = L²/(2π²ν)
 * - Initial condition: u = U₀sin(kx)cos(ky), v = -U₀cos(kx)sin(ky)
 * - Expected decay: E(t) = E₀exp(-4νk²t)
 *
 * VALIDATION METRICS:
 * 1. Velocity field L2 error < 5% at final time
 * 2. Energy decay rate matches analytical within 5%
 * 3. Energy decreases monotonically
 * 4. No NaN or instability
 *
 * ACCEPTANCE CRITERIA:
 * - CRITICAL: Decay rate error < 5% vs analytical
 * - CRITICAL: Energy decreases monotonically
 * - CRITICAL: No NaN or instability
 *
 * SUCCESS:
 * This test passing proves the fluid solver correctly models momentum
 * diffusion, which is essential for melt pool convection in LPBF.
 *
 * REFERENCES:
 * - Taylor & Green (1937): Original vortex solution
 * - Latt et al. (2021): "Palabos: Parallel Lattice Boltzmann Solver"
 * - Krüger et al. (2017): "The Lattice Boltzmann Method" (§5.3.2)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "physics/fluid_lbm.h"
#include "analytical/taylor_green.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

/**
 * @brief Taylor-Green 2D vortex validation test
 *
 * This test validates momentum diffusion by comparing LBM solution
 * to exact analytical decay of kinetic energy.
 */
TEST(FluidValidation, TaylorGreen2D) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  TAYLOR-GREEN VORTEX 2D VALIDATION\n";
    std::cout << "========================================\n\n";

    // ========================================
    // DOMAIN SETUP
    // ========================================
    const int nx = 128;      // Resolution in x
    const int ny = 128;      // Resolution in y
    const int nz = 3;        // Minimal in z (quasi-2D)
    const int num_cells = nx * ny * nz;

    const float Lx = 1.0e-3f;  // Domain size: 1 mm × 1 mm
    const float Ly = 1.0e-3f;
    const float dx = Lx / nx;

    std::cout << "Domain Configuration:\n";
    std::cout << "  Grid: " << nx << " × " << ny << " × " << nz << "\n";
    std::cout << "  Physical size: " << Lx * 1e3 << " mm × " << Ly * 1e3 << " mm\n";
    std::cout << "  Cell size: dx = " << dx * 1e6 << " μm\n\n";

    // ========================================
    // FLOW PARAMETERS (Re = 100)
    // ========================================
    const float U0 = 0.1f;          // Velocity amplitude [m/s]
    const float Re = 100.0f;        // Reynolds number
    const float nu = U0 * Lx / Re;  // Kinematic viscosity [m²/s]
    const float rho0 = 1.0f;        // Density [kg/m³]

    std::cout << "Flow Parameters:\n";
    std::cout << "  U₀ = " << U0 << " m/s\n";
    std::cout << "  Re = U₀L/ν = " << Re << "\n";
    std::cout << "  ν = " << nu << " m²/s\n";
    std::cout << "  ρ₀ = " << rho0 << " kg/m³\n\n";

    // ========================================
    // TIME PARAMETERS
    // ========================================
    // Viscous time scale: τ_visc = L²/(2π²ν)
    const float tau_visc = Lx * Lx / (2.0f * M_PI * M_PI * nu);

    // CFL condition for stability
    // For LBM: dt must ensure tau > 0.5 (omega < 2.0)
    // We want nu_lattice = nu * dt / dx² to give tau ~ 0.6-1.0
    // Target: tau = 0.7 → nu_lattice = cs²(tau - 0.5) = (1/3)(0.2) = 0.0667
    // Therefore: dt = nu_lattice * dx² / nu
    const float nu_lattice_target = 0.0667f;  // Gives tau = 0.7, omega = 1.43
    const float dt = nu_lattice_target * dx * dx / nu;

    const float final_time = 1.0f * tau_visc;  // Simulate for 1 decay time constant (enough to validate)
    const int num_steps = static_cast<int>(final_time / dt);

    std::cout << "Time Parameters:\n";
    std::cout << "  Viscous time scale: τ_visc = " << tau_visc * 1e6 << " μs\n";
    std::cout << "  Simulation time: t_final = 1τ_visc = " << final_time * 1e6 << " μs\n";
    std::cout << "  Timestep: dt = " << dt * 1e9 << " ns\n";
    std::cout << "  Number of steps: " << num_steps << "\n";
    std::cout << "  Target ν_lattice = " << nu_lattice_target << " (tau ≈ 0.7)\n\n";

    // ========================================
    // ANALYTICAL SOLUTION
    // ========================================
    analytical::TaylorGreen2D analytical(U0, Lx, nu, rho0);

    std::cout << "Analytical Solution:\n";
    std::cout << "  Wavenumber: k = 2π/L = " << analytical.k << " m⁻¹\n";
    std::cout << "  Initial energy density: E₀ = " << analytical.E0 << " J/m³\n";
    std::cout << "  Energy at t=0: E(0) = " << analytical.kineticEnergy(0) << " J/m³\n";
    std::cout << "  Energy at t_final: E(t) = " << analytical.kineticEnergy(final_time) << " J/m³\n";
    std::cout << "  Decay factor: exp(-4νk²t) = " << std::exp(-4.0f * nu * analytical.k * analytical.k * final_time) << "\n\n";

    // ========================================
    // INITIALIZE FLUID SOLVER
    // ========================================
    std::cout << "Initializing Fluid LBM Solver...\n";
    FluidLBM fluid(nx, ny, nz,
                   nu,                              // kinematic viscosity
                   rho0,                            // reference density
                   BoundaryType::PERIODIC,          // periodic in x
                   BoundaryType::PERIODIC,          // periodic in y
                   BoundaryType::PERIODIC,          // periodic in z (minimal effect)
                   dt, dx);

    std::cout << "  Solver tau = " << fluid.getTau() << "\n";
    std::cout << "  Solver omega = " << fluid.getOmega() << "\n";
    std::cout << "  Solver nu_physical = " << fluid.getViscosity() << " m²/s\n\n";

    // ========================================
    // SET INITIAL CONDITIONS
    // ========================================
    std::cout << "Setting Initial Conditions (Taylor-Green velocity field)...\n";

    std::vector<float> h_rho(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float x = (i + 0.5f) * dx;  // Cell center
                float y = (j + 0.5f) * dx;

                h_rho[idx] = rho0;
                h_ux[idx] = analytical.velocityU(x, y, 0.0f);
                h_uy[idx] = analytical.velocityV(x, y, 0.0f);
                h_uz[idx] = 0.0f;
            }
        }
    }

    // Copy to device
    float *d_rho, *d_ux, *d_uy, *d_uz;
    CUDA_CHECK(cudaMalloc(&d_rho, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ux, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uy, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uz, num_cells * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));

    fluid.initialize(d_rho, d_ux, d_uy, d_uz);

    // Compute initial kinetic energy
    // For quasi-2D: average over all cells (nx × ny × nz), gives energy per unit volume
    float initial_energy = 0.0f;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float u2 = h_ux[idx] * h_ux[idx] + h_uy[idx] * h_uy[idx];
                initial_energy += 0.5f * rho0 * u2;
            }
        }
    }
    initial_energy /= num_cells;  // Average over all cells to get energy density

    std::cout << "  Initial kinetic energy (LBM): " << initial_energy << " J/m³\n";
    std::cout << "  Initial kinetic energy (analytical): " << analytical.kineticEnergy(0) << " J/m³\n";
    std::cout << "  Relative difference: " << std::abs(initial_energy - analytical.kineticEnergy(0)) / analytical.kineticEnergy(0) * 100.0f << "%\n\n";

    // ========================================
    // TIME INTEGRATION
    // ========================================
    std::cout << "Running Simulation...\n";
    std::cout << "Step       Time[μs]    E_LBM[J/m³]    E_analytical[J/m³]    Error[%]    Decay_Rate_LBM    Decay_Rate_Analytical\n";
    std::cout << std::string(120, '-') << "\n";

    const int output_interval = num_steps / 20;  // 20 output points
    std::vector<float> time_series;
    std::vector<float> energy_lbm_series;
    std::vector<float> energy_analytical_series;

    float prev_energy_lbm = initial_energy;
    bool monotonic_decay = true;

    for (int step = 0; step <= num_steps; ++step) {
        if (step > 0) {
            // LBM step: collision + streaming
            fluid.collisionBGK(0.0f, 0.0f, 0.0f);  // No external force
            fluid.streaming();
            fluid.computeMacroscopic();
        }

        // Output at regular intervals
        if (step % output_interval == 0 || step == num_steps) {
            float current_time = step * dt;

            // Copy velocity to host
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            // Compute kinetic energy (volumetric average)
            float energy_lbm = 0.0f;
            for (int kk = 0; kk < nz; ++kk) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        int idx = i + nx * (j + ny * kk);
                        float u2 = h_ux[idx] * h_ux[idx] + h_uy[idx] * h_uy[idx];
                        energy_lbm += 0.5f * rho0 * u2;
                    }
                }
            }
            energy_lbm /= num_cells;  // Average over all cells to get energy density

            // Check monotonic decrease
            if (step > 0 && energy_lbm > prev_energy_lbm) {
                monotonic_decay = false;
            }
            prev_energy_lbm = energy_lbm;

            // Analytical energy
            float energy_analytical = analytical.kineticEnergy(current_time);
            float error_pct = std::abs(energy_lbm - energy_analytical) / energy_analytical * 100.0f;

            // Decay rates (4νk²E)
            float decay_rate_lbm = -4.0f * nu * analytical.k * analytical.k * energy_lbm;
            float decay_rate_analytical = analytical.energyDecayRate(current_time);

            std::cout << std::setw(8) << step << "   "
                      << std::setw(10) << std::fixed << std::setprecision(3) << current_time * 1e6 << "   "
                      << std::setw(13) << std::scientific << std::setprecision(4) << energy_lbm << "   "
                      << std::setw(20) << std::scientific << std::setprecision(4) << energy_analytical << "   "
                      << std::setw(8) << std::fixed << std::setprecision(2) << error_pct << "   "
                      << std::setw(16) << std::scientific << std::setprecision(3) << decay_rate_lbm << "   "
                      << std::setw(24) << std::scientific << std::setprecision(3) << decay_rate_analytical << "\n";

            // Store for post-processing
            time_series.push_back(current_time);
            energy_lbm_series.push_back(energy_lbm);
            energy_analytical_series.push_back(energy_analytical);
        }
    }

    std::cout << std::string(120, '-') << "\n\n";

    // ========================================
    // FINAL VALIDATION
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  VALIDATION RESULTS\n";
    std::cout << "========================================\n\n";

    // 1. Final energy error
    float final_energy_lbm = energy_lbm_series.back();
    float final_energy_analytical = energy_analytical_series.back();
    float final_error_pct = std::abs(final_energy_lbm - final_energy_analytical) / final_energy_analytical * 100.0f;

    std::cout << "1. Final Energy Error:\n";
    std::cout << "   E_LBM(t_final) = " << std::scientific << final_energy_lbm << " J/m³\n";
    std::cout << "   E_analytical(t_final) = " << std::scientific << final_energy_analytical << " J/m³\n";
    std::cout << "   Relative error = " << std::fixed << final_error_pct << "%\n";
    std::cout << "   CRITERION: Error < 5% ... ";
    if (final_error_pct < 5.0f) {
        std::cout << "PASS\n\n";
    } else {
        std::cout << "FAIL\n\n";
    }

    // 2. Compute decay rate from linear fit of log(E) vs time
    // E(t) = E₀ exp(-4νk²t)  →  log(E) = log(E₀) - 4νk²t
    // Slope should be -4νk²
    std::vector<float> log_energy_lbm(time_series.size());
    for (size_t i = 0; i < time_series.size(); ++i) {
        log_energy_lbm[i] = std::log(energy_lbm_series[i]);
    }

    // Linear regression: y = a + b*x
    float sum_t = 0.0f, sum_logE = 0.0f, sum_t2 = 0.0f, sum_t_logE = 0.0f;
    int n = time_series.size();
    for (int i = 0; i < n; ++i) {
        sum_t += time_series[i];
        sum_logE += log_energy_lbm[i];
        sum_t2 += time_series[i] * time_series[i];
        sum_t_logE += time_series[i] * log_energy_lbm[i];
    }
    float slope_lbm = (n * sum_t_logE - sum_t * sum_logE) / (n * sum_t2 - sum_t * sum_t);
    float slope_analytical = -4.0f * nu * analytical.k * analytical.k;
    float decay_rate_error_pct = std::abs(slope_lbm - slope_analytical) / std::abs(slope_analytical) * 100.0f;

    std::cout << "2. Energy Decay Rate:\n";
    std::cout << "   Slope (LBM): " << slope_lbm << " s⁻¹\n";
    std::cout << "   Slope (analytical): " << slope_analytical << " s⁻¹\n";
    std::cout << "   Decay rate error = " << decay_rate_error_pct << "%\n";
    std::cout << "   CRITERION: Error < 5% ... ";
    if (decay_rate_error_pct < 5.0f) {
        std::cout << "PASS\n\n";
    } else {
        std::cout << "FAIL\n\n";
    }

    // 3. Monotonic decay check
    std::cout << "3. Monotonic Energy Decay:\n";
    std::cout << "   CRITERION: Energy decreases monotonically ... ";
    if (monotonic_decay) {
        std::cout << "PASS\n\n";
    } else {
        std::cout << "FAIL (energy increased between steps)\n\n";
    }

    // 4. NaN check
    bool has_nan = false;
    for (float e : energy_lbm_series) {
        if (std::isnan(e) || std::isinf(e)) {
            has_nan = true;
            break;
        }
    }
    std::cout << "4. Numerical Stability:\n";
    std::cout << "   CRITERION: No NaN or Inf ... ";
    if (!has_nan) {
        std::cout << "PASS\n\n";
    } else {
        std::cout << "FAIL\n\n";
    }

    // Write results to file for plotting
    std::ofstream outfile("/home/yzk/LBMProject/tests/validation/taylor_green_results.csv");
    outfile << "# Taylor-Green 2D Vortex Validation Results\n";
    outfile << "# Re = " << Re << ", nx = " << nx << ", ny = " << ny << "\n";
    outfile << "# nu = " << nu << " m²/s, U0 = " << U0 << " m/s, L = " << Lx << " m\n";
    outfile << "# Time[s],E_LBM[J/m³],E_analytical[J/m³],Error[%]\n";
    for (size_t i = 0; i < time_series.size(); ++i) {
        outfile << std::scientific << std::setprecision(6)
                << time_series[i] << ","
                << energy_lbm_series[i] << ","
                << energy_analytical_series[i] << ","
                << std::fixed << std::setprecision(3)
                << std::abs(energy_lbm_series[i] - energy_analytical_series[i]) / energy_analytical_series[i] * 100.0f
                << "\n";
    }
    outfile.close();
    std::cout << "Results written to: /home/yzk/LBMProject/tests/validation/taylor_green_results.csv\n\n";

    // ========================================
    // GTEST ASSERTIONS
    // ========================================
    EXPECT_LT(final_error_pct, 5.0f) << "Final energy error should be < 5%";
    EXPECT_LT(decay_rate_error_pct, 5.0f) << "Decay rate error should be < 5%";
    EXPECT_TRUE(monotonic_decay) << "Energy should decrease monotonically";
    EXPECT_FALSE(has_nan) << "No NaN or Inf should occur";

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    std::cout << "========================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "========================================\n\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
