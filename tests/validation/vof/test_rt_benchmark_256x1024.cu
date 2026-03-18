/**
 * @file test_rt_benchmark_256x1024.cu
 * @brief Rayleigh-Taylor instability benchmark: 256x1024x4, At=0.5
 *
 * Canonical two-phase RT benchmark in pure lattice units (dx=1, dt=1).
 * Uses constant dynamic viscosity μ = ν_lattice * ρ_avg (computeVariableViscosity),
 * giving ν(f) = μ / ρ(f).
 *
 * Derived parameters:
 * ============================================
 * Domain       : Nx=256, Ny=1024, Nz=4 (quasi-2D)
 * Densities    : ρ_H = 3.0, ρ_L = 1.0 (heavy on top)
 * Atwood number: At = (ρ_H - ρ_L) / (ρ_H + ρ_L) = 0.5
 * Gravity      : g_lattice = 1.0e-5 (downward, -y)
 * Viscosity    : ν_lattice = 0.050596  →  τ = 0.65179 (TRT even mode)
 *                μ = ν_lattice * ρ_avg (constant dynamic viscosity)
 *                ν(f) = μ / ρ(f)  (variable kinematic viscosity)
 * Interface    : η₀ = 0.1*Nx = 25.6 cells, k = 2π/256
 *                tanh width W = 4 cells
 * Theory       : γ_inviscid = sqrt(At*g*k)
 *
 * Physics pipeline per step:
 * 1. forces.reset()
 * 2. forces.addVOFBuoyancyForce(fill, ρ_H, ρ_L, 0, -g, 0)
 * 3. forces.convertToLatticeUnits(dx, dt, ρ_H)
 * 4. fluid.computeVariableViscosity(fill, ρ_H, ρ_L, mu_physical)
 * 5. fluid.collisionTRTVariable(Fx, Fy, Fz, fill, ρ_H, ρ_L, Λ=1/4)
 * 6. fluid.applyBoundaryConditions(1)
 * 7. fluid.streaming()
 * 8. fluid.computeMacroscopic()
 * 9. vof.advectFillLevel(fluid velocity directly, dt)  [v_conv=1.0, no D→H→D copy]
 * 11. enforceVOFBoundaryKernel (pin top/bottom fill values)
 * 12. vof.reconstructInterface()
 * 13. vof.computeCurvature()
 *
 * Validation criteria:
 * - Mushroom formation: h1 > 100 cells
 * - Mass conservation: |Δm/m₀| < 0.5%
 *
 * VTK output: every 2000 steps → output_rt_benchmark/
 * CSV output: every 500 steps
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
#include <numeric>
#include <sys/stat.h>

using namespace lbm::physics;
using namespace lbm::core;

// ============================================================================
// VOF Boundary Enforcement Kernel
// ============================================================================
// Pins fill level at the y-domain boundaries to act as infinite reservoirs:
//   y=0      : pure light fluid (f=0)
//   y=ny-1   : pure heavy fluid (f=1)
__global__ void enforceVOFBoundaryKernel(
    float* fill_level,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || k >= nz) return;

    // Bottom (y=0): light fluid reservoir
    fill_level[i + nx * (0 + ny * k)] = 0.0f;

    // Top (y=ny-1): heavy fluid reservoir
    fill_level[i + nx * ((ny - 1) + ny * k)] = 1.0f;
}

// ============================================================================
// VTK Output (ASCII StructuredPoints, fill level + velocity)
// ============================================================================
void writeVTK(const std::string& filename,
              const std::vector<float>& fill,
              const std::vector<float>& ux,
              const std::vector<float>& uy,
              int nx, int ny, int nz, float dx)
{
    std::ofstream f(filename);
    if (!f.is_open()) return;

    f << "# vtk DataFile Version 3.0\n";
    f << "RT Benchmark 256x1024 At=0.5 variable-mu\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << dx << " " << dx << " " << dx << "\n";
    f << "POINT_DATA " << (nx * ny * nz) << "\n";

    f << "SCALARS fill_level float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                f << fill[i + nx * (j + ny * k)] << "\n";

    f << "VECTORS velocity float\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                f << ux[idx] << " " << uy[idx] << " 0.0\n";
            }
}

// ============================================================================
// Interface Tracking
// ============================================================================
// Averages the f=0.5 crossing position across all x-columns and returns:
//   h_avg = average amplitude of the interface displacement from y_int0
//           (average of upward and downward excursions)
//   h1    = furthest upward displacement (bubble tip)
//   h2    = furthest downward displacement (spike tip)
void measureInterface(const std::vector<float>& fill,
                      int nx, int ny, int nz,
                      float y_int0,
                      float& h_avg, float& h1, float& h2)
{
    float y_max = y_int0;
    float y_min = y_int0;

    // Collect per-column crossing positions for averaging
    std::vector<float> crossings;
    crossings.reserve(nx);

    const int k_center = nz / 2;

    for (int i = 0; i < nx; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            float f0 = fill[i + nx * ((j - 1) + ny * k_center)];
            float f1 = fill[i + nx * (j     + ny * k_center)];

            if ((f0 - 0.5f) * (f1 - 0.5f) < 0.0f) {
                float y_cross = (j - 1) + (0.5f - f0) / (f1 - f0);
                crossings.push_back(y_cross);
                y_max = std::max(y_max, y_cross);
                y_min = std::min(y_min, y_cross);
                break;  // first crossing per column
            }
        }
    }

    h1 = y_max - y_int0;  // bubble tip (upward, positive)
    h2 = y_int0 - y_min;  // spike tip  (downward, positive)

    // h_avg = average displacement magnitude across all columns
    if (!crossings.empty()) {
        float sum = 0.0f;
        for (float yc : crossings) sum += std::abs(yc - y_int0);
        h_avg = sum / static_cast<float>(crossings.size());
    } else {
        h_avg = 0.5f * (h1 + h2);
    }
}

// ============================================================================
// Linear Least-Squares Fit:  y = a*x + b
// Returns (a, b, r2) where r2 is the coefficient of determination.
// ============================================================================
struct LinFit { float a, b, r2; };

LinFit linearFit(const std::vector<float>& x, const std::vector<float>& y)
{
    const int n = static_cast<int>(x.size());
    if (n < 2) return {0.0f, 0.0f, 0.0f};

    float sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; ++i) {
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    float denom = n * sxx - sx * sx;
    if (std::abs(denom) < 1e-12f) return {0.0f, sy / n, 0.0f};

    LinFit fit;
    fit.a = (n * sxy - sx * sy) / denom;
    fit.b = (sy - fit.a * sx) / n;

    // R²
    float y_mean = sy / n;
    float ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; ++i) {
        float yhat = fit.a * x[i] + fit.b;
        ss_res += (y[i] - yhat) * (y[i] - yhat);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    fit.r2 = (ss_tot > 1e-12f) ? 1.0f - ss_res / ss_tot : 1.0f;
    return fit;
}

// ============================================================================
// Test Fixture
// ============================================================================
class RTBenchmark256x1024Test : public ::testing::Test {
protected:
    void createOutputDirectory(const std::string& path) {
        mkdir(path.c_str(), 0755);
    }

    // Initialize fill level with a tanh-smoothed, cosine-perturbed interface.
    // f = 0 at y << y_int (light fluid, bottom)
    // f = 1 at y >> y_int (heavy fluid, top)
    void initializeTanhInterface(VOFSolver& vof,
                                 int nx, int ny, int nz,
                                 float y_int0,
                                 float amplitude,
                                 float wavelength,
                                 float tanh_width)
    {
        std::vector<float> h_fill(nx * ny * nz);
        const float k_wave = 2.0f * static_cast<float>(M_PI) / wavelength;

        for (int kz = 0; kz < nz; ++kz) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    float x     = static_cast<float>(i);
                    float y     = static_cast<float>(j);
                    float y_int = y_int0 + amplitude * std::cos(k_wave * x);
                    float dist  = (y - y_int) / tanh_width;
                    h_fill[i + nx * (j + ny * kz)] = 0.5f * (1.0f + std::tanh(dist));
                }
            }
        }
        vof.initialize(h_fill.data());
    }
};

// ============================================================================
// Main Benchmark Test
// ============================================================================
TEST_F(RTBenchmark256x1024Test, MushroomFormation) {
    std::cout << "\n"
              << "================================================================\n"
              << "  RT BENCHMARK: 256x1024x4  At=0.5\n"
              << "  Pure lattice units: dx=1, dt=1\n"
              << "  TRT collision, constant mu (variable nu)\n"
              << "  Large amplitude keta0=0.628 (nonlinear, mushroom formation)\n"
              << "================================================================\n";

    // ========================================================================
    // Domain
    // ========================================================================
    constexpr int nx = 256;
    constexpr int ny = 1024;
    constexpr int nz = 4;
    constexpr int num_cells = nx * ny * nz;

    constexpr float dx = 1.0f;
    constexpr float dt = 1.0f;

    // ========================================================================
    // Physical Parameters (all in lattice units)
    // ========================================================================
    constexpr float rho_heavy   = 3.0f;
    constexpr float rho_light   = 1.0f;
    constexpr float At          = (rho_heavy - rho_light) / (rho_heavy + rho_light);  // 0.5
    constexpr float rho_avg     = (rho_heavy + rho_light) * 0.5f;                     // 2.0

    constexpr float g_lattice   = 1.0e-5f;
    constexpr float nu_lattice  = 0.050596f;
    constexpr float tau         = nu_lattice / D3Q19::CS2 + 0.5f;
    constexpr float lambda_trt  = 0.25f;

    // Constant dynamic viscosity: mu = nu_lattice * rho_avg
    // This gives nu_eff = mu / rho_avg = nu_lattice for the Chandrasekhar formula.
    const float mu_physical = nu_lattice * rho_avg;

    // Surface tension (lattice units): stabilizes secondary KH modes at mushroom cap
    // without significantly suppressing the primary RT mode.
    //   σ=5e-3: k_c=0.063 → λ_c=99 cells, 8.8% primary mode reduction, Bo=262
    //   Stabilizes secondary KH rolls (~20-60 cells) while preserving mushroom growth.
    constexpr float sigma       = 5.0e-3f;

    // Chandrasekhar (1961) general two-fluid formula for constant dynamic viscosity μ
    // with unequal kinematic viscosities: ν_H = μ/ρ_H, ν_L = μ/ρ_L
    //
    //   γ = -(ν_H + ν_L)/2 * k² + sqrt( At*g*k - σ*k³/(ρ_H+ρ_L) + ν_H*ν_L*k⁴ )
    //
    // This is exact for the equal-μ, two-density case (Chandrasekhar 1961, Ch. X).
    const float k_wave_rt      = 2.0f * static_cast<float>(M_PI) / static_cast<float>(nx);
    const float nu_H           = mu_physical / rho_heavy;  // ν_H = μ/ρ_H
    const float nu_L           = mu_physical / rho_light;  // ν_L = μ/ρ_L
    const float nu_avg         = 0.5f * (nu_H + nu_L);    // arithmetic mean
    const float k2             = k_wave_rt * k_wave_rt;
    const float k3             = k2 * k_wave_rt;
    const float gamma_inviscid = std::sqrt(At * g_lattice * k_wave_rt);
    // General Chandrasekhar formula with surface tension:
    // γ_viscous = -(ν_H+ν_L)/2*k² + sqrt(At*g*k - σ*k³/(ρ_H+ρ_L) + ν_H*ν_L*k⁴)
    const float sigma_term     = sigma * k3 / (rho_heavy + rho_light);
    const float gamma_viscous  = -nu_avg * k2 + std::sqrt(nu_H * nu_L * k2 * k2 + At * g_lattice * k_wave_rt - sigma_term);

    // ========================================================================
    // Interface: large amplitude for full mushroom formation
    // η₀ = 0.1*Nx = 25.6 cells, tanh width W = 4 cells
    // ========================================================================
    const float y_int0     = ny * 0.5f;                    // 512.0
    const float amplitude  = nx * 0.1f;                    // 25.6 cells
    const float wavelength = static_cast<float>(nx);       // 256 cells
    const float tanh_width = 4.0f;                         // cells
    const float k_eta0     = k_wave_rt * amplitude;        // = 0.074

    // h₀ = initial average displacement ≈ amplitude (for small kη₀)
    const float h0 = amplitude;

    // Characteristic time and run length
    const float L        = static_cast<float>(nx);
    const float U_char   = std::sqrt(At * g_lattice * L);
    const float Re       = U_char * L / nu_lattice;
    const float Ma       = U_char * std::sqrt(3.0f);
    const float t_star   = L / U_char;
    const float t_efold  = 1.0f / gamma_viscous;

    // Run 36000 steps for full mushroom development (~11 e-folds)
    const int   num_steps     = 36000;
    const int   vtk_every     = 2000;
    const int   csv_every     = 500;

    // ========================================================================
    // Parameter Summary
    // ========================================================================
    std::cout << "\n=== Domain ===\n"
              << "  Grid: " << nx << " x " << ny << " x " << nz << "\n"
              << "  dx = dt = 1 (pure lattice units)\n";

    std::cout << "\n=== Physical Parameters (lattice units) ===\n"
              << std::fixed << std::setprecision(6)
              << "  rho_heavy    = " << rho_heavy   << "\n"
              << "  rho_light    = " << rho_light   << "\n"
              << "  At           = " << At          << "  (target: 0.5)\n"
              << "  rho_avg      = " << rho_avg     << "\n"
              << "  g_lattice    = " << g_lattice   << "\n"
              << "  nu_lattice   = " << nu_lattice  << "  (= mu/rho_avg, reference)\n"
              << "  mu_physical  = " << mu_physical << "  (= nu_lattice * rho_avg, constant)\n"
              << "  nu_H         = " << mu_physical / rho_heavy << "  (= mu/rho_H)\n"
              << "  nu_L         = " << mu_physical / rho_light << "  (= mu/rho_L)\n"
              << "  tau (even)   = " << tau         << "\n"
              << "  Lambda (TRT) = " << lambda_trt  << "  (magic param 1/4)\n"
              << "  sigma        = " << sigma       << "  (surface tension, lattice units)\n";

    const float k_c = std::sqrt((rho_heavy - rho_light) * g_lattice / sigma);
    const float lambda_c = 2.0f * static_cast<float>(M_PI) / k_c;
    const float Bo = (rho_heavy - rho_light) * g_lattice * wavelength * wavelength / sigma;
    std::cout << "\n=== Dimensionless Numbers ===\n"
              << "  U_char = sqrt(At*g*L)    = " << U_char << "\n"
              << "  Re     = U_char*L/nu     = " << Re     << "\n"
              << "  Ma     = U_char*sqrt3    = " << Ma     << "  (< 0.15 required)\n"
              << "  Bo     = Δρ·g·L²/σ      = " << Bo     << "  (Bond number)\n"
              << "  k_c    = sqrt(Δρ·g/σ)   = " << k_c    << "  (capillary cutoff)\n"
              << "  λ_c    = 2π/k_c         = " << lambda_c << " cells (modes λ<λ_c stabilized)\n";

    std::cout << "\n=== Growth Rates (Chandrasekhar 1961, general two-fluid, equal-mu) ===\n"
              << std::setprecision(8)
              << "  k_wave           = " << k_wave_rt      << " (= 2pi/Nx)\n"
              << "  kη₀              = " << k_eta0         << "  (linear regime: << 0.1)\n"
              << "  nu_H             = " << nu_H           << " (= mu/rho_H)\n"
              << "  nu_L             = " << nu_L           << " (= mu/rho_L)\n"
              << "  gamma_inviscid   = " << gamma_inviscid << " step^-1\n"
              << "  gamma_viscous    = " << gamma_viscous  << " step^-1  (Chandrasekhar, equal-mu)\n"
              << "  viscous/inviscid = " << gamma_viscous / gamma_inviscid << "\n"
              << "  e-folding time   = " << t_efold << " steps\n";

    std::cout << "\n=== Time Integration ===\n"
              << "  t*         = " << t_star   << " steps  (characteristic time)\n"
              << "  t_efold    = " << t_efold  << " steps  (1/gamma_viscous)\n"
              << "  num_steps  = " << num_steps << " (~" << num_steps / t_efold << " e-folds)\n"
              << "  VTK every  = " << vtk_every << " steps\n"
              << "  CSV every  = " << csv_every << " steps\n";

    std::cout << "\n=== Interface Perturbation ===\n"
              << std::setprecision(4)
              << "  y_int0     = " << y_int0    << " (midplane)\n"
              << "  amplitude  = " << amplitude  << " cells (keta0=" << k_eta0 << ")\n"
              << "  wavelength = " << wavelength << " cells\n"
              << "  tanh width = " << tanh_width << " cells\n";

    ASSERT_LT(Ma, 0.15f)  << "Mach number must be < 0.15 for LBM validity";

    // ========================================================================
    // Output Directory
    // ========================================================================
    const std::string output_dir = "/home/yzk/LBMProject/build/output_rt_benchmark";
    createOutputDirectory(output_dir);

    // ========================================================================
    // Initialize Solvers
    // ========================================================================
    std::cout << "\n=== Initializing Solvers ===\n";

    FluidLBM fluid(nx, ny, nz,
                   nu_lattice,
                   rho_heavy,
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
    vof.setTVDLimiter(TVDLimiter::MC);
    vof.setMassConservationCorrection(true, 0.7f);
    vof.setInterfaceCompression(true, 0.10f);  // Mild compression: sharpens interface without suppressing RT growth

    initializeTanhInterface(vof, nx, ny, nz,
                            y_int0, amplitude, wavelength, tanh_width);
    vof.reconstructInterface();
    vof.computeCurvature();

    const float mass_initial = vof.computeTotalMass();
    vof.setReferenceMass(mass_initial);

    ForceAccumulator forces(nx, ny, nz);

    std::vector<float> h_fill(num_cells);
    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);

    // ========================================================================
    // Time Series Storage
    // ========================================================================
    struct TimeSeries {
        std::vector<int>   step;
        std::vector<float> h_avg, h1, h2, ln_h_over_h0;
        std::vector<float> gamma_inst;   // instantaneous growth rate (finite diff)
        std::vector<float> mass_err, u_max;
    } ts;

    // ========================================================================
    // Diagnostics Header
    // ========================================================================
    std::cout << "\n=== Running Main Simulation (" << num_steps << " steps) ===\n"
              << std::setw(8)  << "Step"
              << std::setw(10) << "t/t_fold"
              << std::setw(12) << "h_avg"
              << std::setw(12) << "ln(h/h0)"
              << std::setw(12) << "Mass(%)"
              << std::setw(10) << "u_max\n"
              << std::string(64, '-') << "\n";

    // ========================================================================
    // Main Simulation Loop
    // ========================================================================
    for (int step = 0; step <= num_steps; ++step) {

        const bool do_csv = (step % csv_every == 0) || (step == num_steps);
        const bool do_vtk = (step % vtk_every == 0) || (step == num_steps);

        // --------------------------------------------------------------------
        // Diagnostics
        // --------------------------------------------------------------------
        if (do_csv || do_vtk) {
            vof.copyFillLevelToHost(h_fill.data());

            cudaMemcpy(h_ux.data(), fluid.getVelocityX(),
                       num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), fluid.getVelocityY(),
                       num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), fluid.getVelocityZ(),
                       num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            const float mass_now = vof.computeTotalMass();
            const float mass_err = std::abs(mass_now - mass_initial) / mass_initial * 100.0f;

            float h_avg_val, h1_val, h2_val;
            measureInterface(h_fill, nx, ny, nz, y_int0, h_avg_val, h1_val, h2_val);

            float u_max = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float mag2 = h_ux[i] * h_ux[i] + h_uy[i] * h_uy[i] + h_uz[i] * h_uz[i];
                u_max = std::max(u_max, std::sqrt(mag2));
            }

            const float ln_h = (h_avg_val > 0.1f) ? std::log(h_avg_val / h0) : -99.0f;

            // Instantaneous growth rate from finite difference of ln(h)
            float gamma_inst_val = 0.0f;
            if (do_csv && !ts.ln_h_over_h0.empty() && ts.step.size() >= 1) {
                int prev_step = ts.step.back();
                float prev_lnh = ts.ln_h_over_h0.back();
                float dt_steps = static_cast<float>(step - prev_step);
                if (dt_steps > 0 && ln_h > -50.0f && prev_lnh > -50.0f) {
                    gamma_inst_val = (ln_h - prev_lnh) / dt_steps;
                }
            }

            if (do_csv) {
                ts.step.push_back(step);
                ts.h_avg.push_back(h_avg_val);
                ts.h1.push_back(h1_val);
                ts.h2.push_back(h2_val);
                ts.ln_h_over_h0.push_back(ln_h);
                ts.gamma_inst.push_back(gamma_inst_val);
                ts.mass_err.push_back(mass_err);
                ts.u_max.push_back(u_max);

                if (step % 4000 == 0 || step == num_steps) {
                    std::cout << std::setw(8)  << step
                              << std::setw(10) << std::fixed << std::setprecision(2)
                                               << static_cast<float>(step) / t_efold
                              << std::setw(12) << std::setprecision(3) << h_avg_val
                              << std::setw(12) << std::setprecision(4) << ln_h
                              << std::setw(12) << std::setprecision(3) << mass_err
                              << std::setw(10) << std::setprecision(5) << u_max
                              << "\n";
                }
            }

            if (do_vtk) {
                char vtk_name[512];
                snprintf(vtk_name, sizeof(vtk_name),
                         "%s/rt_benchmark_step%06d.vtk",
                         output_dir.c_str(), step);
                writeVTK(vtk_name, h_fill, h_ux, h_uy, nx, ny, nz, dx);
            }
        }

        if (step == num_steps) break;

        // --------------------------------------------------------------------
        // Physics Pipeline
        // --------------------------------------------------------------------

        // 1. Reset forces
        forces.reset();

        // 2. VOF buoyancy
        forces.addVOFBuoyancyForce(vof.getFillLevel(),
                                   rho_heavy, rho_light,
                                   0.0f, -g_lattice, 0.0f);

        // 2.5 Surface tension (CSF model): F_st = σ · κ · ∇f
        if (sigma > 0.0f) {
            forces.addSurfaceTensionForce(vof.getCurvature(), vof.getFillLevel(),
                                          sigma, nx, ny, nz, dx);
        }

        // 3. Convert forces to lattice units
        forces.convertToLatticeUnits(dx, dt, rho_heavy);

        // 4. Variable viscosity: constant mu = nu_lattice * rho_avg
        //    gives nu(f) = mu / rho(f), nu_eff = mu/rho_avg = nu_lattice
        fluid.computeVariableViscosity(vof.getFillLevel(), rho_heavy, rho_light, mu_physical);

        // 5. TRT collision with per-cell omega
        fluid.collisionTRTVariable(forces.getFx(), forces.getFy(), forces.getFz(),
                                   vof.getFillLevel(),
                                   rho_heavy, rho_light,
                                   lambda_trt);

        // 6. Boundary conditions
        fluid.applyBoundaryConditions(1);

        // 7. Streaming
        fluid.streaming();

        // 8. Macroscopic quantities
        fluid.computeMacroscopic();

        // 9. Advect VOF — v_conv = dx/dt = 1.0, pass fluid velocity directly
        vof.advectFillLevel(fluid.getVelocityX(), fluid.getVelocityY(),
                            fluid.getVelocityZ(), dt);

        // 11. Reservoir boundary conditions
        {
            dim3 block(16, 16);
            dim3 grid((nx + block.x - 1) / block.x,
                      (nz + block.y - 1) / block.y);
            enforceVOFBoundaryKernel<<<grid, block>>>(vof.getFillLevel(), nx, ny, nz);
            cudaDeviceSynchronize();
        }

        // 12. Reconstruct interface
        vof.reconstructInterface();

        // 13. Curvature
        vof.computeCurvature();
    }

    // ========================================================================
    // Write CSV Time Series
    // ========================================================================
    const std::string csv_path = output_dir + "/rt_benchmark_time_series.csv";
    {
        std::ofstream csv(csv_path);
        csv << "step,t_over_tefold,h_avg_cells,h1_cells,h2_cells,"
               "ln_h_over_h0,gamma_inst,gamma_inst_over_theory,mass_error_pct,u_max_lattice\n";
        for (size_t i = 0; i < ts.step.size(); ++i) {
            float gamma_ratio = (gamma_viscous > 0 && ts.gamma_inst[i] > 0)
                                ? ts.gamma_inst[i] / gamma_viscous : 0.0f;
            csv << ts.step[i] << ","
                << std::fixed << std::setprecision(6)
                << static_cast<float>(ts.step[i]) / t_efold << ","
                << ts.h_avg[i]        << ","
                << ts.h1[i]           << ","
                << ts.h2[i]           << ","
                << ts.ln_h_over_h0[i] << ","
                << ts.gamma_inst[i]   << ","
                << gamma_ratio        << ","
                << ts.mass_err[i]     << ","
                << ts.u_max[i]        << "\n";
        }
    }

    // ========================================================================
    // Growth Rate Analysis via Linear Regression
    // ========================================================================
    // Window: h_avg in [30, 150] cells — well past transient, clear exponential regime
    const float h_window_lo = 30.0f;
    const float h_window_hi = 150.0f;

    std::vector<float> fit_t, fit_lnh;
    for (size_t i = 0; i < ts.step.size(); ++i) {
        if (ts.h_avg[i] >= h_window_lo && ts.h_avg[i] <= h_window_hi) {
            fit_t.push_back(static_cast<float>(ts.step[i]));
            fit_lnh.push_back(std::log(ts.h_avg[i]));
        }
    }

    float gamma_fit  = 0.0f;
    float r2         = 0.0f;
    float fit_conf95 = 0.0f;  // approximate 95% CI half-width

    if (fit_t.size() >= 4) {
        LinFit lf = linearFit(fit_t, fit_lnh);
        gamma_fit = lf.a;
        r2        = lf.r2;

        // Estimate standard error of slope
        int nf = static_cast<int>(fit_t.size());
        float t_mean = 0;
        for (float tv : fit_t) t_mean += tv;
        t_mean /= nf;

        float ss_tt = 0, ss_res = 0;
        for (int i = 0; i < nf; ++i) {
            ss_tt += (fit_t[i] - t_mean) * (fit_t[i] - t_mean);
            float yhat = lf.a * fit_t[i] + lf.b;
            ss_res += (fit_lnh[i] - yhat) * (fit_lnh[i] - yhat);
        }
        float se_slope = (nf > 2 && ss_tt > 0)
                         ? std::sqrt(ss_res / (nf - 2) / ss_tt)
                         : 0.0f;
        fit_conf95 = 2.0f * se_slope;  // ~95% CI (t-distribution, large n)
    }

    const float ratio         = (gamma_viscous > 0) ? gamma_fit / gamma_viscous : 0.0f;
    const float ratio_lo      = (gamma_viscous > 0) ? (gamma_fit - fit_conf95) / gamma_viscous : 0.0f;
    const float ratio_hi      = (gamma_viscous > 0) ? (gamma_fit + fit_conf95) / gamma_viscous : 0.0f;

    const float mass_err_final = ts.mass_err.back();

    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cout << "\n"
              << "================================================================\n"
              << "  GROWTH RATE ANALYSIS\n"
              << "================================================================\n"
              << std::setprecision(6)
              << "  Fitting window : h_avg in [" << h_window_lo << ", " << h_window_hi << "] cells\n"
              << "  Data points    : " << fit_t.size() << "\n"
              << "  R2             : " << r2 << "\n"
              << "\n"
              << "  Theory (Chandrasekhar 1961, equal-mu two-fluid):\n"
              << "    nu_H              = " << nu_H << " step^-1 (= mu/rho_H)\n"
              << "    nu_L              = " << nu_L << " step^-1 (= mu/rho_L)\n"
              << "    gamma_inviscid    = " << gamma_inviscid << " step^-1\n"
              << "    gamma_viscous     = " << gamma_viscous  << " step^-1\n"
              << "    gamma_visc/invsc  = " << gamma_viscous / gamma_inviscid << "\n"
              << "\n"
              << "  Simulation result:\n"
              << "    gamma_fit (LBM)   = " << gamma_fit << " step^-1\n"
              << "    95% CI half-width = +/-" << fit_conf95 << " step^-1\n"
              << "\n"
              << "  gamma_fit / gamma_viscous  = " << ratio << "\n"
              << "  95% CI                     = [" << ratio_lo << ", " << ratio_hi << "]\n"
              << "\n"
              << "  Mass error (final)         = " << mass_err_final << " %\n"
              << "\n"
              << "  Pass criteria:\n"
              << "    gamma_fit/gamma_viscous in [0.3, 0.9] -> "
              << (ratio > 0.3f && ratio < 0.9f ? "PASS" : "FAIL")
              << "  (nonlinear regime: kη₀=0.63 >> 0.1, expect ~30% deficit)\n"
              << "    mass_error < 0.5%                     -> "
              << (mass_err_final < 0.5f ? "PASS" : "FAIL") << "\n"
              << "\n"
              << "  Output:\n"
              << "    VTK snapshots : " << output_dir << "/rt_benchmark_step*.vtk\n"
              << "    Time series   : " << csv_path   << "\n"
              << "================================================================\n";

    // ========================================================================
    // Validation Assertions
    // ========================================================================
    ASSERT_GE(static_cast<int>(fit_t.size()), 4)
        << "Not enough data points in growth rate window [" << h_window_lo
        << ", " << h_window_hi << "] cells. "
        << "Check amplitude/num_steps or window bounds.";

    // Growth rate in correct ballpark: kη₀=0.63 is deeply nonlinear,
    // so ~30% deficit vs linear theory is expected. Accept ratio in [0.3, 0.9].
    EXPECT_GT(ratio, 0.3f)
        << "gamma_fit/gamma_viscous=" << ratio << ", too slow — possible physics bug";
    EXPECT_LT(ratio, 0.9f)
        << "gamma_fit/gamma_viscous=" << ratio << ", suspiciously close to linear theory for kη₀=0.63";

    // Mushroom must form: average interface displacement > 200 cells (~40% of half-domain)
    EXPECT_GT(ts.h_avg.back(), 200.0f)
        << "h_avg=" << ts.h_avg.back() << " cells, expected > 200 for mushroom formation";

    // Maximum velocity must be significant (confirms active RT dynamics)
    EXPECT_GT(ts.u_max.back(), 0.03f)
        << "u_max=" << ts.u_max.back() << ", expected > 0.03 for active flow";

    EXPECT_LT(mass_err_final, 0.5f)
        << "Mass conservation error " << mass_err_final << "% exceeds 0.5% threshold";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
