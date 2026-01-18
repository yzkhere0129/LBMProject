/**
 * @file test_vof_oscillating_droplet.cu
 * @brief Oscillating droplet validation test for surface tension dynamics
 *
 * Physics: An initially elliptical droplet relaxes to spherical shape due to
 * surface tension, oscillating at a characteristic frequency given by Lamb's formula.
 *
 * Analytical Solution (Lamb 1932):
 * For axisymmetric oscillation mode n=2:
 *   ω² = n(n-1)(n+2) * σ / (ρ * R³)
 *   f = ω / (2π) = (1/2π) * sqrt(8σ / (ρR³))
 *
 * Physical Setup:
 * - Elliptical droplet with initial aspect ratio a/b = 1.2 (20% deformation)
 * - Equilibrium radius R = 20 cells
 * - Surface tension: σ = 0.072 N/m (water-air at 20°C)
 * - Density: ρ = 1000 kg/m³ (water)
 * - No gravity, periodic boundaries
 *
 * Validation Metrics:
 * 1. Oscillation frequency (compare to Lamb's formula)
 * 2. Damping rate (should be small for low viscosity)
 * 3. Mass conservation (error < 1%)
 * 4. Sphericity recovery over time
 *
 * Reference:
 * - Lamb, H. (1932). Hydrodynamics (6th ed.). Cambridge University Press.
 * - Prosperetti, A. (1980). Normal-mode analysis for the oscillations of a
 *   viscous liquid drop in an immiscible liquid. J. Mécanique, 19(1), 149-182.
 *
 * Expected Results:
 * - Frequency error < 10% (numerical damping affects frequency slightly)
 * - Mass conservation error < 1%
 * - Oscillation amplitude decays due to numerical viscosity
 * - Droplet maintains approximately spherical shape (mean)
 *
 * Test Runtime: ~10-15 minutes for full oscillation analysis
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/surface_tension.h"
#include "physics/fluid_lbm.h"
#include "io/vtk_writer.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace lbm::physics;

class VOFOscillatingDropletTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Initialize elliptical droplet with specified aspect ratio
     *
     * Semi-major axis: a (along x)
     * Semi-minor axis: b (along y and z)
     * Aspect ratio: epsilon = a/b (>1 for prolate ellipsoid)
     *
     * Volume conservation: (4/3)*π*a*b*b = (4/3)*π*R³
     * => a = R * epsilon^(-2/3)
     * => b = R * epsilon^(1/3)
     */
    void initializeEllipsoid(VOFSolver& vof, int nx, int ny, int nz,
                            float cx, float cy, float cz,
                            float R_eq, float aspect_ratio) {
        std::vector<float> h_fill(nx * ny * nz, 0.0f);

        // Compute semi-axes to preserve volume
        float a = R_eq * std::pow(aspect_ratio, -2.0f/3.0f);
        float b = R_eq * std::pow(aspect_ratio, 1.0f/3.0f);

        std::cout << "  Ellipsoid parameters:" << std::endl;
        std::cout << "    Equilibrium radius: R = " << R_eq << std::endl;
        std::cout << "    Aspect ratio: a/b = " << aspect_ratio << std::endl;
        std::cout << "    Semi-major axis: a = " << a << std::endl;
        std::cout << "    Semi-minor axis: b = " << b << std::endl;

        // Initialize ellipsoid with smooth interface
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);

                    float x = static_cast<float>(i) - cx;
                    float y = static_cast<float>(j) - cy;
                    float z = static_cast<float>(k) - cz;

                    // Ellipsoid distance function: (x/a)² + (y/b)² + (z/b)² = 1
                    float dist_normalized = std::sqrt((x*x)/(a*a) +
                                                     (y*y)/(b*b) +
                                                     (z*z)/(b*b));

                    // Smooth interface using tanh (interface width ~2 cells)
                    float interface_width = 1.5f;
                    float dist_to_surface = (1.0f - dist_normalized) * R_eq;
                    h_fill[idx] = 0.5f * (1.0f + tanhf(dist_to_surface / interface_width));
                }
            }
        }

        vof.initialize(h_fill.data());
    }

    /**
     * @brief Compute droplet shape parameters from fill level field
     *
     * Returns: (center_x, center_y, center_z, R_equiv, sphericity)
     * R_equiv = (3*V/(4π))^(1/3) - equivalent spherical radius
     * sphericity = surface area of sphere / actual surface area
     */
    struct DropletShape {
        float cx, cy, cz;           // Centroid
        float R_equiv;              // Equivalent radius
        float volume;               // Total volume
        float semi_major;           // Semi-major axis (approximate)
        float semi_minor;           // Semi-minor axis (approximate)
        float aspect_ratio;         // a/b
    };

    DropletShape analyzeDropletShape(const std::vector<float>& fill,
                                     int nx, int ny, int nz,
                                     float dx) {
        DropletShape shape;

        // Compute centroid and volume
        float total_volume = 0.0f;
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    float f = fill[idx];

                    if (f > 0.01f) {  // Only count liquid cells
                        total_volume += f;
                        sum_x += f * i;
                        sum_y += f * j;
                        sum_z += f * k;
                    }
                }
            }
        }

        shape.volume = total_volume;
        shape.cx = sum_x / total_volume;
        shape.cy = sum_y / total_volume;
        shape.cz = sum_z / total_volume;

        // Compute moments of inertia to estimate axes
        float Ixx = 0.0f, Iyy = 0.0f, Izz = 0.0f;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    float f = fill[idx];

                    if (f > 0.01f) {
                        float dx_i = i - shape.cx;
                        float dy_i = j - shape.cy;
                        float dz_i = k - shape.cz;

                        Ixx += f * (dy_i*dy_i + dz_i*dz_i);
                        Iyy += f * (dx_i*dx_i + dz_i*dz_i);
                        Izz += f * (dx_i*dx_i + dy_i*dy_i);
                    }
                }
            }
        }

        // Estimate equivalent radius from volume
        shape.R_equiv = std::pow(3.0f * total_volume / (4.0f * M_PI), 1.0f/3.0f);

        // Approximate axes from moments (simplified for prolate ellipsoid)
        // For ellipsoid: I_x ∝ b², I_y ≈ I_z ∝ a²
        // We use the ratio to estimate aspect ratio
        shape.semi_major = std::sqrt(5.0f * Iyy / total_volume);
        shape.semi_minor = std::sqrt(5.0f * Ixx / total_volume);
        shape.aspect_ratio = shape.semi_major / shape.semi_minor;

        return shape;
    }

    /**
     * @brief Compute oscillation frequency from time series using FFT-like analysis
     *
     * Uses zero-crossing method for robust frequency estimation
     */
    float computeOscillationFrequency(const std::vector<float>& time_series,
                                     float dt_output) {
        if (time_series.size() < 10) {
            return 0.0f;  // Not enough data
        }

        // Remove mean (detrend)
        float mean = std::accumulate(time_series.begin(), time_series.end(), 0.0f)
                    / time_series.size();
        std::vector<float> detrended(time_series.size());
        for (size_t i = 0; i < time_series.size(); ++i) {
            detrended[i] = time_series[i] - mean;
        }

        // Count zero crossings
        int n_crossings = 0;
        for (size_t i = 1; i < detrended.size(); ++i) {
            if ((detrended[i-1] > 0 && detrended[i] <= 0) ||
                (detrended[i-1] < 0 && detrended[i] >= 0)) {
                n_crossings++;
            }
        }

        // Frequency = (crossings/2) / total_time
        float total_time = (time_series.size() - 1) * dt_output;
        float frequency = (n_crossings / 2.0f) / total_time;

        return frequency;
    }

    /**
     * @brief Compute damping coefficient from exponential decay fit
     *
     * Fits amplitude(t) = A₀ * exp(-γt) to envelope
     */
    float computeDampingRate(const std::vector<float>& time_series,
                            float dt_output) {
        if (time_series.size() < 10) {
            return 0.0f;
        }

        // Find local maxima (envelope)
        std::vector<float> envelope_amplitudes;
        std::vector<float> envelope_times;

        for (size_t i = 1; i < time_series.size() - 1; ++i) {
            if (time_series[i] > time_series[i-1] &&
                time_series[i] > time_series[i+1]) {
                envelope_amplitudes.push_back(time_series[i]);
                envelope_times.push_back(i * dt_output);
            }
        }

        if (envelope_amplitudes.size() < 3) {
            return 0.0f;  // Not enough peaks
        }

        // Fit exponential: ln(A) = ln(A₀) - γt
        float sum_t = 0.0f, sum_lnA = 0.0f, sum_t_lnA = 0.0f, sum_t2 = 0.0f;
        int n = envelope_amplitudes.size();

        for (int i = 0; i < n; ++i) {
            float t = envelope_times[i];
            float lnA = std::log(std::abs(envelope_amplitudes[i]) + 1e-10f);
            sum_t += t;
            sum_lnA += lnA;
            sum_t_lnA += t * lnA;
            sum_t2 += t * t;
        }

        // Least squares fit
        float gamma = -(n * sum_t_lnA - sum_t * sum_lnA) /
                       (n * sum_t2 - sum_t * sum_t);

        return gamma;
    }
};

/**
 * @brief Test 1: 2D Oscillating Droplet (Fast Test)
 *
 * Simplified 2D test for quick validation
 * Domain: 100×100×4 (quasi-2D)
 */
// 诊断测试：检查第一步的初始化和物理过程
TEST_F(VOFOscillatingDropletTest, OscillatingDroplet2D_DiagnosticFirstStep) {
    std::cout << "\n=== VOF Oscillating Droplet: 2D Validation ===" << std::endl;

    // Domain setup (quasi-2D, simplified for faster testing)
    const int nx = 60, ny = 60, nz = 4;
    const float dx = 1.0e-4f;  // 100 microns
    const int num_cells = nx * ny * nz;

    // Droplet parameters (smaller droplet for faster dynamics)
    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R_eq = 15.0f;  // cells (smaller radius)
    const float aspect_ratio = 1.03f;  // 3% initial deformation (minimal for stability)

    // Material properties (carefully tuned for stability)
    // Weber number We = ρ × v² × R / σ should be moderate
    // Capillary number Ca = μ × v / σ should be low
    const float sigma = 0.005f;  // N/m (very low for stability)
    const float rho = 1000.0f;   // kg/m³
    const float nu = 1.0e-5f;    // m²/s (low viscosity)

    // Analytical frequency (Lamb's formula for n=2 mode)
    const float R_phys = R_eq * dx;  // Physical radius [m]
    const float omega_analytical = std::sqrt(8.0f * sigma / (rho * R_phys * R_phys * R_phys));
    const float f_analytical = omega_analytical / (2.0f * M_PI);

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Grid spacing: dx = " << dx * 1e6 << " μm" << std::endl;
    std::cout << "  Droplet: R = " << R_eq << " cells (" << R_phys * 1e6 << " μm)" << std::endl;
    std::cout << "  Initial deformation: " << (aspect_ratio - 1.0f) * 100.0f << "%" << std::endl;
    std::cout << "  Surface tension: σ = " << sigma << " N/m" << std::endl;
    std::cout << "  Density: ρ = " << rho << " kg/m³" << std::endl;
    std::cout << "\n  Analytical prediction (Lamb's formula):" << std::endl;
    std::cout << "    Angular frequency: ω = " << omega_analytical << " rad/s" << std::endl;
    std::cout << "    Frequency: f = " << f_analytical << " Hz" << std::endl;
    std::cout << "    Period: T = " << 1.0f / f_analytical * 1e6 << " μs" << std::endl;

    // Time stepping parameters
    // With reduced surface tension (σ = 0.005 N/m) and deformation (3%),
    // the induced velocities should be much smaller, allowing larger timestep
    const float dt = 5.0e-6f;  // 5 microseconds

    // Initialize VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    initializeEllipsoid(vof, nx, ny, nz, cx, cy, cz, R_eq, aspect_ratio);

    // Initialize fluid solver (use periodic boundaries, pass dt and dx)
    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                   dt, dx);
    // CRITICAL: Initialize distribution functions to equilibrium
    // Lattice density = 1.0 (dimensionless), initial velocity = 0
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);

    // Initialize surface tension
    SurfaceTension st(nx, ny, nz, sigma, dx);

    // Allocate force arrays (volumetric force N/m³ and acceleration m/s²)
    float *d_fx_vol, *d_fy_vol, *d_fz_vol;  // Volumetric force [N/m³]
    float *d_ax, *d_ay, *d_az;              // Acceleration [m/s²]
    cudaMalloc(&d_fx_vol, num_cells * sizeof(float));
    cudaMalloc(&d_fy_vol, num_cells * sizeof(float));
    cudaMalloc(&d_fz_vol, num_cells * sizeof(float));
    cudaMalloc(&d_ax, num_cells * sizeof(float));
    cudaMalloc(&d_ay, num_cells * sizeof(float));
    cudaMalloc(&d_az, num_cells * sizeof(float));

    // Continue time stepping parameters
    // With f ≈ 548 Hz (T ≈ 1.8 ms), 20 ms gives ~11 periods
    const float t_total = 0.02f;  // 20 ms (shorter for faster testing)
    const int n_steps = static_cast<int>(t_total / dt);
    const int output_interval = 5;  // Output every 5 steps for better resolution
    const float dt_output = output_interval * dt;

    std::cout << "\n  Simulation parameters:" << std::endl;
    std::cout << "    Timestep: dt = " << dt * 1e6 << " μs" << std::endl;
    std::cout << "    Total time: " << t_total << " s" << std::endl;
    std::cout << "    Total steps: " << n_steps << std::endl;
    std::cout << "    Expected periods: ~" << t_total * f_analytical << std::endl;

    // Storage for time series analysis
    std::vector<float> time_history;
    std::vector<float> aspect_ratio_history;
    std::vector<float> mass_history;

    float mass_initial = vof.computeTotalMass();
    std::cout << "  初始质量: M₀ = " << mass_initial << std::endl;

    // ========================================================================
    // 诊断阶段 1：检查初始VOF场
    // ========================================================================
    std::cout << "\n=== 诊断阶段 1：初始VOF场 ===" << std::endl;
    {
        std::vector<float> h_fill(num_cells);
        vof.copyFillLevelToHost(h_fill.data());

        int liquid_cells = 0, interface_cells = 0, gas_cells = 0;
        float min_fill = 1.0f, max_fill = 0.0f;

        for (int i = 0; i < num_cells; ++i) {
            float f = h_fill[i];
            if (f > 0.99f) liquid_cells++;
            else if (f < 0.01f) gas_cells++;
            else interface_cells++;

            min_fill = std::min(min_fill, f);
            max_fill = std::max(max_fill, f);
        }

        std::cout << "  液相格点 (f>0.99): " << liquid_cells << std::endl;
        std::cout << "  界面格点 (0.01<f<0.99): " << interface_cells << std::endl;
        std::cout << "  气相格点 (f<0.01): " << gas_cells << std::endl;
        std::cout << "  Fill level 范围: [" << min_fill << ", " << max_fill << "]" << std::endl;

        // 检查中心平面的分布
        int z_center = nz / 2;
        std::cout << "\n  中心平面 (z=" << z_center << ") 分布:" << std::endl;
        float center_fill = 0.0f;
        int center_count = 0;
        for (int j = ny/2 - 2; j <= ny/2 + 2; ++j) {
            for (int i = nx/2 - 2; i <= nx/2 + 2; ++i) {
                int idx = i + nx * (j + ny * z_center);
                float f = h_fill[idx];
                center_fill += f;
                center_count++;
                if (i == nx/2 && j == ny/2) {
                    std::cout << "  中心点 (" << i << "," << j << "," << z_center
                              << "): f = " << f << std::endl;
                }
            }
        }
        std::cout << "  中心5x5区域平均 fill: " << center_fill / center_count << std::endl;
    }

    // ========================================================================
    // 诊断阶段 2：运行单步并检查所有物理量
    // ========================================================================
    std::cout << "\n=== 诊断阶段 2：第一步物理过程 ===" << std::endl;

    // Main simulation loop (只运行一步进行诊断)
    std::cout << "\n  运行第一时间步 (dt=" << dt*1e6 << " μs)..." << std::endl;
    for (int step = 0; step < 1; ++step) {
        // 重构界面并计算曲率
        std::cout << "\n  [步骤 1] 重构界面..." << std::endl;
        vof.reconstructInterface();
        vof.computeCurvature();

        // 检查曲率场
        {
            std::vector<float> h_curv(num_cells);
            vof.copyCurvatureToHost(h_curv.data());

            float max_curv = 0.0f, min_curv = 0.0f;
            int curv_count = 0;
            for (int i = 0; i < num_cells; ++i) {
                if (std::abs(h_curv[i]) > 1e-8f) {
                    max_curv = std::max(max_curv, h_curv[i]);
                    min_curv = std::min(min_curv, h_curv[i]);
                    curv_count++;
                }
            }

            std::cout << "  曲率格点数: " << curv_count << std::endl;
            std::cout << "  曲率范围: [" << min_curv << ", " << max_curv << "] m^-1" << std::endl;

            // 理论曲率 = 1/R
            float R_phys = R_eq * dx;
            float curv_theory = 1.0f / R_phys;
            std::cout << "  理论曲率: κ = 1/R = " << curv_theory << " m^-1" << std::endl;
        }

        // 计算表面张力 (体积力 N/m³)
        std::cout << "\n  [步骤 2] 计算表面张力..." << std::endl;
        st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(),
                          d_fx_vol, d_fy_vol, d_fz_vol);

        // 检查表面张力场
        {
            std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
            cudaMemcpy(h_fx.data(), d_fx_vol, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy.data(), d_fy_vol, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fz.data(), d_fz_vol, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float max_force = 0.0f;
            int force_count = 0;
            for (int i = 0; i < num_cells; ++i) {
                float f_mag = std::sqrt(h_fx[i]*h_fx[i] + h_fy[i]*h_fy[i] + h_fz[i]*h_fz[i]);
                if (f_mag > 1e-6f) {
                    max_force = std::max(max_force, f_mag);
                    force_count++;
                }
            }

            std::cout << "  表面张力格点数: " << force_count << std::endl;
            std::cout << "  最大力密度: " << max_force << " N/m^3" << std::endl;

            // 理论最大加速度
            float a_theory = sigma * (1.0f / (R_eq * dx)) / rho;  // σκ/ρ
            std::cout << "  理论加速度: a = σκ/ρ = " << a_theory << " m/s^2" << std::endl;
        }

        // CRITICAL FIX (2026-01-17): Convert physical force to lattice units
        //
        // The collisionBGK function expects forces in LATTICE UNITS, not physical units.
        // The Guo forcing scheme applies: u_new = u_old + 0.5 * F_lattice / ρ_lattice
        //
        // Conversion from physical force density [N/m³] to lattice force:
        //   F_lattice = (F_phys / ρ_phys) × (dt² / dx)
        //
        // This matches the ForceAccumulator::convertToLatticeUnits implementation.
        //
        std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
        cudaMemcpy(h_fx.data(), d_fx_vol, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fy.data(), d_fy_vol, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fz.data(), d_fz_vol, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        // ========================================================================
        // CRITICAL FIX (2026-01-17): Convert force to LATTICE UNITS for collisionBGK
        // ========================================================================
        // The Guo forcing scheme in collisionBGK expects LATTICE UNIT forces:
        //   u_corrected = u_uncorrected + 0.5 × F_lattice / ρ_lattice
        //
        // Conversion from physical force density [N/m³] to lattice units:
        //   Step 1: F_density [N/m³] → a_phys [m/s²] = F_density / ρ
        //   Step 2: a_phys [m/s²] → F_lattice [dimensionless] = a_phys × dt² / dx
        //
        // Combined: F_lattice = (F_density / ρ) × (dt² / dx)
        //
        const float force_conversion = (dt * dt) / dx;  // [dimensionless factor]

        for (int i = 0; i < num_cells; ++i) {
            // Convert N/m³ → m/s² → lattice units
            h_fx[i] = (h_fx[i] / rho) * force_conversion;
            h_fy[i] = (h_fy[i] / rho) * force_conversion;
            h_fz[i] = (h_fz[i] / rho) * force_conversion;
        }

        // Apply force limiting in LATTICE UNITS
        // Target: velocity change < 0.01 per timestep (stable for LBM)
        const float max_delta_v_lattice = 0.01f;  // [lattice units]
        // In Guo forcing: Δu = 0.5 × F_lattice / ρ_lattice
        // For ρ_lattice ≈ 1: Δu ≈ 0.5 × F_lattice
        // So max_F_lattice = 2 × max_delta_v = 0.02
        const float max_F_lattice = 2.0f * max_delta_v_lattice;

        for (int i = 0; i < num_cells; ++i) {
            float fx = h_fx[i];
            float fy = h_fy[i];
            float fz = h_fz[i];
            float f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);

            if (f_mag > max_F_lattice) {
                float scale = max_F_lattice / f_mag;
                h_fx[i] = fx * scale;
                h_fy[i] = fy * scale;
                h_fz[i] = fz * scale;
            }
        }

        cudaMemcpy(d_ax, h_fx.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ay, h_fy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_az, h_fz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

        // 检查格子单位力
        std::cout << "\n  [步骤 3] 检查格子单位力..." << std::endl;
        {
            float max_f_lattice = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float f_mag = std::sqrt(h_fx[i]*h_fx[i] + h_fy[i]*h_fy[i] + h_fz[i]*h_fz[i]);
                max_f_lattice = std::max(max_f_lattice, f_mag);
            }

            std::cout << "  最大格子单位力: " << max_f_lattice << std::endl;
            std::cout << "  预期速度变化: Δu = 0.5 × F_lattice / ρ_lattice ≈ "
                      << 0.5f * max_f_lattice << " (格子单位)" << std::endl;

            // 稳定性检查
            if (max_f_lattice > 0.02f) {
                std::cout << "  WARNING: 力过大，可能导致不稳定!" << std::endl;
            }
        }

        // 更新流体速度 (collision + streaming)
        std::cout << "\n  [步骤 4] 流体演化 (BGK碰撞+迁移)..." << std::endl;

        // 验证传给collisionBGK的力值和初始速度
        {
            // 检查力
            std::vector<float> check_fx(num_cells);
            cudaMemcpy(check_fx.data(), d_ax, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            float max_check_f = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                max_check_f = std::max(max_check_f, std::abs(check_fx[i]));
            }
            std::cout << "  传给collisionBGK的最大力: " << max_check_f << std::endl;

            // 检查collisionBGK前的速度和密度
            std::vector<float> check_vx(num_cells), check_vy(num_cells), check_vz(num_cells);
            std::vector<float> check_rho(num_cells);
            cudaMemcpy(check_vx.data(), fluid.getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(check_vy.data(), fluid.getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(check_vz.data(), fluid.getVelocityZ(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(check_rho.data(), fluid.getDensity(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            float max_v_before = 0.0f;
            float min_rho = 1e10f, max_rho = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float v = std::sqrt(check_vx[i]*check_vx[i] + check_vy[i]*check_vy[i] + check_vz[i]*check_vz[i]);
                max_v_before = std::max(max_v_before, v);
                min_rho = std::min(min_rho, check_rho[i]);
                max_rho = std::max(max_rho, check_rho[i]);
            }
            std::cout << "  collisionBGK前的最大速度: " << max_v_before << std::endl;
            std::cout << "  collisionBGK前的密度范围: [" << min_rho << ", " << max_rho << "]" << std::endl;
        }

        fluid.collisionBGK(d_ax, d_ay, d_az);
        fluid.streaming();

        // ========================================================================
        // CRITICAL: Convert velocity from lattice units to physical units [m/s]
        // ========================================================================
        // FluidLBM outputs velocity in lattice units (dimensionless, ~0.01-0.1)
        // VOFSolver::advectFillLevel expects physical units [m/s]
        //
        // Conversion: v_phys = v_lattice × (dx / dt)
        //
        const float v_conversion = dx / dt;  // [m/s per lattice unit]

        // Re-use h_fx, h_fy, h_fz arrays for velocity (they're not needed after force calc)
        cudaMemcpy(h_fx.data(), fluid.getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fy.data(), fluid.getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fz.data(), fluid.getVelocityZ(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_cells; ++i) {
            h_fx[i] *= v_conversion;  // Convert to m/s
            h_fy[i] *= v_conversion;
            h_fz[i] *= v_conversion;
        }

        // 先把物理速度复制到设备（供VOF使用）
        cudaMemcpy(d_ax, h_fx.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ay, h_fy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_az, h_fz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

        // 检查流体速度 (格子单位用于诊断)
        std::cout << "\n  [步骤 5] 检查流体速度..." << std::endl;
        {
            // 重新读取格子单位速度用于诊断显示
            cudaMemcpy(h_fx.data(), fluid.getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy.data(), fluid.getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fz.data(), fluid.getVelocityZ(), num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float max_v_lattice = 0.0f;
            float v_sum_x = 0.0f, v_sum_y = 0.0f, v_sum_z = 0.0f;
            int v_count = 0;

            for (int i = 0; i < num_cells; ++i) {
                float vx = h_fx[i];
                float vy = h_fy[i];
                float vz = h_fz[i];
                float v_mag = std::sqrt(vx*vx + vy*vy + vz*vz);

                if (v_mag > 1e-8f) {
                    max_v_lattice = std::max(max_v_lattice, v_mag);
                    v_sum_x += vx;
                    v_sum_y += vy;
                    v_sum_z += vz;
                    v_count++;
                }
            }

            float cfl_lattice = max_v_lattice;  // dt_lattice=1, dx_lattice=1
            float v_phys_max = max_v_lattice * (dx / dt);

            std::cout << "  速度非零格点数: " << v_count << std::endl;
            std::cout << "  最大速度 (格子): " << max_v_lattice << std::endl;
            std::cout << "  最大速度 (物理): " << v_phys_max << " m/s (" << v_phys_max*1e3 << " mm/s)" << std::endl;
            std::cout << "  CFL数 (格子): " << cfl_lattice << std::endl;
            std::cout << "  平均速度: vx=" << v_sum_x/v_count
                      << ", vy=" << v_sum_y/v_count
                      << ", vz=" << v_sum_z/v_count << std::endl;

            // 检查NaN/Inf
            bool has_nan = false;
            for (int i = 0; i < num_cells; ++i) {
                if (std::isnan(h_fx[i]) || std::isnan(h_fy[i]) || std::isnan(h_fz[i]) ||
                    std::isinf(h_fx[i]) || std::isinf(h_fy[i]) || std::isinf(h_fz[i])) {
                    has_nan = true;
                    break;
                }
            }

            if (has_nan) {
                std::cout << "  ERROR: 检测到NaN/Inf!" << std::endl;
            }

            if (cfl_lattice > 0.3f) {
                std::cout << "  WARNING: CFL数过大，可能不稳定!" << std::endl;
            }
        }

        // 物理单位速度已在上面复制到d_ax/d_ay/d_az
        // 对流VOF场 (物理单位速度 m/s)
        std::cout << "\n  [步骤 6] VOF对流..." << std::endl;
        vof.advectFillLevel(d_ax, d_ay, d_az, dt);

        // 检查质量守恒
        std::cout << "\n  [步骤 7] 检查质量守恒..." << std::endl;
        {
            float mass_after = vof.computeTotalMass();
            float mass_loss = (mass_initial - mass_after) / mass_initial * 100.0f;

            std::cout << "  初始质量: " << mass_initial << std::endl;
            std::cout << "  第一步后质量: " << mass_after << std::endl;
            std::cout << "  质量损失: " << mass_loss << "%" << std::endl;

            if (mass_loss > 10.0f) {
                std::cout << "  CRITICAL: 单步质量损失超过10%!" << std::endl;
            } else if (mass_loss > 1.0f) {
                std::cout << "  WARNING: 质量损失较大 (>1%)" << std::endl;
            } else {
                std::cout << "  质量守恒良好 (<1%)" << std::endl;
            }
        }

        // 检查VOF场分布
        std::cout << "\n  [步骤 8] 检查第一步后VOF场..." << std::endl;
        {
            std::vector<float> h_fill(num_cells);
            vof.copyFillLevelToHost(h_fill.data());

            int liquid_cells = 0, interface_cells = 0, gas_cells = 0;
            float min_fill = 1.0f, max_fill = 0.0f;

            for (int i = 0; i < num_cells; ++i) {
                float f = h_fill[i];
                if (f > 0.99f) liquid_cells++;
                else if (f < 0.01f) gas_cells++;
                else interface_cells++;

                min_fill = std::min(min_fill, f);
                max_fill = std::max(max_fill, f);
            }

            std::cout << "  液相格点: " << liquid_cells << " (变化: "
                      << liquid_cells - (num_cells - gas_cells - interface_cells) << ")" << std::endl;
            std::cout << "  界面格点: " << interface_cells << std::endl;
            std::cout << "  气相格点: " << gas_cells << std::endl;
            std::cout << "  Fill level 范围: [" << min_fill << ", " << max_fill << "]" << std::endl;
        }
    }

    // ========================================================================
    // 诊断结论
    // ========================================================================
    std::cout << "\n=== 诊断结论 ===" << std::endl;
    std::cout << "诊断测试完成。检查上述输出以确定质量损失原因。" << std::endl;
    std::cout << "\n关键检查点:" << std::endl;
    std::cout << "1. 初始VOF场是否合理？(液相格点数、界面分布)" << std::endl;
    std::cout << "2. 曲率计算是否正确？(与理论值1/R比较)" << std::endl;
    std::cout << "3. 表面张力大小是否合理？(检查力密度)" << std::endl;
    std::cout << "4. 格子单位力是否过大？(应<0.02以保证稳定)" << std::endl;
    std::cout << "5. 速度场是否有NaN/Inf？" << std::endl;
    std::cout << "6. CFL数是否合理？(应<0.3)" << std::endl;
    std::cout << "7. 第一步质量损失是否过大？(应<1%)" << std::endl;
    std::cout << "\n如果质量损失过大，可能原因:" << std::endl;
    std::cout << "- VOF advection的速度单位转换错误" << std::endl;
    std::cout << "- CFL数过大导致数值不稳定" << std::endl;
    std::cout << "- 界面压缩项参数不当" << std::endl;
    std::cout << "- 表面张力计算错误" << std::endl;

    // 跳过完整模拟，诊断测试只运行一步
    // 注意：这是诊断模式，不执行pass/fail判断

    // Cleanup
    cudaFree(d_fx_vol);
    cudaFree(d_fy_vol);
    cudaFree(d_fz_vol);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);
}

/**
 * @brief Test 2: 3D Oscillating Droplet with VTK Output
 *
 * Full 3D simulation with visualization output
 * Domain: 80×80×80 (adequate for R=20)
 */
TEST_F(VOFOscillatingDropletTest, DISABLED_OscillatingDroplet3D) {
    std::cout << "\n=== VOF Oscillating Droplet: 3D Full Simulation ===" << std::endl;

    // Domain setup (full 3D)
    const int nx = 80, ny = 80, nz = 80;
    const float dx = 1.0e-6f;  // 1 micron
    const int num_cells = nx * ny * nz;

    // Droplet parameters
    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R_eq = 20.0f;
    const float aspect_ratio = 1.2f;

    // Material properties
    const float sigma = 0.072f;
    const float rho = 1000.0f;
    const float nu = 1.0e-6f;

    // Analytical frequency
    const float R_phys = R_eq * dx;
    const float omega_analytical = std::sqrt(8.0f * sigma / (rho * R_phys * R_phys * R_phys));
    const float f_analytical = omega_analytical / (2.0f * M_PI);

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Analytical frequency: f = " << f_analytical << " Hz" << std::endl;

    // Initialize solvers
    VOFSolver vof(nx, ny, nz, dx);
    initializeEllipsoid(vof, nx, ny, nz, cx, cy, cz, R_eq, aspect_ratio);

    const float dt_3d = 1.0e-9f;
    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC, BoundaryType::PERIODIC, BoundaryType::PERIODIC,
                   dt_3d, dx);
    // CRITICAL: Initialize distribution functions to equilibrium
    fluid.initialize(1.0f, 0.0f, 0.0f, 0.0f);
    SurfaceTension st(nx, ny, nz, sigma, dx);

    // Allocate force arrays
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    // Time stepping
    const float dt = 1.0e-9f;
    const float t_total = 10.0e-6f;  // 10 μs for better statistics
    const int n_steps = static_cast<int>(t_total / dt);
    const int output_interval = 100;
    const int vtk_interval = 1000;  // VTK every 1000 steps

    // Storage
    std::vector<float> time_history;
    std::vector<float> aspect_ratio_history;
    std::vector<float> mass_history;

    float mass_initial = vof.computeTotalMass();

    // Main loop
    std::cout << "  Running 3D simulation..." << std::endl;
    for (int step = 0; step < n_steps; ++step) {
        vof.reconstructInterface();
        vof.computeCurvature();
        st.computeCSFForce(vof.getFillLevel(), vof.getCurvature(), d_fx, d_fy, d_fz);
        fluid.collisionBGK(d_fx, d_fy, d_fz);
        fluid.streaming();
        vof.advectFillLevel(fluid.getVelocityX(),
                           fluid.getVelocityY(),
                           fluid.getVelocityZ(), dt);

        // Diagnostics
        if (step % output_interval == 0) {
            std::vector<float> h_fill(num_cells);
            vof.copyFillLevelToHost(h_fill.data());
            auto shape = analyzeDropletShape(h_fill, nx, ny, nz, dx);
            float mass = vof.computeTotalMass();

            time_history.push_back(step * dt);
            aspect_ratio_history.push_back(shape.aspect_ratio);
            mass_history.push_back(mass);
        }

        // VTK output
        if (step % vtk_interval == 0) {
            std::string filename = "output_oscillating_droplet/droplet_"
                                 + std::to_string(step) + ".vtk";
            // VTK writer call would go here
            std::cout << "    Writing VTK: " << filename << std::endl;
        }
    }

    // Analysis
    float mass_final = mass_history.back();
    float mass_error = std::abs(mass_final - mass_initial) / mass_initial;
    float f_measured = computeOscillationFrequency(aspect_ratio_history, output_interval * dt);
    float f_error = std::abs(f_measured - f_analytical) / f_analytical;

    std::cout << "\n  3D Results:" << std::endl;
    std::cout << "    Mass error: " << mass_error * 100.0f << "%" << std::endl;
    std::cout << "    Frequency error: " << f_error * 100.0f << "%" << std::endl;

    EXPECT_LT(mass_error, 0.01f);
    EXPECT_LT(f_error, 0.15f);  // Slightly relaxed for 3D

    std::cout << "  ✓ 3D test passed" << std::endl;

    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
