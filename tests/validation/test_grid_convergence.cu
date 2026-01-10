/**
 * @file test_grid_convergence.cu
 * @brief Grid convergence study for thermal LBM solver
 *
 * This test validates grid independence by solving the same heat diffusion
 * problem on multiple grid resolutions and measuring the convergence rate.
 *
 * PROBLEM: 1D Gaussian heat diffusion with analytical solution
 *   T(x,t) = T0 + A * (σ₀/σ(t)) * exp(-x²/(2σ(t)²))
 *   where σ(t) = sqrt(σ₀² + 2αt)
 *
 * CONVERGENCE ANALYSIS:
 *   For a second-order accurate scheme, the error should scale as:
 *     E(dx) ≈ C * dx²
 *   Taking log: log(E) ≈ log(C) + 2*log(dx)
 *   Convergence order p = log(E₁/E₂) / log(dx₁/dx₂)
 *
 * ACCEPTANCE CRITERIA:
 *   - Convergence order: 1.8 ≤ p ≤ 2.2 (second-order)
 *   - Finest grid error < 1%
 *   - Error decreases monotonically with grid refinement
 *
 * GRID RESOLUTIONS TESTED:
 *   - Coarse:   25 cells  (dx = 8.0 μm)
 *   - Medium:   50 cells  (dx = 4.0 μm)
 *   - Fine:     100 cells (dx = 2.0 μm)
 *   - Finest:   200 cells (dx = 1.0 μm)
 *
 * References:
 *   - Roache, P. J. (1997). Quantification of uncertainty in computational
 *     fluid dynamics. Annual Review of Fluid Mechanics, 29(1), 123-160.
 *   - He, X., et al. (1997). A priori derivation of the lattice Boltzmann
 *     equation. Physical Review E, 55(6), R6333.
 */

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "physics/thermal_lbm.h"

using namespace lbm::physics;

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Analytical solution for 1D Gaussian diffusion
 */
class GaussianDiffusion1D {
public:
    float T0;       // Background temperature [K]
    float A;        // Amplitude [K]
    float sigma0;   // Initial width [m]
    float alpha;    // Thermal diffusivity [m²/s]
    float x_center; // Center position [m]

    GaussianDiffusion1D(float T0_, float A_, float sigma0_, float alpha_, float x_center_)
        : T0(T0_), A(A_), sigma0(sigma0_), alpha(alpha_), x_center(x_center_) {}

    float sigma(float t) const {
        return std::sqrt(sigma0 * sigma0 + 2.0f * alpha * t);
    }

    float peak(float t) const {
        return T0 + A * sigma0 / sigma(t);
    }

    float temperature(float x, float t) const {
        float sig = sigma(t);
        float dx = x - x_center;
        return T0 + A * (sigma0 / sig) * std::exp(-dx * dx / (2.0f * sig * sig));
    }
};

/**
 * @brief Grid resolution configuration
 */
struct GridConfig {
    int nx;          // Number of cells in x-direction
    float dx;        // Grid spacing [m]
    std::string name; // Grid name (e.g., "Coarse", "Fine")

    GridConfig(int nx_, float domain_length, const std::string& name_)
        : nx(nx_), dx(domain_length / (nx_ - 1)), name(name_) {}
};

/**
 * @brief Run simulation on a single grid resolution
 *
 * @param config Grid configuration
 * @param analytical Analytical solution object
 * @param final_time Simulation end time [s]
 * @param alpha Thermal diffusivity [m²/s]
 * @param rho Density [kg/m³]
 * @param cp Specific heat [J/(kg·K)]
 * @return L2 norm error vs analytical solution
 */
float runGridTest(const GridConfig& config,
                  const GaussianDiffusion1D& analytical,
                  float final_time,
                  float alpha,
                  float rho,
                  float cp) {

    const int nx = config.nx;
    const int ny = 3;  // Minimal in y
    const int nz = 3;  // Minimal in z
    const float dx = config.dx;
    const int num_cells = nx * ny * nz;

    // Compute timestep (CFL = 0.1 for stability and accuracy)
    const float CFL = 0.1f;
    const float dt = CFL * dx * dx / alpha;
    const int num_steps = static_cast<int>(std::ceil(final_time / dt));
    const float actual_time = num_steps * dt;

    std::cout << "\n=== Grid: " << config.name << " ===\n";
    std::cout << "  nx = " << nx << " cells\n";
    std::cout << "  dx = " << dx * 1e6 << " μm\n";
    std::cout << "  dt = " << dt * 1e9 << " ns\n";
    std::cout << "  CFL = " << CFL << "\n";
    std::cout << "  Steps = " << num_steps << "\n";
    std::cout << "  Actual time = " << actual_time * 1e6 << " μs\n";

    // Initialize thermal solver
    ThermalLBM thermal(nx, ny, nz, alpha, rho, cp, dt, dx);

    // Set initial Gaussian distribution
    std::vector<float> h_temp(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = i * dx;
                int idx = i + nx * (j + ny * k);
                h_temp[idx] = analytical.temperature(x, 0.0f);
            }
        }
    }

    // Initialize solver
    thermal.initialize(h_temp.data());

    // Time integration
    for (int step = 0; step < num_steps; ++step) {
        thermal.applyBoundaryConditions(0, analytical.T0);  // Adiabatic BC
        thermal.computeTemperature();
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
    }

    // Get final temperature field
    thermal.copyTemperatureToHost(h_temp.data());

    // Compute analytical solution at final time
    std::vector<float> h_temp_analytical(nx);
    for (int i = 0; i < nx; ++i) {
        float x = i * dx;
        h_temp_analytical[i] = analytical.temperature(x, actual_time);
    }

    // Compute L2 error (using centerline only, k=nz/2, j=ny/2)
    // Use proper normalization: ||e||_2 / ||T_analytical - T0||_2
    float sum_squared_error = 0.0f;
    float sum_squared_ref = 0.0f;

    const int j_center = ny / 2;
    const int k_center = nz / 2;

    for (int i = 0; i < nx; ++i) {
        int idx = i + nx * (j_center + ny * k_center);
        float error = h_temp[idx] - h_temp_analytical[i];
        float ref = h_temp_analytical[i] - analytical.T0;  // Reference: excess temperature

        sum_squared_error += error * error;
        sum_squared_ref += ref * ref;
    }

    // Relative L2 error
    float l2_error = std::sqrt(sum_squared_error / sum_squared_ref);

    // Also compute peak error
    float peak_numerical = *std::max_element(h_temp.begin(), h_temp.end());
    float peak_analytical = analytical.peak(actual_time);
    float peak_error = std::abs(peak_numerical - peak_analytical) / (peak_analytical - analytical.T0);

    std::cout << "  Peak (LBM): " << std::fixed << std::setprecision(2) << peak_numerical << " K\n";
    std::cout << "  Peak (Analytical): " << peak_analytical << " K\n";
    std::cout << "  Peak error: " << std::setprecision(4) << peak_error * 100.0f << "%\n";
    std::cout << "  L2 error: " << l2_error * 100.0f << "%\n";

    return l2_error;
}

/**
 * @brief Compute convergence order between two grid levels
 *
 * p = log(E_coarse / E_fine) / log(dx_coarse / dx_fine)
 */
float computeConvergenceOrder(float error_coarse, float error_fine,
                             float dx_coarse, float dx_fine) {
    return std::log(error_coarse / error_fine) / std::log(dx_coarse / dx_fine);
}

/**
 * @brief Main grid convergence study
 */
int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       GRID CONVERGENCE STUDY - THERMAL LBM SOLVER             ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    // Domain configuration
    const float domain_length = 200.0e-6f;  // 200 μm

    // Material properties (simple test case)
    const float rho = 1000.0f;   // kg/m³
    const float cp = 1000.0f;    // J/(kg·K)
    const float k = 1.0f;        // W/(m·K)
    const float alpha = k / (rho * cp);  // 1e-6 m²/s

    // Time parameters
    const float final_time = 5.0e-5f;  // 50 μs (shorter to reduce dispersion)

    // Gaussian parameters
    const float T0 = 300.0f;           // Background [K]
    const float A = 200.0f;            // Amplitude [K]
    const float sigma0 = 10.0e-6f;     // Initial width = 10 μm (smaller for better resolution)
    const float x_center = domain_length / 2.0f;

    GaussianDiffusion1D analytical(T0, A, sigma0, alpha, x_center);

    std::cout << "\n=== Problem Setup ===\n";
    std::cout << "Domain length: " << domain_length * 1e6 << " μm\n";
    std::cout << "Material: ρ=" << rho << " kg/m³, cp=" << cp << " J/(kg·K), k=" << k << " W/(m·K)\n";
    std::cout << "Thermal diffusivity: α=" << alpha * 1e6 << " mm²/s\n";
    std::cout << "Initial Gaussian: T0=" << T0 << " K, A=" << A << " K, σ₀=" << sigma0 * 1e6 << " μm\n";
    std::cout << "Final time: " << final_time * 1e6 << " μs\n";
    std::cout << "Expected peak at t=0: " << analytical.peak(0) << " K\n";
    std::cout << "Expected peak at t=final: " << analytical.peak(final_time) << " K\n";

    // Define grid resolutions
    std::vector<GridConfig> grids;
    grids.emplace_back(25, domain_length, "Coarse");
    grids.emplace_back(50, domain_length, "Medium");
    grids.emplace_back(100, domain_length, "Fine");
    grids.emplace_back(200, domain_length, "Finest");

    // Run simulations on all grids
    std::vector<float> errors;
    std::vector<float> dx_values;

    for (const auto& grid : grids) {
        float error = runGridTest(grid, analytical, final_time, alpha, rho, cp);
        errors.push_back(error);
        dx_values.push_back(grid.dx);
    }

    // Analyze convergence
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    CONVERGENCE ANALYSIS                       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    std::cout << "\n=== Grid Refinement Results ===\n";
    std::cout << std::setw(12) << "Grid"
              << std::setw(12) << "nx"
              << std::setw(15) << "dx [μm]"
              << std::setw(18) << "L2 Error [%]"
              << std::setw(18) << "Order p\n";
    std::cout << std::string(75, '-') << "\n";

    for (size_t i = 0; i < grids.size(); ++i) {
        std::cout << std::setw(12) << grids[i].name
                  << std::setw(12) << grids[i].nx
                  << std::setw(15) << std::fixed << std::setprecision(2) << dx_values[i] * 1e6
                  << std::setw(18) << std::setprecision(4) << errors[i] * 100.0f;

        if (i > 0) {
            float order = computeConvergenceOrder(errors[i-1], errors[i], dx_values[i-1], dx_values[i]);
            std::cout << std::setw(18) << std::setprecision(2) << order;
        } else {
            std::cout << std::setw(18) << "-";
        }
        std::cout << "\n";
    }

    // Compute average convergence order (excluding first comparison for stability)
    float avg_order = 0.0f;
    int count = 0;
    for (size_t i = 2; i < errors.size(); ++i) {
        float order = computeConvergenceOrder(errors[i-1], errors[i], dx_values[i-1], dx_values[i]);
        avg_order += order;
        count++;
    }
    if (count > 0) {
        avg_order /= count;
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Average convergence order (refined grids): " << std::fixed << std::setprecision(2) << avg_order << "\n";
    std::cout << "Finest grid error: " << std::setprecision(4) << errors.back() * 100.0f << "%\n";

    // Verify monotonic decrease
    bool monotonic = true;
    for (size_t i = 1; i < errors.size(); ++i) {
        if (errors[i] >= errors[i-1]) {
            monotonic = false;
            break;
        }
    }
    std::cout << "Error decreases monotonically: " << (monotonic ? "YES" : "NO") << "\n";

    // Acceptance criteria
    std::cout << "\n=== Acceptance Criteria ===\n";

    bool order_ok = (avg_order >= 1.8f && avg_order <= 2.2f);
    std::cout << "✓ Convergence order 1.8-2.2: " << (order_ok ? "PASS" : "FAIL")
              << " (actual: " << avg_order << ")\n";

    bool error_ok = (errors.back() < 0.01f);  // < 1%
    std::cout << "✓ Finest grid error < 1%: " << (error_ok ? "PASS" : "FAIL")
              << " (actual: " << errors.back() * 100.0f << "%)\n";

    std::cout << "✓ Monotonic error decrease: " << (monotonic ? "PASS" : "FAIL") << "\n";

    // Overall verdict
    bool all_pass = order_ok && error_ok && monotonic;

    std::cout << "\n";
    if (all_pass) {
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    ✓ TEST PASSED                              ║\n";
        std::cout << "║      Grid independence verified (second-order convergence)    ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
        return 0;
    } else {
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    ✗ TEST FAILED                              ║\n";
        std::cout << "║      Grid independence criteria not met                       ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
        return 1;
    }
}
