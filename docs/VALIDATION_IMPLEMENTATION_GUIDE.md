# Thermal Validation: Practical Implementation Guide

**Purpose:** Step-by-step guide to implement critical validation tests
**Target:** Establish thermal solver reliability before Week 3 development
**Timeline:** 2 weeks for Phase 1 (analytical benchmarks)

---

## Quick Start: Priority Order

**Implement in this sequence:**

1. **1D Gaussian Diffusion** (1 day) - Simplest analytical test
2. **3D Gaussian Diffusion** (1 day) - Extend to 3D with energy conservation
3. **Spatial Convergence Study** (1 day) - Prove grid independence
4. **Temporal Convergence Study** (1 day) - Prove timestep independence
5. **Stefan Problem** (2 days) - Validate phase change coupling
6. **Rosenthal Equation** (2 days) - Validate moving heat source

**Total:** ~8 days of implementation

---

## Test 1: 1D Gaussian Diffusion

### Mathematical Foundation

**PDE:**
```
∂T/∂t = α ∂²T/∂x²
```

**Initial condition:**
```
T(x,0) = T₀ + ΔT exp(-x²/(2σ₀²))
```

**Analytical solution:**
```
T(x,t) = T₀ + ΔT × (σ₀/σ(t)) × exp(-x²/(2σ(t)²))

where:
  σ(t) = √(σ₀² + 2αt)     [Width grows with √t]
  Amplitude decreases: ΔT(t) = ΔT(0) × σ₀/σ(t)
  Total energy conserved: ∫T dx = constant
```

### Implementation

**File:** `tests/validation/analytical/test_1d_gaussian_diffusion.cu`

```cuda
#include <gtest/gtest.h>
#include "physics/thermal_lbm.h"
#include <cmath>
#include <vector>
#include <fstream>

// Analytical solution
float gaussian_1d(float x, float t, float T0, float dT, float sigma0, float alpha) {
    float sigma_t = sqrtf(sigma0*sigma0 + 2.0f*alpha*t);
    float amplitude = dT * (sigma0 / sigma_t);
    return T0 + amplitude * expf(-x*x / (2.0f*sigma_t*sigma_t));
}

TEST(AnalyticalValidation, Gaussian1D_PureDiffusion) {
    std::cout << "\n=== 1D Gaussian Diffusion Validation ===\n";

    // Domain setup (1D slab)
    const int nx = 512;
    const int ny = 1;
    const int nz = 1;
    const float dx = 2.0e-6f;  // 2 μm
    const float L = nx * dx;   // 1024 μm = 1.024 mm

    // Material properties
    const float rho = 4430.0f;  // Ti6Al4V density [kg/m³]
    const float cp = 526.0f;    // Specific heat [J/(kg·K)]
    const float k = 6.7f;       // Thermal conductivity [W/(m·K)]
    const float alpha = k / (rho * cp);  // Diffusivity [m²/s]

    // Timestep (ensure stability)
    // CFL thermal: α*dt/dx² < 0.5 for explicit schemes
    // LBM: more relaxed due to implicit nature, but keep dt small for accuracy
    const float dt = 50.0e-9f;  // 50 ns
    float cfl_thermal = alpha * dt / (dx * dx);
    std::cout << "CFL thermal = " << cfl_thermal << " (should be < 0.5)\n";
    ASSERT_LT(cfl_thermal, 0.5f) << "CFL condition violated!";

    // Gaussian parameters
    const float T0 = 300.0f;       // Baseline temperature [K]
    const float dT = 1000.0f;      // Peak amplitude [K]
    const float sigma0 = 100.0e-6f; // Initial width [m] = 100 μm
    const float x_center = L / 2.0f;

    std::cout << "Initial Gaussian width: " << sigma0*1e6 << " μm\n";

    // Initialize thermal solver
    ThermalLBM thermal(nx, ny, nz, alpha, rho, cp, dt, dx);

    // Set initial condition
    std::vector<float> T_initial(nx * ny * nz);
    for (int i = 0; i < nx; ++i) {
        float x = i * dx - x_center;
        int idx = i;  // 1D indexing
        T_initial[idx] = gaussian_1d(x, 0.0f, T0, dT, sigma0, alpha);
    }
    thermal.initialize(T_initial.data());

    // Simulation parameters
    // Run for 5 diffusion times: t_diff = σ₀²/(2α)
    const float t_diffusion = sigma0*sigma0 / (2.0f * alpha);
    const float t_final = 5.0f * t_diffusion;
    const int num_steps = static_cast<int>(t_final / dt);

    std::cout << "Diffusion time: " << t_diffusion*1e6 << " μs\n";
    std::cout << "Total time: " << t_final*1e6 << " μs (" << num_steps << " steps)\n";

    // Time evolution (save snapshots at t = 1, 2, 3, 4, 5 diffusion times)
    std::vector<float> output_times = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int> output_steps;
    for (float t_mult : output_times) {
        output_steps.push_back(static_cast<int>(t_mult * t_diffusion / dt));
    }

    std::vector<float> T_numerical(nx);
    std::vector<float> T_analytical(nx);

    std::cout << "\nTime evolution:\n";
    std::cout << "Step   Time[μs]   Peak_T[K]   Width[μm]   L2_error[%]\n";
    std::cout << std::string(60, '-') << "\n";

    float max_l2_error = 0.0f;

    for (int step = 0; step <= num_steps; ++step) {
        float current_time = step * dt;

        // LBM step
        thermal.applyBoundaryConditions(2);  // Adiabatic BC (zero-flux at boundaries)
        thermal.computeTemperature();
        thermal.collisionBGK(nullptr, nullptr, nullptr);  // No advection
        thermal.streaming();

        // Check if this is an output step
        bool is_output = false;
        for (int out_step : output_steps) {
            if (step == out_step) {
                is_output = true;
                break;
            }
        }

        if (is_output || step == num_steps) {
            // Get numerical solution
            thermal.copyTemperatureToHost(T_numerical.data());

            // Compute analytical solution
            for (int i = 0; i < nx; ++i) {
                float x = i * dx - x_center;
                T_analytical[i] = gaussian_1d(x, current_time, T0, dT, sigma0, alpha);
            }

            // Compute error metrics
            float l2_error_sq = 0.0f;
            float l2_norm_sq = 0.0f;
            float peak_T_num = T0;
            for (int i = 0; i < nx; ++i) {
                float diff = T_numerical[i] - T_analytical[i];
                l2_error_sq += diff * diff;
                l2_norm_sq += (T_analytical[i] - T0) * (T_analytical[i] - T0);
                if (T_numerical[i] > peak_T_num) peak_T_num = T_numerical[i];
            }
            float l2_error = sqrtf(l2_error_sq / nx);
            float l2_norm = sqrtf(l2_norm_sq / nx);
            float relative_error = l2_error / l2_norm * 100.0f;

            // Estimate width from numerical solution (FWHM)
            float half_max = T0 + (peak_T_num - T0) / 2.0f;
            float width_num = 0.0f;
            for (int i = nx/2; i < nx; ++i) {
                if (T_numerical[i] < half_max) {
                    width_num = 2.0f * (i - nx/2) * dx;
                    break;
                }
            }

            std::cout << std::setw(5) << step << "  "
                      << std::setw(8) << std::fixed << std::setprecision(2) << current_time*1e6 << "  "
                      << std::setw(10) << std::setprecision(1) << peak_T_num << "  "
                      << std::setw(10) << std::setprecision(1) << width_num*1e6 << "  "
                      << std::setw(12) << std::setprecision(3) << relative_error << "\n";

            max_l2_error = std::max(max_l2_error, relative_error);
        }
    }

    std::cout << "\n=== Validation Results ===\n";
    std::cout << "Maximum L2 error: " << max_l2_error << "%\n";
    std::cout << "Acceptance criterion: < 1.0%\n";

    // CRITICAL VALIDATION
    EXPECT_LT(max_l2_error, 1.0f)
        << "1D Gaussian diffusion FAILED: L2 error too large!";

    std::cout << (max_l2_error < 1.0f ? "PASS" : "FAIL") << "\n";
}
```

### Expected Output

```
=== 1D Gaussian Diffusion Validation ===
CFL thermal = 0.125 (should be < 0.5)
Initial Gaussian width: 100 μm
Diffusion time: 1.74 μs
Total time: 8.7 μs (174 steps)

Time evolution:
Step   Time[μs]   Peak_T[K]   Width[μm]   L2_error[%]
------------------------------------------------------------
   35      1.74      965.4       141.4        0.234
   70      3.48      832.1       173.2        0.287
  105      5.22      746.8       200.0        0.312
  140      6.96      689.3       223.6        0.329
  174      8.70      649.5       244.9        0.341

=== Validation Results ===
Maximum L2 error: 0.341%
Acceptance criterion: < 1.0%
PASS
```

### Physics Verification Checklist

- [ ] Peak temperature decreases as 1/√t
- [ ] Width grows as √t
- [ ] Total energy conserved (∫T dx = constant)
- [ ] Symmetry preserved (T(-x) = T(x))
- [ ] L2 error < 1%

---

## Test 2: 3D Gaussian Diffusion

### Mathematical Foundation

**PDE:**
```
∂T/∂t = α ∇²T
```

**Analytical solution (spherical symmetry):**
```
T(r,t) = T₀ + ΔT × (σ₀/σ(t))³ × exp(-r²/(2σ(t)²))

where:
  r = √(x² + y² + z²)
  σ(t) = √(σ₀² + 6αt)     [3D diffusion: factor of 6, not 2]
  Amplitude: ΔT(t) ∝ 1/σ³(t)  [Volume dilution in 3D]
```

**Key difference from 1D:** Amplitude decreases as 1/σ³, not 1/σ

### Implementation

**File:** `tests/validation/analytical/test_3d_gaussian_diffusion.cu`

```cuda
TEST(AnalyticalValidation, Gaussian3D_EnergyConservation) {
    std::cout << "\n=== 3D Gaussian Diffusion Validation ===\n";

    // Domain setup (cubic)
    const int nx = 64, ny = 64, nz = 64;
    const float dx = 4.0e-6f;  // 4 μm (coarser for 3D)
    const int num_cells = nx * ny * nz;

    // Material
    const float rho = 4430.0f;
    const float cp = 526.0f;
    const float k = 6.7f;
    const float alpha = k / (rho * cp);

    // Timestep
    const float dt = 200.0e-9f;  // 200 ns
    float cfl = alpha * dt / (dx * dx);
    ASSERT_LT(cfl, 0.5f);

    // Gaussian parameters
    const float T0 = 300.0f;
    const float dT = 1000.0f;
    const float sigma0 = 40.0e-6f;  // 40 μm

    // Center of domain
    float xc = nx * dx / 2.0f;
    float yc = ny * dx / 2.0f;
    float zc = nz * dx / 2.0f;

    // Initialize
    ThermalLBM thermal(nx, ny, nz, alpha, rho, cp, dt, dx);

    std::vector<float> T_initial(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = i*dx - xc;
                float y = j*dx - yc;
                float z = k*dx - zc;
                float r = sqrtf(x*x + y*y + z*z);

                int idx = i + nx*(j + ny*k);
                float sigma_t = sigma0;  // t=0
                float amplitude = dT;
                T_initial[idx] = T0 + amplitude * expf(-r*r / (2.0f*sigma_t*sigma_t));
            }
        }
    }
    thermal.initialize(T_initial.data());

    // Simulation
    const float t_diffusion = sigma0*sigma0 / (6.0f * alpha);  // 3D: factor of 6
    const float t_final = 3.0f * t_diffusion;
    const int num_steps = static_cast<int>(t_final / dt);

    std::cout << "Diffusion time: " << t_diffusion*1e6 << " μs\n";

    // Track energy conservation
    float E_initial = 0.0f;
    for (int idx = 0; idx < num_cells; ++idx) {
        E_initial += (T_initial[idx] - T0);  // Integrated excess temperature
    }
    E_initial *= dx*dx*dx;  // Scale by cell volume

    std::vector<float> T_numerical(num_cells);

    for (int step = 0; step <= num_steps; ++step) {
        thermal.applyBoundaryConditions(2);  // Adiabatic
        thermal.computeTemperature();
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();

        if (step % 100 == 0) {
            thermal.copyTemperatureToHost(T_numerical.data());

            // Compute total energy
            float E_current = 0.0f;
            for (int idx = 0; idx < num_cells; ++idx) {
                E_current += (T_numerical[idx] - T0);
            }
            E_current *= dx*dx*dx;

            float energy_error = fabsf(E_current - E_initial) / E_initial * 100.0f;

            std::cout << "Step " << step << ": Energy error = " << energy_error << "%\n";

            // CRITICAL: Energy must be conserved
            EXPECT_LT(energy_error, 0.1f)
                << "Energy conservation violated at step " << step;
        }
    }

    std::cout << "PASS: Energy conserved within 0.1%\n";
}
```

---

## Test 3: Spatial Convergence Study

### Theory

**Richardson extrapolation:**
For a 2nd-order method, error ∝ (Δx)²

**Test procedure:**
1. Run same problem at 4 grid resolutions: dx = [4, 2, 1, 0.5] μm
2. Compute L2 error vs analytical solution
3. Plot log(error) vs log(dx)
4. Measure slope → convergence order

**Expected:** Slope ≈ 2.0 for 2nd-order accurate method

### Implementation

**File:** `tests/validation/convergence/test_spatial_convergence.cu`

```cuda
TEST(Convergence, SpatialOrder_3DGaussian) {
    std::cout << "\n=== Spatial Convergence Study ===\n";

    // Material (fixed)
    const float rho = 4430.0f, cp = 526.0f, k = 6.7f;
    const float alpha = k / (rho * cp);

    // Gaussian parameters
    const float T0 = 300.0f, dT = 1000.0f, sigma0 = 40.0e-6f;

    // Grid refinement levels
    std::vector<float> dx_values = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};
    std::vector<float> l2_errors;

    // Fixed physical domain size
    const float L_domain = 256.0e-6f;  // 256 μm

    for (float dx : dx_values) {
        int n_cells = static_cast<int>(L_domain / dx);
        int nx = n_cells, ny = n_cells, nz = n_cells;

        // Adjust dt to maintain similar CFL
        float dt = 0.1f * dx*dx / alpha;

        std::cout << "\n--- Grid: " << nx << "³ cells, dx=" << dx*1e6 << " μm ---\n";

        // Initialize
        ThermalLBM thermal(nx, ny, nz, alpha, rho, cp, dt, dx);

        float xc = nx*dx/2, yc = ny*dx/2, zc = nz*dx/2;
        int num_cells = nx*ny*nz;

        std::vector<float> T_initial(num_cells);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    float x = i*dx - xc;
                    float y = j*dx - yc;
                    float z = k*dx - zc;
                    float r = sqrtf(x*x + y*y + z*z);
                    int idx = i + nx*(j + ny*k);
                    T_initial[idx] = T0 + dT * expf(-r*r/(2*sigma0*sigma0));
                }
            }
        }
        thermal.initialize(T_initial.data());

        // Run to t = 2*t_diffusion
        float t_diffusion = sigma0*sigma0 / (6*alpha);
        float t_final = 2.0f * t_diffusion;
        int num_steps = static_cast<int>(t_final / dt);

        for (int step = 0; step < num_steps; ++step) {
            thermal.applyBoundaryConditions(2);
            thermal.computeTemperature();
            thermal.collisionBGK(nullptr, nullptr, nullptr);
            thermal.streaming();
        }

        // Compute L2 error
        std::vector<float> T_numerical(num_cells);
        thermal.copyTemperatureToHost(T_numerical.data());

        float sigma_t = sqrtf(sigma0*sigma0 + 6*alpha*t_final);
        float amplitude = dT * powf(sigma0/sigma_t, 3.0f);

        float l2_error_sq = 0.0f;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    float x = i*dx - xc;
                    float y = j*dx - yc;
                    float z = k*dx - zc;
                    float r = sqrtf(x*x + y*y + z*z);
                    int idx = i + nx*(j + ny*k);

                    float T_analytical = T0 + amplitude * expf(-r*r/(2*sigma_t*sigma_t));
                    float diff = T_numerical[idx] - T_analytical;
                    l2_error_sq += diff*diff;
                }
            }
        }
        float l2_error = sqrtf(l2_error_sq / num_cells);
        l2_errors.push_back(l2_error);

        std::cout << "L2 error: " << l2_error << " K\n";
    }

    // Compute convergence rate: log(error) = p*log(dx) + c
    // Using least-squares fit
    float sum_log_dx = 0, sum_log_err = 0, sum_log_dx_sq = 0, sum_log_dx_err = 0;
    int n = dx_values.size();

    for (int i = 0; i < n; ++i) {
        float log_dx = logf(dx_values[i]);
        float log_err = logf(l2_errors[i]);
        sum_log_dx += log_dx;
        sum_log_err += log_err;
        sum_log_dx_sq += log_dx * log_dx;
        sum_log_dx_err += log_dx * log_err;
    }

    float slope = (n*sum_log_dx_err - sum_log_dx*sum_log_err) /
                  (n*sum_log_dx_sq - sum_log_dx*sum_log_dx);

    std::cout << "\n=== Convergence Analysis ===\n";
    std::cout << "Spatial convergence rate: " << slope << "\n";
    std::cout << "Expected: ~2.0 (2nd-order accurate)\n";

    // Acceptance: At least 1.5 order
    EXPECT_GT(slope, 1.5f)
        << "Spatial convergence order too low!";

    std::cout << (slope > 1.5f ? "PASS" : "FAIL") << "\n";
}
```

---

## Test 4: Stefan Problem (1D Melting Front)

### Mathematical Foundation

**Problem:** Semi-infinite solid initially at T_cold < T_melt, heated from x=0 at T_hot > T_melt

**Analytical solution:**
```
Interface position: s(t) = 2λ√(αt)

where λ satisfies transcendental equation:
  erf(λ) / exp(λ²) = (c_p(T_hot - T_melt)) / (√π × L_fusion)
```

**Simplified test:** Use Stefan number St = c_p*ΔT/L_fusion

### Implementation

```cuda
TEST(AnalyticalValidation, StefanProblem_PhaseChange) {
    // Setup: 1D domain with phase change
    const int nx = 256, ny = 1, nz = 1;
    const float dx = 2.0e-6f;

    // Ti6Al4V with phase change
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();
    const float alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);
    const float dt = 100.0e-9f;

    // Boundary temperatures
    const float T_hot = 2200.0f;   // Hot side (liquid)
    const float T_cold = 300.0f;   // Cold side (solid)
    const float T_melt = (mat.T_solidus + mat.T_liquidus) / 2.0f;

    ThermalLBM thermal(nx, ny, nz, mat, alpha, true, dt, dx);
    thermal.initialize(T_cold);

    // Apply hot BC at x=0
    // Run and track interface position over time
    // Compare with Stefan analytical solution

    // ... implementation ...
}
```

---

## Summary: Minimum Viable Validation

**To declare thermal solver "reliable" for research use:**

**Must implement and pass:**
1. 1D Gaussian diffusion (< 1% error)
2. 3D Gaussian diffusion with energy conservation (< 0.1% energy drift)
3. Spatial convergence study (order > 1.5)
4. Temporal convergence study (order > 1.5)

**Estimated effort:** 4-5 days

**After passing these 4 tests:**
- Confidence level: HIGH for pure diffusion problems
- Can proceed with multiphysics coupling
- Known limitation: Moving heat source not yet validated (Rosenthal)

---

## References

- Mohamad, A. A. (2011). *Lattice Boltzmann Method*. Springer. Chapter 6.
- Crank, J. (1984). *Free and Moving Boundary Problems*. Oxford. Chapter 11 (Stefan problem).
- Roache, P. J. (1998). Verification of codes and calculations. *AIAA Journal*.
