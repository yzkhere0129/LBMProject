# Test Implementation Guide: Technical Reference

## Overview

This guide provides implementation details for developers writing the validation tests specified in `TEST_SUITE_DESIGN.md`. It includes code patterns, numerical analysis techniques, common pitfalls, and GPU-specific considerations.

---

## 1. Analytical Solution Implementation Patterns

### 1.1 Computing Error Metrics

**L2 Relative Error (Preferred for field comparisons)**:
```cpp
/**
 * Compute L2 relative error between numerical and analytical solutions
 * L2_rel = sqrt(sum((u_num - u_ana)^2)) / sqrt(sum(u_ana^2))
 */
float computeL2Error(const std::vector<float>& numerical,
                     const std::vector<float>& analytical,
                     int start_idx, int end_idx)
{
    double sum_sq_error = 0.0;
    double sum_sq_analytical = 0.0;

    for (int i = start_idx; i < end_idx; ++i) {
        double error = numerical[i] - analytical[i];
        sum_sq_error += error * error;
        sum_sq_analytical += analytical[i] * analytical[i];
    }

    return std::sqrt(sum_sq_error / (sum_sq_analytical + 1e-15));
}
```

**L-infinity Error (Maximum absolute error)**:
```cpp
/**
 * Compute maximum absolute error and its location
 */
struct LinfError {
    float max_error;
    int location;
    float numerical_value;
    float analytical_value;
};

LinfError computeLinfError(const std::vector<float>& numerical,
                           const std::vector<float>& analytical,
                           int start_idx, int end_idx)
{
    LinfError result = {0.0f, -1, 0.0f, 0.0f};

    for (int i = start_idx; i < end_idx; ++i) {
        float error = std::abs(numerical[i] - analytical[i]);
        if (error > result.max_error) {
            result.max_error = error;
            result.location = i;
            result.numerical_value = numerical[i];
            result.analytical_value = analytical[i];
        }
    }

    return result;
}
```

**Average Relative Error**:
```cpp
/**
 * Compute mean of point-wise relative errors
 * Better for comparing profiles with varying magnitudes
 */
float computeAvgRelativeError(const std::vector<float>& numerical,
                              const std::vector<float>& analytical,
                              int start_idx, int end_idx)
{
    double sum_rel_error = 0.0;
    int count = 0;

    for (int i = start_idx; i < end_idx; ++i) {
        float rel_error = std::abs(numerical[i] - analytical[i])
                        / (std::abs(analytical[i]) + 1e-10f);
        sum_rel_error += rel_error;
        count++;
    }

    return static_cast<float>(sum_rel_error / count);
}
```

---

### 1.2 Profile Extraction and Averaging

**Extracting 1D Profile from 3D Data**:
```cpp
/**
 * Extract velocity profile along y-direction by averaging over x-z planes
 * Useful for: Poiseuille flow, Couette flow, 1D heat conduction
 */
std::vector<float> extract1DProfile(const float* field_3d,
                                    int nx, int ny, int nz,
                                    int direction)  // 0=x, 1=y, 2=z
{
    std::vector<float> profile;

    if (direction == 1) {  // y-direction (most common)
        profile.resize(ny, 0.0f);
        for (int j = 0; j < ny; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < nx; ++i) {
                for (int k = 0; k < nz; ++k) {
                    int idx = i + j * nx + k * nx * ny;
                    sum += field_3d[idx];
                }
            }
            profile[j] = sum / (nx * nz);
        }
    }
    // Similar for other directions...

    return profile;
}
```

**Excluding Boundary Layers**:
```cpp
/**
 * Skip boundary regions where LBM accuracy is reduced
 * Typically exclude 1-2 cells from walls
 */
void excludeBoundaries(int& start_idx, int& end_idx, int n_total, int n_boundary = 2) {
    start_idx = n_boundary;
    end_idx = n_total - n_boundary;
}
```

---

### 1.3 Analytical Solution Implementations

**Poiseuille Flow Velocity Profile**:
```cpp
/**
 * Analytical solution: u(y) = -dp/dx * y*(H-y) / (2*mu)
 */
float poiseuilleVelocity(float y, float H, float dp_dx, float nu, float rho) {
    float mu = nu * rho;  // Dynamic viscosity
    return -dp_dx * y * (H - y) / (2.0f * mu);
}

/**
 * Maximum velocity at centerline
 */
float poiseuilleMaxVelocity(float H, float dp_dx, float nu, float rho) {
    float mu = nu * rho;
    return -dp_dx * H * H / (8.0f * mu);
}

/**
 * Volumetric flow rate per unit depth
 */
float poiseuilleFlowRate(float H, float dp_dx, float nu, float rho) {
    float u_max = poiseuilleMaxVelocity(H, dp_dx, nu, rho);
    return (2.0f / 3.0f) * H * u_max;
}
```

**Couette Flow Velocity Profile**:
```cpp
/**
 * Analytical solution: u(y) = U * (y / H)
 */
float couetteVelocity(float y, float H, float U_top) {
    return U_top * (y / H);
}

/**
 * Wall shear stress
 */
float couetteShearStress(float H, float U_top, float nu, float rho) {
    float mu = nu * rho;
    return mu * U_top / H;
}
```

**1D Heat Conduction (Steady State)**:
```cpp
/**
 * Steady state: T(x) = T_left + (T_right - T_left) * (x / L)
 */
float heatConductionSteadyState(float x, float L, float T_left, float T_right) {
    return T_left + (T_right - T_left) * (x / L);
}

/**
 * Heat flux (constant in steady state)
 */
float heatFlux(float L, float T_left, float T_right, float k) {
    return -k * (T_right - T_left) / L;
}
```

**Taylor-Green Vortex (2D)**:
```cpp
/**
 * Analytical solution for decaying 2D vortex
 */
struct TaylorGreenSolution {
    float u, v, p;
};

TaylorGreenSolution taylorGreenVortex(float x, float y, float t,
                                      float U0, float k, float nu, float rho)
{
    TaylorGreenSolution sol;

    float decay = std::exp(-2.0f * k * k * nu * t);

    sol.u = -U0 * std::cos(k * x) * std::sin(k * y) * decay;
    sol.v =  U0 * std::sin(k * x) * std::cos(k * y) * decay;

    float p0 = 1.0f;  // Reference pressure
    float pressure_term = (rho * U0 * U0 / 4.0f)
                        * (std::cos(2.0f * k * x) + std::cos(2.0f * k * y))
                        * std::exp(-4.0f * k * k * nu * t);
    sol.p = p0 - pressure_term;

    return sol;
}

/**
 * Kinetic energy (should decay exponentially)
 */
float taylorGreenKineticEnergy(const std::vector<float>& ux,
                               const std::vector<float>& uy,
                               float rho)
{
    double E = 0.0;
    for (size_t i = 0; i < ux.size(); ++i) {
        E += ux[i] * ux[i] + uy[i] * uy[i];
    }
    return 0.5f * rho * static_cast<float>(E / ux.size());
}
```

---

## 2. Conservation Law Testing Patterns

### 2.1 Mass Conservation

**Global Mass Calculation**:
```cpp
/**
 * Compute total mass in domain
 * For single-phase: m = sum(rho * dV)
 * For multi-phase: m = sum(rho_mixture * dV)
 */
double computeTotalMass(const float* d_rho,
                        const float* d_vof,  // nullptr if single-phase
                        int num_cells,
                        float cell_volume,
                        float rho_liquid,
                        float rho_gas)
{
    std::vector<float> h_rho(num_cells);
    cudaMemcpy(h_rho.data(), d_rho, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    double total_mass = 0.0;

    if (d_vof == nullptr) {
        // Single phase
        for (int i = 0; i < num_cells; ++i) {
            total_mass += h_rho[i] * cell_volume;
        }
    } else {
        // Multi-phase with VOF
        std::vector<float> h_vof(num_cells);
        cudaMemcpy(h_vof.data(), d_vof, num_cells * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_cells; ++i) {
            float f = h_vof[i];
            float rho_cell = rho_liquid * f + rho_gas * (1.0f - f);
            total_mass += rho_cell * cell_volume;
        }
    }

    return total_mass;
}
```

**Mass Conservation Check**:
```cpp
/**
 * Verify mass conservation over time
 */
struct MassConservationReport {
    double m_initial;
    double m_current;
    double m_evaporated;  // If applicable
    double relative_error;
    bool passed;
};

MassConservationReport checkMassConservation(double m_initial,
                                             double m_current,
                                             double m_evaporated,
                                             float tolerance = 1e-4f)
{
    MassConservationReport report;
    report.m_initial = m_initial;
    report.m_current = m_current;
    report.m_evaporated = m_evaporated;

    // Expected: m_current + m_evaporated = m_initial
    double expected_mass = m_initial;
    double actual_mass = m_current + m_evaporated;

    report.relative_error = std::abs(actual_mass - expected_mass) / expected_mass;
    report.passed = (report.relative_error < tolerance);

    return report;
}
```

---

### 2.2 Momentum Conservation

**Total Momentum Calculation**:
```cpp
/**
 * Compute total momentum: P = sum(rho * u * dV)
 */
struct Momentum3D {
    double Px, Py, Pz;
    double magnitude;
};

Momentum3D computeTotalMomentum(const float* d_rho,
                                const float* d_ux,
                                const float* d_uy,
                                const float* d_uz,
                                int num_cells,
                                float cell_volume)
{
    std::vector<float> h_rho(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);

    cudaMemcpy(h_rho.data(), d_rho, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux.data(), d_ux, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), d_uy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz.data(), d_uz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    Momentum3D P = {0.0, 0.0, 0.0, 0.0};

    for (int i = 0; i < num_cells; ++i) {
        double mass = h_rho[i] * cell_volume;
        P.Px += mass * h_ux[i];
        P.Py += mass * h_uy[i];
        P.Pz += mass * h_uz[i];
    }

    P.magnitude = std::sqrt(P.Px*P.Px + P.Py*P.Py + P.Pz*P.Pz);

    return P;
}
```

**Momentum Balance Verification**:
```cpp
/**
 * Check: dP/dt = F_external
 */
bool verifyMomentumBalance(Momentum3D P_current,
                          Momentum3D P_previous,
                          float dt,
                          Momentum3D F_external,
                          float tolerance = 0.02f)
{
    // Compute dP/dt
    double dPx_dt = (P_current.Px - P_previous.Px) / dt;
    double dPy_dt = (P_current.Py - P_previous.Py) / dt;
    double dPz_dt = (P_current.Pz - P_previous.Pz) / dt;

    // Compare with external force
    double error_x = std::abs(dPx_dt - F_external.Px) / std::abs(F_external.Px + 1e-10);
    double error_y = std::abs(dPy_dt - F_external.Py) / std::abs(F_external.Py + 1e-10);
    double error_z = std::abs(dPz_dt - F_external.Pz) / std::abs(F_external.Pz + 1e-10);

    std::cout << "Momentum balance check:\n";
    std::cout << "  x: dPx/dt = " << dPx_dt << ", Fx = " << F_external.Px
              << ", error = " << error_x * 100 << "%\n";
    std::cout << "  y: dPy/dt = " << dPy_dt << ", Fy = " << F_external.Py
              << ", error = " << error_y * 100 << "%\n";
    std::cout << "  z: dPz/dt = " << dPz_dt << ", Fz = " << F_external.Pz
              << ", error = " << error_z * 100 << "%\n";

    return (error_x < tolerance && error_y < tolerance && error_z < tolerance);
}
```

---

### 2.3 Energy Conservation

**Total Energy Calculation (Thermal)**:
```cpp
/**
 * E_total = E_sensible + E_latent
 * E_sensible = integral[rho * cp * (T - T_ref) dV]
 * E_latent = integral[rho * L_fusion * f_liquid dV]
 */
struct EnergyBreakdown {
    double E_sensible;
    double E_latent;
    double E_total;
    double E_kinetic;  // Optional for coupled simulations
};

EnergyBreakdown computeTotalEnergy(const float* d_T,
                                   const float* d_fl,
                                   const MaterialProperties& mat,
                                   int num_cells,
                                   float cell_volume,
                                   float T_ref = 300.0f)
{
    std::vector<float> h_T(num_cells);
    std::vector<float> h_fl(num_cells);

    cudaMemcpy(h_T.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fl.data(), d_fl, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    EnergyBreakdown energy = {0.0, 0.0, 0.0, 0.0};

    for (int i = 0; i < num_cells; ++i) {
        float T = h_T[i];
        float fl = h_fl[i];

        float rho = mat.getDensity(T);
        float cp = mat.getSpecificHeat(T);

        // Sensible heat
        energy.E_sensible += rho * cp * (T - T_ref) * cell_volume;

        // Latent heat
        energy.E_latent += rho * mat.L_fusion * fl * cell_volume;
    }

    energy.E_total = energy.E_sensible + energy.E_latent;

    return energy;
}
```

**Energy Balance Verification**:
```cpp
/**
 * Check: dE/dt = P_in - P_out
 */
struct EnergyBalance {
    double dE_dt;
    double P_laser;
    double P_evaporation;
    double P_radiation;
    double P_substrate;
    double P_net_expected;
    double balance_error;
    bool passed;
};

EnergyBalance checkEnergyBalance(double E_current,
                                double E_previous,
                                float dt,
                                float P_laser,
                                float P_evap,
                                float P_rad,
                                float P_sub,
                                float tolerance = 0.05f)
{
    EnergyBalance balance;

    balance.dE_dt = (E_current - E_previous) / dt;
    balance.P_laser = P_laser;
    balance.P_evaporation = P_evap;
    balance.P_radiation = P_rad;
    balance.P_substrate = P_sub;

    // Expected: dE/dt = P_in - P_out
    balance.P_net_expected = P_laser - P_evap - P_rad - P_sub;

    balance.balance_error = std::abs(balance.dE_dt - balance.P_net_expected)
                          / std::abs(balance.P_laser + 1e-10);

    balance.passed = (balance.balance_error < tolerance);

    return balance;
}
```

---

## 3. Convergence Study Implementation

### 3.1 Grid Convergence Study

**Framework for Multiple Grid Sizes**:
```cpp
/**
 * Run convergence study at multiple grid resolutions
 */
struct ConvergenceStudyResult {
    int resolution;      // Grid resolution (e.g., ny)
    float h;            // Grid spacing (physical or lattice)
    float error_L2;     // L2 error at this resolution
    float error_max;    // Maximum error
    float runtime_sec;  // Computational time
};

std::vector<ConvergenceStudyResult> runGridConvergenceStudy(
    std::function<float(int)> run_simulation,  // Returns L2 error for given ny
    std::vector<int> resolutions)
{
    std::vector<ConvergenceStudyResult> results;

    for (int ny : resolutions) {
        std::cout << "\n=== Running grid convergence: ny = " << ny << " ===\n";

        auto t_start = std::chrono::high_resolution_clock::now();

        float error = run_simulation(ny);

        auto t_end = std::chrono::high_resolution_clock::now();
        float runtime = std::chrono::duration<float>(t_end - t_start).count();

        ConvergenceStudyResult result;
        result.resolution = ny;
        result.h = 1.0f / ny;  // Normalized grid spacing
        result.error_L2 = error;
        result.runtime_sec = runtime;

        results.push_back(result);

        std::cout << "  h = " << result.h << ", error = " << error
                  << ", time = " << runtime << " s\n";
    }

    return results;
}
```

**Convergence Rate Calculation**:
```cpp
/**
 * Compute convergence order: p = log(e1/e2) / log(h1/h2)
 */
float computeConvergenceOrder(float error1, float error2,
                              float h1, float h2)
{
    return std::log(error1 / error2) / std::log(h1 / h2);
}

/**
 * Analyze convergence study results
 */
void analyzeConvergenceStudy(const std::vector<ConvergenceStudyResult>& results)
{
    std::cout << "\n=== Convergence Analysis ===\n";
    std::cout << "Grid     h        Error      Order\n";
    std::cout << "----------------------------------------\n";

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::setw(4) << results[i].resolution
                  << std::setw(10) << std::fixed << std::setprecision(5)
                  << results[i].h
                  << std::setw(12) << std::scientific << std::setprecision(3)
                  << results[i].error_L2;

        if (i > 0) {
            float order = computeConvergenceOrder(
                results[i-1].error_L2, results[i].error_L2,
                results[i-1].h, results[i].h
            );
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << order;
        } else {
            std::cout << std::setw(10) << " - ";
        }

        std::cout << "\n";
    }

    // Average convergence order (excluding first interval which may be noisy)
    if (results.size() >= 3) {
        float avg_order = 0.0f;
        for (size_t i = 2; i < results.size(); ++i) {
            avg_order += computeConvergenceOrder(
                results[i-1].error_L2, results[i].error_L2,
                results[i-1].h, results[i].h
            );
        }
        avg_order /= (results.size() - 2);
        std::cout << "\nAverage convergence order (excluding first interval): "
                  << avg_order << "\n";
    }
}
```

**Richardson Extrapolation**:
```cpp
/**
 * Estimate exact solution using Richardson extrapolation
 * u_exact ≈ (r^p * u_fine - u_coarse) / (r^p - 1)
 * where r = h_coarse / h_fine, p = convergence order
 */
float richardsonExtrapolation(float u_coarse, float u_fine,
                              float h_coarse, float h_fine,
                              float order)
{
    float r = h_coarse / h_fine;
    float r_p = std::pow(r, order);
    return (r_p * u_fine - u_coarse) / (r_p - 1.0f);
}
```

---

### 3.2 Temporal Convergence Study

**Multiple Time Step Sizes**:
```cpp
/**
 * Run temporal convergence study
 * Keep grid spacing constant, vary dt
 */
std::vector<ConvergenceStudyResult> runTemporalConvergenceStudy(
    std::function<float(float)> run_simulation,  // Returns error for given dt
    float dt_base,
    std::vector<int> refinement_factors)  // e.g., {1, 2, 4, 8}
{
    std::vector<ConvergenceStudyResult> results;

    for (int factor : refinement_factors) {
        float dt = dt_base / factor;
        std::cout << "\n=== Running temporal convergence: dt = " << dt << " ===\n";

        auto t_start = std::chrono::high_resolution_clock::now();
        float error = run_simulation(dt);
        auto t_end = std::chrono::high_resolution_clock::now();

        float runtime = std::chrono::duration<float>(t_end - t_start).count();

        ConvergenceStudyResult result;
        result.resolution = factor;
        result.h = dt;  // Use 'h' field for dt
        result.error_L2 = error;
        result.runtime_sec = runtime;

        results.push_back(result);
    }

    return results;
}
```

---

## 4. Dimensionless Number Testing

### 4.1 Reynolds Number Tests

**Computing Reynolds Number**:
```cpp
/**
 * Re = U * L / nu
 */
struct FlowParameters {
    float U;      // Characteristic velocity
    float L;      // Characteristic length
    float nu;     // Kinematic viscosity
    float Re;     // Reynolds number

    void computeReynolds() {
        Re = U * L / nu;
    }

    // Adjust parameters to achieve target Re while keeping L constant
    void setReynolds(float Re_target, float L_fixed) {
        Re = Re_target;
        L = L_fixed;
        // Choose U = 0.05 (typical for LBM stability)
        U = 0.05f;
        nu = U * L / Re;
    }
};
```

**Drag Coefficient Measurement**:
```cpp
/**
 * Measure drag force on obstacle and compute drag coefficient
 * Cd = F_drag / (0.5 * rho * U^2 * A)
 */
float computeDragCoefficient(float F_drag,
                             float rho,
                             float U,
                             float A_reference)
{
    return F_drag / (0.5f * rho * U * U * A_reference);
}

/**
 * Theoretical drag coefficient for sphere (Stokes + empirical)
 * Cd = 24/Re + 6/(1 + sqrt(Re)) + 0.4
 */
float theoreticalDragCoefficient(float Re)
{
    return 24.0f / Re + 6.0f / (1.0f + std::sqrt(Re)) + 0.4f;
}
```

---

### 4.2 Peclet Number Tests

**Computing Peclet Number**:
```cpp
/**
 * Pe = U * L / alpha
 * where alpha = k / (rho * cp) is thermal diffusivity
 */
struct ThermalFlowParameters {
    float U;      // Velocity
    float L;      // Length scale
    float alpha;  // Thermal diffusivity
    float Pe;     // Peclet number

    void computePeclet() {
        Pe = U * L / alpha;
    }

    // Set parameters for target Pe
    void setPeclet(float Pe_target, float L_fixed, float U_fixed) {
        Pe = Pe_target;
        L = L_fixed;
        U = U_fixed;
        alpha = U * L / Pe;
    }
};
```

**Nusselt Number Measurement**:
```cpp
/**
 * Nusselt number: Nu = h * L / k
 * where h is convective heat transfer coefficient
 *
 * For forced convection over sphere:
 * Nu = 2 + 0.6 * Re^0.5 * Pr^0.33 (empirical)
 */
float computeNusseltNumber(float q,        // Heat flux [W/m^2]
                          float deltaT,    // T_surface - T_infinity
                          float L,         // Characteristic length
                          float k)         // Thermal conductivity
{
    float h = q / deltaT;  // Heat transfer coefficient
    return h * L / k;
}

float theoreticalNusseltNumber(float Re, float Pr) {
    return 2.0f + 0.6f * std::pow(Re, 0.5f) * std::pow(Pr, 0.33f);
}
```

---

### 4.3 Mach Number Tests

**Mach Number and Compressibility**:
```cpp
/**
 * Ma = u / c_s
 * where c_s = 1/sqrt(3) for D3Q19 lattice
 */
float computeMachNumber(float u) {
    const float c_s = 1.0f / std::sqrt(3.0f);
    return u / c_s;
}

/**
 * Check density fluctuations scale as Ma^2
 */
struct CompressibilityCheck {
    float Ma;
    float rms_density_fluctuation;
    float theoretical_fluctuation;
    float error;
};

CompressibilityCheck checkCompressibility(const std::vector<float>& rho,
                                         float rho0,
                                         float Ma)
{
    CompressibilityCheck result;
    result.Ma = Ma;

    // Compute RMS density fluctuation
    double sum_sq = 0.0;
    for (float r : rho) {
        double delta_rho = (r - rho0) / rho0;
        sum_sq += delta_rho * delta_rho;
    }
    result.rms_density_fluctuation = std::sqrt(sum_sq / rho.size());

    // Theoretical: drho/rho ~ Ma^2
    result.theoretical_fluctuation = Ma * Ma;

    result.error = std::abs(result.rms_density_fluctuation - result.theoretical_fluctuation)
                 / result.theoretical_fluctuation;

    return result;
}
```

---

## 5. Stability Testing Patterns

### 5.1 Binary Search for Stability Boundary

**Finding Maximum Stable Parameter**:
```cpp
/**
 * Binary search to find maximum stable velocity, temperature gradient, etc.
 */
float findStabilityBoundary(
    std::function<bool(float)> is_stable,  // Returns true if parameter is stable
    float param_min,
    float param_max,
    float tolerance = 1e-3f)
{
    int max_iter = 50;
    int iter = 0;

    while (param_max - param_min > tolerance && iter < max_iter) {
        float param_mid = 0.5f * (param_min + param_max);

        std::cout << "Testing parameter = " << param_mid << "... ";

        if (is_stable(param_mid)) {
            std::cout << "STABLE\n";
            param_min = param_mid;  // Can go higher
        } else {
            std::cout << "UNSTABLE\n";
            param_max = param_mid;  // Too high
        }

        iter++;
    }

    float stable_param = param_min;
    std::cout << "\nStability boundary: " << stable_param << "\n";
    return stable_param;
}
```

**Stability Check Function**:
```cpp
/**
 * Check if simulation is stable
 * Returns false if: NaN detected, density fluctuations >10%, diverged
 */
bool checkSimulationStability(const float* d_rho,
                              const float* d_ux,
                              const float* d_uy,
                              const float* d_uz,
                              int num_cells,
                              float rho0)
{
    std::vector<float> h_rho(num_cells);
    cudaMemcpy(h_rho.data(), d_rho, num_cells * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Check for NaN
    for (float r : h_rho) {
        if (std::isnan(r) || std::isinf(r)) {
            std::cout << "NaN or Inf detected\n";
            return false;
        }
    }

    // Check density fluctuations
    float rho_min = *std::min_element(h_rho.begin(), h_rho.end());
    float rho_max = *std::max_element(h_rho.begin(), h_rho.end());
    float fluctuation = (rho_max - rho_min) / rho0;

    if (fluctuation > 0.1f) {
        std::cout << "Excessive density fluctuation: " << fluctuation * 100 << "%\n";
        return false;
    }

    return true;
}
```

---

## 6. GPU-Specific Considerations

### 6.1 Precision and Numerical Accuracy

**Double Precision for Error Accumulation**:
```cpp
/**
 * Use double precision for accumulating error metrics
 * Single precision summation can lose accuracy for large N
 */
double computeL2Error_GPU(const float* d_numerical,
                          const float* d_analytical,
                          int N)
{
    // Allocate device arrays for squared errors
    float *d_error_sq;
    cudaMalloc(&d_error_sq, N * sizeof(float));

    // Kernel to compute (num - ana)^2
    compute_squared_error_kernel<<<grid, block>>>(
        d_numerical, d_analytical, d_error_sq, N
    );

    // Use thrust::reduce with double precision
    double sum_sq_error = thrust::reduce(
        thrust::device_pointer_cast(d_error_sq),
        thrust::device_pointer_cast(d_error_sq + N),
        0.0,  // Initial value (double)
        thrust::plus<double>()
    );

    cudaFree(d_error_sq);

    return sum_sq_error;
}
```

---

### 6.2 Memory Transfer Optimization

**Pinned Memory for Faster Transfers**:
```cpp
/**
 * Use pinned (page-locked) memory for host arrays
 * ~2x faster memcpy than pageable memory
 */
class PinnedArray {
private:
    float* data_;
    size_t size_;

public:
    PinnedArray(size_t n) : size_(n) {
        cudaMallocHost(&data_, n * sizeof(float));
    }

    ~PinnedArray() {
        cudaFreeHost(data_);
    }

    float* data() { return data_; }
    size_t size() const { return size_; }
};

// Usage:
PinnedArray h_temp(num_cells);
cudaMemcpy(h_temp.data(), d_temp, num_cells * sizeof(float),
           cudaMemcpyDeviceToHost);
```

---

### 6.3 Asynchronous Operations for Testing

**Overlapping Computation and Data Transfer**:
```cpp
/**
 * Use CUDA streams to overlap operations during testing
 */
void runTestWithPipelining(int num_steps, int check_interval)
{
    // Create streams
    cudaStream_t stream_compute, stream_transfer;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_transfer);

    // Allocate pinned memory for transfers
    float* h_data_current;
    float* h_data_next;
    cudaMallocHost(&h_data_current, num_cells * sizeof(float));
    cudaMallocHost(&h_data_next, num_cells * sizeof(float));

    for (int step = 0; step < num_steps; ++step) {
        // Launch simulation kernel on compute stream
        simulation_kernel<<<grid, block, 0, stream_compute>>>(...);

        // If checkpoint, transfer data asynchronously
        if (step % check_interval == 0) {
            cudaMemcpyAsync(h_data_next, d_data, num_cells * sizeof(float),
                           cudaMemcpyDeviceToHost, stream_transfer);

            // Process previous data on CPU while transfer happens
            if (step > 0) {
                analyzeData(h_data_current);
            }

            // Swap buffers
            std::swap(h_data_current, h_data_next);
        }
    }

    cudaStreamSynchronize(stream_compute);
    cudaStreamSynchronize(stream_transfer);

    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
    cudaFreeHost(h_data_current);
    cudaFreeHost(h_data_next);
}
```

---

## 7. Test Output and Reporting

### 7.1 Structured Test Output

**Standard Output Format**:
```cpp
/**
 * Print test results in consistent format
 */
void printTestHeader(const std::string& test_name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << test_name << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void printTestResult(const std::string& metric_name,
                    float measured,
                    float expected,
                    float tolerance,
                    bool passed)
{
    std::cout << metric_name << ":\n";
    std::cout << "  Measured: " << measured << "\n";
    std::cout << "  Expected: " << expected << "\n";
    std::cout << "  Error:    " << std::abs(measured - expected) << " ("
              << std::abs((measured - expected) / expected) * 100 << "%)\n";
    std::cout << "  Tolerance: " << tolerance * 100 << "%\n";
    std::cout << "  Status:   " << (passed ? "PASS" : "FAIL") << "\n\n";
}

void printTestSummary(int n_passed, int n_total) {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "TEST SUMMARY: " << n_passed << " / " << n_total << " passed";
    if (n_passed == n_total) {
        std::cout << " (ALL PASS)\n";
    } else {
        std::cout << " (" << (n_total - n_passed) << " FAILED)\n";
    }
    std::cout << std::string(80, '=') << "\n\n";
}
```

---

### 7.2 Data Export for Post-Processing

**Export Profile Data**:
```cpp
/**
 * Save profile data to file for visualization and further analysis
 */
void exportProfile(const std::vector<float>& profile,
                  const std::string& filename,
                  const std::vector<float>& coordinates = {})
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file " << filename << "\n";
        return;
    }

    file << "# Profile data\n";
    if (coordinates.empty()) {
        file << "# index\tvalue\n";
        for (size_t i = 0; i < profile.size(); ++i) {
            file << i << "\t" << profile[i] << "\n";
        }
    } else {
        file << "# coordinate\tvalue\n";
        for (size_t i = 0; i < profile.size(); ++i) {
            file << coordinates[i] << "\t" << profile[i] << "\n";
        }
    }

    file.close();
    std::cout << "Profile data saved to " << filename << "\n";
}
```

**Export Comparison Data**:
```cpp
/**
 * Save numerical vs analytical comparison
 */
void exportComparison(const std::vector<float>& numerical,
                     const std::vector<float>& analytical,
                     const std::string& filename)
{
    std::ofstream file(filename);
    file << "# index\tnumerical\tanalytical\terror\n";

    for (size_t i = 0; i < numerical.size(); ++i) {
        float error = numerical[i] - analytical[i];
        file << i << "\t"
             << numerical[i] << "\t"
             << analytical[i] << "\t"
             << error << "\n";
    }

    file.close();
    std::cout << "Comparison data saved to " << filename << "\n";
}
```

---

## 8. Common Pitfalls and Solutions

### 8.1 Boundary Treatment

**Problem**: Errors concentrated at boundaries affect global error metrics.

**Solution**: Exclude boundary cells (typically 1-2 cells) when computing errors:
```cpp
// Exclude boundaries
int n_exclude = 2;
int start_idx = n_exclude;
int end_idx = ny - n_exclude;

float L2_error = computeL2Error(numerical, analytical, start_idx, end_idx);
```

---

### 8.2 Steady-State Detection

**Problem**: Declaring convergence too early leads to inaccurate comparisons.

**Solution**: Monitor convergence history:
```cpp
/**
 * Check if simulation has reached steady state
 */
bool isSteadyState(const std::vector<float>& field_history,
                  int check_length = 5,
                  float tolerance = 1e-5f)
{
    if (field_history.size() < check_length) return false;

    // Check if last 'check_length' values are nearly constant
    size_t n = field_history.size();
    float max_change = 0.0f;

    for (size_t i = n - check_length; i < n - 1; ++i) {
        float change = std::abs(field_history[i+1] - field_history[i]);
        max_change = std::max(max_change, change);
    }

    float avg_value = 0.0f;
    for (size_t i = n - check_length; i < n; ++i) {
        avg_value += field_history[i];
    }
    avg_value /= check_length;

    float relative_change = max_change / (std::abs(avg_value) + 1e-10f);

    return (relative_change < tolerance);
}
```

---

### 8.3 Unit Conversion Errors

**Problem**: Mixing lattice and physical units causes incorrect comparisons.

**Solution**: Always document units and use conversion utilities:
```cpp
/**
 * ALWAYS document units in comments and variable names
 */
// Lattice units
float u_lattice = 0.05f;  // [lattice units]
float L_lattice = 64.0f;  // [lattice spacing]

// Physical units
float u_physical = 1.0f;   // [m/s]
float L_physical = 100e-6f; // [m]

// Conversion
float dx = L_physical / L_lattice;  // [m/lattice spacing]
float dt = dx / u_physical * u_lattice;  // [s/timestep]
```

---

## 9. Checklist for Test Implementation

Before submitting a validation test, verify:

- [ ] Test has clear physics description in header comment
- [ ] Analytical solution is correctly implemented (verify formula source)
- [ ] Boundary conditions match analytical solution assumptions
- [ ] Grid resolution is sufficient (run grid convergence check)
- [ ] Simulation runs to steady state (or correct transient time)
- [ ] Error metrics (L2, Linf, avg) are computed and reported
- [ ] Tolerances are justified based on LBM accuracy expectations
- [ ] Test passes with reasonable margin (not right at tolerance edge)
- [ ] Output files are generated for visualization
- [ ] Test is integrated into CMake build system
- [ ] Test name follows convention: `validation_{category}_{test_name}`
- [ ] GoogleTest assertions use appropriate tolerance
- [ ] CUDA errors are checked after every kernel launch
- [ ] Memory is properly freed (no leaks)
- [ ] Test runs in reasonable time (<5 min on typical GPU)

---

**Document Version**: 1.0
**Date**: 2025-12-02
**Status**: Implementation Reference
