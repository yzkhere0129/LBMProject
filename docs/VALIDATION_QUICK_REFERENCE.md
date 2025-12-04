# Validation Test Quick Reference Guide

Quick reference for implementing validation tests. For full details, see `TEST_SUITE_DESIGN.md` and `TEST_IMPLEMENTATION_GUIDE.md`.

---

## Test Template (Copy-Paste Starter)

```cpp
/**
 * @file test_{category}_{name}.cu
 * @brief [One-line description of what this test validates]
 *
 * Physics: [Brief problem description]
 * Analytical Solution: [Reference or formula]
 * Expected Accuracy: [e.g., L2 < 5%]
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

// Include relevant solver headers
#include "physics/fluid_lbm.h"
#include "physics/thermal_lbm.h"

using namespace lbm::physics;

class TestName : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        // Initialize any required components
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    // Analytical solution function
    float analyticalSolution(float x, /* other params */) {
        // Return analytical value at position x
        return /* formula */;
    }
};

TEST_F(TestName, ValidationTest) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: [Name]\n";
    std::cout << "========================================\n\n";

    // 1. Setup simulation parameters
    const int nx = 64, ny = 64, nz = 3;
    const float dx = 1e-5f;
    const float dt = 1e-8f;

    // 2. Create solver and initialize
    // FluidLBM solver(...);
    // solver.initialize(...);

    // 3. Run simulation
    const int n_steps = 10000;
    for (int step = 0; step < n_steps; ++step) {
        // solver.step();
    }

    // 4. Extract results
    std::vector<float> numerical(ny);
    // Copy from device and extract profile

    // 5. Compute analytical solution
    std::vector<float> analytical(ny);
    for (int j = 0; j < ny; ++j) {
        analytical[j] = analyticalSolution(j * dx);
    }

    // 6. Compute error metrics
    float L2_error = computeL2Error(numerical, analytical, 2, ny-2);

    // 7. Print results
    std::cout << "L2 Error: " << L2_error * 100 << "%\n";
    std::cout << "Tolerance: 5.0%\n";

    // 8. Assertions
    EXPECT_LT(L2_error, 0.05f) << "L2 error exceeds 5%";

    std::cout << "\n========================================\n";
    std::cout << (L2_error < 0.05f ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << "========================================\n\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

---

## Standard Error Metrics

### L2 Relative Error (Most Common)
```cpp
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

### Maximum Error
```cpp
float computeMaxError(const std::vector<float>& numerical,
                     const std::vector<float>& analytical,
                     int start_idx, int end_idx)
{
    float max_err = 0.0f;
    for (int i = start_idx; i < end_idx; ++i) {
        float err = std::abs(numerical[i] - analytical[i]);
        max_err = std::max(max_err, err);
    }
    return max_err;
}
```

### Average Relative Error
```cpp
float computeAvgRelError(const std::vector<float>& numerical,
                        const std::vector<float>& analytical,
                        int start_idx, int end_idx)
{
    double sum = 0.0;
    int count = 0;
    for (int i = start_idx; i < end_idx; ++i) {
        float rel = std::abs(numerical[i] - analytical[i])
                  / (std::abs(analytical[i]) + 1e-10f);
        sum += rel;
        count++;
    }
    return sum / count;
}
```

---

## Common Analytical Solutions

### Poiseuille Flow (Channel)
```cpp
// u(y) = -(dp/dx) * y * (H - y) / (2*mu)
float poiseuilleVelocity(float y, float H, float dp_dx, float nu, float rho) {
    float mu = nu * rho;
    return -dp_dx * y * (H - y) / (2.0f * mu);
}

float poiseuilleMaxVelocity(float H, float dp_dx, float nu, float rho) {
    float mu = nu * rho;
    return -dp_dx * H * H / (8.0f * mu);
}
```

### Couette Flow (Moving Wall)
```cpp
// u(y) = U_top * (y / H)
float couetteVelocity(float y, float H, float U_top) {
    return U_top * (y / H);
}
```

### 1D Heat Conduction (Steady State)
```cpp
// T(x) = T_left + (T_right - T_left) * (x / L)
float heatConductionSteady(float x, float L, float T_left, float T_right) {
    return T_left + (T_right - T_left) * (x / L);
}

// Heat flux: q = -k * dT/dx
float heatFlux(float k, float L, float T_left, float T_right) {
    return -k * (T_right - T_left) / L;
}
```

### Taylor-Green Vortex (2D)
```cpp
// Decaying vortex solution
struct TGV {
    float u, v, p;
};

TGV taylorGreen(float x, float y, float t,
                float U0, float k, float nu, float rho)
{
    TGV sol;
    float decay = std::exp(-2.0f * k * k * nu * t);

    sol.u = -U0 * std::cos(k * x) * std::sin(k * y) * decay;
    sol.v =  U0 * std::sin(k * x) * std::cos(k * y) * decay;

    float p0 = 1.0f;
    sol.p = p0 - (rho * U0 * U0 / 4.0f)
               * (std::cos(2.0f * k * x) + std::cos(2.0f * k * y))
               * std::exp(-4.0f * k * k * nu * t);

    return sol;
}

// Kinetic energy (should decay exponentially)
float kineticEnergy(const std::vector<float>& ux,
                   const std::vector<float>& uy)
{
    double E = 0.0;
    for (size_t i = 0; i < ux.size(); ++i) {
        E += ux[i] * ux[i] + uy[i] * uy[i];
    }
    return 0.5f * E / ux.size();
}
```

---

## Profile Extraction

### 1D Profile from 3D Data (along y)
```cpp
std::vector<float> extract1DProfile_Y(const float* field_3d,
                                      int nx, int ny, int nz)
{
    std::vector<float> profile(ny, 0.0f);

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

    return profile;
}
```

---

## Conservation Checks

### Mass Conservation
```cpp
double computeTotalMass(const float* rho, int N, float cell_volume) {
    double mass = 0.0;
    for (int i = 0; i < N; ++i) {
        mass += rho[i] * cell_volume;
    }
    return mass;
}

bool checkMassConservation(double m_current, double m_initial,
                          float tolerance = 1e-4f)
{
    double rel_error = std::abs(m_current - m_initial) / m_initial;
    return (rel_error < tolerance);
}
```

### Momentum Conservation
```cpp
struct Vec3 {
    double x, y, z;
    double magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

Vec3 computeMomentum(const float* rho, const float* ux,
                    const float* uy, const float* uz,
                    int N, float cell_volume)
{
    Vec3 P = {0, 0, 0};
    for (int i = 0; i < N; ++i) {
        double mass = rho[i] * cell_volume;
        P.x += mass * ux[i];
        P.y += mass * uy[i];
        P.z += mass * uz[i];
    }
    return P;
}

bool checkMomentumBalance(Vec3 P_current, Vec3 P_previous,
                         Vec3 F_external, float dt,
                         float tolerance = 0.02f)
{
    Vec3 dP_dt;
    dP_dt.x = (P_current.x - P_previous.x) / dt;
    dP_dt.y = (P_current.y - P_previous.y) / dt;
    dP_dt.z = (P_current.z - P_previous.z) / dt;

    double error_x = std::abs(dP_dt.x - F_external.x)
                   / (std::abs(F_external.x) + 1e-10);
    double error_y = std::abs(dP_dt.y - F_external.y)
                   / (std::abs(F_external.y) + 1e-10);
    double error_z = std::abs(dP_dt.z - F_external.z)
                   / (std::abs(F_external.z) + 1e-10);

    return (error_x < tolerance && error_y < tolerance && error_z < tolerance);
}
```

### Energy Conservation
```cpp
struct Energy {
    double E_sensible;
    double E_latent;
    double E_total;
};

Energy computeEnergy(const float* T, const float* fl,
                    const MaterialProperties& mat,
                    int N, float cell_volume, float T_ref = 300.0f)
{
    Energy E = {0, 0, 0};

    for (int i = 0; i < N; ++i) {
        float T_i = T[i];
        float fl_i = fl[i];

        float rho = mat.getDensity(T_i);
        float cp = mat.getSpecificHeat(T_i);

        E.E_sensible += rho * cp * (T_i - T_ref) * cell_volume;
        E.E_latent += rho * mat.L_fusion * fl_i * cell_volume;
    }

    E.E_total = E.E_sensible + E.E_latent;
    return E;
}

bool checkEnergyBalance(double E_current, double E_previous,
                       float dt, float P_in, float P_out,
                       float tolerance = 0.05f)
{
    double dE_dt = (E_current - E_previous) / dt;
    double P_net = P_in - P_out;
    double error = std::abs(dE_dt - P_net) / std::abs(P_in + 1e-10);
    return (error < tolerance);
}
```

---

## Convergence Studies

### Grid Convergence
```cpp
struct ConvergenceResult {
    int resolution;
    float h;
    float error_L2;
};

std::vector<ConvergenceResult> runGridConvergence(
    std::function<float(int)> run_simulation,
    std::vector<int> resolutions)
{
    std::vector<ConvergenceResult> results;

    for (int res : resolutions) {
        std::cout << "Running resolution: " << res << "\n";
        float error = run_simulation(res);

        ConvergenceResult r;
        r.resolution = res;
        r.h = 1.0f / res;
        r.error_L2 = error;
        results.push_back(r);
    }

    return results;
}

float computeConvergenceOrder(float e1, float e2, float h1, float h2) {
    return std::log(e1 / e2) / std::log(h1 / h2);
}

void analyzeConvergence(const std::vector<ConvergenceResult>& results) {
    std::cout << "\n=== Convergence Analysis ===\n";
    std::cout << "Res     h          Error      Order\n";
    std::cout << "----------------------------------------\n";

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::setw(4) << results[i].resolution
                  << std::setw(10) << results[i].h
                  << std::setw(12) << results[i].error_L2;

        if (i > 0) {
            float order = computeConvergenceOrder(
                results[i-1].error_L2, results[i].error_L2,
                results[i-1].h, results[i].h
            );
            std::cout << std::setw(10) << order;
        }
        std::cout << "\n";
    }
}
```

---

## Dimensionless Numbers

### Reynolds Number
```cpp
struct FlowParams {
    float U, L, nu;
    float Re;

    void computeRe() { Re = U * L / nu; }

    void setRe(float Re_target, float L_fixed, float U_fixed) {
        Re = Re_target;
        L = L_fixed;
        U = U_fixed;
        nu = U * L / Re;
    }
};

// Drag coefficient: Cd = F_drag / (0.5 * rho * U^2 * A)
float dragCoefficient(float F_drag, float rho, float U, float A) {
    return F_drag / (0.5f * rho * U * U * A);
}

// Theory for sphere: Cd = 24/Re + 6/(1+sqrt(Re)) + 0.4
float theoreticalCd(float Re) {
    return 24.0f / Re + 6.0f / (1.0f + std::sqrt(Re)) + 0.4f;
}
```

### Peclet Number
```cpp
struct ThermalParams {
    float U, L, alpha;
    float Pe;

    void computePe() { Pe = U * L / alpha; }

    void setPe(float Pe_target, float L_fixed, float U_fixed) {
        Pe = Pe_target;
        L = L_fixed;
        U = U_fixed;
        alpha = U * L / Pe;
    }
};

// Nusselt number: Nu = h * L / k
float nusseltNumber(float q, float deltaT, float L, float k) {
    float h = q / deltaT;  // Heat transfer coefficient
    return h * L / k;
}

// Theory for sphere: Nu = 2 + 0.6 * Re^0.5 * Pr^0.33
float theoreticalNu(float Re, float Pr) {
    return 2.0f + 0.6f * std::pow(Re, 0.5f) * std::pow(Pr, 0.33f);
}
```

### Mach Number
```cpp
float machNumber(float u) {
    const float c_s = 1.0f / std::sqrt(3.0f);  // LBM speed of sound
    return u / c_s;
}

// Density fluctuations should scale as Ma^2
bool checkCompressibility(const std::vector<float>& rho,
                         float rho0, float Ma,
                         float tolerance = 0.3f)
{
    // Compute RMS density fluctuation
    double sum_sq = 0.0;
    for (float r : rho) {
        double drho = (r - rho0) / rho0;
        sum_sq += drho * drho;
    }
    float rms = std::sqrt(sum_sq / rho.size());

    // Theoretical: drho/rho ~ Ma^2
    float expected = Ma * Ma;
    float error = std::abs(rms - expected) / expected;

    return (error < tolerance);
}
```

---

## Stability Testing

### Binary Search for Stability Boundary
```cpp
float findStabilityBoundary(
    std::function<bool(float)> is_stable,
    float param_min, float param_max,
    float tolerance = 1e-3f)
{
    int max_iter = 50;
    int iter = 0;

    while (param_max - param_min > tolerance && iter < max_iter) {
        float param_mid = 0.5f * (param_min + param_max);

        std::cout << "Testing parameter = " << param_mid << "... ";

        if (is_stable(param_mid)) {
            std::cout << "STABLE\n";
            param_min = param_mid;
        } else {
            std::cout << "UNSTABLE\n";
            param_max = param_mid;
        }

        iter++;
    }

    return param_min;
}

bool checkStability(const float* rho, int N, float rho0) {
    // Check for NaN
    for (int i = 0; i < N; ++i) {
        if (std::isnan(rho[i]) || std::isinf(rho[i])) {
            return false;
        }
    }

    // Check density fluctuations < 10%
    float rho_min = *std::min_element(rho, rho + N);
    float rho_max = *std::max_element(rho, rho + N);
    float fluctuation = (rho_max - rho_min) / rho0;

    return (fluctuation < 0.1f);
}
```

---

## Output and Reporting

### Export Profile Data
```cpp
void exportProfile(const std::vector<float>& profile,
                  const std::string& filename)
{
    std::ofstream file(filename);
    file << "# index\tvalue\n";
    for (size_t i = 0; i < profile.size(); ++i) {
        file << i << "\t" << profile[i] << "\n";
    }
    file.close();
}

void exportComparison(const std::vector<float>& numerical,
                     const std::vector<float>& analytical,
                     const std::string& filename)
{
    std::ofstream file(filename);
    file << "# index\tnumerical\tanalytical\terror\n";
    for (size_t i = 0; i < numerical.size(); ++i) {
        float error = numerical[i] - analytical[i];
        file << i << "\t" << numerical[i] << "\t"
             << analytical[i] << "\t" << error << "\n";
    }
    file.close();
}
```

### Standard Output Format
```cpp
void printTestHeader(const std::string& name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << name << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void printTestResult(const std::string& metric, float measured,
                    float expected, float tolerance, bool passed)
{
    std::cout << metric << ":\n";
    std::cout << "  Measured:  " << measured << "\n";
    std::cout << "  Expected:  " << expected << "\n";
    std::cout << "  Tolerance: " << tolerance * 100 << "%\n";
    std::cout << "  Status:    " << (passed ? "PASS ✓" : "FAIL ✗") << "\n\n";
}
```

---

## Typical Tolerances

| Test Type | Tolerance | Rationale |
|-----------|-----------|-----------|
| Analytical solution (L2 error) | 5-6% | LBM second-order discretization |
| Mass conservation (closed) | 0.01% | Should be exact (numerical precision) |
| Mass conservation (with sources) | 1% | Accounting for evaporation, boundaries |
| Momentum conservation | 2% | Force application accuracy |
| Energy conservation | 5% | Multiple sources/sinks, coupling errors |
| Convergence order (spatial) | 1.8 < p < 2.2 | Second-order scheme |
| Convergence order (temporal) | 0.85 < p < 1.15 | First-order explicit time integration |
| Reynolds number scaling | 8-10% | Drag coefficient correlations |
| Peclet number scaling | 10% | Heat transfer correlations |

---

## Common Pitfalls

### 1. Not Reaching Steady State
**Problem**: Compare to analytical before transients decay.

**Solution**: Monitor convergence, run longer, or check `max|u^n - u^{n-100}| / max|u|`.

---

### 2. Including Boundary Cells in Error
**Problem**: Boundary treatment introduces larger errors at walls.

**Solution**: Exclude 1-2 boundary cells:
```cpp
int start = 2;
int end = ny - 2;
float L2_error = computeL2Error(numerical, analytical, start, end);
```

---

### 3. Wrong Units
**Problem**: Mixing lattice and physical units.

**Solution**: Document units explicitly:
```cpp
float u_lattice = 0.05f;  // [lattice units]
float u_physical = 1.0f;   // [m/s]
// Conversion: u_physical = u_lattice * (dx / dt)
```

---

### 4. Insufficient Resolution
**Problem**: Grid too coarse to resolve features.

**Solution**: Run grid convergence study, ensure at least 16 cells across characteristic length.

---

### 5. Forgetting CUDA Error Checks
**Problem**: Kernel launch failures go undetected.

**Solution**: Always check:
```cpp
kernel<<<grid, block>>>(...);
cudaError_t err = cudaGetLastError();
ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
```

---

## Quick Checklist

Before submitting test:
- [ ] Header comment has physics description and analytical solution
- [ ] Steady state reached (or correct transient time)
- [ ] Error metrics computed (L2, max, avg)
- [ ] Tolerances justified and documented
- [ ] Boundary cells excluded from error calculation
- [ ] Output files generated for visualization
- [ ] GoogleTest assertions present
- [ ] CUDA errors checked
- [ ] Memory freed (no leaks)
- [ ] Test added to CMakeLists.txt
- [ ] Test passes consistently (run 3+ times)

---

**For complete details, refer to**:
- `TEST_SUITE_DESIGN.md` - Full test specifications
- `TEST_IMPLEMENTATION_GUIDE.md` - Detailed implementation patterns
- `VALIDATION_ROADMAP.md` - Implementation schedule and priorities
