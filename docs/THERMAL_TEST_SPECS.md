# ThermalLBM Comprehensive Test Specifications

## Overview

This document provides detailed specifications for comprehensive standalone tests of the ThermalLBM module. Each test validates a specific physical phenomenon with analytical solutions or known benchmarks.

**Module Under Test:**
- Source: `/home/yzk/LBMProject/src/physics/thermal/thermal_lbm.cu`
- Header: `/home/yzk/LBMProject/include/physics/thermal_lbm.h`
- Lattice: D3Q7 (7 discrete velocities for scalar transport)

**Testing Framework:** Google Test (GTest)

**Test Organization:**
- Unit tests: Individual functions and kernels
- Integration tests: Multi-step physics simulations
- Validation tests: Comparison with analytical solutions

---

## Test Suite 1: Pure Heat Conduction

### Test 1.1: 1D Steady-State Conduction (Linear Profile)

**Physical Phenomenon:** Heat conduction between two constant-temperature walls reaches steady state with linear temperature profile.

**Domain Configuration:**
```cpp
int nx = 100;    // 1D domain
int ny = 1;
int nz = 1;
float L = 100e-6;  // 100 microns
float dx = L / (nx - 1);
```

**Material Properties:**
- Ti-6Al-4V solid phase
- k = 21.9 W/(m·K)
- rho = 4430 kg/m³
- cp = 546 J/(kg·K)
- alpha = k/(rho*cp) = 9.05e-6 m²/s

**Initial Conditions:**
- Uniform temperature T = 1000 K

**Boundary Conditions:**
- x=0: Dirichlet BC, T_cold = 1000 K
- x=L: Dirichlet BC, T_hot = 2000 K
- y, z: Periodic (single cell)

**Simulation Parameters:**
```cpp
float dt = 0.1 * dx * dx / alpha;  // CFL condition
int num_steps = 100000;  // Run until steady state
```

**Analytical Solution:**
```cpp
T(x) = T_cold + (T_hot - T_cold) * x / L
```

**Validation Metrics:**
```cpp
// L2 norm error
float L2_error = sqrt(sum((T_num[i] - T_ana[i])^2) / sum(T_ana[i]^2));

// Max absolute error
float max_error = max(abs(T_num[i] - T_ana[i]));

// Check linearity via R² coefficient
float R_squared = 1 - SSres/SStot;
```

**Success Criteria:**
- L2 error < 1% after steady state
- Max error < 10 K at any point
- R² > 0.999 (confirms linear profile)
- Convergence time matches theoretical estimate: t_steady ≈ L²/α

**GTest Implementation:**
```cpp
TEST_F(ThermalLBMTest, SteadyStateConduction1D) {
    // Setup
    ThermalLBM solver(nx, ny, nz, material, alpha, false, dt, dx);
    solver.initialize(1000.0f);

    // Apply boundary conditions
    applyDirichletBC(solver, FACE_X_MIN, 1000.0f);
    applyDirichletBC(solver, FACE_X_MAX, 2000.0f);

    // Run to steady state
    for (int step = 0; step < num_steps; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.applyBoundaryConditions(1, 0.0f);  // Constant T
        solver.computeTemperature();
    }

    // Validate
    std::vector<float> T_num(nx);
    solver.copyTemperatureToHost(T_num.data());

    float L2_error = computeL2Error(T_num, analyticalLinearProfile);
    EXPECT_LT(L2_error, 0.01f) << "L2 error exceeds 1%";

    float R_squared = computeRSquared(T_num);
    EXPECT_GT(R_squared, 0.999f) << "Profile is not linear";
}
```

---

### Test 1.2: 1D Transient Conduction (Gaussian Diffusion)

**Physical Phenomenon:** A Gaussian temperature pulse diffuses and broadens according to the heat equation.

**Domain Configuration:**
```cpp
int nx = 200;
int ny = 1;
int nz = 1;
float L = 400e-6;  // 400 microns
float dx = L / (nx - 1);
```

**Initial Conditions:**
```cpp
// Gaussian pulse centered at x = L/2
float x_center = L / 2.0f;
float sigma0 = L / 20.0f;  // Initial width
T(x, t=0) = T_ambient + (T_peak - T_ambient) * exp(-(x - x_center)² / (2*sigma0²))
```
- T_ambient = 300 K
- T_peak = 1943 K (Ti-6Al-4V melting point)

**Boundary Conditions:**
- All boundaries: Adiabatic (zero flux)

**Analytical Solution:**
```cpp
// Gaussian diffusion solution
float sigma_t = sqrt(sigma0² + 2*alpha*t);
T(x,t) = T_ambient + (T_peak - T_ambient) * (sigma0/sigma_t) *
         exp(-(x - x_center)² / (2*sigma_t²))
```

**Test Times:**
- t₁ = 0.1 ms
- t₂ = 0.5 ms
- t₃ = 1.0 ms

**Validation Metrics:**
```cpp
// Peak temperature (should decrease as 1/sqrt(t))
float T_peak_expected = T_ambient + (T_peak - T_ambient) * sigma0 / sigma_t;

// Width (should increase as sqrt(t))
float width_expected = sigma_t;

// Total energy (should be conserved)
float E_total = integral(rho * cp * T * dx);
```

**Success Criteria:**
- L2 error < 5% at all test times
- Peak temperature error < 10 K
- Width error < 5%
- Energy conservation: |E(t) - E(0)| / E(0) < 0.1%

**GTest Implementation:**
```cpp
TEST_F(ThermalLBMTest, TransientGaussianDiffusion) {
    // Setup with Gaussian initial condition
    ThermalLBM solver(nx, ny, nz, material, alpha, false, dt, dx);
    std::vector<float> T_initial(nx);
    for (int i = 0; i < nx; ++i) {
        float x = i * dx;
        T_initial[i] = gaussianProfile(x, 0.0f);
    }
    solver.initialize(T_initial.data());

    float E_initial = computeTotalEnergy(solver);

    // Test at multiple times
    float test_times[] = {0.1e-3f, 0.5e-3f, 1.0e-3f};
    for (float t_target : test_times) {
        int steps = (int)(t_target / dt);
        for (int s = 0; s < steps; ++s) {
            solver.collisionBGK();
            solver.streaming();
            solver.computeTemperature();
        }

        // Validate against analytical solution
        float L2_error = validateGaussianProfile(solver, t_target);
        EXPECT_LT(L2_error, 0.05f) << "L2 error at t=" << t_target;

        float E_current = computeTotalEnergy(solver);
        float E_change = fabs(E_current - E_initial) / E_initial;
        EXPECT_LT(E_change, 0.001f) << "Energy drift at t=" << t_target;
    }
}
```

---

## Test Suite 2: Thermal Diffusion with Variable Properties

### Test 2.1: Temperature-Dependent Conductivity

**Physical Phenomenon:** Heat diffusion with k(T), testing the material property interpolation in the mushy zone.

**Domain Configuration:**
```cpp
int nx = 100;
int ny = 1;
int nz = 1;
```

**Material Properties:**
```cpp
// Ti-6Al-4V with phase change
T_solidus = 1878 K
T_liquidus = 1943 K
k_solid = 21.9 W/(m·K)
k_liquid = 28.5 W/(m·K)
```

**Initial Conditions:**
- Temperature range spanning solid, mushy, and liquid zones
- T(x) = 1500 + 1000 * x/L (linear gradient crossing phase boundaries)

**Test Objective:**
- Verify that getThermalDiffusivity(T) correctly interpolates in mushy zone
- Check tau_T updates properly when alpha changes

**Validation:**
```cpp
TEST_F(ThermalLBMTest, TemperatureDependentDiffusivity) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();

    // Test points
    float T_solid = 1500.0f;    // Below solidus
    float T_mushy = 1910.0f;    // Between solidus and liquidus
    float T_liquid = 2000.0f;   // Above liquidus

    // Expected values
    float alpha_solid = mat.getThermalDiffusivity(T_solid);
    float alpha_mushy = mat.getThermalDiffusivity(T_mushy);
    float alpha_liquid = mat.getThermalDiffusivity(T_liquid);

    // Verify interpolation
    EXPECT_FLOAT_EQ(alpha_solid, k_solid / (rho_solid * cp_solid));
    EXPECT_FLOAT_EQ(alpha_liquid, k_liquid / (rho_liquid * cp_liquid));

    // Mushy zone should be between solid and liquid
    EXPECT_GT(alpha_mushy, alpha_solid);
    EXPECT_LT(alpha_mushy, alpha_liquid);

    // Verify tau_T calculation
    float tau_solid = ThermalLBM::computeThermalTau(alpha_solid, dx, dt);
    float tau_liquid = ThermalLBM::computeThermalTau(alpha_liquid, dx, dt);
    EXPECT_GT(tau_solid, 0.5f);
    EXPECT_LT(tau_solid, 2.0f);  // Stability range
}
```

---

## Test Suite 3: Stefan Problem (Phase Change)

### Test 3.1: 1D Stefan Problem (Moving Interface)

**Physical Phenomenon:** Solid-liquid interface moves as s(t) ∝ √t when melting occurs.

**Domain Configuration:**
```cpp
int nx = 200;
int ny = 1;
int nz = 1;
float L = 400e-6;
```

**Initial Conditions:**
- Left half: T = T_liquidus + 100 K (liquid)
- Right half: T = T_solidus - 100 K (solid)
- Sharp interface at x = L/2

**Boundary Conditions:**
- x=0: T = T_liquidus + 100 K (hot)
- x=L: T = T_solidus - 100 K (cold)

**Analytical Solution (Stefan Number):**
```cpp
// Stefan number
float Ste = cp_solid * (T_hot - T_melt) / L_fusion;

// Interface position (approximate)
float s(t) = 2 * lambda * sqrt(alpha * t);
// where lambda is found from: lambda * exp(lambda²) * erf(lambda) = Ste / sqrt(pi)
```

**Validation Metrics:**
```cpp
// Track interface position via liquid fraction
float x_interface = findInterfacePosition(liquid_fraction);

// Check sqrt(t) scaling
float slope = d(x_interface²) / dt;  // Should be constant
```

**Success Criteria:**
- Interface position follows √t scaling
- Latent heat is properly absorbed (temperature plateau in mushy zone)
- Energy balance: Q_conducted = rho * L * (dx_interface/dt)

**GTest Implementation:**
```cpp
TEST_F(ThermalLBMTest, StefanProblem1D) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();
    ThermalLBM solver(nx, ny, nz, mat, alpha, true, dt, dx);  // Enable phase change

    // Initialize with step function
    initializeStepFunction(solver);

    // Track interface over time
    std::vector<float> interface_positions;
    std::vector<float> times;

    for (int step = 0; step < 10000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.computeTemperature();
        solver.applyPhaseChangeCorrection(dt);

        if (step % 100 == 0) {
            float x_int = findInterfacePosition(solver);
            interface_positions.push_back(x_int);
            times.push_back(step * dt);
        }
    }

    // Verify sqrt(t) scaling
    bool follows_sqrt_t = checkSqrtTScaling(interface_positions, times);
    EXPECT_TRUE(follows_sqrt_t) << "Interface does not follow s(t) ~ sqrt(t)";
}
```

---

## Test Suite 4: Advection-Diffusion

### Test 4.1: Passive Scalar Advection (Plug Flow)

**Physical Phenomenon:** Temperature pulse advects downstream with constant velocity while diffusing.

**Domain Configuration:**
```cpp
int nx = 200;
int ny = 10;
int nz = 1;
```

**Velocity Field:**
```cpp
float U = 0.1;  // Constant velocity in x-direction (lattice units)
float ux = U;
float uy = 0.0f;
float uz = 0.0f;
```

**Initial Conditions:**
```cpp
// Gaussian pulse at x = L/4
T(x,y,z) = T_ambient + ΔT * exp(-((x-x0)² + (y-y0)²) / (2*sigma²))
```

**Analytical Solution:**
```cpp
// Advection-diffusion of Gaussian
float x_center_t = x0 + U * t;
float sigma_t = sqrt(sigma0² + 2*alpha*t);
T(x,y,t) = T_ambient + ΔT * (sigma0/sigma_t)² *
           exp(-((x-x_center_t)² + (y-y0)²) / (2*sigma_t²))
```

**Peclet Number:**
```cpp
float Pe = U * L / alpha;  // Should be < 10 for stability
```

**Validation Metrics:**
- Peak position: x_peak(t) = x0 + U*t
- Peak temperature: T_peak(t) = T0 * (sigma0/sigma_t)²
- L2 error against analytical solution

**Success Criteria:**
- Peak position error < 2 cells
- L2 error < 10% (advection-diffusion has larger errors than pure diffusion)
- No oscillations or negative diffusion

**GTest Implementation:**
```cpp
TEST_F(ThermalLBMTest, AdvectionDiffusionPlugFlow) {
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);

    // Initialize Gaussian pulse
    initializeGaussianPulse(solver, x0, y0);

    // Create constant velocity field
    float* d_ux = createConstantVelocityField(nx, ny, nz, U, 0.0f, 0.0f);

    // Simulate
    for (int step = 0; step < num_steps; ++step) {
        solver.collisionBGK(d_ux, nullptr, nullptr);
        solver.streaming();
        solver.computeTemperature();
    }

    // Find peak position
    float x_peak_num = findPeakPosition(solver);
    float x_peak_ana = x0 + U * (num_steps * dt);

    float position_error = fabs(x_peak_num - x_peak_ana) / dx;
    EXPECT_LT(position_error, 2.0f) << "Peak advected incorrectly";

    cudaFree(d_ux);
}
```

---

## Test Suite 5: Energy Conservation

### Test 5.1: Isolated System Energy Conservation

**Physical Phenomenon:** Total internal energy should remain constant in an isolated system (no heat sources/sinks).

**Domain Configuration:**
```cpp
int nx = 50;
int ny = 50;
int nz = 50;
```

**Initial Conditions:**
- Random temperature field: T ∈ [300, 2000] K
- Includes solid, mushy, and liquid regions

**Boundary Conditions:**
- All boundaries: Adiabatic (zero flux)

**Energy Definition:**
```cpp
float E_total = sum(rho(T) * cp(T) * (T - T_ref) * dV +
                    f_liquid(T) * rho * L_fusion * dV)
// Sensible energy + Latent energy
```

**Validation:**
```cpp
float dE = |E(t) - E(0)|;
float dE_relative = dE / E(0);
```

**Success Criteria:**
- Relative energy drift < 0.1% after 10,000 steps
- No energy accumulation or loss trends

**GTest Implementation:**
```cpp
TEST_F(ThermalLBMTest, EnergyConservationIsolated) {
    ThermalLBM solver(nx, ny, nz, mat, alpha, true, dt, dx);

    // Random initial condition
    initializeRandomTemperatureField(solver);

    float E_initial = solver.computeTotalThermalEnergy(dx);

    // Run for many steps
    for (int step = 0; step < 10000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.computeTemperature();

        if (solver.hasPhaseChange()) {
            solver.applyPhaseChangeCorrection(dt);
        }
    }

    float E_final = solver.computeTotalThermalEnergy(dx);
    float E_drift = fabs(E_final - E_initial) / E_initial;

    EXPECT_LT(E_drift, 0.001f) << "Energy drift exceeds 0.1%";
}
```

---

### Test 5.2: Energy Balance with Heat Source

**Physical Phenomenon:** Energy added via heat source should equal increase in internal energy.

**Domain Configuration:**
```cpp
int nx = 30;
int ny = 30;
int nz = 30;
```

**Heat Source:**
```cpp
// Constant volumetric heat source
float Q = 1e12;  // W/m³ (laser-like intensity)
// Applied to central cell only
```

**Energy Balance:**
```cpp
dE/dt = Q * V_heated
// Integrated energy increase should match Q * V * t
```

**Validation:**
```cpp
float E_added_expected = Q * V_heated * t_total;
float E_added_actual = E(t) - E(0);
float error = |E_added_actual - E_added_expected| / E_added_expected;
```

**Success Criteria:**
- Energy balance error < 1%
- No energy loss to boundaries (check adiabatic BC)

**GTest Implementation:**
```cpp
TEST_F(ThermalLBMTest, EnergyBalanceHeatSource) {
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);
    solver.initialize(300.0f);

    // Create heat source (central cell only)
    float* d_Q = createCentralHeatSource(nx, ny, nz, Q);

    float E_initial = solver.computeTotalThermalEnergy(dx);

    // Apply heat source for many steps
    int num_steps = 1000;
    for (int step = 0; step < num_steps; ++step) {
        solver.addHeatSource(d_Q, dt);
        solver.collisionBGK();
        solver.streaming();
        solver.computeTemperature();
    }

    float E_final = solver.computeTotalThermalEnergy(dx);
    float E_added_actual = E_final - E_initial;

    float V_cell = dx * dx * dx;
    float E_added_expected = Q * V_cell * num_steps * dt;

    float error = fabs(E_added_actual - E_added_expected) / E_added_expected;
    EXPECT_LT(error, 0.01f) << "Energy balance error exceeds 1%";

    cudaFree(d_Q);
}
```

---

## Test Suite 6: Boundary Conditions

### Test 6.1: Dirichlet BC (Constant Temperature)

**Setup:**
- 1D domain with T_left = 1000 K, T_right = 2000 K
- Steady state should give linear profile

**Validation:**
```cpp
TEST_F(ThermalLBMTest, DirichletBoundaryCondition) {
    ThermalLBM solver(nx, 1, 1, mat, alpha, false, dt, dx);
    solver.initialize(1500.0f);

    // Run to steady state
    for (int step = 0; step < 100000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.applyBoundaryConditions(1, 0.0f);  // Type 1 = constant T
        applyDirichletBC(solver, FACE_X_MIN, 1000.0f);
        applyDirichletBC(solver, FACE_X_MAX, 2000.0f);
        solver.computeTemperature();
    }

    // Check boundary values
    std::vector<float> T(nx);
    solver.copyTemperatureToHost(T.data());

    EXPECT_NEAR(T[0], 1000.0f, 1.0f) << "Left BC not enforced";
    EXPECT_NEAR(T[nx-1], 2000.0f, 1.0f) << "Right BC not enforced";

    // Check linearity
    float R2 = computeLinearityMetric(T);
    EXPECT_GT(R2, 0.999f) << "Profile not linear";
}
```

---

### Test 6.2: Neumann BC (Adiabatic)

**Setup:**
- All boundaries adiabatic
- Initial hot spot in center
- Heat should not escape

**Validation:**
```cpp
TEST_F(ThermalLBMTest, NeumannAdiabaticBC) {
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);

    // Hot spot in center
    initializeHotSpot(solver, nx/2, ny/2, nz/2);

    float E_initial = solver.computeTotalThermalEnergy(dx);

    // Diffuse for many steps
    for (int step = 0; step < 5000; ++step) {
        solver.collisionBGK();
        solver.streaming();  // Adiabatic BC via bounce-back
        solver.computeTemperature();
    }

    float E_final = solver.computeTotalThermalEnergy(dx);
    float E_drift = fabs(E_final - E_initial) / E_initial;

    EXPECT_LT(E_drift, 0.001f) << "Energy leaked through adiabatic BC";
}
```

---

### Test 6.3: Radiation BC (Stefan-Boltzmann)

**Physical Phenomenon:** Surface cooling via radiation: q = ε σ (T⁴ - T_amb⁴)

**Setup:**
```cpp
int nx = 30;
int ny = 30;
int nz = 30;
float epsilon = 0.35f;
float T_ambient = 300.0f;
```

**Initial Conditions:**
- Uniform high temperature T = 3000 K
- Top surface (z=nz-1) has radiation BC

**Expected Behavior:**
- Surface temperature should decrease faster than interior
- Cooling rate proportional to T⁴

**Validation:**
```cpp
TEST_F(ThermalLBMTest, RadiationBoundaryCondition) {
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);
    solver.initialize(3000.0f);
    solver.setEmissivity(0.35f);

    // Track surface temperature
    std::vector<float> T_surface_history;
    std::vector<float> times;

    for (int step = 0; step < 1000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.applyRadiationBC(dt, dx, 0.35f, 300.0f);
        solver.computeTemperature();

        if (step % 10 == 0) {
            float T_surf = getSurfaceTemperature(solver, nz-1);
            T_surface_history.push_back(T_surf);
            times.push_back(step * dt);
        }
    }

    // Verify cooling occurred
    EXPECT_LT(T_surface_history.back(), T_surface_history[0]);

    // Verify T⁴ scaling at early times
    bool follows_T4_law = checkT4CoolingRate(T_surface_history, times);
    EXPECT_TRUE(follows_T4_law) << "Radiation BC does not follow T⁴ law";
}
```

---

### Test 6.4: Substrate Cooling BC (Convective)

**Physical Phenomenon:** Bottom surface cooling via convection: q = h (T - T_substrate)

**Setup:**
```cpp
float h_conv = 50000.0f;  // W/(m²·K)
float T_substrate = 300.0f;
```

**Validation:**
```cpp
TEST_F(ThermalLBMTest, SubstrateCoolingBC) {
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);
    solver.initialize(2000.0f);

    float E_initial = solver.computeTotalThermalEnergy(dx);

    for (int step = 0; step < 1000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.applySubstrateCoolingBC(dt, dx, h_conv, T_substrate);
        solver.computeTemperature();
    }

    // Energy should decrease due to cooling
    float E_final = solver.computeTotalThermalEnergy(dx);
    EXPECT_LT(E_final, E_initial) << "No cooling detected";

    // Compute expected cooling power
    float P_substrate_expected = computeExpectedSubstratePower(solver);
    float P_substrate_actual = solver.computeSubstratePower(dx, h_conv, T_substrate);

    float power_error = fabs(P_substrate_actual - P_substrate_expected) / P_substrate_expected;
    EXPECT_LT(power_error, 0.05f) << "Substrate power calculation error > 5%";
}
```

---

## Test Suite 7: Stability and Robustness

### Test 7.1: High Peclet Number Stability (Pe > 10)

**Challenge:** BGK collision becomes unstable when advection dominates diffusion.

**Setup:**
```cpp
float U = 0.3;  // High velocity (lattice units)
float alpha_small = 0.01;  // Low diffusivity
float Pe = U * L / alpha_small;  // Pe ≈ 30
```

**Expected Behavior:**
- Omega capping should activate (ω → 1.85)
- No oscillations or blow-up

**Validation:**
```cpp
TEST_F(ThermalLBMTest, HighPecletStability) {
    // Create low diffusivity material
    float alpha_low = 0.001;
    ThermalLBM solver(nx, ny, nz, mat, alpha_low, false, dt, dx);

    float omega = solver.getThermalOmega();
    EXPECT_LE(omega, 1.85f) << "Omega not capped for stability";

    // High velocity field
    float* d_ux = createConstantVelocityField(nx, ny, nz, 0.3f, 0.0f, 0.0f);

    // Run simulation
    bool remained_stable = true;
    for (int step = 0; step < 1000; ++step) {
        solver.collisionBGK(d_ux, nullptr, nullptr);
        solver.streaming();
        solver.computeTemperature();

        if (!checkFieldValidity(solver)) {
            remained_stable = false;
            break;
        }
    }

    EXPECT_TRUE(remained_stable) << "Simulation became unstable at high Pe";
    cudaFree(d_ux);
}
```

---

### Test 7.2: Large Temperature Gradients (No Oscillations)

**Challenge:** Sharp temperature gradients can cause numerical oscillations in BGK.

**Setup:**
```cpp
// Step function: T = 2000 K (x < L/2), T = 300 K (x >= L/2)
```

**Expected Behavior:**
- Interface should diffuse smoothly
- No overshoots or undershoots

**Validation:**
```cpp
TEST_F(ThermalLBMTest, SharpGradientStability) {
    ThermalLBM solver(nx, 1, 1, mat, alpha, false, dt, dx);

    // Initialize step function
    initializeStepFunction(solver, nx/2);

    // Check for oscillations
    bool has_oscillations = false;
    for (int step = 0; step < 1000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.computeTemperature();

        if (detectOscillations(solver)) {
            has_oscillations = true;
            break;
        }
    }

    EXPECT_FALSE(has_oscillations) << "Oscillations detected near sharp gradient";

    // Check monotonicity
    bool is_monotonic = checkMonotonicity(solver);
    EXPECT_TRUE(is_monotonic) << "Profile is not monotonic";
}
```

---

### Test 7.3: Omega Capping Verification

**Objective:** Verify that omega is limited to 1.85 when thermal diffusivity is very high.

**Validation:**
```cpp
TEST_F(ThermalLBMTest, OmegaCappingLogic) {
    // Very high diffusivity → omega would be > 1.9
    float alpha_high = 1.0;  // Unphysically high for testing

    ThermalLBM solver(nx, ny, nz, mat, alpha_high, false, dt, dx);

    float omega = solver.getThermalOmega();
    float tau = solver.getThermalTau();

    EXPECT_LE(omega, 1.85f) << "Omega exceeded stability limit";
    EXPECT_GE(tau, 1.0f/1.85f) << "Tau too small";

    std::cout << "High alpha = " << alpha_high << " → omega = " << omega << std::endl;
}
```

---

## Test Suite 8: Phase Change Physics

### Test 8.1: Latent Heat Absorption (Melting)

**Physical Phenomenon:** When material melts, temperature should plateau in mushy zone while latent heat is absorbed.

**Setup:**
```cpp
// Heat a solid block crossing the melting point
T_initial = 1800 K  // Below T_solidus = 1878 K
// Apply constant heat source
```

**Expected Behavior:**
- Temperature rises linearly until T_solidus
- Slows down in mushy zone (1878 - 1943 K)
- Resumes linear rise above T_liquidus

**Validation:**
```cpp
TEST_F(ThermalLBMTest, LatentHeatAbsorptionMelting) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();
    ThermalLBM solver(nx, ny, nz, mat, alpha, true, dt, dx);
    solver.initialize(1800.0f);

    // Constant heat source
    float Q = 1e12;  // W/m³
    float* d_Q = createUniformHeatSource(nx, ny, nz, Q);

    std::vector<float> T_avg_history;
    std::vector<float> fl_avg_history;

    for (int step = 0; step < 5000; ++step) {
        solver.addHeatSource(d_Q, dt);
        solver.collisionBGK();
        solver.streaming();
        solver.computeTemperature();
        solver.applyPhaseChangeCorrection(dt);

        T_avg_history.push_back(computeAverageTemperature(solver));
        fl_avg_history.push_back(computeAverageLiquidFraction(solver));
    }

    // Verify temperature plateau in mushy zone
    bool has_plateau = detectTemperaturePlateau(T_avg_history,
                                                 mat.T_solidus, mat.T_liquidus);
    EXPECT_TRUE(has_plateau) << "No temperature plateau during melting";

    // Verify liquid fraction increases
    EXPECT_GT(fl_avg_history.back(), 0.9f) << "Material did not fully melt";

    cudaFree(d_Q);
}
```

---

### Test 8.2: Latent Heat Release (Solidification)

**Physical Phenomenon:** During cooling/solidification, temperature should plateau as latent heat is released.

**Validation:**
```cpp
TEST_F(ThermalLBMTest, LatentHeatReleaseSolidification) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();
    ThermalLBM solver(nx, ny, nz, mat, alpha, true, dt, dx);
    solver.initialize(2100.0f);  // Start fully liquid

    // Apply cooling (radiation + substrate)
    for (int step = 0; step < 10000; ++step) {
        solver.collisionBGK();
        solver.streaming();
        solver.applyRadiationBC(dt, dx, 0.35f, 300.0f);
        solver.applySubstrateCoolingBC(dt, dx, 50000.0f, 300.0f);
        solver.computeTemperature();
        solver.applyPhaseChangeCorrection(dt);

        // Track temperature during solidification
        if (step % 100 == 0) {
            float T_avg = computeAverageTemperature(solver);
            float fl_avg = computeAverageLiquidFraction(solver);

            // When in mushy zone, temperature should change slowly
            if (fl_avg > 0.1f && fl_avg < 0.9f) {
                // This is the plateau region
            }
        }
    }

    // Final state should be mostly solid
    float fl_final = computeAverageLiquidFraction(solver);
    EXPECT_LT(fl_final, 0.1f) << "Material did not solidify";
}
```

---

### Test 8.3: Mushy Zone Liquid Fraction

**Objective:** Verify that liquid fraction is correctly computed in the mushy zone.

**Validation:**
```cpp
TEST_F(ThermalLBMTest, MushyZoneLiquidFraction) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();

    // Test liquid fraction calculation
    float T_solidus = mat.T_solidus;
    float T_liquidus = mat.T_liquidus;

    float T_below = T_solidus - 100.0f;
    float T_mid = (T_solidus + T_liquidus) / 2.0f;
    float T_above = T_liquidus + 100.0f;

    EXPECT_FLOAT_EQ(mat.liquidFraction(T_below), 0.0f);
    EXPECT_FLOAT_EQ(mat.liquidFraction(T_mid), 0.5f);
    EXPECT_FLOAT_EQ(mat.liquidFraction(T_above), 1.0f);

    // Linear interpolation in mushy zone
    float fl_quarter = mat.liquidFraction(T_solidus + 0.25f * (T_liquidus - T_solidus));
    EXPECT_NEAR(fl_quarter, 0.25f, 1e-6f);
}
```

---

## Test Suite 9: Evaporation Physics

### Test 9.1: Evaporation Mass Flux (Hertz-Knudsen)

**Physical Phenomenon:** Evaporation mass flux follows J = α P_sat / sqrt(2πRT/M)

**Setup:**
```cpp
float T_test[] = {3000, 3500, 4000, 4500, 5000};  // K
float T_boil = 3560 K;  // Ti-6Al-4V
```

**Validation:**
```cpp
TEST_F(ThermalLBMTest, EvaporationMassFlux) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);

    // Create interface (VOF fill level = 0.5)
    float* d_fill = createUniformFillLevel(nx, ny, nz, 0.5f);
    float* d_J_evap = allocateDeviceArray(nx * ny * nz);

    // Test at different temperatures
    for (float T_test : {3000, 3500, 4000}) {
        solver.initialize(T_test);
        solver.computeEvaporationMassFlux(d_J_evap, d_fill);

        float J_avg = computeAverageFlux(d_J_evap);

        // Verify flux increases exponentially with temperature
        std::cout << "T = " << T_test << " K → J_evap = " << J_avg << " kg/(m²·s)" << std::endl;

        // Flux should be positive and finite
        EXPECT_GT(J_avg, 0.0f);
        EXPECT_LT(J_avg, 1000.0f);  // Physical upper limit
    }

    cudaFree(d_fill);
    cudaFree(d_J_evap);
}
```

---

### Test 9.2: Evaporation Cooling Power

**Objective:** Verify that evaporation power P_evap = J * L_vap * A is correctly computed.

**Validation:**
```cpp
TEST_F(ThermalLBMTest, EvaporationCoolingPower) {
    MaterialProperties mat = MaterialDatabase::getTi6Al4V();
    ThermalLBM solver(nx, ny, nz, mat, alpha, false, dt, dx);
    solver.initialize(4000.0f);  // Above T_boil

    float* d_fill = createTopSurfaceFill(nx, ny, nz);

    float P_evap = solver.computeEvaporationPower(d_fill, dx);

    // Power should be positive (cooling)
    EXPECT_GT(P_evap, 0.0f) << "No evaporation cooling";

    // Order of magnitude check (for typical LPBF)
    EXPECT_LT(P_evap, 1000.0f) << "Evaporation power unphysically high";

    std::cout << "Evaporation power = " << P_evap << " W" << std::endl;

    cudaFree(d_fill);
}
```

---

## Test Suite 10: Kernel-Level Unit Tests

### Test 10.1: Equilibrium Distribution Computation

**Objective:** Verify D3Q7::computeThermalEquilibrium() correctness.

**Validation:**
```cpp
TEST(D3Q7Test, ThermalEquilibriumDistribution) {
    // At rest (u = 0)
    float T = 1000.0f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;

    float g_eq[7];
    for (int q = 0; q < 7; ++q) {
        g_eq[q] = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
    }

    // Sum should equal temperature
    float sum = 0.0f;
    for (int q = 0; q < 7; ++q) {
        sum += g_eq[q];
    }
    EXPECT_NEAR(sum, T, 1e-5f);

    // Weights: w0=1/4, w1-6=1/8
    EXPECT_NEAR(g_eq[0], T * 0.25f, 1e-5f);
    for (int q = 1; q < 7; ++q) {
        EXPECT_NEAR(g_eq[q], T * 0.125f, 1e-5f);
    }

    // With velocity
    ux = 0.1f;
    for (int q = 0; q < 7; ++q) {
        g_eq[q] = D3Q7::computeThermalEquilibrium(q, T, ux, uy, uz);
    }

    // Sum should still equal temperature
    sum = 0.0f;
    for (int q = 0; q < 7; ++q) {
        sum += g_eq[q];
    }
    EXPECT_NEAR(sum, T, 1e-5f);
}
```

---

### Test 10.2: Temperature Clamping

**Objective:** Verify computeTemperatureKernel clamps to [T_MIN, T_MAX].

**Validation:**
```cpp
TEST_F(ThermalLBMTest, TemperatureClamping) {
    ThermalLBM solver(10, 10, 10, mat, alpha, false, dt, dx);

    // Try to initialize with extreme temperatures
    std::vector<float> T_extreme(1000);
    for (int i = 0; i < 500; ++i) {
        T_extreme[i] = -100.0f;  // Below T_MIN
    }
    for (int i = 500; i < 1000; ++i) {
        T_extreme[i] = 20000.0f;  // Above T_MAX
    }

    solver.initialize(T_extreme.data());
    solver.computeTemperature();

    std::vector<float> T_result(1000);
    solver.copyTemperatureToHost(T_result.data());

    // Check clamping
    for (int i = 0; i < 1000; ++i) {
        EXPECT_GE(T_result[i], 0.0f);
        EXPECT_LE(T_result[i], 7000.0f);
    }
}
```

---

## Test Execution Summary

### Recommended Test Sequence

1. **Kernel Unit Tests** (Test Suite 10)
   - Fast, deterministic
   - Catch low-level bugs early

2. **Pure Conduction** (Test Suite 1)
   - Fundamental physics validation
   - Establishes baseline accuracy

3. **Boundary Conditions** (Test Suite 6)
   - Critical for all other tests
   - Must work correctly first

4. **Energy Conservation** (Test Suite 5)
   - Cross-cutting validation
   - Reveals subtle bugs

5. **Phase Change** (Test Suite 8)
   - Complex physics
   - Depends on previous suites

6. **Stability Tests** (Test Suite 7)
   - Edge cases and robustness
   - Final validation before deployment

### Performance Metrics

Expected runtime (NVIDIA RTX 3090):
- Unit tests: < 5 seconds
- Integration tests: < 2 minutes
- Full suite: < 10 minutes

### CI/CD Integration

```bash
# Run all thermal tests
cd /home/yzk/LBMProject/build
ctest -R "ThermalLBM*" --output-on-failure

# Run specific test suite
./test_thermal_conduction
./test_thermal_phase_change
./test_thermal_boundaries
```

---

## Appendix A: Helper Functions

### A.1: Common Test Utilities

```cpp
// Compute L2 error
float computeL2Error(const std::vector<float>& num,
                     const std::vector<float>& ana) {
    float sse = 0.0f, ssa = 0.0f;
    for (size_t i = 0; i < num.size(); ++i) {
        float err = num[i] - ana[i];
        sse += err * err;
        ssa += ana[i] * ana[i];
    }
    return sqrt(sse / ssa);
}

// Find peak position
int findPeakPosition(const std::vector<float>& T) {
    int peak_idx = 0;
    float peak_val = T[0];
    for (size_t i = 1; i < T.size(); ++i) {
        if (T[i] > peak_val) {
            peak_val = T[i];
            peak_idx = i;
        }
    }
    return peak_idx;
}

// Detect oscillations
bool detectOscillations(const std::vector<float>& T) {
    int sign_changes = 0;
    for (size_t i = 2; i < T.size(); ++i) {
        float d2T = T[i] - 2*T[i-1] + T[i-2];
        if (i > 2) {
            float d2T_prev = T[i-1] - 2*T[i-2] + T[i-3];
            if (d2T * d2T_prev < 0) sign_changes++;
        }
    }
    return sign_changes > 10;  // Threshold for noise
}

// Check monotonicity
bool checkMonotonicity(const std::vector<float>& T) {
    for (size_t i = 1; i < T.size(); ++i) {
        if (T[i] > T[i-1]) return false;  // Expect decreasing
    }
    return true;
}
```

### A.2: Analytical Solutions

```cpp
// 1D Gaussian diffusion
float gaussianDiffusion(float x, float t, float x0, float sigma0,
                       float T_ambient, float T_peak, float alpha) {
    float sigma_t = sqrt(sigma0*sigma0 + 2*alpha*t);
    float spatial = exp(-(x-x0)*(x-x0) / (2*sigma_t*sigma_t));
    return T_ambient + (T_peak - T_ambient) * (sigma0/sigma_t) * spatial;
}

// Linear steady-state profile
float linearProfile(float x, float L, float T_cold, float T_hot) {
    return T_cold + (T_hot - T_cold) * x / L;
}

// Stefan problem interface position
float stefanInterface(float t, float Ste, float alpha) {
    // Simplified: lambda from Stefan number
    float lambda = Ste / 2.0f;  // Approximation
    return 2 * lambda * sqrt(alpha * t);
}
```

---

## Appendix B: Material Properties for Testing

```cpp
// Ti-6Al-4V (default test material)
MaterialProperties mat = MaterialDatabase::getTi6Al4V();
// T_solidus = 1878 K
// T_liquidus = 1943 K
// T_vaporization = 3560 K
// L_fusion = 286,000 J/kg
// L_vaporization = 9,830,000 J/kg
// k_solid = 21.9 W/(m·K)
// rho_solid = 4430 kg/m³
// cp_solid = 546 J/(kg·K)

// Simplified test material (no phase change)
MaterialProperties simple_mat;
simple_mat.rho_solid = 8000.0f;
simple_mat.cp_solid = 500.0f;
simple_mat.k_solid = 20.0f;
simple_mat.T_solidus = 1e6f;  // Effectively disable phase change
simple_mat.T_liquidus = 1e6f + 1.0f;
```

---

## Appendix C: Debugging Checklist

When a test fails, check:

1. **Unit conversion:**
   - Is alpha in physical units (m²/s) or lattice units?
   - Is dt computed correctly from CFL condition?
   - Are temperatures in Kelvin?

2. **Boundary conditions:**
   - Are BCs applied in correct order?
   - Does streaming handle boundaries properly?

3. **Energy conservation:**
   - Check T_ref for energy calculation
   - Verify no energy leaks at boundaries

4. **Phase change:**
   - Is phase change solver initialized?
   - Is latent heat correction applied after computeTemperature()?

5. **Stability:**
   - Check omega value (should be ≤ 1.85)
   - Verify Peclet number is reasonable
   - Look for NaN or Inf in output

---

## Document Version
- Version: 1.0
- Date: 2025-12-03
- Author: Claude Code (Testing Specialist)
- Project: LBM-CUDA CFD Framework
