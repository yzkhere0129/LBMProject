# VOF Solver Test Specifications

**Project:** LBM-CUDA CFD Framework for Metal AM
**Module:** Volume of Fluid (VOF) Solver
**Author:** CFD Test Architecture Team
**Date:** 2025-12-03

---

## Overview

This document specifies comprehensive standalone tests for the VOF (Volume of Fluid) solver module. Each test validates a specific physical phenomenon or numerical property with clearly defined expected behavior and acceptance criteria.

**VOF Module Components:**
- **Advection:** `/home/yzk/LBMProject/src/physics/vof/vof_solver.cu`
- **Surface Tension:** `/home/yzk/LBMProject/src/physics/vof/surface_tension.cu`
- **Marangoni Effect:** `/home/yzk/LBMProject/src/physics/vof/marangoni.cu`
- **Recoil Pressure:** `/home/yzk/LBMProject/src/physics/vof/recoil_pressure.cu`

---

## Test 1: Zalesak's Disk (Advection Accuracy)

### Purpose
Validate VOF advection accuracy using a standard benchmark with sharp gradients and complex geometry.

### Physical Description
Zalesak's disk is a slotted circle (notched disk) advected in a rotating velocity field. After one full rotation (360°), the disk should return to its original position and shape. This is a canonical test for interface-capturing schemes.

### Setup Parameters

**Domain:**
- Grid: 100 × 100 × 1 cells (2D simulation)
- Physical size: 200 μm × 200 μm × 2 μm
- Grid spacing: dx = 2 μm

**Initial Condition:**
- Disk center: (50 μm, 75 μm)
- Disk radius: R = 15 μm
- Slot width: 5 μm
- Slot depth: 25 μm (extends from bottom of disk upward)
- Slot centered at x = 50 μm

**Velocity Field (Solid Body Rotation):**
```
u(x,y) = -ω(y - y_center)
v(x,y) =  ω(x - x_center)
```
where:
- ω = 2π / T_period (angular velocity)
- (x_center, y_center) = (100 μm, 100 μm)
- T_period = 2000 time steps

**Numerical Parameters:**
- Time step: dt = 1e-7 s
- Total time: 2000 steps (one full rotation)
- CFL number: max(u,v) × dt / dx ≈ 0.4

### Expected Behavior

**Shape Preservation:**
- After 360° rotation, disk returns to original position
- Shape error E_shape < 5% (see metric below)
- Slot remains visible and sharp

**Mass Conservation:**
- Total mass M = Σ f_i remains constant
- Relative mass error: |M_final - M_initial| / M_initial < 0.5%

**Interface Sharpness:**
- Interface thickness should remain ~2-3 cells (diffusion limited)
- No excessive smearing or staircasing

### Validation Metrics

**1. Shape Error:**
```
E_shape = Σ|f_final(x,y) - f_initial(x,y)| / Σf_initial(x,y)
```

**2. Centroid Error:**
```
x_c = Σ(x_i × f_i) / Σf_i
y_c = Σ(y_i × f_i) / Σf_i
ΔC = √[(x_c,final - x_c,initial)² + (y_c,final - y_c,initial)²]
```
Acceptance: ΔC < 1 cell

**3. Mass Conservation:**
```
ΔM_rel = |M_final - M_initial| / M_initial
```
Acceptance: ΔM_rel < 0.005

### Implementation Guidelines

**Initialization:**
```cpp
for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
        float x = i * dx;
        float y = j * dx;
        float x_rel = x - x_disk;
        float y_rel = y - y_disk;
        float r = sqrt(x_rel*x_rel + y_rel*y_rel);

        // Inside disk
        bool in_disk = (r < R_disk);

        // Inside slot
        bool in_slot = (abs(x_rel) < slot_width/2) && (y_rel < 0) && (y_rel > -slot_depth);

        // Disk with slot removed
        fill_level[idx] = (in_disk && !in_slot) ? 1.0f : 0.0f;
    }
}
```

**Velocity Field:**
```cpp
for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
        float x = i * dx;
        float y = j * dx;
        ux[idx] = -omega * (y - y_center);
        uy[idx] =  omega * (x - x_center);
        uz[idx] = 0.0f;
    }
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_zalesak_disk.cu`

---

## Test 2: Spherical Droplet (Surface Tension - Laplace Pressure)

### Purpose
Validate surface tension implementation by verifying the Laplace pressure jump across a spherical interface.

### Physical Description
A spherical droplet in equilibrium with surrounding gas experiences a pressure jump across its interface given by the Young-Laplace equation:
```
ΔP = P_liquid - P_gas = 2σ / R
```
For σ = 1.5 N/m (Ti6Al4V at 2000 K) and R = 10 μm:
```
ΔP = 2 × 1.5 / (10e-6) = 300,000 Pa = 0.3 MPa
```

### Setup Parameters

**Domain:**
- Grid: 64 × 64 × 64 cells
- Physical size: 128 μm × 128 μm × 128 μm
- Grid spacing: dx = 2 μm

**Droplet:**
- Center: (64 μm, 64 μm, 64 μm)
- Radius: R = 10 μm (5 cells)
- Surface tension: σ = 1.5 N/m

**Fluid Properties:**
- Liquid density: ρ_l = 4110 kg/m³ (Ti6Al4V)
- Gas density: ρ_g = 1.0 kg/m³
- Viscosity: μ = 0.005 Pa·s (both phases)

**Numerical Parameters:**
- Time step: dt = 1e-8 s (small for stability)
- Total time: 1000 steps (allow equilibration)
- Initial velocity: u = v = w = 0 (quiescent)

### Expected Behavior

**Pressure Jump:**
- Theoretical: ΔP = 2σ/R = 300,000 Pa
- Measured: Average pressure difference across interface
- Tolerance: ±10% (numerical diffusion affects interface sharpness)

**Droplet Shape:**
- Remains spherical (aspect ratio < 1.05)
- Radius unchanged: |R_final - R_initial| < 0.5 cells

**Spurious Currents:**
- Parasitic velocities due to CSF errors
- Maximum velocity: u_max < 0.01 m/s (< 1% of capillary velocity)
- Capillary velocity scale: U_cap = σ / μ = 300 m/s

### Validation Metrics

**1. Pressure Jump Measurement:**
```
P_liquid_avg = average pressure at r < R/2 (droplet center)
P_gas_avg = average pressure at r > 2R (far field)
ΔP_measured = P_liquid_avg - P_gas_avg
error = |ΔP_measured - ΔP_theory| / ΔP_theory
```
Acceptance: error < 0.15

**2. Sphericity:**
```
R_x = radius along x-axis
R_y = radius along y-axis
R_z = radius along z-axis
sphericity = max(R_x, R_y, R_z) / min(R_x, R_y, R_z)
```
Acceptance: sphericity < 1.05

**3. Spurious Currents:**
```
u_max = max(|u|, |v|, |w|) over all cells
U_cap = σ / μ
ratio = u_max / U_cap
```
Acceptance: ratio < 0.01

### Implementation Guidelines

**Pressure Extraction (via Momentum Equation):**

In LBM, pressure is related to density:
```cpp
float computePressure(float rho, float cs2) {
    return rho * cs2;  // cs2 = (1/3) for D3Q19
}
```

For VOF with CSF, measure pressure from LBM macroscopic fields after equilibration.

**Radial Averaging:**
```cpp
float averagePressureInRegion(const float* pressure, float r_min, float r_max,
                                float3 center, int nx, int ny, int nz, float dx) {
    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = i * dx - center.x;
                float y = j * dx - center.y;
                float z = k * dx - center.z;
                float r = sqrt(x*x + y*y + z*z);

                if (r >= r_min && r <= r_max) {
                    int idx = i + nx * (j + ny * k);
                    sum += pressure[idx];
                    count++;
                }
            }
        }
    }
    return sum / count;
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_laplace_pressure.cu`

---

## Test 3: Oscillating Droplet (Dynamic Surface Tension)

### Purpose
Validate dynamic surface tension effects by measuring the oscillation frequency of a perturbed droplet.

### Physical Description
An ellipsoidal droplet oscillates between oblate and prolate shapes due to surface tension restoring forces. The oscillation frequency for mode n=2 (ellipsoidal) is:
```
f = (1/2π) × √[n(n-1)(n+2)σ / (ρR³)]
```
For n=2:
```
f = (1/2π) × √[8σ / (ρR³)]
```

For Ti6Al4V: σ = 1.5 N/m, ρ = 4110 kg/m³, R = 10 μm:
```
f = (1/2π) × √[8 × 1.5 / (4110 × (10e-6)³)]
  = (1/2π) × √[2.92e9]
  = 8600 Hz
  ≈ 8.6 kHz
```
Period: T = 1/f ≈ 116 μs

### Setup Parameters

**Domain:**
- Grid: 64 × 64 × 64 cells
- Physical size: 128 μm × 128 μm × 128 μm
- Grid spacing: dx = 2 μm

**Initial Droplet (Ellipsoid):**
- Center: (64 μm, 64 μm, 64 μm)
- Semi-axes: a = 12 μm (x), b = 10 μm (y), c = 10 μm (z)
- Equivalent spherical radius: R = 10 μm
- Perturbation: 20% elongation along x-axis

**Fluid Properties:**
- Density: ρ = 4110 kg/m³
- Surface tension: σ = 1.5 N/m
- Viscosity: μ = 0.001 Pa·s (low viscosity for multiple oscillations)

**Numerical Parameters:**
- Time step: dt = 1e-8 s
- Total time: 500 μs (≈4 periods)
- Output frequency: every 10 steps (1 μs)

### Expected Behavior

**Oscillation Frequency:**
- Theoretical: f = 8600 Hz
- Measured: Peak frequency from FFT of aspect ratio time series
- Tolerance: ±15% (viscous damping affects frequency)

**Damping:**
- Amplitude decays exponentially: A(t) = A₀ exp(-γt)
- Damping coefficient: γ ≈ μ / (ρR²)
- For μ = 0.001 Pa·s: γ ≈ 2400 s⁻¹

**Shape Evolution:**
- Smooth oscillation between oblate (a < b,c) and prolate (a > b,c)
- No fragmentation or instability

### Validation Metrics

**1. Frequency Measurement:**
```
aspect_ratio(t) = max(R_x, R_y, R_z) / min(R_x, R_y, R_z)
FFT[aspect_ratio(t)] → frequency spectrum
f_measured = frequency of dominant peak
error_freq = |f_measured - f_theory| / f_theory
```
Acceptance: error_freq < 0.15

**2. Damping Rate:**
```
Extract amplitude envelope: A(t) = max(aspect_ratio) - 1
Fit exponential: A(t) = A₀ exp(-γt)
γ_measured from fit
γ_theory = μ / (ρR²)
error_damping = |γ_measured - γ_theory| / γ_theory
```
Acceptance: error_damping < 0.3 (qualitative check)

**3. Volume Conservation:**
```
V(t) = Σf_i × dx³
ΔV_rel = |V(t) - V(0)| / V(0)
```
Acceptance: ΔV_rel < 0.01 for all t

### Implementation Guidelines

**Ellipsoidal Initialization:**
```cpp
void initializeEllipsoid(float* fill_level, float3 center,
                         float a, float b, float c,
                         int nx, int ny, int nz, float dx) {
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                float x = i * dx - center.x;
                float y = j * dx - center.y;
                float z = k * dx - center.z;

                float dist = sqrt((x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c));

                // Smooth interface
                float interface_width = 2.0f * dx;
                fill_level[idx] = 0.5f * (1.0f - tanh((dist - 1.0f) / (interface_width / a)));
            }
        }
    }
}
```

**Aspect Ratio Tracking:**
```cpp
struct DropletMoments {
    float x_cm, y_cm, z_cm;  // Center of mass
    float Ixx, Iyy, Izz;     // Moments of inertia
    float R_x, R_y, R_z;     // Principal radii
};

DropletMoments computeMoments(const float* fill_level, int nx, int ny, int nz, float dx);
```

**FFT Analysis:**
Use cuFFT or host-side FFTW for frequency extraction.

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_oscillating_droplet.cu`

---

## Test 4: Thermocapillary Migration (Marangoni Effect)

### Purpose
Validate Marangoni force implementation by measuring droplet migration in a temperature gradient.

### Physical Description
A liquid droplet in a temperature gradient migrates due to surface tension gradients (thermocapillary effect). The migration velocity for a spherical droplet in creeping flow is given by the Young-Goldstein-Block (YGB) theory:
```
U_migr = (2/3) × |dσ/dT| × |∇T| × R / μ
```

For Ti6Al4V:
- dσ/dT = -0.26e-3 N/(m·K)
- Temperature gradient: ∇T = 1e6 K/m (100 K across 100 μm)
- Droplet radius: R = 10 μm
- Viscosity: μ = 0.005 Pa·s

Expected velocity:
```
U_migr = (2/3) × 0.26e-3 × 1e6 × 10e-6 / 0.005
       = 0.347 m/s ≈ 35 cm/s
```

### Setup Parameters

**Domain:**
- Grid: 100 × 50 × 50 cells
- Physical size: 200 μm × 100 μm × 100 μm
- Grid spacing: dx = 2 μm

**Temperature Field:**
- Linear gradient along x-axis
- T(x) = T₀ + (ΔT/L) × x
- T₀ = 2000 K (left boundary)
- ΔT = 100 K (across domain)
- ∇T = ΔT / L = 100 / (200e-6) = 5e5 K/m

**Droplet:**
- Initial center: (50 μm, 50 μm, 50 μm)
- Radius: R = 10 μm
- Initially at rest

**Material Properties:**
- Surface tension: σ(T) = σ₀ - |dσ/dT| × (T - T₀)
- dσ/dT = -0.26e-3 N/(m·K)
- Viscosity: μ = 0.005 Pa·s
- Density: ρ = 4110 kg/m³

**Numerical Parameters:**
- Time step: dt = 1e-8 s
- Total time: 10 μs (allow steady migration to establish)
- Marangoni gradient limiter: 5e8 K/m (safety factor)

### Expected Behavior

**Migration Velocity:**
- Theory (YGB): U = 0.347 m/s (for ∇T = 5e5 K/m)
- Direction: Toward cooler region (negative x if dσ/dT < 0)
- Tolerance: ±30% (YGB assumes Stokes flow, creeping conditions)

**Velocity Field Pattern:**
- Internal recirculation inside droplet
- Flow from hot to cold along interface (surface flow)
- Flow from cold to hot in droplet core (return flow)

**Steady State:**
- Migration velocity reaches constant value after ~5 μs
- Droplet shape remains spherical (no significant deformation)

### Validation Metrics

**1. Migration Velocity:**
```
Track droplet centroid position:
x_cm(t) = Σ(x_i × f_i) / Σf_i

Velocity from finite difference:
U_migr(t) = [x_cm(t+Δt) - x_cm(t)] / Δt

Average over steady state (t > 5 μs):
U_avg = mean(U_migr(t)) for t ∈ [5μs, 10μs]

U_theory = (2/3) × |dσ/dT| × |∇T| × R / μ
error = |U_avg - U_theory| / U_theory
```
Acceptance: error < 0.3

**2. Flow Pattern Check:**
```
Sample interface velocity magnitude at top of droplet (hot side):
u_interface_hot = velocity at (x_cm, y_cm + R, z_cm)

Sample core velocity at droplet center:
u_core = velocity at (x_cm, y_cm, z_cm)

Expected: u_interface_hot and u_core should have opposite signs (recirculation)
```

**3. Shape Preservation:**
```
sphericity = max(R_x, R_y, R_z) / min(R_x, R_y, R_z)
```
Acceptance: sphericity < 1.1

### Implementation Guidelines

**Temperature Field Initialization:**
```cpp
__global__ void setLinearTemperatureGradient(float* temperature,
                                              float T0, float dT_dx,
                                              int nx, int ny, int nz, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);
    float x = i * dx;
    temperature[idx] = T0 + dT_dx * x;
}
```

**Centroid Tracking:**
```cpp
float3 computeCentroid(const float* fill_level, int nx, int ny, int nz, float dx) {
    float sum_fx = 0.0f, sum_fy = 0.0f, sum_fz = 0.0f;
    float sum_f = 0.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_level[idx];
                sum_fx += f * i * dx;
                sum_fy += f * j * dx;
                sum_fz += f * k * dx;
                sum_f += f;
            }
        }
    }

    return make_float3(sum_fx / sum_f, sum_fy / sum_f, sum_fz / sum_f);
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_thermocapillary_migration.cu`

---

## Test 5: Contact Angle (Wall Wetting)

### Purpose
Validate contact angle boundary condition implementation.

### Physical Description
A liquid droplet on a solid surface forms a static contact angle θ determined by the Young equation:
```
σ_sv - σ_sl = σ_lv × cos(θ)
```
where σ_sv, σ_sl, σ_lv are solid-vapor, solid-liquid, and liquid-vapor surface tensions.

In VOF simulations, contact angle is imposed by adjusting the interface normal at the wall.

### Setup Parameters

**Domain:**
- Grid: 64 × 64 × 32 cells (horizontal substrate)
- Physical size: 128 μm × 128 μm × 64 μm
- Grid spacing: dx = 2 μm
- Bottom wall: z = 0 (substrate)

**Droplet:**
- Initial: Spherical droplet just touching substrate
- Center: (64 μm, 64 μm, 10 μm)
- Radius: R = 10 μm

**Contact Angles to Test:**
- θ = 30° (hydrophilic)
- θ = 90° (neutral)
- θ = 150° (hydrophobic)

**Numerical Parameters:**
- Time step: dt = 1e-8 s
- Relaxation time: 20 μs (allow droplet to equilibrate)
- Surface tension: σ = 1.5 N/m

### Expected Behavior

**Equilibrium Shape:**
- Droplet deforms to satisfy contact angle at wall
- Spherical cap geometry: height h and base radius r_base satisfy:
  ```
  θ = arctan(h / r_base)  (approximate for small caps)
  ```

**Contact Line:**
- Three-phase contact line is circular
- Contact angle measured consistently around perimeter

**Volume Conservation:**
- Total volume unchanged during relaxation
- |V_final - V_initial| / V_initial < 1%

### Validation Metrics

**1. Contact Angle Measurement:**

Method: Fit circular arc to interface near wall and measure tangent angle.

```
For cells at z=1 (just above substrate):
  - Extract interface contour (f ≈ 0.5)
  - Fit circle to contour points
  - Compute tangent angle at contact point
  - Average around contact line perimeter

θ_measured = average tangent angle at wall
error = |θ_measured - θ_input|
```
Acceptance: error < 5°

**2. Axial Symmetry:**
```
For each radial distance r from droplet center:
  - Measure height h(r,φ) for all azimuthal angles φ
  - Compute standard deviation σ_h(r)

symmetry_error = max(σ_h(r)) / h_max
```
Acceptance: symmetry_error < 0.05

**3. Volume Conservation:**
```
V_initial = Σf_i × dx³ (before relaxation)
V_final = Σf_i × dx³ (after relaxation)
ΔV_rel = |V_final - V_initial| / V_initial
```
Acceptance: ΔV_rel < 0.01

### Implementation Guidelines

**Contact Angle Measurement:**
```cpp
float measureContactAngle(const float* fill_level, int nx, int ny, int nz, float dx,
                           float3 droplet_center) {
    // Sample at z=1 (one cell above substrate)
    int k_sample = 1;

    std::vector<float> angles;

    // Loop around contact line
    for (int theta_deg = 0; theta_deg < 360; theta_deg += 10) {
        float theta = theta_deg * M_PI / 180.0f;

        // Trace radially from center at this angle
        for (float r = 0; r < 20 * dx; r += dx) {
            int i = static_cast<int>((droplet_center.x + r * cos(theta)) / dx);
            int j = static_cast<int>((droplet_center.y + r * sin(theta)) / dx);

            if (i < 0 || i >= nx || j < 0 || j >= ny) continue;

            // Find interface crossing (f ≈ 0.5)
            int idx = i + nx * (j + ny * k_sample);
            if (fill_level[idx] > 0.4f && fill_level[idx] < 0.6f) {
                // Compute local normal
                float3 normal = computeNormal(fill_level, i, j, k_sample, nx, ny, nz, dx);

                // Contact angle from normal (angle between normal and horizontal plane)
                float angle = acos(fabs(normal.z));  // Angle from vertical
                angle = M_PI / 2.0f - angle;  // Convert to angle from horizontal
                angles.push_back(angle * 180.0f / M_PI);
                break;
            }
        }
    }

    // Return average
    return std::accumulate(angles.begin(), angles.end(), 0.0f) / angles.size();
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_contact_angle_static.cu`

---

## Test 6: Evaporation Mass Loss

### Purpose
Validate VOF-thermal coupling for evaporative mass loss.

### Physical Description
When liquid evaporates, the VOF fill level decreases according to:
```
df/dt = -J_evap / (ρ × dx)
```
where:
- J_evap: evaporation mass flux [kg/(m²·s)]
- ρ: liquid density [kg/m³]
- dx: grid spacing [m]

For constant evaporation flux, the fill level decreases linearly with time.

### Setup Parameters

**Domain:**
- Grid: 32 × 32 × 32 cells
- Physical size: 64 μm × 64 μm × 64 μm
- Grid spacing: dx = 2 μm

**Initial Configuration:**
- Uniform liquid layer: f = 1.0 for z < 20 cells
- Gas above: f = 0.0 for z ≥ 20 cells
- Flat interface at z = 20 cells (40 μm)

**Evaporation:**
- Constant flux: J_evap = 10 kg/(m²·s) (applied at interface cells)
- Liquid density: ρ = 4110 kg/m³
- Expected df/dt = -10 / (4110 × 2e-6) = -1.216e3 s⁻¹

**Numerical Parameters:**
- Time step: dt = 1e-7 s
- Total time: 10 μs (100 steps)
- Expected total mass loss: ΔM = J_evap × A_interface × t

### Expected Behavior

**Fill Level Evolution:**
- Interface cells (z=20) lose mass according to df = df/dt × dt
- After 100 steps: Δf_total = -1.216e3 × 1e-7 × 100 = -0.01216 ≈ -1.2%
- Interface moves downward as material evaporates

**Mass Conservation:**
- Total mass lost matches evaporative flux:
  ```
  ΔM_theory = J_evap × A_interface × t
  ΔM_measured = M_initial - M_final
  ```

### Validation Metrics

**1. Mass Loss Rate:**
```
M(t) = Σf_i × ρ × dx³
dM/dt = [M(t+dt) - M(t)] / dt

Measure over t ∈ [1μs, 10μs] (skip transient):
dM/dt_measured = average rate

Expected:
dM/dt_theory = -J_evap × A_interface
where A_interface ≈ nx × ny × dx² (flat interface)

error = |dM/dt_measured - dM/dt_theory| / |dM/dt_theory|
```
Acceptance: error < 0.1

**2. Interface Position:**
```
Track interface height (z-coordinate where f ≈ 0.5):
z_interface(t) = find z where f ≈ 0.5

Velocity of interface regression:
v_interface = dz/dt

Expected from mass balance:
v_interface = -J_evap / ρ = -10 / 4110 = -2.43e-3 m/s = -2.43 mm/s
```

**3. Stability Check:**
- All fill levels remain in [0, 1]
- No NaN or Inf values
- Evaporation limiter (2% per step) prevents numerical blowup

### Implementation Guidelines

**Constant Flux Kernel:**
```cpp
__global__ void applyConstantEvaporationFlux(float* fill_level,
                                              float J_evap,
                                              float rho, float dx, float dt,
                                              int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    float f = fill_level[idx];

    // Only apply at interface cells
    if (f < 0.01f || f > 0.99f) return;

    // Compute fill level change
    float df = -J_evap * dt / (rho * dx);

    // Apply limiter (max 2% per step)
    const float MAX_DF = 0.02f * f;
    if (df < -MAX_DF) df = -MAX_DF;

    fill_level[idx] = fmaxf(0.0f, f + df);
}
```

**Test File:** `/home/yzk/LBMProject/tests/unit/vof/test_vof_evaporation_mass_loss.cu` (already exists)

---

## Test 7: Recoil Pressure (Keyhole Formation)

### Purpose
Validate recoil pressure force implementation for high-temperature evaporation.

### Physical Description
At high temperatures (T > T_boil), rapid evaporation generates recoil pressure that pushes the liquid surface downward:
```
P_recoil = C_r × P_sat(T)
P_sat(T) = P_ref × exp[(L_vap × M / R) × (1/T_boil - 1/T)]
```
where:
- C_r = 0.54 (recoil coefficient)
- P_ref = 101325 Pa (atmospheric pressure)
- L_vap = 8.878e6 J/kg (latent heat, Ti6Al4V)
- M = 0.0479 kg/mol (molar mass)
- R = 8.314 J/(mol·K) (gas constant)
- T_boil = 3560 K

For T = 4000 K:
```
P_sat = 101325 × exp[51096 × (1/3560 - 1/4000)]
      = 101325 × exp[1.57]
      = 486,000 Pa
P_recoil = 0.54 × 486,000 = 262,000 Pa ≈ 0.26 MPa
```

### Setup Parameters

**Domain:**
- Grid: 64 × 64 × 64 cells
- Physical size: 128 μm × 128 μm × 128 μm
- Grid spacing: dx = 2 μm

**Initial Configuration:**
- Flat liquid surface at z = 32 cells (64 μm)
- Liquid below (f=1), gas above (f=0)

**Temperature:**
- Hot spot at center of surface: T = 4000 K
- Gaussian profile: T(r) = T₀ + ΔT × exp(-r²/w²)
  - T₀ = 2000 K (background)
  - ΔT = 2000 K (peak excess)
  - w = 10 μm (spot width)
- Temperature above T_activation = 3060 K triggers recoil

**Material Properties:**
- Density: ρ = 4110 kg/m³
- Viscosity: μ = 0.003 Pa·s (low for faster response)
- Surface tension: σ = 1.5 N/m

**Numerical Parameters:**
- Time step: dt = 1e-9 s (small for stability with large force)
- Total time: 5 μs (observe initial depression)
- Recoil coefficient: C_r = 0.54
- Pressure limiter: P_max = 1e8 Pa (100 MPa)

### Expected Behavior

**Surface Depression:**
- At hot spot center, recoil force pushes surface downward
- Depression depth increases with time
- Typical depth after 5 μs: ~2-5 cells (qualitative, depends on flow dynamics)

**Pressure Distribution:**
- P_recoil peaks at hot spot center
- Decays radially following temperature profile
- No recoil outside hot region (T < T_activation)

**Force Direction:**
- Force points downward (into liquid) at interface
- Magnitude: F = -P_recoil × |∇f| × n [N/m³]

### Validation Metrics

**1. Recoil Pressure Magnitude:**
```
At hot spot center (x=64μm, y=64μm, z=64μm):
T_center = 4000 K
P_theory = C_r × P_sat(T_center) = 262,000 Pa

P_measured = recoil pressure computed by kernel

error = |P_measured - P_theory| / P_theory
```
Acceptance: error < 0.05 (Clausius-Clapeyron is exact)

**2. Force Application:**
```
At interface cells with T > T_activation:
F_z should be negative (downward)

Check force magnitude:
F_theory = P_recoil × |∇f|
F_measured = force_z[idx]  (from kernel output)

Qualitative check: F_measured should have correct sign and order of magnitude
```

**3. Surface Response:**
```
Track interface height at center:
z_interface(t) = z-coordinate where f ≈ 0.5

After 5 μs:
Δz = z_interface(5μs) - z_interface(0)

Expected: Δz < 0 (depression)
Magnitude: |Δz| = 2-10 cells (depends on flow solver coupling)
```

### Implementation Guidelines

**Temperature Field Setup:**
```cpp
__global__ void setGaussianHotSpot(float* temperature,
                                    float T0, float dT, float w,
                                    float3 center,
                                    int nx, int ny, int nz, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    float x = i * dx - center.x;
    float y = j * dx - center.y;
    float z = k * dx - center.z;

    float r = sqrt(x*x + y*y);  // Radial distance in xy-plane

    // Gaussian profile (applied at surface layer only)
    if (k >= 30 && k <= 34) {  // Near z=32 interface
        temperature[idx] = T0 + dT * exp(-r*r / (w*w));
    } else {
        temperature[idx] = T0;
    }
}
```

**Recoil Pressure Verification:**
```cpp
// Test saturation pressure calculation directly
float testSaturationPressure() {
    float T = 4000.0f;
    float T_boil = 3560.0f;
    float P_ref = 101325.0f;
    float CC_factor = 51096.0f;  // L_vap * M / R

    float exponent = CC_factor * (1.0f/T_boil - 1.0f/T);
    float P_sat = P_ref * exp(exponent);
    float P_recoil = 0.54f * P_sat;

    printf("T = %.1f K\n", T);
    printf("P_sat = %.1f Pa\n", P_sat);
    printf("P_recoil = %.1f Pa (%.2f MPa)\n", P_recoil, P_recoil / 1e6);

    return P_recoil;
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_recoil_pressure_depression.cu`

---

## Test 8: Interface Reconstruction (PLIC - Piecewise Linear Interface Calculation)

### Purpose
Validate interface normal computation accuracy.

### Physical Description
VOF uses central differences to compute interface normals:
```
n = -∇f / |∇f|
```
For known interface geometries (plane, sphere, cylinder), the computed normal should match the analytical normal.

### Setup Parameters

**Test Cases:**

**Case 8A: Planar Interface**
- Domain: 32 × 32 × 32 cells
- Interface: Vertical plane at x = 16
- Expected normal: n = (-1, 0, 0) everywhere at interface

**Case 8B: Spherical Interface**
- Domain: 64 × 64 × 64 cells
- Sphere center: (32, 32, 32) cells
- Sphere radius: R = 10 cells
- Expected normal: n = -(r - r_c) / |r - r_c| (radial, outward from liquid)

**Case 8C: Cylindrical Interface**
- Domain: 64 × 64 × 32 cells
- Cylinder axis: z-axis through (32, 32, z)
- Cylinder radius: R = 10 cells
- Expected normal: n_x = -(x-32)/R, n_y = -(y-32)/R, n_z = 0

### Expected Behavior

**Accuracy:**
- Angular error: |θ_error| < 5° for cells within 2-4 cells of interface
- At exact interface (f ≈ 0.5): error < 2°
- In bulk (f ≈ 0 or f ≈ 1): normal should be ~zero (no interface)

**Consistency:**
- Normal magnitude: |n| = 1 at interface cells
- Normal direction: Points from liquid to gas

### Validation Metrics

**Angular Error:**
```
For each interface cell (0.1 < f < 0.9):
  n_computed = interface_normal[idx]
  n_exact = analyticalNormal(position)

  cos_angle = dot(n_computed, n_exact) / (|n_computed| × |n_exact|)
  angle_error = acos(cos_angle) × 180 / π

Metrics:
- mean_error = average(angle_error)
- max_error = max(angle_error)
- rms_error = sqrt(mean(angle_error²))
```

Acceptance:
- mean_error < 5°
- max_error < 10°
- rms_error < 6°

### Implementation Guidelines

**Analytical Normals:**
```cpp
float3 analyticalNormalSphere(float3 position, float3 center) {
    float3 r = make_float3(position.x - center.x,
                           position.y - center.y,
                           position.z - center.z);
    float mag = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);

    // Normal points from liquid (inside) to gas (outside)
    return make_float3(-r.x / mag, -r.y / mag, -r.z / mag);
}

float3 analyticalNormalCylinder(float3 position, float3 axis_point) {
    float dx = position.x - axis_point.x;
    float dy = position.y - axis_point.y;
    float r = sqrt(dx*dx + dy*dy);

    return make_float3(-dx / r, -dy / r, 0.0f);
}

float3 analyticalNormalPlane(float3 normal_direction) {
    // Already known
    return normal_direction;
}
```

**Error Computation:**
```cpp
struct NormalError {
    float mean_error;
    float max_error;
    float rms_error;
    int num_interface_cells;
};

NormalError computeNormalError(const float* fill_level,
                                 const float3* interface_normal,
                                 std::function<float3(float3)> analytical_normal_fn,
                                 int nx, int ny, int nz, float dx) {
    std::vector<float> errors;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_level[idx];

                // Only check interface cells
                if (f < 0.1f || f > 0.9f) continue;

                float3 pos = make_float3(i * dx, j * dx, k * dx);
                float3 n_comp = interface_normal[idx];
                float3 n_exact = analytical_normal_fn(pos);

                // Compute angular error
                float dot_prod = n_comp.x * n_exact.x +
                                 n_comp.y * n_exact.y +
                                 n_comp.z * n_exact.z;
                float mag_comp = sqrt(n_comp.x*n_comp.x + n_comp.y*n_comp.y + n_comp.z*n_comp.z);
                float mag_exact = sqrt(n_exact.x*n_exact.x + n_exact.y*n_exact.y + n_exact.z*n_exact.z);

                float cos_angle = dot_prod / (mag_comp * mag_exact + 1e-10f);
                cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle));  // Clamp

                float angle_error_rad = acos(cos_angle);
                float angle_error_deg = angle_error_rad * 180.0f / M_PI;

                errors.push_back(angle_error_deg);
            }
        }
    }

    NormalError result;
    result.num_interface_cells = errors.size();
    result.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0f) / errors.size();
    result.max_error = *std::max_element(errors.begin(), errors.end());

    float sum_sq = 0.0f;
    for (float e : errors) sum_sq += e * e;
    result.rms_error = sqrt(sum_sq / errors.size());

    return result;
}
```

**Test File:** `/home/yzk/LBMProject/tests/unit/vof/test_vof_reconstruction.cu` (already exists)

---

## Test 9: Curvature Calculation

### Purpose
Validate curvature computation for known geometries.

### Physical Description
Curvature κ is computed from the divergence of the interface normal:
```
κ = ∇·n
```

For known geometries:
- **Sphere:** κ = 2/R (constant)
- **Cylinder:** κ = 1/R (constant in xy-plane)
- **Plane:** κ = 0 (no curvature)

### Setup Parameters

**Test Cases:**

**Case 9A: Sphere**
- Radius: R = 10 cells (20 μm with dx=2μm)
- Expected: κ = 2/R = 2/(20e-6) = 1e5 m⁻¹

**Case 9B: Cylinder**
- Radius: R = 10 cells
- Expected: κ = 1/R = 1/(20e-6) = 5e4 m⁻¹

**Case 9C: Plane**
- Flat interface
- Expected: κ = 0

### Expected Behavior

**Accuracy:**
- Relative error: |κ_computed - κ_exact| / κ_exact < 10%
- Best accuracy at R > 5 cells (well-resolved)

**Sensitivity:**
- Curvature degrades for small radii (R < 3 cells)
- Numerical noise in second derivatives affects accuracy

### Validation Metrics

**Curvature Error:**
```
For interface cells (0.1 < f < 0.9):
  κ_computed = curvature[idx]
  κ_exact = analytical curvature

  error_rel = |κ_computed - κ_exact| / κ_exact

Statistics:
- mean_error
- max_error
- std_dev (measure of consistency)
```

Acceptance:
- mean_error < 0.1 (10%)
- max_error < 0.2 (20%)

### Implementation Guidelines

**Analytical Curvature:**
```cpp
float analyticalCurvatureSphere(float R) {
    return 2.0f / R;
}

float analyticalCurvatureCylinder(float R) {
    return 1.0f / R;
}

float analyticalCurvaturePlane() {
    return 0.0f;
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/vof/test_vof_curvature_sphere.cu` (already exists)

---

## Test 10: Mass Conservation (Long-Term Advection)

### Purpose
Validate global mass conservation over extended simulations with complex velocity fields.

### Physical Description
VOF should conserve total liquid volume (mass) regardless of velocity field complexity:
```
M(t) = Σ f_i(t) × dx³ = constant
```

### Setup Parameters

**Domain:**
- Grid: 64 × 64 × 64 cells
- Physical size: 128 μm × 128 μm × 128 μm
- Periodic boundaries in x, y, z

**Initial Condition:**
- Spherical droplet at center
- Radius: R = 15 μm
- Initial mass: M₀ = (4/3)πR³ × ρ

**Velocity Field:**
- Rotating shear flow (time-dependent):
  ```
  u(x,y,z,t) = -ω(t) × (y - y_c)
  v(x,y,z,t) =  ω(t) × (x - x_c)
  w(x,y,z,t) = 0
  ω(t) = ω₀ × sin(2πt / T_period)
  ```
- ω₀ = 1e5 rad/s
- T_period = 10 μs

**Numerical Parameters:**
- Time step: dt = 1e-8 s
- Total time: 50 μs (5 periods)
- CFL < 0.5 maintained

### Expected Behavior

**Mass Conservation:**
- Mass variation: |M(t) - M₀| / M₀ < 1% for all t
- No systematic drift (accumulation or loss)

**Droplet Deformation:**
- Droplet elongates and compresses due to shear
- Shape changes are reversible (returns near initial after full period)

### Validation Metrics

**Mass Tracking:**
```
Record M(t) at every 100 time steps
Compute:
  ΔM_abs(t) = |M(t) - M₀|
  ΔM_rel(t) = ΔM_abs / M₀

Acceptance:
  max(ΔM_rel) < 0.01 over entire simulation
```

**Monotonicity Check:**
```
Check if mass is systematically increasing or decreasing:
  Linear fit: M(t) = M₀ + α×t

If |α| > threshold: systematic leak detected
Acceptance: |α| < 0.001 × M₀ / t_total
```

### Implementation Guidelines

**Mass Tracking:**
```cpp
std::vector<float> mass_history;
std::vector<float> time_history;

for (int step = 0; step < num_steps; ++step) {
    vof.advectFillLevel(d_ux, d_uy, d_uz, dt);

    if (step % 100 == 0) {
        float mass = vof.computeTotalMass();
        mass_history.push_back(mass);
        time_history.push_back(step * dt);

        float rel_error = fabs(mass - mass_initial) / mass_initial;
        printf("Step %d: M = %.6f, error = %.4f%%\n", step, mass, rel_error * 100);
    }
}

// Final validation
float max_error = 0.0f;
for (float m : mass_history) {
    float err = fabs(m - mass_initial) / mass_initial;
    max_error = fmaxf(max_error, err);
}

EXPECT_LT(max_error, 0.01);
```

**Test File:** `/home/yzk/LBMProject/tests/unit/vof/test_vof_mass_conservation.cu` (already exists)

---

## Test 11: CFL Stability

### Purpose
Validate VOF advection stability under CFL (Courant-Friedrichs-Lewy) limit violations and conformance.

### Physical Description
The CFL condition for explicit advection is:
```
CFL = v_max × dt / dx < 0.5
```
where v_max is the maximum velocity magnitude.

VOF advection should:
- **Remain stable** for CFL < 0.5
- **Not produce f > 1 or f < 0** even at CFL ≈ 0.5
- **Detect and warn** for CFL > 0.5

### Setup Parameters

**Test Cases:**

**Case 11A: Safe CFL (CFL = 0.25)**
- Domain: 32 × 32 × 32 cells, dx = 2 μm
- Velocity: u = 0.25 m/s
- Time step: dt = 2e-6 / 0.25 = 8e-6 s → CFL = 0.25 × 8e-6 / 2e-6 = 1.0 (too high, recalculate)
  - Corrected: dt = 2e-6 × 0.25 = 5e-7 s → CFL = 0.25
- Run 1000 steps
- Expected: Stable, no warnings

**Case 11B: Marginal CFL (CFL = 0.49)**
- Same domain
- dt adjusted for CFL = 0.49
- Expected: Stable, fill ∈ [0,1], possible minor warnings

**Case 11C: CFL Violation (CFL = 0.8)**
- Same domain
- dt increased to violate CFL
- Expected: Warning issued, possible instability (f > 1 or f < 0)

### Expected Behavior

**Case 11A (Safe):**
- No CFL warnings
- All fill levels remain in [0, 1]
- Mass conserved within 1%

**Case 11B (Marginal):**
- Possible CFL warning (implementation-dependent)
- Fill levels clamped to [0, 1] if necessary
- Mass conserved within 2%

**Case 11C (Violation):**
- CFL warning issued
- Fill levels may exceed [0, 1] before clamping
- Advection errors accumulate

### Validation Metrics

**Stability:**
```
For each time step:
  Check: min(fill_level) >= 0
  Check: max(fill_level) <= 1

If violated: stability_failure = true
```

**Warning Detection:**
```
Capture stdout/stderr for CFL warnings
Expected warning format: "WARNING: VOF CFL violation: {CFL} > 0.5"
```

**Mass Error vs CFL:**
```
Plot mass error as function of CFL number
Expected: error increases sharply for CFL > 0.5
```

### Implementation Guidelines

**CFL Computation:**
```cpp
float computeMaxVelocity(const float* ux, const float* uy, const float* uz, int num_cells) {
    std::vector<float> h_ux(num_cells), h_uy(num_cells), h_uz(num_cells);
    cudaMemcpy(h_ux.data(), ux, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), uy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uz.data(), uz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float v_max = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float v = sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        v_max = fmaxf(v_max, v);
    }
    return v_max;
}

float computeCFL(float v_max, float dt, float dx) {
    return v_max * dt / dx;
}
```

**Test File:** `/home/yzk/LBMProject/tests/validation/test_cfl_stability.cu` (already exists)

---

## Test 12: Marangoni Force Verification (Isolated)

### Purpose
Validate Marangoni force computation in isolation (without flow solver coupling).

### Physical Description
Marangoni force in CSF formulation:
```
F_Marangoni = (dσ/dT) × ∇_s T × |∇f|  [N/m³]
```
where:
- ∇_s T = (I - n⊗n) · ∇T (tangential gradient)
- |∇f| = interface delta function

For a planar interface with known temperature gradient, the force magnitude is analytically calculable.

### Setup Parameters

**Domain:**
- Grid: 32 × 32 × 32 cells
- Physical size: 64 μm × 64 μm × 64 μm
- dx = 2 μm

**Interface:**
- Flat vertical plane at x = 16 cells
- Normal: n = (-1, 0, 0)
- Fill level: f = 1 for i < 16, f = 0 for i >= 16

**Temperature:**
- Linear gradient along y-axis: T(y) = T₀ + (dT/dy) × y
- T₀ = 2000 K
- dT/dy = 1e6 K/m
- Tangential gradient at interface: ∇_s T = (0, 1e6, 0) K/m

**Material:**
- dσ/dT = -0.26e-3 N/(m·K)

**Expected Force:**
```
|∇f| ≈ 1/dx = 1/(2e-6) = 5e5 m⁻¹ (approximate, depends on diffusion)

F_y = |dσ/dT| × |∇_s T| × |∇f|
    = 0.26e-3 × 1e6 × 5e5
    = 1.3e8 N/m³
    = 130 MN/m³
```

### Expected Behavior

**Force Direction:**
- Force points in +y direction (toward higher temperature if dσ/dT < 0)
- Force components: F_x ≈ 0, F_y > 0, F_z ≈ 0

**Force Magnitude:**
- At interface cells (15 ≤ i ≤ 17): F_y ≈ 1e8 N/m³
- Outside interface: F_y ≈ 0

### Validation Metrics

**Force Direction:**
```
At interface cells:
  Check: force_y[idx] > 0  (correct sign)
  Check: |force_x[idx]| < 0.01 × |force_y[idx]|
  Check: |force_z[idx]| < 0.01 × |force_y[idx]|
```

**Force Magnitude:**
```
F_y_avg = average force_y at interface cells (0.1 < f < 0.9)
F_y_theory = |dσ/dT| × |∇T_tangential| × |∇f|

error = |F_y_avg - F_y_theory| / F_y_theory
```
Acceptance: error < 0.2 (20%, due to |∇f| approximation)

### Implementation Guidelines

**Setup:**
```cpp
// Initialize flat interface
std::vector<float> h_fill(num_cells, 0.0f);
for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * k);
            h_fill[idx] = (i < nx/2) ? 1.0f : 0.0f;
        }
    }
}

// Set linear temperature gradient in y
std::vector<float> h_temp(num_cells);
for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * k);
            float y = j * dx;
            h_temp[idx] = T0 + dT_dy * y;
        }
    }
}

// Reconstruct interface normals
vof.reconstructInterface();

// Compute Marangoni force
marangoni.computeMarangoniForce(d_temperature, d_fill_level, d_interface_normal,
                                 d_force_x, d_force_y, d_force_z);
```

**Test File:** `/home/yzk/LBMProject/tests/unit/vof/test_vof_marangoni.cu` (already exists)

---

## Summary Table

| Test # | Test Name | Physics Validated | Key Metric | Acceptance Criterion | Test File |
|--------|-----------|-------------------|------------|----------------------|-----------|
| 1 | Zalesak's Disk | Advection accuracy | Shape error | E_shape < 5% | test_vof_zalesak_disk.cu |
| 2 | Spherical Droplet | Surface tension (Laplace) | Pressure jump | ΔP error < 15% | test_vof_laplace_pressure.cu |
| 3 | Oscillating Droplet | Dynamic surface tension | Oscillation frequency | f error < 15% | test_vof_oscillating_droplet.cu |
| 4 | Thermocapillary Migration | Marangoni effect | Migration velocity | U error < 30% | test_vof_thermocapillary_migration.cu |
| 5 | Contact Angle | Wall wetting | Static contact angle | θ error < 5° | test_vof_contact_angle_static.cu |
| 6 | Evaporation Mass Loss | VOF-thermal coupling | Mass loss rate | dM/dt error < 10% | test_vof_evaporation_mass_loss.cu (exists) |
| 7 | Recoil Pressure | Keyhole formation | Pressure magnitude | P error < 5% | test_vof_recoil_pressure_depression.cu |
| 8 | Interface Reconstruction | PLIC normal | Angular error | mean error < 5° | test_vof_reconstruction.cu (exists) |
| 9 | Curvature Calculation | Geometric curvature | Relative error | κ error < 10% | test_vof_curvature_sphere.cu (exists) |
| 10 | Mass Conservation | Global conservation | Mass variation | ΔM < 1% | test_vof_mass_conservation.cu (exists) |
| 11 | CFL Stability | Numerical stability | Bound violations | No f > 1 or f < 0 | test_cfl_stability.cu (exists) |
| 12 | Marangoni Force | Force computation | Force magnitude | F error < 20% | test_vof_marangoni.cu (exists) |

---

## Testing Workflow

### 1. Unit Tests (Fast)
Run individual component tests to verify isolated functionality:
```bash
cd /home/yzk/LBMProject/build
./tests/unit/vof/test_vof_reconstruction
./tests/unit/vof/test_vof_curvature
./tests/unit/vof/test_vof_marangoni
./tests/unit/vof/test_vof_evaporation_mass_loss
./tests/unit/vof/test_vof_mass_conservation
```

### 2. Validation Tests (Comprehensive)
Run physics validation tests with analytical benchmarks:
```bash
./tests/validation/vof/test_vof_zalesak_disk
./tests/validation/vof/test_vof_laplace_pressure
./tests/validation/vof/test_vof_oscillating_droplet
./tests/validation/vof/test_vof_thermocapillary_migration
./tests/validation/vof/test_vof_contact_angle_static
./tests/validation/vof/test_vof_recoil_pressure_depression
./tests/validation/test_cfl_stability
```

### 3. Integration Tests
Run coupled multiphysics tests:
```bash
./tests/integration/multiphysics/test_vof_fluid_coupling
./tests/integration/multiphysics/test_vof_subcycling_convergence
```

---

## Implementation Priorities

**Phase 1 (Core Functionality):**
1. Test 8: Interface Reconstruction (verify PLIC accuracy)
2. Test 9: Curvature Calculation (foundation for surface tension)
3. Test 10: Mass Conservation (critical numerical property)

**Phase 2 (Physical Validation):**
4. Test 2: Laplace Pressure (validate surface tension)
5. Test 1: Zalesak's Disk (advection benchmark)
6. Test 12: Marangoni Force (isolated force check)

**Phase 3 (Advanced Physics):**
7. Test 4: Thermocapillary Migration (Marangoni dynamics)
8. Test 7: Recoil Pressure (high-T evaporation)
9. Test 3: Oscillating Droplet (dynamic effects)

**Phase 4 (Boundary Conditions):**
10. Test 5: Contact Angle (wall interactions)
11. Test 11: CFL Stability (robustness)

---

## References

1. **VOF Method:**
   - Hirt, C.W. & Nichols, B.D. (1981). "Volume of fluid (VOF) method for the dynamics of free boundaries." Journal of Computational Physics, 39(1), 201-225.

2. **CSF Surface Tension:**
   - Brackbill, J.U., Kothe, D.B., & Zemach, C. (1992). "A continuum method for modeling surface tension." Journal of Computational Physics, 100(2), 335-354.

3. **Zalesak's Disk:**
   - Zalesak, S.T. (1979). "Fully multidimensional flux-corrected transport algorithms for fluids." Journal of Computational Physics, 31(3), 335-362.

4. **Thermocapillary Migration:**
   - Young, N.O., Goldstein, J.S., & Block, M.J. (1959). "The motion of bubbles in a vertical temperature gradient." Journal of Fluid Mechanics, 6(3), 350-356.

5. **Recoil Pressure:**
   - Anisimov, S.I. (1968). "Vaporization of metal absorbing laser radiation." Soviet Physics JETP, 27, 182-183.
   - Khairallah, S.A. et al. (2016). "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." Acta Materialia, 108, 36-45.

6. **Contact Angle:**
   - Young, T. (1805). "An essay on the cohesion of fluids." Philosophical Transactions of the Royal Society of London, 95, 65-87.

---

**Document Version:** 1.0
**Last Updated:** 2025-12-03
**Status:** Ready for Implementation
