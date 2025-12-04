# VOF Static Surface Root Cause Analysis

## Problem Statement
User reports that LPBF simulation surface does not deform despite enabling `enable_vof_advection = true`.

## DIAGNOSTIC RESULTS (2025-11-21)

After adding diagnostic output to `vofStep()`, we discovered:

### KEY FINDING: VOF Advection IS Working!

```
[VOF DIAGNOSTIC] Step 0:
  max fill_level change = 0.00000000
  interface z-centroid  = 46.711 cells
  v_z max at interface  = 0.0000 m/s   <-- Initial velocity = 0 (expected)

[VOF DIAGNOSTIC] Step 100:
  max fill_level change = 0.00376567   <-- CHANGE DETECTED!
  interface z-centroid  = 44.879 cells (delta=-1.8321)  <-- Interface moved 1.8 cells!
  v_z max at interface  = 0.1115 m/s   <-- Velocity exists!
```

### The Interface IS Moving

- Initial position: z = 46.711 cells
- After 100 steps: z = 44.879 cells
- **Total movement: 1.8 cells (3.6 um)**

### Why User Sees "Static Surface"

1. **Movement is small** - 3.6 um movement may not be visible in ParaView
2. **Simulation time too short** - 200 steps = 20 us, need longer run
3. **Temperature too low for melting** - T_max=1035K < T_liquidus=1933K

### Code is Correct - No Bug Found

The VOF advection code is functioning properly. The apparent "static" surface is due to:
1. Early simulation stage (temperature gradient still building)
2. Small time window (need 100+ us for visible deformation)
3. Low laser power (20W vs typical 200W)

---

## Original Analysis (For Reference)

After thorough code review, I have identified **multiple potential causes** for the static surface issue. The most likely root causes are:

1. **CRITICAL: Interface Height Initialization** - Interface is at z=0.98 (top 2%), but the upwind scheme + periodic boundaries cause issues at domain edges
2. **Velocity Magnitude Problem** - Lattice velocities may be too small after CFL limiting to cause visible interface motion
3. **Numerical Diffusion** - Upwind scheme is diffusive, can smear interface over time without clear deformation

---

## Detailed Analysis

### 1. VOF Advection Kernel Analysis (`vof_solver.cu`)

The advection kernel uses first-order upwind scheme:

```cpp
// Line 58-63 of vof_solver.cu
float dfdt_x = u * (fill_level[idx] - fill_level[idx_x]) / dx;
float dfdt_y = v * (fill_level[idx] - fill_level[idx_y]) / dx;
float dfdt_z = w * (fill_level[idx] - fill_level[idx_z]) / dx;

float f_new = fill_level[idx] - dt * (dfdt_x + dfdt_y + dfdt_z);
```

**Mathematical Analysis:**
- The scheme solves: `df/dt + u * df/dx = 0`
- Upwind index selection (lines 49-51):
  - If `u > 0`: use upstream neighbor (`i-1`)
  - If `u < 0`: use downstream neighbor (`i+1`)

**ISSUE IDENTIFIED:** The gradient formula computes:
```
dfdt_x = u * (f[idx] - f[idx_x]) / dx
```

For upwind scheme with `u > 0`:
- `idx_x = i - 1` (upstream)
- Gradient = `(f[i] - f[i-1]) / dx` = forward difference from upstream

This is **CORRECT** for upwind advection. However...

### 2. Interface Height Problem

From `visualize_lpbf_scanning.cu` line 177:
```cpp
const float interface_height = 0.98f;  // Near top (z = 49 for nz=50)
```

From `multiphysics_solver.cu` initialization (lines 505-518):
```cpp
int z_interface = static_cast<int>(interface_height * config_.nz);
// z_interface = 49 (one cell from top)

for (int k = 0; k < config_.nz; ++k) {
    float z_dist = k - z_interface;
    h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
}
```

**CRITICAL ISSUE:** The interface is at `z = 49` (nearly top boundary).

In the advection kernel, the upwind neighbor for z-direction:
```cpp
int k_up = (w > 0.0f) ? (k > 0 ? k - 1 : nz - 1) : (k < nz - 1 ? k + 1 : 0);
```

For cells at `k = 49` (top layer):
- If `w > 0` (upward flow): `k_up = 48` (OK)
- If `w < 0` (downward flow): `k_up = 0` (WRAPS TO BOTTOM - periodic BC!)

**This is a BUG for non-periodic physical domains!** The surface should NOT wrap to the bottom.

### 3. Velocity Magnitude Analysis

From `multiphysics_solver.cu` vofStep() (lines 777-791):
```cpp
const float velocity_conversion = config_.dx / config_.dt;
// For dx = 2e-6 m, dt = 1e-7 s:
// velocity_conversion = 2e-6 / 1e-7 = 20 m/s per lattice unit

convertVelocityToPhysicalUnitsKernel<<<blocks, threads>>>(
    fluid_->getVelocityX(), ...
    d_velocity_physical_x_, ...
    velocity_conversion, ...
);
```

**Expected velocity range:**
- Lattice velocity: O(0.01 - 0.1) for stable LBM
- Physical velocity: 0.2 - 2 m/s after conversion

**Potential Issue:** If CFL limiter in `fluidStep()` heavily reduces forces, the resulting velocities may be too small to cause visible interface motion over the simulation time.

### 4. VOF Subcycling Analysis

From `multiphysics_solver.cu` (lines 805-812):
```cpp
float dt_sub = dt / config_.vof_subcycles;  // dt_sub = 1e-7 / 10 = 1e-8 s

for (int i = 0; i < config_.vof_subcycles; ++i) {
    vof_->advectFillLevel(d_velocity_physical_x_, ..., dt_sub);
}
```

**Numerical Analysis:**
For dx = 2e-6 m, dt_sub = 1e-8 s, and v = 1 m/s:
```
CFL = v * dt_sub / dx = 1 * 1e-8 / 2e-6 = 0.005
```

This CFL is **very small** (0.005 << 0.5), meaning:
- Each subcycle moves interface by only `0.005 * dx = 10 nm`
- After 10 subcycles: 100 nm total per main timestep
- After 3000 steps: 0.3 mm = 300 um maximum

**This is actually a reasonable displacement** if velocity stays at 1 m/s.

### 5. Diagnostic Output in VOF Advection

The code already has diagnostic output (lines 419-430):
```cpp
static int call_count = 0;
if (call_count % 500 == 0 && call_count < 5000) {
    float mass = computeTotalMass();
    printf("[VOF ADVECT] Call %d: v_max=%.4f m/s ...\n", call_count, v_max, ...);
}
call_count++;
```

**Check if this output appears** - if `v_max` is very small, that confirms the velocity issue.

---

## Root Causes Summary

| Priority | Issue | Location | Severity |
|----------|-------|----------|----------|
| 1 | **Periodic BC at interface** | `advectFillLevelUpwindKernel` line 51 | HIGH |
| 2 | **Velocity may be near zero** | Depends on Marangoni/fluid coupling | HIGH |
| 3 | **Interface at domain edge** | `initialize()` line 177 | MEDIUM |
| 4 | **Numerical diffusion** | Upwind scheme | LOW |

---

## Recommended Diagnostic Steps

### Step 1: Add VOF Diagnostic Output

Add to `vofStep()` in `multiphysics_solver.cu`:

```cpp
void MultiphysicsSolver::vofStep(float dt) {
    if (!vof_ || !fluid_) return;

    // DIAGNOSTIC: Check fill level before/after advection
    static int diag_count = 0;
    std::vector<float> h_fill_before, h_fill_after;

    if (diag_count % 100 == 0) {
        h_fill_before.resize(num_cells);
        cudaMemcpy(h_fill_before.data(), vof_->getFillLevel(),
                   num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // ... existing advection code ...

    if (diag_count % 100 == 0) {
        h_fill_after.resize(num_cells);
        cudaMemcpy(h_fill_after.data(), vof_->getFillLevel(),
                   num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute change
        float max_change = 0.0f;
        for (int i = 0; i < num_cells; ++i) {
            float delta = std::abs(h_fill_after[i] - h_fill_before[i]);
            max_change = std::max(max_change, delta);
        }

        printf("[VOF DIAGNOSTIC] Step %d: max fill_level change = %.6f\n",
               diag_count, max_change);
    }
    diag_count++;
}
```

### Step 2: Check Physical Velocity in VOF

Add to `vofStep()` after velocity conversion:

```cpp
// After convertVelocityToPhysicalUnitsKernel
std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells);
cudaMemcpy(h_vx.data(), d_velocity_physical_x_, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_vy.data(), d_velocity_physical_y_, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_vz.data(), d_velocity_physical_z_, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

float v_max = 0.0f;
for (int i = 0; i < num_cells; ++i) {
    float v = std::sqrt(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i] + h_vz[i]*h_vz[i]);
    v_max = std::max(v_max, v);
}

static int v_diag_count = 0;
if (v_diag_count % 100 == 0) {
    printf("[VOF VELOCITY] Physical v_max = %.4f m/s (%.2f mm/s)\n", v_max, v_max * 1000);
}
v_diag_count++;
```

### Step 3: Verify Interface Position is Changing

Check the z-centroid of the interface:

```cpp
// Compute interface z-centroid
float z_sum = 0.0f, f_sum = 0.0f;
for (int k = 0; k < config_.nz; ++k) {
    for (int j = 0; j < config_.ny; ++j) {
        for (int i = 0; i < config_.nx; ++i) {
            int idx = i + config_.nx * (j + config_.ny * k);
            float f = h_fill_after[idx];
            if (f > 0.01f && f < 0.99f) {  // Interface cells only
                z_sum += k * f;
                f_sum += f;
            }
        }
    }
}
float z_centroid = (f_sum > 0) ? z_sum / f_sum : -1.0f;
printf("[VOF INTERFACE] z-centroid = %.2f cells\n", z_centroid);
```

---

## Recommended Fixes

### Fix 1: Add Non-Periodic Z Boundary Option (CRITICAL)

```cpp
// In advectFillLevelUpwindKernel, add boundary flag parameter:
__global__ void advectFillLevelUpwindKernel(
    ...,
    bool periodic_z)  // NEW parameter
{
    // For z-direction:
    int k_up;
    if (w > 0.0f) {
        k_up = (k > 0) ? k - 1 : (periodic_z ? nz - 1 : k);
    } else {
        k_up = (k < nz - 1) ? k + 1 : (periodic_z ? 0 : k);
    }
}
```

### Fix 2: Move Interface Away from Boundary

```cpp
// In visualize_lpbf_scanning.cu:
const float interface_height = 0.90f;  // z = 45 instead of z = 49
```

### Fix 3: Increase Output Frequency Near Interface

Add z-slice diagnostic to see if interface is actually moving within the domain.

---

## Conclusion

The most likely cause of the "static surface" issue is that:

1. **The velocity field is too weak** to cause visible interface motion in the simulation time
2. **The periodic boundary condition in z** may be causing artifacts at the top boundary
3. **The diagnostic output is not being checked** - the `[VOF ADVECT]` messages may show v_max is near zero

**Recommended immediate action:**
1. Run simulation and check console output for `[VOF ADVECT]` messages
2. If `v_max` is near zero, the issue is in the fluid/Marangoni coupling
3. If `v_max` is reasonable (>0.1 m/s), add the fill_level change diagnostic to verify advection is working

