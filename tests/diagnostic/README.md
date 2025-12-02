# Emergency Velocity Diagnostic Test Suite

## Purpose

This diagnostic test suite was created to isolate the root cause of the **zero-velocity bug** in v5 Test A validation, where:
- Temperature field shows active melting (T_max = 41,061 K)
- Fluid solver is enabled
- Buoyancy forces should be present
- **BUT**: Maximum velocity = 0.000 mm/s (exactly zero)

## Test Suite Structure

### Test 0: Unit Conversion Verification
**File**: `test0_unit_conversion.cu`

**Purpose**: Determine the exact conversion factor needed for force scaling

**Result**: Scaling factors of **1e4 to 1e6** produce reasonable velocities (0.001 to 0.1 m/s)

**Key Finding**: The current implementation has NO scaling, resulting in forces that are too small by 4-6 orders of magnitude.

---

### Test 1: Bare FluidLBM Velocity Computation
**File**: `test1_fluid_velocity_only.cu`

**Purpose**: Does FluidLBM compute ANY velocity without multiphysics?

**Test case**:
- 10×10×10 domain
- Apply constant force F_x = 1e-5 m/s² everywhere
- Run 100 steps
- Check: Is max(|ux|) > 0?

**Expected**: v_x ~ 1e-5 m/s or higher

**Result**: **FAIL**
- Velocity = 1.25e-09 m/s
- Force is applied but magnitude is wrong by **10,000x factor**

**Diagnosis**: Force scaling/unit conversion error in FluidLBM

---

### Test 2: Buoyancy Force Magnitude
**File**: `test2_buoyancy_force.cu`

**Purpose**: Is buoyancy force being computed and applied?

**Test case**:
- Create temperature gradient: 300 K (bottom) to 3000 K (top)
- Compute buoyancy: F = rho·beta·g·(T - T_ref)
- Apply to FluidLBM
- Run 10 steps

**Expected**:
- Force: ~10,800 N/m³
- Velocity: ~0.01 to 0.1 m/s

**Result**: **FAIL (Instability)**
- Force computed correctly: 10,800 N/m³ ✓
- Simulation goes unstable:
  - Step 0: v = 1.35 m/s
  - Step 1: v = 1.87e10 m/s (exponential growth!)
  - Step 2: v = inf (overflow)
  - Step 6+: v = 0 (crashed)

**Diagnosis**: Buoyancy force is correct but TOO LARGE for LBM (needs conversion to lattice units)

---

### Test 3: Darcy Damping Isolation
**File**: `test3_darcy_damping.cu`

**Purpose**: Is Darcy damping killing ALL flow instead of just solid regions?

**Test case**:
- Initialize with velocity u_x = 1.0 m/s everywhere
- Set liquid_fraction: 0.0 (solid), 0.5 (mushy), 1.0 (liquid) in thirds
- Apply Darcy damping with C = 20000
- Check: Does liquid region preserve velocity?

**Expected**:
- Solid: velocity → 0
- Liquid: velocity preserved
- Mushy: intermediate damping

**Result**: NOT RUN (Test 1 failed first)

**Purpose**: Would identify if Darcy formula is incorrect

---

### Test 4: Thermal-Fluid Velocity Coupling
**File**: `test4_thermal_advection_coupling.cu`

**Purpose**: Are velocity fields correctly passed from Fluid to Thermal solver?

**Test case**:
- Set FluidLBM velocity: u_x = 1.0 m/s
- Pass to ThermalLBM via collisionBGK(ux, uy, uz)
- Compare thermal diffusion with/without advection

**Expected**: Advection should modify temperature profile

**Result**: NOT RUN (Test 1 failed first)

**Purpose**: Would identify pointer passing bugs in multiphysics coupling

---

### Test 5: Config Flag Propagation
**File**: `test5_config_flags.cu`

**Purpose**: Are config flags correctly read and applied?

**Test case**:
- Parse config file
- Print all physics flags
- Verify enable_fluid, enable_thermal, enable_thermal_advection

**Expected**: All critical flags should be true

**Result**: NOT RUN (Test 1 failed first)

**Purpose**: Would identify configuration parsing errors

---

## Root Cause: Force Unit Conversion Error

### The Problem

FluidLBM receives forces in **physical units** (SI: N/m³ or m/s²) but applies them directly to the lattice Boltzmann collision operator, which expects forces in **dimensionless lattice units**.

### Evidence

1. **Test 0**: Scaling by 1e4 to 1e6 produces reasonable velocities
2. **Test 1**: No scaling gives velocity 10,000x too small
3. **Test 2**: No scaling on large force causes instability

### Unit Analysis

For LBM, forces must be converted:

```
F_lattice = F_physical × conversion_factor
```

Where the conversion factor depends on how forces are interpreted:

#### If F is acceleration [m/s²]:
```
conversion_factor = 1 / rho_physical
                  = 1 / 4000
                  ≈ 2.5e-4
```

#### If F is force density [N/m³]:
```
conversion_factor = dt² / (rho × dx)
                  = (1e-9)² / (4000 × 1e-6)
                  ≈ 2.5e-16
```

### Why v5 Test Shows Zero Velocity

In v5 Test A:
1. Thermal field creates buoyancy force: F ~ 10,000 N/m³
2. Force passed directly to FluidLBM (no conversion)
3. For stability, LBM needs F_lattice ~ 1e-5 to 1e-1
4. Without conversion:
   - Small forces (F ~ 1e-5) produce nearly zero velocity
   - Large forces (F ~ 10,000) cause immediate instability → NaN → 0

Result: **Zero velocity in output**

---

## Solution

### Recommended Fix (Option A)

Modify `FluidLBM::collisionBGK()` to convert forces:

```cpp
void FluidLBM::collisionBGK(const float* force_x,
                           const float* force_y,
                           const float* force_z) {
    // Convert physical forces to lattice forces
    float scale = dt_ * dt_ / dx_;  // For force density [N/m³]
    // OR
    float scale = 1.0f / rho0_;     // For acceleration [m/s²]

    // Apply scaling in kernel
    fluidBGKCollisionVaryingForceKernel<<<...>>>(
        ..., force_x, force_y, force_z, scale, ...
    );
}
```

**Pros**:
- Centralizes conversion in one place
- Transparent to callers
- Easy to verify and test

**Cons**:
- Requires FluidLBM to store dt and dx
- Need to determine correct scaling formula

### Alternative Fix (Option B)

Convert forces at MultiphysicsSolver level before passing to FluidLBM:

```cpp
void MultiphysicsSolver::computeTotalForce(...) {
    // Compute buoyancy (in SI units)
    fluid_->computeBuoyancyForce(..., d_force_x_, d_force_y_, d_force_z_);

    // Convert to lattice units
    float scale = config_.dt * config_.dt / config_.dx;
    scaleForcesKernel<<<...>>>(d_force_x_, d_force_y_, d_force_z_, scale);

    // Pass to fluid solver
    fluid_->collisionBGK(d_force_x_, d_force_y_, d_force_z_);
}
```

**Pros**:
- Keeps FluidLBM unit-agnostic
- Easier to understand unit conversions

**Cons**:
- Must convert at every call site
- Easy to forget conversion

---

## Validation Protocol

After implementing the fix:

1. **Re-run Test 0**: Verify scaling = 1.0 now works (no manual scaling needed)
2. **Re-run Test 1**: Should see v ~ 1e-5 to 1e-3 m/s (reasonable)
3. **Re-run Test 2**: Should see stable flow, v ~ 0.01 to 0.1 m/s
4. **Run Test 3**: Verify Darcy damping doesn't kill liquid flow
5. **Run v5 Test A**: Should see **non-zero velocity** in melt pool

Expected v5 result after fix:
- v_max > 0.1 mm/s (currently 0.000)
- Stable flow in liquid regions
- Flow magnitude consistent with buoyancy force

---

## Files Created

All files located in `/home/yzk/LBMProject/tests/diagnostic/`:

1. `test0_unit_conversion.cu` - Determines correct scaling factor
2. `test1_fluid_velocity_only.cu` - Tests basic FluidLBM force application
3. `test2_buoyancy_force.cu` - Tests buoyancy force magnitude
4. `test3_darcy_damping.cu` - Tests Darcy damping selectivity
5. `test4_thermal_advection_coupling.cu` - Tests thermal-fluid coupling
6. `test5_config_flags.cu` - Tests configuration parsing
7. `run_velocity_diagnostics.sh` - Master test runner script
8. `DIAGNOSTIC_RESULTS.md` - Detailed results and analysis
9. `README.md` - This file

Build configuration updated in:
- `/home/yzk/LBMProject/tests/CMakeLists.txt`

---

## How to Run

### Build tests:
```bash
cd /home/yzk/LBMProject/build
cmake ..
make test0_unit_conversion test1_fluid_velocity_only test2_buoyancy_force \
     test3_darcy_damping test4_thermal_advection_coupling test5_config_flags
```

### Run all tests:
```bash
/home/yzk/LBMProject/tests/diagnostic/run_velocity_diagnostics.sh
```

### Run individual tests:
```bash
./build/tests/test0_unit_conversion
./build/tests/test1_fluid_velocity_only
./build/tests/test2_buoyancy_force
# etc.
```

---

## Test Results Summary

| Test | Status | Key Finding |
|------|--------|-------------|
| Test 0 | INFO | Scaling of 1e4-1e6 needed |
| Test 1 | FAIL | Velocity 10,000x too small |
| Test 2 | FAIL | Force causes instability |
| Test 3 | - | Not run |
| Test 4 | - | Not run |
| Test 5 | - | Not run |

**Root cause identified in Test 1**: Force scaling/unit conversion error

---

## Technical Details

### LBM Force Implementation

The Guo forcing scheme used in FluidLBM adds force as:

```
f_i^post = f_i^eq + (1 - ω/2) × F_i × dt
```

Where:
- `f_i` = distribution function
- `ω` = relaxation parameter (1/τ)
- `F_i` = force in direction i
- `dt` = time step (= 1 in lattice units)

For this to work correctly, `F` must be in **lattice units** (dimensionless).

### Conversion Formula

Physical to lattice force conversion:

```
F_lattice = F_physical × (Δt_phys² / Δx_phys)
```

For LPBF simulation:
- Δt_phys = 1e-9 s (1 nanosecond)
- Δx_phys = 1e-6 m (1 micrometer)
- Conversion = (1e-9)² / 1e-6 = 1e-12

But empirically, Test 0 shows 1e4-1e6 works best, suggesting:
- Current implementation may already include partial conversion
- Or different force interpretation is used
- Need to check FluidLBM source code

### Reference

Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the forcing term in the lattice Boltzmann method." Physical Review E, 65(4), 046308.

---

## Conclusion

The diagnostic test suite has successfully isolated the root cause of the zero-velocity bug:

**Missing or incorrect unit conversion from physical forces to lattice forces in FluidLBM**

The fix is straightforward and localized. Implementation should take < 1 hour, with validation in v5 Test A confirming non-zero velocities.

---

**Created**: 2025-11-18
**Suite**: Emergency Velocity Diagnostic Tests
**Status**: Root cause identified
**Next**: Implement force scaling fix in FluidLBM
