# Bug Report: Zero Velocity in Phase 5 Simulation

## Executive Summary

**Root Cause**: Incorrect force unit conversion and scaling in `visualize_laser_melting_with_flow.cu` causes numerical instability and velocity collapse.

**Location**: `/home/yzk/LBMProject/apps/visualize_laser_melting_with_flow.cu`, lines 286-294

**Severity**: Critical - Renders Phase 5 simulation unusable

---

## Test-Driven Investigation Results

### Test Ladder Results

| Test Level | Test Name | Result | Conclusion |
|------------|-----------|--------|------------|
| 1.1 | Collision Kernel Direct | ✅ PASS | Kernel applies forces correctly |
| 1.2 | Macroscopic Kernel | ✅ PASS | Velocity extraction works |
| 2.0 | Force→Velocity Pipeline | ✅ PASS | Basic force application works |
| 3.1 | Buoyancy Force Check | ✅ PASS | Buoyancy forces computed correctly |
| 3.2 | Darcy Damping Survival | ✅ PASS | Darcy damping works (produces large forces) |
| 4.0 | Phase 5 Instrumented | ❌ FAIL | **Catastrophic force scaling error** |

### Critical Findings from Instrumented Test

```
Step  | T_max | Melt% | max|f_buoy| | max|f_darcy| | max|f_final| | max|u|
------|-------|-------|--------------|---------------|---------------|-------
  600 |   882 |   0.0 |    -5.50e+02 |     -5.50e+02 |     -2.20e+09 | 4.72e+05
  700 |   963 |   0.0 |    -5.49e+02 |     -5.49e+02 |     -2.20e+09 | 0.00e+00
```

**Observations**:
1. Buoyancy force: ~550 N/m³ (reasonable)
2. Force after Darcy: ~550 N/m³ (reasonable - limited damping in solid regions)
3. **Force after scaling: -2.2×10⁹** (CATASTROPHIC - 9 orders of magnitude too large!)
4. Step 600: Velocity = 472,000 m/s (numerical overflow)
5. Step 700: Velocity = 0 (collapsed due to NaN/instability)

---

## Root Cause Analysis

### The Buggy Code

```cpp
// File: visualize_laser_melting_with_flow.cu, lines 201, 285-294

float force_conversion = (dt * dt) / dx;  // = (5e-10)² / 2e-6 = 1.25e-13

float target_force_magnitude = 1e-4f;
float estimated_phys_force = 200.0f;
float force_scale = target_force_magnitude / (estimated_phys_force * force_conversion);
// force_scale = 1e-4 / (200 * 1.25e-13) = 4e+06  ← BUG HERE

scaleForceArrayKernel<<<grid_size, block_size>>>(
    d_fx, d_fy, d_fz, force_scale, num_cells
);
// Result: forces multiplied by 4 million!
```

### Why This is Wrong

1. **Unit Conversion Formula** (`dt²/dx`):
   - Converts force from [N/m³] to lattice units
   - This part is CORRECT in principle
   - Result: forces on order of 10⁻¹⁰ (very small in lattice units)

2. **The "Rescaling Hack"**:
   - Code attempts to "boost" forces to "visible" magnitude (1e-4)
   - Divides by `(estimated_phys_force * force_conversion)`
   - **This is fundamentally flawed logic!**
   - Creates forces 10⁹ times too large
   - LBM requires forces ~10⁻⁴ for stability, not 10⁹

3. **Consequence**:
   - Forces → 2×10⁹ in lattice units
   - Velocity explodes to 472,000 m/s in first 100 steps
   - Causes numerical overflow, NaN propagation
   - Velocity field collapses to zero

### The Correct Approach

**Option 1: Dimensionless LBM (Recommended)**

Don't use physical units in LBM simulation. Instead:

```cpp
// Set dimensionless parameters
float g_lattice = 1e-4f;  // Gravity in lattice units
float beta_lattice = 1e-2f;  // Thermal expansion in lattice units
float T_ref_lattice = 0.5f;  // Reference temperature (0-1 range)

// Compute dimensionless forces directly
fluid.computeBuoyancyForce(
    thermal.getTemperature(),  // Already dimensionless in LBM
    T_ref_lattice,
    beta_lattice,
    0.0f, g_lattice, 0.0f,  // Dimensionless gravity
    d_fx, d_fy, d_fz
);

// NO SCALING NEEDED - forces are already in lattice units
fluid.collisionBGK(d_fx, d_fy, d_fz);
```

**Option 2: Fix the Unit Conversion**

If physical units must be used:

```cpp
// Step 1: Compute forces in physical units [N/m³]
fluid.computeBuoyancyForce(
    thermal.getTemperature(), T_ref, beta,
    0.0f, g_physical, 0.0f,
    d_fx, d_fy, d_fz
);

// Step 2: Convert to lattice units using ONLY the conversion factor
// F_lattice = F_physical * (dt² / dx)
// But this gives ~10⁻¹⁰, too small for LBM stability

// Step 3: Adjust LBM parameters instead of forces
// Increase viscosity or grid spacing to match physical Reynolds number
// See: Krüger et al. (2017), "The Lattice Boltzmann Method", Chapter 9
```

**Option 3: Scale Physical Parameters (Best for coupled simulations)**

```cpp
// Scale the PHYSICS, not the forces!
float g_scaled = 1e-4f / (rho0 * beta * dT_typical);  // Target force magnitude
float beta_scaled = beta * (g_physical / g_scaled);    // Compensate

// Now forces will naturally be ~1e-4
fluid.computeBuoyancyForce(
    thermal.getTemperature(), T_ref, beta_scaled,
    0.0f, g_scaled, 0.0f,  // Scaled gravity
    d_fx, d_fy, d_fz
);

fluid.collisionBGK(d_fx, d_fy, d_fz);  // NO additional scaling!
```

---

## The Fix

### Minimal Fix (Remove the Buggy Scaling)

```cpp
// REMOVE these lines:
// float target_force_magnitude = 1e-4f;
// float estimated_phys_force = 200.0f;
// float force_scale = target_force_magnitude / (estimated_phys_force * force_conversion);
//
// scaleForceArrayKernel<<<grid_size, block_size>>>(
//     d_fx, d_fy, d_fz, force_scale, num_cells
// );

// REPLACE with dimensionless approach:
float g_lattice = 1e-4f;  // Adjust to control convection strength
float beta_lattice = 1e-2f;
float T_min = 300.0f;
float T_max = 3000.0f;
float T_ref_lattice = (T_ref - T_min) / (T_max - T_min);

// Normalize temperature to 0-1
normalizeTemperatureKernel<<<grid, block>>>(
    thermal.getTemperature(), T_min, T_max, num_cells
);

// Compute dimensionless forces
fluid.computeBuoyancyForce(
    thermal.getTemperature(), T_ref_lattice, beta_lattice,
    0.0f, g_lattice, 0.0f,
    d_fx, d_fy, d_fz
);

fluid.applyDarcyDamping(
    thermal.getLiquidFraction(), darcy_constant,
    d_fx, d_fy, d_fz
);

// NO SCALING - use forces directly
fluid.collisionBGK(d_fx, d_fy, d_fz);
```

---

## Verification

After applying the fix, re-run `test_phase5_instrumented`:

**Expected Output**:
```
Step  | T_max | Melt% | max|f_buoy| | max|f_darcy| | max|f_final| | max|u|
------|-------|-------|--------------|---------------|---------------|-------
  600 |   882 |   0.0 |     1.00e-04 |      5.00e-05 |      5.00e-05 | 1.00e-03
  700 |   963 |   0.5 |     2.00e-04 |      1.50e-04 |      1.50e-04 | 2.50e-03
  800 |  1150 |   2.1 |     5.00e-04 |      4.00e-04 |      4.00e-04 | 5.00e-03
```

Forces should be on order 10⁻⁴, velocities on order 10⁻³ to 10⁻².

---

## References

1. **Guo, Z., Zheng, C., & Shi, B. (2002).** "Discrete lattice effects on the forcing term in the lattice Boltzmann method." *Physical Review E*, 65(4), 046308.
   - Correct forcing scheme implemented in code

2. **Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., & Viggen, E. M. (2017).** "The Lattice Boltzmann Method: Principles and Practice." *Springer*.
   - Chapter 9: Non-dimensionalization and unit conversion

3. **Shan, X., & Chen, H. (1993).** "Lattice Boltzmann model for simulating flows with multiple phases and components." *Physical Review E*, 47(3), 1815.
   - Proper force scaling for multiphase flows

---

## Test Files Created

All test files are located in `/home/yzk/LBMProject/tests/debug/`:

1. `test_collision_kernel_direct.cu` - Level 1.1: Kernel-level test
2. `test_macroscopic_kernel.cu` - Level 1.2: Velocity extraction test
3. `test_force_velocity_pipeline.cu` - Level 2: End-to-end pipeline test
4. `test_buoyancy_force_check.cu` - Level 3.1: Buoyancy computation test
5. `test_darcy_survival.cu` - Level 3.2: Darcy damping test
6. `test_phase5_instrumented.cu` - Level 4: Full simulation with instrumentation

Run with:
```bash
cd /home/yzk/LBMProject/build
./tests/test_collision_kernel_direct
./tests/test_macroscopic_kernel
./tests/test_force_velocity_pipeline
./tests/test_buoyancy_force_check
./tests/test_darcy_survival
./tests/test_phase5_instrumented
```

---

## Conclusion

The zero velocity issue in Phase 5 is NOT due to:
- ❌ Broken collision kernel
- ❌ Broken macroscopic computation
- ❌ Broken buoyancy calculation
- ❌ Broken Darcy damping
- ❌ Missing unit conversion

It IS due to:
- ✅ **Catastrophic force over-scaling by factor of 10⁹**
- ✅ **Incorrect "rescaling hack" logic**
- ✅ **Numerical instability from forces 2×10⁹ in lattice units**

**Recommendation**: Remove the force scaling entirely and use dimensionless LBM formulation.
