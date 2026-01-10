# VTK Simulation Verification Guide

## Overview

This guide describes the tools and procedures for verifying LBM thermal solver correctness after critical bug fixes. The verification suite is designed to catch common issues that arise from:

- CUDA errors and memory issues
- Evaporation division-by-zero
- Material property validation failures
- Uninitialized memory

## Quick Reference

### Quick Single-File Check
```bash
python3 /home/yzk/LBMProject/scripts/quick_vtk_check.py simulation.vtk
```

### Comprehensive Multi-Timestep Validation
```bash
python3 /home/yzk/LBMProject/scripts/verify_simulation_correctness.py ./vtk_out/
```

## Tools

### 1. quick_vtk_check.py

**Purpose:** Rapid diagnostic for a single VTK file

**When to use:**
- During development to spot-check individual timesteps
- Quick sanity check after code changes
- First-pass investigation of suspicious results

**What it checks:**
- NaN/Inf values in all fields
- Temperature bounds (0 < T < 50,000 K)
- Fill level bounds (0 ≤ f ≤ 1)
- Basic statistics (min, max, mean, std)

**Example output:**
```
QUICK VTK CHECK: simulation_0100.vtk
============================================================
Grid: 100 x 50 x 50 = 250000 cells
Spacing: 1.000e-04 x 1.000e-04 x 1.000e-04

Fields found: ['temperature', 'fill_level']

  temperature:
    Range: [2.930000e+02, 2.547891e+03]
    Mean:  4.128456e+02
    Std:   3.124567e+02

  fill_level:
    Range: [0.000000e+00, 1.000000e+00]
    Mean:  6.234567e-01
    Std:   4.567891e-01

[PASS] No issues detected
```

### 2. verify_simulation_correctness.py

**Purpose:** Comprehensive validation across multiple timesteps

**When to use:**
- After completing a simulation run
- Before publishing or analyzing results
- Validating bug fixes
- Checking conservation properties

**What it checks:**

1. **Field Validation** (per timestep)
   - NaN/Inf detection
   - Physical bounds
   - Statistical summaries

2. **Gradient Analysis** (per timestep)
   - Temperature gradient magnitude
   - Detection of extreme gradients (> 10^8 K/m)
   - Numerical stability indicators

3. **Conservation Properties** (across timesteps)
   - Energy conservation (E = Σ ρ·cp·T·dV)
   - Mass conservation (M = Σ f·ρ·dV)
   - Trend analysis

**Output files:**
```
verification_results/
├── validation_results.json          # Complete results
├── energy_conservation.png          # Energy vs time plot
├── mass_conservation.png            # Mass vs time plot
├── gradients_t0.png                 # Gradient histogram (t=0)
├── gradients_t10.png                # Gradient histogram (t=10)
└── ...
```

## Validation Workflow

### Standard Workflow

1. **Run simulation**
   ```bash
   cd /home/yzk/walberla/build/apps/showcases/LaserHeating
   ./LaserHeating
   ```

2. **Quick check first timestep**
   ```bash
   python3 /home/yzk/LBMProject/scripts/quick_vtk_check.py vtk_out/simulation_0000.vtk
   ```

3. **If OK, run comprehensive validation**
   ```bash
   python3 /home/yzk/LBMProject/scripts/verify_simulation_correctness.py vtk_out/
   ```

4. **Review results**
   ```bash
   # Check exit code
   echo $?  # 0 = pass, 1 = fail

   # View summary
   cat verification_results/validation_results.json | jq '.summary'

   # View plots
   xdg-open verification_results/energy_conservation.png
   ```

### Debugging Workflow

If validation fails:

1. **Identify which check failed**
   - Read console output for specific failures
   - Check JSON for detailed statistics

2. **Examine field statistics**
   ```bash
   cat verification_results/validation_results.json | jq '.results[] | select(.passed == false)'
   ```

3. **Look at gradient plots**
   - High gradients → mesh resolution issues
   - Spike patterns → numerical instability
   - Localized extreme values → boundary condition issues

4. **Analyze conservation trends**
   - Linear drift → boundary flux imbalance
   - Oscillations → time step too large
   - Sudden jumps → phase change errors

5. **Correlate with simulation parameters**
   - Check material properties
   - Verify boundary conditions
   - Review time step size
   - Check mesh resolution

## Common Issues and Solutions

### Issue: NaN values detected

**Likely causes:**
- Division by zero in evaporation model
- Uninitialized memory
- Numerical overflow

**Actions:**
1. Check evaporation threshold (avoid f=0)
2. Verify memory initialization
3. Add CUDA error checking
4. Review material property validation

### Issue: Temperature out of bounds

**Likely causes:**
- Unrealistic heat source power
- Wrong material properties (wrong units)
- Time step too large

**Actions:**
1. Verify laser power in physical units (W)
2. Check thermal conductivity, specific heat units
3. Reduce time step by 50%
4. Check boundary conditions

### Issue: Fill level out of bounds

**Likely causes:**
- VOF advection errors
- Interface reconstruction issues
- Floating point precision

**Actions:**
1. Enable fill level clamping (0 ≤ f ≤ 1)
2. Check CFL condition
3. Reduce time step
4. Review interface compression scheme

### Issue: Extreme temperature gradients

**Likely causes:**
- Insufficient mesh resolution
- Discontinuous material properties at interface
- Boundary condition artifacts

**Actions:**
1. Refine mesh near heat source
2. Implement gradient limiting
3. Smooth material property transitions
4. Check boundary condition implementation

### Issue: Energy not conserved

**Likely causes:**
- Boundary condition flux errors
- Phase change latent heat not properly accounted for
- Evaporation mass loss

**Actions:**
1. Verify boundary conditions conserve energy
2. Check latent heat implementation
3. Account for vapor energy loss
4. Verify source term integration

### Issue: Mass not conserved

**Likely causes:**
- VOF advection errors
- Boundary condition mass flux errors
- Evaporation model inconsistency

**Actions:**
1. Check VOF advection scheme
2. Verify inflow/outflow boundary conditions
3. Verify evaporation mass flux matches VOF changes
4. Check for fill level clipping artifacts

## Validation Criteria

### Critical Failures (Must Fix)

- Any NaN or Inf values
- Temperature < 0 K or > 100,000 K
- Fill level < 0 or > 1
- Extreme gradients > 10^8 K/m
- Energy change > 10%
- Mass change > 5%

### Warnings (Investigate)

- Temperature > 10,000 K (unless expected)
- Energy change > 5%
- Mass change > 1%
- Large standard deviation in conservation

### Acceptable (Monitor)

- Small numerical noise (< 1e-6)
- Energy drift < 1% over long simulations
- Mass drift < 0.1%

## Advanced Usage

### Custom Thresholds

Edit parameters at top of `verify_simulation_correctness.py`:

```python
TEMPERATURE_MIN = 0.0          # K
TEMPERATURE_MAX = 50000.0      # K
FILL_LEVEL_MIN = 0.0
FILL_LEVEL_MAX = 1.0
MAX_GRADIENT = 1e8             # K/m
ENERGY_CHANGE_THRESHOLD = 0.05 # 5%
```

### Specific Timesteps Only

```bash
python3 verify_simulation_correctness.py vtk_out/ --timesteps 0,100,200,300
```

### Different Material Properties

Modify in script (lines 336-337, 430):

```python
rho = 8900.0  # kg/m³ (e.g., copper)
cp = 385.0    # J/(kg·K)
```

### Automation in CI/CD

```bash
#!/bin/bash
# Run simulation
./LaserHeating

# Validate output
python3 /home/yzk/LBMProject/scripts/verify_simulation_correctness.py vtk_out/

# Check exit code
if [ $? -eq 0 ]; then
    echo "Validation passed"
    exit 0
else
    echo "Validation failed"
    exit 1
fi
```

## Testing

### Generate Test Data

```bash
python3 /home/yzk/LBMProject/scripts/test_verification_script.py
```

Creates synthetic test cases in `/home/yzk/LBMProject/scripts/test_vtk_data/`:
- Valid data (should pass all checks)
- Invalid data (should fail specific checks)

### Run Tests

```bash
# Test on valid data (should pass)
python3 verify_simulation_correctness.py test_vtk_data/ --timesteps 0,10,20,30

# Test quick check
python3 quick_vtk_check.py test_vtk_data/valid_0000.vtk
python3 quick_vtk_check.py test_vtk_data/invalid_0000.vtk
```

## File Format Requirements

Scripts support legacy VTK ASCII format:

```
# vtk DataFile Version 3.0
Description
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS nx ny nz
ORIGIN x0 y0 z0
SPACING dx dy dz
POINT_DATA npoints
SCALARS temperature float 1
LOOKUP_TABLE default
T1
T2
...
SCALARS fill_level float 1
LOOKUP_TABLE default
f1
f2
...
```

**Note:** XML VTK (.vtu) and binary formats are not currently supported by the simple reader. Use ParaView or VTK Python bindings to convert if needed.

## Performance

- **quick_vtk_check.py**: < 1 second per file
- **verify_simulation_correctness.py**: ~5-10 seconds for 10 timesteps with 100k cells each

Memory usage scales linearly with number of cells and timesteps loaded.

## References

- Main verification script: `/home/yzk/LBMProject/scripts/verify_simulation_correctness.py`
- Quick check script: `/home/yzk/LBMProject/scripts/quick_vtk_check.py`
- Test generator: `/home/yzk/LBMProject/scripts/test_verification_script.py`
- README: `/home/yzk/LBMProject/scripts/README_VERIFICATION.md`
