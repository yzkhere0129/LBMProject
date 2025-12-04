# Cavity Diagnostic Report: Melt Pool Depression Analysis

## Problem Description
Abnormal cavity/depression observed behind the melt pool during LPBF simulation.

## Diagnostic Approach
Created three test configurations to isolate the root cause:
1. **Baseline** - All effects enabled (reference)
2. **No Shrinkage** - `enable_solidification_shrinkage = false`
3. **No Evaporation** - `enable_evaporation_mass_loss = false`

## Key Files Modified

### Configuration Files Created
- `/home/yzk/LBMProject/config/diagnostic_baseline.cfg`
- `/home/yzk/LBMProject/config/diagnostic_no_shrinkage.cfg`
- `/home/yzk/LBMProject/config/diagnostic_no_evaporation.cfg`
- `/home/yzk/LBMProject/config/diagnostic_solidification_test.cfg`
- `/home/yzk/LBMProject/config/diagnostic_quick_test.cfg`

### Code Modifications

1. **Config Loader** (`/home/yzk/LBMProject/include/config/lpbf_config_loader.h`):
   - Added support for `enable_evaporation_mass_loss` parameter
   - Added support for `enable_solidification_shrinkage` parameter
   - Added support for `enable_recoil_pressure` parameter
   - Added VOF diagnostic flags to config summary output

2. **Multiphysics Solver** (`/home/yzk/LBMProject/src/physics/multiphysics/multiphysics_solver.cu`):
   - Enhanced evaporation diagnostic output (every 100 steps):
     - Active evaporation cells count
     - Max evaporation flux and location
     - Expected fill_level change per timestep
   - Enhanced solidification shrinkage diagnostic output:
     - Total solidifying/melting cells
     - Interface-only solidifying cells (these are the ones affected)
     - Max solidification rate and location
     - Expected fill_level change with CORRECTED formula
     - Comparison with old incorrect /dx formula

### Diagnostic Script
- `/home/yzk/LBMProject/scripts/run_cavity_diagnostic.sh`
  - Automates running all three test cases
  - Extracts and summarizes diagnostic output
  - Provides ParaView visualization commands

## Key Findings (Test Results: 2024-11-22)

### Test Configuration
- Domain: 100 x 75 x 50 cells (200 x 150 x 100 um)
- Laser: 300W, 50um spot, stationary
- Laser ON: 0-50us, Laser OFF: 50-150us
- Total simulation: 1500 steps (150 us)

### Mass Conservation Analysis
- Initial mass: 303,749 (fill_level sum)
- Final mass: 292,402
- Total mass loss: 11,347 (3.73%)

### Evaporation Mass Loss (Step 400, peak evaporation)
- Active evaporation cells: 5,954
- Max J_evap: 391 kg/(m^2*s)
- Expected df per step: -4.77e-3 (0.48% fill_level reduction)
- Contribution: **SIGNIFICANT during laser ON phase**

### Solidification Shrinkage (Step 500, solidification begins)
- Beta (shrinkage factor): 0.0701 (7.01% volume contraction)
- Solidifying cells: 330
- Interface solidifying cells: 222 (only these get shrinkage applied)
- Max solidification rate: -1.24e6 1/s
- Expected df per step (CORRECTED): -8.67e-3 (0.87%)
- OLD formula would give df: -4336 (CATASTROPHIC BUG - fixed!)
- Contribution: **MODERATE during solidification phase**

### Surface Depression Analysis
- Initial z-centroid: 38.07 cells
- Final z-centroid: 36.85 cells
- Total depression: 1.22 cells (2.44 um)
- Depression timeline:
  - Steps 0-400: Rapid depression (-0.16 cells/100 steps) - EVAPORATION dominant
  - Steps 500+: Slower depression (-0.08 cells/100 steps) - SHRINKAGE dominant

### Formula Correction Status
The VOF solver (`vof_solver.cu`) now uses the **CORRECTED** dimensionless formula:
```cpp
df = beta * dfl_dt * dt  // Correct: [dimensionless]
```
Previously incorrect formula was:
```cpp
df = rate * beta * dt / dx  // WRONG: units [1/m]
```

### Additional Constraints (Already Implemented)
1. Only interface cells (0.01 < f < 0.99) can shrink
2. Only solidifying cells (rate < 0) experience shrinkage
3. Max 1% fill_level change per timestep (limiter)

## How to Run Diagnostic Tests

```bash
cd /home/yzk/LBMProject/build

# Run all tests
bash ../scripts/run_cavity_diagnostic.sh

# Or run individual tests:
./visualize_lpbf_scanning ../config/diagnostic_baseline.cfg --output lpbf_baseline
./visualize_lpbf_scanning ../config/diagnostic_no_shrinkage.cfg --output lpbf_no_shrinkage
./visualize_lpbf_scanning ../config/diagnostic_no_evaporation.cfg --output lpbf_no_evaporation
```

## Visualization

Open results in ParaView:
```bash
paraview lpbf_baseline/lpbf_*.vtk &
paraview lpbf_no_shrinkage/lpbf_*.vtk &
paraview lpbf_no_evaporation/lpbf_*.vtk &
```

Compare:
1. Color by `fill_level` to see cavity formation
2. Look at region behind melt pool (where solidification occurs)
3. Check if cavity depth differs between test cases

## Interpretation Guide

| Result | Cavity Present? | Conclusion |
|--------|----------------|------------|
| Baseline | Yes | Reference case |
| No Shrinkage | No/Reduced | Shrinkage is the cause |
| No Evaporation | No/Reduced | Evaporation is the cause |
| Both No-X | Yes | Other physics (recoil, advection) |

## Recommendations

1. If shrinkage is the cause:
   - Consider reducing beta or limiter threshold
   - Verify shrinkage formula is physically correct
   - Check if bulk cells are incorrectly being modified

2. If evaporation is the cause:
   - Review evaporation model parameters
   - Check if evaporation is occurring in correct cells
   - Verify temperature thresholds

3. If neither eliminates cavity:
   - Check recoil pressure implementation
   - Review VOF advection scheme
   - Verify velocity field near solidification front

## Files Summary

| File | Purpose |
|------|---------|
| `diagnostic_baseline.cfg` | Reference with all effects |
| `diagnostic_no_shrinkage.cfg` | Shrinkage disabled |
| `diagnostic_no_evaporation.cfg` | Evaporation disabled |
| `run_cavity_diagnostic.sh` | Automation script |
| `lpbf_config_loader.h` | Config loading with new flags |
| `multiphysics_solver.cu` | Enhanced diagnostics |

---
Generated: 2024-11-22
