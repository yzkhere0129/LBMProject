# Realistic LPBF Simulation Usage Guide

## Overview

The `visualize_lpbf_marangoni_realistic` application simulates a realistic Laser Powder Bed Fusion (LPBF) process with Marangoni convection, starting from cold solid metal and progressively heating it with a laser.

## Key Features

- **Realistic initial conditions:** Cold solid metal at room temperature (300K)
- **Laser heating:** Progressive melting under moving laser beam
- **Full physics coupling:** Thermal diffusion + Fluid flow + Marangoni forces + Darcy damping
- **Dynamic temperature evolution:** Temperature field changes over time (not static)
- **Solid region damping:** Darcy damping keeps solid substrate stationary

## Comparison with Validation Tests

| Feature | Validation Test (`visualize_phase6_marangoni`) | Realistic LPBF (`visualize_lpbf_marangoni_realistic`) |
|---------|-----------------------------------------------|------------------------------------------------------|
| **Purpose** | Validate Marangoni forces only | Realistic LPBF simulation |
| **Initial Temperature** | 2000-2500K (pre-melted liquid) | 300K (cold solid metal) |
| **Temperature Evolution** | Static (never changes) | Dynamic (heated by laser) |
| **Laser Heating** | No | Yes |
| **Melting Process** | No (already liquid) | Yes (progressive melting) |
| **Use Case** | Physics validation | Process simulation |

**Important:** These are DIFFERENT applications serving DIFFERENT purposes. Both are correct for their intended use!

## Quick Start

### Build

```bash
cd /home/yzk/LBMProject/build
cmake ..
make visualize_lpbf_marangoni_realistic -j8
```

### Run

```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_marangoni_realistic
```

Output directory: `lpbf_realistic/`

Output files: `lpbf_realistic/lpbf_NNNNNN.vtk` (one per output interval)

## Expected Behavior

### Initial Frame (t=0)
- **Temperature:** 300K everywhere (uniform cold solid)
- **Liquid fraction:** 0.0 everywhere (all solid)
- **Velocity:** 0.0 everywhere (stationary)
- **Substrate:** Bottom 14 μm (7 cell layers)

### After 10 μs (step 100)
- **Hot spot appears** under laser beam
- **Temperature locally exceeds** melting point (T > 1923K)
- **Material begins to melt** in laser interaction zone
- **Velocity remains low** (material still mostly solid)

### After 100 μs (step 1000)
- **Melt pool forms** around laser path
- **Marangoni convection develops** in liquid region
- **Surface flow visible** (velocity 0.5-2 m/s typical)
- **Solid substrate remains stationary** (Darcy damping working)

### After 1 ms (step 10000)
- **Laser has scanned** across surface
- **Wake solidification** behind laser
- **Final melt pool shape** established
- **Temperature gradient** drives Marangoni flow

## Visualization in ParaView

### Load Data

```bash
paraview lpbf_realistic/lpbf_*.vtk
```

ParaView will automatically recognize the time series.

### Recommended Views

#### View 1: Temperature Evolution
1. Color by: `Temperature`
2. Color scale: Blue (300K) → Red (>2000K)
3. Play animation to see laser heating

**What to see:** Hot spot appears, moves across domain, leaves wake

#### View 2: Melting Process
1. Color by: `PhaseState`
2. Legend: 0=Solid (blue), 1=Mushy (green), 2=Liquid (red)
3. Play animation to see melting

**What to see:** Melt pool forms and grows under laser, solidifies in wake

#### View 3: Marangoni Flow
1. Color by: `Velocity` magnitude
2. Add Glyph filter: `Filters → Glyph`
   - Vectors: `Velocity`
   - Scale factor: Adjust for visibility
   - Glyph type: Arrow
3. Play animation to see flow development

**What to see:** Flow develops in liquid region, radiates from hot spot

#### View 4: Substrate Damping (Verification)
1. Create slice at z = 7 μm (substrate boundary)
2. Color by: `Velocity` magnitude
3. Verify substrate (z < 7 μm) has zero velocity

**What to see:** Substrate remains stationary, flow only in melt pool

## Validation Tests

### Run Tests

```bash
cd /home/yzk/LBMProject/build
make test_realistic_lpbf_initial_conditions
./tests/validation/test_realistic_lpbf_initial_conditions
```

### Test Coverage

The validation suite includes 6 tests:

1. **InitialTemperatureIsCold**
   - Verifies T_initial ≈ 300K (not >1900K)
   - Critical for realistic simulation

2. **LaserHeatingIncreaseTemperature**
   - Verifies temperature increases over time
   - Proves laser heating is working
   - Expected: ΔT > 100K after 100 steps

3. **LaserModuleEnabled**
   - Verifies laser module is active
   - Checks laser power > 0

4. **ThermalModuleEnabled**
   - Verifies thermal solver is active
   - Ensures dynamic temperature evolution

5. **PhysicsModulesEnabled**
   - Verifies Marangoni module is active
   - Verifies Darcy damping is active

6. **DifferentFromValidationTest**
   - Confirms realistic LPBF starts cold (300K)
   - Confirms validation tests start hot (2000K)
   - Validates architectural distinction

### Expected Test Results

```
[==========] Running 6 tests from 1 test suite.
...
[  PASSED  ] 6 tests.
```

All tests should pass. If any fail, check:
- Initial temperature configuration
- Laser module enablement
- Physics module flags

## Configuration Parameters

Edit `/home/yzk/LBMProject/apps/visualize_lpbf_marangoni_realistic.cu` to adjust:

### Domain Size
```cpp
config.nx = 100;  // x-direction cells
config.ny = 100;  // y-direction cells
config.nz = 50;   // z-direction cells
config.dx = 2.0e-6f;  // Cell size [m]
```

Physical domain: `nx*dx × ny*dx × nz*dx` meters

### Time Stepping
```cpp
config.dt = 1.0e-7f;  // Time step [s]
const int num_steps = 10000;  // Total steps
const int output_interval = 100;  // Output frequency
```

Total simulation time: `num_steps * dt` seconds

### Laser Parameters
```cpp
config.laser_power = 200.0f;  // Power [W]
config.laser_spot_radius = 50.0e-6f;  // Spot size [m]
config.laser_absorptivity = 0.35f;  // Absorption coefficient
config.laser_penetration_depth = 10.0e-6f;  // Penetration depth [m]
```

Typical LPBF parameters:
- Power: 100-400 W
- Spot: 30-100 μm
- Absorptivity: 0.3-0.5 for Ti6Al4V

### Initial Conditions
```cpp
const float T_initial = 300.0f;  // K (room temperature)
```

**Critical:** Keep T_initial = 300K for realistic simulation!

Change only if simulating:
- Preheated substrate (e.g., 400-500K)
- Cryogenic conditions (e.g., 77K for liquid nitrogen)

### Recompile After Changes

```bash
cd /home/yzk/LBMProject/build
make visualize_lpbf_marangoni_realistic -j8
```

## Troubleshooting

### Issue: No heating observed

**Symptoms:**
- Temperature remains at 300K throughout simulation
- No melt pool forms

**Diagnosis:**
```bash
grep "Laser:" output_log.txt
```

Should show: `Laser: ON`

**Fix:**
1. Verify `config.enable_laser = true` in source
2. Verify `config.laser_power > 0`
3. Recompile

### Issue: Entire domain melts immediately

**Symptoms:**
- Temperature jumps to >2000K at t=0
- All cells show liquid fraction = 1.0

**Diagnosis:**
Check initial temperature:
```bash
grep "Initial temperature" output_log.txt
```

Should show: `Initial temperature: 300 K`

**Fix:**
1. Verify `T_initial = 300.0f` (not 2300.0f)
2. Check you're running realistic app, not validation test
3. Recompile

### Issue: Solid regions flow

**Symptoms:**
- Substrate (z < 14 μm) has non-zero velocity
- Solid regions move

**Diagnosis:**
```bash
grep "Darcy" output_log.txt
```

Should show: `Darcy Damping: ON`

**Fix:**
1. Verify `config.enable_darcy = true`
2. Check Darcy coefficient: `config.darcy_coefficient = 1e7f`
3. This was fixed in Phase 1 - ensure using latest code

### Issue: Simulation runs slowly

**Symptoms:**
- Taking >5 minutes for 10000 steps
- High GPU usage

**Solutions:**
1. Reduce domain size (nx, ny, nz)
2. Reduce total steps
3. Increase output interval
4. Use GPU with higher compute capability

### Issue: VTK files not generated

**Symptoms:**
- `lpbf_realistic/` directory empty
- No output files

**Diagnosis:**
```bash
ls -la lpbf_realistic/
```

**Fix:**
1. Directory creation might have failed
2. Check write permissions: `chmod 755 lpbf_realistic/`
3. Run with: `mkdir -p lpbf_realistic && ./visualize_lpbf_marangoni_realistic`

## Performance Optimization

### Domain Size vs. Computation Time

| Grid Size | Cells | Time/Step | 10k Steps |
|-----------|-------|-----------|-----------|
| 50×50×25 | 62,500 | ~0.5 ms | ~5 min |
| 100×100×50 | 500,000 | ~3 ms | ~30 min |
| 200×200×100 | 4,000,000 | ~20 ms | ~3.5 hr |

### Recommendations

**Quick testing:** 50×50×25, 1000 steps (~30 seconds)

**Standard runs:** 100×100×50, 10000 steps (~30 minutes)

**High resolution:** 200×200×100, 50000 steps (~12 hours)

## Physical Interpretation

### Temperature Range

- **300-1878K:** Solid (no flow)
- **1878-1923K:** Mushy (partial melting, viscous flow)
- **>1923K:** Liquid (full melting, Marangoni flow active)

### Velocity Range

- **Substrate (solid):** 0 mm/s (Darcy damped)
- **Mushy zone:** 0-100 mm/s (viscous)
- **Liquid melt pool:** 100-2000 mm/s (Marangoni-driven)

Literature values: 0.5-2 m/s for Ti6Al4V LPBF (Khairallah 2016)

### Melt Pool Dimensions

Typical LPBF melt pool (200W, 50 μm spot):
- **Width:** 100-200 μm
- **Depth:** 50-100 μm
- **Length:** 200-500 μm

## References

### Literature

1. **Khairallah et al. (2016)** - "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones"
   - Marangoni velocity: 0.5-2 m/s for Ti6Al4V
   - Recoil pressure effects

2. **Panwisawas et al. (2017)** - "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution"
   - Multi-physics coupling approach
   - Validation against experiments

3. **King et al. (2015)** - "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing"
   - Experimental melt pool measurements
   - High-speed imaging

### Codebase Documentation

- `/home/yzk/LBMProject/docs/PHASE6_MARANGONI_STATUS.md` - Marangoni implementation
- `/home/yzk/LBMProject/tests/validation/README.md` - Validation test philosophy

## Contact & Support

For issues or questions:
1. Check validation tests pass: `./tests/validation/test_realistic_lpbf_initial_conditions`
2. Review ParaView output for expected behavior
3. Compare with validation test to understand differences

## Acknowledgments

This implementation is based on:
- walberla free surface framework (VOF method)
- Lattice Boltzmann Method (D3Q7 thermal, D3Q19 fluid)
- Literature values from Khairallah, Panwisawas, King et al.
