# Surface Deformation Optimization Guide

## Overview

This document explains how to configure the LPBF simulation to maximize visible surface deformation in ParaView. The key insight is that **`enable_vof_advection = true`** must be set for the interface to move.

## Root Cause Analysis

**Problem**: Interface remains static (flat) despite Marangoni forces and melt pool formation.

**Root Cause**: `enable_vof_advection` defaults to `false` in `MultiphysicsConfig()`:

```cpp
// From include/physics/multiphysics_solver.h:144
enable_vof_advection(false),  // Disabled by default (Step 1)
```

**Solution**: Explicitly set `enable_vof_advection = true` in configuration file.

---

## Physics of Surface Deformation

### 1. What Causes Surface Depression?

In real LPBF without recoil pressure, surface depression is driven by:

1. **Marangoni Convection** (dominant)
   - Temperature gradient creates surface tension gradient
   - Fluid flows from hot (low surface tension) to cold (high surface tension)
   - Creates radially outward flow at surface
   - Conservation of mass causes surface to depress at center

2. **Buoyancy/Natural Convection**
   - Hot liquid rises, cold liquid sinks
   - Creates circulation that reinforces surface depression

3. **Surface Tension Relaxation**
   - Curved interface creates pressure difference (Young-Laplace)
   - Helps maintain smooth interface shape

### 2. Key Dimensionless Numbers

| Number | Definition | Typical LPBF | Significance |
|--------|------------|--------------|--------------|
| Marangoni (Ma) | `(dsigma/dT * DeltaT * L) / (mu * alpha)` | 100-1000 | Ma > 100: Strong surface flow |
| Capillary (Ca) | `mu * v / sigma` | 0.01-0.1 | Ca < 0.1: Surface tension dominates |
| Reynolds (Re) | `rho * v * L / mu` | 10-100 | Re < 100: Laminar, stable |
| CFL | `u * dt / dx` | < 0.5 | Stability criterion |

---

## Parameter Optimization Strategy

### 1. Laser Parameters

#### Power (laser_power)

```
Higher power -> Larger melt pool -> Stronger temperature gradient -> More Marangoni
```

| Power (W) | Melt Pool Size | Surface Depression | Stability |
|-----------|----------------|-------------------|-----------|
| 150 | Small (~80 um) | Minimal (3-8 um) | Excellent |
| 200 | Medium (~120 um) | Moderate (8-15 um) | Good |
| **280** | **Large (~160 um)** | **Visible (15-30 um)** | **Good** |
| 350+ | Very large | Deep (30+ um) | May need smaller dt |

**Recommendation**: 250-300W for optimal visibility without instability.

#### Spot Radius (laser_spot_radius)

```
Larger spot -> Wider melt pool -> More area for surface depression
```

| Radius (um) | Effect |
|-------------|--------|
| 40 | Concentrated heating, deep narrow pool |
| **60-70** | **Good balance: wide pool, visible depression** |
| 100+ | Very wide but shallow |

**Recommendation**: 60-70 um for maximum visible area.

#### Stationary vs Moving Laser

- **Stationary**: Maximum depression at one location (best for demos)
- **Moving**: Elongated trailing depression (more realistic but less dramatic)

### 2. Marangoni Coefficient (dsigma_dT)

This is the **most critical** parameter for surface-driven flow.

```
F_marangoni = dsigma_dT * grad(T)
```

| dsigma_dT (N/m/K) | Effect | Accuracy |
|-------------------|--------|----------|
| -0.26e-3 | Literature Ti6Al4V | High |
| -0.35e-3 | Enhanced visibility | Medium |
| **-0.40e-3** | **Strong visible flow** | **Acceptable** |
| -0.50e-3+ | Dramatic effect | Low (may cause instability) |

**Physical Range for Ti6Al4V**: -0.26e-3 to -0.35e-3 N/(m*K)

**Recommendation**: -0.40e-3 for visualization, -0.26e-3 for validation.

### 3. Darcy Coefficient (darcy_coefficient)

Controls damping in mushy zone and solid regions.

```
F_darcy = -darcy_coefficient * (1 - f_liquid)^2 / f_liquid^3 * velocity
```

| Value | Effect |
|-------|--------|
| 1e7+ | Very strong damping (nearly rigid solid) |
| 1000-5000 | Moderate damping |
| **300-500** | **Low damping, dynamic flow** |
| < 200 | Too low, may cause instability |

**Recommendation**: 400 for maximum visible dynamics.

### 4. VOF Parameters

#### enable_vof_advection (CRITICAL)

```
enable_vof_advection = true    # Interface moves with fluid
enable_vof_advection = false   # Interface stays static (useless for visualization)
```

**Must be TRUE for any visible surface deformation.**

#### vof_subcycles

Number of VOF subcycles per LBM timestep. Higher = more stable but slower.

| Value | Use Case |
|-------|----------|
| 10 | Conservative physics |
| **15-20** | **Strong dynamics (recommended)** |
| 25+ | Extreme forces (may be needed) |

### 5. Surface Tension (enable_surface_tension)

**Should be TRUE** for realistic surface deformation.

- Smooths interface (prevents spurious oscillations)
- Provides restoring force (interface springs back when laser off)
- Creates capillary waves (visible surface ripples)

---

## Recommended Configurations

### Configuration A: Maximum Visibility (Recommended)

File: `configs/lpbf_MAX_SURFACE_DEFORMATION.conf`

```
laser_power = 280.0
laser_spot_radius = 65.0e-6
dsigma_dT = -0.40e-3
darcy_coefficient = 400.0
enable_vof_advection = true
vof_subcycles = 20
enable_surface_tension = true
```

**Expected Results**:
- Depression depth: 15-30 um
- Max velocity: 0.5-2 m/s
- Melt pool: ~160 um diameter

### Configuration B: Physical Accuracy

File: `configs/lpbf_SURFACE_DEFORMATION_CONSERVATIVE.conf`

```
laser_power = 200.0
laser_spot_radius = 50.0e-6
dsigma_dT = -0.26e-3
darcy_coefficient = 1000.0
enable_vof_advection = true
vof_subcycles = 15
```

**Expected Results**:
- Depression depth: 5-15 um
- Max velocity: 0.3-1.0 m/s
- Matches literature data

### Configuration C: Extreme Visual Effect

File: `configs/lpbf_SURFACE_DEFORMATION_EXTREME.conf`

```
laser_power = 350.0
laser_spot_radius = 80.0e-6
dsigma_dT = -0.50e-3
darcy_coefficient = 250.0
enable_vof_advection = true
vof_subcycles = 25
```

**Expected Results**:
- Depression depth: 30-50 um
- Chaotic dynamics
- NOT physically accurate

---

## ParaView Visualization Guide

### 1. Load Data

```
File -> Open -> output_dir/lpbf_*.vtk
Click "Apply"
```

### 2. View Surface (Contour Filter)

1. Select data in Pipeline Browser
2. Filters -> Common -> Contour
3. Contour By: **FillLevel**
4. Isosurfaces: **0.5**
5. Click Apply

### 3. Color by Temperature

1. Select Contour in Pipeline Browser
2. Coloring: **Temperature**
3. Adjust color map: Cool to Warm or similar

### 4. Animate

1. Click Play button (or use arrow keys)
2. Adjust animation speed if needed

### 5. What to Look For

| Time | Observation |
|------|-------------|
| 0-50 us | Surface begins to depress |
| 50-200 us | Maximum depression develops |
| 200-300 us | Steady state (if laser on) |
| After shutoff | Surface relaxes, capillary waves |

---

## Troubleshooting

### Problem: Surface doesn't move

**Causes**:
1. `enable_vof_advection = false` (most common)
2. `enable_vof = false`
3. No melting occurring (check temperature)

**Solution**: Verify in config file:
```
enable_vof = true
enable_vof_advection = true
```

### Problem: Simulation explodes (NaN)

**Causes**:
1. Time step too large
2. Forces too strong (Darcy too low, dsigma_dT too high)
3. CFL violation

**Solutions**:
- Reduce `dt` by factor of 2
- Increase `darcy_coefficient`
- Increase `vof_subcycles`
- Reduce `dsigma_dT`

### Problem: Surface looks jagged/noisy

**Causes**:
1. Surface tension disabled
2. Resolution too low
3. VOF subcycles too low

**Solutions**:
- Enable surface tension: `enable_surface_tension = true`
- Increase resolution (smaller dx)
- Increase `vof_subcycles`

### Problem: Depression too small to see

**Causes**:
1. Laser power too low
2. Marangoni too weak
3. Darcy damping too strong

**Solutions**:
- Increase `laser_power`
- Increase `|dsigma_dT|`
- Decrease `darcy_coefficient`

---

## Summary: Critical Settings Checklist

```
[x] enable_vof_advection = true    # Interface motion
[x] enable_vof = true              # VOF solver
[x] enable_surface_tension = true  # Smooth interface
[x] enable_marangoni = true        # Surface flow driver
[x] vof_subcycles >= 15            # Stability
[x] laser_power >= 250             # Sufficient heating
[x] darcy_coefficient <= 500       # Allow flow
[x] dsigma_dT <= -0.35e-3          # Strong Marangoni
```

---

## References

1. Khairallah et al. (2016) "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones"
2. Panwisawas et al. (2017) "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution"
3. Wei et al. (2015) "Origin of spattering during selective laser melting of metallic powder"
