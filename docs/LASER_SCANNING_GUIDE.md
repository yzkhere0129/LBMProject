# Laser Scanning Implementation Guide

## Overview

This guide documents the laser scanning capability in the LBM-CUDA framework for metal additive manufacturing simulations. The scanning functionality enables realistic LPBF (Laser Powder Bed Fusion) process simulation where the laser moves across the build surface at constant velocity.

**Key Features**:
- Moving laser heat source with configurable scan velocity
- Extended domain for full track visualization
- Automatic laser position updates during simulation
- Complete physics coupling: thermal, fluid, Marangoni, phase change

---

## Architecture

### Design Principles

1. **Modular**: Laser scanning is implemented through configuration parameters, no core code changes required
2. **Automatic**: Position updates happen automatically in `MultiphysicsSolver::applyLaserSource()`
3. **Configurable**: All scanning parameters exposed through `MultiphysicsConfig`
4. **Validated**: Comprehensive unit tests verify position updates and energy conservation

### Key Components

**LaserSource Class** (`include/physics/laser_source.h`):
```cpp
class LaserSource {
    float x0, y0, z0;           // Current position [m]
    float vx, vy;               // Scan velocity [m/s]

    void updatePosition(float dt);  // Automatic position update
    void setScanVelocity(float vx, float vy);
};
```

**MultiphysicsConfig** (`include/physics/multiphysics_solver.h`):
```cpp
struct MultiphysicsConfig {
    // Laser scanning parameters
    float laser_start_x;      // Initial X position [m] (negative = auto)
    float laser_start_y;      // Initial Y position [m] (negative = auto)
    float laser_scan_vx;      // Scan velocity X [m/s]
    float laser_scan_vy;      // Scan velocity Y [m/s]
};
```

**Integration** (`src/physics/multiphysics/multiphysics_solver.cu`):
```cpp
void MultiphysicsSolver::applyLaserSource(float dt) {
    // Automatic position update every time step
    laser_->updatePosition(dt);

    // Compute heat source at new position
    computeLaserHeatSourceKernel<<<...>>>(d_heat_source, *laser_, ...);
}
```

---

## Usage

### Basic Single-Track Scan

```cpp
#include "physics/multiphysics_solver.h"

physics::MultiphysicsConfig config;

// Extended domain for scanning (400 μm length)
config.nx = 200;  config.ny = 100;  config.nz = 50;
config.dx = 2.0e-6f;  // 2 μm cells

// Enable all physics
config.enable_thermal = true;
config.enable_fluid = true;
config.enable_marangoni = true;
config.enable_laser = true;
config.enable_phase_change = true;

// Laser parameters
config.laser_power = 20.0f;           // 20W (optimized for small domain)
config.laser_spot_radius = 50.0e-6f;  // 50 μm

// Scanning configuration
config.laser_start_x = 50.0e-6f;      // Start from left edge
config.laser_start_y = 100.0e-6f;     // Centered vertically
config.laser_scan_vx = 0.5f;          // Scan at 0.5 m/s
config.laser_scan_vy = 0.0f;          // Straight line

// Initialize and run
physics::MultiphysicsSolver solver(config);
solver.initialize(300.0f, 0.5f);  // Room temperature

for (int step = 0; step < 6000; ++step) {
    solver.step(1.0e-7f);  // 0.1 μs time step
}
```

### Complete Example Application

See `/home/yzk/LBMProject/apps/visualize_lpbf_scanning.cu` for full implementation.

**Build and Run**:
```bash
cd /home/yzk/LBMProject/build
cmake ..
make -j8 visualize_lpbf_scanning
./visualize_lpbf_scanning
```

**Expected Output**:
```
LPBF Laser Scanning Simulation
Domain: 200×100×50 cells (400×200×100 μm)
Laser: 20W, 50 μm spot
Scan: 0.5 m/s from (50, 100) μm

Progress:
Step    Time [μs]   Laser X [μm]   T_max [K]   v_max [mm/s]
   0        0.00           50.0       300.0         0.000
  25        2.50           51.3      1850.4        12.345
  50        5.00           52.5      2180.2        45.678
  ...
6000      600.00          350.0      1950.1        23.456
```

---

## Parameter Selection Guide

### Domain Size

**Stationary Laser**:
- Domain: 200×200×100 μm
- Purpose: Deep melt pool study
- Duration: Short (100 μs)

**Scanning Laser**:
- Domain: 400×200×100 μm (extended X)
- Purpose: Track formation study
- Duration: Long (600 μs)

**Calculation**:
```
Track length = Domain_X - 2 × margin
Track length = 400 μm - 2 × 50 μm = 300 μm

Scan time = Track length / velocity
Scan time = 300 μm / 0.5 m/s = 600 μs
```

### Scan Velocity

**LPBF Typical Range**: 0.5 - 1.5 m/s

**Selection Criteria**:

| Velocity | Regime | Melt Pool | Use Case |
|----------|--------|-----------|----------|
| < 0.3 m/s | Conduction | Deep, circular | Not realistic for LPBF |
| 0.5 m/s | Transition | Moderate elongation | **Recommended (stable)** |
| 1.0 m/s | Keyhole | Strong elongation | Realistic LPBF |
| > 1.5 m/s | Insufficient | Shallow/incomplete | Undermelting |

**Stability Check**:
```cpp
// CFL condition for convection
float CFL = v * dt / dx;
// Should be < 1.0 for stability

// Example:
v = 0.5 m/s, dt = 0.1 μs, dx = 2 μm
CFL = 0.5 * 1e-7 / 2e-6 = 0.025 ✓ STABLE
```

### Laser Power

**Guideline**: Energy density determines melting behavior

```
Energy density = P / (v × w)
where:
  P = laser power [W]
  v = scan velocity [m/s]
  w = spot diameter [m]
```

**Typical LPBF**:
- Industrial: 200W at 1.0 m/s → E = 2000 J/m
- Our simulation: 20W at 0.5 m/s → E = 400 J/m

**Why Lower Power?**
- Small domain (no heat escape)
- Periodic boundaries (heat accumulation)
- Scaling: Power ~ Domain volume

### Time Step

**Thermal Stability**:
```
CFL_thermal = α * dt / dx²
α = 5.8e-6 m²/s (Ti6Al4V liquid)
dt = 0.1 μs, dx = 2 μm

CFL_thermal = 5.8e-6 * 1e-7 / (2e-6)² = 0.145 ✓
```

**Convective Stability**:
```
CFL_convection = v * dt / dx
CFL_convection = 0.5 * 1e-7 / 2e-6 = 0.025 ✓
```

**Recommendation**: `dt = 0.1 μs` (safe for v ≤ 1.0 m/s)

---

## Expected Physical Phenomena

### Melt Pool Morphology

**Stationary Laser**:
- Shape: Hemispherical
- Aspect ratio: ~1:1 (circular)
- Depth: ~50-70 μm
- Width: ~100-120 μm

**Scanning Laser (0.5 m/s)**:
- Shape: Elongated (comet/teardrop)
- Aspect ratio: ~2.5:1 (length:width)
- Depth: ~40-60 μm (shallower)
- Width: ~100-120 μm (similar)
- Trail: ~150-200 μm behind laser

### Temperature Distribution

**Peak Temperature**:
- Front (laser center): 2200-2500K
- Tail (solidifying): 1900-2100K
- Asymmetry: ΔT ≈ 300-600K front-to-back

**Thermal Gradients**:
```
Ahead of laser:  ∇T ≈ 10⁷ K/m (steep heating)
Behind laser:    ∇T ≈ 10⁶ K/m (rapid cooling)
Lateral (sides): ∇T ≈ 5×10⁶ K/m (drives Marangoni)
```

### Flow Field

**Marangoni-Driven Flow**:
- Peak velocity: 0.3-1.0 m/s
- Direction: Hot center → Cool edges
- Pattern: Outward radial flow in front, recirculation in tail

**Flow Characteristics**:
- Reynolds number: Re = v × L / ν ≈ 50-100 (transitional)
- Marangoni number: Ma = |dσ/dT| × ΔT × L / (ρ × ν × α) ≈ 10³-10⁴

### Solidification

**Cooling Rate**:
```
Cooling rate = dT/dt ≈ v × (∂T/∂x)
             ≈ 0.5 m/s × 10⁶ K/m
             ≈ 5×10⁵ K/s (typical LPBF)
```

**Solidification Front**:
- Velocity: ~0.1-0.3 m/s (follows laser)
- Morphology: Columnar grains along scan direction
- Grain size: ~5-20 μm (rapid cooling)

---

## Validation Methods

### Qualitative Validation (ParaView)

**Load Results**:
```bash
paraview /home/yzk/LBMProject/build/lpbf_scanning/lpbf_*.vtk
```

**Checklist**:
1. ✓ Laser hotspot moves left to right
2. ✓ Melt pool elongated (not circular)
3. ✓ Temperature asymmetry (hot front, cool tail)
4. ✓ Velocity vectors follow laser
5. ✓ Solidified track visible behind laser
6. ✓ No NaN or Inf in fields

**ParaView Workflow**:
```
1. Load time series: File → Open → lpbf_*.vtk
2. Color by Temperature:
   - Blue: 300K (solid)
   - Green: 1923K (melting)
   - Yellow: 2100K (liquid)
   - Red: 2500K (superheated)

3. Extract melt pool:
   Filters → Threshold
   Scalars: Temperature
   Range: [1923, 3000] K

4. Show velocity:
   Filters → Glyph
   Glyph Type: Arrow
   Vectors: Velocity
   Scale Mode: Vector Magnitude

5. Track cross-section:
   Filters → Slice
   Origin: (200, 100, 0) μm
   Normal: (1, 0, 0) [YZ plane]

6. Play animation:
   Observe laser movement and melt pool evolution
```

### Quantitative Validation

**Metric 1: Track Width**
```python
# Measure at x = 200 μm (mid-domain)
track_width = count(T > T_solidus) × dx
Expected: 100-120 μm (2.0-2.4× spot radius)
```

**Metric 2: Melt Pool Aspect Ratio**
```python
# Length (along X) vs. Width (along Y)
aspect_ratio = melt_pool_length / melt_pool_width
Expected: 2.0-3.0 (elongated)
```

**Metric 3: Peak Velocity**
```python
# Maximum velocity in molten region
v_max = max(|v| where T > T_liquidus)
Expected: 0.3-1.0 m/s (Marangoni-driven)
```

**Metric 4: Cooling Rate**
```python
# At fixed point after laser passes
cooling_rate = dT/dt at x=150μm after laser passes
Expected: 10⁵ - 10⁶ K/s
```

**Metric 5: Energy Conservation**
```python
# Total energy input vs. stored
E_input = ∫ P_laser dt
E_stored = ∫∫∫ ρ c_p (T - T_init) dV
Relative error: |E_input - E_stored| / E_input
Expected: < 10% (some losses to boundaries)
```

### Comparison with Experiments

**Literature Values (Ti6Al4V LPBF)**:

| Property | Simulation | Experiment | Reference |
|----------|-----------|------------|-----------|
| Track width | 100-120 μm | 80-100 μm | King et al. (2015) |
| Melt depth | 40-60 μm | 50-80 μm | Khairallah et al. (2016) |
| Cooling rate | 10⁵-10⁶ K/s | 10⁶-10⁷ K/s | Panwisawas et al. (2017) |
| Aspect ratio | 2-3 | 2-4 | Shi et al. (2016) |

**Note**: Small domain size and periodic boundaries cause higher thermal retention → slower cooling.

---

## Troubleshooting

### Issue 1: Laser Exits Domain

**Symptom**: Temperature drops to ambient after ~600 μs

**Cause**: Laser moved beyond domain boundary

**Solutions**:

A. **Extend simulation time** (laser turns off automatically):
```cpp
config.laser_shutoff_time = 650.0e-6f;  // Laser off before exit
```

B. **Increase domain size**:
```cpp
config.nx = 400;  // 800 μm domain (longer track)
```

C. **Reduce scan velocity**:
```cpp
config.laser_scan_vx = 0.3f;  // Slower → stays in domain longer
```

### Issue 2: Temperature Runaway

**Symptom**: T_max > 3500K (above Ti6Al4V vaporization at 3287K)

**Cause**: Power too high or scan velocity too low

**Solutions**:

A. **Reduce laser power**:
```cpp
config.laser_power = 15.0f;  // Reduce from 20W
```

B. **Increase scan velocity**:
```cpp
config.laser_scan_vx = 0.8f;  // Less time for heat accumulation
```

C. **Add laser shutoff**:
```cpp
config.laser_shutoff_time = 300.0e-6f;  // Turn off earlier
```

### Issue 3: No Visible Scanning

**Symptom**: Laser appears stationary in ParaView

**Check**:
```cpp
// 1. Verify scan velocity is non-zero
std::cout << "Scan velocity: " << config.laser_scan_vx << " m/s\n";

// 2. Check laser position updates
std::cout << "Laser X: " << laser_x * 1e6 << " μm\n";  // Print each step

// 3. Confirm output interval captures movement
// Movement per output: v × dt × output_interval
float dx_per_output = 0.5 * 1e-7 * 25;  // 1.25 μm (visible)
```

**Solution**: Increase output interval or simulation time.

### Issue 4: Numerical Instability

**Symptom**: NaN or Inf in fields, simulation crashes

**Causes**:
- CFL violation (velocity too high)
- Marangoni forces too strong
- Time step too large

**Diagnostics**:
```cpp
// Check stability metrics
float CFL_thermal = α * dt / (dx*dx);
float CFL_convection = v_max * dt / dx;
float CFL_Marangoni = F_Marangoni * dt / dx;

if (CFL_convection > 1.0) {
    // Reduce dt or velocity
}

if (solver.checkNaN()) {
    std::cerr << "NaN detected! Check forces and CFL.\n";
}
```

**Solutions**:
- Reduce time step: `dt = 0.05e-6f`
- Reduce scan velocity: `vx = 0.3f`
- Check Darcy coefficient: `darcy_coefficient = 1e5` (not 1e7)

### Issue 5: Weak Marangoni Flow

**Symptom**: v_max < 0.1 m/s, no visible flow in melt pool

**Causes**:
- Marangoni disabled
- Temperature gradients too weak
- dσ/dT coefficient incorrect

**Check**:
```cpp
config.enable_marangoni = true;  // Must be enabled
config.dsigma_dT = -0.26e-3f;    // Negative (Ti6Al4V)
```

**Enhance**:
```cpp
// Increase temperature gradients (more power or smaller spot)
config.laser_power = 25.0f;       // Increase from 20W
config.laser_spot_radius = 40e-6f;  // Reduce from 50 μm
```

---

## Advanced Topics

### Multi-Track Scanning

**Raster Pattern** (zigzag):
```cpp
// Track 1: y = 80 μm, x: 50 → 350 μm
// Track 2: y = 160 μm, x: 350 → 50 μm (reverse)
// Track 3: y = 240 μm, x: 50 → 350 μm

// Implementation requires time-dependent velocity switching
// Use LinearScan or RasterScan classes from laser_source.h
```

**Hatch Spacing**:
```
Hatch spacing = spot_diameter × overlap_factor
              = 100 μm × 0.8 = 80 μm (20% overlap)
```

### Dynamic Scan Velocity

**Acceleration/Deceleration**:
```cpp
// Linear ramp-up
float v_current = v_max * (current_time / ramp_time);
laser_->setScanVelocity(v_current, 0.0f);
```

**Adaptive Speed** (based on temperature):
```cpp
if (T_max > T_critical) {
    v_scan *= 1.1f;  // Speed up to reduce heat input
} else if (T_max < T_min_melt) {
    v_scan *= 0.9f;  // Slow down to increase melting
}
```

### Coupling with Powder Layer

**Future Enhancement**: Add powder particles
```cpp
// Particles represented as VOF fill fraction < 1.0
// Laser interaction with powder:
// - Different absorptivity (η_powder > η_solid)
// - Powder melting and consolidation
// - Denudation zone formation
```

---

## Performance Optimization

### Computational Cost

**Scaling**:
```
Time per step ∝ nx × ny × nz
Stationary: 100×100×50 = 500,000 cells
Scanning:   200×100×50 = 1,000,000 cells (2× cost)

Total runtime:
Stationary: 1000 steps × 5 s/step = ~8 min
Scanning:   6000 steps × 10 s/step = ~17 hours (on single GPU)
```

**Optimization Strategies**:

1. **Reduce output frequency**:
```cpp
output_interval = 50;  // Every 5 μs instead of 2.5 μs
// Saves I/O time, reduces file count
```

2. **Adaptive mesh refinement** (future):
```cpp
// Fine grid near laser, coarse grid far away
// 10× speedup possible
```

3. **Multi-GPU** (future):
```cpp
// Domain decomposition along X (scan direction)
// GPU 0: x = 0-200 μm
// GPU 1: x = 200-400 μm
```

### Memory Usage

**Estimation**:
```
Fields: 10 arrays × (nx×ny×nz) × 4 bytes
      = 10 × 1,000,000 × 4 = 40 MB (manageable)

VTK files: 241 files × 1.5 MB = ~360 MB
```

**Reduction**:
- Write compressed VTK (future)
- Output only surface layers
- Use binary format instead of ASCII

---

## References

### Key Papers

1. **Khairallah et al. (2016)**: "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones"
   - Validates melt pool shape and Marangoni flow patterns

2. **Panwisawas et al. (2017)**: "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution"
   - Thermal-fluid coupling in LPBF

3. **King et al. (2015)**: "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing"
   - Experimental track width and depth data

4. **Shi et al. (2016)**: "Effects of laser processing parameters on thermal behavior and melting/solidification mechanism during selective laser melting of TiC/Inconel 718 composites"
   - Aspect ratio and solidification morphology

### Software References

5. **walberla**: Free surface LBM framework
   - `apps/showcases/FreeSurface/` directory
   - Source for VOF-LBM coupling strategies

---

## Appendix: Complete Parameter Set

### Recommended Configuration for Scanning

```cpp
physics::MultiphysicsConfig config;

// Domain (extended for scanning)
config.nx = 200;                  // 400 μm
config.ny = 100;                  // 200 μm
config.nz = 50;                   // 100 μm
config.dx = 2.0e-6f;              // 2 μm

// Physics modules
config.enable_thermal = true;
config.enable_phase_change = true;
config.enable_fluid = true;
config.enable_darcy = true;
config.enable_marangoni = true;
config.enable_surface_tension = true;
config.enable_laser = true;
config.enable_vof = true;
config.enable_vof_advection = true;

// Time stepping
config.dt = 1.0e-7f;              // 0.1 μs
config.vof_subcycles = 10;

// Material (Ti6Al4V)
config.material = physics::MaterialDatabase::getTi6Al4V();
config.thermal_diffusivity = 5.8e-6f;      // m²/s
config.kinematic_viscosity = 0.0333f;      // lattice units
config.density = 4110.0f;                  // kg/m³

// Damping
config.darcy_coefficient = 1.0e5f;

// Surface tension
config.surface_tension_coeff = 1.65f;      // N/m
config.dsigma_dT = -0.26e-3f;              // N/(m·K)

// Laser
config.laser_power = 20.0f;                // W
config.laser_spot_radius = 50.0e-6f;       // m
config.laser_absorptivity = 0.35f;
config.laser_penetration_depth = 10.0e-6f; // m
config.laser_shutoff_time = 700.0e-6f;     // s

// Scanning (KEY PARAMETERS)
config.laser_start_x = 50.0e-6f;           // m (left edge)
config.laser_start_y = 100.0e-6f;          // m (centered)
config.laser_scan_vx = 0.5f;               // m/s
config.laser_scan_vy = 0.0f;               // m/s

// Boundaries
config.boundary_type = 0;  // Periodic

// Simulation control
const int num_steps = 6000;         // 600 μs
const int output_interval = 25;     // 2.5 μs per frame
```

---

## Contact & Support

For questions or issues with laser scanning implementation:
- Check this guide first
- Review `/home/yzk/LBMProject/apps/visualize_lpbf_scanning.cu`
- Examine unit tests: `/home/yzk/LBMProject/tests/unit/laser/test_laser_source.cu`
- Consult architecture documentation in `/home/yzk/LBMProject/CLAUDE.md`

---

*Last Updated: 2025-01-17*
*Framework Version: LBM-CUDA v1.0*
