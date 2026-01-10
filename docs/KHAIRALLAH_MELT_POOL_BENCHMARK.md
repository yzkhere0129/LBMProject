# Khairallah 2016 Melt Pool Size Benchmark Test

## Overview

This validation test replicates the landmark laser powder-bed fusion (LPBF) simulations from:

**Khairallah, S. A., Anderson, A. T., Rubenchik, A., & King, W. E. (2016).**
*"Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones."*
**Acta Materialia, 108, 36-45.**

This paper established key benchmarks for computational modeling of LPBF processes, particularly melt pool dimensions and keyhole formation dynamics.

## Scientific Context

### Importance of Melt Pool Validation

The Khairallah 2016 paper is seminal in LPBF computational research because:

1. **Direct comparison to experiments**: The simulations were validated against high-speed X-ray imaging of actual LPBF processes
2. **Physics completeness**: Includes thermal diffusion, Marangoni convection, surface tension, recoil pressure, and evaporation
3. **Wide parameter space**: Covers conduction mode and keyhole mode regimes
4. **Industrial relevance**: Uses real LPBF parameters (316L, typical laser powers, scan speeds)

### Key Physical Insights from Paper

1. **Melt pool geometry**: Width ~100-150 μm, depth ~50-100 μm in conduction mode
2. **Marangoni convection**: Dominates melt pool fluid flow with velocities 0.5-2 m/s
3. **Denudation zone**: Vapor-driven powder displacement creates ~100 μm zone
4. **Keyhole formation**: Recoil pressure depression forms at high powers (>300 W)
5. **Pore formation**: Result of keyhole collapse and vapor entrainment

## Test Configuration

### Domain Setup

```
Grid:             300 × 200 × 100 cells
Grid spacing:     2.0 μm
Domain size:      600 × 400 × 200 μm³
Coordinate system: X (scan direction), Y (transverse), Z (vertical, upward)
```

### Material: 316L Stainless Steel

From Khairallah 2016 and ASM Metals Handbook:

| Property | Value | Units |
|----------|-------|-------|
| Density (solid) | 7990 | kg/m³ |
| Density (liquid) | 6900 | kg/m³ |
| Specific heat (solid) | 500 | J/(kg·K) |
| Thermal conductivity | 16.2 | W/(m·K) |
| Melting point (solidus) | 1658 | K |
| Melting point (liquidus) | 1700 | K |
| Boiling point | 3090 | K |
| Latent heat of fusion | 260 | kJ/kg |
| Surface tension | 1.75 | N/m |
| dσ/dT | -0.43 | mN/(m·K) |

### Laser Parameters

Following Khairallah 2016:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Power | 195 W | Conduction mode regime |
| Spot radius (1/e²) | 40 μm | FWHM ≈ 94 μm |
| Absorptivity | 0.38 | 316L typical at 1064 nm |
| Penetration depth | 10 μm | Beer-Lambert absorption |
| Scan speed | 1.0 m/s | Middle of experimental range |
| Scan length | 400 μm | Sufficient for steady state |

**Energy density calculation:**
```
E = P / (v × D) = 195 W / (1.0 m/s × 94 μm) ≈ 2.07 J/mm²
```

This is in the typical LPBF range (0.5-5 J/mm²).

### Physics Enabled

Full multiphysics coupling:

- ✅ Thermal diffusion (D3Q7 LBM)
- ✅ Thermal advection (v·∇T coupling)
- ✅ Phase change (melting/solidification with enthalpy method)
- ✅ Incompressible Navier-Stokes (D3Q19 LBM)
- ✅ VOF interface tracking
- ✅ Marangoni convection (thermocapillary)
- ✅ Surface tension (Laplace pressure)
- ✅ Darcy damping in mushy/solid zones
- ✅ Buoyancy (Boussinesq approximation)
- ✅ Evaporative mass loss
- ✅ Solidification shrinkage
- ❌ Recoil pressure (disabled for conduction mode test)

### Numerical Parameters

```
Timestep:           100 ns
Total time:         600 μs (400 μs scan + 200 μs solidification)
Total steps:        6,000,000
Output interval:    10 μs (every 100 steps)
VTK output:         Every 50 μs
```

**Stability verification:**
```
Thermal CFL: ω = 1/(0.5 + 3αΔt/Δx²) ≈ 1.4 < 1.9 ✓
where α = k/(ρcp) = 2.57×10⁻⁶ m²/s
```

## Validation Metrics

### Literature Values (Khairallah 2016)

From paper's Table 1 and Figures 3-5:

| Metric | Conduction Mode (195 W, 1.0 m/s) | Notes |
|--------|-----------------------------------|-------|
| Melt pool width | 100-150 μm | Transverse dimension |
| Melt pool depth | 50-100 μm | Below initial surface |
| Melt pool length | 150-250 μm | Along scan direction |
| Peak temperature | 2500-3500 K | Approaching boiling |
| Marangoni velocity | 0.5-2.0 m/s | Surface flow |
| Denudation width | ~100 μm | Powder displacement |

### Success Criteria (±50% Tolerance)

Our validation criteria account for parameter uncertainties:

| Metric | Target Range | Rationale |
|--------|-------------|-----------|
| Width | 80-200 μm | ±50% of 100-150 μm |
| Depth | 30-150 μm | ±50% of 50-100 μm |
| Peak T | 2000-4000 K | ±20% of 2500-3500 K |
| Velocity | 0.3-3.0 m/s | ±50% of 0.5-2.0 m/s |

**Why ±50% tolerance?**

1. **Material property uncertainty**: Thermophysical properties (especially high-T) vary by 10-30% in literature
2. **Absorptivity variation**: Surface condition affects absorptivity (0.3-0.5 for 316L)
3. **Beam profile differences**: Gaussian assumption vs. actual measured profiles
4. **Grid resolution effects**: 2 μm resolution vs. finer grids in original work
5. **Powder vs. solid**: Original simulations include powder bed porosity effects

## Test Execution

### Build and Run

```bash
# From build directory
cmake --build . --target test_khairallah_melt_pool -j8
ctest -R KhairallahMeltPoolBenchmark -V
```

### Expected Runtime

- **Serial execution**: ~60 minutes
- **Memory usage**: ~2 GB
- **GPU required**: CUDA-capable GPU (SM 5.2+)
- **Output size**: ~500 MB (VTK files)

## Output Analysis

### Time Series Data

Generated file: `output_khairallah_melt_pool/melt_pool_metrics.csv`

Columns:
- `time_us`: Simulation time [μs]
- `laser_x_um`: Laser position along scan [μm]
- `width_um`: Melt pool width [μm]
- `depth_um`: Melt pool depth [μm]
- `length_um`: Melt pool length [μm]
- `peak_temp_K`: Maximum temperature [K]
- `surface_temp_K`: Maximum surface temperature [K]
- `max_velocity_mps`: Maximum fluid velocity [m/s]

### VTK Visualization

Generated files: `output_khairallah_melt_pool/khairallah_t*.vtk`

**ParaView visualization guide:**

1. **Temperature field**:
   - Colormap: Rainbow (300 K blue → 3000 K red)
   - Contour at T_liquidus = 1700 K (melt pool boundary)
   - Contour at T_boiling = 3090 K (keyhole boundary)

2. **Velocity vectors**:
   - Glyph filter → 3D arrows
   - Scale factor: 10 for visibility
   - Color by velocity magnitude
   - Filter: Show only where fill_level > 0.5 (liquid phase)

3. **Free surface**:
   - Contour filter on fill_level at 0.5
   - Shows liquid-gas interface deformation

4. **Phase state**:
   - Colormap: Blue (solid=0) → Green (mushy=1) → Red (liquid=2)

### Python Post-Processing

Example script to analyze results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('output_khairallah_melt_pool/melt_pool_metrics.csv')

# Plot melt pool evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Width vs time
axes[0, 0].plot(df['time_us'], df['width_um'])
axes[0, 0].axhline(125, color='r', linestyle='--', label='Literature (100-150 μm)')
axes[0, 0].set_xlabel('Time [μs]')
axes[0, 0].set_ylabel('Width [μm]')
axes[0, 0].legend()

# Depth vs time
axes[0, 1].plot(df['time_us'], df['depth_um'])
axes[0, 1].axhline(75, color='r', linestyle='--', label='Literature (50-100 μm)')
axes[0, 1].set_xlabel('Time [μs]')
axes[0, 1].set_ylabel('Depth [μm]')
axes[0, 1].legend()

# Temperature vs time
axes[1, 0].plot(df['time_us'], df['peak_temp_K'])
axes[1, 0].axhline(3090, color='orange', linestyle='--', label='Boiling point')
axes[1, 0].set_xlabel('Time [μs]')
axes[1, 0].set_ylabel('Peak Temperature [K]')
axes[1, 0].legend()

# Velocity vs time
axes[1, 1].plot(df['time_us'], df['max_velocity_mps'])
axes[1, 1].set_xlabel('Time [μs]')
axes[1, 1].set_ylabel('Max Velocity [m/s]')

plt.tight_layout()
plt.savefig('khairallah_validation.png', dpi=300)
```

## Expected Results

### Typical Behavior

1. **Initial transient (0-50 μs)**:
   - Rapid heating at laser position
   - Melt pool formation and expansion
   - Width and depth increase to steady state

2. **Steady-state scanning (50-400 μs)**:
   - Melt pool translates with laser
   - Dimensions stabilize (quasi-steady)
   - Marangoni recirculation established

3. **Solidification phase (400-600 μs)**:
   - Laser turns off
   - Rapid cooling and freezing
   - Melt pool shrinks and disappears

### Physics Observations

**Marangoni flow pattern:**
- Outward flow at surface (hot center → cold periphery)
- Downward flow at pool center
- Inward return flow at bottom
- Recirculation time: ~10-50 μs

**Temperature gradients:**
- Surface: 100-500 K/μm (strong Marangoni driving force)
- Depth: 50-200 K/μm

**Free surface deformation:**
- Depression at laser spot (recoil + Marangoni)
- Typical depth: 5-20 μm (without strong recoil)

## Troubleshooting

### Common Issues

**Issue: Melt pool too small**
- Check laser absorptivity (should be 0.35-0.42 for 316L)
- Verify penetration depth (10 μm is standard)
- Check substrate cooling BC (may be too strong)

**Issue: Temperature too high (> 4000 K)**
- Verify grid spacing (2 μm recommended)
- Check timestep stability (ω < 1.9)
- Enable radiation BC if disabled

**Issue: NaN or instability**
- Reduce timestep (try 50 ns instead of 100 ns)
- Check CFL velocity limits (should be < 0.2)
- Verify VOF subcycles (10 recommended)

**Issue: Melt pool depth zero**
- Check phase change is enabled
- Verify material properties loaded correctly
- Check T_liquidus threshold (1700 K for 316L)

## Comparison to Other Benchmarks

### Rosenthal Analytical Solution

The Rosenthal solution provides a simpler analytical benchmark for melt pool size:

```
Width (Rosenthal) = 4√(αP/(πρcpv)) ≈ 120 μm
```

For our parameters:
- α = 2.57×10⁻⁶ m²/s
- P = 195 W × 0.38 = 74 W (absorbed)
- v = 1.0 m/s
- Predicted width ≈ 120 μm

**Comparison:**
- Rosenthal: ~120 μm (steady-state, no Marangoni)
- Khairallah: 100-150 μm (includes Marangoni, surface effects)
- Our test: Should match Khairallah within ±50%

### King et al. 2015 Experiments

Experimental validation from:
King, W. E., et al. (2015). "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing." J. Mater. Process. Technol., 214(12), 2915-2925.

**Experimental observations:**
- Melt pool width: 80-120 μm (conduction, 316L)
- Keyhole depth: 200-400 μm (keyhole mode, >400 W)
- Denudation: 50-150 μm

## Extensions and Future Work

### Additional Test Cases

The test file includes disabled test cases for:

1. **Lower scan speed (0.8 m/s)**:
   - Expected: Wider, deeper melt pool
   - Higher energy density: 2.6 J/mm²

2. **Higher scan speed (1.5 m/s)**:
   - Expected: Shallower melt pool
   - Lower energy density: 1.4 J/mm²

Enable with:
```cpp
TEST(KhairallahBenchmark, MeltPoolSize_316L_195W_0p8mps) {
    // Remove DISABLED_ prefix
}
```

### Keyhole Mode Test

For high-power keyhole validation:

```cpp
config.laser_power = 400.0f;  // W
config.enable_recoil_pressure = true;
```

Expected:
- Keyhole depth: 150-300 μm
- Width: 100-150 μm (similar to conduction)
- Recoil depression: 50-150 μm

## References

1. **Khairallah, S. A., et al.** (2016). Laser powder-bed fusion additive manufacturing: Physics of complex melt flow. *Acta Materialia*, 108, 36-45.

2. **King, W. E., et al.** (2015). Observation of keyhole-mode laser melting. *J. Mater. Process. Technol.*, 214(12), 2915-2925.

3. **Körner, C., et al.** (2011). Mesoscopic simulation of selective beam melting processes. *Modelling Simul. Mater. Sci. Eng.*, 19, 064001.

4. **Matthews, M. J., et al.** (2016). Denudation of metal powder layers in laser powder bed fusion processes. *Acta Materialia*, 114, 33-42.

5. **Rosenthal, D.** (1946). The theory of moving sources of heat. *Trans. ASME*, 68, 849-866.

## Contact and Support

For questions or issues with this validation test:
- Check existing documentation in `docs/`
- Review test implementation: `tests/validation/test_khairallah_melt_pool.cu`
- Examine CMake configuration: `tests/validation/CMakeLists.txt`

## Revision History

- **2026-01-10**: Initial implementation
  - 316L stainless steel
  - 195 W, 1.0 m/s scan speed
  - Full multiphysics coupling
  - Conduction mode validation
