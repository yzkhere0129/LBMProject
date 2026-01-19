# Rayleigh-Taylor Mass Loss Analysis

**Date**: 2026-01-19
**Simulation**: RT Mushroom (256x1024x4, dt=0.00390625 m)
**Time Range**: t=0.00 to t=1.00 s

---

## Executive Summary

The Rayleigh-Taylor simulation exhibits **20.73% total mass loss** over 1 second of simulation time. Mass loss is predominantly occurring in **bulk fluid regions** (initially filled cells with f > 0.9), with a moderate spatial correlation to high-velocity regions.

---

## Key Findings

### 1. Total Mass Loss

| Time (s) | Total Mass | Loss (%) | Loss Rate (%/s) |
|----------|-----------|----------|-----------------|
| 0.00 | 3.122e-02 | 0.00 | - |
| 0.20 | 2.848e-02 | -8.78 | -43.9 |
| 0.70 | 2.506e-02 | -19.73 | -21.9 |
| 1.00 | 2.475e-02 | -20.73 | -3.3 |

**Observations:**
- Initial rapid loss rate of 43.9%/s in first 0.2 seconds
- Loss rate decreases over time (stabilizing effect)
- Total loss exceeds acceptable threshold (typically <1% for VOF)

---

### 2. Regional Mass Distribution

**Mass change by initial region classification:**

| Region | Initial Mass | Final Mass | Change (%) |
|--------|-------------|------------|-----------|
| **Bulk Fluid** (f>0.9) | 3.108e-02 | 9.261e-03 | **-70.2%** |
| **Interface** (0.1<f<0.9) | 1.338e-04 | 1.527e-02 | +11,317% |
| **Boundary** | 1.577e-02 | 1.256e-02 | -20.4% |
| **High Velocity** (\|v\|>0.15) | 0.000e+00 | 2.389e-02 | - |

**Critical Finding:**
The bulk fluid loses **70.2% of its mass**, while the interface region gains mass (appears to "smear" as fluid cells lose volume fraction). This suggests:
- Numerical diffusion at the interface
- Under-compressive advection scheme
- Possible under-resolved gradients

---

### 3. Spatial Distribution

**Statistics:**
- Total cells losing mass: **467,464** (44.6% of domain)
- Total cells gaining mass: **199,608** (19.0% of domain)
- Maximum local loss: **Δf = -1.0** at cell (33, 798, 3)
  - This is a complete fluid-to-empty transition
- Net mass change: **-1.086e+05** (arbitrary units)

**Spatial Pattern:**
- Loss concentrated where initial fluid (heavy, top) descends
- Gain occurs in the rising bubble region (lighter fluid ascending)
- Boundary regions show moderate loss (-20.4%)

**Visualization Available:**
- `/home/yzk/LBMProject/build/output_rt_mushroom/mass_loss_analysis_spatial_distribution.png`
  - (a) Initial fill level
  - (b) Final fill level
  - (c) Change in fill (Δf)
  - (d) Absolute change magnitude

---

### 4. Correlation with Velocity

**Pearson Correlation Coefficient:** r = **-0.2086**
→ Weak negative correlation (higher velocity → more mass loss)

**Regional Velocity Analysis:**

| Velocity Region | Average Δf | Interpretation |
|----------------|-----------|----------------|
| High (\|v\| > 0.15 m/s) | **-0.1088** | Significant loss |
| Low (\|v\| < 0.05 m/s) | **-0.0230** | Moderate loss |

**Finding:**
High-velocity regions lose **4.7x more mass** than low-velocity regions. This suggests:
- Advection errors compound with flow speed
- CFL-related discretization errors
- Under-resolved interface in rapidly moving regions

**Visualization:**
- Scatter plot shows mass loss increases slightly with velocity
- Binned average confirms trend is real but noisy

---

### 5. Average Mass Change by Region Type

| Region Type | Avg Δf per cell | Total Mass Loss |
|-------------|----------------|-----------------|
| **Bulk Fluid** | -0.421 | **-1.309e-02** |
| **Interface** | -0.034 | -8.997e-06 |
| Bulk Empty | +0.213 | +6.629e-03 |
| **Boundary** | -0.102 | **-3.210e-03** |

**Dominant Loss Region:** Bulk Fluid (77.5% of total loss)

**Analysis:**
1. Bulk fluid cells lose on average 42.1% of their fill level
2. Empty cells gain mass (non-physical, indicates numerical diffusion)
3. Boundary cells lose 10.2% per cell (significant boundary effects)
4. Interface cells have smaller per-cell loss but cover larger area over time

---

## Root Cause Hypotheses

### Primary Suspects

1. **VOF Advection Scheme**
   - Current: PLIC (Piecewise Linear Interface Calculation)
   - Issue: Not mass-conservative or under-compressive
   - Evidence: Bulk fluid → interface transfer, empty cells gaining mass

2. **Boundary Conditions**
   - 20.4% loss in boundary regions suggests improper BC treatment
   - Possible flux imbalance at domain edges
   - Check: Are ghost cells properly filled/updated?

3. **Time Stepping / CFL Violation**
   - Rapid initial loss (43.9%/s) suggests instability
   - Check: CFL number = |v|Δt/Δx for max velocity
   - Possible under-resolved timestep

4. **High Velocity Regions**
   - 4.7x higher loss rate in fast-moving regions
   - Advection scheme fails under large Courant numbers
   - May need operator splitting or subcycling

### Secondary Factors

- Density ratio effects (ρ_heavy/ρ_light = 3.0)
- Interface reconstruction errors in complex topology
- Lack of interface compression/sharpening
- Numerical viscosity/diffusion

---

## Recommended Actions

### Immediate (Critical)

1. **Verify VOF Advection Algorithm**
   - Confirm mass conservation in 1D/2D test cases
   - Check for sign errors in flux calculations
   - Review interface reconstruction near boundaries

2. **Check Boundary Conditions**
   - Visualize fill level at boundaries (all 6 faces)
   - Verify no-flux conditions for VOF field
   - Ensure ghost cells mirror interior or enforce gradient=0

3. **Analyze CFL Number**
   - Extract max velocity at each timestep
   - Compute CFL = max(|u|Δt/Δx)
   - Target CFL < 0.5 for stability

### Short-term (Important)

4. **Test Simpler Case**
   - Run Zalesak disk (pure advection, no flow)
   - Should have <1% mass loss over full rotation
   - Isolates advection issues from flow coupling

5. **Implement Interface Compression**
   - Add artificial compression term (e.g., Weller's approach)
   - Keeps interface sharp, reduces diffusion
   - Standard practice in OpenFOAM, interFoam

6. **Reduce Timestep**
   - Try Δt/2 and check if mass loss decreases
   - If yes → CFL-limited; if no → algorithmic issue

### Long-term (Best Practice)

7. **Switch to Conservative VOF Scheme**
   - Geometric VOF (Youngs, Rider-Kothe)
   - Ensures exact mass conservation to machine precision
   - More complex but industry standard

8. **Add Diagnostic Output**
   - Track total mass at every timestep
   - Output CFL number, max velocity
   - Flag cells with |Δf| > threshold

---

## Visualization Files Generated

All files located in: `/home/yzk/LBMProject/build/output_rt_mushroom/`

1. **mass_loss_analysis_mass_vs_time.png**
   - Shows exponential-like decay of total mass
   - Clearly demonstrates 20.73% loss

2. **mass_loss_analysis_regional_mass.png**
   - Tracks mass evolution in interface, bulk fluid, boundary, high-velocity regions
   - Highlights bulk fluid collapse and interface growth

3. **mass_loss_analysis_spatial_distribution.png**
   - 4-panel visualization:
     - Initial/final fill levels
     - Change map (Δf)
     - Absolute change magnitude

4. **mass_loss_analysis_velocity_correlation.png**
   - Scatter plot and binned average
   - Shows weak but real velocity-loss correlation

5. **mass_loss_analysis_region_comparison.png**
   - Bar charts: average and total mass change per region
   - Clearly shows bulk fluid dominates loss

---

## Next Steps

1. **Human Review**: Inspect visualizations for unexpected patterns
2. **Code Audit**: Review VOF advection and boundary handling
3. **Parameter Sweep**: Test different Δt, resolution, CFL limits
4. **Benchmark**: Compare to reference (Gerris, OpenFOAM) at same conditions

---

## Script Location

**Analysis Script:** `/home/yzk/LBMProject/scripts/analyze_rt_mass_loss_vtk.py`

**Rerun Analysis:**
```bash
cd /home/yzk/LBMProject
python3 scripts/analyze_rt_mass_loss_vtk.py
```

**Modify Parameters:**
Edit lines 18-24 in script to change:
- Interface threshold range
- Velocity threshold for "high velocity"
- Output directory

---

## Technical Notes

- **Grid**: 256 (x) × 1024 (y) × 4 (z) cells
- **Spacing**: 0.00390625 m (uniform)
- **Domain**: 1.0 m × 4.0 m × 0.015625 m (quasi-2D)
- **Arrays**: fill_level, velocity (both present and valid)
- **Precision**: Analysis performed in float64 (double precision)

---

**End of Report**
