# VTK Analysis for Marangoni and Multiphysics Simulations

Comprehensive Python analysis tools for VTK output from CUDA-LBM multiphysics simulations.

---

## Quick Start

### 1. Analyze Existing VTK Files (Fastest)

```bash
# Activate Python environment
source /home/yzk/LBMProject/venv/bin/activate

# Quick check of latest timestep
python quick_marangoni_check.py

# Full analysis with plots
python analyze_marangoni_comprehensive.py
python analyze_multiphysics_coupling.py

# View results
ls -lh marangoni_results/
ls -lh multiphysics_results/
```

**Results:** See `/home/yzk/LBMProject/MARANGONI_MULTIPHYSICS_VISUALIZATION_REPORT.md`

---

### 2. Generate Fresh VTK Files

```bash
# Run Marangoni visualization test
cd /home/yzk/LBMProject/build
./visualize_phase6_marangoni

# Then analyze
source /home/yzk/LBMProject/venv/bin/activate
cd /home/yzk/LBMProject/analysis
python analyze_marangoni_comprehensive.py
```

**Guide:** See `RUN_MARANGONI_TESTS_GUIDE.md`

---

## Available Analysis Scripts

### 1. `quick_marangoni_check.py`

**Purpose:** Rapid validation of single VTK file

**Features:**
- Extracts key metrics (velocity, temperature, phase)
- Validates Marangoni velocity range
- Assesses surface flow
- No plotting (text output only)

**Usage:**
```bash
# Analyze most recent file
python quick_marangoni_check.py

# Analyze specific file
python quick_marangoni_check.py /path/to/file.vtk
```

**Output:** Terminal text summary

**Runtime:** <5 seconds

---

### 2. `analyze_marangoni_comprehensive.py`

**Purpose:** Full time-series analysis of Marangoni flow

**Features:**
- Analyzes all timesteps (56 VTK files)
- Extracts velocity evolution
- Computes surface temperature gradients
- Validates against expected Marangoni velocities
- Generates time evolution plots
- Creates final state visualization

**Usage:**
```bash
python analyze_marangoni_comprehensive.py
```

**Output:**
- `marangoni_results/marangoni_time_evolution.png` (6-panel time series)
- `marangoni_results/marangoni_final_state.png` (4-panel final state)
- Terminal summary report

**Runtime:** ~2-3 minutes

**Customization:**
Edit these parameters at top of file:
```python
VTK_DIR = "/path/to/vtk/files"
SOLIDUS_TEMP = 1650.0  # K
LIQUID_FRACTION_THRESHOLD = 0.5
SURFACE_FILL_LEVEL = (0.4, 0.6)
```

---

### 3. `analyze_multiphysics_coupling.py`

**Purpose:** Thermal-fluid coupling analysis

**Features:**
- Analyzes 5 selected timesteps
- Computes vorticity (convection cells)
- Calculates Reynolds and Peclet numbers
- Generates streamline plots
- Extracts vertical temperature/velocity profiles
- Creates multi-field coupling visualizations

**Usage:**
```bash
python analyze_multiphysics_coupling.py
```

**Output:**
- `multiphysics_results/coupling_analysis_t*.png` (6-panel, 5 timesteps)
- `multiphysics_results/streamlines_t*.png` (flow field, 5 timesteps)
- `multiphysics_results/vertical_profiles_t*.png` (1D profiles, 5 timesteps)
- Terminal summary with dimensionless numbers

**Runtime:** ~1-2 minutes

**Customization:**
Edit these parameters:
```python
VTK_DIR = "/path/to/vtk/files"
TIMESTEPS_TO_ANALYZE = [1000, 3000, 5000, 7000, 10000]
```

---

## Key Results from Current Analysis

### Marangoni Flow (phase6_test2c_visualization)

**Dataset:**
- 56 VTK files (timesteps 0-10000)
- Grid: 100 x 100 x 50 (500k points)
- Domain: 0.2 x 0.2 x 0.1 mm
- File size: ~14 MB each

**Key Findings:**
- **Peak Marangoni velocity:** 1.03 m/s (timestep 600) ✓ EXPECTED RANGE
- **Surface temperature gradient:** 2.46 × 10^7 K/m
- **Liquid volume fraction:** 4.85%
- **Reynolds number:** 2-40 (transitional flow)
- **Peclet number:** 0.1-2.0 (diffusion-dominated)

**Status:** Marangoni effect successfully demonstrated and validated

---

## Understanding the Outputs

### Time Evolution Plot (`marangoni_time_evolution.png`)

6-panel figure showing:

1. **Velocity magnitude** - Global max, liquid max, liquid mean vs time
   - Expected: Peak ~1 m/s early, decay to ~0.1 m/s
   - Orange/red lines show expected range (0.5-2.0 m/s)

2. **Surface velocity** - Marangoni-driven flow at free surface
   - Expected: Follows global velocity trend
   - Peak when temperature gradient maximum

3. **Temperature evolution** - Max and liquid mean temperature
   - Shows laser heating and subsequent cooling
   - Blue/cyan lines mark solidus/liquidus temperatures

4. **Surface temperature** - Max and min at free surface
   - Shaded region shows temperature variation
   - Larger variation → stronger Marangoni driving force

5. **Surface temperature gradient** - Drives Marangoni stress
   - Max and mean gradient vs time
   - Peak gradient → peak Marangoni velocity

6. **Liquid volume fraction** - Size of melt pool
   - Shows melt pool growth and shrinkage
   - Typically 2-10% for laser melting

### Final State Plot (`marangoni_final_state.png`)

4-panel figure at final timestep:

1. **Temperature + Velocity field** - Center slice (XZ plane)
   - Hot colors: Temperature
   - Cyan arrows: Velocity vectors
   - Shows flow driven from hot center outward

2. **Velocity distribution** - Histogram
   - Blue: All points
   - Red: Liquid only
   - Orange/darkred lines: Expected Marangoni range

3. **Liquid fraction field** - Phase distribution
   - Red: Liquid (fraction = 1)
   - Blue: Solid (fraction = 0)
   - Yellow: Mushy zone

4. **Temperature-velocity correlation** - Surface only
   - Shows coupling between thermal and fluid fields
   - Points should show correlation: hotter → faster

### Coupling Analysis (`coupling_analysis_t*.png`)

6-panel figure at specific timestep:

1. **Temperature field** - Thermal distribution
2. **Velocity magnitude** - Flow intensity
3. **Vorticity** - Convection cells (log scale)
4. **Reynolds number** - Inertia/viscous ratio
5. **Peclet number** - Advection/diffusion ratio
6. **Liquid fraction** - Phase field

### Streamline Plots (`streamlines_t*.png`)

Flow field visualization:
- **Background color:** Temperature (hot colormap)
- **Arrows:** Velocity vectors (colored by magnitude)
- Shows flow patterns and thermal coupling

### Vertical Profiles (`vertical_profiles_t*.png`)

1D profiles along vertical centerline:
- **Left:** Temperature vs height
- **Right:** Velocity vs height

---

## Physical Interpretation Guide

### Marangoni Effect

**What to look for:**
1. **Outward surface flow** from hot center
2. **Peak velocity** when temperature gradient maximum
3. **Correlation** between surface temp gradient and velocity
4. **Recirculation** bringing cooler fluid back

**Expected values:**
- Velocity: 0.5-2.0 m/s for laser melting
- Surface temp gradient: 10^6-10^8 K/m
- Temperature delta at surface: 100-1000 K

### Reynolds Number

**Re = ρ U L / μ**

**Interpretation:**
- Re < 1: Viscous-dominated (Stokes flow)
- Re ~ 1-100: Transitional (viscous and inertia both matter)
- Re > 100: Inertia-dominated (turbulent if Re >> 1000)

**Expected for metal melt pools:** Re ~ 1-100

### Peclet Number

**Pe = U L / α** (thermal)

**Interpretation:**
- Pe < 1: Diffusion-dominated (heat spreads by conduction)
- Pe ~ 1: Comparable advection and diffusion
- Pe > 1: Advection-dominated (heat carried by flow)

**Expected for liquid metals:** Pe ~ 0.1-10 (high thermal diffusivity)

### Vorticity

**ω = ∇ × u** (curl of velocity)

**Interpretation:**
- High vorticity → strong rotation/circulation
- Concentrated at free surface → Marangoni-driven
- Organized patterns → convection cells

**Expected:** ω ~ 10^4-10^6 1/s for laser melting

---

## Customizing Analysis

### Change VTK Input Directory

Edit in script:
```python
VTK_DIR = "/your/path/to/vtk/files"
VTK_PATTERN = "your_pattern_*.vtk"
```

### Modify Physical Parameters

```python
SOLIDUS_TEMP = 1650.0  # K (material-specific)
LIQUIDUS_TEMP = 1700.0  # K
LIQUID_FRACTION_THRESHOLD = 0.5  # Define liquid region
```

### Select Different Timesteps

In `analyze_multiphysics_coupling.py`:
```python
TIMESTEPS_TO_ANALYZE = [500, 1000, 2000, 5000, 10000]
```

### Change Output Directory

```python
OUTPUT_DIR = "/your/output/path"
```

---

## Troubleshooting

### Issue: "No module named 'pyvista'"

**Solution:**
```bash
source /home/yzk/LBMProject/venv/bin/activate
pip install pyvista matplotlib scipy numpy
```

### Issue: "No VTK files found"

**Solution:**
1. Check VTK_DIR path is correct
2. Check VTK_PATTERN matches your files
3. Run a test to generate VTK files (see `RUN_MARANGONI_TESTS_GUIDE.md`)

### Issue: "Data array not present"

**Solution:**
- The scripts now automatically skip corrupted files
- Check that simulation ran to completion
- Verify VTK file size (should be ~10-20 MB)

### Issue: Analysis is slow

**Solutions:**
- Use `quick_marangoni_check.py` for single file
- Reduce number of VTK files to analyze
- Reduce `TIMESTEPS_TO_ANALYZE` in coupling script

---

## File Organization

```
/home/yzk/LBMProject/analysis/
├── README_ANALYSIS.md                    # This file
├── RUN_MARANGONI_TESTS_GUIDE.md          # How to generate VTK
├── quick_marangoni_check.py              # Quick validation
├── analyze_marangoni_comprehensive.py    # Full time-series
├── analyze_multiphysics_coupling.py      # Coupling analysis
├── marangoni_results/                    # Marangoni output
│   ├── marangoni_time_evolution.png
│   └── marangoni_final_state.png
└── multiphysics_results/                 # Coupling output
    ├── coupling_analysis_t*.png
    ├── streamlines_t*.png
    └── vertical_profiles_t*.png
```

---

## Dependencies

**Python packages** (in `/home/yzk/LBMProject/venv/`):
- `pyvista` - VTK file I/O and processing
- `numpy` - Numerical arrays and operations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing (gradients, filters)

**Installation:**
```bash
python3 -m venv /home/yzk/LBMProject/venv
source /home/yzk/LBMProject/venv/bin/activate
pip install pyvista matplotlib scipy numpy
```

---

## References

**Main Report:**
`/home/yzk/LBMProject/MARANGONI_MULTIPHYSICS_VISUALIZATION_REPORT.md`

**Test Execution Guide:**
`/home/yzk/LBMProject/analysis/RUN_MARANGONI_TESTS_GUIDE.md`

**VTK Data Location:**
`/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/`

---

## Contact

**Project:** LBMProject - CUDA-based Lattice Boltzmann CFD
**Location:** `/home/yzk/LBMProject`
**Analysis Scripts Created:** 2025-12-04

---

**Quick Commands:**

```bash
# Activate environment
source /home/yzk/LBMProject/venv/bin/activate

# Quick check
python quick_marangoni_check.py

# Full analysis
python analyze_marangoni_comprehensive.py
python analyze_multiphysics_coupling.py

# Generate fresh VTK
cd /home/yzk/LBMProject/build && ./visualize_phase6_marangoni
```
