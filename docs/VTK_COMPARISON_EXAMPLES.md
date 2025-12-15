# VTK File Comparison Examples

**Author:** LBMProject Team
**Date:** 2025-12-04
**Purpose:** Comprehensive guide for comparing and visualizing VTK simulation outputs between LBMProject and WalBerla

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [VTK File Formats](#vtk-file-formats)
3. [Available Tools](#available-tools)
4. [Comparison Examples](#comparison-examples)
5. [ParaView Visualization](#paraview-visualization)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Comparison

Compare two VTK files and print metrics:

```bash
cd /home/yzk/LBMProject
python scripts/compare_vtk_files.py \
    build/config_parser_test/output_000000.vtk \
    /home/yzk/walberla/sim_output/sim_output_00000000.vtk
```

### With Visualization

Generate comparison plots automatically:

```bash
python scripts/compare_vtk_files.py \
    build/config_parser_test/output_000000.vtk \
    /home/yzk/walberla/sim_output/sim_output_00000000.vtk \
    --plot-velocity \
    --plot-temperature \
    --output-dir comparison_plots
```

### Slice Comparison

Compare specific 2D slices:

```bash
python scripts/compare_vtk_files.py \
    file1.vtk file2.vtk \
    --slice-field Velocity \
    --slice-axis 2 \
    --slice-index 25 \
    --output-dir slice_comparison
```

---

## VTK File Formats

### LBMProject VTK Format

**Location:** `/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/`

**Structure:**
```
# vtk DataFile Version 3.0
LBM Multiphysics Simulation with Flow
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 100 100 50
ORIGIN 0.0 0.0 0.0
SPACING 2e-06 2e-06 2e-06

POINT_DATA 500000
VECTORS Velocity float
<velocity data>

SCALARS Temperature float 1
LOOKUP_TABLE default
<temperature data>

SCALARS LiquidFraction float 1
LOOKUP_TABLE default
<liquid fraction data>

SCALARS PhaseState float 1
LOOKUP_TABLE default
<phase state data>

SCALARS FillLevel float 1
LOOKUP_TABLE default
<fill level data>
```

**Fields:**
- `Velocity` (vector): Fluid velocity field (m/s)
- `Temperature` (scalar): Temperature field (K)
- `LiquidFraction` (scalar): Liquid fraction (0-1)
- `PhaseState` (scalar): Phase indicator
- `FillLevel` (scalar): VOF fill level (0-1)

### WalBerla VTK Format

**Location:** `/home/yzk/walberla/sim_output/`

**Structure:**
```
# vtk DataFile Version 3.0
LBM-CUDA-Framework Simulation Data
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 32 32 32
ORIGIN 0 0 0
SPACING 1 1 1
POINT_DATA 32768

SCALARS rho double 1
LOOKUP_TABLE default
<density data>

SCALARS temperature double 1
LOOKUP_TABLE default
<temperature data>

VECTORS u double
<velocity data>

VECTORS force double
<force data>
```

**Fields:**
- `rho` (scalar): Density field
- `temperature` (scalar): Temperature field (K)
- `u` (vector): Velocity field (m/s)
- `force` (vector): Body force field

### Key Differences

| Aspect | LBMProject | WalBerla |
|--------|-----------|----------|
| Velocity field name | `Velocity` | `u` |
| Temperature name | `Temperature` | `temperature` |
| Data type | float | double |
| Additional fields | Phase change, VOF | Density, Force |
| Domain size | Variable (µm scale) | Variable (LBM units) |

---

## Available Tools

### 1. General Comparison Tool

**File:** `/home/yzk/LBMProject/scripts/compare_vtk_files.py`

**Features:**
- Load and parse VTK structured points
- Extract velocity, temperature, and scalar fields
- Compute L2, max, RMSE, and relative errors
- Generate comparison plots (line, histogram, scatter)
- Support for 2D slice visualization

**Usage:**
```bash
python scripts/compare_vtk_files.py [options] file1.vtk file2.vtk

Options:
  --fields FIELD1 FIELD2    Fields to compare (default: Velocity Temperature)
  --plot-velocity           Generate velocity comparison plots
  --plot-temperature        Generate temperature comparison plots
  --slice-field FIELD       Field name for slice comparison
  --slice-axis {0,1,2}      Axis for slice (0=X, 1=Y, 2=Z)
  --slice-index INDEX       Index along slice axis
  --output-dir DIR          Output directory for plots
```

**Output:**
- Console report with error metrics
- PNG plots in specified directory

### 2. Poiseuille Flow Comparison

**File:** `/home/yzk/LBMProject/scripts/compare_poiseuille.py`

**Features:**
- Extract velocity profiles along channel centerline
- Compare with analytical solution
- Compute normalized error metrics
- Generate profile comparison plots

**Usage:**
```bash
python scripts/compare_poiseuille.py \
    lbm_poiseuille.vtk \
    walberla_poiseuille.vtk \
    --output-dir poiseuille_results
```

**Output:**
- `poiseuille_comparison.png`: Velocity profile comparison
- `poiseuille_error_metrics.png`: Error metrics bar chart

### 3. Marangoni Flow Visualization

**File:** `/home/yzk/LBMProject/scripts/visualize_marangoni.py`

**Features:**
- Comprehensive multi-panel visualization
- Temperature and velocity fields
- Surface analysis (if fill level available)
- Time evolution comparison

**Usage:**

Single timestep:
```bash
python scripts/visualize_marangoni.py \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk \
    --output-dir marangoni_vis
```

Time evolution:
```bash
python scripts/visualize_marangoni.py \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_*.vtk \
    --time-evolution \
    --output-dir marangoni_evolution
```

**Output:**
- `marangoni_visualization.png`: Multi-panel field visualization
- `marangoni_surface_analysis.png`: Surface-specific analysis
- `marangoni_time_evolution.png`: Time series comparison

---

## Comparison Examples

### Example 1: Basic LBMProject vs WalBerla Comparison

**Objective:** Compare baseline thermal-fluid simulation

**Files:**
- LBMProject: `/home/yzk/LBMProject/build/config_parser_test/output_010000.vtk`
- WalBerla: `/home/yzk/walberla/sim_output/sim_output_00000450.vtk`

**Command:**
```bash
cd /home/yzk/LBMProject

python scripts/compare_vtk_files.py \
    build/config_parser_test/output_010000.vtk \
    /home/yzk/walberla/sim_output/sim_output_00000450.vtk \
    --fields Velocity u Temperature temperature \
    --plot-velocity \
    --plot-temperature \
    --output-dir results/comparison_basic
```

**Expected Output:**
```
================================================================================
VTK FILE COMPARISON REPORT
================================================================================

File 1: build/config_parser_test/output_010000.vtk
File 2: /home/yzk/walberla/sim_output/sim_output_00000450.vtk

METADATA COMPARISON:
  Dimensions: (32, 32, 16) vs (32, 32, 32)
  Origin:     (0.0, 0.0, 0.0) vs (0, 0, 0)
  Spacing:    (5e-06, 5e-06, 5e-06) vs (1, 1, 1)

FIELD COMPARISON:
  File 1 fields: ['Velocity', 'Temperature', 'LiquidFraction', 'PhaseState']
  File 2 fields: ['rho', 'temperature', 'u', 'force']

ERROR METRICS:
Field                L2 Error        Max Error       RMSE            Rel Error
--------------------------------------------------------------------------------
Velocity             1.234567e-04    5.678901e-04    2.345678e-04    3.456789e-03
Temperature          2.345678e-02    1.234567e-01    4.567890e-02    5.678901e-03
================================================================================
```

**Generated Plots:**
- `results/comparison_basic/velocity_comparison.png`
- `results/comparison_basic/temperature_comparison.png`

---

### Example 2: Poiseuille Flow Validation

**Objective:** Validate Poiseuille flow implementation against analytical solution

**Files:**
- LBMProject: Use dedicated Poiseuille test output (if available)
- WalBerla: `/home/yzk/walberla/apps/benchmarks/PoiseuilleChannel/output_*.vtk`

**Command:**
```bash
# If LBMProject Poiseuille test exists
python scripts/compare_poiseuille.py \
    build/tests/integration/poiseuille_final.vtk \
    /home/yzk/walberla/poiseuille_output.vtk \
    --output-dir results/poiseuille
```

**Alternative:** Generate VTK from text output:
```bash
# If only text output available, use analytical comparison
python scripts/compare_poiseuille.py \
    build/tests/validation/config_parser_test/output_010000.vtk \
    /home/yzk/walberla/sim_output/sim_output_00000000.vtk \
    --output-dir results/poiseuille_test
```

**Expected Metrics:**
- L2 Error: < 1e-3 (excellent agreement)
- Max Error: < 5e-3
- Relative Error: < 1% at centerline

---

### Example 3: Thermal Conduction Comparison

**Objective:** Validate pure thermal diffusion

**Files:**
- LBMProject: Any steady-state thermal output
- Analytical solution: Built into comparison script

**Command:**
```bash
python scripts/compare_vtk_files.py \
    build/tests/validation/pure_conduction_final.vtk \
    /home/yzk/walberla/thermal_diffusion.vtk \
    --fields Temperature temperature \
    --plot-temperature \
    --slice-field Temperature \
    --slice-axis 2 \
    --slice-index 8 \
    --output-dir results/thermal
```

**Visualization:**
```bash
# View in ParaView
paraview build/tests/validation/pure_conduction_final.vtk &
```

**Expected:**
- Linear temperature gradient for 1D conduction
- Symmetric pattern for 2D/3D with centered heat source

---

### Example 4: Marangoni Flow Analysis

**Objective:** Visualize surface tension-driven convection

**Files:**
- LBMProject: `/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/`

**Single Timestep Analysis:**
```bash
python scripts/visualize_marangoni.py \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk \
    --output-dir results/marangoni_t10000
```

**Time Evolution Analysis:**
```bash
# Analyze multiple timesteps
python scripts/visualize_marangoni.py \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_002000.vtk \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_004000.vtk \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_006000.vtk \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_008000.vtk \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk \
    --time-evolution \
    --output-dir results/marangoni_evolution
```

**Expected Features:**
- Temperature gradient from laser heating
- Outward surface flow (hot to cold)
- Return flow in bulk (cold to hot)
- Recirculation cells

---

### Example 5: Slice Comparison at Multiple Locations

**Objective:** Compare fields at different spatial locations

**Command:**
```bash
#!/bin/bash
# Create multiple slice comparisons

OUTPUT_DIR="results/slice_study"
mkdir -p $OUTPUT_DIR

FILE1="build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk"
FILE2="/home/yzk/walberla/sim_output/sim_output_00000450.vtk"

# Z-slices (top to bottom)
for i in 5 10 15 20 25; do
    python scripts/compare_vtk_files.py \
        $FILE1 $FILE2 \
        --slice-field Velocity \
        --slice-axis 2 \
        --slice-index $i \
        --output-dir ${OUTPUT_DIR}/z_slice_${i}
done

# Y-slices (front to back)
for i in 10 20 30 40 50; do
    python scripts/compare_vtk_files.py \
        $FILE1 $FILE2 \
        --slice-field Temperature \
        --slice-axis 1 \
        --slice-index $i \
        --output-dir ${OUTPUT_DIR}/y_slice_${i}
done

echo "Slice study complete. Results in $OUTPUT_DIR"
```

---

## ParaView Visualization

### Quick Launch Commands

**Single File:**
```bash
# LBMProject output
paraview /home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk &

# WalBerla output
paraview /home/yzk/walberla/sim_output/sim_output_00000450.vtk &
```

**Time Series:**
```bash
# Load all timesteps
paraview --data=/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/marangoni_flow_*.vtk &
```

**State File:**
```bash
# Load pre-configured visualization
paraview --state=marangoni_visualization.pvsm &
```

### ParaView Workflow for Comparison

#### Step 1: Load Both Files

1. Open ParaView
2. File → Open → Select LBMProject VTK file
3. File → Open → Select WalBerla VTK file
4. Click "Apply" for both

#### Step 2: Side-by-Side Comparison

1. View → Split Horizontal
2. Select LBMProject in left view
3. Select WalBerla in right view
4. Apply same colormap and range to both

#### Step 3: Velocity Field Visualization

**LBMProject (left):**
```
1. Select "Velocity" in array selector
2. Filters → Glyph
   - Glyph Type: Arrow
   - Scale Mode: vector
   - Scale Factor: adjust for visibility
3. Filters → Stream Tracer
   - Seed Type: Point Cloud
   - Maximum Streamline Length: adjust
4. Color by: Velocity magnitude
```

**WalBerla (right):**
```
1. Select "u" in array selector
2. Apply same filters as left
3. Synchronize colormap range with left view
```

#### Step 4: Temperature Field Visualization

**Both views:**
```
1. Select Temperature/temperature field
2. Representation: Surface
3. Coloring: Temperature
4. Color Map: "Blue to Red Rainbow" or "Hot"
5. Rescale to Data Range
6. Add contour lines:
   Filters → Contour
   - Isosurfaces: 5-10 values
```

#### Step 5: Slice Comparison

**Create slice filter:**
```
1. Filters → Slice
2. Slice Type: Plane
3. Origin: Domain center
4. Normal: Z-axis [0, 0, 1]
5. Apply
6. Color by desired field
```

**Animation:**
```
1. View → Animation View
2. Add track: Slice Origin Z
3. Key frames: Bottom to top
4. Play animation
```

#### Step 6: Export Comparison Images

```
File → Save Screenshot
- Resolution: 1920x1080 or higher
- Background: White (for papers)
- Transparent: Optional
```

### ParaView Python Scripting

**Automated Comparison Script:**

```python
# save as paraview_compare.py
from paraview.simple import *

# Load files
lbm_file = "/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk"
walberla_file = "/home/yzk/walberla/sim_output/sim_output_00000450.vtk"

lbm = LegacyVTKReader(FileNames=[lbm_file])
walberla = LegacyVTKReader(FileNames=[walberla_file])

# Create layout
layout = GetLayout()
layout.SplitHorizontal(0, 0.5)

# Left view: LBMProject
SetActiveView(GetViews()[0])
lbm_display = Show(lbm)
lbm_display.Representation = 'Surface'
ColorBy(lbm_display, ('POINTS', 'Velocity', 'Magnitude'))

# Right view: WalBerla
SetActiveView(GetViews()[1])
walberla_display = Show(walberla)
walberla_display.Representation = 'Surface'
ColorBy(walberla_display, ('POINTS', 'u', 'Magnitude'))

# Synchronize camera
cam1 = GetActiveCamera()
cam2 = GetViews()[0].GetActiveCamera()
cam2.DeepCopy(cam1)

# Save screenshot
SaveScreenshot('comparison.png', layout, ImageResolution=[1920, 1080])
```

**Run script:**
```bash
pvpython paraview_compare.py
```

### ParaView Filter Recipes

#### 1. Velocity Magnitude

```
Calculator:
  - Name: VelMag
  - Formula: mag(Velocity)  # or mag(u) for WalBerla
  - Result Array: VelocityMagnitude
```

#### 2. Temperature Gradient

```
GradientOfUnstructuredDataSet:
  - Scalar Array: Temperature
  - Gradient Name: TempGradient
  - Compute Divergence: Off
  - Compute Vorticity: Off
```

#### 3. Streamlines

```
StreamTracer:
  - Vectors: Velocity (or u)
  - Seed Type: High Resolution Line Source
  - Resolution: 100
  - Maximum Streamline Length: 1000
  - Integration Direction: BOTH
```

#### 4. Iso-surfaces

```
Contour:
  - Contour By: Temperature
  - Isosurfaces: [300, 500, 700, 900, 1100]  # K
  - Compute Normals: On
```

#### 5. Volume Rendering

```
Representation: Volume
- Scalar: Temperature
- Color Transfer Function: Hot
- Opacity Transfer Function: Ramp
- Ambient: 0.2
- Diffuse: 0.8
- Specular: 0.2
```

---

## Troubleshooting

### Issue 1: "Could not extract velocity data"

**Cause:** Field name mismatch between files

**Solution:**
```bash
# Check available fields
grep "VECTORS\|SCALARS" file.vtk

# Use correct field names
python scripts/compare_vtk_files.py file1.vtk file2.vtk \
    --fields Velocity u  # LBMProject uses "Velocity", WalBerla uses "u"
```

### Issue 2: "Shape mismatch" error

**Cause:** Different grid dimensions

**Solution:**
```bash
# Check dimensions
grep "DIMENSIONS" file1.vtk file2.vtk

# If dimensions differ, compare derived quantities instead
python scripts/compare_vtk_files.py file1.vtk file2.vtk \
    --plot-velocity  # Plots handle different sizes
```

### Issue 3: ParaView crashes on large files

**Cause:** Insufficient memory

**Solution:**
```bash
# Decimate data before loading
python -c "
from compare_vtk_files import VTKData
vtk = VTKData('large_file.vtk')
# Extract subset or reduce resolution
"

# Or use ParaView in client-server mode
pvserver &
paraview --server=localhost
```

### Issue 4: No output plots generated

**Cause:** Missing matplotlib or numpy

**Solution:**
```bash
# Install dependencies
pip install numpy matplotlib

# Or use conda
conda install numpy matplotlib
```

### Issue 5: "File not found" errors

**Cause:** Incorrect paths

**Solution:**
```bash
# Use absolute paths
python scripts/compare_vtk_files.py \
    /home/yzk/LBMProject/build/output.vtk \
    /home/yzk/walberla/sim_output/output.vtk

# Or check if files exist
ls -la /home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/*.vtk
ls -la /home/yzk/walberla/sim_output/*.vtk
```

### Issue 6: Temperature values seem wrong

**Cause:** Different unit systems or reference temperatures

**Solution:**
```python
# Check temperature range
python -c "
from compare_vtk_files import VTKData
vtk = VTKData('file.vtk')
temp = vtk.get_field('Temperature')
print(f'Min: {temp.min()}, Max: {temp.max()}, Mean: {temp.mean()}')
"

# Convert units if needed
# LBMProject: Kelvin
# WalBerla: May be Kelvin or dimensionless
```

---

## Advanced Workflows

### Batch Comparison Script

```bash
#!/bin/bash
# compare_all.sh - Compare all available VTK outputs

LBM_DIR="/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization"
WALBERLA_DIR="/home/yzk/walberla/sim_output"
OUTPUT_DIR="results/batch_comparison"

mkdir -p $OUTPUT_DIR

# Get file lists
LBM_FILES=($LBM_DIR/marangoni_flow_*.vtk)
WALBERLA_FILES=($WALBERLA_DIR/sim_output_*.vtk)

# Compare corresponding timesteps
for i in "${!LBM_FILES[@]}"; do
    if [ $i -lt ${#WALBERLA_FILES[@]} ]; then
        echo "Comparing timestep $i..."
        python scripts/compare_vtk_files.py \
            "${LBM_FILES[$i]}" \
            "${WALBERLA_FILES[$i]}" \
            --plot-velocity \
            --plot-temperature \
            --output-dir "$OUTPUT_DIR/timestep_$i"
    fi
done

echo "Batch comparison complete. Results in $OUTPUT_DIR"
```

### Create Comparison Report

```python
#!/usr/bin/env python3
# generate_report.py - Automated comparison report

import sys
from pathlib import Path
sys.path.insert(0, '/home/yzk/LBMProject/scripts')
from compare_vtk_files import VTKComparator

# File pairs to compare
comparisons = [
    ("LBMProject t=0", "build/output_000000.vtk", "/home/yzk/walberla/sim_output/sim_output_00000000.vtk"),
    ("LBMProject t=100", "build/output_000100.vtk", "/home/yzk/walberla/sim_output/sim_output_00000100.vtk"),
    ("LBMProject t=500", "build/output_000500.vtk", "/home/yzk/walberla/sim_output/sim_output_00000500.vtk"),
]

# Generate report
with open("comparison_report.md", "w") as f:
    f.write("# VTK Comparison Report\n\n")

    for name, file1, file2 in comparisons:
        f.write(f"## {name}\n\n")

        comp = VTKComparator(file1, file2)

        # Compute metrics
        vel_l2 = comp.compute_l2_error('Velocity')
        temp_l2 = comp.compute_l2_error('Temperature')

        f.write(f"- Velocity L2 Error: {vel_l2:.6e}\n")
        f.write(f"- Temperature L2 Error: {temp_l2:.6e}\n\n")

print("Report generated: comparison_report.md")
```

---

## File Path Reference

### LBMProject VTK Locations

```
/home/yzk/LBMProject/build/
├── config_parser_test/
│   ├── output_000000.vtk
│   └── output_010000.vtk
├── test_output/
│   └── laser_melting_final.vtk
└── tests/validation/
    ├── config_parser_test/
    │   ├── output_000000.vtk
    │   └── output_010000.vtk
    ├── visualization_output/
    │   └── output_000000.vtk
    └── phase6_test2c_visualization/
        ├── marangoni_flow_002000.vtk
        ├── marangoni_flow_004000.vtk
        ├── marangoni_flow_006000.vtk
        ├── marangoni_flow_008000.vtk
        └── marangoni_flow_010000.vtk
```

### WalBerla VTK Locations

```
/home/yzk/walberla/
├── cavity_output_00000000.vtk
└── sim_output/
    ├── sim_output_00000000.vtk
    ├── sim_output_00000050.vtk
    ├── sim_output_00000100.vtk
    ├── sim_output_00000150.vtk
    ├── sim_output_00000200.vtk
    ├── sim_output_00000250.vtk
    ├── sim_output_00000300.vtk
    ├── sim_output_00000350.vtk
    ├── sim_output_00000400.vtk
    └── sim_output_00000450.vtk
```

### Script Locations

```
/home/yzk/LBMProject/scripts/
├── compare_vtk_files.py       # General comparison tool
├── compare_poiseuille.py      # Poiseuille-specific comparison
└── visualize_marangoni.py     # Marangoni visualization
```

---

## References

1. **VTK File Format Specification:**
   https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

2. **ParaView Guide:**
   https://docs.paraview.org/en/latest/

3. **LBMProject Documentation:**
   `/home/yzk/LBMProject/docs/`

4. **WalBerla Documentation:**
   https://www.walberla.net/

---

## Contact and Support

For issues or questions:
- Check existing documentation in `/home/yzk/LBMProject/docs/`
- Review test cases in `/home/yzk/LBMProject/tests/`
- Examine example configs in `/home/yzk/LBMProject/configs/`

---

**Last Updated:** 2025-12-04
**Version:** 1.0
