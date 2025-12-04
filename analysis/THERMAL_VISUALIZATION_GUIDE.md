# Thermal Physics Visualization Quick Reference

## How to View Thermal VTK Files in ParaView

### Loading Data

1. Open ParaView
2. File → Open → Navigate to VTK directory
3. Select all files with same prefix (e.g., `output_*.vtk`)
4. Click "Apply" in Properties panel

### Essential Visualizations for Thermal Analysis

#### 1. Temperature Field (Scalar)

**Setup:**
- Coloring: Select "Temperature" from dropdown
- Color scale: Edit → Choose "Black-Body Radiation" or "Rainbow"
- Range: Set manually (300-3000 K for full range)

**What to look for:**
- Hot spot location (peak temperature)
- Melt pool boundaries (around 1923 K for Ti6Al4V)
- Temperature gradients (sharp vs. smooth)
- Symmetry (or lack thereof)

**ParaView commands:**
```
Coloring: Temperature
Color Map: Rainbow Desaturated
Range: [300, 3000]
Opacity: 1.0
```

#### 2. Melt Pool Boundary (Isosurface)

**Setup:**
- Filters → Common → Contour
- Contour By: Temperature
- Value: 1923 (solidus temperature)

**What to look for:**
- Melt pool shape (elliptical, hemispherical, keyhole)
- Width and depth dimensions
- Smoothness of interface

#### 3. Temperature Slices

**Setup:**
- Filters → Common → Slice
- Slice Type: Plane
- Origin: Center of domain
- Normal: (0, 0, 1) for horizontal, (0, 1, 0) for vertical

**What to look for:**
- Cross-sectional temperature distribution
- Depth of heat penetration
- Laser spot size and intensity profile

#### 4. Volume Rendering (Advanced)

**Setup:**
- Representation: Volume
- Scalar: Temperature
- Edit opacity transfer function
  - Low temps (300-500 K): Transparent
  - Medium temps (1000-1923 K): Semi-transparent
  - High temps (>1923 K): Opaque

**What to look for:**
- 3D melt pool shape
- Internal temperature structure
- Hot regions buried below surface

---

## Key Temperature Thresholds for Ti6Al4V

```
300 K     ████  Ambient temperature
500 K     ████  Substrate heating
1000 K    ████  Significant thermal expansion
1878 K    ████  Solidus (melting begins)
1923 K    ████  Melting point
1928 K    ████  Liquidus (fully liquid)
2500 K    ████  Typical peak in LPBF
3560 K    ████  Boiling point (WARNING: evaporation!)
5000 K+   ████  UNPHYSICAL - check evaporation model
```

---

## Analysis Checklist

### For Each VTK File Series:

- [ ] Load time series and animate
- [ ] Check temperature range (min/max)
- [ ] Identify melt pool location and size
- [ ] Measure peak temperature
- [ ] Look for artifacts (grid noise, NaN regions)
- [ ] Check velocity field (if available) for flow patterns
- [ ] Examine liquid fraction field for phase boundaries

### Physical Realism Checks:

- [ ] Peak temperature below boiling point (3560 K)?
- [ ] Melt pool dimensions reasonable (50-200 µm width)?
- [ ] Temperature gradients smooth (no discontinuities)?
- [ ] Substrate remains mostly solid?
- [ ] Energy appears conserved (no runaway heating)?

### Comparison to Simulation Output:

Run the analysis script first:
```bash
python3 /home/yzk/LBMProject/analysis/analyze_thermal_vtk.py
```

Then compare ParaView observations to printed statistics.

---

## Common Issues and Fixes

### Issue: Entire domain same color

**Cause:** Color scale range doesn't match data range

**Fix:**
- Click "Rescale to Data Range" button (⟲)
- Or manually set range in Color Map Editor

### Issue: No melt pool visible

**Cause 1:** Temperature never exceeds melting point
- Check simulation ran long enough
- Verify laser power settings

**Cause 2:** Melt pool exists but color scale hides it
- Use "Threshold" filter: 1878 < Temperature < 2500
- Show only molten region

### Issue: Blocky, pixelated appearance

**Cause:** Structured grid rendering
**Fix:**
- Under Properties → Styling → Interpolate Scalars Before Mapping: ON
- Increases smoothness of color transitions

### Issue: Animation too fast/slow

**Cause:** ParaView time step settings
**Fix:**
- View → Animation View
- Adjust duration or frame rate

---

## Quantitative Measurements in ParaView

### 1. Peak Temperature
- Use "Find Data" icon
- Sort by Temperature column
- Read maximum value

### 2. Melt Pool Volume
- Filters → Threshold
- Temperature: 1923 to 5000
- Filters → Integrate Variables
- Read "Cell Volume"

### 3. Centerline Temperature Profile
- Filters → Plot Over Line
- Point 1: (center_x, center_y, 0)
- Point 2: (center_x, center_y, max_z)
- Resolution: 100 points
- Plot X Axis: arc_length
- Plot Y Axis: Temperature

### 4. Temperature Gradient Magnitude
- Filters → Gradient Of Unstructured DataSet
- Scalars: Temperature
- Results: Gradient vectors
- Filters → Calculator
- Formula: `mag(Gradients)`

---

## Recommended Color Maps by Field

| Field | Color Map | Reason |
|-------|-----------|--------|
| Temperature | Black-Body Radiation | Intuitive (red=hot) |
| Temperature (alternative) | Rainbow Desaturated | Shows subtle variations |
| Liquid Fraction | Cool to Warm | Blue=solid, Red=liquid |
| Velocity Magnitude | Viridis | Perceptually uniform |
| Phase State | Discrete (3 colors) | Distinct phases |

---

## Example Analysis Workflow

### Scenario: Investigate Laser Melting Simulation

1. **Load data:**
   ```
   File: build/test_output/laser_melting_final.vtk
   ```

2. **Initial view:**
   - Coloring: Temperature
   - Representation: Surface
   - Rescale to data range

3. **Observe:**
   - Peak temperature: 6999 K (VERY HIGH!)
   - Entire domain red (above 5000 K)

4. **Quantify:**
   - Run Python analysis script
   - Confirm: 100% of domain > 5000 K

5. **Diagnose:**
   - Compare to Marangoni test (peak 2500 K)
   - Check evaporation cooling in code
   - Review laser power parameter

6. **Report:**
   - Screenshot of temperature field
   - Note in THERMAL_VTK_ANALYSIS_REPORT.md
   - Flag for physics team review

---

## Advanced: Creating Publication-Quality Figures

### For Papers/Presentations:

1. **Set white background:**
   - View → Color Palette → Print

2. **Add color bar:**
   - Properties → Color Map Editor → Show Color Legend

3. **Adjust resolution:**
   - View → Save Screenshot
   - Set resolution: 1920x1080 or higher
   - Transparent background: Optional

4. **Annotate:**
   - Sources → Text
   - Add labels for key features

5. **Multiple views:**
   - Split View (horizontal/vertical)
   - Show different fields side-by-side

### Example Figure Caption:

> "Temperature field from LBM simulation of LPBF Ti6Al4V at t = 0.5 ms.
> (a) Top view showing melt pool geometry (200 µm width).
> (b) Cross-section revealing depth penetration (100 µm).
> (c) 3D volume rendering of molten region (T > 1923 K).
> Simulation parameters: 200 W laser, 50 µm spot, 2 µm grid resolution."

---

## Python Analysis Script Integration

The automated analysis script generates:
- Time-evolution plots
- Statistical summaries
- Energy balance data
- Melt pool geometry

**Use this first**, then use ParaView for:
- Visual confirmation
- Detailed spatial investigation
- Identifying features the script might miss
- Creating presentation graphics

**Workflow:**
1. Run Python script → Get quantitative overview
2. Open ParaView → Visual inspection
3. Iterate between script and ParaView as needed

---

## Shortcuts and Tips

### ParaView Keyboard Shortcuts:
- `r`: Rescale to data range
- `s`: Surface representation
- `w`: Wireframe representation
- `p`: Point representation
- `Ctrl+Space`: Play/pause animation
- `1,2,3`: Standard camera views (XY, XZ, YZ)

### Performance Tips:
- For large datasets (>1M points): Use "Outline" first to check extents
- Enable "Use Cache" in Animation View for smooth playback
- Reduce "Resolution" in slice/clip filters if slow
- Use "Extract Subset" to analyze smaller regions first

---

## References

**ParaView Guide:** https://docs.paraview.org/
**VTK File Format:** https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

**This Project's Analysis Tools:**
- Script: `/home/yzk/LBMProject/analysis/analyze_thermal_vtk.py`
- Report: `/home/yzk/LBMProject/analysis/THERMAL_VTK_ANALYSIS_REPORT.md`
- Data: `/home/yzk/LBMProject/build/visualization_output/`

**For questions on thermal physics interpretation:**
- Review material properties in `src/physics/materials/material_database.cu`
- Check simulation parameters in config files under `config/` or `configs/`
- Consult validation test results in project documentation

---

**Last updated:** December 4, 2025
**Visualization tool:** ParaView 5.x recommended
**Data format:** VTK ASCII structured points
