# Validation Analysis Scripts

This directory contains standalone Python scripts for validating thermal and VOF simulation results from VTK output files.

## Scripts Overview

### 1. Thermal Validation: `thermal_walberla_comparison.py`

Compares thermal LBM results against walberla finite difference solver.

**What it analyzes:**
- Peak temperature evolution over time
- Spatial temperature profiles (centerline, depth)
- Error metrics (RMSE, max error, relative error)
- Temperature distribution comparison

**Usage:**
```bash
# Using default paths
python thermal_walberla_comparison.py

# Custom paths
python thermal_walberla_comparison.py \
    --lbm /path/to/lbm/vtk \
    --walberla /path/to/walberla/vtk \
    --dt 100e-9 \
    --output ./results
```

**Output:**
- `thermal_peak_evolution.png` - Peak temperature vs time with error plot
- `thermal_spatial_profiles.png` - Centerline and depth profiles
- `thermal_comparison_summary.txt` - Numerical error metrics

**Expected Results:**
- Peak temperature error < 5%
- RMSE < 100 K for typical laser melting (4000 K peak)
- Spatial profiles should overlay closely

---

### 2. VOF Advection Validation: `vof_advection_validation.py`

Tracks interface position and validates mass conservation in VOF advection tests.

**What it analyzes:**
- Interface position (center of mass) over time
- Interface displacement vs expected (u*t)
- Mass conservation error
- Interface sharpness degradation

**Usage:**
```bash
# Basic usage
python vof_advection_validation.py /path/to/vof/vtk

# With advection velocity for error computation
python vof_advection_validation.py /path/to/vtk \
    --velocity 0.001 \
    --dt 1e-6 \
    --dx 1e-6 \
    --output ./results
```

**Output:**
- `vof_interface_tracking.png` - Position vs time, trajectory, error plot
- `vof_conservation_metrics.png` - Volume conservation and sharpness
- `vof_validation_summary.txt` - Numerical results

**Expected Results:**
- Interface position error < 0.5 cells after full simulation
- Mass conservation error < 0.1%
- Interface should remain relatively sharp (< 5% smearing)

---

### 3. Grid Convergence Analysis: `grid_convergence_analysis.py`

Analyzes convergence order from multiple grid resolutions.

**What it analyzes:**
- Log-log convergence plot (error vs dx)
- Convergence order p from least-squares fit
- Grid Convergence Index (GCI)
- Error reduction ratios between grids
- Richardson extrapolation estimate

**Usage:**

**Option A: From VTK directories (auto-compute errors)**
```bash
python grid_convergence_analysis.py \
    /path/to/coarse/vtk \
    /path/to/medium/vtk \
    /path/to/fine/vtk \
    /path/to/finest/vtk \
    --output ./results
```

**Option B: From pre-computed errors**
```bash
python grid_convergence_analysis.py \
    --resolutions 25,50,100,200 \
    --errors 1.2e-2,3.1e-3,7.8e-4,1.9e-4 \
    --domain-length 200e-6 \
    --output ./results
```

**Output:**
- `grid_convergence.png` - Log-log plot with fitted convergence order
- `error_reduction.png` - Bar chart of error reduction ratios
- `convergence_summary.txt` - Numerical convergence metrics

**Expected Results:**
- Second-order schemes: 1.8 < p < 2.2
- First-order schemes: 0.8 < p < 1.2
- Errors should decrease monotonically
- GCI < 5% indicates grid-converged solution

---

## Dependencies

All scripts require:
```bash
pip install numpy pyvista matplotlib scipy
```

- `numpy` - Numerical operations
- `pyvista` - VTK file I/O and mesh operations
- `matplotlib` - Plotting
- `scipy` - Interpolation and spatial queries

---

## General Usage Notes

### VTK File Naming Convention

Scripts expect VTK files with timestep numbers in filenames:
- `simulation_step_001.vtu`
- `output_step_0050.vtk`
- `result_0100.vti`

Supported extensions: `.vtu`, `.vtk`, `.vti`, `.pvtu`

### Field Names

Scripts automatically detect common field names:
- **Temperature**: `T`, `temperature`, `Temperature`, `scalar`
- **VOF**: `fill_level`, `vof`, `F`, `alpha`, `fill`

### Modifying Parameters

Key parameters are at the top of each script in the `=== PARAMETERS ===` section:
- Default file paths
- Timestep size
- Grid spacing
- Field name lists
- Output directories

### Output Structure

All scripts create a `results/` directory with:
- PNG figures (150 dpi, publication quality)
- Text summaries with numerical metrics
- Clear labels, legends, and grid lines

---

## Examples

### Example 1: Thermal validation workflow

```bash
# Run thermal LBM test
cd /home/yzk/LBMProject/build
./tests/validation/test_thermal_walberla_match

# Run walberla reference
cd /home/yzk/walberla/build/apps/showcases/LaserHeating
./LaserHeating

# Compare results
cd /home/yzk/LBMProject/scripts/validation
python thermal_walberla_comparison.py \
    --lbm ../../tests/validation/output_thermal_walberla \
    --walberla /home/yzk/walberla/build/apps/showcases/LaserHeating/vtk_out/laser_heating

# Results in: results/thermal_peak_evolution.png
```

### Example 2: VOF advection validation

```bash
# Run VOF rotation test
cd /home/yzk/LBMProject/build
./tests/validation/test_vof_advection_rotation

# Analyze results
cd /home/yzk/LBMProject/scripts/validation
python vof_advection_validation.py \
    ../../tests/validation/vof/output_rotation \
    --velocity 0.0 \
    --dt 1e-6 \
    --dx 2e-6

# Check mass conservation in results/vof_conservation_metrics.png
```

### Example 3: Grid convergence study

```bash
# Run simulations at multiple resolutions
./test_grid_convergence  # Outputs to grid_25, grid_50, grid_100, grid_200

# Analyze convergence
python grid_convergence_analysis.py \
    output_grid_25 \
    output_grid_50 \
    output_grid_100 \
    output_grid_200

# Or use pre-computed errors from test output
python grid_convergence_analysis.py \
    --resolutions 25,50,100,200 \
    --errors 0.0143,0.0038,0.0009,0.0002

# Check convergence order in results/grid_convergence.png
```

---

## Interpreting Results

### Thermal Validation

**Good results:**
- Peak temperature curves overlay closely
- Relative error < 5% throughout simulation
- Spatial profiles show smooth agreement

**Potential issues:**
- Large oscillations → CFL number too high
- Systematic offset → source term or BC error
- Growing divergence → stability issue

### VOF Validation

**Good results:**
- Interface tracks expected position within 0.5 cells
- Volume error stays below 0.1%
- Interface sharpness remains < 5%

**Potential issues:**
- Interface diffusion → need compression/sharpening
- Volume loss → advection scheme not conservative
- Large position error → velocity field incorrect

### Grid Convergence

**Good results:**
- Convergence order near theoretical (p ≈ 2 for LBM)
- Monotonic error decrease
- GCI < 5% for production simulations

**Potential issues:**
- p < expected order → bug or under-resolved
- Non-monotonic → solution not converged in time
- p > expected → fortuitous cancellation or bug

---

## Customization

### Adding New Field Names

Edit the field name lists at the top of scripts:
```python
TEMP_FIELD_NAMES = ['T', 'temperature', 'my_custom_temp']
VOF_FIELD_NAMES = ['fill_level', 'vof', 'my_vof']
```

### Analytical Solutions

For `grid_convergence_analysis.py`, you can add analytical solution comparison:
```python
def analytical_temperature(x, y, z):
    # Your analytical solution here
    return T0 + A * np.exp(-r**2 / (2 * sigma**2))

# In compute_l2_error call:
error = compute_l2_error(mesh_fine, mesh_coarse, analytical_func=analytical_temperature)
```

### Custom Error Metrics

Add new metrics to the analysis scripts:
```python
def compute_custom_metric(mesh):
    field = get_temperature(mesh)
    # Your custom metric
    return np.percentile(field, 95)  # Example: 95th percentile
```

---

## Troubleshooting

### "No VTK files found"
- Check directory path exists
- Verify VTK extension matches (vtu/vtk/vti)
- Ensure filenames contain timestep numbers

### "No temperature field"
- Add your field name to TEMP_FIELD_NAMES
- Use `quick_vtk_check.py` to see available fields

### "Time series lengths differ"
- VTK outputs may have different frequencies
- Use `--dt` to override default timestep
- Interpolate to common time points if needed

### Large errors
- Check units (temperature in K, lengths in m)
- Verify both simulations use same parameters
- Check if solution is time-converged

---

## References

**Grid Convergence:**
- Roache, P. J. (1997). Quantification of uncertainty in CFD. Annual Review of Fluid Mechanics.

**VOF Validation:**
- Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport. J. Comp. Phys.

**Thermal LBM:**
- He, X., et al. (1997). A priori derivation of the lattice Boltzmann equation. Phys. Rev. E.

---

## Contact

For questions or issues with these scripts, see project documentation or examine the script source code comments.
