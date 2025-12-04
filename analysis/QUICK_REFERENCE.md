# VTK Analysis Quick Reference

## Files Generated

### Analysis Scripts (in /home/yzk/LBMProject/analysis/)
- `analyze_poiseuille_flow.py` - Poiseuille flow validation
- `analyze_velocity_vtk_simple.py` - Velocity field analysis from VTK
- `analyze_vof_vtk_simple.py` - VOF field analysis from VTK
- `run_all_analyses.py` - Master script (runs all)

### Results (in /home/yzk/LBMProject/analysis/results/)
**Plots (10 total):**
1. `poiseuille_analysis.png` - Velocity profile vs analytical
2. `poiseuille_parabolic_fit.png` - Parabolic fit validation
3. `velocity_profiles.png` - Velocity along X and Y centerlines
4. `velocity_distribution.png` - Velocity histogram
5. `velocity_time_series.png` - Max/mean velocity evolution
6. `vof_profiles.png` - Fill level along centerlines
7. `vof_distribution.png` - Fill level histogram
8. `vof_slice_z16.png` - 2D VOF slice (initial state)
9. `vof_slice_z25.png` - 2D VOF slice (final state)
10. `vof_mass_conservation.png` - Mass conservation over time

**Reports:**
- `analysis_report.txt` - Text summary
- `COMPREHENSIVE_ANALYSIS_REPORT.md` - Full detailed report

## Quick Run Commands

```bash
# Run all analyses
cd /home/yzk/LBMProject/analysis
python3 run_all_analyses.py

# Run individual analyses
python3 analyze_poiseuille_flow.py
python3 analyze_velocity_vtk_simple.py
python3 analyze_vof_vtk_simple.py

# View results
ls -lh results/*.png
cat results/analysis_report.txt
```

## Key Results Summary

### Poiseuille Flow Validation
- **L2 Error:** 4.06% ✓ (< 5% target)
- **Max Error:** 1.50e-03
- **Symmetry:** 3.34e-08 ✓
- **Boundary Conditions:** Perfect no-slip ✓
- **Parabolic Fit:** R² = 0.9998 ✓

### Velocity Field (Marangoni Flow)
- **Initial Max Velocity:** 14.1 mm/s
- **Final Max Velocity:** 418 mm/s (30× amplification)
- **Active Flow Cells:** 12-14%
- **Numerical Quality:** No NaN/Inf ✓
- **Grid:** 64×64×32 → 100×100×50 (AMR)

### VOF Field
- **Liquid Cells:** 2.9-3.1%
- **Gas Cells:** 81-85%
- **Interface Cells:** 12-16% (sharp interface)
- **Bound Compliance:** All F ∈ [0,1] ✓
- **Mass Conservation:** Need normalization for AMR

## Tests Run
1. **test_poiseuille_flow_fluidlbm** - Passed (4.06% error)
2. **test_vof_advection** - Passed (6/6 tests)

## Data Sources
- Poiseuille: `/home/yzk/LBMProject/build/tests/integration/poiseuille_profile_fluidlbm.txt`
- VTK Files: `/home/yzk/LBMProject/build/phase6_test2c_visualization/marangoni_flow_*.vtk`
- Total VTK Files: 56 timesteps (0 to 10000)

## Modifying Parameters

Edit top of each script:
```python
# In analyze_velocity_vtk_simple.py or analyze_vof_vtk_simple.py
VTK_DIR = "/path/to/vtk/files"
VTK_PATTERN = "simulation_*.vtk"
OUTPUT_DIR = "/path/to/output"
VELOCITY_THRESHOLD = 1e-6  # m/s
```

## Dependencies
- Python 3
- numpy
- matplotlib

**No pyvista required** - uses custom ASCII VTK parser

## File Locations (Absolute Paths)
- Analysis scripts: `/home/yzk/LBMProject/analysis/`
- Results: `/home/yzk/LBMProject/analysis/results/`
- VTK data: `/home/yzk/LBMProject/build/phase6_test2c_visualization/`
- Poiseuille data: `/home/yzk/LBMProject/build/tests/integration/`

## Viewing Plots

```bash
# On Linux with GUI
xdg-open /home/yzk/LBMProject/analysis/results/velocity_time_series.png

# Copy to local machine for viewing
# On your local machine:
scp user@server:/home/yzk/LBMProject/analysis/results/*.png ./
```

## Next Steps for User
1. Review all 10 plots in `results/` directory
2. Read `COMPREHENSIVE_ANALYSIS_REPORT.md` for detailed findings
3. Address mass conservation normalization for AMR
4. Run grid convergence study
5. Compare with experimental data
