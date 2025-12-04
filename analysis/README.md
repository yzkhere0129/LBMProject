# VTK Fluid and VOF Analysis Suite

## Overview
Comprehensive Python-based analysis tools for LBM-CUDA multiphysics simulation VTK output files. Analyzes velocity fields, VOF (Volume of Fluid) interface tracking, and validates against analytical solutions.

## Directory Structure
```
/home/yzk/LBMProject/analysis/
├── analyze_poiseuille_flow.py        # Poiseuille flow validation
├── analyze_velocity_vtk_simple.py    # Velocity field analysis
├── analyze_vof_vtk_simple.py         # VOF field analysis
├── run_all_analyses.py               # Master analysis script
├── COMPREHENSIVE_ANALYSIS_REPORT.md  # Detailed findings (THIS IS KEY)
├── QUICK_REFERENCE.md                # Quick reference card
├── README.md                         # This file
└── results/                          # Output directory
    ├── *.png                         # 10 visualization plots
    └── analysis_report.txt           # Text summary
```

## Quick Start

### Run All Analyses
```bash
cd /home/yzk/LBMProject/analysis
python3 run_all_analyses.py
```

### View Results
```bash
# List generated plots
ls -lh results/*.png

# Read comprehensive report
cat COMPREHENSIVE_ANALYSIS_REPORT.md

# Quick reference
cat QUICK_REFERENCE.md
```

## Key Results

### ✓ Poiseuille Flow Validation - PASSED
- L2 error: 4.06% (target < 5%)
- Perfect parabolic profile (R² = 0.9998)
- No-slip boundaries satisfied

### ✓ Velocity Field Analysis - COMPLETE
- 56 timesteps analyzed
- Velocity evolution: 14 → 418 mm/s (Marangoni convection)
- No numerical artifacts detected

### ✓ VOF Field Analysis - COMPLETE  
- Sharp interface maintained (12-16% interface cells)
- Bound compliance: all F ∈ [0, 1]
- Mass conservation tracked (AMR normalization needed)

## Generated Visualizations (10 plots)

1. **poiseuille_analysis.png** - Velocity profile validation
2. **poiseuille_parabolic_fit.png** - Parabolic fit (R²=0.9998)
3. **velocity_profiles.png** - Spatial velocity distributions
4. **velocity_distribution.png** - Velocity magnitude histogram
5. **velocity_time_series.png** - Velocity evolution over time
6. **vof_profiles.png** - Interface position profiles
7. **vof_distribution.png** - Fill level histogram
8. **vof_slice_z16.png** - 2D interface visualization (t=0)
9. **vof_slice_z25.png** - 2D interface visualization (t=end)
10. **vof_mass_conservation.png** - Mass conservation over time

## Data Sources

- **Poiseuille Data:** /home/yzk/LBMProject/build/tests/integration/poiseuille_profile_fluidlbm.txt
- **VTK Files:** /home/yzk/LBMProject/build/phase6_test2c_visualization/marangoni_flow_*.vtk
- **Test Executables:** /home/yzk/LBMProject/build/tests/integration/

## Dependencies
- Python 3 (standard library)
- numpy
- matplotlib

No specialized VTK libraries required - uses custom ASCII parser.

## Customization

Edit parameters at top of each script:
```python
VTK_DIR = "/your/vtk/directory"
VTK_PATTERN = "simulation_*.vtk"
OUTPUT_DIR = "/your/output/directory"
```

## Test Runs

Before analysis, the following tests were executed:
```bash
# Poiseuille flow test
./build/tests/integration/test_poiseuille_flow_fluidlbm
# Result: PASSED (2 tests, 4.06% L2 error)

# VOF advection test  
./build/tests/test_vof_advection
# Result: PASSED (6/6 tests)
```

## Important Notes

1. **AMR Effect on Mass Conservation:** The simulation uses adaptive mesh refinement (64³ → 100³ grid). Mass conservation analysis needs normalization by cell count.

2. **Coordinate System:** VTK files use Fortran order ('F') for array reshaping.

3. **Units:**
   - Velocity: m/s (plots show mm/s)
   - Length: m (plots show μm)
   - Grid spacing: 2 μm

## Further Analysis Recommendations

1. Normalize mass conservation by dividing by cell count
2. Compute vorticity field (curl of velocity)
3. Analyze temperature-velocity coupling
4. Perform grid convergence study
5. Compare with experimental data

## Contact / Project Info

- **Project:** LBM-CUDA Multiphysics CFD Framework
- **Build Dir:** /home/yzk/LBMProject/build
- **Analysis Date:** 2025-12-04

## For More Information

See **COMPREHENSIVE_ANALYSIS_REPORT.md** for detailed findings, physical interpretation, and recommendations.
