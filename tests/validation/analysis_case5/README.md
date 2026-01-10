# Case 5 Laser Melting - Analysis Results

**Analysis Date**: 2025-12-22
**VTK Files Analyzed**: 100
**Validation Status**: **FAIL**

---

## Quick Start

**View Summary**:
```bash
cat /home/yzk/LBMProject/tests/validation/analysis_case5/SUMMARY.txt
```

**View Detailed Report**:
```bash
cat /home/yzk/LBMProject/tests/validation/CASE5_ANALYSIS_REPORT.md
```

**Re-run Analysis**:
```bash
cd /home/yzk/LBMProject/tests/validation
./quick_vtk_analysis.sh
```

---

## Critical Findings

### Validation Failure
- **Peak Temperature Error**: 19.62% (threshold: 15%)
- **Melt Pool Depth Error**: 37.20% (threshold: 15%)

### Physical Instability
- **81 of 100 timesteps** show temperatures above boiling point (3560 K)
- **Maximum temperature**: 11,847 K (unphysical)
- **Instability onset**: Timestep 80 (early in simulation)

### Phase Change Issues
- **Maximum liquid fraction**: 1.61% (very low for laser melting)
- Suggests latent heat absorption problems
- VOF-thermal coupling may be incorrect

---

## Analysis Files

### Reports
| File | Description |
|------|-------------|
| `SUMMARY.txt` | Quick reference summary (text format) |
| `CASE5_ANALYSIS_REPORT.md` | Comprehensive analysis report |
| `README.md` | This file |

### Data
| File | Description |
|------|-------------|
| `timeseries_metrics.json` | Complete metrics for all 100 timesteps |

### Visualizations
| File | Description |
|------|-------------|
| `detailed_analysis.png` | 4-panel overview: temperature, depth, liquid fraction, location |
| `thermal_diagnostics.png` | 4-panel diagnostics: regime analysis, rate of change, correlations |
| `peak_temperature_evolution.png` | Peak temperature vs time |
| `liquid_fraction_evolution.png` | Liquid fraction vs time |
| `rosenthal_comparison.png` | Comparison with analytical solution (NOTE: parameters may not match) |

---

## Key Metrics

### At Peak Temperature Event (Timestep 1700)

| Metric | Value |
|--------|-------|
| Peak Temperature | 3506.19 K |
| Peak Location | (75.0, 150.0, 146.25) μm |
| Liquid Fraction | 1.175% |
| Melt Pool Width | 71.25 μm |
| Melt Pool Depth | 56.25 μm |
| Melt Pool Height | 67.50 μm |

### Comparison with Expected Values

| Metric | Expected | Actual | Error |
|--------|----------|--------|-------|
| Peak Temperature | 2931 K | 3506 K | +19.62% |
| Melt Pool Depth | 41 μm | 56.25 μm | +37.20% |

---

## Diagnostic Summary

### Temperature Regimes (100 timesteps)

| Regime | Count | Percentage | Status |
|--------|-------|------------|--------|
| Below melting (<1923 K) | 2 | 2% | Normal |
| Melting zone (1923-3560 K) | 17 | 17% | Valid |
| Above boiling (>3560 K) | 81 | 81% | CRITICAL ISSUE |
| Extreme (>5000 K) | 63 | 63% | CRITICAL ISSUE |

### Instability Characteristics

- **First unstable timestep**: 80
- **Temperature at onset**: 3690 K
- **Largest temperature jump**: +1228 K (timestep 0→20)
- **Pattern**: Monotonic temperature increase followed by oscillations

---

## Root Cause Analysis

### 1. Thermal Solver Stability
**Symptoms**: Early instability onset (timestep 80), extreme temperatures
**Likely Issues**:
- CFL condition violated
- Timestep too large for thermal LBM
- Source term discretization incorrect

### 2. Laser Power Deposition
**Symptoms**: Very high temperatures, large initial jump
**Likely Issues**:
- Power too concentrated spatially
- Absorption coefficient incorrect
- Energy units may be wrong (check W vs mW)

### 3. VOF-Thermal Coupling
**Symptoms**: Low liquid fraction despite high temperatures
**Likely Issues**:
- Latent heat not properly absorbed during phase change
- Energy transfer between phases incorrect
- Temperature boundary conditions at interface wrong

### 4. Energy Conservation
**Symptoms**: Monotonic temperature increase, widespread overheating
**Likely Issues**:
- Energy accumulating without dissipation
- Boundary conditions not removing heat properly
- Thermal conductivity may be too low

---

## Recommendations

### Immediate Debugging (Priority 1)

1. **Reduce laser power to 50 W** (from 200 W)
   - Check if temperatures stay below boiling
   - Verify basic thermal response

2. **Add energy conservation monitoring**
   - Track total energy in system
   - Monitor energy input vs output
   - Check for energy accumulation

3. **Review thermal CFL condition**
   - Calculate actual CFL number
   - Reduce timestep if needed
   - Verify relaxation time calculation

### Validation Testing (Priority 2)

4. **Pure thermal conduction test**
   - No laser, no VOF
   - Known analytical solution
   - Validates basic thermal solver

5. **Phase change test separately**
   - Fixed temperature boundary
   - Stefan problem setup
   - Validates VOF-thermal coupling

6. **Simplified laser test**
   - Lower power (20-50 W)
   - Shorter simulation time
   - Monitor for stability

### Code Review (Priority 3)

7. **Check source term implementation**
   - Verify Gaussian beam formula
   - Check absorption coefficient
   - Verify energy units (W not mW)

8. **Review material properties**
   - Thermal conductivity
   - Specific heat
   - Latent heat of fusion
   - Verify temperature-dependent properties

---

## Tools and Scripts

### Analysis Scripts
Located in: `/home/yzk/LBMProject/tests/validation/`

| Script | Purpose |
|--------|---------|
| `quick_vtk_analysis.sh` | Run complete analysis (one command) |
| `analyze_case5_detailed.py` | Generate detailed analysis and plots |
| `compare_with_rosenthal.py` | Compare with analytical solution |
| `diagnose_thermal_issues.py` | Identify thermal solver problems |

### VTK Comparison Framework
Located in: `/home/yzk/LBMProject/benchmark/vtk_comparison/`

```bash
# Activate virtual environment
source /home/yzk/LBMProject/benchmark/vtk_comparison/venv/bin/activate

# Run timeseries analysis
python3 vtk_compare.py timeseries <directory> --pattern "*.vtk"

# Extract metrics from single file
python3 vtk_compare.py metrics <file.vtk> --solidus 1923 --liquidus 1973

# Compare profiles
python3 vtk_compare.py profiles <file1.vtk> <file2.vtk> --axis x
```

---

## Next Steps

1. **DO NOT** use current results for validation
2. **DEBUG** thermal solver using recommendations above
3. **RE-RUN** Case 5 after fixes
4. **RE-ANALYZE** using this framework to verify improvements
5. **ITERATE** until validation passes (errors < 15%)

---

## Contact / Support

For questions about this analysis:
- Analysis framework: `/home/yzk/LBMProject/benchmark/vtk_comparison/README.md`
- VTK data format: Check VTK file headers
- Simulation setup: `/home/yzk/LBMProject/tests/validation/case5_rosenthal_validation.cpp`

---

**Analysis Framework Version**: 1.0
**Generated by**: VTK Comparison Framework
**Last Updated**: 2025-12-22
