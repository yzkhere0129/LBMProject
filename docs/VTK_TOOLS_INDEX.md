# VTK Comparison Tools - Documentation Index

**Last Updated:** 2025-12-04
**Status:** Production Ready

---

## Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [VTK_TOOLS_SUMMARY.md](#summary) | Overview and quick start | Everyone |
| [VTK_COMPARISON_EXAMPLES.md](#examples) | Comprehensive guide | Users |
| [README_VTK_TOOLS.md](#readme) | Technical reference | Developers |
| [QUICK_VTK_COMMANDS.sh](#commands) | Shell utilities | Power users |

---

## Document Details

### 1. VTK_TOOLS_SUMMARY.md <a name="summary"></a>

**Location:** `/home/yzk/LBMProject/VTK_TOOLS_SUMMARY.md`

**Purpose:** Executive summary and delivery documentation

**Contents:**
- Deliverables overview
- Quick start guide
- File locations
- Verification status
- Usage examples

**Best For:**
- First-time users
- Project managers
- Quick reference

**Read Time:** 5 minutes

---

### 2. VTK_COMPARISON_EXAMPLES.md <a name="examples"></a>

**Location:** `/home/yzk/LBMProject/docs/VTK_COMPARISON_EXAMPLES.md`

**Purpose:** Comprehensive user guide with detailed examples

**Contents:**
- VTK format specifications
- Tool usage documentation
- 10+ comparison examples
- ParaView workflows
- Troubleshooting guide
- File path reference

**Best For:**
- Regular users
- Learning workflows
- Solving specific problems

**Read Time:** 20-30 minutes (browse as needed)

---

### 3. README_VTK_TOOLS.md <a name="readme"></a>

**Location:** `/home/yzk/LBMProject/scripts/README_VTK_TOOLS.md`

**Purpose:** Technical reference for scripts directory

**Contents:**
- Script API documentation
- Advanced usage examples
- Performance optimization
- Testing procedures
- Integration guides

**Best For:**
- Developers
- Script customization
- Advanced workflows

**Read Time:** 15-20 minutes

---

### 4. QUICK_VTK_COMMANDS.sh <a name="commands"></a>

**Location:** `/home/yzk/LBMProject/scripts/QUICK_VTK_COMMANDS.sh`

**Purpose:** Shell functions and batch operations

**Contents:**
- 30+ pre-configured commands
- Batch processing utilities
- ParaView shortcuts
- File inspection tools

**Best For:**
- Command-line users
- Batch processing
- Quick operations

**Usage:**
```bash
source scripts/QUICK_VTK_COMMANDS.sh
show_help
```

---

## Quick Start by Task

### I want to compare two VTK files

**Read:** Section "Quick Start" in VTK_TOOLS_SUMMARY.md

**Run:**
```bash
cd /home/yzk/LBMProject
python3 scripts/compare_vtk_files.py file1.vtk file2.vtk
```

**Documentation:** VTK_COMPARISON_EXAMPLES.md → Example 1

---

### I want to visualize Marangoni flow

**Read:** Section "Marangoni Flow Visualization" in VTK_COMPARISON_EXAMPLES.md

**Run:**
```bash
python3 scripts/visualize_marangoni.py \
    build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk \
    --output-dir results
```

**Documentation:** VTK_COMPARISON_EXAMPLES.md → Example 4

---

### I want to use ParaView

**Read:** Section "ParaView Visualization" in VTK_COMPARISON_EXAMPLES.md

**Run:**
```bash
paraview build/tests/validation/phase6_test2c_visualization/marangoni_flow_010000.vtk &
```

**Documentation:** VTK_COMPARISON_EXAMPLES.md → ParaView workflows

---

### I want to compare LBMProject vs WalBerla

**Read:** Section "Example 1" in VTK_COMPARISON_EXAMPLES.md

**Run:**
```bash
python3 scripts/compare_vtk_files.py \
    build/config_parser_test/output_010000.vtk \
    /home/yzk/walberla/sim_output/sim_output_00000450.vtk \
    --plot-velocity --plot-temperature --output-dir results
```

**Documentation:** VTK_COMPARISON_EXAMPLES.md → Examples section

---

### I want to batch process many files

**Read:** Section "Batch Operations" in README_VTK_TOOLS.md

**Run:**
```bash
source scripts/QUICK_VTK_COMMANDS.sh
batch_compare_marangoni
```

**Documentation:** QUICK_VTK_COMMANDS.sh → help function

---

### I want to customize the scripts

**Read:** README_VTK_TOOLS.md → "Advanced Usage" section

**Study:**
- VTKData class in compare_vtk_files.py
- VTKComparator class implementation
- Custom analysis examples

**Documentation:** README_VTK_TOOLS.md → entire document

---

## File Structure

```
/home/yzk/LBMProject/
│
├── VTK_TOOLS_SUMMARY.md              # This is your starting point
│
├── docs/
│   ├── VTK_COMPARISON_EXAMPLES.md    # Comprehensive guide
│   └── VTK_TOOLS_INDEX.md            # This file
│
└── scripts/
    ├── compare_vtk_files.py          # General comparison tool
    ├── compare_poiseuille.py         # Poiseuille-specific tool
    ├── visualize_marangoni.py        # Marangoni visualization
    ├── QUICK_VTK_COMMANDS.sh         # Shell utilities
    └── README_VTK_TOOLS.md           # Technical reference
```

---

## Learning Path

### Beginner (1 hour)

1. Read VTK_TOOLS_SUMMARY.md (5 min)
2. Try basic comparison example (10 min)
3. Browse VTK_COMPARISON_EXAMPLES.md → Quick Start (10 min)
4. Open VTK in ParaView (15 min)
5. Experiment with slice comparisons (20 min)

### Intermediate (3 hours)

1. Complete beginner path
2. Read VTK_COMPARISON_EXAMPLES.md → Examples 1-5 (30 min)
3. Try Marangoni visualization (20 min)
4. Learn ParaView workflows (45 min)
5. Practice with shell functions (45 min)
6. Create custom comparison workflow (30 min)

### Advanced (1 day)

1. Complete intermediate path
2. Read README_VTK_TOOLS.md (30 min)
3. Study script implementation (2 hours)
4. Customize Python analysis (2 hours)
5. Set up batch processing (1 hour)
6. Create automated reporting (2 hours)

---

## Common Workflows

### Daily Development

```bash
# Quick comparison of latest build
source scripts/QUICK_VTK_COMMANDS.sh
compare_lbm_vs_walberla_final
```

### Weekly Testing

```bash
# Comprehensive validation
python3 scripts/compare_vtk_files.py \
    latest_test.vtk reference.vtk \
    --plot-velocity --plot-temperature \
    --output-dir weekly_validation
```

### Publication Figures

```bash
# High-quality visualizations
python3 scripts/visualize_marangoni.py \
    final_results.vtk \
    --output-dir publication_figures

# Edit DPI in script for higher resolution
```

### Batch Analysis

```bash
# Process all simulation outputs
source scripts/QUICK_VTK_COMMANDS.sh
batch_compare_lbm_walberla
```

---

## Reference Tables

### Script Comparison

| Script | Input | Output | Best For |
|--------|-------|--------|----------|
| compare_vtk_files.py | 2 VTK files | Metrics + plots | General comparison |
| compare_poiseuille.py | 2 VTK files | Profiles + metrics | Flow validation |
| visualize_marangoni.py | 1+ VTK files | Multi-panel plots | Surface flows |
| QUICK_VTK_COMMANDS.sh | Various | Various | Automation |

### Output Files

| File | Type | Purpose |
|------|------|---------|
| velocity_comparison.png | 4-panel plot | Velocity field analysis |
| temperature_comparison.png | 4-panel plot | Temperature field analysis |
| poiseuille_comparison.png | 2x2 plot | Profile comparison |
| marangoni_visualization.png | 6-panel plot | Field visualization |
| marangoni_surface_analysis.png | 4-panel plot | Surface analysis |
| marangoni_time_evolution.png | Grid plot | Time series |

### Error Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| L2 Error | ‖u₁ - u₂‖₂ / √N | Overall magnitude |
| Max Error | max\|u₁ - u₂\| | Worst-case difference |
| RMSE | √(mean((u₁ - u₂)²)) | Average error |
| Relative | mean(\|u₁ - u₂\|/\|u₁\|) | Normalized error |

---

## Troubleshooting Quick Reference

| Issue | Solution | Documentation |
|-------|----------|---------------|
| "python: command not found" | Use `python3` | README → Troubleshooting |
| "Could not extract velocity" | Check field names | VTK_COMPARISON_EXAMPLES → Troubleshooting |
| Shape mismatch warnings | Expected (different grids) | README → Common Issues |
| Memory errors | Use slice comparisons | VTK_COMPARISON_EXAMPLES → Performance |
| ParaView crashes | Reduce data size | VTK_COMPARISON_EXAMPLES → ParaView section |

---

## External Resources

### VTK Format
- **VTK File Format Specification:** https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

### ParaView
- **ParaView Guide:** https://docs.paraview.org/en/latest/
- **Tutorial:** https://www.paraview.org/Wiki/ParaView/Users_Guide/List_of_filters

### Python Libraries
- **NumPy Documentation:** https://numpy.org/doc/stable/
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/index.html

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-04 | Initial release with all tools |

---

## Feedback and Updates

For issues, suggestions, or updates:
- Check existing documentation
- Review examples in `/home/yzk/LBMProject/tests/`
- Examine configuration files in `/home/yzk/LBMProject/configs/`

---

**Quick Start:** Read VTK_TOOLS_SUMMARY.md first

**Comprehensive Guide:** See VTK_COMPARISON_EXAMPLES.md

**Technical Reference:** See scripts/README_VTK_TOOLS.md

**Shell Utilities:** Source scripts/QUICK_VTK_COMMANDS.sh
