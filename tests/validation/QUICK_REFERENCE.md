# Validation Framework - Quick Reference Card

**Version:** 1.0 | **Date:** 2025-01-19

---

## Essential Commands

### Run All Validation Tests
```bash
cd /home/yzk/LBMProject/tests/validation
./run_all_validation.sh                    # Standard (~1 hour)
./run_all_validation.sh --quick            # Quick check (~30 min)
./run_all_validation.sh --full             # Including benchmark (~3 hours)
```

### Run Individual Tests
```bash
./test_grid_convergence.sh                 # 30-45 min
./test_peclet_sweep.sh                     # 20-30 min
./test_energy_conservation.sh              # 15-20 min
./test_literature_benchmark.sh             # 2-3 hours
./test_flux_limiter_impact.sh              # 20-30 min
```

### Analyze Results
```bash
cd /home/yzk/LBMProject/analysis
jupyter lab validation_analysis.ipynb      # Interactive analysis
```

---

## Success Criteria Quick Check

| Test | Metric | Pass If |
|------|--------|---------|
| Grid Convergence | Order | ≥ 1.0 |
| Peclet Sweep | T_max | < 10,000 K (all Pe) |
| Energy | Error | < 5% average |
| Literature | T_peak | ± 20% of 2600 K |
| Flux Limiter | Accuracy & Speedup | < 5% loss, > 2× faster |

---

## Results Location

```
/home/yzk/LBMProject/validation_results/
├── run_YYYYMMDD_HHMMSS/validation_summary.txt  ← Start here
├── grid_convergence/convergence_report.txt
├── peclet_sweep/peclet_sweep_report.txt
├── energy_conservation/energy_report.txt
├── literature_benchmark/benchmark_comparison.txt
└── flux_limiter_impact/comparison_report.txt
```

---

## Interpretation

### Exit Code 0 (All Pass)
→ Fixes are treating **ROOT CAUSE (治本)**
→ Deploy to production

### Exit Code 1 (Some Fail)
→ Fixes may be **SUPERFICIAL (治标)**
→ Debug failed tests

---

## Common Issues

**Out of Memory:**
Reduce grid resolution in test scripts (decrease nx, ny, nz)

**Tests Timeout:**
Increase timeout or run on more powerful GPU

**Missing VTK Files:**
Check disk space: `df -h`

**Diagnostics Missing:**
Verify `enable_diagnostics = true` in config

---

## File Locations

| File | Path |
|------|------|
| Test Scripts | `/home/yzk/LBMProject/tests/validation/` |
| Diagnostic Kernels | `/home/yzk/LBMProject/include/physics/thermal/validation_diagnostics.h` |
| Analysis Notebook | `/home/yzk/LBMProject/analysis/validation_analysis.ipynb` |
| Report Template | `/home/yzk/LBMProject/docs/validation/VALIDATION_REPORT_TEMPLATE.md` |
| Full README | `/home/yzk/LBMProject/tests/validation/README.md` |

---

## Help

**Detailed Documentation:**
```bash
less /home/yzk/LBMProject/tests/validation/README.md
```

**Command Help:**
```bash
./run_all_validation.sh --help
```

---

**End of Quick Reference**
