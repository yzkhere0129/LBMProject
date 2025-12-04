# Marangoni Benchmark - Documentation Index

## Overview

Complete architectural design for a standalone Marangoni convection benchmark that validates our implementation against waLBerla's reference code.

**Created:** 2025-12-03
**Status:** Design complete, ready for implementation
**Estimated Time:** 3-4 days

---

## Core Documentation (Read These)

### 1. README (Start Here)
**File:** `docs/README_MARANGONI_BENCHMARK.md` (307 lines)

Quick navigation guide to all documentation. Read this first.

### 2. Quick Start Guide
**File:** `docs/MARANGONI_BENCHMARK_QUICK_START.md` (425 lines)

Practical implementation guide with:
- Physical problem description
- Key parameters
- Test cases
- Debugging checklist
- Timeline

**Read this when ready to code.**

### 3. Full Design Document
**File:** `docs/MARANGONI_BENCHMARK_DESIGN.md` (1293 lines)

Complete technical specification:
- waLBerla analysis
- Architecture design
- Implementation roadmap
- Code templates
- Validation methodology

**Reference during implementation.**

### 4. Executive Summary
**File:** `MARANGONI_BENCHMARK_SUMMARY.md` (96 lines)

High-level overview of the benchmark design.

---

## Quick Reference

### What We're Building
A 2D microchannel thermocapillary flow test:
- Domain: 512 × 256 × 1 cells
- Horizontal interface at y=128
- Temperature gradient drives Marangoni flow
- Compare with waLBerla analytical solution

### Key Parameters (Case 1)
```
ρ = 1.0, μ = 0.2, ν = 0.2
σ₀ = 0.025, dσ/dT = -5×10⁻⁴
κ = 0.2 (both fluids)
T_top = 10, T_bottom = 20 + 4·cos(...)
Ma ≈ 12.8, Re ≈ 12.8
```

### Success Criteria
- L2_T < 0.01 (temperature error)
- L2_U < 0.05 (velocity error)
- Matches waLBerla within 10%

---

## Implementation Checklist

### Files to Create
- [ ] `tests/validation/test_marangoni_microchannel.cu`
- [ ] `include/physics/marangoni_force.h`
- [ ] `src/physics/marangoni/marangoni_force.cu`
- [ ] `scripts/analytical_marangoni.py`
- [ ] `scripts/validate_marangoni.py`

### Key Components
- [ ] Phase field initialization (tanh profile)
- [ ] Temperature gradient kernel
- [ ] Marangoni force kernel
- [ ] Sinusoidal thermal BC
- [ ] ThermalLBM + FluidLBM coupling
- [ ] VTK output
- [ ] L2 error computation

---

## waLBerla Reference

**Location:**
```
/home/yzk/walberla/apps/showcases/Thermocapillary/
├── microchannel2D.py
├── thermocapillary.cpp
└── InitializerFunctions.cpp
```

**Run:**
```bash
cd /home/yzk/walberla/apps/showcases/Thermocapillary
python microchannel2D.py
```

**Analytical Solution:**
```python
from lbmpy.phasefield_allen_cahn.analytical import analytical_solution_microchannel
```

---

## Reading Order

1. **This index** (2 min) - Overview
2. **README_MARANGONI_BENCHMARK.md** (5 min) - Navigation guide
3. **MARANGONI_BENCHMARK_QUICK_START.md** (20 min) - Implementation prep
4. **MARANGONI_BENCHMARK_DESIGN.md** (as needed) - Technical details

---

## Timeline

- **Day 1:** Infrastructure (domain, phase field, VTK)
- **Day 2:** Thermal solver (temperature field + BCs)
- **Day 3:** Marangoni force (force kernel + coupling)
- **Day 4:** Validation (analytical comparison, plots)

---

## Key Design Choices

1. **Simple phase field** (no VOF) - interface is stationary
2. **Local tau computation** - handle varying thermal conductivity
3. **Guo forcing** - use existing FluidLBM force interface
4. **Python validation** - use lbmpy analytical solution initially

---

## Questions to Resolve Before Starting

1. Does ThermalLBM support spatially varying thermal conductivity?
2. How is `setBodyForce()` implemented in FluidLBM?
3. Are periodic BCs already working in both solvers?
4. Do we need harmonic averaging at interface cells?

---

## Success Metrics

**MVP (Day 3):**
- Runs and converges
- Qualitatively correct pattern
- L2_T < 0.05

**Production (Day 4):**
- L2_T < 0.01
- L2_U < 0.05
- Matches waLBerla ±10%

---

## References

**Paper:** Chai et al. (2013), JCP - Analytical solution
**waLBerla:** `/home/yzk/walberla/apps/showcases/Thermocapillary/`
**Our Code:** ThermalLBM, FluidLBM in `/home/yzk/LBMProject/include/physics/`

---

**Next Action:** Read README_MARANGONI_BENCHMARK.md, then start Day 1 tasks.
