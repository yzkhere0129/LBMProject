# Marangoni Benchmark Documentation Index

## Overview

Complete architectural design for a standalone Marangoni (thermocapillary) convection benchmark test that validates our LBM implementation against waLBerla's reference code.

**Status:** Design complete, ready for implementation
**Estimated Implementation Time:** 3-4 days
**Complexity:** Medium
**Dependencies:** ThermalLBM, FluidLBM (existing)

---

## Documentation Files

### 1. Executive Summary
**File:** `/home/yzk/LBMProject/MARANGONI_BENCHMARK_SUMMARY.md`

**Quick overview:**
- What the benchmark tests
- Key findings from waLBerla
- Implementation roadmap
- File locations

**Read this first** for high-level understanding.

**Length:** ~300 lines

---

### 2. Quick Start Guide
**File:** `/home/yzk/LBMProject/docs/MARANGONI_BENCHMARK_QUICK_START.md`

**Practical implementation guide:**
- Physical problem description
- Key parameters and expected results
- Test cases (Case 1 & 2)
- Debugging checklist
- Command reference
- Timeline with daily breakdown

**Read this second** when ready to implement.

**Length:** ~400 lines

---

### 3. Full Architecture Design
**File:** `/home/yzk/LBMProject/docs/MARANGONI_BENCHMARK_DESIGN.md`

**Complete technical specification:**

**Section 1:** waLBerla Reference Analysis
- Test case details (microchannel geometry)
- Physical parameters
- Analytical solution
- Validation metrics

**Section 2:** Our Implementation Architecture
- Design philosophy
- Module integration strategy
- Code structure
- Domain configuration

**Section 3:** Implementation Roadmap
- 6-phase development plan
- Detailed task breakdown

**Section 4:** Expected Results
- Physical behavior
- Quantitative metrics
- Visualization checklist

**Section 5:** Comparison Methodology
- Direct waLBerla comparison
- Validation protocol
- Success criteria

**Section 6:** Architectural Decisions
- Phase field vs VOF choice
- Thermal conductivity handling
- Force application method
- BC strategy

**Section 7:** Testing and Debugging
- Unit tests
- Integration tests
- Debugging workflow

**Section 8:** File Structure
- New files to create
- Modified components

**Section 9:** Success Metrics
- Technical validation
- Code quality
- Scientific impact

**Section 10:** Risk Assessment
- Key risks and mitigations

**Section 11:** Next Steps
- Immediate, short-term, medium-term actions

**Section 12:** References
- Papers, waLBerla code, our existing code

**Appendices:**
- A: Parameter tables (Case 1 & 2)
- B: Dimensional analysis (Ma, Re, Pr)
- C: Code snippet templates

**Read this** for detailed implementation guidance.

**Length:** ~1000 lines

---

## Quick Reference

### Test Parameters (Case 1)
```
Domain:  512 × 256 × 1
ρ = 1.0, μ = 0.2, ν = 0.2
σ₀ = 0.025, dσ/dT = -5×10⁻⁴
κ = 0.2 (both fluids)
T_top = 10, T_bottom = 20 + 4·cos(...)
```

### Expected Results
```
L2_T < 0.01  (temperature error)
L2_U < 0.05  (velocity error)
u_max ~ 0.008 (maximum velocity)
Ma ≈ 12.8, Re ≈ 12.8
```

### Implementation Tasks
1. Create `test_marangoni_microchannel.cu`
2. Implement phase field initialization
3. Implement Marangoni force kernel
4. Configure thermal BC (sinusoidal)
5. Run coupled simulation
6. Validate against analytical solution

---

## waLBerla Reference Code

### Location
```
/home/yzk/walberla/apps/showcases/Thermocapillary/
├── microchannel2D.py           - Test script
├── thermocapillary.cpp         - C++ implementation
└── InitializerFunctions.cpp    - Initialization
```

### Run waLBerla Test
```bash
cd /home/yzk/walberla/apps/showcases/Thermocapillary
python microchannel2D.py
```

Output: VTK files, CSV with L2 errors

### Analytical Solution
```python
from lbmpy.phasefield_allen_cahn.analytical import analytical_solution_microchannel

x, y, u_x, u_y, T = analytical_solution_microchannel(
    reference_length=256,
    length_x=512, length_y=256,
    kappa_top=0.2, kappa_bottom=0.2,
    t_h=20, t_c=10, t_0=4,
    sigma_t=-5e-4, mu=0.2
)
```

---

## Files to Create

### Core Implementation
```
/home/yzk/LBMProject/
├── tests/validation/
│   └── test_marangoni_microchannel.cu    [Main test driver]
│
├── include/physics/
│   └── marangoni_force.h                 [Force interface]
│
└── src/physics/marangoni/
    └── marangoni_force.cu                [CUDA kernels]
```

### Validation Scripts
```
/home/yzk/LBMProject/scripts/
├── analytical_marangoni.py               [Analytical solution]
└── validate_marangoni.py                 [Comparison script]
```

---

## Key Design Decisions

### 1. Simple Phase Field (Not VOF)
**Reason:** Interface is flat and stationary
**Benefit:** Simpler, faster, isolates Marangoni physics
**Trade-off:** Can't handle moving interfaces (add VOF later)

### 2. Spatially Varying Thermal Conductivity
**Approach:** Compute local tau from phase field
**Implementation:** In collision kernel: κ(φ) = (1-φ)·κ_gas + φ·κ_liquid

### 3. Marangoni Force via Guo Forcing
**Interface:** FluidLBM::setBodyForce(fx, fy, fz)
**Computation:** External kernel computes F = (dσ/dT)·∇_s T

### 4. Analytical Validation
**Short-term:** Python script for comparison
**Long-term:** Port to C++ for self-contained test

---

## Success Criteria

### Minimum Viable Product (Day 3)
- Test runs and converges
- Qualitatively correct flow pattern
- L2_T < 0.05

### Production Quality (Day 4)
- L2_T < 0.01
- L2_U < 0.05
- Matches waLBerla within 10%
- VTK output for visualization

### Stretch Goals
- Grid convergence O(h²)
- Matches waLBerla within 1%
- Performance benchmark
- 3D droplet migration test

---

## Implementation Timeline

### Day 1: Infrastructure
- Create test file skeleton
- Implement phase field initialization
- Set up VTK output
- Verify domain and BCs

### Day 2: Thermal Solver
- Configure ThermalLBM
- Implement sinusoidal Dirichlet BC
- Run thermal-only simulation
- Validate temperature pattern

### Day 3: Coupled Solver
- Implement Marangoni force kernel
- Configure FluidLBM
- Run coupled thermal-fluid-Marangoni
- Debug and achieve MVP

### Day 4: Validation
- Port analytical solution (or use Python)
- Compute L2 errors
- Compare with waLBerla
- Generate publication-quality plots

---

## Debugging Quick Reference

### Velocity Issues
1. Check force direction (cold→hot for σ_t<0)
2. Verify force at interface only
3. Check no-slip BC at walls

### Temperature Issues
1. Verify sinusoidal BC at bottom
2. Check thermal diffusivity
3. Monitor for NaN

### Convergence Issues
1. Check CFL conditions
2. Monitor max velocity
3. Reduce time step if needed

---

## Questions to Resolve

1. Does ThermalLBM support spatially varying κ?
2. How is setBodyForce() implemented in FluidLBM?
3. Are periodic BCs already working?
4. Do we need harmonic averaging at interface?

---

## Contact and Support

**Documentation Author:** CFD Platform Architect
**Date Created:** 2025-12-03
**Status:** Complete, ready for implementation

**For implementation questions:**
- Review MARANGONI_BENCHMARK_DESIGN.md Section 7 (Testing/Debugging)
- Check MARANGONI_BENCHMARK_QUICK_START.md debugging checklist
- Refer to waLBerla reference code for clarification

**For validation questions:**
- See Section 5 (Comparison Methodology) in design doc
- Use analytical solution from lbmpy package
- Compare VTK outputs visually in ParaView

---

## References

**Papers:**
- Chai et al. (2013), "A comparative study of local and nonlocal Allen-Cahn equations with mass conservation", JCP

**waLBerla Code:**
- Thermocapillary showcase: `/home/yzk/walberla/apps/showcases/Thermocapillary/`
- Analytical solution: `lbmpy.phasefield_allen_cahn.analytical`

**Our Codebase:**
- ThermalLBM: `/home/yzk/LBMProject/include/physics/thermal_lbm.h`
- FluidLBM: `/home/yzk/LBMProject/include/physics/fluid_lbm.h`
- Marangoni: `/home/yzk/LBMProject/src/physics/vof/marangoni.cu`

---

## How to Use This Documentation

1. **Start here** - Read this index
2. **Get overview** - Read MARANGONI_BENCHMARK_SUMMARY.md
3. **Practical guide** - Read MARANGONI_BENCHMARK_QUICK_START.md
4. **Deep dive** - Read MARANGONI_BENCHMARK_DESIGN.md as needed during implementation
5. **Reference waLBerla** - Run their test for comparison
6. **Implement** - Follow roadmap, refer back to docs

**Recommended reading order:**
1. This file (5 min)
2. Summary (10 min)
3. Quick Start (20 min)
4. Design Doc (as needed during implementation)

---

**Ready to implement?** Start with MARANGONI_BENCHMARK_QUICK_START.md and follow the Day 1 tasks.

**Questions about architecture?** See MARANGONI_BENCHMARK_DESIGN.md Section 6.

**Need debugging help?** Check MARANGONI_BENCHMARK_QUICK_START.md debugging checklist.

**Want to see reference?** Run waLBerla microchannel2D.py.
