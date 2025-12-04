# Marangoni Benchmark - Quick Start Guide

## What is This Test?

A standalone validation benchmark for thermocapillary (Marangoni) convection that can be directly compared with waLBerla's reference implementation.

**Purpose:** Validate that our Marangoni force implementation is correct before using it in complex LPBF simulations.

---

## The Physical Problem

### Geometry
```
         Top Wall (T = 10, u = 0)
    ================================
           ↓ slow flow ↓
    ~~~~ Interface (y = 128) ~~~~~  ← Marangoni force acts here
           ↑ slow flow ↑
    ================================
       Bottom Wall (T varies, u = 0)
```

**Domain:** 512 × 256 × 1 cells (quasi-2D)
**Interface:** Horizontal, stationary (no free surface tracking)
**Driving Force:** Temperature gradient along interface creates surface tension gradient

### Temperature Boundary Conditions
- **Top (y=255):** T = 10 (cold, constant)
- **Bottom (y=0):** T = 20 + 4·cos(π(x-256)/256) (hot, sinusoidal)
- **X-direction:** Periodic

The sinusoidal temperature at the bottom creates hot and cold spots along the interface, which drives thermocapillary flow.

### Physics
When temperature varies along an interface:
1. Surface tension σ varies: σ(T) = σ₀ + (dσ/dT)·T
2. Surface tension gradient creates tangential force: F = (dσ/dT)·∇T
3. This force drives flow from cold to hot regions (for dσ/dT < 0)

**No buoyancy, no phase change, no free surface deformation** - just pure Marangoni effect.

---

## Key Parameters (Case 1)

```cpp
// Domain
NX = 512, NY = 256, NZ = 1

// Fluid properties (both phases equal)
rho = 1.0
mu = 0.2
nu = 0.2

// Surface tension
sigma_ref = 0.025
sigma_t = -5e-4  // dσ/dT (negative: flow from cold to hot)

// Thermal properties
kappa = 0.2  // Thermal conductivity
alpha = 0.2  // Thermal diffusivity (kappa/rho/cp)

// Temperature
T_ref = 10   // Top wall
T_h = 20     // Bottom wall baseline
T_0 = 4      // Bottom wall amplitude
```

**Non-dimensional parameters:**
- Marangoni number: Ma ≈ 12.8
- Reynolds number: Re ≈ 12.8
- Prandtl number: Pr = 1.0

---

## Expected Results

### Temperature Field
- Sinusoidal variation along bottom (wavelength = 512 cells)
- Constant T=10 at top
- Smooth gradient between top and bottom
- Different gradients above/below interface (for Case 2)

### Velocity Field
- Maximum velocity at interface: u_max ~ 0.001-0.01 (lattice units)
- Flow from cold to hot along interface (→ direction)
- Return flow in bulk fluid above/below interface (← direction)
- Creates counter-rotating vortex cells
- Zero velocity at walls (no-slip BC)

### Validation Metrics
- **L2_T < 0.01** (1% error in temperature field)
- **L2_U < 0.05** (5% error in velocity field)

Compared against analytical solution from Chai et al., JCP 2013.

---

## Implementation Strategy

### Option A: Simple Phase Field (Recommended)
```
Components:
├── ThermalLBM (D3Q7)
├── FluidLBM (D3Q19)
├── Marangoni force kernel
└── Static phase field (no VOF)

Complexity: Low
Time: 2-3 days
Validation: Direct
```

**Why this approach:**
- Interface is flat and stationary - no need for VOF
- Validates Marangoni physics in isolation
- Fast debugging
- Direct comparison with analytical solution

### What We Need to Implement

**New Kernels:**
1. **Phase field initialization**
   - Tanh profile: φ = 0.5 + 0.5·tanh((y-128)/2.5)

2. **Temperature gradient computation**
   - Central difference: dT/dx, dT/dy, dT/dz

3. **Marangoni force**
   - F = (dσ/dT)·∇_s T (tangent to interface)
   - Only active where 0.4 < φ < 0.6

4. **Thermal Dirichlet BC**
   - Top: T = 10
   - Bottom: T = 20 + 4·cos(π(x-256)/256)

**Modified Components:**
- ThermalLBM: May need spatially varying thermal conductivity
- FluidLBM: Already supports body force (Guo forcing)

---

## Test Cases

### Case 1: Equal Thermal Conductivity
```
kappa_top = 0.2
kappa_bottom = 0.2

Expected: Symmetric temperature profile
          Moderate Marangoni flow
```

### Case 2: Different Thermal Conductivity
```
kappa_top = 0.04  (5x less conductive)
kappa_bottom = 0.2

Expected: Asymmetric temperature profile
          Different gradients above/below interface
          Similar flow speed but different pattern
```

---

## Comparison with waLBerla

### Their Implementation
- Location: `/home/yzk/walberla/apps/showcases/Thermocapillary/`
- Benchmark: `microchannel2D.py`
- Analytical solution: `lbmpy.phasefield_allen_cahn.analytical.analytical_solution_microchannel()`

### What to Match
- [x] Exact same domain size (512, 256, 1)
- [x] Exact same parameters (Case 1 or 2)
- [x] Same boundary conditions
- [x] Same interface position and thickness
- [x] Same analytical solution for validation

### Validation Protocol
1. Run our implementation
2. Run waLBerla microchannel2D.py
3. Compare VTK outputs visually
4. Compare L2 errors (should be within 10%)
5. Compare velocity profiles along vertical lines
6. Compare temperature contours

---

## Success Criteria

### Minimum Viable Product (Day 3)
- [x] Test compiles and runs
- [x] Reaches steady state
- [x] Shows qualitatively correct flow pattern
- [x] Temperature shows sinusoidal variation
- [x] L2_T < 0.05

### Production Quality (Day 4)
- [x] L2_T < 0.01
- [x] L2_U < 0.05
- [x] Both Case 1 and Case 2 pass
- [x] VTK output for visualization
- [x] Matches waLBerla within 10%

### Stretch Goals (Week 2)
- [x] Grid convergence study (128, 256, 512)
- [x] Matches waLBerla within 1%
- [x] Performance benchmark (MLUPS)
- [x] 3D droplet migration test

---

## Debugging Checklist

### If Velocity is Wrong
1. Check Marangoni force direction
   - For sigma_t < 0, flow should be from cold to hot
   - Print force field, verify it's tangent to interface
2. Check force magnitude
   - Should be O(10⁻⁴) for these parameters
3. Verify no-slip BC at walls
   - u should be exactly zero at j=0 and j=NY-1

### If Temperature is Wrong
1. Check boundary conditions
   - Plot T at j=0 vs x (should be sinusoidal)
   - Plot T at j=NY-1 (should be constant = 10)
2. Check thermal diffusivity
   - If too high: temperature over-smoothed
   - If too low: sharp gradients, instability
3. Check for NaN
   - Usually from unstable relaxation time

### If Test Doesn't Converge
1. Reduce time step (unlikely needed)
2. Check CFL condition
   - Thermal: α·dt/dx² < 0.5 ✓ (0.2 for our params)
   - Viscous: ν·dt/dx² < 0.5 ✓ (0.2 for our params)
   - Convective: u·dt/dx < 1.0 ✓ (0.01 for our params)
3. Monitor max velocity
   - Should grow then plateau around 0.001-0.01
   - If exploding: relaxation time issue

---

## File Locations

### Design Documents
- `/home/yzk/LBMProject/docs/MARANGONI_BENCHMARK_DESIGN.md` - Full architecture (this was comprehensive!)
- `/home/yzk/LBMProject/docs/MARANGONI_BENCHMARK_QUICK_START.md` - This file

### Implementation (to be created)
```
/home/yzk/LBMProject/
├── tests/validation/
│   └── test_marangoni_microchannel.cu          [NEW - main test]
├── include/physics/
│   └── marangoni_force.h                       [NEW - force interface]
├── src/physics/marangoni/
│   └── marangoni_force.cu                      [NEW - CUDA kernels]
└── scripts/
    ├── analytical_marangoni.py                 [NEW - analytical solution]
    └── validate_marangoni.py                   [NEW - compare with waLBerla]
```

### waLBerla Reference
```
/home/yzk/walberla/apps/showcases/Thermocapillary/
├── microchannel2D.py                 - Test configuration
├── thermocapillary.cpp               - Main C++ code
└── InitializerFunctions.cpp          - Phase/temp initialization

/home/yzk/walberla/build/venv/lib/python3.12/site-packages/lbmpy/
└── phasefield_allen_cahn/analytical.py - Analytical solution
```

---

## Quick Command Reference

### Run waLBerla Test
```bash
cd /home/yzk/walberla/apps/showcases/Thermocapillary
python microchannel2D.py

# Output:
# - VTK files in results_*/
# - CSV with L2 errors
```

### Run Our Test (when implemented)
```bash
cd /home/yzk/LBMProject/build
./tests/validation/test_marangoni_microchannel

# Expected output:
# Step 0: max_u = 0.000000e+00
# Step 100: max_u = 2.341e-04
# ...
# Step 10000: max_u = 8.234e-04
# Validation results:
#   L2_T = 0.008432 (target < 0.01)
#   L2_U = 0.042156 (target < 0.05)
# Test PASSED
```

### Visualize Results
```bash
paraview velocity_sim.vtk

# In ParaView:
# 1. Load temperature and velocity fields
# 2. Add "Glyph" filter for velocity vectors
# 3. Add "Contour" filter for temperature iso-lines
# 4. Compare with waLBerla VTK
```

---

## Key Insights from waLBerla

### What They Do Well
1. **Clean separation:** Benchmark script (Python) vs solver (C++)
2. **Analytical validation:** Built into the test, auto-computes L2 errors
3. **Multiple cases:** Tests both equal and different thermal conductivity
4. **VTK output:** Both velocity and temperature fields for visual validation

### What We Can Learn
1. Use same parameters exactly → direct comparison
2. Port their analytical solution → identical validation
3. Match their domain setup → 1:1 correspondence
4. Use their L2 error metric → quantitative comparison

### What We Do Better
1. **CUDA acceleration:** Our GPU kernels vs their CPU
2. **Modular physics:** Clean ThermalLBM/FluidLBM separation
3. **Material properties:** Our MaterialProperties class is more flexible
4. **Modern C++:** Our code uses C++17 features

---

## Timeline

### Day 1: Setup
- [x] Read design documents
- [ ] Create test file skeleton
- [ ] Implement phase field initialization
- [ ] Set up VTK output
- [ ] Verify domain and BCs

### Day 2: Thermal
- [ ] Configure ThermalLBM
- [ ] Implement sinusoidal Dirichlet BC
- [ ] Run thermal-only test
- [ ] Validate temperature field pattern
- [ ] Check steady state convergence

### Day 3: Coupled
- [ ] Implement Marangoni force kernel
- [ ] Configure FluidLBM
- [ ] Run coupled simulation
- [ ] Debug and iterate
- [ ] First validation attempt

### Day 4: Validation
- [ ] Port analytical solution
- [ ] Compute L2 errors
- [ ] Compare with waLBerla
- [ ] Generate plots
- [ ] Document results

---

## Questions to Resolve

1. **Does ThermalLBM support spatially varying thermal conductivity?**
   - Check: Can we have different kappa for top/bottom fluids?
   - If no: Need to implement local tau computation

2. **How is Guo forcing applied in FluidLBM?**
   - Check: `setBodyForce()` interface
   - Verify: Force is added correctly in collision step

3. **Are periodic BCs already implemented?**
   - Check: X-direction periodicity for both thermal and fluid
   - If no: Need to implement periodic wrapping

4. **Do we need to handle interface cells specially?**
   - Check: Is harmonic averaging needed for thermal conductivity?
   - Probably: Yes for high accuracy, but can start without it

---

## References

**Main Paper:**
- Chai et al. (2013), "A comparative study of local and nonlocal Allen-Cahn equations with mass conservation", JCP
- Provides analytical solution for microchannel thermocapillary flow

**waLBerla Implementation:**
- `/home/yzk/walberla/apps/showcases/Thermocapillary/`
- Working reference code, can be compiled and run

**lbmpy Analytical Solution:**
- `analytical_solution_microchannel()` function
- Python implementation, can be ported to C++

**Our Existing Code:**
- ThermalLBM: `/home/yzk/LBMProject/include/physics/thermal_lbm.h`
- FluidLBM: `/home/yzk/LBMProject/include/physics/fluid_lbm.h`
- Marangoni: `/home/yzk/LBMProject/src/physics/vof/marangoni.cu` (may need modification)

---

**Ready to implement? Start with:**
1. Create `tests/validation/test_marangoni_microchannel.cu`
2. Copy structure from `test_pure_conduction.cu` or similar
3. Add phase field initialization kernel
4. Add Marangoni force kernel
5. Wire up ThermalLBM + FluidLBM
6. Run and debug!

**Expected first result:** Should see temperature gradient and weak flow within 1-2 days of coding.
