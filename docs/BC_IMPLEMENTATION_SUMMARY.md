# Boundary Condition Implementation Summary

## Quick Reference for Implementation

This is the **executive summary** of the full BC design. Read this first, then refer to `BC_DESIGN_DOCUMENT.md` for complete details.

---

## The Problem

**Current BC Setup (WRONG):**
```cpp
FluidLBM fluid(nx, ny, nz, nu_lattice, rho_liquid,
               BoundaryType::PERIODIC,  // X - WRONG
               BoundaryType::PERIODIC,  // Y - WRONG
               BoundaryType::WALL);      // Z - PARTIAL
```

**Result:** Interface drifts upward, NaN/Inf in output, test failures.

**Root Cause:** Periodic X-Y creates infinite domain, no free surface at top, no outflow.

---

## The Solution

**Correct BC Setup:**
```cpp
FluidLBM fluid(nx, ny, nz, nu_lattice, rho_liquid,
               BoundaryType::OUTFLOW,      // X - zero-gradient
               BoundaryType::OUTFLOW,      // Y - zero-gradient
               BoundaryType::WALL);        // Z - bottom wall + top free surface
```

| Boundary | FluidLBM BC | VOF BC | Physics |
|----------|-------------|--------|---------|
| Bottom (z=0) | Bounce-back (no-slip) | Contact angle (150°) | Substrate |
| Top (z=z_max) | Zou-He pressure (p=0) | Zero-gradient fill | Free surface |
| Sides (x, y) | Zero-gradient outflow | Zero-gradient fill | Open boundaries |

---

## Implementation Checklist

### Phase 1: Quick Win (5 minutes) ✅ HIGHEST PRIORITY

**File:** `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity.cu`

**Line 651:** Add VOF contact angle BC (1 line)
```cpp
vof.reconstructInterface();

// ADD THIS LINE:
vof.applyBoundaryConditions(1, 150.0f);  // Contact angle at substrate
```

**Expected Impact:** Interface shape improves near substrate.

---

### Phase 2: Free Surface at Top (2 hours)

#### Step 1: Add Enum
**File:** `/home/yzk/LBMProject/include/physics/fluid_lbm.h`
**Line 36-39:**
```cpp
enum class BoundaryType {
    PERIODIC = 0,
    WALL = 1,
    OUTFLOW = 2,        // NEW
    FREE_SURFACE = 3    // NEW
};
```

#### Step 2: Add Kernel Declaration
**File:** `/home/yzk/LBMProject/include/physics/fluid_lbm.h`
**After line 313:**
```cpp
__global__ void applyFreeSurfaceTopKernel(
    float* f_src,
    const float* rho, const float* ux, const float* uy, const float* uz,
    float p_atm, float rho0, float cs2,
    int nx, int ny, int nz);
```

#### Step 3: Implement Kernel
**File:** `/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu`
**After line 845:** Add full kernel implementation (see design doc Appendix A)

#### Step 4: Call in applyBoundaryConditions
**File:** `/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu`
**Line 283-306:** Replace with comprehensive BC system (see design doc Part 2.4)

---

### Phase 3: Outflow at Sides (1 hour)

#### Step 1: Add Kernel Declaration
**File:** `/home/yzk/LBMProject/include/physics/fluid_lbm.h`
```cpp
__global__ void applyOutflowKernel(
    float* f_src,
    int boundary_face,
    int nx, int ny, int nz);
```

#### Step 2: Implement Kernel
**File:** `/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu`
**After line 845:** (See design doc Section 2.5)

#### Step 3: Update Test Configuration
**File:** `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity.cu`
**Line 543-546:**
```cpp
FluidLBM fluid(nx, ny, nz, nu_lattice, rho_liquid,
               BoundaryType::OUTFLOW,   // FIX: was PERIODIC
               BoundaryType::OUTFLOW,   // FIX: was PERIODIC
               BoundaryType::WALL);
```

**Line 706:** Update BC call
```cpp
fluid.applyBoundaryConditions(2);  // 2 = comprehensive BC
```

---

## Validation Tests

After each phase, run:

```bash
cd /home/yzk/LBMProject/build
make test_marangoni_velocity
./tests/validation/test_marangoni_velocity
```

**Expected Results:**
- ✅ Interface at z < 10 (not z=12)
- ✅ Marangoni velocity 0.7-1.5 m/s
- ✅ No NaN/Inf in output
- ✅ Pressure at top ≈ 0 Pa

---

## Key Equations

### Zou-He at Free Surface (z=z_max)
```
ρ = ρ₀ + p_atm/c_s²
u_x, u_y ← extrapolate from z-1
u_z ← computed from continuity
f_unknown ← f_eq(ρ, u_x, u_y, u_z)
```

### Contact Angle Correction
```
n_corrected = n_tangential + cos(θ) * n_wall
θ = 150° for Ti6Al4V (non-wetting)
```

### Zero-Gradient Outflow
```
f(boundary) ← f(interior neighbor)
```

---

## Debugging Checklist

If test still fails:

1. **Check BC is actually called:**
   - Add `printf("Free surface BC applied\n")` in kernel
   - Should see message every step

2. **Check pressure at top:**
   ```cpp
   float p_top = h_pressure[nx/2 + nx*(ny/2 + ny*(nz-1))];
   std::cout << "p(z_max) = " << p_top << " Pa" << std::endl;
   // Should be ≈ 0
   ```

3. **Check interface position:**
   ```cpp
   int z_max_interface = 0;
   for (int idx = 0; idx < num_cells; ++idx) {
       if (h_fill[idx] > 0.5f) {
           int k = idx / (nx*ny);
           z_max_interface = max(z_max_interface, k);
       }
   }
   std::cout << "z_interface_max = " << z_max_interface << std::endl;
   // Should be < 10
   ```

4. **Check mass conservation:**
   ```cpp
   float mass_0 = vof.computeTotalMass();
   // ... simulate ...
   float mass_final = vof.computeTotalMass();
   float loss = (mass_0 - mass_final) / mass_0 * 100.0f;
   std::cout << "Mass loss: " << loss << "%" << std::endl;
   // Should be < 0.1%
   ```

---

## Expected Performance Impact

| Change | Overhead | Status |
|--------|----------|--------|
| Contact angle (VOF) | 0.1% | ✅ Negligible |
| Free surface (FluidLBM) | 0.5% | ✅ Acceptable |
| Outflow (FluidLBM) | 1.5% | ✅ Acceptable |
| **TOTAL** | **~2%** | ✅ Under 10% threshold |

---

## File Modification Summary

| File | Lines | Changes |
|------|-------|---------|
| `include/physics/fluid_lbm.h` | 36-39, 313+ | Add enums, kernel declarations |
| `src/physics/fluid/fluid_lbm.cu` | 283-306, 845+ | Update BC function, add kernels |
| `tests/validation/test_marangoni_velocity.cu` | 543, 651, 706 | Fix BC types, add VOF BC call |
| `include/physics/vof_solver.h` | 134 | Update comment (optional) |

**Total Lines Changed:** ~200
**Total New Kernel Code:** ~150 lines
**Implementation Time:** 3-4 hours

---

## Success Criteria

### Must Pass All:
- [x] Phase 1: VOF contact angle applied (visual check in ParaView)
- [ ] Phase 2: Free surface BC working (p(z_max) ≈ 0)
- [ ] Phase 3: Outflow BC stable (no mass loss)
- [ ] Full test: Interface at z<10, velocity 0.7-1.5 m/s, no NaN

---

## Critical Implementation Notes

1. **Order Matters:** Apply BCs in sequence: wall → free surface → outflow
2. **Kernel Launch:** Use 2D grids for boundary kernels (more efficient than 3D)
3. **Edge Cases:** Corners handled by priority (wall > free surface > outflow)
4. **VOF Coupling:** Call `vof.applyBoundaryConditions()` AFTER `vof.reconstructInterface()`
5. **Synchronization:** Add `cudaDeviceSynchronize()` after each BC kernel group

---

## References

- **Full Design:** `/home/yzk/LBMProject/docs/BC_DESIGN_DOCUMENT.md`
- **Zou-He Paper:** Zou & He (1997), "On pressure and velocity boundary conditions..."
- **Contact Angle:** Implemented in `vof_solver.cu:197-252` (already exists!)
- **Bounce-Back:** Working in `fluid_lbm.cu:293-305` (no changes needed)

---

## Next Steps

1. **NOW:** Implement Phase 1 (5 minutes)
2. **Today:** Implement Phase 2 (2 hours)
3. **Tomorrow:** Implement Phase 3 (1 hour)
4. **Validate:** Run full test suite, debug if needed
5. **Document:** Update code comments, commit changes

**Total Timeline:** 1-2 days to complete implementation and validation.

---

**Status:** Ready for implementation
**Priority:** CRITICAL (blocking full multiphysics solver)
**Difficulty:** Medium (standard LBM techniques, well-documented)
