# VOF Higher-Order Advection Architectural Design
## TVD Scheme Implementation for Rayleigh-Taylor Simulations

**Date:** 2026-01-19
**Author:** Chief LBM Architect
**Status:** Design Review - Pending Implementation
**Priority:** Medium (Current subcycling solution working)

---

## Executive Summary

### Current State
- **First-order upwind advection** with CFL-adaptive subcycling
- **Mass loss:** Reduced from 20.75% to <7% via subcycling (CFL_target=0.10)
- **Performance cost:** +20% simulation time due to aggressive subcycling
- **Interface quality:** 3-4 cells thick, acceptable for RT but suboptimal for surface tension

### Proposed Upgrade
- **TVD (Total Variation Diminishing) scheme** for 2nd-order spatial accuracy
- **Expected mass loss:** <2% without subcycling (10× better than current)
- **Performance:** Similar or better than current (fewer substeps needed)
- **Interface quality:** 2-3 cells thick, improved for all physics

### Strategic Decision
**Recommendation:** Proceed with TVD implementation as **Phase 2 enhancement** after completing M1-M3 validation milestones. Current subcycling solution is adequate for RT validation.

---

## Problem Analysis

### Why First-Order Upwind Fails

**Numerical diffusion analysis:**

```
Upwind truncation error:
∂f/∂t + u·∇f = -u·dx/2·(1 - 2·CFL)·∂²f/∂x² + O(dx²)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
                  Numerical diffusion term
```

**Mass loss mechanism:**
1. Interface smears by ~1 cell per step at CFL=0.15
2. Over 5000 steps: interface spreads from 2 → 100+ cells
3. Clipping to [0,1] and truncation errors accumulate as mass loss
4. Cannot be fully eliminated by subcycling (only mitigated)

**Current workaround (subcycling to CFL=0.10):**
- Reduces diffusion but doesn't eliminate it
- Requires 2-5× substeps → +20% computational cost
- Still loses ~7% mass in RT test
- Interface still 3-4 cells thick

---

## TVD Scheme Theory

### What is TVD?

**Total Variation Diminishing** schemes ensure:
```
TV(f^{n+1}) ≤ TV(f^n)

where TV(f) = Σ|f_{i+1} - f_i|  (measure of interface sharpness)
```

**Key property:** Prevents spurious oscillations while maintaining 2nd-order accuracy in smooth regions.

### Flux Limiter Formulation

**General TVD flux:**
```
F_{i+1/2} = u·[f_i + 0.5·φ(r)·(f_i - f_{i-1})]  for u > 0
          = u·[f_{i+1} - 0.5·φ(r)·(f_{i+2} - f_{i+1})]  for u < 0

where:
  r = (f_{i+1} - f_i) / (f_i - f_{i-1} + ε)  (smoothness indicator)
  φ(r) = limiter function
  ε = 10^-10 (division by zero protection)
```

**Limiter functions (in TVD region):**

| Limiter | Formula | Properties |
|---------|---------|-----------|
| **minmod** | φ(r) = max(0, min(1, r)) | Most diffusive, most robust |
| **van Leer** | φ(r) = (r + \|r\|)/(1 + \|r\|) | Smooth, good for VOF |
| **superbee** | φ(r) = max(0, min(2r, 1), min(r, 2)) | Least diffusive, sharp interfaces |
| **MC** | φ(r) = max(0, min(2r, 0.5(1+r), 2)) | Monotonized central difference |

**Recommendation for VOF:** Start with **van Leer** (smooth, well-tested), provide **superbee** option for sharpest interfaces.

### Why TVD for VOF?

**Advantages over first-order upwind:**
1. **2nd-order accuracy** in smooth regions → O(dx²) vs O(dx)
2. **Monotonicity preserving** → no new extrema, bounds-preserving [0,1]
3. **Reduced diffusion** → 10× better mass conservation
4. **Sharper interfaces** → 2-3 cells vs 3-4 cells
5. **Higher CFL allowed** → CFL ≤ 0.5 without aggressive subcycling

**Comparison to Olsson-Kreiss compression:**
- TVD: Prevents diffusion upfront (2nd-order advection)
- Olsson-Kreiss: Fixes diffusion afterward (compression step)
- **Best practice:** TVD advection + optional light compression (C=0.1)

---

## Architectural Design

### Design Principle 1: Minimize Code Disruption

**Philosophy:** TVD should integrate seamlessly into existing VOFSolver architecture without breaking current functionality.

**Implementation strategy:**
- Extend existing `advectFillLevelUpwindKernel` → create `advectFillLevelTVDKernel`
- Add TVD as optional mode, keep upwind as fallback
- Reuse existing buffer management and boundary handling
- Maintain backward compatibility for all tests

### Design Principle 2: GPU Performance by Design

**Memory access pattern (critical for coalescing):**
- Load 4 neighbors per direction (im2, im, i, ip, ip2) → **structured access**
- Compute limiter for x, y, z fluxes independently → **no dependencies**
- Write single output per thread → **coalesced writes**

**Register pressure:**
- Limiter computation: ~10 registers per direction × 3 = 30 registers
- Neighbor loads: 4 per direction × 3 = 12 loads
- Target: <32 registers per thread for 100% occupancy

**Shared memory strategy:**
- Option A: Load 8×8×8 block + 2-cell halo into shared memory
- Option B: Direct global memory access (simpler, may be faster with L1 cache)
- **Recommendation:** Start with Option B, profile, then try Option A if needed

### Design Principle 3: Modularity and Extensibility

**Limiter as template parameter (compile-time selection):**
```cpp
template<typename LimiterFunc>
__global__ void advectFillLevelTVDKernel(...) {
    float phi = LimiterFunc::compute(r);
    // Use phi in flux computation
}

// Limiter functors
struct VanLeerLimiter {
    __device__ static float compute(float r) {
        return (r + fabsf(r)) / (1.0f + fabsf(r));
    }
};

struct SuperbeeLimiter {
    __device__ static float compute(float r) {
        return fmaxf(0.0f, fmaxf(fminf(2.0f*r, 1.0f), fminf(r, 2.0f)));
    }
};
```

**Why template over enum:**
- Compile-time dispatch → zero overhead
- Limiter inlined → optimal register usage
- Easy to add new limiters → just define new functor
- Clean separation of limiter logic from flux computation

---

## Detailed Implementation Plan

### Architecture Option A: Separate TVD Kernel (Recommended)

**Pros:**
- Clean separation of 1st-order and 2nd-order code
- Easy to A/B test performance
- No risk to existing upwind implementation
- Can optimize each kernel independently

**Cons:**
- Some code duplication (boundary handling, flux assembly)
- Slightly larger binary size

**Verdict:** Choose this for robustness and maintainability.

### Architecture Option B: Unified Kernel with Template

**Pros:**
- Single code path for maintenance
- Shared boundary handling logic

**Cons:**
- More complex kernel code
- Harder to debug if issues arise
- Template instantiation increases compile time

**Verdict:** Defer to Phase 3 refactoring if needed.

### Memory Layout for TVD

**Neighbor stencil (per direction):**
```
X-direction: [i-2][i-1][i][i+1][i+2]  (5-point for 2nd-order upwind)
Y-direction: [j-2][j-1][j][j+1][j+2]
Z-direction: [k-2][k-1][k][k+1][k+2]
```

**Index computation (critical for performance):**
```cpp
// X-direction neighbors
int im2 = (i >= 2) ? i - 2 : (bc_x == 0) ? nx - 2 + i : i;
int im1 = (i >= 1) ? i - 1 : (bc_x == 0) ? nx - 1 : i;
int ip1 = (i < nx - 1) ? i + 1 : (bc_x == 0) ? 0 : i;
int ip2 = (i < nx - 2) ? i + 2 : (bc_x == 0) ? i + 2 - nx : i;

// Global indices
int idx_im2 = im2 + nx * (j + ny * k);
int idx_im1 = im1 + nx * (j + ny * k);
int idx = i + nx * (j + ny * k);
int idx_ip1 = ip1 + nx * (j + ny * k);
int idx_ip2 = ip2 + nx * (j + ny * k);
```

**Memory coalescing:**
- Threads in same warp access consecutive i indices → coalesced loads
- Y and Z neighbors have strided access → acceptable with L1 cache
- Total memory traffic: 15 loads per cell (vs 7 for upwind) → 2× bandwidth

### Boundary Condition Handling

**Challenge:** TVD requires 2 upstream neighbors, but walls only have 1.

**Solution 1: Fallback to 1st-order at boundaries (recommended)**
```cpp
// At boundary cells, use upwind instead of TVD
if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2 || k < 2 || k >= nz - 2) {
    // Fall back to first-order upwind
    float flux_xp = (u_face_xp >= 0.0f) ? u_face_xp * fill_level[idx]
                                        : u_face_xp * fill_level[idx_ip1];
} else {
    // Use TVD flux
    float r = (fill_level[idx_ip1] - fill_level[idx]) /
              (fill_level[idx] - fill_level[idx_im1] + 1e-10f);
    float phi = VanLeerLimiter::compute(r);
    float flux_xp = u_face_xp * (fill_level[idx] + 0.5f * phi * (fill_level[idx] - fill_level[idx_im1]));
}
```

**Solution 2: Extrapolation for phantom cells**
```cpp
// At i=0, extrapolate f[-1] = 2*f[0] - f[1]
if (i == 0 && bc_x == 1) {  // Wall boundary
    f_im1 = 2.0f * fill_level[idx] - fill_level[idx_ip1];
}
```

**Recommendation:** Use Solution 1 (fallback) initially. Boundary cells are <1% of domain, minimal impact on global accuracy.

### Flux Computation (Conservative Form)

**X-direction flux at face i+1/2:**
```cpp
// Face velocity (interpolated)
float u_face_xp = 0.5f * (ux[idx] + ux[idx_ip1]);

// Upwind direction check
if (u_face_xp >= 0.0f) {
    // Flow from i → i+1, upwind is i
    float f_i = fill_level[idx];
    float f_im1 = fill_level[idx_im1];
    float f_im2 = fill_level[idx_im2];

    // Smoothness indicator
    float r = (f_i - f_im1) / (f_im1 - f_im2 + 1e-10f);

    // Limiter
    float phi = VanLeerLimiter::compute(r);

    // 2nd-order upwind reconstruction
    float f_face = f_i - 0.5f * phi * (f_i - f_im1);

    // Flux
    flux_xp = u_face_xp * f_face;
} else {
    // Flow from i+1 → i, upwind is i+1
    float f_ip1 = fill_level[idx_ip1];
    float f_ip2 = fill_level[idx_ip2];
    float f_i = fill_level[idx];

    float r = (f_ip1 - f_ip2) / (f_i - f_ip1 + 1e-10f);
    float phi = VanLeerLimiter::compute(r);
    float f_face = f_ip1 + 0.5f * phi * (f_ip1 - f_ip2);

    flux_xp = u_face_xp * f_face;
}
```

**Sign convention note:** Carefully check flux direction vs gradient sign to ensure upwinding is correct!

### Conservative Update (Same as Upwind)

```cpp
// Flux divergence
float div_flux = (flux_xp - flux_xm + flux_yp - flux_ym + flux_zp - flux_zm) / dx;

// Update
float f_new = fill_level[idx] - dt * div_flux;

// Clamp to [0, 1]
fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
```

**Boundedness guarantee:** TVD limiters ensure f_new ∈ [0, 1] if f^n ∈ [0, 1] and CFL ≤ 1.

---

## Interface Changes to VOFSolver Class

### Minimal API Changes (Backward Compatible)

**Option 1: Add advection scheme parameter to constructor**
```cpp
enum class AdvectionScheme {
    UPWIND_1ST = 0,      // Current implementation
    TVD_VAN_LEER = 1,    // TVD with van Leer limiter
    TVD_SUPERBEE = 2,    // TVD with superbee limiter
    TVD_MINMOD = 3       // TVD with minmod limiter
};

class VOFSolver {
public:
    VOFSolver(int nx, int ny, int nz, float dx = 1.0f,
              BoundaryType bc_x = BoundaryType::PERIODIC,
              BoundaryType bc_y = BoundaryType::PERIODIC,
              BoundaryType bc_z = BoundaryType::PERIODIC,
              AdvectionScheme scheme = AdvectionScheme::UPWIND_1ST);  // Default: current behavior

private:
    AdvectionScheme advection_scheme_;
};
```

**Option 2: Runtime configuration setter (more flexible)**
```cpp
class VOFSolver {
public:
    // Keep existing constructor
    VOFSolver(int nx, int ny, int nz, float dx = 1.0f, ...);

    // Add scheme selection
    void setAdvectionScheme(AdvectionScheme scheme) {
        advection_scheme_ = scheme;
    }

    AdvectionScheme getAdvectionScheme() const {
        return advection_scheme_;
    }
};
```

**Recommendation:** Use Option 2 for maximum flexibility. Tests can switch schemes without recreating solver.

### Internal Dispatch in advectFillLevel()

```cpp
void VOFSolver::advectFillLevel(const float* velocity_x,
                                 const float* velocity_y,
                                 const float* velocity_z,
                                 float dt) {
    // ... CFL computation and subcycling logic (unchanged) ...

    for (int substep = 0; substep < n_substeps; ++substep) {
        // Dispatch based on selected scheme
        switch (advection_scheme_) {
            case AdvectionScheme::UPWIND_1ST:
                advectFillLevelUpwindKernel<<<gridSize, blockSize>>>(
                    d_fill_level_, d_fill_level_tmp_,
                    velocity_x, velocity_y, velocity_z,
                    dt_sub, dx_, nx_, ny_, nz_,
                    static_cast<int>(bc_x_), static_cast<int>(bc_y_), static_cast<int>(bc_z_));
                break;

            case AdvectionScheme::TVD_VAN_LEER:
                advectFillLevelTVDKernel<VanLeerLimiter><<<gridSize, blockSize>>>(
                    d_fill_level_, d_fill_level_tmp_,
                    velocity_x, velocity_y, velocity_z,
                    dt_sub, dx_, nx_, ny_, nz_,
                    static_cast<int>(bc_x_), static_cast<int>(bc_y_), static_cast<int>(bc_z_));
                break;

            case AdvectionScheme::TVD_SUPERBEE:
                advectFillLevelTVDKernel<SuperbeeLimiter><<<gridSize, blockSize>>>(...);
                break;

            // ... other limiters ...
        }
        CUDA_CHECK_KERNEL();

        // Copy result back (unchanged)
        cudaMemcpy(d_fill_level_, d_fill_level_tmp_, num_cells_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    // ... interface compression (unchanged) ...
}
```

**Compile-time optimization:** Switch statement on enum dispatches to correct template instantiation, zero runtime overhead.

---

## Performance Analysis

### Memory Bandwidth Comparison

**First-order upwind:**
- Loads: 7 neighbors (im, ip, jm, jp, km, kp, center) × 4 bytes = 28 bytes/cell
- Stores: 1 output × 4 bytes = 4 bytes/cell
- **Total:** 32 bytes/cell

**TVD (van Leer):**
- Loads: 15 neighbors (im2, im, i, ip, ip2 per direction × 3) × 4 bytes = 60 bytes/cell
- Stores: 1 output × 4 bytes = 4 bytes/cell
- **Total:** 64 bytes/cell

**Bandwidth ratio:** 64/32 = **2× memory traffic**

**BUT:** With CFL_target=0.10 subcycling (current solution), effective cost:
- Upwind with 2-5× substeps → 32 × 3 = 96 bytes/cell (average)
- TVD with 1-2× substeps → 64 × 1.5 = 96 bytes/cell (average)

**Net result:** TVD is **same or better** performance due to fewer substeps needed.

### Computational Intensity

**Upwind:**
- ~20 FLOPs per cell (flux assembly, divergence)

**TVD:**
- ~60 FLOPs per cell (limiter computation, 2nd-order reconstruction)

**Ratio:** 3× more compute per cell

**BUT:** Modern GPUs are memory-bound, not compute-bound. Extra FLOPs hidden by memory latency.

**Occupancy:**
- Upwind: ~20 registers → 100% occupancy
- TVD: ~35 registers → 75% occupancy (acceptable)

### Expected Speedup

**Current solution (upwind + subcycling):**
- CFL_target = 0.10 → 2-5× substeps
- VOF advection: 10% of total time
- Overhead: 10% × (3-1) = +20% simulation time

**TVD solution:**
- CFL_target = 0.25 (or even 0.50) → 1-2× substeps
- Each substep 2× memory + 3× compute → ~2.5× cost per substep
- But 2-3× fewer substeps → net 2.5/2.5 = **1.0× (same cost)**
- Better mass conservation → can disable compression (C=0) → save 25%
- **Net speedup: 10-15% faster than current**

---

## Integration with Existing CFL-Adaptive Subcycling

**Good news:** TVD integrates seamlessly with existing subcycling infrastructure!

**Updated CFL target:**
```cpp
// Current (first-order upwind)
const float CFL_target = 0.10f;  // Very conservative

// With TVD (can relax)
const float CFL_target = 0.25f;  // Standard for 2nd-order schemes
// Or even
const float CFL_target = 0.40f;  // Aggressive but stable
```

**Subcycling behavior with TVD:**
```
Time    v_max   CFL     CFL_target   n_substeps   CFL_sub   Comment
0.0s    0.0     0.00    0.25         1            0.00      No subcycling
0.5s    3.5     0.18    0.25         1            0.18      No subcycling
0.8s    6.0     0.31    0.25         2            0.16      Light subcycling
1.0s    8.5     0.44    0.25         2            0.22      2× substeps
```

**Comparison to current (upwind + CFL=0.10):**
- Current: 2-5× substeps throughout most of simulation
- TVD: 1-2× substeps, only at high CFL
- **TVD subcycling cost: 50% less than current**

---

## Testing and Validation Strategy

### Phase 1: Unit Tests (1 day)

**Test 1: Zalesak's disk rotation**
- Standard VOF benchmark
- Exact solution after 1 full rotation (2π)
- Success: Mass loss < 1%, shape preservation > 95%

**Test 2: Diagonal translation**
- Worst-case for dimensional splitting
- TVD should show no grid alignment
- Success: < 2% mass loss over 100 steps

**Test 3: Interface sharpness**
- Measure interface thickness over time
- Success: Width < 3 cells after 1000 steps (vs 5-6 for upwind)

**Test 4: Limiter comparison**
- Run same test with all 3 limiters (van Leer, superbee, minmod)
- Document trade-off: superbee (sharpest) vs minmod (most robust)

### Phase 2: Integration Tests (2 days)

**Test 5: Rayleigh-Taylor mushroom**
- Current benchmark case
- Success: Mass loss < 2% (vs current 7%)
- Success: h₁ = 1.37 m ± 0.05 m (physics unchanged)
- Success: Simulation time ≤ current (no slowdown)

**Test 6: Oscillating droplet (M2 benchmark)**
- Surface tension dominated
- TVD should reduce spurious currents
- Success: Oscillation amplitude correct, < 2% mass loss

**Test 7: Rising bubble (M1 benchmark)**
- Buoyancy + surface tension
- Success: Terminal velocity correct, mass conserved

### Phase 3: Performance Profiling (1 day)

**Profile metrics:**
1. Kernel execution time (advect vs upwind)
2. Memory bandwidth utilization (should be 80-90%)
3. Occupancy (should be >75%)
4. Total simulation time vs baseline

**Tools:**
- `nvprof --metrics gld_throughput,gst_throughput`
- `nsys profile --stats=true`

**Success criteria:**
- TVD kernel < 3× upwind kernel time
- Overall simulation time ≤ current (due to fewer substeps)

---

## Risk Assessment and Mitigation

### Risk 1: Implementation Complexity (Medium)

**Risk:** TVD is more complex than upwind, higher chance of bugs.

**Mitigation:**
- Implement van Leer limiter first (well-tested, smooth)
- Extensive unit testing on canonical cases (Zalesak, translation)
- Compare against published results for validation
- Keep upwind as fallback if TVD fails

**Likelihood:** Medium
**Impact:** Low (can fall back to upwind)

### Risk 2: Performance Regression (Low)

**Risk:** TVD slower than expected, negates mass conservation benefit.

**Mitigation:**
- Profile early and optimize memory access patterns
- Use shared memory if global memory too slow
- Can always use TVD with higher CFL_target to reduce substeps
- Worst case: TVD + subcycling still better than upwind + aggressive subcycling

**Likelihood:** Low
**Impact:** Medium

### Risk 3: Numerical Instability (Low)

**Risk:** TVD limiters fail in corner cases (extreme gradients, etc).

**Mitigation:**
- Van Leer limiter is robust (smooth, monotone)
- Add fallback to upwind at problematic cells (if f < 0.01 or f > 0.99)
- Clamp limiter output: phi ∈ [0, 2]
- Extensive testing on pathological cases

**Likelihood:** Low
**Impact:** Medium

### Risk 4: Breaking Existing Tests (Low)

**Risk:** TVD changes behavior of existing simulations.

**Mitigation:**
- Default to UPWIND_1ST for backward compatibility
- Existing tests pass unchanged
- New TVD tests added separately
- Documentation clearly states when to use each scheme

**Likelihood:** Very Low
**Impact:** Low

---

## Implementation Timeline

### Phase 1: Core TVD Kernel (3-4 days)

**Tasks:**
1. Define limiter functors (VanLeerLimiter, SuperbeeLimiter, MinmodLimiter)
2. Implement `advectFillLevelTVDKernel` with template limiter
3. Handle boundary conditions (fallback to 1st-order)
4. Add AdvectionScheme enum and dispatch logic
5. Unit test: Zalesak disk, diagonal translation

**Deliverable:** Working TVD kernel passing unit tests

### Phase 2: Integration and Validation (2-3 days)

**Tasks:**
1. Integrate into VOFSolver::advectFillLevel()
2. Add scheme selection API (setAdvectionScheme)
3. Run RT mushroom test with TVD
4. Run oscillating droplet test with TVD
5. Compare mass conservation and performance

**Deliverable:** TVD validated on RT and M2 benchmarks

### Phase 3: Optimization and Documentation (1-2 days)

**Tasks:**
1. Profile kernel performance (nvprof, nsys)
2. Optimize memory access if needed (shared memory)
3. Tune CFL_target for TVD (test 0.25, 0.40)
4. Document usage in VOFSolver API
5. Update test expectations and tolerances

**Deliverable:** Optimized TVD ready for production

### Total Effort: 6-9 days

**Risk buffer:** +2 days for unexpected issues
**Realistic timeline:** 8-11 days (1.5-2 weeks)

---

## Configuration Recommendations

### When to Use Each Scheme

**Use first-order upwind (current) when:**
- Rapid prototyping (fastest implementation)
- Extremely diffusive flow (high CFL, low resolution)
- Debugging (simplest to understand)

**Use TVD (van Leer) when:**
- Mass conservation critical (< 2% error required)
- Interface quality matters (surface tension, droplets)
- Moderate CFL (0.2-0.4)
- Production simulations

**Use TVD (superbee) when:**
- Sharpest possible interface needed
- Willing to accept slight oscillations near discontinuities
- High resolution (interface > 10 cells wide)

**Use TVD (minmod) when:**
- Maximum robustness required (guaranteed monotone)
- Low resolution (interface ~ 3 cells wide)
- Extreme Atwood numbers (At > 0.9)

### Recommended CFL Targets

| Scheme | CFL_target | Subcycling Frequency | Mass Loss (RT) |
|--------|-----------|---------------------|----------------|
| Upwind | 0.10 | Heavy (2-5×) | ~7% |
| Upwind | 0.25 | Light (1-2×) | ~20% |
| TVD van Leer | 0.25 | Light (1-2×) | ~2% |
| TVD van Leer | 0.40 | Minimal (1×) | ~3-4% |
| TVD superbee | 0.25 | Light (1-2×) | ~1% |

**Production recommendation:** TVD van Leer with CFL_target=0.25

---

## Code Organization

### File Structure

**New files:**
```
include/physics/vof_limiters.h          // Limiter functor definitions
src/physics/vof/vof_advection_tvd.cu    // TVD kernel implementation
tests/unit/vof/test_vof_tvd.cu          // TVD unit tests
tests/validation/vof/test_vof_tvd_comparison.cu  // Upwind vs TVD comparison
```

**Modified files:**
```
include/physics/vof_solver.h            // Add AdvectionScheme enum, setter
src/physics/vof/vof_solver.cu           // Add scheme dispatch in advectFillLevel()
```

**Documentation:**
```
docs/VOF_TVD_IMPLEMENTATION.md          // Implementation details
docs/VOF_TVD_USER_GUIDE.md              // Usage guide for end users
docs/VOF_TVD_BENCHMARK_RESULTS.md       // Performance comparison data
```

### Header: vof_limiters.h

```cpp
#pragma once

namespace lbm {
namespace physics {
namespace limiters {

/**
 * @brief Van Leer flux limiter (smooth, well-balanced)
 *
 * Formula: φ(r) = (r + |r|) / (1 + |r|)
 *
 * Properties:
 * - 2nd-order accurate in smooth regions
 * - TVD (total variation diminishing)
 * - Smooth (no kinks)
 * - Good for general-purpose VOF
 */
struct VanLeerLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;  // Upwind fallback
        return (r + fabsf(r)) / (1.0f + fabsf(r));
    }
};

/**
 * @brief Superbee flux limiter (least diffusive, sharpest interfaces)
 *
 * Formula: φ(r) = max(0, min(2r, 1), min(r, 2))
 *
 * Properties:
 * - Least diffusive TVD limiter
 * - Sharpest interface preservation
 * - May produce slight oscillations near discontinuities
 * - Best for high-resolution simulations
 */
struct SuperbeeLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;
        float term1 = fminf(2.0f * r, 1.0f);
        float term2 = fminf(r, 2.0f);
        return fmaxf(term1, term2);
    }
};

/**
 * @brief Minmod flux limiter (most diffusive, most robust)
 *
 * Formula: φ(r) = max(0, min(1, r))
 *
 * Properties:
 * - Most robust (guaranteed monotone)
 * - Most diffusive (but still better than upwind)
 * - Best for low-resolution or extreme cases
 */
struct MinmodLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;
        return fminf(1.0f, r);
    }
};

/**
 * @brief MC (Monotonized Central) flux limiter (good balance)
 *
 * Formula: φ(r) = max(0, min(2r, 0.5(1+r), 2))
 *
 * Properties:
 * - Balance between superbee and van Leer
 * - Good for most applications
 */
struct MCLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;
        return fmaxf(0.0f, fminf(fminf(2.0f * r, 0.5f * (1.0f + r)), 2.0f));
    }
};

} // namespace limiters
} // namespace physics
} // namespace lbm
```

---

## Backward Compatibility Guarantee

**Promise:** All existing code continues to work unchanged after TVD implementation.

**How:**
1. Default scheme is UPWIND_1ST (current behavior)
2. Existing tests pass without modification
3. API extension only (no breaking changes)
4. Performance of upwind unchanged (separate kernel)

**Migration path:**
```cpp
// Old code (still works)
VOFSolver vof(nx, ny, nz, dx);
vof.advectFillLevel(ux, uy, uz, dt);  // Uses upwind (default)

// New code (opt-in to TVD)
VOFSolver vof(nx, ny, nz, dx);
vof.setAdvectionScheme(VOFSolver::AdvectionScheme::TVD_VAN_LEER);
vof.advectFillLevel(ux, uy, uz, dt);  // Uses TVD van Leer
```

---

## Future Enhancements (Phase 4+)

### WENO (Weighted Essentially Non-Oscillatory) Schemes

**What:** 5th-order accurate reconstruction with adaptive stencil selection.

**Advantages:**
- Even less diffusion than TVD
- Better for turbulent flows
- Captures thin filaments

**Challenges:**
- 7-point stencil (vs 5 for TVD)
- More complex implementation
- Higher memory bandwidth

**Timeline:** Phase 4 (after TVD proven successful)

### Adaptive Mesh Refinement (AMR) Integration

**What:** Refine grid near interface, coarsen in bulk regions.

**Synergy with TVD:**
- TVD provides high accuracy on refined regions
- Less subcycling needed with local refinement

**Timeline:** Long-term (requires major architecture changes)

### PLIC (Piecewise Linear Interface Calculation)

**What:** Geometric VOF reconstruction for exact interface tracking.

**Synergy with TVD:**
- TVD advects PLIC-reconstructed interface
- Best of both worlds: geometric accuracy + conservative transport

**Timeline:** Phase 5 (research-grade implementation)

---

## Conclusion and Recommendation

### Summary of Design

1. **Separate TVD kernel** with template limiter dispatch
2. **Minimal API changes** (setAdvectionScheme() setter)
3. **Backward compatible** (default to upwind)
4. **Integrated with subcycling** (use higher CFL_target)
5. **Limiter choice:** Start with van Leer, add superbee/minmod as options

### Expected Outcomes

| Metric | Current (Upwind + Subcycling) | TVD (van Leer) | Improvement |
|--------|-------------------------------|----------------|-------------|
| Mass loss (RT) | ~7% | ~2% | **3.5× better** |
| Interface width | 3-4 cells | 2-3 cells | **1.5× sharper** |
| Subcycling overhead | +20% | +5% | **3× less overhead** |
| Total simulation time | Baseline + 20% | Baseline + 5% | **12% faster** |
| Code complexity | Low | Medium | Acceptable |
| Implementation time | N/A | 8-11 days | Reasonable |

### Strategic Recommendation

**Proceed with TVD implementation as Phase 2 enhancement:**

**Rationale:**
1. Current subcycling solution is adequate for M1-M3 validation (7% mass loss acceptable)
2. TVD is a known, proven upgrade with clear benefits
3. Implementation risk is low (can fall back to upwind)
4. Performance improvement is moderate but measurable (12% faster)
5. Opens door for future enhancements (WENO, PLIC)

**Priority:** Medium (after M1-M3 validation complete)

**Confidence:** High (80%) - Standard technique with well-understood trade-offs

---

## References

### Flux Limiter Theory

1. **Sweby, P. K. (1984).** "High resolution schemes using flux limiters for hyperbolic conservation laws."
   *SIAM Journal on Numerical Analysis*, 21(5), 995-1011.
   - Original TVD theory and limiter design

2. **Van Leer, B. (1979).** "Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method."
   *Journal of Computational Physics*, 32(1), 101-136.
   - Van Leer limiter derivation

3. **Roe, P. L. (1986).** "Characteristic-based schemes for the Euler equations."
   *Annual Review of Fluid Mechanics*, 18, 337-365.
   - Superbee limiter and comparison

### VOF with Higher-Order Schemes

4. **Ubbink, O., & Issa, R. I. (1999).** "A method for capturing sharp fluid interfaces on arbitrary meshes."
   *Journal of Computational Physics*, 153(1), 26-50.
   - TVD schemes for VOF advection (OpenFOAM basis)

5. **Muzaferija, S., & Peric, M. (1999).** "Computation of free surface flows using interface-tracking and interface-capturing methods."
   *Nonlinear Water Wave Interaction*, Computational Mechanics Publications, 59-100.
   - PLIC + TVD combination

### Benchmarks

6. **Zalesak, S. T. (1979).** "Fully multidimensional flux-corrected transport algorithms for fluids."
   *Journal of Computational Physics*, 31(3), 335-362.
   - Zalesak disk (standard VOF benchmark)

7. **Rider, W. J., & Kothe, D. B. (1998).** "Reconstructing volume tracking."
   *Journal of Computational Physics*, 141(2), 112-152.
   - Interface reconstruction accuracy study

---

**Document Status:** Complete - Ready for Review
**Next Action:** Present to team, get approval, begin Phase 1 implementation
**Estimated Start:** After M1-M3 validation complete (ETA: TBD)

---

**Files:**
- Header: `/home/yzk/LBMProject/include/physics/vof_limiters.h` (to be created)
- Implementation: `/home/yzk/LBMProject/src/physics/vof/vof_advection_tvd.cu` (to be created)
- Interface: `/home/yzk/LBMProject/include/physics/vof_solver.h` (to be modified)
- Solver: `/home/yzk/LBMProject/src/physics/vof/vof_solver.cu` (to be modified)
