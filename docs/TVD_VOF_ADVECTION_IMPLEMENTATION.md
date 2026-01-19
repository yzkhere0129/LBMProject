# TVD VOF Advection Implementation Guide

**Date:** 2026-01-19
**Author:** CFD Development Team
**Status:** Implementation Complete

---

## Executive Summary

Implemented a **Total Variation Diminishing (TVD)** advection scheme for the VOF solver to fix the ~20% mass loss issue in Rayleigh-Taylor instability simulations. The TVD scheme is **second-order accurate** in smooth regions while maintaining **first-order stability** near discontinuities, significantly reducing numerical diffusion compared to first-order upwind.

### Key Results (Expected)
- **Mass conservation:** < 1% error (vs 20% with first-order upwind)
- **Interface sharpness:** 2-3 cells (vs 4-5 cells with upwind + subcycling)
- **Stability:** Same CFL limit (< 0.5) as first-order upwind
- **Performance:** ~20% overhead due to larger stencil (5-point vs 3-point)

---

## Problem Analysis

### Root Cause of Mass Loss
The first-order upwind scheme causes **numerical diffusion** proportional to:
```
ε_diffusion ≈ (1 - 2·CFL) + CFL²
```

For CFL = 0.1 (current subcycling target):
- Diffusion error: ~80% per timestep
- Over 10,000 timesteps: **20% cumulative mass loss**

### Why Subcycling Alone Is Insufficient
Even with aggressive subcycling (CFL = 0.1), first-order upwind remains **fundamentally diffusive**. The only way to eliminate this diffusion is to use a **higher-order scheme**.

---

## TVD Scheme Theory

### Conservative Flux Formulation
```
∂f/∂t + ∇·(uf) = 0

f^{n+1} = f^n - (dt/dx) × (F_{i+1/2} - F_{i-1/2})
```

### TVD Flux Formula
```
F_{i+1/2} = F_low + φ(r) × (F_high - F_low)
```

Where:
- **F_low:** First-order upwind flux (stable, diffusive)
- **F_high:** Second-order flux (accurate, oscillatory)
- **φ(r):** Flux limiter function (0 ≤ φ ≤ 2)
- **r:** Gradient ratio = (f_i - f_{i-1}) / (f_{i+1} - f_i)

### Gradient Ratio Interpretation
For velocity u > 0 (left-to-right flow):
```
r = (f_i - f_{i-1}) / (f_{i+1} - f_i + ε)
```

- **r ≈ 1:** Smooth linear variation → use 2nd-order (φ ≈ 1)
- **r ≈ 0 or r < 0:** Discontinuity → use 1st-order (φ ≈ 0)
- **r >> 1:** Sharp gradient change → limit flux (φ < 1)

### TVD Property
The TVD property ensures **no spurious oscillations**:
```
TV(f^{n+1}) ≤ TV(f^n)

where TV(f) = Σ |f_{i+1} - f_i|
```

This means:
1. No new maxima or minima are created
2. Monotonicity is preserved
3. Solution remains bounded without clipping

---

## Flux Limiter Functions

### 1. Minmod (Most Stable)
```
φ(r) = max(0, min(1, r))
```
- **Characteristics:** Most diffusive, most stable
- **Use case:** Initial testing, very sharp discontinuities
- **Accuracy:** ~1.5-order (between 1st and 2nd)

### 2. van Leer (Recommended)
```
φ(r) = (r + |r|) / (1 + |r|)
```
- **Characteristics:** Balanced accuracy and stability
- **Use case:** General-purpose, Rayleigh-Taylor simulations
- **Accuracy:** ~2nd-order in smooth regions

### 3. Superbee (Most Compressive)
```
φ(r) = max(0, min(2r, 1), min(r, 2))
```
- **Characteristics:** Least diffusive, sharpest interfaces
- **Use case:** Interface tracking, droplet simulations
- **Accuracy:** ~2nd-order, can overshoot slightly

### 4. MC (Monotonized Central)
```
φ(r) = max(0, min((1+r)/2, 2, 2r))
```
- **Characteristics:** Prevents overshoot, smooth profiles
- **Use case:** Smooth flows, thermal problems
- **Accuracy:** ~2nd-order with minimal overshoot

### Comparison Table

| Limiter   | Diffusion | Stability | Compression | Overshoot | Recommended Use |
|-----------|-----------|-----------|-------------|-----------|-----------------|
| Minmod    | High      | Excellent | Low         | None      | Testing, sharp shocks |
| van Leer  | Medium    | Excellent | Medium      | None      | **General purpose** |
| Superbee  | Low       | Good      | High        | Slight    | Interface tracking |
| MC        | Medium    | Excellent | Medium      | None      | Smooth flows |

---

## Implementation Details

### File Structure
```
/home/yzk/LBMProject/
├── include/physics/vof_solver.h         # Header with enum declarations
└── src/physics/vof/vof_solver.cu        # Implementation
    ├── Flux limiter functions (lines 21-75)
    ├── advectFillLevelTVDKernel (lines 297-640)
    └── Modified advectFillLevel (lines 1357-1379)
```

### Key Components

#### 1. Flux Limiter Functions (Device Functions)
```cuda
__device__ __forceinline__ float fluxLimiterVanLeer(float r) {
    if (r <= 0.0f) return 0.0f;
    return (r + fabsf(r)) / (1.0f + fabsf(r));
}
```
- Declared as `__device__ __forceinline__` for performance
- Optimized with early return for negative r
- Handles division by zero with epsilon (1e-10)

#### 2. TVD Kernel Stencil
```
5-point stencil per direction:
[imm] [im] [i] [ip] [ipp]

Required for gradient ratio computation:
- Upwind-upwind: f_{i-1} - f_{i-2}
- Upwind-center: f_i - f_{i-1}
```

#### 3. Boundary Condition Handling
- **Periodic:** Wrap-around indexing
- **Wall:** Extrapolate with same index (zero-gradient)
- **Zero-flux:** Enforced at wall boundaries

#### 4. Scheme Selection API
```cpp
// Set advection scheme
vof_solver.setAdvectionScheme(VOFAdvectionScheme::TVD);

// Set flux limiter
vof_solver.setTVDLimiter(TVDLimiter::VAN_LEER);
```

---

## Usage Examples

### Example 1: Rayleigh-Taylor Instability
```cpp
// Create VOF solver with wall boundaries
VOFSolver vof_solver(nx, ny, nz, dx,
                     VOFSolver::BoundaryType::PERIODIC,  // x
                     VOFSolver::BoundaryType::PERIODIC,  // y
                     VOFSolver::BoundaryType::WALL);     // z (top/bottom)

// Enable TVD scheme with van Leer limiter
vof_solver.setAdvectionScheme(VOFAdvectionScheme::TVD);
vof_solver.setTVDLimiter(TVDLimiter::VAN_LEER);

// Run simulation
for (int step = 0; step < n_steps; ++step) {
    vof_solver.advectFillLevel(ux, uy, uz, dt);
    // ... other physics
}
```

### Example 2: Droplet Oscillation (Sharp Interface)
```cpp
// Use Superbee for sharpest interface
vof_solver.setAdvectionScheme(VOFAdvectionScheme::TVD);
vof_solver.setTVDLimiter(TVDLimiter::SUPERBEE);
```

### Example 3: Conservative First-Order (Baseline)
```cpp
// Keep first-order upwind for comparison
vof_solver.setAdvectionScheme(VOFAdvectionScheme::UPWIND);
```

---

## Performance Characteristics

### Computational Cost

| Component | First-Order Upwind | TVD Scheme | Overhead |
|-----------|-------------------|------------|----------|
| Stencil size | 3-point | 5-point | +67% reads |
| Flux computation | 6 ops/face | ~15 ops/face | +150% |
| Overall kernel | 1.0× | ~1.2× | +20% |

### Memory Access Pattern
```
Upwind:  [im] [i] [ip]              → 3 reads per direction
TVD:     [imm] [im] [i] [ip] [ipp]  → 5 reads per direction
```

### Register Pressure
- **Upwind:** ~20 registers
- **TVD:** ~30 registers (more intermediate values)
- **Impact:** Negligible (well below 64 register limit)

---

## CFL Condition and Stability

### Theoretical CFL Limit
```
CFL = |u| × dt/dx < 0.5  (same as first-order upwind)
```

### Practical CFL Recommendations

| Application | Recommended CFL | Subcycling | Comments |
|------------|----------------|------------|----------|
| RT instability | 0.25 | Yes (2-4×) | Balance accuracy and speed |
| Droplet sim | 0.10 | Yes (5-10×) | Sharp interface, high accuracy |
| Smooth flow | 0.40 | Minimal | TVD handles smooth regions well |

### CFL-Adaptive Subcycling
The existing subcycling mechanism (lines 1315-1379) works seamlessly with TVD:
```cpp
int n_substeps = max(1, ceil(CFL_current / CFL_target));
float dt_sub = dt / n_substeps;
```

---

## Validation Tests

### Test 1: Mass Conservation (1D Advection)
```
Initial: Square wave, f=1 in [0.4, 0.6], f=0 elsewhere
Velocity: u=1.0 m/s (constant)
Domain: 128 cells, periodic BC
Time: 10 advection cycles

Expected Results:
- First-order upwind: 15-20% mass loss
- TVD (van Leer):     < 1% mass loss
- TVD (superbee):     < 0.1% mass loss
```

### Test 2: Interface Sharpness (RT Instability)
```
Domain: 64×64×128, wall BC (top/bottom)
Physics: ρ_heavy=1000, ρ_light=100, g=-9.8
Time: 0.3 s (bubble formation)

Expected Results:
- Upwind: Interface thickness ~5 cells
- TVD (van Leer): Interface thickness ~2-3 cells
- TVD (superbee): Interface thickness ~2 cells
```

### Test 3: Numerical Oscillations (Step Function)
```
Initial: Sharp step, f=1 (x<0.5), f=0 (x>0.5)
Velocity: u=1.0 m/s
Check: max(f) should remain ≤ 1.0 (no overshoot)

Expected Results:
- Upwind: No overshoot (but diffuses to f_max~0.8)
- TVD (minmod): No overshoot (f_max~0.9)
- TVD (van Leer): No overshoot (f_max~0.95)
- TVD (superbee): Minimal overshoot (f_max~1.02, acceptable)
```

---

## Troubleshooting

### Issue 1: Excessive Diffusion (TVD Not Working)
**Symptoms:** Mass loss similar to first-order upwind

**Diagnosis:**
```cpp
// Check if TVD is actually enabled
printf("Scheme: %d\n", static_cast<int>(vof_solver.getAdvectionScheme()));
// Should print 1 (TVD), not 0 (UPWIND)
```

**Solution:**
```cpp
vof_solver.setAdvectionScheme(VOFAdvectionScheme::TVD);
```

### Issue 2: Numerical Oscillations
**Symptoms:** Fill level exceeds [0, 1], spurious ripples

**Possible Causes:**
1. CFL condition violated (CFL > 0.5)
2. Superbee limiter too aggressive for this flow

**Solutions:**
```cpp
// 1. Reduce CFL target
const float CFL_target = 0.10f;  // More conservative

// 2. Switch to van Leer or minmod
vof_solver.setTVDLimiter(TVDLimiter::VAN_LEER);
```

### Issue 3: Performance Degradation
**Symptoms:** Simulation 50%+ slower with TVD

**Diagnosis:**
- Expected overhead: ~20%
- If > 30%: Check GPU occupancy with `nvprof`

**Optimization:**
```bash
# Profile kernel
nvprof --metrics achieved_occupancy ./test_vof_advection

# Target: > 50% occupancy
# If lower: Consider reducing register usage
```

---

## Future Enhancements

### Short-Term (Next Release)
1. **WENO5 Scheme:** 5th-order accuracy in smooth regions
2. **Adaptive Limiter:** Automatically select limiter based on local flow
3. **Performance Tuning:** Optimize memory access pattern

### Medium-Term
1. **PLIC Reconstruction:** Combine TVD with piecewise linear interface
2. **Multi-Material:** Extend to 3+ phases
3. **GPU Optimization:** Shared memory for stencil data

### Long-Term
1. **Adaptive Mesh Refinement:** Higher resolution at interfaces
2. **Implicit TVD:** Remove CFL restriction
3. **Machine Learning:** Learn optimal limiter function from data

---

## References

### Foundational Papers
1. **Sweby, P.K. (1984).** "High resolution schemes using flux limiters for hyperbolic conservation laws." *SIAM Journal on Numerical Analysis*, 21(5), 995-1011.
   - Defines TVD property and flux limiter theory

2. **Harten, A. (1983).** "High resolution schemes for hyperbolic conservation laws." *Journal of Computational Physics*, 49(3), 357-393.
   - Original TVD scheme formulation

3. **Hirt, C.W., & Nichols, B.D. (1981).** "Volume of fluid (VOF) method for the dynamics of free boundaries." *Journal of Computational Physics*, 39(1), 201-225.
   - VOF method foundation

### Implementation References
4. **LeVeque, R.J. (2002).** *Finite Volume Methods for Hyperbolic Problems.* Cambridge University Press.
   - Chapter 6: TVD methods

5. **Toro, E.F. (2009).** *Riemann Solvers and Numerical Methods for Fluid Dynamics.* Springer.
   - Chapter 13: High-resolution methods

### VOF-Specific
6. **Rider, W.J., & Kothe, D.B. (1998).** "Reconstructing volume tracking." *Journal of Computational Physics*, 141(2), 112-152.
   - VOF reconstruction methods

7. **Ubbink, O., & Issa, R.I. (1999).** "A method for capturing sharp fluid interfaces on arbitrary meshes." *Journal of Computational Physics*, 153(1), 26-50.
   - CICSAM scheme (another TVD variant for VOF)

---

## Appendix A: Mathematical Derivation

### TVD Flux Construction
Starting from the conservative form:
```
∂f/∂t + ∂(uf)/∂x = 0
```

Integrate over control volume [x_{i-1/2}, x_{i+1/2}]:
```
f_i^{n+1} = f_i^n - (dt/dx) × (F_{i+1/2} - F_{i-1/2})
```

First-order upwind flux (stable, diffusive):
```
F_{i+1/2}^{low} = u_{i+1/2} × f_i    (if u > 0)
```

Second-order flux (accurate, oscillatory):
```
F_{i+1/2}^{high} = u_{i+1/2} × f_{i+1/2}^*
where f_{i+1/2}^* = f_i + 0.5 × φ(r) × (f_{i+1} - f_i)
```

Gradient ratio:
```
r = (f_i - f_{i-1}) / (f_{i+1} - f_i + ε)
```

Flux limiter φ(r) chosen from TVD region:
```
0 ≤ φ(r) ≤ min(2, 2r)    (Sweby diagram)
```

Final TVD flux:
```
F_{i+1/2} = F^{low} + φ(r) × (F^{high} - F^{low})
          = u × [f_i + 0.5 × φ(r) × (f_{i+1} - f_i)]
```

---

## Appendix B: Code Snippets

### Complete Example: RT Simulation with TVD
```cpp
#include "physics/vof_solver.h"
#include "physics/fluid_lbm.h"

int main() {
    // Domain setup
    const int nx = 64, ny = 64, nz = 128;
    const float dx = 2e-3f;  // 2 mm
    const float dt = 1e-5f;  // 10 µs

    // Create VOF solver
    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::PERIODIC,  // x, y
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL);     // z

    // Enable TVD with van Leer limiter
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::VAN_LEER);

    // Initialize two-layer stratification
    std::vector<float> fill(nx * ny * nz);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                fill[idx] = (k < nz/2) ? 1.0f : 0.0f;  // Heavy on bottom
            }
        }
    }
    vof.initialize(fill.data());

    // Time integration
    float t = 0.0f;
    const float t_end = 0.3f;  // 300 ms
    int step = 0;

    while (t < t_end) {
        // Advect VOF field
        vof.advectFillLevel(ux, uy, uz, dt);

        // Reconstruct interface
        vof.reconstructInterface();
        vof.computeCurvature();

        // Check mass conservation
        if (step % 100 == 0) {
            float mass = vof.computeTotalMass();
            printf("Step %d: t=%.4f, mass=%.2f\n", step, t, mass);
        }

        t += dt;
        step++;
    }

    return 0;
}
```

### Switching Between Schemes at Runtime
```cpp
// Start with upwind for stability during initialization
vof.setAdvectionScheme(VOFAdvectionScheme::UPWIND);

for (int step = 0; step < 100; ++step) {
    vof.advectFillLevel(ux, uy, uz, dt);
}

// Switch to TVD after flow develops
vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
vof.setTVDLimiter(TVDLimiter::VAN_LEER);

for (int step = 100; step < n_steps; ++step) {
    vof.advectFillLevel(ux, uy, uz, dt);
}
```

---

## Appendix C: Sweby Diagram

The Sweby diagram defines the TVD region for flux limiters:

```
φ(r)
 2 |     /SUPERBEE\
   |    /          \
   |   /  MC    VAN_LEER
 1 |--+----*----+----
   | /MINMOD  /
   |/        /
 0 +--------+--------- r
   0        1        2
```

**Key regions:**
- **Below φ=1:** First-order (diffusive)
- **φ=1:** Second-order central (optimal)
- **Above φ=1:** Compressive (anti-diffusive)
- **Outside TVD region:** Unstable (oscillatory)

**Limiter characteristics:**
- **Minmod:** Stays at lower bound (most stable)
- **van Leer:** Smooth curve through φ=1
- **MC:** Prevents overshoot (stays below φ=1 line)
- **Superbee:** Maximum compression (upper bound)

---

**End of Document**
