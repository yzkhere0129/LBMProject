# VOF TVD Implementation Quick Reference

**Purpose:** Fast lookup for implementing TVD advection in VOFSolver
**Audience:** Developer implementing the TVD kernel
**Related:** `/home/yzk/LBMProject/docs/VOF_TVD_ARCHITECTURAL_DESIGN.md` (full design)

---

## Implementation Checklist

- [ ] Phase 1: Limiter Functors (1 day)
  - [ ] Create `include/physics/vof_limiters.h`
  - [ ] Define VanLeerLimiter, SuperbeeLimiter, MinmodLimiter
  - [ ] Add unit tests for limiter functions
- [ ] Phase 2: TVD Kernel (2 days)
  - [ ] Create `src/physics/vof/vof_advection_tvd.cu`
  - [ ] Implement `advectFillLevelTVDKernel<Limiter>(...)`
  - [ ] Handle 5-point stencil with boundary fallback
  - [ ] Verify conservative flux formulation
- [ ] Phase 3: Integration (1 day)
  - [ ] Add AdvectionScheme enum to vof_solver.h
  - [ ] Add dispatch in VOFSolver::advectFillLevel()
  - [ ] Add setAdvectionScheme() method
- [ ] Phase 4: Testing (2-3 days)
  - [ ] Zalesak disk rotation test
  - [ ] Diagonal translation test
  - [ ] RT mushroom test (mass < 2%)
  - [ ] Performance profiling
- [ ] Phase 5: Documentation (1 day)
  - [ ] Update API docs
  - [ ] Benchmark results
  - [ ] User guide

**Total: 7-8 days**

---

## TVD Kernel Template Structure

```cpp
// File: src/physics/vof/vof_advection_tvd.cu

template<typename LimiterFunc>
__global__ void advectFillLevelTVDKernel(
    const float* __restrict__ fill_level,
    float* __restrict__ fill_level_new,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float dt, float dx,
    int nx, int ny, int nz,
    int bc_x, int bc_y, int bc_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // === STEP 1: LOAD 5-POINT STENCIL ===
    // X-direction: [i-2][i-1][i][i+1][i+2]
    // Y-direction: [j-2][j-1][j][j+1][j+2]
    // Z-direction: [k-2][k-1][k][k+1][k+2]

    // === STEP 2: CHECK IF INTERIOR CELL ===
    bool is_interior = (i >= 2 && i < nx-2) &&
                       (j >= 2 && j < ny-2) &&
                       (k >= 2 && k < nz-2);

    if (!is_interior) {
        // FALLBACK: Use first-order upwind at boundaries
        // ... (copy from existing advectFillLevelUpwindKernel)
        return;
    }

    // === STEP 3: COMPUTE TVD FLUXES ===
    // X-direction
    float flux_xp = computeTVDFlux<LimiterFunc>(
        ux[idx], ux[idx_ip1],
        fill_level[idx_im2], fill_level[idx_im1],
        fill_level[idx], fill_level[idx_ip1], fill_level[idx_ip2]
    );
    float flux_xm = computeTVDFlux<LimiterFunc>(
        ux[idx_im1], ux[idx],
        // ... similar neighbor pattern
    );

    // Y-direction
    float flux_yp = computeTVDFlux<LimiterFunc>(...);
    float flux_ym = computeTVDFlux<LimiterFunc>(...);

    // Z-direction
    float flux_zp = computeTVDFlux<LimiterFunc>(...);
    float flux_zm = computeTVDFlux<LimiterFunc>(...);

    // === STEP 4: CONSERVATIVE UPDATE ===
    float div_flux = (flux_xp - flux_xm +
                      flux_yp - flux_ym +
                      flux_zp - flux_zm) / dx;

    float f_new = fill_level[idx] - dt * div_flux;

    // === STEP 5: CLAMP TO [0, 1] ===
    if (f_new < 1e-9f) f_new = 0.0f;
    fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}
```

---

## TVD Flux Computation (Device Function)

```cpp
template<typename LimiterFunc>
__device__ __forceinline__ float computeTVDFlux(
    float u_left,      // Velocity at left cell center
    float u_right,     // Velocity at right cell center
    float f_im2,       // Fill level at i-2
    float f_im1,       // Fill level at i-1
    float f_i,         // Fill level at i
    float f_ip1,       // Fill level at i+1
    float f_ip2)       // Fill level at i+2
{
    // Face velocity (interpolated)
    float u_face = 0.5f * (u_left + u_right);

    // Upwind direction check
    if (u_face >= 0.0f) {
        // Flow from i → i+1, upwind is i
        // Smoothness indicator: r = (f_i - f_{i-1}) / (f_{i-1} - f_{i-2})
        float denominator = f_im1 - f_im2;
        float numerator = f_i - f_im1;

        if (fabsf(denominator) < 1e-10f) {
            // Constant gradient or zero gradient → fall back to 1st-order
            return u_face * f_i;
        }

        float r = numerator / denominator;

        // Apply limiter
        float phi = LimiterFunc::compute(r);

        // 2nd-order upwind reconstruction
        // f_face = f_i - 0.5 * phi * (f_i - f_{i-1})
        float f_face = f_i - 0.5f * phi * numerator;

        // Clamp reconstructed value to physical bounds
        f_face = fminf(1.0f, fmaxf(0.0f, f_face));

        return u_face * f_face;

    } else {
        // Flow from i+1 → i, upwind is i+1
        float denominator = f_i - f_ip1;
        float numerator = f_ip1 - f_ip2;

        if (fabsf(denominator) < 1e-10f) {
            return u_face * f_ip1;
        }

        float r = numerator / denominator;
        float phi = LimiterFunc::compute(r);

        // f_face = f_{i+1} + 0.5 * phi * (f_{i+1} - f_{i+2})
        float f_face = f_ip1 + 0.5f * phi * numerator;
        f_face = fminf(1.0f, fmaxf(0.0f, f_face));

        return u_face * f_face;
    }
}
```

---

## Index Computation for 5-Point Stencil

```cpp
// X-direction neighbors with boundary handling
int im2, im1, ip1, ip2;

if (bc_x == 0) {  // PERIODIC
    im2 = (i >= 2) ? i - 2 : nx - 2 + i;
    im1 = (i >= 1) ? i - 1 : nx - 1;
    ip1 = (i < nx - 1) ? i + 1 : i + 1 - nx;
    ip2 = (i < nx - 2) ? i + 2 : i + 2 - nx;
} else {  // WALL - use interior cell fallback check instead
    im2 = max(i - 2, 0);
    im1 = max(i - 1, 0);
    ip1 = min(i + 1, nx - 1);
    ip2 = min(i + 2, nx - 1);
}

// Global indices
int idx_im2 = im2 + nx * (j + ny * k);
int idx_im1 = im1 + nx * (j + ny * k);
int idx = i + nx * (j + ny * k);
int idx_ip1 = ip1 + nx * (j + ny * k);
int idx_ip2 = ip2 + nx * (j + ny * k);

// Repeat for Y and Z directions...
```

**Note:** For wall boundaries, recommend using `is_interior` check and fallback to first-order upwind at boundary cells (simpler and safer).

---

## Limiter Function Implementations

### Van Leer (Recommended for VOF)

```cpp
struct VanLeerLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;  // Upwind fallback
        return (r + fabsf(r)) / (1.0f + fabsf(r));
    }
};
```

**Properties:**
- Smooth (no kinks)
- 2nd-order accurate
- Good balance of accuracy and robustness

### Superbee (Sharpest Interfaces)

```cpp
struct SuperbeeLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;
        float term1 = fminf(2.0f * r, 1.0f);
        float term2 = fminf(r, 2.0f);
        return fmaxf(term1, term2);
    }
};
```

**Properties:**
- Least diffusive
- Sharpest interface preservation
- May produce slight oscillations

### Minmod (Most Robust)

```cpp
struct MinmodLimiter {
    __device__ __forceinline__ static float compute(float r) {
        if (r <= 0.0f) return 0.0f;
        return fminf(1.0f, r);
    }
};
```

**Properties:**
- Most diffusive (but still better than upwind)
- Most robust (guaranteed monotone)
- Best for low resolution

---

## API Integration

### Add to vof_solver.h

```cpp
// After CellFlag enum, add:
enum class AdvectionScheme : uint8_t {
    UPWIND_1ST = 0,      // First-order upwind (current)
    TVD_VAN_LEER = 1,    // TVD with van Leer limiter
    TVD_SUPERBEE = 2,    // TVD with superbee limiter
    TVD_MINMOD = 3       // TVD with minmod limiter
};

// In VOFSolver class private section:
AdvectionScheme advection_scheme_;

// In VOFSolver class public section:
void setAdvectionScheme(AdvectionScheme scheme) {
    advection_scheme_ = scheme;
}

AdvectionScheme getAdvectionScheme() const {
    return advection_scheme_;
}
```

### Add to vof_solver.cu Constructor

```cpp
VOFSolver::VOFSolver(int nx, int ny, int nz, float dx,
                     BoundaryType bc_x, BoundaryType bc_y, BoundaryType bc_z)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz), dx_(dx),
      bc_x_(bc_x), bc_y_(bc_y), bc_z_(bc_z),
      advection_scheme_(AdvectionScheme::UPWIND_1ST),  // <-- ADD THIS
      d_fill_level_(nullptr), d_cell_flags_(nullptr),
      // ... rest of initialization
```

### Dispatch in advectFillLevel()

```cpp
// Inside subcycling loop:
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
        advectFillLevelTVDKernel<SuperbeeLimiter><<<gridSize, blockSize>>>(/* ... */);
        break;

    case AdvectionScheme::TVD_MINMOD:
        advectFillLevelTVDKernel<MinmodLimiter><<<gridSize, blockSize>>>(/* ... */);
        break;

    default:
        throw std::runtime_error("Unknown advection scheme");
}
```

---

## Testing Strategy

### Test 1: Zalesak Disk Rotation (Unit Test)

**Setup:**
```cpp
// Domain: 100 × 100 × 4 (quasi-2D)
// Disk: R=15, centered at (50, 75)
// Velocity: solid body rotation, ω = 2π/628 (period = 628 steps)
// Run: 1 full rotation (628 steps)
```

**Success criteria:**
- Mass error < 1% (vs 5-10% for upwind)
- Shape preservation: > 95% of disk intact
- No oscillations (all f ∈ [0, 1])

**File:** `tests/unit/vof/test_vof_zalesak_disk.cu`

### Test 2: Diagonal Translation (Unit Test)

**Setup:**
```cpp
// Domain: 64 × 64 × 4
// Initial: sharp interface at x=32
// Velocity: (1, 1, 0) diagonal
// Run: 100 steps
```

**Success criteria:**
- Mass error < 2%
- No grid alignment (interface stays sharp in diagonal direction)

**File:** `tests/unit/vof/test_vof_diagonal_translation.cu`

### Test 3: RT Mushroom (Integration Test)

**Setup:**
```cpp
// Same as current RT test
// Change: vof.setAdvectionScheme(VOFSolver::AdvectionScheme::TVD_VAN_LEER);
```

**Success criteria:**
- Mass error < 2% (vs current 7%)
- h₁ = 1.37 ± 0.05 m (physics unchanged)
- Simulation time ≤ current (no slowdown)

**File:** `tests/validation/vof/test_rayleigh_taylor_tvd.cu`

---

## Performance Profiling

### Metrics to Track

```bash
# Memory bandwidth
nvprof --metrics gld_throughput,gst_throughput ./test_rayleigh_taylor_tvd

# Occupancy
nvprof --metrics achieved_occupancy ./test_rayleigh_taylor_tvd

# Kernel time
nsys profile --stats=true ./test_rayleigh_taylor_tvd
```

**Expected results:**
- Memory bandwidth: 70-80% of peak
- Occupancy: 75-85% (30-35 registers per thread)
- TVD kernel time: 2-3× upwind kernel time
- Total simulation time: ≤ current (due to fewer substeps)

### Optimization Targets

**If TVD kernel > 3× upwind:**
1. Check memory access patterns (should be coalesced)
2. Try shared memory for 8×8×8 block + halo
3. Reduce register pressure (move constant computations out)

**If occupancy < 70%:**
1. Reduce register usage (simplify limiter computation)
2. Adjust block size (try 4×4×4 or 16×16×2)

---

## Debugging Tips

### Check 1: Limiter Function Correctness

```cpp
// Host-side unit test
void test_limiter() {
    VanLeerLimiter limiter;
    assert(limiter.compute(-1.0f) == 0.0f);  // Downwind → upwind fallback
    assert(limiter.compute(0.0f) == 0.0f);   // Zero gradient
    assert(fabs(limiter.compute(1.0f) - 1.0f) < 1e-6f);  // Smooth region
    // Van Leer: φ(1) = (1 + 1) / (1 + 1) = 1
}
```

### Check 2: Mass Conservation

```cpp
// Before and after advection
float mass_before = vof.computeTotalMass();
vof.advectFillLevel(ux, uy, uz, dt);
float mass_after = vof.computeTotalMass();
float mass_error = fabsf(mass_after - mass_before) / mass_before;
printf("Mass error: %.4f%%\n", mass_error * 100.0f);
// Should be < 0.1% for single step
```

### Check 3: Boundedness

```cpp
// After advection, check all f ∈ [0, 1]
std::vector<float> h_fill(num_cells);
vof.copyFillLevelToHost(h_fill.data());

int violations = 0;
for (int i = 0; i < num_cells; ++i) {
    if (h_fill[i] < -1e-6f || h_fill[i] > 1.0f + 1e-6f) {
        violations++;
        printf("Violation at cell %d: f = %.6f\n", i, h_fill[i]);
    }
}
// Should be zero violations
```

### Check 4: Interface Sharpness

```cpp
// Measure interface thickness (number of cells with 0.01 < f < 0.99)
int interface_cells = 0;
for (int i = 0; i < num_cells; ++i) {
    if (h_fill[i] > 0.01f && h_fill[i] < 0.99f) {
        interface_cells++;
    }
}
float interface_thickness = sqrtf(interface_cells);  // Approx width
printf("Interface thickness: %.1f cells\n", interface_thickness);
// TVD should be 2-3 cells, upwind 3-4 cells
```

---

## Common Pitfalls

### Pitfall 1: Sign Error in Flux Direction

**Symptom:** Mass grows instead of conserved.

**Cause:** Flux sign convention wrong.

**Fix:** Carefully check:
```cpp
// Correct: flux points OUTWARD from cell
div_flux = (flux_xp - flux_xm) / dx;
f_new = f - dt * div_flux;  // <-- MINUS sign
```

### Pitfall 2: Limiter Domain Error

**Symptom:** NaN values appear, simulation crashes.

**Cause:** Division by zero in r = num / denom.

**Fix:** Add epsilon to denominator:
```cpp
float r = numerator / (denominator + 1e-10f);
```

### Pitfall 3: Boundary Fallback Not Working

**Symptom:** Artifacts at domain boundaries.

**Cause:** Forgot to implement fallback to upwind at boundaries.

**Fix:** Check `is_interior` condition:
```cpp
bool is_interior = (i >= 2 && i < nx-2) && (j >= 2 && j < ny-2) && (k >= 2 && k < nz-2);
if (!is_interior) {
    // Use first-order upwind (copy existing code)
    return;
}
```

### Pitfall 4: Template Not Instantiated

**Symptom:** Linker error "undefined reference to advectFillLevelTVDKernel".

**Cause:** Template kernel must be instantiated for each limiter.

**Fix:** Add explicit instantiations at end of .cu file:
```cpp
// Explicit template instantiations
template __global__ void advectFillLevelTVDKernel<VanLeerLimiter>(...);
template __global__ void advectFillLevelTVDKernel<SuperbeeLimiter>(...);
template __global__ void advectFillLevelTVDKernel<MinmodLimiter>(...);
```

### Pitfall 5: Subcycling CFL Too High

**Symptom:** TVD produces oscillations or instabilities.

**Cause:** CFL_target set too high (e.g., 0.8).

**Fix:** Keep CFL_target ≤ 0.5 for TVD:
```cpp
const float CFL_target = 0.25f;  // Safe for TVD
```

---

## Quick Decision Guide

**Question:** Which limiter should I use?

| Use Case | Limiter | Rationale |
|----------|---------|-----------|
| General VOF (default) | van Leer | Best balance, smooth |
| Surface tension simulations | van Leer | Minimizes spurious currents |
| Sharp interfaces (RT, KH) | superbee | Least diffusive |
| Low resolution (< 32 cells) | minmod | Most robust |
| Debugging | minmod | Simplest, most stable |

**Question:** What CFL_target should I use?

| Scheme | CFL_target | Comment |
|--------|-----------|---------|
| Upwind | 0.10 | Current (conservative) |
| TVD van Leer | 0.25 | Recommended |
| TVD van Leer (aggressive) | 0.40 | For speed, slight mass loss |
| TVD superbee | 0.20 | Sharper → more restrictive |

**Question:** Should I disable interface compression?

| Simulation Type | Compression C | Rationale |
|----------------|--------------|-----------|
| RT instability | C = 0.0 | Disable (compression suppresses physics) |
| Oscillating droplet | C = 0.1 | Light compression (reduces spurious currents) |
| Rising bubble | C = 0.1 | Light compression |
| General AM (laser melting) | C = 0.0 | Disable (TVD sufficient) |

---

## Files Checklist

**New files to create:**
- [ ] `include/physics/vof_limiters.h` (limiter functors)
- [ ] `src/physics/vof/vof_advection_tvd.cu` (TVD kernel)
- [ ] `tests/unit/vof/test_vof_zalesak_disk.cu` (Zalesak test)
- [ ] `tests/unit/vof/test_vof_diagonal_translation.cu` (diagonal test)
- [ ] `tests/validation/vof/test_rayleigh_taylor_tvd.cu` (RT comparison)
- [ ] `docs/VOF_TVD_USER_GUIDE.md` (end-user documentation)
- [ ] `docs/VOF_TVD_BENCHMARK_RESULTS.md` (performance data)

**Files to modify:**
- [ ] `include/physics/vof_solver.h` (add enum, setAdvectionScheme)
- [ ] `src/physics/vof/vof_solver.cu` (add dispatch, constructor)
- [ ] `tests/validation/vof/CMakeLists.txt` (add new test targets)

---

## Summary: 3 Key Implementation Points

1. **Separate kernel with template dispatch** → Clean, maintainable, zero overhead
2. **Boundary fallback to first-order** → Robust, simple, <1% cells affected
3. **Conservative flux formulation** → Mass conservation guaranteed

**Expected outcome:** 3.5× better mass conservation, 12% faster simulation, 2-3 cells sharp interface.

---

**Status:** Design complete, ready for implementation
**Estimated time:** 7-8 days
**Risk:** Low (proven technique, fallback available)
**Priority:** Medium (after M1-M3 validation)

---

**Related Files:**
- Full design: `/home/yzk/LBMProject/docs/VOF_TVD_ARCHITECTURAL_DESIGN.md`
- Current implementation: `/home/yzk/LBMProject/src/physics/vof/vof_solver.cu`
- Current header: `/home/yzk/LBMProject/include/physics/vof_solver.h`
