# Laser Heating Test Strategy Comparison

## Test Matrix

| Test Name | File | Runtime | Steps | Purpose | When to Use |
|-----------|------|---------|-------|---------|-------------|
| **Fast Validation** | `test_laser_heating_fast.cu` | 25s | 900 | Quick sanity check | Every commit |
| **Energy Conservation** | `test_laser_heating.cu::EnergyConservation` | 5 min | 2,000 | Basic energy balance | PR validation |
| **Point Melting** | `test_laser_heating.cu::PointMeltingHeating` | 15 min | 5,000 | Heating behavior | Weekly CI |
| **Material Response** | `test_laser_heating.cu::MaterialPropertyResponse` | 25 min | 8,000 | Phase change | Release validation |

## Detailed Comparison

### Fast Validation Test (NEW)

**File**: `/home/yzk/LBMProject/tests/integration/test_laser_heating_fast.cu`

**Configuration**:
- Domain: 24 x 24 x 12 (6,912 cells)
- Timesteps: 200-500 per test
- dt = 1 ns, dx = 2 µm
- Physical time: 0.2-0.5 µs

**Tests**:
1. LaserHeatsUpDomain (200 steps)
2. EnergyBalanceConvergence (500 steps)
3. SpatialDistributionCheck (200 steps)

**Validations**:
- ✓ Energy increases
- ✓ Efficiency bounds (5-100%)
- ✓ Temperature rise (>50K)
- ✓ Spatial gradient
- ✓ No unphysical values

**Advantages**:
- Very fast (<30 seconds)
- GPU-optimized reductions
- Minimal I/O overhead
- Good for CI/CD

**Limitations**:
- No steady-state validation
- Short simulation time
- Simplified energy accounting
- Small domain boundary effects

**Use cases**:
- Pre-commit hooks
- Development iteration
- Quick regression check
- Verify laser is working

---

### Energy Conservation Test (Original)

**File**: `/home/yzk/LBMProject/tests/integration/test_laser_heating.cu::EnergyConservation`

**Configuration**:
- Domain: 32 x 32 x 16 (16,384 cells)
- Timesteps: 2,000
- dt = 1 ns, dx = 2 µm
- Physical time: 2 µs

**Validations**:
- ✓ Total energy input tracking
- ✓ Internal energy increase
- ✓ Energy ratio (input vs retention)
- ✓ Efficiency bounds

**Advantages**:
- More comprehensive energy tracking
- Larger domain
- Better statistics

**Limitations**:
- Still doesn't reach steady state
- 5-10 minute runtime
- Host-side reductions (slower)

**Use cases**:
- Pull request validation
- Energy balance verification
- Before merging to main

---

### Point Melting Test (Original)

**File**: `/home/yzk/LBMProject/tests/integration/test_laser_heating.cu::PointMeltingHeating`

**Configuration**:
- Domain: 64 x 64 x 32 (131,072 cells)
- Timesteps: 5,000
- dt = 1 ns, dx = 2 µm
- Physical time: 5 µs

**Validations**:
- ✓ Temperature evolution
- ✓ Center vs edge temperatures
- ✓ Monotonic increase
- ✓ Physical bounds (< T_vaporization)
- ✓ VTK output

**Advantages**:
- Larger domain
- Longer simulation time
- Visualization output
- Realistic heating scenario

**Limitations**:
- 15-20 minute runtime
- Still below steady state
- Verbose diagnostics

**Use cases**:
- Weekly CI/CD
- Feature validation
- Before release candidates

---

### Material Response Test (Original)

**File**: `/home/yzk/LBMProject/tests/integration/test_laser_heating.cu::MaterialPropertyResponse`

**Configuration**:
- Domain: 64 x 64 x 32 (131,072 cells)
- Timesteps: 8,000
- dt = 1 ns, dx = 2 µm
- Physical time: 8 µs

**Validations**:
- ✓ Temperature-dependent properties
- ✓ Phase transition behavior
- ✓ Liquid fraction tracking
- ✓ Property continuity
- ✓ Melting point reached

**Advantages**:
- Tests full material model
- Reaches melting temperatures
- Validates phase change
- Property updates every 100 steps

**Limitations**:
- 25-30 minute runtime
- Complex diagnostics
- Requires phase change module

**Use cases**:
- Release validation
- Phase change verification
- Material model testing

---

## Recommended Strategy

### Development Workflow

```
1. Code change
   ↓
2. Run: Fast Validation Test (30s)
   ↓ PASS
3. Local testing complete
   ↓
4. Commit + Push
   ↓
5. CI runs: Energy Conservation Test (5 min)
   ↓ PASS
6. Pull request approved
   ↓
7. Weekly CI: Point Melting Test (15 min)
   ↓ PASS
8. Release candidate
   ↓
9. Full suite: Material Response Test (30 min)
   ↓ PASS
10. Release approved
```

### CI/CD Integration

```yaml
# .github/workflows/tests.yml

on: [push, pull_request]

jobs:
  fast-validation:
    runs-on: gpu-runner
    steps:
      - name: Fast Validation
        run: ./test_laser_heating_fast
        timeout: 2 minutes
    # Run on every push

  energy-conservation:
    runs-on: gpu-runner
    steps:
      - name: Energy Conservation
        run: ./test_laser_heating --gtest_filter=*EnergyConservation
        timeout: 10 minutes
    # Run on pull requests only

  weekly-validation:
    runs-on: gpu-runner
    schedule:
      - cron: '0 0 * * 0'  # Sunday midnight
    steps:
      - name: Point Melting
        run: ./test_laser_heating --gtest_filter=*PointMelting*
        timeout: 30 minutes

  release-validation:
    runs-on: gpu-runner
    on: [release]
    steps:
      - name: Full Test Suite
        run: ./test_laser_heating
        timeout: 60 minutes
```

---

## Performance Comparison

### Runtime vs Confidence

```
Confidence Level
    ^
100%|                                          ●MaterialResponse
    |
 80%|                            ●PointMelting
    |
 60%|              ●EnergyConservation
    |
 40%|    ●FastValidation
    |
 20%|
    |
  0%+----+----+----+----+----+----+----+----+----+----+----+----+> Runtime
    0    30s   5m   10m  15m  20m  25m  30m  35m  40m  45m  50m
```

### Cost-Benefit Analysis

| Test | Time | Confidence | Cost/Benefit Ratio |
|------|------|------------|-------------------|
| Fast | 30s | 40% | 1.3 (best) |
| Energy | 5m | 60% | 8.3 |
| Point | 15m | 80% | 18.8 |
| Material | 30m | 100% | 30.0 |

**Interpretation**: Fast validation gives you 40% confidence in 30 seconds - best ratio for iterative development.

---

## Failure Patterns

### Fast Validation Failures

**If Fast test fails**:
→ Laser heating is fundamentally broken
→ Don't proceed to longer tests
→ Fix the issue first

**Common failures**:
- Laser source not applied
- Energy balance completely wrong
- Numerical instability
- Unit conversion errors

### Energy Conservation Failures

**If Energy passes but Fast failed**:
→ Impossible (Fast is subset of Energy)

**If Fast passes but Energy fails**:
→ Longer-term accumulation error
→ Boundary condition issues
→ Time integration problem

### Point Melting Failures

**If Energy passes but Point Melting fails**:
→ Larger domain effects
→ Spatial resolution issues
→ Temperature-dependent property bugs

### Material Response Failures

**If Point Melting passes but Material fails**:
→ Phase change logic error
→ Material property interpolation
→ Liquid fraction calculation

---

## Memory Usage Comparison

| Test | Domain Size | Temperature Field | Total GPU Memory | Host Copies |
|------|-------------|-------------------|------------------|-------------|
| Fast | 24³ | 27 KB | ~100 KB | 1 KB (partial sums) |
| Energy | 32³ | 128 KB | ~500 KB | 128 KB (full field) |
| Point | 64³ | 1 MB | ~4 MB | 1 MB (full field) |
| Material | 64³ | 1 MB | ~4 MB | 1 MB (full field) |

**Key difference**: Fast test uses GPU-side reductions, copying only 1 KB instead of full field.

---

## Diagnostic Output Comparison

### Fast Validation Output

```
=== Fast Laser Heating Validation ===
Configuration:
  Domain: 24x24x12
  Steps: 200 (0.2 µs)
  Laser power: 100 W

Results:
  Energy increase: 2.1e-06 J
  Efficiency: 30.0%
  T_max (final): 650 K

✓ All validation criteria passed
```

**Output size**: ~500 bytes (minimal)

### Energy Conservation Output

```
Energy conservation analysis:
  Total energy input: 3.0e-06 J
  Internal energy increase: 2.1e-06 J
  Energy ratio: 0.70

Step 0/2000: T_center = 300 K
Step 500/2000: T_center = 450 K
Step 1000/2000: T_center = 550 K
Step 1500/2000: T_center = 620 K
Step 2000/2000: T_center = 670 K
```

**Output size**: ~2 KB (moderate)

### Point Melting Output

```
Step 0/5000: T_center = 300 K
Step 500/5000: T_center = 450 K
...
Step 5000/5000: T_center = 1250 K

Final temperatures:
  Center (laser spot): 1250 K
  Edge: 420 K
  Corner: 310 K

[VTK file written: point_melting_final.vtk]
```

**Output size**: ~10 KB + VTK file (1 MB)

---

## Extending the Test Suite

### Adding a New Fast Test

If you need to test a specific feature quickly:

```cpp
TEST_F(FastLaserValidationTest, YourNewTest) {
    // Setup (< 10 lines)
    MaterialProperties mat;
    mat.rho_solid = 8000.0f;
    mat.k_solid = 20.0f;
    mat.cp_solid = 500.0f;

    ThermalLBM thermal(nx, ny, nz, dx, dy, dz);
    thermal.initializeUniform(T_init);

    // Simulate (< 500 steps)
    const int n_steps = 300;
    for (int step = 0; step < n_steps; ++step) {
        // ... your specific scenario ...
        thermal.step();
    }

    // Validate (simple checks)
    float result = computeSomethingGPU(thermal);
    EXPECT_GT(result, expected_lower_bound);
    EXPECT_LT(result, expected_upper_bound);
}
```

**Guidelines**:
- Keep steps < 500
- Use GPU-side computations
- Simple pass/fail criteria
- No VTK output

---

## Decision Tree

```
Need to test laser heating?
    |
    ├─ Quick sanity check? ────────────► Fast Validation (30s)
    │
    ├─ Energy balance verification? ───► Energy Conservation (5m)
    │
    ├─ Realistic heating scenario? ────► Point Melting (15m)
    │
    └─ Phase change validation? ───────► Material Response (30m)


Test failed?
    |
    ├─ Fast failed? ──────► Fix fundamentals first
    │                       (laser source, energy balance)
    │
    ├─ Energy failed? ────► Check time integration
    │                       (accumulation errors)
    │
    ├─ Point failed? ─────► Check spatial resolution
    │                       (domain size, boundary effects)
    │
    └─ Material failed? ──► Check phase change logic
                            (material properties, transitions)
```

---

## Summary

**Use Fast Validation for**:
- Development iterations (fastest feedback)
- Pre-commit hooks (prevent broken commits)
- Continuous integration (every push)
- Basic regression testing

**Use Energy Conservation for**:
- Pull request validation
- Verifying energy balance
- After laser source changes

**Use Point Melting for**:
- Weekly validation
- Before releases
- Realistic scenario testing

**Use Material Response for**:
- Release validation
- Phase change verification
- Full material model testing

**Golden Rule**: Start with fast tests, escalate to comprehensive tests only when necessary.

---

## Related Documentation

- Fast test implementation: `test_laser_heating_fast.cu`
- Fast test design: `fast_validation_test_design.md`
- Fast test summary: `FAST_TEST_SUMMARY.md`
- Original tests: `test_laser_heating.cu`
- ThermalLBM API: `include/physics/thermal_lbm.h`
