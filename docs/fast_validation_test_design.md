# Fast Laser Heating Validation Test Design

## Problem Statement

The current `test_laser_heating.cu::EnergyConservation` test takes too long (~150k steps) for routine validation. We need a fast alternative that verifies laser heating functionality and energy balance without the overhead of full steady-state convergence.

## Design Overview

### Test File
`/home/yzk/LBMProject/tests/integration/test_laser_heating_fast.cu`

### Philosophy
**"Fast validation over exhaustive verification"**

- Verify correctness in <30 seconds total runtime
- Simple pass/fail criteria
- Minimal diagnostic overhead
- GPU-resident computations

## Configuration Parameters

### Domain Size
```
Original test: 32 x 32 x 16 = 16,384 cells
Fast test:     24 x 24 x 12 = 6,912 cells  (58% reduction)
```

**Rationale**: Smaller domain reduces memory bandwidth and computation time while maintaining sufficient resolution to capture laser heating profile.

### Timesteps

| Test | Steps | Physical Time | Estimated Runtime |
|------|-------|---------------|-------------------|
| Test 1: LaserHeatsUpDomain | 200 | 0.2 µs | ~5 seconds |
| Test 2: EnergyBalanceConvergence | 500 | 0.5 µs | ~10 seconds |
| Test 3: SpatialDistributionCheck | 200 | 0.2 µs | ~5 seconds |
| **Total** | **900** | **0.9 µs** | **~25 seconds** |

**Original test**: 150,000 steps (~30+ minutes)

**Speedup**: ~70x faster

### Grid/Block Dimensions
```cpp
dim3 blockSize(256);  // Standard, good occupancy
dim3 gridSize((nx * ny * nz + 255) / 256);
```

### Output Intervals
```
Original: Every 500-1000 steps (verbose diagnostics)
Fast:     Only at test completion (summary only)
```

## Disabled Features

### 1. Verbose Diagnostics
**Original behavior:**
- Print every check_interval (500 steps)
- Temperature field copies to host
- Console output for monitoring

**Fast test behavior:**
- No intermediate output
- Only final summary printed

### 2. VTK Export
**Original behavior:**
- Export full 3D temperature fields to VTK files
- Write to disk

**Fast test behavior:**
- NO VTK export (add manually if needed for debugging)
- Comment indicates where to add if debugging

### 3. Host-Side Energy Computation
**Original behavior:**
```cpp
// CPU-side reduction over entire domain
std::vector<float> h_temperature(nx * ny * nz);
thermal.getTemperatureField(h_temperature.data());  // SLOW: Device->Host copy
for (int idx = 0; idx < nx * ny * nz; ++idx) {
    float T = h_temperature[idx];
    // ... compute energy ...
}
```

**Fast test behavior:**
```cpp
// GPU-side reduction with minimal host transfer
computeTotalEnergyKernel<<<...>>>(temperature, partial_sums, ...);
// Only copy partial_sums[num_blocks] to host (~few KB instead of MB)
```

### 4. Detailed Temperature Profiles
**Original behavior:**
- Extract full temperature profiles
- Analyze FWHM, Gaussian shape
- Multiple snapshots

**Fast test behavior:**
- Only sample key points (center, near, far, corner)
- No profile analysis
- Single final snapshot

## Validation Criteria

### Test 1: LaserHeatsUpDomain
**Purpose**: Verify laser is depositing energy

**Checks**:
1. ✓ Energy increases: `energy_final > energy_initial`
2. ✓ Reasonable efficiency: `5% < (energy_increase / energy_input) < 100%`
3. ✓ Significant heating: `T_max > T_init + 50 K`
4. ✓ Physical bounds: `T_max < 2 * T_melt`

**Expected Results**:
- Energy increase: ~0.5-5 µJ (depending on configuration)
- Efficiency: 10-40% (rest lost through boundaries)
- Peak temperature: 400-800 K (initial heating phase)

### Test 2: EnergyBalanceConvergence
**Purpose**: Verify energy balance is converging toward steady state

**Checks**:
1. ✓ Monotonic energy increase
2. ✓ Power retention decreases over time (approaching steady state)
3. ✓ Reasonable retention: `0% < final_retention < 100%`

**Expected Results**:
- Phase 1 retention: ~40-60% (cold start)
- Phase 5 retention: ~20-40% (warming up, more boundary losses)
- Trend: Decreasing retention rate (approaching equilibrium)

**Physical interpretation**:
```
dE/dt = P_laser - P_boundaries
```
As system heats up, `P_boundaries` increases (T⁴ radiation, conduction losses), so `dE/dt` decreases.

### Test 3: SpatialDistributionCheck
**Purpose**: Verify temperature distribution is physically reasonable

**Checks**:
1. ✓ Hottest point at laser center: `T_center > T_near > T_far`
2. ✓ Strong gradient: `T_center - T_far > 50 K`
3. ✓ No unphysical values: `T_init ≤ T ≤ 5000 K`

**Expected Results**:
- T_center: 450-800 K
- T_near: 350-500 K
- T_far: 300-350 K

## Performance Optimization Techniques

### 1. GPU-Side Reductions
```cpp
// Instead of copying entire temperature field:
float computeTotalEnergyGPU(const ThermalLBM& thermal, ...) {
    computeTotalEnergyKernel<<<...>>>(  // Parallel reduction on GPU
        temperature,
        partial_sums,  // Only num_blocks floats
        ...
    );
    // Copy only partial_sums (~256 bytes) instead of full field (~16 MB)
    cudaMemcpy(h_partial, d_partial_sums, gridSize.x * sizeof(float), ...);
    // Finish reduction on CPU (trivial cost)
    return sum(h_partial);
}
```

**Speedup**: ~1000x for energy computation

### 2. Shared Memory Reductions
```cpp
__global__ void computeTotalEnergyKernel(...) {
    extern __shared__ float s_data[];

    // Load to shared memory
    s_data[tid] = energy_per_cell[idx];
    __syncthreads();

    // Tree reduction in shared memory (fast!)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // One value per block
    if (tid == 0) partial_sums[blockIdx.x] = s_data[0];
}
```

### 3. Minimal Synchronization
```cpp
// Only synchronize when absolutely necessary
for (int step = 0; step < n_steps; ++step) {
    computeLaserHeatSourceKernel<<<...>>>(/* async */);
    thermal.addHeatSource(/* async */);
    thermal.step(/* async */);
    // NO cudaDeviceSynchronize() inside loop!
}
// Synchronize only at end
cudaDeviceSynchronize();
```

### 4. Reduced Precision Where Appropriate
Currently using `float` throughout (sufficient for validation).

## Expected Runtime Breakdown

Hardware assumption: NVIDIA V100 or similar

| Component | Time | Percentage |
|-----------|------|------------|
| Kernel execution (900 steps) | ~15 s | 60% |
| Memory transfers | ~2 s | 8% |
| Host-side reductions | ~1 s | 4% |
| Test overhead (setup/teardown) | ~3 s | 12% |
| Google Test framework | ~4 s | 16% |
| **Total** | **~25 s** | **100%** |

## Compilation

Add to `CMakeLists.txt`:

```cmake
# Fast laser heating validation test
cuda_add_executable(test_laser_heating_fast
    tests/integration/test_laser_heating_fast.cu
    src/physics/thermal/thermal_lbm.cu
    src/physics/laser/laser_source.cu
    src/physics/materials/material_properties.cu
    src/lattice/lattice_d3q7.cu
)
target_link_libraries(test_laser_heating_fast
    ${CUDA_LIBRARIES}
    gtest
    gtest_main
    pthread
)
target_include_directories(test_laser_heating_fast PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
set_target_properties(test_laser_heating_fast PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"
)
```

Build:
```bash
cd build
cmake .. -DBUILD_TESTING=ON
make test_laser_heating_fast
```

Run:
```bash
./test_laser_heating_fast
```

## Usage Guidelines

### When to Use This Test
- ✓ Pre-commit validation (fast CI/CD)
- ✓ Regression testing after code changes
- ✓ Quick sanity check during development
- ✓ Verifying laser heating is working

### When NOT to Use This Test
- ✗ Validating steady-state accuracy
- ✗ Detailed energy accounting
- ✗ Long-term stability testing
- ✗ Production simulations

### Recommended Test Strategy
```
1. Fast validation test (30 seconds) ← Use this test
2. Medium validation test (5 minutes) ← Intermediate checks
3. Full validation test (30 minutes) ← Before major releases
```

## Interpreting Results

### Pass Scenario
```
=== Fast Laser Heating Validation ===
Configuration:
  Domain: 24x24x12
  Steps: 200 (0.2 µs)
  Laser power: 100 W
  Absorbed power: 35 W

Results:
  Energy input (laser): 7.0e-06 J
  Energy increase (domain): 2.1e-06 J
  Efficiency: 30.0%
  T_max (initial): 300 K
  T_max (final): 650 K
  Temperature rise: 350 K

✓ All validation criteria passed
```

**Interpretation**: Laser is working correctly, energy balance is reasonable.

### Fail Scenario: No Heating
```
Results:
  Energy increase (domain): 0.0 J
  T_max (final): 300 K

FAIL: Energy must increase when laser is on
```

**Diagnosis**:
- Laser source not being applied
- Check `addHeatSource()` implementation
- Verify laser position is within domain

### Fail Scenario: Excessive Efficiency
```
Results:
  Efficiency: 150.0%

FAIL: Cannot gain more energy than input
```

**Diagnosis**:
- Energy calculation bug
- Unit conversion error
- Check `computeTotalEnergyGPU()` implementation

### Fail Scenario: Unphysical Temperature
```
Results:
  T_max (final): 8500 K

FAIL: Temperature unreasonably high (possible numerical issue)
```

**Diagnosis**:
- Numerical instability (check CFL condition)
- Heat source too large (check absorption calculation)
- Timestep too large

## Limitations

1. **No steady-state validation**: 200-500 steps is insufficient to reach steady state (requires ~50,000+ steps)

2. **Small domain effects**: 24x24x12 domain may have proportionally larger boundary effects than realistic domains

3. **No convergence proof**: Energy increasing ≠ converging to correct solution

4. **Limited spatial resolution**: Cannot validate fine details of temperature profile

5. **No long-term stability**: Cannot detect slow divergence or drift

## Future Improvements

1. **Adaptive step count**: Automatically determine steps needed based on heating rate

2. **Reference solution comparison**: Compare against analytical solution for simple cases

3. **Multi-resolution testing**: Run same test at different domain sizes to verify scaling

4. **Parametric sweeps**: Test various laser powers, spot sizes automatically

5. **Performance benchmarking**: Track runtime over time to detect performance regressions

## References

- Original test: `/home/yzk/LBMProject/tests/integration/test_laser_heating.cu`
- ThermalLBM implementation: `/home/yzk/LBMProject/src/physics/thermal/thermal_lbm.cu`
- LaserSource implementation: `/home/yzk/LBMProject/src/physics/laser/laser_source.cu`

## Revision History

- 2025-11-20: Initial design (Expert 2 fast validation specialist)
