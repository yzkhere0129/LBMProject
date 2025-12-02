# MultiphysicsSolver Tests - Quick Start Guide

## 5-Minute Quick Start

```bash
# 1. Build the tests
cd /home/yzk/LBMProject
cmake -B build -S .
cmake --build build --target test_energy_conservation_full

# 2. Run a single test
cd build
./tests/integration/multiphysics/test_energy_conservation_full

# 3. Run critical test suite
ctest -L "multiphysics.*critical" --output-on-failure
```

## Available Test Categories

### 1. Energy Conservation (Critical)
```bash
# All energy tests
ctest -R "test_energy_conservation" --output-on-failure

# Individual tests
./tests/integration/multiphysics/test_energy_conservation_no_source
./tests/integration/multiphysics/test_energy_conservation_laser_only
./tests/integration/multiphysics/test_energy_conservation_full
```

### 2. Physics Coupling
```bash
# All coupling tests
ctest -L "coupling" --output-on-failure

# Individual tests
./tests/integration/multiphysics/test_thermal_fluid_coupling
./tests/integration/multiphysics/test_vof_fluid_coupling
./tests/integration/multiphysics/test_thermal_vof_coupling
./tests/integration/multiphysics/test_phase_fluid_coupling
```

### 3. Stability & Robustness
```bash
# CFL and stability tests
./tests/integration/multiphysics/test_cfl_limiting_effectiveness
./tests/integration/multiphysics/test_nan_detection
```

## Common Commands

### Build All Tests
```bash
cmake --build build --parallel 4
```

### Run All Multiphysics Tests
```bash
cd build
ctest -L "multiphysics" -j4 --output-on-failure
```

### Run Fast Tests Only (< 60s each)
```bash
ctest -L "multiphysics.*critical" -j4
```

### Run Single Test with Verbose Output
```bash
ctest -R test_energy_conservation_full -V
```

### Run Tests Matching Pattern
```bash
ctest -R "energy" --output-on-failure
```

## Test Output Interpretation

### Success Output
```
========================================
TEST: Energy Conservation - Full System
========================================

Configuration:
  Domain: 60×60×30
  dx = 2.0 μm
  dt = 10.0 ns
  Laser power: 100.0 W

...

Final Results:
  P_laser     = 35.000 W (INPUT)
  dE/dt       = 33.123 W
  Balance error = 2.34 %

Energy Balance Check:
  Tolerance: 5.0%
  Measured:  2.34%
  Status: PASS ✓

========================================
TEST PASSED ✓
========================================
```

### Failure Output
```
test_energy_conservation_full.cu:123: Failure
Expected: (balance_error) <= (tolerance), actual: 7.23 > 5.0
Energy balance violated: error = 7.23%

========================================
TEST FAILED ✗
========================================
```

## Troubleshooting

### Test Fails to Build
```bash
# Check for missing dependencies
cmake --build build --target test_energy_conservation_full --verbose

# Rebuild from scratch
rm -rf build
cmake -B build -S .
cmake --build build
```

### Test Fails with NaN
1. Check configuration parameters in test file
2. Reduce timestep (dt)
3. Enable CFL limiting
4. Check for extreme gradients

### Test Timeout
1. Reduce domain size (nx, ny, nz)
2. Reduce number of steps
3. Increase timeout in CMakeLists.txt

### All Tests Fail
```bash
# Check CUDA availability
nvidia-smi

# Rebuild entire project
cd /home/yzk/LBMProject
rm -rf build
cmake -B build -S .
cmake --build build
```

## Performance Tips

### Parallel Testing
```bash
# Run 4 tests simultaneously
ctest -j4
```

### Fast Iteration During Development
```bash
# Build and run single test
cmake --build build --target test_energy_conservation_full && \
  ./build/tests/integration/multiphysics/test_energy_conservation_full
```

### Skip Long Tests
```bash
# Run only tests < 60s
ctest -L "multiphysics" --timeout 60
```

## Test Development

### Create New Test (from template)
```bash
cd /home/yzk/LBMProject/tests/integration/multiphysics
python3 generate_test_stubs.py
```

### Implement Stub Test
1. Open test file (e.g., `test_force_balance_static.cu`)
2. Find TODO markers
3. Implement validation logic
4. Build and test
5. Update documentation

### Add to CMakeLists.txt
```cmake
add_multiphysics_test(test_my_new_test test_my_new_test.cu)
set_tests_properties(test_my_new_test PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;my_category")
```

## CI/CD Integration

### Pre-Commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
cd build && ctest -L "multiphysics.*critical" -j4 --output-on-failure
```

### GitHub Actions
```yaml
- name: Run Multiphysics Tests
  run: |
    cd build
    ctest -L "multiphysics" -j4 --output-on-failure --timeout 3600
```

## Documentation

- **Full Guide**: See [README.md](README.md)
- **Test Summary**: See [TEST_SUMMARY.md](TEST_SUMMARY.md)
- **Main Project**: See [/home/yzk/LBMProject/README.md](../../../README.md)

## Support

For issues with tests:
1. Check test documentation in source file
2. Review README.md for detailed information
3. Examine similar working tests
4. Check MultiphysicsSolver API documentation

## Test Statistics

- **Total Tests**: 30+
- **Fully Implemented**: 10
- **Critical Tests Runtime**: ~5 minutes
- **Full Suite Runtime**: ~30 minutes
- **Test Code**: ~4,000 lines

---

*Last Updated: 2025-12-02*
