# MultiphysicsSolver Integration Test Suite

Comprehensive test suite for the MultiphysicsSolver, validating all coupling mechanisms and ensuring robust, physically correct simulation of metal additive manufacturing processes.

## Overview

**Coverage Target**: 100% of multiphysics coupling mechanisms

**Test Count**: 30+ integration tests + validation tests

**Run Time**: ~30 minutes for full suite, < 5 minutes for critical tests

## Test Organization

### 1. Energy Conservation Tests (3 tests)
Validate that energy is properly conserved across all physics modules.

- `test_energy_conservation_no_source.cu`
  - **Physics**: Thermal diffusion only, uniform temperature
  - **Criteria**: dE/dt ≈ 0 (< 1% over 100 steps)
  - **Runtime**: ~30s

- `test_energy_conservation_laser_only.cu`
  - **Physics**: Laser heating with no cooling
  - **Criteria**: P_laser = dE/dt (within 10%)
  - **Runtime**: ~60s

- `test_energy_conservation_full.cu`
  - **Physics**: All energy terms (laser, evaporation, radiation, substrate)
  - **Criteria**: Energy balance within 5%
  - **Runtime**: ~90s

### 2. Coupling Correctness Tests (4 tests)
Verify that physics modules interact correctly.

- `test_thermal_fluid_coupling.cu`
  - **Physics**: Buoyancy-driven natural convection
  - **Criteria**: Hot fluid rises, cold sinks
  - **Runtime**: ~60s

- `test_vof_fluid_coupling.cu`
  - **Physics**: VOF advects with fluid velocity
  - **Criteria**: Mass conservation, interface moves correctly
  - **Runtime**: ~60s

- `test_thermal_vof_coupling.cu`
  - **Physics**: Evaporation at interface
  - **Criteria**: Evaporation only at interface, mass loss matches energy loss
  - **Runtime**: ~90s

- `test_phase_fluid_coupling.cu`
  - **Physics**: Darcy damping in mushy zone
  - **Criteria**: v_solid ≈ 0, v_mushy < v_liquid
  - **Runtime**: ~60s

### 3. Force Balance Tests (3 tests)
Validate force computation and physical correctness.

- `test_force_balance_static.cu`
  - **Criteria**: Forces sum to zero at equilibrium
  - **Runtime**: ~30s

- `test_force_magnitude_ordering.cu`
  - **Criteria**: Verify force magnitudes are physical
  - **Runtime**: ~30s

- `test_force_direction.cu`
  - **Criteria**: Buoyancy upward, Marangoni hot→cold
  - **Runtime**: ~30s

### 4. CFL Stability Tests (3 tests)
Ensure numerical stability under extreme conditions.

- `test_cfl_limiting_effectiveness.cu`
  - **Criteria**: Velocity never exceeds target
  - **Runtime**: ~60s

- `test_cfl_limiting_conservation.cu`
  - **Criteria**: CFL limiting preserves conservation
  - **Runtime**: ~60s

- `test_extreme_gradients.cu`
  - **Criteria**: Survive 10^7 K/m gradients
  - **Runtime**: ~120s

### 5. Subcycling Tests (2 tests)
Validate VOF subcycling for temporal accuracy.

- `test_vof_subcycling_convergence.cu`
  - **Criteria**: Results converge with more subcycles
  - **Runtime**: ~180s

- `test_subcycling_1_vs_10.cu`
  - **Criteria**: Compare N=1 vs N=10 subcycles
  - **Runtime**: ~90s

### 6. Unit Conversion Tests (2 tests)
Critical validation of lattice ↔ physical unit conversions.

- `test_unit_conversion_roundtrip.cu`
  - **Criteria**: lattice→physical→lattice is identity
  - **Runtime**: ~15s

- `test_unit_conversion_consistency.cu`
  - **Criteria**: All modules use same conversions
  - **Runtime**: ~30s

### 7. Steady State Tests (3 tests)
Validate long-time behavior and equilibrium.

- `test_steady_state_temperature.cu`
  - **Criteria**: Temperature reaches equilibrium
  - **Runtime**: ~180s

- `test_steady_state_flow.cu`
  - **Criteria**: Flow reaches steady state
  - **Runtime**: ~180s

- `test_melt_pool_dimensions.cu`
  - **Criteria**: Melt pool size matches analytical estimates
  - **Runtime**: ~180s

### 8. Robustness Tests (3 tests)
Stress testing under extreme conditions.

- `test_high_power_laser.cu`
  - **Criteria**: 500W laser doesn't crash
  - **Runtime**: ~120s

- `test_rapid_solidification.cu`
  - **Criteria**: Fast cooling doesn't diverge
  - **Runtime**: ~120s

- `test_nan_detection.cu`
  - **Criteria**: NaN detected and reported
  - **Runtime**: ~30s

### 9. Module Enable/Disable Tests (3 tests)
Verify configurability and modularity.

- `test_disable_marangoni.cu`
  - **Criteria**: Works without Marangoni
  - **Runtime**: ~30s

- `test_disable_vof.cu`
  - **Criteria**: Works without VOF (single phase)
  - **Runtime**: ~30s

- `test_minimal_config.cu`
  - **Criteria**: Only thermal, no fluid
  - **Runtime**: ~30s

### 10. Regression Tests (2 tests)
Prevent regressions and ensure determinism.

- `test_known_good_output.cu`
  - **Criteria**: Compare with saved golden output
  - **Runtime**: ~60s

- `test_deterministic.cu`
  - **Criteria**: Same input → same output
  - **Runtime**: ~120s

## Running Tests

### Quick Critical Tests (< 5 minutes)
```bash
cd build
make multiphysics_critical_tests
```

### Energy Conservation Suite
```bash
cd build
make multiphysics_energy_tests
```

### Full Test Suite (~30 minutes)
```bash
cd build
make multiphysics_all_tests
```

### Individual Test
```bash
cd build
./tests/integration/multiphysics/test_energy_conservation_full
```

## Success Criteria Summary

### Critical (Must Pass)
- ✓ Energy conservation (< 5% error)
- ✓ Mass conservation (< 1% error)
- ✓ No NaN in all tests
- ✓ CFL stability (v < v_target)
- ✓ Unit conversions exact

### Important (Should Pass)
- ✓ Force directions physically correct
- ✓ Coupling magnitudes reasonable
- ✓ Steady states reached
- ✓ Deterministic results

### Informational (Nice to Have)
- Comparison with literature values
- Performance benchmarks
- Scalability tests

## Test Implementation Status

### Fully Implemented (10 tests)
1. test_energy_conservation_no_source.cu ✓
2. test_energy_conservation_laser_only.cu ✓
3. test_energy_conservation_full.cu ✓
4. test_thermal_fluid_coupling.cu ✓
5. test_vof_fluid_coupling.cu ✓
6. test_thermal_vof_coupling.cu ✓
7. test_phase_fluid_coupling.cu ✓
8. test_cfl_limiting_effectiveness.cu ✓
9. test_nan_detection.cu ✓
10. CMakeLists.txt (build system) ✓

### Stub Implementation (19 tests)
Tests have skeleton structure with TODOs for:
- Specific physics configuration
- Validation logic
- Success criteria implementation

See individual test files for TODO markers.

## Development Workflow

### Adding New Tests

1. **Create test file**:
   ```cpp
   // tests/integration/multiphysics/test_new_feature.cu
   TEST(MultiphysicsTest, NewFeature) { ... }
   ```

2. **Add to CMakeLists.txt**:
   ```cmake
   add_multiphysics_test(test_new_feature test_new_feature.cu)
   set_tests_properties(test_new_feature PROPERTIES
       TIMEOUT 120
       LABELS "integration;multiphysics;new_feature")
   ```

3. **Build and run**:
   ```bash
   cmake --build build
   cd build && ctest -R test_new_feature -V
   ```

### Test Template
See `generate_test_stubs.py` for the standard test template.

## Physics Validation Reference

### Literature Values (Ti6Al4V LPBF)

**Marangoni Velocity**: 0.5-2 m/s (Khairallah et al. 2016)

**Melt Pool Depth**: 50-150 μm @ 200W (King et al. 2015)

**Melt Pool Width**: 100-200 μm @ 200W (King et al. 2015)

**Evaporation Rate**: ~10^-3 kg/(m²·s) @ 3287K (Hertz-Knudsen)

**Surface Tension**: 1.65 N/m (Ti6Al4V liquid)

**dσ/dT**: -0.26×10^-3 N/(m·K) (Ti6Al4V)

## Troubleshooting

### Test Fails with NaN
1. Check CFL limiting parameters
2. Verify timestep size (dt < dx²/(2α))
3. Check for extreme temperature gradients
4. Review force magnitudes

### Test Fails Energy Balance
1. Check boundary conditions (radiation, substrate)
2. Verify laser power calculation
3. Check evaporation power formulation
4. Review temporal integration accuracy

### Test Timeout
1. Reduce domain size (nx, ny, nz)
2. Increase timestep (if stable)
3. Reduce number of steps
4. Increase timeout in CMakeLists.txt

## References

1. Khairallah et al. (2016). "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." *Acta Materialia* 108:36-45.

2. King et al. (2015). "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing." *Journal of Materials Processing Technology* 214(12):2915-2925.

3. Panwisawas et al. (2017). "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution." *Computational Materials Science* 126:479-490.

## Contributing

When adding new tests:
1. Follow existing test structure
2. Use descriptive test names
3. Document success criteria clearly
4. Keep runtime < 2 minutes per test
5. Add to appropriate test suite
6. Update this README

## Contact

For questions about the test suite:
- See main project README
- Review existing test implementations
- Check MultiphysicsSolver documentation
