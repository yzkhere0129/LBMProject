# LBM Multiphysics Test Matrix

## Overview

This document provides a comprehensive matrix of all physics tests, showing what physical phenomena each test validates, the testing method used, and key validation criteria.

---

## Test Matrix Legend

**Test Type**:
- **U**: Unit test (component level)
- **I**: Integration test (module interaction)
- **V**: Validation test (physics vs analytical/literature)
- **R**: Regression test (prevent known bugs)

**Speed**:
- **F**: Fast (< 30s)
- **M**: Medium (30s - 5min)
- **S**: Slow (> 5min)

**Priority**:
- **C**: Critical (must pass before merge)
- **H**: High (should pass before release)
- **N**: Normal (validation/documentation)

---

## 1. VOF Solver Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_vof_advection | U | F | H | Interface advection ∂f/∂t + ∇·(fu) = 0 | Unit conversion, basic advection | Displacement error < 0.5 cells |
| test_vof_advection_uniform | U | F | H | Uniform velocity advection | Analytical displacement | Displacement < 0.5 cells, mass < 1% |
| test_vof_advection_shear | U | M | H | Shear flow advection | Interface tilt angle | Angle within 30% of theory |
| test_vof_advection_rotation | V | S | N | Zalesak's disk benchmark | Classic VOF benchmark | L1 error < 0.15, mass < 5% |
| test_vof_reconstruction | U | F | H | PLIC reconstruction | Volume consistency | Volume error < 5% |
| test_vof_curvature | U | F | H | Curvature κ = ∇·n | Analytical geometries | Plane: κ≈0, Sphere: error < 10% |
| test_vof_curvature_sphere | V | M | N | Spherical curvature | Multi-resolution | R=20: <10%, R=12: <20%, R=8: <30% |
| test_vof_curvature_cylinder | V | M | N | Cylindrical curvature | 2D geometry | R=16: <15%, R=10: <25% |
| test_vof_mass_conservation | U | F | C | Mass conservation ∫f dV = const | Static and dynamic | Static: <0.1%, dynamic: <1-2% |
| test_vof_surface_tension | U | F | H | Surface tension F = σκn | CSF method, Laplace pressure | ΔP error < 20% |
| test_vof_contact_angle | U | M | H | Contact angle BC | Young's equation | Angle within 5° |
| test_vof_evaporation_mass_loss | U | F | C | Evaporation mass loss df = -J/(ρdx)dt | Single & progressive steps | Single: <5%, cumulative: <10% |
| test_vof_marangoni | U | F | H | VOF-Marangoni coupling | Force application | Force magnitude & direction |
| test_recoil_pressure | U | F | H | Recoil pressure P = 0.54 P_sat | Temperature dependence | Pressure within 20% |
| test_evaporation_temperature_check | U | F | H | Evaporation T threshold | T > T_boil activation | Zero flux below T_boil |
| test_interface_geometry | U | F | H | Normal & tangent vectors | Orthogonality | \|n\| = 1, \|t·n\| < 0.01 |
| test_vof_cell_conversion | U | F | N | VOF cell type detection | Cell flagging | Correct GAS/LIQUID/INTERFACE |
| test_static_droplet | I | M | N | Static droplet equilibrium | Surface tension balance | No spurious currents |
| test_laser_surface_deformation | I | M | N | Laser-driven deformation | Recoil + surface tension | Realistic deformation |

---

## 2. Marangoni Effect Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_marangoni_force_magnitude | U | F | C | Force F = \|dσ/dT\|·\|∇T_t\|·\|∇f\| | Simple gradient case | Magnitude within 20%, order 10^8-10^9 N/m³ |
| test_marangoni_force_direction | U | F | C | Force direction (hot → cold) | Tangential projection | Normal component < 1%, correct direction |
| test_marangoni_gradient_limiter | U | F | H | Gradient limiter stability | Extreme gradients | Limiter activates, force capped |
| test_marangoni_interface_localization | U | F | H | Force localization to interface | Spatial distribution | Force only at interface cells |
| test_marangoni_material_properties | U | F | N | dσ/dT coefficient | Ti6Al4V properties | dσ/dT ≈ -0.00026 N/(m·K) |
| test_marangoni_stability | U | M | H | Long-term stability | 1000 timesteps | No NaN, forces bounded |
| test_marangoni_tangential_projection | U | F | H | Tangent projection accuracy | Orthogonality check | \|F·n\| < 0.01·\|F\| |
| test_marangoni_velocity_scale | U | F | H | Velocity scaling | Force → velocity | Reasonable velocity magnitude |
| test_marangoni_flow | I | M | H | Full Marangoni-driven flow | Temperature gradient setup | Flow hot→cold, velocity 0.1-10 m/s |
| test_marangoni_system | I | M | N | System-level integration | Full melt pool | Physical flow patterns |
| test_marangoni_velocity | V | S | N | Literature velocity validation | Khairallah et al. 2016 | 0.5-2 m/s for Ti6Al4V |

---

## 3. Thermal Solver Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_thermal_lbm | U | F | C | Core thermal LBM | Initialization, diffusion | Uniform init, gradient preserve |
| test_lattice_d3q7 | U | F | C | D3Q7 lattice properties | Weight sum, isotropy | Σw_i = 1, Σc_i = 0 |
| test_flux_limiter | U | F | **C** | TVD flux limiter | Extreme velocities (v=5) | All g_eq ≥ 0, T conserved |
| test_temperature_bounds | U | F | **C** | Temperature clamping | Extreme heating/cooling | T ∈ [0,7000] K, no NaN |
| test_omega_reduction | R | F | **C** | Omega cap ω_T ≤ 1.50 | Various diffusivities | ω_T ≤ 1.50 always |
| test_pure_conduction | V | M | H | Diffusion equation ∂T/∂t = α∇²T | Gaussian analytical | L2 error < 5% |
| test_stefan_problem | V | S | H | Phase change moving boundary | Stefan analytical solution | Interface error < 10%, T error < 15% |
| test_high_pe_stability | I | M | **C** | High-Pe stability (Pe~10) | 500 steps, v=5, T=2500K | No divergence, T_max < 15000K |
| test_laser_source | U | F | H | Gaussian laser beam | Beer-Lambert absorption | Power error < 1%, profile correct |
| test_laser_heating | I | M | N | Laser heating integration | Temperature rise | Realistic heating rate |
| test_laser_melting | I | M | N | Laser melting with phase change | Melt pool formation | Melting occurs, realistic pool |
| test_laser_heating_simplified | I | M | N | Simplified laser test | Reduced complexity | Completes without error |
| test_laser_melting_convection | I | M | N | Melt pool convection | Phase 5 integration | Flow in melt pool |
| test_substrate_cooling_bc | V | M | H | Substrate cooling BC | Heat extraction rate | Cooling rate matches theory |
| test_substrate_temperature_reduction | I | M | H | Temperature reduction via substrate | Integration test | Temperature decreases correctly |

---

## 4. Fluid LBM Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_fluid_lbm | U | F | C | Core fluid LBM | Equilibrium, collision | Density/velocity recovery |
| test_d3q19 | U | F | C | D3Q19 lattice properties | Weights, isotropy | Σw_i = 1, Σc_i c_j = cs²δ_ij |
| test_bgk | U | F | C | BGK collision operator | Relaxation to equilibrium | Correct relaxation, ν = cs²(τ-0.5) |
| test_streaming | U | F | C | Streaming step | Periodic BC, propagation | Conservation during streaming |
| test_boundary | U | F | H | Boundary conditions | Various BC types | BC correctly applied |
| test_fluid_boundaries | U | F | H | Fluid-specific boundaries | Inlet/outlet/wall | Mass conserved at boundaries |
| test_no_slip_boundary | U | F | H | No-slip wall BC | Bounce-back | Wall velocity < 1e-6 |
| test_poiseuille_flow | I | M | H | Poiseuille channel flow | Parabolic profile | Profile error < 5%, peak < 2% |
| test_poiseuille_flow_fluidlbm | I | M | H | Poiseuille via FluidLBM | High-level API | Same as above |
| test_uniform_flow_fluidlbm | I | M | N | Uniform flow maintenance | Constant velocity | Velocity drift < 0.1% |
| test_divergence_free | V | M | **C** | Incompressibility ∇·u = 0 | Various force fields | \|∇·u\| < 1e-3 |
| test_fluid_debug | U | F | N | Debug utilities | Diagnostic output | No crash, useful output |

---

## 5. Phase Change Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_liquid_fraction | U | F | H | Liquid fraction f_l(T) | Three regimes | Solid: f_l=0, liquid: f_l=1, mushy: 0<f_l<1 |
| test_enthalpy | U | F | H | Enthalpy method H = c_p T + f_l L | Newton-Raphson T(H) | Temperature recovery < 1 K |
| test_phase_properties | U | F | H | Phase-dependent properties | Density, viscosity, k | Physical ranges, smooth interpolation |
| test_stefan_1d | I | S | H | 1D Stefan problem | Moving boundary | Interface position error < 10% |
| test_phase_change_robustness | V | M | C | Robustness in extreme cases | Rapid heating/cooling | No NaN, T physical, stable transitions |
| test_materials | U | F | N | Material database | Property lookup | Correct values for all materials |
| test_darcy_damping_solid | U | F | H | Darcy damping F = -C(1-f_l)²u | Velocity in solid/mushy | Solid: v≈0, mushy: gradual damping |
| test_liquid_fraction_copy | U | F | N | Device-to-host transfer | CUDA memory copy | Correct data transfer |

---

## 6. Multiphysics Coupling Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_thermal_fluid_coupling | I | M | C | Buoyancy F = ρβg(T-T_ref) | Natural convection | Flow upward in hot region |
| test_mp_thermal_fluid_coupling | I | M | C | Same via MultiphysicsSolver | High-level API | Same as above |
| test_vof_fluid_coupling | I | M | C | Surface tension → fluid velocity | CSF force coupling | Force affects velocity, interface advects |
| test_thermal_vof_coupling | I | M | C | Evaporation thermal-VOF coupling | Mass loss + cooling | Evaporation at T>T_boil, mass decreases |
| test_phase_fluid_coupling | I | M | C | Darcy damping in solid/mushy | Phase-dependent viscosity | Velocity ~0 in solid, damped in mushy |
| test_force_balance_static | I | F | H | Force equilibrium | Static balance | Net force < 1% of individual forces |
| test_force_magnitude_ordering | I | F | H | Realistic force magnitudes | Literature comparison | Order of magnitude correct |
| test_force_direction | I | F | C | Force direction correctness | Vector analysis | Buoyancy: up, Marangoni: hot→cold, ST: inward |
| test_cfl_limiting_effectiveness | I | M | C | CFL limiter prevents instability | With/without limiter | Stable with limiter |
| test_cfl_limiting_conservation | I | M | H | CFL limiter preserves conservation | Energy/mass tracking | Conservation maintained |
| test_extreme_gradients | I | M | H | Stability under extremes | Steep gradients | No divergence |
| test_vof_subcycling_convergence | I | S | H | VOF subcycling accuracy | 1, 5, 10 subcycles | Converges with more subcycles |
| test_subcycling_1_vs_10 | I | M | H | Direct subcycle comparison | Accuracy improvement | 10 subcycles more accurate |
| test_unit_conversion_roundtrip | I | F | H | Unit conversion consistency | Phys→lattice→phys | Roundtrip error < 1e-6 |
| test_unit_conversion_consistency | I | F | H | Conversion across modules | All modules consistent | No conversion mismatch |
| test_deterministic | I | M | N | Deterministic behavior | Multiple runs | Identical results |
| test_nan_detection | I | F | H | NaN detection system | Inject NaN | Detection works, simulation stops |
| test_minimal_config | I | F | N | Minimal configuration runs | Simplest setup | Completes without error |
| test_known_good_output | I | M | N | Known good baseline | Reference output | Matches baseline |
| test_melt_pool_dimensions | I | M | N | Realistic melt pool size | Literature comparison | Depth/width in realistic range |
| test_high_power_laser | I | M | N | High laser power stability | 200W+ laser | Stable, realistic behavior |
| test_rapid_solidification | I | M | N | Rapid cooling phase change | Laser shutoff | Solidification rate realistic |
| test_steady_state_flow | I | M | N | Steady state flow achievement | Long run | Reaches steady state |
| test_steady_state_temperature | I | M | N | Steady state temperature | Long run | Reaches thermal steady state |
| test_disable_marangoni | I | M | N | Disable Marangoni module | Feature toggle | Runs without Marangoni |
| test_disable_vof | I | M | N | Disable VOF module | Feature toggle | Runs without VOF |

---

## 7. Energy Conservation Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_energy_conservation_no_source | I | F | C | Energy conservation dE/dt = 0 | Isolated system | Drift < 1% per 1000 steps |
| test_energy_conservation_laser_only | I | M | C | Energy balance with laser | ΔE = Q_in - Q_out | Balance error < 5% |
| test_energy_conservation_full | I | M | C | Full energy accounting | All sources/sinks | Balance error < 10% |
| test_evaporation_energy_balance | I | M | H | Evaporative cooling Q = J·L_v | Latent heat removal | Cooling rate matches theory |
| test_evaporation_hertz_knudsen | V | F | H | Hertz-Knudsen formula | J = α_e P_sat/√(2πRT) | Mass flux error < 10% |
| test_energy_conservation_timestep | V | S | H | Energy conservation vs timestep | Convergence study | Error decreases with smaller dt |
| test_bug3_energy_diagnostic | R | M | H | Energy diagnostic dt-scaling | Bug 3 regression | Fine dt → best error (not worst) |
| test_evaporation_rate | V | M | N | Evaporation rate validation | Mass flux measurement | Rate matches formula |

---

## 8. Stability & Validation Test Matrix

| Test Name | Type | Speed | Priority | Physics Validated | Method | Pass Criteria |
|-----------|------|-------|----------|-------------------|--------|---------------|
| test_cfl_stability | V | F | H | CFL condition u·dt/dx < CFL_max | Analytical check | Timestep selection correct |
| test_flux_limiter_overhead | U | F | N | Performance overhead | Benchmark throughput | Overhead < 20% |
| test_realistic_temperature | R | M | N | Realistic T range | Temperature tracking | T stays in physical range |
| test_substrate_bc_stability | R | M | H | Substrate BC stability | Long-term run | No oscillations, correct cooling |
| test_grid_convergence | V | S | H | Grid independence | Three grid sizes | Results converge with finer grid |
| test_timestep_convergence | V | S | H | Timestep independence | Three timestep sizes | Results converge with smaller dt |
| test_regression_50W | R | S | H | 50W baseline regression | Known good case | Matches baseline within tolerance |
| test_week3_readiness | V | M | N | Comprehensive readiness | Scoring system | Score ≥ 85/100 |
| test_realistic_lpbf_initial_conditions | V | M | H | LPBF starting conditions | T_init ≈ 300K | Starts cold, heats with laser |
| test_laser_shutoff_configurable | V | M | H | Laser shutoff parameter | Timed shutoff | Laser turns off at specified time |
| test_config_parser | V | M | N | Configuration parsing | Various configs | Correct parameter loading |
| test_powder_bed_generation | V | M | N | Powder bed initialization | Random packing | Realistic packing density |

---

## Test Coverage Summary by Physics Module

| Module | Unit Tests | Integration Tests | Validation Tests | Total | Critical Tests |
|--------|-----------|-------------------|------------------|-------|----------------|
| VOF Solver | 13 | 3 | 3 | 19 | 2 |
| Marangoni | 8 | 2 | 1 | 11 | 2 |
| Thermal | 6 | 4 | 4 | 14 | 4 |
| Fluid LBM | 7 | 3 | 2 | 12 | 4 |
| Phase Change | 5 | 1 | 2 | 8 | 1 |
| Multiphysics | 0 | 25 | 0 | 25 | 8 |
| Energy | 0 | 5 | 3 | 8 | 3 |
| Stability | 3 | 5 | 6 | 14 | 5 |
| **TOTAL** | **42** | **48** | **21** | **111** | **29** |

---

## Critical Test Dependencies

### Thermal Stability Chain
```
test_flux_limiter (MUST PASS)
    ↓
test_temperature_bounds (MUST PASS)
    ↓
test_omega_reduction (MUST PASS)
    ↓
test_high_pe_stability (MUST PASS)
```

If any fail → Thermal solver UNSTABLE → DO NOT MERGE

### VOF Mass Conservation Chain
```
test_divergence_free (MUST PASS)
    ↓ (incompressibility required)
test_vof_mass_conservation (MUST PASS)
    ↓ (mass conservation required)
test_vof_advection_* (should pass)
```

### Energy Balance Chain
```
test_energy_conservation_no_source (MUST PASS)
    ↓ (base conservation)
test_energy_conservation_laser_only (MUST PASS)
    ↓ (add laser)
test_energy_conservation_full (should pass)
```

### Force Balance Chain
```
test_force_direction (MUST PASS)
    ↓ (correct directions)
test_force_magnitude_ordering (should pass)
    ↓ (realistic magnitudes)
test_force_balance_static (should pass)
```

---

## Physics Equations Tested

### VOF Solver
1. **Advection**: ∂f/∂t + ∇·(f·u) = 0
2. **Curvature**: κ = ∇·n where n = -∇f/|∇f|
3. **Surface Tension**: F = σκn
4. **Evaporation**: df/dt = -J/(ρ·dx)
5. **Mass Conservation**: M = ∫f dV = constant

### Marangoni Effect
1. **Force**: F = |dσ/dT| × |∇T_tangential| × |∇f|
2. **Material**: dσ/dT ≈ -0.00026 N/(m·K) for Ti6Al4V

### Thermal Solver
1. **Diffusion**: ∂T/∂t = α∇²T
2. **LBM**: ∂g_i/∂t + c_i·∇g_i = -(g_i - g_i^eq)/τ
3. **Tau-Diffusivity**: α = (τ - 0.5)cs²
4. **Stefan Problem**: Interface position vs time

### Fluid Solver
1. **Navier-Stokes**: ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u + F
2. **Incompressibility**: ∇·u = 0
3. **LBM**: ∂f_i/∂t + c_i·∇f_i = -(f_i - f_i^eq)/τ
4. **Poiseuille**: u(y) = (dp/dx)·y(H-y)/(2μ)

### Phase Change
1. **Enthalpy**: H = c_p T + f_l L
2. **Liquid Fraction**: f_l = (T - T_s)/(T_l - T_s) for T_s < T < T_l
3. **Darcy Damping**: F = -C(1 - f_l)² u

### Energy Conservation
1. **Balance**: dE/dt = Q_laser - Q_cond - Q_rad - Q_evap
2. **Evaporation**: Q_evap = J_evap × L_v
3. **Hertz-Knudsen**: J = α_e × P_sat(T)/√(2πRT)

---

## Test Execution Strategy

### Daily Development (< 1 min)
```bash
# Run critical stability tests
ctest -R "flux_limiter|temperature_bounds|omega_reduction"
```

### Before Commit (< 2 min)
```bash
# Add unit tests for changed module
ctest -R "flux_limiter|temperature_bounds|omega_reduction|vof_mass_conservation|divergence_free"
```

### Before Push (< 10 min)
```bash
# Add integration tests
ctest -R "stability|high_pe|test_.*_coupling|energy_conservation_no_source"
```

### Nightly/Release (30-60 min)
```bash
# Full validation suite
./RUN_ALL_PHYSICS_TESTS.sh
```

---

## Test Priority Matrix

### P0 - Critical (Block Merge)
- test_flux_limiter
- test_temperature_bounds
- test_omega_reduction
- test_high_pe_stability
- test_divergence_free
- test_vof_mass_conservation
- test_energy_conservation_no_source
- test_force_direction
- test_thermal_fluid_coupling
- test_vof_fluid_coupling
- test_thermal_vof_coupling
- test_phase_fluid_coupling

### P1 - High (Block Release)
- All unit tests
- test_marangoni_flow
- test_poiseuille_flow
- test_pure_conduction
- test_stefan_problem
- test_evaporation_energy_balance
- test_phase_change_robustness
- test_cfl_stability

### P2 - Normal (Validation/Documentation)
- All validation tests
- Benchmark tests
- Regression tests
- Performance tests

---

**Test Matrix Version 1.0**
**Last Updated**: 2025-12-02
