# Benchmark Comparison Test Plan: LBM-CUDA vs waLBerla

**Date:** 2025-11-22
**Version:** 1.0
**Author:** Validation Test Specialist

---

## 1. Executive Summary

This document defines a comprehensive validation test suite to benchmark the LBM-CUDA thermal solver against both analytical solutions and the waLBerla framework. The focus is on a **simple flat plate with laser heating** (no powder bed) to establish baseline accuracy before LPBF complexity.

### Key Objectives
1. Validate thermal LBM accuracy against analytical solutions
2. Compare performance and accuracy with waLBerla
3. Establish grid convergence (2nd order accuracy)
4. Verify energy conservation

---

## 2. Existing Test Coverage Analysis

### 2.1 Current LBMProject Validation Tests

| Test | File | Status | Analytical Solution |
|------|------|--------|---------------------|
| 1D Pure Conduction | `tests/validation/test_pure_conduction.cu` | Exists | Gaussian diffusion |
| Stefan Problem | `tests/validation/test_stefan_problem.cu` | Exists | Phase change front |
| Grid Convergence | `tests/validation/test_grid_convergence.cu` | Exists | Relative error |
| Energy Conservation | `tests/unit/thermal/test_thermal_lbm.cu` | Exists | Sum(T) constant |

### 2.2 Current LBMProject Unit Tests (Thermal)

| Test | Coverage |
|------|----------|
| `test_lattice_d3q7.cu` | D3Q7 lattice operations |
| `test_thermal_lbm.cu` | Solver initialization, diffusion, advection |
| `test_laser_source.cu` | Gaussian distribution, Beer-Lambert law |

### 2.3 waLBerla Reference Tests

| Test | Location | Relevance |
|------|----------|-----------|
| DiffusionTest | `tests/lbm/DiffusionTest.cpp` | Advection-diffusion LBM |
| Thermocapillary | `apps/showcases/Thermocapillary/` | Thermal + Marangoni |

---

## 3. Validation Test Suite Design

### Test 1: 1D Heat Conduction (Analytical Solution)

**Purpose:** Validate thermal diffusion without advection

**Analytical Solution:**
```
T(x,t) = T_amb + (T_peak - T_amb) * sqrt(t0/(t+t0)) * exp(-x^2/(4*alpha*(t+t0)))
```

**Configuration:**
```yaml
domain:
  nx: 200
  ny: 1
  nz: 1
  dx: 2.0e-6  # 2 um

physics:
  T_ambient: 300.0  # K
  T_peak: 1943.0    # K (Ti6Al4V melting point)
  thermal_diffusivity: 9.05e-6  # m^2/s (Ti6Al4V)

test_times: [0.1e-3, 0.5e-3, 1.0e-3]  # seconds
```

**Success Criteria:**
- L2 norm error < 5% at each test time
- Energy conservation < 0.1%

**Existing Implementation:** `/home/yzk/LBMProject/tests/validation/test_pure_conduction.cu`

---

### Test 2: 2D/3D Thermal Diffusion Convergence

**Purpose:** Verify 2nd order spatial accuracy

**Setup:**
- Initial condition: Gaussian temperature pulse at center
- Run on 3 grid resolutions: 4um, 2um, 1um
- Compare at fixed physical time

**Analytical Solution:**
```
T(r,t) = T_amb + Q/(4*pi*alpha*t) * exp(-r^2/(4*alpha*t))
```

**Grid Convergence Ratio:**
```
Order = log2(error_coarse / error_baseline) / log2(2)
```

**Success Criteria:**
- Convergence order >= 1.9 (expect 2.0 for LBM)
- Richardson extrapolation error < 2%

**Configuration Files:**
- `/home/yzk/LBMProject/configs/grid_convergence/thermal_2D_4um.conf`
- `/home/yzk/LBMProject/configs/grid_convergence/thermal_2D_2um.conf`
- `/home/yzk/LBMProject/configs/grid_convergence/thermal_2D_1um.conf`

---

### Test 3: Stefan Problem (Phase Change)

**Purpose:** Validate phase change with moving boundary

**Analytical Solution:**
```
Front position: s(t) = 2 * lambda * sqrt(alpha * t)

where lambda solves: lambda * exp(lambda^2) * erf(lambda) = St / sqrt(pi)
Stefan number: St = cp * dT / L_fusion
```

**For Ti6Al4V:**
- dT = T_liquidus - T_solidus = 100 K
- cp = 546 J/(kg*K)
- L_fusion = 286,000 J/kg
- St = 0.191
- lambda = 0.139

**Configuration:**
```yaml
domain:
  nx: 500
  ny: 1
  nz: 1
  length: 2000e-6  # 2 mm

boundary:
  x_min: fixed_temperature, T_liquidus
  x_max: adiabatic

phase_change: enabled
```

**Success Criteria:**
- Front position error < 5% at t = 0.5, 1.0, 2.0 ms
- Latent heat storage > 0 (phase change working)

**Existing Implementation:** `/home/yzk/LBMProject/tests/validation/test_stefan_problem.cu`

---

### Test 4: Gaussian Heat Source Response

**Purpose:** Validate laser heat source coupling with thermal solver

**Setup:**
- Flat plate substrate
- Gaussian laser beam (no movement)
- Compare steady-state temperature with analytical/semi-analytical solution

**Semi-Analytical Solution (3D steady state):**
```
For surface point source: T_max = P * absorptivity / (2 * pi * k * r_spot)
```

**Configuration:**
```yaml
domain:
  nx: 128
  ny: 128
  nz: 64
  dx: 2.0e-6

laser:
  power: 50.0  # W
  spot_radius: 50.0e-6  # 50 um
  absorptivity: 0.35
  position: [center, center, 0]
  velocity: [0, 0, 0]  # stationary

physics:
  k_solid: 21.9  # W/(m*K)
  material: Ti6Al4V
```

**Success Criteria:**
- Peak temperature within 10% of semi-analytical
- Energy input = absorbed power (within 5%)
- Temperature distribution Gaussian-like at surface

---

## 4. Comparison Metrics Framework

### 4.1 Error Metrics

```python
# L2 Norm (global accuracy)
L2_error = sqrt(sum((T_numerical - T_analytical)^2) / sum(T_analytical^2))

# L_inf Norm (worst case error)
Linf_error = max(abs(T_numerical - T_analytical)) / max(T_analytical)

# RMS Error
RMS_error = sqrt(mean((T_numerical - T_analytical)^2))
```

### 4.2 Grid Convergence Metrics

```python
# Richardson extrapolation
p = log2(error_h / error_h2)  # Convergence order

# Grid Convergence Index (GCI)
GCI = Fs * abs(f_h2 - f_h) / (r^p - 1)
# where Fs = 1.25 (safety factor), r = refinement ratio
```

### 4.3 Conservation Metrics

```python
# Energy conservation
energy_initial = sum(T * rho * cp * dx * dy * dz)
energy_final = sum(T * rho * cp * dx * dy * dz)
energy_input = integral(P * absorptivity * dt)
energy_loss = integral(heat_flux_boundaries * dt)

conservation_error = abs(energy_final - energy_initial - energy_input + energy_loss) / energy_input
```

---

## 5. waLBerla Comparison Approach

### 5.1 Equivalent Parameter Mapping

| LBMProject | waLBerla | Conversion |
|------------|----------|------------|
| `thermal_diffusivity` | `d` | Same (m^2/s) |
| `dx` | `dx` | Same (m) |
| `dt` | `dt` | Must match: dt = dt_lattice * (dx^2 / alpha) |
| `tau_T` | `tau` | Same: tau = 3*D + 0.5 where D = alpha*dt/dx^2 |
| `omega_T` | `omega` | omega = 1/tau |

### 5.2 Boundary Condition Equivalence

| BC Type | LBMProject | waLBerla |
|---------|------------|----------|
| Periodic | Default streaming | `periodic: (1, 1, 1)` |
| Dirichlet | `applyBoundaryConditions(1, T)` | `DiffusionDirichletStatic` |
| Neumann | Not fully implemented | `NeumannBC` |
| No-slip | `BounceBack` | `NoSlip` |

### 5.3 Time Stepping Comparison

**LBMProject:**
```cpp
// Explicit steps
solver.collisionBGK();
solver.streaming();
solver.computeTemperature();
```

**waLBerla:**
```cpp
// Uses sweeps in timeloop
timeloop.add() << Sweep(lbm::makeCellwiseAdvectionDiffusionSweep(...));
```

---

## 6. Test Implementation Scripts

### 6.1 Run Both Codes Script

**File:** `/home/yzk/LBMProject/scripts/benchmark/run_comparison.sh`

```bash
#!/bin/bash
# Benchmark comparison: LBM-CUDA vs waLBerla

TEST_NAME=$1
GRID_SIZE=$2

echo "=========================================="
echo "Benchmark: $TEST_NAME"
echo "Grid: ${GRID_SIZE}um"
echo "=========================================="

# Run LBM-CUDA
echo "Running LBM-CUDA..."
cd /home/yzk/LBMProject/build
./run_simulation ../configs/benchmark/${TEST_NAME}_${GRID_SIZE}um.conf \
    > /tmp/lbmcuda_${TEST_NAME}_${GRID_SIZE}um.log 2>&1

# Run waLBerla (if configured)
echo "Running waLBerla..."
cd /home/yzk/walberla/build
./apps/benchmarks/DiffusionTest \
    -d ${DIFFUSIVITY} -dx ${DX} -dt ${DT} -t ${SIM_TIME} \
    > /tmp/walberla_${TEST_NAME}_${GRID_SIZE}um.log 2>&1

# Extract and compare
echo "Comparing results..."
python3 /home/yzk/LBMProject/scripts/benchmark/compare_results.py \
    /tmp/lbmcuda_${TEST_NAME}_${GRID_SIZE}um.log \
    /tmp/walberla_${TEST_NAME}_${GRID_SIZE}um.log \
    --analytical ${ANALYTICAL_SOLUTION}
```

### 6.2 Python Comparison Script

**File:** `/home/yzk/LBMProject/scripts/benchmark/compare_results.py`

```python
#!/usr/bin/env python3
"""
Compare LBM-CUDA and waLBerla simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def compute_L2_error(numerical, analytical):
    """Compute L2 norm error"""
    num = np.sum((numerical - analytical)**2)
    den = np.sum(analytical**2)
    return np.sqrt(num / den)

def compute_grid_convergence_order(errors, grid_sizes):
    """Compute convergence order from multiple grid levels"""
    ratios = grid_sizes[:-1] / grid_sizes[1:]
    orders = np.log(errors[:-1] / errors[1:]) / np.log(ratios)
    return orders

def analytical_1d_diffusion(x, t, T_peak, T_amb, alpha, t0=1e-5):
    """Analytical solution for 1D Gaussian diffusion"""
    t_eff = t + t0
    spatial = np.exp(-x**2 / (4 * alpha * t_eff))
    temporal = np.sqrt(t0 / t_eff)
    return T_amb + (T_peak - T_amb) * temporal * spatial

def analytical_stefan_front(t, St, alpha):
    """Analytical melting front position for Stefan problem"""
    # Solve lambda * exp(lambda^2) * erf(lambda) = St / sqrt(pi)
    from scipy.special import erf
    from scipy.optimize import fsolve

    def stefan_eq(lam):
        return lam * np.exp(lam**2) * erf(lam) - St / np.sqrt(np.pi)

    lam = fsolve(stefan_eq, 0.15)[0]
    return 2 * lam * np.sqrt(alpha * t)

def plot_comparison(x, T_lbmcuda, T_walberla, T_analytical, output_file):
    """Generate comparison plot"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Temperature profiles
    ax1 = axes[0]
    ax1.plot(x * 1e6, T_analytical, 'k-', label='Analytical', linewidth=2)
    ax1.plot(x * 1e6, T_lbmcuda, 'b--', label='LBM-CUDA', linewidth=1.5)
    if T_walberla is not None:
        ax1.plot(x * 1e6, T_walberla, 'r:', label='waLBerla', linewidth=1.5)
    ax1.set_xlabel('x (um)')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Temperature Profile Comparison')

    # Error plot
    ax2 = axes[1]
    error_lbmcuda = np.abs(T_lbmcuda - T_analytical)
    ax2.semilogy(x * 1e6, error_lbmcuda, 'b-', label='LBM-CUDA error')
    if T_walberla is not None:
        error_walberla = np.abs(T_walberla - T_analytical)
        ax2.semilogy(x * 1e6, error_walberla, 'r-', label='waLBerla error')
    ax2.set_xlabel('x (um)')
    ax2.set_ylabel('Absolute Error (K)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Error Distribution')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare LBM simulation results')
    parser.add_argument('lbmcuda_output', help='LBM-CUDA output file (VTK or CSV)')
    parser.add_argument('walberla_output', nargs='?', help='waLBerla output file')
    parser.add_argument('--analytical', choices=['diffusion', 'stefan', 'gaussian'],
                       help='Analytical solution type')
    parser.add_argument('--output', default='comparison.png', help='Output plot file')
    args = parser.parse_args()

    print("="*50)
    print("LBM-CUDA vs waLBerla Comparison")
    print("="*50)

    # Load data and compute metrics (placeholder)
    print(f"LBM-CUDA file: {args.lbmcuda_output}")
    print(f"waLBerla file: {args.walberla_output}")
    print(f"Analytical: {args.analytical}")

    # Example output format
    print("\nResults Summary:")
    print("-"*50)
    print(f"{'Metric':<25} {'LBM-CUDA':<15} {'waLBerla':<15}")
    print("-"*50)
    print(f"{'L2 Error (%)':<25} {'TBD':<15} {'TBD':<15}")
    print(f"{'L_inf Error (K)':<25} {'TBD':<15} {'TBD':<15}")
    print(f"{'Energy Conservation (%)':<25} {'TBD':<15} {'TBD':<15}")
    print(f"{'Runtime (s)':<25} {'TBD':<15} {'TBD':<15}")

if __name__ == '__main__':
    main()
```

---

## 7. Potential Issues and Mitigations

### 7.1 Unit Conversion Differences

**Issue:** LBMProject uses SI units internally with conversion to lattice units. waLBerla may use different conventions.

**Mitigation:**
1. Document unit systems explicitly
2. Verify tau computation matches: `tau = alpha_lattice / cs2 + 0.5`
3. Cross-check with known analytical solutions

### 7.2 Boundary Condition Implementations

**Issue:** BC implementations may differ subtly (ghost layer treatment, boundary position).

**Mitigation:**
1. Use periodic BCs for core validation (no ambiguity)
2. Test Dirichlet BCs separately with simple 1D case
3. Compare boundary flux calculations

### 7.3 Time Stepping Schemes

**Issue:** LBMProject uses explicit BGK; waLBerla has multiple options (SRT, MRT, RK).

**Mitigation:**
1. Force waLBerla to use SRT (`omega` based) for fair comparison
2. Match CFL conditions
3. Run to steady state for time-independent comparison

### 7.4 Initial Condition Setup

**Issue:** Equilibrium initialization may differ.

**Mitigation:**
1. Initialize with known analytical solution at t > 0
2. Compare after sufficient relaxation time
3. Use simple step/Gaussian profiles

---

## 8. Configuration Files for Benchmark

### 8.1 Pure Conduction Benchmark

**File:** `/home/yzk/LBMProject/configs/benchmark/pure_conduction_2um.conf`

```conf
# Pure 1D Heat Conduction Benchmark
# Compare with analytical Gaussian diffusion solution

[domain]
nx = 200
ny = 1
nz = 1
dx = 2.0e-6

[physics]
material = Ti6Al4V
thermal_diffusivity = 9.05e-6

[initial_condition]
type = gaussian
T_ambient = 300.0
T_peak = 1943.0
x_center = 200.0e-6
sigma = 20.0e-6

[boundary]
x_min = periodic
x_max = periodic

[simulation]
dt_multiplier = 0.1
num_steps = 5000
output_interval = 500

[output]
vtk_enabled = true
csv_enabled = true
```

### 8.2 Gaussian Heat Source Benchmark

**File:** `/home/yzk/LBMProject/configs/benchmark/gaussian_source_2um.conf`

```conf
# Gaussian Laser Heat Source Benchmark
# Flat plate, no powder, stationary laser

[domain]
nx = 128
ny = 128
nz = 64
dx = 2.0e-6

[physics]
material = Ti6Al4V
enable_phase_change = false

[laser]
power = 50.0
spot_radius = 50.0e-6
absorptivity = 0.35
penetration_depth = 10.0e-6
x_position = 128.0e-6
y_position = 128.0e-6
velocity_x = 0.0
velocity_y = 0.0

[initial_condition]
type = uniform
T_initial = 300.0

[boundary]
z_min = fixed_temperature
z_min_value = 300.0
all_other = adiabatic

[simulation]
dt_multiplier = 0.05
num_steps = 10000
output_interval = 1000
```

---

## 9. waLBerla Configuration Equivalent

### 9.1 DiffusionTest Parameters

```bash
# waLBerla DiffusionTest equivalent for pure conduction
./DiffusionTest \
    -d 9.05e-6 \           # thermal diffusivity (m^2/s)
    -dx 2.0e-6 \           # grid spacing (m)
    -dt 4.4e-9 \           # time step (s), computed from CFL
    -t 0.001 \             # simulation time (s)
    -dim 0 \               # x-direction
    -err 0.05 \            # max allowed error
    --vtk                  # enable VTK output
```

### 9.2 Thermocapillary Scenario (Reference)

The waLBerla Thermocapillary showcase provides a reference for coupled thermal-Marangoni flow. Key parameters from `/home/yzk/walberla/apps/showcases/Thermocapillary/microchannel2D.py`:

```python
# Relevant parameters for comparison
kappa_h = 0.2           # thermal conductivity (liquid)
kappa_l = 0.04 or 0.2   # thermal conductivity (gas/solid)
temperature_ref = 10
temperature_h = 20
temperature_l = 4
sigma_t = -5e-4         # surface tension temperature coefficient
```

---

## 10. Execution Schedule

### Phase 1: Analytical Validation (Week 1)

| Day | Task |
|-----|------|
| 1 | Run existing pure conduction test, verify results |
| 2 | Run Stefan problem test, compare front position |
| 3 | Implement Gaussian heat source benchmark |
| 4 | Grid convergence study (3 levels) |
| 5 | Document results, compute convergence order |

### Phase 2: waLBerla Comparison (Week 2)

| Day | Task |
|-----|------|
| 1 | Set up waLBerla DiffusionTest with matching parameters |
| 2 | Run parallel tests on same hardware |
| 3 | Extract and compare temperature fields |
| 4 | Compute error metrics |
| 5 | Generate comparison plots and report |

---

## 11. Expected Results Summary

### Test 1: Pure Conduction
- Expected L2 error: < 5%
- Expected convergence order: 2.0
- Energy conservation: < 0.1%

### Test 2: Grid Convergence
- Expected order: 1.9 - 2.0
- GCI < 5%

### Test 3: Stefan Problem
- Front position error: < 5%
- Latent heat correctly tracked

### Test 4: Gaussian Source
- Peak T within 10% of semi-analytical
- Energy balance within 5%

### waLBerla Comparison
- Temperature profiles within 5% (same physics)
- Performance: LBM-CUDA faster on GPU hardware

---

## 12. Appendix: Key File Locations

### LBMProject Files
```
Tests:
  /home/yzk/LBMProject/tests/validation/test_pure_conduction.cu
  /home/yzk/LBMProject/tests/validation/test_stefan_problem.cu
  /home/yzk/LBMProject/tests/validation/test_grid_convergence.cu
  /home/yzk/LBMProject/tests/unit/thermal/test_thermal_lbm.cu
  /home/yzk/LBMProject/tests/unit/laser/test_laser_source.cu

Source:
  /home/yzk/LBMProject/src/physics/thermal/
  /home/yzk/LBMProject/include/physics/thermal_lbm.h
  /home/yzk/LBMProject/include/physics/laser_source.h

Configs:
  /home/yzk/LBMProject/configs/validation/
  /home/yzk/LBMProject/configs/grid_convergence/
```

### waLBerla Files
```
Tests:
  /home/yzk/walberla/tests/lbm/DiffusionTest.cpp

Showcases:
  /home/yzk/walberla/apps/showcases/Thermocapillary/
  /home/yzk/walberla/apps/showcases/Thermocapillary/microchannel2D.py
  /home/yzk/walberla/apps/showcases/Thermocapillary/thermocapillary.cpp
```

---

## 13. References

1. Qian, Y. H., et al. (1992). Lattice BGK models for Navier-Stokes equation
2. He, X., & Luo, L. S. (1997). Theory of the lattice Boltzmann method
3. Chopard, B., et al. (2009). The lattice Boltzmann advection-diffusion model revisited
4. waLBerla documentation: https://walberla.net
5. LPBF validation: Khairallah et al. (2016)
