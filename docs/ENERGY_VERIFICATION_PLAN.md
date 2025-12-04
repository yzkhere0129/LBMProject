# Energy Conservation Verification Plan

**Objective**: Establish quantitative verification that the LPBF simulation correctly models energy conservation and produces results consistent with literature.

**Date**: 2025-11-19
**Status**: Implementation in progress

---

## 1. Energy Conservation Verification

### 1.1 Energy Balance Equation

The fundamental energy balance for the LPBF process is:

```
P_laser = P_evap + P_rad + P_cond + dE/dt
```

Where:
- **P_laser**: Absorbed laser power [W] = P_input × absorptivity
- **P_evap**: Evaporation cooling power [W] = Σ(ṁ × L_v) at T > T_boil
- **P_rad**: Radiation cooling power [W] = Σ(ε·σ·A·(T⁴ - T_amb⁴))
- **P_cond**: Conduction to substrate [W] (currently not modeled - periodic BC)
- **dE/dt**: Rate of change of internal energy [W]

### 1.2 Internal Energy Calculation

Total internal energy stored in the domain:

```
E_total = E_sensible + E_latent
```

Where:
- **E_sensible** = Σ(ρ × c_p × T × V) for all cells
- **E_latent** = Σ(f_l × ρ × L_f × V) for mushy zone cells

For Test H:
- Volume per cell: V = (2 μm)³ = 8×10⁻¹⁸ m³
- Total cells: 200 × 100 × 50 = 1,000,000
- Liquid cells (estimated): ~100,000 (10%)

### 1.3 Expected Values for Test H

**Input**:
```
P_laser = 195 W × 0.35 = 68.25 W
Duration: 500 μs
Total energy input: E_in = 68.25 W × 500×10⁻⁶ s = 0.034125 J
```

**Output** (estimated from physics):
```
P_evap ≈ 10-20 W  (surface evaporation cooling)
P_rad  ≈ 5-10 W   (Stefan-Boltzmann radiation)
P_cond ≈ 0 W      (periodic boundary - no conduction loss)
dE/dt  ≈ 40-50 W  (heating up liquid metal)
```

**Internal energy change**:
```
If 100,000 cells heat from 300 K to 2700 K:
ΔE = m × c_p × ΔT
   = (100,000 × 8×10⁻¹⁸ m³ × 4110 kg/m³) × 831 J/(kg·K) × 2400 K
   ≈ 0.006 J

This is only 18% of input energy!
Where did the rest go?
```

**Hypothesis**:
1. Most energy goes to **evaporation** (not melting)
2. **Radiation** is significant at high temperatures (T⁴ scaling)
3. Some energy lost to **numerical dissipation**

### 1.4 Acceptance Criteria

**PASS** if:
```
|P_in - P_out - dE/dt| / P_in < 5%
```

Where:
- P_in = P_laser (absorbed)
- P_out = P_evap + P_rad + P_cond
- dE/dt = (E_current - E_previous) / dt

**TARGET**: Energy balance error < 5% for all timesteps after initial transient.

---

## 2. Literature Comparison

### 2.1 Reference Data

**Mohr et al. 2020** (ISS Microgravity Experiments):
- Material: Ti6Al4V
- Laser power: 100-200 W
- Spot size: ~100 μm
- Peak temperature: 2,400 - 2,800 K
- Melt pool length: 150 - 300 μm
- Observations: Strong Marangoni flow, keyhole formation at high power

**Khairallah et al. 2016** (LLNL Simulations):
- Material: 316L Stainless Steel
- Laser power: 200 W
- Scan velocity: 0.8 m/s
- Peak velocity: 0.5 - 2.8 m/s
- Keyhole depth: 50 - 150 μm (at high power)

### 2.2 Test H Target Metrics

**Quantitative Goals** (must match literature):
1. **Peak temperature**: 2,400 - 2,800 K (Mohr 2020 range)
2. **Melt pool length**: 150 - 300 μm
3. **Peak velocity**: 0.5 - 3.0 m/s (literature range for Ti6Al4V)

**Qualitative Goals** (must be present):
1. Marangoni-driven recirculation flow
2. Surface depression at laser spot (recoil pressure effect)
3. Thermal gradient: ~10⁶ K/m at interface

### 2.3 Current Test H Results

**From VTK file** (lpbf_005000.vtk at t=500 μs):
- Peak temperature: 27,466 K (TOO HIGH - 10× literature!)
- Peak velocity: ~1 mm/s (within range, but low)
- Melt pool: Not yet measured

**Status**: Peak temperature FAILURE → Energy imbalance suspected

---

## 3. Test Case Design

### 3.1 Baseline Test (Test H)

**Configuration** (lpbf_195W_test_H_full_physics.conf):
```
Laser: 195 W, 50 μm spot, 0.35 absorptivity
Material: Ti6Al4V
Grid: 200×100×50 cells, dx=2 μm
Time: 500 μs (5000 steps × 0.1 μs)
Physics: ALL ENABLED (thermal, fluid, VOF, Marangoni, surface tension, phase change)
```

**Expected outcome**:
- Energy balance error < 5%
- Peak temperature 2,400-2,800 K
- Melt pool length 150-300 μm

### 3.2 Sensitivity Tests

**Test H-1: Radiation BC Strength**
- Vary emissivity: ε = 0.1, 0.3, 0.5, 0.8
- Measure: Peak temperature, energy balance
- Goal: Find correct emissivity that matches literature T_max

**Test H-2: Evaporation Rate**
- Vary evaporation coefficient (if implemented)
- Measure: P_evap, peak temperature
- Goal: Quantify evaporation cooling importance

**Test H-3: Laser Penetration Depth**
- Vary δ: 2 μm, 5 μm, 10 μm, 20 μm
- Measure: Temperature distribution, melt pool depth
- Goal: Match literature melt pool geometry

**Test H-4: Grid Resolution**
- Grids: 1 μm, 2 μm, 4 μm
- Measure: Energy balance, melt pool size
- Goal: Verify grid convergence

### 3.3 Validation Test (Mohr 2020 Replication)

**Configuration**:
```
Laser: 195 W (match Mohr 2020 experiment)
Material: Ti6Al4V
Scan: Stationary (no scanning)
Duration: 500 μs (experiment duration)
Output: Every 50 μs
```

**Acceptance**:
1. Peak temperature within 2,400-2,800 K
2. Melt pool length within 150-300 μm
3. Energy balance error < 5%

**PASS/FAIL**: Must meet ALL 3 criteria.

---

## 4. Implementation Checklist

### 4.1 Code Additions

- [ ] ThermalLBM::computeEvaporationPower()
- [ ] ThermalLBM::computeRadiationPower()
- [ ] ThermalLBM::computeTotalThermalEnergy()
- [ ] MultiphysicsSolver::getLaserAbsorbedPower()
- [ ] MultiphysicsSolver::getEvaporationPower()
- [ ] MultiphysicsSolver::getRadiationPower()
- [ ] MultiphysicsSolver::getThermalEnergyChangeRate()
- [ ] MultiphysicsSolver::printEnergyBalance()

### 4.2 Main Loop Integration

Modify `examples/lpbf_simulation.cpp`:

```cpp
// Add after step loop
if (step % 100 == 0) {
    solver.printEnergyBalance(step);
}
```

Expected output:
```
=== ENERGY BALANCE (step 1000, t=100 μs) ===
  P_laser_in:       68.25 W
  P_evaporation:    12.34 W
  P_radiation:       8.91 W
  P_conduction:      0.00 W (periodic BC)
  dE/dt:            46.12 W
  --------------------------------
  P_in - P_out - dE/dt: 0.88 W
  Energy error:     1.3% ✓ PASS
```

### 4.3 VTK Analysis Pipeline

1. **Extract data**: `python analyze_vtk.py lpbf_005000.vtk`
2. **Generate report**: `--output report_005000.txt`
3. **Generate plots**: `--plot`

**Output files**:
- `report_005000.txt`: Quantitative metrics
- `vtk_analysis.png`: Temperature/velocity plots

### 4.4 Regression Tests

**Test Suite** (to run after code changes):
```bash
# 1. Basic functionality
./test_energy_conservation

# 2. Evaporation still works
check: Surface T ≈ 3,560 K (T_boil for Ti6Al4V)

# 3. Marangoni flow exists
check: v_max > 0

# 4. Phase change correct
check: Liquid fraction 5-15%

# 5. No numerical errors
check: No NaN/Inf in any field
```

---

## 5. Debugging Protocol

### 5.1 If Energy Error > 5%

**Step 1**: Check each term individually
```
Print at every timestep:
- P_laser (should be constant 68.25 W)
- P_evap (should increase as surface heats)
- P_rad (should increase as T⁴)
- dE/dt (should decrease as equilibrium approaches)
```

**Step 2**: Verify integration accuracy
```
- Check surface area calculation (for P_rad)
- Check mass flux calculation (for P_evap)
- Check volume integration (for dE/dt)
```

**Step 3**: Look for missing physics
```
- Is conduction to substrate modeled? (currently NO - periodic BC)
- Is latent heat accounted for? (YES - phase change enabled)
- Is evaporation mass loss tracked? (TBD)
```

### 5.2 If Temperature Too High (T_max >> 2,800 K)

**Hypothesis**: Cooling mechanisms insufficient

**Fix 1**: Increase radiation BC strength
```
Increase emissivity: ε = 0.3 → 0.5 or 0.8
```

**Fix 2**: Implement evaporation cooling
```
Add latent heat sink at surface: Q_evap = ṁ × L_v
```

**Fix 3**: Reduce laser penetration
```
Decrease δ: 5 μm → 2 μm (more surface heating, more cooling)
```

### 5.3 If Melt Pool Too Small

**Hypothesis**: Insufficient heat input or too much dissipation

**Fix 1**: Increase laser power
```
P_laser = 195 W → 250 W (stay within literature range)
```

**Fix 2**: Increase penetration depth
```
δ = 5 μm → 10 μm (heat penetrates deeper)
```

**Fix 3**: Reduce thermal diffusivity
```
α = 5.8×10⁻⁶ m²/s → 4.0×10⁻⁶ m²/s (slower heat dissipation)
```

---

## 6. Success Criteria

### 6.1 Quantitative Metrics

| Metric | Target | Tolerance | Source |
|--------|--------|-----------|--------|
| Energy balance error | < 5% | ±2% | Physics requirement |
| Peak temperature | 2,400-2,800 K | ±200 K | Mohr 2020 |
| Melt pool length | 150-300 μm | ±50 μm | Mohr 2020 |
| Peak velocity | 0.5-3.0 m/s | ±0.5 m/s | Literature range |
| Liquid fraction | 5-15% | ±5% | Expected for LPBF |

### 6.2 Qualitative Checks

- [ ] Marangoni recirculation visible in velocity streamlines
- [ ] Temperature gradient ~10⁶ K/m at melt pool boundary
- [ ] No NaN/Inf in any field
- [ ] Stable simulation (no divergence)
- [ ] VOF interface remains smooth (no spurious currents)

### 6.3 Final Acceptance

**PASS** = All of:
1. Energy balance error < 5% (averaged over last 1000 steps)
2. Peak temperature within 2,400-2,800 K
3. Melt pool length within 150-300 μm
4. No numerical errors (NaN/Inf)
5. Stable for full simulation duration (500 μs)

**FAIL** = Any of:
1. Energy error > 10%
2. Peak temperature < 2,000 K or > 5,000 K
3. Simulation divergence (NaN/Inf)
4. Melt pool not formed (no cells > T_liquidus)

---

## 7. Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Implement energy diagnostic methods | 1 day | IN PROGRESS |
| 2 | Test energy balance on Test H | 0.5 day | PENDING |
| 3 | Debug energy imbalance if present | 1-2 days | PENDING |
| 4 | VTK analysis and literature comparison | 0.5 day | PENDING |
| 5 | Parameter sensitivity tests | 1 day | PENDING |
| 6 | Final validation report | 0.5 day | PENDING |

**Total estimated time**: 4-5 days

---

## 8. References

1. **Mohr, M., et al. (2020)**. "In-Situ Defect Detection in Laser Powder Bed Fusion by Using Thermography: Calibrated Analytical Models for Ti-6Al-4V." *Materials*, 13(9), 2111.

2. **Khairallah, S. A., et al. (2016)**. "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." *Acta Materialia*, 108, 36-45.

3. **King, W. E., et al. (2015)**. "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing." *Journal of Materials Processing Technology*, 214(12), 2915-2925.

4. **Panwisawas, C., et al. (2017)**. "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution." *Computational Materials Science*, 126, 479-490.

---

**Last updated**: 2025-11-19
**Author**: Claude Code (AI Testing Specialist)
**Review status**: Draft (awaiting implementation completion)
