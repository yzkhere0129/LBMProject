# Quick Start: Energy Verification

**Goal**: Verify energy conservation and validate against literature in 5 steps.

---

## Step 1: Compile Code (2 minutes)

```bash
cd /home/yzk/LBMProject/build
cmake ..
make -j$(nproc)
```

**Expected**: Clean compilation, no errors.

---

## Step 2: Run Test H with Energy Diagnostics (10-30 minutes)

**Option A**: Quick test (100 steps, 10 μs)
```bash
cd /home/yzk/LBMProject/build

# Edit config to reduce steps
sed -i 's/total_steps = 5000/total_steps = 100/' ../configs/lpbf_195W_test_H_full_physics.conf

# Run
./lpbf_simulation ../configs/lpbf_195W_test_H_full_physics.conf
```

**Option B**: Full test (5000 steps, 500 μs) - use existing results if already run
```bash
# Already completed if you have lpbf_test_H_full_physics/ directory
ls -lh lpbf_test_H_full_physics/lpbf_005000.vtk
```

**Check**: Look for energy balance output in console:
```
=== ENERGY BALANCE (step 100, t=10 μs) ===
  P_laser_in:       68.25 W
  P_evaporation:    X.XX W
  P_radiation:      X.XX W
  dE/dt:            X.XX W
  Energy error:     X.X % [PASS/FAIL]
```

---

## Step 3: Analyze VTK Output (1 minute)

```bash
cd /home/yzk/LBMProject

# Basic analysis
python tools/analyze_vtk.py build/lpbf_test_H_full_physics/lpbf_005000.vtk

# With plots (optional)
python tools/analyze_vtk.py build/lpbf_test_H_full_physics/lpbf_005000.vtk --plot

# Save report
python tools/analyze_vtk.py build/lpbf_test_H_full_physics/lpbf_005000.vtk --output results_test_H.txt
```

**Expected Output**:
```
=== MELT POOL ANALYSIS ===
  Length: XXX.X μm (target: 150-300 μm)
  Width:  XX.X μm
  Depth:  XX.X μm
  Maximum temperature: XXXX.X K (target: 2400-2800 K)

=== LITERATURE COMPARISON ===
Peak Temperature:
  Literature: 2400-2800 K
  Simulation: XXXX.X K
  Status: [WITHIN RANGE ✓ / TOO LOW / TOO HIGH]

Melt Pool Length:
  Literature: 150-300 μm
  Simulation: XXX.X μm
  Status: [WITHIN RANGE ✓ / TOO SHORT / TOO LONG]
```

---

## Step 4: Interpret Results

### Case A: PASS (Energy error < 5%, T in range)

**Result**: Simulation is validated. Proceed to sensitivity tests.

**Next steps**:
1. Run H-POW-1 (100W) and H-POW-2 (200W) to match literature exactly
2. Generate final validation report
3. Publish results

---

### Case B: Energy Error > 5% (Energy imbalance)

**Diagnosis**:

1. **Check where energy goes**:
   - If `dE/dt ≈ P_laser` → cooling mechanisms not working
   - If `P_evap + P_rad << P_laser` → need stronger cooling
   - If `error` oscillates → numerical instability

2. **Fix strategy**:
   - **Too much heating**: Increase radiation (ε=0.8) or evaporation
   - **Too little heating**: Decrease cooling or increase laser power
   - **Numerical issue**: Reduce timestep (dt=0.05 μs)

3. **Run sensitivity test**:
   ```bash
   # Try higher emissivity
   cp configs/lpbf_195W_test_H_full_physics.conf configs/lpbf_test_H_rad_high.conf
   sed -i 's/emissivity = 0.3/emissivity = 0.8/' configs/lpbf_test_H_rad_high.conf
   ./lpbf_simulation configs/lpbf_test_H_rad_high.conf
   ```

---

### Case C: Temperature Out of Range (T not 2400-2800 K)

**Too High (T > 2800 K)**:

1. **Increase cooling**:
   - Increase emissivity: ε = 0.3 → 0.8
   - Check evaporation is active (T_surface > 3560 K for Ti6Al4V)

2. **Reduce heat input**:
   - Increase penetration depth: δ = 5 μm → 10 μm (spreads heat)

**Too Low (T < 2400 K)**:

1. **Decrease cooling**:
   - Decrease emissivity: ε = 0.3 → 0.1

2. **Increase heat input**:
   - Increase laser power: P = 195 W → 200 W
   - Decrease penetration depth: δ = 5 μm → 2 μm (concentrates heat)

---

### Case D: Melt Pool Size Wrong

**Too Small (Length < 150 μm)**:

1. **Increase heat input**: P = 195 W → 200 W
2. **Increase penetration**: δ = 5 μm → 10 μm
3. **Decrease diffusivity**: α (slower heat dissipation)

**Too Large (Length > 300 μm)**:

1. **Decrease heat input**: P = 195 W → 150 W
2. **Decrease penetration**: δ = 5 μm → 2 μm
3. **Increase diffusivity**: α (faster heat dissipation)

---

## Step 5: Document Results

Create final report:

```markdown
# Test H Results

## Energy Balance
- Energy error: X.X% [PASS/FAIL]
- P_laser: 68.25 W
- P_evap: X.XX W
- P_rad: X.XX W
- dE/dt: X.XX W

## Literature Comparison
- Peak temperature: XXXX K (target: 2400-2800 K) [PASS/FAIL]
- Melt pool length: XXX μm (target: 150-300 μm) [PASS/FAIL]
- Peak velocity: X.XX mm/s (target: 0.5-3.0 m/s) [PASS/FAIL]

## Overall Status
[PASS / FAIL with adjustments needed / FAIL fundamental issue]

## Recommendations
[List any parameter adjustments needed]
```

---

## Quick Reference: Key Files

### Configuration
- Test H config: `configs/lpbf_195W_test_H_full_physics.conf`
- Modify: `nano configs/lpbf_195W_test_H_full_physics.conf`

### Running Simulation
- Executable: `build/lpbf_simulation`
- Usage: `./lpbf_simulation <config_file>`

### Analysis Tools
- VTK analyzer: `tools/analyze_vtk.py`
- Energy balance: Built into simulation (prints every 100 steps)

### Output Locations
- VTK files: `build/lpbf_test_H_full_physics/lpbf_XXXXXX.vtk`
- Console output: Energy balance diagnostics
- Reports: Custom location (use `--output` flag)

---

## Troubleshooting

### Issue: Script won't run (import error)

```bash
# Install dependencies
pip install numpy

# Optional: for plotting
pip install matplotlib

# Optional: for robust VTK reading
pip install vtk
```

### Issue: Compilation error

```bash
# Clean rebuild
cd build
rm -rf *
cmake ..
make -j$(nproc)
```

### Issue: Simulation crashes

1. Check for NaN/Inf in output
2. Reduce timestep: `dt = 5.0e-8` (0.05 μs)
3. Check CFL condition: `v_max * dt / dx < 0.5`

### Issue: Energy balance not printing

1. Check that energy diagnostics are called in main loop
2. Verify thermal solver is enabled: `enable_thermal = true`
3. Check for compilation warnings

---

## Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Compile code | 2 min | ⬜ |
| Run Test H (100 steps) | 2 min | ⬜ |
| Analyze VTK | 1 min | ⬜ |
| Interpret results | 5 min | ⬜ |
| **TOTAL (quick test)** | **10 min** | |
| | | |
| Run Test H (5000 steps) | 30 min | ⬜ |
| Debug if needed | 30-60 min | ⬜ |
| Sensitivity tests | 1-2 hours | ⬜ |
| **TOTAL (full validation)** | **2-3 hours** | |

---

## Success Criteria

**Minimum (PASS):**
- ✓ Energy balance error < 5%
- ✓ Peak temperature 2400-2800 K
- ✓ No NaN/Inf errors

**Ideal (EXCELLENT):**
- ✓ Energy balance error < 2%
- ✓ Peak temperature 2400-2800 K
- ✓ Melt pool length 150-300 μm
- ✓ Peak velocity 0.5-3.0 m/s
- ✓ All sensitivity tests pass

---

**Good luck! If you need help, consult the detailed documentation:**
- `docs/ENERGY_VERIFICATION_PLAN.md`
- `docs/TEST_CASES_ENERGY_VERIFICATION.md`
- `docs/IMPLEMENTATION_SUMMARY.md`
