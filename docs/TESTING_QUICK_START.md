# Quick Start: Steady-State Testing for Week 3 GO/NO-GO

**Goal:** Validate laser heating and energy conservation in < 1 hour
**Decision:** Proceed to Week 3 vapor phase or fix bugs first

---

## TL;DR: Run This Now

```bash
cd /home/yzk/LBMProject/build

# Tier 1: Smoke test (5 min)
./visualize_lpbf_scanning --config ../configs/calibration/tier1_smoke_test.conf

# Check: Did T_max exceed 1928 K? If YES, continue:

# Tier 2: Energy balance (30 min)
./visualize_lpbf_scanning --config ../configs/calibration/tier2_energy_balance.conf

# Analyze results:
cd tier2_energy_balance
tail -20 energy_balance.csv   # Check if dE/dt is decreasing
```

**Decision:**
- If Tier 1 passes + Tier 2 shows energy error < 10% → **GO for Week 3**
- If Tier 1 fails → **NO GO**, fix laser heating first
- If Tier 2 shows divergence → **NO GO**, fix energy conservation

---

## Tier 1: Smoke Test (5 minutes)

**What:** Verify laser heats material to melting point
**Runtime:** < 5 min
**Pass criteria:** T_max > 1928 K, no crash

```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/tier1_smoke_test.conf
```

**Check output:**
```
Step      Time [μs]   T_max [K]   v_max [mm/s]
   0           0.00      300.0          0.000
 100          10.00      800.0          0.500
 ...
1000         100.00     2100.0          3.500  ← Must be > 1928 K ✓
```

**PASS:** T_max > 1928 K (melting achieved)
**FAIL:** T_max < 1500 K (laser not working)

---

## Tier 2: Energy Balance (30 minutes)

**What:** Verify energy conservation and steady-state trends
**Runtime:** 25-35 min
**Pass criteria:** Error < 10%, dE/dt decreasing

```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/tier2_energy_balance.conf
```

**After completion, check energy file:**
```bash
cd tier2_energy_balance
tail -20 energy_balance.csv

# Look for:
# - Column 11 (dE/dt_computed): Should decrease over time
# - Column 13 (Error %): Should be < 10%, ideally < 5%
# - Temperature should stabilize (T_max variation < 50 K)
```

**PASS indicators:**
- Last 10 lines show dE/dt < 2 W (down from ~10 W initially)
- Error < 10% consistently
- No runaway heating (T_max < 3000 K)

**FAIL indicators:**
- dE/dt increasing or staying constant > 5 W
- Error > 20% persistently
- Simulation crashes

---

## Tier 3: Full Steady-State (OPTIONAL, 2-3 hours)

**Only run if:** Tier 2 is borderline (error 10-15%, slow convergence)
**Skip if:** Tier 2 clearly passes (error < 5%, dE/dt < 1 W)

```bash
cd /home/yzk/LBMProject/build

# Use original config but with reduced output:
# Edit: configs/calibration/lpbf_50W_dt_010us_STEADY_STATE.conf
# Change: output_interval = 5000  (instead of 500)

./visualize_lpbf_scanning --config ../configs/calibration/lpbf_50W_dt_010us_STEADY_STATE.conf
```

**This runs 150,000 steps = 15 ms physical time**
**Runtime: ~2-3 hours (good for overnight)**

---

## Decision Matrix

| Tier 1 | Tier 2 | Decision | Action |
|--------|--------|----------|--------|
| PASS | PASS (error < 5%) | **FULL GO** | Proceed to Week 3 |
| PASS | PASS (error 5-10%) | **GO** | Proceed, monitor in Week 3 |
| PASS | CONDITIONAL (10-15%) | **CONDITIONAL GO** | Consider Tier 3 or proceed with caution |
| PASS | FAIL (> 15%) | **NO GO** | Fix energy conservation |
| FAIL | - | **NO GO** | Fix laser heating first |

---

## Common Issues

### Tier 1 fails: No heating

**Check:**
1. Laser power in config: `laser_power = 50.0` (not 0)
2. Absorptivity: `laser_absorptivity = 0.20` (20%)
3. Laser position: `laser_start_x = 200e-6, laser_start_y = 100e-6` (domain center)
4. Laser duration: `laser_duration = 20000e-6` (20 ms, longer than simulation)

**Fix:** Verify Bug #1 and Bug #2 fixes are in code

### Tier 2 fails: High energy error

**Check:**
1. Energy diagnostics file exists: `tier2_energy_balance/energy_balance.csv`
2. If file missing → app not calling `computeEnergyBalance()` (see full doc)
3. Power terms make sense: `P_laser ≈ 10 W`, `P_substrate + P_radiation > 0`

**Fix:** Add energy diagnostic calls to app (see STEADY_STATE_TESTING_STRATEGY.md)

### Energy file not written

**Symptom:** No `energy_balance.csv` in output directory

**Cause:** App doesn't call `writeEnergyBalanceHistory()`

**Quick fix:**
```bash
# Check if energy output is implemented
cd /home/yzk/LBMProject
grep -n "writeEnergyBalance" apps/visualize_lpbf_scanning.cu

# If nothing found, see Action Item #1 in full strategy doc
```

---

## Files Created

**Configs:**
- `/home/yzk/LBMProject/configs/calibration/tier1_smoke_test.conf`
- `/home/yzk/LBMProject/configs/calibration/tier2_energy_balance.conf`

**Docs:**
- `/home/yzk/LBMProject/docs/STEADY_STATE_TESTING_STRATEGY.md` (full details)
- `/home/yzk/LBMProject/docs/TESTING_QUICK_START.md` (this file)

**Outputs (after running):**
- `/home/yzk/LBMProject/build/tier1_smoke_test/` (VTK + energy CSV)
- `/home/yzk/LBMProject/build/tier2_energy_balance/` (VTK + energy CSV)

---

## Next Steps

1. **Verify energy diagnostics are integrated** (see full doc Action Item #1)
2. **Run Tier 1** (5 min)
3. **If pass, run Tier 2** (30 min)
4. **Make GO/NO-GO decision**

**Total time investment: < 1 hour to confident decision**

For detailed analysis, failure modes, and debugging steps, see:
`/home/yzk/LBMProject/docs/STEADY_STATE_TESTING_STRATEGY.md`
