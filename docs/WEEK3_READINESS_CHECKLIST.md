# Week 3 Readiness Checklist

**Date:** 2025-11-20
**Target:** GO/NO-GO decision for vapor phase implementation

---

## Pre-Flight Checklist

### 1. Code Verification

- [ ] Bug #1 fixed: Laser position calculation correct
- [ ] Bug #2 fixed: Laser scanning velocity units (m/s vs μm/s)
- [ ] Executable built: `/home/yzk/LBMProject/build/visualize_lpbf_scanning` exists
- [ ] Timestamp recent: Executable built after Nov 20 bug fixes

**Verify:**
```bash
ls -lh /home/yzk/LBMProject/build/visualize_lpbf_scanning
# Should show timestamp: Nov 20 or later
```

---

### 2. Energy Diagnostics Integration

**CRITICAL:** Verify app writes energy balance CSV

- [ ] App calls `solver.computeEnergyBalance()` in time loop
- [ ] App calls `solver.writeEnergyBalanceHistory()` after loop
- [ ] Config loader reads `enable_energy_diagnostics` flag
- [ ] Config loader reads `energy_output_interval` parameter

**Verify:**
```bash
cd /home/yzk/LBMProject
grep -n "computeEnergyBalance\|writeEnergyBalance" apps/visualize_lpbf_scanning.cu
# Should show at least 2 lines (compute + write calls)
```

**If not found:** See Action Item #1 in `STEADY_STATE_TESTING_STRATEGY.md`

---

### 3. Configuration Files

- [ ] Tier 1 config exists: `configs/calibration/tier1_smoke_test.conf`
- [ ] Tier 2 config exists: `configs/calibration/tier2_energy_balance.conf`
- [ ] Tier 3 config exists: `configs/calibration/lpbf_50W_dt_010us_STEADY_STATE.conf`

**Verify:**
```bash
ls -lh /home/yzk/LBMProject/configs/calibration/tier*.conf
```

---

### 4. Output Directories

- [ ] Build directory exists: `/home/yzk/LBMProject/build`
- [ ] Output dirs prepared: `tier1_smoke_test/`, `tier2_energy_balance/`

**Prepare:**
```bash
cd /home/yzk/LBMProject/build
mkdir -p tier1_smoke_test tier2_energy_balance
```

---

## Execution Checklist

### Tier 1: Smoke Test (5 min)

**Run:**
```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/tier1_smoke_test.conf
```

**Check during run:**
- [ ] No crash or segfault
- [ ] Temperature increasing in stdout
- [ ] No NaN or Inf messages

**Check after completion:**
- [ ] Final T_max > 1928 K (melting point)
- [ ] Final T_max < 4000 K (no runaway heating)
- [ ] VTK files written: `tier1_smoke_test/lpbf_*.vtk` (10 files)

**PASS Criteria:**
- [x] All checks above pass

**If FAIL → STOP. Fix laser heating before Tier 2.**

---

### Tier 2: Energy Balance (30 min)

**Run:**
```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/tier2_energy_balance.conf
```

**Check during run:**
- [ ] No crash throughout 30k steps
- [ ] Temperature stabilizing (T_max not increasing rapidly after 1000 μs)
- [ ] Velocity bounded (v_max < 10 m/s)

**Check after completion:**
- [ ] Energy CSV exists: `tier2_energy_balance/energy_balance.csv`
- [ ] VTK files written: `tier2_energy_balance/lpbf_*.vtk` (30 files)

**Analyze results:**
```bash
cd tier2_energy_balance
tail -20 energy_balance.csv

# Run automated analysis (if Python available):
python3 ../../scripts/analyze_energy_balance.py energy_balance.csv
```

**PASS Criteria (manual check):**
- [ ] Last 10 lines show dE/dt < 2 W (column 11)
- [ ] Last 10 lines show error < 10% (column 13)
- [ ] dE/dt is decreasing over time (not constant or increasing)
- [ ] T_max in range [2200 K, 3000 K]

**PASS Criteria (automated analysis):**
- [ ] Script reports "FULL GO" or "CONDITIONAL GO"
- [ ] Score ≥ 70/100

**If CONDITIONAL GO → Consider Tier 3 for confirmation**
**If FAIL → Investigate energy conservation issues**

---

### Tier 3: Full Steady-State (OPTIONAL, 2-3 hours)

**Only run if:** Tier 2 is borderline (60-80 score) OR high confidence required

**Prepare config (reduce output for speed):**
```bash
cd /home/yzk/LBMProject/configs/calibration
cp lpbf_50W_dt_010us_STEADY_STATE.conf lpbf_50W_dt_010us_STEADY_STATE_FAST.conf

# Edit FAST version: change output_interval from 500 to 5000
# This reduces VTK files from 300 to 30 (10× speedup)
```

**Run:**
```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/lpbf_50W_dt_010us_STEADY_STATE_FAST.conf
```

**PASS Criteria:**
- [ ] dE/dt < 0.5 W over last 2000 μs
- [ ] Error < 5% persistently
- [ ] T_max variation < 10 K over last 2000 μs
- [ ] Final T_steady_state ∈ [2400 K, 2700 K]

---

## GO/NO-GO Decision

### FULL GO ✓

**Criteria:**
- [x] Tier 1 PASS
- [x] Tier 2 PASS (score ≥ 85/100 OR error < 5%)

**Action:** Proceed to Week 3 vapor phase implementation with high confidence

---

### CONDITIONAL GO ~

**Criteria:**
- [x] Tier 1 PASS
- [x] Tier 2 CONDITIONAL (score 70-84/100 OR error 5-10%)

**Action:**
- Proceed to Week 3 with caution
- Monitor energy balance during vapor implementation
- OR run Tier 3 for confirmation (if time permits)

---

### NO GO ✗

**Criteria (any of):**
- [x] Tier 1 FAIL (laser not heating)
- [x] Tier 2 FAIL (score < 70/100 OR error > 15%)
- [x] Tier 3 FAIL (if run, not converging)

**Action:**
- Debug and fix issues before Week 3
- Re-run failed tier after fixes
- Document root cause and solution

---

## Post-Test Actions

### If GO Decision

1. **Archive results:**
```bash
cd /home/yzk/LBMProject
tar -czf week3_baseline_$(date +%Y%m%d).tar.gz \
    build/tier1_smoke_test \
    build/tier2_energy_balance
```

2. **Document baseline:**
   - Record final T_steady_state
   - Record energy balance error
   - Save plots for comparison with Week 3 results

3. **Proceed to Week 3:**
   - Begin vapor phase module design
   - Identify integration points with existing multiphysics solver
   - Plan validation tests for vapor effects

### If NO GO Decision

1. **Diagnose failure mode:**
   - Check stdout logs for error messages
   - Verify config parameters
   - Review recent code changes

2. **Prioritize fixes:**
   - Laser heating issues (Tier 1 fail) → highest priority
   - Energy conservation (Tier 2 fail) → high priority
   - Convergence speed (Tier 3 fail) → medium priority

3. **Re-test after fixes:**
   - Start from Tier 1 again
   - Document what was changed and why
   - Verify fix doesn't break other physics

---

## Quick Reference

**Execution Time:**
- Tier 1: 5 min
- Tier 2: 30 min
- Tier 3: 2-3 hours (optional)

**Fastest GO Decision:** 35 minutes (Tier 1 + Tier 2)

**Key Files:**
- Full strategy: `docs/STEADY_STATE_TESTING_STRATEGY.md`
- Quick start: `docs/TESTING_QUICK_START.md`
- This checklist: `docs/WEEK3_READINESS_CHECKLIST.md`
- Analysis script: `scripts/analyze_energy_balance.py`

**Expected Results (Tier 2):**
```
Final state (~3000 μs):
  T_max:        2550 ± 50 K
  dE/dt:        1.5 ± 1.0 W
  Error:        5 ± 3%
  P_laser:      10.0 W
  P_out_total:  ~9 W (substrate + radiation + evap)
```

---

## Emergency Contacts / References

**If energy CSV not generated:**
- See Action Item #1 in `STEADY_STATE_TESTING_STRATEGY.md`
- Code modification required in `apps/visualize_lpbf_scanning.cu`

**If energy error very high (> 30%):**
- Check boundary conditions active (radiation, substrate, evaporation)
- Verify material properties loaded correctly
- Check for NaN in power terms

**If simulation crashes:**
- Check GPU memory: `nvidia-smi`
- Verify CUDA version compatibility
- Check for recent code changes that broke compilation

---

**Status:** READY FOR TESTING
**Next Step:** Run Tier 1 smoke test (5 min)
