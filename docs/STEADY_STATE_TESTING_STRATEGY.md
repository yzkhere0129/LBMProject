# Multi-Tier Testing Strategy for LPBF Steady-State Validation
## Week 3 GO/NO-GO Decision Framework

**Date:** 2025-11-20
**Status:** READY FOR EXECUTION
**Target:** Validate laser heating, energy balance, and steady-state convergence

---

## Executive Summary

This document defines a pragmatic, three-tier testing strategy to validate the LPBF simulation framework before Week 3 vapor phase implementation. The strategy balances speed (rapid feedback) with rigor (comprehensive validation).

**Key Insight:** Previous 150k-step runs were dominated by I/O overhead. The new strategy uses aggressive output reduction and tiered validation to achieve 5-minute smoke tests while maintaining scientific rigor.

**Decision Criteria:**
- **FULL GO:** All Tier 1 + Tier 2 tests pass (expect ~35-40 min total)
- **CONDITIONAL GO:** Tier 1 passes, Tier 2 shows positive trends (requires expert judgment)
- **NO GO:** Tier 1 fails or Tier 2 shows divergence/non-physical behavior

---

## Background: Current Status

**Recent Fixes (Nov 20):**
1. Bug #1: Laser position - fixed incorrect center calculation
2. Bug #2: Laser scanning - fixed velocity units (m/s vs μm/s)

**Current Bottleneck:**
- 150k steps at dt=0.1μs = 15,000 μs (15 ms) physical time
- Output interval = 500 steps (every 50 μs)
- Result: 300 VTK files → excessive I/O overhead → hours of runtime

**Files:**
- **Executable:** `/home/yzk/LBMProject/build/visualize_lpbf_scanning`
- **Config:** `/home/yzk/LBMProject/configs/calibration/lpbf_50W_dt_010us_STEADY_STATE.conf`
- **Material:** Ti6Al4V (T_solidus=1878K, T_liquidus=1928K, T_vaporization=3560K)

---

## Tier 1: Smoke Test (5 minutes)
**Purpose:** Verify laser heating works and physics are numerically stable
**Runtime Target:** < 5 minutes
**Decision:** GO/NO-GO for Tier 2

### Configuration

Create new config file: `/home/yzk/LBMProject/configs/calibration/tier1_smoke_test.conf`

```ini
# TIER 1: Smoke Test - Laser Heating Verification
# Target: Verify laser turns on, heats material, reaches melting
# Physical time: 1,000 steps × 0.1μs = 100 μs

num_steps = 1000           # 100 μs (enough to see heating)
dt = 0.1e-6                # 0.10 μs timestep
output_interval = 100      # Every 10 μs → 10 VTK files

# Domain configuration (SAME as steady-state)
nx = 200
ny = 100
nz = 50
dx = 2.0e-6                # 2 μm grid spacing
                           # Domain: 400×200×100 μm³

# Material
material = Ti6Al4V

# Laser parameters (SAME as steady-state)
laser_power = 50.0                 # 50 W
laser_absorptivity = 0.20          # 20% absorption → 10 W absorbed
laser_radius = 50.0e-6             # 50 μm spot
laser_duration = 20000.0e-6        # 20,000 μs (always on)
laser_start_x = 200.0e-6           # Center of domain
laser_start_y = 100.0e-6           # Center of domain
laser_scan_vx = 0.0                # NO SCANNING
laser_scan_vy = 0.0

# Initial and ambient conditions
T_initial = 300.0
T_ambient = 300.0

# Boundary conditions (SAME as steady-state)
enable_substrate_cooling = true
substrate_h_conv = 50000.0         # 50,000 W/(m²·K)
substrate_temperature = 300.0

enable_radiation = true
emissivity = 0.30

enable_evaporation = true

# Output configuration (REDUCED OUTPUT)
output_directory = tier1_smoke_test
save_temperature = true
save_velocity = false              # Disabled for speed
save_vof = false                   # Disabled for speed

# Energy diagnostics
enable_energy_diagnostics = true
energy_output_interval = 100       # Every 10 μs
```

### Execution

```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/tier1_smoke_test.conf
```

### Pass/Fail Criteria

**PASS if ALL conditions met:**

1. **Simulation Completes:** No crashes, NaN, or Inf values
2. **Laser Heats Material:** T_max > T_liquidus (1928 K) by end of run
3. **Numerical Stability:** Temperature stays below unphysical limits (T_max < 4000 K)
4. **Energy Input Verified:** Laser power in stdout matches expected 10 W absorbed

**Expected Output:**
```
Step      Time [μs]   T_max [K]   v_max [mm/s]
   0           0.00      300.0          0.000
 100          10.00      800.0          0.500
 200          20.00     1200.0          1.200
 ...
1000         100.00     2100.0          3.500  ← Should exceed T_liquidus
```

**FAIL if ANY condition met:**
- Simulation crashes or produces NaN/Inf
- T_max < T_liquidus after 100 μs (laser not working)
- T_max > 5000 K (runaway heating, numerical instability)

### What This Validates

- ✓ Laser position fix (Bug #1): Laser heating occurs at domain center
- ✓ Physics modules interact correctly
- ✓ Energy input pathway functional
- ✓ Numerical stability at short timescales

### What This Does NOT Validate

- ✗ Energy conservation (too short for meaningful balance)
- ✗ Steady-state convergence
- ✗ Long-time numerical stability

---

## Tier 2: Energy Balance Check (30 minutes)
**Purpose:** Verify energy conservation trends and approach to steady-state
**Runtime Target:** 25-35 minutes
**Decision:** GO/NO-GO for Week 3, determine if Tier 3 needed

### Configuration

Create new config file: `/home/yzk/LBMProject/configs/calibration/tier2_energy_balance.conf`

```ini
# TIER 2: Energy Balance Check - Medium Duration
# Target: Verify energy conservation and steady-state trends
# Physical time: 30,000 steps × 0.1μs = 3,000 μs = 3 ms

num_steps = 30000          # 3,000 μs (10× longer than Tier 1)
dt = 0.1e-6                # 0.10 μs timestep
output_interval = 1000     # Every 100 μs → 30 VTK files (SPARSE OUTPUT)

# Domain configuration (SAME as steady-state)
nx = 200
ny = 100
nz = 50
dx = 2.0e-6                # 2 μm grid spacing

# Material
material = Ti6Al4V

# Laser parameters (SAME as steady-state)
laser_power = 50.0
laser_absorptivity = 0.20
laser_radius = 50.0e-6
laser_duration = 20000.0e-6
laser_start_x = 200.0e-6
laser_start_y = 100.0e-6
laser_scan_vx = 0.0
laser_scan_vy = 0.0

# Initial and ambient conditions
T_initial = 300.0
T_ambient = 300.0

# Boundary conditions (SAME as steady-state)
enable_substrate_cooling = true
substrate_h_conv = 50000.0
substrate_temperature = 300.0

enable_radiation = true
emissivity = 0.30

enable_evaporation = true

# Output configuration (AGGRESSIVE REDUCTION)
output_directory = tier2_energy_balance
save_temperature = true
save_velocity = false
save_vof = false

# Energy diagnostics (CRITICAL FOR THIS TIER)
enable_energy_diagnostics = true
energy_output_interval = 100       # Every 10 μs (dense for analysis)
```

### Execution

```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning --config ../configs/calibration/tier2_energy_balance.conf

# IMPORTANT: The app must call solver.computeEnergyBalance() and
# solver.writeEnergyBalanceHistory() - verify this in code first!
```

### Pass/Fail Criteria

**CRITICAL ASSUMPTION:** The application writes energy balance to CSV file. If not, this tier requires code modification first (see Action Items below).

**PASS if ALL conditions met:**

1. **Energy Conservation (Early Phase, t < 500 μs):**
   - Energy balance error < 10% during rapid heating phase
   - No systematic drift (error does not monotonically increase)

2. **Steady-State Trend (Late Phase, t > 2000 μs):**
   - dE/dt decreases over time (system stabilizing)
   - T_max variation < 50 K over last 500 μs
   - Energy balance error < 5% after initial transient

3. **Physical Reasonableness:**
   - 2200 K < T_max < 3000 K (below vaporization, above melting)
   - P_laser ≈ 10 W (consistent with input)
   - P_substrate + P_radiation + P_evaporation ≈ P_laser at late times

4. **Numerical Stability:**
   - No NaN or Inf throughout run
   - Velocity field remains bounded (v_max < 10 m/s)

**Expected Behavior:**

```
Time [μs]    T_max [K]    dE/dt [W]    Error [%]    Phase
    0          300         +10.0         N/A        Initial heating
  500         2400          +8.5         3.2%       Rapid heating
 1000         2550          +5.2         2.8%       Approaching balance
 2000         2620          +1.5         1.5%       Near steady-state
 3000         2640          +0.3         1.2%       Steady-state ✓
```

**CONDITIONAL PASS if:**
- Energy error < 15% but trends are correct (decreasing over time)
- T_max shows slow convergence but no runaway heating
- **Action:** Recommend Tier 3 for confirmation

**FAIL if ANY condition met:**
- Energy error > 20% persistently
- dE/dt increases with time (diverging, not converging)
- T_max runaway (> 3500 K) or unrealistic (< 1500 K)
- Simulation crashes before completion

### What This Validates

- ✓ Energy conservation over multiphysics coupling
- ✓ Steady-state convergence trends
- ✓ Thermal boundary conditions (radiation, substrate, evaporation) functioning correctly
- ✓ Medium-term numerical stability (3 ms physical time)

### What This Does NOT Validate

- ✗ Final steady-state convergence (requires ~15 ms)
- ✗ Statistical steady-state (requires longer averaging window)

### Analysis: Post-Run Diagnostics

After Tier 2 completes, analyze energy balance CSV:

```bash
# Assuming energy output is written to tier2_energy_balance/energy_balance.csv
cd /home/yzk/LBMProject/build/tier2_energy_balance

# Quick check: Last 10 lines (should show decreasing dE/dt)
tail -20 energy_balance.csv

# Plot energy balance (requires Python + matplotlib)
python3 << EOF
import numpy as np
import matplotlib.pyplot as plt

# Load data (skip header lines starting with #)
data = np.loadtxt('energy_balance.csv', comments='#')
time_us = data[:, 0] * 1e6  # Convert s → μs
E_total = data[:, 5]         # Column 6: E_total [J]
dE_dt_computed = data[:, 10] # Column 11: dE/dt computed [W]
dE_dt_balance = data[:, 11]  # Column 12: dE/dt balance [W]
error_pct = data[:, 12]      # Column 13: Error [%]

# Plot 1: Total energy vs time
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(time_us, E_total * 1e6, 'b-', linewidth=1.5)
plt.xlabel('Time [μs]')
plt.ylabel('Total Energy [μJ]')
plt.title('Total Energy vs Time')
plt.grid(True)

# Plot 2: dE/dt comparison
plt.subplot(2, 2, 2)
plt.plot(time_us, dE_dt_computed, 'b-', label='dE/dt (computed)', linewidth=1.5)
plt.plot(time_us, dE_dt_balance, 'r--', label='dE/dt (balance)', linewidth=1.5)
plt.xlabel('Time [μs]')
plt.ylabel('Power [W]')
plt.title('Energy Balance Comparison')
plt.legend()
plt.grid(True)

# Plot 3: Error percentage
plt.subplot(2, 2, 3)
plt.plot(time_us, error_pct, 'r-', linewidth=1.5)
plt.axhline(y=5.0, color='g', linestyle='--', label='5% threshold')
plt.axhline(y=10.0, color='orange', linestyle='--', label='10% threshold')
plt.xlabel('Time [μs]')
plt.ylabel('Error [%]')
plt.title('Energy Balance Error')
plt.legend()
plt.grid(True)
plt.ylim([0, max(20, error_pct.max() * 1.1)])

# Plot 4: dE/dt trend (should decrease)
plt.subplot(2, 2, 4)
plt.plot(time_us, dE_dt_computed, 'b-', linewidth=1.5)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.xlabel('Time [μs]')
plt.ylabel('dE/dt [W]')
plt.title('Energy Rate (should approach 0)')
plt.grid(True)

plt.tight_layout()
plt.savefig('energy_balance_analysis.png', dpi=150)
print("✓ Energy balance plots saved to energy_balance_analysis.png")

# Quantitative checks
print("\n=== TIER 2 QUANTITATIVE ANALYSIS ===")
print(f"Total runtime: {time_us[-1]:.1f} μs")
print(f"Final E_total: {E_total[-1]*1e6:.2f} μJ")
print(f"Final dE/dt: {dE_dt_computed[-1]:.3f} W")
print(f"Final error: {error_pct[-1]:.2f}%")

# Convergence check: Compare first half vs second half variability
mid_idx = len(time_us) // 2
dE_dt_std_early = np.std(dE_dt_computed[:mid_idx])
dE_dt_std_late = np.std(dE_dt_computed[mid_idx:])
print(f"\ndE/dt variability (early): {dE_dt_std_early:.3f} W")
print(f"dE/dt variability (late):  {dE_dt_std_late:.3f} W")

if dE_dt_std_late < dE_dt_std_early * 0.5:
    print("✓ PASS: System is converging (late variability < 50% of early)")
else:
    print("⚠ WARNING: System may not be converging")

# Steady-state check: Last 500 μs should have low dE/dt
last_500us_mask = time_us > (time_us[-1] - 500)
mean_dE_dt_late = np.mean(np.abs(dE_dt_computed[last_500us_mask]))
print(f"\nMean |dE/dt| (last 500 μs): {mean_dE_dt_late:.3f} W")

if mean_dE_dt_late < 1.0:
    print("✓ PASS: Approaching steady-state (mean |dE/dt| < 1 W)")
elif mean_dE_dt_late < 2.0:
    print("⚠ CONDITIONAL: May need Tier 3 for full convergence")
else:
    print("✗ FAIL: Not converging (mean |dE/dt| > 2 W)")
EOF
```

---

## Tier 3: Full Steady-State (OPTIONAL, 2-3 hours)
**Purpose:** Confirm final steady-state convergence if Tier 2 is inconclusive
**Runtime Target:** 2-3 hours (use overnight or weekend)
**Decision:** Final validation before Week 3

### When to Run Tier 3

**RUN Tier 3 if:**
- Tier 2 CONDITIONAL PASS (trending correctly but not fully converged)
- Week 3 GO decision requires high confidence in steady-state
- Time permits (overnight run acceptable)

**SKIP Tier 3 if:**
- Tier 2 PASS with clear steady-state (dE/dt < 0.5 W, error < 3%)
- Time-critical decision (Tier 2 sufficient for GO/NO-GO)
- Tier 2 FAIL (no point running longer if fundamentally broken)

### Configuration

Use original config: `/home/yzk/LBMProject/configs/calibration/lpbf_50W_dt_010us_STEADY_STATE.conf`

**Recommended Modifications:**
```ini
# Keep num_steps = 150000 (15 ms physical time)
# BUT reduce output for speed:
output_interval = 5000      # Every 500 μs → 30 VTK files (was 500 → 300 files)
energy_output_interval = 100  # Every 10 μs (dense for analysis)
```

### Pass/Fail Criteria

**PASS if:**
- dE/dt < 0.5 W over last 2000 μs (< 5% of 10 W input)
- T_max variation < 10 K over last 2000 μs
- Energy balance error < 3% persistently
- Final T_steady_state in range [2400 K, 2700 K] (physically reasonable)

**Expected Final State:**
```
Time: 15,000 μs
T_max: 2550 ± 10 K (steady)
dE/dt: +0.3 W (< 0.5 W threshold)
Error: 1.8%
P_laser: 10.0 W
P_out (total): 9.7 W (substrate + radiation + evap)
```

### What This Validates

- ✓ Full steady-state convergence
- ✓ Long-term numerical stability
- ✓ Final energy balance closure
- ✓ Platform ready for Week 3 vapor phase

---

## Decision Logic: GO/NO-GO Framework

```
┌─────────────────────────────────────────────────────────────┐
│                        START                                │
│                   Run Tier 1 (5 min)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Tier 1 PASS? │
                    └───────────────┘
                      │           │
                  YES │           │ NO
                      ▼           ▼
            ┌─────────────┐   ┌──────────────┐
            │ Run Tier 2  │   │  NO GO       │
            │  (30 min)   │   │  Fix bugs    │
            └─────────────┘   └──────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Tier 2 PASS? │
              └───────────────┘
                /      |      \
           PASS/   COND │       \FAIL
              /        |        \
             ▼         ▼         ▼
        ┌────────┐  ┌────────┐  ┌──────────┐
        │FULL GO │  │Time OK?│  │  NO GO   │
        │Week 3  │  └────────┘  │Fix energy│
        └────────┘   /      \   └──────────┘
                  YES       NO
                   │         │
                   ▼         ▼
             ┌──────────┐  ┌──────────┐
             │COND. GO  │  │Run Tier 3│
             │(proceed) │  │(2-3 hrs) │
             └──────────┘  └──────────┘
                                 │
                                 ▼
                          ┌───────────┐
                          │Tier 3 PASS│
                          └───────────┘
                            │       │
                        YES │       │ NO
                            ▼       ▼
                      ┌────────┐  ┌──────────┐
                      │FULL GO │  │  NO GO   │
                      └────────┘  └──────────┘
```

---

## Action Items: BEFORE Running Tests

### 1. Verify Energy Diagnostics Integration

**Check if `visualize_lpbf_scanning.cu` writes energy output:**

```bash
cd /home/yzk/LBMProject
grep -n "computeEnergyBalance\|writeEnergyBalance" apps/visualize_lpbf_scanning.cu
```

**If NOT found:** Energy diagnostics infrastructure exists but is not called by the app.

**Required Code Addition** (add to time integration loop):

```cpp
// In visualize_lpbf_scanning.cu, inside the time integration loop
for (int step = 0; step <= num_steps; ++step) {
    // ... existing VTK output code ...

    // NEW: Energy diagnostics (if enabled in config)
    if (metadata.enable_energy_diagnostics &&
        step % metadata.energy_output_interval == 0) {
        solver.computeEnergyBalance();
        solver.getEnergyBalanceTracker().record();  // Store in history

        // Optional: Print to stdout for real-time monitoring
        const auto& balance = solver.getCurrentEnergyBalance();
        std::cout << "  [Energy] t=" << std::setw(8) << std::fixed
                  << std::setprecision(1) << step * config.dt * 1e6
                  << " μs, dE/dt=" << std::setw(8) << std::setprecision(2)
                  << balance.dE_dt_computed << " W, error="
                  << std::setw(5) << std::setprecision(1)
                  << balance.error_percent << "%\n";
    }

    // ... step forward ...
}

// AFTER loop completes, write energy history to file
if (metadata.enable_energy_diagnostics) {
    std::string energy_file = output_dir + "/energy_balance.csv";
    solver.writeEnergyBalanceHistory(energy_file);
    std::cout << "\n✓ Energy balance written to: " << energy_file << "\n";
}
```

**Verify config loader handles energy flags:**

```bash
grep -n "enable_energy_diagnostics\|energy_output_interval" \
    /home/yzk/LBMProject/src/config/lpbf_config_loader.cpp
```

If missing, add to config parser.

### 2. Create Test Configuration Files

Create three new config files as specified in Tier 1, Tier 2, Tier 3 sections.

### 3. Verify Executable is Up-to-Date

```bash
cd /home/yzk/LBMProject/build
ls -lh visualize_lpbf_scanning
# Check timestamp - should be recent (after Nov 20 bug fixes)

# If needed, rebuild:
make visualize_lpbf_scanning
```

### 4. Prepare Output Directories

```bash
cd /home/yzk/LBMProject/build
mkdir -p tier1_smoke_test tier2_energy_balance
# Tier 3 uses existing steady_state_verification directory
```

---

## Estimated Timeline

| Tier | Task | Runtime | Analysis | Total |
|------|------|---------|----------|-------|
| **Tier 1** | Smoke test | 5 min | 2 min | **7 min** |
| **Tier 2** | Energy balance | 30 min | 10 min | **40 min** |
| **Tier 3** | Full steady-state (optional) | 150 min | 15 min | **165 min** |

**Fastest Path to GO Decision:** Tier 1 + Tier 2 = **47 minutes**
**High-Confidence Path:** Tier 1 + Tier 2 + Tier 3 = **3.5 hours** (can run Tier 3 overnight)

---

## Success Metrics: What "PASS" Means for Week 3

A successful completion of this testing strategy validates:

1. **Laser heating works correctly** (Tier 1)
   - Position fix verified
   - Energy input pathway functional

2. **Energy conservation holds** (Tier 2)
   - Multiphysics coupling preserves energy to < 5% error
   - No systematic drift or non-physical behavior

3. **Steady-state convergence achievable** (Tier 2/3)
   - System approaches thermal equilibrium
   - dE/dt → 0 as expected physically

4. **Platform numerically stable** (All tiers)
   - No NaN, Inf, or crashes over multi-millisecond runs
   - Bounded velocity and temperature fields

**This provides a solid foundation for Week 3 vapor phase implementation.**

---

## Failure Modes and Debugging

### If Tier 1 Fails: Laser Not Heating

**Symptoms:** T_max barely rises above 300 K after 100 μs

**Possible Causes:**
1. Laser position still wrong (check Bug #1 fix)
2. Laser absorptivity = 0 or not applied
3. Laser module disabled in config
4. Timestep too large (CFL violation in thermal solver)

**Debug Steps:**
```bash
# Check laser parameters in stdout
grep -i "laser" tier1_smoke_test/stdout.log

# Verify laser power in code
grep -n "laser_power\|absorptivity" apps/visualize_lpbf_scanning.cu

# Check thermal solver tau
# Should be tau < 2.0 for stability
```

### If Tier 2 Fails: Energy Error > 20%

**Symptoms:** Energy balance error persists > 20% throughout run

**Possible Causes:**
1. Energy computation kernels have bugs
2. Boundary conditions not accounted in energy balance
3. Timestep violates CFL in one of the solvers
4. Latent heat term missing or double-counted

**Debug Steps:**
```bash
# Check individual energy components
grep "E_thermal\|E_kinetic\|E_latent" tier2_energy_balance/energy_balance.csv | tail -10

# Verify power terms
grep "P_laser\|P_radiation\|P_substrate" tier2_energy_balance/energy_balance.csv | tail -10

# Check if any term is NaN or extremely large
awk '{if ($6 > 1e-3 || $6 < -1e-10) print "E_total anomaly:", $0}' \
    tier2_energy_balance/energy_balance.csv
```

### If Tier 2 Fails: Non-Convergence

**Symptoms:** dE/dt stays large (> 5 W) or increases with time

**Possible Causes:**
1. Boundary cooling insufficient (h_conv too small)
2. Numerical instability in fluid solver
3. Evaporation model broken (negative mass flux)
4. Domain too small (lacks thermal mass)

**Debug Steps:**
```bash
# Check if T_max is runaway heating
grep "T_max" tier2_energy_balance/stdout.log | tail -20

# Verify boundary cooling is active
grep "substrate\|radiation\|evaporation" tier2_energy_balance/stdout.log

# Check velocity field (if too large, fluid solver unstable)
# v_max should be < 5 m/s for stable melt pool
```

---

## Integration with Week 3 Planning

**After Successful Tier 2 PASS:**

The Week 3 vapor phase module can be implemented with confidence that:
- Energy accounting is correct
- Thermal boundary conditions work
- Multiphysics coupling is stable
- Steady-state validation framework is in place

**Vapor Phase Energy Terms to Add:**
- Latent heat of vaporization (∆H_vap ≈ 9.83 MJ/kg for Ti6Al4V)
- Recoil pressure energy transfer
- Vapor plume convective losses

**Testing Strategy for Week 3:**
- Use same multi-tier approach
- Add vapor-specific pass/fail criteria (e.g., keyhole formation)
- Validate vapor momentum coupling

---

## Conclusion

This multi-tier testing strategy provides:

1. **Speed:** 5-minute smoke tests for rapid iteration
2. **Rigor:** 30-minute energy balance checks for confidence
3. **Completeness:** Optional 3-hour full validation for final sign-off
4. **Actionability:** Clear GO/NO-GO decision criteria at each tier
5. **Debuggability:** Specific failure modes and diagnostic steps

**Recommended Next Steps:**

1. Verify energy diagnostics integration (Action Item #1)
2. Create three config files (Action Item #2)
3. Run Tier 1 (5 min)
4. If pass, run Tier 2 (30 min)
5. Make GO/NO-GO decision based on Tier 2 results

**Total time to decision: < 1 hour** (assuming no major failures)

---

**Document Status:** READY FOR EXECUTION
**Last Updated:** 2025-11-20
**Author:** LBM-CUDA Architect (Expert 3)
