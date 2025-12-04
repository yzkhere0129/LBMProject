# Test E Coordination Plan: VOF Advection Validation

## Executive Summary

**Objective**: Validate if enabling VOF advection (free surface deformation) unlocks higher Marangoni velocities by enabling interface-flow feedback loops.

**Hypothesis**: Static VOF interface in Test C/D artificially limits Marangoni flow to ~3 mm/s, preventing the physical reinforcement mechanism where interface deformation creates stronger thermal gradients.

**Configuration**: `/home/yzk/LBMProject/configs/lpbf_195W_test_E_vof_advection.conf`

**Key Change**: `enable_vof_advection = true` (Test D had `false`)

---

## 1. VTK Analysis Plan (vtk-simulation-analyzer)

### Primary Metrics to Extract

#### A. Interface Deformation Tracking
**Field**: `fill_level` (VOF field, 0 = gas, 1 = liquid)

**Analysis Tasks**:
```
1. Extract iso-surface at fill_level = 0.5 for all 51 frames
2. Compute interface position z_interface(x,y,t)
3. Calculate deformation amplitude: δz(t) = max(z_interface) - min(z_interface)
4. Track interface normal vectors: n = ∇(fill_level) / |∇(fill_level)|
5. Measure interface curvature: κ = ∇·n
```

**Expected Evolution** (if VOF working):
- **t = 100 μs**: δz ~ 0.5-1 μm (initial deformation)
- **t = 250 μs**: δz ~ 2-4 μm (significant asymmetry)
- **t = 500 μs**: δz ~ 4-8 μm (established Marangoni depression)

**Failure Indicators**:
- δz < 0.2 μm at 500 μs → VOF advection not working
- Interface smeared (fill_level = 0.3-0.7 over wide region) → Numerical diffusion
- Interface unchanged from Test D → Advection kernel not being called

#### B. Velocity Field Analysis
**Fields**: `velocity_x`, `velocity_y`, `velocity_z`

**Analysis Tasks**:
```
1. Compute velocity magnitude: v_mag = sqrt(vx² + vy² + vz²)
2. Extract v_max(t) for all frames
3. Identify Marangoni circulation cell: streamlines in melt pool
4. Calculate velocity gradients near interface: ∇v at fill_level = 0.5
5. Compare velocity patterns to Test D (static VOF baseline)
```

**Expected Patterns** (if hypothesis correct):
- **Early phase (0-100 μs)**: Similar to Test D, v ~ 1-2 mm/s
- **VOF feedback onset (100-250 μs)**: Velocity accelerates, v ~ 4-8 mm/s
- **Established circulation (250-500 μs)**: v ~ 6-12 mm/s, stable cell

**Diagnostic Checks**:
- Velocity maximum location: Should move with deforming interface
- Flow direction: Outward from hot center (Marangoni), downward return flow
- Vorticity: Strong near interface where thermal gradients are steepest

#### C. Thermal Field Coupling
**Field**: `temperature`

**Analysis Tasks**:
```
1. Extract temperature gradient at interface: ∇T at fill_level = 0.5
2. Calculate |∇T| magnitude along interface
3. Compare to Test D: expect HIGHER gradients if interface deforms
4. Correlate interface curvature with temperature distribution
```

**Physical Expectation**:
- Static interface (Test D): Flat → uniform thermal layer → weak ∇T
- Dynamic interface (Test E): Deformed → compressed hot region → strong ∇T
- Marangoni force ∝ |∇T| → higher gradients → stronger flow

#### D. Mass Conservation Validation
**Field**: `fill_level`

**Analysis Tasks**:
```
1. Integrate total liquid volume: V_liquid(t) = ∫∫∫ fill_level dV
2. Calculate mass drift: ΔV = |V_liquid(500μs) - V_liquid(0)| / V_liquid(0)
3. Check for interface smearing: histogram of fill_level values
```

**Acceptance Criteria**:
- ΔV < 1% over 500 μs (VOF advection conserves mass)
- Sharp interface: 95% of cells have fill_level < 0.1 or > 0.9
- No artificial liquid creation/destruction

**Failure Modes**:
- ΔV > 5% → VOF advection numerically unstable
- Broad histogram (many cells with 0.2 < fill_level < 0.8) → Excessive diffusion

---

## 2. GPU Performance Monitoring (cfd-cuda-architect)

### Kernel Performance Metrics

#### A. VOF Advection Kernel
**Kernel**: `advect_vof_kernel` (or equivalent)

**Metrics to Track**:
```
- Kernel execution time per step
- Memory bandwidth utilization
- Occupancy (should be > 50%)
- CFL condition: max(u·Δt/Δx) < 1.0
```

**Performance Expectations**:
- Execution time: 0.5-2 ms per step (for 200×100×50 domain)
- Bandwidth: 50-80% of peak (VOF advection is memory-bound)
- No divergence (uniform workload across domain)

**Alert Conditions**:
- Execution time > 5 ms → Kernel inefficiency
- Occupancy < 30% → Poor parallelization
- CFL > 0.9 → Stability risk, reduce timestep

#### B. Interface Reconstruction
**If using PLIC (Piecewise Linear Interface Calculation)**:

**Metrics**:
```
- Interface normal calculation time
- Number of interface cells (should be ~surface area / dx²)
- Reconstruction accuracy (angle errors)
```

**Diagnostic**:
- Too many interface cells → Interface smearing
- Slow normal calculation → Consider optimizing gradient stencil

#### C. Overall Simulation Performance
**Compare to Test D** (both 5000 steps, same domain):

| Metric | Test D (static VOF) | Test E (dynamic VOF) | Expected Overhead |
|--------|---------------------|----------------------|-------------------|
| Time/step | ~15 ms | ~18-20 ms | +20-33% |
| Total runtime | ~75 s | ~90-100 s | Acceptable if v_max improves |

**Unacceptable Performance**:
- Test E > 2× slower than Test D → VOF implementation needs optimization
- GPU memory overflow → Domain too large for VOF advection

---

## 3. Physical Validation (test-debug-validator)

### Comparison Matrix: Test D vs Test E

#### A. Velocity Evolution Comparison
**Checkpoint Analysis**:

| Time | Test D (static) | Test E Target | Status | Interpretation |
|------|-----------------|---------------|--------|----------------|
| 100 μs | 1.8 mm/s | 1.8-2.5 mm/s | Early phase | Interface deformation starting |
| 250 μs | 2.5 mm/s | 4-8 mm/s | VOF effect | Feedback loop active |
| 500 μs | 3.3 mm/s | 6-12 mm/s | Full circulation | Marangoni reinforced |

**Success Criteria**:
- **v_max(500μs) ≥ 8 mm/s**: FULL SUCCESS → VOF was the bottleneck
- **v_max(500μs) = 5-8 mm/s**: PARTIAL SUCCESS → VOF helps but not sufficient
- **v_max(500μs) = 3-5 mm/s**: MARGINAL → Minimal improvement
- **v_max(500μs) < 3 mm/s**: FAILURE → VOF makes it worse
- **Crash/NaN**: INSTABILITY → Advection scheme broken

#### B. Interface Deformation Metrics
**Quantitative Checks**:

```python
# Extract from VTK at 500 μs
interface_deformation = max(z_interface) - min(z_interface)
interface_rms_deviation = sqrt(mean((z_interface - z_mean)²))
interface_gradient_max = max(|∇T|) at fill_level = 0.5

# Compare to Test D
if interface_deformation > 2 μm:
    print("SUCCESS: Interface deforming significantly")
elif interface_deformation > 0.5 μm:
    print("MARGINAL: Minor deformation")
else:
    print("FAILURE: Interface still static")
```

#### C. Energy Balance Validation
**Check physical consistency**:

```
Marangoni work rate: W_M = ∫ F_M · v dV
Viscous dissipation: Q_visc = ∫ μ (∇v)² dV
Interface energy: E_surf = ∫ σ κ dA

# Energy conservation (should hold):
d(KE)/dt + Q_visc = W_M + W_buoyancy
```

**Expected**:
- Test E: Higher W_M (stronger Marangoni work) due to larger ∇T
- Test E: Higher Q_visc (more dissipation) due to higher velocities
- Balance: Net energy increase should match increased laser input utilization

#### D. Numerical Stability Checks
**At each checkpoint (100 μs, 250 μs, 500 μs)**:

```
1. Check for NaN/Inf in all fields
2. Verify CFL conditions:
   - CFL_advection = max(v·Δt/Δx) < 0.5
   - CFL_diffusion = α·Δt/Δx² < 0.25
3. Monitor fill_level bounds: 0 ≤ fill_level ≤ 1
4. Check temperature positivity: T > 0 K
5. Verify velocity doesn't exceed physical limits: v < 10 m/s
```

**Failure Actions**:
- NaN detected → Abort, investigate kernel at failure step
- CFL violation → Reduce dt or implement adaptive timestep
- fill_level < 0 or > 1 → VOF advection has mass conservation bug

---

## 4. Decision Matrix: Next Steps Based on Results

### Scenario A: FULL SUCCESS (v_max ≥ 8 mm/s)

**Interpretation**:
- VOF advection was the primary bottleneck
- Interface deformation enables Marangoni feedback loop
- Static VOF artificially constrained flow in Test C/D

**Evidence Required**:
- ✓ Interface deformation > 3 μm at 500 μs
- ✓ Velocity increases correlate with interface deformation onset
- ✓ Thermal gradients at interface 2-3× stronger than Test D
- ✓ Mass conservation maintained (ΔV < 1%)

**Next Step**: **Test F - Enable Surface Tension**
```
Config: Test E + enable_surface_tension = true
Goal: Complete VOF physics with curvature-driven forces
Expected: v_max = 8-20 mm/s (surface tension may increase or dampen)
```

---

### Scenario B: PARTIAL SUCCESS (v_max = 5-8 mm/s)

**Interpretation**:
- VOF advection helps but not sufficient
- Interface deformation is occurring but reinforcement is weaker than expected
- Likely need BOTH VOF + another fix (thermal or surface tension)

**Evidence Required**:
- ✓ Interface deformation = 1-3 μm (moderate)
- ✓ Velocity improvement = +50% to +140% vs Test D
- ✓ Marangoni circulation cell established but flow weaker than target

**Next Step**: **Test F - Combined VOF + Enhanced Thermal**
```
Config: Test E + enhanced thermal gradient calculation
Options:
  1. Increase laser power: 195W → 250W (more realistic for Ti6Al4V)
  2. Sharper laser profile: reduce spot radius 50μm → 40μm
  3. Higher dsigma_dT: -0.26 → -0.40 mN/(m·K) (check literature)
Expected: v_max = 10-20 mm/s
```

---

### Scenario C: MARGINAL (v_max = 3-5 mm/s)

**Interpretation**:
- VOF advection active but effect is small
- Interface deformation insufficient to change gradients significantly
- Bottleneck may be elsewhere (thermal diffusion, viscosity, laser model)

**Evidence Required**:
- ✓ Interface deformation = 0.5-1 μm (small)
- ✓ Velocity improvement = +0% to +50% vs Test D (marginal)
- ✓ Interface deforms but doesn't affect thermal field much

**Diagnostic Actions**:
1. Check thermal gradient calculation: Is ∇T accurate at interface?
2. Verify Marangoni force magnitude: F_M = dsigma_dT · ∇T
3. Test with higher laser power to create stronger gradients
4. Consider if thermal diffusion timestep is washing out gradients

**Next Step**: **Test F - Thermal Diagnostic**
```
Config: Test E + print detailed thermal gradient diagnostics
Add: Output ∇T field, Marangoni force field, thermal diffusion rate
Goal: Identify why interface deformation doesn't affect flow
```

---

### Scenario D: FAILURE (v_max < 3 mm/s)

**Interpretation**:
- VOF advection makes it WORSE
- Possible causes:
  1. VOF advection introduces excessive numerical diffusion
  2. Interface smearing destroys sharp thermal gradients
  3. Mass conservation errors disrupt flow field
  4. Kernel bug causing incorrect advection

**Evidence Required**:
- ✗ Velocity DECREASED vs Test D
- ✗ Interface smeared (many cells with 0.2 < fill_level < 0.8)
- ✗ Mass drift > 5%
- ✗ Temperature gradients WEAKER than Test D (diffusion)

**Diagnostic Actions**:
1. **Immediate**: Check VOF advection implementation
   - Verify upwind scheme is correctly implemented
   - Check for donor-acceptor errors
   - Test on simple advection benchmark (rotating disk)

2. **Interface Reconstruction**: If using PLIC, check:
   - Normal vector calculation accuracy
   - Interface angle reconstruction
   - Flux calculation at cell faces

3. **Numerical Diffusion**: Quantify diffusion:
   - Track interface sharpness: width of 0.4 < fill_level < 0.6 region
   - Should be 1-2 cells if sharp, >5 cells indicates excessive diffusion

**Next Step**: **Revert to Static VOF, Fix Advection Kernel**
```
Action: Keep Test D configuration until VOF advection debugged
Parallel: Implement higher-order VOF scheme (CICSAM, HRIC, or Geo-Reconstruct)
Test: Validate on Zalesak's disk or dam break benchmark
```

---

### Scenario E: INSTABILITY (Crash/NaN)

**Interpretation**:
- VOF advection causes numerical instability
- CFL violation, mass conservation breakdown, or kernel bug

**Diagnostic Actions**:
1. **Identify failure step**: When did NaN first appear?
   - Use binary search: output every 10 steps until crash isolated

2. **Check CFL condition**:
   ```
   CFL_vof = max(v · Δt / Δx)
   If CFL > 1.0 → Interface advection unstable
   ```

3. **Inspect field at failure**:
   - Look for velocity spikes near interface
   - Check if fill_level goes out of bounds [0,1]
   - Verify temperature field is still physical

4. **Kernel-level debugging**:
   - Add printf statements in VOF advection kernel
   - Check for division by zero, sqrt of negative, etc.
   - Verify array bounds (no out-of-bounds memory access)

**Next Step**: **Fix Stability, Reduce Timestep**
```
Option 1: Reduce dt from 1e-7 to 5e-8 (halve timestep)
Option 2: Implement CFL limiter specifically for VOF advection
Option 3: Add interface smoothing (1-2 iterations of Gaussian filter)
Rerun: Test E with stable parameters
```

---

## 5. Monitoring Checkpoints

### Checkpoint 1: 100 μs (Step 1000)

**Quick Checks**:
```bash
# Extract velocity max
vtk-simulation-analyzer extract-velocity lpbf_test_E_vof_advection --step 1000
Expected: v_max ≈ 1.8-2.5 mm/s

# Check interface deformation
vtk-simulation-analyzer extract-interface lpbf_test_E_vof_advection --step 1000
Expected: δz ≈ 0.5-1 μm (starting to deform)
```

**Alert Conditions**:
- v_max < 1.5 mm/s → Slower than Test D, investigate immediately
- δz < 0.1 μm → Interface not moving, VOF advection not working
- NaN/Inf → Abort, debug kernel

**Action if Alert**:
- STOP simulation
- Check console output for CFL violations or warnings
- Inspect VTK at step 1000 for anomalies

---

### Checkpoint 2: 250 μs (Step 2500)

**Detailed Analysis**:
```bash
# Full VTK analysis
vtk-simulation-analyzer compare-to-testD lpbf_test_E_vof_advection --step 2500
Expected:
  - v_max ≈ 4-8 mm/s (+60% to +220% vs Test D)
  - δz ≈ 2-4 μm (significant deformation)
  - ∇T at interface: 2-3× stronger than Test D
```

**Decision Point**:
- **v_max > 6 mm/s**: ON TRACK for full success, continue to 500 μs
- **v_max = 4-6 mm/s**: PARTIAL success likely, continue to confirm
- **v_max < 4 mm/s**: MARGINAL or FAILURE, consider early abort
  - If interface not deforming → VOF not working, stop and debug
  - If interface deforming but v low → Physics issue, continue to 500 μs for full data

---

### Checkpoint 3: 500 μs (Step 5000) - FINAL

**Comprehensive Analysis**:

#### Quantitative Metrics
```python
# vtk-simulation-analyzer final report
metrics = {
    "v_max": extract_velocity_max(step=5000),
    "interface_deformation": compute_interface_amplitude(step=5000),
    "thermal_gradient_max": compute_gradient_at_interface(step=5000),
    "mass_conservation": compute_mass_drift(step=5000),
    "marangoni_cell_size": measure_circulation_cell(step=5000)
}

# Classification
if metrics["v_max"] >= 8.0:
    result = "FULL SUCCESS"
elif metrics["v_max"] >= 5.0:
    result = "PARTIAL SUCCESS"
elif metrics["v_max"] >= 3.0:
    result = "MARGINAL"
else:
    result = "FAILURE"
```

#### Qualitative Assessment
```
1. Interface morphology: Smooth? Rough? Symmetric?
2. Velocity field structure: Single vortex? Multiple cells?
3. Temperature distribution: Hotspot centered? Asymmetric?
4. Flow stability: Steady? Oscillating? Turbulent?
```

#### Comparative Visualization
Generate comparison plots:
```
Figure 1: Velocity evolution
  - Test C (Darcy=1000, static VOF): v_max vs time
  - Test D (Darcy=500, static VOF): v_max vs time
  - Test E (Darcy=1000, dynamic VOF): v_max vs time

Figure 2: Interface profile at 500 μs
  - Test D: z_interface(x) (should be nearly flat)
  - Test E: z_interface(x) (should show deformation)

Figure 3: Temperature gradient at interface
  - Test D: |∇T| along interface
  - Test E: |∇T| along interface (should be higher if deformed)

Figure 4: Velocity streamlines at 500 μs
  - Test D: Marangoni cell structure (constrained by flat interface)
  - Test E: Marangoni cell structure (enhanced by deformed interface)
```

---

## 6. Expected Physical Mechanisms

### If VOF Advection Enables Feedback Loop

**Mechanism**:
```
Step 1: Laser heats interface → Temperature gradient ∇T
Step 2: Marangoni force F_M = dsigma_dT · ∇T → Flow outward from hot center
Step 3: Flow advects interface → Depression forms at hot center
Step 4: Depression compresses thermal boundary layer → STRONGER ∇T
Step 5: Stronger ∇T → STRONGER F_M → FASTER flow
Step 6: Faster flow → MORE deformation → LOOP REINFORCES
```

**Mathematical Expression**:
```
Static VOF:  F_M = dsigma_dT · ∇T_static → v = v_0
Dynamic VOF: ∇T_dynamic = ∇T_static / (1 - ε·v/v_ref)
             F_M_dynamic = dsigma_dT · ∇T_dynamic = F_M_static / (1 - ε·v/v_ref)
             v = v_0 / (1 - ε·v/v_ref) → v = v_0 / (1 - ε) if ε·v/v_ref << 1

Amplification: v_dynamic / v_static = 1 / (1 - ε)
               If ε = 0.5 → 2× amplification
               If ε = 0.75 → 4× amplification
```

**Evidence in VTK Data**:
- Interface curvature κ highest at center (where T is max)
- Thermal gradient |∇T| inversely proportional to local interface height
- Velocity maximum location moves with interface deformation
- Correlation: v_max(t) increases as δz(t) increases

---

### If VOF Advection Does NOT Help

**Possible Reasons**:

1. **Thermal Diffusion Too Strong**:
   ```
   Peclet number: Pe = v·L/α
   If Pe << 1 → Diffusion dominates, interface deformation doesn't matter
   Check: α·Δt/Δx² (should be ~0.1-0.2, not >0.5)
   ```

2. **Viscous Damping Too Strong**:
   ```
   Reynolds number: Re = v·L/ν
   If Re << 1 → Viscous forces suppress flow
   Check: ν value (should be ~3e-7 m²/s for molten Ti6Al4V, not 3e-6)
   ```

3. **Laser Power Too Low**:
   ```
   If |∇T| at interface is too weak:
   F_M = dsigma_dT · ∇T is small even with interface deformation
   Check: T_max > 3000 K? If not, increase laser power
   ```

4. **Interface Deformation Too Small**:
   ```
   If δz < 1 μm, effect on ∇T is negligible
   Possible causes:
   - Surface tension too strong (if enabled, check σ value)
   - Viscosity too high (damping interface motion)
   - VOF advection timestep too conservative (undershoot)
   ```

---

## 7. Timeline Estimate

| Phase | Duration | Cumulative | Task |
|-------|----------|------------|------|
| **Setup** | 5 min | 5 min | Copy Test E config, verify parameters |
| **Compilation** | 2 min | 7 min | Rebuild if needed (VOF advection already in code) |
| **Simulation** | ~90 sec | 9 min | 5000 steps @ ~18 ms/step |
| **VTK Loading** | 2 min | 11 min | Load 51 frames into ParaView/custom analyzer |
| **Quick Check** | 3 min | 14 min | Extract v_max(t), classify result |
| **Detailed Analysis** | 15 min | 29 min | Interface deformation, thermal gradients, comparison plots |
| **Report Generation** | 10 min | 39 min | Synthesize findings, recommendation |

**Total: ~40 minutes** from start to final decision

**Parallel Efficiency**:
- vtk-simulation-analyzer can start loading VTK while simulation runs
- cfd-cuda-architect monitors GPU during simulation (real-time)
- test-debug-validator prepares comparison scripts in parallel

---

## 8. Success Metrics Summary

| Metric | Measurement | Target | Source |
|--------|-------------|--------|--------|
| **v_max @ 500 μs** | Velocity magnitude | ≥ 8 mm/s | VTK velocity field |
| **Interface deformation** | max(z) - min(z) | ≥ 3 μm | VTK fill_level isosurface |
| **Thermal gradient increase** | max(|∇T|) at interface | 2-3× Test D | VTK temperature field |
| **Mass conservation** | \|ΔV\| / V_0 | < 1% | VTK fill_level integral |
| **Interface sharpness** | Width of 0.4 < fill_level < 0.6 | < 3 cells (6 μm) | VTK fill_level histogram |
| **CFL compliance** | max(v·Δt/Δx) | < 0.5 | Console output |
| **Simulation time** | GPU execution time | < 2× Test D (~150 s) | CUDA profiler |

---

## 9. Communication Protocol

### Real-time Updates (Every 100 μs during simulation)

**cfd-cuda-architect** reports:
```
Step 1000: GPU time = 18.2 ms/step, CFL_max = 0.32, Memory OK
```

**test-debug-validator** reports:
```
Step 1000: v_max = 2.1 mm/s, T_max = 3200 K, No NaN detected
```

### Checkpoint Reports (100 μs, 250 μs, 500 μs)

**vtk-simulation-analyzer** reports:
```
Checkpoint 2500 (250 μs):
  - v_max = 6.2 mm/s (+86% vs Test D @ 250 μs)
  - Interface deformation: δz = 2.8 μm
  - Thermal gradient: |∇T|_max = 4.2e6 K/m (2.3× Test D)
  - Mass conservation: ΔV = 0.3%
  Status: ON TRACK for FULL SUCCESS
```

### Final Report (500 μs)

**Lead architect synthesizes**:
```
TEST E RESULT: FULL SUCCESS

v_max = 9.4 mm/s (EXCEEDED 8 mm/s target, +182% vs Test D)

Conclusion: VOF advection was the primary bottleneck limiting Marangoni flow

Evidence:
  ✓ Interface deformed 4.1 μm (significant)
  ✓ Thermal gradients 2.7× stronger than static interface (Test D)
  ✓ Velocity correlates with interface deformation onset
  ✓ Mass conserved (ΔV = 0.6%)
  ✓ Marangoni circulation cell fully established

Recommendation: Proceed to TEST F - Enable surface tension
  - Config: Test E + enable_surface_tension = true
  - Goal: Complete VOF physics with curvature forces
  - Expected: v_max = 10-25 mm/s (surface tension may regulate flow)

Physical Insight:
  Static VOF artificially constrained Marangoni feedback loop.
  Dynamic interface enables thermal compression → stronger gradients → reinforced flow.
  This explains why Test C/D (3 mm/s) << physical LPBF (100s mm/s).
```

---

## 10. Deliverables Checklist

- [x] Test E configuration file created: `/home/yzk/LBMProject/configs/lpbf_195W_test_E_vof_advection.conf`
- [x] VTK analysis plan documented (Section 1)
- [x] GPU monitoring plan documented (Section 2)
- [x] Physical validation criteria defined (Section 3)
- [x] Decision matrix for all scenarios (Section 4)
- [x] Expected interface deformation patterns (Section 6)
- [x] Timeline estimate (Section 7)
- [x] Success metrics table (Section 8)
- [ ] **EXECUTE SIMULATION** (Ready to launch)
- [ ] **ANALYZE RESULTS** (Post-simulation task)
- [ ] **GENERATE FINAL REPORT** (Post-analysis task)

---

## Appendix A: Quick Command Reference

### Run Test E
```bash
cd /home/yzk/LBMProject/build
./LBMSolver ../configs/lpbf_195W_test_E_vof_advection.conf
```

### Quick Velocity Check
```bash
vtk-simulation-analyzer extract-max-velocity \
  --dir /home/yzk/LBMProject/output/lpbf_test_E_vof_advection \
  --step 5000
```

### Interface Deformation Analysis
```bash
vtk-simulation-analyzer extract-interface-profile \
  --dir /home/yzk/LBMProject/output/lpbf_test_E_vof_advection \
  --step 5000 \
  --output interface_profile_E.csv
```

### Compare to Test D
```bash
vtk-simulation-analyzer compare-tests \
  --test-d /home/yzk/LBMProject/output/lpbf_test_D_darcy500 \
  --test-e /home/yzk/LBMProject/output/lpbf_test_E_vof_advection \
  --output comparison_D_vs_E.png
```

---

## Appendix B: Physics Equations Reference

### Marangoni Force
```
F_M = ∇_s·σ = (dσ/dT)·∇_s T
where ∇_s is surface gradient (tangent to interface)
```

### Interface Deformation (Kinematic Condition)
```
∂h/∂t + u·∇h = v_normal
where h is interface height, v_normal is normal velocity
```

### VOF Advection Equation
```
∂φ/∂t + ∇·(φ·v) = 0
where φ is fill_level (volume fraction)
```

### Thermal Gradient Compression by Interface Deformation
```
If interface depression δz at hot center:
  Boundary layer thickness: δ_T ∝ sqrt(α·t)
  Effective thickness: δ_T_eff = δ_T - δz (compressed)
  Gradient: ∇T ~ ΔT / δ_T_eff = ΔT / (δ_T - δz)

Amplification: ∇T_dynamic / ∇T_static = δ_T / (δ_T - δz)
               If δz = 0.5·δ_T → 2× stronger gradient
```

---

**Test E is READY for execution.**

**Lead Architect Recommendation**: Proceed with simulation. Monitor checkpoints at 100 μs, 250 μs. If v_max ≥ 6 mm/s at 250 μs, continue to full 500 μs. If v_max < 4 mm/s at 250 μs, consider early termination for diagnostics.
