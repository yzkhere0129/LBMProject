# URGENT ACTION PLAN - November 18, 2025, 17:45

## SITUATION REPORT

**Status**: CRITICAL DISCOVERY - Test C vs Test E comparison was INVALID

**Root Cause**: Binary version mismatch
- Test C (16:02): OLD binary ignoring config flags → Surface Tension ON, Phase Change ENABLED
- Test E (17:20): NEW binary respecting config flags → Surface Tension OFF, Phase Change OFF

**Impact**: We compared apples (enhanced physics) to oranges (minimal physics)

**Velocity Results**:
- Test C: 3.190 mm/s (with hidden physics boosts)
- Test E: 0.777 mm/s (minimal physics, plateau)

**Hypothesis**: VOF advection requires surface tension to maintain interface quality for Marangoni

---

## IMMEDIATE ACTIONS (Next 2 Hours)

### Priority 1: Test F - VOF + Surface Tension

**Purpose**: Validate if surface tension enables VOF-Marangoni coupling

**Command**:
```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning \
    --config ../configs/lpbf_195W_test_F_vof_surface.conf \
    > test_F_vof_surface.log 2>&1 &

# Monitor progress
tail -f test_F_vof_surface.log
```

**Expected Runtime**: ~3 minutes (5000 steps)

**Success Criteria**:
- v_max > 2 mm/s at 500 μs → Surface tension is critical
- v_max < 1 mm/s at 500 μs → Hypothesis wrong, deeper issues

---

### Priority 2: Test H - Full Physics Stack

**Purpose**: Test production configuration with all physics enabled

**Command**:
```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_scanning \
    --config ../configs/lpbf_195W_test_H_full_physics.conf \
    > test_H_full_physics.log 2>&1 &

# Monitor progress
tail -f test_H_full_physics.log
```

**Expected Runtime**: ~4 minutes (5000 steps, more physics)

**Success Criteria**:
- v_max > 5 mm/s at 500 μs → Production ready!
- v_max 2-5 mm/s → Acceptable, tune parameters
- v_max < 2 mm/s → Major physics bugs exist

---

## DECISION TREE (After Tests Complete)

### Scenario A: Test F > 2 mm/s AND Test H > 5 mm/s ✅ SUCCESS PATH

**Interpretation**:
- Surface tension is critical for VOF-Marangoni coupling
- Full physics stack achieves realistic velocity
- Architecture validated, coupling constraint identified

**Next Actions**:
1. Document coupling requirement: VOF advection REQUIRES surface tension
2. Update architecture docs with module dependency matrix
3. Add config validation to enforce constraint
4. Proceed to Week 2: Multi-spot simulations
5. **Timeline**: Back on track, Friday delivery

**Deliverables**:
- VOF-Marangoni coupling analysis report
- Updated architectural constraints document
- Config validation implementation
- Multi-spot test plan

---

### Scenario B: Test F < 1 mm/s OR Test H < 3 mm/s ❌ FAILURE PATH

**Interpretation**:
- VOF-Marangoni coupling fundamentally broken
- Physics stack has deeper implementation bugs
- Need major debugging effort

**Next Actions**:
1. Extract Marangoni force magnitude from VTK files
2. Analyze interface normal quality (∇φ smoothness)
3. Compare force fields: Test C (old binary) vs Test F/H (new binary)
4. Identify numerical instability or implementation error
5. **Timeline**: 2-3 day slip, extend Week 1 to Monday

**Diagnostics Required**:
```bash
# Extract force fields
cd /home/yzk/LBMProject/build

# Check Marangoni force magnitude evolution
python3 ../scripts/analyze_marangoni_force.py \
    lpbf_test_F_vof_surface/lpbf_*.vtk

# Check interface quality
python3 ../scripts/analyze_interface_normals.py \
    lpbf_test_F_vof_surface/lpbf_*.vtk
```

---

### Scenario C: Test F > 2 mm/s BUT Test H CRASHES ⚠️ STABILITY PATH

**Interpretation**:
- VOF + Surface Tension works
- Phase change causes numerical instability
- Need to fix phase change implementation

**Next Actions**:
1. Review phase change solver stability
2. Check latent heat source term implementation
3. Validate mushy zone Darcy damping coupling
4. Run Test G (phase change + VOF, no surface tension) to isolate
5. **Timeline**: 1-2 day slip, weekend work required

---

## BACKUP PLAN: Abandon VOF Approach

**If ALL tests fail (v_max < 1 mm/s or crashes)**:

### Alternative Strategy: Thermal Parameter Tuning

**Rationale**:
- Test C (old binary) achieved 3.2 mm/s with surface tension + phase change
- Maybe the solution is NOT VOF advection
- Focus on optimizing thermal parameters instead

**Test I Config**:
```
enable_phase_change = true       # Re-enable
enable_surface_tension = true    # Re-enable
enable_vof_advection = false     # DISABLE (back to static VOF)
thermal_diffusivity = 8.0e-6     # Increase (faster heat spread)
laser_penetration_depth = 10.0e-6  # Increase (deeper heating)
```

**Expected**: v_max = 4-6 mm/s (if thermal limits were bottleneck)

**Timeline**: 1 additional day, Friday delivery at risk

---

## MONITORING CHECKLIST

### During Test F/H Runs

Watch for:
- [ ] Velocity grows beyond 1 mm/s by 100 μs
- [ ] Temperature reaches 45,000+ K (melting confirmed)
- [ ] No NaN errors in log
- [ ] No "EXCEEDED MAX ITERATIONS" warnings
- [ ] Liquid fraction increases over time
- [ ] Output files generated correctly

### Red Flags

Stop immediately if:
- Velocity diverges (> 100 mm/s)
- Temperature > 100,000 K (unphysical)
- NaN detected in any field
- Simulation crashes mid-run

**Action**: Analyze last good timestep VTK file to identify divergence source

---

## COMMUNICATION PLAN

### If Tests Succeed (Scenario A)

**Message to Team**:
> "VOF-Marangoni coupling validated! Critical finding: VOF advection requires surface tension for interface stability. Full physics stack achieves v = X.X mm/s (within literature range). Proceeding to multi-spot simulations. Week 1 objective met."

**Deliverables Ready**:
- Coupling constraint documentation
- Production config template
- Multi-spot test plan

---

### If Tests Fail (Scenario B/C)

**Message to Team**:
> "VOF-Marangoni coupling issue identified. Root cause under investigation: [force magnitude drop / interface corruption / phase change instability]. Extending Week 1 debug phase to Monday. Multi-spot simulations pushed to next week."

**Escalation Required**:
- Request extension of Week 1 timeline
- Re-evaluate VOF approach viability
- Consider thermal parameter tuning as alternative

---

## RESOURCE REQUIREMENTS

### Computational

- **Test F**: ~3 min runtime, 51 VTK files (~2.5 GB)
- **Test H**: ~4 min runtime, 51 VTK files (~2.5 GB)
- **Total**: ~10 GB disk space, 10 minutes compute

### Personnel

- **Architect** (you): Monitor tests, analyze results, make decision (2 hours)
- **Physics expert**: Validate force field analysis if tests fail (4 hours, contingency)
- **Numerics expert**: Debug stability if Test H crashes (8 hours, contingency)

---

## SUCCESS METRICS

### Week 1 Objectives (Original)

1. Validate Marangoni convection (v = 5-15 mm/s) ← PENDING Test H
2. Identify optimal physics configuration ← PENDING Test F/H
3. Document coupling requirements ← PENDING analysis
4. Deliver production config template ← PENDING validation

### Revised Success Criteria (Post-Discovery)

1. **Minimum (Pass)**: Test H achieves v > 3 mm/s, no crashes
2. **Target (Good)**: Test H achieves v > 5 mm/s, stable to 500 μs
3. **Stretch (Excellent)**: Test H achieves v > 8 mm/s, matches literature

---

## LESSONS LEARNED (Preliminary)

### Process Failures

1. **Binary versioning not tracked** → Tests used different code
2. **Config validation missing** → Flags silently ignored for weeks
3. **Module coupling assumptions wrong** → VOF requires surface tension

### Architecture Gaps

1. **No hard coupling constraints enforced**
2. **No runtime config validation**
3. **No build timestamp logging**
4. **No automated test result comparison**

### Fixes Required (Post-Week 1)

1. Add `validatePhysicsConfig()` function
2. Log build timestamp and git commit in output
3. Document module dependency matrix
4. Create automated test regression suite

---

## APPENDIX: Quick Results Extraction

### After Test F/H Complete

```bash
cd /home/yzk/LBMProject/build

# Extract final velocities
grep "5000" test_F_vof_surface.log | tail -1
grep "5000" test_H_full_physics.log | tail -1

# Extract velocity at 100 μs
grep "1000" test_F_vof_surface.log | tail -1
grep "1000" test_H_full_physics.log | tail -1

# Check for errors
grep -i "nan\|error\|exceeded" test_F_vof_surface.log
grep -i "nan\|error\|exceeded" test_H_full_physics.log

# Check final statistics
grep -A5 "Final statistics" test_F_vof_surface.log
grep -A5 "Final statistics" test_H_full_physics.log
```

---

## CONTACT INFO

**Architect**: LBM Platform Architect (this agent)
**Escalation**: Project Lead (if timeline slip required)
**Technical Backup**: Physics/Numerics experts (if deep dive needed)

---

## STATUS TRACKING

- [ ] Test F launched (ETA: 17:50)
- [ ] Test H launched (ETA: 17:55)
- [ ] Test F completed (ETA: 17:53)
- [ ] Test H completed (ETA: 17:59)
- [ ] Results analyzed (ETA: 18:05)
- [ ] Decision made (ETA: 18:10)
- [ ] Action plan updated (ETA: 18:15)

**Next Check-In**: 18:00 (after both tests complete)

---
**END OF ACTION PLAN**
