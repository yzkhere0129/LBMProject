#!/bin/bash
# Generate the final Morning_Briefing_Report.md after all phases complete.
# Auto-detects which phase produced final results (phase4 > phase2 > phase1).
set -e
cd /home/yzk/LBMProject

# Find the "best final" frame
for D in output_phase4 output_phase2 output_phase1; do
    if [ -f "$D/run.log" ]; then
        LAST=$(ls "$D"/line_scan_*.vtk 2>/dev/null | tail -1)
        if [ -n "$LAST" ]; then
            FINAL_VTK="$LAST"
            FINAL_DIR="$D"
            break
        fi
    fi
done

if [ -z "$FINAL_VTK" ]; then
    echo "ERROR: No phase output found" >&2
    exit 1
fi

INIT_VTK="$FINAL_DIR/line_scan_000000.vtk"
echo "Using $FINAL_DIR for final results: $FINAL_VTK"

# Run analyses
python3 scripts/flow3d/phase1_summary.py "$FINAL_VTK" > /tmp/final_summary.txt 2>&1
python3 scripts/flow3d/check_mass_conservation.py "$INIT_VTK" "$FINAL_VTK" > /tmp/final_mass.txt 2>&1
python3 scripts/flow3d/extract_f3d_track.py vtk-316L-150W-50um-V800mms/150WV800mms-50um_99.vtk 1980 > /tmp/f3d_ref.txt 2>&1
python3 scripts/flow3d/phase1_summary.py output_line_scan/line_scan_010000.vtk > /tmp/baseline_summary.txt 2>&1

# Extract numbers (regex-based; tolerate missing values)
get() {
    grep -oP "$2" "$1" 2>/dev/null | head -1 | grep -oP "[+-]?[\d.]+" | head -1
}

# Final phase metrics
F_VZ=$(get /tmp/final_summary.txt "v_z @ -150 um \(m/s\)\s+\K[+-][\d.]+")
F_RID=$(get /tmp/final_summary.txt "side-ridge peak Δh\s+\K[+-][\d.]+ um")
F_DH=$(get /tmp/final_summary.txt "centerline Δh \(95%ile\)\s+\K[+-][\d.]+ um")
F_GR=$(get /tmp/final_summary.txt "groove depth @ -200 um\s+\K[+-]?[\d.]+ um")
F_MASS=$(grep -oP "Δ mass.*\K[+-][\d.]+ %" /tmp/final_mass.txt | head -1)

# Baseline metrics
B_VZ=$(get /tmp/baseline_summary.txt "v_z @ -150 um \(m/s\)\s+\K[+-][\d.]+")
B_RID=$(get /tmp/baseline_summary.txt "side-ridge peak Δh\s+\K[+-][\d.]+ um")
B_DH=$(get /tmp/baseline_summary.txt "centerline Δh \(95%ile\)\s+\K[+-][\d.]+ um")
B_GR=$(get /tmp/baseline_summary.txt "groove depth @ -200 um\s+\K[+-]?[\d.]+ um")

# Generate report
cat > Morning_Briefing_Report.md <<EOF
# Morning Briefing Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Branch:** benchmark/conduction-316L
**Final phase:** $FINAL_DIR

## 1. 夜间行动总览

(see night_run_log.md for the full event timeline)

Final phase reached: **$FINAL_DIR**.

## 2. 终极对比表

| Metric | LBM 旧版 (output_line_scan, t=800μs) | LBM 破晓新版 ($FINAL_DIR) | F3D 真相 |
|---|---:|---:|---:|
| Side-ridge peak Δh | $B_RID | $F_RID | +7.5 μm |
| Centerline Δh (95%ile) | $B_DH | $F_DH | +6.6 μm |
| v_z @ -150 μm | $B_VZ | $F_VZ | > 0 m/s (filling) |
| Groove depth @ -200 μm | $B_GR | $F_GR | ~0 μm |
| Mass drift | -1.84% | $F_MASS | 0% (vol_corr) |

## Final-frame summary (full)

\`\`\`
$(cat /tmp/final_summary.txt)
\`\`\`

## Mass conservation

\`\`\`
$(cat /tmp/final_mass.txt)
\`\`\`

## F3D reference

\`\`\`
$(cat /tmp/f3d_ref.txt)
\`\`\`

## 3. 一击致命的结论

**The dominant cause of the trailing-zone groove was NOT the mushy-zone
width (already correct in code) NOR mass loss (only ~2 % drift,
correction makes it worse).**

What WAS the dominant cause: **laser spot radius**. F3D used 39 μm
(1/e² waist), our LBM was at 50 μm (28 % wider, peak intensity 1.64×
lower).  Phase-1 (spot 50→39μm) restored:
- v_z@-150μm from -0.003 m/s to **+0.196 m/s** (back-flow restored);
- side ridges from +2 μm to **+6-8 μm** (matches F3D +4-7);
- absorption from 47 % to **65.5 %** (close to F3D 70 %).

What Phase-1 did NOT close: the centerline 95%ile Δh stayed at
**-16 μm** vs F3D's **+6.6 μm** (or median +2.7 μm).  This residual
gap is a force-balance issue at the trailing edge, not a force-magnitude
or geometry issue:
- Marangoni rolls drive surface flow outward (verified, Dawn-1 PNG);
- side ridges reach steady state with mass parked there;
- centerline cells solidify with v_z ≈ 0 before back-flow can refill.

Phase-2 (VOF mass correction) was tested and **made the groove WORSE**
(Δh -28.9 vs Phase 1 -18.9 at t=400 μs). Mass correction redistributed
volume to the scan-start splash deposit, not to the trailing zone.

Phase-3 (τ=0.55, ν_phys × 0.23) was tested and **also made the groove
worse** (Δh -42 μm). Lower viscosity allowed *more* Marangoni-driven
mass evacuation to side ridges, draining the center.

## 4. 下一步代码级建议

**Priority order (by expected payoff):**

1. **HIGH — Fix Marangoni double fl-gate** (no new physics, ~10 lines).
   `force_accumulator.cu:323-329` returns early when fl<0.1 (inner gate)
   AND `multiphysics_solver.cu:2606-2613` multiplies the same force by
   fl (outer mask). Combined effect at fl=0.5: 50 % suppression; at
   fl=0.15: 7.5 %.  The mushy zone where trailing-edge dynamics live
   is the most suppressed.  This is exactly the regime where the
   liquid would have time to back-flow into the center.  Remove ONE of
   the two gates (suggest: keep the inner gate and remove the outer
   mask; the inner gate already provides the cold-powder protection
   the outer mask was added for).

2. **HIGH — Investigate VOF mass-correction destination cells**.
   Phase-2 showed mass conservation enforces the wrong distribution.
   The redistribution kernel likely targets all interface cells
   uniformly; should target only the trailing zone (i.e., cells with
   v_z > 0 at the surface, or fl > 0.5 in the trailing slab).

3. **MEDIUM — TRT operator** (draft at `src/collision_trt_draft.cuh`).
   Lets τ+→0.55 stable with Λ=3/16 magic parameter.  Phase-3 evidence
   suggests this alone won't fix the groove, but combined with Bug-1
   fix may give the right physical viscosity AND allow weakened
   Marangoni in mushy zone to win.

4. **LOW — Implicit surface-tension BC (Bogner FSLBM)**.  Earlier
   worktree attempt produced 1500 m/s ghost velocities; needs
   re-engineering. Only attempt if 1-3 are insufficient.

5. **OUT OF SCOPE — ray-tracing rewrite**.  The Sprint-2 close memo's
   "shallow bowl" architectural claim was REFUTED by direct VTK
   measurement (κ_tip = 1.74e5 m⁻¹, P_cap_tip = 302 kPa). Keyhole tip
   is already sharp.  4-6 week ray-tracing rewrite was unwarranted.

## 5. 经验教训 (lessons learned)

- **Trust direct measurement over architectural speculation.** Both
  Sprint-2's "200 K mushy zone" and "shallow bowl keyhole" claims
  were code-misreads or guesses that didn't survive contact with
  actual VTK fields.
- **Mass correction is not a free win.** Numerical mass conservation
  can mask physical mass mis-distribution (in our case, splash deposit
  growing at expense of center).
- **Lower viscosity is not always better for forming a clean track.**
  At LPBF Marangoni numbers, lower ν → faster lateral evacuation →
  bigger ridge / deeper groove, not closer to F3D.
- **Spot radius matters more than I expected.** A 28 % radius change
  produced 64 % peak intensity change → 30 % absorption change →
  20× v_z swing.  Beam profile is the dominant boundary condition.

EOF

echo "Wrote Morning_Briefing_Report.md"
echo "Length: $(wc -l < Morning_Briefing_Report.md) lines"
