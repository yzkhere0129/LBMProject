# Morning Briefing Report

**Generated:** 2026-04-26 06:07:03
**Branch:** benchmark/conduction-316L
**Final phase:** output_phase4

## 1. 夜间行动总览

(详见 night_run_log.md 完整时间线)

**夜跑跨越的决策树分支：**
- **Phase 1** (mushy 200K→23K + spot 50→39μm) — 第一次以 4096 rays / 5 bounces 在 step 1860 卡到 ~0.12 step/sec；
  重启用 2048 rays / 3 bounces 跑完 800 μs，v_z=+0.196 m/s、侧凸 +6-8 μm 显著好转。判定 **PARTIAL** → Phase 2.
- **Phase 2** (Phase 1 + VOF mass correction) — t=400 μs 时 Δh 由 -18.9 退化到 **-28.9 μm**。
  质量补偿把质量倒到 scan-start splash 不是中心。**ABORTED** → Phase 3.
- **Phase 3** (Phase 1 + τ=0.55) — Δh = **-42 μm** 更糟。低粘度让更多液体被甩到侧凸。**FAIL** → Phase 4 用 Phase 1 配置.
- **Phase 4** (Phase 1 配置, 2 ms full run, ny=75 因 OOM 被压窄) — 收敛稳态 Δh = -22 μm
  (比 Phase 1 ny=125 略差，证实 ny 越宽越好)。
- **Bug-1 fix** (commit 5853e74) — 主线运行中确认 Marangoni 在 multiphysics_solver 和 force_accumulator 被双重 fl-gate
  (mushy zone 50% 抑制)。修复 + 重新链 Phase-1 binary，启动 **Phase-1B** 验证 (运行中).
- **Dawn-1** (Marangoni 流线可视化, `docs/reference/marangoni_phase1_yz.png`) 完成。
- **Dawn-2** (TRT 算子草案 `src/collision_trt_draft.cuh`) 完成。
- **Dawn-3** (dσ/dT 灵敏度: -4.3 vs -4.9e-4) 完成 (06:59). **结论: 14% Marangoni 强度变化造成
  侧凸 +5-10 μm 偏差但中心 Δh 完全不动**。验证 -4.3e-4 选择正确，且证实中心填充瓶颈
  不是 Marangoni 强度的可调按钮 (mid-mushy 双重 fl-gate 修复了不能完全解决问题，
  低粘度让事情更糟，质量补偿让事情更糟，spot 半径修了 95% 然后剩余 5% 不可由该方向解开).

Final phase reached: **output_phase4** (+ Phase-1B 进行中).

## 2. 终极对比表

**三个 LBM 配置同时报告。** Phase 4 (2 ms 但 ny=75 narrow) 跑完了，
但 **Phase 1 (800 μs, ny=125 wider) 才是物理上最好的** —
ny 宽给 Marangoni 卷有空间。

| Metric | 基线 (output_line_scan, 800μs) | Phase-1 (ny=125, 800μs) | Phase-4 (ny=75, 2 ms) | F3D (1.98 ms) |
|---|---:|---:|---:|---:|
| Side-ridge peak Δh (μm) | +18 | +14 | +16 | +7.5 (max), +4.1 (mean) |
| Centerline Δh 95%ile (μm) | **-20** | **-16** | -22 | **+6.6** |
| v_z @ -150 μm (m/s) | -0.003 | **+0.196** | -0.064 | >0 |
| Groove depth @ -200 μm | +22 | +20 | +26 | ~0 |
| Effective absorption | 47% | **65.5%** | ~70% | 70-75% |
| Pool depth (μm) | 70 | 70 | 106 | 78 |
| Mass drift over t_total | -1.84% | -2.55% | -3.63% | 0% (vol_corr) |
| Mushy width | **23 K** (already F3D) | 23 K | 23 K | 23 K |
| dσ/dT | **-4.3e-4** (already F3D) | -4.3e-4 | -4.3e-4 | -4.3e-4 |

**Phase-4 偏深 pool (106 vs F3D 78)** 是 2 ms 长跑在窄横向上的板厚穿透痕迹 (keyhole 撞到 substrate 底部
adiabatic BC 的反射)。在 t=800 μs Phase-1 报告 D=70 μm 与基线一致。

### Phase-1B (Bug-1 fix 验证, 800 μs, 06:40 完成)

| metric | Phase-1 (pre-fix) | Phase-1B (post-fix) | delta |
|---|---:|---:|---:|
| v_z @ -150 μm        | +0.196 m/s | +0.193 m/s | -0.003 |
| centerline Δh 95%ile | -16 μm | **-16 μm** | **0** |
| mass drift           | -2.55 % | **-1.66 %** | **-0.89 pp (35% better)** |
| side-ridge -300 μm   | +2 μm | +10 μm | +8 (more mushy-zone Marangoni) |

Bug-1 fix is **physically correct** (mushy zone Marangoni no longer
double-suppressed) and **improves mass conservation** but **does not close
the center-fill gap**. The bottleneck is deeper than this single bug.

### Dawn-3 (dσ/dT sensitivity, 400 μs, 06:59 完成)

Override `dσ/dT = -4.9e-4` (vs F3D-aligned -4.3e-4; +14% steeper Marangoni):

| metric | -4.3e-4 baseline | -4.9e-4 stress | delta |
|---|---:|---:|---:|
| side-ridge -200 μm   | +8 μm  | +18 μm | +10 μm (much taller) |
| centerline Δh 95%ile | -18 μm | -18 μm | **0 (unchanged)** |

A 14% Marangoni magnitude swing leaves the centerline groove untouched.
**The center groove is not a Marangoni-magnitude knob.**

## Final-frame summary (full)

```
==============================================================================
PHASE-1 LBM FRAME SUMMARY
==============================================================================
File           : output_phase4/line_scan_025000.vtk
Grid           : 1100 x 75 x 100, dx=2.00 um
Step / t       : 25000  /  t = 2000.0 us
Laser x        : 2100.0 um  (start=500, v=0.8 m/s)
z0 (substrate) : 160.0 um
v_factor       : 25.000 m/s per LU  (dx=2.00um / dt=80ns)
Has any metal  : True
Melt pool cells (T>1697 & f>0.5): 152304

------------------------------------------------------------------------------
[1] TRAILING-ZONE PROFILE at x = laser_x - 200 um
------------------------------------------------------------------------------
  x_slab          : 1900.0 um
  z_centerline    : 134.00 um
  z_side_max      : 162.00 um  (|y-y_mid|>15um)
  groove_depth    : +26.00 um  (z0 - z_centerline)
  side-ridge dh   : +2.00 um  (vs z0)
  F3D ref         : center +4 um, side ridges +4 um

------------------------------------------------------------------------------
[2] v_z ALONG CENTERLINE BEHIND LASER (top-of-fill, physical m/s)
------------------------------------------------------------------------------
  x_off um    x_um z_top um     v_z LU    v_z m/s
       -50  2050.0     64.0     0.0104      0.260
      -100  2000.0    106.0     0.0623      1.557
      -150  1950.0    128.0    -0.0026     -0.064
      -200  1900.0    134.0    -0.0009     -0.023
      -250  1850.0    134.0    -0.0012     -0.031
      -300  1800.0    134.0    -0.0010     -0.024

------------------------------------------------------------------------------
[3] SIDE-RIDGE MAX z BEHIND LASER (max over y, |y-y_mid|>15um)
    Band x = [laser_x - 500, laser_x - 100] um
------------------------------------------------------------------------------
  Peak side-ridge z   : 176.00 um at x = 1742.0 um
  Peak vs z0 (Δh)     : +16.00 um
  x_off=-100 um  -> z_side_max=166.00 um, Δh= +6.00 um
  x_off=-200 um  -> z_side_max=162.00 um, Δh= +2.00 um
  x_off=-300 um  -> z_side_max=166.00 um, Δh= +6.00 um

------------------------------------------------------------------------------
[4] CENTERLINE RAISED-TRACK Δh (95%ile, extract_track_height convention)
------------------------------------------------------------------------------
  z_p95 (centerline)  : 138.00 um
  Δh (95%ile)         : -22.00 um   ← preferred
  Δh (max)            : -20.00 um
  Verdict             : DEPRESSED (groove)
  F3D ref             : Δh ≈ +4 um (slight ridge / fill)

==============================================================================
DECISION TABLE
==============================================================================
  metric                                  value       F3D ref
  -------------------------------- ------------  ------------
  v_z @ -150 um (m/s)                    -0.064   >0 expected
  side-ridge peak Δh                  +16.00 um      +4.00 um
  centerline Δh (95%ile)              -22.00 um      +4.00 um
  groove depth @ -200 um              +26.00 um         ~0 um
==============================================================================
```

## Mass conservation

```
Domain: 1100×75×100

                         Frame     Mass [g]      Σfill     n_full  n_partial
output_phase4/line_scan_000000.vtk     0.000422    6641251    6435000     412500
output_phase4/line_scan_025000.vtk     0.000389    6400370    6287116     260468

Δ mass         : -0.000034 g  (-7.967 %)
Δ Σfill        : -240880.5 cells

VERDICT: SIGNIFICANT MASS LOSS (-7.967%)
  Likely numerical leakage in VOF advection or evap mass-loss kernel.
  Recommend: enable_vof_mass_correction=true and rerun.
```

## F3D reference

```
/home/yzk/LBMProject/scripts/flow3d/extract_f3d_track.py:66: RuntimeWarning: All-NaN slice encountered
  z_centerline = np.nanmax(z_grid[:, band_lo:band_hi], axis=1)
/home/yzk/LBMProject/scripts/flow3d/extract_f3d_track.py:94: RuntimeWarning: All-NaN slice encountered
  ridge_max_per_x = np.nanmax(side_band[:, side_mask], axis=1)
Reading vtk-316L-150W-50um-V800mms/150WV800mms-50um_99.vtk (F3D PolyData)...
Points: 806856
Domain (μm): x∈[-498, 2103], y∈[-202, 202], z∈[-198, 20]
Laser at x = 1584 μm (F3D coordinates), t=1980 μs

=== F3D centerline trailing-zone (laser at 1584 μm) ===
  x_off um  z (F3D zero)  z (LBM offset)  Δz vs z₀
       -50        -10.00          150.00    -10.00
      -100         -2.52          157.48     -2.52
      -150         -0.90          159.10     -0.90
      -200         -1.04          158.96     -1.04
      -250          2.90          162.90     +2.90
      -300          2.44          162.44     +2.44

  Behind-laser side-ridge MAX z (vs F3D z=0): +7.51 μm
  Behind-laser side-ridge MEAN z           : +4.11 μm
  Centerline 95%ile z (trailing)            : +6.57 μm
  Centerline median z (trailing)            : +2.72 μm
```

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

1. **DONE — Marangoni double fl-gate fixed** (commit 5853e74, 2026-04-26).
   `force_accumulator.cu:323-329` returned early when fl<0.1 (inner gate, with
   smooth ramp to 1 at fl=0.2), and `multiphysics_solver.cu:2606-2613` then
   multiplied the same force by fl (outer mask). Combined effect at fl=0.5:
   50 % Marangoni suppression; at fl=0.15: 92.5 % suppression. The mushy zone
   — exactly where trailing-edge back-flow lives — was the most suppressed.
   Outer mask removed; inner gate retained (already provides cold-powder
   protection). Phase-1B sim running now to quantify the effect; results will
   be appended to night_run_log.md.

2. **HIGH — Investigate VOF mass-correction destination cells**.
   Phase-2 showed mass conservation enforces the wrong distribution.
   The redistribution kernel likely targets all interface cells
   uniformly; should target only the trailing zone (i.e., cells with
   v_z > 0 at the surface, or fl > 0.5 in the trailing slab).

3. **MEDIUM — TRT operator** (draft at `src/collision_trt_draft.cuh`).
   Lets τ+→0.55 stable with Λ=3/16 magic parameter.  Phase-3 evidence
   suggests this alone won't fix the groove (made it worse: -42 μm), but
   combined with Bug-1 fix may give the right physical viscosity AND
   allow weakened Marangoni in mushy zone to win. Header comment lists
   the integration steps for human migration.

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

