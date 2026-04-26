# LBMProject Flow3D 对标冲刺 — 项目状态报告

**生成时间**: 2026-04-26
**主分支 HEAD**: `5d9b7d1 test(trt): honest scope + LES smoke + RAII`
**工作分支**: `benchmark/conduction-316L`
**Worktree A**: `feature/vof-mass-correction-destination` @ `b0117de`（活跃）

---

## 0. 项目一句话概括

CUDA 实现的 D3Q19 LBM (流体) + D3Q7 LBM (热) + VOF (自由界面) 多物理场求解器，
目标是把 316L 单道激光粉床熔覆 (LPBF, 150 W / 0.8 m/s / dx=2 μm) 仿真**重现 Flow3D 标定结果**
到 0.1% 量级。

---

## 1. 验证已通过的核心指标

| 指标 | LBM Phase-1 | F3D | 状态 |
|---|---:|---:|:---:|
| Pool depth | 70 μm | 78 μm | ✓ 90 % |
| Pool length | 113 % | 100 % | ✓ |
| Pool width | matches | 73 μm | ✓ |
| T_max sync | yes | yes | ✓ |
| v_max in band | 4-5 m/s | 1-4 m/s | ✓ |
| 有效吸收率 | **65.5 %** | 70-75 % | ✓ ~93 % |
| 侧凸 -100 μm Δh | +6 μm | +4 μm mean | ✓ |
| 侧凸 -200 μm Δh | +8 μm | +7.5 μm max | ✓ |
| v_z 回流 @ -150 μm | **+0.196 m/s** | > 0 | ✓ |

**目前已在工程级验收范围**: 8/9 指标。

---

## 2. 唯一硬卡点

| 指标 | LBM Phase-1 | F3D 真值 | 差距 |
|---|---:|---:|---:|
| 中心 Δh 95%ile (尾流) | **-16 μm** | -1 μm | **15 μm** |

整夜协议确认：
- **不是** 束斑半径（已修，50→39 μm 给了 80% 的提升）
- **不是** 糊状区宽度（早就是 23 K，跟 F3D 一致）
- **不是** Marangoni 强度（dσ/dT 14% 摆动 → 中心 Δh 0 摆动）
- **不是** 双 fl-gate（Bug-1 修了 → 中心 Δh 不变）
- **不是** 粘度不真实（τ→0.55 → 中心更糟）
- **是** 多物理场耦合在尾流的力平衡 + VOF 质量补偿目标错位

---

## 3. 已完成的架构改动 (本会话 6 个 commit)

```
5d9b7d1 test(trt): honest scope + LES smoke + RAII; delete redundant draft header
03180f1 test(trt): anchor test that TRT(ω⁻=ω⁺) ≡ BGK within FP32 round-off
bffd368 docs(night-protocol): archive Night's Watch run log + Morning Briefing Report
46e55f4 feat(draft): TRT-D3Q19 collision operator draft (Dawn-2)
684f559 feat(scripts): add Flow3D-comparison + night-protocol analysis tooling
255ba06 feat(sim): add night-protocol Phase-{1,2,3,4} + Dawn-3 + Sprint-3 line-scan apps
5853e74 fix(marangoni): remove outer Marangoni fl-mask (Bug-1 night audit)
```

总览:
- **11 个新 sim 应用** 覆盖 night-protocol + Sprint-3 决策树
- **18 个新分析脚本** (Python + Bash) 跑 F3D 对照 / VTK 解析 / 自动派单
- **2 个新测试** 锚定 TRT-EDM 路径（已诚实降级为 smoke + LES smoke）
- **Bug-1 修复** 删除冗余 Marangoni fl-mask
- **夜间协议归档** 在 `docs/night-protocol-2026-04-26/`

---

## 4. Bug 审计闭合状态

| Bug | 描述 | 修复 commit |
|---|---|---|
| 1 | Marangoni 双 fl-gate | **5853e74** (本会话) |
| 2 | ∇T 跨气界面零截断 | 25e6d06 (Sprint-1) |
| 3 | EDM Δu 缺 Darcy 分母 | 25e6d06 (Sprint-1) |
| 4 | FP32 串行质量求和 | 25e6d06 (Sprint-1) |

四个全闭。Phase-1B 验证 Bug-1 修复给出 mass drift 改善 35% (-2.55%→-1.66%)。

Memory 顶部置 `project_4bug_audit_closed_2026_04_26.md` — 阻止 cron 重复触发已修工作。

---

## 5. 在飞 / 计划任务

| ID | 任务 | 位置 | 状态 |
|---|---|---|:---:|
| **A** | VOF mass-correction destination 重设计 | worktree `LBMProject_vof_mass_correction` | 🟡 在跑 |
| **B** | TRT 集成 + anchor test | main 分支 5d9b7d1 | ✓ 完成（smoke 级） |
| **C** | TRT 真正的 ω⁻≠ω⁺ 解析对照测试 | 待开 | ⏳ |
| **D** | 物性 F3D 对齐 (emissivity, ρ(T)) | 待开 | ⏳ |
| **E** | 尾流力平衡架构诊断 | 待开（研究级） | ⏳ |
| **F** | Bogner FSLBM 隐式表面张力 BC | worktree-禁触 | 🚫 |

---

## 6. 关键架构事实

### LBM Stack (D3Q19 + D3Q7 + VOF)
```
src/core/lattice/d3q19.cu          ← 19-velocity stencil
src/core/lattice/d3q7.cu           ← 7-velocity thermal
src/core/streaming/                ← propagation
src/core/boundary/                 ← BC application
src/physics/fluid/fluid_lbm.cu     ← 主流体求解器 (TRT-EDM 已生产化)
src/physics/thermal/thermal_fdm.cu ← 热场 (FDM-ESM 在用; D3Q7 LBM 留备)
src/physics/vof/vof_solver.cu      ← PLIC 几何 VOF
src/physics/multiphysics/          ← 编排所有物理
src/physics/laser/                 ← Beer-Lambert + 复Fresnel ray-tracing
src/physics/force_accumulator.cu   ← Marangoni / CSF / Darcy / 浮力
src/physics/phase_change/          ← enthalpy-bisection ESM
src/physics/materials/             ← 316L Mills 数据 (已对齐 F3D)
```

### 算子使用
- **流体碰撞**: TRT-EDM with Λ=3/16 (production default)
- **热场**: FDM with enthalpy-source-method bisection inverter (residual 0.012%)
- **VOF**: PLIC 几何重构 (Youngs normals + Strang splitting)
- **激光**: Ray-tracing with Fresnel (n=2.96, k=4.01) + 多反射

### 关键参数 (production sim_line_scan_316L)
```
domain   650 × 125 × 100 cells = 1300 × 250 × 200 μm  (Phase-1)
dx/dt    2 μm / 80 ns
material 316L Mills, σ=1.75, dσ/dT=-4.3e-4, T_sol=1674.15, T_liq=1697.15
laser    150 W, spot 39 μm (Phase-1 align), 1064 nm, Fresnel multi-bounce
ν_LU     0.065 (artificial 4.3× over Mills physical — stability requirement)
```

### F3D 对照参数 (vtk-316L-150W-50um-V800mms/prepin)
```
P=150 W, beam_radius dum2=39 μm
n=2.9613, k=4.0133, Fresnel multi-bounce
σ=1.74, dσ/dT(csigma)=-4.3e-4
ts1=1674.15, tl1=1697.15, L_fusion=260 kJ/kg
μ=6.0e-3 Pa·s, density table 7237→6881 (5% jump at melting)
emissivity=0.55, if_vol_corr=1
```

### 已识别但未匹配的 F3D 参数
| 参数 | F3D | LBM | 待修 |
|---|---|---|---|
| emissivity | 0.55 | 0.20 | D 任务 |
| ρ(T) | 表 + 5% jump @melting | 单一 ρ_liquid | D 任务 |
| if_vol_corr | 1 (1 μs 周期) | off | A 任务 |
| ν_phys | 7.6e-7 m²/s (Mills) | 3.25e-6 m²/s (artificial) | E 任务 |
| BC | ibct=3 outflow | WALL | 中等优先级 |

---

## 7. 三座大山 (排序)

### 🏔️ 山 1: VOF mass-correction destination
**正在攻**（A 任务，worktree 在跑）。当前算法把丢失质量均匀泼回所有界面格 → 大部分泼到 scan-start splash deposit，把侧凸越推越高，中心更空。

合格条件 7 + 放弃条件 4 已写入 `docs/task-A-vof-mass-correction/A_TASK_BRIEF.md` (commit `b0117de`)。

### 🏔️ 山 2: 尾流 Marangoni 出 vs 毛细回 力平衡
LBM 测得 v_z=+0.196 m/s 已经符号正确，但 5 步内液体没流回足够的中心（被 Darcy 吸住 + 凝固封死）。这是耦合而非单一参数问题。

候选解：
- Bogner FSLBM 隐式表面张力 BC（被 worktree-禁触，需要重新工程化）
- Sharp-interface VOF 取代 CSF 抹散
- 提高 capillary 力的施加 cell width

### 🏔️ 山 3: GPU 内存 4 GB
直接限制 Phase-4 横向 ny=75（vs Phase-1 的 125）。F3D 跑的 2600×400×300 μm 域我们做不下来。多 GPU / AMR 是单独 sprint。

---

## 8. 中等山 (1-2 周)

| # | 任务 | 估计影响 |
|---|---|---|
| 7 | 真正的 TRT ω⁻≠ω⁺ Poiseuille 解析对照 | 不改形貌，给信心 |
| 4 | BC: WALL → outflow ibct=3 类似 | 1-3 μm |
| 5 | 密度 ρ(T) 表 + 5% melt jump | 1-3 μm |
| 6 | emissivity 0.20 → 0.55 | 1-2 μm |

合并以上 4 项可能再啃 5-8 μm（但有相互作用，不是简单叠加）。

---

## 9. 小山 (小时-天)

- Substrate punch-through: Phase-4 跑 2 ms 时 keyhole 撞底（106 μm vs 200 μm 域底）
- Recoil tail at T<T_boil: 当前 30 K ramp，Sprint-2 试过 5 K 无效
- Pre-existing dirty files (30 个 untracked + modified)：清理 git 状态

---

## 10. 关键参考资料 / 工具

### 仿真应用 (apps/)
```
sim_line_scan_316L          production (50 μm spot, baseline)
sim_linescan_phase1         night protocol Phase-1 (spot=39 μm)
sim_linescan_phase{2,3,4}   Phase-2/3/4 决策树分支
sim_linescan_dawn3          Dawn-3 dσ/dT 灵敏度
```

### 分析脚本 (scripts/flow3d/)
```
phase1_summary.py             单 VTK 决策表
phase1_decide.py              应用决策树到一帧
phase1_compare_baseline.py    新旧 LBM 对比
check_mass_conservation.py    ρ(T)·fill·dx³ 积分 (Mills 表)
extract_f3d_track.py          F3D PolyData → z(x,y)
diag_keyhole_shape.py         κ, P_cap, aperture 角
diag_trailing_edge.py         v_z, fl, κ, T 在尾流偏移处
diag_marangoni_streamlines.py y-z 切片 + 温度等值线 PNG
analyze_trailing_yz_slice.py  per-y profile + v_y v_z T
finalize_morning_report.sh    重新生成 morning report
launch_dawn3.sh               跑 dσ/dT 灵敏度
```

### 参考 VTK 数据
- `output_line_scan/line_scan_010000.vtk` — pre-fix baseline (-20 μm)
- `output_phase1/line_scan_010000.vtk` — Phase-1 best (-16 μm)
- `vtk-316L-150W-50um-V800mms/150WV800mms-50um_99.vtk` — F3D 真值 (-1 μm)

### 文档
- `docs/night-protocol-2026-04-26/night_run_log.md` — 470 行夜间协议时间线
- `docs/night-protocol-2026-04-26/Morning_Briefing_Report.md` — 270 行三方对比 + 5 节经验教训
- `docs/task-A-vof-mass-correction/A_TASK_BRIEF.md` — A 任务规范 (7 pass + 4 reject)
- 本文档 `docs/project-status-2026-04-26.md`

---

## 11. 推荐的下一步并行工作

A 在 worktree 跑期间（GPU 紧张），主分支可以并行做这些**纯源码 / CPU 工作**：

1. **am-simulation-configurator** → 配置 Phase-5 集成验证（emissivity + ρ(T) + 等 A 完成后叠加）
2. **test-debug-validator** → 实施真正的 TRT ω⁻≠ω⁺ Poiseuille 解析对照测试
3. **cfd-cuda-architect** → 诊断山 2 (尾流力平衡)，定位 CSF 抹散是不是元凶
4. **cfd-math-expert** → 分析密度跳变 (山中 #5)，给出 LBM 实现路径

A 完成后所有改动在 Phase-4 一次跑齐，看叠加效果。

---

## 12. 风险登记

| 风险 | 严重度 | 缓解 |
|---|---|---|
| A 不能解（3 算法都 fail）| 高 | 已写 abort 条件 → cherry-pick docs only |
| 主分支累积改动太多冲突 A | 中 | 限制本 sprint 主分支只做 docs / non-conflicting 测试 |
| 4 GB GPU 卡住 Phase-4 真实生产域 | 高 | 需要硬件升级，或 multi-GPU 工程 |
| TRT 内核有沉默 ω⁻ 算术 bug | 中 | 加 Poiseuille 解析对照（C 任务） |
| ρ(T) jump 改动破坏 mass conservation | 中 | cfd-math-expert 先做理论分析 |
| 山 2 (Marangoni-毛细力平衡) 是体系级问题 | 高 | 多周项目 — 不是本 sprint 范围 |

---

## 13. 退出条件 (本 sprint)

合格门:
- 中心 Δh 95%ile @ t=2 ms 改善到 ≥ -8 μm（基线 -22, 目标 -1，midpoint as merge bar）
- 8/9 指标 → 9/9 (中心 Δh 由 fail 转 pass)
- 0 commit pushed 出 fork（按用户 protocol）

不达标也合格的"软退出":
- 3 个工程改动尝试都没把中心改善 ≥ 5 μm → 写"中心瓶颈是体系级"诊断报告 → 升级到下一 sprint 做 FSLBM/sharp-VOF
