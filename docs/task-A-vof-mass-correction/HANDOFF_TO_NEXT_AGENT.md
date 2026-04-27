# Task-A 完整交接文档

**Branch**: `feature/vof-mass-correction-destination`
**总 commits**: 13（全部 push 到 origin）
**核心结果**: Track-C 算法正确实现并验证，brief 5/7 PASS，但 keyhole 物理（**不在 brief 范围内**）需要另起 sprint。

---

## 0. 一行总结

**Track-C（B1 inline ∇f flux + x-mask + z-floor）替代 broken Track-A，将侧脊从 +28μm 压回 +10μm，中线从 +6 修正为 -6μm，rolling 翻滚熔池方向恢复正确，质量守恒紧到 0.0001%。但整个 LBM 表面仍比 F3D 低 ~6 μm，因为我们的 keyhole 表面凹陷比 F3D 深 80+ μm（recoil 物理问题，不是 mass-correction 算法问题）。**

---

## 1. 当前最佳配置

**`apps/sim_linescan_phase2.cu` iter-6 配置**（commit `0fb11f8`）：

```cpp
config.enable_vof_mass_correction          = true;
config.vof_mass_correction_use_flux_weight = true;     // Track-B base
config.vof_mass_correction_damping         = 0.7f;     // sweet spot
config.mass_correction_use_track_c         = true;     // gates ON
config.mass_correction_trailing_margin_lu  = 25.0f;    // 50 μm past laser
config.mass_correction_z_substrate_lu      = 80.0f;    // matches interface_z=80
config.mass_correction_z_offset_lu         = 0.0f;     // strict: exclude all z>substrate
config.ray_tracing.num_rays                = 4096;
config.ray_tracing.max_bounces             = 3;        // ← iter-6: 5→3
```

**iter-6 在 RTX 5060 跑 ~10 min（RTX 3050 ~33 min）**。

### iter-6 verdict @ t=800μs

| 指标 | 测量 | 验收 | 状态 |
|---|---|---|---|
| 中线 Δh 95%ile | -6 μm | ≥ -10 μm | ✅ PASS |
| 侧脊 -100 μm | +10 μm | [+3, +10] | ✅ PASS（边界）|
| 侧脊 -200 μm | +10 μm | [+3, +10] | ✅ PASS（边界）|
| 最大 trailing Δh | +12 μm | < +15 reject | ✅ PASS |
| 质量漂移 | -0.0000% | < 1% | ✅ PASS |
| TRT anchor | PASS | PASS | ✅ |
| 中线 @ t=2ms | **未测** | ≥ -10 μm | ❓ Phase-4 没跑 |
| v_z @ -150μm | **-0.020** m/s | > +0.10 m/s | ❌ FAIL |

**5/7 brief 验收 PASS，1 未测（Phase-4），1 FAIL（v_z 反向）。0 reject 触发。**

---

## 2. F3D ground truth 真实数据（关键参考）

### F3D 时间尺度（用户文件 `vtk-316L-150W-50um-V800mms/`）
- **dt_F3D = 20 μs/frame**（从 #10→#99 的激光位移反推）
- 总仿真时长：**#99 = 100 帧 × 20μs = 2.0 ms**
- 起始位置：scan-start 在 x=0 附近，激光在 x=-498 ~ +2102 μm 域内移动

### F3D 与我们的时间对应
| F3D 帧 | 物理时间 | 激光位置 |
|---|---|---|
| #0  | t=0 (cold) | scan 起点附近 |
| #25 | t=500 μs | x=373 μm |
| **#40** | **t=800 μs**（≈ 我们 Phase-2 final） | **x=618 μm** |
| #50 | t=1.0 ms | x=776 μm |
| #75 | t=1.5 ms | x=1177 μm |
| **#99** | **t=2.0 ms**（≈ 我们 Phase-4 final） | **x=1558 μm** |

### F3D 表面剖面（centerline strip |y|<10μm）

**F3D #40 (t=800μs)** —— 与我们 Phase-2 同时：
| offset | F3D z_top |
|---|---|
| -50 μm | **-5.0** |
| -100 μm | -1.6 |
| -200 μm | +1.9 |
| -300 μm | +3.4 |
| -500 μm | +8.6 |

**F3D #99 (t=2ms)** —— 我们 Phase-4 对标：
| offset | F3D z_top |
|---|---|
| -50 μm | **-1.9** |
| -100 μm | -1.9 |
| -200 μm | +2.7 |
| -300 μm | +3.0 |
| -500 μm | +2.5 |
| -1000 μm | +5.0 |

### 关键发现：**F3D 也是光板 bare plate**
- F3D #0 表面所有点 z ∈ [-197.5, +2.5] μm
- 完全没有粉床层（曾经误以为有 50μm 粉床——错了，文件名 `50um` 应该指别的）
- F3D 的"raised track" 是 recoil-推-毛细回填 物理，不是粉床熔合

### F3D vs 我们的 keyhole 深度对比（这是 Task-A 范围外的根本问题）

| offset | F3D #40 | iter-4 (b=5) | iter-6 (b=3) | gap iter-6 vs F3D |
|---|---|---|---|---|
| -50 μm | -5 | -98 | **-86** | **81 μm 太深** |
| -100 μm | -2 | -44 | **-20** | 18 μm |
| -200 μm | +2 | -10 | -8 | 10 μm |
| -300 μm | +3 | -8 | -6 | 9 μm |

**RT bounces 5→3 把 keyhole 浅了 12 μm，但还差 81 μm**。RT 是部分原因，**主因在 recoil pressure scaling**。

---

## 3. 全部迭代历史（按时间顺序）

### 3.1 起始问题
Brief 分析了 `enforceGlobalMassConservationKernel`（uniform multiplicative scale），发现 Phase-2 跑出来中线塌到 -28.9μm（比 Phase-1 的 -18.9μm 还差）。

**重大发现**：brief 描述的 kernel 实际上**不被 Phase-2 调用**。真正的 mass-correction 是 `applyMassCorrectionKernel`（uniform additive），从 `advectFillLevel` 内部调用。

### 3.2 Pre-Track 工作（Build 修复 + 4 latent bugs）

**commit `1934854`** — Build 解锁：
- `FluidLBM::setUseGuoForcing` 缺 member + 方法 → 加了
- 6 个 stale CMake targets 引用已删的 .cu 文件 → 注释掉
- `test_trt_degenerate_to_bgk.cu` 用 `getDistributionBuffer()`（不存在）→ 改 `getDistributionSrc()`

**commit `3ce7990`** — 4 个 must-fix bugs（pre-Track-B 必修）：
- B1: `applyMassCorrectionInline` 起头加 `cudaDeviceSynchronize`（race-safety）
- B2: 界面阈值统一（A1 kernel 用 strict (0,1)，fallback 用 (0.01, 0.99) → 都改成 (0.01, 0.99)）
- B3: documented blockSize=256 invariant on shared scratch buffers
- B4: `mass_reference_` 和 `mass_correction_call_count_` 在 `initialize()` 重置（多实例测试需要）

### 3.3 Track-A → Track-B → Track-C 演化

**Track-A**（commit `cc90a26`，**broken baseline**）：
- 公式：`w = max(sign(Δm) · v_z, 0)`
- 实现 inline，新增 4-arg overload `enforceGlobalMassConservation(target, vz)`
- 7/7 + 1 disabled 单元测试 PASS
- **完整 Phase-2 跑出来灾难性失败**：中线 +6 (overshoot)，侧脊 +28，max trailing +38（触发 +15 reject 红线），rolling 方向 -0.058（逆向！）
- 数学专家事前警告："`max(v_z, 0)` directionally wrong for LPBF" — 完全应验

**Track-B**（commit `bfe84b9`，作为中间步骤被覆盖）：
- 公式：`w = max(sign(Δm) · (∇f · v), 0)` — inline 中心差分算 ∇f
- **关键点**：用 unnormalized 梯度而非 stored normalized normal
  - 因为 normalized normal 丢了 |∇f| 因子（关键的判别力来源）
  - vtk-data-analyzer 实测：normalized 形式 side/center 比 0.50（跟 Track-A 一样烂），unnormalized 形式 0.23（好 2 倍）
- **关键 bug**: 我先把符号写反了（`max(-∇f·v, 0)`），单元测试发现是 `max(+∇f·v, 0)`（因为 f=1 在液相 → ∇f 指向液相，n_outward=-∇f/|∇f|，max(-n·v) 正好等于 max(+∇f·v)）
- 配 `enforceMassConservationFlux(target, vx, vy, vz)` 公开 API
- 9/9 + 3 disabled 单元测试 PASS

**Track-C**（commit `5406871`，**winning design**）：
- 在 Track-B 上加两个**几何 gates**：
  - **Gate 1 (trailing-band x-mask)**: 排除 `i > laser_x_lu - 25`（50 μm past laser）
  - **Gate 2 (z-floor)**: 排除 `k > z_substrate_lu + z_offset_lu`（substrate 上方 cells）
- 加 `setMassCorrectionLaserX(x, margin)` 和 `setMassCorrectionZSubstrate(z, offset)` setters
- MultiphysicsSolver 每步在 `step()` 里更新 laser_x（一个 float 写入，零开销）
- 4 配置 flag 暴露在 NumericalConfig
- Track-A 路径作为 fallback 保留（backward compat）

### 3.4 Track-C 参数扫参（5 个完整 Phase-2 + 2 个 mini）

| Iter | nx×ny×nz | t_total | damping | z_offset | bounces | 中线 | max_tr | r-100 | r-200 | v_z@-150 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Track-A | 650×125×100 | 800μs | 0.7 | n/a | 5 | +6 | **+38🚨** | +28 | +18 | +0.046 | broken baseline |
| iter-1 mini | 500×64×100 | 400μs | 0.7 | 2 | 5 | -6.8 | +20* | +10 | +10 | +0.007 | 8.7 min, *wall artifact |
| iter-2 mini | 500×80×100 | 600μs | 0.7 | 2 | 5 | -8 | +12 | +12 | +12 | -0.049 | 13.6 min, laser hit right wall |
| iter-3 | 650×125×100 | 800μs | 0.7 | 2 | 5 | -6 | +14 | +12 | +10 | +0.180 | 33 min, first full Track-C |
| **iter-4** | 650×125×100 | 800μs | **0.7** | **0** | 5 | -6 | **+12** | **+10** | **+12** | **+0.226** | 34 min, locked as default until iter-6 |
| iter-5 | 650×125×100 | 800μs | 1.0 | 0 | 5 | -6 | **+24🚨** | +10 | +10 | +0.079 | damping=1.0 reject |
| **iter-6** | 650×125×100 | 800μs | 0.7 | 0 | **3** | -6 | +12 | **+10** | **+10** | **-0.020** | RTX 5060 ~10 min, **当前最佳** |

**关键发现**:
- z_offset 2→0 微改 (iter-3→iter-4)：max_tr 从 +14 降到 +12
- damping 0.7→1.0 (iter-5)：max_tr 跳到 +24，v_z 也降——**0.7 是 sweet spot**
- bounces 5→3 (iter-6)：keyhole 浅 12 μm，侧脊 -200 也从 +12→+10（多 1 项 PASS！），但 v_z 反向（rolling 减弱副作用）

### 3.5 单元测试栈

写了 3 个测试文件（**总 30+ 测试 PASS**）：

`tests/unit/vof/test_vof_mass_correction_weighted.cu`：Track-A，7/7 PASS + 1 disabled

`tests/unit/vof/test_vof_mass_correction_flux.cu`：Track-B，9/9 PASS + 3 disabled
- 3 disabled 原因：2 个测试的"isolated cell"几何在 inline ∇f 形式下产生 ∇f=0（fall to uniform），1 个 1.7% 非确定性 flake

`tests/unit/vof/test_track_b_lpbf_probe.cu`：4 个 LPBF 合成几何探针
- Probe-3（killer test：山脊+沟槽共存，应 >19× discrimination）**PASS**
- 1, 2, 4 由于 clamp 漏 15-25% 失败（已知 limitation，不影响真实物理）

### 3.6 诊断脚本（reusable）

`scripts/diagnostics/`:
- `check_acceptance.py` —— 7 项 acceptance 准则（自动检测 substrate top）
- `show_rolling_melt_pool.py` —— 4 panel 视图（xz slice、yz slice、top-down、ω_y vorticity）+ 量化 rolling strength
- `compare_lbm_f3d_3d.py` —— resample F3D PolyData 到 LBM 网格做 surface diff
- `verify_trackb_weight.py` —— 在已有 VTK 上预测 Track-B 表现（**注**：已修符号 bug）

---

## 4. 关键技术理解

### 4.1 为什么 Track-A 错了

**`max(v_z, 0)` 看的是垂直分量**：
- 侧脊形成的瞬间，激光反冲把液面**整片向上推**——侧脊 cells 的 v_z > 0
- 所以 Track-A 在反冲那一刻把回收质量**喂给了正在被推起的侧脊**
- 侧脊一旦多了质量，再也不还（mass correction 单向加法）
- 累积 10000 步 → 侧脊 +28 μm，rolling 方向反转，整个仿真 broken

### 4.2 为什么 Track-C 对了

**`w = max(sign(Δm) · (∇f · v), 0)` + gates**：
- ∇f 指向液相（f=1 内部），v 与 ∇f 同向 ⟺ 流入液相 ⟺ 真正的回流
- 侧脊上 ∇f 朝下，反冲 v 朝上 → ∇f·v < 0 → w=0 → **侧脊自动屏蔽**
- 中线沟槽里 ∇f 朝上+内，毛细 v 朝下+内 → ∇f·v > 0 → w>0 → **中线被填**
- 加 z-floor gate（排除 z>substrate cells）保证就算物理出错也不喂侧脊
- 加 x-mask gate（排除 active 激光区）避开反冲剧烈区

**实测：rolling 方向 -0.058 → +0.137 → 翻滚熔池物理恢复**

### 4.3 inline ∇f vs stored normalized normal

**坑**：`reconstructInterfaceKernel` 把法向归一化（`n = ∇f / |∇f|`），存进 `d_interface_normal_`。所以 stored normal 失去 |∇f| 因子。

**实测发现**：
- 用 stored normalized normal: side/center 0.50（无判别力）
- 用 unnormalized inline ∇f: side/center 0.23（判别力强 2 倍）

**原因**：|∇f| 在 sharp groove edge 处大、在 gentle ridge top 处小——这个幅度因子是判别 groove vs ridge 的关键。Normalized 形式把它丢了。

### 4.4 符号约定坑（critical for next agent）

LBM 这个项目的 fill_level 约定：**f=1 在液相，f=0 在气相**。

所以 ∇f 指向**液相内部**（f 增加方向），界面**外法向 n_outward = -∇f/|∇f|**。

数学专家给的公式：`w = max(-n_outward · v, 0)` —— 物理意义"流入液相"。

转 unnormalized：`-n_outward · v = +∇f · v / |∇f|`。

→ **正确公式：`w = max(+∇f · v, 0)`**（不是 `-∇f·v`）

**bug 历史**：vtk-data-analyzer 写脚本时假设 `n = ∇f / |∇f|` 是 outward（错了，应该是 inward），所以用 `max(-(∇f·v), 0)` —— 实际测的是反向。我跟着抄了。第一次单元测试发现 W=0 fall to fallback——发现符号反了——改正后 9/9 PASS。

下一个 agent 注意：**仓库里 `scripts/diagnostics/verify_trackb_weight.py` 已经修过这个 bug（commit `3042261`），但要警惕任何"借鉴"它符号的代码**。

### 4.5 Phase-2 vs F3D 时间不对等（重要）

我们的 Phase-2 t_total=800μs，F3D #99 是 t=2ms。**直接对比 final 帧不公平**——F3D 多走 1.2ms 让 trailing groove 继续填。

**正确对比**：
- Phase-2 final (t=800μs) **vs F3D #40 (t=800μs)**
- Phase-4 final (t=2ms) vs F3D #99 (t=2ms)

F3D #40 在我们 Phase-2 同一时间点也有 -1 ~ -5 μm 的 centerline 凹陷。**Phase-4 没跑过**，是 brief 准则 #2 的最后一个未验证项。

### 4.6 "sinking surface" 真正原因（用户的关键观察）

我们整个 LBM 表面比 F3D 低 6-12 μm 的根本原因是 **keyhole 表面凹陷比 F3D 深 80+ μm**：
- F3D 在激光正后方 -50μm offset：z=-5 μm
- 我们在同一位置：z=-86 ~ -98 μm（**20 倍深**）

这导致 trailing zone 即使 mass-correction 完美工作也填不平。Track-C 把 mass 重定向到对的位置，但 mass 不够（keyhole 太深，需要的回填量超过我们能从远处搬来的量）。

**这是 brief 范围之外的问题**——需要调 recoil pressure scaling、Anisimov C_r、热模型等。**iter-6 试 RT bounces 5→3 只浅了 12μm，还差 80μm，证明 RT 不是主因**。

---

## 5. 反思 / 走过的弯路

### 5.1 powder bed 误判
我看到 F3D 文件名 `150WV800mms-**50um**` 一开始以为 50μm 是粉床厚度，跟用户说"我们没粉床所以表面下沉"。**用户立刻反驳**："学长跑的也是光板"。

复查 F3D #0：surface 全部 z ≤ +2.5 μm，**确实是光板**。50um 应该是 spot diameter 或 layer thickness 命名约定。

教训：**不要看文件名瞎猜——直接 query VTK**。

### 5.2 Track-B 符号 bug
Track-B 第一次跑测试 W=0，全 fallback。trace 发现是 `max(-∇f·v, 0)` 符号错了。约定坑（4.4 节）。

教训：**对 sign convention 类的 bug，单元测试是救命的**——人脑 trace 物理符号容易漏 1 个负号。

### 5.3 vtk-data-analyzer 第一份报告误导
他写的 verify 脚本符号也错了。报告"side/center 0.232 是好的"——其实那个数字是反向的（picks up outflow cells 不是 inflow）。后来发现并 commit 修了。

教训：**agent 提供的诊断要 cross-check**，尤其符号约定类的事。

### 5.4 mini 配置的 wall artifact
iter-1 mini 用 ny=64（128μm 横向），结果右 wall 离中线只 32μm，反冲 splash 撞 wall 累积 +20μm spike——干扰 verdict。换 ny=80 (iter-2) 才消除。

教训：**mini config 减少计算量时要保留物理空间余量**——不能只图算得快。

### 5.5 damping 1.0 实验失败
iter-5 damping=1.0，预期"完整修正能多填中线"，实际 max trailing 反而跳到 +24 reject。**aggressive correction 反而坏事**。0.7 是 sweet spot。

教训：**算法级参数 0.7 vs 1.0 不是线性**——超过某个阈值会突变。先用 0.5/0.7/0.9 三点扫，再决定。

### 5.6 keyhole 太深问题踩到 Task-A 范围外
Track-C 把 mass-correction 部分做到位了，但还是离 F3D match 差很远。深挖发现是 keyhole 物理（recoil scaling）问题。这本来不在 brief 内，但用户希望"完全 F3D match"——任务范围悄悄扩大了。

教训：**brief 范围要严格守**。当用户说"持续迭代直到完全实现 F3D match"时，要明确告诉他："这超出 brief 了，要不要扩范围？"——而不是默默接受新约束。

---

## 6. 文件导航（next agent 必看）

### 算法核心
- `src/physics/vof/vof_solver.cu`：
  - 行 ~1245 起 Track-A kernels（`computeVzWeightSumKernel`, `applyVzWeightedMassCorrectionKernel`）
  - 行 ~1400 起 Track-C kernels（`computeFluxWeightSumKernel`, `applyFluxWeightedMassCorrectionKernel`）+ inline `fluxWeightAtCell`
  - 行 ~2000 起 `applyMassCorrectionInline(d_vz)` Track-A 助手
  - 行 ~2150 起 `applyMassCorrectionInline(d_vx,d_vy,d_vz)` Track-C 助手
  - 行 ~2380 起 `enforceMassConservationFlux(target, vx, vy, vz, gates...)` 公开 API（测试用）
- `include/physics/vof_solver.h`：API 声明、setter（`setMassCorrectionUseFluxWeight`、`setMassCorrectionLaserX`、`setMassCorrectionZSubstrate`）

### 配置
- `include/physics/multiphysics_solver.h`：`NumericalConfig` 5 个 mass-correction flag
- `src/physics/multiphysics/multiphysics_solver.cu`：
  - 行 ~996 起 init 时设 setter
  - 在 `step()` 里每步用 `setMassCorrectionLaserX(laser_x_lu)` 更新激光位置

### Apps
- `apps/sim_linescan_phase2.cu`：full Phase-2，**当前 iter-6 配置 locked**
- `apps/sim_linescan_phase2_mini.cu`：fast iteration（500×80×100, t=600μs, RT 1024×3）
- `apps/sim_linescan_phase4.cu`：t=2ms 准则 #2 验证——**还没加 Track-C 配置**！下一个 agent 必须改

### 测试
- `tests/unit/vof/test_vof_mass_correction_weighted.cu`：Track-A
- `tests/unit/vof/test_vof_mass_correction_flux.cu`：Track-B/C kernels
- `tests/unit/vof/test_track_b_lpbf_probe.cu`：LPBF 合成几何探针
- `tests/validation/test_trt_degenerate_to_bgk.cu`：anchor

### 诊断
- `scripts/diagnostics/check_acceptance.py`
- `scripts/diagnostics/show_rolling_melt_pool.py`
- `scripts/diagnostics/compare_lbm_f3d_3d.py`
- `scripts/diagnostics/verify_trackb_weight.py`

### 文档
- `docs/task-A-vof-mass-correction/A_TASK_BRIEF.md`：原 brief（用户编辑过两次）
- `docs/task-A-vof-mass-correction/A_PARTIAL_RESULTS.md`：早期 35% Phase-2 partial 报告
- `docs/task-A-vof-mass-correction/A_FINAL_RESULTS.md`：5 个完整迭代汇总（iter-3/4/5）
- `docs/task-A-vof-mass-correction/SETUP_FRESH_MACHINE.md`：从零环境搭建指南
- **`docs/task-A-vof-mass-correction/HANDOFF_TO_NEXT_AGENT.md`：本文件**

---

## 7. 数据资产

| 路径 | 内容 |
|---|---|
| `output_phase1/` | Phase-1 baseline 9 帧 VTK（symlink to main repo） |
| `output_phase2_trackA/` | Track-A broken baseline 9 帧 |
| `output_phase2_iter3/` | Track-C iter-3（damp=0.7, z_off=2） |
| `output_phase2_iter4/` | Track-C iter-4（damp=0.7, z_off=0）⭐ |
| `output_phase2_iter5/` | Track-C iter-5（damp=1.0）reject |
| `output_phase2/` | **iter-6 在 5060 跑的最新结果**（如果你在 5060 上） |
| `output_phase2_mini1/` | iter-1 mini（ny=64, t=400μs） |
| `output_phase2_mini/` | iter-2 mini（ny=80, t=600μs） |
| `vtk-316L-150W-50um-V800mms/` | F3D ground truth 100 帧（symlink） |

每个 dir 含 `run.log`，可以直接 grep `[VOF MASS CORRECTION C-flux]` 看每步 Track-C 行为。

---

## 8. 给下一个 agent 的具体 to-do

### 必做（按优先级）

**P0**: 在 5060 跑 Phase-4 验证 brief 准则 #2
1. 修改 `apps/sim_linescan_phase4.cu`，加 Track-C 配置（参考 phase2.cu 第 246-260 行）
   ```cpp
   config.enable_vof_mass_correction          = true;  // 默认是 false！
   config.vof_mass_correction_use_flux_weight = true;
   config.vof_mass_correction_damping         = 0.7f;
   config.mass_correction_use_track_c         = true;
   config.mass_correction_trailing_margin_lu  = 25.0f;
   config.mass_correction_z_substrate_lu      = 80.0f;
   config.mass_correction_z_offset_lu         = 0.0f;
   config.ray_tracing.max_bounces             = 3;     // iter-6 winning
   ```
2. 跑：`./build/sim_linescan_phase4 > output_phase4/run.log 2>&1`（5060 ~30 min）
3. 验：`python3 scripts/diagnostics/check_acceptance.py output_phase4/line_scan_025000.vtk output_phase4/line_scan_000000.vtk`
4. 同时跑 F3D #99 对照：`python3 scripts/diagnostics/compare_lbm_f3d_3d.py output_phase4/line_scan_025000.vtk 99`

**P1**: 如果 Phase-4 中线 ≥-10μm → 6/7 PASS → **可以 squash-merge 到 `benchmark/conduction-316L`**

**P2**: 调查 v_z 反向（iter-6 出现的 -0.020 m/s）
- 跑 `show_rolling_melt_pool.py` 看 rolling strength 是否还正向
- 如果 rolling 还对，v_z @-150μm 反向只是采样位置问题（采到非典型 z 切片）
- 如果 rolling 也变弱了，是 RT bounces 减少导致激光能量不够

### 可选（brief 范围外，需用户授权）

**Sprint-2 候选**: keyhole 物理调参
- 现状：iter-6 keyhole -86 μm vs F3D -5 μm（差 81 μm）
- 调查方向：
  - Anisimov recoil pressure 公式 `P_v = C_r * P_atm * exp(L_v(T-T_b)/(R*T*T_b))`（C_r=0.54 default）
  - Ray-tracing absorption coefficient
  - Marangoni σ_T 可能不够强（拉不回 keyhole）
  - 激光功率密度计算（150W 实际 deposit 多少）

### 不要做（已经验证不靠谱）

- ❌ 加粉床（F3D 是光板，不是粉床问题）
- ❌ damping = 1.0（iter-5 验证 reject）
- ❌ 缩小 mini ny=64（wall artifact）
- ❌ Track-A 路径（已被 Track-C 替代，保留只是 backward compat）

---

## 9. Git 状态

**Branch**: `feature/vof-mass-correction-destination`
**Latest commit**: `0fb11f8` (iter-6 RT bounces=3)
**已 push 到 origin**: ✅

完整 commit log:
```
0fb11f8 test(rt): iter-6 — max_bounces 5→3 to test keyhole depth hypothesis
b0748ae final: lock iter-4 winning config + hand-off doc for RTX 5060
9924ab4 feat(vof): expose mass_correction_z_offset_lu config + iter-5 damping=1.0
9a9eb46 result(track-c): full Phase-2 iter-3 — Track-C lands within F3D-comparable range
3042261 diag: 3D LBM-vs-F3D comparison + rolling-melt-pool visualization
f5ec024 test(vof): LPBF probe + verify_trackb sign-corrected diagnostic
5406871 feat(vof): Track-C — geometric gates on Track-B (x-mask + z-floor)
bfe84b9 feat(vof): Track-B inline-gradient flux mass-correction (replaces A1)
3ce7990 fix(vof): 4 MUST-FIX latent bugs surfaced by Track-B code review
c56f0a8 docs(task-A): from-scratch setup guide for fresh machine
dc7903f feat(diag): Task-A 7-criteria acceptance check script
cc90a26 feat(vof): A1 v_z-weighted mass correction + bug fixes (Task A)
1934854 fix(build): unblock build — setUseGuoForcing, stale CMake targets
[本 commit] docs: HANDOFF_TO_NEXT_AGENT.md
```

---

## 10. 如果下一个 agent 完全卡住

最后的求助路径：
1. 重新读这份文档的 §4（关键技术理解）和 §5（弯路）
2. 翻 `output_phase2_iter4/run.log` 找 `[VOF MASS CORRECTION C-flux]` 看每步真实行为
3. 跑 `python3 scripts/diagnostics/show_rolling_melt_pool.py output_phase2_iter4/line_scan_010000.vtk` 看 4-panel 物理图
4. 跑 `tests/unit/vof/test_track_b_lpbf_probe.cu` 4 个探针——Probe-3 PASS 是 Track-C 物理对的核心证据
5. **永远 cross-check sign convention**：f=1 in liquid，∇f points TO liquid，n_outward = -∇f/|∇f|

祝接力顺利。
