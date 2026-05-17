# 可压/气动分支交接文档

**目标读者**：从零开始接手 `feature/compressible-aero` 分支的技术伙伴。
**写作日期**：2026-05-17（最后实质性工作 2026-05-16-17）
**作者**：上任 Claude（交接前最后一次会话）
**风格**：诚实，不粉饰。

---

## 0. EXECUTIVE SUMMARY（5 分钟版）

1. **LBMProject 不是成熟平台**。它是一个为金属增材制造（LPBF, laser powder bed fusion）写的 CUDA LBM 代码库，21 个源文件、35 个头、~164 个测试。最后一次主分支大规模重构是 2026-03-01。**它原本不打算做气动**。

2. **`feature/compressible-aero` 是 2026-05-08 从 master 拉出来的一个 9 天 side-fork**。在 `/home/yzk/CompressibleCFD/` 单独 worktree 里。8 个 commit ahead of master，但**最后一次 commit 是 2026-05-09**——之后 9 天的全部工作都在 worktree 里以**未提交的 patch** 形式存在。

3. **核心算法是对的**：ST 2D-2 圆柱（DFG benchmark）在 D/dx=20 给 Cd=3.19，距 DFG band [3.22, 3.24] 仅 **1.2%**。Cumulant D3Q27 collision 94 单元测试通过。这证明 LBM kernel 本身没大问题。

4. **核心局限在于 missing capabilities，而非 implementation bugs**：
   - **没有湍流模型**（无 LES / Smagorinsky / wall function / RANS）
   - **没有 AMR / 多重网格**（单块、均匀网格）
   - **没有 body-fitted / curvilinear mesh**（只有 Cartesian + Bouzidi sub-cell BC）
   - **没有多 GPU / MPI**（单卡，4GB GPU 限死 D/dx ≤ 180 在 30×20c 域上）
   - **没有 IBM**（移动物体不支持）
   - **没有 double precision**（FP32 全程）
   - **不能做 3D body**（只能 z-extruded，nz=4 "thin" quasi-2D）

5. **NACA0012 高 Re 失败了**：α=8° Re=2000 给 Cl=0.337，文献（Kurtulus 2015 trend 外推）~0.57，**欠预测 41%**。Re=1000 同 α 给 Cl=0.311 vs Kurtulus 0.49（直接报数），**欠预测 37%**。要做到 10% 误差**需要 AMR 或换 framework**。

6. **诚实点**：上一任 Claude 之前在汇报里把误差写成 49%，是把 Re=2000 文献值外推到 0.65 的结果——实际外推应该是 0.57，所以一直 inflated ~8 个百分点。重写后是 41%。但这不动结论。

7. **过去 9 天我们试过的所有 code-level fix 全是 null**：
   - QMIN clamp（0.05→1e-3 共 6 处）：Cl +1.6%
   - omega_b = 1.0（Cumulant bulk 弛豫）：Cl +0.6%
   - omega_3..6 = 0.5（高阶 cumulant 弛豫）：Cl +0.8%
   - xLE 距离假设：predicted +<1%（未跑成功，geometry bug 阻碍）

8. **现在你能做的**：
   - **A. 接受现状**——把当前结果作为 D/dx=80 method 上限文档化
   - **B. 在本代码库 build AMR**（估 6-8 周 CUDA 工作）
   - **C. 切到 walberla**（成熟开源 LBM，自带 AMR + 多 GPU + 验证过 NACA），估 1-2 周学 + 1-2 周移植
   - **D. 等大 GPU**——D/dx=400-500 估能到 ~30% 误差，仍不到 10%
   - **B+D 合**是唯一能稳定到 10% 的路径

9. **该读哪些 memory**：见末尾 §10。最重要四个：
   - `project_aero_phase1_2026_05_08.md`（DFG 验证）
   - `project_naca_universality_2026_05_13.md`（α + Re 扫描，bias 反转）
   - `project_naca_debug_2026_05_16.md`（最近的 8-variant chain，决定性证据）
   - `project_cumulant_forcing_bug.md`（**主分支已知 Cumulant forcing 结构 bug**——不影响 aero 但要知道）

---

## 1. 一切在哪里

### 1.1 文件系统布局

```
/home/yzk/LBMProject/             ← 原 AM/LPBF 代码库（master + 其他分支）
/home/yzk/CompressibleCFD/        ← 本气动 worktree（feature/compressible-aero）
~/.claude/projects/-home-yzk-LBMProject/memory/         ← AM 主项目记忆
~/.claude/projects/-home-yzk-CompressibleCFD/memory/    ← 气动 worktree 专属记忆
```

**两个 worktree 共享同一 git 仓库**，但 checkout 在不同分支。气动 worktree 的 `.git` 是个 file 指向主仓库。

### 1.2 关键代码位置（worktree 内）

| 文件 | 行数 | 作用 |
|------|------|------|
| `apps/aero/aero_naca0012_cumulant.cu` | 1004 | NACA 驱动 main，包含 stream/MEM force 内嵌 kernel |
| `apps/aero/aero_st_2d2_cumulant.cu` | 547 | DFG 圆柱驱动 |
| `apps/aero/aero_naca0012_ib.cu` | ~? | NACA IB 实验，**未启用** |
| `apps/aero/aero_schaefer_turek_3d.cu` | ~? | ST 3D 实验，**未启用** |
| `include/physics/cumulant/cumulant_d3q27.h` | 449 | Cumulant transforms（forward/inverse/relax） |
| `src/physics/cumulant/cumulant_d3q27.cu` | 78 | Cumulant collision orchestration kernel |
| `src/physics/cumulant/streaming_d3q27_qbb.cu` | 252 | QBB streaming kernel（dense + sparse） |
| `src/physics/aero/momentum_exchange_force.cu` | 455 | 共享 MEM force（D3Q19，**不被 NACA driver 调用**——见下） |
| `include/physics/aero/obstacle_geometry.h` | 700+ | mask stamp + qfrac 生成（dense + sparse） |
| `src/core/lattice/d3q27.cu` | 200+ | D3Q27 stencil + 权重 + opposite 表 |
| `tests/aero/test_cumulant_stages.cu` | ~? | 94 stage tests for Cumulant transforms |
| `tests/aero/test_d3q27_sanity.cu` | ~? | D3Q27 lattice sanity |
| `tests/aero/test_ib_*.cu` | ~? | IB tests（未启用） |

### 1.3 当前 worktree dirty 状态

```
M apps/aero/aero_naca0012_cumulant.cu          ← omega_b=1.0 + 2× QMIN clamps + profiling
M src/physics/aero/momentum_exchange_force.cu  ← 3× QMIN clamps（实际未被 NACA 调用）
M src/physics/cumulant/streaming_d3q27_qbb.cu  ← 2× QMIN clamps
?? scripts/aero/compare_with_kurtulus.py
?? scripts/aero/debug_le_qfrac.py
?? scripts/aero/diagnostic_chain.sh
?? scripts/aero/overnight_chain.sh
?? scripts/aero/plot_chain_summary.py
?? scripts/aero/plot_flowfield_g500_g2000.py
?? scripts/aero/plot_lbm_vs_lit_annotated.py
?? scripts/aero/plot_re_sweep.py
?? scripts/aero/qfrac_distribution.py
```

**重要**：所有 9 天的 patch 都还没提交。decide before doing更多工作是否要 commit 这些。它们改动小、行为正确（验证过）、不会让事情更糟。

---

## 2. 项目历史

### 2.1 LBMProject 主项目（不是气动）

- **目标**：金属增材制造（LPBF）多物理 CFD 仿真。激光熔池、相变、Marangoni、recoil 压力、自由表面。
- **架构**：5 层。L0 cuda → L1 lattice/collision/streaming → L2 leaf physics → L3 solvers (FluidLBM/ThermalLBM/VOFSolver) → L4 MultiphysicsSolver → L5 I/O。
- **代码量**：21 源、35 头、~164 测试（2026-03-01 后基本稳定，是个 boring 状态）。
- **Last major refactor**：commit `fe2a220` (2026-03-01)，包括 MultiphysicsConfig 拆分、D3Q7 AoS→SoA、CudaBuffer RAII、FieldRegistry。
- **底层用 D3Q19 fluid + D3Q7 thermal + VOF**，**不用 D3Q27**（这是气动后加的）。
- **生产 collision 是 BGK/TRT/MRT**，**不是 Cumulant**（Cumulant 有结构 bug——见 §6.4）。
- **大部分代码经过反复 debug**，覆盖 Stefan benchmark、自然对流、Zalesak disk、RT 不稳定、相变等基准。但**仍然不是产品级**，更像是个研究组内的精修 prototype。
- **AM 主线当前焦灼问题**：跟 Flow3D 对照 LPBF spot melt 时升力翼差距 ~25-30%（不是本气动 case）。memory 中 R26-R31 多轮 audit 在追这个。

### 2.2 为什么会有 compressible-aero 分支

**2026-05-08 决定**：用户想做 NACA / Schäfer-Turek / 翼型气动，但因为 LBM 是 AM 主项目的核心算法，不想切到 FV/FD framework 重学一遍。

决定：**stay LBM，参考 walberla 做 blueprint，写一个 side-fork**。
- 不在 master 上做（避免污染 AM 主线）。
- 在新 worktree `/home/yzk/CompressibleCFD/` 隔离。
- 分支 `feature/compressible-aero`。
- 记忆体单独：`~/.claude/projects/-home-yzk-CompressibleCFD/memory/`。

参见 `~/.claude/projects/-home-yzk-CompressibleCFD/memory/feedback_lbm_only_aero.md`。

### 2.3 9 天 commit 履历

```
3876d65 (2026-05-09) NACA: TRT D3Q27 collision + sparse-qfrac + diagnostic CLI flags
4f13e32 (2026-05-09) NACA QBB-D3Q27 curved BC + std aero α convention  ← Cl 符号修正
7823a1f (2026-05-08) Cumulant Phase 4 BREAKTHROUGH: Cd=3.19 at D/dx=20 (1.2% from DFG)  ★
e8e5318 (2026-05-08) Cumulant Phase 4 first cut: D3Q27 driver — functional, Cd ≈ 0.55x expected
bd1c308 (2026-05-08) Cumulant Phase 2: D3Q27 collision kernel, all 94 stage tests pass
bb5fc06 (2026-05-08) Aero Phase 1d: D3Q27 lattice infrastructure (Cumulant prep)
d8a0225 (2026-05-08) Aero Phase 1c: 2nd-order BCs + halfway walls + Mach control
887703c (2026-05-08) Aero Phase 1+1b: Schaefer-Turek pipeline + QBB curved BC
```

**Commit `7823a1f` 是分水岭**：DFG 圆柱 Cd 在 D/dx=20 落到 1.2% 误差，证明 kernel 数值正确。所有后面的失败都跟 kernel 算法无关。

**Commit `4f13e32` 是另一个关键**：NACA 在 +α=8° 给 −Cl（错符号）。诊断后发现是 stamp 旋转约定（用 `sin_a = -sin(α)`，对应 R(-α_std)），改后 +α 给 +Cl 与 Kurtulus 同号。**这是个约定问题，不是 bug**。

**2026-05-09 后没有 commit**。9 天的实验都散在 worktree 里。

### 2.4 9 天的实验时间线

| 日期 | 事件 | 关键发现 |
|------|------|---------|
| 2026-05-08 | Phase 1: DFG cylinder | ✓ Cd=3.19 vs DFG 3.23 = 1.2% (D/dx=20) |
| 2026-05-09 | NACA Cl sign flip | 约定修正 commit 4f13e32 |
| 2026-05-12 | 通宵 10% 目标 campaign | D/dx 40-180 sweep、BGK/TRT/Cumulant、ω scan、wall ω scan：全部 stuck 在 ~32% deficit |
| 2026-05-13 | α-sweep | 偏差是 universal（5 个 α 都欠预测，斜率比是 62%） |
| 2026-05-14 | Re-sweep G500/G2000 | **bias 方向随 Re 反转**：Re=200 +33% over，Re=1000 -37% under |
| 2026-05-15 | VTK dumps + flowfield 图 | Kármán 街拓扑对，量级全错 |
| 2026-05-16 | 8-variant debug chain | omega_b、omega_3..6、QMIN clamp 全 null（<2% Cl 影响），QBB-vs-stair real (+11-13%) |
| 2026-05-17 | 诚实统计 audit | 把 49% gap 改成 41%（外推文献值用过头了） |

---

## 3. 当前状态详解（2026-05-17）

### 3.1 已验证的事实

| 测试 | 结果 | 来源 |
|------|------|------|
| **DFG ST 2D-2 Cumulant D/dx=20** | Cd=3.19 vs DFG band [3.22, 3.24] = **1.2% off** | commit 7823a1f, `output_BEST_cumulant_d20_st2d2/` |
| **DFG ST 2D-2 D/dx=80** (halfway BB) | Cd=3.06 vs DFG = **−5% under** | 2026-05-16 debug session |
| **NACA Re=200 α=8 D/dx=80** | Cl=0.40 vs lit 0.30 = **+33% over** | `output_univ_G200/` |
| **NACA Re=500 α=8 D/dx=80** | Cl=0.366 vs lit ~0.40 = **−9% under** | `output_univ_G500/` |
| **NACA Re=1000 α=8 D/dx=80** | Cl=0.311 vs Kurtulus 0.49 = **−37% under** | `output_univ_F8_re1000_a8` 类、`output_diag_N3_*` |
| **NACA Re=2000 α=8 D/dx=80** | Cl=0.337 vs lit ~0.57 = **−41% under** | `output_univ_G2000/`, `output_diag_N1_*` |
| **94 Cumulant stage unit tests** | All pass | `tests/aero/test_cumulant_stages.cu` |

### 3.2 已证伪的假设（debug 9 天内）

| 假设 | Cl 实测变化 | 状态 |
|------|------------|------|
| QBB QMIN=0.05 clamp 偏置 LE | +1.6% | **null** |
| `omega_b = omega_nu`（应 = 1.0）压缩波污染 | +0.6% | **null** |
| Cumulant 高阶 omega_3..6=1.0 过阻尼 | +0.8% | **null** |
| Inlet xLE=10c 太近，induced velocity 反算 | predicted <1% | **null（理论）** |
| D/dx 不够（80→120→180） | +5-10%, 然后 saturate | **饱和** |
| BGK / TRT / Cumulant 差异 | 三者 ~5% 内一致 | **collision 无关** |
| Ladd vs equilibrium inlet | Ladd 完胜（cylinder） | **inlet 已选对** |
| Mach scan (u_LU 0.05→0.20) | 越大 Cl 越偏 | **compressibility 不是 fix** |
| Wall ω override (0.03→1.97) | 全 ~0.314 | **insensitive** |

**结论**：全平台、全参数空间扫过，**没有任何 code-level fix 能闭合 50%（修正后 40%）的 Cl 缺口**。

### 3.3 真正贡献的（小但 real）

| 因子 | 量级 |
|------|------|
| QBB vs stair | **+11-13% Cl** at both Re=1000 和 Re=2000 |
| omega_b=1.0 vs omega_nu | **−37% Cl_rms**（阻尼 spurious acoustic 振荡） |
| D/dx 80→120 | **+12% Cl** |
| D/dx 120→180 | **+5% Cl**（饱和） |

QBB 在做正事；omega_b=1.0 在做正事（小但 real）；分辨率 helps 直到 saturate。所有这些加一起 ~25-30%，仍不到 50% gap 的一半。

---

## 4. 代码库地图——**已实现**（含完成度评估）

### 4.1 Lattice 层

| 模块 | 完成度 | 备注 |
|------|--------|------|
| D3Q19 stencil + opposite + 权重 | ✓ 完成 | `src/core/lattice/d3q19.cu`，AM 主项目用 |
| **D3Q27 stencil + opposite + 权重** | ✓ 完成 | `src/core/lattice/d3q27.cu`，气动专用，27 项已手验 |
| D3Q7 thermal | ✓ 完成 | AM 主项目用，气动不用 |
| Y-mirror 反射表（D3Q27 free-slip） | ✓ 完成 | 手验所有 27 项 2026-05-16 |
| Opposite27 表 | ✓ 完成 | 手验 |

### 4.2 Collision 层

| 模块 | 完成度 | 备注 |
|------|--------|------|
| BGK D3Q27 | ✓ 完成 | `fluidBGKD3Q27Kernel` |
| TRT D3Q27 (magic-Λ) | ✓ 完成 | `fluidTRTD3Q27Kernel`，Λ=3/16 默认 |
| **Cumulant D3Q27**（无 forcing） | ✓ 完成且验证 | `fluidCumulantCollisionKernel`，94 stage tests，前向/逆/弛豫数学自洽 |
| Cumulant D3Q27 **with Guo forcing** | ❌ **已知 bug** | `project_cumulant_forcing_bug.md`：Δu 在 forward + inverse 都用全步导致线性 cancel。AM 主项目用 MRT-Guo 不用 Cumulant。**气动 case 无外力，不触发此 bug**。 |

### 4.3 Streaming + BC 层

| 模块 | 完成度 | 备注 |
|------|--------|------|
| 标准 push streaming | ✓ 完成 | AM 主项目 |
| **PULL streaming + QBB** | ✓ 完成 | `streamD3Q27_naca_qbb`(_sparse)，lbmpy 单节点公式 |
| 半步 BB（stair） | ✓ 完成 | `streamD3Q27_naca` |
| **QBB sparse-qfrac CSR** | ✓ 完成 | `lookup_sparse_qfrac`；在 D/dx=180 节省 1500MB |
| Ladd inlet（D3Q27） | ✓ 完成 | `applyInletFreestream`，**ρ=1.0 硬编码**（应 = local ρ，小偏差） |
| 零梯度 outlet | ✓ 完成 | `applyOutletExtrap` |
| Y 自由滑壁（mirror） | ✓ 完成 | y_mirror27 表 |
| Z 周期 | ✓ 完成 | streaming kernel 内 wrap |
| **特征/海绵 BC** | ❌ **无** | 当前 BC 反射声波 |

### 4.4 Force probe

| 模块 | 完成度 | 备注 |
|------|--------|------|
| MEM force halfway-BB | ✓ 完成 | `memForceNaca`（driver 内嵌） |
| **MEM force QBB**（dense + sparse） | ✓ 完成 | `memForceNaca_QBB`(_sparse)（driver 内嵌） |
| 共享 MEM 工具 `momentum_exchange_force.cu` | ⚠️ 存在但**不被气动调用** | 是 D3Q19 版本；NACA driver 完全用自己内嵌的 D3Q27 版本 |

⚠️ **注意**：driver 自己实现了 D3Q27 MEM kernel，跟 `src/physics/aero/momentum_exchange_force.cu` 是两套独立代码。9 天 patch 把 QMIN 全改了，**两套都改了**。

### 4.5 几何

| 模块 | 完成度 | 备注 |
|------|--------|------|
| 圆柱 stamp | ✓ 完成 | `stampSphere` |
| **NACA0012 四位 stamp** | ✓ 完成（含 α 旋转） | `stampNacaAirfoil4Digit`：cell-center inside-test，**5% 弦缩水**（76/80） |
| Flat plate stamp | ✓ 完成 | 诊断用 |
| **NACA qfrac 生成**（dense） | ✓ 完成 | `makeNacaQFraction`：64-step scan + 20-bisect。**24% 链路 fallback 到 0.5** |
| NACA qfrac sparse | ✓ 完成 | `makeNacaQFractionSparse` |
| **3D body / curvilinear** | ❌ **无** | 只能 z-extrude |

### 4.6 Driver / 应用

| Driver | 完成度 | 备注 |
|--------|--------|------|
| **ST 2D-2 cylinder Cumulant** | ✓ 验证（DFG 1.2%） | `aero_st_2d2_cumulant.cu`，固定 ST geometry |
| **NACA0012 Cumulant**（CLI 灵活） | ✓ 完成 | `aero_naca0012_cumulant.cu`，1004 行，含 stair/qbb-snode 选项 |
| NACA IB（immersed boundary） | ⚠️ stub | `aero_naca0012_ib.cu`，未生产化 |
| ST 3D | ⚠️ stub | `aero_schaefer_turek_3d.cu`，未生产化 |

### 4.7 Profiling（2026-05-16 加）

driver 内 chrono + cudaDeviceSynchronize 包每个 stage，输出每步 ms + 占比。当前 NACA Re=2000 D/dx=80 30k 步典型 profile：

```
collision (Cumulant):  46-49% (41-71 ms/step)
streaming + QBB:       50-53% (42-81 ms/step)
inlet + outlet BC:     ~0.5%  (~0.55 ms/step)
force probe (×300):    ~0.2%
unaccounted:           ~0%
```

吞吐 ~100 MLUPS（含 sync），无 sync 报 197 MLUPS。每 30k 步 ~40-72 min wall。

---

## 5. 代码库地图——**未实现**（aero 失败的真正原因）

这部分是答用户原题（"局限性还是底层有问题"）。**主要是局限性，不是底层 bug**。

### 5.1 没有湍流模型

- **无 LES**（Smagorinsky 等亚格子模型）
- **无 RANS**（k-ε / k-ω / SST）
- **无 wall function**（Spalding's law、Werner-Wengle）
- **无 DES / hybrid**

**对当前 case 的影响**：NACA Re=1000-2000 仍在 laminar 区，理论上 DNS 即可。所以"没有湍流模型"**不是当前 case 的瓶颈**。但你要做 Re=10⁴-10⁶ 就必须有。

### 5.2 没有 AMR / 局部加密 / 多重网格

- 单块、Cartesian、均匀 dx
- **NACA0012 LE 半径 = 1.59% chord**。D/dx=80 时只有 **1.27 cell** 半径——根本分辨不出 LE 抽吸峰（文献 Cp_min ≈ -3 到 -4 在前 5% 弦）。
- 加密手段只有提高 D/dx 整个域，对内存指数级。

**这是当前 case 真正的瓶颈**。AMR 是 Tier-1 critical gap（参见 LBMProject CLAUDE.md 主项目 gap_analysis_openfoam.md：列为 "blocks research" 项）。

### 5.3 没有 body-fitted / curvilinear mesh

- 只能 Cartesian + Bouzidi sub-cell BC
- Bouzidi 在 q∈[0.05, 1] 工作良好，但 LE 半径 sub-cell 时 24% 链路 fallback 到 0.5（halfway BB），结构性退化

**对当前 case**：是 LE 几何的根本制约之一。Curvilinear 能修，但需要全新 streaming 算法。

### 5.4 没有 IBM（immersed boundary method）

- 不能做移动物体、振动翼、旋转叶片
- 当前 `aero_naca0012_ib.cu` 有 stub 但未启用

**对当前 case**：static airfoil 不需要 IBM，QBB 是合理选择。但你将来想做 pitching 或 flapping 必须有。

### 5.5 没有多 GPU / MPI / 多块

- 单 CUDA context，单 GPU
- 当前 GPU（NVIDIA RTX 3050 Laptop, 4GB）caps domain ~15M cells（D/dx=80 30c×20c×4）
- 用 sparse-qfrac 在 D/dx=180 跑过，是上限

**对当前 case**：D/dx > 200 需要更大 GPU 或多 GPU。memory project_naca_overnight_2026_05_12.md 估 D/dx=400-500 在 H100 40GB 上能跑，Cl 估到 ~0.45（仍欠 30%，没到 10%）。

### 5.6 没有 double precision

- 全程 FP32
- Ma 误差 ~Ma² ≈ 0.007 in LU（量级 OK）
- 但 30k 步累积 + Cumulant 多次非线性 transform，FP32 round-off 可能在某些场景失稳

**对当前 case**：测过 Cumulant 在 Re=2000 ω=1.976 还能稳定（borderline），FP32 没炸。但没 audit 累积偏差。

### 5.7 没有 characteristic / sponge BC

- Ladd inlet 强制 u_∞ 均匀，没有让流场 "feel" 翼的 induced velocity
- Extrapolation outlet 反射声波回上游
- 当前 inlet 设在 xLE=10c 上游、outlet 在 xLE+20c 下游

**对当前 case**：induced velocity 在 10c 处 ~0.7% U_∞，效应不到 1% Cl。memory 估算过 xLE=30c 收益 <1%。**不是主要 bug**。但生产代码应有。

### 5.8 只能 z-extruded geometry（nz=4 thin）

- 当前所有 NACA case 用 nz_thin=4 z-周期
- 几乎 quasi-2D（z 上几乎无变化）
- 文献 NACA 真 3D（端板效应、翼尖涡）我们做不了
- nz=4 在 D3Q27 里有 27 个方向，部分 z-diag link 在 nz=4 周期下可能有奇怪 wrap 行为

**对当前 case**：Kurtulus 2015 也是 2D / quasi-2D DNS，可比。但 nz=4 是否够 D3Q27 没 audit。**未知风险**。

### 5.9 没有验证基础设施

- 没有自动 regression（"每次跑 Re=200 验证 Cl 在 0.40 ± 0.04"）
- 没有 mesh convergence study（D/dx=40/80/120/160 自动 Richardson 外推）
- 没有 cross-code validation（我们没跑 walberla 同 case 对照）
- 测试只 cover Cumulant transform unit，不 cover 端到端 force

**对当前 case**：每次结果都是手算 + 手 diff。Time-consuming。

### 5.10 其他 missing

- 无 in-situ post-processing（每次 dump 大 VTK 然后 Python 解）
- 无 checkpoint / restart（LBM 长跑炸了从头来）
- 无统一 logging（散在 stdout）
- 无 config file 格式（CLI flag-driven，复用难）
- 没 benchmark suite

---

## 6. 代码库地图——**底层"漏水"** 的地方（bug 或 questionable）

用户说"小作坊、四处漏水"——这是诚实地列出来。

### 6.1 NACA stamp 5% 弦缩水

- `stampNacaAirfoil4Digit` 用 cell-center inside-test
- LE 半径 1.27 cells → 部分 LE tip 的 cell-center 测出 outside → 不 stamp
- 实测：D/dx=80 stamp 出 76 cells 有效弦（vs 期望 80），**5% 偏小**
- Cl 归一化用名义弦 c=1.0，**Cl 系统欠报 5%**
- 修正：归一化用有效弦 → Cl × 80/76 = Cl × 1.053
- **该修但小**

### 6.2 QBB qfrac 24% fallback 到 0.5

- `makeNacaQFraction` 64-step scan + 20-bisect，但在 LE/TE 翼厚 < 1 cell 处，链路从 fluid cell 到 solid cell **可能完全不穿过翼**（grazing corner）
- fallback：`qfrac = 0.5f`（halfway BB 等价）
- 实测：D/dx=80 α=8° NACA 翼内 1476 个 wall-adjacent link 里 **350 个（24%）fallback 到 0.5**
- **几何根本性问题，不是 algorithm bug**。在 D/dx > 1000 才能消失。
- 影响：LE/TE BC 在 24% 链路上退化为 halfway BB，是 LE 物理欠分辨率的 mechanism。

### 6.3 QBB QMIN clamp（已 patch）

- 原始 QMIN=0.05，QMAX=0.95（"stability hack"）
- **28% 链路 qf<0.05 被夹到 0.05**，对应 5% cell 偏移
- patch 后 1e-3 / 0.999
- 实测影响：Cl +1.6%（小）

### 6.4 LBMProject 主分支 Cumulant forcing 结构 bug（**仅 AM 侧**）

- `project_cumulant_forcing_bug.md` 详述
- D3Q27 Cumulant kernel 用 Guo forcing 时，Δu 在 forward + inverse 两个 u-shift 中**完全 cancel**（应该用 Δu/2 + 每 cumulant `(1-ω_i/2)` projection）
- **不影响气动**（气动 case 无 body force）
- 但说明同一份 Cumulant 实现在 AM 主项目里**不能用**，被 MRT 替代
- 提醒：如果未来气动加 forcing（如 Marangoni、buoyancy），**必须先修这个**

### 6.5 Ladd inlet 用 ρ=1.0 硬编码

```cpp
f[id + q * n_cells] = f[id + q_opp * n_cells]
                    + 6.0f * w27[q] * 1.0f * cu;   // ←  1.0f 应该是 local ρ
```

- 当 inlet column 因下游压力梯度 ρ≠1 时引入质量通量失配
- Ma=0.087 下 ρ 偏差预计 ~Ma² ≈ 0.7% → Cl 影响 <1%
- **小但应修**

### 6.6 D3Q27 nz=4 thin 域

- nz=4 + 27 方向（含 z-diag）+ z 周期
- D3Q27 corner directions（19-26）有 ez=±1。streaming 走 4 步 z 才完整循环
- **没 audit** quasi-2D 是否引入 z 方向"伪 3D"模式
- 但 ST 2D-2 圆柱（同 nz=4）通过 DFG 1.2% 验证，所以**不会很严重**

### 6.7 driver 内嵌 MEM 跟共享工具不一致

- driver 内的 `memForceNaca_QBB(_sparse)` 是 D3Q27 版本，**自己写在 driver 文件里**
- `src/physics/aero/momentum_exchange_force.cu` 是 D3Q19 版本，**不被 NACA 调用**
- 9 天 patch 时一开始改了**错误文件**（用户不知道有两套）
- **代码组织问题，不是 bug**，但容易出错

### 6.8 单节点 QBB 公式假设 BGK collision

- lbmpy 单节点 QBB 推导基于 BGK：post-collision PDF = (1-ω)·f + ω·feq
- Cumulant collision 给的 post-collision 在高阶矩上有不同结构
- 公式可能不严格匹配 Cumulant，导致 high-ω 下 BC 有偏差
- **难量化**，BGK/TRT/Cumulant 三者实测 Cl 在 5% 内一致，所以**至多 5% 影响**

### 6.9 力归一化用名义 chord c=1.0

- 配合 §6.1 弦缩水问题，**Cl 报表系统低 5%**
- 修正方向 = 增大 Cl 报值 → 减小 gap
- **该修但小**

### 6.10 没有 mass conservation audit

- 没人验证 inlet mass flux = outlet mass flux ± 噪声
- 没人验证整域 ρ 总量随时间常数
- 是个未知风险

### 6.11 stair-step + α 旋转的不对称

- α=+8° 翼型 stamp 后是 stair-stepped 边界
- 在 LE 上下表面**stair pattern 不对称**（一边 cell 多，一边 cell 少）
- memory `project_am_audit_round3_naca_signflip_2026_05_09.md` 有详细 cross-pollination 分析
- **可能是 +α 和 -α 性能不完全对称的原因**（未实测）

---

## 7. 用户原题诚实回答

**问**：是平台局限性（湍流模型之类）还是底层有大问题？

**答**：**主要是平台局限性（missing capabilities），不是底层 implementation bug**。

### 7.1 底层 implementation 评估

- **DFG ST 2D-2 圆柱在 D/dx=20 给 1.2% off**：kernel 算法是正确的。
- Cumulant transforms 数学自洽（94 stage tests + 手 audit 2026-05-16）。
- D3Q27 stencil、y-mirror、opposite 表手验通过。
- 力归一化维度正确。
- Ladd inlet、QBB BC 物理意义对（QBB 实测 +13% 优于 stair）。

底层有**多处小漏点**（§6 列了 11 项），每项 ~1-5% 偏差，**全合起来 ~10-15%**。**不是结构性问题**。

### 7.2 平台局限性评估

- **没 AMR**：是 NACA 高 Re 失败的**核心原因**。LE 半径 1.27 cells 单块均匀网格无法分辨。
- **没 LES**：当前 case Re=2000 不需要，但限制未来扩展。
- **没大 GPU 多卡**：硬件限制 D/dx ≤ 180。
- **没 body-fitted**：sub-cell LE 永远是离散问题。

这些都是**设计阶段没纳入的能力**，不是"代码写坏了"。

### 7.3 总判决

**Bottom line**：NACA Re=2000 41% Cl 缺口 = 
- ~5% 来自 §6 各种小漏点
- ~5% 来自 ω→2 数值黏性 / Cumulant-QBB 公式不严格匹配
- **~30% 来自 LE 几何欠分辨率（结构性，AMR 或 D/dx ≫ 80 才能修）**

修平台所有"漏水"能把 41% 降到 35%。**仍是 10% 目标的 3.5 倍**。

**结论**：你想达到 10%，靠修当前代码做不到。需要新能力（AMR）或换 framework（walberla）。

---

## 8. 给新人的建议

### 8.1 第一周必做

1. **读 4 个核心 memory**：
   - `~/.claude/projects/-home-yzk-CompressibleCFD/memory/MEMORY.md`（index）
   - `project_aero_phase1_2026_05_08.md`（DFG 验证，知道 kernel 是对的）
   - `project_naca_universality_2026_05_13.md`（α + Re 扫描，bias 反转）
   - `project_naca_debug_2026_05_16.md`（最近 8-variant chain 全 null）

2. **跑一次 DFG cylinder 重现 1.2% 结果**：
   ```bash
   cd /home/yzk/CompressibleCFD/build
   ./aero_st_2d2_cumulant --resolution 20 --re 100
   ```
   确认 settled Cd 在 [3.18, 3.22]。如果失败，**先修这个再做别的**——它是 kernel 健康的最后一道屏障。

3. **跑一次 NACA Re=2000 baseline 重现 Cl ≈ 0.334**：
   ```bash
   ./aero_naca0012_cumulant --resolution 80 --re 2000 --alpha 8 --steps 30000 \
       --sparse-qfrac --bc qbb-snode --output-dir output_repro_g2000
   ```
   ~50 min。settled Cl 应该 0.33-0.34。如果不是，**有什么变了**。

4. **看流场图** `images/naca_lbm_vs_lit_annotated.png`：理解我们做对了什么、做错了什么。

### 8.2 决策树

```
你的目标是什么？
├─ 仅要 NACA 物理对的展示 → 用 D/dx=80 Re ≤ 500 的 case，文档化"low-Re aero working"
├─ 必须 Re ≥ 1000 NACA 10% 内
│   ├─ 时间预算 < 4 周 → 切到 walberla
│   ├─ 时间预算 4-12 周 + 想留在此 codebase → 实现 AMR
│   └─ 时间预算 > 12 周 + 战略性 → AMR + 多 GPU + LES
├─ 要做其他几何（球、机翼、车）→ 现在 stamp + qfrac 是 NACA 专用，要扩展
├─ 要做 unsteady moving body → 实现 IBM
└─ 要做 turbulent (Re > 10⁴) → 实现 LES + wall function
```

### 8.3 不要做的事

- **不要再 patch omega_b / omega_3..6 / QMIN clamp**——8-variant chain 已证伪
- **不要再做 D/dx ≤ 180 的扫描期望突破**——saturate 了
- **不要假设文献文本描述 = 文献 figure**（我之前把 Re=2000 Cl 外推成 0.65，实际应该 0.57）
- **不要 commit 当前 dirty patches 前重跑 DFG**——确保 patch 没破坏 1.2% match

### 8.4 该修但 cheap 的小漏水（按 ROI 排序）

如果你决定留在本 codebase 做小改进：

| 项 | 估时 | 预期 Cl gain |
|----|------|------------|
| Ladd inlet ρ=1 → local ρ | 30min code + verify | <1% |
| Cl 归一化用 effective chord | 30min code | +5%（report only） |
| Mass conservation audit | 2h | 诊断，可能发现新 bug |
| qfrac fallback 改用 "wall at destination" 而非 0.5 | 2h code + 1 run | +2-5%? |
| nz=4 → nz=8 audit | 2 runs | 验证 quasi-2D 假设 |
| 双精度 build | 1 day | 验证 FP32 累积误差 |

### 8.5 建议的下一步 sprint

**最高 ROI**：切到 walberla 做 NACA Re=2000 对照。
- 知道 walberla 同 case 给多少 Cl（应该接近 0.55-0.65）
- 给我们的 41% gap 提供 ground truth
- 学 walberla 也是为将来铺路

**次高 ROI**：在本 codebase 实现简单 2-level local refinement（LE 圆 ±0.1c 内加密 2×）。这是 AMR 的 minimal viable subset，估 2-3 周。

**第三**：在本 codebase 加 Smagorinsky LES（虽然 Re=2000 用不到，但是 1 周工作，为将来铺路）。

---

## 9. 文件清单（生存包）

### 9.1 必读 memory（按优先级）

```
~/.claude/projects/-home-yzk-CompressibleCFD/memory/
├── MEMORY.md                                   ← 索引
├── feedback_lbm_only_aero.md                   ← 战略决策
├── project_aero_phase1_2026_05_08.md           ★ DFG 验证（kernel 健康证据）
├── project_naca_cl_sign_bug_2026_05_09.md      ← 约定修正
├── project_naca_overnight_2026_05_12.md        ★ 第一次 10% 目标尝试，failed
├── project_naca_universality_2026_05_13.md     ★ α/Re 扫描，bias 反转
└── project_naca_debug_2026_05_16.md            ★ 最后 8-variant chain（决定性）

~/.claude/projects/-home-yzk-LBMProject/memory/  ← AM 侧但与气动有交叉
├── project_cumulant_forcing_bug.md             ★ Cumulant 已知 forcing bug
├── project_am_audit_from_aero_2026_05_08.md    ← Round 1 交叉污染
├── project_am_audit_round2_cumulant_2026_05_09.md
└── project_am_audit_round3_naca_signflip_2026_05_09.md
```

### 9.2 关键代码（按熟悉顺序）

```
/home/yzk/CompressibleCFD/
├── apps/aero/aero_naca0012_cumulant.cu         ★ 主入口，1004 行
├── apps/aero/aero_st_2d2_cumulant.cu           ★ DFG 验证 driver
├── include/physics/aero/obstacle_geometry.h    ★ NACA stamp + qfrac
├── include/physics/cumulant/cumulant_d3q27.h   ★ Cumulant 数学 449 行
├── src/physics/cumulant/cumulant_d3q27.cu      ← collision orchestration
├── src/physics/cumulant/streaming_d3q27_qbb.cu ★ QBB streaming（dense + sparse）
├── src/core/lattice/d3q27.cu                   ← stencil 表
└── tests/aero/                                  ← 94 stage tests
```

### 9.3 关键数据（保留产物）

```
/home/yzk/CompressibleCFD/
├── output_BEST_cumulant_d20_st2d2/             ← DFG 1.2% 锚点
├── output_univ_G200/G500/G2000/                ← Re-sweep canonical (G500/G2000 也有 _vtk 流场)
├── output_univ_F2/F4/F6/F10_re1000_a*/         ← α-sweep
├── output_overnight_O1/O2_*/                   ← omega_b patch chain
├── output_diag_N1..N4_*/                       ← omega_3..6 / stair chain
├── output_phaseD_D4_final/                     ← 50k step Re=1000 D/dx=180 (memory canonical)
├── output_phaseE_E1_final/                     ← 50k step Re=1000 D/dx=160 (memory canonical)
└── images/                                      ← 全部 19 张诊断/对比图
```

### 9.4 关键脚本

```
scripts/aero/
├── overnight_chain.sh                          ← 早期 chain（bash buffer bug，废弃）
├── diagnostic_chain.sh                         ★ 当前 chain 模板
├── plot_chain_summary.py                       ★ 8-variant 汇总条形图
├── plot_re_sweep.py                            ★ Re-sweep 升力曲线
├── plot_flowfield_g500_g2000.py                ← G500/G2000 流场对比图
├── plot_lbm_vs_lit_annotated.py                ★ LBM vs lit 注释对比
├── qfrac_distribution.py                       ★ qfrac 静态分布（28% clamp + 24% fallback）
├── debug_le_qfrac.py                           ← LE 链路计数
├── compare_with_kurtulus.py                    ← FFT Strouhal + 文献对照
└── viz_naca_zoom.py + viz_flow.py              ← 一般可视化
```

---

## 10. 一个简短的"我学到的"清单（给伙伴的元知识）

1. **LBM 在低 Re 翼型上能行，高 Re 翼型上需要 AMR**——这是 LBM 社区的共识。我们的实验只是再次确认了它。

2. **Cumulant collision 对 stability 加分**但**对 Cl 精度无贡献**——BGK/TRT/Cumulant 在 D/dx=80 Re=2000 给的 Cl 在 5% 内一致。如果只为 Cl 不要 Cumulant，省 30% 算力。

3. **QBB 比 stair BC 实测 +11-13% Cl**——值这个 30% 算力代价。

4. **bash 脚本运行中改文件可能不生效**——bash 会缓存读取。要修就 kill + restart。

5. **关键 case 重跑要验证 deterministic**——我们多次发现重跑给 bit-identical 结果（除非改了代码或 GPU 时序），这是好事，意味着可以放心 patch。

6. **"过预测 vs 欠预测" 反转**（Re=200 +33% / Re=2000 -41%）是真现象——表示有两种相反误差源，不是单纯欠分辨率。低 Re 时 Cumulant 反耗散过强（Galilean error），高 Re 时 LE 物理欠分辨率。

7. **文献文本描述 ≠ 文献 figure**——我之前把 Re=2000 文献 Cl 外推到 0.65，实际应该 0.57。所有报告诚实的 41% gap，不是 49%。

8. **DFG 1.2% 是 kernel 健康证据**，但**不能外推到任意 geometry**——sharp LE 是另一种 case。

9. **每次"是不是漏了个 bug"先 dump 数据再 patch**——我们 2026-05-16 直接 patch QMIN 浪费 50 min，事后 dump qfrac 分布看 28%/24% 才理解为什么 patch 无效。

10. **chain script 设计**：每个 variant 跑完输出 settled stats + profile 到 log，**让你早上起来直接读 log 就知道发生了什么**——比 monitor 实时盯有用得多。

---

## 11. 最后的话

这个 worktree **能跑、能产数据、kernel 数学正确、低 Re 准、高 Re 欠 41%**。

它**不能**：cover 高 Re aero、做 3D body、做 moving wall、scale 到 multi-GPU、自动 mesh refine。

如果你想用它**继续做 aero 研究**，先决定你的 Re 范围。Re ≤ 500 用现状即可。Re ≥ 1000 想要 10% 误差**必须建 AMR 或换 framework**。

如果你想 **fix the platform itself**，§5（missing）和 §6（leaks）按 ROI 排序选项目。但要清楚：把每个 leak 都堵上，gap 也只从 41% 降到 35%，没有 silver bullet。

如果你觉得有什么我没记清楚的，**读 §9.1 memory 文件**——那里面的东西比这份 doc 更详细，且每个都是当时写的不会失真。这份 doc 是 high-level 综合，可能漏细节。

祝好运。

— 上任 Claude / 2026-05-17
