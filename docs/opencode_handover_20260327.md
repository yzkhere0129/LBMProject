# Opencode 项目交接报告

> **日期**: 2026-03-27  
> **分支**: `opencode/sandbox-20260327`  
> **基础提交**: `d56b2ad` (Gas phase thermal fixes + evap cooling compensation)  
> **原始分支**: `benchmark/conduction-316L`

---

## 一、代码架构概览

本项目是一个 **GPU加速的LBM多物理场求解器**，专注于金属增材制造（LPBF）仿真，采用 **5层无循环依赖架构**：

```
L0: CUDA Primitives (CudaBuffer, error check)
 └→ L1: Lattice Infrastructure (D3Q19, D3Q7, BCs)
     └→ L2: Leaf Physics (FluidLBM, ThermalLBM, VOFSolver, PhaseChangeSolver, ...)
         └→ L3: Force Pipeline (ForceAccumulator)
             └→ L4: Coupling Orchestration (MultiphysicsSolver)
                 └→ L5: I/O & Diagnostics (VTK, FieldRegistry, EnergyBalance)
```

**关键技术特征**：
- **内存布局**：SoA（`f[q * num_cells + cell_idx]`），保证GPU warp内合并访存
- **编译模式**：CUDA分离编译（`-rdc=true`），所有设备常量使用`__device__`
- **核心模块数**：~10,300行代码，45个源文件/头文件

---

## 二、核心物理模块位置

| 模块 | 头文件 | 源文件 | 状态 |
|------|--------|--------|------|
| **流体求解器** | `include/physics/fluid_lbm.h` | `src/physics/fluid/fluid_lbm.cu` | ✅ 成熟 |
| **热传导求解器** | `include/physics/thermal_lbm.h` | `src/physics/thermal/thermal_lbm.cu` | ✅ 成熟 |
| **VOF自由面** | `include/physics/vof_solver.h` | `src/physics/vof/vof_solver.cu` | ✅ 成熟 |
| **相变求解器** | `include/physics/phase_change.h` | `src/physics/phase_change/phase_change.cu` | ✅ 成熟 |
| **Marangoni力** | `include/physics/marangoni.h` | `src/physics/vof/marangoni.cu` | ✅ 成熟 |
| **力管线** | `include/physics/force_accumulator.h` | `src/physics/force_accumulator.cu` | ✅ 成熟 |
| **激光热源** | `include/physics/laser_source.h` | `src/physics/laser/laser_source.cu` | ✅ 成熟 |
| **光线追踪激光** | `include/physics/ray_tracing_laser.h` | N/A | 🆕 新增 |
| **反冲压力** | `include/physics/recoil_pressure.h` | `src/physics/vof/recoil_pressure.cu` | ✅ 成熟 |
| **粉末床生成** | `include/physics/powder_bed.h` | `src/physics/powder/powder_bed.cu` | ✅ 成熟 |
| **多物理场耦合器** | `include/physics/multiphysics_solver.h` | `src/physics/multiphysics/multiphysics_solver.cu` | ✅ 核心 |

---

## 三、最近攻坚方向（Git历史分析）

最近20个提交集中在 **LPBF粉末床仿真的数值稳定性**：

| 提交 | 关键修复 |
|------|----------|
| `d56b2ad` | **气相热修复**：两阶段气相重置（f<0.01→600K, 0.01-0.05→无扩散缓冲区）消除弹道逃逸伪影 |
| `24a13cd` | **Carman-Kozeny Darcy** + 底部绝热BC |
| `205fb8e` | **反冲力8×放大** + 界面温度审计 |
| `effe5a1` | **力相位掩码**：`F × liquid_fraction` 抑制固相蠕变 |
| `572c43d` | **移除硬冻结** + FLOW-3D方法平滑CSF曲率 |
| `453f414` | **接触角润湿**（10°）+ Darcy气相豁免 |
| `d4d80aa` | **气相解绑**：冻结核 + Darcy跳过气相元胞 |
| `8b16fb6` | **几何光线追踪激光** + 气相热隔离 |

**已修复的关键Bug**：
1. ✅ 气相非物理吸热（相场掩码修复）
2. ✅ 气相LBM弹道逃逸伪影（Gas Wipe）
3. ✅ VOF涂抹导致的蒸发过冷（evap_cooling_factor=0.25）
4. ✅ 固相力泄漏（力相位掩码）

---

## 四、Benchmark对标进展

**已验证的Benchmark**：
| 模块 | Benchmark | 状态 | 误差 |
|------|-----------|------|------|
| Thermal | 3D高斯扩散 | PASS | L2 < 0.1% |
| Thermal | 自然对流 Ra=1e3, 1e4 | PASS | Nu误差 < 7% |
| Thermal + PhaseChange | Stefan问题（ESM） | PASS | 前沿位置 < 5% |
| Fluid | Couette-Poiseuille | PASS | 解析解匹配 |
| Fluid | Taylor-Green涡衰减 | PASS | 2阶收敛 |
| Fluid | Re400方腔驱动流 | PASS | Ghia数据匹配 |
| VOF | Zalesak圆盘（TVD） | PASS | 质量损失0.00013% |
| VOF | Zalesak圆盘（PLIC） | PASS | 保持尖角 |
| VOF | Rayleigh-Taylor不稳定性 | PASS | γ/γ_vis ∈ [0.3, 0.9] |
| VOF | 上升气泡 | PASS | V_terminal合理，质量0.24% |
| Fluid + Thermal | Marangoni 1D回流 | PASS | L2 = 0.99% |
| Fluid + Thermal | Marangoni 2D方腔 (Ma=1000) | PASS (定性) | Nu 88% of ref |
| **全耦合** | **LPBF激光熔化** | **⚠️ 进行中** | 最新提交显示已稳定 |

**当前Benchmark配置**：
- `benchmark_conduction.json`：纯传导+相变的Spot Melting基准（跨平台验证用）
- `apps/benchmark_spot_melt_316L.cu`：传导+Marangoni对流的Spot Melting基准

---

## 五、当前开发阶段判断

**处于阶段**：**Phase 6→7过渡期 — 全耦合LPBF仿真稳定性攻坚**

已完成：
- ✅ 独立物理模块验证（L1-L3层）
- ✅ Marangoni对流耦合（Phase 6）
- ✅ 几何光线追踪激光（Phase 7部分）
- ✅ 粉末床生成与加载
- ✅ 反冲压力与蒸发冷却
- ✅ 气相热修复（最新提交）

进行中：
- 🔄 LPBF粉末床仿真的长期稳定性（5000+步）
- 🔄 跨平台Benchmark验证（对标OpenFOAM）
- 🔄 能量守恒精度优化

---

## 六、接下来1-2个最紧迫的任务

### 任务1：完成LPBF Powder Bed Benchmark验证
- **紧迫性**：高
- **原因**：最新提交已修复气相热问题，现在是验证完整LPBF仿真的最佳时机
- **具体行动**：
  1. 运行`apps/sim_powder_bed_316L.cu`，确认10,000+步无NaN
  2. 验证熔池深度、宽度与文献值对比
  3. 检查能量守恒误差（目标：< 5%）
  4. 生成标准化Benchmark JSON配置（类似`benchmark_conduction.json`）

### 任务2：光线追踪激光在匙孔模式下的验证
- **紧迫性**：中高
- **原因**：几何光线追踪已实现（`8b16fb6`），但缺乏匙孔模式验证
- **具体行动**：
  1. 创建高功率密度测试用例（P>300W, 小光斑）
  2. 验证多反射吸收率与Beer-Lambert对比
  3. 确认匙孔深度与Rosenthal解析解的偏差
  4. 添加到验证测试套件（`tests/validation/`）

### 潜在优化点：
- **VOF涂抹补偿**：当前`evap_cooling_factor=0.25`是经验参数，需理论校准
- **Darcy阻尼极端参数**：全耦合LPBF测试曾因Darcy发散而失败（`C > 10^13`），需进一步调优
- **多GPU扩展**：当前单GPU限制大域计算，MPI并行化是长期需求

---

## 七、架构亮点（值得保留）

1. **力管线设计**：`ForceAccumulator`作为汇聚点，统一处理SI→lattice单位转换，避免各模块不一致
2. **ESM相变方法**：Jiaung (2001)焓源项法，纯金属Stefan问题误差<5%
3. **D3Q7热格子**：`c_s² = 1/4`（非标准1/3），经3D高斯扩散校准验证
4. **配置层级**：`MultiphysicsConfig`采用嵌套结构体+向后兼容的扁平访问器
5. **积木式架构**：每个物理模块可独立实例化、测试、替换

---

## 八、分支管理

### 当前分支结构
```
benchmark/conduction-316L  (原始分支，未改动)
      ↑
opencode/sandbox-20260327  (当前分支，隔离测试)
```

### 退回指令

**随时一键回到原始状态**：
```bash
git checkout benchmark/conduction-316L
```

**如果要完全删除这个沙箱分支**：
```bash
git checkout benchmark/conduction-316L
git branch -D opencode/sandbox-20260327
```

### 分支用途
- 🧪 **沙箱分支** `opencode/sandbox-20260327`：所有opencode的更改都在这里
- 🔒 **原始分支** `benchmark/conduction-316L`：保持不变，随时可切回

---

## 九、关键文件快速参考

### 配置文件
- `benchmark_conduction.json` - 纯传导基准配置
- `CLAUDE.md` - 代码哲学和开发原则
- `CMakeLists.txt` - 构建配置

### 核心应用
- `apps/benchmark_spot_melt_316L.cu` - Spot Melting基准
- `apps/sim_powder_bed_316L.cu` - LPBF粉末床仿真
- `apps/demo_phase6_marangoni.cu` - Marangoni耦合测试

### 文档
- `docs/architecture_whitepaper.md` - 架构白皮书（详细）
- `docs/opencode_handover_20260327.md` - 本交接报告

---

## 总结

项目架构清晰、模块化良好，已完成80%+的核心功能开发。当前处于**全耦合LPBF仿真稳定性攻坚**的最后阶段，最紧迫的任务是完成粉末床Benchmark验证和光线追踪激光的匙孔模式验证。代码质量高，物理模型严谨，已具备生产级仿真的基础。

---

*报告生成时间: 2026-03-27 21:22*  
*分支: opencode/sandbox-20260327*
