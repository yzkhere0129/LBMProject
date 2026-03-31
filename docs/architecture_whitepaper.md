# LBM 求解器架构白皮书

> 基于代码库物理现状的独立探索，2026-03-12
> 代码版本: `05249e2` (master)

---

## 总览

本项目是面向金属增材制造 (LPBF) 的 CUDA LBM 求解器，包含 ~10,300 行核心代码（45 个源文件/头文件），采用五层无循环依赖架构：

```
L0: CUDA Primitives (CudaBuffer, error check)
 └→ L1: Lattice Infrastructure (D3Q19, D3Q7, BCs)
     └→ L2: Leaf Physics (FluidLBM, ThermalLBM, VOFSolver, PhaseChangeSolver, ...)
         └→ L3: Force Pipeline (ForceAccumulator)
             └→ L4: Coupling Orchestration (MultiphysicsSolver)
                 └→ L5: I/O & Diagnostics (VTK, FieldRegistry, EnergyBalance)
```

所有分布函数采用 **SoA 布局**: `f[q * num_cells + cell_idx]`，保证 GPU warp 内合并访存。
全局编译方式为 **CUDA 分离编译 (`-rdc=true`)**，因此所有设备常量使用 `__device__` 而非 `__constant__`。

---

## 子系统 1: D3Q19 流体求解器

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/fluid_lbm.h` | 641 | 类接口 |
| `src/physics/fluid/fluid_lbm.cu` | 2004 | 碰撞/流动/宏观量内核 |
| `include/core/lattice_d3q19.h` | 120 | 格子常量 + 平衡态 |
| `src/core/lattice/d3q19.cu` | 120 | 设备端初始化 |
| `include/core/collision_bgk.h` | 80 | BGK 碰撞接口 |
| `include/core/streaming.h` | 60 | 流动方向定义 |
| `include/core/boundary_conditions.h` | 180 | 边界节点结构 |
| `src/core/boundary/boundary_conditions.cu` | 190 | Bounce-back, Zou-He 实现 |

### 核心算法

| 算法 | 内核函数 | 说明 |
|------|---------|------|
| **BGK 碰撞** | `fluidBGKCollisionKernel` | $f_i^{out} = f_i - \omega(f_i - f_i^{eq}) + F_i^{Guo}$ |
| **TRT 碰撞** | `fluidTRTCollisionKernel` | 对称/反对称分裂, $\Lambda = 3/16$ (magic parameter) |
| **变黏度 TRT** | `fluidTRTCollisionVariableOmegaKernel` | 逐格点 $\omega$ 场（两相流） |
| **Push 流动** | `fluidStreamingKernelWithWalls` | 混合边界 (PERIODIC / WALL / VELOCITY) |
| **宏观量** | `computeMacroscopicKernel` | $\rho = \sum f_i$, $\mathbf{u} = \sum \mathbf{c}_i f_i / \rho$ |
| **Guo 修正** | `computeMacroscopicWithForceKernel` | $\mathbf{u} = \sum \mathbf{c}_i f_i / \rho + 0.5 \mathbf{F}/\rho$ |
| **浮力** | `computeBuoyancyForceKernel` | Boussinesq: $\mathbf{F} = \rho_0 \beta \Delta T \mathbf{g}$ |
| **Darcy 阻尼** | `applyDarcyDampingKernel` | $\mathbf{F}_{darcy} = -C \frac{(1-f_l)^2}{f_l^3 + \varepsilon} \mathbf{u}$ |
| **力补偿** | `compensateForceForOmegaKernel` | $\mathbf{F} \times 2/(2-\omega)$ 消除松弛时间依赖 |

力项格式: **Guo forcing scheme** — 碰撞时添加 $S_i = (1 - 1/(2\tau)) w_i \left[ \frac{(\mathbf{c}_i - \mathbf{u})}{c_s^2} + \frac{(\mathbf{c}_i \cdot \mathbf{u})}{c_s^4} \mathbf{c}_i \right] \cdot \mathbf{F}$

### 设备内存布局

```
d_f_src[19 * NC]    // 当前时步分布函数 (SoA)
d_f_dst[19 * NC]    // 流动后分布函数
d_rho[NC]           // 密度场
d_ux[NC], d_uy[NC], d_uz[NC]  // 速度场
d_pressure[NC]      // p = cs²(ρ - ρ₀)
d_omega_field_[NC]  // 逐格点松弛参数（变黏度模式）
```

### 数据输入与输出

| 方向 | 场 | 来源/去向 |
|------|---|----------|
| **读取** | `d_force_x/y/z` | ← ForceAccumulator (浮力 + Darcy + 表面张力 + Marangoni + 反冲) |
| **读取** | `d_temperature` | ← ThermalLBM (浮力计算用) |
| **读取** | `d_liquid_fraction` | ← PhaseChangeSolver (Darcy 阻尼用) |
| **读取** | `d_fill_level` | ← VOFSolver (变黏度用) |
| **输出** | `d_ux, d_uy, d_uz` | → ThermalLBM (对流项), VOFSolver (界面推进), ForceAccumulator (CFL限制) |
| **输出** | `d_rho` | → 状态方程压力 |

### 与其他模块的耦合节点

- **→ ThermalLBM**: 提供速度场 `(ux, uy, uz)` 用于热对流
- **→ VOFSolver**: 提供速度场用于 fill level 推进
- **→ ForceAccumulator**: 提供速度场用于 CFL 限制和 Darcy 阻尼
- **← ForceAccumulator**: 接收合力数组 `(d_force_x/y/z)` 注入碰撞步

---

## 子系统 2: D3Q7 热传导求解器

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/thermal_lbm.h` | 590 | 类接口 + 内核前向声明 |
| `src/physics/thermal/thermal_lbm.cu` | 2499 | 碰撞/流动/相变/蒸发内核 |
| `include/physics/lattice_d3q7.h` | 90 | 格子常量 ($c_s^2 = 1/4$) |
| `src/physics/thermal/lattice_d3q7.cu` | 80 | 设备端初始化 |

### 核心算法

| 算法 | 内核函数 | 说明 |
|------|---------|------|
| **热BGK碰撞** | `thermalBGKCollisionKernel` | $g_i^{out} = g_i - \omega_T(g_i - g_i^{eq})$, 可选 apparent Cp |
| **热流动** | `thermalStreamingKernel` | Push-based, 带辐射边界处理 |
| **温度计算** | `computeTemperatureKernel` | $T = \sum g_i$ |
| **Dirichlet BC** | `applyConstantTemperatureBoundary` | 逐面指定温度 |
| **绝热 BC** | `applyAdiabaticBoundary` | Bounce-back (零通量) |
| **对流 BC** | `applyConvectiveBCKernel` | Newton 冷却: $q = h(T - T_\infty)$ |
| **辐射 BC** | `applyRadiationBCFaceKernel` | Stefan-Boltzmann: $q = \varepsilon \sigma (T^4 - T_{amb}^4)$ |
| **体积热源** | `addHeatSourceKernel` | $g_i \mathrel{+}= w_i \cdot \dot{q} \cdot dt / (\rho c_p)$ |
| **焓源项** | `enthalpySourceTermKernel` | **Jiaung (2001)**: $H = c_p T^* + f_l^{old} L$ → 解码 $(T_{new}, f_l^{new})$ → $g_i \mathrel{+}= w_i \Delta T$ |
| **蒸发质量通量** | `computeEvaporationMassFluxKernel` | Hertz-Knudsen-Langmuir 模型 |

**关键设计**: $c_s^2 = 1/4$（非标准 $1/3$），经 3D 高斯扩散校准验证，L2 误差 0.09% (vs $c_s^2 = 1/3$ 的 0.64%)。
因此 $\tau = 4\alpha_{LU} + 0.5$（不是 $3\alpha + 0.5$）。

### 设备内存布局

```
d_g_src[7 * NC]     // 热分布函数 (SoA)
d_g_dst[7 * NC]     // 流动后
d_temperature[NC]   // 温度场
phase_solver_       // → PhaseChangeSolver (optional, unique_ptr)
```

### 数据输入与输出

| 方向 | 场 | 来源/去向 |
|------|---|----------|
| **读取** | `d_ux, d_uy, d_uz` | ← FluidLBM (对流速度) |
| **读取** | `d_heat_source` | ← LaserSource (激光能量) |
| **读取** | `d_fill_level` | ← VOFSolver (蒸发仅在界面) |
| **输出** | `d_temperature` | → FluidLBM (浮力), ForceAccumulator (Marangoni), VOFSolver (蒸发质量通量) |
| **输出** | `d_liquid_fraction` | → FluidLBM (Darcy 阻尼), VOFSolver (凝固收缩) |
| **输出** | `d_evap_mass_flux` | → VOFSolver (界面质量损失) |

### 与其他模块的耦合节点

- **← FluidLBM**: 接收速度场用于对流项 $g_i^{eq} = w_i T (1 + \mathbf{c}_i \cdot \mathbf{u} / c_s^2)$
- **← LaserSource**: 接收体积热源
- **→ PhaseChangeSolver**: 内置耦合 — `computeTemperature()` 后自动调用 `enthalpySourceTermKernel`
- **→ ForceAccumulator**: 温度场用于浮力 + Marangoni 力计算
- **→ VOFSolver**: 蒸发质量通量用于界面退缩

---

## 子系统 3: VOF 自由面追踪

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/vof_solver.h` | 562 | 类接口 + 枚举定义 |
| `src/physics/vof/vof_solver.cu` | 2970 | 推进/重构/曲率/质量守恒内核 |

### 核心算法

| 算法 | 内核函数 | 说明 |
|------|---------|------|
| **一阶迎风** | `advectFillLevelUpwindKernel` | 稳定但耗散 (~0.5% 质量误差) |
| **二阶 TVD** | `advectFillLevelTVDKernel` | 通量限制器: MINMOD / VAN_LEER / **MC** / SUPERBEE |
| **几何 PLIC** | `computeYoungsNormalsKernel` → `computeAlphaInversionKernel` → `advectPLICKernel` | Youngs 法向 + Scardovelli-Zaleski α 反演 + Strang 分裂 |
| **界面重构** | `reconstructInterfaceKernel` | 中心差分 $\nabla f$ → 单位法向 |
| **曲率** | `computeCurvatureKernel` | $\kappa = \nabla \cdot \hat{n}$ (Height-function or divergence) |
| **格元标记** | `convertCellsKernel` | GAS ($f < \varepsilon$) / INTERFACE / LIQUID ($f > 1-\varepsilon$) / OBSTACLE |
| **界面压缩** | `applyInterfaceCompressionKernel` | Olsson-Kreiss: $\nabla \cdot (\varepsilon f(1-f) \hat{n})$ |
| **全局质量修正** | `enforceGlobalMassConservationKernel` | $f_i \mathrel{\times}= M_{target} / M_{current}$ |
| **蒸发质量损失** | `applyEvaporationMassLossKernel` | $\Delta f = -dt \cdot J_{evap} / (\rho \cdot dx)$ |
| **凝固收缩** | `applySolidificationShrinkageKernel` | $\Delta f = \beta \cdot (df_l/dt) \cdot dt$ |
| **接触角** | `applyContactAngleBoundaryKernel` | 壁面润湿角 |

### 设备内存布局

```
d_fill_level_[NC]          // f ∈ [0, 1]
d_fill_level_tmp_[NC]      // 推进中间态
d_cell_flags_[NC]          // CellFlag 枚举
d_interface_normal_[NC]    // float3 法向
d_curvature_[NC]           // κ [1/m]
// PLIC 专用:
plic_nx_, plic_ny_, plic_nz_  // PLIC 法向分量
plic_alpha_[NC]            // 截距
plic_flux_[NC]             // 几何通量
plic_face_vel_[NC]         // 面速度
```

### 数据输入与输出

| 方向 | 场 | 来源/去向 |
|------|---|----------|
| **读取** | `d_ux, d_uy, d_uz` | ← FluidLBM (界面推进速度) |
| **读取** | `d_evap_mass_flux` | ← ThermalLBM (蒸发导致的质量损失) |
| **读取** | `d_dfl_dt` | ← PhaseChangeSolver (凝固收缩率) |
| **输出** | `d_fill_level` | → FluidLBM (变黏度), ForceAccumulator (表面力) |
| **输出** | `d_curvature` | → ForceAccumulator (表面张力力 CSF) |
| **输出** | `d_interface_normal` | → ForceAccumulator (Marangoni + 反冲方向) |
| **输出** | `d_cell_flags` | → 可视化、诊断 |

### 与其他模块的耦合节点

- **← FluidLBM**: 速度场驱动 fill level 推进
- **← ThermalLBM**: 蒸发质量通量减少界面处 fill level
- **← PhaseChangeSolver**: 凝固收缩率调整 fill level
- **→ ForceAccumulator**: 提供曲率 + 法向 + fill level 梯度用于表面力
- **→ FluidLBM**: fill level 用于变密度/变黏度两相流

---

## 子系统 4: 相变求解器

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/phase_change.h` | 335 | 类接口 + 焓方法 |
| `src/physics/phase_change/phase_change.cu` | 536 | Newton-Raphson 求解内核 |

### 核心算法

| 算法 | 内核函数 | 说明 |
|------|---------|------|
| **焓→温度** | `solveTemperatureFromEnthalpyKernel` | Newton-Raphson: $H = \rho c_p T + f_l(T) \rho L$ 隐式求解 $T(H)$ |
| **温度→液相率** | `updateLiquidFractionKernel` | 线性插值: $f_l = (T - T_s)/(T_l - T_s)$ 在糊状区 |
| **焓源项** (ESM) | `enthalpySourceTermKernel` (在 thermal_lbm.cu 中) | **Jiaung (2001)**: 碰撞后 $T^* = \sum g_i$ → 焓守恒解码 $(T_{new}, f_l^{new})$ → 修正分布 |
| **液相率变化率** | `computeLiquidFractionRateKernel` | $df_l/dt = (f_l^{n} - f_l^{n-1}) / dt$ |
| **总能量** | `computeTotalEnergyKernel` | 并行归约 $\sum H_i \cdot V_{cell}$ |

**两种相变方法现存**:
1. **ESM (Enthalpy Source Term, Jiaung 2001)** — 当前激活的方法。在 `ThermalLBM::computeTemperature()` 后执行。纯金属 Stefan 问题误差 < 5%
2. **Apparent Heat Capacity** — `thermalBGKCollisionKernel` 中可选。当 ESM 激活时自动禁用以防双重计算

### 设备内存布局

```
d_enthalpy[NC]              // H [J/m³]
d_liquid_fraction[NC]       // fl ∈ [0, 1]
d_liquid_fraction_prev_[NC] // fl(t-dt) 用于速率和 ESM
d_dfl_dt_[NC]               // dfl/dt [1/s]
```

### 数据输入与输出

| 方向 | 场 | 来源/去向 |
|------|---|----------|
| **读取** | `d_temperature` | ← ThermalLBM ($T^*$ 碰撞后温度) |
| **读取** | `MaterialProperties` | ← 材料数据库 ($T_s, T_l, L, c_p, \rho$) |
| **输出** | `d_liquid_fraction` | → FluidLBM (Darcy), VOFSolver (凝固收缩) |
| **输出** | `d_temperature` (修正后) | → 覆写 ThermalLBM 的温度场 |
| **输出** | `d_dfl_dt` | → VOFSolver (凝固收缩源项) |
| **输出 (分布修正)** | `g[q]` | → 直接修改 ThermalLBM 的分布函数: $g_i \mathrel{+}= w_i \Delta T$ |

### 与其他模块的耦合节点

- **⊂ ThermalLBM**: 作为 ThermalLBM 的内置子组件 (`unique_ptr<PhaseChangeSolver> phase_solver_`)
- **→ FluidLBM**: 液相率用于 Darcy 阻尼（糊状区动量衰减）
- **→ VOFSolver**: 液相率变化率用于凝固收缩

---

## 子系统 5: 力管线

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/force_accumulator.h` | 277 | 力累加管线接口 |
| `src/physics/force_accumulator.cu` | ~500 | 各物理力内核 |
| `include/physics/surface_tension.h` + `.cu` | 80 + 317 | CSF 表面张力 |
| `include/physics/marangoni.h` + `.cu` | 70 + 380 | 热毛细力 |
| `include/physics/recoil_pressure.h` + `.cu` | 60 + 602 | 蒸发反冲压力 |

### 执行管线（严格顺序）

```
ForceAccumulator::reset()                    // d_fx = d_fy = d_fz = 0
    ↓
addBuoyancyForce(T, T_ref, β, ρ, g)         // += ρ₀βΔTg
    ↓
addDarcyDamping(fl, ux, uy, uz, C, ρ)       // -= C(1-fl)²/(fl³+ε)·u
    ↓
addSurfaceTensionForce(κ, f, σ)             // += σ·κ·∇f (CSF)
    ↓
addMarangoniForce(T, f, n, dσ/dT)           // += (dσ/dT)·∇ₛT·|∇f|
    ↓
addRecoilPressureForce(T, f, n, T_boil, Lv, M)  // += P_sat(T)·n (Clausius-Clapeyron)
    ↓
convertToLatticeUnits(dx, dt, ρ)             // SI → lattice: F_LU = F_phys·dt²/(dx·ρ)
    ↓
applyCFLLimiting(ux, uy, uz, dx, dt)        // 限制力导致的速度增量 < Ma_target
```

### 数据输入与输出

| 方向 | 场 | 来源 |
|------|---|------|
| **读取** | `d_temperature` | ← ThermalLBM |
| **读取** | `d_liquid_fraction` | ← PhaseChangeSolver |
| **读取** | `d_fill_level, d_curvature, d_interface_normal` | ← VOFSolver |
| **读取** | `d_ux, d_uy, d_uz` | ← FluidLBM (CFL + Darcy) |
| **输出** | `d_force_x, d_force_y, d_force_z` | → FluidLBM 碰撞步 (Guo forcing) |

### 耦合特点

ForceAccumulator 是所有物理模块的 **汇聚节点**（convergence point）：
- 它读取来自 Thermal、VOF、PhaseChange、Fluid 四个模块的场
- 它仅输出到 FluidLBM 一个模块
- 它负责 SI → lattice 单位转换，避免各模块自行转换导致不一致

---

## 子系统 6: 材料数据库

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/material_properties.h` | 440 | `MaterialProperties` 结构体 + 温度依赖方法 |
| `src/physics/materials/material_database.cu` | 278 | 5 种预置材料工厂方法 |

### 材料属性

```cpp
struct MaterialProperties {
    // 固/液态热物性
    float rho_solid, rho_liquid;         // 密度 [kg/m³]
    float cp_solid, cp_liquid;           // 比热 [J/(kg·K)]
    float k_solid, k_liquid;             // 导热系数 [W/(m·K)]
    float mu_liquid;                     // 动力黏度 [Pa·s]

    // 相变温度与潜热
    float T_solidus, T_liquidus;         // 固/液相线 [K]
    float T_vaporization;                // 沸点 [K]
    float L_fusion, L_vaporization;      // 潜热 [J/kg]
    float molar_mass;                    // 摩尔质量 [kg/mol]

    // 表面效应
    float surface_tension, dsigma_dT;    // 表面张力 + 温度系数

    // 光学/辐射
    float absorptivity_solid/liquid;     // 激光吸收率
    float emissivity;                    // 发射率
};
```

**预置材料**: Ti6Al4V, 316L, Inconel 718, AlSi10Mg, Steel (纯铁)

**温度依赖方法**: `getDensity(T)`, `getSpecificHeat(T)`, `getThermalConductivity(T)`, `liquidFraction(T)`, `getApparentHeatCapacity(T)`

### 耦合方式

材料属性通过 **值拷贝** 传入各内核（作为 kernel 参数或 `__device__` 变量），不通过指针引用。
- → ThermalLBM: 构造时传入 `MaterialProperties`
- → PhaseChangeSolver: 构造时传入
- → ForceAccumulator: 各力函数参数中传入 σ, dσ/dT, β 等

---

## 子系统 7: 激光热源

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/laser_source.h` | 307 | LaserSource 类 + 扫描路径 |
| `src/physics/laser/laser_source.cu` | 256 | 体积热源内核 |

### 核心算法

- **高斯光束**: $I(r) = \frac{2P}{\pi r_0^2} \exp\left(-\frac{2r^2}{r_0^2}\right)$
- **Beer-Lambert 吸收**: $\dot{q}(z) = \eta \cdot I(r) \cdot \frac{1}{d_p} \exp(-z/d_p)$
- **扫描路径**: `LinearScan` (直线), `RasterScan` (光栅, 带 hatch spacing)

### 数据输入与输出

| 方向 | 场 | 说明 |
|------|---|------|
| **读取** | 光斑位置 $(x_0, y_0)$ | 每步更新: $x_0 \mathrel{+}= v_x \cdot dt$ |
| **读取** | `d_fill_level` | ← VOFSolver (仅在材料区施加热源) |
| **输出** | `d_heat_source[NC]` | → ThermalLBM (`addHeatSource`) |

---

## 子系统 8: 粉末床生成

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/powder_bed.h` | 491 | 粒子结构 + 配置 + 生成器 |
| `src/physics/powder/powder_bed.cu` | 682 | 随机填充 + 沉降 + 热物性 |

### 核心算法

- **粒子生成**: RANDOM_SEQUENTIAL (无重叠随机放置), RAIN_DEPOSITION (模拟沉降), REGULAR_PERTURBED (规则+扰动)
- **粒径分布**: 对数正态 $D_{50}, \sigma_g, D_{min}, D_{max}$
- **热初始化**: Zehner-Bauer-Schlünder 有效导热系数 (粉末层)
- **VOF 初始化**: 将粒子几何映射到 fill level 场

### 数据输入与输出

| 方向 | 场 | 说明 |
|------|---|------|
| **输出** | `d_fill_level` | → VOFSolver (初始界面几何) |
| **输出** | `d_k_effective` | → ThermalLBM (粉末层等效导热) |
| **输出** | 粒子列表 `particles_[]` | 主机端统计 |

---

## 子系统 9: 耦合编排器 (MultiphysicsSolver)

### 包含的核心文件

| 文件 | 行数 | 角色 |
|------|------|------|
| `include/physics/multiphysics_solver.h` | 878 | 配置层级 + 编排接口 |
| `src/physics/multiphysics/multiphysics_solver.cu` | 2764 | 耦合时步 + 能量诊断 |

### 配置层级

```
MultiphysicsConfig
├── DomainConfig        {nx, ny, nz, dx}
├── PhysicsFlags        {enable_thermal, enable_fluid, enable_vof, ...}
├── NumericalConfig     {dt, vof_subcycles, cfl_limit, mass_correction}
├── FluidConfig         {kinematic_viscosity, density, darcy_coeff}
├── ThermalConfig       {thermal_diffusivity, substrate_cooling}
├── SurfaceConfig       {surface_tension, dsigma_dT, recoil, molar_mass}
├── BuoyancyConfig      {beta, gravity_vec, T_ref}
├── LaserConfig         {power, spot_radius, absorptivity, scan_path}
├── PhaseChangeConfig   {newton_tolerance, max_iterations}
├── FaceBoundaryConfig  {per-face BC type for 6 faces × fluid + thermal}
└── MaterialProperties  {Ti6Al4V default, or custom}
```

### 时步执行顺序

```
step(dt)
│
├─ 1. LASER → d_heat_source
│
├─ 2. THERMAL
│   ├─ addHeatSource(d_heat_source)
│   ├─ collisionBGK(d_ux, d_uy, d_uz)
│   ├─ streaming()
│   ├─ applyBoundaryConditions()
│   ├─ computeTemperature()         // 内含 ESM 相变修正
│   ├─ applyRadiationBC()
│   └─ applySubstrateCoolingBC()
│
├─ 3. VOF (with subcycling, default 10x)
│   ├─ advectFillLevel(d_ux, d_uy, d_uz, dt_sub)  × n_sub
│   ├─ reconstructInterface()
│   ├─ computeCurvature()
│   ├─ convertCells()
│   ├─ applyEvaporationMassLoss()
│   └─ enforceGlobalMassConservation()
│
├─ 4. FORCES
│   ├─ ForceAccumulator::reset()
│   ├─ += buoyancy(d_temperature)
│   ├─ += darcy(d_liquid_fraction, d_ux, d_uy, d_uz)
│   ├─ += surface_tension(d_curvature, d_fill_level)
│   ├─ += marangoni(d_temperature, d_fill_level, d_normals)
│   ├─ += recoil(d_temperature, d_fill_level, d_normals)
│   ├─ convertToLatticeUnits()
│   └─ applyCFLLimiting()
│
└─ 5. FLUID
    ├─ collisionTRT(d_force_x, d_force_y, d_force_z)
    ├─ streaming()
    ├─ applyBoundaryConditions()
    └─ computeMacroscopic(d_force_x, d_force_y, d_force_z)  // Guo 修正
```

### 数据流全景图

```
                    ┌──────────┐
                    │  Laser   │
                    │  Source  │
                    └────┬─────┘
                         │ d_heat_source
                         ▼
┌──────────┐    d_ux   ┌──────────┐   d_temperature   ┌───────────────┐
│          │ ────────→ │ Thermal  │ ─────────────────→ │    Force      │
│  Fluid   │           │   LBM    │                    │ Accumulator   │
│   LBM    │           │ + ESM    │   d_liquid_frac    │               │
│ (D3Q19)  │           │ (D3Q7)   │ ─────────────────→ │ buoyancy      │
│          │           └──────────┘                    │ darcy         │
│          │    d_ux   ┌──────────┐   d_fill_level     │ surf.tension  │
│          │ ────────→ │   VOF    │ ─────────────────→ │ marangoni     │
│          │           │  Solver  │   d_curvature      │ recoil        │
│          │           │          │ ─────────────────→ │               │
│          │           └──────────┘   d_normals        │               │
│          │                        ─────────────────→ │               │
│          │  d_force_x/y/z                            │               │
│          │ ◄──────────────────────────────────────── │               │
└──────────┘                                           └───────────────┘
```

---

## 子系统 10: I/O 与诊断

### 包含的核心文件

| 文件 | 角色 |
|------|------|
| `include/io/vtk_writer.h` + `src/io/vtk_writer.cu` | VTK XML 非结构网格输出 |
| `include/io/field_registry.h` | 可配置输出场注册 |
| `include/diagnostics/energy_balance.h` + `.cu` | 能量守恒追踪 |
| `src/config/simulation_config.cpp` | 运行时配置文件加载 |
| `include/config/lpbf_config_loader.h` | LPBF 参数加载器 |

### 诊断能力

```
EnergyBalanceTracker:
  E_thermal    = Σ ρ·cp·T·V
  E_latent     = Σ fl·ρ·L·V
  P_laser      = 激光功率 × 吸收率
  P_evap       = Σ J_evap · L_vap · dA
  P_radiation  = Σ ε·σ·(T⁴ - T_amb⁴) · dA
  P_substrate  = Σ h·(T - T_sub) · dA
  balance_err  = |dE/dt - (P_in - P_out)| / P_in
```

---

## 已验证的 Benchmark 状态

| 模块 | Benchmark | 状态 | 误差 |
|------|-----------|------|------|
| Thermal | 3D 高斯扩散 | PASS | L2 < 0.1% |
| Thermal | 自然对流 Ra=1e3, 1e4 | PASS | Nu 误差 < 7% |
| Thermal + PhaseChange | Stefan 问题 (ESM, 纯金属) | PASS | 前沿位置 < 5% |
| Fluid | Couette-Poiseuille | PASS | 解析解匹配 |
| Fluid | Taylor-Green 涡衰减 | PASS | 2阶收敛 |
| Fluid | Re400 方腔驱动流 | PASS | Ghia 数据匹配 |
| VOF | Zalesak 圆盘 (TVD) | PASS | 质量损失 0.00013% |
| VOF | Zalesak 圆盘 (PLIC) | PASS | 保持尖角 |
| VOF | Rayleigh-Taylor 不稳定性 | PASS | γ/γ_vis ∈ [0.3, 0.9] |
| VOF | 上升气泡 | PASS | V_terminal 合理，质量 0.24% |
| Fluid + Thermal | Marangoni 1D 回流 | PASS | L2 = 0.99% |
| Fluid + Thermal | Marangoni 2D 方腔 (Ma=1000) | PASS (定性) | Nu 88% of ref |
| 全耦合 | LPBF 激光熔化 | FAIL | NaN (Darcy 阻尼过大) |

---

## 架构限制（当前已知）

1. **单 GPU**: 无 MPI / multi-GPU 支持
2. **无 AMR**: 均匀网格，大域计算昂贵
3. **无光线追踪激光**: Beer-Lambert 仅支持传导模式，不支持匙孔
4. **Push-based 流动**: FluidLBM 库使用 push streaming (viz_marangoni_cavity 自建 pull kernel)
5. **顺序耦合**: Thermal → VOF → Force → Fluid 无迭代子循环
6. **全场 Darcy**: 极端参数 ($C > 10^{13}$) 导致数值发散（已知 LPBF 测试失败根因）
