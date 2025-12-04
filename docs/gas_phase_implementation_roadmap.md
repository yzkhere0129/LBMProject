# LBM-CUDA 气相建模实现技术路线书

## 1. 项目背景与目标

### 1.1 现状分析

当前LBM-CUDA仿真平台已实现：
- D3Q7热传导LBM求解器
- D3Q19流体动力学LBM求解器
- VOF自由表面追踪
- 激光热源建模
- Marangoni效应（表面张力驱动流动）
- 蒸发冷却（能量守恒层面）

**缺失功能**：
- 气相动力学建模
- 反冲压力对熔池形态的影响
- 蒸发质量损失耦合
- 气孔形成预测
- 深熔/匙孔模式模拟

### 1.2 实现目标

构建完整的气-液-固三相耦合仿真能力，支持：
1. LPBF中的匙孔动力学模拟
2. 气孔缺陷预测
3. 材料飞溅行为
4. 更精确的熔池几何形态

---

## 2. 物理模型

### 2.1 蒸发模型

#### Clausius-Clapeyron 方程（饱和蒸气压）

$$P_{sat}(T) = P_0 \exp\left[-\frac{M_a L_v}{R}\left(\frac{1}{T} - \frac{1}{T_b}\right)\right]$$

参数说明：
| 符号 | 含义 | Ti-6Al-4V典型值 |
|------|------|----------------|
| $P_0$ | 参考压力 | 101325 Pa |
| $M_a$ | 摩尔质量 | 0.0479 kg/mol |
| $L_v$ | 汽化潜热 | 8.9×10⁶ J/kg |
| $R$ | 气体常数 | 8.314 J/(mol·K) |
| $T_b$ | 沸点 | 3560 K |

#### Hertz-Knudsen-Langmuir 蒸发速率

$$\dot{m}'' = \sigma_e \sqrt{\frac{M_a}{2\pi R T}} P_{sat}(T)$$

其中 $\sigma_e$ 为蒸发系数（0.1-1.0）。

### 2.2 反冲压力模型

#### Anisimov模型

$$P_{recoil} = 0.54 \times P_{sat}(T)$$

物理解释：
- 蒸发产生的动量反作用力
- 0.54系数来源于气体动力学分析
- 作用于液-气界面法向

#### 压力分布

$$P_{recoil}(\mathbf{x}) = 0.54 \times P_{sat}(T(\mathbf{x})) \cdot \mathbf{n}$$

其中 $\mathbf{n}$ 为界面法向量（从液相指向气相）。

### 2.3 质量守恒耦合

蒸发导致的质量损失：
$$\frac{\partial \rho}{\partial t} = -\dot{m}'' \cdot |\nabla \phi|$$

其中 $\phi$ 为VOF函数。

---

## 3. 实现架构设计

### 3.1 模块结构

```
src/physics/
├── gas_phase/
│   ├── gas_phase_solver.cuh          # 气相求解器接口
│   ├── gas_phase_solver.cu           # 主实现
│   ├── evaporation_model.cuh         # 蒸发模型接口
│   ├── evaporation_model.cu          # 蒸发计算
│   ├── recoil_pressure.cuh           # 反冲压力接口
│   ├── recoil_pressure.cu            # 反冲压力计算
│   ├── vapor_properties.cuh          # 蒸汽属性
│   └── kernels/
│       ├── evaporation_kernels.cu    # 蒸发CUDA核函数
│       └── recoil_pressure_kernels.cu # 反冲压力核函数
├── multiphysics/
│   └── multiphysics_solver.cu        # 修改：集成气相耦合
└── thermal/
    └── thermal_lbm.cu                # 修改：蒸发热损失
```

### 3.2 类设计

```cpp
// gas_phase_solver.cuh
class GasPhaseSolver {
public:
    struct Config {
        bool enable_recoil_pressure = true;
        bool enable_mass_loss = true;
        float evaporation_coefficient = 0.82f;  // σ_e
        float reference_pressure = 101325.0f;    // P_0
    };

    void initialize(const SimulationConfig& config);
    void computeRecoilPressure(const float* T, const float* vof,
                                float* P_recoil, cudaStream_t stream);
    void computeEvaporationRate(const float* T, const float* vof,
                                 float* mdot, cudaStream_t stream);
    void applyMassLoss(float* vof, const float* mdot,
                       float dt, cudaStream_t stream);

private:
    Config config_;
    MaterialProperties* material_;

    // 设备端数据
    float* d_P_sat_;        // 饱和蒸气压场
    float* d_P_recoil_;     // 反冲压力场
    float* d_evap_rate_;    // 蒸发速率场
};

// evaporation_model.cuh
class EvaporationModel {
public:
    __device__ float computeSaturationPressure(float T);
    __device__ float computeEvaporationRate(float T, float P_sat);
    __device__ float computeRecoilPressure(float P_sat);

private:
    float P0_, Ma_, Lv_, R_, Tb_;
    float sigma_e_;  // 蒸发系数
};
```

### 3.3 CUDA核函数设计

```cpp
// recoil_pressure_kernels.cu
__global__ void computeRecoilPressureKernel(
    const float* __restrict__ T,
    const float* __restrict__ vof,
    const float* __restrict__ vof_gradient_x,
    const float* __restrict__ vof_gradient_y,
    const float* __restrict__ vof_gradient_z,
    float* __restrict__ P_recoil_x,
    float* __restrict__ P_recoil_y,
    float* __restrict__ P_recoil_z,
    int nx, int ny, int nz,
    MaterialProperties mat
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    float vof_val = vof[idx];
    // 只在界面处计算（0.01 < VOF < 0.99）
    if (vof_val < 0.01f || vof_val > 0.99f) {
        P_recoil_x[idx] = 0.0f;
        P_recoil_y[idx] = 0.0f;
        P_recoil_z[idx] = 0.0f;
        return;
    }

    float T_local = T[idx];
    if (T_local < mat.T_melt) {
        P_recoil_x[idx] = 0.0f;
        P_recoil_y[idx] = 0.0f;
        P_recoil_z[idx] = 0.0f;
        return;
    }

    // Clausius-Clapeyron
    float P_sat = mat.P0 * expf(-mat.Ma * mat.Lv / mat.R *
                                 (1.0f/T_local - 1.0f/mat.Tb));

    // Anisimov反冲压力
    float P_rec = 0.54f * P_sat;

    // 计算界面法向
    float gx = vof_gradient_x[idx];
    float gy = vof_gradient_y[idx];
    float gz = vof_gradient_z[idx];
    float mag = sqrtf(gx*gx + gy*gy + gz*gz) + 1e-10f;

    // 反冲压力沿法向方向（指向液相内部）
    P_recoil_x[idx] = -P_rec * gx / mag;
    P_recoil_y[idx] = -P_rec * gy / mag;
    P_recoil_z[idx] = -P_rec * gz / mag;
}
```

---

## 4. 分阶段实施计划

### 阶段一：反冲压力边界条件（优先级：高）

**目标**：实现反冲压力对熔池形态的影响

**工作内容**：
1. 实现 `EvaporationModel` 类
   - Clausius-Clapeyron饱和蒸气压计算
   - Anisimov反冲压力模型
2. 实现 `computeRecoilPressureKernel`
3. 修改 `FluidSolver` 添加压力边界条件接口
4. 集成到 `MultiphysicsSolver` 时间推进循环

**关键代码修改**：
- `src/physics/fluid/fluid_lbm.cu`: 添加外部压力源项
- `src/physics/multiphysics/multiphysics_solver.cu`: 调用反冲压力计算

**验证测试**：
- 静态液滴蒸发测试
- 匙孔深度vs激光功率关系验证
- 与文献数据对比

**预计工作量**：1-2周

### 阶段二：蒸发质量耦合（优先级：中）

**目标**：实现蒸发导致的质量损失

**工作内容**：
1. 实现 Hertz-Knudsen-Langmuir 蒸发速率计算
2. 实现VOF场质量损失更新
3. 蒸发热损失耦合到热求解器

**关键代码修改**：
- `src/physics/vof/vof_solver.cu`: 添加质量损失项
- `src/physics/thermal/thermal_lbm.cu`: 蒸发热损失源项

**验证测试**：
- 质量守恒验证
- 蒸发速率vs温度验证
- 能量平衡验证

**预计工作量**：2-3周

### 阶段三：完整气相LBM（优先级：低，可选）

**目标**：添加气相流场求解

**工作内容**：
1. 实现D3Q19气相LBM求解器
2. 气-液界面耦合
3. 蒸汽羽流动力学

**关键考虑**：
- 气相密度比液相低3-4个数量级
- 需要特殊数值处理避免不稳定
- 计算成本显著增加

**预计工作量**：4-6周

---

## 5. 接口设计

### 5.1 配置文件扩展

```ini
# Gas Phase Configuration
[gas_phase]
enable_gas_phase = true
enable_recoil_pressure = true
enable_mass_loss = true

# Evaporation parameters
evaporation_coefficient = 0.82
recoil_pressure_factor = 0.54

# Optional: Full gas phase LBM
enable_gas_lbm = false
gas_viscosity = 1.0e-5
gas_density = 1.0
```

### 5.2 MultiphysicsSolver集成

```cpp
// multiphysics_solver.cu 修改
void MultiphysicsSolver::step() {
    // 1. 热场求解
    thermal_solver_->step();

    // 2. 气相计算（新增）
    if (config_.enable_gas_phase) {
        gas_phase_solver_->computeRecoilPressure(
            thermal_solver_->getTemperature(),
            vof_solver_->getVOF(),
            recoil_pressure_,
            stream_
        );

        if (config_.enable_mass_loss) {
            gas_phase_solver_->computeEvaporationRate(
                thermal_solver_->getTemperature(),
                vof_solver_->getVOF(),
                evaporation_rate_,
                stream_
            );
        }
    }

    // 3. 流场求解（包含反冲压力）
    fluid_solver_->setExternalPressure(recoil_pressure_);
    fluid_solver_->step();

    // 4. VOF更新（包含质量损失）
    vof_solver_->step();
    if (config_.enable_mass_loss) {
        gas_phase_solver_->applyMassLoss(
            vof_solver_->getVOF(),
            evaporation_rate_,
            dt_,
            stream_
        );
    }
}
```

---

## 6. 验证与测试计划

### 6.1 单元测试

| 测试项 | 验证内容 | 预期结果 |
|--------|----------|----------|
| `test_clausius_clapeyron` | 饱和蒸气压计算 | 与查表数据误差<1% |
| `test_evaporation_rate` | HKL蒸发速率 | 与解析解匹配 |
| `test_recoil_pressure` | 反冲压力量级 | Ti-6Al-4V@3500K: ~100 kPa |

### 6.2 集成测试

| 测试案例 | 验证目标 | 参考数据 |
|----------|----------|----------|
| 静态液滴蒸发 | 质量守恒、能量守恒 | 解析解对比 |
| 单点激光熔化 | 匙孔形成阈值 | P>150W for Ti-6Al-4V |
| 扫描激光 | 匙孔深度vs功率 | Cunningham 2019 X-ray |

### 6.3 验证标准

1. **物理合理性**
   - 反冲压力随温度指数增长
   - 匙孔深度随功率增加
   - 蒸发速率在沸点附近急剧增加

2. **数值稳定性**
   - 长时间仿真无发散
   - CFL条件满足
   - 能量守恒误差<5%

3. **文献对比**
   - 匙孔深宽比
   - 临界功率密度
   - 熔池几何形态

---

## 7. 参考文献

1. Anisimov, S. I., & Khokhlov, V. A. (1995). *Instabilities in Laser-Matter Interaction*. CRC Press.

2. Khairallah, S. A., et al. (2016). "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." *Acta Materialia*, 108, 36-45.

3. Cunningham, R., et al. (2019). "Keyhole threshold and morphology in laser melting revealed by ultrahigh-speed x-ray imaging." *Science*, 363(6429), 849-852.

4. Tan, W., et al. (2013). "Investigation of keyhole plume and molten pool based on a three-dimensional dynamic model with sharp interface formulation." *Journal of Physics D: Applied Physics*, 46(5), 055501.

5. Körner, C., et al. (2011). "Mesoscopic simulation of selective beam melting processes." *Journal of Materials Processing Technology*, 211(6), 978-987.

---

## 8. 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 反冲压力导致数值不稳定 | 仿真发散 | 压力限幅、自适应时间步 |
| 气液密度比过大 | 计算精度下降 | 伪压缩性方法 |
| 蒸发速率计算误差 | 物理不准确 | 与实验数据校准 |
| 性能下降 | 仿真变慢 | CUDA流并行优化 |

---

## 9. 里程碑与时间表

| 里程碑 | 内容 | 预计完成 |
|--------|------|----------|
| M1 | 反冲压力模型实现与测试 | 第2周 |
| M2 | MultiphysicsSolver集成 | 第3周 |
| M3 | 蒸发质量耦合实现 | 第5周 |
| M4 | 验证测试与文献对比 | 第6周 |
| M5 | 文档与代码审查 | 第7周 |

---

*文档版本: 1.0*
*创建日期: 2025-11-21*
*作者: LBM-CUDA Development Team*
