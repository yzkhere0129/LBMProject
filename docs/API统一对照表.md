# API统一对照表

**用途**：快速查找现有API与run_simulation.cu期望API的对应关系

---

## 1. ThermalLBM

| run_simulation期望 | 现有API | 解决方案 | 状态 |
|-------------------|---------|----------|------|
| `thermal->collision()` | `thermal->collisionBGK()` | 添加便捷方法 | 阶段1 |
| `thermal->update()` | `collisionBGK()` + `streaming()` | 添加`timestep()`方法 | 阶段1 |
| `thermal->getTemperature()` | ✅ 存在 | 无需修改 | ✅ |
| `thermal->getLiquidFraction()` | ✅ 存在 | 无需修改 | ✅ |
| `thermal->addHeatSource()` | ✅ 存在 | 无需修改 | ✅ |

**实现代码**：
```cpp
// include/physics/thermal_lbm.h
void collision();  // 新增
void timestep(const float* ux = nullptr, ...);  // 新增

// src/physics/thermal_lbm.cu
void ThermalLBM::collision() {
    collisionBGK(nullptr, nullptr, nullptr);
}

void ThermalLBM::timestep(const float* ux, const float* uy, const float* uz) {
    collisionBGK(ux, uy, uz);
    streaming();
    computeTemperature();
}
```

---

## 2. FluidLBM

| run_simulation期望 | 现有API | 解决方案 | 状态 |
|-------------------|---------|----------|------|
| `fluid->collision()` | `fluid->collisionBGK(0,0,0)` | 添加便捷方法 | 阶段2 |
| `fluid->collisionWithForce(fx,fy,fz)` | `fluid->collisionBGK(fx,fy,fz)` | 添加别名方法 | 阶段2 |
| `fluid->applyDarcyDamping(fl, C, dt)` | `applyDarcyDamping(fl, C, fx, fy, fz)` | 添加重载版本 | 阶段2 |
| `fluid->getVelocityX/Y/Z()` | ✅ 存在 | 无需修改 | ✅ |
| `fluid->computeMacroscopic()` | ✅ 存在 | 无需修改 | ✅ |

**实现代码**：
```cpp
// include/physics/fluid_lbm.h
void collision();  // 新增
void collisionWithForce(const float* fx, const float* fy, const float* fz);  // 新增
void applyDarcyDamping(const float* liquid_fraction, float C, float dt);  // 新增重载

// src/physics/fluid_lbm.cu
void FluidLBM::collision() {
    collisionBGK(0.0f, 0.0f, 0.0f);
}

void FluidLBM::collisionWithForce(const float* fx, const float* fy, const float* fz) {
    collisionBGK(fx, fy, fz);
}

void FluidLBM::applyDarcyDamping(const float* liquid_fraction, float C, float dt) {
    // 临时实现：打印警告
    std::cerr << "Warning: 3-param applyDarcyDamping not implemented. Use 5-param version.\n";
}
```

---

## 3. LaserSource / LaserHeatSource

| run_simulation期望 | 现有实现 | 解决方案 | 状态 |
|-------------------|---------|----------|------|
| `LaserSource(nx,ny,nz,dx,dy,dz,...)` | ❌ 不存在（POD构造） | 创建新类`LaserHeatSource` | 阶段3 |
| `laser->setPosition(x,y,z)` | ✅ `LaserSource::setPosition()` | 包装到新类 | 阶段3 |
| `laser->applyHeatSource(T, dt)` | ❌ 不存在 | 新类提供此方法 | 阶段3 |

**现有实现**：
```cpp
// laser_source.h (当前)
class LaserSource {
    __host__ __device__ LaserSource(float P, float w0, float eta, float delta);
    __host__ __device__ void setPosition(float x, float y, float z);
    __host__ __device__ float computeVolumetricHeatSource(x, y, z);
};

__global__ void computeLaserHeatSourceKernel(float* heat_source, LaserSource laser, ...);
```

**新增封装**：
```cpp
// include/physics/laser_heat_source.h (新文件)
namespace lbm {
namespace physics {

class LaserHeatSource {
public:
    LaserHeatSource(int nx, int ny, int nz,
                    float dx, float dy, float dz,
                    float power, float spot_radius,
                    float absorption, float penetration_depth);
    ~LaserHeatSource();

    void setPosition(float x, float y, float z);
    const float* getHeatSource() const;  // 获取热源场
    void updateHeatSource();             // 重新计算（调用kernel）

private:
    int nx_, ny_, nz_;
    float dx_, dy_, dz_;
    LaserSource laser_;       // 底层POD
    float* d_heat_source_;    // GPU内存
};

}} // namespace
```

**使用方式对比**：
```cpp
// 现有方式（Phase 5工作代码）
LaserSource laser(1200.0f, 50e-6f, 0.35f, 15e-6f);
laser.setPosition(80e-6, 80e-6, 0.0);
float* d_heat_source;
cudaMalloc(&d_heat_source, num_cells * sizeof(float));
computeLaserHeatSourceKernel<<<grid, block>>>(d_heat_source, laser, dx, dy, dz, nx, ny, nz);
thermal.addHeatSource(d_heat_source, dt);

// 新方式（run_simulation期望）
auto laser = std::make_unique<LaserHeatSource>(
    nx, ny, nz, dx, dy, dz,
    1200.0f, 50e-6f, 0.35f, 15e-6f
);
laser->setPosition(80e-6, 80e-6, 0.0);
// 每个时间步
laser->updateHeatSource();  // 内部调用kernel
thermal.addHeatSource(laser->getHeatSource(), dt);
```

---

## 4. VTKWriter

| run_simulation期望 | 现有实现 | 解决方案 | 状态 |
|-------------------|---------|----------|------|
| `VTKWriter writer(file, nx, ny, nz, dx, dy, dz)` | ❌ 静态方法 | 添加Builder类 | 阶段4 |
| `writer.addScalarField(name, data)` | ❌ 不存在 | 新增方法 | 阶段4 |
| `writer.addVectorField(name, ux, uy, uz)` | ❌ 不存在 | 新增方法 | 阶段4 |
| `writer.write()` | ❌ 不存在 | 新增方法 | 阶段4 |

**现有实现**：
```cpp
// io/vtk_writer.h (当前)
class VTKWriter {
public:
    static void writeStructuredPoints(filename, data, nx, ny, nz, ...);
    static void writeStructuredGridWithVectors(filename, T, fl, phase, ux, uy, uz, ...);
    // ... 所有方法都是静态的
};
```

**新增Builder模式**：
```cpp
// io/vtk_writer.h (扩展)
class VTKWriter {
public:
    // 新增构造函数
    VTKWriter(const std::string& filename, int nx, int ny, int nz,
              float dx, float dy, float dz);

    // Builder方法
    void addScalarField(const std::string& name, const float* d_data);
    void addVectorField(const std::string& name,
                        const float* d_ux, const float* d_uy, const float* d_uz);
    void write();

    // 保留现有静态方法（向后兼容）
    static void writeStructuredPoints(...);
    static void writeStructuredGridWithVectors(...);
    // ...

private:
    std::string filename_;
    int nx_, ny_, nz_;
    float dx_, dy_, dz_;
    std::vector<ScalarField> scalar_fields_;
    std::vector<VectorField> vector_fields_;
};
```

**使用方式对比**：
```cpp
// 现有方式（Phase 5）
VTKWriter::writeStructuredGridWithVectors(
    "output.vtk", h_temp, h_fl, h_phase, h_ux, h_uy, h_uz,
    nx, ny, nz, dx, dy, dz
);

// 新方式（run_simulation期望）
VTKWriter writer("output.vtk", nx, ny, nz, dx, dy, dz);
writer.addScalarField("Temperature", d_temp);
writer.addScalarField("LiquidFraction", d_fl);
writer.addVectorField("Velocity", d_ux, d_uy, d_uz);
writer.write();
```

---

## 5. MaterialProperties

| run_simulation期望 | 现有实现 | 解决方案 | 状态 |
|-------------------|---------|----------|------|
| `MaterialDatabase::getTi6Al4V()` | ✅ 存在 | 无需修改 | ✅ |
| `mat.rho_solid` | ✅ 存在 | 无需修改 | ✅ |
| `mat.T_solidus` | ✅ 存在 | 无需修改 | ✅ |

**注意**：run_simulation.cu中使用了简化的材料获取方式：
```cpp
// run_simulation.cu (行112-122)
physics::MaterialProperties mat;
if (cfg.material.type == "Ti6Al4V") {
    mat = physics::Ti6Al4V();  // ❌ 这可能不存在
} else if (...) {
    ...
}
```

**现有正确方式**：
```cpp
// 正确的API
physics::MaterialProperties ti64 = physics::MaterialDatabase::getTi6Al4V();
```

**修复**：
```cpp
// 修改run_simulation.cu
physics::MaterialProperties mat;
if (cfg.material.type == "Ti6Al4V") {
    mat = physics::MaterialDatabase::getTi6Al4V();
} else if (cfg.material.type == "SS316L") {
    mat = physics::MaterialDatabase::getSS316L();
} else if (cfg.material.type == "Inconel718") {
    mat = physics::MaterialDatabase::getInconel718();
} else {
    throw std::runtime_error("不支持的材料: " + cfg.material.type);
}
```

---

## 6. 其他辅助函数

### 6.1 enforceZeroVelocityInSolid
| 位置 | 状态 |
|------|------|
| run_simulation.cu (行27-40) | ✅ 已定义 |
| 使用 | ✅ 正确（行245-252） |

### 6.2 scaleForceArrayKernel
| 位置 | 状态 |
|------|------|
| run_simulation.cu (行43-52) | ✅ 已定义 |
| 使用 | ✅ 正确（行217-219） |

---

## 7. 配置系统

| 功能 | 状态 | 备注 |
|------|------|------|
| `SimulationConfig::loadFromFile()` | ✅ 已实现 | `/home/yzk/LBMProject/src/config/simulation_config.cpp` |
| `SimulationConfig::getPreset()` | ✅ 已实现 | 支持"phase5", "stefan_problem", "thermal_only" |
| `cfg.printSummary()` | ✅ 已实现 | |
| `cfg.validate()` | ✅ 已实现 | 基础验证 |

---

## 8. 核心修改汇总表

| 文件 | 修改类型 | 工作量 | 阶段 |
|------|----------|--------|------|
| `include/physics/thermal_lbm.h` | 添加方法声明 | 5分钟 | 1 |
| `src/physics/thermal_lbm.cu` | 添加方法实现 | 10分钟 | 1 |
| `include/physics/fluid_lbm.h` | 添加方法声明 | 5分钟 | 2 |
| `src/physics/fluid_lbm.cu` | 添加方法实现 | 15分钟 | 2 |
| `include/physics/laser_heat_source.h` | **新文件** | 15分钟 | 3 |
| `src/physics/laser_heat_source.cu` | **新文件** | 15分钟 | 3 |
| `include/io/vtk_writer.h` | 扩展类定义 | 10分钟 | 4 |
| `src/io/vtk_writer.cpp` | 实现Builder方法 | 20分钟 | 4 |
| `apps/run_simulation.cu` | 修复API调用 | 20分钟 | 5 |
| `src/physics/CMakeLists.txt` | 添加laser_heat_source.cu | 2分钟 | 3 |

**总计**：约2小时

---

## 9. 测试检查清单

### 阶段1测试（ThermalLBM）
```bash
cd /home/yzk/LBMProject/build
make -j8
# 编译通过 = 成功
```

### 阶段2测试（FluidLBM）
```bash
make -j8
./visualize_laser_melting_with_flow  # 现有代码应仍正常
```

### 阶段3测试（LaserHeatSource）
```bash
make test_laser_heat_source  # 需创建测试程序
./test_laser_heat_source
```

### 阶段4测试（VTKWriter）
```bash
make test_vtk_builder
./test_vtk_builder
ls -l test_output.vtk
```

### 阶段5测试（run_simulation）
```bash
make run_simulation
# 编译通过 = 成功（不一定运行）
```

### 阶段6测试（配置驱动）
```bash
./run_simulation ../config/laser_melting_phase5.cfg
ls -l visualization_output/*.vtk
# 生成VTK文件 = 成功
```

---

## 10. 常见问题（FAQ）

### Q1: 为什么不直接修改现有类，而要添加新方法？
**A**: 保持向后兼容。Phase 5的`visualize_laser_melting_with_flow.cu`等程序依赖现有API，直接修改会破坏它们。

### Q2: LaserHeatSource是否会取代LaserSource？
**A**: 否。LaserHeatSource是高层封装，内部仍使用LaserSource POD + kernel。两者共存。

### Q3: VTKWriter Builder模式会不会影响性能？
**A**: 会有轻微开销（额外的内存拷贝），但可以通过延迟拷贝优化。当前优先正确性。

### Q4: 如果阶段5失败怎么办？
**A**:
1. 检查编译错误信息
2. 逐个修复API不匹配
3. 如果遇到无法解决的问题，回退到阶段4，使用静态方法临时替代

### Q5: applyDarcyDamping的3参数版本为什么不完整实现？
**A**: 当前设计需要外部管理force数组（见Phase 5代码）。3参数版本需要内部分配或直接修改速度，这需要更多设计决策。暂时保留警告，未来优化。

---

**文档版本**: v1.0
**最后更新**: 2025-11-01
**维护者**: LBM-CFD架构团队
