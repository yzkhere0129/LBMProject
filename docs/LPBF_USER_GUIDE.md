# LPBF 仿真用户指南

本指南介绍如何运行 LPBF (激光粉末床熔融) 仿真程序并修改参数。

---

## 1. 快速开始

### 1.1 编译

```bash
cd /home/yzk/LBMProject/build
make visualize_lpbf_scanning -j4
```

### 1.2 运行

**使用默认参数运行:**
```bash
./visualize_lpbf_scanning
```

**使用配置文件运行:**
```bash
./visualize_lpbf_scanning --config ../config/high_power_195w_radiation.cfg
```

**覆盖部分参数:**
```bash
./visualize_lpbf_scanning --config ../config/high_power_195w_radiation.cfg --steps 5000 --output my_output
```

### 1.3 命令行参数

| 参数 | 说明 |
|------|------|
| `--config <file>` | 指定配置文件 |
| `--steps <N>` | 覆盖仿真步数 |
| `--output <dir>` | 覆盖输出目录 |
| `--help` | 显示帮助信息 |

---

## 2. 关键参数说明

### 2.1 域大小参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nx` | 200 | X方向网格数 |
| `ny` | 150 | Y方向网格数 |
| `nz` | 100 | Z方向网格数 |
| `dx` | 2.0e-6 m | 网格尺寸 (2 um) |

**物理尺寸:** 默认 400 x 300 x 200 um

### 2.2 激光参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `laser_power` | 300 W | 激光功率 |
| `laser_spot_radius` | 50e-6 m | 光斑半径 (50 um) |
| `laser_absorptivity` | 0.35 | 吸收率 (35%) |
| `laser_penetration_depth` | 10e-6 m | 穿透深度 |
| `laser_start_x` | 100e-6 m | 激光起始X位置 |
| `laser_start_y` | 150e-6 m | 激光起始Y位置 |
| `laser_scan_vx` | 0.5 m/s | X方向扫描速度 |
| `laser_scan_vy` | 0.0 m/s | Y方向扫描速度 |
| `laser_shutoff_time` | 700e-6 s | 激光关闭时间 |

### 2.3 材料参数 (Ti6Al4V)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `thermal_diffusivity` | 5.8e-6 m^2/s | 热扩散系数 |
| `kinematic_viscosity` | 0.0333 (格子单位) | 运动粘度 |
| `density` | 4110 kg/m^3 | 密度 |
| `surface_tension_coeff` | 1.65 N/m | 表面张力系数 |
| `dsigma_dT` | -0.26e-3 N/(m*K) | 表面张力温度系数 |

### 2.4 时间参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dt` | 1.0e-7 s | 时间步长 (0.1 us) |
| `num_steps` | 3000 | 仿真步数 |
| `output_interval` | 25 | 输出间隔 |

**默认仿真时间:** 3000 * 0.1 us = 300 us

---

## 3. 修改参数的方法

### 方法一: 使用配置文件 (推荐)

创建或修改 `.cfg` 文件:

```bash
# 文件: my_config.cfg

# 域参数
nx = 100
ny = 100
nz = 50
dx = 2.0e-6

# 时间参数
dt = 1.0e-7
num_steps = 5000
output_interval = 50

# 激光参数
laser_power = 200.0
laser_spot_radius = 50.0e-6
laser_absorptivity = 0.35
laser_scan_vx = 0.8

# 物理开关
enable_thermal = true
enable_phase_change = true
enable_fluid = true
enable_marangoni = true
enable_darcy = true
enable_vof = true
enable_recoil_pressure = true

# 输出
output_directory = my_simulation_output
```

运行:
```bash
./visualize_lpbf_scanning --config my_config.cfg
```

### 方法二: 修改源代码

1. 编辑源文件:
```bash
nano /home/yzk/LBMProject/apps/visualize_lpbf_scanning.cu
```

2. 找到并修改 `config.xxx` 参数 (约在第 50-150 行)

3. 重新编译:
```bash
cd /home/yzk/LBMProject/build
make visualize_lpbf_scanning -j4
```

---

## 4. 输出文件

### 4.1 输出位置

VTK 文件保存在: `<output_dir>/lpbf_XXXXXX.vtk`

默认: `lpbf_realistic/lpbf_000000.vtk`, `lpbf_000025.vtk`, ...

### 4.2 VTK 文件内容

每个 VTK 文件包含:
- **Temperature**: 温度场 (K)
- **LiquidFraction**: 液相分数 (0-1)
- **Phase**: 相态 (0=固体, 1=糊状, 2=液体)
- **FillLevel**: VOF 填充率 (0=气体, 1=金属)
- **Velocity**: 速度矢量场 (m/s)

### 4.3 使用 ParaView 可视化

```bash
paraview lpbf_realistic/lpbf_*.vtk
```

**推荐可视化步骤:**

1. 打开所有 VTK 文件 (File -> Open -> 选择全部)
2. 点击 "Apply"
3. 颜色映射: 选择 "Temperature" 查看温度分布
4. 添加速度箭头: Filters -> Glyph -> Vectors: Velocity
5. 播放动画查看熔池演化

---

## 5. 常见修改示例

### 5.1 减小域大小 (加快计算)

```bash
# 配置文件
nx = 100
ny = 100
nz = 50
```

或修改源代码:
```cpp
config.nx = 100;
config.ny = 100;
config.nz = 50;
```

### 5.2 改变激光功率

低功率 (传导模式):
```bash
laser_power = 150.0
```

高功率 (匙孔模式):
```bash
laser_power = 400.0
```

### 5.3 改变扫描速度

慢速扫描 (深熔池):
```bash
laser_scan_vx = 0.2
```

快速扫描 (浅熔池):
```bash
laser_scan_vx = 1.0
```

### 5.4 延长仿真时间

```bash
num_steps = 10000
output_interval = 100
```

### 5.5 静止激光 (点熔)

```bash
laser_scan_vx = 0.0
laser_scan_vy = 0.0
```

---

## 6. 示例配置文件

项目提供了多个预设配置文件:

| 文件 | 用途 |
|------|------|
| `config/high_power_195w_radiation.cfg` | 195W 高功率测试 |
| `config/laser_melting_phase5.cfg` | 激光熔化测试 |
| `config/thermal_only.cfg` | 纯热传导测试 |
| `config/laser_pulse_quick_test.cfg` | 快速测试 |

---

## 7. 故障排除

### 问题: 编译失败
```bash
# 清理后重新编译
cd /home/yzk/LBMProject/build
make clean
cmake ..
make visualize_lpbf_scanning -j4
```

### 问题: 温度过高 (>10000 K)
- 启用辐射边界条件: `enable_radiation_bc = true`
- 降低激光功率或增大光斑

### 问题: 仿真太慢
- 减小域大小 (nx, ny, nz)
- 增大 output_interval

### 问题: 数值不稳定
- 减小时间步长 dt
- 检查 CFL 限制参数

---

## 8. 联系方式

如有问题，请查阅项目文档或联系开发团队。
