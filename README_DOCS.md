# 项目文档索引

**最后更新**: 2025-10-31

---

## 🚀 快速开始

### 👤 用户重启后
1. 阅读: `QUICK_START_AFTER_REBOOT.md`（5 分钟）
2. 检查环境并运行

### 🤖 Claude 重启后
1. **必读**: `CLAUDE_CONTEXT.md`（5 分钟）⭐⭐⭐⭐⭐
2. 然后查看 `PROJECT_STATUS.md` 了解全貌

### 第一次使用
1. 阅读: `PROJECT_STATUS.md`（15 分钟）
2. 运行: `cd build && ./visualize_laser_melting_with_flow`

---

## 📚 文档分类

### 1. 项目概览文档

#### `CLAUDE_CONTEXT.md` ⭐⭐⭐⭐⭐ 🤖
**位置**: `/home/yzk/LBMProject/CLAUDE_CONTEXT.md`

**内容**:
- **专为 Claude 设计的上下文恢复文档**
- 最近修复的 4 个 bug（详细说明）
- 坐标系统（关键！）
- 当前参数配置
- 已知问题和解决方案
- 用户交互模式
- 关键记忆点

**何时阅读**: **Claude 新会话开始时必读！**

---

#### `PROJECT_STATUS.md` ⭐⭐⭐⭐⭐
**位置**: `/home/yzk/LBMProject/PROJECT_STATUS.md`

**内容**:
- 项目完整状态
- Phase 1-5 完成情况
- 所有 bug 修复记录
- 参数配置速查
- 故障排除指南
- 性能基准

**何时阅读**: 用户或 Claude 想全面了解项目

---

#### `QUICK_START_AFTER_REBOOT.md` ⭐⭐⭐⭐⭐
**位置**: `/home/yzk/LBMProject/QUICK_START_AFTER_REBOOT.md`

**内容**:
- 5 步快速启动（< 5 分钟）
- 环境检查
- 重新编译（如需）
- 运行验证
- 常见问题

**何时阅读**: 每次重启后

---

### 2. Phase 5 技术文档

#### `build/SESSION_SUMMARY_2025_10_31.md` ⭐⭐⭐⭐
**位置**: `/home/yzk/LBMProject/build/SESSION_SUMMARY_2025_10_31.md`

**内容**:
- 本次会话发现的所有 bug
- 4 个关键 bug 的详细修复
- 修复前后对比
- 用户贡献总结
- 教训与收获

**何时阅读**: 想了解 2025-10-31 修复了什么

---

#### `build/PHASE5_VALIDATION_REPORT.md` ⭐⭐⭐⭐
**位置**: `/home/yzk/LBMProject/build/PHASE5_VALIDATION_REPORT.md`

**内容**:
- 理论背景（Rayleigh-Bénard 对流）
- 物理验证（定性 + 定量）
- Rayleigh 数计算
- 速度量级分析
- 与文献对比
- 优化建议

**何时阅读**: 需要理解物理原理或验证结果

---

#### `build/SOLID_VELOCITY_FIX_COMPLETE.md` ⭐⭐⭐
**位置**: `/home/yzk/LBMProject/build/SOLID_VELOCITY_FIX_COMPLETE.md`

**内容**:
- 固体速度问题详细分析
- `enforceZeroVelocityInSolid` 实现
- 修复效果验证
- 液相分数分布

**何时阅读**: 需要了解固体速度约束实现

---

#### `build/PHASE5_FINAL_STATUS_REPORT.md` ⭐⭐
**位置**: `/home/yzk/LBMProject/build/PHASE5_FINAL_STATUS_REPORT.md`

**内容**:
- 速度为零问题的早期诊断
- NaN 修复记录
- 单位转换问题
- Darcy 阻尼分析

**何时阅读**: 了解早期调试历史（部分内容已过时）

---

### 3. 日志文件

#### 最新运行日志
**位置**: `/home/yzk/LBMProject/build/*.log`

**重要日志**:
- `adiabatic_boundary_test.log`: 绝热边界修复后的运行
- `gravity_fix_test.log`: 重力方向修复后的运行
- `solid_velocity_fix_test.log`: 固体速度修复后的运行

**何时查看**: 调试或验证运行结果

---

### 4. 输出数据

#### VTK 可视化文件
**位置**: `/home/yzk/LBMProject/build/visualization_output/`

**文件格式**: `laser_melting_flow_XXXXXX.vtk`（81 个文件）

**包含数据**:
- Temperature (温度场)
- LiquidFraction (液相分数)
- PhaseState (相态: 0=固, 1=糊, 2=液)
- Velocity (速度矢量)

**何时使用**: ParaView 可视化

---

## 🔍 按需求查找文档

### 需求: "我想快速恢复工作"
→ 阅读 `QUICK_START_AFTER_REBOOT.md`

### 需求: "我想了解项目整体状态"
→ 阅读 `PROJECT_STATUS.md`

### 需求: "我想知道今天修复了什么 bug"
→ 阅读 `build/SESSION_SUMMARY_2025_10_31.md`

### 需求: "我想验证结果是否物理正确"
→ 阅读 `build/PHASE5_VALIDATION_REPORT.md`

### 需求: "我想知道如何修复固体速度问题"
→ 阅读 `build/SOLID_VELOCITY_FIX_COMPLETE.md`

### 需求: "我想调整参数优化速度"
→ 查看 `PROJECT_STATUS.md` 的"参数配置速查"部分

### 需求: "我想查看运行结果"
→ 打开 `build/visualization_output/` 用 ParaView

### 需求: "我遇到了问题需要调试"
→ 查看 `PROJECT_STATUS.md` 的"故障排除"部分

---

## 📊 文档重要性排序

### ⭐⭐⭐⭐⭐ 必读
1. `QUICK_START_AFTER_REBOOT.md` - 重启后必读
2. `PROJECT_STATUS.md` - 项目全貌

### ⭐⭐⭐⭐ 强烈推荐
3. `SESSION_SUMMARY_2025_10_31.md` - 今天做了什么
4. `PHASE5_VALIDATION_REPORT.md` - 物理验证

### ⭐⭐⭐ 参考资料
5. `SOLID_VELOCITY_FIX_COMPLETE.md` - 固体速度修复细节

### ⭐⭐ 历史记录
6. `PHASE5_FINAL_STATUS_REPORT.md` - 早期调试（部分过时）

---

## 🗂️ 文档层次结构

```
/home/yzk/LBMProject/
│
├── README_DOCS.md                    ← 本文件（文档索引）
├── PROJECT_STATUS.md                 ← 项目整体状态
├── QUICK_START_AFTER_REBOOT.md      ← 快速启动指南
├── CLAUDE.md                         ← 代码风格规范
│
└── build/
    ├── SESSION_SUMMARY_2025_10_31.md     ← 本次会话总结
    ├── PHASE5_VALIDATION_REPORT.md       ← 物理验证报告
    ├── SOLID_VELOCITY_FIX_COMPLETE.md    ← 固体速度修复
    ├── PHASE5_FINAL_STATUS_REPORT.md     ← 早期状态报告
    │
    ├── *.log                              ← 运行日志
    │
    └── visualization_output/
        └── laser_melting_flow_*.vtk      ← 可视化数据
```

---

## 📝 建议的阅读顺序

### 第一次使用项目
1. `PROJECT_STATUS.md`（15 分钟）- 了解全貌
2. `SESSION_SUMMARY_2025_10_31.md`（10 分钟）- 了解最新修复
3. `QUICK_START_AFTER_REBOOT.md`（5 分钟）- 学习如何运行

### 重启后恢复
1. `QUICK_START_AFTER_REBOOT.md`（5 分钟）- 快速启动
2. （可选）`PROJECT_STATUS.md` - 如需回顾

### 深入研究
1. `PHASE5_VALIDATION_REPORT.md` - 理解物理
2. `SOLID_VELOCITY_FIX_COMPLETE.md` - 实现细节
3. 源代码 + 注释

---

## 🔧 维护建议

### 添加新文档时
1. 在本文件中添加条目
2. 标注重要性（⭐）
3. 说明何时阅读

### 更新现有文档时
1. 更新"最后更新"日期
2. 在本索引中标注"已更新"

### 过时文档
- 标注 ⚠️ 或移到 `archive/` 目录
- 在索引中说明"已过时"

---

## 💡 提示

**文档太多？**
- 优先阅读 ⭐⭐⭐⭐⭐ 文档
- 需要时再查阅其他文档

**找不到信息？**
- 使用 `grep -r "关键词" *.md`
- 或询问 Claude Code

**想贡献文档？**
- 遵循现有格式
- 更新本索引
- 添加清晰的标题和总结

---

**这个索引帮助您快速找到需要的信息！** 📖
