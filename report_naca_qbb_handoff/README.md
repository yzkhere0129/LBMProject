# NACA D3Q27-Cumulant + QBB-Bouzidi 交接包

**分支**: `feature/compressible-aero` @ 2026-05-09
**Worktree**: `/home/yzk/CompressibleCFD` (从 master 切出)
**目的**: 给伙伴 review 当前进度、QBB 实现细节、待审问题。仿真**未运行**，等批准。

---

## 一句话现状

**D3Q27 Cumulant + Ladd 入口在 ST 圆柱 D/dx=20 上 Cd=3.19（DFG 3.22-3.24，1.2% off）**——
↓ 切到 NACA0012 →
**stair-step 离散倾斜弦破坏 lattice mirror-symmetry，α=8° Cl 反号 (-0.35 vs Kurtulus +0.49)**——
↓ 已写 ↓
**D3Q27 single-node QBB Bouzidi 纸面端完成（streaming kernel + qfrac builder + `--bc qbb-snode` CLI），编译干净**，等批准跑 acceptance test。

---

## 包内文件 (6 个)

1. **README.md** — 本文件，入口 + 导览
2. **[STATUS.md](STATUS.md)** — Pipeline 时间线、ST 圆柱收敛数据、NACA bug 物理诊断、QBB 实现摘要
3. **[QUESTIONS.md](QUESTIONS.md)** — 5 个待伙伴 audit 的问题（**Q1 PULL↔PUSH 映射** 和 **Q2 omega 选择** 是最高 leverage）
4. **[KEY_CODE.md](KEY_CODE.md)** — QBB streaming kernel + qfrac builder 关键代码段 + 设计文档要点
5. **01_st_cylinder_cumulant_d20_vorticity.png** — Cumulant + Ladd 在 ST 2D-2 上的涡量场（成功 baseline，对称卡门街，St=0.30 范围）
6. **02_naca_a8_stair_cl_signflip_vorticity.png** — NACA0012 α=+8° stair-step 涡量场（**尾流上偏 → 升力下指 → Cl 错号**的 visual evidence）

---

## 给伙伴的最小动作清单

如果只有 30 分钟：
1. 读 README + STATUS §2（NACA bug evidence）
2. 看图 02：注意涡量场上下不对称、尾流明显上偏（α=+8° 应该尾流下偏才对）
3. 读 QUESTIONS §Q1，独立推一遍 PULL↔PUSH 映射
4. 给 Q1 + Q2 一个 yes/no/uncertain

如果有 2 小时：
5. 读 KEY_CODE，audit qfrac builder 的 inside_airfoil 测试 + bisection
6. 读 STATUS §3.5（仍未做的）+ §5（验收），给改进建议
7. 独立推 acceptance bar 的合理性（QUESTIONS §Q4）

---

## 关键数据速查

| 项目 | 期望 (DFG / Kurtulus) | 当前 (LBM) | 偏差 | Status |
|---|---|---|---|---|
| ST 2D-2 圆柱 Cd, D/dx=20 | 3.22-3.24 | **3.19** | -1.2% | ✓ |
| ST 2D-2 圆柱 Cl_max | 0.99-1.01 | ~0.91 | -9% (D/dx=20 mesh limit) | △ |
| ST 2D-2 圆柱 St | 0.295-0.305 | TBD (FFT) | - | OK |
| NACA0012 α=0 Cl, stair | 0 | 0 ± 0.001 | ✓ | OK |
| NACA0012 α=+8° Cl, stair | +0.49 | **−0.35** | 反号 | **BUG** |
| NACA0012 α=+8° Cl, **QBB (待跑)** | +0.49 | TBD | TBD | 等批准 |

---

## 关键 commit 链

```
7823a1f  Cumulant Phase 4 BREAKTHROUGH: Cd=3.19 at D/dx=20 (1.2% from DFG)
e8e5318  Cumulant Phase 4 first cut: Cd ≈ 0.55x (entry mass-flux loss)
bd1c308  Cumulant Phase 2: D3Q27 collision, 94 stage tests pass
bb5fc06  Aero Phase 1d: D3Q27 lattice infrastructure
d8a0225  Aero Phase 1c: 2nd-order BCs + halfway walls + Mach control
887703c  Aero Phase 1+1b: Schaefer-Turek pipeline + QBB curved BC (D3Q19)
```

QBB-D3Q27 这一条**未提交**（未运行 → 未提交）。所有改动在 working tree。
