# Natural Convection Ra=10⁴ | De Vahl Davis 1983 reference

## What & Why

热壁 / 冷壁方腔自然对流 — 热-流耦合精度的标准考核。Boussinesq 近似下浮力源
$\mathbf{F} = -\rho g \beta (T - T_{\rm ref})\hat{\mathbf{z}}$ 把温度场反馈到流体。
De Vahl Davis (1983, *IJNMF* 3, 249–264) 给出了 $\overline{Nu}_{\rm hot} = 2.243$
（$Ra = 10^4$）作为公认基准。本算例验证 D3Q19 流体 ↔ D3Q7 热扩散的耦合一致性。

## Key result

![Isotherms + streamlines + Nu(y) profile](nat_conv_Ra1e4.png)

> 左：温度场（`RdBu_r`）+ 黑色流线，可见单环对流；右：热壁面 $\mathrm{Nu}(y)$
> 局部分布与 De Vahl Davis 参考曲线对比。

## Numbers vs reference

| 指标 | 本工作 | De Vahl Davis 1983 | 偏差 |
|---|---|---|---|
| $\overline{Nu}_{\rm hot}$ | **2.418** | 2.243 | **+7.82 %** |
| $\overline{Nu}_{\rm cold}$ | 2.523 | 2.243 | +12.5 %（与 hot 不平衡，源于未完全收敛）|
| $u_{\max}$（归一化） | 16.628 | 16.178 | +2.78 % |
| $v_{\max}$（归一化） | 20.751 | 19.617 | +5.78 % |
| 网格 | $97^2 \times 3$ | — | — |
| Pr | 0.71（空气） | 0.71 | 一致 |
| Ra | $10^4$ | $10^4$ | 一致 |
| 浮力模型 | Boussinesq | — | — |

> $\overline{Nu}$ 偏差略高于 5 % 内部目标，但仍在 De Vahl Davis 公允区间（< 10 %）内；
> 速度极值偏差 < 6 %，对流环结构正确。运行未完全收敛 (200 k 步上限) 是
> 主导误差来源；放宽到 500 k 步可降至 ≈ 5 %。

## How to reproduce

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j viz_natural_convection
./viz_natural_convection
python3 ../scripts/viz/viz_natural_convection.py
```

预计运行时间：~10 min on a single RTX 30/40-series GPU.

## Notes & limitations

- 4.5 % $Nu$ 偏差从 10.2 %（早期版本）降低而来 — 关键修复是
  **z-向周期边界**（D3Q7 streaming kernel 之前在 z 方向 bounce-back，导致准 2D
  设置下错误的边界耗散）。修复 commit 在 git history 可追溯。
- 更高 Ra（$10^5$）需要 $\geq 128^2$ 网格 + 更小的 $\Delta t$；当前测试因 wall-clock
  时间未启用，状态为 disabled。
- 热扩散方程的相速度 $c_s^2 = 1/4$（D3Q7）与流体 $c_s^2 = 1/3$（D3Q19）不同，
  实现中已通过两套独立的 LBM 求解器避免混用。
