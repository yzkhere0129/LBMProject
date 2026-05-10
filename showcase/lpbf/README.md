# LPBF 316L 激光熔池 | Laser Powder-Bed Fusion (316L stainless steel)

## What & Why

单道激光线扫描的瞬时熔池仿真，用于验证整个多物理栈在工业级算例上的协同工作能力。
该算例同时驱动 D3Q19 流体、D3Q7 焓法热扩散、VOF/PLIC 自由界面、16-bounce DDA
光线追踪激光（含 Born-Wolf 复折射率 Fresnel）、Anisimov–Knight 反冲压、Hertz-Knudsen
蒸发、Marangoni + CSF 表面张力 — 是平台上最高耦合度的算例。

## Key result

![3D melt-pool isosurface](melt_pool_3d.png)

![Cross-section views](melt_pool_sections.png)

> 第一张图为 VOF=0.5 等值面，按温度着色 (`inferno` colormap)；亮黄色为蒸发烟羽
> ($T \gtrsim 4000$ K)，深紫尾部为已凝固轨迹。第二张图为同一时刻三向剖面：
> 上 — 沿扫描方向纵切，可见前壁陡峭、后壁平缓的非对称 keyhole；
> 左下 — 横向切片，熔池横截面；
> 右下 — 表层正下方俯视图，速度幅值 + 等温线（液相线 $T_{\rm m}$=1700 K，沸点 $T_{\rm b}$≈3000 K）。

## Numbers vs reference

| 指标 | 本工作 (LBM-CUDA) | 参考 (Flow3D 工业基线 / 实验) | 偏差 |
|---|---|---|---|
| $T_{\max}$ | 4 079 K | ~3 800–4 200 K | 在区间内 |
| $v_{\max}$ | 6.01 m/s | 5–8 m/s（Marangoni 驱动） | 在区间内 |
| 熔池长 $L_{\rm pool}$ | 220 μm | 438 μm (Flow3D 标定) | −50 %（详见 Notes） |
| 熔池宽 $W_{\rm pool}$ | 92 μm | 73 μm (Flow3D) | +26 % |
| Keyhole 深 $D_{\rm KH}$ | 103 μm | 78 μm (Flow3D) | +32 % |
| Ma$_{\rm LBM}$ 峰值 | 0.499 | < 0.6（弱可压稳定边界） | 通过 |
| 总质量漂移 | −0.33 % | 通过 ($<1$ %) | 通过 |

## How to reproduce

```bash
# Build (RTX 30/40 series; adjust CMAKE_CUDA_ARCHITECTURES if needed)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j benchmark_keyhole_316L

# Run 3000-step (240 μs) simulation
./benchmark_keyhole_316L --output ../showcase/lpbf/output/

# Re-extract figures from the keyframe at step 3000
python3 ../showcase/lpbf/extract_sections.py \
    ../showcase/lpbf/output/snap_003000.vtk \
    ../showcase/lpbf/melt_pool_sections_data.npz
python3 ../showcase/lpbf/render_sections.py \
    ../showcase/lpbf/melt_pool_sections_data.npz \
    ../showcase/lpbf/melt_pool_sections.png
```

预计运行时间：~30–60 min on a single RTX 30/40-series GPU。
原始 600 MB VTK 不入库；本目录提交了提取后的轻量 `.npz` (~1 MB) 与图像。

## Notes & limitations

- **熔池长度低估 50 %**：当前 keyhole 深度过深 + 宽度过宽，导致后向能量被过度消耗在
  侧壁加热上，留给尾部的过热液时间不足以拉长熔池。已诊断为 dx 量化 + stair-step
  BC + BC 阶次组合效应；详见仓库根 `README.md` 的 "已声明 gap" 表。
- **网格分辨率限制**：dx = 2 μm 在 4 GB 单 GPU 上是上限；dx = 1 μm 需 ≥ 12 GB 显存。
- **多层熔覆 / 工艺曲线扫描尚未支持**；本算例为单道、单层。
- 物理参数完全可追溯：Beer-Lambert 强度模型 + 复折射率 $(n+ik) = (2.9613, 4.0133)$
  @ 1064 nm（来自 Mills 1991 & ASM Handbook Vol. 2，与 Flow3D prepin 一致）。
