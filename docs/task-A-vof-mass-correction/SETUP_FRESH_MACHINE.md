# 从零搭建：在一台没装过 LBMProject 的机器上跑 Task A 验证

**目的**：在快机器（RTX 30/40 系，desktop GPU）上做完整的 Phase-2 + Phase-4 验证。
**预期总时间**：~2 小时（环境装 30 min + 编译 5 min + 测试 5 min + Phase-2 30 min + Phase-4 90 min）。

---

## 0. 系统要求

| 项 | 最低 | 推荐 |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) 或 WSL2 | Ubuntu 22.04 native |
| GPU | NVIDIA, ≥4 GB VRAM, sm_70+ | RTX 3090/4080/4090 |
| RAM | 8 GB | 16 GB |
| 磁盘 | 30 GB（含 VTK 输出） | 50 GB |
| 网络 | 能访问 github.com | |

---

## 1. 装依赖

```bash
# === Ubuntu / WSL2 系统包 ===
sudo apt update
sudo apt install -y build-essential git cmake python3 python3-pip

# === CUDA Toolkit (12.x 推荐) ===
# 跳过如果已有。验证：
nvcc --version       # 应显示 release 11.4+ (12.x 更佳)
nvidia-smi           # 应能看到 GPU + driver

# 没装的话从 https://developer.nvidia.com/cuda-downloads 选 OS + version
# Ubuntu 22.04 + CUDA 12.4 例：
# wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
# sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent

# === Python 分析依赖 ===
pip3 install --user numpy pyvista
# 验证：
python3 -c "import pyvista; print(pyvista.__version__)"

# === gcc 11+（C++17 + CUDA 17 标准要求）===
gcc --version       # 应 ≥ 11
# 如果 < 11：
# sudo apt install gcc-11 g++-11
# export CC=gcc-11 CXX=g++-11
```

---

## 2. 拉代码

```bash
mkdir -p ~/work && cd ~/work
git clone git@github.com:yzkhere0129/LBMProject.git LBMProject_vof_mass_correction
# 如果没配 SSH key，用 HTTPS：
# git clone https://github.com/yzkhere0129/LBMProject.git LBMProject_vof_mass_correction

cd LBMProject_vof_mass_correction
git checkout feature/vof-mass-correction-destination

# 验证看到 4 个 commits（Task A 的核心）
git log --oneline -5
# 期望：
#   304610e docs(task-A): partial validation report + hand-off recipe
#   dc7903f feat(diag): Task-A 7-criteria acceptance check script
#   cc90a26 feat(vof): A1 v_z-weighted mass correction + bug fixes (Task A)
#   1934854 fix(build): unblock build — ...
#   b0117de docs(brief): tighten A acceptance criteria
```

---

## 3. 编译

```bash
cd ~/work/LBMProject_vof_mass_correction

# 配置（30 秒）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_CUDA_ARCHITECTURES="86"   # RTX 30 系；40 系用 89

# 编译（首次 5-10 min，并行 8）
cmake --build build -j 8 2>&1 | tail -20

# 验证关键 binaries 都生成了
ls -la build/sim_linescan_phase2 build/sim_linescan_phase4 \
       build/tests/test_vof_mass_correction_weighted \
       build/tests/validation/test_trt_degenerate_to_bgk
```

如果报 `CMAKE_CUDA_ARCHITECTURES` 不识别，去 CMakeLists.txt 第 16 行改成你的 GPU 架构号：
| GPU | Architecture |
|---|---|
| RTX 20xx (Turing) | 75 |
| A100 (Ampere) | 80 |
| RTX 30xx (Ampere) | 86 |
| RTX 40xx (Ada) | 89 |

---

## 4. 烟雾测试（5 分钟）

每一步都该秒过，任何 FAIL 立刻停下。

```bash
# === A1 单元测试（应 7/7 PASS, 1 disabled）===
./build/tests/test_vof_mass_correction_weighted
# 期望最后看到：
#   [  PASSED  ] 7 tests.
#   YOU HAVE 1 DISABLED TEST   ← 这个是设计内的，A5 ClampOverflowRedistributed

# === TRT 等价性 anchor（FP32 round-off）===
./build/tests/validation/test_trt_degenerate_to_bgk
# 期望：[  PASSED  ] 1 test.

# === VOF 9 个单元测试（合计 ~10 秒）===
for t in test_vof_advection test_vof_reconstruction test_vof_curvature \
         test_vof_cell_conversion test_vof_surface_tension test_vof_marangoni \
         test_vof_contact_angle test_vof_mass_conservation test_vof_interface_compression; do
  ./build/tests/$t > /tmp/$t.log 2>&1 && echo "  ✅ $t" || { echo "  ❌ $t"; tail -10 /tmp/$t.log; break; }
done
```

如果上面任何一个 FAIL，**不要继续**——先把问题定位修好。可能原因：
- CUDA 版本 / 驱动版本不匹配
- GPU 架构号 (`CMAKE_CUDA_ARCHITECTURES`) 设错了
- gcc < 11 不支持 C++17 nvcc 集成

---

## 5. 跑 Phase-2（核心验证，~30 min on RTX 3090）

```bash
cd ~/work/LBMProject_vof_mass_correction

rm -rf output_phase2 && mkdir output_phase2

# 后台跑 + 实时可看 log
stdbuf -oL ./build/sim_linescan_phase2 > output_phase2/run.log 2>&1 &
PHASE2_PID=$!

# 监督：每 60 秒看一次步数 + VTK 帧
while kill -0 $PHASE2_PID 2>/dev/null; do
  step=$(grep -oE "Step [0-9]+" output_phase2/run.log | awk '{print $2}' | sort -un | tail -1)
  vtks=$(ls output_phase2/*.vtk 2>/dev/null | wc -l)
  echo "$(date +%H:%M:%S)  step=$step  vtks=$vtks"
  sleep 60
done

# 跑完后期望 11 个 VTK 帧（含 t=0），最大 step=10000
ls output_phase2/*.vtk | wc -l   # = 11
```

跑完后检查 7 项验收：

```bash
# t=400 μs（中段）
python3 scripts/diagnostics/check_acceptance.py \
    output_phase2/line_scan_005000.vtk \
    output_phase2/line_scan_000000.vtk

# t=800 μs（最关键）
python3 scripts/diagnostics/check_acceptance.py \
    output_phase2/line_scan_010000.vtk \
    output_phase2/line_scan_000000.vtk
```

**Phase-2 决策点**：

| t=800μs 看到 | 接下来怎么办 |
|---|---|
| 7 项全 PASS | → 跑 Phase-4，全过就 merge |
| 中线 Δh PASS 但侧脊 -100μm > +15μm | → 加 A2 trailing-band mask（约 30 行代码，下一节）|
| 中线 Δh ≤ -14μm（没改善）| → 换权重函数（数学专家方案），或加 A2 |
| t<500μs 出 NaN | → reject，写 `lessons_learned.md` |

---

## 6. 跑 Phase-4（2 ms 验证, ~90 min on RTX 3090）

**只在 Phase-2 通过中线条件后再跑**——Phase-4 跑 25000 步是为验证 t=2ms 稳态条件。

```bash
rm -rf output_phase4 && mkdir output_phase4
stdbuf -oL ./build/sim_linescan_phase4 > output_phase4/run.log 2>&1 &

# 等完成（同样的监督脚本）

# 关键帧：t=2 ms 是 line_scan_025000.vtk
python3 scripts/diagnostics/check_acceptance.py \
    output_phase4/line_scan_025000.vtk \
    output_phase4/line_scan_000000.vtk
```

---

## 7. 如果要加 A2 trailing-band mask（备用方案）

`docs/task-A-vof-mass-correction/A_PARTIAL_RESULTS.md` 已经记录了 yellow flag：t=200μs 时 -100μm 侧脊 +14 μm。如果 t=800μs 真的破了 +15μm，需要在 `applyMassCorrectionInline` 加 trailing-band 掩码。

具体改动（约 30 行）：

```cpp
// include/physics/vof_solver.h —— 给 helper 加 laser_x 参数
void applyMassCorrectionInline(const float* d_vz, float laser_x_lu = -1.0f);

// src/physics/vof/vof_solver.cu —— 在 computeVzWeightSumKernel 里加掩码
// 当 laser_x_lu >= 0 时，只允许 i*1.0 < laser_x_lu - margin_lu 的 cells 参与
__global__ void computeVzWeightSumKernel(
    const float* fill_level, const float* velocity_z, float sign_dm,
    float laser_x_lu, float margin_lu,    // 新参数
    float* partial_sums, int num_cells, int nx, int ny)
{
    int idx = ...;
    int i = idx % nx;     // 解出 x 坐标
    if (laser_x_lu >= 0 && i >= laser_x_lu - margin_lu) {
        sdata[tid] = 0;   // 不在 trailing band
        return;
    }
    // ... 余下不变
}

// MultiphysicsSolver 调用处——传入当前 laser_x（已知量）
vof_->applyMassCorrectionInline(d_velocity_physical_z_, laser_x_lattice);
```

完整示例 patch 我可以根据 Phase-2 t=800μs 的具体测量值来写——不用现在猜。

---

## 8. 可选：参考数据传输（不强制）

Phase-1 完整参考数据 (~4 GB) 在原机器 `/home/yzk/LBMProject/output_phase1/`。F3D ground truth (~7 GB) 在 `/home/yzk/LBMProject/vtk-316L-150W-50um-V800mms/`。

7 项 acceptance 都是**绝对阈值**，**不需要** Phase-1 参考。但如果想做轨迹对比图：

```bash
# 选项 A：rsync 过来（快网络上 ~5 min）
mkdir -p output_phase1
rsync -avh --progress 用户@原机器IP:/home/yzk/LBMProject/output_phase1/ ./output_phase1/

# 选项 B：在新机器上重新跑 Phase-1
cmake --build build --target sim_linescan_phase1 -j 8
rm -rf output_phase1 && mkdir output_phase1
stdbuf -oL ./build/sim_linescan_phase1 > output_phase1/run.log 2>&1
# RTX 3090 约 25 min，完成后有 11 帧
```

---

## 9. 故障排查速查

### `nvcc fatal: Unsupported gpu architecture 'compute_xx'`
GPU 太新或太旧。改 `CMAKE_CUDA_ARCHITECTURES` 为你 GPU 的正确架构号（见 §3 表）。

### `error: identifier "use_guo_forcing_" is undefined`
旧版 commit `1934854` 没合进来，重 `git pull` 一下。

### Build 卡在 `lbm_physics`，没报错但 5 分钟没动静
正常——`vof_solver.cu` 是大文件，nvcc 编它要 2-3 分钟。看 `top` 应该有一个 `cicc` 进程在跑。

### Phase-2 跑了 5 分钟还在 step 0
检查 `output_phase2/run.log` 头几行，看是否报 `cudaMalloc failed: out of memory`。如果是，GPU 显存不够（需 ≥ 3 GB free），关掉浏览器/桌面再跑。

### `[VOF MASS CORRECTION A1-fallback-uniform]` 出现
W ≈ 0 触发了均匀回退路径——说明那一步没有上流的界面 cells。LPBF 跑久了不该常发生。如果 `tail -100 run.log | grep "fallback"` 频繁出现，说明 v_z 加权失效，A1 算法对这个工况无效。

---

## 10. 一行总结：成功的样子

跑完整套验证应该看到：

```
$ python3 scripts/diagnostics/check_acceptance.py output_phase2/line_scan_010000.vtk output_phase2/line_scan_000000.vtk

[1/2] Centerline Δh (95%ile, n=??? cells): -X.XX μm  (target ≥ −10 μm)  PASS
[3]   Side ridge Δh @ −100 μm: +Y μm, @ −200 μm: +Z μm  (target both ∈ [+3, +10])  PASS
[4]   Mass drift |ΔM/M₀|: ±0.0001%  (target < ±1.0%)  PASS
[7]   v_z @ centerline x=??? μm: +X.XX m/s  (target > +0.10)  PASS
```

四项全 PASS + Phase-4 同样四项全 PASS + TRT anchor + 75 phase-change tests → 走 brief 的 merge 协议。
