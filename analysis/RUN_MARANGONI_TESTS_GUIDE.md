# Guide: Running Marangoni Tests and Generating VTK Output

This guide shows how to run Marangoni flow tests to generate fresh VTK visualization files.

---

## Available Marangoni Test Executables

Located in `/home/yzk/LBMProject/build/`:

### Production Tests (Recommended for Visualization)

1. **`visualize_phase6_marangoni`** - Full multiphysics Marangoni simulation
   - Generates time-series VTK output
   - Most comprehensive test
   - Output: `build/tests/validation/phase6_test2c_visualization/`

2. **`visualize_lpbf_marangoni_realistic`** - Realistic LPBF parameters
   - Uses literature-validated parameters
   - Good for comparing to experiments

3. **`test_marangoni_velocity`** (in `build/tests/validation/`)
   - Validates Marangoni velocity magnitudes
   - Quick test with VTK output

### Unit/Integration Tests

4. **`test_marangoni_system`** (in `build/tests/`)
   - System-level Marangoni test

5. **`test_marangoni_flow`** (in `build/tests/`)
   - Basic Marangoni flow test

6. **`test_marangoni_force`** (in `build/tests/`)
   - Marangoni force computation test

7. **`test_vof_marangoni`** (in `build/tests/`)
   - VOF + Marangoni coupling test

### Specialized Tests

8. **`test_marangoni_benchmark`**
   - Benchmark performance test

9. **`test_marangoni_gradient_limiter`**
   - Tests gradient limiting for stability

10. **`test_marangoni_velocity_scale`**
    - Tests velocity scaling

11. **`test_disable_marangoni`** (in `build/tests/integration/multiphysics/`)
    - Control test with Marangoni disabled

---

## Running Tests to Generate VTK Files

### Method 1: Run Full Visualization Test (Recommended)

This is the test that generated the 56 VTK files we just analyzed.

```bash
cd /home/yzk/LBMProject/build
./visualize_phase6_marangoni
```

**Output location:**
```
/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/
├── marangoni_flow_000000.vtk
├── marangoni_flow_000200.vtk
├── ...
└── marangoni_flow_010000.vtk
```

**Runtime:** ~30-60 seconds (56 timesteps with output every 200 steps)

---

### Method 2: Run Validation Test

```bash
cd /home/yzk/LBMProject/build
./tests/validation/test_marangoni_velocity
```

**Output location:**
```
/home/yzk/LBMProject/build/tests/validation/marangoni_output/
```

---

### Method 3: Run Realistic LPBF Test

```bash
cd /home/yzk/LBMProject/build
./visualize_lpbf_marangoni_realistic
```

**Output location:** Check console output for VTK file paths

---

## Analyzing New VTK Output

After running a test, use the analysis scripts:

### Quick Check (Single File)

```bash
# Activate virtual environment
source /home/yzk/LBMProject/venv/bin/activate

# Quick analysis of most recent file
python /home/yzk/LBMProject/analysis/quick_marangoni_check.py

# Or specify a file
python /home/yzk/LBMProject/analysis/quick_marangoni_check.py /path/to/file.vtk
```

### Comprehensive Analysis (All Timesteps)

```bash
# Activate virtual environment
source /home/yzk/LBMProject/venv/bin/activate

# Marangoni flow analysis
python /home/yzk/LBMProject/analysis/analyze_marangoni_comprehensive.py

# Multiphysics coupling analysis
python /home/yzk/LBMProject/analysis/analyze_multiphysics_coupling.py
```

**Note:** You may need to modify the `VTK_DIR` parameter in the scripts if output location differs.

---

## Customizing VTK Output

### Modify Output Frequency

Edit the test source code to change VTK output frequency:

```cpp
// In the test file (e.g., visualize_phase6_marangoni.cu)
const int VTK_OUTPUT_INTERVAL = 200;  // Output every 200 timesteps
```

### Change Domain Size

```cpp
// Modify grid dimensions
const int NX = 100;
const int NY = 100;
const int NZ = 50;
```

### Adjust Simulation Time

```cpp
// Modify total timesteps
const int TOTAL_TIMESTEPS = 10000;
```

After modifying, rebuild:

```bash
cd /home/yzk/LBMProject/build
cmake --build . --target visualize_phase6_marangoni
```

---

## Understanding VTK Output Fields

Each VTK file contains:

1. **Velocity** (vector) - Fluid velocity field in m/s
   - Use for: Flow patterns, Marangoni-driven circulation

2. **Temperature** (scalar) - Temperature field in Kelvin
   - Use for: Heat distribution, thermal gradients

3. **LiquidFraction** (scalar) - Phase field (0=solid, 1=liquid)
   - Use for: Melt pool size and shape

4. **PhaseState** (scalar) - Phase indicator
   - Use for: Phase boundaries

5. **FillLevel** (scalar) - Volume-of-fluid (VOF)
   - Use for: Free surface location

---

## Expected Results

Based on current analysis of `phase6_test2c_visualization`:

| Metric | Expected Value | Physical Meaning |
|--------|----------------|------------------|
| Max velocity | 0.5 - 2.0 m/s | Marangoni-driven flow |
| Surface velocity | 0.3 - 1.0 m/s | Surface tension gradient effect |
| Temperature gradient | 10^6 - 10^7 K/m | Drives Marangoni stress |
| Liquid fraction | 2 - 10% | Small melt pool |
| Reynolds number | 2 - 40 | Transitional flow |
| Peclet number | 0.1 - 2.0 | Diffusion-dominated |

---

## Troubleshooting

### VTK Files Not Generated

**Check:**
1. Test ran to completion (no crashes)
2. Output directory exists and is writable
3. VTK output enabled in code (check `ENABLE_VTK` flag)

**Solution:**
```bash
# Create output directory manually
mkdir -p /home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization
```

### Analysis Scripts Fail to Read VTK

**Error:** "Data array not present" or "Error reading ascii data"

**Cause:** Incomplete or corrupted VTK file

**Solution:**
- The analysis scripts now skip corrupted files automatically
- Check that simulation ran to completion
- Verify VTK file size (should be ~14 MB for 500k points)

```bash
# Check file sizes
ls -lh /path/to/vtk/output/*.vtk
```

### Python Dependencies Missing

**Error:** "ModuleNotFoundError: No module named 'pyvista'"

**Solution:**
```bash
# Virtual environment already created at:
source /home/yzk/LBMProject/venv/bin/activate

# Or recreate if needed:
python3 -m venv /home/yzk/LBMProject/venv
source /home/yzk/LBMProject/venv/bin/activate
pip install pyvista matplotlib scipy numpy
```

---

## Analysis Workflow

```
┌─────────────────────────┐
│  Run Marangoni Test     │
│  (generate VTK files)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Quick Check            │
│  (validate output)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Comprehensive Analysis │
│  (time evolution,       │
│   coupling, etc.)       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Generate Visualizations│
│  (PNG plots)            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  ParaView 3D Rendering  │
│  (optional)             │
└─────────────────────────┘
```

---

## File Locations Reference

### Executables
```
/home/yzk/LBMProject/build/
├── visualize_phase6_marangoni          # Main visualization test
├── visualize_lpbf_marangoni_realistic  # Realistic LPBF
└── tests/
    └── validation/
        └── test_marangoni_velocity     # Validation test
```

### VTK Output
```
/home/yzk/LBMProject/build/tests/validation/phase6_test2c_visualization/
└── marangoni_flow_*.vtk  (56 files)
```

### Analysis Scripts
```
/home/yzk/LBMProject/analysis/
├── quick_marangoni_check.py              # Quick validation
├── analyze_marangoni_comprehensive.py    # Full time-series analysis
└── analyze_multiphysics_coupling.py      # Coupling analysis
```

### Analysis Results
```
/home/yzk/LBMProject/analysis/
├── marangoni_results/
│   ├── marangoni_time_evolution.png
│   └── marangoni_final_state.png
└── multiphysics_results/
    ├── coupling_analysis_t*.png
    ├── streamlines_t*.png
    └── vertical_profiles_t*.png
```

---

## Next Steps

1. **Run fresh simulation:**
   ```bash
   cd /home/yzk/LBMProject/build
   ./visualize_phase6_marangoni
   ```

2. **Quick check:**
   ```bash
   source /home/yzk/LBMProject/venv/bin/activate
   python /home/yzk/LBMProject/analysis/quick_marangoni_check.py
   ```

3. **Full analysis:**
   ```bash
   python /home/yzk/LBMProject/analysis/analyze_marangoni_comprehensive.py
   ```

4. **Review visualizations:**
   ```bash
   ls -lh /home/yzk/LBMProject/analysis/marangoni_results/
   ls -lh /home/yzk/LBMProject/analysis/multiphysics_results/
   ```

---

**Last Updated:** 2025-12-04
**Validated With:** phase6_test2c_visualization dataset (56 timesteps, 10000 iterations)
