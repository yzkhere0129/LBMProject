# LBMProject vs waLBerla Benchmark Plan

## Executive Summary

This document outlines a fair performance and accuracy comparison between the LBMProject (CUDA-based) and waLBerla (CPU-based with optional GPU) frameworks for a simple flat plate laser heating benchmark case.

## 1. Benchmark Test Case Design

### 1.1 Physics Scenario: Flat Plate Laser Heating

**Simplified benchmark without powder bed complexity:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Geometry | Solid flat plate | No powder particles |
| Material | Ti-6Al-4V | Standard LPBF material |
| Initial Temperature | 300 K | Room temperature |
| Laser Power | 200 W | Moderate power (no keyhole) |
| Laser Spot Radius | 50 um | Standard LPBF spot size |
| Absorptivity | 0.35 | Ti-6Al-4V value |
| Scan Speed | 0.5 m/s | Static or slow scan |

### 1.2 Domain Size Recommendations

For fair comparison, use identical domain sizes:

| Configuration | Cells | Physical Size | Memory (est.) |
|---------------|-------|---------------|---------------|
| **Small** | 100 x 100 x 50 | 200 x 200 x 100 um | ~100 MB |
| **Medium** | 200 x 200 x 100 | 400 x 400 x 200 um | ~800 MB |
| **Large** | 400 x 400 x 200 | 800 x 800 x 400 um | ~6 GB |

**Recommended grid spacing:** dx = 2.0 um (standard LPBF resolution)

### 1.3 Physics Modules Comparison

| Module | LBMProject | waLBerla | Benchmark Enabled |
|--------|------------|----------|-------------------|
| Thermal diffusion | D3Q7 LBM | Explicit FD | YES |
| Laser heat source | Gaussian + Beer-Lambert | Gaussian | YES |
| Phase change | Enthalpy method | Not implemented | OPTIONAL |
| Fluid flow | D3Q19 LBM | D3Q19 LBM | NO (thermal only) |
| Marangoni | Surface tension gradient | Not in LaserHeating | NO |
| VOF | Interface tracking | Free surface module | NO |

### 1.4 Benchmark Modes

**Mode A: Pure Thermal (Fair Comparison)**
- Thermal diffusion only
- Static laser at domain center
- No phase change
- Duration: 100 us

**Mode B: Thermal + Phase Change (LBMProject Feature)**
- Thermal diffusion with melting/solidification
- Moving laser scan
- Phase change enabled
- Duration: 500 us

**Mode C: Full Multiphysics (LBMProject Showcase)**
- All physics enabled
- Demonstrates CUDA acceleration advantage
- Duration: 1000 us

## 2. Performance Measurement Plan

### 2.1 MLUPS Calculation

**Definition:** Million Lattice Updates Per Second

```
MLUPS = (nx * ny * nz * num_steps) / (wall_time_seconds * 1e6)
```

**For thermal LBM (D3Q7):**
```
MLUPS_thermal = domain_size / wall_time * 1e-6
```

**For fluid LBM (D3Q19):**
```
MLUPS_fluid = domain_size / wall_time * 1e-6
```

### 2.2 Timing Instrumentation

**Key timers to add:**

```cpp
// In benchmark_thermal_lpbf.cu
struct BenchmarkTimers {
    float total_time_ms;           // Total wall clock time
    float kernel_time_ms;          // GPU kernel execution
    float memory_transfer_time_ms; // Host-device transfers
    float collision_time_ms;       // Collision step
    float streaming_time_ms;       // Streaming step
    float boundary_time_ms;        // Boundary conditions
    float laser_time_ms;           // Laser heat source
    float vtk_output_time_ms;      // I/O time (excluded from MLUPS)
};
```

### 2.3 GPU Memory Tracking

```cpp
// Memory usage tracking
size_t getGPUMemoryUsage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem - free_mem;
}
```

### 2.4 Performance Output Format

```
=============== BENCHMARK RESULTS ===============
Configuration:
  Domain: 200 x 200 x 100 cells
  Total cells: 4,000,000
  Time steps: 10,000
  Physics: Thermal + Laser (no phase change)

Timing:
  Total wall time: 45.3 s
  Kernel time: 42.1 s (93%)
  Memory transfers: 2.1 s (5%)
  I/O time: 1.1 s (excluded)

Performance:
  MLUPS (thermal): 885.2
  Throughput: 1.77 GB/s
  GPU Utilization: 87%

Memory:
  Peak GPU memory: 1.2 GB
  Allocated fields: 8
  Bytes per cell: 312

Accuracy:
  Max temperature: 2156.3 K
  Melt pool depth: 45.2 um
  Energy balance error: 0.3%
================================================
```

## 3. Accuracy Comparison Metrics

### 3.1 Temperature Probes

Place probe points at fixed locations:

```cpp
struct ProbePoint {
    float x, y, z;    // Position [m]
    std::string name;
};

std::vector<ProbePoint> probes = {
    {0.0002f, 0.0002f, 0.00008f, "center_surface"},   // Laser center
    {0.0003f, 0.0002f, 0.00008f, "offset_50um"},      // 50um from center
    {0.0002f, 0.0002f, 0.00006f, "depth_20um"},       // 20um below surface
    {0.0002f, 0.0002f, 0.00004f, "depth_40um"},       // 40um below surface
};
```

### 3.2 Output Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| T_max(t) | Maximum temperature vs time | K |
| T_probe[i](t) | Temperature at probe points | K |
| melt_pool_width | Width at T > T_liquidus | um |
| melt_pool_depth | Depth at T > T_liquidus | um |
| total_energy | Integral of rho*cp*T*dV | J |
| energy_balance | P_laser - P_loss - dE/dt | W |

### 3.3 Comparison Output Format

```csv
# Benchmark comparison: LBMProject vs waLBerla
# Test case: flat_plate_laser_200W
# Date: 2025-11-22
time_us,LBM_Tmax,WLB_Tmax,LBM_probe0,WLB_probe0,LBM_depth,WLB_depth
0.0,300.0,300.0,300.0,300.0,0.0,0.0
10.0,1523.4,1518.2,1456.3,1451.0,12.3,11.8
20.0,1856.7,1849.3,1789.4,1782.1,28.5,27.9
...
```

## 4. Implementation Files

### 4.1 New Files to Create

| File | Purpose |
|------|---------|
| `/home/yzk/LBMProject/apps/benchmark_thermal_lpbf.cu` | Main benchmark application |
| `/home/yzk/LBMProject/include/utils/benchmark_timer.h` | High-resolution timing utilities |
| `/home/yzk/LBMProject/configs/benchmark_flat_plate.yaml` | Benchmark configuration |
| `/home/yzk/LBMProject/scripts/run_benchmark.sh` | Automated benchmark runner |
| `/home/yzk/LBMProject/scripts/compare_results.py` | Results comparison script |

### 4.2 CMakeLists.txt Addition

```cmake
# Benchmark application
add_executable(benchmark_thermal_lpbf apps/benchmark_thermal_lpbf.cu)
set_target_properties(benchmark_thermal_lpbf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(benchmark_thermal_lpbf
    lbm_physics
    lbm_io
    CUDA::cudart
)
```

## 5. Benchmark Application Code Outline

### 5.1 Main Structure

```cpp
// benchmark_thermal_lpbf.cu
#include <cuda_runtime.h>
#include <chrono>
#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "io/vtk_writer.h"

struct BenchmarkConfig {
    // Domain
    int nx, ny, nz;
    float dx;

    // Time
    float dt;
    int num_steps;
    int warmup_steps;

    // Physics
    bool enable_phase_change;
    bool enable_laser_scan;

    // Output
    int output_interval;
    bool output_vtk;
    std::string output_dir;
};

struct BenchmarkResults {
    // Performance
    double total_time_s;
    double mlups_thermal;
    size_t peak_memory_bytes;

    // Accuracy
    std::vector<float> T_max_history;
    std::vector<float> probe_temperatures;
    float final_melt_depth;
    float energy_balance_error;
};

int main(int argc, char** argv) {
    // 1. Parse configuration
    BenchmarkConfig config = parseArgs(argc, argv);

    // 2. Initialize solver
    auto solver = createSolver(config);

    // 3. Warmup phase (not timed)
    for (int i = 0; i < config.warmup_steps; i++) {
        solver.step(config.dt);
    }
    cudaDeviceSynchronize();

    // 4. Benchmark phase
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < config.num_steps; step++) {
        solver.step(config.dt);

        if (step % config.output_interval == 0) {
            recordMetrics(solver, results);
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // 5. Compute results
    results.total_time_s = std::chrono::duration<double>(end - start).count();
    results.mlups_thermal = computeMLUPS(config, results);

    // 6. Output results
    printResults(config, results);
    writeResultsCSV(config, results);

    return 0;
}
```

## 6. waLBerla Comparison Setup

### 6.1 Matching Configuration

To ensure fair comparison, the waLBerla LaserHeating showcase should use:

```
DomainSetup
{
   blocks          < 1, 1, 1 >;
   cellsPerBlock   < 200, 200, 100 >;  // Match LBMProject
   periodic        < 0, 0, 0 >;
   domainSize      0.0004;  // 400 um domain
}

Parameters
{
   timesteps          10000;
   dt                 1e-7;    // 0.1 us (match LBMProject)
   vtkWriteFrequency  100;
}

Material
{
   density         4430.0;    // Ti-6Al-4V
   specificHeat    526.0;
   conductivity    6.7;
   meltingTemp     1923.0;
   ambientTemp     300.0;
}

Laser
{
   power           200.0;
   radius          50e-6;
   absorptivity    0.35;
   scanSpeed       0.0;       // Static for benchmark
   startPosition   < 0.0002, 0.0002, 0 >;  // Center
   direction       < 1, 0, 0 >;
}
```

### 6.2 Running waLBerla Benchmark

```bash
cd /home/yzk/walberla/build
./apps/showcases/LaserHeating/LaserHeating \
    ../apps/showcases/LaserHeating/benchmark_comparison.cfg
```

## 7. Expected Results

### 7.1 Performance Expectations

| Metric | LBMProject (GPU) | waLBerla (CPU) | Speedup |
|--------|------------------|----------------|---------|
| MLUPS (200x200x100) | 500-1000 | 10-50 | 20-50x |
| Memory bandwidth | 100-300 GB/s | 20-50 GB/s | 5-10x |
| Peak memory | 1-2 GB | 0.5-1 GB | Similar |

### 7.2 Accuracy Expectations

| Metric | Tolerance |
|--------|-----------|
| T_max difference | < 5% |
| Melt pool depth | < 10% |
| Energy balance | < 1% |

## 8. Execution Checklist

- [ ] Create `benchmark_thermal_lpbf.cu`
- [ ] Add timing instrumentation
- [ ] Create matching waLBerla config
- [ ] Run LBMProject benchmark
- [ ] Run waLBerla benchmark
- [ ] Compare results
- [ ] Generate comparison plots
- [ ] Document findings

## 9. References

1. LBMProject existing apps:
   - `/home/yzk/LBMProject/apps/visualize_laser_heating.cu`
   - `/home/yzk/LBMProject/apps/visualize_lpbf_scanning.cu`

2. waLBerla LaserHeating:
   - `/home/yzk/walberla/apps/showcases/LaserHeating/LaserHeating.cpp`
   - `/home/yzk/walberla/apps/showcases/LaserHeating/laser_heating.cfg`

3. Material properties:
   - `/home/yzk/LBMProject/include/physics/material_properties.h`
